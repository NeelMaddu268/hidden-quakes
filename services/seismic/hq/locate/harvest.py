"""Pick harvest at predicted arrivals (LOC-10): picks in no association event added to located
events, which are then relocated.

Why: PyOcto associates without station terms (``hq.associate``), so at a station-phase whose
term is large (the reference terms reach 0.9 s, against an association tolerance of 0.3 s) a real
arrival can stay in no association event. Once an event is located with statics, its predicted
arrival ``tPred = t0 + T(MAP) + static`` is known at every locator station and phase.

``harvest_and_relocate`` (called by ``hq.locate.locate_detailed`` when asked and
``harvest.enabled``) adds to an event every pick that
- is free: in no association event, ``prob >= harvest.minProb``, on a locator station;
- carries the slot's phase label and lies within ``harvest.windowS[phase]`` of the event's tPred
  there (boundary included) at a station-phase the event has no pick for; an associated pick,
  used or dropped by the outlier pass, fills the slot, so it is never replaced or duplicated;
and skips (each counted)
- a pick within the window of the tPred of two or more located events at its station-phase
  (filled slots and untrusted events included): likely another event's arrival;
- a slot with two or more such candidates;
- the slots of events whose location is not trusted (``UNTRUSTED``); their windows still count
  for the rule above.
Every event that gained picks is relocated once, with the same statics map as before (a
reference event's held-out map included) and the normal outlier pass; a harvested pick that pass
drops stays in arrivals with ``usedInLocation`` false, as an associated one does. Harvest runs
once, never iterated, and the station terms are never estimated from harvested picks
(``hq.locate.statics`` reads the association's picks only).

Caveat: a harvested pick is chosen because it agrees with the current location, so the pick
counts, gap, rmsS and formal errors it changes are not independent evidence of a better
location; offsets from independent hypocentres (the held-out reference offsets) are.

Chance: ``chance`` holds the analytic estimate (sum over the empty slots of trusted events of the
station-phase's free-pick rate x 2 windowS) and a control: the same rules applied with every
tPred moved by each of ``CONTROL_SHIFTS_S``, which counts same-label free picks near, but not
at, the predicted arrivals.
"""

import logging
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hq.config.seismology import HarvestConfig, PhaseSigma
from hq.locate.locator import EventLocation, Locator, LocatorSetup, locate_many
from hq.locate.tt_grid import PHASES

log = logging.getLogger(__name__)

Statics = Mapping[tuple[str, str], float]
UNTRUSTED = ("mapOnVolumeTop", "mapOnVolumeBottom", "pdfTruncated", "depthOnEdge")
CONTROL_SHIFTS_S = (-1.0, -0.6, 0.6, 1.0)
SLOT_COLUMNS = ("event", "stationId", "phase", "tPred", "filled", "eligible")
FREE_COLUMNS = ("id", "stationId", "phase", "t", "prob")
CHOSEN_COLUMNS = ("event", "pickId", "stationId", "phase", "t", "prob", "tPred", "offsetS")
HARVEST_COLUMNS = ("assocId", "pickId", "stationId", "phase", "t", "prob", "tPred", "offsetS",
                   "usedInLocation", "residualS")


def untrusted_flags(loc: EventLocation) -> list[str]:
    """The ``UNTRUSTED`` flags set on ``loc``."""
    hits = (loc.map_on_volume_top, loc.map_on_volume_bottom, loc.pdf_truncated,
            loc.depth_on_edge)
    return [name for name, hit in zip(UNTRUSTED, hits, strict=True) if hit]


def slot_table(
    locator: Locator, located: Sequence[EventLocation], frames: Sequence[pd.DataFrame],
    per_event: Sequence[Statics],
) -> pd.DataFrame:
    """One row per event x locator station x phase (``SLOT_COLUMNS``): ``event`` (index into
    ``located``), ``tPred`` (t0 + T(MAP) + the event's static, as arrivals.parquet holds it),
    ``filled`` (the event's picks frame has a pick there) and ``eligible`` (no ``UNTRUSTED``
    flag)."""
    rows = []
    for k, (loc, frame, statics) in enumerate(zip(located, frames, per_event, strict=True)):
        picked = set(zip(frame["stationId"].astype(str), frame["phase"].astype(str),
                         strict=True))
        eligible = not untrusted_flags(loc)
        pred = locator.travel_times(loc.e_m, loc.n_m, loc.elev_m)
        for sid, ph, tt in pred.itertuples(index=False):
            rows.append((k, sid, ph, loc.t0 + float(tt) + float(statics.get((sid, ph), 0.0)),
                         (sid, ph) in picked, eligible))
    return pd.DataFrame(rows, columns=list(SLOT_COLUMNS))


def free_picks(
    picks: pd.DataFrame, associated: Collection[str], station_ids: Collection[str],
    min_prob: float,
) -> pd.DataFrame:
    """Picks in no association event (``associated``: every associated pick id), with
    ``prob >= min_prob``, on ``station_ids`` and phase P or S (``FREE_COLUMNS``)."""
    table = pd.DataFrame({
        "id": picks["id"].astype(str).to_numpy(dtype=object),
        "stationId": picks["stationId"].astype(str).to_numpy(dtype=object),
        "phase": picks["phase"].astype(str).to_numpy(dtype=object),
        "t": picks["t"].to_numpy(dtype=np.float64),
        "prob": picks["prob"].to_numpy(dtype=np.float64),
    })
    keep = ((table["prob"] >= min_prob) & ~table["id"].isin(set(associated))
            & table["stationId"].isin(set(station_ids)) & table["phase"].isin(PHASES))
    return table[keep].sort_values(["stationId", "phase", "t", "id"], kind="stable").reset_index(
        drop=True)


def select(
    slots: pd.DataFrame, free: pd.DataFrame, window: PhaseSigma
) -> tuple[pd.DataFrame, dict[str, int]]:
    """The harvest rules on ``slot_table`` rows and ``free_picks`` rows (no locator involved).

    Returns the chosen picks (``CHOSEN_COLUMNS``, sorted by stationId, phase, t, pickId) and the
    counts ``ambiguousPicks``, ``skippedUntrustedSlots`` and ``skippedMultiCandidateSlots``. The
    result depends only on the rows, not on their order.
    """
    free = free.sort_values(["stationId", "phase", "t", "id"], kind="stable")
    by_sp = {key: g for key, g in free.groupby(["stationId", "phase"], sort=True)}
    chosen: list[tuple[Any, ...]] = []
    ambiguous: set[str] = set()
    untrusted = multi = 0
    for (sid, ph), g in slots.groupby(["stationId", "phase"], sort=True):
        fr = by_sp.get((sid, ph))
        if fr is None:
            continue
        w = float(getattr(window, str(ph)))
        t = fr["t"].to_numpy(dtype=np.float64)
        ids = fr["id"].to_numpy(dtype=object)
        tp = g["tPred"].to_numpy(dtype=np.float64)
        lo = np.searchsorted(t, tp - w, side="left")
        hi = np.searchsorted(t, tp + w, side="right")  # picks lo..hi-1 lie in the slot's window
        cover = np.zeros(t.size + 1, dtype=np.int64)
        np.add.at(cover, lo, 1)
        np.add.at(cover, hi, -1)
        amb = np.cumsum(cover)[:-1] >= 2  # in the windows of two or more events' rows
        ambiguous.update(ids[amb].tolist())
        for j, (event, filled, eligible) in enumerate(zip(
                g["event"], g["filled"].astype(bool), g["eligible"].astype(bool), strict=True)):
            if filled:
                continue
            cand = [i for i in range(int(lo[j]), int(hi[j])) if not amb[i]]
            if not cand:
                continue
            if not eligible:
                untrusted += 1
                continue
            if len(cand) > 1:
                multi += 1
                continue
            i = cand[0]
            chosen.append((event, ids[i], sid, ph, float(t[i]), float(fr["prob"].iloc[i]),
                           float(tp[j]), float(t[i] - tp[j])))
    out = pd.DataFrame(chosen, columns=list(CHOSEN_COLUMNS)).sort_values(
        ["stationId", "phase", "t", "pickId"], kind="stable").reset_index(drop=True)
    # A chosen pick lies in exactly one row's window, so it goes to one event, once.
    assert not out["pickId"].duplicated().any(), "a harvested pick went to two slots"
    assert not out.duplicated(["event", "stationId", "phase"]).any(), "a slot filled twice"
    return out, {"ambiguousPicks": len(ambiguous), "skippedUntrustedSlots": untrusted,
                 "skippedMultiCandidateSlots": multi}


def analytic_chance(slots: pd.DataFrame, free: pd.DataFrame, window: PhaseSigma,
                    span_s: float) -> float:
    """Expected chance picks: sum over the empty slots of trusted events of the station-phase's
    free-pick rate (count / ``span_s``) x 2 x window."""
    if span_s <= 0:
        return math.nan
    rate = free.groupby(["stationId", "phase"]).size() / span_s
    open_ = slots[~slots["filled"].astype(bool) & slots["eligible"].astype(bool)]
    total = 0.0
    for (sid, ph), n in open_.groupby(["stationId", "phase"]).size().items():
        total += float(rate.get((sid, ph), 0.0)) * 2.0 * float(getattr(window, str(ph))) * n
    return total


def _pick_stats(located: Sequence[EventLocation], frames: Sequence[pd.DataFrame]) -> dict:
    """``hq.locate.synthetic.measured_pick_stats``' two numbers over these locations."""
    ratio = [loc.n_s / loc.n_stations for loc in located]
    probs = []
    for loc, frame in zip(located, frames, strict=True):
        prob = dict(zip(frame["id"].astype(str), frame["prob"], strict=True))
        used = loc.arrivals.loc[loc.arrivals["usedInLocation"].to_numpy(dtype=bool), "pickId"]
        probs += [float(prob[str(p)]) for p in used]
    return {"sKeepProb": float(np.median(ratio)), "pickProb": float(np.median(probs))}


@dataclass(frozen=True, eq=False)
class HarvestReport:
    """What the harvest did, for the counts, run.json and diagnostics.md."""

    config: HarvestConfig
    added: pd.DataFrame  # HARVEST_COLUMNS; usedInLocation / residualS after the relocation
    before: dict[str, EventLocation]  # assocId -> location before harvest (events that gained)
    counts: dict[str, int]
    chance: dict[str, Any]
    pick_stats_before: dict[str, float]  # sKeepProb / pickProb over the pre-harvest locations

    def to_record(self, event_ids: Mapping[str, str]) -> dict[str, Any]:
        """``ProcessingRun.locator["harvest"]``; ``event_ids`` maps assocId to eventId."""
        a = self.added
        picks = [
            {"eventId": event_ids[str(r.assocId)], "assocId": str(r.assocId),
             "pickId": str(r.pickId), "stationId": str(r.stationId), "phase": str(r.phase),
             "offsetS": float(r.offsetS), "prob": float(r.prob),
             "usedInLocation": bool(r.usedInLocation),
             "residualS": float(r.residualS)}
            for r in a.itertuples(index=False)
        ]
        by_phase = {ph: {"added": int((a["phase"] == ph).sum()),
                         "used": int(((a["phase"] == ph) & a["usedInLocation"]).sum())}
                    for ph in PHASES}
        return {
            "config": self.config.model_dump(mode="json"),
            "note": "picks in no association event within windowS of an event's tPred (statics "
            "included) at a station-phase it had no pick for, added and the event relocated "
            "once (hq.locate.harvest); station terms never use them. Chosen by agreement with "
            "the current location, so not independent evidence of a better one. Recover them "
            "as events_located pickIds minus the event's assoc_picks.",
            "counts": dict(self.counts), "byPhase": by_phase, "chance": dict(self.chance),
            "pickStatsBeforeHarvest": dict(self.pick_stats_before), "picks": picks,
        }


def harvest_and_relocate(
    setup: LocatorSetup,
    locator: Locator,
    assoc_ids: Sequence[str],
    frames: Sequence[pd.DataFrame],
    located: Sequence[EventLocation],
    per_event: Sequence[Statics],
    picks: pd.DataFrame,
    associated: Collection[str],
    cfg: HarvestConfig,
) -> tuple[list[pd.DataFrame], list[EventLocation], HarvestReport]:
    """Harvest (see the module docstring) and relocate the events that gained picks.

    ``assoc_ids``, ``frames``, ``located`` and ``per_event`` are parallel (``locate_detailed``'s
    order); ``associated`` holds the pick ids of every association event, so ``frames`` must be
    the whole association. Returns the frames with the harvested picks added (sorted by pick
    id), the locations with the gained events relocated, and the report.
    """
    slots = slot_table(locator, located, frames, per_event)
    free = free_picks(picks, associated, locator.station_ids, cfg.minProb)
    chosen, rule_counts = select(slots, free, cfg.windowS)
    control = {f"{s:+g}": len(select(slots.assign(tPred=slots["tPred"] + s), free,
                                     cfg.windowS)[0]) for s in CONTROL_SHIFTS_S}
    t_all = picks["t"].to_numpy(dtype=np.float64)
    span = float(t_all.max() - t_all.min()) if t_all.size else 0.0
    chance = {"analyticPicks": analytic_chance(slots, free, cfg.windowS, span),
              "pickSpanS": span, "controlShiftsS": list(CONTROL_SHIFTS_S),
              "controlPicks": control,
              "controlMeanPicks": float(np.mean(list(control.values())))}

    out_frames, out_located = list(frames), list(located)
    gained = sorted({int(k) for k in chosen["event"]})
    for k in gained:
        rows = chosen.loc[chosen["event"] == k]
        add = pd.DataFrame({"id": rows["pickId"].to_numpy(dtype=object),
                            "stationId": rows["stationId"].to_numpy(dtype=object),
                            "phase": rows["phase"].to_numpy(dtype=object),
                            "t": rows["t"].to_numpy(dtype=np.float64),
                            "prob": rows["prob"].to_numpy(dtype=np.float64)})
        out_frames[k] = pd.concat([frames[k], add], ignore_index=True).sort_values(
            "id", kind="stable").reset_index(drop=True)
    relocated = locate_many(setup, [out_frames[k] for k in gained],
                            event_statics=[per_event[k] for k in gained],
                            locator=locator) if gained else []
    before = {}
    for k, loc in zip(gained, relocated, strict=True):
        before[str(assoc_ids[k])] = located[k]
        out_located[k] = loc

    used, residual = [], []
    for r in chosen.itertuples(index=False):
        arr = out_located[int(r.event)].arrivals
        row = arr[arr["pickId"].astype(str) == str(r.pickId)].iloc[0]
        used.append(bool(row["usedInLocation"]))
        residual.append(float(row["residualS"]))
    added = pd.DataFrame({
        "assocId": [str(assoc_ids[int(k)]) for k in chosen["event"]],
        **{c: chosen[c].to_numpy() for c in HARVEST_COLUMNS[1:8]},
        "usedInLocation": np.array(used, dtype=bool),
        "residualS": np.array(residual, dtype=np.float64),
    }, columns=list(HARVEST_COLUMNS))
    counts = {"picks": len(added), "picksUsed": int(added["usedInLocation"].sum()),
              "events": len(gained), "freePicks": len(free), **rule_counts}
    log.info(
        "harvest: %d picks added (%d used after relocation) to %d events from %d free picks; "
        "skipped %d ambiguous picks, %d slots of untrusted events, %d multi-candidate slots; "
        "chance: analytic %.1f, control %s",
        counts["picks"], counts["picksUsed"], counts["events"], counts["freePicks"],
        counts["ambiguousPicks"], counts["skippedUntrustedSlots"],
        counts["skippedMultiCandidateSlots"], chance["analyticPicks"], control,
    )
    return out_frames, out_located, HarvestReport(
        config=cfg, added=added, before=before, counts=counts, chance=chance,
        pick_stats_before=_pick_stats(located, frames))

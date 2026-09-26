"""Station statics (LOC-05): one additive term per station and phase, applied in the locator.

The locator predicts ``tPred = t0 + T + static`` (``hq.locate.locator``). ``statics.mode`` in
``seismology.yaml`` picks how the statics are estimated:

selfConsistent (the lane doc's Locator step 5, inside ``locate()``)
    Pass 1 locates every event with no statics. Then ``iterations`` times: each station-phase's
    static becomes its current static plus the median residual of its used picks over the
    well-constrained events (``wellConstrained``; never depthOnEdge, never a truncated PDF),
    capped at +/- ``capS``, and 0 when fewer than ``minEvents`` such events have a used pick
    there; every event is relocated with the new statics.

referenceEvents (the showcase default)
    LOC-04 found the 1D model can't hold the region's lateral structure: at the public regional
    catalog's hypocentres, stations on one side arrive early and on the other late, and
    self-consistent statics trade that against the locations. Here the terms come from the
    public regional catalog events matched to the pass-1 locations (the internal match below),
    with every hypocentre fixed at the catalog's (latitude/longitude to ENU through
    ``hq.locate.coords``; the catalog's ``elevM`` already carries its stated depth datum).
    Per reference event, ``d = t_obs - T(catalog hypocentre)`` over its associated picks. Median
    polish, ``polishIterations`` times: origin time per event = the locator's weighted median of
    ``d - term`` (weights prob / sigma); term per station-phase = median over reference events of
    ``d - t0``, capped at +/- ``referenceCapS`` and 0 when fewer than ``minReferenceEvents``
    reference events have a pick there.
    Circularity: every reference event is relocated with terms computed WITHOUT it (``folds``
    null: leave-one-out; k: k-fold, dealt round-robin in catalog origin-time order). Unmatched
    candidate events use the terms from every reference event; those are the statics table. The
    held-out relocations' offsets from the catalog are the cross-validated agreement with the
    public regional catalog's frame (reported next to an in-sample relocation with every term),
    not absolute accuracy, and absolute positions are then tied to that frame.

    One call of stage ``locate`` (``locate_with_statics`` with the run's ``catalog.parquet``):
        1. pass 1: every event located without statics;
        2. an internal one-to-one match of the pass-1 locations against the catalog
           (``match_pass_one``: ``hq.match.match`` with the same ``matching`` config, after the
           same ENU-frame check of the catalog, so the rules stage match applies);
        3. the reference pairs of its matched rows (``reference_pairs``);
        4. pass 2: terms from those pairs, every reference event relocated with held-out terms;
        5. the stage writes the pass-2 outputs, and ``statics_reference.parquet`` (H2-internal:
           the internal match's pairs, ``hq.locate.result.REFERENCE_DTYPES``).
    Then stage match -> tier as usual: stage match writes ``matches.parquet`` for the final
    locations and tier checks it is current. Stage locate never reads ``matches.parquet``, so
    rerunning it on a run dir gives the same outputs whatever match or tier files are there.
    Without a catalog, or when every fold would leave fewer than ``minReferenceEvents``
    reference events to estimate held-out terms from, ``referenceFallback`` decides: ``fail``
    (the default) raises with the reason; ``noStatics`` keeps pass 1 and records a WARNING
    (``StaticsReport.fallback``). ``locate()`` (docs/02 §5) has no catalog: in this mode it
    locates without statics (pass 1 only).

Both modes: ``statics.parquet`` holds the station-phase terms (``nEvents`` = the events each was
estimated from, so a term zeroed for too few events shows it), ``quality.statics`` is true on
events whose used picks carry a non-zero static, every static above
``diagnostics.stationResidualFlagS`` gets a written explanation from ``explain_terms``, and
``residual_sigma`` compares the post-statics residual spread with ``locator.pickSigmaS``.
"""

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig, StaticsConfig, StaticsExplainConfig
from hq.locate import LocateDetails, event_picks, locate_detailed
from hq.locate.coords import to_enu
from hq.locate.locator import Locator, weighted_median
from hq.locate.result import REFERENCE_DTYPES, typed_frame
from hq.locate.tt_grid import PHASES
from hq.locate.velocity import LayerModel

if TYPE_CHECKING:
    from hq.associate.result import AssocResult

log = logging.getLogger(__name__)

SELF_CONSISTENT = "selfConsistent"
REFERENCE_EVENTS = "referenceEvents"
FAIL = "fail"  # statics.referenceFallback
PIPELINE_ORDER = (
    "locate (one call: pass 1 without statics -> internal one-to-one match of pass 1 with "
    "catalog.parquet -> pass 2 with reference terms) -> match -> tier"
)
# |t_event - t_catalog - matches.dtS| above this means the matches were computed for other
# located events (float64 kept exactly; this only absorbs float round-off).
ROUNDOFF_S = 1e-6
NO_CATALOG = "no public regional catalog (catalog.parquet) to match the pass-1 locations against"
# StaticsReport.previous_source: where the no-statics median rmsS came from.
PREVIOUS_RUN_DIR = "the run dir's previous events_located.parquet, located without statics"
PREVIOUS_PASS_ONE = "pass 1 of this call (the same events, located without statics)"
ALL_EVENTS = "all located events"
GAUSS_MAD = 1.4826  # robust sigma = GAUSS_MAD * MAD (consistent with a Gaussian's sigma)
TERM_COLUMNS = ["stationId", "phase", "staticS", "rawS", "nEvents", "madS"]
Statics = Mapping[tuple[str, str], float]


# --- reference events -----------------------------------------------------------------------------


def reference_pairs(
    matches: pd.DataFrame,
    events_located: pd.DataFrame,
    flags: pd.DataFrame,
    catalog: pd.DataFrame,
    run: RunSection,
) -> pd.DataFrame:
    """Matched public events with their located event, association event and catalog hypocentre.

    Columns ``catalogId, eventId, assocId, catalogT, catalogE, catalogN, catalogElevM`` and
    ``pickIds`` (the located event's), sorted by catalog origin time. ``matches`` must have been
    computed for ``events_located`` (every matched event present, ``t_event - t_catalog ==
    dtS``), or this raises.
    """
    matched = matches[matches["eventId"].notna()]
    ev = events_located.set_index(events_located["id"].astype(str))
    link = flags.set_index(flags["eventId"].astype(str))["assocId"].astype(str)
    cat = catalog.set_index(catalog["id"].astype(str))
    ids = matched["eventId"].astype(str)
    cids = matched["catalogId"].astype(str)
    missing = sorted((set(ids) - set(ev.index)) | (set(ids) - set(link.index)))
    if missing:
        raise ValueError(
            f"the matches name located events that the located events / flags lack "
            f"({missing[:5]}): they were computed for other located events"
        )
    unknown = sorted(set(cids) - set(cat.index))
    if unknown:
        raise ValueError(f"the matches name public events not in the catalog: {unknown[:5]}")
    ct = cat.loc[cids, "t"].to_numpy(dtype=np.float64)
    dt = ev.loc[ids, "t"].to_numpy(dtype=np.float64) - ct
    stale = np.abs(dt - matched["dtS"].to_numpy(dtype=np.float64)) > ROUNDOFF_S
    if stale.any():
        raise ValueError(
            f"the matches are stale: {int(stale.sum())} matched event(s) have another origin "
            f"time in the located events than when matched (e.g. "
            f"{ids.iloc[int(np.argmax(stale))]})"
        )
    e, n, _ = to_enu(cat.loc[cids, "latitude"], cat.loc[cids, "longitude"],
                     cat.loc[cids, "elevM"], run.origin)
    out = pd.DataFrame({
        "catalogId": cids.to_numpy(dtype=object), "eventId": ids.to_numpy(dtype=object),
        "assocId": link.loc[ids].to_numpy(dtype=object), "catalogT": ct,
        "catalogE": np.asarray(e, dtype=np.float64), "catalogN": np.asarray(n, dtype=np.float64),
        "catalogElevM": cat.loc[cids, "elevM"].to_numpy(dtype=np.float64),
        "pickIds": [list(p) for p in ev.loc[ids, "pickIds"]],
    })
    return out.sort_values(["catalogT", "catalogId"], kind="stable").reset_index(drop=True)


@dataclass(frozen=True, eq=False)
class InternalMatch:
    """Stage locate's own one-to-one match of its pass-1 locations against the catalog."""

    matches: pd.DataFrame  # hq.match.match rows (docs/02 Match), one per public event
    pairs: pd.DataFrame  # reference_pairs of its matched rows

    def table(self) -> pd.DataFrame:
        """``statics_reference.parquet``: one row per matched public event, catalog order."""
        assoc = dict(zip(self.pairs["catalogId"].astype(str), self.pairs["assocId"].astype(str),
                         strict=True))
        rows = [{"pass1EventId": str(r.eventId), "catalogId": str(r.catalogId),
                 "assocId": assoc[str(r.catalogId)], "dtS": float(r.dtS), "distM": float(r.distM)}
                for r in self.matches[self.matches["eventId"].notna()].itertuples(index=False)]
        return typed_frame(rows, REFERENCE_DTYPES)

    def summary(self, n_reference: int, skipped: Sequence[str]) -> dict[str, Any]:
        """Run-record summary: the rule, recovered / public events, reference events used."""
        return {
            "rule": "hq.match.match of the pass-1 locations (no statics) against "
            "catalog.parquet with the matching config stage match uses (one-to-one, the same "
            "admissibility, cost and catalog ENU-frame check)",
            "publicEvents": len(self.matches),
            "recovered": int(self.matches["eventId"].notna().sum()),
            "referenceEvents": n_reference,
            "skippedOutsideGrid": list(skipped),
        }


def match_pass_one(
    pass_one: LocateDetails, catalog: pd.DataFrame, cfg: SeismologyConfig, run: RunSection
) -> InternalMatch:
    """``hq.match.match`` of the pass-1 events against ``catalog`` with ``cfg`` (stage match's
    rules, after its ENU-frame check of the catalog), and the reference pairs of the matches."""
    from hq.match import match  # at call time: hq.match imports hq.locate
    from hq.match.run import check_enu_frame

    check_enu_frame(catalog, "catalog", run, cfg.matching.enuConsistencyM)
    events = pass_one.result.events
    matches = match(events, catalog, cfg).matches
    return InternalMatch(matches, reference_pairs(matches, events, pass_one.flags, catalog, run))


def catalog_residuals(
    locator: Locator, stations: pd.DataFrame, pairs: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """``d = t_obs - T`` of each reference event's associated picks at the catalog hypocentre.

    ``stations``: the locator's stations (``enu_e``, ``enu_n``). Returns (``assocId, stationId,
    phase, d, w`` with ``w = prob / sigma``, the catalog ids left out because their hypocentre
    lies outside the travel-time grid).
    """
    grid = locator.tables.grid
    st_e = stations["enu_e"].to_numpy(dtype=np.float64)
    st_n = stations["enu_n"].to_numpy(dtype=np.float64)
    parts, skipped = [], []
    for r in pairs.itertuples(index=False):
        reach = float(np.max(np.hypot(st_e - r.catalogE, st_n - r.catalogN)))
        if not (grid.bottom_elev_m <= r.catalogElevM <= grid.top_elev_m
                and reach <= grid.r_max_m):
            skipped.append(str(r.catalogId))
            continue
        f = frames[str(r.assocId)]
        tt = locator.travel_times(r.catalogE, r.catalogN, r.catalogElevM).set_index(
            ["stationId", "phase"])["travelTimeS"]
        keys = list(zip(f["stationId"].astype(str), f["phase"].astype(str), strict=True))
        sigma = np.array([locator.pick_sigma(s, p) for s, p in keys])  # type: ignore[arg-type]
        parts.append(pd.DataFrame({
            "assocId": str(r.assocId), "stationId": [k[0] for k in keys],
            "phase": [k[1] for k in keys],
            "d": f["t"].to_numpy(dtype=np.float64) - tt.loc[keys].to_numpy(dtype=np.float64),
            "w": f["prob"].to_numpy(dtype=np.float64) / sigma,
        }))
    if skipped:
        log.warning("statics: %d reference event(s) outside the travel-time grid left out: %s",
                    len(skipped), skipped)
    empty = pd.DataFrame({"assocId": [], "stationId": [], "phase": [], "d": [], "w": []})
    return (pd.concat(parts, ignore_index=True) if parts else empty), skipped


def _capped(raw: pd.Series, n: pd.Series, cap_s: float, min_events: int) -> pd.Series:
    return raw.clip(-cap_s, cap_s).where(n >= min_events, 0.0)


def polish_terms(
    res: pd.DataFrame, *, iterations: int, cap_s: float, min_events: int
) -> pd.DataFrame:
    """Station-phase terms from residuals at fixed hypocentres (median polish), ``TERM_COLUMNS``.

    ``res``: ``assocId, stationId, phase, d, w``. Each iteration: per event ``t0`` = weighted
    median of ``d - term`` (weights ``w``); per station-phase ``rawS`` = median over events of
    ``d - t0``, ``staticS`` = ``rawS`` capped at +/- ``cap_s``, 0 below ``min_events`` events;
    ``madS`` = MAD of ``d - t0`` around ``rawS`` (how consistent the term is across events).
    """
    if res.empty:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
            TERM_COLUMNS, ["object", "object", "float64", "float64", "int64", "float64"],
            strict=True)})
    sp = res.groupby(["stationId", "phase"], sort=True).ngroup().to_numpy()
    d = res["d"].to_numpy(dtype=np.float64)
    w = res["w"].to_numpy(dtype=np.float64)
    events = res["assocId"].astype(str).to_numpy()
    rows = {aid: np.flatnonzero(events == aid) for aid in pd.unique(events)}
    term = np.zeros(d.size)
    table = pd.DataFrame()
    for _ in range(iterations):
        t0 = np.empty(d.size)
        for idx in rows.values():
            t0[idx] = weighted_median(d[idx] - term[idx], w[idx])
        r = pd.DataFrame({"stationId": res["stationId"], "phase": res["phase"], "r": d - t0,
                          "assocId": events})
        table = _term_table(r, "assocId", cap_s, min_events)
        term = table["staticS"].to_numpy()[sp]
    return table


def _term_table(r: pd.DataFrame, event: str, cap_s: float, min_events: int) -> pd.DataFrame:
    """Per station-phase: median ``r`` (rawS), capped static, event count, MAD around rawS."""
    g = r.groupby(["stationId", "phase"], sort=True)
    table = pd.DataFrame({
        "rawS": g["r"].median(), "nEvents": g[event].nunique(),
        "madS": g["r"].agg(lambda x: float(np.median(np.abs(x - np.median(x))))),
    })
    table["staticS"] = _capped(table["rawS"], table["nEvents"], cap_s, min_events)
    return table.reset_index()[TERM_COLUMNS].astype({"nEvents": "int64"})


def fold_of(pairs: pd.DataFrame, folds: int | None) -> pd.Series:
    """Fold index per reference assocId: leave-one-out (``folds`` null) or round-robin k-fold
    in catalog origin-time order (``pairs`` is sorted that way)."""
    k = len(pairs) if folds is None else folds
    return pd.Series(np.arange(len(pairs)) % max(k, 1), index=pairs["assocId"].astype(str))


def held_out_terms(
    res: pd.DataFrame, folds: pd.Series, **polish: Any
) -> dict[str, dict[tuple[str, str], float]]:
    """Per reference assocId: the statics map from ``polish_terms`` over every OTHER fold's
    residuals, so no reference event is relocated with terms it contributed to."""
    out: dict[str, dict[tuple[str, str], float]] = {}
    for f in sorted(set(folds)):
        held = set(folds.index[folds == f])
        terms = statics_map(polish_terms(res[~res["assocId"].astype(str).isin(held)], **polish))
        out.update({aid: terms for aid in held})
    return out


def statics_map(terms: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {(str(s), str(p)): float(v)
            for s, p, v in zip(terms["stationId"], terms["phase"], terms["staticS"], strict=True)}


# --- self-consistent statics ----------------------------------------------------------------------


def well_constrained(details: LocateDetails, cfg: SeismologyConfig) -> pd.Series:
    """Event ids of the well-constrained events (``statics.wellConstrained``)."""
    ev = details.result.events
    wc = cfg.statics.wellConstrained
    keep = (
        (ev["quality_nStations"] >= wc.minStations) & (ev["quality_nS"] >= wc.minS)
        & (ev["quality_gapDeg"] <= wc.maxGapDeg) & ~ev["quality_depthOnEdge"].astype(bool)
        & ev["quality_hErrM"].notna()
    )
    return ev.loc[keep, "id"].astype(str)


def self_consistent_terms(
    details: LocateDetails, current: Statics, cfg: SeismologyConfig
) -> pd.DataFrame:
    """Next statics (``TERM_COLUMNS``): current static + median used-pick residual over the
    well-constrained events, capped at +/- ``capS``, 0 below ``minEvents`` events."""
    scfg = cfg.statics
    arr = details.result.arrivals
    ids = set(well_constrained(details, cfg))
    use = arr[arr["usedInLocation"].to_numpy(dtype=bool) & arr["eventId"].astype(str).isin(ids)]
    keys = zip(use["stationId"].astype(str), use["phase"].astype(str), strict=True)
    total = use["residualS"].to_numpy(dtype=np.float64) + np.array(
        [float(current.get(k, 0.0)) for k in keys])
    frame = pd.DataFrame({"stationId": use["stationId"].astype(str),
                          "phase": use["phase"].astype(str), "r": total,
                          "eventId": use["eventId"].astype(str)})
    return _term_table(frame, "eventId", scfg.capS, scfg.minEvents)


# --- explanations and residual spread -------------------------------------------------------------


def _neighbour_median(
    terms: pd.DataFrame, pos: pd.DataFrame, station: str, phase: str, k: int, min_events: int,
    max_dist_m: float,
) -> tuple[float, list[str]]:
    """Median ``phase`` term of the ``k`` stations nearest ``station`` (epicentrally, within
    ``max_dist_m``) that have an estimated term (``nEvents >= min_events``), and their ids; NaN
    when none has one."""
    others = terms[(terms["phase"] == phase) & (terms["stationId"] != station)
                   & (terms["nEvents"] >= min_events)]
    if others.empty:
        return math.nan, []
    here = pos.loc[station]
    p = pos.loc[others["stationId"]]
    d = np.hypot(p["enu_e"].to_numpy() - here["enu_e"], p["enu_n"].to_numpy() - here["enu_n"])
    near = [i for i in np.argsort(d, kind="stable") if d[i] <= max_dist_m][:k]
    if not near:
        return math.nan, []
    ids = others["stationId"].to_numpy()[near].tolist()
    return float(np.median(others["staticS"].to_numpy(dtype=np.float64)[near])), ids


def explain_terms(
    terms: pd.DataFrame,
    stations: pd.DataFrame,
    model: LayerModel,
    centre: tuple[float, float],
    flag_s: float,
    min_events: int,
    cfg: StaticsExplainConfig,
) -> pd.DataFrame:
    """One row per static with ``|staticS| > flag_s``: its evidence and a written explanation.

    Evidence, each computed from the terms, the station geometry and the model. Each verdict
    says what the term is consistent with; none proves a cause:
    - neighbours: the median term of the ``neighbours`` nearest other stations within
      ``neighbourMaxDistM`` with an estimated term (``nEvents >= min_events``, same phase).
      The same sign and at least ``lateralFraction`` of the term: nearby stations share the
      delay, consistent with lateral structure the 1D model can't hold (diagnostics row 7)
      rather than a station fault. The only verdict that draws on other stations' terms.
    - S/P: the station's S term over its P term (same sign, ``|P| >= minRatioTermS``) against
      the model's Vp/Vs at the sensor. Within a factor ``ratioBand`` of it: P and S changed in
      proportion, consistent with a velocity anomaly near the station. Above that: S changed
      proportionally more than P, consistent with near-station rock whose Vp/Vs differs from the
      model's (slower and higher where late, as in unconsolidated sediment; faster and lower
      where early, as in crystalline rock under the model's sediment-like top layers); such rock
      moves both phases, so this covers the P term too. Within a factor ``ratioBand`` of 1:
      equal delays, a station timing offset or pick bias is possible. These bands leave few
      same-sign ratios without a label (only ratios below ``1 / ratioBand`` or between
      ``ratioBand`` and ``Vp/Vs / ratioBand``), so they label consistency, not a tested cause.
    - distance: an early term at a station more than ``farStationM`` from ``centre`` (the
      events' median epicentre): most of its ray length lies in the model's half-space, where
      the sources sit and which the layer file marks as extrapolation, so a faster half-space
      would make distant stations early. Another station beyond ``farStationM`` whose same-phase
      term is late by more than ``flag_s`` contradicts that, and then the verdict is not far.
    - elevation: the sensor against the source model's own top (above it the top layer is
      extended upward and stands in for whatever rock is there); context, not a verdict.
    ``verdict``: the first of lateral / path / vpvs / timing / far that holds, else unexplained.
    """
    columns = ["stationId", "phase", "staticS", "nEvents", "neighbourMedianS", "neighbours",
               "spRatio", "modelVpVs", "distanceM", "sensorElevM", "aboveModelTopM",
               "farContradictedBy", "verdict", "explanation"]
    pos = stations.set_index(stations["id"].astype(str))
    source_top = (model.top_extension.from_elev_m if model.top_extension is not None
                  else model.top_of_model_elev_m)
    by_key = {(str(s), str(p)): float(v) for s, p, v in
              zip(terms["stationId"], terms["phase"], terms["staticS"], strict=True)}
    sids = terms["stationId"].astype(str)
    dist_of = pd.Series(np.hypot(pos.loc[sids, "enu_e"].to_numpy() - centre[0],
                                 pos.loc[sids, "enu_n"].to_numpy() - centre[1]),
                        index=terms.index)
    far_late = terms[(dist_of > cfg.farStationM) & (terms["staticS"] > flag_s)
                     & (terms["nEvents"] >= min_events)]
    rows = []
    for r in terms[terms["staticS"].abs() > flag_s].itertuples(index=False):
        sid, ph, term = str(r.stationId), str(r.phase), float(r.staticS)
        near, near_ids = _neighbour_median(terms, pos, sid, ph, cfg.neighbours, min_events,
                                           cfg.neighbourMaxDistM)
        p_term, s_term = by_key.get((sid, "P"), 0.0), by_key.get((sid, "S"), 0.0)
        ratio = (s_term / p_term if abs(p_term) >= cfg.minRatioTermS and s_term * p_term > 0
                 else math.nan)
        elev = float(pos.loc[sid, "sensorElevM"])
        vpvs = float(model.vp_at(elev)) / float(model.vs_at(elev))
        dist = float(np.hypot(pos.loc[sid, "enu_e"] - centre[0], pos.loc[sid, "enu_n"] - centre[1]))
        lateral = (math.isfinite(near) and np.sign(near) == np.sign(term)
                   and abs(near) >= cfg.lateralFraction * abs(term))
        has_ratio = math.isfinite(ratio)
        path = has_ratio and vpvs / cfg.ratioBand <= ratio <= vpvs * cfg.ratioBand
        vpvs_differs = has_ratio and ratio > vpvs * cfg.ratioBand
        timing = has_ratio and 1.0 / cfg.ratioBand <= ratio <= cfg.ratioBand
        far_hypothesis = term < 0 and dist > cfg.farStationM
        counter = far_late[(far_late["phase"] == ph) & (far_late["stationId"] != sid)]
        far = far_hypothesis and counter.empty
        verdict = next((name for name, hit in (("lateral", lateral), ("path", path),
                                                ("vpvs", vpvs_differs), ("timing", timing),
                                                ("far", far)) if hit), "unexplained")
        above = elev - source_top
        side = "early" if term < 0 else "late"
        text = [f"{ph} arrives {abs(term):.3f} s {side} against the 1D model (n {int(r.nEvents)})."]
        if near_ids:
            text.append(
                f"Its nearest stations with a {ph} term within "
                f"{cfg.neighbourMaxDistM / 1000:g} km ({', '.join(near_ids)}) have a median "
                f"{near:+.3f} s: " + (
                    "they share it, consistent with lateral structure the 1D model can't hold "
                    "(row 7) rather than a station fault." if lateral else "they don't share it.")
            )
        else:
            text.append(f"No other station with a {ph} term lies within "
                        f"{cfg.neighbourMaxDistM / 1000:g} km.")
        if has_ratio:
            faster = "faster" if term < 0 else "slower"
            text.append(
                f"S/P term ratio {ratio:.2f} against the model's Vp/Vs {vpvs:.2f} at the sensor: "
                + (f"P and S {'slowed' if term > 0 else 'sped up'} in proportion, consistent "
                   "with a velocity anomaly near the station." if path else
                   f"S changed proportionally more than P, consistent with {faster} "
                   "near-station rock with a "
                   + ("higher Vp/Vs than the model's (as in unconsolidated sediment), which "
                      "delays P and delays S more." if term > 0
                      else "lower Vp/Vs than the model's (as in crystalline rock under the "
                      "model's sediment-like top layers), which advances P and advances S more.")
                   if vpvs_differs else
                   "equal P and S delays: a station timing offset or pick bias is possible; "
                   "check the station's timing." if timing else
                   "neither proportional to the slownesses nor equal.")
            )
        elif p_term * s_term < 0 and abs(p_term) >= cfg.minRatioTermS:
            text.append("P and S terms have opposite signs.")
        if far_hypothesis:
            head = (f"The station lies {dist / 1000:.1f} km from the events' median epicentre "
                    f"(more than {cfg.farStationM / 1000:g} km), so most of its ray length lies "
                    "in the model's half-space, which the layer file marks as extrapolation; a "
                    "faster half-space would make distant stations early.")
            if counter.empty:
                text.append(head + f" No other station that far has a {ph} term later than "
                            f"+{flag_s:g} s, so the table doesn't contradict that; nothing here "
                            "tests it further.")
            else:
                cites = ", ".join(
                    f"{c.stationId} at {dist_of[i] / 1000:.1f} km has a late {ph} term "
                    f"{c.staticS:+.3f} s" for i, c in zip(counter.index,
                                                          counter.itertuples(index=False),
                                                          strict=True))
                text.append(head + f" This table contradicts that: {cites}.")
        text.append(
            f"Sensor at {elev:.0f} m ASL, "
            + (f"{above:.0f} m above the velocity model's own top ({source_top:.0f} m ASL), where "
               "its top layer is extended upward and stands in for whatever rock is there."
               if above > 0 else f"{-above:.0f} m below the velocity model's own top.")
        )
        if verdict == "unexplained":
            text.append("No evidence here explains it: an unexplained static (depth gate).")
        rows.append({"stationId": sid, "phase": ph, "staticS": term, "nEvents": int(r.nEvents),
                     "neighbourMedianS": near, "neighbours": ",".join(near_ids),
                     "spRatio": ratio, "modelVpVs": vpvs, "distanceM": dist, "sensorElevM": elev,
                     "aboveModelTopM": above,
                     "farContradictedBy": (",".join(counter["stationId"].astype(str))
                                           if far_hypothesis else ""),
                     "verdict": verdict, "explanation": " ".join(text)})
    return pd.DataFrame(rows, columns=columns)


def residual_sigma(
    arrivals: pd.DataFrame, event_ids: Sequence[str], cfg: SeismologyConfig, label: str
) -> pd.DataFrame:
    """Per phase: robust sigma (``GAUSS_MAD`` x MAD) of the used-pick residuals of ``event_ids``
    (``label`` names them) against ``locator.pickSigmaS``; ``wellAbove`` past
    ``statics.sigmaFlagRatio``; ``recommendedS`` the robust sigma rounded to ms."""
    use = arrivals[arrivals["usedInLocation"].to_numpy(dtype=bool)
                   & arrivals["eventId"].astype(str).isin(set(event_ids))]
    rows = []
    for ph in PHASES:
        r = use.loc[use["phase"].astype(str) == ph, "residualS"].to_numpy(dtype=np.float64)
        conf = float(getattr(cfg.locator.pickSigmaS, ph))
        sigma = GAUSS_MAD * float(np.median(np.abs(r - np.median(r)))) if r.size else math.nan
        ratio = sigma / conf
        rows.append({"events": label, "phase": ph, "nPicks": int(r.size), "configuredS": conf,
                     "robustSigmaS": sigma, "ratio": ratio,
                     "wellAbove": bool(math.isfinite(ratio) and ratio > cfg.statics.sigmaFlagRatio),
                     "recommendedS": round(sigma, 3) if math.isfinite(sigma) else math.nan})
    return pd.DataFrame(rows)


# --- orchestration --------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class StaticsReport:
    """What the statics pass did, for ``diagnostics.md`` and the run record."""

    mode: str
    pass_number: int  # 1: located without statics; 2: located with statics
    note: str
    terms: pd.DataFrame  # TERM_COLUMNS: the statics table's terms (rawS uncapped)
    cap_s: float
    min_events: int
    explanations: pd.DataFrame  # explain_terms
    sigma: pd.DataFrame  # residual_sigma: the calibration events first, then all events
    sigma_events: str  # the calibration events (the recommendation comes from them)
    history: pd.DataFrame  # per pass: iteration, medianRmsS, nCalibration, nNonZero, maxAbsS
    # referenceEvents pass 2: per reference event, catalogId, eventId (after), assocId, fold,
    # offsets before (no statics), after (held-out terms) and inSample (all-reference terms):
    # h/de/dn/dz (m), rmsS (s)
    reference: pd.DataFrame | None = None
    skipped: tuple[str, ...] = ()  # reference events outside the travel-time grid
    previous_median_rms_s: float | None = None  # all events located without statics
    previous_source: str = PREVIOUS_RUN_DIR  # where previous_median_rms_s came from
    # referenceEvents with referenceFallback noStatics: why no statics were applied (a WARNING)
    fallback: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def offsets_summary(self) -> dict[str, dict[str, float]]:
        """Median and p90 of the horizontal and |vertical| offsets: before (no statics), after
        (held-out terms) and inSample (all-reference terms, the held-out events included)."""
        ref = self.reference
        if ref is None or ref.empty:
            return {}
        out = {}
        for when in ("before", "after", "inSample"):
            if f"{when}HM" not in ref.columns:
                continue
            h = ref[f"{when}HM"].to_numpy(dtype=np.float64)
            v = np.abs(ref[f"{when}DzM"].to_numpy(dtype=np.float64))
            out[when] = {"medianHM": float(np.median(h)), "p90HM": float(np.quantile(h, 0.9)),
                         "medianAbsDzM": float(np.median(v)),
                         "p90AbsDzM": float(np.quantile(v, 0.9)),
                         "medianRmsS": float(np.median(ref[f"{when}RmsS"]))}
        return out

    def to_record(self) -> dict[str, Any]:
        def rows(frame: pd.DataFrame | None) -> list[dict[str, Any]] | None:
            if frame is None:
                return None
            return [{k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                     for k, v in rec.items()} for rec in frame.to_dict(orient="records")]

        return {
            "mode": self.mode, "pass": self.pass_number, "note": self.note,
            "pipelineOrder": PIPELINE_ORDER if self.mode == REFERENCE_EVENTS else None,
            "capS": self.cap_s, "minEvents": self.min_events, "terms": rows(self.terms),
            "explanations": rows(self.explanations), "sigma": rows(self.sigma),
            "sigmaEvents": self.sigma_events, "history": rows(self.history),
            "reference": rows(self.reference), "skippedOutsideGrid": list(self.skipped),
            "crossValidatedOffsets": self.offsets_summary() or None,
            "previousMedianRmsS": self.previous_median_rms_s,
            "previousMedianRmsSource": self.previous_source, "fallback": self.fallback,
            **self.extra,
        }


@dataclass(frozen=True, eq=False)
class StaticsOutcome:
    details: LocateDetails
    report: StaticsReport
    internal: InternalMatch | None = None  # referenceEvents: the internal match of pass 1


@dataclass(frozen=True, eq=False)
class _Subset:
    events: pd.DataFrame
    picks: pd.DataFrame


def _history_row(k: int, details: LocateDetails, n_cal: int, terms: pd.DataFrame | None) -> dict:
    s = terms["staticS"].to_numpy(dtype=np.float64) if terms is not None else np.zeros(0)
    return {"iteration": k, "medianRmsS": float(details.result.events["quality_rmsS"].median()),
            "nCalibration": n_cal, "nNonZero": int(np.count_nonzero(s)),
            "maxAbsS": float(np.max(np.abs(s))) if s.size else 0.0}


def _report(
    cfg: SeismologyConfig, details: LocateDetails, terms: pd.DataFrame, *, mode: str,
    pass_number: int, note: str, cap_s: float, min_events: int, sigma_ids: Sequence[str],
    sigma_events: str, history: list[dict], **kw: Any,
) -> StaticsReport:
    ev = details.result.events
    centre = ((float(ev["enu_e"].median()), float(ev["enu_n"].median())) if len(ev)
              else (0.0, 0.0))
    explained = explain_terms(terms, details.stations, details.locator.tables.model, centre,
                              cfg.diagnostics.stationResidualFlagS, min_events,
                              cfg.statics.explain)
    return StaticsReport(
        mode=mode, pass_number=pass_number, note=note, terms=terms, cap_s=cap_s,
        min_events=min_events, explanations=explained,
        sigma=pd.concat([
            residual_sigma(details.result.arrivals, sigma_ids, cfg, sigma_events),
            *([] if sigma_events == ALL_EVENTS else [residual_sigma(
                details.result.arrivals, ev["id"].astype(str).tolist(), cfg, ALL_EVENTS)]),
        ], ignore_index=True), sigma_events=sigma_events,
        history=pd.DataFrame(history), **kw,
    )


def _offsets(details: LocateDetails, pairs: pd.DataFrame, prefix: str) -> pd.DataFrame:
    at = {aid: k for k, aid in enumerate(details.assoc_ids)}
    ids = details.result.events["id"].astype(str).tolist()
    rows = []
    for r in pairs.itertuples(index=False):
        loc = details.locations[at[str(r.assocId)]]
        de, dn = loc.e_m - r.catalogE, loc.n_m - r.catalogN
        rows.append({"assocId": str(r.assocId), f"{prefix}EventId": ids[at[str(r.assocId)]],
                     f"{prefix}HM": math.hypot(de, dn), f"{prefix}DeM": de, f"{prefix}DnM": dn,
                     f"{prefix}DzM": loc.elev_m - r.catalogElevM, f"{prefix}RmsS": loc.rms_s})
    return pd.DataFrame(rows)


def _previous_rms(previous: pd.DataFrame | None) -> float | None:
    if previous is None or previous.empty or previous["quality_statics"].astype(bool).any():
        return None
    return float(previous["quality_rmsS"].median())


def locate_with_statics(
    assoc: "AssocResult",
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    cfg: SeismologyConfig,
    run: RunSection,
    *,
    run_id: str | None = None,
    cache_dir: Path | None = None,
    catalog: pd.DataFrame | None = None,
    previous_events: pd.DataFrame | None = None,
    model: LayerModel | None = None,
) -> StaticsOutcome:
    """Locate with the configured statics (see the module docstring).

    ``catalog``: the run's ``catalog.parquet`` rows (``CatalogEvent``), used by mode
    referenceEvents only; None means there is none (``referenceFallback`` decides).
    ``previous_events``: mode selfConsistent only, the run dir's events_located.parquet before
    this call, whose median rmsS is reported as the no-statics value when it was located without
    statics (referenceEvents reports its own pass 1 instead).
    """
    started = time.perf_counter()
    scfg = cfg.statics
    kw: dict[str, Any] = {"run_id": run_id, "cache_dir": cache_dir, "model": model}
    if scfg.mode == SELF_CONSISTENT:
        previous = _previous_rms(previous_events)
        details = locate_detailed(assoc, picks, stations, cfg, run, **kw)
        current = polish_terms(pd.DataFrame(), iterations=1, cap_s=scfg.capS, min_events=1)
        cal = well_constrained(details, cfg)
        history = [_history_row(0, details, len(cal), None)]
        for k in range(1, scfg.iterations + 1):
            current = self_consistent_terms(details, statics_map(current), cfg)
            details = locate_detailed(assoc, picks, stations, cfg, run,
                                      statics=statics_map(current),
                                      static_events=_counts(current), **kw)
            cal = well_constrained(details, cfg)
            history.append(_history_row(k, details, len(cal), current))
            log.info("statics selfConsistent iteration %d: median rmsS %.3f s, %d non-zero "
                     "statics (max |static| %.3f s)", k, history[-1]["medianRmsS"],
                     history[-1]["nNonZero"], history[-1]["maxAbsS"])
        report = _report(
            cfg, details, current, mode=SELF_CONSISTENT, pass_number=2,
            note=f"{scfg.iterations} iteration(s) of the median used-pick residual over the "
            "well-constrained events", cap_s=scfg.capS, min_events=scfg.minEvents,
            sigma_ids=list(cal), sigma_events="well-constrained events", history=history,
            previous_median_rms_s=previous)
        log.info("statics: selfConsistent done in %.1f s", time.perf_counter() - started)
        return StaticsOutcome(details, report)

    # referenceEvents (module docstring): pass 1, internal match, pass 2, in this one call.
    if catalog is None:
        _fallback_or_fail(scfg, NO_CATALOG)  # before pass 1: with "fail" nothing is located
    base = locate_detailed(assoc, picks, stations, cfg, run, **kw)  # pass 1, no statics
    pass_one = base.result.events
    pass_one_rms = float(pass_one["quality_rmsS"].median()) if len(pass_one) else None
    if catalog is None:
        return _pass_one_only(cfg, base, NO_CATALOG, pass_one_rms)
    internal = match_pass_one(base, catalog, cfg, run)
    order, frames = event_picks(assoc, picks, cfg.locator.minPicks)
    res, skipped = catalog_residuals(base.locator, base.stations, internal.pairs,
                                     dict(zip(order, frames, strict=True)))
    pairs = internal.pairs[~internal.pairs["catalogId"].isin(skipped)].reset_index(drop=True)
    folds = fold_of(pairs, scfg.folds)
    k_desc = "leave-one-out" if scfg.folds is None else f"{scfg.folds}-fold"
    matched = internal.summary(len(pairs), skipped)
    log.info("statics: internal match of pass 1: recovered %d / %d public regional catalog "
             "events; %d reference event(s) inside the travel-time grid", matched["recovered"],
             matched["publicEvents"], len(pairs))
    # Each held-out term is estimated from the reference events outside its fold.
    fewest = len(pairs) - int(folds.value_counts().max()) if len(pairs) else 0
    if fewest < scfg.minReferenceEvents:
        reason = (
            f"the internal match of pass 1 recovered {matched['recovered']} of "
            f"{matched['publicEvents']} public regional catalog events, {len(pairs)} of them "
            f"usable as reference events"
            + (f" ({len(skipped)} outside the travel-time grid: {', '.join(skipped)})"
               if skipped else "")
            + f"; held out {k_desc}, a fold leaves {fewest} to estimate its terms from, fewer "
            f"than minReferenceEvents {scfg.minReferenceEvents}"
        )
        _fallback_or_fail(scfg, reason)
        return _pass_one_only(cfg, base, reason, pass_one_rms, internal, skipped,
                              {"internalMatch": matched})

    # Pass 2: terms at the catalog hypocentres, held out per fold for the reference events.
    polish = {"iterations": scfg.polishIterations, "cap_s": scfg.referenceCapS,
              "min_events": scfg.minReferenceEvents}
    terms = polish_terms(res, **polish)
    held_out = held_out_terms(res, folds, **polish)
    ref_assoc = set(pairs["assocId"])
    subset = _Subset(events=assoc.events[assoc.events["assocId"].astype(str).isin(ref_assoc)],
                     picks=assoc.picks[assoc.picks["assocId"].astype(str).isin(ref_assoc)])
    # In-sample comparison only (never written): the reference events with every term.
    in_sample = locate_detailed(subset, picks, stations, cfg, run, statics=statics_map(terms),
                                static_events=_counts(terms), **kw)
    details = locate_detailed(assoc, picks, stations, cfg, run, statics=statics_map(terms),
                              event_statics=held_out, static_events=_counts(terms), **kw)
    ref = pairs[["catalogId", "assocId"]].assign(fold=folds.reindex(pairs["assocId"]).to_numpy())
    # "before" is pass 1 (each event is located independently, so it is the no-statics location).
    ref = ref.merge(_offsets(base, pairs, "before"), on="assocId").merge(
        _offsets(details, pairs, "after"), on="assocId").merge(
        _offsets(in_sample, pairs, "inSample").drop(columns="inSampleEventId"), on="assocId")
    report = _report(
        cfg, details, terms, mode=REFERENCE_EVENTS, pass_number=2,
        note=f"terms from {len(pairs)} reference events at the public regional catalog's "
        f"hypocentres; reference events relocated {k_desc} (held-out terms)",
        cap_s=scfg.referenceCapS, min_events=scfg.minReferenceEvents,
        sigma_ids=ref["afterEventId"].tolist(),
        sigma_events="reference events (held-out terms)",
        history=[_history_row(0, base, len(pairs), None),
                 _history_row(1, details, len(pairs), terms)],
        reference=ref, skipped=tuple(skipped), previous_median_rms_s=pass_one_rms,
        previous_source=PREVIOUS_PASS_ONE,
        extra={"foldScheme": k_desc, "nReferenceResiduals": len(res), "internalMatch": matched},
    )
    summary = report.offsets_summary()
    if summary:
        b, a, i = summary["before"], summary["after"], summary["inSample"]
        log.info(
            "statics: referenceEvents pass 2 on %d reference events (%s): horizontal offset from "
            "the catalog median %.0f -> %.0f m (in-sample %.0f m), p90 %.0f -> %.0f m (in-sample "
            "%.0f m); |dz| median %.0f -> %.0f m (in-sample %.0f m); %d non-zero terms; %.1f s",
            len(pairs), k_desc, b["medianHM"], a["medianHM"], i["medianHM"], b["p90HM"],
            a["p90HM"], i["p90HM"], b["medianAbsDzM"], a["medianAbsDzM"], i["medianAbsDzM"],
            int(np.count_nonzero(terms["staticS"])), time.perf_counter() - started,
        )
    return StaticsOutcome(details, report, internal)


def _fallback_or_fail(scfg: StaticsConfig, reason: str) -> None:
    """Raise with ``reason`` under referenceFallback fail; otherwise log the WARNING."""
    if scfg.referenceFallback == FAIL:
        raise ValueError(
            f"statics.mode referenceEvents: {reason}. Stage locate estimates its station terms "
            "from reference events: give it the run's catalog.parquet (stage catalog) and enough "
            "matched events, or set statics.referenceFallback: noStatics to locate without "
            "statics"
        )
    log.warning("statics: referenceEvents: %s; statics.referenceFallback is noStatics: every "
                "event is located WITHOUT statics", reason)


def _pass_one_only(
    cfg: SeismologyConfig, base: LocateDetails, reason: str, pass_one_rms: float | None,
    internal: InternalMatch | None = None, skipped: Sequence[str] = (),
    extra: dict[str, Any] | None = None,
) -> StaticsOutcome:
    """referenceEvents fallback (referenceFallback noStatics): pass 1 is the result."""
    scfg = cfg.statics
    report = _report(
        cfg, base, polish_terms(pd.DataFrame(), iterations=1, cap_s=1.0, min_events=1),
        mode=REFERENCE_EVENTS, pass_number=1,
        note=f"WARNING: {reason}; statics.referenceFallback is noStatics, so every event is "
        "located without statics", cap_s=scfg.referenceCapS,
        min_events=scfg.minReferenceEvents,
        sigma_ids=base.result.events["id"].astype(str).tolist(), sigma_events=ALL_EVENTS,
        history=[_history_row(0, base, 0, None)], skipped=tuple(skipped),
        previous_median_rms_s=pass_one_rms, previous_source=PREVIOUS_PASS_ONE, fallback=reason,
        extra=extra or {},
    )
    return StaticsOutcome(base, report, internal)


def _counts(terms: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {(str(s), str(p)): int(n) for s, p, n in
            zip(terms["stationId"], terms["phase"], terms["nEvents"], strict=True)}

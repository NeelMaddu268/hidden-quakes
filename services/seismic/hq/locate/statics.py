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
    public regional catalog events matched to located events (``matches.parquet`` of a prior
    match pass), with every hypocentre fixed at the catalog's (latitude/longitude to ENU through
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

    Pipeline order with this mode:
        locate (pass 1, no statics) -> match -> locate (pass 2, reference terms) -> match -> tier
    Stage ``locate`` runs pass 2 when ``matches.parquet`` is in the run dir and pass 1 otherwise
    (logged: the statics pass needs a match first). ``locate()`` (docs/02 §5) has no matches, so
    in this mode it always runs pass 1 (no statics).

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
from hq.config.seismology import SeismologyConfig, StaticsExplainConfig
from hq.locate import (
    LocateDetails,
    default_cache_dir,
    event_picks,
    locate_detailed,
    used_stations,
)
from hq.locate.coords import to_enu
from hq.locate.locator import Locator, LocatorSetup, build_locator, weighted_median
from hq.locate.tt_grid import PHASES
from hq.locate.tt_grid3d import Model3dSource
from hq.locate.velocity import LayerModel, load_configured_model

if TYPE_CHECKING:
    from hq.associate.result import AssocResult

log = logging.getLogger(__name__)

SELF_CONSISTENT = "selfConsistent"
REFERENCE_EVENTS = "referenceEvents"
PIPELINE_ORDER = (
    "locate (pass 1, no statics) -> match -> locate (pass 2, reference terms from "
    "matches.parquet) -> match -> tier"
)
# |t_event - t_catalog - matches.dtS| above this means matches.parquet was written for another
# events_located.parquet (parquet keeps float64 exactly; this only absorbs float round-off).
ROUNDOFF_S = 1e-6
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
    written for ``events_located`` (every matched event present, ``t_event - t_catalog == dtS``),
    or this raises: rerun stage match first.
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
            f"matches.parquet names located events that events_located.parquet / "
            f"locate_flags.parquet lack ({missing[:5]}): rerun stage match on these events first"
        )
    unknown = sorted(set(cids) - set(cat.index))
    if unknown:
        raise ValueError(f"matches.parquet names public events not in catalog.parquet: {unknown[:5]}")
    ct = cat.loc[cids, "t"].to_numpy(dtype=np.float64)
    dt = ev.loc[ids, "t"].to_numpy(dtype=np.float64) - ct
    stale = np.abs(dt - matched["dtS"].to_numpy(dtype=np.float64)) > ROUNDOFF_S
    if stale.any():
        raise ValueError(
            f"matches.parquet is stale: {int(stale.sum())} matched event(s) have another origin "
            f"time in events_located.parquet than when matched (e.g. {ids.iloc[int(np.argmax(stale))]})"
            ": rerun stage match on these located events first"
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


def check_same_association(pairs: pd.DataFrame, assoc_picks: pd.DataFrame) -> None:
    """Raise unless every reference event's located picks belong to its association event in
    ``assoc_picks``: an association rerun since the match renumbers or regroups events."""
    groups: dict[str, set[str]] = {}
    for aid, pid in zip(assoc_picks["assocId"].astype(str), assoc_picks["pickId"].astype(str),
                        strict=True):
        groups.setdefault(aid, set()).add(pid)
    bad = [str(r.catalogId) for r in pairs.itertuples(index=False)
           if not set(map(str, r.pickIds)) <= groups.get(str(r.assocId), set())]
    if bad:
        raise ValueError(
            f"reference events {bad[:5]} were located from another association than "
            "assoc_picks.parquet holds now, so matches.parquet is stale: rerun stage associate "
            "(it removes the stale matches.parquet), then locate (pass 1) and match on this "
            "association first"
        )


def catalog_residuals(
    locator: Locator, stations: pd.DataFrame, pairs: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """``d = t_obs - T`` of each reference event's associated picks at the catalog hypocentre.

    ``stations``: the locator's stations (``enu_e``, ``enu_n``). Returns (``assocId, stationId,
    phase, d, w`` with ``w = prob / sigma``, the catalog ids left out because their hypocentre
    lies outside the travel-time grid).
    """
    parts, skipped = [], []
    for r in pairs.itertuples(index=False):
        if not locator.covers(r.catalogE, r.catalogN, r.catalogElevM):
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


@dataclass(frozen=True)
class ModelFacts:
    """What ``explain_terms`` says about the travel-time model at one station."""

    against: str  # the travel times the station's terms are relative to
    vpvs: float  # the model's Vp/Vs at the sensor
    above_top_m: float  # sensor above the model's own top (NaN: the model has no such top)
    elevation: str  # the sentence placing the sensor against the model's top or ground
    deep: str  # where a distant station's rays mostly run (the far-station hypothesis)


DEEP_1D = "the model's half-space, which the layer file marks as extrapolation"


def facts_1d(model: LayerModel, stations: pd.DataFrame,
             against: str = "the 1D model") -> dict[str, ModelFacts]:
    """Per station: the 1D layer model at its sensor (``model`` as the tables hold it: its
    ``top_extension``, when set, says where the source model's own top was)."""
    source_top = (model.top_extension.from_elev_m if model.top_extension is not None
                  else model.top_of_model_elev_m)
    out = {}
    for sid, elev in zip(stations["id"].astype(str), stations["sensorElevM"].astype(float),
                         strict=True):
        above = elev - source_top
        out[sid] = ModelFacts(
            against=against,
            vpvs=float(model.vp_at(elev)) / float(model.vs_at(elev)),
            above_top_m=above,
            elevation=f"Sensor at {elev:.0f} m ASL, " + (
                f"{above:.0f} m above the velocity model's own top ({source_top:.0f} m ASL), "
                "where its top layer is extended upward and stands in for whatever rock is there."
                if above > 0 else f"{-above:.0f} m below the velocity model's own top."),
            deep=DEEP_1D,
        )
    return out


def facts_grid3d(locator: Locator, stations: pd.DataFrame) -> dict[str, ModelFacts]:
    """Per station: the 3D model's column at a station with a 3D table (Vp/Vs at the sensor as
    its table's near-receiver seed holds it; where the file puts the ground), the 1D facts at a
    station on its 1D tables (outside the 3D model, or a constant column under fallback1d)."""
    t3 = locator.tables3d
    if t3 is None:
        raise ValueError("facts_grid3d needs a grid3d locator")
    out = facts_1d(locator.tables.model, stations,
                   against="its 1D tables (no 3D table: 1D fallback)")
    for sid, elev in zip(stations["id"].astype(str), stations["sensorElevM"].astype(float),
                         strict=True):
        if not t3.has(sid):
            continue
        col = t3.column_model_of(sid)
        c = t3.columns[sid]
        vp, vs = float(col.vp_at(elev)), float(col.vs_at(elev))
        if c.kind == "basin" and c.ground_low_elev_m is not None:
            where = (f"the 3D model's ground in its column lies between {c.ground_low_elev_m:.0f} "
                     f"and {c.ground_high_elev_m:.0f} m ASL (Vp {vp:.0f}, Vs {vs:.0f} m/s at the "
                     "sensor).")
        elif c.kind == "constant":
            where = (f"in a 3D model column that holds one velocity at every elevation (Vp "
                     f"{vp:.0f}, Vs {vs:.0f} m/s): the file has no ground surface or basin data "
                     "there, so the 3D times put basement rock right up to the sensor.")
        else:
            where = (f"in a 3D model column with no air value on top (Vp {vp:.0f}, Vs {vs:.0f} "
                     "m/s at the sensor): the file marks no ground surface there.")
        out[sid] = ModelFacts(
            against="the 3D model", vpvs=vp / vs, above_top_m=math.nan,
            elevation=f"Sensor at {elev:.0f} m ASL; " + where,
            deep="the 3D model's basement, which the file holds at one velocity everywhere",
        )
    return out


def explain_terms(
    terms: pd.DataFrame,
    stations: pd.DataFrame,
    model: LayerModel,
    centre: tuple[float, float],
    flag_s: float,
    min_events: int,
    cfg: StaticsExplainConfig,
    facts: Mapping[str, ModelFacts] | None = None,
) -> pd.DataFrame:
    """One row per static with ``|staticS| > flag_s``: its evidence and a written explanation.

    ``facts`` says, per station, which travel times its terms are relative to and what that
    model holds at the sensor (grid3d: ``facts_grid3d``); None: ``facts_1d(model, stations)``.

    Evidence, each computed from the terms, the station geometry and the model. Each verdict
    says what the term is consistent with; none proves a cause:
    - neighbours: the median term of the ``neighbours`` nearest other stations within
      ``neighbourMaxDistM`` with an estimated term (``nEvents >= min_events``, same phase).
      The same sign and at least ``lateralFraction`` of the term: nearby stations share the
      delay, consistent with lateral structure the model can't hold (diagnostics row 7)
      rather than a station fault. The only verdict that draws on other stations' terms.
    - S/P: the station's S term over its P term (same sign, ``|P| >= minRatioTermS``) against
      the model's Vp/Vs at the sensor (grid3d: in the station's 3D column, air handled as for its
      table). Within a factor ``ratioBand`` of it: P and S changed in proportion, consistent
      with a velocity anomaly near the station. Above that: S changed
      proportionally more than P, consistent with near-station rock whose Vp/Vs differs from the
      model's (slower and higher where late, as in unconsolidated sediment; faster and lower
      where early, as in crystalline rock under the model's sediment-like top layers); such rock
      moves both phases, so this covers the P term too. Within a factor ``ratioBand`` of 1:
      equal delays, a station timing offset or pick bias is possible. These bands leave few
      same-sign ratios without a label (only ratios below ``1 / ratioBand`` or between
      ``ratioBand`` and ``Vp/Vs / ratioBand``), so they label consistency, not a tested cause.
    - distance: an early term at a station more than ``farStationM`` from ``centre`` (the
      events' median epicentre): most of its ray length lies deep in the model (1D: the
      half-space, where the sources sit and which the layer file marks as extrapolation; 3D: the
      file's one-velocity basement), so faster rock there would make distant stations early.
      Another station beyond ``farStationM`` whose same-phase term is late by more than
      ``flag_s`` contradicts that, and then the verdict is not far.
    - elevation: the sensor against the source model's own top (above it the top layer is
      extended upward and stands in for whatever rock is there; grid3d: where the station's 3D
      column puts the ground, or that it has none); context, not a verdict.
    ``verdict``: the first of lateral / path / vpvs / timing / far that holds, else unexplained.
    """
    columns = ["stationId", "phase", "staticS", "nEvents", "neighbourMedianS", "neighbours",
               "spRatio", "modelVpVs", "distanceM", "sensorElevM", "aboveModelTopM",
               "farContradictedBy", "verdict", "explanation"]
    pos = stations.set_index(stations["id"].astype(str))
    at = facts_1d(model, stations) if facts is None else facts
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
        here = at[sid]
        vpvs = here.vpvs
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
        side = "early" if term < 0 else "late"
        text = [(f"{ph} arrives {abs(term):.3f} s {side} against {here.against} "
                 f"(n {int(r.nEvents)}).")]
        if near_ids:
            text.append(
                f"Its nearest stations with a {ph} term within "
                f"{cfg.neighbourMaxDistM / 1000:g} km ({', '.join(near_ids)}) have a median "
                f"{near:+.3f} s: " + (
                    f"they share it, consistent with lateral structure {here.against} can't hold "
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
                    f"in {here.deep}; faster rock there would make distant stations early.")
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
        text.append(here.elevation)
        if verdict == "unexplained":
            text.append("No evidence here explains it: an unexplained static (depth gate).")
        rows.append({"stationId": sid, "phase": ph, "staticS": term, "nEvents": int(r.nEvents),
                     "neighbourMedianS": near, "neighbours": ",".join(near_ids),
                     "spRatio": ratio, "modelVpVs": vpvs, "distanceM": dist, "sensorElevM": elev,
                     "aboveModelTopM": here.above_top_m,
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
    previous_median_rms_s: float | None = None  # stored no-statics events_located, all events
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
            "previousMedianRmsS": self.previous_median_rms_s, **self.extra,
        }


@dataclass(frozen=True, eq=False)
class StaticsOutcome:
    details: LocateDetails
    report: StaticsReport


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
    locator = details.locator
    facts = None if locator.tables3d is None else facts_grid3d(locator, details.stations)
    explained = explain_terms(terms, details.stations, locator.tables.model, centre,
                              cfg.diagnostics.stationResidualFlagS, min_events,
                              cfg.statics.explain, facts)
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
    reference: pd.DataFrame | None = None,
    previous_events: pd.DataFrame | None = None,
    model: LayerModel | None = None,
    model3d: Model3dSource | None = None,
) -> StaticsOutcome:
    """Locate with the configured statics (see the module docstring).

    ``reference``: ``reference_pairs`` output (mode referenceEvents; None runs pass 1).
    ``previous_events``: the run dir's events_located.parquet before this pass, whose median
    rmsS is reported as the no-statics value when it was located without statics.
    """
    started = time.perf_counter()
    scfg = cfg.statics
    kw: dict[str, Any] = {"run_id": run_id, "cache_dir": cache_dir, "model": model,
                          "model3d": model3d}
    previous = _previous_rms(previous_events)
    if scfg.mode == SELF_CONSISTENT:
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

    if reference is None or reference.empty:
        base = locate_detailed(assoc, picks, stations, cfg, run, **kw)
        note = ("no matches.parquet in the run dir: the statics pass needs a match first ("
                + PIPELINE_ORDER + "); located without statics" if reference is None else
                "matches.parquet has no matched public event: located without statics")
        log.info("statics: referenceEvents pass 1: %s", note)
        report = _report(
            cfg, base, polish_terms(pd.DataFrame(), iterations=1, cap_s=1.0, min_events=1),
            mode=REFERENCE_EVENTS, pass_number=1, note=note, cap_s=scfg.referenceCapS,
            min_events=scfg.minReferenceEvents,
            sigma_ids=base.result.events["id"].astype(str).tolist(),
            sigma_events=ALL_EVENTS, history=[_history_row(0, base, 0, None)],
            previous_median_rms_s=previous)
        return StaticsOutcome(base, report)

    # Pass 2: terms at the catalog hypocentres, held out per fold for the reference events.
    setup = LocatorSetup(
        stations=used_stations(stations, run, cfg.locator.enuConsistencyTolM),
        model=load_configured_model(cfg.velocity) if model is None else model,
        config=cfg, run=run,
        cache_dir=Path(cache_dir) if cache_dir is not None else default_cache_dir(cfg),
        model3d=model3d,
    )
    check_same_association(reference, assoc.picks)
    locator = build_locator(setup)
    order, frames = event_picks(assoc, picks, cfg.locator.minPicks)
    res, skipped = catalog_residuals(locator, setup.stations, reference,
                                     dict(zip(order, frames, strict=True)))
    pairs = reference[~reference["catalogId"].isin(skipped)].reset_index(drop=True)
    if pairs.empty:
        raise ValueError(f"every reference event's catalog hypocentre lies outside the "
                         f"travel-time grid ({skipped}): no station terms can be estimated")
    polish = {"iterations": scfg.polishIterations, "cap_s": scfg.referenceCapS,
              "min_events": scfg.minReferenceEvents}
    terms = polish_terms(res, **polish)
    folds = fold_of(pairs, scfg.folds)
    held_out = held_out_terms(res, folds, **polish)
    ref_assoc = set(pairs["assocId"])
    subset = _Subset(events=assoc.events[assoc.events["assocId"].astype(str).isin(ref_assoc)],
                     picks=assoc.picks[assoc.picks["assocId"].astype(str).isin(ref_assoc)])
    before = locate_detailed(subset, picks, stations, cfg, run, **kw)
    # In-sample comparison only (never written): the reference events with every term.
    in_sample = locate_detailed(subset, picks, stations, cfg, run, statics=statics_map(terms),
                                static_events=_counts(terms), **kw)
    details = locate_detailed(assoc, picks, stations, cfg, run, statics=statics_map(terms),
                              event_statics=held_out, static_events=_counts(terms), **kw)
    ref = pairs[["catalogId", "assocId"]].assign(fold=folds.reindex(pairs["assocId"]).to_numpy())
    ref = ref.merge(_offsets(before, pairs, "before"), on="assocId").merge(
        _offsets(details, pairs, "after"), on="assocId").merge(
        _offsets(in_sample, pairs, "inSample").drop(columns="inSampleEventId"), on="assocId")
    k_desc = "leave-one-out" if scfg.folds is None else f"{scfg.folds}-fold"
    report = _report(
        cfg, details, terms, mode=REFERENCE_EVENTS, pass_number=2,
        note=f"terms from {len(pairs)} reference events at the public regional catalog's "
        f"hypocentres; reference events relocated {k_desc} (held-out terms)",
        cap_s=scfg.referenceCapS, min_events=scfg.minReferenceEvents,
        sigma_ids=ref["afterEventId"].tolist(),
        sigma_events="reference events (held-out terms)",
        history=[_history_row(0, before, len(pairs), None),
                 _history_row(1, details, len(pairs), terms)],
        reference=ref, skipped=tuple(skipped), previous_median_rms_s=previous,
        extra={"foldScheme": k_desc, "nReferenceResiduals": len(res)},
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
    return StaticsOutcome(details, report)


def _counts(terms: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {(str(s), str(p)): int(n) for s, p, n in
            zip(terms["stationId"], terms["phase"], terms["nEvents"], strict=True)}

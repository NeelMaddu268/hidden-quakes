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
    held-out relocations' offsets from the catalog are the honest accuracy number, and absolute
    positions are then tied to the public regional catalog's frame.

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
    _process_cache_dir,
    event_picks,
    locate_detailed,
    used_stations,
)
from hq.locate.coords import to_enu
from hq.locate.locator import Locator, LocatorSetup, build_locator, weighted_median
from hq.locate.tt_grid import PHASES
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

    Columns ``catalogId, eventId, assocId, catalogT, catalogE, catalogN, catalogElevM``, sorted by
    catalog origin time. ``matches`` must have been written for ``events_located`` (every matched
    event present, ``t_event - t_catalog == dtS``), or this raises: rerun stage match first.
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
    })
    return out.sort_values(["catalogT", "catalogId"], kind="stable").reset_index(drop=True)


def catalog_residuals(
    locator: Locator, pairs: pd.DataFrame, frames: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, list[str]]:
    """``d = t_obs - T`` of each reference event's associated picks at the catalog hypocentre.

    Returns (``assocId, stationId, phase, d, w`` with ``w = prob / sigma``, the catalog ids left
    out because their hypocentre lies outside the travel-time grid).
    """
    grid = locator.tables.grid
    st_e = locator._e
    st_n = locator._n
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


def _trend_prediction(
    terms: pd.DataFrame, pos: pd.DataFrame, station: str, phase: str, min_stations: int
) -> float:
    """The plane fit (term ~ a + b e + c n) of the OTHER stations' non-zero terms of ``phase``,
    evaluated at ``station``; NaN with fewer than ``min_stations`` of them."""
    others = terms[(terms["phase"] == phase) & (terms["stationId"] != station)
                   & (terms["staticS"] != 0.0)]
    if len(others) < min_stations:
        return math.nan
    p = pos.loc[others["stationId"]]
    design = np.column_stack([np.ones(len(p)), p["enu_e"] / 1000.0, p["enu_n"] / 1000.0])
    coef, *_ = np.linalg.lstsq(design, others["staticS"].to_numpy(dtype=np.float64), rcond=None)
    here = pos.loc[station]
    return float(coef[0] + coef[1] * here["enu_e"] / 1000.0 + coef[2] * here["enu_n"] / 1000.0)


def explain_terms(
    terms: pd.DataFrame,
    stations: pd.DataFrame,
    model: LayerModel,
    flag_s: float,
    cfg: StaticsExplainConfig,
) -> pd.DataFrame:
    """One row per static with ``|staticS| > flag_s``: its evidence and a written explanation.

    Evidence: (1) the position trend of the other stations' terms of that phase (a plane over
    station ENU) predicted at this station: the same sign and at least ``lateralFraction`` of the
    term reads as lateral structure the 1D model can't hold (diagnostics row 7); (2) the station's
    S/P term ratio (when ``|P term| >= minRatioTermS``) against the model's Vp/Vs at the sensor:
    within a factor ``ratioBand`` of it reads as a velocity anomaly along the path near the
    station, within a factor ``ratioBand`` of 1 as equal delays (a timing offset is possible);
    (3) the sensor's elevation against the source model's own top (above it, the model's top
    layer is extended upward and stands in for whatever rock is there). ``verdict`` is the first
    of lateral / path / timing that holds, else unexplained.
    """
    columns = ["stationId", "phase", "staticS", "nEvents", "trendPredS", "spRatio", "modelVpVs",
               "sensorElevM", "aboveModelTopM", "verdict", "explanation"]
    pos = stations.set_index(stations["id"].astype(str))
    source_top = (model.top_extension.from_elev_m if model.top_extension is not None
                  else model.top_of_model_elev_m)
    by_key = {(str(s), str(p)): float(v) for s, p, v in
              zip(terms["stationId"], terms["phase"], terms["staticS"], strict=True)}
    rows = []
    for r in terms[terms["staticS"].abs() > flag_s].itertuples(index=False):
        sid, ph, term = str(r.stationId), str(r.phase), float(r.staticS)
        pred = _trend_prediction(terms, pos, sid, ph, cfg.minTrendStations)
        p_term, s_term = by_key.get((sid, "P"), 0.0), by_key.get((sid, "S"), 0.0)
        ratio = s_term / p_term if abs(p_term) >= cfg.minRatioTermS and s_term != 0.0 else math.nan
        elev = float(pos.loc[sid, "sensorElevM"])
        vpvs = float(model.vp_at(elev)) / float(model.vs_at(elev))
        lateral = (math.isfinite(pred) and np.sign(pred) == np.sign(term)
                   and abs(pred) >= cfg.lateralFraction * abs(term))
        path = math.isfinite(ratio) and vpvs / cfg.ratioBand <= ratio <= vpvs * cfg.ratioBand
        timing = math.isfinite(ratio) and 1.0 / cfg.ratioBand <= ratio <= cfg.ratioBand
        verdict = ("lateral" if lateral else "path" if path else "timing" if timing
                   else "unexplained")
        above = elev - source_top
        side = "early" if term < 0 else "late"
        text = [f"{ph} arrives {abs(term):.3f} s {side} against the 1D model (n {int(r.nEvents)})."]
        if math.isfinite(pred):
            text.append(
                f"The other stations' {ph} terms, fit as a plane over station position, predict "
                f"{pred:+.3f} s here: " + (
                    "the same sign and most of the term, so it is part of the regional trend: "
                    "lateral structure the 1D model can't hold (row 7)." if lateral else
                    "not enough of the term to call it the regional trend.")
            )
        else:
            text.append(f"Too few other {ph} terms (< {cfg.minTrendStations}) for a position trend.")
        if math.isfinite(ratio):
            text.append(
                f"S/P term ratio {ratio:.2f} against the model's Vp/Vs {vpvs:.2f} at the sensor: "
                + ("P and S delayed in proportion to the slownesses, a velocity anomaly along "
                   "the path near the station, not a clock offset." if path else
                   "equal P and S delays: a station timing offset or pick bias is possible; "
                   "check the station's timing." if timing else
                   "neither proportional to the slownesses nor equal.")
            )
        text.append(
            f"Sensor at {elev:.0f} m ASL, "
            + (f"{above:.0f} m above the velocity model's own top ({source_top:.0f} m ASL), where "
               "its top layer is extended upward and stands in for whatever rock is there."
               if above > 0 else f"{-above:.0f} m below the velocity model's own top.")
        )
        if verdict == "unexplained":
            text.append("No evidence here explains it: an unexplained static (depth gate).")
        rows.append({"stationId": sid, "phase": ph, "staticS": term, "nEvents": int(r.nEvents),
                     "trendPredS": pred, "spRatio": ratio, "modelVpVs": vpvs, "sensorElevM": elev,
                     "aboveModelTopM": above, "verdict": verdict, "explanation": " ".join(text)})
    return pd.DataFrame(rows, columns=columns)


def residual_sigma(
    arrivals: pd.DataFrame, event_ids: Sequence[str], cfg: SeismologyConfig
) -> pd.DataFrame:
    """Per phase: robust sigma (``GAUSS_MAD`` x MAD) of the used-pick residuals of ``event_ids``
    against ``locator.pickSigmaS``; ``wellAbove`` past ``statics.sigmaFlagRatio``."""
    use = arrivals[arrivals["usedInLocation"].to_numpy(dtype=bool)
                   & arrivals["eventId"].astype(str).isin(set(event_ids))]
    rows = []
    for ph in PHASES:
        r = use.loc[use["phase"].astype(str) == ph, "residualS"].to_numpy(dtype=np.float64)
        conf = float(getattr(cfg.locator.pickSigmaS, ph))
        sigma = GAUSS_MAD * float(np.median(np.abs(r - np.median(r)))) if r.size else math.nan
        ratio = sigma / conf
        rows.append({"phase": ph, "nPicks": int(r.size), "configuredS": conf,
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
    sigma: pd.DataFrame  # residual_sigma over the calibration events
    sigma_events: str  # which events the sigma rows are over
    history: pd.DataFrame  # per pass: iteration, medianRmsS, nCalibration, nNonZero, maxAbsS
    # referenceEvents pass 2: per reference event, catalogId, eventId (after), assocId, fold,
    # offsets before (no statics) and after (held-out terms): h/de/dn/dz (m), rmsS (s)
    reference: pd.DataFrame | None = None
    skipped: tuple[str, ...] = ()  # reference events outside the travel-time grid
    previous_median_rms_s: float | None = None  # stored no-statics events_located, all events
    extra: dict[str, Any] = field(default_factory=dict)

    def offsets_summary(self) -> dict[str, dict[str, float]]:
        """Median and p90 of the horizontal and |vertical| offsets, before and after."""
        ref = self.reference
        if ref is None or ref.empty:
            return {}
        out = {}
        for when in ("before", "after"):
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
    explained = explain_terms(terms, details.stations, details.locator.tables.model,
                              cfg.diagnostics.stationResidualFlagS, cfg.statics.explain)
    return StaticsReport(
        mode=mode, pass_number=pass_number, note=note, terms=terms, cap_s=cap_s,
        min_events=min_events, explanations=explained,
        sigma=residual_sigma(details.result.arrivals, sigma_ids, cfg), sigma_events=sigma_events,
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
) -> StaticsOutcome:
    """Locate with the configured statics (see the module docstring).

    ``reference``: ``reference_pairs`` output (mode referenceEvents; None runs pass 1).
    ``previous_events``: the run dir's events_located.parquet before this pass, whose median
    rmsS is reported as the no-statics value when it was located without statics.
    """
    started = time.perf_counter()
    scfg = cfg.statics
    kw: dict[str, Any] = {"run_id": run_id, "cache_dir": cache_dir, "model": model}
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
            sigma_events="all located events", history=[_history_row(0, base, 0, None)],
            previous_median_rms_s=previous)
        return StaticsOutcome(base, report)

    # Pass 2: terms at the catalog hypocentres, held out per fold for the reference events.
    setup = LocatorSetup(
        stations=used_stations(stations, run, cfg.locator.enuConsistencyTolM),
        model=load_configured_model(cfg.velocity) if model is None else model,
        config=cfg, run=run,
        cache_dir=Path(cache_dir) if cache_dir is not None else _process_cache_dir(),
    )
    locator = build_locator(setup)
    order, frames = event_picks(assoc, picks, cfg.locator.minPicks)
    res, skipped = catalog_residuals(locator, reference, dict(zip(order, frames, strict=True)))
    pairs = reference[~reference["catalogId"].isin(skipped)].reset_index(drop=True)
    polish = {"iterations": scfg.polishIterations, "cap_s": scfg.referenceCapS,
              "min_events": scfg.minReferenceEvents}
    terms = polish_terms(res, **polish)
    folds = fold_of(pairs, scfg.folds)
    held_out: dict[str, dict[tuple[str, str], float]] = {}
    for f in sorted(set(folds)):
        out = set(folds.index[folds == f])
        fold_terms = statics_map(polish_terms(res[~res["assocId"].isin(out)], **polish))
        held_out.update({aid: fold_terms for aid in out})
    ref_assoc = set(pairs["assocId"])
    subset = _Subset(events=assoc.events[assoc.events["assocId"].astype(str).isin(ref_assoc)],
                     picks=assoc.picks[assoc.picks["assocId"].astype(str).isin(ref_assoc)])
    before = locate_detailed(subset, picks, stations, cfg, run, **kw)
    details = locate_detailed(assoc, picks, stations, cfg, run, statics=statics_map(terms),
                              event_statics=held_out, static_events=_counts(terms), **kw)
    ref = pairs[["catalogId", "assocId"]].assign(fold=folds.reindex(pairs["assocId"]).to_numpy())
    ref = ref.merge(_offsets(before, pairs, "before"), on="assocId").merge(
        _offsets(details, pairs, "after"), on="assocId")
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
        b, a = summary["before"], summary["after"]
        log.info(
            "statics: referenceEvents pass 2 on %d reference events (%s): horizontal offset from "
            "the catalog median %.0f -> %.0f m, p90 %.0f -> %.0f m; |dz| median %.0f -> %.0f m; "
            "%d non-zero terms; %.1f s", len(pairs), k_desc, b["medianHM"], a["medianHM"],
            b["p90HM"], a["p90HM"], b["medianAbsDzM"], a["medianAbsDzM"],
            int(np.count_nonzero(terms["staticS"])), time.perf_counter() - started,
        )
    return StaticsOutcome(details, report)


def _counts(terms: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {(str(s), str(p)): int(n) for s, p, n in
            zip(terms["stationId"], terms["phase"], terms["nEvents"], strict=True)}

"""``diagnostics.md``: the lane doc's depth diagnostics (rows 1-7), each with a computed result.

Every number in the report comes from the run's own tables and locator; the thresholds come from
``seismology.yaml`` (``diagnostics``, ``locator``, ``matching``). A row that the data can't settle
says so and gives the counts. Rows (docs/lanes/H2-seismology.md, "Diagnostics for smeared depths"):

1. Borehole sensor depths: ``surfaceElevM``, ``sensorDepthM``, ``sensorElevM`` per station; a
   borehole at depth 0, a row where ``sensorElevM != surfaceElevM - sensorDepthM``, or a table
   not at ``sensorElevM`` fails the row.
2. Datum check: noise-free P and S picks from the exact 1D layered times at every used station
   for synthetic events at ``diagnostics.datumCheck`` (known elevM), located by the run's locator
   and tables; errors above ``passTolM`` fail the row. The check uses the run's own model, sensor
   elevations and tables, so it tests the locator's elevM bookkeeping only; the known public
   events' depth offsets (catalog comparison) are the independent datum evidence the row cites.
3. PyOcto elevM (association) vs relocated elevM, per event, joined through ``assocId``.
4. Formal ``vErrM`` and the elevM spread for ``nS >= minSForDepth`` vs ``nS < minSForDepth``.
5. Residuals split by ``Station.preprocessProfile``, per profile and phase: MAD (unscaled, as the
   outlier pass) over every associated pick at the final location (before the outlier pass
   removes any, so a noisy profile can't look clean by losing its worst picks), MAD over the used
   picks, and the fraction the outlier pass dropped.
6. Median residual per station and phase over used picks (input for LOC-05 statics).
7. Residual trend: least squares ``r = a + b cos(az) + c sin(az)`` per phase over used picks
   (az: station azimuth from the epicentre, grid north), the per-station medians against station
   position and ``sensorElevM``, and the outlier pass's drop fraction west vs east of the
   epicentre (input for LOC-07). With the public catalog in the run dir, the same fit with the
   hypocentre fixed at the catalog's; its S/P amplitude ratio against the model's Vp/Vs at the
   source depths separates an S-heavy (structural) trend from what a mislocated catalog
   epicentre alone would give. The test can't exclude the public catalog's own model and
   locations as a cause, and the conclusion says so.

Residuals are ``tObs - tPred`` (positive: the pick is later than the model predicts). Also
reported: the pick sigma against the observed residual spread (``hErrM`` / ``vErrM`` are formal
errors at that sigma, without model error), the synthetic recovery test when the stage ran it,
the travel-time tables' error against the exact layered times at the located hypocentres (the
shallow-interface bias LOC-02 recorded per layer), and a comparison with the public regional
catalog when the run dir holds it.

With ``locator.method`` grid3d (LOC-07) the report adds a 1D vs 3D section: the 3D model's frame
and air evidence, where each station sits in it (3D table or 1D fallback, and for a basin column
where the model's ground lies against the station's surfaceElevM), the 3D minus 1D table time
per station and phase at the final hypocentres, how many events lie above the 3D model's ground,
and against the same association located in the same stage run with grid1d and the same statics
(``DiagnosticsInputs.grid1d``): the depth and epicentre shift per event, rmsS, the row 7
azimuthal residual trend, both sets' offsets from the public regional catalog, the statics above
the flag and the per-station median residuals. Row 2's synthetic picks then come from the 3D
tables themselves (so does the synthetic test's forward model: no table error in it, unlike the
1D test), and the table-error section covers the 1D fallback stations.

When the pick harvest ran (LOC-10, ``LocateDetails.harvest``), a Pick harvest section reports
what it added, skipped and moved, its chance estimates, and the reference events' held-out
offsets with and without it; the header counts harvested picks apart from associated ones.
Without it the report is unchanged.
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig
from hq.locate.coords import origin_utm
from hq.locate.locator import GRID3D
from hq.locate.tt_grid import PHASES, layered_first_arrival
from hq.locate.tt_grid3d import ground_elev_m, nearest_columns

# H1's known-event windows in the run dir (stage locate reads the known ids from it).
KNOWN_WINDOWS_FILE = "known/windows.json"

if TYPE_CHECKING:
    from hq.locate import LocateDetails
    from hq.locate.statics import StaticsReport
    from hq.locate.synthetic import SyntheticResult

log = logging.getLogger(__name__)

HEADER = ("#", "Suspect", "Quick test", "Result (this run)", "Conclusion", "Fix")
SUSPECTS = {
    1: ("Borehole sensor depth ignored",
        "surfaceElevM, sensorDepthM, sensorElevM per station; a borehole at depth 0 is a bug",
        "Request H1 fix; use sensorElevM"),
    2: ("Datum mixing", "Locate a synthetic event at a known elevM and check the output",
        "Everything in elevM; convert only at display"),
    3: ("Coarse or homogeneous model used as the final location",
        "Compare PyOcto depth with relocated depth", "Relocate with the layered model"),
    4: ("Too few S picks, so depth trades off against origin time",
        "Depth spread for nS >= {k} vs nS < {k}", "Require S for Tier A"),
    5: ("Borehole picks degraded by resampling", "Residuals split by preprocessing profile",
        "Ask H1 to try borehole-B; per-profile sigma"),
    6: ("Station timing offsets or local sediment", "Median residual per station and phase",
        "Statics (LOC-05)"),
    7: ("1D can't represent the dipping basement", "Residuals trend with azimuth or position",
        "3D grids (LOC-07)"),
}


@dataclass(frozen=True)
class Row:
    number: int
    result: str
    conclusion: str


@dataclass(frozen=True, eq=False)
class Comparison1d:
    """grid3d runs: the same association located in the same stage run with grid1d and the same
    statics (mode, reference events, knobs): the 1D side of the 1D vs 3D section."""

    details: "LocateDetails"
    statics: "StaticsReport"
    runtime_s: float


@dataclass(frozen=True)
class DiagnosticsInputs:
    run_id: str
    run: RunSection
    cfg: SeismologyConfig
    stations: pd.DataFrame  # stations.parquet rows, all of them
    details: "LocateDetails"
    assoc_events: pd.DataFrame  # assoc_events (assocId, latitude, longitude, elevM)
    catalog: pd.DataFrame | None = None  # catalog.parquet rows when present
    catalog_errors: pd.DataFrame | None = None  # catalog_uncertainties() when catalog.quakeml is
    known_ids: tuple[str, ...] | None = None  # known/windows.json ids when present
    synthetic: "SyntheticResult | None" = None  # the stage's synthetic recovery test
    statics: "StaticsReport | None" = None  # the statics pass (LOC-05)
    grid1d: Comparison1d | None = None  # grid3d runs: the same events located with grid1d


def _cell(text: str) -> str:
    return " ".join(str(text).replace("|", "/").split())


def _f(value: float, fmt: str = ".0f") -> str:
    return "n/a" if value is None or not math.isfinite(float(value)) else format(float(value), fmt)


def _q(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else math.nan


def used_arrivals(details: "LocateDetails", stations: pd.DataFrame) -> pd.DataFrame:
    """Arrivals of picks used in a location, with event epicentre and station position/profile."""
    arr = details.result.arrivals
    arr = arr[arr["usedInLocation"].to_numpy(dtype=bool)]
    ev = details.result.events[["id", "enu_e", "enu_n", "elevM"]].rename(
        columns={"id": "eventId", "enu_e": "evE", "enu_n": "evN", "elevM": "evElevM"}
    )
    st = stations[["id", "enu_e", "enu_n", "sensorElevM", "preprocessProfile"]].rename(
        columns={"id": "stationId", "enu_e": "stE", "enu_n": "stN"}
    )
    st = st.assign(stationId=st["stationId"].astype(str),
                   preprocessProfile=st["preprocessProfile"].astype(str))
    out = arr.assign(eventId=arr["eventId"].astype(str), stationId=arr["stationId"].astype(str),
                     phase=arr["phase"].astype(str))
    out = out.merge(ev.assign(eventId=ev["eventId"].astype(str)), on="eventId", how="left")
    out = out.merge(st, on="stationId", how="left", validate="many_to_one")
    return out.reset_index(drop=True)


# --- row 1 ----------------------------------------------------------------------------------------


def row_borehole(inputs: DiagnosticsInputs) -> tuple[Row, list[str]]:
    st = inputs.stations
    tol = inputs.cfg.locator.enuConsistencyTolM
    kind = st["kind"].astype(str)
    depth = st["sensorDepthM"].to_numpy(dtype=np.float64)
    surface = st["surfaceElevM"].to_numpy(dtype=np.float64)
    sensor = st["sensorElevM"].to_numpy(dtype=np.float64)
    ids = st["id"].astype(str).to_numpy()
    bore = (kind == "borehole").to_numpy()
    zero = bore & (depth == 0.0)
    mismatch = np.abs(surface - depth - sensor) > tol
    receivers = inputs.details.locator.tables.receiver_elev_m
    used = inputs.details.stations
    off_table = [
        str(sid) for sid, z in zip(used["id"], used["sensorElevM"], strict=True)
        if receivers.get(str(sid)) != float(z)
    ]
    other_depth = ~bore & (depth > 0.0)
    result = (
        f"{len(st)} stations ({len(used)} used in the run): {int(bore.sum())} borehole sensors "
        f"with sensorDepthM {_f(depth[bore].min() if bore.any() else math.nan, 'g')} to "
        f"{_f(depth[bore].max() if bore.any() else math.nan, 'g')} m; {int(zero.sum())} borehole(s) "
        f"at depth 0; {int(mismatch.sum())} row(s) with sensorElevM != surfaceElevM - "
        f"sensorDepthM (tolerance {tol:g} m); travel-time tables at sensorElevM for "
        f"{len(used) - len(off_table)} of {len(used)} used stations; {int(other_depth.sum())} "
        f"non-borehole sensor(s) with sensorDepthM > 0 ({', '.join(ids[other_depth]) or 'none'}). "
        "Per-station values: appendix A."
    )
    bad = sorted(set(ids[zero]) | set(ids[mismatch]) | set(off_table))
    if bad:
        conclusion = (
            f"FAIL for {', '.join(bad)}: a borehole at depth 0, an inconsistent sensorElevM or a "
            "table off the sensor puts receivers at the wrong elevation. Request an H1 fix "
            "(docs/requests/H1.md) before trusting depths."
        )
    else:
        conclusion = (
            "Not the cause: every borehole sensor carries a non-zero channel depth, sensorElevM = "
            "surfaceElevM - sensorDepthM holds on every row, and every table puts its receiver at "
            "sensorElevM."
        )
    lines = [
        (
            "| id | kind | preprocessProfile | usedInRun | surfaceElevM | sensorDepthM | "
            "sensorElevM | note |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for k, sid in enumerate(ids):
        note = []
        if zero[k]:
            note.append("borehole at depth 0")
        if mismatch[k]:
            note.append("sensorElevM inconsistent")
        if sid in off_table:
            note.append("table not at sensorElevM")
        lines.append(
            f"| {sid} | {kind.iloc[k]} | {st['preprocessProfile'].astype(str).iloc[k]} | "
            f"{bool(st['usedInRun'].iloc[k])} | {surface[k]:.1f} | {depth[k]:g} | {sensor[k]:.1f} | "
            f"{'; '.join(note)} |"
        )
    return Row(1, result, conclusion), lines


# --- row 2 ----------------------------------------------------------------------------------------


def datum_check(inputs: DiagnosticsInputs) -> pd.DataFrame:
    """Noise-free synthetic events at the configured elevations, located by the run's locator."""
    dc = inputs.cfg.diagnostics.datumCheck
    locator = inputs.details.locator
    stations = inputs.details.stations
    top = locator.volume.top_elev_m
    if max(dc.elevM) > top:
        raise ValueError(
            f"diagnostics.datumCheck elevM {max(dc.elevM)} lies above the search volume top {top}"
        )
    t0 = inputs.run.window_start_s
    rows = []
    for z in dc.elevM:
        picks = []
        for sid, se, sn, zr in stations[["id", "enu_e", "enu_n", "sensorElevM"]].itertuples(
            index=False
        ):
            r = math.hypot(dc.eM - float(se), dc.nM - float(sn))
            for ph in PHASES:
                tt = (float(locator.station_times(str(sid), ph, dc.eM, dc.nM, z))
                      if locator.method == GRID3D else
                      float(layered_first_arrival(locator.tables.model, ph, float(zr), r, z)))
                picks.append({"id": f"datum-check:{sid}:{ph}", "stationId": str(sid),
                              "phase": ph, "t": t0 + tt, "prob": 1.0})
        loc = locator.locate(pd.DataFrame(picks))
        rows.append(
            {
                "trueElevM": z,
                "elevM": loc.elev_m,
                "dzM": loc.elev_m - z,
                "dhM": math.hypot(loc.e_m - dc.eM, loc.n_m - dc.nM),
                "dtS": loc.t0 - t0,
                "depthKm": (inputs.run.refSurfaceElevM - loc.elev_m) / 1000.0,
                "trueDepthKm": (inputs.run.refSurfaceElevM - z) / 1000.0,
            }
        )
    return pd.DataFrame(rows)


KNOWN_COLUMNS = ["catalogId", "dzM", "catalogZErrM", "distM", "catalogHErrM"]


def _known_depths(comp: pd.DataFrame | None) -> pd.DataFrame:
    """Known public events with a candidate: dz (ours minus catalog elevM) and the epicentral
    offset, each with the catalog's stated error."""
    if comp is None:
        return pd.DataFrame(columns=KNOWN_COLUMNS)
    return comp[comp["known"] & comp["eventId"].notna()][KNOWN_COLUMNS]


def row_datum(inputs: DiagnosticsInputs, comp: pd.DataFrame | None = None) -> Row:
    dc = inputs.cfg.diagnostics.datumCheck
    check = datum_check(inputs)
    worst_z = float(check["dzM"].abs().max())
    worst_h = float(check["dhM"].max())
    per = "; ".join(
        f"{r.trueElevM:g} m ASL -> {r.elevM:.0f} m (dz {r.dzM:+.0f} m, dh {r.dhM:.0f} m, dt "
        f"{r.dtS * 1000:+.1f} ms; depthKm {r.depthKm:.3f} vs {r.trueDepthKm:.3f})"
        for r in check.itertuples(index=False)
    )
    datum = ""
    if inputs.catalog is not None and len(inputs.catalog):
        labels = sorted(set(inputs.catalog["depthDatum"].astype(str)))
        datum = f" Public regional catalog depths were converted to elevM with: {'; '.join(labels)}."
    source = ("the 3D tables themselves (grid3d)" if inputs.details.locator.method == GRID3D
              else "the exact 1D layered times")
    result = (
        f"Noise-free P+S picks from {source} at {len(inputs.details.stations)} used stations for "
        f"synthetic events at (e {dc.eM:g}, n {dc.nM:g}) m: {per}. Largest abs(dz) "
        f"{worst_z:.0f} m, largest dh {worst_h:.0f} m (passTolM {dc.passTolM:g} m).{datum}"
    )
    known = _known_depths(comp)
    if len(known):
        z_err = known["catalogZErrM"].to_numpy(dtype=np.float64)
        within = np.abs(known["dzM"].to_numpy(dtype=np.float64)) <= z_err
        within_h = known["distM"].to_numpy(dtype=np.float64) <= known["catalogHErrM"].to_numpy(
            dtype=np.float64)
        evidence = (
            " Independent check: relocated minus public regional catalog elevM for the known "
            "events: " + ", ".join(
                f"{r.catalogId} {r.dzM:+.0f} m (catalog depth error {_f(r.catalogZErrM)} m)"
                for r in known.itertuples(index=False)
            ) + f"; {int(within.sum())} of {len(known)} within the catalog's stated depth error. "
            "Horizontally (not a datum test; the LOC-04 acceptance 'within their catalog "
            "uncertainty' has both halves): " + ", ".join(
                f"{r.catalogId} {r.distM:.0f} m (catalog horizontal error {_f(r.catalogHErrM)} m)"
                for r in known.itertuples(index=False)
            ) + f"; {int(within_h.sum())} of {len(known)} within the catalog's stated horizontal "
            "error."
        )
    elif not inputs.known_ids:
        evidence = (" No known public event was compared, so no independent datum check on this "
                    f"run: {KNOWN_WINDOWS_FILE} (H1's known-event windows) is not in the run dir.")
    else:
        evidence = (" No known public event was compared, so no independent datum check on this "
                    f"run: none of the known events in {KNOWN_WINDOWS_FILE} has a located "
                    "candidate within the matching tolerances.")
    if worst_z <= dc.passTolM and worst_h <= dc.passTolM:
        conclusion = (
            "Not the cause within what this check tests: the locator reports elevM consistently "
            "(synthetic picks from the run's own model, sensorElevM and tables come back at their "
            "elevM; depth appears only as depthKm = (refSurfaceElevM - elevM) / 1000). It can't "
            "detect a wrong datum in the layer file, in H1's sensor elevations or in the catalog "
            "depths." + evidence
        )
    else:
        conclusion = (
            f"FAIL: a noise-free event comes back {worst_z:.0f} m off in elevM or {worst_h:.0f} m "
            "off horizontally; look for datum mixing (depth vs elevM, wellhead vs sensor) before "
            "trusting any depth."
        )
    return Row(2, result, conclusion)


# --- row 3 ----------------------------------------------------------------------------------------


def pyocto_shift(inputs: DiagnosticsInputs) -> pd.DataFrame:
    """Per located event: relocated minus PyOcto (association) position, joined via assocId."""
    from hq.locate.coords import to_enu

    flags = inputs.details.flags[["eventId", "assocId"]].astype(str)
    ev = inputs.details.result.events
    assoc = inputs.assoc_events.assign(assocId=inputs.assoc_events["assocId"].astype(str))
    joined = flags.merge(assoc[["assocId", "latitude", "longitude", "elevM"]], on="assocId",
                         how="left", validate="one_to_one")
    if joined["elevM"].isna().any():
        raise ValueError("located events whose assocId is missing from assoc_events")
    pe, pn, _ = to_enu(joined["latitude"].to_numpy(dtype=np.float64),
                       joined["longitude"].to_numpy(dtype=np.float64),
                       joined["elevM"].to_numpy(dtype=np.float64), inputs.run.origin)
    loc = ev.set_index(ev["id"].astype(str)).loc[joined["eventId"]]
    return pd.DataFrame(
        {
            "eventId": joined["eventId"].to_numpy(),
            "dzM": loc["elevM"].to_numpy(dtype=np.float64) - joined["elevM"].to_numpy(dtype=float),
            "deM": loc["enu_e"].to_numpy(dtype=np.float64) - pe,
            "dnM": loc["enu_n"].to_numpy(dtype=np.float64) - pn,
        }
    )


def row_pyocto(inputs: DiagnosticsInputs) -> Row:
    shift = pyocto_shift(inputs)
    n = len(shift)
    n_assoc = inputs.details.counts["assocEvents"]
    if n == 0:
        return Row(3, f"No located events (0 of {n_assoc} association events).",
                   "Can't conclude on this data: nothing was located to compare.")
    dz = shift["dzM"].to_numpy()
    dh = np.hypot(shift["deM"], shift["dnM"]).to_numpy()
    result = (
        f"{n} events: relocated minus PyOcto elevM median {np.median(dz):+.0f} m (p10 "
        f"{_q(dz, 0.1):+.0f}, p90 {_q(dz, 0.9):+.0f} m), median abs(dz) {np.median(np.abs(dz)):.0f} m; "
        f"epicentre shift median {np.median(dh):.0f} m (p90 {_q(dh, 0.9):.0f} m), median de "
        f"{np.median(shift['deM']):+.0f} m, dn {np.median(shift['dnM']):+.0f} m."
    )
    conclusion = (
        f"Not the cause: the final locations are the {inputs.details.locator.method} "
        f"relocation ({inputs.cfg.locator.fineSpacingM:g} m fine grid, PDF errors); PyOcto's "
        f"locations (the 1D layer model, {inputs.cfg.associator.minNodeSizeLocationKm:g} km "
        "location nodes) only seed association. The shift above is what keeping PyOcto's "
        "positions would have added."
    )
    return Row(3, result, conclusion)


# --- row 4 ----------------------------------------------------------------------------------------


def row_ns(inputs: DiagnosticsInputs) -> Row:
    dcfg = inputs.cfg.diagnostics
    k = dcfg.minSForDepth
    ev = inputs.details.result.events
    flags = inputs.details.flags
    n_s = ev["quality_nS"].to_numpy(dtype=np.int64)
    groups = {f"nS >= {k}": n_s >= k, f"nS < {k}": n_s < k}
    stats = {}
    parts = []
    for label, mask in groups.items():
        verr = ev.loc[mask, "quality_vErrM"].to_numpy(dtype=np.float64)
        verr = verr[np.isfinite(verr)]
        elev = ev.loc[mask, "elevM"].to_numpy(dtype=np.float64)
        med = float(np.median(verr)) if verr.size else math.nan
        stats[label] = (int(mask.sum()), med)
        parts.append(
            f"{label}: {int(mask.sum())} events, median vErrM {_f(med)} m ({int(verr.size)} with "
            f"formal errors), elevM IQR {_f(_q(elev, 0.75) - _q(elev, 0.25))} m, depthOnEdge "
            f"{int(ev.loc[mask, 'quality_depthOnEdge'].sum())}, MAP on volume top "
            f"{int(flags.loc[mask, 'mapOnVolumeTop'].sum())}"
        )
    result = "; ".join(parts) + "."
    (n_hi, med_hi), (n_lo, med_lo) = stats.values()
    if min(n_hi, n_lo) < dcfg.minGroupSize or not (math.isfinite(med_hi) and math.isfinite(med_lo)):
        conclusion = (
            f"Can't conclude on this data: {n_hi} events with nS >= {k} and {n_lo} with nS < {k}; "
            f"each group needs at least minGroupSize {dcfg.minGroupSize} events with formal errors."
        )
    else:
        ratio = med_lo / med_hi
        if ratio > dcfg.degradationRatio:
            conclusion = (
                f"Too few S picks widen depth here: median vErrM is {ratio:.1f}x larger with nS < "
                f"{k}. Require nS >= {k} for Tier A (LOC-06)."
            )
        else:
            conclusion = (
                f"No sign of this suspect: median vErrM with nS < {k} is {ratio:.2f}x that with nS "
                f">= {k} (degraded above {dcfg.degradationRatio:g}x)."
            )
    return Row(4, result, conclusion)


# --- rows 5 and 6 ---------------------------------------------------------------------------------


def _mad(values: np.ndarray) -> float:
    return float(np.median(np.abs(values - np.median(values)))) if values.size else math.nan


def residual_stats(ua: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """n, median, MAD (unscaled) and rms of residualS per group."""
    rows = []
    for key, g in ua.groupby(by, sort=True):
        r = g["residualS"].to_numpy(dtype=np.float64)
        keys = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, keys, strict=True)), "n": int(r.size),
                     "medianS": float(np.median(r)), "madS": _mad(r),
                     "rmsS": float(np.sqrt(np.mean(r**2)))})
    return pd.DataFrame(rows, columns=[*by, "n", "medianS", "madS", "rmsS"])


def picked_arrivals(details: "LocateDetails", stations: pd.DataFrame) -> pd.DataFrame:
    """Every arrival with a pick (used or outlier-dropped), with station profile and position.

    Residuals are at the final location, so the dropped picks keep the residual that got them
    dropped.
    """
    arr = details.result.arrivals
    arr = arr[arr["pickId"].notna().to_numpy(dtype=bool)]
    ev = details.result.events[["id", "enu_e"]].rename(columns={"id": "eventId", "enu_e": "evE"})
    st = stations[["id", "enu_e", "preprocessProfile"]].rename(
        columns={"id": "stationId", "enu_e": "stE"}
    )
    st = st.assign(stationId=st["stationId"].astype(str),
                   preprocessProfile=st["preprocessProfile"].astype(str))
    out = arr.assign(eventId=arr["eventId"].astype(str), stationId=arr["stationId"].astype(str),
                     phase=arr["phase"].astype(str),
                     used=arr["usedInLocation"].to_numpy(dtype=bool))
    out = out.merge(ev.assign(eventId=ev["eventId"].astype(str)), on="eventId", how="left")
    out = out.merge(st, on="stationId", how="left", validate="many_to_one")
    return out.reset_index(drop=True)


def profile_stats(pa: pd.DataFrame) -> pd.DataFrame:
    """Per phase and profile: picks, used picks, drop fraction, MAD over all and over used."""
    rows = []
    for (phase, profile), g in pa.groupby(["phase", "preprocessProfile"], sort=True):
        r = g["residualS"].to_numpy(dtype=np.float64)
        used = g["used"].to_numpy(dtype=bool)
        rows.append({"phase": phase, "preprocessProfile": profile, "nAll": int(r.size),
                     "nUsed": int(used.sum()), "dropFraction": 1.0 - used.sum() / r.size,
                     "madAllS": _mad(r), "madUsedS": _mad(r[used]),
                     "medianUsedS": float(np.median(r[used])) if used.any() else math.nan})
    return pd.DataFrame(rows, columns=["phase", "preprocessProfile", "nAll", "nUsed",
                                       "dropFraction", "madAllS", "madUsedS", "medianUsedS"])


def row_profiles(inputs: DiagnosticsInputs, pa: pd.DataFrame) -> Row:
    dcfg = inputs.cfg.diagnostics
    stats = profile_stats(pa)
    if stats.empty:
        return Row(5, "No associated picks.", "Can't conclude on this data: nothing was located.")
    result = "; ".join(
        f"{r.phase} {r.preprocessProfile}: {r.nUsed} of {r.nAll} picks used "
        f"({r.dropFraction:.0%} dropped by the outlier pass), MAD {r.madAllS:.3f} s over all, "
        f"{_f(r.madUsedS, '.3f')} s over used, median used {_f(r.medianUsedS, '+.3f')} s"
        for r in stats.itertuples(index=False)
    ) + (". MAD is unscaled (as the outlier pass); 'all' is every associated pick at the final "
         "location, before the outlier pass drops any.")
    notes = []
    degraded = []
    for phase, g in stats.groupby("phase", sort=True):
        ok = g[g["nAll"] >= dcfg.minGroupSize]
        if len(ok) < 2:
            notes.append(
                f"{phase}: only {len(ok)} profile(s) with >= minGroupSize {dcfg.minGroupSize} picks "
                f"({', '.join(f'{p} n {n}' for p, n in zip(g['preprocessProfile'], g['nAll'], strict=True))})"
            )
            continue
        best = ok.loc[ok["madAllS"].idxmin()]
        for r in ok.itertuples(index=False):
            if best["madAllS"] > 0 and r.madAllS > dcfg.degradationRatio * float(best["madAllS"]):
                text = (f"{phase} {r.preprocessProfile} MAD {r.madAllS:.3f} s = "
                        f"{r.madAllS / float(best['madAllS']):.1f}x {best['preprocessProfile']} "
                        f"({r.dropFraction:.0%} vs {float(best['dropFraction']):.0%} dropped)")
                degraded.append((str(r.preprocessProfile), text))
    prefix = dcfg.boreholeProfilePrefix
    bore = sorted({p for p in stats["preprocessProfile"] if p.startswith(prefix)})
    bore_bad = sorted({p for p, _ in degraded if p.startswith(prefix)})
    scope = (f"borehole profiles (preprocessProfile {prefix}*): {', '.join(bore) or 'none'}")
    if degraded:
        verdict = (
            f"borehole picks degraded ({', '.join(bore_bad)})" if bore_bad else
            f"no borehole profile among them ({scope}), so this suspect does not explain it; "
            "rows 6 and 7 look at station and lateral terms"
        )
        conclusion = (
            f"Degraded: {'; '.join(t for _, t in degraded)} (MAD over all associated picks, so "
            f"the outlier pass can't hide it); {verdict}. Set per-profile sigma "
            "(locator.profilePickSigmaS) for those profiles"
            + ("; ask H1 whether borehole-B picks those stations better." if bore_bad else ".")
        )
    elif notes and len(notes) == stats["phase"].nunique():
        conclusion = "Can't conclude on this data: " + "; ".join(notes) + "."
    else:
        conclusion = (
            f"No profile's MAD over all its associated picks exceeds {dcfg.degradationRatio:g}x "
            f"the best profile's for the same phase ({scope}): resampling does not visibly degrade "
            "borehole picks at this level" + (f" ({'; '.join(notes)})" if notes else "") + "."
        )
    return Row(5, result, conclusion)


def station_medians(inputs: DiagnosticsInputs, ua: pd.DataFrame) -> pd.DataFrame:
    stats = residual_stats(ua, ["stationId", "phase"])
    elev = inputs.details.stations.set_index("id")["sensorElevM"]
    stats["sensorElevM"] = elev.reindex(stats["stationId"]).to_numpy(dtype=np.float64)
    stats["enough"] = stats["n"] >= inputs.cfg.diagnostics.minGroupSize
    stats["flagged"] = stats["enough"] & (
        stats["medianS"].abs() > inputs.cfg.diagnostics.stationResidualFlagS
    )
    return stats


def row_stations(inputs: DiagnosticsInputs, ua: pd.DataFrame) -> tuple[Row, list[str]]:
    dcfg = inputs.cfg.diagnostics
    stats = station_medians(inputs, ua)
    ok = stats[stats["enough"]] if len(stats) else stats
    total = len(inputs.details.stations) * len(PHASES)
    lines = [
        "| stationId | phase | n used picks | median residual (s) | MAD (s) | sensorElevM | flag |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in stats.itertuples(index=False):
        flag = f"> {dcfg.stationResidualFlagS:g} s" if r.flagged else (
            "" if r.enough else f"< {dcfg.minGroupSize} picks")
        lines.append(f"| {r.stationId} | {r.phase} | {r.n} | {r.medianS:+.3f} | {r.madS:.3f} | "
                     f"{r.sensorElevM:.0f} | {flag} |")
    if ok.empty:
        return Row(6, f"No station-phase has >= minGroupSize {dcfg.minGroupSize} used picks "
                   f"({len(stats)} of {total} have any).",
                   "Can't conclude on this data: too few located events per station."), lines
    worst = ok.loc[ok["medianS"].abs().idxmax()]
    flagged = ok[ok["flagged"]]
    listed = ", ".join(f"{r.stationId} {r.phase} {r.medianS:+.3f} s (n {r.n})"
                       for r in flagged.itertuples(index=False))
    result = (
        f"{len(ok)} of {total} station-phases have >= {dcfg.minGroupSize} used picks; "
        f"abs(median) > {dcfg.stationResidualFlagS:g} s at {len(flagged)}"
        + (f": {listed}" if listed else "")
        + f". Largest abs(median) {worst['medianS']:+.3f} s at {worst['stationId']} {worst['phase']}; "
        f"P medians span {_f(ok.loc[ok.phase == 'P', 'medianS'].min(), '+.3f')} to "
        f"{_f(ok.loc[ok.phase == 'P', 'medianS'].max(), '+.3f')} s, S "
        f"{_f(ok.loc[ok.phase == 'S', 'medianS'].min(), '+.3f')} to "
        f"{_f(ok.loc[ok.phase == 'S', 'medianS'].max(), '+.3f')} s. All station-phases: appendix B."
    )
    rep = inputs.statics
    if rep is not None and rep.pass_number == 2:
        conclusion = (
            f"Residuals after statics: {len(flagged)} station-phase median(s) still exceed "
            f"{dcfg.stationResidualFlagS:g} s. The statics themselves, and a written "
            "explanation of each above that, are in the Station statics section."
        )
    elif len(flagged):
        conclusion = (
            f"Station terms matter: {len(flagged)} station-phase median(s) exceed "
            f"{dcfg.stationResidualFlagS:g} s. LOC-05 statics absorb them, and each static above "
            f"{dcfg.stationResidualFlagS:g} s needs a written explanation (sensor elevation above "
            "the model's own top, local sediment, timing)."
        )
    else:
        conclusion = (
            f"No station-phase median exceeds {dcfg.stationResidualFlagS:g} s: LOC-05 statics stay "
            "under the explanation threshold; the offsets above are its starting point."
        )
    return Row(6, result, conclusion), lines


# --- row 7 ----------------------------------------------------------------------------------------


def azimuth_fit(ua: pd.DataFrame) -> pd.DataFrame:
    """Per phase: least squares residual = a + b cos(az) + c sin(az); amplitude and late azimuth."""
    rows = []
    for phase, g in ua.groupby("phase", sort=True):
        az = np.arctan2(g["stE"] - g["evE"], g["stN"] - g["evN"]).to_numpy(dtype=np.float64)
        r = g["residualS"].to_numpy(dtype=np.float64)
        amp = late = math.nan
        if r.size >= 3:
            design = np.column_stack([np.ones_like(az), np.cos(az), np.sin(az)])
            (_, b, c), *_ = np.linalg.lstsq(design, r, rcond=None)
            amp = float(math.hypot(b, c))
            late = float(math.degrees(math.atan2(c, b)) % 360.0)
        rows.append({"phase": phase, "n": int(r.size), "amplitudeS": amp, "lateAzDeg": late})
    return pd.DataFrame(rows, columns=["phase", "n", "amplitudeS", "lateAzDeg"])


def position_fit(medians: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Per phase: station medians ~ a + ge * e_km + gn * n_km, and r(median, sensorElevM)."""
    pos = stations.set_index("id")[["enu_e", "enu_n", "sensorElevM"]]
    rows = []
    for phase, g in medians[medians["enough"]].groupby("phase", sort=True):
        p = pos.reindex(g["stationId"])
        m = g["medianS"].to_numpy(dtype=np.float64)
        grad = gaz = corr = math.nan
        if m.size >= 4:
            design = np.column_stack([np.ones(m.size), p["enu_e"] / 1000.0, p["enu_n"] / 1000.0])
            (_, ge, gn), *_ = np.linalg.lstsq(design, m, rcond=None)
            grad = float(math.hypot(ge, gn))
            gaz = float(math.degrees(math.atan2(ge, gn)) % 360.0)
            z = p["sensorElevM"].to_numpy(dtype=np.float64)
            if np.std(z) > 0 and np.std(m) > 0:
                corr = float(np.corrcoef(m, z)[0, 1])
        rows.append({"phase": phase, "nStations": int(m.size), "gradientSPerKm": grad,
                     "gradientAzDeg": gaz, "corrWithSensorElevM": corr})
    return pd.DataFrame(rows, columns=["phase", "nStations", "gradientSPerKm", "gradientAzDeg",
                                       "corrWithSensorElevM"])


def _fit_text(az: pd.DataFrame) -> str:
    return "; ".join(
        f"{r.phase} amplitude {_f(r.amplitudeS, '.3f')} s, latest toward az {_f(r.lateAzDeg)} deg "
        f"(n {r.n})" for r in az.itertuples(index=False)
    )


def _flagged(az: pd.DataFrame, min_n: int, flag_s: float) -> pd.DataFrame:
    ok = az[(az["n"] >= min_n) & np.isfinite(az["amplitudeS"])]
    return ok[ok["amplitudeS"] > flag_s]


def side_drop_text(pa: pd.DataFrame) -> str:
    """Outlier-pass drop fraction per phase for stations west vs east of the epicentre and of
    the run origin (the two splits differ when the epicentres sit off the origin)."""
    if pa.empty:
        return ""
    splits = []
    for split, east_of in (("the epicentre", pa["evE"]), ("the run origin", 0.0)):
        bits = []
        for phase in sorted(pa["phase"].unique()):
            g = pa["phase"] == phase
            east = (pa["stE"] > east_of).to_numpy(dtype=bool)
            used = pa["used"].to_numpy(dtype=bool)
            parts = []
            for label, mask in (("west", g.to_numpy() & ~east), ("east", g.to_numpy() & east)):
                n = int(mask.sum())
                parts.append(f"{label} {1.0 - used[mask].sum() / n:.0%} of {n}" if n else
                             f"{label} none")
            bits.append(f"{phase} " + ", ".join(parts))
        splits.append(f"of {split}: " + "; ".join(bits))
    return "; outlier pass drop fraction by station side " + " / ".join(splits)


def sp_ratio(inputs: DiagnosticsInputs, cat_az: pd.DataFrame,
             at_catalog: pd.DataFrame) -> tuple[float, float, float]:
    """(S/P trend amplitude ratio, model Vp/Vs at the median catalog source elevM, that elevM)."""
    amp = cat_az.set_index("phase")["amplitudeS"]
    if not {"P", "S"} <= set(amp.index) or not (amp["P"] > 0):
        return math.nan, math.nan, math.nan
    z = float(np.median(at_catalog["evElevM"].to_numpy(dtype=np.float64)))
    t3 = inputs.details.locator.tables3d
    if t3 is None:
        model = inputs.details.locator.tables.model
        vpvs = float(model.vp_at(z)) / float(model.vs_at(z))
    else:  # the 3D model's Vp/Vs at each compared catalog hypocentre (nearest node), median
        hyp = at_catalog.drop_duplicates("catalogId")
        i, j = nearest_columns(t3.source, hyp["evE"], hyp["evN"], origin_utm(inputs.run.origin))
        vp, vs = t3.source.columns(i, j)
        k = np.clip(np.rint((hyp["evElevM"].to_numpy(dtype=np.float64) - t3.source.elev_m[0])
                            / t3.source.spacing_m[2]).astype(np.intp), 0, vp.shape[1] - 1)
        rows = np.arange(len(hyp))
        vpvs = float(np.median(vp[rows, k] / vs[rows, k]))
    return float(amp["S"] / amp["P"]), vpvs, z


def _model_label(inputs: DiagnosticsInputs) -> str:
    """The travel-time model this run located with, for the report's wording."""
    return "the 3D model" if inputs.details.locator.method == GRID3D else "the 1D model"


def row_trend(
    inputs: DiagnosticsInputs,
    ua: pd.DataFrame,
    pa: pd.DataFrame,
    at_catalog: pd.DataFrame | None,
    comp: pd.DataFrame | None,
    before_statics: pd.DataFrame | None = None,
) -> Row:
    """Row 7. ``before_statics``: ``at_catalog`` with the statics removed (pass 2 only)."""
    dcfg = inputs.cfg.diagnostics
    rep = inputs.statics
    after_statics = rep is not None and rep.pass_number == 2
    grid3d = inputs.details.locator.method == GRID3D
    az = azimuth_fit(ua)
    pos = position_fit(station_medians(inputs, ua), inputs.details.stations)
    if az.empty:
        return Row(7, "No used picks.", "Can't conclude on this data: nothing was located.")
    result = "At our locations, residual vs azimuth: " + _fit_text(az)
    if len(pos):
        result += "; " + "; ".join(
            f"{r.phase} station medians vs position ({r.nStations} stations): gradient "
            f"{_f(r.gradientSPerKm * 1000, '.1f')} ms/km, increasing toward az "
            f"{_f(r.gradientAzDeg)} deg, r(median, sensorElevM) {_f(r.corrWithSensorElevM, '+.2f')}"
            for r in pos.itertuples(index=False)
        )
    result += side_drop_text(pa)
    cat_az = None
    ratio = vpvs = src_z = math.nan
    if at_catalog is not None and len(at_catalog):
        cat_az = azimuth_fit(at_catalog)
        corr = {
            ph: float(np.corrcoef(g["residualS"], g["sensorElevM"])[0, 1]) if len(g) > 2 else math.nan
            for ph, g in at_catalog.groupby("phase", sort=True)
        }
        ratio, vpvs, src_z = sp_ratio(inputs, cat_az, at_catalog)
        result += (
            f". With the hypocentre fixed at the public regional catalog's for the "
            f"{at_catalog['catalogId'].nunique()} compared event(s) (origin time by weighted "
            "median): " + _fit_text(cat_az) + "; r(residual, sensorElevM) "
            + ", ".join(f"{ph} {_f(v, '+.2f')}" for ph, v in corr.items())
            + f"; S/P amplitude ratio {_f(ratio, '.2f')} vs the model's Vp/Vs "
            f"{_f(vpvs, '.2f')} at the catalog's median source elevM {_f(src_z)} m"
        )
    result += ". Azimuths are grid (UTM) azimuths from the epicentre to the station."
    ours = _flagged(az, dcfg.minGroupSize, dcfg.trendFlagS)
    cat = (_flagged(cat_az, dcfg.minGroupSize, dcfg.trendFlagS) if cat_az is not None
           else cat_az)
    if cat is not None and len(cat):
        listed = ", ".join(f"{r.phase} {r.amplitudeS:.3f} s, later toward az {r.lateAzDeg:.0f} and "
                           f"earlier toward az {(r.lateAzDeg + 180) % 360:.0f} deg"
                           for r in cat.itertuples(index=False))
        shift = ""
        have = comp[comp["eventId"].notna()] if comp is not None else None
        if have is not None and len(have):
            de, dn = float(have["deM"].median()), float(have["dnM"].median())
            shift_az = math.degrees(math.atan2(de, dn)) % 360.0
            early = (float(cat.loc[cat["amplitudeS"].idxmax(), "lateAzDeg"]) + 180.0) % 360.0
            side = "the early side" if abs((shift_az - early + 180) % 360 - 180) <= 45 else (
                "not the early side")
            shift = (f" Our candidates sit a median {math.hypot(de, dn):.0f} m from the catalog's "
                     f"epicentres toward az {shift_az:.0f} deg, {side}.")
        alternative = (
            "The test fixes the public catalog's hypocentres, so the catalog's own velocity model "
            "and locations remain a possible cause it can't exclude."
        )
        if math.isfinite(ratio) and ratio > vpvs * dcfg.trendSPRatioExcess:
            verdict = (
                f"Consistent with lateral structure {_model_label(inputs)} can't hold "
                f"(near-surface or deeper): with the hypocentre at the public regional catalog's, residuals trend "
                f"with azimuth ({listed}; above trendFlagS {dcfg.trendFlagS:g} s), and the trend "
                f"is S-heavy: S/P amplitude ratio {ratio:.2f}, while a mislocated catalog "
                f"epicentre alone would give about the model's Vp/Vs {vpvs:.2f} at the source "
                f"depths (S-heavy above {dcfg.trendSPRatioExcess:g}x that)."
            )
        else:
            verdict = (
                f"Lateral structure or the public catalog's own locations: with the hypocentre at "
                f"the catalog's, residuals trend with azimuth ({listed}; above trendFlagS "
                f"{dcfg.trendFlagS:g} s), but the S/P amplitude ratio {_f(ratio, '.2f')} is not "
                f"above {dcfg.trendSPRatioExcess:g}x the model's Vp/Vs {_f(vpvs, '.2f')}, which is "
                "what a mislocated catalog epicentre alone would give; this test can't separate "
                "the two."
            )
        conclusion = (
            f"{verdict}{shift} {alternative} Fix: "
            + ("station terms fixed at reference hypocentres (LOC-05); this run already locates "
               "with the 3D model (LOC-07)." if grid3d else
               "3D grids (LOC-07), or station terms fixed at reference hypocentres (LOC-05).")
        )
    elif len(ours):
        listed = ", ".join(f"{r.phase} {r.amplitudeS:.3f} s toward az {r.lateAzDeg:.0f} deg"
                           for r in ours.itertuples(index=False))
        conclusion = (
            f"Residuals trend with azimuth ({listed}, above trendFlagS {dcfg.trendFlagS:g} s): "
            + ("either the 3D model misses lateral structure" if grid3d else
               "either the 1D model misses lateral structure (dipping basement; LOC-07 3D grids)")
            + " or station terms line up with azimuth (LOC-05 statics first, then recheck)."
        )
    elif az["n"].max() < dcfg.minGroupSize:
        conclusion = (f"Can't conclude on this data: fewer than minGroupSize {dcfg.minGroupSize} "
                      "used picks per phase.")
    elif after_statics:
        conclusion = (
            f"No azimuthal trend above trendFlagS {dcfg.trendFlagS:g} s after statics (largest "
            f"{float(az['amplitudeS'].max()):.3f} s)."
        )
    else:
        conclusion = (
            f"No azimuthal trend above trendFlagS {dcfg.trendFlagS:g} s (largest "
            f"{float(az['amplitudeS'].max()):.3f} s)"
            + ("." if grid3d else ": nothing here asks for 3D grids (LOC-07) yet.")
        )
    if after_statics:
        by_construction = (
            " The reference terms were fitted at the public regional catalog's hypocentres (each "
            "reference event's from the other reference events at theirs), so a flat trend at "
            "those hypocentres after statics is expected by construction, not evidence that "
            f"{_model_label(inputs)} holds." if rep.reference is not None else "")
        before = ""
        if before_statics is not None and len(before_statics):
            b_az = azimuth_fit(before_statics)
            b_flag = _flagged(b_az, dcfg.minGroupSize, dcfg.trendFlagS)
            before = (
                " Before statics (the same picks with the statics removed), at the catalog's "
                "hypocentres: " + _fit_text(b_az) + (
                    f"; above trendFlagS {dcfg.trendFlagS:g} s for "
                    + ", ".join(str(p) for p in b_flag["phase"]) if len(b_flag)
                    else f"; none above trendFlagS {dcfg.trendFlagS:g} s") + ".")
        conclusion = (
            "Residuals here are after statics, which absorb a per-station delay: "
            + conclusion + by_construction + before + " The station terms themselves carry any "
            "lateral structure (Station statics section: each term above the flag is tested "
            "against its nearest stations), so this row can no longer show it"
            + ("." if grid3d else ", and 3D grids (LOC-07) remain the model-side fix.")
        )
    return Row(7, result, conclusion)


def catalog_hypocentre_residuals(
    inputs: DiagnosticsInputs, comp: pd.DataFrame, *, with_statics: bool = True
) -> pd.DataFrame:
    """Residuals of each compared candidate's picks with the hypocentre fixed at the catalog's.

    The origin time is the locator's weighted median (weights prob / sigma, statics as applied,
    or none with ``with_statics`` false) over every associated pick of the candidate; then the
    locator's outlier rule is applied at that hypocentre (``|residual| > max(madK * MAD,
    floorS)`` left out), so a pick dropped only because our location moved is kept. Public
    events outside the travel-time grid are left out.
    """
    from hq.locate.locator import weighted_median

    columns = ["catalogId", "eventId", "stationId", "phase", "residualS", "evE", "evN",
               "evElevM", "stE", "stN", "sensorElevM"]
    if inputs.catalog is None or comp.empty:
        return pd.DataFrame(columns=columns)
    locator = inputs.details.locator
    outlier = inputs.cfg.locator.outlier
    by_event = dict(zip(inputs.details.result.events["id"].astype(str), inputs.details.locations,
                        strict=True))
    cat = inputs.catalog.set_index(inputs.catalog["id"].astype(str))
    st = inputs.details.stations.set_index("id")
    parts = []
    for r in comp[comp["eventId"].notna()].itertuples(index=False):
        c = cat.loc[r.catalogId]
        e, n, z = float(c["enu_e"]), float(c["enu_n"]), float(c["elevM"])
        if not locator.covers(e, n, z):
            continue
        a = by_event[str(r.eventId)].arrivals
        tt = locator.travel_times(e, n, z).set_index(["stationId", "phase"])["travelTimeS"]
        keys = list(zip(a["stationId"].astype(str), a["phase"].astype(str), strict=True))
        d = a["tObs"].to_numpy(dtype=np.float64) - tt.loc[keys].to_numpy()
        if with_statics:
            d = d - a["staticS"].to_numpy(dtype=np.float64)
        res = d - weighted_median(d, a["weight"].to_numpy(dtype=np.float64))
        keep = np.abs(res) <= max(outlier.madK * _mad(res), outlier.floorS)
        sid = a["stationId"].astype(str)[keep]
        parts.append(pd.DataFrame({
            "catalogId": r.catalogId, "eventId": r.eventId, "stationId": sid.to_numpy(),
            "phase": a["phase"].astype(str).to_numpy()[keep], "residualS": res[keep], "evE": e,
            "evN": n, "evElevM": z, "stE": st.loc[sid, "enu_e"].to_numpy(),
            "stN": st.loc[sid, "enu_n"].to_numpy(),
            "sensorElevM": st.loc[sid, "sensorElevM"].to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


# --- pick sigma vs residual spread, synthetic test -----------------------------------------------


def sigma_section(inputs: DiagnosticsInputs, pa: pd.DataFrame) -> list[str]:
    """Configured pick sigma next to the observed residual spread; hErrM/vErrM are formal."""
    lcfg = inputs.cfg.locator
    lines = ["## Pick sigma vs observed residual spread (hErrM / vErrM are formal errors)", ""]
    if pa.empty:
        lines.append("No associated picks: no residual spread to compare with the pick sigma.")
        return lines
    parts = []
    above = []
    for phase, g in pa.groupby("phase", sort=True):
        sigma = lcfg.pickSigmaS.P if phase == "P" else lcfg.pickSigmaS.S
        r_used = g.loc[g["used"], "residualS"].to_numpy(dtype=np.float64)
        r_all = g["residualS"].to_numpy(dtype=np.float64)
        # Laplace scale (maximum likelihood): mean |r - median(r)|
        scale = (float(np.mean(np.abs(r_used - np.median(r_used)))) if r_used.size
                 else math.nan)
        parts.append(
            f"{phase}: pickSigmaS {sigma:g} s; used picks (n {r_used.size}) MAD "
            f"{_f(_mad(r_used), '.3f')} s, mean abs deviation {_f(scale, '.3f')} s = "
            f"{_f(scale / sigma, '.1f')}x sigma; all associated picks (n {r_all.size}) MAD "
            f"{_f(_mad(r_all), '.3f')} s"
        )
        above.append(bool(scale > sigma))
    overrides = ", ".join(f"{k} P {v.P:g} / S {v.S:g} s"
                          for k, v in sorted(lcfg.profilePickSigmaS.items())) or "none"
    lines.append("; ".join(parts) + f". Per-profile sigma overrides: {overrides}.")
    lines.append("")
    lines.append(
        "hErrM and vErrM in events_located.parquet are formal errors: the spread of the PDF "
        "exp(-misfit) with each pick weighted by prob / pickSigmaS. They assume residuals on the "
        "pickSigmaS scale and contain no model error. "
        + ("The used-pick residual spread above is larger than pickSigmaS, so the formal errors "
           "are too small even before model error" if all(above) else
           "The used-pick residual spread above is at or below pickSigmaS for every phase, so "
           "on the pick-noise side the formal errors are not too small; they still hold no "
           "model error" if not any(above) else
           "The used-pick residual spread above is larger than pickSigmaS for "
           + ", ".join(ph for ph, a in zip(sorted(pa["phase"].unique()), above, strict=True) if a)
           + ", so those formal errors are too small even before model error")
        + "; the public regional catalog comparison below shows how far model error can move a "
        "hypocentre. pickSigmaS is not updated from the residual spread in this run (the lane "
        "doc's Locator step 2): the Station statics section compares the robust sigma with it "
        "and flags only a ratio above statics.sigmaFlagRatio, and a config change is the lead's "
        "call. Compare formal errors only with formal errors (synthetic "
        "noisy.medianFormalVErrM)."
    )
    return lines


def _left_out(params: dict[str, Any]) -> str:
    """The synthetic test's used stations left out for recording no pick in this run, if any."""
    dropped = params.get("stationsWithoutPicks") or []
    return f" (left out for recording no pick in this run: {', '.join(dropped)})" if dropped else ""


def synthetic_section(inputs: DiagnosticsInputs) -> list[str]:
    """The stage's synthetic recovery test (synthetic.json) and the pick stats it used."""
    lines = ["## Synthetic recovery test (synthetic.json)", ""]
    syn = inputs.synthetic
    if syn is None:
        lines.append("Not run with this report.")
        return lines
    rep, params = syn.report, syn.params
    stats = params["pickStats"]
    noisy, clean = params["noisy"], params["noiseFree"]
    lines.append(
        f"{rep.nEvents} synthetic events on {params['nStations']} of the "
        f"{len(inputs.details.stations)} used stations{_left_out(params)}, "
        + ("the locator's own 3D tables (1D tables at 1D-fallback stations; no table error in "
           "the test, unlike the grid1d test's exact layered times)"
           if params.get("method") == GRID3D else "exact 1D layered times")
        + f" plus Gaussian noise at pickSigmaS (P {rep.pickSigmaS['P']:g} s, S "
        f"{rep.pickSigmaS['S']:g} s); every station has a P pick, S kept with probability "
        f"{stats['sKeepProb']:.2f}, every pick at prob {stats['pickProb']:.2f} (source: "
        f"{stats['source']}). True errors (noisy): median h {rep.medianHErrM:.0f} m, median v "
        f"{rep.medianVErrM:.0f} m, p90 v {rep.p90VErrM:.0f} m, median depth bias "
        f"{rep.medianDepthBiasM:+.0f} m ({params['depthBiasSign']}); noise-free depth bias "
        f"{clean['medianDepthBiasM']:+.0f} m. Formal errors (noisy): median hErrM "
        f"{_f(noisy['medianFormalHErrM'])} m, vErrM {_f(noisy['medianFormalVErrM'])} m; true error "
        f"within them for {_f(100 * (noisy['fracHWithinHErrM'] or math.nan))}% (h) and "
        f"{_f(100 * (noisy['fracVWithinVErrM'] or math.nan))}% (v)."
    )
    lines.append("")
    measured = stats.get("measured") or {}
    typical = (
        f" A real located event has a median of {measured['medianNStations']:g} stations and "
        f"{measured['medianNS']:g} S picks, so these numbers describe an event recorded on every "
        "station, like the best-recorded candidates, not a typical one."
        if "medianNStations" in measured and "medianNS" in measured else ""
    )
    lines.append(
        "The synthetic picks carry noise at pickSigmaS, below the observed residual spread (section "
        f"above), and a P pick at every one of the {params['nStations']} stations, and the test "
        "has no model error, so these errors are optimistic for the real candidate events."
        + typical
    )
    return lines


# --- table error exposure ------------------------------------------------------------------------


def table_errors(inputs: DiagnosticsInputs, ua: pd.DataFrame) -> pd.DataFrame:
    """Per used pick: table travel time minus the exact layered time at the located hypocentre."""
    tables = inputs.details.locator.tables
    parts = []
    for (sid, phase), g in ua.groupby(["stationId", "phase"], sort=True):
        r = np.hypot(g["evE"] - g["stE"], g["evN"] - g["stN"]).to_numpy(dtype=np.float64)
        z = g["evElevM"].to_numpy(dtype=np.float64)
        zr = tables.receiver_elev_m[str(sid)]
        exact = layered_first_arrival(tables.model, phase, zr, r, z)
        table = tables.table(str(sid), phase).lookup(r, z)
        parts.append(pd.DataFrame({"eventId": g["eventId"].to_numpy(), "stationId": sid,
                                   "phase": phase, "errS": table - exact}))
    if not parts:
        return pd.DataFrame(columns=["eventId", "stationId", "phase", "errS"])
    return pd.concat(parts, ignore_index=True)


def table_error_section(inputs: DiagnosticsInputs, ua: pd.DataFrame) -> list[str]:
    dcfg = inputs.cfg.diagnostics
    locator = inputs.details.locator
    tables = locator.tables
    by_layer = tables.accuracy_by_layer()
    sigma = inputs.cfg.locator.pickSigmaS
    lines = ["## Shallow-interface bias exposure (travel-time tables vs exact layered times)", ""]
    if locator.tables3d is not None:
        fallback = sorted(locator.tables3d.fallback)
        lines.append(
            "grid3d: the 3D tables have no exact reference; their accuracy on laterally uniform "
            "and on analytic laterally varying models is measured by tests/seismology/"
            "test_tt_grid3d.py (hq.locate.tt_grid3d docstring). The rest of this section covers "
            "the stations on 1D tables (fallback): " + (", ".join(fallback) or "none") + ".")
        lines.append("")
        ua = ua[ua["stationId"].isin(fallback)]
    layer_bits = []
    for phase in PHASES:
        worst = [f"top {row['topElevM']:g} m: {row['maxErrS'] * 1000:.1f} ms"
                 for row in by_layer[phase] if row["maxErrS"] > dcfg.tableErrorFlagS]
        layer_bits.append(f"{phase}: " + ("; ".join(worst) if worst else
                                          f"no layer above {dcfg.tableErrorFlagS * 1000:g} ms"))
    lines.append("LOC-02 accuracy record, largest table error per model layer over all tables "
                 f"(layers above tableErrorFlagS {dcfg.tableErrorFlagS * 1000:g} ms): "
                 + " / ".join(layer_bits) + ".")
    lines.append("")
    err = table_errors(inputs, ua)
    if err.empty:
        lines.append("No used picks: no exposure at located hypocentres to report.")
        return lines
    parts = []
    exposed_events = set()
    for phase, g in err.groupby("phase", sort=True):
        a = g["errS"].abs().to_numpy()
        exposed = g[g["errS"].abs() > dcfg.tableErrorFlagS]
        exposed_events |= set(exposed["eventId"])
        sig = sigma.P if phase == "P" else sigma.S
        parts.append(
            f"{phase}: {len(g)} used picks, table - exact median {np.median(g['errS']) * 1000:+.1f} "
            f"ms, max |error| {a.max() * 1000:.1f} ms ({a.max() / sig:.2f} of pickSigmaS "
            f"{sig * 1000:g} ms); {len(exposed)} picks above {dcfg.tableErrorFlagS * 1000:g} ms"
        )
    n_ev = len(inputs.details.result.events)
    lines.append("At the located hypocentres, " + "; ".join(parts) + f". {len(exposed_events)} of "
                 f"{n_ev} located events have at least one used pick above "
                 f"{dcfg.tableErrorFlagS * 1000:g} ms.")
    lines.append("")
    worst = float(err["errS"].abs().max())
    if worst <= min(sigma.P, sigma.S) / 2:
        lines.append(f"Conclusion: the tables' interface bias at these hypocentres (at most "
                     f"{worst * 1000:.1f} ms) stays under half the smallest pick sigma; it can't "
                     "move depths by more than a fraction of the formal errors.")
    else:
        lines.append(f"Conclusion: table errors reach {worst * 1000:.1f} ms at these hypocentres, "
                     "comparable to the pick sigma; finer tables (grids.dzM) or exact layered "
                     "times near interfaces would remove that bias.")
    return lines


# --- public regional catalog comparison ----------------------------------------------------------


def catalog_uncertainties(quakeml: Path) -> pd.DataFrame:
    """Per catalog event id: preferred-origin horizontal and depth uncertainty (m), when stated."""
    from obspy import read_events

    from hq.match.catalog import comcat_id  # at call time: hq.match imports hq.locate

    rows = []
    for event in read_events(str(quakeml), format="QUAKEML"):
        origin = event.preferred_origin()
        if origin is None:
            continue
        unc = origin.origin_uncertainty
        h = None if unc is None else unc.horizontal_uncertainty
        z = origin.depth_errors.uncertainty if origin.depth_errors is not None else None
        rows.append({"id": comcat_id(event),
                     "horizontalErrorM": math.nan if h is None else float(h),
                     "depthErrorM": math.nan if z is None else float(z)})
    return pd.DataFrame(rows, columns=["id", "horizontalErrorM", "depthErrorM"])


def compare_with_catalog(
    events: pd.DataFrame,
    catalog: pd.DataFrame,
    errors: pd.DataFrame | None,
    cfg: SeismologyConfig,
    known_ids: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Per public event: the lowest-cost located event within matching.maxDtS / maxDistM.

    Not one-to-one (the match stage is); offsets are ours minus the catalog's: ``dtS``, ``deM``,
    ``dnM`` (ENU), ``distM`` (epicentral), ``dzM`` (elevM; the catalog's elevM uses its stated
    depth datum). ``withinH`` / ``withinZ``: ``distM`` / ``|dzM|`` within the catalog's stated
    horizontal / depth uncertainty (null where the catalog states none).
    """
    m = cfg.matching
    cat = catalog.sort_values(["t", "id"], kind="stable").reset_index(drop=True)
    err = (errors.set_index("id") if errors is not None and len(errors)
           else pd.DataFrame(columns=["horizontalErrorM", "depthErrorM"]))
    ev_t = events["t"].to_numpy(dtype=np.float64)
    ev_e = events["enu_e"].to_numpy(dtype=np.float64)
    ev_n = events["enu_n"].to_numpy(dtype=np.float64)
    rows = []
    for c in cat.itertuples(index=False):
        cid = str(c.id)
        h_err = float(err.loc[cid, "horizontalErrorM"]) if cid in err.index else math.nan
        z_err = float(err.loc[cid, "depthErrorM"]) if cid in err.index else math.nan
        row: dict[str, Any] = {"catalogId": cid, "known": bool(known_ids and cid in known_ids),
                               "catalogT": float(c.t), "catalogElevM": float(c.elevM),
                               "catalogMag": c.mag, "catalogMagType": c.magType,
                               "catalogHErrM": h_err, "catalogZErrM": z_err, "eventId": None}
        dt = ev_t - float(c.t)
        dist = np.hypot(ev_e - float(c.enu_e), ev_n - float(c.enu_n))
        ok = (np.abs(dt) <= m.maxDtS) & (dist <= m.maxDistM)
        if ok.any():
            cost = np.where(ok, np.abs(dt) / m.dtScaleS + dist / m.distScaleM, np.inf)
            j = int(np.argmin(cost))
            e = events.iloc[j]
            dz = float(e["elevM"]) - float(c.elevM)
            row.update(
                eventId=str(e["id"]), dtS=float(dt[j]), distM=float(dist[j]),
                deM=float(ev_e[j] - float(c.enu_e)), dnM=float(ev_n[j] - float(c.enu_n)),
                dzM=dz, elevM=float(e["elevM"]), hErrM=float(e["quality_hErrM"]),
                vErrM=float(e["quality_vErrM"]), nStations=int(e["quality_nStations"]),
                nP=int(e["quality_nP"]), nS=int(e["quality_nS"]), rmsS=float(e["quality_rmsS"]),
                withinH=None if math.isnan(h_err) else bool(dist[j] <= h_err),
                withinZ=None if math.isnan(z_err) else bool(abs(dz) <= z_err),
            )
        rows.append(row)
    columns = ["catalogId", "known", "catalogT", "catalogElevM", "catalogMag", "catalogMagType",
               "catalogHErrM", "catalogZErrM", "eventId", "dtS", "distM", "deM", "dnM", "dzM",
               "elevM", "hErrM", "vErrM", "nStations", "nP", "nS", "rmsS", "withinH", "withinZ"]
    return pd.DataFrame(rows, columns=columns)


def catalog_section(
    inputs: DiagnosticsInputs, comp: pd.DataFrame | None, lateral: bool,
    lateral_before: bool = False,
) -> list[str]:
    """The comparison table; ``lateral``: row 7 flags a trend at the catalog hypocentres;
    ``lateral_before``: it did with the statics removed (pass 2)."""
    lines = ["## Comparison with the public regional catalog", ""]
    if inputs.catalog is None or comp is None:
        lines.append("catalog.parquet is not in the run dir: no comparison.")
        return lines
    m = inputs.cfg.matching
    have = comp[comp["eventId"].notna()]
    source = ("catalog.quakeml preferred origins" if inputs.catalog_errors is not None
              else "none (catalog.quakeml absent)")
    lines.append(
        f"Per public event, the lowest-cost located candidate event within {m.maxDtS:g} s and "
        f"{m.maxDistM / 1000:g} km (not one-to-one; the match stage assigns). Offsets are ours "
        "minus the catalog's; dz compares elevM (the catalog's converted with its stated datum). "
        f"Catalog uncertainties: {source}. {len(have)} of {len(comp)} public events have a "
        "candidate" + (f"; known events (known/windows.json): {', '.join(inputs.known_ids)}"
                       if inputs.known_ids else "") + "."
        + (" Located with reference statics: a reference event's position here is its held-out "
           "relocation (Station statics section), and every position is tied to the catalog's "
           "frame through the terms, so these offsets are not independent of the catalog."
           if inputs.statics is not None and inputs.statics.reference is not None else "")
    )
    lines.append("")
    if len(have):
        within_h = have["withinH"].eq(True)
        within_z = have["withinZ"].eq(True)
        stated_h = have["withinH"].notna()
        no_h = int((~stated_h).sum())
        no_z = int(have["withinZ"].isna().sum())
        lines.append(
            f"Median offsets over those {len(have)}: dt {have['dtS'].median():+.2f} s, de "
            f"{have['deM'].median():+.0f} m, dn {have['dnM'].median():+.0f} m, dz "
            f"{have['dzM'].median():+.0f} m. Within the catalog's stated horizontal uncertainty: "
            f"{int(within_h.sum())} of {len(have)}; depth: {int(within_z.sum())} of {len(have)}; "
            f"both: {int((within_h & within_z).sum())} of {len(have)}"
            + (f" (no stated horizontal error for {no_h}, no stated depth error for {no_z}; those "
               "count as neither within nor outside)" if no_h or no_z else "") + ". 'Within' "
            "compares the offset with the catalog's stated error alone; our formal hErrM / vErrM "
            "(see the pick sigma section: formal, no model error) are not added."
        )
        lines.append("")
        lines.append("| catalogId | known | eventId | dt (s) | dist (m) | de (m) | dn (m) | dz (m) "
                     "| our formal hErrM / vErrM (m) | catalog h / z err (m) | nSta / nP / nS | "
                     "rmsS (s) | within catalog stated error h / z |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
                     "--- |")
        for r in have.itertuples(index=False):
            lines.append(
                f"| {r.catalogId} | {'yes' if r.known else ''} | {r.eventId} | {r.dtS:+.2f} | "
                f"{r.distM:.0f} | {r.deM:+.0f} | {r.dnM:+.0f} | {r.dzM:+.0f} | {_f(r.hErrM)} / "
                f"{_f(r.vErrM)} | {_f(r.catalogHErrM)} / {_f(r.catalogZErrM)} | "
                f"{int(r.nStations)} / {int(r.nP)} / {int(r.nS)} | {r.rmsS:.3f} | {r.withinH} / "
                f"{r.withinZ} |"
            )
        lines.append("")
        outside = int(have["withinH"].eq(False).sum())
        if outside and lateral:
            lines.append(
                f"Conclusion: {outside} of {len(have)} compared events lie outside the catalog's "
                f"horizontal uncertainty (median de {have['deM'].median():+.0f} m, dn "
                f"{have['dnM'].median():+.0f} m). Row 7 tests the cause: with the hypocentre at "
                "the catalog's, residuals trend with azimuth beyond trendFlagS. Its S/P amplitude "
                "ratio against the model's Vp/Vs tells whether that trend is S-heavy (consistent "
                f"with lateral structure {_model_label(inputs)} can't hold); the public "
                "catalog's own model "
                "and locations remain the alternative the test can't exclude."
            )
        elif outside and inputs.statics is not None and inputs.statics.pass_number == 2:
            stated = have["catalogHErrM"].dropna()
            lines.append(
                f"Conclusion: after statics, {outside} of {len(have)} compared events lie outside "
                f"the catalog's stated horizontal uncertainty (median horizontal offset "
                f"{have['distM'].median():.0f} m; median stated catalog horizontal error "
                + (f"{stated.median():.0f} m" if len(stated) else "none stated") + "). "
                + ("Row 7 tested this before statics: with the statics removed, residuals at "
                   "the catalog's hypocentres trend with azimuth beyond trendFlagS, consistent "
                   f"with lateral structure {_model_label(inputs)} can't hold; the public "
                   "catalog's own model and locations remain the alternative the test can't "
                   "exclude. The station terms now carry whatever made that trend (Station "
                   "statics section), so row 7 no longer shows it: absorbed, not absent."
                   if lateral_before else
                   "Rows 1-7 flag no cause on this data, before or after statics.")
                + (" With reference statics these offsets are not independent of the catalog "
                   "(Station statics section)." if inputs.statics.reference is not None else "")
            )
        elif outside:
            lines.append(
                f"Conclusion: {outside} of {len(have)} compared events lie outside the catalog's "
                "horizontal uncertainty, and rows 1-7 flag no cause on this data."
            )
        elif int(stated_h.sum()) == 0:
            lines.append("Conclusion: the catalog states no horizontal uncertainty for the "
                         "compared events, so none can be judged within or outside it.")
        else:
            lines.append(f"Conclusion: every compared event with a stated horizontal uncertainty "
                         f"({int(stated_h.sum())} of {len(have)}) lies within it.")
    missing = comp[comp["eventId"].isna()]
    if len(missing):
        lines.append("")
        lines.append(f"No located candidate within tolerance for {len(missing)} public event(s): "
                     + ", ".join(missing["catalogId"].astype(str)) + ".")
    return lines


# --- station statics (LOC-05) ----------------------------------------------------------------------


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _offsets_lines(rep: "StaticsReport") -> list[str]:
    summary = rep.offsets_summary()
    ref = rep.reference
    if not summary or ref is None:
        return []
    fold = rep.extra.get("foldScheme", "")
    lines = [
        (f"Cross-validated offsets from the public regional catalog over the {len(ref)} "
         "reference events (ours minus the catalog's; vertical = elevM, the catalog's from its "
         f"stated datum). The held-out ({fold}) row is the cross-validated agreement with the "
         "public regional catalog's frame, not absolute accuracy: the catalog's own systematic "
         "error is shared by every fold. The in-sample row relocates the same events with the "
         "terms from all reference events, themselves included:"),
        "",
        _row(["locations", "horizontal median (m)", "horizontal p90 (m)", "abs(dz) median (m)",
              "abs(dz) p90 (m)", "median rmsS (s)"]),
        _row(["---"] * 6),
    ]
    for when, label in (("before", "no statics"), ("after", f"held-out terms ({fold})"),
                        ("inSample", "all-reference terms (in-sample)")):
        if when not in summary:
            continue
        o = summary[when]
        lines.append(_row([label, _f(o["medianHM"]), _f(o["p90HM"]), _f(o["medianAbsDzM"]),
                           _f(o["p90AbsDzM"]), _f(o["medianRmsS"], ".3f")]))
    if "inSample" in summary:
        a, i = summary["after"], summary["inSample"]
        lines += ["", (
            f"Held-out minus in-sample: horizontal median {a['medianHM'] - i['medianHM']:+.0f} m, "
            f"p90 {a['p90HM'] - i['p90HM']:+.0f} m; abs(dz) median "
            f"{a['medianAbsDzM'] - i['medianAbsDzM']:+.0f} m, p90 "
            f"{a['p90AbsDzM'] - i['p90AbsDzM']:+.0f} m. Holding a fold out removes about "
            f"{len(ref) / max(ref['fold'].nunique(), 1):.0f} of {len(ref)} reference events from "
            "each median term; the folds are dealt by origin "
            "time, not by position, so the held-out offsets measure agreement inside the "
            "reference events' cloud, not how the terms transfer to candidate events away from "
            "it."
        )]
    lines += ["", (
        f"Median signed offsets with held-out terms: de {ref['afterDeM'].median():+.0f} m, dn "
        f"{ref['afterDnM'].median():+.0f} m, dz {ref['afterDzM'].median():+.0f} m (no statics: de "
        f"{ref['beforeDeM'].median():+.0f} m, dn {ref['beforeDnM'].median():+.0f} m, dz "
        f"{ref['beforeDzM'].median():+.0f} m)."
    ), ""]
    return lines


def statics_section(inputs: DiagnosticsInputs) -> list[str]:
    """The statics pass: method, terms, a written explanation per term above the flag, spread."""
    lines = ["## Station statics (LOC-05)", ""]
    rep = inputs.statics
    if rep is None:
        lines.append("No statics pass ran with this report.")
        return lines
    scfg = inputs.cfg.statics
    flag = inputs.cfg.diagnostics.stationResidualFlagS
    terms = rep.terms
    lines.append(f"Mode `{rep.mode}`, pass {rep.pass_number}: {rep.note}.")
    lines.append("")
    if rep.mode == "referenceEvents":
        lines += [
            ("Pipeline order in this mode: locate (pass 1, no statics) -> match -> locate (pass "
             "2, reference terms read through matches.parquet) -> match -> tier. Stage locate "
             "runs pass 2 when matches.parquet is in the run dir, pass 1 otherwise; `locate()` "
             "(docs/02) has no matches and runs pass 1."), ""]
    if rep.pass_number == 1:
        lines.append("No statics applied in this pass.")
        lines += ["", *_sigma_lines(rep, scfg.sigmaFlagRatio)]
        return lines
    if rep.mode == "referenceEvents":
        lines += [
            "Method: each reference event is a public regional catalog event matched to a "
            "candidate event (matches.parquet). Its hypocentre is fixed at the catalog's "
            "(latitude/longitude to ENU through hq.locate.coords; elevM from the catalog's stated "
            "depth datum), and our associated picks are compared with the travel times from "
            "there. Median polish over the reference events: per event the origin time is the "
            "locator's weighted median, per station-phase the term is the median residual "
            f"({scfg.polishIterations} alternations), capped at +/- referenceCapS "
            f"{scfg.referenceCapS:g} s, and 0 with fewer than minReferenceEvents "
            f"{scfg.minReferenceEvents} reference events. Every reference event was relocated "
            "with terms computed without it; unmatched candidate events use the terms from all "
            "reference events (statics.parquet)."
            + (f" Left out, hypocentre outside the travel-time grid: {', '.join(rep.skipped)}."
               if rep.skipped else ""),
            "",
            ("Absolute positions are therefore tied to the public regional catalog's (UUSS) "
             "frame: the terms carry the catalog's own velocity model and locations into ours, "
             "so agreement with the catalog is no longer independent evidence of accuracy. The "
             "held-out (cross-validated) offsets below measure agreement with that frame, not "
             "absolute accuracy. The terms are calibrated for sources near the reference events; "
             "they are less valid for candidate events far from them."),
            "",
            *_offsets_lines(rep),
        ]
    else:
        hist = rep.history
        lines += [
            (f"Method: pass 1 without statics, then {scfg.iterations} iteration(s): each "
             "station-phase static = current static + the median used-pick residual over the "
             f"well-constrained events (nStations >= {scfg.wellConstrained.minStations}, nS >= "
             f"{scfg.wellConstrained.minS}, gapDeg <= {scfg.wellConstrained.maxGapDeg:g}, no "
             f"depthOnEdge, formal errors present), capped at +/- capS {scfg.capS:g} s, 0 with "
             f"fewer than minEvents {scfg.minEvents} events; every event relocated with the new "
             "statics."), "",
            _row(["iteration", "median rmsS (s)", "well-constrained events", "non-zero statics",
                  "max abs static (s)"]), _row(["---"] * 5),
            *[_row([str(int(h.iteration)), _f(h.medianRmsS, ".3f"), str(int(h.nCalibration)),
                    str(int(h.nNonZero)), _f(h.maxAbsS, ".3f")]) for h in hist.itertuples()],
            "",
        ]
    lines += [
        ("Terms are relative delays: each event's origin time absorbs any constant shared by all "
         "its picks, so a term is a station-phase's delay against the event's weighted-median "
         "pick, not an absolute time correction. Their common constant is not fixed by the data "
         "(the polish starts from zero terms), and it moves every origin time by about the "
         "same amount: the median dt of the public regional catalog comparison includes it. It "
         "moves no location and no predicted arrival (tPred includes the static)."), ""]
    after_all = float(inputs.details.result.events["quality_rmsS"].median())
    prev = rep.previous_median_rms_s
    lines.append(
        f"Median rmsS over all {len(inputs.details.result.events)} located events with statics: "
        f"{after_all:.3f} s" + (f"; {rep.previous_median_rms_from}: {prev:.3f} s."
                                if prev is not None else
                                "; no no-statics events_located.parquet was in the run dir to "
                                "compare with, and run.json held no value for this association.")
    )
    lines.append("")
    active = terms[terms["nEvents"] >= rep.min_events]
    below = terms[terms["nEvents"] < rep.min_events]
    clipped = active[active["rawS"].abs() > rep.cap_s]
    if len(active):
        big = active.loc[active["rawS"].abs().idxmax()]
        lines.append(
            f"Cap: +/- {rep.cap_s:g} s clips {len(clipped)} of {len(active)} estimated terms. "
            f"Largest uncapped term {big['rawS']:+.3f} s ({big['stationId']} {big['phase']}, "
            f"{int(big['nEvents'])} events, MAD over those events {big['madS']:.3f} s); the "
            f"largest MAD over events of any estimated term is {active['madS'].max():.3f} s, so "
            "each term is a consistent delay across events, not the scatter of a few picks."
            + (" The cap bounds a term from a station with systematically wrong picks; one "
               "that clips no term this consistent leaves the lateral structure in place."
               if clipped.empty
               else " Clipped terms leave part of the station's delay in the residuals.")
        )
    if len(below):
        lines.append("")
        lines.append(
            f"Below the minimum of {rep.min_events} events, set to 0 (flagged; nEvents in "
            "statics.parquet shows the count): " + ", ".join(
                f"{r.stationId} {r.phase} (n {int(r.nEvents)})"
                for r in below.itertuples(index=False)) + "."
        )
    lines += ["", _row(["stationId", "phase", "staticS (s)", "uncapped (s)", "nEvents",
                        "MAD over events (s)", "note"]), _row(["---"] * 7)]
    for r in terms.itertuples(index=False):
        note = (f"< {rep.min_events} events: 0" if r.nEvents < rep.min_events else
                f"capped at {rep.cap_s:g}" if abs(r.rawS) > rep.cap_s else
                f"> {flag:g} s: explained below" if abs(r.staticS) > flag else "")
        lines.append(_row([str(r.stationId), str(r.phase), f"{r.staticS:+.3f}",
                           f"{r.rawS:+.3f}", str(int(r.nEvents)), _f(r.madS, ".3f"), note]))
    ex = rep.explanations
    xcfg = scfg.explain
    n_un = int((ex["verdict"] == "unexplained").sum())
    n_far = int((ex["verdict"] == "far").sum())
    n_contra = int((ex["farContradictedBy"].astype(str) != "").sum())
    counts = ex["verdict"].value_counts()
    lines += ["", f"### Written explanation of every static above {flag:g} s", "",
              f"{len(ex)} static(s) above {flag:g} s: " + (", ".join(
                  f"{k} {int(v)}" for k, v in counts.items()) if len(ex) else "none") + ". "
              + (f"{n_un} unexplained (the depth gate asks for none). " if n_un else
                 "None unexplained. ")
              + (f"For {n_contra} early term(s) at stations beyond farStationM "
                 f"{xcfg.farStationM / 1000:g} km, another station that far has a late term above "
                 f"{flag:g} s, which contradicts the far-station hypothesis (named in the "
                 "explanation). " if n_contra else "")
              + (f"{n_far} rest on the far-station hypothesis, which no other distant station "
                 "contradicts but nothing here tests further. " if n_far else "")
              + "Each verdict says what the term is consistent with, not a tested cause: "
              f"lateral = the nearest stations within {xcfg.neighbourMaxDistM / 1000:g} km share "
              f"the delay (structure {_model_label(inputs)} can't hold, row 7; the only verdict "
              "that draws "
              "on other stations' terms); path = P and S changed in proportion to the model's "
              "Vp/Vs; vpvs = S changed proportionally more than P (near-station rock whose Vp/Vs "
              "differs from the model's, which moves both phases); timing = equal P and S "
              "delays; far = early at a distant station. The S/P bands (ratioBand "
              f"{xcfg.ratioBand:g}) leave few same-sign ratios without a label, so a path, vpvs "
              "or timing verdict shows a term is consistent with near-station structure or "
              "timing, not that this cause was tested. Evidence rules: seismology.yaml "
              "`statics.explain`; the verdict is the first that holds, in that order.", ""]
    if len(ex):
        lines += [_row(["stationId", "phase", "staticS (s)", "verdict", "explanation"]),
                  _row(["---"] * 5)]
        lines += [_row([str(r.stationId), str(r.phase), f"{r.staticS:+.3f}", str(r.verdict),
                        str(r.explanation)]) for r in ex.itertuples(index=False)]
        lines.append("")
    lines += _sigma_lines(rep, scfg.sigmaFlagRatio)
    return lines


def _sigma_lines(rep: "StaticsReport", ratio_flag: float) -> list[str]:
    sig = rep.sigma
    when = "after" if rep.pass_number == 2 else "without"
    lines = [f"### Residual spread vs pick sigma ({when} statics, used picks)", "",
             _row(["events", "phase", "picks", "pickSigmaS (s)", "robust sigma 1.4826 x MAD (s)",
                   "ratio"]), _row(["---"] * 6)]
    lines += [_row([str(r.events), str(r.phase), str(int(r.nPicks)), f"{r.configuredS:g}",
                    _f(r.robustSigmaS, ".3f"), _f(r.ratio, ".2f")])
              for r in sig.itertuples(index=False)]
    above = sig[sig["wellAbove"]]
    cal = sig[sig["events"] == rep.sigma_events]
    lines.append("")
    if len(above):
        lines.append(
            f"Well above the configured sigma (ratio > statics.sigmaFlagRatio {ratio_flag:g}): "
            + ", ".join(f"{r.phase} over the {r.events} {r.robustSigmaS:.3f} s vs "
                        f"{r.configuredS:g} s" for r in above.itertuples(index=False))
            + f". Recommended locator.pickSigmaS (robust sigma of the {rep.sigma_events}): "
            + ", ".join(f"{r.phase} {r.robustSigmaRoundedS:.3f} s"
                        for r in cal.itertuples(index=False))
            + ". Not applied: the lead decides whether to change config. Until then hErrM / vErrM "
            "are formal errors at the configured sigma, smaller than this spread supports."
        )
    else:
        lines.append(
            f"No phase's robust sigma exceeds {ratio_flag:g}x its pickSigmaS, so no change is "
            "recommended. The robust sigma is a lower bound on pick noise: it comes from the "
            "residuals of the used picks (after the outlier pass) at locations fitted to those "
            "same picks, so a ratio below 1 does not show the configured sigma is too large. "
            "The formal errors hold no model error either way."
        )
    return lines


# --- 1D vs 3D (LOC-07) -----------------------------------------------------------------------------


def _residual_frame(events: pd.DataFrame, arrivals: pd.DataFrame,
                    stations: pd.DataFrame) -> pd.DataFrame:
    """Used picks' residuals with the event epicentre and station position (row 7's inputs)."""
    arr = arrivals[arrivals["usedInLocation"].to_numpy(dtype=bool)]
    ev = events[["id", "enu_e", "enu_n"]].rename(
        columns={"id": "eventId", "enu_e": "evE", "enu_n": "evN"})
    st = stations[["id", "enu_e", "enu_n"]].rename(
        columns={"id": "stationId", "enu_e": "stE", "enu_n": "stN"})
    out = arr.assign(eventId=arr["eventId"].astype(str), stationId=arr["stationId"].astype(str),
                     phase=arr["phase"].astype(str))
    out = out.merge(ev.assign(eventId=ev["eventId"].astype(str)), on="eventId", how="inner")
    return out.merge(st.assign(stationId=st["stationId"].astype(str)), on="stationId",
                     how="inner")


def model_time_differences(inputs: DiagnosticsInputs, ua: pd.DataFrame) -> pd.DataFrame:
    """Per station and phase over the used picks: 3D minus 1D table time at the final hypocentres."""
    locator = inputs.details.locator
    rows = []
    for (sid, phase), g in ua.groupby(["stationId", "phase"], sort=True):
        e, n = g["evE"].to_numpy(dtype=np.float64), g["evN"].to_numpy(dtype=np.float64)
        z = g["evElevM"].to_numpy(dtype=np.float64)
        t3 = locator.station_times(str(sid), phase, e, n, z)
        t1 = locator.tables.table(str(sid), phase).lookup(
            np.hypot(e - g["stE"].to_numpy(dtype=np.float64),
                     n - g["stN"].to_numpy(dtype=np.float64)), z)
        rows.append({"stationId": sid, "phase": phase, "n": len(g),
                     "medianDiffS": float(np.median(t3 - t1)),
                     "has3d": bool(locator.tables3d is not None and locator.tables3d.has(str(sid)))})
    return pd.DataFrame(rows, columns=["stationId", "phase", "n", "medianDiffS", "has3d"])


def _pct(values: np.ndarray, q: float) -> str:
    return _f(_q(values, q), "+.0f")


def events_above_ground(inputs: DiagnosticsInputs) -> tuple[int, int, int]:
    """grid3d: (events above the 3D model's ground at their epicentre, events whose epicentre's
    model column has no ground surface, events). The ground is ``tt_grid3d.ground_elev_m`` of the
    nearest model column."""
    t3 = inputs.details.locator.tables3d
    ev = inputs.details.result.events
    if t3 is None or ev.empty:
        return 0, 0, len(ev)
    i, j = nearest_columns(t3.source, ev["enu_e"], ev["enu_n"], origin_utm(inputs.run.origin))
    vp, _ = t3.source.columns(i, j)
    ground = ground_elev_m(vp, t3.source)
    z = ev["elevM"].to_numpy(dtype=np.float64)
    has = np.isfinite(ground)
    return int(np.sum(has & (z > np.where(has, ground, np.inf)))), int(np.sum(~has)), len(ev)


def _verdicts(rep: "StaticsReport | None", flag: float) -> str:
    if rep is None or not bool((rep.terms["staticS"] != 0.0).any()):
        return "no statics"
    ex = rep.explanations
    counts = ex["verdict"].value_counts()
    un = ex[ex["verdict"] == "unexplained"]
    listed = ", ".join(f"{k} {int(v)}" for k, v in counts.items()) or "none"
    unexplained = ", ".join(f"{r.stationId} {r.phase} {r.staticS:+.3f} s"
                            for r in un.itertuples(index=False))
    return (f"{len(ex)} above {flag:g} s ({listed})"
            + (f": unexplained {unexplained}" if len(un) else ""))


def grid3d_section(inputs: DiagnosticsInputs, ua: pd.DataFrame,
                   comp: pd.DataFrame | None) -> list[str]:
    """The 1D vs 3D section (grid3d only; one line otherwise)."""
    d = inputs.details
    locator = d.locator
    lines = ["## 1D vs 3D travel times (LOC-07)", ""]
    t3 = locator.tables3d
    if t3 is None:
        lines.append(f"Not run: locator.method is `{locator.method}` (grid3d writes this section).")
        return lines
    rec = t3.source.to_record()
    ev = rec.get("evidence", {})
    res = dict(t3.resampling)
    g = t3.grid
    lines.append(
        f"Model `{t3.source.name}` (SHA-256 {t3.source.sha256[:12]}...; "
        f"{rec.get('source', {}).get('citation', 'no citation') if isinstance(rec.get('source'), dict) else 'n/a'}"
        f"), resampled to {g.spacing_m:g} m cells (mean slowness) on the run's ENU lattice e "
        f"{g.e0_m:.0f}..{g.e_max_m:.0f}, n {g.n0_m:.0f}..{g.n_max_m:.0f}, elevM {g.z0_m:.0f}.."
        f"{g.top_elev_m:.0f} m ({g.n_e} x {g.n_n} x {g.n_z} nodes); air handling "
        f"`{t3.air_handling}` ({res.get('airNodes', 'n/a')} air nodes in "
        f"{res.get('columnsWithAir', 'n/a')} columns; {res.get('constantColumns', 'n/a')} columns "
        f"with one value at every elevation); constant columns `{t3.constant_columns}`; seed "
        f"radius {t3.seed_radius_m:g} m; {t3.n_built} tables built, {t3.n_loaded} loaded from the "
        f"cache in {t3.build_s:.1f} s."
    )
    lines.append("")
    for label in ("crs", "vertical", "air"):
        if label in ev:
            lines.append(f"- {label}: {ev[label]}")
    lines.append("")
    st = inputs.stations.set_index(inputs.stations["id"].astype(str))
    fallback = t3.fallback
    lines.append(
        f"Stations on 3D tables: {len(t3.stations_3d)} of {len(d.stations)} used; on 1D tables: "
        + (", ".join(f"{k} ({v})" for k, v in fallback.items()) or "none") + ".")
    lines.append("")
    lines += [("| station | column in the 3D model | tables | model ground between (m ASL) | "
               "surfaceElevM | sensorElevM | ground check |"),
              "| --- | --- | --- | --- | --- | --- | --- |"]
    inside = outside = 0
    worst = 0.0
    constant = []
    for sid in sorted(t3.columns):
        c = t3.columns[sid]
        surface = float(st.loc[sid, "surfaceElevM"])
        sensor = float(st.loc[sid, "sensorElevM"])
        if c.kind == "basin" and c.ground_low_elev_m is not None:
            lo, hi = c.ground_low_elev_m, float(c.ground_high_elev_m or math.nan)
            off = max(lo - surface, surface - hi, 0.0)
            check = "surfaceElevM inside" if off == 0.0 else f"surfaceElevM {off:.1f} m outside"
            inside += int(off == 0.0)
            outside += int(off > 0.0)
            worst = max(worst, off)
            span = f"{lo:.0f} .. {hi:.0f}"
        else:
            span = "n/a"
            check = ("no ground surface in the file (one value at every elevation)"
                     if c.kind == "constant" else
                     "varies with elevation, no air value on top: no ground surface to check"
                     if c.kind == "other" else str(c.reason))
            if c.kind == "constant":
                constant.append(sid)
        tables = "3D" if t3.has(sid) else "1D"
        lines.append(f"| {sid} | {c.kind or 'outside'} | {tables} | {span} | {surface:.1f} | "
                     f"{sensor:.1f} | {check} |")
    lines.append("")
    on_1d = [sid for sid in constant if not t3.has(sid)]
    lines.append(
        (f"Ground check: in {inside} of {inside + outside} basin columns the station's "
         "surfaceElevM (3DEP) lies between the model's last ground node and first air node"
         + (f"; the others lie within {worst:.1f} m of that bracket" if outside else "")
         + ". That places the model's ground surface in the run's frame at those stations. "
         if inside + outside else "Ground check: no station sits in a basin column (the model "
         "holds no ground surface at any station), so nothing places the model's ground in the "
         "run's frame here. ") + (
            f"{len(constant)} station(s) sit in columns with one value at every elevation "
            f"({', '.join(constant)}): the file holds one velocity there from its bottom to its "
            "top (basement velocity in the GDR 1800 file) and no ground surface or basin data, "
            "so it can't tell outcrop from missing data; grid3d.constantColumns "
            f"`{t3.constant_columns}` "
            + (f"puts {', '.join(on_1d)} on their 1D tables." if on_1d else
               "gives them 3D tables from those values (basement rock right up to the sensor).")
            if constant else ""))
    lines.append("")
    n_above, n_no_ground, n_ev = events_above_ground(inputs)
    lines.append(
        f"Events above the 3D model's ground at their epicentre (nearest model column; ground "
        f"midway between its last ground node and first air node): {n_above} of {n_ev}; "
        f"{n_no_ground} epicentres sit in columns with no ground surface in the file. The locator "
        "does not mask above-ground nodes (the search volume's top is "
        f"{locator.volume.top_elev_m:.0f} m ASL everywhere); locate_flags.parquet flags events "
        "above their nearest used station's surfaceElevM, as for grid1d.")
    lines.append("")

    diff = model_time_differences(inputs, ua)
    if len(diff):
        lines.append("3D minus 1D table time at this run's final hypocentres (median over each "
                     "station-phase's used picks; negative = the 3D model predicts an earlier "
                     "arrival; stations on 1D tables show 0):")
        lines.append("")
        lines += ["| station | P (s) | S (s) | picks |", "| --- | --- | --- | --- |"]
        wide = diff.pivot_table(index="stationId", columns="phase", values="medianDiffS")
        counts = diff.groupby("stationId")["n"].sum()
        for sid in wide.index:
            p = wide.loc[sid].get("P", math.nan)
            s_ = wide.loc[sid].get("S", math.nan)
            tag = "" if t3.has(str(sid)) else " (1D)"
            lines.append(f"| {sid}{tag} | {_f(p, '+.3f')} | {_f(s_, '+.3f')} | "
                         f"{int(counts[sid])} |")
        lines.append("")

    lines.append(
        "Synthetic recovery test: with grid3d its forward model is the locator's own 3D tables, "
        "so its errors hold no table error; the grid1d test uses exact layered times, so its "
        "errors include the 1D tables' error. The two tests' numbers are not directly comparable "
        "(synthetic.json has no field for the forward model; run.json's synthetic.forwardModel "
        "records it).")
    lines.append("")

    cmp1 = inputs.grid1d
    if cmp1 is None:
        lines.append("No grid1d relocation with this report: no per-event 1D vs 3D comparison.")
        return lines
    d1 = cmp1.details
    cur = d.result.events.merge(d.flags[["eventId", "assocId"]], left_on="id", right_on="eventId")
    old = d1.result.events.merge(d1.flags[["eventId", "assocId"]], left_on="id",
                                 right_on="eventId")
    both = cur.merge(old, on="assocId", suffixes=("", "1d"))
    if both.empty:
        lines.append("The grid1d relocation located none of these events: no per-event comparison.")
        return lines
    dz = (both["elevM"] - both["elevM1d"]).to_numpy(dtype=np.float64)
    de = (both["enu_e"] - both["enu_e1d"]).to_numpy(dtype=np.float64)
    dn = (both["enu_n"] - both["enu_n1d"]).to_numpy(dtype=np.float64)
    dh = np.hypot(de, dn)
    statics_1d = bool(d1.result.events["quality_statics"].astype(bool).any())
    statics_3d = bool(d.result.events["quality_statics"].astype(bool).any())
    rep3 = inputs.statics
    lines.append(
        f"Against grid1d locations of the same association, made in this stage run with the same "
        f"statics configuration ({len(both)} events; statics pass "
        f"{rep3.pass_number if rep3 is not None else 'n/a'}, applied: 1D "
        f"{'yes' if statics_1d else 'no'}, 3D {'yes' if statics_3d else 'no'}; the 1D relocation "
        f"took {cmp1.runtime_s:.0f} s): depth shift 3D minus 1D elevM median {np.median(dz):+.0f} "
        f"m (p10 {_pct(dz, 0.1)}, p90 {_pct(dz, 0.9)} m; negative = deeper in 3D), median abs "
        f"{np.median(np.abs(dz)):.0f} m; epicentre shift median {np.median(dh):.0f} m (p90 "
        f"{_q(dh, 0.9):.0f} m), median de {np.median(de):+.0f} m, dn {np.median(dn):+.0f} m; "
        f"median rmsS 1D {both['quality_rmsS1d'].median():.3f} s, 3D "
        f"{both['quality_rmsS'].median():.3f} s.")
    lines.append("")
    stations = d.stations
    r1 = _residual_frame(d1.result.events, d1.result.arrivals, stations)
    r3 = _residual_frame(d.result.events, d.result.arrivals, stations)
    a1, a3 = azimuth_fit(r1), azimuth_fit(r3)
    lines.append("Row 7 residual trend vs azimuth at each method's own locations: 1D "
                 + (_fit_text(a1) or "n/a") + "; 3D " + (_fit_text(a3) or "n/a") + ".")
    lines.append("")
    if comp is not None and inputs.catalog is not None:
        comp1 = compare_with_catalog(d1.result.events, inputs.catalog, inputs.catalog_errors,
                                     inputs.cfg, inputs.known_ids)
        h1, h3 = comp1.dropna(subset=["distM"]), comp.dropna(subset=["distM"])
        if len(h1) and len(h3):
            lines.append(
                "Offsets from the public regional catalog (nearest candidate within matching."
                f"maxDtS / maxDistM, as in the catalog section): 1D {len(h1)} events, "
                f"horizontal median {h1['distM'].median():.0f} m (p90 {h1['distM'].quantile(0.9):.0f}"
                f" m), median de {h1['deM'].median():+.0f}, dn {h1['dnM'].median():+.0f}, dz "
                f"{h1['dzM'].median():+.0f} m, median abs dz {h1['dzM'].abs().median():.0f} m; 3D "
                f"{len(h3)} events, horizontal median {h3['distM'].median():.0f} m (p90 "
                f"{h3['distM'].quantile(0.9):.0f} m), median de {h3['deM'].median():+.0f}, dn "
                f"{h3['dnM'].median():+.0f}, dz {h3['dzM'].median():+.0f} m, median abs dz "
                f"{h3['dzM'].abs().median():.0f} m.")
            lines.append("")
    flag = inputs.cfg.diagnostics.stationResidualFlagS
    lines.append(f"Statics above the flag: 1D {_verdicts(cmp1.statics, flag)}; 3D "
                 f"{_verdicts(rep3, flag)}.")
    lines.append("")
    m1 = r1.groupby(["stationId", "phase"])["residualS"].median()
    m3 = r3.groupby(["stationId", "phase"])["residualS"].median()
    table = pd.concat({"1d": m1, "3d": m3}, axis=1).reset_index()
    lines.append("Median residual per station and phase over used picks (tObs - tPred, each at "
                 "its own locations, statics included as applied):")
    lines.append("")
    lines += ["| station | phase | 1D (s) | 3D (s) |", "| --- | --- | --- | --- |"]
    for r in table.itertuples(index=False):
        lines.append(f"| {r.stationId} | {r.phase} | {_f(r[2], '+.3f')} | {_f(r[3], '+.3f')} |")
    return lines


# --- report ---------------------------------------------------------------------------------------


def _mad_s(values: pd.Series) -> float:
    x = values.to_numpy(dtype=np.float64)
    return float(np.median(np.abs(x - np.median(x)))) if x.size else math.nan


def harvest_section(inputs: DiagnosticsInputs) -> list[str]:
    """The pick harvest (LOC-10), only when it ran: rules, counts, chance, per station-phase
    offsets, relocation shifts and the held-out reference offsets with and without it."""
    d = inputs.details
    rep = d.harvest
    if rep is None:
        return []
    cfg, c, a = rep.config, rep.counts, rep.added
    lines = [
        "## Pick harvest (LOC-10)", "",
        (f"After the statics-corrected locate, a pick in no association event (prob >= "
         f"{cfg.minProb:g}, on a used station; {c['freePicks']} such picks) was added to an "
         f"event when it lay within {cfg.windowS.P:g} s (P) / {cfg.windowS.S:g} s (S) of the "
         "event's predicted arrival (statics included) at a station-phase the event had no pick "
         f"for, with the same phase label; the {c['events']} events that gained picks were "
         f"relocated once (`hq.locate.harvest`): {c['picks']} picks added, {c['picksUsed']} "
         f"used after the relocation. Skipped: {c['ambiguousPicks']} picks within the window "
         "of two or more "
         f"events' predicted arrivals, {c['skippedUntrustedSlots']} slots with a candidate in "
         "events whose location is not trusted (depthOnEdge, MAP on a volume face, truncated "
         "PDF) and "
         f"{c['skippedMultiCandidateSlots']} slots with two or more candidates. The station "
         "terms never use harvested picks."),
        "",
        ("Harvested picks are chosen because they agree with the current location, so the pick "
         "counts, gap, rmsS and formal errors they change, and the tiers built on them, are not "
         "independent evidence of a better location. The held-out reference offsets below are."),
        "",
        _row(["phase", "added", "used", "dropped by the outlier pass", "median offset (s)",
              "MAD offset (s)"]),
        _row(["---"] * 6),
    ]
    for ph in ("P", "S"):
        g = a[a["phase"] == ph]
        lines.append(_row([ph, str(len(g)), str(int(g["usedInLocation"].sum())),
                           str(int((~g["usedInLocation"].astype(bool)).sum())),
                           _f(g["offsetS"].median() if len(g) else math.nan, "+.3f"),
                           _f(_mad_s(g["offsetS"]), ".3f")]))
    ch = rep.chance
    control = ", ".join(f"{k} s: {v}" for k, v in ch["controlPicks"].items())
    lines += ["", (
        f"Chance: {_f(ch['analyticPicks'], '.1f')} picks expected analytically (each "
        "station-phase's free-pick rate over the picks' time span x the window, summed over the "
        "empty slots of trusted events); control, the same rules with every predicted arrival "
        f"moved by a fixed shift: {control} (mean {_f(ch['controlMeanPicks'], '.1f')}), "
        f"against {c['picks']} harvested at the predicted arrivals."), ""]
    if len(a):
        lines += [_row(["station", "phase", "added", "used", "median offset (s)",
                        "MAD offset (s)"]), _row(["---"] * 6)]
        per = a.groupby(["stationId", "phase"], sort=True)
        order = sorted(per.groups, key=lambda k: (-len(per.groups[k]), k))
        for sid, ph in order:
            g = per.get_group((sid, ph))
            lines.append(_row([str(sid), str(ph), str(len(g)), str(int(g["usedInLocation"].sum())),
                               _f(g["offsetS"].median(), "+.3f"), _f(_mad_s(g["offsetS"]), ".3f")]))
        lines.append("")
    at = {aid: k for k, aid in enumerate(d.assoc_ids)}
    shifts = [(math.hypot(d.locations[at[aid]].e_m - b.e_m, d.locations[at[aid]].n_m - b.n_m),
               abs(d.locations[at[aid]].elev_m - b.elev_m), b.rms_s, d.locations[at[aid]].rms_s)
              for aid, b in rep.before.items()]
    if shifts:
        sh = np.array(shifts, dtype=np.float64)
        lines += [(
            f"Relocation of the {len(shifts)} events that gained picks: horizontal shift median "
            f"{_f(np.median(sh[:, 0]))} m, p90 {_f(_q(sh[:, 0], 0.9))} m; abs(dz) median "
            f"{_f(np.median(sh[:, 1]))} m, p90 {_f(_q(sh[:, 1], 0.9))} m; median rmsS "
            f"{_f(np.median(sh[:, 2]), '.3f')} s before, {_f(np.median(sh[:, 3]), '.3f')} s "
            "after."), ""]
    st = inputs.statics
    ref = st.reference if st is not None else None
    if st is not None and ref is not None and "afterNoHarvestHM" in ref.columns and len(ref):
        summary = st.offsets_summary()
        gained = int(ref["assocId"].astype(str).isin(set(rep.before)).sum())
        lines += [
            (f"Held-out offsets of the {len(ref)} reference events from the public regional "
             f"catalog ({gained} of them gained picks), with the same held-out terms before and "
             "after the harvest:"), "",
            _row(["locations", "horizontal median (m)", "horizontal p90 (m)",
                  "abs(dz) median (m)", "abs(dz) p90 (m)", "median rmsS (s)"]),
            _row(["---"] * 6),
        ]
        for when, label in (("afterNoHarvest", "held-out terms, before the harvest"),
                            ("after", "held-out terms, after the harvest")):
            o = summary[when]
            lines.append(_row([label, _f(o["medianHM"]), _f(o["p90HM"]), _f(o["medianAbsDzM"]),
                               _f(o["p90AbsDzM"]), _f(o["medianRmsS"], ".3f")]))
        h = ref["afterHM"].to_numpy(dtype=np.float64) - ref["afterNoHarvestHM"].to_numpy(
            dtype=np.float64)
        v = (np.abs(ref["afterDzM"].to_numpy(dtype=np.float64))
             - np.abs(ref["afterNoHarvestDzM"].to_numpy(dtype=np.float64)))
        lines += ["", (
            f"Per reference event, the harvest moved the horizontal offset closer for "
            f"{int((h < 0).sum())} and farther for {int((h > 0).sum())}, and abs(dz) closer for "
            f"{int((v < 0).sum())} and farther for {int((v > 0).sum())}."), ""]
    syn = inputs.cfg.synthetic
    if syn.sKeepProb is None or syn.pickProb is None:
        b = rep.pick_stats_before
        lines += [(
            "synthetic.json's measured sKeepProb / pickProb come from the events after the "
            f"harvest; before it they were {b['sKeepProb']:.3f} / {b['pickProb']:.3f} (set them "
            "in seismology.yaml to compare the synthetic test with and without the harvest)."),
            ""]
    return lines[:-1] if lines[-1] == "" else lines


def build_rows(inputs: DiagnosticsInputs) -> tuple[list[Row], dict[str, list[str]]]:
    ua = used_arrivals(inputs.details, inputs.details.stations)
    pa = picked_arrivals(inputs.details, inputs.details.stations)
    comp = (
        compare_with_catalog(inputs.details.result.events, inputs.catalog,
                             inputs.catalog_errors, inputs.cfg, inputs.known_ids)
        if inputs.catalog is not None else None
    )
    at_catalog = catalog_hypocentre_residuals(inputs, comp) if comp is not None else None
    after_statics = inputs.statics is not None and inputs.statics.pass_number == 2
    before_statics = (catalog_hypocentre_residuals(inputs, comp, with_statics=False)
                      if comp is not None and after_statics else None)
    row1, stations_table = row_borehole(inputs)
    row6, medians_table = row_stations(inputs, ua)
    rows = [row1, row_datum(inputs, comp), row_pyocto(inputs), row_ns(inputs),
            row_profiles(inputs, pa), row6,
            row_trend(inputs, ua, pa, at_catalog, comp, before_statics)]
    dcfg = inputs.cfg.diagnostics

    def trends(frame: pd.DataFrame | None) -> bool:
        return frame is not None and len(frame) > 0 and len(
            _flagged(azimuth_fit(frame), dcfg.minGroupSize, dcfg.trendFlagS)) > 0

    return rows, {"stations": stations_table, "medians": medians_table,
                  "sigma": sigma_section(inputs, pa), "synthetic": synthetic_section(inputs),
                  "tableErrors": table_error_section(inputs, ua),
                  "catalog": catalog_section(inputs, comp, trends(at_catalog),
                                             trends(before_statics)),
                  "statics": statics_section(inputs), "harvest": harvest_section(inputs),
                  "grid3d": grid3d_section(inputs, ua, comp)}


def build_diagnostics(inputs: DiagnosticsInputs) -> str:
    """The Markdown report (see the module docstring)."""
    rows, extra = build_rows(inputs)
    d = inputs.details
    c = d.counts
    rep = inputs.statics
    applied = (f"mode `{rep.mode}` pass {rep.pass_number}"
               + (" (applied)" if rep.pass_number == 2 else " (none applied)")
               if rep is not None else
               "applied" if d.record["locate"]["statics"]["applied"] else "none")
    picks_used = (
        f"{c['picksUsed']} of {c['picksIn']} associated picks used" if d.harvest is None else
        f"{c['picksUsed'] - c['picksHarvestedUsed']} of {c['picksIn']} associated and "
        f"{c['picksHarvestedUsed']} of {c['picksHarvested']} harvested picks used; Pick harvest "
        "section"
    )
    lines = [
        "# Location diagnostics (LOC-04)",
        "",
        (
            f"Run `{inputs.run_id}`, stage `locate`: {c['events']} of {c['assocEvents']} "
            f"association events located as candidate events on {c['stations']} stations "
            f"({picks_used}), method "
            f"`{d.locator.method}` on the `{d.velocity_model.get('name', '?')}` "
            f"{'3D model (1D vs 3D section)' if d.locator.method == GRID3D else 'layer model'}, "
            f"statics {applied} (Station statics section). depthOnEdge "
            f"{c['eventsDepthOnEdge']}, MAP on the volume top {c['eventsMapOnVolumeTop']}, PDF "
            f"truncated {c['eventsPdfTruncated']}."
        ),
        "",
        (
            "Every result below is computed by `hq.locate.diagnostics` from this run's tables and "
            "locator; thresholds are in seismology.yaml (`diagnostics`). Residual = tObs - tPred "
            "at the final location, over the picks used in it unless a row says it also counts "
            "the picks the outlier pass dropped. hErrM / vErrM are formal errors (see the pick "
            "sigma section)."
        ),
        "",
        "## Depth diagnostics (docs/lanes/H2-seismology.md)",
        "",
        "| " + " | ".join(HEADER) + " |",
        "| " + " | ".join("---" for _ in HEADER) + " |",
    ]
    for row in rows:
        suspect, test, fix = SUSPECTS[row.number]
        test = test.format(k=inputs.cfg.diagnostics.minSForDepth)
        cells = [str(row.number), suspect, test, row.result, row.conclusion, fix]
        lines.append("| " + " | ".join(_cell(x) for x in cells) + " |")
    harvest = [*extra["harvest"], ""] if extra["harvest"] else []
    lines += ["", *extra["grid3d"], "", *extra["statics"], "", *harvest, *extra["sigma"], "",
              *extra["synthetic"], "",
              *extra["tableErrors"], "",
              *extra["catalog"], "",
              "## Appendix A: stations (row 1)", "", *extra["stations"], "",
              "## Appendix B: residuals per station and phase (row 6, input for LOC-05)", "",
              *extra["medians"], "",
              "## Configuration", "", "```json",
              json.dumps(inputs.cfg.diagnostics.model_dump(mode="json"), indent=1), "```", ""]
    return "\n".join(lines)

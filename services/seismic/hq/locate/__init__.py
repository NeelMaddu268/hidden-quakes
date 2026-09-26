"""Location (LOC-04): associated events -> located candidate events, arrivals, statics (docs/02 §5).

``locate(assoc, picks, stations, cfg, run) -> LocateResult`` is the docs/02 §5 API; the stage
wrapper (run dir in, tables and ``diagnostics.md`` out) is ``hq.locate.run``, and ``run`` here is
that stage function, because the stage registry (H4's ``hq.runs.STAGES``) resolves stage
``locate`` as the attribute ``hq.locate.run``.

Inputs
    ``assoc``: anything with ``.events`` (assoc_events: ``assocId``, ``t``) and ``.picks``
    (assoc_picks: ``assocId``, ``pickId``), e.g. ``hq.associate.AssocResult``. ``picks``: docs/02
    ``Pick`` rows holding every associated pick id; only ``id, stationId, phase, t, prob`` are
    read, never ``picker``, so PhaseNet and STA/LTA picks run through unchanged. ``stations``:
    ``stations.parquet`` rows; the ``usedInRun`` ones are located against, at ``sensorElevM``,
    and their stored ENU must agree with latitude/longitude/``sensorElevM`` in the run frame.

Per event
    Its associated picks go through ``hq.locate.locator.Locator`` (coarse then fine grid search,
    weighted L1 with the origin time removed analytically, one outlier pass, PDF errors) on the
    per-station 1D tables of the configured layer model (``hq.locate.tt_grid``). Tables are
    cached under ``<cache_dir>/ttgrids/``; without ``cache_dir`` (the docs/02 call) they go to a
    temporary directory kept for the life of the process, so repeated validation reruns in one
    process build them once.

Outputs (``hq.locate.result`` has the dtypes)
    ``events`` (``events_located.parquet``): ``SeismicEvent`` fields except ``tier``,
    ``tierReasons``, ``catalogMatch``, ``magnitude``, flattened (``enu_*``, ``quality_*``). Each
    row is validated as a ``SeismicEvent`` (with a placeholder tier that is dropped). ``id`` is
    ``hq-<runId>-NNNNNN``, numbered from 000000 in origin-time order (ties: assocId); ``runId`` is
    ``run_id`` (the docs/02 call has none, so ``run.name`` stands in); ``source`` is
    ``hq-pipeline``; latitude/longitude come from ``hq.locate.coords.from_enu``; ``depthKm`` is
    ``(run.refSurfaceElevM - elevM) / 1000``; ``quality.method`` is ``grid1d`` and
    ``quality.statics`` is true only when a used pick carried a non-zero static (never before
    LOC-05); ``meanPickProb`` is the mean ``prob`` of the picks used in the final location, which
    are ``pickIds``; ``revealOrder`` is -1.
    ``arrivals`` (``arrivals.parquet``): one row per located event x used station x phase.
    Station-phases without a pick keep ``tPred`` and have null ``tObs``, ``residualS`` and
    ``pickId``; ``usedInLocation`` is false for them and for outlier-dropped picks.
    ``statics`` (``statics.parquet``): one row per used station x phase, ``staticS`` the static
    the locator applied (0.0 until LOC-05) and ``nEvents`` the located events that used a pick of
    that station-phase. With no located event every table has zero rows.

H2-internal table ``locate_flags.parquet`` (not in docs/02; LOC-06 and the depth gate read it)
    One row per located event: ``eventId``, ``assocId`` and the per-event flags docs/02 has no
    column for (``hq.locate.result.FLAG_DTYPES``): MAP on the volume top/bottom
    (``mapOnVolumeTop``/``mapOnVolumeBottom``, the depth gate's "z = 0 collapse" even when a broad
    PDF leaves ``depthOnEdge`` false), which face ``depthOnEdge`` fired on and the face masses,
    PDF truncation (``pdfTruncated``: hErrM/vErrM null), the node budget, grid-limited formal
    errors, and the outlier pass. It is never written into ``events_located.parquet``.

Statics hook (LOC-05)
    ``locate_detailed(..., statics={(stationId, phase): s})`` passes additive statics to the
    locator; they enter every ``tPred`` and the ``staticS`` column. ``locate`` applies none.
"""

import atexit
import logging
import math
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from hq_contracts.io import to_frame
from hq_contracts.models import SeismicEvent

from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig
from hq.locate.coords import from_enu, to_enu
from hq.locate.locator import (
    METHOD,
    PICK_COLUMNS,
    EventLocation,
    Locator,
    LocatorSetup,
    build_locator,
    locate_many,
)
from hq.locate.result import (
    ARRIVAL_DTYPES,
    EVENT_COLUMNS,
    FLAG_DTYPES,
    STATIC_DTYPES,
    LocateResult,
    typed_frame,
)
from hq.locate.tt_grid import PHASES
from hq.locate.velocity import LayerModel, load_configured_model

if TYPE_CHECKING:
    from hq.associate.result import AssocResult

log = logging.getLogger(__name__)

STATION_COLUMNS = (
    "id",
    "latitude",
    "longitude",
    "sensorElevM",
    "usedInRun",
    "enu_e",
    "enu_n",
    "enu_u",
    "preprocessProfile",
)
LOCATOR_STATION_COLUMNS = ("id", "enu_e", "enu_n", "enu_u", "sensorElevM", "preprocessProfile")
SOURCE = "hq-pipeline"  # SeismicEvent.source
REVEAL_ORDER_UNSET = -1  # docs/02: H2 writes -1, the exporter assigns the real order
EVENT_ID_DIGITS = 6
PLACEHOLDER_TIER = "C"  # only to validate rows as SeismicEvent; tier columns are dropped
Statics = Mapping[tuple[str, str], float]

_process_cache: list[Path] = []  # the process-lifetime table cache when no cache_dir is given


def _process_cache_dir() -> Path:
    if not _process_cache:
        path = Path(tempfile.mkdtemp(prefix="hq-locate-ttgrids-"))
        atexit.register(shutil.rmtree, path, True)
        _process_cache.append(path)
        log.warning(
            "locate: no cache_dir given; travel-time tables go to %s for the life of this process "
            "(pass cache_dir=<data>/cache to reuse <data>/cache/ttgrids/)", path,
        )
    return _process_cache[0]


def _missing(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} lacks columns {missing}")


def used_stations(stations: pd.DataFrame, run: RunSection, tol_m: float) -> pd.DataFrame:
    """The ``usedInRun`` stations as the Locator takes them, ENU checked against lat/lon/elev.

    ENU is recomputed from latitude/longitude and ``sensorElevM`` (the sensor, not the wellhead)
    and must agree with the stored ``enu_*`` within ``tol_m``, so a table in another origin,
    projection or elevation fails here instead of shifting every location.
    """
    _missing(stations, STATION_COLUMNS, "stations")
    used = stations[stations["usedInRun"].astype(bool)].reset_index(drop=True)
    if used.empty:
        raise ValueError("stations table has no usedInRun station")
    ids = used["id"].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"duplicate usedInRun station ids: {sorted(ids[ids.duplicated()])}")
    e, n, u = to_enu(
        used["latitude"].to_numpy(dtype=np.float64),
        used["longitude"].to_numpy(dtype=np.float64),
        used["sensorElevM"].to_numpy(dtype=np.float64),
        run.origin,
    )
    stored = used[["enu_e", "enu_n", "enu_u"]].to_numpy(dtype=np.float64)
    off = np.max(np.abs(np.column_stack([e, n, u]) - stored), axis=1)
    if not np.all(off <= tol_m):
        worst = int(np.nanargmax(np.where(np.isfinite(off), off, np.inf)))
        raise ValueError(
            f"station {ids.iloc[worst]}: stored enu differs by {off[worst]:.3f} m from latitude/"
            f"longitude/sensorElevM in the run frame (locator.enuConsistencyTolM {tol_m} m)"
        )
    out = used[list(LOCATOR_STATION_COLUMNS)].copy()
    out["id"] = ids.to_numpy(dtype=object)
    out["preprocessProfile"] = used["preprocessProfile"].astype(str).to_numpy(dtype=object)
    for col in ("enu_e", "enu_n", "enu_u", "sensorElevM"):
        out[col] = out[col].to_numpy(dtype=np.float64)
    return out


def event_picks(
    assoc: "AssocResult", picks: pd.DataFrame, min_picks: int
) -> tuple[list[str], list[pd.DataFrame]]:
    """Per association event (by origin time, then assocId): its id and its picks frame.

    Each picks frame holds ``id, stationId, phase, t, prob`` sorted by pick id. Fails on a pick id
    missing from ``picks``, a pick in two events, an unknown assocId, or an event with fewer than
    ``min_picks`` picks (the config forbids associator minimums that allow one).
    """
    events, links = assoc.events, assoc.picks
    _missing(events, ("assocId", "t"), "assoc events")
    _missing(links, ("assocId", "pickId"), "assoc picks")
    _missing(picks, PICK_COLUMNS, "picks")
    ev = pd.DataFrame(
        {"assocId": events["assocId"].astype(str), "t": events["t"].to_numpy(dtype=np.float64)}
    )
    if ev["assocId"].duplicated().any():
        raise ValueError(f"duplicate assocIds: {sorted(ev['assocId'][ev['assocId'].duplicated()])}")
    link = pd.DataFrame(
        {"assocId": links["assocId"].astype(str), "pickId": links["pickId"].astype(str)}
    )
    unknown = sorted(set(link["assocId"]) - set(ev["assocId"]))
    if unknown:
        raise ValueError(f"assoc picks name events missing from assoc events: {unknown[:5]}")
    if link["pickId"].duplicated().any():
        raise ValueError(
            f"picks in more than one association event: {sorted(link['pickId'][link['pickId'].duplicated()])[:5]}"
        )
    table = picks[list(PICK_COLUMNS)].copy()
    table["id"] = table["id"].astype(str)
    table["stationId"] = table["stationId"].astype(str)
    table["phase"] = table["phase"].astype(str)
    table["t"] = table["t"].to_numpy(dtype=np.float64)
    table["prob"] = table["prob"].to_numpy(dtype=np.float64)
    if table["id"].duplicated().any():
        raise ValueError(f"duplicate pick ids, e.g. {table['id'][table['id'].duplicated()].iloc[0]!r}")
    by_id = table.set_index("id", drop=False)
    missing = sorted(set(link["pickId"]) - set(by_id.index))
    if missing:
        raise ValueError(f"{len(missing)} associated pick ids are not in the picks table: {missing[:5]}")
    groups = {aid: sorted(g) for aid, g in link.groupby("assocId")["pickId"]}
    ordered = ev.sort_values(["t", "assocId"], kind="stable")["assocId"].tolist()
    frames = []
    for aid in ordered:
        ids = groups.get(aid, [])
        if len(ids) < min_picks:
            raise ValueError(
                f"association event {aid} has {len(ids)} picks; the locator needs at least "
                f"locator.minPicks {min_picks}"
            )
        frames.append(by_id.loc[ids].reset_index(drop=True))
    return ordered, frames


@dataclass(frozen=True, eq=False)
class LocateDetails:
    """``locate`` plus what the stage and the diagnostics need."""

    result: LocateResult
    flags: pd.DataFrame  # locate_flags.parquet (FLAG_DTYPES)
    locations: tuple[EventLocation, ...]  # in result.events order
    assoc_ids: tuple[str, ...]  # in result.events order
    locator: Locator
    stations: pd.DataFrame  # the usedInRun stations, as the Locator took them
    counts: dict[str, int]
    record: dict[str, Any]  # ProcessingRun.locator params
    velocity_model: dict[str, Any]  # ProcessingRun.velocityModel: the top-extended model
    runtime_s: float


def _event_row(
    k: int, loc: EventLocation, lat: float, lon: float, probs: dict[str, float],
    run: RunSection, run_id: str,
) -> SeismicEvent:
    used = loc.arrivals.loc[loc.arrivals["usedInLocation"].to_numpy(dtype=bool), "pickId"]
    pick_ids = [str(p) for p in used]
    return SeismicEvent.model_validate(
        {
            "id": f"hq-{run_id}-{k:0{EVENT_ID_DIGITS}d}",
            "runId": run_id,
            "source": SOURCE,
            "t": loc.t0,
            "latitude": lat,
            "longitude": lon,
            "elevM": loc.elev_m,
            "depthKm": (run.refSurfaceElevM - loc.elev_m) / 1000.0,
            "enu": {"e": loc.e_m, "n": loc.n_m, "u": loc.u_m},
            "quality": loc.quality(),
            "tier": PLACEHOLDER_TIER,
            "tierReasons": [],
            "meanPickProb": float(np.mean([probs[p] for p in pick_ids])),
            "revealOrder": REVEAL_ORDER_UNSET,
            "pickIds": pick_ids,
        }
    )


def _flag_row(event_id: str, assoc_id: str, loc: EventLocation, edge_fraction: float) -> dict:
    pdf = loc.pdf
    return {
        "eventId": event_id,
        "assocId": assoc_id,
        "mapOnVolumeTop": loc.map_on_volume_top,
        "mapOnVolumeBottom": loc.map_on_volume_bottom,
        "depthOnEdgeTop": pdf.top_face_mass > edge_fraction,
        "depthOnEdgeBottom": pdf.bottom_face_mass > edge_fraction,
        "topFaceMass": pdf.top_face_mass,
        "bottomFaceMass": pdf.bottom_face_mass,
        "pdfTruncated": loc.pdf_truncated,
        "nodeBudgetHit": bool(loc.search["nodeBudgetHit"]),
        "hErrGridLimited": pdf.h_err_floored,
        "vErrGridLimited": pdf.v_err_floored,
        "nDroppedPicks": len(loc.dropped_pick_ids),
        "outlierPassSkipped": loc.outlier_note is not None,
    }


def _arrival_rows(
    event_id: str, loc: EventLocation, locator: Locator, statics: Statics
) -> list[dict[str, Any]]:
    picked = {(str(a.stationId), str(a.phase)): a for a in loc.arrivals.itertuples(index=False)}
    rows = []
    pred = locator.travel_times(loc.e_m, loc.n_m, loc.elev_m)
    for sid, ph, tt in pred.itertuples(index=False):
        a = picked.get((sid, ph))
        if a is None:
            rows.append(
                {
                    "eventId": event_id,
                    "stationId": sid,
                    "phase": ph,
                    "tPred": loc.t0 + float(tt) + float(statics.get((sid, ph), 0.0)),
                    "tObs": math.nan,
                    "residualS": math.nan,
                    "pickId": None,
                    "usedInLocation": False,
                }
            )
        else:
            rows.append(
                {
                    "eventId": event_id,
                    "stationId": sid,
                    "phase": ph,
                    "tPred": float(a.tPred),
                    "tObs": float(a.tObs),
                    "residualS": float(a.residualS),
                    "pickId": str(a.pickId),
                    "usedInLocation": bool(a.usedInLocation),
                }
            )
    return rows


def locate_detailed(
    assoc: "AssocResult",
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    cfg: SeismologyConfig,
    run: RunSection,
    *,
    run_id: str | None = None,
    cache_dir: Path | None = None,
    statics: Statics | None = None,
    model: LayerModel | None = None,
) -> LocateDetails:
    """``locate`` plus flags, the per-event locations, counts and the run record.

    ``statics`` maps (stationId, phase) to an additive static (s), the LOC-05 hook; ``model``
    replaces the configured layer file (tests).
    """
    started = time.perf_counter()
    rid = run.name if run_id is None else run_id
    applied: Statics = dict(statics or {})
    used = used_stations(stations, run, cfg.locator.enuConsistencyTolM)
    assoc_ids, frames = event_picks(assoc, picks, cfg.locator.minPicks)
    setup = LocatorSetup(
        stations=used,
        model=load_configured_model(cfg.velocity) if model is None else model,
        config=cfg,
        run=run,
        cache_dir=Path(cache_dir) if cache_dir is not None else _process_cache_dir(),
    )
    locator = build_locator(setup)
    located = locate_many(setup, frames, statics=applied, locator=locator) if frames else []

    order = sorted(range(len(located)), key=lambda k: (located[k].t0, assoc_ids[k]))
    locations = tuple(located[k] for k in order)
    ordered_assoc = tuple(assoc_ids[k] for k in order)
    probs = [dict(zip(frames[k]["id"], frames[k]["prob"], strict=True)) for k in order]
    lat, lon, _ = from_enu(
        [loc.e_m for loc in locations],
        [loc.n_m for loc in locations],
        [loc.u_m for loc in locations],
        run.origin,
    )
    models: list[SeismicEvent] = []
    arrivals: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    n_events: Counter[tuple[str, str]] = Counter()
    edge = cfg.locator.depthOnEdgeMassFraction
    for k, loc in enumerate(locations):
        event = _event_row(k, loc, float(lat[k]), float(lon[k]), probs[k], run, rid)
        models.append(event)
        arrivals += _arrival_rows(event.id, loc, locator, applied)
        flags.append(_flag_row(event.id, ordered_assoc[k], loc, edge))
        use = loc.arrivals[loc.arrivals["usedInLocation"].to_numpy(dtype=bool)]
        n_events.update(zip(use["stationId"].astype(str), use["phase"].astype(str), strict=True))
    static_rows = (
        [
            {
                "stationId": sid,
                "phase": ph,
                "staticS": float(applied.get((sid, ph), 0.0)),
                "nEvents": int(n_events[(sid, ph)]),
            }
            for sid in locator.station_ids
            for ph in PHASES
        ]
        if locations
        else []
    )
    events_frame = to_frame(models, SeismicEvent)[list(EVENT_COLUMNS)].reset_index(drop=True)
    result = LocateResult(
        events=events_frame,
        arrivals=typed_frame(arrivals, ARRIVAL_DTYPES),
        statics=typed_frame(static_rows, STATIC_DTYPES),
    )
    flag_frame = typed_frame(flags, FLAG_DTYPES)
    n_picks = sum(len(f) for f in frames)
    n_used = sum(int(loc.arrivals["usedInLocation"].sum()) for loc in locations)
    counts = {
        "assocEvents": len(assoc_ids),
        "events": len(locations),
        "stations": len(used),
        "picksIn": n_picks,
        "picksUsed": n_used,
        "picksDroppedAsOutliers": n_picks - n_used,
        "eventsRelocatedAfterOutliers": sum(loc.relocated for loc in locations),
        "eventsOutlierPassSkipped": int(flag_frame["outlierPassSkipped"].sum()),
        "eventsDepthOnEdge": int(result.events["quality_depthOnEdge"].sum()),
        "eventsMapOnVolumeTop": int(flag_frame["mapOnVolumeTop"].sum()),
        "eventsMapOnVolumeBottom": int(flag_frame["mapOnVolumeBottom"].sum()),
        "eventsPdfTruncated": int(flag_frame["pdfTruncated"].sum()),
        "eventsNodeBudgetHit": int(flag_frame["nodeBudgetHit"].sum()),
        "eventsStaticsApplied": int(result.events["quality_statics"].sum()),
        "arrivals": len(result.arrivals),
        "arrivalsWithPick": int(result.arrivals["pickId"].notna().sum()),
        "statics": len(result.statics),
    }
    pickers = (
        sorted(str(p) for p in picks["picker"].dropna().unique()) if "picker" in picks else None
    )
    record = {
        **locator.to_record(),
        "locate": {
            "runId": rid,
            "eventId": f"hq-<runId>-<{EVENT_ID_DIGITS} digits>, numbered from 0 in origin-time "
            "order (ties: assocId)",
            "source": SOURCE,
            "depthKm": "(run.refSurfaceElevM - elevM) / 1000",
            "refSurfaceElevM": run.refSurfaceElevM,
            "latLon": "hq.locate.coords.from_enu (EPSG:32612 minus the run origin)",
            "pickIds": "picks used in the final location (outlier-dropped picks left out)",
            "meanPickProb": "mean prob of the picks in pickIds (picker confidence)",
            "arrivals": "one row per event x used station x phase; tPred = origin time + table "
            "travel time + static; tObs/residualS/pickId null without a pick; usedInLocation "
            "false for outlier-dropped picks and rows without a pick",
            "statics": {
                "applied": bool(applied),
                "nonZero": int(sum(1 for v in applied.values() if v != 0.0)),
                "note": "additive per (stationId, phase); none applied until LOC-05",
            },
            "flagsTable": "locate_flags.parquet (H2-internal): " + ", ".join(FLAG_DTYPES),
            "stations": {"nUsed": len(used), "ids": locator.station_ids},
            "pickers": pickers,
            "method": METHOD,
        },
    }
    runtime = time.perf_counter() - started
    log.info(
        "locate: %d of %d association events located on %d stations (%d of %d picks used, %d "
        "relocated after outliers; depthOnEdge %d, MAP on volume top %d, PDF truncated %d) in "
        "%.1f s",
        counts["events"], counts["assocEvents"], counts["stations"], n_used, n_picks,
        counts["eventsRelocatedAfterOutliers"], counts["eventsDepthOnEdge"],
        counts["eventsMapOnVolumeTop"], counts["eventsPdfTruncated"], runtime,
    )
    return LocateDetails(
        result=result,
        flags=flag_frame,
        locations=locations,
        assoc_ids=ordered_assoc,
        locator=locator,
        stations=used,
        counts=counts,
        record=record,
        velocity_model=locator.velocity_model_record(),
        runtime_s=runtime,
    )


def locate(
    assoc: "AssocResult",
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    cfg: SeismologyConfig,
    run: RunSection,
    *,
    run_id: str | None = None,
    cache_dir: Path | None = None,
) -> LocateResult:
    """Locate every association event (docs/02 §5); see the module docstring.

    ``run_id`` (keyword only; the docs/02 call leaves it out, and ``run.name`` stands in) goes
    into event ids and ``runId``. ``cache_dir`` caches the travel-time tables under
    ``<cache_dir>/ttgrids/``.
    """
    return locate_detailed(
        assoc, picks, stations, cfg, run, run_id=run_id, cache_dir=cache_dir
    ).result


# Last, so the package attribute ``run`` is the stage function, not the submodule.
from hq.locate.run import run

__all__ = ["LocateDetails", "LocateResult", "locate", "locate_detailed", "run"]

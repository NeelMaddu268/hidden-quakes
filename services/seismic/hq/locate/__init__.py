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
    travel-time tables ``locator.method`` names: the per-station 1D tables of the configured layer
    model (``grid1d``, ``hq.locate.tt_grid``) or the per-station 3D tables of the 3D model
    (``grid3d``, ``hq.locate.tt_grid3d``; 1D tables for stations outside it). Tables are cached
    under ``<cache_dir>/ttgrids/``; without ``cache_dir`` (the docs/02 call) grid1d tables go to a
    temporary directory kept for the life of the process, so repeated validation reruns in one
    process build them once. grid3d reads the 3D model from ``<cache_dir>/velocity/``; without
    ``cache_dir`` it uses the data dir's cache (``default_cache_dir``), where the model lives.

Outputs (``hq.locate.result`` has the dtypes)
    ``events`` (``events_located.parquet``): ``SeismicEvent`` fields except ``tier``,
    ``tierReasons``, ``catalogMatch``, ``magnitude``, flattened (``enu_*``, ``quality_*``). Each
    row is validated as a ``SeismicEvent`` (with a placeholder tier that is dropped). ``id`` is
    ``hq-<runId>-NNNNNN``, numbered from 000000 in origin-time order (ties: assocId); ``runId`` is
    ``run_id`` (the docs/02 call has none, so ``run.name`` stands in); ``source`` is
    ``hq-pipeline``; latitude/longitude come from ``hq.locate.coords.from_enu``; ``depthKm`` is
    ``(run.refSurfaceElevM - elevM) / 1000``; ``quality.method`` is ``locator.method`` and
    ``quality.statics`` is true only when a used pick carried a non-zero static;
    ``meanPickProb`` is the mean ``prob`` of the picks used in the final location, which
    are ``pickIds``; ``revealOrder`` is -1.
    ``arrivals`` (``arrivals.parquet``): one row per located event x used station x phase.
    Station-phases without a pick keep ``tPred`` and have null ``tObs``, ``residualS`` and
    ``pickId``; ``usedInLocation`` is false for them and for outlier-dropped picks.
    ``statics`` (``statics.parquet``): one row per used station x phase, ``staticS`` the static
    the locator applied to every event without its own (0.0 without statics) and ``nEvents`` the
    events that static was estimated from (``hq.locate.statics``); without statics, the located
    events that used a pick of that station-phase. With no located event every table has zero
    rows.

H2-internal table ``locate_flags.parquet`` (not in docs/02; LOC-06 and the depth gate read it)
    One row per located event: ``eventId``, ``assocId`` and the per-event flags docs/02 has no
    column for (``hq.locate.result.FLAG_DTYPES``): MAP on the volume top/bottom
    (``mapOnVolumeTop``/``mapOnVolumeBottom``, the depth gate's "z = 0 collapse" even when a broad
    PDF leaves ``depthOnEdge`` false), which face ``depthOnEdge`` fired on and the face masses,
    PDF truncation (``pdfTruncated``: hErrM/vErrM null), the node budget, grid-limited formal
    errors, the outlier pass, and the local-ground check: with no DEM, the ground above an event
    is proxied by the ``surfaceElevM`` of the epicentrally nearest used station
    (``nearestStationSurfaceElevM``), and ``aboveNearestStationSurface`` flags a hypocentre above
    it. The search volume's top is ``run.refSurfaceElevM``, which lies above the ground at the
    lower stations, so this flag catches what ``mapOnVolumeTop`` can't. It is never written into
    ``events_located.parquet``.

Statics (LOC-05, ``hq.locate.statics``)
    ``locate_detailed(..., statics={(stationId, phase): s})`` passes additive statics to the
    locator; they enter every ``tPred`` and the ``staticS`` column; ``event_statics`` gives
    single events (by assocId) their own. ``locate`` applies what ``statics.mode`` says. Pipeline
    order with the default mode (referenceEvents): locate (pass 1, no statics) -> match -> locate
    (pass 2, terms at the public-catalog hypocentres of the matched events, each matched event
    relocated with terms computed without it) -> match -> tier. Stage ``locate`` runs pass 2 when
    ``matches.parquet`` is in the run dir; ``locate()`` has no matches and runs pass 1.

Pick harvest (LOC-10, ``hq.locate.harvest``, off unless ``harvest.enabled``)
    Every statics-corrected locate of a whole association (stage locate pass 2, selfConsistent's
    last iteration, ``locate(statics=...)`` and the tier sweep) passes ``harvest=True``: picks in
    no association event within ``harvest.windowS`` of an event's predicted arrival at a
    station-phase it has no pick for are added and the events that gained picks relocated. They
    appear only as extra ``pickIds``, filled ``arrivals`` rows (``tObs``, ``residualS``,
    ``pickId``) and more ``event_picks`` rows: recover them as an event's ``pickIds`` minus its
    association's picks. ``counts`` and the record (``harvest``) gain harvest keys only when it
    ran, so with it off every output is as before.
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
    GRID3D,
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
from hq.locate.tt_grid3d import Model3dSource
from hq.locate.velocity import LayerModel, load_configured_model

if TYPE_CHECKING:
    from hq.associate.result import AssocResult
    from hq.locate.harvest import HarvestReport

log = logging.getLogger(__name__)

STATION_COLUMNS = (
    "id",
    "latitude",
    "longitude",
    "surfaceElevM",
    "sensorElevM",
    "usedInRun",
    "enu_e",
    "enu_n",
    "enu_u",
    "preprocessProfile",
)
LOCATOR_STATION_COLUMNS = (
    "id", "enu_e", "enu_n", "enu_u", "sensorElevM", "surfaceElevM", "preprocessProfile"
)
SOURCE = "hq-pipeline"  # SeismicEvent.source
REVEAL_ORDER_UNSET = -1  # docs/02: H2 writes -1, the exporter assigns the real order
EVENT_ID_DIGITS = 6
PLACEHOLDER_TIER = "C"  # only to validate rows as SeismicEvent; tier columns are dropped
Statics = Mapping[tuple[str, str], float]

_process_cache: list[Path] = []  # the process-lifetime table cache when no cache_dir is given
_data_cache_logged: list[Path] = []  # grid3d data caches already logged (default_cache_dir)


def default_cache_dir(cfg: SeismologyConfig) -> Path:
    """The table cache when the caller passes no ``cache_dir`` (the docs/02 call).

    grid1d: a temporary directory kept for the life of the process. grid3d: the data dir's cache
    as ``hq run`` resolves it (``hq.cli.resolve_data_dir``: ``$HQ_DATA_DIR``, else
    ``<checkout root>/data``), because the 3D model file lives in its ``velocity/`` and the 3D
    tables (minutes to build at 100 m) are shared through its ``ttgrids/3d/``. Raises when no
    data dir resolves; a missing model file fails when the model is opened.
    """
    if cfg.locator.method != GRID3D:
        return _process_cache_dir()
    from hq.cli import resolve_data_dir  # H4's data-dir rule; imported here to avoid a cycle
    from hq.runs import CACHE_DIRNAME

    cache = resolve_data_dir(None, Path(__file__).resolve().parent) / CACHE_DIRNAME
    if cache not in _data_cache_logged:
        _data_cache_logged.append(cache)
        log.warning("locate: no cache_dir given and locator.method is grid3d: the 3D model and "
                    "the tables come from %s", cache)
    return cache


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
    for col in ("enu_e", "enu_n", "enu_u", "sensorElevM", "surfaceElevM"):
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
    harvest: "HarvestReport | None" = None  # set only when the pick harvest ran (LOC-10)


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


def nearest_station_surface(stations: pd.DataFrame, e_m: float, n_m: float) -> float:
    """``surfaceElevM`` of the station epicentrally nearest to (e, n): the local-ground proxy."""
    d = np.hypot(stations["enu_e"].to_numpy(dtype=np.float64) - e_m,
                 stations["enu_n"].to_numpy(dtype=np.float64) - n_m)
    return float(stations["surfaceElevM"].to_numpy(dtype=np.float64)[int(np.argmin(d))])


def _flag_row(
    event_id: str, assoc_id: str, loc: EventLocation, edge_fraction: float, surface_m: float
) -> dict:
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
        "nearestStationSurfaceElevM": surface_m,
        "aboveNearestStationSurface": loc.elev_m > surface_m,
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
    event_statics: Mapping[str, Statics] | None = None,
    static_events: Mapping[tuple[str, str], int] | None = None,
    model: LayerModel | None = None,
    model3d: Model3dSource | None = None,
    harvest: bool = False,
) -> LocateDetails:
    """``locate`` plus flags, the per-event locations, counts and the run record.

    ``statics`` maps (stationId, phase) to an additive static (s) for every event;
    ``event_statics`` (keyed by assocId) gives those events their own map instead (LOC-05's
    held-out reference terms). ``static_events`` is the ``nEvents`` column of the statics table:
    how many events each static was estimated from (default: the located events that used a
    pick of that station-phase). ``model`` replaces the configured layer file and ``model3d``
    the configured 3D model file (tests).

    ``harvest`` (LOC-10): when true and ``harvest.enabled``, picks in no association event are
    harvested at the events' predicted arrivals and the events that gained picks relocated
    (``hq.locate.harvest``); ``details.harvest`` then holds the report, and the counts and the
    record gain harvest keys. Callers pass it only on a statics-corrected locate of the whole
    association (every association event, so the free picks are those of no event); otherwise,
    and with ``harvest.enabled`` false, nothing changes.
    """
    started = time.perf_counter()
    rid = run.name if run_id is None else run_id
    applied: Statics = dict(statics or {})
    own = {str(k): dict(v) for k, v in (event_statics or {}).items()}
    used = used_stations(stations, run, cfg.locator.enuConsistencyTolM)
    assoc_ids, frames = event_picks(assoc, picks, cfg.locator.minPicks)
    unknown = sorted(set(own) - set(assoc_ids))
    if unknown:
        raise ValueError(f"event_statics name association events not in assoc: {unknown[:5]}")
    per_event = [own.get(aid, applied) for aid in assoc_ids]
    setup = LocatorSetup(
        stations=used,
        model=load_configured_model(cfg.velocity) if model is None else model,
        config=cfg,
        run=run,
        cache_dir=Path(cache_dir) if cache_dir is not None else default_cache_dir(cfg),
        model3d=model3d,
    )
    locator = build_locator(setup)
    located = (
        locate_many(setup, frames, event_statics=per_event, locator=locator) if frames else []
    )
    n_picks = sum(len(f) for f in frames)  # associated picks
    harvested: HarvestReport | None = None
    if harvest and cfg.harvest.enabled and located:
        from hq.locate.harvest import harvest_and_relocate  # imports hq.locate.locator only

        frames, located, harvested = harvest_and_relocate(
            setup, locator, assoc_ids, frames, located, per_event, picks,
            set(assoc.picks["pickId"].astype(str)), cfg.harvest,
        )

    order = sorted(range(len(located)), key=lambda k: (located[k].t0, assoc_ids[k]))
    locations = tuple(located[k] for k in order)
    ordered_assoc = tuple(assoc_ids[k] for k in order)
    probs = [dict(zip(frames[k]["id"], frames[k]["prob"], strict=True)) for k in order]
    event_maps = [per_event[k] for k in order]
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
        arrivals += _arrival_rows(event.id, loc, locator, event_maps[k])
        surface = nearest_station_surface(used, loc.e_m, loc.n_m)
        flags.append(_flag_row(event.id, ordered_assoc[k], loc, edge, surface))
        use = loc.arrivals[loc.arrivals["usedInLocation"].to_numpy(dtype=bool)]
        n_events.update(zip(use["stationId"].astype(str), use["phase"].astype(str), strict=True))
    static_rows = (
        [
            {
                "stationId": sid,
                "phase": ph,
                "staticS": float(applied.get((sid, ph), 0.0)),
                "nEvents": int(
                    n_events[(sid, ph)] if static_events is None
                    else static_events.get((sid, ph), 0)
                ),
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
    n_all = sum(len(f) for f in frames)  # associated plus harvested picks
    n_used = sum(int(loc.arrivals["usedInLocation"].sum()) for loc in locations)
    counts = {
        "assocEvents": len(assoc_ids),
        "events": len(locations),
        "stations": len(used),
        "picksIn": n_picks,
        "picksUsed": n_used,
        "picksDroppedAsOutliers": n_all - n_used,
        "eventsRelocatedAfterOutliers": sum(loc.relocated for loc in locations),
        "eventsOutlierPassSkipped": int(flag_frame["outlierPassSkipped"].sum()),
        "eventsDepthOnEdge": int(result.events["quality_depthOnEdge"].sum()),
        "eventsMapOnVolumeTop": int(flag_frame["mapOnVolumeTop"].sum()),
        "eventsMapOnVolumeBottom": int(flag_frame["mapOnVolumeBottom"].sum()),
        "eventsPdfTruncated": int(flag_frame["pdfTruncated"].sum()),
        "eventsNodeBudgetHit": int(flag_frame["nodeBudgetHit"].sum()),
        "eventsAboveNearestStationSurface": int(flag_frame["aboveNearestStationSurface"].sum()),
        "eventsStaticsApplied": int(result.events["quality_statics"].sum()),
        "arrivals": len(result.arrivals),
        "arrivalsWithPick": int(result.arrivals["pickId"].notna().sum()),
        "statics": len(result.statics),
    }
    if harvested is not None:
        hc = harvested.counts
        counts.update({
            "picksHarvested": hc["picks"], "picksHarvestedUsed": hc["picksUsed"],
            "eventsHarvested": hc["events"], "harvestAmbiguousPicks": hc["ambiguousPicks"],
            "harvestSkippedUntrustedSlots": hc["skippedUntrustedSlots"],
            "harvestSkippedMultiCandidateSlots": hc["skippedMultiCandidateSlots"],
        })
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
                "applied": bool(applied) or any(bool(v) for v in own.values()),
                "nonZero": int(sum(1 for v in applied.values() if v != 0.0)),
                "eventsWithOwnStatics": len(own),
                "note": "additive per (stationId, phase); the statics table holds the map every "
                "event without its own used (hq.locate.statics)",
            },
            "flagsTable": "locate_flags.parquet (H2-internal): " + ", ".join(FLAG_DTYPES),
            "stations": {"nUsed": len(used), "ids": locator.station_ids},
            "pickers": pickers,
            "method": locator.method,
        },
    }
    if harvested is not None:
        record["harvest"] = harvested.to_record(
            dict(zip(ordered_assoc, result.events["id"].astype(str), strict=True)))
    runtime = time.perf_counter() - started
    log.info(
        "locate: %d of %d association events located on %d stations (%d of %d picks used, %d "
        "relocated after outliers; depthOnEdge %d, MAP on volume top %d, PDF truncated %d) in "
        "%.1f s",
        counts["events"], counts["assocEvents"], counts["stations"], n_used, n_all,
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
        harvest=harvested,
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
    statics: pd.DataFrame | None = None,
) -> LocateResult:
    """Locate every association event (docs/02 §5); see the module docstring.

    ``run_id`` (keyword only; the docs/02 call leaves it out, and ``run.name`` stands in) goes
    into event ids and ``runId``. ``cache_dir`` caches the travel-time tables under
    ``<cache_dir>/ttgrids/``. Without ``statics``, statics follow ``statics.mode``
    (``hq.locate.statics``): selfConsistent iterates them here; referenceEvents needs a match
    pass this call has no access to, so it locates without statics (pass 1).

    ``statics`` (keyword only): a ``statics.parquet``-shaped table (``stationId``, ``phase``,
    ``staticS``) of fixed station terms, applied additively to every event instead of
    ``statics.mode``; station-phases not in it get 0. Validation reruns pass the showcase run's
    own ``statics.parquet`` so their events are located, and so graded by the run's tier bars,
    on the same scale as the run's events (REQ-H1-5). With ``harvest.enabled`` this path also
    harvests picks (LOC-10), as stage locate's pass 2 does.

    With ``locator.nWorkers`` above 1 the events are located in spawned worker processes, which
    re-import the caller's ``__main__``: call this from a script whose top level is guarded by
    ``if __name__ == "__main__":`` (the CLI entry points are), or the pool breaks
    (``BrokenProcessPool``). The results do not depend on ``nWorkers``.
    """
    if statics is not None:
        from hq.locate.statics import statics_map  # imports this package

        missing = {"stationId", "phase", "staticS"} - set(statics.columns)
        if missing:
            raise ValueError(f"locate: statics table lacks columns {sorted(missing)}")
        if statics["staticS"].isna().any():
            raise ValueError("locate: statics table has null staticS values")
        return locate_detailed(
            assoc, picks, stations, cfg, run, run_id=run_id, cache_dir=cache_dir,
            statics=statics_map(statics), harvest=True,
        ).result

    from hq.locate.statics import locate_with_statics  # imports this package

    return locate_with_statics(
        assoc, picks, stations, cfg, run, run_id=run_id, cache_dir=cache_dir
    ).details.result


# Last, so the package attribute ``run`` is the stage function, not the submodule.
from hq.locate.run import run

__all__ = ["LocateDetails", "LocateResult", "locate", "locate_detailed", "run"]

"""Association with PyOcto 0.2.0: inputs, the PyOcto call, duplicate merge and outputs (LOC-03).

Flow for one setting of the knobs:

1. Picks: validated (docs/02 ``Pick`` columns), kept at ``prob >= minPickProb``. PyOcto 0.2.0 has
   no per-pick weight (its ``Pick`` is index, time, station, phase), so ``prob`` is used only as
   that threshold. Nothing depends on ``picker``: STA/LTA picks run through unchanged.
2. PyOcto with ``StationSpecificVelocityModel1D`` tables built from the layer model
   (``hq.associate.tables``), stations at ``sensorElevM``, in the frame of ``hq.associate.frame``.
3. Duplicate merge: events whose origin times differ by at most ``mergeWithinS`` and that share at
   least ``mergeMinSharedFraction`` of the smaller event's picks become one (transitively). The
   member with the most picks (then the earlier, then the lower PyOcto index) keeps its origin;
   the others add the picks of station-phases it lacks.
4. Shared picks: a pick still in two events stays with the one where its PyOcto residual is
   smallest in magnitude (ties: more picks, earlier, lower index).
5. Minimums: every event is recounted and must meet PyOcto's pick minimums and ``minStations``.
6. Output: ``elevM = -z * 1000``, latitude/longitude from ``(x, y) * 1000`` through
   ``hq.locate.coords``, ``assocId`` numbered in origin-time order.
"""

import logging
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyocto

from hq.associate.frame import (
    SearchVolume,
    elev_m_from_pyocto_z,
    enu_from_pyocto_xy,
    pyocto_xy_km,
    pyocto_z_km,
    search_volume,
)
from hq.associate.result import (
    EVENT_DTYPES,
    PICK_DTYPES,
    AssocResult,
    empty_result,
    typed_frame,
)
from hq.associate.tables import TableSet, build_tables
from hq.config.run import Origin, RunSection
from hq.config.seismology import AssociatorConfig, SeismologyConfig
from hq.locate.coords import from_enu, to_enu
from hq.locate.velocity import LayerModel, load_configured_model

log = logging.getLogger(__name__)

PICK_COLUMNS = ("id", "stationId", "phase", "t", "prob")  # the docs/02 Pick fields used here
STATION_COLUMNS = ("id", "latitude", "longitude", "sensorElevM", "usedInRun", "enu_e", "enu_n", "enu_u")
PHASES = ("P", "S")
ASSOC_ID_PREFIX = "assoc-"
ASSOC_ID_DIGITS = 6


@dataclass(frozen=True)
class Setup:
    """Everything that does not change between association runs on one station set."""

    stations: pd.DataFrame  # PyOcto station frame: id, x, y, z (km)
    volume: SearchVolume
    tables: TableSet
    origin: Origin
    velocity_model: dict[str, Any]  # LayerModel.to_record() of the (extended) model used
    station_note: dict[str, Any]


@dataclass(frozen=True)
class RawAssociation:
    """PyOcto's output for one pick set, before the merge (reused across ``minStations``)."""

    events: pd.DataFrame  # eid, t, x, y, z
    assignments: pd.DataFrame  # eid, pickId, station, phase, residual
    picks_in: int
    picks_used: int
    runtime_s: float


# --- inputs ------------------------------------------------------------------------------------


def _missing(frame: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} table lacks columns {missing}")


def station_geometry(
    stations: pd.DataFrame, run: RunSection, tol_m: float
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Stations used in the run: ids, ENU e/n (m) and sensorElevM, checked against stored ENU.

    ENU is recomputed from latitude/longitude and ``sensorElevM`` (the sensor, not the wellhead)
    and must agree with the stored ``enu_*`` within ``tol_m``, so a table built with another
    origin, projection or elevation fails here instead of shifting every location.
    """
    _missing(stations, STATION_COLUMNS, "stations")
    used = stations[stations["usedInRun"].astype(bool)]
    ids = [str(s) for s in used["id"]]
    if len(set(ids)) != len(ids):
        raise ValueError("stations table has duplicate ids among usedInRun stations")
    if not ids:
        raise ValueError("stations table has no usedInRun station")
    lat = used["latitude"].to_numpy(dtype=np.float64)
    lon = used["longitude"].to_numpy(dtype=np.float64)
    elev = used["sensorElevM"].to_numpy(dtype=np.float64)
    e, n, u = to_enu(lat, lon, elev, run.origin)
    stored = used[["enu_e", "enu_n", "enu_u"]].to_numpy(dtype=np.float64)
    off = np.max(np.abs(np.column_stack([e, n, u]) - stored), axis=1)
    if not np.all(off <= tol_m):
        worst = int(np.argmax(off))
        raise ValueError(
            f"station {ids[worst]}: stored enu differs by {off[worst]:.3f} m from latitude/"
            f"longitude/sensorElevM in the run frame (tolerance {tol_m} m)"
        )
    note = {
        "nStations": len(ids),
        "nNotUsedInRun": int(len(stations) - len(used)),
        "elevationField": "sensorElevM",
        "stationTermsS": "none: PyOcto adds 0 s for every station (statics come after location)",
    }
    return ids, np.asarray(e), np.asarray(n), elev, note


def select_picks(picks: pd.DataFrame, min_prob: float, station_ids: set[str]) -> pd.DataFrame:
    """Validated picks at ``prob >= min_prob`` in PyOcto's format plus ``pickId``."""
    _missing(picks, PICK_COLUMNS, "picks")
    ids = picks["id"].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"duplicate pick ids, e.g. {ids[ids.duplicated()].iloc[0]!r}")
    phase = picks["phase"].astype(str)
    bad_phase = sorted(set(phase) - set(PHASES))
    if bad_phase:
        raise ValueError(f"pick phases must be P or S, got {bad_phase}")
    t = picks["t"].to_numpy(dtype=np.float64)
    prob = picks["prob"].to_numpy(dtype=np.float64)
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(prob))):
        raise ValueError("pick t and prob must be finite")
    if np.any((prob < 0) | (prob > 1)):
        raise ValueError("pick prob must lie in [0, 1]")
    station = picks["stationId"].astype(str)
    unknown = sorted(set(station) - station_ids)
    if unknown:
        raise ValueError(f"picks reference stations not used in the run: {unknown}")
    keep = prob >= min_prob
    return pd.DataFrame(
        {
            "station": station[keep].to_numpy(dtype=object),
            "phase": phase[keep].to_numpy(dtype=object),
            "time": t[keep],
            "pickId": ids[keep].to_numpy(dtype=object),
        }
    )


# --- setup ---------------------------------------------------------------------------------------


@contextmanager
def prepared(
    stations: pd.DataFrame,
    cfg: SeismologyConfig,
    run: RunSection,
    *,
    model: LayerModel | None = None,
    cache_dir: Path | None = None,
) -> Iterator[Setup]:
    """Stations, search volume and PyOcto tables, valid inside the ``with`` block.

    ``model`` defaults to the configured layer file. ``cache_dir`` None builds the tables in a
    temporary directory removed on exit; otherwise they are cached under ``<cache_dir>/ttgrids/``.
    """
    acfg = cfg.associator
    ids, e, n, elev, note = station_geometry(stations, run, acfg.enuConsistencyTolM)
    volume = search_volume(acfg.volume, run)
    base = load_configured_model(cfg.velocity) if model is None else model
    extended = base.with_top_extended_to(
        max(float(elev.max()), volume.top_elev_m), max_extension_m=cfg.velocity.maxTopExtensionM
    )
    farthest = volume.farthest_horizontal_m(e, n)
    x, y = pyocto_xy_km(e, n)
    frame = pd.DataFrame({"id": ids, "x": x, "y": y, "z": pyocto_z_km(elev)})

    def _setup(root: Path) -> Setup:
        tables = build_tables(extended, ids, elev, farthest, volume, acfg.tables, root)
        return Setup(
            stations=frame,
            volume=volume,
            tables=tables,
            origin=run.origin,
            velocity_model=extended.to_record(),
            station_note=note,
        )

    if cache_dir is None:
        with tempfile.TemporaryDirectory(prefix="hq-pyocto-") as tmp:
            yield _setup(Path(tmp))
    else:
        yield _setup(Path(cache_dir))


# --- PyOcto --------------------------------------------------------------------------------------


def velocity_kwargs(acfg: AssociatorConfig, setup: Setup) -> dict[str, Any]:
    """Every argument passed to ``pyocto.StationSpecificVelocityModel1D``."""
    return {
        "path": str(setup.tables.directory),
        "tolerance": acfg.toleranceS,
        "association_cutoff_distance": acfg.associationCutoffDistanceKm,
    }


def associator_kwargs(acfg: AssociatorConfig, setup: Setup) -> dict[str, Any]:
    """Every argument passed to ``pyocto.OctoAssociator`` except ``velocity_model``."""
    vol = setup.volume
    return {
        "xlim": vol.xlim_km,
        "ylim": vol.ylim_km,
        "zlim": vol.zlim_km,
        "time_before": acfg.timeBeforeS,
        "min_node_size": acfg.minNodeSizeKm,
        "min_node_size_location": acfg.minNodeSizeLocationKm,
        "pick_match_tolerance": acfg.pickMatchToleranceS,
        "min_interevent_time": acfg.minIntereventTimeS,
        "exponential_edt": acfg.exponentialEdt,
        "edt_pick_std": acfg.edtPickStdS,
        "max_pick_overlap": acfg.maxPickOverlap,
        "n_picks": acfg.nPicks,
        "n_p_picks": acfg.nPPicks,
        "n_s_picks": acfg.nSPicks,
        "n_p_and_s_picks": acfg.nPAndSPicks,
        "refinement_iterations": acfg.refinementIterations,
        "time_slicing": acfg.timeSlicingS,
        "node_log_interval": acfg.nodeLogInterval,
        "queue_memory_protection_dfs_size": acfg.queueMemoryProtectionDfsSize,
        "location_split_depth": acfg.locationSplitDepth,
        "location_split_return": acfg.locationSplitReturn,
        "min_pick_fraction": acfg.minPickFraction,
        "second_pass_overwrites": None,
        "n_threads": acfg.nThreads,
        "velocity_model_location": None,  # PyOcto: same model as for association
        "crs": None,  # coordinates are projected here (hq.associate.frame)
    }


def check_time_before(acfg: AssociatorConfig, setup: Setup) -> None:
    largest = setup.tables.max_s_time_in_volume_s
    if acfg.timeBeforeS < largest:
        raise ValueError(
            f"timeBeforeS {acfg.timeBeforeS} s is below the largest S travel time from a station "
            f"to the search volume ({largest:.2f} s); PyOcto would miss events across time slices"
        )


def run_pyocto(picks: pd.DataFrame, setup: Setup, acfg: AssociatorConfig) -> RawAssociation:
    """Filter ``picks`` (docs/02 Pick rows) at ``acfg.minPickProb`` and run PyOcto on them."""
    started = time.perf_counter()
    check_time_before(acfg, setup)
    selected = select_picks(picks, acfg.minPickProb, set(setup.stations["id"]))
    empty_events = pd.DataFrame(
        {"eid": pd.Series([], dtype="int64"), **{c: pd.Series([], dtype="float64")
                                               for c in ("t", "x", "y", "z")}}
    )
    empty_assign = pd.DataFrame(
        {
            "eid": pd.Series([], dtype="int64"),
            "pickId": pd.Series([], dtype=object),
            "station": pd.Series([], dtype=object),
            "phase": pd.Series([], dtype=object),
            "residual": pd.Series([], dtype="float64"),
        }
    )
    if selected.empty:
        # PyOcto 0.2.0 indexes the first pick unconditionally; never call it without picks.
        return RawAssociation(empty_events, empty_assign, len(picks), 0,
                              time.perf_counter() - started)
    associator = pyocto.OctoAssociator(
        velocity_model=pyocto.StationSpecificVelocityModel1D(**velocity_kwargs(acfg, setup)),
        **associator_kwargs(acfg, setup),
    )
    events, assignments = associator.associate(selected, setup.stations)
    if len(events) == 0:
        raw_events, raw_assign = empty_events, empty_assign
    else:
        raw_events = pd.DataFrame(
            {
                "eid": events["idx"].to_numpy(dtype=np.int64),
                "t": events["time"].to_numpy(dtype=np.float64),
                "x": events["x"].to_numpy(dtype=np.float64),
                "y": events["y"].to_numpy(dtype=np.float64),
                "z": events["z"].to_numpy(dtype=np.float64),
            }
        )
        raw_assign = pd.DataFrame(
            {
                "eid": assignments["event_idx"].to_numpy(dtype=np.int64),
                "pickId": assignments["pickId"].to_numpy(dtype=object),
                "station": assignments["station"].to_numpy(dtype=object),
                "phase": assignments["phase"].to_numpy(dtype=object),
                "residual": assignments["residual"].to_numpy(dtype=np.float64),
            }
        )
    runtime = time.perf_counter() - started
    log.info(
        "associate: PyOcto found %d events with %d picks from %d picks at prob >= %s "
        "(%d in) in %.2f s",
        len(raw_events), len(raw_assign), len(selected), acfg.minPickProb, len(picks), runtime,
    )
    return RawAssociation(raw_events, raw_assign, len(picks), len(selected), runtime)


# --- post-processing -----------------------------------------------------------------------------


def _check_one_pick_per_station_phase(assign: pd.DataFrame) -> None:
    dup = assign.duplicated(["eid", "station", "phase"])
    if dup.any():
        row = assign[dup].iloc[0]
        raise RuntimeError(
            f"event {row['eid']} holds two {row['phase']} picks from {row['station']}"
        )


def merge_duplicates(
    events: pd.DataFrame, assign: pd.DataFrame, within_s: float, min_shared_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Merge duplicate events (module docstring, step 3). Returns events, assignments, n merged."""
    if events.empty:
        return events, assign, 0
    picks_of = {eid: set(g) for eid, g in assign.groupby("eid")["pickId"]}
    n_picks = {int(eid): len(picks_of.get(eid, ())) for eid in events["eid"]}
    order = events.sort_values(["t", "eid"], kind="stable")
    eids = order["eid"].to_numpy(dtype=np.int64)
    times = order["t"].to_numpy(dtype=np.float64)
    parent = {int(e): int(e) for e in eids}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(eids)):
        a = int(eids[i])
        for j in range(i + 1, len(eids)):
            if times[j] - times[i] > within_s:
                break
            b = int(eids[j])
            shared = len(picks_of.get(a, set()) & picks_of.get(b, set()))
            if shared and shared >= min_shared_fraction * min(n_picks[a], n_picks[b]):
                parent[find(a)] = find(b)

    t_of = dict(zip(events["eid"].astype(int), events["t"], strict=True))
    groups: dict[int, list[int]] = {}
    for e in eids:
        groups.setdefault(find(int(e)), []).append(int(e))
    rank = {e: (-n_picks[e], t_of[e], e) for e in n_picks}
    keep_rows: list[pd.DataFrame] = []
    keep_events: list[int] = []
    merged = 0
    for members in groups.values():
        members.sort(key=lambda e: rank[e])
        primary = members[0]
        keep_events.append(primary)
        if len(members) == 1:
            keep_rows.append(assign[assign["eid"] == primary])
            continue
        merged += len(members) - 1
        rows = assign[assign["eid"].isin(members)].copy()
        rows["_rank"] = rows["eid"].map({e: k for k, e in enumerate(members)})
        rows = rows.sort_values(["_rank", "pickId"], kind="stable")
        rows = rows.drop_duplicates("pickId").drop_duplicates(["station", "phase"])
        rows = rows.drop(columns="_rank").assign(eid=primary)
        keep_rows.append(rows)
    out_events = events[events["eid"].isin(keep_events)].reset_index(drop=True)
    out_assign = (
        pd.concat(keep_rows, ignore_index=True) if keep_rows else assign.iloc[0:0]
    )
    _check_one_pick_per_station_phase(out_assign)
    return out_events, out_assign, merged


def resolve_shared_picks(
    events: pd.DataFrame, assign: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Keep every pick in one event only (module docstring, step 4). Returns rows, n removed."""
    if assign.empty:
        return assign, 0
    size = assign.groupby("eid")["pickId"].transform("size")
    t_of = assign["eid"].map(dict(zip(events["eid"], events["t"], strict=True)))
    ranked = assign.assign(
        _abs=assign["residual"].abs(), _size=-size, _t=t_of
    ).sort_values(["pickId", "_abs", "_size", "_t", "eid"], kind="stable")
    kept = ranked.drop_duplicates("pickId").drop(columns=["_abs", "_size", "_t"])
    return kept.sort_values(["eid", "pickId"], kind="stable").reset_index(drop=True), len(
        assign
    ) - len(kept)


def pick_counts(assign: pd.DataFrame) -> pd.DataFrame:
    """Per event: nPicks, nP, nS (stations with that phase), nPS (stations with both), nStations."""
    if assign.empty:
        return pd.DataFrame(columns=["nPicks", "nP", "nS", "nPS", "nStations"], dtype="int64")
    per = assign.groupby(["eid", "station", "phase"]).size().unstack("phase", fill_value=0)
    per = per.reindex(columns=list(PHASES), fill_value=0)
    has_p, has_s = per["P"] > 0, per["S"] > 0
    grouped = pd.DataFrame({"P": has_p, "S": has_s, "PS": has_p & has_s}).groupby(level="eid")
    counts = pd.DataFrame(
        {
            "nPicks": assign.groupby("eid").size(),
            "nP": grouped["P"].sum(),
            "nS": grouped["S"].sum(),
            "nPS": grouped["PS"].sum(),
            "nStations": grouped.size(),
        }
    )
    return counts.astype("int64")


def apply_minimums(
    events: pd.DataFrame, assign: pd.DataFrame, acfg: AssociatorConfig
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Drop events below PyOcto's minimums or ``minStations`` after the merge and resolution.

    Returns events, assignments, the number dropped for PyOcto's minimums and for minStations.
    """
    counts = pick_counts(assign).reindex(events["eid"], fill_value=0)
    pyocto_ok = (
        (counts["nPicks"] >= acfg.nPicks)
        & (counts["nP"] >= acfg.nPPicks)
        & (counts["nS"] >= acfg.nSPicks)
        & (counts["nPS"] >= acfg.nPAndSPicks)
    )
    station_ok = counts["nStations"] >= acfg.minStations
    keep = counts.index[pyocto_ok & station_ok]
    dropped_pyocto = int((~pyocto_ok).sum())
    dropped_stations = int((pyocto_ok & ~station_ok).sum())
    out_events = events[events["eid"].isin(keep)].reset_index(drop=True)
    out_assign = assign[assign["eid"].isin(keep)].reset_index(drop=True)
    return out_events, out_assign, dropped_pyocto, dropped_stations


def to_result(events: pd.DataFrame, assign: pd.DataFrame, origin: Origin) -> AssocResult:
    """Build the docs/02 frames; ``assocId`` numbers events by origin time, then first pick id."""
    if events.empty:
        return empty_result()
    first_pick = assign.groupby("eid")["pickId"].min()
    ordered = events.assign(_first=events["eid"].map(first_pick)).sort_values(
        ["t", "_first"], kind="stable"
    )
    ids = [f"{ASSOC_ID_PREFIX}{k:0{ASSOC_ID_DIGITS}d}" for k in range(len(ordered))]
    id_of = dict(zip(ordered["eid"], ids, strict=True))
    e_m, n_m = enu_from_pyocto_xy(ordered["x"], ordered["y"])
    elev_m = elev_m_from_pyocto_z(ordered["z"])
    lat, lon, _ = from_enu(e_m, n_m, elev_m - origin.elevM, origin)
    counts = pick_counts(assign).reindex(ordered["eid"])
    events_out = typed_frame(
        {
            "assocId": ids,
            "t": ordered["t"].to_numpy(dtype=np.float64),
            "latitude": lat,
            "longitude": lon,
            "elevM": elev_m,
            "nPicks": counts["nPicks"].to_numpy(),
            "nP": counts["nP"].to_numpy(),
            "nS": counts["nS"].to_numpy(),
        },
        EVENT_DTYPES,
    )
    picks_out = assign.assign(assocId=assign["eid"].map(id_of)).sort_values(
        ["assocId", "pickId"], kind="stable"
    )
    return AssocResult(
        events_out,
        typed_frame(
            {"assocId": picks_out["assocId"].to_numpy(), "pickId": picks_out["pickId"].to_numpy()},
            PICK_DTYPES,
        ),
    )


def finish(
    raw: RawAssociation, acfg: AssociatorConfig, origin: Origin
) -> tuple[AssocResult, dict[str, int]]:
    """Steps 3-6 on PyOcto's output; returns the result and its counts."""
    started = time.perf_counter()
    events, assign, merged = merge_duplicates(
        raw.events, raw.assignments, acfg.mergeWithinS, acfg.mergeMinSharedFraction
    )
    assign, shared = resolve_shared_picks(events, assign)
    events, assign, low_pyocto, low_stations = apply_minimums(events, assign, acfg)
    result = to_result(events, assign, origin)
    counts = {
        "picksIn": raw.picks_in,
        "picksAboveMinProb": raw.picks_used,
        "pyoctoEvents": len(raw.events),
        "pyoctoAssignments": len(raw.assignments),
        "mergedDuplicates": merged,
        "sharedPicksRemoved": shared,
        "droppedBelowPickMinimums": low_pyocto,
        "droppedBelowMinStations": low_stations,
        "events": len(result.events),
        "assocPicks": len(result.picks),
    }
    log.info(
        "associate: %d events (%d picks) at minStations %d, nSPicks %d, minPickProb %s: "
        "%d duplicates merged, %d shared picks removed, %d below pick minimums, "
        "%d below minStations (%.2f s)",
        counts["events"], counts["assocPicks"], acfg.minStations, acfg.nSPicks, acfg.minPickProb,
        merged, shared, low_pyocto, low_stations, time.perf_counter() - started,
    )
    return result, counts


def associate_setup(
    picks: pd.DataFrame, setup: Setup, acfg: AssociatorConfig
) -> tuple[AssocResult, dict[str, int]]:
    """One association run on a prepared setup."""
    raw = run_pyocto(picks, setup, acfg)
    return finish(raw, acfg, setup.origin)


def record(acfg: AssociatorConfig, setup: Setup) -> dict[str, Any]:
    """The ``ProcessingRun.associator`` params: every PyOcto argument and how inputs were built."""
    assoc_args = associator_kwargs(acfg, setup)
    return {
        "pyoctoVersion": pyocto.__version__,
        "pyocto": {
            "OctoAssociator": {
                **{k: list(v) if isinstance(v, tuple) else v for k, v in assoc_args.items()},
                "velocity_model": {
                    "class": "StationSpecificVelocityModel1D",
                    **velocity_kwargs(acfg, setup),
                },
            },
        },
        "frame": {
            "x": "ENU e / 1000 (km; UTM 12N EPSG:32612 minus the run origin)",
            "y": "ENU n / 1000 (km)",
            "z": "-elevM / 1000 (km below sea level, down positive); stations at sensorElevM",
            "elevM": "-z * 1000 (m ASL)",
            "origin": setup.origin.model_dump(),
        },
        "volume": setup.volume.to_record(),
        "stations": setup.station_note,
        "picks": {
            "minPickProb": acfg.minPickProb,
            "weights": "none: PyOcto 0.2.0 has no per-pick weight; prob is only the threshold",
        },
        "postprocess": {
            "mergeWithinS": acfg.mergeWithinS,
            "mergeMinSharedFraction": acfg.mergeMinSharedFraction,
            "mergeRule": "origin times within mergeWithinS and shared picks >= fraction of the "
            "smaller event's picks, transitively; the member with most picks keeps its origin",
            "sharedPickRule": "a pick in two events stays with the smaller |PyOcto residual| "
            "(ties: more picks, earlier, lower index)",
            "minStations": acfg.minStations,
            "minimumsReapplied": "nPicks, nPPicks, nSPicks, nPAndSPicks and minStations after the "
            "merge and the shared-pick rule",
        },
        "tables": setup.tables.to_record(),
        "velocityModel": setup.velocity_model,
        "config": acfg.model_dump(mode="json"),
    }

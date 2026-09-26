"""Association with PyOcto 0.2.0: inputs, the PyOcto call, duplicate merge and outputs (LOC-03).

Flow for one setting of the knobs:

1. Picks: validated (docs/02 ``Pick`` columns), kept at ``prob >= minPickProb``. PyOcto 0.2.0 has
   no per-pick weight (its ``Pick`` is index, time, station, phase), so ``prob`` is used only as
   that threshold. Nothing depends on ``picker``: STA/LTA picks run through unchanged. Picks from
   a station that is in ``stations.parquet`` with ``usedInRun`` false (H1's known-event picker
   picks those on purpose) are dropped and counted, or rejected, per
   ``picksFromUnusedStations``; a pick from a station missing from the table always fails.
2. PyOcto with ``StationSpecificVelocityModel1D`` tables built from the layer model
   (``hq.associate.tables``), stations at ``sensorElevM``, in the frame of ``hq.associate.frame``.
3. Duplicate merge. PyOcto itself first deletes the smaller of two events sharing more than
   ``maxPickOverlap`` picks (C++ ``deduplicate_events``, no time condition, not counted here).
   Then, pairwise: events are taken in rank order (most picks, then earlier, then lower PyOcto
   index); each event not yet merged keeps its origin and absorbs every later-ranked event whose
   origin time is within ``mergeWithinS`` of its own and that shares at least
   ``mergeMinSharedFraction`` of the smaller event's picks with it (not transitively). Absorbed
   events add the picks of station-phases the primary lacks; those picks keep the residual PyOcto
   computed for the absorbed event's hypocentre.
3b. Run window: events whose origin time lies outside ``[windowStart, windowEnd)`` (run.yaml; the
   public catalog uses the same bounds) are dropped and counted. PyOcto places an origin before
   its first arrival, so in-window picks can form an event up to one S travel time before
   ``windowStart``. Their picks are then free for the other events in step 4. The bbox is not
   applied here (the search volume extends ``volume.horizontalMarginM`` beyond it); association
   locations are preliminary, so that bound belongs after location.
4. Shared picks and minimums: events that fail PyOcto's pick minimums or ``minStations`` even with
   all their picks are dropped. Then a pick still in two events stays with the one where its
   PyOcto residual is smallest in magnitude (ties: more picks, earlier, lower index), and events
   are recounted against the minimums. If some now fail (they lost shared picks), only the
   lowest-ranked of them (fewest picks, then later, then higher index) is dropped, its picks go
   back to the other events that held them, and the resolution runs again, until none fails. So
   a pick is never lost to an event that is then dropped.
5. Output: ``elevM = -z * 1000``, latitude/longitude from ``(x, y) * 1000`` through
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
from hq.associate.tables import PYOCTO_VERSION, TableSet, build_tables
from hq.config.run import Origin, RunSection
from hq.config.seismology import AssociatorConfig, SeismologyConfig
from hq.locate.coords import from_enu, to_enu
from hq.locate.velocity import LayerModel, load_configured_model

log = logging.getLogger(__name__)


def require_pyocto(version: str) -> None:
    """Raise unless ``version`` is the PyOcto release this package was written for.

    PyOcto 0.1.x (what uv resolves on Python 3.11) has no ``StationSpecificVelocityModel1D`` and
    no ``second_pass_overwrites``; a newer release may change the table format or the z datum.
    """
    if version != PYOCTO_VERSION:
        raise ImportError(
            f"hq.associate needs pyocto=={PYOCTO_VERSION} (its travel-time table format, z datum "
            f"and arguments were checked against that release); pyocto {version} is installed. "
            "PyOcto 0.2.0 needs Python >= 3.12 in this lock."
        )


require_pyocto(pyocto.__version__)

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
    window_s: tuple[float, float]  # run window [start, end), epoch s UTC
    velocity_model: dict[str, Any]  # LayerModel.to_record() of the (extended) model used
    station_note: dict[str, Any]
    unused_station_ids: frozenset[str]  # in stations.parquet with usedInRun false


@dataclass(frozen=True)
class RawAssociation:
    """PyOcto's output for one pick set, before the merge (reused across ``minStations``)."""

    events: pd.DataFrame  # eid, t, x, y, z
    assignments: pd.DataFrame  # eid, pickId, station, phase, residual
    picks_in: int
    picks_used: int
    picks_unused_stations: int  # dropped: from stations with usedInRun false
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


def select_picks(
    picks: pd.DataFrame,
    min_prob: float,
    station_ids: set[str],
    unused_station_ids: frozenset[str],
    unused_policy: str,
) -> tuple[pd.DataFrame, int]:
    """Validated picks at ``prob >= min_prob`` in PyOcto's format plus ``pickId``.

    Also returns how many picks were dropped for coming from a ``usedInRun`` false station
    (``unused_policy`` "drop"; "reject" raises instead).
    """
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
    unknown = sorted(set(station) - station_ids - unused_station_ids)
    if unknown:
        raise ValueError(f"picks reference stations missing from the stations table: {unknown}")
    from_unused = station.isin(unused_station_ids).to_numpy()
    if from_unused.any():
        names = sorted(set(station[from_unused]))
        if unused_policy != "drop":
            raise ValueError(
                f"{int(from_unused.sum())} picks come from stations with usedInRun false {names} "
                f"(associator.picksFromUnusedStations: {unused_policy})"
            )
        log.warning(
            "associate: dropped %d picks from stations with usedInRun false %s "
            "(associator.picksFromUnusedStations: drop)",
            int(from_unused.sum()), names,
        )
    keep = (prob >= min_prob) & ~from_unused
    return pd.DataFrame(
        {
            "station": station[keep].to_numpy(dtype=object),
            "phase": phase[keep].to_numpy(dtype=object),
            "time": t[keep],
            "pickId": ids[keep].to_numpy(dtype=object),
        }
    ), int(from_unused.sum())


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
            window_s=(run.window_start_s, run.window_end_s),
            velocity_model=extended.to_record(),
            station_note=note,
            unused_station_ids=frozenset(
                str(s) for s in stations.loc[~stations["usedInRun"].astype(bool), "id"]
            ),
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
    selected, from_unused = select_picks(
        picks,
        acfg.minPickProb,
        set(setup.stations["id"]),
        setup.unused_station_ids,
        acfg.picksFromUnusedStations,
    )
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
        return RawAssociation(empty_events, empty_assign, len(picks), 0, from_unused,
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
    return RawAssociation(raw_events, raw_assign, len(picks), len(selected), from_unused, runtime)


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
    picks_of = {int(eid): set(g) for eid, g in assign.groupby("eid")["pickId"]}
    t_of = {int(e): float(t) for e, t in zip(events["eid"], events["t"], strict=True)}
    n_picks = {e: len(picks_of.get(e, ())) for e in t_of}
    rank = {e: (-n_picks[e], t_of[e], e) for e in t_of}
    by_time = sorted(t_of, key=lambda e: (t_of[e], e))
    times = np.array([t_of[e] for e in by_time], dtype=np.float64)
    owner: dict[int, int] = {}
    groups: dict[int, list[int]] = {}
    for a in sorted(t_of, key=lambda e: rank[e]):
        if a in owner:
            continue
        owner[a] = a
        groups[a] = [a]
        lo = int(np.searchsorted(times, t_of[a] - within_s, side="left"))
        hi = int(np.searchsorted(times, t_of[a] + within_s, side="right"))
        for b in by_time[lo:hi]:
            if b in owner:  # a itself, or ranked before a
                continue
            shared = len(picks_of.get(a, set()) & picks_of.get(b, set()))
            if shared and shared >= min_shared_fraction * min(n_picks[a], n_picks[b]):
                owner[b] = a
                groups[a].append(b)
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
    """Keep every pick in one event only (smallest |residual|). Returns rows, n removed."""
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


def drop_outside_window(
    events: pd.DataFrame, assign: pd.DataFrame, window_s: tuple[float, float]
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Drop events whose origin time is outside ``[start, end)`` (module docstring, step 3b)."""
    start, end = window_s
    inside = (events["t"] >= start) & (events["t"] < end)
    kept = events[inside].reset_index(drop=True)
    return kept, assign[assign["eid"].isin(kept["eid"])].reset_index(drop=True), int((~inside).sum())


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


def minimum_checks(
    events: pd.DataFrame, assign: pd.DataFrame, acfg: AssociatorConfig
) -> tuple["pd.Series[bool]", "pd.Series[bool]"]:
    """Per event (index eid): meets PyOcto's pick minimums; meets ``minStations``."""
    counts = pick_counts(assign).reindex(events["eid"], fill_value=0)
    pyocto_ok = (
        (counts["nPicks"] >= acfg.nPicks)
        & (counts["nP"] >= acfg.nPPicks)
        & (counts["nS"] >= acfg.nSPicks)
        & (counts["nPS"] >= acfg.nPAndSPicks)
    )
    return pyocto_ok, counts["nStations"] >= acfg.minStations


def apply_minimums(
    events: pd.DataFrame, assign: pd.DataFrame, acfg: AssociatorConfig
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Drop events below PyOcto's minimums or ``minStations``.

    Returns events, assignments, the number dropped for PyOcto's minimums and for minStations.
    """
    pyocto_ok, station_ok = minimum_checks(events, assign, acfg)
    keep = pyocto_ok.index[pyocto_ok & station_ok]
    dropped_pyocto = int((~pyocto_ok).sum())
    dropped_stations = int((pyocto_ok & ~station_ok).sum())
    out_events = events[events["eid"].isin(keep)].reset_index(drop=True)
    out_assign = assign[assign["eid"].isin(keep)].reset_index(drop=True)
    return out_events, out_assign, dropped_pyocto, dropped_stations


def resolve_with_minimums(
    events: pd.DataFrame, assign: pd.DataFrame, acfg: AssociatorConfig
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int]:
    """Shared picks and minimums until stable (module docstring, step 4).

    Returns events, assignments, shared picks removed, and the events dropped for PyOcto's
    minimums and for ``minStations``.
    """
    # An event's own rows never change below (only other events' rows leave), so an event that
    # passes here with all its picks keeps passing that check.
    events, held, low_pyocto, low_stations = apply_minimums(events, assign, acfg)
    n_held = held.groupby("eid").size()
    while True:
        kept, shared = resolve_shared_picks(events, held)
        pyocto_ok, station_ok = minimum_checks(events, kept, acfg)
        failing = events[~(pyocto_ok & station_ok).reindex(events["eid"]).to_numpy()]
        if failing.empty:
            return events, kept, shared, low_pyocto, low_stations
        worst = int(
            failing.assign(_n=failing["eid"].map(n_held))
            .sort_values(["_n", "t", "eid"], ascending=[True, False, False], kind="stable")
            .iloc[0]["eid"]
        )
        low_pyocto += int(not pyocto_ok[worst])
        low_stations += int(bool(pyocto_ok[worst]))
        # the dropped event's picks go back to the other events that held them
        events = events[events["eid"] != worst].reset_index(drop=True)
        held = held[held["eid"] != worst].reset_index(drop=True)


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
    raw: RawAssociation, acfg: AssociatorConfig, setup: Setup
) -> tuple[AssocResult, dict[str, int]]:
    """Steps 3-5 on PyOcto's output; returns the result and its counts."""
    started = time.perf_counter()
    events, assign, merged = merge_duplicates(
        raw.events, raw.assignments, acfg.mergeWithinS, acfg.mergeMinSharedFraction
    )
    events, assign, outside = drop_outside_window(events, assign, setup.window_s)
    events, assign, shared, low_pyocto, low_stations = resolve_with_minimums(events, assign, acfg)
    result = to_result(events, assign, setup.origin)
    counts = {
        "picksIn": raw.picks_in,
        "picksFromUnusedStationsDropped": raw.picks_unused_stations,
        "picksAboveMinProb": raw.picks_used,
        "pyoctoEvents": len(raw.events),
        "pyoctoAssignments": len(raw.assignments),
        "mergedDuplicates": merged,
        "droppedOutsideWindow": outside,
        "sharedPicksRemoved": shared,
        "droppedBelowPickMinimums": low_pyocto,
        "droppedBelowMinStations": low_stations,
        "events": len(result.events),
        "assocPicks": len(result.picks),
    }
    log.info(
        "associate: %d events (%d picks) at minStations %d, nSPicks %d, minPickProb %s: "
        "%d duplicates merged, %d outside the run window, %d shared picks removed, "
        "%d below pick minimums, %d below minStations (%.2f s)",
        counts["events"], counts["assocPicks"], acfg.minStations, acfg.nSPicks, acfg.minPickProb,
        merged, outside, shared, low_pyocto, low_stations, time.perf_counter() - started,
    )
    return result, counts


def associate_setup(
    picks: pd.DataFrame, setup: Setup, acfg: AssociatorConfig
) -> tuple[AssocResult, dict[str, int]]:
    """One association run on a prepared setup."""
    raw = run_pyocto(picks, setup, acfg)
    return finish(raw, acfg, setup)


def record(acfg: AssociatorConfig, setup: Setup, picks: pd.DataFrame) -> dict[str, Any]:
    """The ``ProcessingRun.associator`` params: every PyOcto argument and how inputs were built.

    ``picks``: the input picks (docs/02 ``Pick`` rows), whose count and time span are recorded.
    """
    assoc_args = associator_kwargs(acfg, setup)
    t = picks["t"].to_numpy(dtype=np.float64) if "t" in picks.columns else np.empty(0)
    return {
        "pyoctoVersion": pyocto.__version__,
        "pyocto": {
            "OctoAssociator": {
                **{k: list(v) if isinstance(v, tuple) else v for k, v in assoc_args.items()},
                "velocity_model": {
                    "class": "StationSpecificVelocityModel1D",
                    **velocity_kwargs(acfg, setup),
                    "path": setup.tables.relative_directory,  # absolute path: log only
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
            "fromUnusedStations": f"{acfg.picksFromUnusedStations}: picks from stations with "
            "usedInRun false (counts.picksFromUnusedStationsDropped)",
            "nIn": len(picks),
            "tMinS": float(t.min()) if t.size else None,
            "tMaxS": float(t.max()) if t.size else None,
        },
        "window": {
            "startS": setup.window_s[0],
            "endS": setup.window_s[1],
            "rule": "events with origin time outside [startS, endS) are dropped after the "
            "duplicate merge (counts.droppedOutsideWindow); the bbox is not applied here",
        },
        "postprocess": {
            "mergeWithinS": acfg.mergeWithinS,
            "mergeMinSharedFraction": acfg.mergeMinSharedFraction,
            "pyoctoDeduplication": "PyOcto deletes the smaller of two events sharing more than "
            "max_pick_overlap picks before this merge (no time condition, not counted here)",
            "mergeRule": "pairwise against a primary taken in rank order (most picks, earlier, "
            "lower index): origin times within mergeWithinS and shared picks >= fraction of the "
            "smaller event's picks; not transitive; absorbed picks keep the absorbed event's "
            "PyOcto residual",
            "sharedPickRule": "a pick in two events stays with the smaller |PyOcto residual| "
            "(ties: more picks, earlier, lower index)",
            "minStations": acfg.minStations,
            "minimumsReapplied": "nPicks, nPPicks, nSPicks, nPAndSPicks and minStations before "
            "and after the shared-pick rule; an event dropped there returns its picks to the "
            "other events holding them and the rule reruns until no event is dropped",
        },
        "tables": setup.tables.to_record(),
        "velocityModel": setup.velocity_model,
        "config": acfg.model_dump(mode="json"),
    }

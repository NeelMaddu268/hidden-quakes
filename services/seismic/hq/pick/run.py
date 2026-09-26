"""Full-window PhaseNet picking (SEIS-06): stage ``pick`` -> ``runs/<id>/picks.parquet``.

For every ``stations.parquet`` row with ``usedInRun``, the run window (``run.yaml``) is read one
chunk at a time through the shared iterator ``hq.preprocess.chunks.iter_model_chunks`` with the
station's ``preprocessProfile`` and ``channels``, so this picker and the STA/LTA baseline
(SEIS-07) see identical preprocessed traces. Each chunk goes through SEIS-04's gap-safe machinery
unchanged: ``prepare_station`` splits it into contiguous three-component blocks (never zero-filled;
blocks shorter than the model window are skipped and counted) and ``pick_prepared`` classifies each
block with every ``classify`` argument from ``picker`` (thresholds, ``strict``, ``blinding``,
``overlap``) and converts each pick to real time with the chunk's ``TimeMap``, exactly once.

The keep rule is the shared one (``pick_is_kept``), applied to that real time:

* ``outside_keep``: the pick lies in the chunk's read overlap and belongs to the neighbouring
  chunk. It is discarded and never counted as a drop (the report lists it as ``outsideKeep``).
* ``near_gap_edge``: within ``picker.gapEdgeS`` real seconds of a raw data edge (a gap, or where
  the cached data starts or stops). Dropped and counted (``droppedNearGap``).

SEIS-04's own block-edge drop is switched off here (``gapEdgeS`` 0 inside ``pick_prepared``),
because a block edge inside the read overlap is a chunk cut, not a gap; the data-edge rule above
replaces it. Blinding still applies: seisbench returns nothing within ``blinding`` samples of a
block edge. At 100 Hz model rate that is 2.5 s, more than ``gapEdgeS`` (1 s), so on the
``surface-100``, ``surface-hi`` and ``borehole-A`` profiles a gap that splits a block cannot
cost a pick through ``gapEdgeS``: nothing is produced that close to it. ``droppedNearGap`` there
counts picks near raw data edges that do NOT split a block. The known case is a raw gap shorter
than one model sample (one missing sample at 1,000 Hz): ``for_picking`` tapers the two segments
separately, their 100 Hz outputs abut, ``split_blocks`` joins them into one block, and PhaseNet
sees the two tapered ends. The report states per station how many seconds were blinded inside
the window (``blindedS``) and the resulting edge exclusion (``edgeExclusionS``), so a small or
zero ``droppedNearGap`` is read correctly.

Weights: ``picker.weightsByProfile[profile]``; a profile missing there falls back to
``picker.defaultWeights`` with a warning, and the report records which (``weightsSource``).

Execution: stations run in parallel in spawned worker processes (``picker.run.workers``); each
worker loads each weight set once and runs torch with ``picker.run.torchThreadsPerWorker``
intra-op threads. Results are collected per station and assembled in station-id order, so the
output never depends on which worker finished first. Picks are deduplicated by ``id`` (the
highest ``prob`` wins; the count is logged and recorded) and sorted by
``(t, stationId, phase, prob desc)``.

Outputs: ``picks.parquet`` (``Pick`` rows through ``hq_contracts.io``; an empty table is valid),
``pick_report.json`` (per-station counts, blinding, edge exclusion, runtimes and a
``zeroPickReason`` for every station without picks), and ``ctx.record("pick", ...)``.

CLI (a local context: writes picks.parquet and pick_report.json into ``--run-dir``, which must
hold stations.parquet, and does not touch run.json)::

    uv run python -m hq.pick.run --run-dir <dir> --config-dir configs/showcase --cache-dir <dir>
        [--stations UU.FORK,UU.NMU] [--start 2026-09-10T09:00:00Z --end 2026-09-10T11:00:00Z]
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol

import obspy
import yaml

from hq.config.run import RunSection
from hq.config.signal import PickerConfig, PickerRunConfig, SignalConfig
from hq.ingest.cache import CacheMissError, station_keys
from hq.pick.ab import StationInfo, load_signal_config, load_stations, write_picks
from hq.pick.phasenet import load_model, pick_prepared, prepare_station
from hq.preprocess.chunks import (
    NEAR_GAP_EDGE,
    OUTSIDE_KEEP,
    ChunkStats,
    ModelChunk,
    iter_model_chunks,
    pick_is_kept,
)
from hq.preprocess.profiles import TimeMap

log = logging.getLogger(__name__)

STAGE = "pick"
PICKS_FILE = "picks.parquet"
REPORT_FILE = "pick_report.json"
STATIONS_FILE = "stations.parquet"
_SECONDS_PER_HOUR = 3600.0  # unit conversion for the report, not a knob
_LOG_FORMAT = "%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s"

WeightsSource = Literal["weightsByProfile", "defaultWeights"]


# --- tasks and results ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationTask:
    """One station to pick: what the worker needs besides the shared ``StationJob``."""

    stationId: str
    profile: str
    channels: tuple[str, ...]
    weights: str
    weightsSource: WeightsSource


@dataclass(frozen=True)
class StationJob:
    """Settings shared by every station of one run (picklable: sent to each worker)."""

    cfg: SignalConfig
    t0: float  # real epoch s, window start (inclusive)
    t1: float  # real epoch s, window end (exclusive)
    cacheDir: Path


@dataclass
class StationReport:
    """Per-station counts for ``pick_report.json`` (real seconds throughout)."""

    stationId: str
    profile: str
    weights: str
    weightsSource: WeightsSource
    channels: list[str]
    windowS: float
    gapEdgeS: float
    cacheMiss: bool = False
    chunks: int = 0  # chunks the iterator yielded (had data after preprocessing)
    chunksPlanned: int = 0
    chunksEmpty: int = 0  # no samples on the station's channels
    chunksNoSegments: int = 0  # samples, but every segment too short for preprocessing
    chunksMissingComponents: int = 0  # a chunk without one of Z/N/E: not picked
    blocks: int = 0
    blocksPicked: int = 0
    blocksTooShort: int = 0  # shorter than the model window (3001 samples): skipped
    nOverlaps: int = 0
    nP: int = 0
    nS: int = 0
    droppedNearGapP: int = 0
    droppedNearGapS: int = 0
    outsideKeep: int = 0  # read-overlap picks owned by a neighbouring chunk; not a drop
    droppedBelowThreshold: int = 0
    duplicates: int = 0
    dataEdges: int = 0  # raw data edges inside the window (gaps, data start / end)
    secondsPicked: float = 0.0  # data the model saw, inside the window
    secondsTooShort: float = 0.0  # data in blocks shorter than the model window, inside the window
    blindedS: float = 0.0  # inside the window, at block edges, where blinding leaves no output
    blindingEdgeS: float = 0.0  # blinded seconds at one block edge (max over blocks)
    edgeExclusionS: float = 0.0  # no pick survives this close to a data edge: max(blinding, gap)
    runtimeS: float = 0.0
    zeroPickReason: str | None = None

    @property
    def droppedNearGap(self) -> int:
        return self.droppedNearGapP + self.droppedNearGapS

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["droppedNearGap"] = self.droppedNearGap
        return out


@dataclass(frozen=True)
class StationResult:
    picks: list[dict[str, Any]]
    report: StationReport


@dataclass(frozen=True)
class PickRunResult:
    picks: list[dict[str, Any]]  # deduplicated and sorted
    reports: list[StationReport]  # station-id order
    duplicates: int
    workers: int  # worker processes used (1 = in-process)
    runtimeS: float

    def counts(self) -> dict[str, int]:
        return {
            "stations": len(self.reports),
            "picksP": sum(1 for p in self.picks if p["phase"] == "P"),
            "picksS": sum(1 for p in self.picks if p["phase"] == "S"),
            "droppedNearGap": sum(r.droppedNearGap for r in self.reports),
            "duplicates": self.duplicates,
            "zeroPickStations": sum(1 for r in self.reports if r.zeroPickReason is not None),
            "cacheMissStations": sum(1 for r in self.reports if r.cacheMiss),
            "chunksPlanned": sum(r.chunksPlanned for r in self.reports),
            "chunksPicked": sum(r.chunks for r in self.reports),
            "blocksPicked": sum(r.blocksPicked for r in self.reports),
            "blocksTooShort": sum(r.blocksTooShort for r in self.reports),
        }


# --- injectable I/O ------------------------------------------------------------------------------


class IterChunks(Protocol):
    """Signature of ``hq.preprocess.chunks.iter_model_chunks`` as used here."""

    def __call__(
        self,
        station_id: str,
        channels: Sequence[str],
        profile: str,
        t0: float,
        t1: float,
        cfg: SignalConfig,
        *,
        cache_dir: Path,
        stats: ChunkStats | None = None,
    ) -> Iterator[ModelChunk]: ...


ModelLoader = Callable[[str, PickerConfig], Any]
CacheCheck = Callable[[str, Path], None]  # raises CacheMissError when nothing is cached


def check_cached(station_id: str, cache_dir: Path) -> None:
    """Raise ``CacheMissError`` if nothing at all is cached for the station (any day)."""
    station_keys(station_id, cache_dir=cache_dir)


@dataclass(frozen=True)
class PickIO:
    """What a station worker reads; module-level functions, so it pickles into spawned workers."""

    iter_chunks: IterChunks
    load_model: ModelLoader
    check_cached: CacheCheck


def default_io() -> PickIO:
    """Real implementations: SEIS-07 chunk iterator over the SEIS-05 cache, seisbench PhaseNet."""
    return PickIO(iter_chunks=iter_model_chunks, load_model=load_model, check_cached=check_cached)


@dataclass(frozen=True)
class StageIO:
    """What the stage reads and writes in the run directory (``hq_contracts.io``)."""

    load_stations: Callable[[Path], dict[str, StationInfo]]
    write_picks: Callable[[list[dict[str, Any]], Path], None]


def default_stage_io() -> StageIO:
    return StageIO(load_stations=load_stations, write_picks=write_picks)


StationFn = Callable[[StationTask], StationResult]
Runner = Callable[[StationFn, Sequence[StationTask]], list[StationResult]]


# --- one station ---------------------------------------------------------------------------------


def _inside(a: float, b: float, keep: tuple[float, float]) -> float:
    """Length of ``[a, b]`` inside ``keep``."""
    return max(0.0, min(b, keep[1]) - max(a, keep[0]))


def _chunk_as_prepared(chunk: ModelChunk) -> Callable[..., tuple[obspy.Stream, TimeMap]]:
    """A ``for_picking`` stand-in for ``prepare_station``: the chunk is already preprocessed."""

    def already_prepared(
        st: obspy.Stream, profile: str, cfg: SignalConfig
    ) -> tuple[obspy.Stream, TimeMap]:
        if profile != chunk.profile:
            raise ValueError(f"{chunk.stationId}: chunk is {chunk.profile}, asked for {profile}")
        return st, chunk.timemap

    return already_prepared


def without_block_edge_drop(cfg: SignalConfig) -> SignalConfig:
    """``cfg`` with SEIS-04's block-edge drop off: here the shared data-edge rule replaces it."""
    return cfg.model_copy(update={"picker": cfg.picker.model_copy(update={"gapEdgeS": 0.0})})


def zero_pick_reason(rep: StationReport) -> str | None:
    """Why a station has no picks (None when it has some)."""
    if rep.nP + rep.nS:
        return None
    if rep.cacheMiss:
        return "nothing cached for this station"
    if rep.chunks == 0:
        if rep.chunksNoSegments:
            return "all segments too short for preprocessing (minSegmentModelS / filter padding)"
        return "no data in window"
    if rep.blocks == 0:
        return "no three-component data (a component is missing)"
    if rep.blocksPicked == 0:
        return "all segments shorter than model window"
    if rep.droppedNearGap:
        return "every pick was within gapEdgeS of a data edge"
    return "no picks above threshold"


def pick_station(task: StationTask, *, job: StationJob, io: PickIO) -> StationResult:
    """Pick one station over the window, chunk by chunk (runs inside a worker)."""
    began = time.perf_counter()
    cfg = job.cfg
    picker = cfg.picker
    model = io.load_model(
        task.weights,
        picker.model_copy(update={"torchThreads": picker.run.torchThreadsPerWorker}),
    )
    model_sr = float(model.sampling_rate)
    min_samples = int(model.in_samples)
    blind_pre, blind_post = picker.seisbench.blinding
    block_cfg = without_block_edge_drop(cfg)
    rep = StationReport(
        stationId=task.stationId,
        profile=task.profile,
        weights=task.weights,
        weightsSource=task.weightsSource,
        channels=list(task.channels),
        windowS=job.t1 - job.t0,
        gapEdgeS=picker.gapEdgeS,
    )
    stats = ChunkStats()
    picks: list[dict[str, Any]] = []
    for chunk in io.iter_chunks(
        task.stationId,
        task.channels,
        task.profile,
        job.t0,
        job.t1,
        cfg,
        cache_dir=job.cacheDir,
        stats=stats,
    ):
        if chunk.stationId != task.stationId or chunk.profile != task.profile:
            raise ValueError(
                f"chunk for {chunk.stationId} ({chunk.profile}) while picking "
                f"{task.stationId} ({task.profile})"
            )
        rep.chunks += 1
        keep = chunk.keep
        rep.dataEdges += sum(1 for e in chunk.dataEdges if keep[0] <= e < keep[1])
        prepared = prepare_station(
            chunk.stream, task.stationId, task.profile, cfg, for_picking=_chunk_as_prepared(chunk)
        )
        if prepared.missingComponents:
            rep.chunksMissingComponents += 1
        rep.blocks += len(prepared.blocks)
        rep.nOverlaps += prepared.nOverlaps
        for block in prepared.blocks:
            inside = _inside(block.startReal, block.endReal, keep)
            if block.npts < min_samples:
                rep.blocksTooShort += 1
                rep.secondsTooShort += inside
                continue
            rep.blocksPicked += 1
            rep.secondsPicked += inside
            real_per_sample = block.realPerModelS / model_sr
            pre_s, post_s = blind_pre * real_per_sample, blind_post * real_per_sample
            rep.blindedS += _inside(block.startReal, block.startReal + pre_s, keep)
            rep.blindedS += _inside(block.endReal - post_s, block.endReal, keep)
            rep.blindingEdgeS = max(rep.blindingEdgeS, pre_s, post_s)

        # SEIS-04: classify each block, threshold, convert to real time with chunk.timemap once.
        chunk_picks, diag = pick_prepared(prepared, block_cfg, model, task.weights)
        if diag.droppedNearEdge:  # gapEdgeS is 0 here, so only a pick outside its block could
            raise RuntimeError(
                f"{task.stationId}: {diag.droppedNearEdge} picks fell outside their own block"
            )
        rep.droppedBelowThreshold += diag.droppedBelowThreshold
        for p in chunk_picks:
            kept, reason = pick_is_kept(p["t"], chunk, picker.gapEdgeS)
            if kept:
                picks.append(p)
            elif reason == NEAR_GAP_EDGE:
                if p["phase"] == "P":
                    rep.droppedNearGapP += 1
                else:
                    rep.droppedNearGapS += 1
            elif reason == OUTSIDE_KEEP:
                rep.outsideKeep += 1
            else:
                raise RuntimeError(f"pick_is_kept returned an unknown reason {reason!r}")

    rep.chunksPlanned = stats.planned
    rep.chunksEmpty = stats.empty
    rep.chunksNoSegments = stats.noSegments
    rep.nP = sum(1 for p in picks if p["phase"] == "P")
    rep.nS = sum(1 for p in picks if p["phase"] == "S")
    rep.edgeExclusionS = max(picker.gapEdgeS, rep.blindingEdgeS)
    rep.zeroPickReason = zero_pick_reason(rep)
    rep.runtimeS = time.perf_counter() - began
    log.info(
        "%s %s %s: chunks %d/%d, blocks %d (too short %d), P %d, S %d, dropped near gap %d, "
        "blinded %.1f s, edge exclusion %.2f s, picked %.2f h in %.1f s%s",
        task.stationId,
        task.profile,
        task.weights,
        rep.chunks,
        rep.chunksPlanned,
        rep.blocks,
        rep.blocksTooShort,
        rep.nP,
        rep.nS,
        rep.droppedNearGap,
        rep.blindedS,
        rep.edgeExclusionS,
        rep.secondsPicked / _SECONDS_PER_HOUR,
        rep.runtimeS,
        f" (zero picks: {rep.zeroPickReason})" if rep.zeroPickReason else "",
    )
    return StationResult(picks=picks, report=rep)


# --- runners -------------------------------------------------------------------------------------


def run_in_process(fn: StationFn, tasks: Sequence[StationTask]) -> list[StationResult]:
    return [fn(task) for task in tasks]


def _init_worker(log_level: int) -> None:
    """Spawned workers start with no logging config; give them the parent's level."""
    logging.basicConfig(level=log_level, format=_LOG_FORMAT)


def process_pool_runner(workers: int) -> Runner:
    """Run station tasks in ``workers`` spawned processes; results come back in any order."""

    def run(fn: StationFn, tasks: Sequence[StationTask]) -> list[StationResult]:
        results: list[StationResult] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(logging.getLogger().getEffectiveLevel(),),
        ) as pool:
            futures: dict[Future[StationResult], str] = {
                pool.submit(fn, task): task.stationId for task in tasks
            }
            try:
                for fut in as_completed(futures):
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        raise RuntimeError(f"{futures[fut]}: picking failed") from exc
            except BaseException:
                for fut in futures:
                    fut.cancel()
                raise
        return results

    return run


def default_runner(run_cfg: PickerRunConfig, n_tasks: int) -> tuple[Runner, int]:
    workers = min(run_cfg.workers, n_tasks)
    if workers <= 1:
        return run_in_process, 1
    return process_pool_runner(workers), workers


# --- the run -------------------------------------------------------------------------------------


def plan_tasks(
    stations: Mapping[str, StationInfo],
    picker: PickerConfig,
    station_ids: Sequence[str] | None = None,
) -> list[StationTask]:
    """``usedInRun`` stations in station-id order, with their weights (optionally a subset)."""
    used = sorted(sid for sid, s in stations.items() if s.usedInRun)
    if station_ids is None:
        selected = used
    else:
        unknown = sorted(set(station_ids) - set(stations))
        if unknown:
            raise ValueError(f"stations {unknown} are not in {STATIONS_FILE}")
        unused = sorted(sid for sid in station_ids if not stations[sid].usedInRun)
        if unused:
            raise ValueError(f"stations {unused} have usedInRun = false")
        selected = sorted(set(station_ids))
    log.info(
        "%d of %d stations have usedInRun; picking %d", len(used), len(stations), len(selected)
    )
    tasks: list[StationTask] = []
    for sid in selected:
        info = stations[sid]
        profile = info.preprocessProfile
        source: WeightsSource
        if profile in picker.weightsByProfile:
            weights, source = picker.weightsByProfile[profile], "weightsByProfile"
        else:
            weights, source = picker.defaultWeights, "defaultWeights"
            log.warning(
                "%s: profile %s has no picker.weightsByProfile entry; using defaultWeights %s",
                sid,
                profile,
                weights,
            )
        if not info.channels:
            raise ValueError(f"{sid}: no channels in {STATIONS_FILE}")
        tasks.append(
            StationTask(
                stationId=sid,
                profile=profile,
                channels=tuple(info.channels),
                weights=weights,
                weightsSource=source,
            )
        )
    return tasks


def dedupe_and_sort(picks: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """One pick per ``id`` (highest prob, then earliest t), sorted by (t, station, phase, -prob)."""
    best: dict[str, dict[str, Any]] = {}
    for p in picks:
        cur = best.get(p["id"])
        if cur is None or (p["prob"], -p["t"]) > (cur["prob"], -cur["t"]):
            best[p["id"]] = p
    out = sorted(
        best.values(), key=lambda p: (p["t"], p["stationId"], p["phase"], -p["prob"], p["id"])
    )
    return out, len(picks) - len(out)


def _cache_miss_report(task: StationTask, job: StationJob) -> StationReport:
    rep = StationReport(
        stationId=task.stationId,
        profile=task.profile,
        weights=task.weights,
        weightsSource=task.weightsSource,
        channels=list(task.channels),
        windowS=job.t1 - job.t0,
        gapEdgeS=job.cfg.picker.gapEdgeS,
        cacheMiss=True,
    )
    rep.zeroPickReason = zero_pick_reason(rep)
    return rep


def pick_window(
    tasks: Sequence[StationTask],
    cfg: SignalConfig,
    t0: float,
    t1: float,
    *,
    cache_dir: Path,
    io: PickIO | None = None,
    runner: Runner | None = None,
) -> PickRunResult:
    """Pick every task over ``[t0, t1)``; output independent of worker count and finish order."""
    began = time.perf_counter()
    if not t1 > t0:
        raise ValueError(f"picking window must satisfy t0 < t1, got [{t0}, {t1})")
    ids = [task.stationId for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate station tasks: {ids}")
    io = io if io is not None else default_io()
    job = StationJob(cfg=cfg, t0=t0, t1=t1, cacheDir=Path(cache_dir))

    missing: dict[str, str] = {}
    for task in tasks:  # before any picking, so onCacheMiss=error fails in seconds
        try:
            io.check_cached(task.stationId, job.cacheDir)
        except CacheMissError as exc:
            if cfg.picker.run.onCacheMiss == "error":
                raise
            missing[task.stationId] = str(exc)
            log.warning("%s: %s; zero picks (picker.run.onCacheMiss = report)", task.stationId, exc)
    todo = [task for task in tasks if task.stationId not in missing]

    workers = 1
    if runner is None:
        runner, workers = default_runner(cfg.picker.run, len(todo))
    log.info(
        "picking %d stations over [%s, %s) with %d worker(s) x %d torch threads",
        len(todo),
        obspy.UTCDateTime(t0),
        obspy.UTCDateTime(t1),
        workers,
        cfg.picker.run.torchThreadsPerWorker,
    )
    fn: StationFn = partial(pick_station, job=job, io=io)
    results = runner(fn, todo) if todo else []

    by_id: dict[str, StationResult] = {}
    for result in results:
        sid = result.report.stationId
        if sid in by_id:
            raise RuntimeError(f"{sid}: picked twice")
        by_id[sid] = result
    if set(by_id) != {task.stationId for task in todo}:
        raise RuntimeError(
            f"station results {sorted(by_id)} do not match the tasks "
            f"{sorted(task.stationId for task in todo)}"
        )

    reports: list[StationReport] = []
    collected: list[dict[str, Any]] = []
    for task in tasks:  # station-id order, whatever order the workers finished in
        if task.stationId in missing:
            reports.append(_cache_miss_report(task, job))
            continue
        result = by_id[task.stationId]
        reports.append(result.report)
        collected.extend(result.picks)
    picks, duplicates = dedupe_and_sort(collected)
    if duplicates:
        log.warning("dropped %d picks with a duplicate id (kept the highest prob)", duplicates)
        for rep in reports:
            station_picks = [p for p in picks if p["stationId"] == rep.stationId]
            n_p = sum(1 for p in station_picks if p["phase"] == "P")
            n_s = len(station_picks) - n_p
            rep.duplicates = rep.nP + rep.nS - n_p - n_s
            rep.nP, rep.nS = n_p, n_s
    return PickRunResult(
        picks=picks,
        reports=reports,
        duplicates=duplicates,
        workers=workers,
        runtimeS=time.perf_counter() - began,
    )


# --- report --------------------------------------------------------------------------------------


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")


def totals(result: PickRunResult, t0: float, t1: float) -> dict[str, Any]:
    station_hours = len(result.reports) * (t1 - t0) / _SECONDS_PER_HOUR
    worker_s = sum(r.runtimeS for r in result.reports)
    return {
        **result.counts(),
        "workers": result.workers,
        "stationHours": station_hours,
        "hoursPicked": sum(r.secondsPicked for r in result.reports) / _SECONDS_PER_HOUR,
        "blindedS": sum(r.blindedS for r in result.reports),
        "workerRuntimeS": worker_s,
        "workerSecondsPerStationHour": worker_s / station_hours if station_hours else 0.0,
        "wallRuntimeS": result.runtimeS,
    }


def build_report(result: PickRunResult, cfg: SignalConfig, t0: float, t1: float) -> dict[str, Any]:
    picker = cfg.picker
    blinding_over_gap = sorted(
        {r.profile for r in result.reports if r.blocksPicked and r.blindingEdgeS >= r.gapEdgeS}
    )
    notes = []
    if blinding_over_gap:
        notes.append(
            f"On {', '.join(blinding_over_gap)} the blinding at each block edge "
            f"(blindingEdgeS) is at least gapEdgeS ({picker.gapEdgeS} s): PhaseNet returns no "
            "pick that close to a block edge, so a gap that splits a block cannot drop a pick "
            "there. droppedNearGap on these profiles counts picks near raw data edges inside a "
            "block, e.g. a raw gap shorter than one model sample whose two separately tapered "
            "segments abut on the model grid and form one block."
        )
    return {
        "stage": STAGE,
        "window": {"start": _iso(t0), "end": _iso(t1), "t0": t0, "t1": t1},
        "model": picker.model,
        "weightsVersion": picker.weightsVersion,
        "thresholds": {"P": picker.pThreshold, "S": picker.sThreshold},
        "gapEdgeS": picker.gapEdgeS,
        "blindingSamples": list(picker.seisbench.blinding),
        "chunks": cfg.preprocess.chunks.model_dump(mode="json"),
        "execution": {
            "workers": result.workers,
            "torchThreadsPerWorker": picker.run.torchThreadsPerWorker,
        },
        "totals": totals(result, t0, t1),
        "notes": notes,
        "stations": [r.as_dict() for r in result.reports],
    }


def format_table(result: PickRunResult, t0: float, t1: float) -> str:
    head = (
        f"{'station':<10} {'profile':<11} {'weights':<9} {'chunks':>6} {'blocks':>6} "
        f"{'nP':>6} {'nS':>6} {'nearGap':>7} {'blindS':>7} {'edgeExS':>7} {'pickedH':>7} "
        f"{'runS':>7}  zeroPickReason"
    )
    lines = [head, "-" * len(head)]
    for r in result.reports:
        lines.append(
            f"{r.stationId:<10} {r.profile:<11} {r.weights:<9} {r.chunks:>6} {r.blocksPicked:>6} "
            f"{r.nP:>6} {r.nS:>6} {r.droppedNearGap:>7} {r.blindedS:>7.1f} "
            f"{r.edgeExclusionS:>7.2f} {r.secondsPicked / _SECONDS_PER_HOUR:>7.2f} "
            f"{r.runtimeS:>7.1f}  {r.zeroPickReason or ''}"
        )
    tot = totals(result, t0, t1)
    lines.append("-" * len(head))
    lines.append(
        f"{'TOTAL':<10} {len(result.reports):>2} stations          {tot['chunksPicked']:>6} "
        f"{tot['blocksPicked']:>6} {tot['picksP']:>6} {tot['picksS']:>6} "
        f"{tot['droppedNearGap']:>7} {tot['blindedS']:>7.1f} {'':>7} {tot['hoursPicked']:>7.2f} "
        f"{tot['workerRuntimeS']:>7.1f}  zero-pick stations {tot['zeroPickStations']}, "
        f"duplicates {tot['duplicates']}"
    )
    lines.append(
        f"window {_iso(t0)} .. {_iso(t1)}; {tot['stationHours']:.1f} station-hours; "
        f"worker time {tot['workerSecondsPerStationHour']:.1f} s per station-hour; "
        f"wall {tot['wallRuntimeS']:.1f} s with {tot['workers']} worker(s)"
    )
    return "\n".join(lines)


# --- stage ---------------------------------------------------------------------------------------


class StageContext(Protocol):
    """The part of H4's ``hq.runs.RunContext`` this stage uses (docs/02 -> Stage API)."""

    @property
    def cache_dir(self) -> Path: ...

    @property
    def config(self) -> Any: ...

    def path(self, name: str) -> Path: ...

    def record(
        self,
        stage: str,
        *,
        runtime_s: float,
        counts: dict[str, int],
        params: dict[str, Any] | None = None,
    ) -> None: ...


def run_picking(
    ctx: StageContext,
    *,
    station_ids: Sequence[str] | None = None,
    window: tuple[float, float] | None = None,
    io: PickIO | None = None,
    runner: Runner | None = None,
    stage_io: StageIO | None = None,
) -> PickRunResult:
    """The stage body; the CLI narrows it to some stations or a sub-window."""
    began = time.perf_counter()
    cfg: SignalConfig = ctx.config.signal
    if window is None:
        run_section: RunSection = ctx.config.run
        window = (run_section.window_start_s, run_section.window_end_s)
    t0, t1 = window
    sio = stage_io if stage_io is not None else default_stage_io()
    stations = sio.load_stations(ctx.path(STATIONS_FILE))
    tasks = plan_tasks(stations, cfg.picker, station_ids)
    result = pick_window(tasks, cfg, t0, t1, cache_dir=ctx.cache_dir, io=io, runner=runner)

    picks_path = ctx.path(PICKS_FILE)
    sio.write_picks(result.picks, picks_path)
    report_path = ctx.path(REPORT_FILE)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(build_report(result, cfg, t0, t1), indent=2) + "\n", encoding="utf-8"
    )
    counts = result.counts()
    log.info("wrote %s (%d picks) and %s", picks_path, len(result.picks), report_path)
    ctx.record(
        STAGE,
        runtime_s=time.perf_counter() - began,
        counts=counts,
        params={
            **cfg.picker.model_dump(mode="json"),
            "chunks": cfg.preprocess.chunks.model_dump(mode="json"),
        },
    )
    update_run = getattr(ctx, "update_run", None)
    if callable(update_run):
        update_run(pickerModel=cfg.picker.model, pickerWeights=cfg.picker.defaultWeights)
        log.info(
            "run.json: pickerModel=%s pickerWeights=%s", cfg.picker.model, cfg.picker.defaultWeights
        )
    else:
        log.info("context has no update_run; pickerModel / pickerWeights not set in run.json")
    print(format_table(result, t0, t1))
    return result


def run(ctx: StageContext) -> None:
    """Stage entry point (docs/02 -> Stage API)."""
    run_picking(ctx)


# --- CLI -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalConfig:
    run: RunSection
    signal: SignalConfig


@dataclass
class LocalContext:
    """CLI stand-in for ``RunContext``: records are logged and kept, run.json is not touched."""

    run_id: str
    run_dir: Path
    cache_dir: Path
    config: LocalConfig
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def record(
        self,
        stage: str,
        *,
        runtime_s: float,
        counts: dict[str, int],
        params: dict[str, Any] | None = None,
    ) -> None:
        self.records[stage] = {"runtime_s": runtime_s, "counts": counts, "params": params}
        log.info("record %s: %.1f s %s (run.json not updated by the CLI)", stage, runtime_s, counts)


def _parse_utc(text: str) -> float:
    value = datetime.fromisoformat(text)
    offset = value.utcoffset()
    if offset is None or offset.total_seconds() != 0.0:
        raise ValueError(f"{text!r} must be UTC with an explicit Z or +00:00")
    return value.timestamp()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hq.pick.run", description="Full-window PhaseNet picking (SEIS-06)."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="holds stations.parquet")
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--stations", help="comma-separated station ids (default: all usedInRun)")
    parser.add_argument("--start", help="ISO UTC; with --end, overrides the run.yaml window")
    parser.add_argument("--end", help="ISO UTC, exclusive")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format=_LOG_FORMAT)
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end go together")
    with (args.config_dir / "run.yaml").open(encoding="utf-8") as fh:
        run_section = RunSection.model_validate(yaml.safe_load(fh))
    ctx = LocalContext(
        run_id=args.run_dir.name,
        run_dir=args.run_dir,
        cache_dir=args.cache_dir,
        config=LocalConfig(run=run_section, signal=load_signal_config(args.config_dir)),
    )
    window = None if args.start is None else (_parse_utc(args.start), _parse_utc(args.end))
    station_ids = None
    if args.stations:
        station_ids = [s.strip() for s in args.stations.split(",") if s.strip()]
    run_picking(ctx, station_ids=station_ids, window=window)
    print(f"picks: {ctx.path(PICKS_FILE)}")
    print(f"report: {ctx.path(REPORT_FILE)}")
    print("(CLI run: run.json is not updated; run the stage through hq to record it)")
    return 0


if __name__ == "__main__":
    # Delegate to the imported module (hq.pick imports it for the stage registry), so what is
    # pickled for spawned workers is hq.pick.run.*, never __main__.*.
    from hq.pick.run import main as _main

    sys.exit(_main())

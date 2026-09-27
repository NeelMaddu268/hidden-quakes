"""Scoring the STA/LTA threshold sweep (SEIS-07) on VAL-01's scale.

A grid point's picks go through exactly the rerun H4's baseline comparison (VAL-01,
``hq.validate.baseline``) makes for its ``BaselineRow``, so a sweep row and a VAL-01 row are on
one scale:

- H2's four functions from ``hq.validate.lanes.real_seismology_api(cache_dir=ctx.cache_dir,
  run_id=ctx.run_id)`` (REQ-H2-8: ``locate`` reads the run's travel-time table cache);
- association profile ``full`` (``hq.validate.baseline.GAIN_PROFILE``, the profile
  ``BaselineGain`` is quoted on; the lane doc's objective is the baseline's own Tier A count),
  through ``select_profile`` / ``profile_config``. VAL-01 claims its gain only when it also holds
  in ``p_only``, which this sweep does not optimise (recorded as ``profileNote``);
- ``rerun_tables(..., thresholds=<the run's ProcessingRun.tiering>)`` (REQ-H2-9): every point is
  tiered against the run's own bars, checked with ``require_thresholds``; a run without them
  (stage ``tier`` has not run) fails loudly naming H2's tier stage, and no bars are invented;
- ``summarize_row("stalta", "full", events, matches)``: the point's ``BaselineRow``.

Statics. A scoring run through this module locates WITHOUT station statics, while the run's bars
come from statics-corrected events. The reason is H2's ``statics.mode: referenceEvents``
(seismology.yaml): its terms need a match pass, which ``hq.locate.locate`` does not have, so
``locate()`` runs its pass 1 with no statics (H4's ``NO_STATICS_NOTE``); ``real_api`` binds only
``cache_dir`` and ``run_id`` and passes no statics table. That stays true until the run's
``statics.parquet`` reaches ``locate`` here, the way REQ-H1-5 decided (option (a):
``locate(..., statics=<the run's table>)``, H2 PR #94; H4 wires it into VAL-01's reruns).
Meanwhile H4's ``rerun_notes`` records ``staticsApplied``: check ``baseline_reference.json``
``notes`` and the PhaseNet reference row against the run's own tiers before copying a best point
into ``baseline.chosen``. The run's terms are residuals of PhaseNet picks at the public regional
catalog's hypocentres of matched events, so they absorb PhaseNet's station timing bias and favour
PhaseNet; report that beside any comparison. Option (b), bars from a no-statics PhaseNet rerun,
is superseded; the showcase run's ``baseline_tuning_b.json`` is its historical record and no
stage reads it.

The PhaseNet ``picks.parquet`` of the same stations and window is scored once through the same
path (method ``phasenet``) as the reference the baseline's best point is read against (docs/03
baseline kill switch). It is scored alone, before any grid point: its ``locate`` builds the
run's travel-time tables once, so points scored in parallel never race to write them into a
cold ``<cache_dir>/ttgrids/``.

Objective and diagnostics. The objective is Tier A over every candidate event of a rerun
(matched and additional, as ``BaselineRow.tiers`` counts them). When every scored point has the
same Tier A count the objective is flat: the sweep ranks nothing, no point is reported as the
best (``best`` null, a WARNING), and ``baseline.chosen`` is not validated by the sweep. Per
rerun the record keeps H2's tier counts split into matched and additional events, and
``barsMet``: how many candidate events meet each metric bar of the run's record, and all of them
at once (a diagnostic of which bar the candidates miss; the tiers themselves come from H2's
``assign_tiers`` alone, which also applies the depthOnEdge and nearest-station rules).

Search. ``scoreMode`` ``all`` scores every grid point (exhaustive). ``coordinate`` is a
coordinate descent from ``baseline.chosen``: vary ``pOn`` over its list at the incumbent's
``sOn`` and off level, move to the point with the most Tier A events; then ``sOn``; then the off
level; repeat whole passes until a pass moves nowhere, at most ``maxPasses``. Ties keep the
incumbent, then the first point in grid order. A point already scored is never scored again. Its
result is a coordinate-wise local optimum: its Tier A is a lower bound on the grid maximum.
``none`` scores nothing. Unscored points stay null in ``baseline_sweep.parquet``.

Parallelism. Points are scored in ``scoreWorkers`` spawned processes (each H2 ``locate`` call
spawns ``seismology.locator.nWorkers`` more; the product is logged against the CPU count). A
point's picks are built in the parent and sent to its worker; at most ``scoreWorkers`` points are
in flight, so the parent never holds more pick sets than that. Results are keyed by grid index,
so the output never depends on completion order. Any error from H2 stops the stage, with the
point named; points already running cannot be interrupted and finish in their workers before the
process exits (the error is logged first).
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from hq_contracts.models import BaselineRow

from hq.config.run import RunSection
from hq.config.validate import POnlyAssociatorConfig
from hq.validate.baseline import GAIN_PROFILE, Method, rerun_tables, summarize_row
from hq.validate.lanes import SeismologyApi, real_seismology_api
from hq.validate.notes import rerun_notes
from hq.validate.null_test import profile_config, require_thresholds, select_profile

log = logging.getLogger(__name__)

PROFILE = GAIN_PROFILE  # "full": VAL-01's profile for the quoted gain
STALTA: Method = "stalta"
PHASENET: Method = "phasenet"
REFERENCE_KEY = "phasenet"  # job key of the PhaseNet reference
REFERENCE_FILE = "baseline_reference.json"
PHASENET_PICKS_FILE = "picks.parquet"
CATALOG_FILE = "catalog.parquet"
THRESHOLDS_WHAT = "the baseline sweep"  # how require_thresholds names the caller
AXES = ("pOn", "sOn", "off")  # coordinate order of a grid point and of the descent
H2_MISSING_MESSAGE = "H2 pipeline not merged; sweep has pick counts only"
H2_OWNER = "H2 Seismology"
BARRED_TIERS = ("A", "B")  # tiers with metric bars in ProcessingRun.tiering["thresholds"] (H2)
TIER_COUNT_SETS = ("all", "matched", "additional")  # keys of H2's tiering["counts"]
QUALITY_PREFIX = "quality_"  # docs/02 section 2: SeismicEvent.quality flattened
OBJECTIVE = (
    "tiers.A over every candidate event of a rerun (matched and additional), on the run's own "
    "bars (ProcessingRun.tiering)"
)
FLAT_NOTE = (
    "Every scored point has the same Tier A count: the sweep ranks no thresholds, no point is "
    "reported as the best, and baseline.chosen is not validated by it. barsMet shows which of "
    "the run's bars the candidates miss."
)
PROFILE_NOTE = (
    "Thresholds are tuned on association profile full only; VAL-01 claims its gain only when it "
    "also holds in p_only, which this sweep does not optimise."
)
SCOPE = {
    "all": "exhaustive: every grid point is scored",
    "coordinate": (
        "coordinate-wise local optimum from baseline.chosen: its Tier A is a lower bound on the "
        "grid maximum"
    ),
}

# H2's library API (docs/02 section 5). Only checked for presence here: the calls go through
# ``real_seismology_api`` (REQ-H2-8 bindings).
_H2_API = (
    ("hq.associate", "associate"),
    ("hq.locate", "locate"),
    ("hq.match", "match"),
    ("hq.tier", "assign_tiers"),
)

Coords = tuple[float, float, float]  # (pOn, sOn, off) of one grid point
ScoreMode = Literal["all", "coordinate", "none"]
TieringProvider = Callable[[], Mapping[str, Any] | None]  # the run's ProcessingRun.tiering
ApiFactory = Callable[[Path, str], SeismologyApi]


class H2PipelineMissingError(RuntimeError):
    """H2's associate/locate/match/assign_tiers (docs/02 section 5) are not importable."""


def check_h2_merged() -> None:
    """``H2PipelineMissingError`` unless H2's four library functions import.

    A module that exists but fails to import for another reason (a missing dependency, a bug)
    raises its own error: only a missing H2 module or function counts as "not merged".
    """
    missing: list[str] = []
    for module_name, attr in _H2_API:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            missing.append(module_name)
            continue
        if not callable(getattr(module, attr, None)):
            missing.append(f"{module_name}.{attr}")
    if missing:
        raise H2PipelineMissingError(f"{H2_MISSING_MESSAGE} (missing: {', '.join(missing)})")


def context_tiering(ctx: Any) -> TieringProvider:
    """The default provider: ``ctx.read_run().tiering`` (H4's ``RunContext``, docs/02 §4)."""
    read_run = getattr(ctx, "read_run", None)
    if read_run is None:
        raise TypeError(
            f"{type(ctx).__name__} has no read_run(): the baseline sweep needs the run's "
            "ProcessingRun.tiering (pass tiering= to run_baseline)"
        )

    def provide() -> Mapping[str, Any] | None:
        tiering: Mapping[str, Any] | None = read_run().tiering
        return tiering

    return provide


def real_api(cache_dir: Path, run_id: str) -> SeismologyApi:
    """H2's functions as VAL-01 binds them (top level: the spawned workers call it)."""
    return real_seismology_api(cache_dir=cache_dir, run_id=run_id)


def seismology_value(seismology: Any, section: str, name: str) -> Any:
    """``seismology.<section>.<name>`` of H2's ``SeismologyConfig``, or an error naming it."""
    value = getattr(getattr(seismology, section, None), name, None)
    if value is None:
        raise ValueError(
            f"the baseline sweep needs seismology.{section}.{name} (seismology.yaml, owner "
            f"{H2_OWNER}); got {type(seismology).__name__}"
        )
    return value


# --- one scoring rerun -------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreShared:
    """What every rerun of one sweep shares; pickled to the workers."""

    stations: pd.DataFrame
    catalog: pd.DataFrame
    seismology: Any  # H2's SeismologyConfig
    run: RunSection
    thresholds: Mapping[str, Any]  # the run's ProcessingRun.tiering (REQ-H2-9)
    cache_dir: Path
    run_id: str
    api_factory: ApiFactory = real_api  # must be a top-level function for the process runner


@dataclass(frozen=True)
class ScoreJob:
    """One pick set to score: a grid point (``index``) or the PhaseNet reference."""

    key: str
    method: Method
    picks: pd.DataFrame
    index: int | None = None  # grid index; None for the reference


@dataclass(frozen=True)
class JobResult:
    key: str
    method: Method
    index: int | None
    nPicks: int
    row: BaselineRow
    tiering: dict[str, Any] | None  # H2's TierResult.tiering; None when nothing was tiered
    barsMet: dict[str, dict[str, Any]]  # bars_met of the rerun's final events
    runtimeS: float


def _meets_bar(events: pd.DataFrame, metric: str, bar: Mapping[str, Any]) -> np.ndarray:
    """Which events meet one bar of the record (``op`` and ``value``); a null never does."""
    if len(events) == 0:
        return np.zeros(0, dtype=bool)
    column = QUALITY_PREFIX + metric
    if column not in events.columns:
        raise ValueError(
            f"assign_tiers events lack {column} for the {metric} bar (docs/02 section 2 "
            f"events.parquet, owner {H2_OWNER})"
        )
    values = pd.to_numeric(events[column]).to_numpy(dtype=np.float64, na_value=np.nan)
    op, value = bar.get("op"), float(bar["value"])
    if op == ">=":
        return values >= value
    if op == "<=":
        return values <= value
    raise ValueError(f"the {metric} bar has op {op!r}; H2's tiering record uses '>=' or '<='")


def bars_met(events: pd.DataFrame, record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per barred tier of the run's ``tiering["thresholds"]`` record: how many of ``events``
    meet each metric bar (``perBar``) and every one at once (``everyBar``). A diagnostic only:
    the depthOnEdge and nearest-station rules of Tier A are not applied here."""
    out: dict[str, dict[str, Any]] = {}
    for tier in BARRED_TIERS:
        bars = record.get(tier)
        if not isinstance(bars, Mapping):
            raise TypeError(
                f"the run's tier bars have no {tier!r} record (ProcessingRun.tiering.thresholds, "
                f"owner {H2_OWNER})"
            )
        every = np.ones(len(events), dtype=bool)
        per_bar: dict[str, int] = {}
        for metric, bar in bars.items():
            meets = _meets_bar(events, str(metric), bar)
            per_bar[str(metric)] = int(meets.sum())
            every &= meets
        out[tier] = {"perBar": per_bar, "everyBar": int(every.sum())}
    return out


def score_job(shared: ScoreShared, job: ScoreJob, api: SeismologyApi) -> JobResult:
    """VAL-01's rerun of one pick set (module docstring)."""
    began = perf_counter()
    picks = select_profile(job.picks, PROFILE)
    # The p_only overrides only apply to the p_only profile; the sweep scores "full".
    cfg = profile_config(shared.seismology, PROFILE, POnlyAssociatorConfig())
    events, matches, tiering = rerun_tables(
        picks, shared.stations, shared.catalog, api, cfg, shared.run, thresholds=shared.thresholds
    )
    row = summarize_row(job.method, PROFILE, events, matches)
    return JobResult(
        key=job.key,
        method=job.method,
        index=job.index,
        nPicks=len(picks),
        row=row,
        tiering=tiering,
        barsMet=bars_met(events, shared.thresholds["thresholds"]),
        runtimeS=perf_counter() - began,
    )


def tier_counts(tiering: Mapping[str, Any] | None) -> dict[str, dict[str, int]] | None:
    """H2's tier counts of one rerun, split into all, matched and additional (unmatched) events;
    None when the rerun reached no ``assign_tiers``."""
    if tiering is None:
        return None
    counts = tiering.get("counts")
    if not isinstance(counts, Mapping) or any(
        not isinstance(counts.get(k), Mapping) for k in TIER_COUNT_SETS
    ):
        raise ValueError(
            f"assign_tiers returned a tiering record without counts {list(TIER_COUNT_SETS)} "
            f"(owner: {H2_OWNER})"
        )
    return {k: {t: int(n) for t, n in counts[k].items()} for k in TIER_COUNT_SETS}


# --- runners: where the reruns execute ----------------------------------------------------------


class Runner(Protocol):
    """Scores ``jobs`` (consumed lazily) and yields each result as it completes, in any order."""

    def __call__(self, shared: ScoreShared, jobs: Iterable[ScoreJob]) -> Iterator[JobResult]: ...


@dataclass
class SerialRunner:
    """In this process, one job at a time (``scoreWorkers`` 1). One instance serves one sweep:
    H2's API is built on the first call and reused for every later batch."""

    api: SeismologyApi | None = None

    def __call__(self, shared: ScoreShared, jobs: Iterable[ScoreJob]) -> Iterator[JobResult]:
        if self.api is None:
            self.api = shared.api_factory(shared.cache_dir, shared.run_id)
        for job in jobs:
            try:
                result = score_job(shared, job, self.api)
            except Exception as exc:  # not handled: only name the pick set, as ProcessRunner does
                exc.add_note(f"baseline sweep: scoring {job.key}")
                raise
            yield result


_worker_api: list[SeismologyApi] = []  # one API per worker process


def _init_worker(log_level: int) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
    )


def _score_in_worker(shared: ScoreShared, job: ScoreJob) -> JobResult:
    if not _worker_api:
        _worker_api.append(shared.api_factory(shared.cache_dir, shared.run_id))
    return score_job(shared, job, _worker_api[0])


@dataclass(frozen=True)
class ProcessRunner:
    """``workers`` spawned processes; at most ``workers`` jobs are built and in flight.

    A job's error is re-raised at once with the pick set named. Jobs not started yet are
    cancelled; jobs already running cannot be interrupted and finish in their workers before the
    process exits (logged).
    """

    workers: int

    def __call__(self, shared: ScoreShared, jobs: Iterable[ScoreJob]) -> Iterator[JobResult]:
        pending = iter(jobs)
        in_flight: dict[Future[JobResult], tuple[int, ScoreJob]] = {}
        submitted = 0
        finished = False
        pool = ProcessPoolExecutor(
            self.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(logging.getLogger().getEffectiveLevel(),),
        )
        try:
            while True:
                while len(in_flight) < self.workers:
                    job = next(pending, None)
                    if job is None:
                        break
                    in_flight[pool.submit(_score_in_worker, shared, job)] = (submitted, job)
                    submitted += 1
                if not in_flight:
                    break
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in sorted(done, key=lambda f: in_flight[f][0]):
                    _, job = in_flight.pop(fut)
                    exc = fut.exception()
                    if exc is not None:
                        exc.add_note(f"baseline sweep: scoring {job.key}")
                        raise exc
                    yield fut.result()
            finished = True
        finally:
            if finished:
                pool.shutdown(wait=True)
            else:  # an error, or the consumer stopped: do not wait here for running points
                running = [job.key for fut, (_, job) in in_flight.items() if not fut.done()]
                pool.shutdown(wait=False, cancel_futures=True)
                if running:
                    log.warning(
                        "baseline sweep: stopping; %s already running cannot be interrupted and "
                        "finish in their worker processes before this process exits",
                        ", ".join(running),
                    )


def make_runner(workers: int) -> Runner:
    return SerialRunner() if workers == 1 else ProcessRunner(workers)


# --- search ------------------------------------------------------------------------------------


def best_of(candidates: Sequence[int], tier_a: Mapping[int, int], incumbent: int | None) -> int:
    """The candidate with the most Tier A events; ties keep ``incumbent``, then grid order."""
    top = max(tier_a[k] for k in candidates)
    if incumbent is not None and incumbent in candidates and tier_a[incumbent] == top:
        return incumbent
    return min(k for k in candidates if tier_a[k] == top)


@dataclass(frozen=True)
class SearchStep:
    """One coordinate step: ``axis`` varied over ``candidates`` (grid order) at the others."""

    passNo: int
    axis: str
    candidates: tuple[int, ...]
    tierA: tuple[int, ...]  # aligned with candidates
    best: int
    moved: bool


@dataclass(frozen=True)
class SearchResult:
    best: int
    steps: tuple[SearchStep, ...]
    converged: bool  # a pass moved nowhere (False: stopped at maxPasses)


def axis_line(coords: Sequence[Coords], at: int, axis: int) -> list[int]:
    """Grid indices that differ from point ``at`` only on ``axis``, in grid order."""
    fixed = coords[at]
    return [
        k
        for k, c in enumerate(coords)
        if all(c[j] == fixed[j] for j in range(len(AXES)) if j != axis)
    ]


def coordinate_search(
    coords: Sequence[Coords],
    start: int,
    max_passes: int,
    score: Callable[[Sequence[int]], Mapping[int, int]],
) -> SearchResult:
    """Coordinate descent on Tier A from grid point ``start`` (module docstring).

    ``score(indices)`` scores every index not scored yet and returns the Tier A count of each
    index asked for.
    """
    incumbent = start
    steps: list[SearchStep] = []
    for pass_no in range(1, max_passes + 1):
        moved_in_pass = False
        for axis, name in enumerate(AXES):
            line = axis_line(coords, incumbent, axis)
            tier_a = score(line)
            best = best_of(line, tier_a, incumbent)
            moved = best != incumbent
            steps.append(
                SearchStep(pass_no, name, tuple(line), tuple(tier_a[k] for k in line), best, moved)
            )
            log.info(
                "baseline sweep search: pass %d, vary %s at %s: Tier A %s; %s",
                pass_no,
                name,
                {a: coords[incumbent][j] for j, a in enumerate(AXES) if j != axis},
                {coords[k][axis]: tier_a[k] for k in line},
                f"move to {name} {coords[best][axis]}" if moved else "stay",
            )
            incumbent = best
            moved_in_pass |= moved
        if not moved_in_pass:
            return SearchResult(incumbent, tuple(steps), converged=True)
    log.warning(
        "baseline sweep search: still moving after baseline.sweep.maxPasses = %d passes; the "
        "best point so far is reported",
        max_passes,
    )
    return SearchResult(incumbent, tuple(steps), converged=False)


# --- the sweep ---------------------------------------------------------------------------------


def _bars_summary(bars: Mapping[str, Mapping[str, Any]]) -> str:
    a = bars["A"]
    per_bar = ", ".join(f"{m} {n}" for m, n in a["perBar"].items())
    return f"{per_bar or 'no bars'}; every A bar {a['everyBar']}"


@dataclass
class SweepScorer:
    """Scores grid points on demand (the reference alone first); keeps the results."""

    shared: ScoreShared
    runner: Runner
    coords: Sequence[Coords]
    params: Sequence[Mapping[str, float]]  # per grid index, for the logs
    point_picks: Callable[[int], pd.DataFrame]  # the STA/LTA picks of a grid point
    reference_picks: pd.DataFrame
    results: dict[int, JobResult] = field(default_factory=dict)
    reference: JobResult | None = None

    def _point_jobs(self, todo: Sequence[int]) -> Iterator[ScoreJob]:
        for k in todo:  # built lazily: the runner pulls one when a worker is free
            yield ScoreJob(f"grid point {k}", STALTA, self.point_picks(k), index=k)

    def _run(self, jobs: Iterable[ScoreJob], what: str) -> None:
        began = perf_counter()
        for res in self.runner(self.shared, jobs):
            self._store(res)
        log.info(
            "baseline sweep: scored %s in %.1f s (%d point(s) scored so far)",
            what,
            perf_counter() - began,
            len(self.results),
        )

    def score(self, indices: Sequence[int]) -> dict[int, int]:
        if self.reference is None:
            # Alone, before any point: its locate builds the run's travel-time tables once, so
            # points in parallel never race to write them into a cold cache.
            self._run([ScoreJob(REFERENCE_KEY, PHASENET, self.reference_picks)], "the reference")
            if self.reference is None:
                raise RuntimeError("baseline sweep: the runner returned no PhaseNet reference")
        todo = [k for k in dict.fromkeys(indices) if k not in self.results]
        if todo:
            self._run(self._point_jobs(todo), f"a batch of {len(todo)} point(s)")
            missing = [k for k in todo if k not in self.results]
            if missing:
                raise RuntimeError(f"baseline sweep: the runner returned no result for {missing}")
        return {k: self.results[k].row.tiers.A for k in indices}

    def _store(self, res: JobResult) -> None:
        if res.index is None:
            if res.key != REFERENCE_KEY or self.reference is not None:
                raise RuntimeError(f"baseline sweep: unexpected result {res.key}")
            self.reference = res
            what = "PhaseNet reference (picks.parquet)"
        else:
            if res.index in self.results:
                raise RuntimeError(f"baseline sweep: {res.key} scored twice")
            self.results[res.index] = res
            what = f"point {res.index + 1}/{len(self.coords)} {dict(self.params[res.index])}"
        r = res.row
        split = tier_counts(res.tiering)
        log.info(
            "baseline sweep %s: %d picks -> %d candidates, %d recovered public, tiers A %d / "
            "B %d / C %d (Tier A matched %s, additional %s), median rmsS %.3f s, median stations "
            "%.1f; A bars met: %s; %.1f s",
            what,
            res.nPicks,
            r.candidates,
            r.recoveredPublic,
            r.tiers.A,
            r.tiers.B,
            r.tiers.C,
            "-" if split is None else split["matched"].get("A", 0),
            "-" if split is None else split["additional"].get("A", 0),
            r.medianRmsS,
            r.medianStations,
            _bars_summary(res.barsMet),
            res.runtimeS,
        )


@dataclass(frozen=True)
class SweepScores:
    """What scoring produced: every scored point's result by grid index, the reference, the best
    Tier A point (the ``chosen`` incumbent on ties) and how it was found."""

    mode: ScoreMode
    grid_size: int
    results: Mapping[int, JobResult]
    reference: JobResult
    best: int
    chosen: int
    steps: tuple[SearchStep, ...]
    converged: bool
    runtimeS: float

    @property
    def rows(self) -> list[BaselineRow | None]:
        """One row per grid index; None where the point was not scored."""
        return [self.results[k].row if k in self.results else None for k in range(self.grid_size)]

    @property
    def scored(self) -> int:
        return len(self.results)

    @property
    def tier_a_values(self) -> list[int]:
        """The distinct Tier A counts over the scored points."""
        return sorted({r.row.tiers.A for r in self.results.values()})

    @property
    def flat(self) -> bool:
        """Every scored point has the same Tier A count: nothing is ranked."""
        return len(self.tier_a_values) <= 1

    @property
    def best_row(self) -> BaselineRow:
        if self.best not in self.results:
            raise RuntimeError(f"baseline sweep: best point {self.best} has no score")
        return self.results[self.best].row

    @property
    def tierings(self) -> list[dict[str, Any] | None]:
        """Every rerun's tiering record: the reference, then the points in grid order."""
        return [self.reference.tiering] + [self.results[k].tiering for k in sorted(self.results)]


def score_sweep(mode: ScoreMode, chosen: int, max_passes: int, scorer: SweepScorer) -> SweepScores:
    """Score the grid in ``mode`` (``all`` or ``coordinate``) and name the best Tier A point
    (``chosen``: the grid index of ``baseline.chosen``, the incumbent on ties)."""
    began = perf_counter()
    coords = scorer.coords
    if mode == "all":
        tier_a = scorer.score(range(len(coords)))
        best = best_of(range(len(coords)), tier_a, chosen)
        steps: tuple[SearchStep, ...] = ()
        converged = True
    elif mode == "coordinate":
        found = coordinate_search(coords, chosen, max_passes, scorer.score)
        best, steps, converged = found.best, found.steps, found.converged
    else:
        raise ValueError(f"score_sweep needs scoreMode all or coordinate, got {mode!r}")
    if scorer.reference is None:  # scored before the first point
        raise RuntimeError("baseline sweep: the PhaseNet reference was never scored")
    return SweepScores(
        mode=mode,
        grid_size=len(coords),
        results=dict(scorer.results),
        reference=scorer.reference,
        best=best,
        chosen=chosen,
        steps=steps,
        converged=converged,
        runtimeS=perf_counter() - began,
    )


def point_record(params: Mapping[str, float], row: BaselineRow | None) -> dict[str, Any]:
    return {"params": dict(params), "row": None if row is None else row.model_dump(mode="json")}


def job_diagnostics(res: JobResult) -> dict[str, Any]:
    """A rerun's pick count, H2's tier counts split by matched/additional, and ``barsMet``."""
    return {"picks": res.nPicks, "tierCounts": tier_counts(res.tiering), "barsMet": res.barsMet}


def scoring_record(
    scores: SweepScores,
    params: Sequence[Mapping[str, float]],
    thresholds: Mapping[str, Any],
    workers: int,
    window: tuple[float, float],
    station_ids: Sequence[str],
) -> dict[str, Any]:
    """Contents of ``baseline_reference.json``; ``record_view`` of it goes to
    ``ProcessingRun.picker["baseline"]["sweepScoring"]``. No runtimes: the file is deterministic
    (they are logged, and the total is recorded as ``scoringRuntimeS``)."""
    ref = scores.reference.row
    top = max(scores.tier_a_values)
    notes = rerun_notes(scores.tierings, thresholds["thresholds"], {}, THRESHOLDS_WHAT)
    return {
        "mode": scores.mode,
        "associationProfile": PROFILE,
        "profileNote": PROFILE_NOTE,
        "scoreWorkers": workers,
        "window": {"t0": window[0], "t1": window[1]},
        "stationIds": list(station_ids),
        "gridPoints": scores.grid_size,
        "pointsScored": scores.scored,
        "objective": {
            "metric": OBJECTIVE,
            "maxTierA": top,
            "tierAValues": scores.tier_a_values,
            "flat": scores.flat,
            "note": FLAT_NOTE if scores.flat else None,
        },
        "phasenetReference": {
            "row": ref.model_dump(mode="json"),
            **job_diagnostics(scores.reference),
        },
        "best": None if scores.flat else point_record(params[scores.best], scores.best_row),
        "chosenAtScoring": point_record(params[scores.chosen], scores.rows[scores.chosen]),
        # The most Tier A events over PhaseNet's: docs/03's kill switch reads "within ~20%".
        "tierARatioToPhasenet": None if ref.tiers.A == 0 else top / ref.tiers.A,
        "search": {
            "scope": SCOPE[scores.mode],
            "converged": scores.converged,
            "path": [
                {
                    "pass": s.passNo,
                    "axis": s.axis,
                    "points": [
                        {"params": dict(params[k]), "tierA": a}
                        for k, a in zip(s.candidates, s.tierA, strict=True)
                    ],
                    "best": dict(params[s.best]),
                    "moved": s.moved,
                }
                for s in scores.steps
            ],
        },
        "points": [
            {
                "index": k,
                **point_record(params[k], scores.results[k].row),
                **job_diagnostics(scores.results[k]),
            }
            for k in sorted(scores.results)
        ],
        "notes": notes.model_dump(mode="json"),
    }


def record_view(scoring: Mapping[str, Any]) -> dict[str, Any]:
    """``scoring`` without its per-point list (kept in ``baseline_reference.json`` only, so
    ``run.json`` stays small)."""
    return {k: v for k, v in scoring.items() if k != "points"}


def reference_picks(
    picks: pd.DataFrame, station_ids: Sequence[str], t0: float, t1: float
) -> pd.DataFrame:
    """PhaseNet picks of the sweep's stations in its window ``[t0, t1)``, logged."""
    keep = (
        picks["stationId"].astype(str).isin(set(station_ids))
        & (picks["t"] >= t0)
        & (picks["t"] < t1)
    )
    out = picks[keep.to_numpy(dtype=bool)].reset_index(drop=True)
    log.info(
        "baseline sweep: PhaseNet reference keeps %d of %d picks.parquet picks (the sweep's %d "
        "stations and window)",
        len(out),
        len(picks),
        len(station_ids),
    )
    return out


@dataclass(frozen=True)
class ScorePlan:
    """Everything scoring needs, gathered before any picking (so a bad run fails fast)."""

    mode: ScoreMode
    shared: ScoreShared
    phasenet_picks: pd.DataFrame  # all of picks.parquet; reference_picks narrows it


def check_pick_prob(pick_prob: float, seismology: Any) -> None:
    """Every STA/LTA pick must reach the associator (``prob >= associator.minPickProb``)."""
    min_prob = float(seismology_value(seismology, "associator", "minPickProb"))
    if pick_prob < min_prob:
        raise ValueError(
            f"baseline.prob {pick_prob} is below seismology.associator.minPickProb {min_prob}: "
            "associate would drop every STA/LTA pick"
        )


def log_process_budget(score_workers: int, seismology: Any) -> None:
    """Log scoring workers x locate processes against the CPU count; WARNING above it."""
    locate_workers = int(seismology_value(seismology, "locator", "nWorkers"))
    total = score_workers * locate_workers
    cpus = os.cpu_count()
    log.info(
        "baseline sweep: %d scoring worker(s) x seismology.locator.nWorkers %d = up to %d locate "
        "processes at once, on %s CPUs",
        score_workers,
        locate_workers,
        total,
        cpus,
    )
    if cpus is not None and total > cpus:
        log.warning(
            "baseline sweep: %d locate processes at once exceed the %d CPUs; lower "
            "baseline.sweep.scoreWorkers (each worker also holds a point's picks and H2's "
            "travel-time tables in memory)",
            total,
            cpus,
        )


def prepare_scoring(
    ctx: Any,
    mode: ScoreMode,
    stations: pd.DataFrame,
    tiering: TieringProvider | None,
    *,
    pick_prob: float,
    score_workers: int,
    h2_required: bool,
) -> ScorePlan | None:
    """``None`` when ``mode`` is ``none``, or when H2 is not merged and ``h2_required`` is false
    (WARNING); else the plan.

    ``stations`` is the run's whole ``stations.parquet``, as VAL-01 passes it. Raises (naming
    who writes the missing piece) when H2 is not merged and ``h2_required`` (an explicit scoring
    request that would otherwise end in a null sweep), or when the run has no tier bars, no
    ``catalog.parquet``, no PhaseNet ``picks.parquet`` or no seismology config, or when
    ``pick_prob`` is below the associator's ``minPickProb``.
    """
    from hq_contracts.io import read_table

    if mode == "none":
        log.info("baseline sweep: scoreMode none; candidates, recoveredPublic and tiers are null")
        return None
    try:
        check_h2_merged()
    except H2PipelineMissingError as exc:
        if h2_required:
            raise
        log.warning("baseline sweep: %s; every score column is null", exc)
        return None
    provider = context_tiering(ctx) if tiering is None else tiering
    thresholds = require_thresholds(provider(), THRESHOLDS_WHAT)
    seismology = getattr(ctx.config, "seismology", None)
    if seismology is None:
        raise ValueError(
            f"the baseline sweep needs the seismology config section (seismology.yaml, owner "
            f"{H2_OWNER})"
        )
    check_pick_prob(pick_prob, seismology)
    log_process_budget(score_workers, seismology)
    tables: dict[str, pd.DataFrame] = {}
    for name, writer in (
        (CATALOG_FILE, f"stage catalog, owner {H2_OWNER}"),
        (PHASENET_PICKS_FILE, "stage pick, owner H1 Signal"),
    ):
        path = ctx.path(name)
        if not Path(path).is_file():
            raise FileNotFoundError(f"the baseline sweep needs {path} ({writer})")
        tables[name] = read_table(path)
    picks = tables[PHASENET_PICKS_FILE]
    if picks.attrs.get("model") != "Pick":
        raise ValueError(f"{PHASENET_PICKS_FILE} holds {picks.attrs.get('model')} rows, not Pick")
    log.info(
        "baseline sweep: scoreMode %s, profile %s, tier bars from the run's ProcessingRun.tiering "
        "(%s matched events), %d catalog events, %d PhaseNet picks",
        mode,
        PROFILE,
        thresholds["thresholds"].get("nMatched"),
        len(tables[CATALOG_FILE]),
        len(picks),
    )
    shared = ScoreShared(
        stations=stations,
        catalog=tables[CATALOG_FILE],
        seismology=seismology,
        run=ctx.config.run,
        thresholds=dict(thresholds),
        cache_dir=Path(ctx.cache_dir),
        run_id=str(ctx.run_id),
    )
    return ScorePlan(mode=mode, shared=shared, phasenet_picks=picks)

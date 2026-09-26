"""Scoring the STA/LTA threshold sweep (SEIS-07) on VAL-01's scale.

A grid point's picks go through exactly the rerun H4's baseline comparison (VAL-01,
``hq.validate.baseline``) makes for its ``BaselineRow``, so a sweep row and a VAL-01 row are on
one scale:

- H2's four functions from ``hq.validate.lanes.real_seismology_api(cache_dir=ctx.cache_dir,
  run_id=ctx.run_id)`` (REQ-H2-8: ``locate`` reads the run's travel-time table cache);
- association profile ``full`` (``hq.validate.baseline.GAIN_PROFILE``, the profile
  ``BaselineGain`` is quoted on; the lane doc's objective is the baseline's own Tier A count),
  through ``select_profile`` / ``profile_config``;
- ``rerun_tables(..., thresholds=<the run's ProcessingRun.tiering>)`` (REQ-H2-9): every point is
  tiered against the run's own bars, checked with ``require_thresholds``; a run without them
  (stage ``tier`` has not run) fails loudly naming H2's tier stage, and no bars are invented;
- ``summarize_row("stalta", "full", events, matches)``: the point's ``BaselineRow``.

The PhaseNet ``picks.parquet`` of the same stations and window is scored once through the same
path (method ``phasenet``) as the reference the baseline's best point is read against (docs/03
baseline kill switch).

Search. ``scoreMode`` ``all`` scores every grid point. ``coordinate`` is a coordinate descent
from ``baseline.chosen``: vary ``pOn`` over its list at the incumbent's ``sOn`` and off level,
move to the point with the most Tier A events; then ``sOn``; then the off level; repeat whole
passes until a pass moves nowhere, at most ``maxPasses``. Ties keep the incumbent, then the first
point in grid order. A point already scored is never scored again. ``none`` scores nothing.
Unscored points stay null in ``baseline_sweep.parquet``.

Parallelism. Points are scored in ``scoreWorkers`` spawned processes (each H2 ``locate`` call
spawns ``seismology.locator.nWorkers`` more). A point's picks are built in the parent and sent to
its worker; at most ``scoreWorkers`` points are in flight, so the parent never holds more pick
sets than that. Results are keyed by grid index, so the output never depends on completion
order. Any error from H2 stops the stage, with the point named.
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

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
    runtimeS: float


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
        runtimeS=perf_counter() - began,
    )


# --- runners: where the reruns execute ----------------------------------------------------------


class Runner(Protocol):
    """Scores ``jobs`` (consumed lazily) and yields each result as it completes, in any order."""

    def __call__(self, shared: ScoreShared, jobs: Iterable[ScoreJob]) -> Iterator[JobResult]: ...


def serial_runner(shared: ScoreShared, jobs: Iterable[ScoreJob]) -> Iterator[JobResult]:
    """In this process, one job at a time (``scoreWorkers`` 1)."""
    api = shared.api_factory(shared.cache_dir, shared.run_id)
    for job in jobs:
        try:
            result = score_job(shared, job, api)
        except Exception as exc:  # not handled: only name the pick set, as the process runner does
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

    A job's error is re-raised with the pick set named; pending jobs are cancelled first, and the
    ones already running finish before it propagates.
    """

    workers: int

    def __call__(self, shared: ScoreShared, jobs: Iterable[ScoreJob]) -> Iterator[JobResult]:
        pending = iter(jobs)
        in_flight: dict[Future[JobResult], tuple[int, ScoreJob]] = {}
        submitted = 0
        with ProcessPoolExecutor(
            self.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(logging.getLogger().getEffectiveLevel(),),
        ) as pool:
            try:
                while True:
                    while len(in_flight) < self.workers:
                        job = next(pending, None)
                        if job is None:
                            break
                        in_flight[pool.submit(_score_in_worker, shared, job)] = (submitted, job)
                        submitted += 1
                    if not in_flight:
                        return
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in sorted(done, key=lambda f: in_flight[f][0]):
                        _, job = in_flight.pop(fut)
                        exc = fut.exception()
                        if exc is not None:
                            exc.add_note(f"baseline sweep: scoring {job.key}")
                            raise exc
                        yield fut.result()
            except BaseException:
                for fut in in_flight:  # running ones finish; nothing new starts
                    fut.cancel()
                raise


def make_runner(workers: int) -> Runner:
    return serial_runner if workers == 1 else ProcessRunner(workers)


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


@dataclass
class SweepScorer:
    """Scores grid points on demand (plus the reference, with the first batch); keeps results."""

    shared: ScoreShared
    runner: Runner
    coords: Sequence[Coords]
    params: Sequence[Mapping[str, float]]  # per grid index, for the logs
    point_picks: Callable[[int], pd.DataFrame]  # the STA/LTA picks of a grid point
    reference_picks: pd.DataFrame
    results: dict[int, JobResult] = field(default_factory=dict)
    reference: JobResult | None = None

    def _jobs(self, todo: Sequence[int]) -> Iterator[ScoreJob]:
        if self.reference is None:
            yield ScoreJob(REFERENCE_KEY, PHASENET, self.reference_picks)
        for k in todo:  # built lazily: the runner pulls one when a worker is free
            yield ScoreJob(f"grid point {k}", STALTA, self.point_picks(k), index=k)

    def score(self, indices: Sequence[int]) -> dict[int, int]:
        todo = [k for k in dict.fromkeys(indices) if k not in self.results]
        if todo or self.reference is None:
            began = perf_counter()
            for res in self.runner(self.shared, self._jobs(todo)):
                self._store(res)
            missing = [k for k in todo if k not in self.results]
            if missing or self.reference is None:
                raise RuntimeError(f"baseline sweep: the runner returned no result for {missing}")
            log.info(
                "baseline sweep: batch of %d point(s) scored in %.1f s (%d scored so far)",
                len(todo),
                perf_counter() - began,
                len(self.results),
            )
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
        log.info(
            "baseline sweep %s: %d picks -> %d candidates, %d recovered public, tiers A %d / "
            "B %d / C %d, median rmsS %.3f s, median stations %.1f, %.1f s",
            what,
            res.nPicks,
            r.candidates,
            r.recoveredPublic,
            r.tiers.A,
            r.tiers.B,
            r.tiers.C,
            r.medianRmsS,
            r.medianStations,
            res.runtimeS,
        )


@dataclass(frozen=True)
class SweepScores:
    """What scoring produced: one row per grid index (None: not scored), the reference row,
    the best Tier A point and how it was found."""

    mode: ScoreMode
    rows: list[BaselineRow | None]
    reference: JobResult
    best: int
    chosen: int
    steps: tuple[SearchStep, ...]
    converged: bool
    tierings: list[dict[str, Any] | None]
    runtimeS: float

    @property
    def scored(self) -> int:
        return sum(1 for r in self.rows if r is not None)

    @property
    def best_row(self) -> BaselineRow:
        row = self.rows[self.best]
        if row is None:
            raise RuntimeError(f"baseline sweep: best point {self.best} has no score")
        return row


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
    if scorer.reference is None:  # scored with the first batch
        raise RuntimeError("baseline sweep: the PhaseNet reference was never scored")
    rows = [scorer.results[k].row if k in scorer.results else None for k in range(len(coords))]
    tierings = [scorer.reference.tiering] + [
        scorer.results[k].tiering for k in sorted(scorer.results)
    ]
    return SweepScores(
        mode=mode,
        rows=rows,
        reference=scorer.reference,
        best=best,
        chosen=chosen,
        steps=steps,
        converged=converged,
        tierings=tierings,
        runtimeS=perf_counter() - began,
    )


def point_record(params: Mapping[str, float], row: BaselineRow | None) -> dict[str, Any]:
    return {"params": dict(params), "row": None if row is None else row.model_dump(mode="json")}


def scoring_record(
    scores: SweepScores,
    params: Sequence[Mapping[str, float]],
    thresholds: Mapping[str, Any],
    workers: int,
) -> dict[str, Any]:
    """``baseline_reference.json`` and ``ProcessingRun.picker["baseline"]["sweepScoring"]``."""
    best_row = scores.best_row
    ref = scores.reference.row
    notes = rerun_notes(scores.tierings, thresholds["thresholds"], {}, THRESHOLDS_WHAT)
    return {
        "mode": scores.mode,
        "associationProfile": PROFILE,
        "scoreWorkers": workers,
        "pointsScored": scores.scored,
        "phasenetReference": {
            "picks": scores.reference.nPicks,
            "row": ref.model_dump(mode="json"),
        },
        "best": point_record(params[scores.best], best_row),
        "chosen": point_record(params[scores.chosen], scores.rows[scores.chosen]),
        # Tier A of the best point over PhaseNet's: docs/03's kill switch reads "within ~20%".
        "tierARatioToPhasenet": None if ref.tiers.A == 0 else best_row.tiers.A / ref.tiers.A,
        "search": {
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
        "notes": notes.model_dump(mode="json"),
    }


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


def prepare_scoring(
    ctx: Any, mode: ScoreMode, stations: pd.DataFrame, tiering: TieringProvider | None
) -> ScorePlan | None:
    """``None`` when ``mode`` is ``none`` or H2 is not merged (WARNING); else the plan.

    ``stations`` is the run's whole ``stations.parquet``, as VAL-01 passes it. Raises (naming
    who writes the missing piece) when the run has no tier bars, no ``catalog.parquet``, no
    PhaseNet ``picks.parquet`` or no seismology config.
    """
    from hq_contracts.io import read_table

    if mode == "none":
        log.info("baseline sweep: scoreMode none; candidates, recoveredPublic and tiers are null")
        return None
    try:
        check_h2_merged()
    except H2PipelineMissingError as exc:
        log.warning("baseline sweep: %s; every score column is null", exc)
        return None
    provider = context_tiering(ctx) if tiering is None else tiering
    thresholds = require_thresholds(provider(), THRESHOLDS_WHAT)
    seismology = getattr(ctx.config, "seismology", None)
    if seismology is None:
        raise ValueError(
            "the baseline sweep needs the seismology config section (seismology.yaml, owner H2 "
            "Seismology)"
        )
    tables: dict[str, pd.DataFrame] = {}
    for name, writer in (
        (CATALOG_FILE, "stage catalog, owner H2 Seismology"),
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

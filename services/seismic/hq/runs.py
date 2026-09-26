"""Run directories, ``RunContext`` and the stage runner (docs/02 §4, docs/01 → Pipeline).

A run is a directory ``<data>/<mode>/runs/<runId>/`` holding ``run.json`` (a
``hq_contracts.models.ProcessingRun``), a ``stages.json`` sidecar with per-stage runtimes and
counts, and the tables each stage writes. A stage is a function ``run(ctx: RunContext) -> None``
in one module per pipeline step. ``STAGES`` names every module and its owning lane, so a stage
that isn't merged yet fails with a message that says who ships it.

``run.json`` is read and written only through the ``ProcessingRun`` model, never as loose dicts.
"""

import importlib
import importlib.metadata
import logging
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, get_args

from hq_contracts.models import DataMode, ProcessingRun
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from hq.config import RunConfig, load_config

log = logging.getLogger(__name__)

RUN_JSON = "run.json"
STAGES_JSON = "stages.json"
CACHE_DIRNAME = "cache"
RUNS_DIRNAME = "runs"
NO_GIT_SHA = "nogit00"  # stands in for the 7-char sha when the config dir isn't in a git repo
GIT_SHA_LEN = 7
DATA_MODES: tuple[str, ...] = get_args(DataMode.__value__)

# Packages whose versions go into ProcessingRun.softwareVersions (missing ones are skipped).
VERSIONED_PACKAGES: tuple[str, ...] = (
    "obspy",
    "seisbench",
    "pyocto",
    "torch",
    "scikit-fmm",
    "numpy",
    "pandas",
    "scipy",
    "pyarrow",
    "pyproj",
    "pydantic",
)

# ProcessingRun dict fields that hold stage parameters, straight from the contract model.
PARAM_FIELDS: tuple[str, ...] = tuple(
    name for name, field in ProcessingRun.model_fields.items() if field.annotation is dict
)

# Which ProcessingRun dict field a stage's ``record(params=...)`` merges into by default.
# Stages that share a field nest their params under their own key (``{"catalog": {...}}``,
# ``{"inventory": {...}}``) so keys never collide; the pick and match stages own the top level.
STAGE_PARAM_FIELDS: dict[str, str] = {
    "inventory": "picker",
    "download": "picker",
    "pick": "picker",
    "baseline": "picker",
    "associate": "associator",
    "locate": "locator",
    "tier": "tiering",
    "match": "matching",
    "catalog": "matching",
}

# The only top-level ProcessingRun fields a stage may set with ``RunContext.update_run``.
# Everything else (id, mode, createdAt, gitSha, window*, bbox, isSynthetic) is fixed by
# ``create_run``; runtimes and stage params go through ``record``.
UPDATABLE_FIELDS: tuple[str, ...] = (
    "stationIds",
    "pickerModel",
    "pickerWeights",
    "softwareVersions",
)


class RunError(RuntimeError):
    """A run directory, run.json or stage problem the CLI reports without a traceback."""


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage: registry name, module exposing ``run(ctx)``, owning lane."""

    name: str
    module: str
    owner: str


# Pipeline order and ownership, exactly docs/01 → Pipeline.
STAGES: tuple[StageSpec, ...] = (
    StageSpec("inventory", "hq.ingest.inventory", "H1 Signal"),
    StageSpec("catalog", "hq.match.catalog", "H2 Seismology"),
    StageSpec("download", "hq.ingest.download", "H1 Signal"),
    StageSpec("pick", "hq.pick", "H1 Signal"),
    StageSpec("baseline", "hq.baseline", "H1 Signal"),
    StageSpec("associate", "hq.associate", "H2 Seismology"),
    StageSpec("locate", "hq.locate", "H2 Seismology"),
    StageSpec("match", "hq.match", "H2 Seismology"),
    StageSpec("tier", "hq.tier", "H2 Seismology"),
    StageSpec("magnitude", "hq.magnitude", "H2 Seismology"),
    StageSpec("validate", "hq.validate", "H4 Platform"),
    StageSpec("export", "hq.export", "H4 Platform"),
)


class StageMissingError(RunError):
    """The stage's module or its ``run(ctx)`` isn't merged yet; the message names the owner."""

    def __init__(self, spec: StageSpec, reason: str) -> None:
        self.spec = spec
        self.reason = reason
        super().__init__(
            f"stage {spec.name!r} is not implemented yet: {reason} (owner: {spec.owner})"
        )


class UnknownStageError(RunError):
    """A stage name that isn't in the registry."""

    def __init__(self, name: str, registry: Sequence[StageSpec]) -> None:
        super().__init__(
            f"unknown stage {name!r}; stages in order: {', '.join(s.name for s in registry)}"
        )


class StageRecord(BaseModel):
    """One entry of ``stages.json``: what ``RunContext.record`` stored for a stage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    runtimeS: float
    counts: dict[str, int]


_STAGE_RECORDS = TypeAdapter(dict[str, StageRecord])


# --- run.json / stages.json ---------------------------------------------------------------------


def _write_text(path: Path, text: str) -> None:
    """Write through a temp file in the same directory and ``os.replace`` it into place, so a
    crash never leaves a half-written JSON. One process writes a run at a time; there is no
    cross-process locking."""
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def read_run_json(run_dir: Path) -> ProcessingRun:
    path = Path(run_dir) / RUN_JSON
    if not path.is_file():
        raise RunError(f"{path} not found; is {run_dir} a run directory?")
    try:
        return ProcessingRun.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise RunError(f"{path} is not a valid ProcessingRun:\n{exc}") from exc


def write_run_json(run_dir: Path, run: ProcessingRun) -> None:
    _write_text(Path(run_dir) / RUN_JSON, run.model_dump_json(indent=2) + "\n")


def read_stage_records(run_dir: Path) -> dict[str, StageRecord]:
    path = Path(run_dir) / STAGES_JSON
    if not path.is_file():
        return {}
    try:
        return _STAGE_RECORDS.validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise RunError(f"{path} is malformed:\n{exc}") from exc


def write_stage_records(run_dir: Path, records: dict[str, StageRecord]) -> None:
    text = _STAGE_RECORDS.dump_json(records, indent=2).decode()
    _write_text(Path(run_dir) / STAGES_JSON, text + "\n")


# --- RunContext -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunContext:
    """What every stage gets: where the run lives, the shared cache, every lane's config."""

    run_id: str
    run_dir: Path  # data/<mode>/runs/<run_id>
    cache_dir: Path  # data/cache
    config: RunConfig

    def path(self, name: str) -> Path:
        """A file in the run directory (``docs/02`` §2 names them)."""
        return self.run_dir / name

    def read_run(self) -> ProcessingRun:
        """The current ``run.json``."""
        return read_run_json(self.run_dir)

    def record(
        self,
        stage: str,
        *,
        runtime_s: float,
        counts: dict[str, int],
        params: dict[str, Any] | None = None,
        field: str | None = None,
    ) -> None:
        """Merge a stage's runtime, counts and parameters into the run.

        ``runtimeS[stage]`` goes into ``run.json``. ``params`` merge into the ``ProcessingRun``
        dict field for the stage (``STAGE_PARAM_FIELDS``), or into ``field`` when given; a stage
        with neither is an error. ``ProcessingRun`` has no counts field, so ``counts`` (and the
        runtime) also go into the ``stages.json`` sidecar, and both are logged at INFO.
        """
        stage_spec(stage)  # an unknown stage name is an error, not a new key in run.json
        if runtime_s < 0:
            raise ValueError(f"runtime_s must be >= 0, got {runtime_s}")
        try:  # validate the sidecar entry first so a bad call touches nothing on disk
            entry = StageRecord(runtimeS=float(runtime_s), counts=dict(counts))
        except ValidationError as exc:
            raise RunError(f"stage {stage!r}: counts must be {{name: int}}:\n{exc}") from exc
        run = self.read_run()
        updates: dict[str, Any] = {"runtimeS": {**run.runtimeS, stage: float(runtime_s)}}
        if params is not None:
            target = field if field is not None else STAGE_PARAM_FIELDS.get(stage)
            if target is None:
                raise RunError(
                    f"stage {stage!r} has no ProcessingRun field for params; "
                    f"pass field= one of {list(PARAM_FIELDS)}"
                )
            if target not in PARAM_FIELDS:
                raise RunError(
                    f"field {target!r} is not a ProcessingRun params dict; one of {list(PARAM_FIELDS)}"
                )
            updates[target] = {**getattr(run, target), **params}
        self._update(run, updates)

        records = read_stage_records(self.run_dir)
        records[stage] = entry
        write_stage_records(self.run_dir, records)

        summary = ", ".join(f"{value} {key}" for key, value in counts.items()) or "done"
        log.info("stage %s: %s in %.1f s", stage, summary, runtime_s)

    def update_run(self, **fields: Any) -> None:
        """Set top-level ``ProcessingRun`` fields (``stationIds``, ``pickerModel``, ...).

        Runtimes and stage parameters go through ``record``; ``id`` never changes.
        """
        unknown = [name for name in fields if name not in UPDATABLE_FIELDS]
        if unknown:
            raise RunError(
                f"cannot set {unknown} on ProcessingRun; update_run accepts {list(UPDATABLE_FIELDS)}"
            )
        self._update(self.read_run(), dict(fields))

    def _update(self, run: ProcessingRun, updates: dict[str, Any]) -> None:
        """Re-validate the merged document through the model, then write it."""
        try:
            merged = ProcessingRun.model_validate({**run.model_dump(), **updates})
        except ValidationError as exc:
            raise RunError(f"run {self.run_id}: invalid update {list(updates)}:\n{exc}") from exc
        write_run_json(self.run_dir, merged)


# --- creating and loading runs --------------------------------------------------------------------


def runs_dir(data_dir: Path, mode: str) -> Path:
    return Path(data_dir) / mode / RUNS_DIRNAME


def git_sha(cwd: Path) -> str:
    """Full HEAD sha of the repo containing ``cwd``, or ``NO_GIT_SHA`` when there is none."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        log.warning("git is not installed; run id uses %s", NO_GIT_SHA)
        return NO_GIT_SHA
    if proc.returncode != 0:
        log.warning("%s is not in a git repo; run id uses %s", cwd, NO_GIT_SHA)
        return NO_GIT_SHA
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=False
    )
    if status.returncode == 0 and status.stdout.strip():
        log.warning(
            "working tree has uncommitted changes; gitSha alone does not pin this run's code"
        )
    return proc.stdout.strip()


def software_versions() -> dict[str, str]:
    """Interpreter and package versions for ``ProcessingRun.softwareVersions``."""
    versions = {"python": platform.python_version()}
    for name in VERSIONED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            log.debug("package %s not installed; omitted from softwareVersions", name)
    return versions


def window_label(start: datetime, end: datetime) -> str:
    """``"2026-09-10 00:00-24:00 UTC"``: a same-day window; a whole day ends at 24:00."""
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    next_midnight = datetime(start.year, start.month, start.day, tzinfo=UTC) + timedelta(days=1)
    if end == next_midnight:
        end_label = "24:00"
    elif end.date() == start.date():
        end_label = f"{end:%H:%M}"
    else:
        end_label = f"{end:%Y-%m-%d %H:%M}"
    return f"{start:%Y-%m-%d %H:%M}-{end_label} UTC"


def create_run(
    config_dir: Path,
    data_dir: Path,
    mode: DataMode = "showcase",
    *,
    now: datetime | None = None,
) -> RunContext:
    """Make ``<data>/<mode>/runs/<runId>/`` with its initial ``run.json``.

    ``run_id`` is ``YYYYMMDD-HHMM-<gitsha7>`` in UTC. ``now`` exists for tests; it must be aware.
    """
    config_dir, data_dir = Path(config_dir).resolve(), Path(data_dir).resolve()
    if mode not in DATA_MODES:
        raise RunError(f"mode must be one of {DATA_MODES}, got {mode!r}")
    if now is not None and now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    config = load_config(config_dir)
    created = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    sha = git_sha(config_dir)
    run_id = f"{created:%Y%m%d-%H%M}-{sha[:GIT_SHA_LEN]}"
    run_dir = runs_dir(data_dir, mode) / run_id
    section = config.run
    run = ProcessingRun(
        id=run_id,
        mode=mode,
        createdAt=created.isoformat().replace("+00:00", "Z"),
        gitSha=sha,
        windowStart=section.window_start_s,
        windowEnd=section.window_end_s,
        windowLabel=window_label(section.windowStart, section.windowEnd),
        bbox=section.bbox,
        stationIds=[],
        pickerModel="",
        pickerWeights="",
        softwareVersions=software_versions(),
        runtimeS={},
        picker={},
        associator={},
        velocityModel={},
        locator={},
        tiering={},
        matching={},
        isSynthetic=False,
    )
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RunError(
            f"run directory already exists: {run_dir} "
            "(run ids have minute resolution; wait a minute or remove it)"
        ) from exc
    cache_dir = data_dir / CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_run_json(run_dir, run)
    log.info("created run %s (mode %s, git %s) at %s", run_id, mode, sha[:GIT_SHA_LEN], run_dir)
    return RunContext(run_id=run_id, run_dir=run_dir, cache_dir=cache_dir, config=config)


def find_run_dir(run_id: str, data_dir: Path, mode: str | None = None) -> Path:
    """Locate ``<data>/<mode>/runs/<run_id>``, searching every mode when ``mode`` is None."""
    data_dir = Path(data_dir)
    if mode is not None:
        candidates = [runs_dir(data_dir, mode) / run_id]
    else:
        candidates = sorted(data_dir.glob(f"*/{RUNS_DIRNAME}/{run_id}"))
    found = [path for path in candidates if path.is_dir()]
    if not found:
        raise RunError(f"run {run_id!r} not found under {data_dir}/<mode>/{RUNS_DIRNAME}/")
    if len(found) > 1:
        raise RunError(f"run {run_id!r} exists in several modes: {found}; pass mode=")
    return found[0]


def load_run(run_id: str, data_dir: Path, config_dir: Path, mode: str | None = None) -> RunContext:
    """A ``RunContext`` for an existing run, for ``hq stage``."""
    data_dir = Path(data_dir).resolve()
    run_dir = find_run_dir(run_id, data_dir, mode)
    run = read_run_json(run_dir)
    if run.id != run_id:
        raise RunError(f"{run_dir / RUN_JSON} has id {run.id!r}, expected {run_id!r}")
    config = load_config(Path(config_dir).resolve())
    section = config.run
    expected = (section.window_start_s, section.window_end_s, tuple(section.bbox))
    actual = (run.windowStart, run.windowEnd, tuple(run.bbox))
    if expected != actual:
        raise RunError(
            f"{config_dir} no longer matches run {run_id}: config (windowStart, windowEnd, bbox) "
            f"= {expected} but run.json has {actual}; start a new run instead"
        )
    return RunContext(
        run_id=run_id, run_dir=run_dir, cache_dir=data_dir / CACHE_DIRNAME, config=config
    )


# --- stage runner ---------------------------------------------------------------------------------

StageFn = Callable[[RunContext], None]


def stage_spec(name: str, registry: Sequence[StageSpec] = STAGES) -> StageSpec:
    for spec in registry:
        if spec.name == name:
            return spec
    raise UnknownStageError(name, registry)


def select_stages(
    names: Sequence[str] | None, registry: Sequence[StageSpec] = STAGES
) -> tuple[StageSpec, ...]:
    """Registry entries for ``names`` in registry order (every name), or all of them."""
    if names is None:
        return tuple(registry)
    wanted = set()
    for name in names:
        stage_spec(name, registry)  # raises UnknownStageError
        wanted.add(name)
    return tuple(spec for spec in registry if spec.name in wanted)


def resolve_stage(spec: StageSpec) -> StageFn:
    """Import the stage module lazily and return its ``run``; missing pieces name the owner."""
    try:
        module = importlib.import_module(spec.module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if spec.module == missing or spec.module.startswith(missing + "."):
            raise StageMissingError(spec, f"module {spec.module} not found") from exc
        raise  # the stage module exists but one of its imports is broken: a real error
    fn = getattr(module, "run", None)
    if not callable(fn):
        raise StageMissingError(spec, f"{spec.module} has no run(ctx)")
    return fn


def stage_status(spec: StageSpec) -> str:
    """``"implemented"`` or ``"missing: <reason>"`` for ``hq stages``."""
    try:
        resolve_stage(spec)
    except StageMissingError as exc:
        return f"missing: {exc.reason}"
    return "implemented"


def run_stage(ctx: RunContext, name: str, registry: Sequence[StageSpec] = STAGES) -> float:
    """Run one stage, time it, and return its wall time in seconds.

    The stage records its own runtime and counts through ``ctx.record``; if it doesn't, the wall
    time measured here is recorded with no counts and a warning.
    """
    spec = stage_spec(name, registry)
    fn = resolve_stage(spec)
    log.info("stage %s: start (%s, owner %s)", name, spec.module, spec.owner)
    t0 = time.perf_counter()
    fn(ctx)
    runtime_s = time.perf_counter() - t0
    log.info("stage %s: finished in %.1f s", name, runtime_s)
    if name not in ctx.read_run().runtimeS:
        log.warning(
            "stage %s did not call ctx.record(); recording %.1f s wall time with no counts",
            name,
            runtime_s,
        )
        ctx.record(name, runtime_s=runtime_s, counts={})
    return runtime_s


def run_stages(
    ctx: RunContext, names: Sequence[str] | None = None, registry: Sequence[StageSpec] = STAGES
) -> list[str]:
    """Run stages in registry order, stopping at the first failure. Returns the names that ran."""
    specs = select_stages(names, registry)
    done: list[str] = []
    for spec in specs:
        try:
            run_stage(ctx, spec.name, registry)
        except Exception as exc:  # the CLI logs it once; the note says where and how to rerun
            exc.add_note(
                f"run {ctx.run_id} failed at stage {spec.name!r} after {len(done)}/{len(specs)} "
                f"stages; rerun it with: hq stage {spec.name} --run {ctx.run_id}"
            )
            raise
        done.append(spec.name)
    log.info("run %s: %d stage(s) finished: %s", ctx.run_id, len(done), ", ".join(done))
    return done

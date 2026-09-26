"""Live worker knobs (``services/api/config.yaml``). Unknown keys are an error.

Every value the worker or the API uses is a field here: the rolling window, the scheduler, the
pipeline stages a window runs, where state and bundles live, what the API serves and on which
origins, and the latency the Live kill switch in docs/03 watches. The whole config is recorded
in every window record the API serves (``/health``), so a served window traces to its knobs.

Paths are relative to the checkout root (the first ancestor of this package holding ``.git``,
like ``hq.export.checkout_root``) unless absolute. ``dataDir: null`` follows ``hq.cli``: the
main checkout's ``data/``, shared by every worktree on a laptop.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from hq import runs

DEFAULT_CONFIG_FILE = "config.yaml"
PIPELINE_STAGE_NAMES: tuple[str, ...] = tuple(spec.name for spec in runs.STAGES)
# The exporter runs inside the worker (``hq.export.export_bundle``) and the validation stage
# belongs to the frozen showcase run, so neither may be listed as a window stage.
WORKER_OWNED_STAGES: frozenset[str] = frozenset({"validate", "export"})


class LiveConfigError(ValueError):
    """``config.yaml`` is missing, malformed or holds unknown keys."""


class WindowConfig(BaseModel):
    """The rolling window one pipeline run covers, and how often it runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    windowS: float = Field(gt=0.0)  # seconds of data per window (LiveStatus.windowS)
    everyS: float = Field(gt=0.0)  # seconds between window starts
    # Seconds the window end trails the wall clock, so the last samples had time to reach the
    # data center. Counted into latencyS (data end -> results ready).
    dataLagS: float = Field(default=0.0, ge=0.0)
    # Pipeline stages a window runs, in registry order (``hq stages``); validate and export are
    # the worker's own business and are refused here.
    stages: list[str] = Field(min_length=1)
    # A window whose latencyS exceeds this is logged as slow and flagged in /health: the Live
    # kill switch in docs/03 ("latency over ~10 min -> cut the LIVE pill").
    maxLatencyS: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> "WindowConfig":
        if self.everyS > self.windowS:
            raise ValueError(
                f"everyS {self.everyS} must not exceed windowS {self.windowS}: consecutive "
                "windows would leave gaps in the data covered"
            )
        unknown = [s for s in self.stages if s not in PIPELINE_STAGE_NAMES]
        if unknown:
            raise ValueError(f"unknown stages {unknown}; hq stages lists {PIPELINE_STAGE_NAMES}")
        owned = [s for s in self.stages if s in WORKER_OWNED_STAGES]
        if owned:
            raise ValueError(f"stages {owned} are run by the worker itself; remove them")
        if len(set(self.stages)) != len(self.stages):
            raise ValueError(f"stages must not repeat, got {self.stages}")
        order = [s for s in PIPELINE_STAGE_NAMES if s in self.stages]
        if order != self.stages:
            raise ValueError(f"stages must be in pipeline order: {order}")
        return self


class PathsConfig(BaseModel):
    """Where the worker reads its pipeline config and writes runs, bundles and state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # The showcase config directory; each window copies it and rewrites run.yaml's window.
    configDir: str = Field(min_length=1)
    # Runs root (``<dataDir>/live/runs/<runId>``) and waveform cache (``<dataDir>/cache``);
    # null means the main checkout's data/ (the hq CLI rule), shared across worktrees.
    dataDir: str | None = Field(default=None, min_length=1)
    # Per-window config directories the worker generates (run.yaml with the live window).
    windowConfigDir: str = Field(min_length=1)
    # The last good window and the recent history, so a restart serves the last window.
    stateFile: str = Field(min_length=1)
    # One exported ``live`` bundle per window: ``<bundlesDir>/<runId>/``. The API serves these.
    bundlesDir: str = Field(min_length=1)
    # The ``snapshot`` bundle the web app fails over to (API-05).
    snapshotDir: str = Field(min_length=1)
    # Finished windows older than the newest ``keepBundles`` are deleted from bundlesDir.
    keepBundles: int = Field(ge=1)


class SnapshotConfig(BaseModel):
    """When the snapshot bundle is rewritten after a window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    # Also freeze a window with no candidate events. False keeps the last snapshot that showed
    # something; the API still serves the empty window live.
    writeEmptyWindows: bool


class ServeConfig(BaseModel):
    """What the API serves and to whom."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Cap on GET /api/live/events, taken in reveal order (Tier A first).
    maxEvents: int = Field(ge=1)
    # A served window whose updatedAt is older than this is reported stale in /health.
    staleAfterS: float = Field(gt=0.0)
    # Window records kept in the state file and shown by /health, newest first.
    historyN: int = Field(ge=1)
    # Origins allowed by CORS (the static site); "*" allows every origin.
    corsOrigins: list[str] = Field(min_length=1)


class ServerConfig(BaseModel):
    """uvicorn settings for ``python -m hq_api`` / ``make api``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    # Seconds an idle keep-alive connection stays open (uvicorn timeout_keep_alive).
    keepAliveS: float = Field(gt=0.0)
    # Seconds the server waits for in-flight requests on shutdown (uvicorn timeout_graceful_shutdown).
    gracefulShutdownS: float = Field(gt=0.0)


class LiveConfig(BaseModel):
    """Contents of ``config.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    window: WindowConfig
    paths: PathsConfig
    snapshot: SnapshotConfig
    serve: ServeConfig
    server: ServerConfig

    def dump(self) -> dict[str, Any]:
        """Plain JSON-ready dict, recorded in every window record."""
        return self.model_dump(mode="json")


def checkout_root(start: Path | None = None) -> Path:
    """The checkout holding this package: the first ancestor with a ``.git`` entry (a linked
    worktree's ``.git`` file counts, like ``hq.export.checkout_root``)."""
    start = (start or Path(__file__)).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise LiveConfigError(f"no .git above {start}; use absolute paths in config.yaml")


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_live_config(path: Path) -> LiveConfig:
    """Read and validate ``config.yaml``; unknown keys and bad values are errors."""
    path = Path(path)
    if not path.is_file():
        raise LiveConfigError(f"live config not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LiveConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise LiveConfigError(f"{path}: top level must be a mapping")
    try:
        return LiveConfig.model_validate(data)
    except ValidationError as exc:
        raise LiveConfigError(f"{path}:\n{exc}") from exc

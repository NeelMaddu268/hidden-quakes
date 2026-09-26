"""What the API serves and what survives a restart.

A ``WindowRecord`` is one attempt at a window: ok (a valid ``live`` bundle exists for it),
empty (ok with no candidate events) or failed (the error, and nothing served from it). The
``LiveState`` file (``paths.stateFile``, ``data/live/latest.json``) holds the last good record
and the recent history, written atomically after every attempt. ``ServedWindow`` is the last
good window loaded into memory from its bundle directory: ``BundleMeta`` (``mode: "live"``),
the events in reveal order and the evidence ids, straight from the files the exporter wrote
and checked. Evidence is read from the bundle on request.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hq_contracts.models import BundleMeta, EventEvidence, LiveStatus, SeismicEvent
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from hq.export.files import EVENTS_JSON, EVIDENCE_DIR, META_JSON

log = logging.getLogger(__name__)

WindowOutcome = Literal["ok", "empty", "failed"]
GOOD_OUTCOMES: frozenset[str] = frozenset({"ok", "empty"})
STATE_VERSION = 1

# ``LiveStatus`` without ``events`` (docs/02 §7, GET /api/live/status), derived from the contract
# model so the two never drift: the same fields, types and order, minus the event list.
LiveStatusSummary: type[BaseModel] = create_model(
    "LiveStatusSummary",
    __config__=ConfigDict(extra="forbid"),
    **{
        name: (field.annotation, ...)
        for name, field in LiveStatus.model_fields.items()
        if name != "events"
    },
)


class WindowRecord(BaseModel):
    """One attempt at a window, as stored in the state file and shown by ``/health``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: WindowOutcome
    windowStart: float  # epoch s UTC, data start
    windowEnd: float  # epoch s UTC, data end
    startedAt: float  # epoch s UTC, when the pipeline started (wall clock)
    updatedAt: float  # epoch s UTC, results ready (or the failure time)
    runtimeS: float  # pipeline + export, monotonic clock
    latencyS: float | None  # data end -> results ready; null when the window failed
    slow: bool  # latencyS > window.maxLatencyS (the Live kill switch in docs/03)
    runId: str | None
    stagesRan: list[str]
    eventCount: int
    stationsOnline: int
    evidenceIds: list[str]
    bundleDir: str | None  # the live bundle this record's events are served from
    snapshotWritten: bool
    error: str | None
    config: dict[str, Any]  # the LiveConfig this window ran with


class LiveState(BaseModel):
    """The state file: last good window plus the newest records first."""

    model_config = ConfigDict(extra="forbid")

    version: int = STATE_VERSION
    latest: WindowRecord | None = None
    history: list[WindowRecord] = Field(default_factory=list)


class StateError(RuntimeError):
    """The state file or a bundle it points at is unusable."""


def write_json_atomic(path: Path, text: str) -> None:
    """Temp file in the same directory, then ``os.replace``: never a half-written state."""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def read_state(path: Path) -> LiveState:
    """The persisted state, or an empty one when the file does not exist yet."""
    path = Path(path)
    if not path.is_file():
        return LiveState()
    try:
        state = LiveState.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, json.JSONDecodeError) as exc:
        raise StateError(f"{path} is not a valid live state file:\n{exc}") from exc
    if state.version != STATE_VERSION:
        raise StateError(f"{path} has state version {state.version}, expected {STATE_VERSION}")
    return state


def write_state(path: Path, state: LiveState) -> None:
    write_json_atomic(Path(path), state.model_dump_json(indent=2) + "\n")


@dataclass(frozen=True)
class ServedWindow:
    """The last good window, loaded from its live bundle directory."""

    record: WindowRecord
    bundle_dir: Path
    meta: BundleMeta
    events: list[SeismicEvent]  # reveal order
    evidence_ids: frozenset[str]

    def status(self, window_s: float) -> BaseModel:
        """``GET /api/live/status``: ``LiveStatus`` without ``events``."""
        latency = self.record.latencyS
        if latency is None:  # a good record always has one; the type says otherwise
            raise StateError(f"window {self.record.runId} has no latencyS")
        return LiveStatusSummary(
            updatedAt=self.record.updatedAt,
            windowS=window_s,
            latencyS=latency,
            stationsOnline=self.record.stationsOnline,
        )

    def live_status(self, window_s: float, max_events: int) -> LiveStatus:
        """The full ``LiveStatus`` (the events capped like ``GET /api/live/events``)."""
        summary = self.status(window_s)
        return LiveStatus(**summary.model_dump(), events=self.events[:max_events])

    def evidence(self, event_id: str) -> EventEvidence | None:
        """The event's evidence from the bundle, or None when the window has none for it."""
        if event_id not in self.evidence_ids:
            return None
        path = self.bundle_dir / EVIDENCE_DIR / f"{event_id}.json"
        try:
            return EventEvidence.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            raise StateError(f"{path} is not a valid EventEvidence: {exc}") from exc


def load_served_window(record: WindowRecord, bundle_dir: Path) -> ServedWindow:
    """Read ``meta.json`` and ``events.json`` of a checked live bundle into memory."""
    bundle_dir = Path(bundle_dir)
    if record.outcome not in GOOD_OUTCOMES:
        raise StateError(f"window {record.runId} is {record.outcome}; nothing to serve")
    if not bundle_dir.is_dir():
        raise StateError(f"live bundle {bundle_dir} for window {record.runId} is gone")
    try:
        meta = BundleMeta.model_validate_json((bundle_dir / META_JSON).read_text("utf-8"))
        raw = json.loads((bundle_dir / EVENTS_JSON).read_text("utf-8"))
        events = [SeismicEvent.model_validate(item) for item in raw]
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        raise StateError(f"live bundle {bundle_dir} is not readable: {exc}") from exc
    if meta.run.id != record.runId:
        raise StateError(
            f"live bundle {bundle_dir} holds run {meta.run.id!r}, record says {record.runId!r}"
        )
    events.sort(key=lambda e: e.revealOrder)
    evidence_dir = bundle_dir / EVIDENCE_DIR
    evidence_ids = frozenset(p.stem for p in evidence_dir.glob("*.json") if evidence_dir.is_dir())
    return ServedWindow(
        record=record,
        bundle_dir=bundle_dir,
        meta=meta,
        events=events,
        evidence_ids=evidence_ids,
    )

"""The Live API (docs/02 §7), served from the worker's last good window.

    GET /api/live/meta            BundleMeta for the latest processed window (mode: "live")
    GET /api/live/events          SeismicEvent[] in reveal order, capped by serve.maxEvents
    GET /api/live/evidence/{id}   EventEvidence
    GET /api/live/status          LiveStatus without events
    GET /health                   worker state, the served and latest window records, history

Response models are the contract models themselves. Before the first good window every
``/api/live/*`` route answers 503, which the web ``LiveProvider`` treats as a failed fetch and
API-05 turns into snapshot failover. An empty window is a normal answer: an empty event list
and a status the shell derives its copy from; the API never carries that copy.
"""

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from hq_contracts.models import BundleMeta, EventEvidence, SeismicEvent
from pydantic import BaseModel, ConfigDict

from hq_api.config import LiveConfig
from hq_api.runner import PipelineRunner
from hq_api.state import LiveStatusSummary, ServedWindow, WindowRecord
from hq_api.worker import LiveWorker

log = logging.getLogger(__name__)

API_PREFIX = "/api/live"
HEALTH_PATH = "/health"
NO_WINDOW_STATUS = 503
NOT_FOUND_STATUS = 404
NO_WINDOW_DETAIL = "no live window has been processed yet; the worker runs the first one at startup"

ServiceStatus = Literal["waiting", "ok", "stale"]


class WorkerStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    running: bool  # a window is being processed right now
    attempts: int  # windows started since this process began
    failures: int
    skipped: int  # ticks that found a window still running
    nextRunAt: float | None  # epoch s UTC of the next scheduled tick


class HealthReport(BaseModel):
    """``GET /health``: everything the kill switch and a restart need to know."""

    model_config = ConfigDict(extra="forbid")

    status: ServiceStatus  # of the served window: waiting (none yet), ok, or stale
    served: WindowRecord | None  # the window /api/live/* answers from
    lastAttempt: WindowRecord | None  # the most recent window, good or failed
    worker: WorkerStats
    history: list[WindowRecord]  # newest first, at most serve.historyN
    config: dict[str, Any]


def _served(worker: LiveWorker) -> ServedWindow:
    served = worker.current()
    if served is None:
        raise HTTPException(NO_WINDOW_STATUS, NO_WINDOW_DETAIL)
    return served


def create_app(
    config: LiveConfig,
    runner: PipelineRunner,
    *,
    root: Path | None = None,
    scheduler: bool = True,
    clock: Callable[[], float] = time.time,
    worker: LiveWorker | None = None,
) -> FastAPI:
    """The ASGI app with its worker attached (``app.state.worker``). ``scheduler=False`` leaves
    the ticker off (tests drive the worker directly)."""
    live = worker if worker is not None else LiveWorker(config, runner, root=root, clock=clock)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if scheduler:
            live.start()
            log.info(
                "live worker started: windowS %.0f every %.0f s, stages %s",
                config.window.windowS,
                config.window.everyS,
                ", ".join(config.window.stages),
            )
        yield
        await live.stop()

    app = FastAPI(
        title="Hidden Quakes live worker",
        description="Runs the hq pipeline on a rolling window and serves docs/02 §7.",
        lifespan=lifespan,
    )
    app.state.worker = live
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.serve.corsOrigins),
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get(f"{API_PREFIX}/meta", response_model=BundleMeta)
    def live_meta(request: Request) -> BundleMeta:
        return _served(request.app.state.worker).meta

    @app.get(f"{API_PREFIX}/events", response_model=list[SeismicEvent])
    def live_events(request: Request) -> list[SeismicEvent]:
        return _served(request.app.state.worker).events[: config.serve.maxEvents]

    @app.get(f"{API_PREFIX}/evidence/{{event_id}}", response_model=EventEvidence)
    def live_evidence(event_id: str, request: Request) -> EventEvidence:
        evidence = _served(request.app.state.worker).evidence(event_id)
        if evidence is None:
            raise HTTPException(
                NOT_FOUND_STATUS,
                f"no evidence for candidate event {event_id} in the current live window",
            )
        return evidence

    @app.get(f"{API_PREFIX}/status", response_model=LiveStatusSummary)
    def live_status(request: Request) -> BaseModel:
        return _served(request.app.state.worker).status(config.window.windowS)

    @app.get(HEALTH_PATH, response_model=HealthReport)
    def health(request: Request) -> HealthReport:
        worker_: LiveWorker = request.app.state.worker
        served = worker_.current()
        state = worker_.snapshot_state()
        if served is None:
            status: ServiceStatus = "waiting"
        elif worker_.is_stale(served):
            status = "stale"
        else:
            status = "ok"
        return HealthReport(
            status=status,
            served=served.record if served is not None else None,
            lastAttempt=state.history[0] if state.history else None,
            worker=WorkerStats(
                running=worker_.running,
                attempts=worker_.attempts,
                failures=worker_.failures,
                skipped=worker_.skipped,
                nextRunAt=worker_.next_run_at,
            ),
            history=state.history,
            config=config.dump(),
        )

    return app

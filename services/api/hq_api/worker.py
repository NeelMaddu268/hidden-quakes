"""The live worker: one window at a time, on a fixed cadence, in a thread, never overlapping.

``LiveWorker.run_window_now`` is the whole job for one window, synchronous and safe to call
from a thread: run the pipeline through the ``PipelineRunner``, export the run to a ``live``
bundle with ``hq.export.export_bundle`` (which checks the bundle before swapping it in),
measure ``latencyS`` (data end -> results ready, monotonic clock around the run plus the data
lag), rewrite the ``snapshot`` bundle for API-05, swap the served window, and persist the state
file. A failure anywhere records a failed window and keeps serving the last good one.

The scheduler is an ``asyncio`` task: it fires a tick at startup and then every ``everyS``
seconds of wall time, each tick handing ``run_window_now`` to a single-thread executor so the
API stays responsive. A tick that arrives while a window is still running is skipped and
logged, never queued: the next tick runs a fresh window.
"""

import asyncio
import logging
import math
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hq.config.export import ExportConfig, ExportMode
from hq.export import RunTables, export_bundle, load_run_tables
from hq.export.bundle import FeaturesLoader, load_features_lazily
from hq.export.files import EVIDENCE_DIR
from hq_api.config import LiveConfig, checkout_root, resolve_path
from hq_api.runner import LiveWindow, PipelineRun, PipelineRunner
from hq_api.state import (
    GOOD_OUTCOMES,
    LiveState,
    ServedWindow,
    StateError,
    WindowRecord,
    load_served_window,
    read_state,
    write_state,
)

log = logging.getLogger(__name__)

LIVE_MODE: ExportMode = "live"
SNAPSHOT_MODE: ExportMode = "snapshot"
Clock = Callable[[], float]


def error_text(exc: BaseException) -> str:
    """The message plus any notes (``hq.runs`` adds the rerun hint as a note)."""
    return "; ".join([f"{type(exc).__name__}: {exc}", *getattr(exc, "__notes__", [])])


class LiveWorker:
    """Owns the served window, the state file, the overlap guard and the scheduler."""

    def __init__(
        self,
        config: LiveConfig,
        runner: PipelineRunner,
        *,
        root: Path | None = None,
        clock: Clock = time.time,
        monotonic: Clock = time.perf_counter,
        features_loader: FeaturesLoader = load_features_lazily,
    ) -> None:
        self.config = config
        self.runner = runner
        self.clock = clock
        self.monotonic = monotonic
        self.features_loader = features_loader
        root = root if root is not None else checkout_root()
        paths = config.paths
        self.state_file = resolve_path(paths.stateFile, root)
        self.bundles_dir = resolve_path(paths.bundlesDir, root)
        self.snapshot_dir = resolve_path(paths.snapshotDir, root)

        self._run_lock = threading.Lock()  # held while a window runs; never waited on
        self._swap_lock = threading.Lock()  # guards served/state swaps and reads
        self._executor: ThreadPoolExecutor | None = None
        self._ticker: asyncio.Task[None] | None = None
        self._pending: set[asyncio.Future[WindowRecord | None]] = set()
        self._stop = asyncio.Event()
        self.attempts = 0
        self.skipped = 0
        self.failures = 0
        self.next_run_at: float | None = None

        self.state: LiveState = read_state(self.state_file)
        self.served: ServedWindow | None = self._restore(self.state.latest)

    # --- restart ---------------------------------------------------------------------------------

    def _restore(self, record: WindowRecord | None) -> ServedWindow | None:
        """Serve the persisted last good window again, if its bundle is still there."""
        if record is None:
            log.info("live: no previous window in %s; waiting for the first one", self.state_file)
            return None
        if record.bundleDir is None:
            log.warning("live: last record %s has no bundle; nothing restored", record.runId)
            return None
        try:
            served = load_served_window(record, Path(record.bundleDir))
        except StateError as exc:
            log.warning("live: last window %s not restored: %s", record.runId, exc)
            return None
        log.info(
            "live: restored window %s from %s: %d candidate events, %d evidence files",
            record.runId,
            record.bundleDir,
            len(served.events),
            len(served.evidence_ids),
        )
        return served

    # --- one window -----------------------------------------------------------------------------

    def window_now(self) -> LiveWindow:
        """``[now - dataLagS - windowS, now - dataLagS]`` on whole seconds."""
        cfg = self.config.window
        end = math.floor(self.clock() - cfg.dataLagS)
        return LiveWindow(start=float(end - cfg.windowS), end=float(end))

    @property
    def running(self) -> bool:
        return self._run_lock.locked()

    def run_window_now(self) -> WindowRecord | None:
        """Process the current window; None (logged) when one is already running."""
        if not self._run_lock.acquire(blocking=False):
            self.skipped += 1
            log.warning(
                "live: a window is still running; this tick is skipped (%d skipped so far)",
                self.skipped,
            )
            return None
        try:
            return self.process(self.window_now())
        finally:
            self._run_lock.release()

    def process(self, window: LiveWindow) -> WindowRecord:
        """Pipeline -> live bundle -> snapshot -> served window, or a failed record."""
        self.attempts += 1
        started_at = self.clock()
        t0 = self.monotonic()
        log.info("live window %s: start (attempt %d)", window.label, self.attempts)
        run: PipelineRun | None = None
        try:
            run = self.runner.run_window(window)
            record, served = self._finish(window, run, started_at, t0)
        except Exception as exc:
            self.failures += 1
            runtime_s = self.monotonic() - t0
            log.exception("live window %s: failed after %.1f s", window.label, runtime_s)
            record = WindowRecord(
                outcome="failed",
                windowStart=window.start,
                windowEnd=window.end,
                startedAt=started_at,
                updatedAt=self.clock(),
                runtimeS=runtime_s,
                latencyS=None,
                slow=False,
                runId=run.run_id if run is not None else None,
                stagesRan=list(run.stages_ran) if run is not None else [],
                eventCount=0,
                stationsOnline=0,
                evidenceIds=[],
                bundleDir=None,
                snapshotWritten=False,
                error=error_text(exc),
                config=self.config.dump(),
            )
            served = None
        self._commit(record, served)
        return record

    def _finish(
        self, window: LiveWindow, run: PipelineRun, started_at: float, t0: float
    ) -> tuple[WindowRecord, ServedWindow]:
        """Export the finished run, measure latency, write the snapshot, load what to serve."""
        cfg = self.config
        tables = load_run_tables(run.run_dir, gr_cfg=run.config.validate.gr)
        export_cfg = run.config.export.model_copy(update={"modes": [LIVE_MODE, SNAPSHOT_MODE]})
        bundle_dir = self.bundles_dir / run.run_id
        result = export_bundle(
            tables,
            export_cfg,
            run.config.run,
            LIVE_MODE,
            run.waveforms,
            cache_dir=run.cache_dir,
            out_dir=bundle_dir,
            features_loader=self.features_loader,
            baseline_cfg=run.config.validate.baseline,
        )
        runtime_s = self.monotonic() - t0
        updated_at = self.clock()
        # Results-ready minus data end: the run measured on the monotonic clock, plus how far the
        # data end trailed the wall clock when the run started (dataLagS, by construction).
        latency_s = runtime_s + (started_at - window.end)
        slow = latency_s > cfg.window.maxLatencyS
        n_events = result.counts["events"]
        outcome = "ok" if n_events else "empty"
        log.info(
            "live window %s: %s, %d candidate events, %d evidence files, latencyS %.1f "
            "(maxLatencyS %.0f%s), run %s",
            window.label,
            outcome,
            n_events,
            result.counts["evidenceFiles"],
            latency_s,
            cfg.window.maxLatencyS,
            ", SLOW" if slow else "",
            run.run_id,
        )
        max_events = cfg.serve.maxEvents
        if n_events > max_events:
            log.warning(
                "live window %s: %d candidate events; /api/live/events serves the first %d in "
                "reveal order (serve.maxEvents)",
                window.label,
                n_events,
                max_events,
            )
        if slow:
            log.warning(
                "live window %s: latencyS %.1f exceeds maxLatencyS %.0f; docs/03 says cut the "
                "LIVE pill if this persists",
                window.label,
                latency_s,
                cfg.window.maxLatencyS,
            )
        snapshot_written, snapshot_error = self._write_snapshot(run, tables, export_cfg, n_events)
        record = WindowRecord(
            outcome=outcome,
            windowStart=window.start,
            windowEnd=window.end,
            startedAt=started_at,
            updatedAt=updated_at,
            runtimeS=runtime_s,
            latencyS=latency_s,
            slow=slow,
            runId=run.run_id,
            stagesRan=list(run.stages_ran),
            eventCount=n_events,
            stationsOnline=sum(1 for s in tables.stations if s.usedInRun),
            evidenceIds=sorted(
                name.removeprefix(f"{EVIDENCE_DIR}/").removesuffix(".json")
                for name in result.sizes
                if name.startswith(f"{EVIDENCE_DIR}/")
            ),
            bundleDir=str(bundle_dir),
            snapshotWritten=snapshot_written,
            error=snapshot_error,
            config=cfg.dump(),
        )
        return record, load_served_window(record, bundle_dir)

    def _write_snapshot(
        self, run: PipelineRun, tables: RunTables, export_cfg: ExportConfig, n_events: int
    ) -> tuple[bool, str | None]:
        """Freeze the window to ``paths.snapshotDir`` (API-05). A snapshot failure never fails
        the window: the live bundle is already good; the error is recorded and logged."""
        cfg = self.config.snapshot
        if not cfg.enabled:
            return False, None
        if n_events == 0 and not cfg.writeEmptyWindows:
            log.info("live: window has no candidate events; snapshot kept as it was")
            return False, None
        try:
            export_bundle(
                tables,
                export_cfg,
                run.config.run,
                SNAPSHOT_MODE,
                run.waveforms,
                cache_dir=run.cache_dir,
                out_dir=self.snapshot_dir,
                features_loader=self.features_loader,
                baseline_cfg=run.config.validate.baseline,
            )
        except Exception as exc:  # the live bundle is already good; only the snapshot is lost
            log.exception("live: snapshot bundle not written to %s", self.snapshot_dir)
            return False, f"snapshot: {error_text(exc)}"
        log.info("live: snapshot bundle written to %s", self.snapshot_dir)
        return True, None

    def _commit(self, record: WindowRecord, served: ServedWindow | None) -> None:
        """Swap in the served window (good records only), persist, prune old bundles."""
        history_n = self.config.serve.historyN
        with self._swap_lock:
            if served is not None:
                self.served = served
                self.state.latest = record
            self.state.history = [record, *self.state.history][:history_n]
            write_state(self.state_file, self.state)
        if served is not None:
            self._prune_bundles()

    def _prune_bundles(self) -> None:
        """Keep the newest ``keepBundles`` live bundles (run ids sort by time), never the served."""
        if not self.bundles_dir.is_dir():
            return
        serving = self.served.bundle_dir.resolve() if self.served is not None else None
        dirs = sorted(
            (p for p in self.bundles_dir.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.name,
            reverse=True,
        )
        for old in dirs[self.config.paths.keepBundles :]:
            if old.resolve() == serving:
                continue
            shutil.rmtree(old, ignore_errors=True)
            log.info("live: removed old bundle %s", old)

    # --- reads (any thread) ---------------------------------------------------------------------

    def current(self) -> ServedWindow | None:
        with self._swap_lock:
            return self.served

    def snapshot_state(self) -> LiveState:
        with self._swap_lock:
            return self.state.model_copy(deep=True)

    def is_stale(self, served: ServedWindow) -> bool:
        return self.clock() - served.record.updatedAt > self.config.serve.staleAfterS

    def last_good(self) -> WindowRecord | None:
        state = self.snapshot_state()
        return next((r for r in state.history if r.outcome in GOOD_OUTCOMES), None)

    # --- scheduler (event loop) -----------------------------------------------------------------

    def start(self) -> None:
        """Start the ticker on the running loop: one window now, then every ``everyS``."""
        if self._ticker is not None:
            raise RuntimeError("the live worker is already started")
        self._stop = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hq-live")
        self._ticker = asyncio.create_task(self._tick_forever(), name="hq-live-ticker")

    async def run_once(self) -> WindowRecord | None:
        """One window in the executor; None when a window was already running (skipped)."""
        loop = asyncio.get_running_loop()
        executor = self._executor
        if executor is None:
            executor = self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="hq-live"
            )
        future = loop.run_in_executor(executor, self.run_window_now)
        self._pending.add(future)
        try:
            return await future
        finally:
            self._pending.discard(future)

    async def _tick_forever(self) -> None:
        every_s = self.config.window.everyS
        due = self.clock()
        while not self._stop.is_set():
            if self.running:
                self.skipped += 1
                log.warning(
                    "live: tick at %.0f skipped, the previous window is still running "
                    "(%d skipped so far)",
                    due,
                    self.skipped,
                )
            else:
                loop = asyncio.get_running_loop()
                task = loop.create_task(self.run_once())
                task.add_done_callback(_log_task_error)
            due += every_s
            self.next_run_at = due
            wait_s = max(0.0, due - self.clock())
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=wait_s)
            except TimeoutError:
                continue
        self.next_run_at = None

    async def stop(self) -> None:
        """Stop the ticker. A window still running finishes in its thread and is committed;
        the executor is released without waiting for it."""
        self._stop.set()
        if self._ticker is not None:
            self._ticker.cancel()
            try:
                await self._ticker
            except asyncio.CancelledError:
                pass
            self._ticker = None
        if self.running:
            log.warning(
                "live: a window is still running at shutdown; it finishes in the background"
            )
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


def _log_task_error(task: "asyncio.Task[WindowRecord | None]") -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:  # process() catches pipeline errors; this is a worker bug
        log.error("live: window task crashed: %s", error_text(exc), exc_info=exc)

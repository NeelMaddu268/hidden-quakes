"""API-04 acceptance: the live worker serves docs/02 §7 from the last good window, measures and
stores ``latencyS``, answers an empty window with an empty event list and an ok status, keeps
serving the previous window when a run fails (naming the missing stage's owner), restores the
last window after a restart, skips an overlapping tick, writes a snapshot bundle that passes
``check_bundle(mode="snapshot")`` and rejects unknown config keys. API-05: shutdown abandons a
running window promptly, and ``hq-api freeze-snapshot`` copies the last good live bundle into
the snapshot directory. Offline and fast."""

import asyncio
import json
import math
import shutil
import threading
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from hq_contracts.models import SCHEMA_VERSION, BundleMeta, EventEvidence, LiveStatus, SeismicEvent
from pydantic import ValidationError

from hq.export import check_bundle
from hq_api import __main__ as cli
from hq_api.app import create_app
from hq_api.config import DEFAULT_CONFIG_FILE, LiveConfig, LiveConfigError, load_live_config
from hq_api.runner import LiveWindow, window_run_section
from hq_api.snapshot import FreezeError, freezable_window, freeze_snapshot
from hq_api.state import LiveState, LiveStatusSummary, read_state
from hq_api.worker import LiveWorker
from tests.conftest import (
    API_DIR,
    MISSING_STAGE_OWNER,
    T_START,
    FakeClock,
    FakeRunner,
    no_features,
)

pytestmark = pytest.mark.smoke

API = "/api/live"
STATUS_FIELDS = tuple(name for name in LiveStatus.model_fields if name != "events")
SCHEDULER_WAIT_S = 20.0


def make_worker(
    config: LiveConfig, runner: FakeRunner, clock: FakeClock, tmp_path: Path
) -> LiveWorker:
    return LiveWorker(config, runner, root=tmp_path, clock=clock, features_loader=no_features)


def client_for(worker: LiveWorker, config: LiveConfig, runner: FakeRunner) -> TestClient:
    return TestClient(create_app(config, runner, worker=worker, scheduler=False))


# --- config --------------------------------------------------------------------------------------


def test_config_yaml_loads_and_unknown_keys_are_rejected(tmp_path: Path) -> None:
    cfg = load_live_config(API_DIR / DEFAULT_CONFIG_FILE)
    assert cfg.window.everyS <= cfg.window.windowS
    assert "export" not in cfg.window.stages and "validate" not in cfg.window.stages
    data = yaml.safe_load((API_DIR / DEFAULT_CONFIG_FILE).read_text())
    data["window"]["bogusKnob"] = 1
    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(LiveConfigError, match="bogusKnob"):
        load_live_config(bad)
    data["window"].pop("bogusKnob")
    data["window"]["stages"] = ["pick", "export"]
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(LiveConfigError, match="export"):
        load_live_config(bad)
    data["window"]["stages"] = ["pick", "inventory"]
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(LiveConfigError, match="pipeline order"):
        load_live_config(bad)
    with pytest.raises(LiveConfigError, match="not found"):
        load_live_config(tmp_path / "missing.yaml")


def test_window_run_section_replaces_only_the_window() -> None:
    source = {"name": "showcase", "windowStart": "a", "windowEnd": "b", "bbox": [1, 2, 3, 4]}
    out = window_run_section(source, LiveWindow(start=T_START - 7200.0, end=T_START))
    assert out["bbox"] == [1, 2, 3, 4]
    assert (
        out["windowStart"] == "2026-09-26T10:00:00Z" and out["windowEnd"] == "2026-09-26T12:00:00Z"
    )
    assert out["name"].startswith("live-")


def test_live_status_summary_mirrors_the_contract() -> None:
    assert tuple(LiveStatusSummary.model_fields) == STATUS_FIELDS
    with pytest.raises(ValidationError):
        LiveStatusSummary(updatedAt=1.0, windowS=1.0, latencyS=1.0, stationsOnline=1, events=[])


# --- endpoints from a window with events ---------------------------------------------------------


def test_window_with_events_is_served_and_latency_stored(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    runner = FakeRunner(live_config, n_events=3)
    worker = make_worker(live_config, runner, clock, tmp_path)
    record = worker.run_window_now()
    assert record is not None and record.outcome == "ok"
    window = runner.windows[0]
    assert window.end == math.floor(T_START - live_config.window.dataLagS)
    assert window.length_s == live_config.window.windowS
    assert record.latencyS is not None and math.isfinite(record.latencyS)
    assert record.latencyS >= live_config.window.dataLagS
    assert record.runtimeS > 0.0 and record.latencyS >= record.runtimeS
    assert record.stagesRan == ["tier"] and record.stationsOnline == 4

    client = client_for(worker, live_config, runner)
    meta = BundleMeta.model_validate(client.get(f"{API}/meta").json())
    assert meta.mode == "live" and meta.schemaVersion == SCHEMA_VERSION
    assert meta.run.id == record.runId and meta.run.mode == "live"
    assert meta.run.windowStart == window.start and meta.run.windowEnd == window.end
    assert meta.summary.candidateCount == 3 and meta.scene.heroEventId is not None
    events = [SeismicEvent.model_validate(e) for e in client.get(f"{API}/events").json()]
    assert len(events) == 3 and [e.revealOrder for e in events] == [0, 1, 2]
    assert all(e.runId == record.runId for e in events)
    evidence = EventEvidence.model_validate(client.get(f"{API}/evidence/{events[0].id}").json())
    assert evidence.eventId == events[0].id and 1 <= len(evidence.traces) <= 4
    assert client.get(f"{API}/evidence/hq-nope-000001").status_code == 404
    status = client.get(f"{API}/status").json()
    assert tuple(status) == STATUS_FIELDS
    assert status["latencyS"] == record.latencyS
    assert status["windowS"] == live_config.window.windowS
    assert status["updatedAt"] == record.updatedAt and status["stationsOnline"] == 4
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["served"]["runId"] == record.runId
    assert health["config"] == live_config.dump() and health["worker"]["attempts"] == 1

    state = read_state(worker.state_file)
    assert state.latest is not None and state.latest.latencyS == record.latencyS
    assert state.latest.evidenceIds == sorted(e.id for e in events)
    assert state.history[0] == state.latest
    assert not list(worker.state_file.parent.glob("*.tmp"))


def test_snapshot_bundle_is_written_and_checks(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    runner = FakeRunner(live_config, n_events=2)
    worker = make_worker(live_config, runner, clock, tmp_path)
    record = worker.run_window_now()
    assert record is not None and record.snapshotWritten and record.error is None
    counts = check_bundle(worker.snapshot_dir, mode="snapshot")
    assert counts["events"] == 2 and counts["evidenceFiles"] == 2
    snapshot_meta = json.loads((worker.snapshot_dir / "meta.json").read_text())
    live_meta = json.loads((Path(str(record.bundleDir)) / "meta.json").read_text())
    assert snapshot_meta["mode"] == "snapshot" and live_meta["mode"] == "live"
    assert snapshot_meta["run"] == live_meta["run"]
    assert check_bundle(Path(str(record.bundleDir)), mode="live")["events"] == 2


def test_events_are_capped_by_max_events(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    config = live_config.model_copy(
        update={"serve": live_config.serve.model_copy(update={"maxEvents": 2})}
    )
    runner = FakeRunner(config, n_events=5)
    worker = make_worker(config, runner, clock, tmp_path)
    assert worker.run_window_now() is not None
    events = client_for(worker, config, runner).get(f"{API}/events").json()
    assert [e["revealOrder"] for e in events] == [0, 1]


# --- empty window --------------------------------------------------------------------------------


def test_empty_window_serves_no_events_and_ok_status(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    runner = FakeRunner(live_config, n_events=0)
    worker = make_worker(live_config, runner, clock, tmp_path)
    record = worker.run_window_now()
    assert record is not None and record.outcome == "empty" and record.eventCount == 0
    assert record.latencyS is not None and math.isfinite(record.latencyS)
    client = client_for(worker, live_config, runner)
    assert client.get(f"{API}/events").json() == []
    status = client.get(f"{API}/status")
    assert status.status_code == 200 and status.json()["latencyS"] == record.latencyS
    meta = BundleMeta.model_validate(client.get(f"{API}/meta").json())
    assert meta.summary.candidateCount == 0 and meta.scene.heroEventId is None
    assert client.get("/health").json()["status"] == "ok"
    # The default snapshot policy never freezes an empty window (it would replace the committed
    # failover bundle with zero events), so nothing was written.
    assert not live_config.snapshot.writeEmptyWindows
    assert not record.snapshotWritten and not worker.snapshot_dir.exists()


def test_empty_window_keeps_the_last_snapshot_by_default(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    runner = FakeRunner(live_config, n_events=2)
    worker = make_worker(live_config, runner, clock, tmp_path)
    assert worker.run_window_now() is not None
    before = (worker.snapshot_dir / "meta.json").read_bytes()
    runner.n_events = 0
    clock.advance(live_config.window.everyS)
    record = worker.run_window_now()
    assert record is not None and record.outcome == "empty" and not record.snapshotWritten
    assert (worker.snapshot_dir / "meta.json").read_bytes() == before
    assert check_bundle(worker.snapshot_dir, mode="snapshot")["events"] == 2
    assert client_for(worker, live_config, runner).get(f"{API}/events").json() == []


def test_empty_window_is_frozen_when_configured(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    config = live_config.model_copy(
        update={"snapshot": live_config.snapshot.model_copy(update={"writeEmptyWindows": True})}
    )
    runner = FakeRunner(config, n_events=0)
    worker = make_worker(config, runner, clock, tmp_path)
    record = worker.run_window_now()
    assert record is not None and record.outcome == "empty" and record.snapshotWritten
    assert check_bundle(worker.snapshot_dir, mode="snapshot")["events"] == 0


def test_snapshot_failure_never_fails_a_good_window(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    """The live export pass reads each of the 4 stations once; the snapshot pass then hits the
    synthetic cache failure (a RuntimeError, not an ExportError)."""
    runner = FakeRunner(live_config, n_events=1, fail_reads_after=4)
    worker = make_worker(live_config, runner, clock, tmp_path)
    record = worker.run_window_now()
    assert record is not None and record.outcome == "ok" and record.eventCount == 1
    assert not record.snapshotWritten
    assert record.error is not None and record.error.startswith("snapshot: RuntimeError")
    assert not worker.snapshot_dir.exists()
    client = client_for(worker, live_config, runner)
    assert len(client.get(f"{API}/events").json()) == 1
    assert client.get("/health").json()["served"]["runId"] == record.runId
    assert read_state(worker.state_file).latest == record


# --- failures keep the last good window ----------------------------------------------------------


def test_no_window_yet_answers_503(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    runner = FakeRunner(live_config)
    worker = make_worker(live_config, runner, clock, tmp_path)
    client = client_for(worker, live_config, runner)
    for route in ("meta", "events", "status", "evidence/x"):
        assert client.get(f"{API}/{route}").status_code == 503
    health = client.get("/health").json()
    assert health["status"] == "waiting" and health["served"] is None and health["history"] == []


def test_failing_run_is_recorded_and_previous_window_still_served(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    runner = FakeRunner(live_config, n_events=2)
    worker = make_worker(live_config, runner, clock, tmp_path)
    good = worker.run_window_now()
    assert good is not None and good.outcome == "ok"
    snapshot_before = (worker.snapshot_dir / "meta.json").read_bytes()

    runner.fail = True
    clock.advance(live_config.window.everyS)
    failed = worker.run_window_now()
    assert failed is not None and failed.outcome == "failed"
    assert failed.error is not None and MISSING_STAGE_OWNER in failed.error
    assert "not implemented yet" in failed.error and failed.latencyS is None
    assert failed.runId is None and failed.bundleDir is None and not failed.snapshotWritten

    client = client_for(worker, live_config, runner)
    assert client.get(f"{API}/meta").json()["run"]["id"] == good.runId
    assert len(client.get(f"{API}/events").json()) == 2
    assert client.get(f"{API}/status").json()["updatedAt"] == good.updatedAt
    health = client.get("/health").json()
    assert health["served"]["runId"] == good.runId
    assert health["lastAttempt"]["outcome"] == "failed"
    assert [r["outcome"] for r in health["history"]] == ["failed", "ok"]
    assert health["worker"]["failures"] == 1 and health["worker"]["attempts"] == 2
    assert (worker.snapshot_dir / "meta.json").read_bytes() == snapshot_before
    state = read_state(worker.state_file)
    assert state.latest == good and [r.outcome for r in state.history] == ["failed", "ok"]


def test_first_window_failing_serves_nothing(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    runner = FakeRunner(live_config, fail=True)
    worker = make_worker(live_config, runner, clock, tmp_path)
    record = worker.run_window_now()
    assert record is not None and record.outcome == "failed"
    client = client_for(worker, live_config, runner)
    assert client.get(f"{API}/status").status_code == 503
    health = client.get("/health").json()
    assert health["status"] == "waiting" and health["lastAttempt"]["outcome"] == "failed"
    assert read_state(worker.state_file).latest is None


# --- restart -------------------------------------------------------------------------------------


def test_restart_restores_the_last_window(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    runner = FakeRunner(live_config, n_events=2)
    first = make_worker(live_config, runner, clock, tmp_path)
    record = first.run_window_now()
    assert record is not None
    events_before = client_for(first, live_config, runner).get(f"{API}/events").json()

    restarted = make_worker(live_config, FakeRunner(live_config), clock, tmp_path)
    assert restarted.attempts == 0
    client = client_for(restarted, live_config, runner)
    assert client.get(f"{API}/events").json() == events_before
    assert client.get(f"{API}/status").json()["latencyS"] == record.latencyS
    assert client.get(f"{API}/evidence/{events_before[0]['id']}").status_code == 200
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["served"]["runId"] == record.runId

    clock.advance(live_config.serve.staleAfterS + 1.0)
    assert client.get("/health").json()["status"] == "stale"


def test_restart_with_a_missing_bundle_serves_nothing(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    runner = FakeRunner(live_config, n_events=1)
    first = make_worker(live_config, runner, clock, tmp_path)
    record = first.run_window_now()
    assert record is not None and record.bundleDir is not None
    shutil.rmtree(record.bundleDir)
    restarted = make_worker(live_config, runner, clock, tmp_path)
    assert restarted.current() is None
    assert client_for(restarted, live_config, runner).get(f"{API}/meta").status_code == 503


def test_corrupt_state_file_fails_loudly(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    runner = FakeRunner(live_config)
    state_file = Path(str(live_config.paths.stateFile))
    state_file.parent.mkdir(parents=True)
    state_file.write_text("{not json")
    with pytest.raises(Exception, match="not a valid live state file"):
        make_worker(live_config, runner, clock, tmp_path)
    state_file.write_text(LiveState(version=99).model_dump_json())
    with pytest.raises(Exception, match="state version"):
        make_worker(live_config, runner, clock, tmp_path)


# --- overlap, pruning, scheduler -----------------------------------------------------------------


def test_overlapping_run_is_skipped(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    gate = threading.Event()
    runner = FakeRunner(live_config, n_events=1, block=gate)
    worker = make_worker(live_config, runner, clock, tmp_path)
    results: list[object] = []
    thread = threading.Thread(target=lambda: results.append(worker.run_window_now()))
    thread.start()
    try:
        assert runner.started.wait(SCHEDULER_WAIT_S)
        assert worker.running
        assert worker.run_window_now() is None  # skipped, not queued
        assert worker.skipped == 1 and worker.attempts == 1
    finally:
        gate.set()  # a failed assertion must never leave the worker thread blocked at exit
    thread.join(SCHEDULER_WAIT_S)
    assert not thread.is_alive() and not worker.running
    assert results and getattr(results[0], "outcome", None) == "ok"
    assert len(runner.windows) == 1


def test_old_bundles_are_pruned_but_never_the_served_one(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    config = live_config.model_copy(
        update={"paths": live_config.paths.model_copy(update={"keepBundles": 2})}
    )
    runner = FakeRunner(config, n_events=1)
    worker = make_worker(config, runner, clock, tmp_path)
    ids: list[str] = []
    for _ in range(3):
        record = worker.run_window_now()
        assert record is not None and record.runId is not None
        ids.append(record.runId)
        clock.advance(config.window.everyS)
    kept = sorted(p.name for p in worker.bundles_dir.iterdir())
    assert kept == sorted(ids[1:])
    assert worker.current() is not None and worker.current().record.runId == ids[-1]


def test_stop_clears_next_run_at_on_every_python(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    """The ticker's ``next_run_at`` is None after ``stop()`` whether the loop ends on the stop
    event or through cancellation. On Python 3.12+ a cancel issued right after the stop event
    wins the race inside ``asyncio.wait_for``, which used to skip the reset (REQ-H2-4)."""
    runner = FakeRunner(live_config, n_events=1)

    async def clean_stop() -> float | None:
        worker = make_worker(live_config, runner, clock, tmp_path / "clean")
        worker.start()
        await asyncio.sleep(0.05)
        assert worker.next_run_at == pytest.approx(T_START + live_config.window.everyS)
        await worker.stop()
        return worker.next_run_at

    async def cancel_first() -> float | None:
        worker = make_worker(live_config, runner, clock, tmp_path / "cancel")
        worker.start()
        await asyncio.sleep(0.05)
        assert worker._ticker is not None
        worker._ticker.cancel()  # the worst case: cancellation reaches the loop before the event
        await worker.stop()
        return worker.next_run_at

    assert asyncio.run(clean_stop()) is None
    assert asyncio.run(cancel_first()) is None


def test_scheduler_runs_a_window_at_startup(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    runner = FakeRunner(live_config, n_events=1)
    worker = make_worker(live_config, runner, clock, tmp_path)
    app = create_app(live_config, runner, worker=worker, scheduler=True)
    with TestClient(app) as client:
        assert runner.started.wait(SCHEDULER_WAIT_S)
        for _ in range(int(SCHEDULER_WAIT_S * 20)):
            if worker.current() is not None:
                break
            threading.Event().wait(0.05)
        assert worker.current() is not None
        assert client.get(f"{API}/status").status_code == 200
        health = client.get("/health").json()
        assert health["worker"]["attempts"] == 1 and health["worker"]["nextRunAt"] is not None
        assert health["worker"]["nextRunAt"] == pytest.approx(T_START + live_config.window.everyS)
    assert worker.next_run_at is None  # the ticker stopped with the app


# --- API-05: shutdown abandons a running window ---------------------------------------------------


def test_shutdown_abandons_a_running_window_promptly(
    live_config: LiveConfig,
    clock: FakeClock,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ctrl-C mid-window: stop() returns at once, the tick task is gone, the window is logged
    as abandoned and, when its thread finally finishes, it is not committed."""
    gate = threading.Event()
    runner = FakeRunner(live_config, n_events=1, block=gate)
    worker = make_worker(live_config, runner, clock, tmp_path)
    app = create_app(live_config, runner, worker=worker, scheduler=True)
    caplog.set_level("WARNING", logger="hq_api.worker")
    try:
        with TestClient(app):
            assert runner.started.wait(SCHEDULER_WAIT_S)
            assert worker.running
            t0 = time.perf_counter()
        stop_s = time.perf_counter() - t0
        assert stop_s < SCHEDULER_WAIT_S / 4, f"stop() waited {stop_s:.1f} s for the window"
        assert worker.abandoned and worker.running  # the thread is still blocked on the gate
        assert not worker._tasks and not worker._pending and worker.next_run_at is None
        assert any("abandoned" in r.getMessage() for r in caplog.records)
    finally:
        gate.set()  # a failed assertion must never leave the worker thread blocked at exit
    for _ in range(int(SCHEDULER_WAIT_S * 20)):
        if not worker.running:
            break
        threading.Event().wait(0.05)
    assert not worker.running
    assert worker.current() is None  # finished after shutdown: not served ...
    assert read_state(worker.state_file).history == []  # ... and not in the state file
    assert any("not committed" in r.getMessage() for r in caplog.records)


def test_stop_without_a_running_window_is_clean_and_idempotent(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    runner = FakeRunner(live_config, n_events=1)
    worker = make_worker(live_config, runner, clock, tmp_path)
    app = create_app(live_config, runner, worker=worker, scheduler=True)
    with TestClient(app):
        for _ in range(int(SCHEDULER_WAIT_S * 20)):
            if worker.current() is not None and not worker._tasks:
                break
            threading.Event().wait(0.05)
    assert not worker.abandoned and not worker.running
    assert worker.current() is not None and len(read_state(worker.state_file).history) == 1
    asyncio.run(worker.stop())  # a second stop is a no-op
    assert not worker.abandoned


# --- API-05: freeze-snapshot ---------------------------------------------------------------------


def without_runtime_snapshot(config: LiveConfig) -> LiveConfig:
    """The worker's own snapshot write off, so what lands in snapshotDir is the command's."""
    return config.model_copy(
        update={"snapshot": config.snapshot.model_copy(update={"enabled": False})}
    )


def test_freeze_snapshot_copies_the_last_good_window_with_events(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    config = without_runtime_snapshot(live_config)
    runner = FakeRunner(config, n_events=3)
    worker = make_worker(config, runner, clock, tmp_path)
    good = worker.run_window_now()
    assert good is not None and good.outcome == "ok" and good.bundleDir is not None
    assert not worker.snapshot_dir.exists()

    # A later empty window is served live but is not what gets frozen by default.
    clock.advance(config.window.everyS)
    runner.n_events = 0
    empty = worker.run_window_now()
    assert empty is not None and empty.outcome == "empty"
    assert worker.current() is not None and worker.current().record.runId == empty.runId

    result = freeze_snapshot(config, tmp_path)
    assert result.record.runId == good.runId and result.target == worker.snapshot_dir
    assert result.counts["events"] == 3
    counts = check_bundle(worker.snapshot_dir, mode="snapshot")
    assert counts["events"] == 3
    meta = BundleMeta.model_validate_json((worker.snapshot_dir / "meta.json").read_text())
    assert meta.mode == "snapshot" and meta.run.id == good.runId
    live_dir = Path(good.bundleDir)
    assert (worker.snapshot_dir / "events.json").read_bytes() == (
        live_dir / "events.json"
    ).read_bytes()
    live_evidence = sorted(p.name for p in (live_dir / "evidence").iterdir())
    assert sorted(p.name for p in (worker.snapshot_dir / "evidence").iterdir()) == live_evidence
    # The live bundle is untouched (still mode live, still served).
    assert check_bundle(live_dir, mode="live")["events"] == 3
    assert not [p for p in worker.snapshot_dir.parent.iterdir() if p.name.startswith(".")]

    # --allow-empty freezes the newest good window even without candidate events.
    result = freeze_snapshot(config, tmp_path, allow_empty=True)
    assert result.record.runId == empty.runId
    assert check_bundle(worker.snapshot_dir, mode="snapshot")["events"] == 0


def test_freeze_snapshot_refuses_without_a_freezable_window(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    config = without_runtime_snapshot(live_config)
    with pytest.raises(FreezeError, match="no good live window"):
        freeze_snapshot(config, tmp_path)  # no state file at all
    runner = FakeRunner(config, n_events=2)
    worker = make_worker(config, runner, clock, tmp_path)
    record = worker.run_window_now()
    assert record is not None and record.bundleDir is not None
    shutil.rmtree(record.bundleDir)  # pruned or lost: skipped, with nothing else to freeze
    assert freezable_window(read_state(worker.state_file)) is None
    with pytest.raises(FreezeError, match="no good live window"):
        freeze_snapshot(config, tmp_path)
    assert not worker.snapshot_dir.exists()


def test_freeze_snapshot_keeps_the_previous_snapshot_when_the_copy_fails(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path
) -> None:
    config = without_runtime_snapshot(live_config)
    runner = FakeRunner(config, n_events=2)
    worker = make_worker(config, runner, clock, tmp_path)
    first = worker.run_window_now()
    assert first is not None
    freeze_snapshot(config, tmp_path)
    before = (worker.snapshot_dir / "meta.json").read_bytes()
    clock.advance(config.window.everyS)
    second = worker.run_window_now()
    assert second is not None and second.bundleDir is not None
    (Path(second.bundleDir) / "events.json").write_text("[]")  # corrupt the newest live bundle
    with pytest.raises(FreezeError, match="does not check"):
        freeze_snapshot(config, tmp_path)
    assert (worker.snapshot_dir / "meta.json").read_bytes() == before
    assert not [p for p in worker.snapshot_dir.parent.iterdir() if p.name.startswith(".")]


def test_cli_parses_serve_by_default_and_freeze_snapshot(
    live_config: LiveConfig, clock: FakeClock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.parse_args([]).command == "serve"
    args = cli.parse_args(["--port", "8001", "--no-scheduler"])
    assert args.command == "serve" and args.port == 8001 and args.no_scheduler
    args = cli.parse_args(["freeze-snapshot", "--allow-empty"])
    assert args.command == "freeze-snapshot" and args.allow_empty
    with pytest.raises(SystemExit):
        cli.parse_args(["bogus-command"])

    config = without_runtime_snapshot(live_config)
    config_yaml = tmp_path / "live.yaml"
    config_yaml.write_text(yaml.safe_dump(config.dump()))
    assert cli.main(["freeze-snapshot", "--config", str(config_yaml)]) == 1  # nothing to freeze
    runner = FakeRunner(config, n_events=1)
    worker = make_worker(config, runner, clock, tmp_path)
    assert worker.run_window_now() is not None
    assert cli.main(["freeze-snapshot", "--config", str(config_yaml)]) == 0
    out = capsys.readouterr().out
    assert "snapshot bundle written" in out and "1 candidate events" in out
    assert check_bundle(worker.snapshot_dir, mode="snapshot")["events"] == 1
    assert cli.main(["freeze-snapshot", "--config", str(tmp_path / "missing.yaml")]) == 1

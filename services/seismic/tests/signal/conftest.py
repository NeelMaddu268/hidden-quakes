"""Shared fixtures for the signal lane's tests.

``FakeRunContext`` is a light stand-in for H4's ``hq.runs.RunContext`` (docs/02 -> Stage API): it
records what a stage passes to ``record()`` instead of writing run.json, and rejects a stage name
H4's registry (``hq.runs.STAGES``) would reject.

Every test here is offline (CLAUDE.md rule 10): the autouse ``_no_network`` guard refuses any
connection or DNS lookup that leaves the machine.
"""

import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from hq.config.run import RunSection
from hq.config.signal import SignalConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "showcase"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


def _host(address: Any) -> Any:
    """The host of a socket address ((host, port[, flowinfo, scope_id]) for AF_INET/AF_INET6)."""
    host = address[0] if isinstance(address, tuple) and address else address
    return host.decode() if isinstance(host, bytes) else host


def _refuse_remote(what: str, host: Any) -> None:
    if host not in _LOOPBACK:
        raise AssertionError(f"network access in an offline test: {what} {host!r}")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every non-loopback connection and DNS lookup: each service is injected.

    Loopback must stay allowed: on Windows, asyncio's ProactorEventLoop (seisbench's ``classify``
    runs one) connects a self-pipe socketpair to 127.0.0.1 when the loop starts.
    """
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex
    create_connection, getaddrinfo = socket.create_connection, socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _refuse_remote("connect", _host(address))
        connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _refuse_remote("connect_ex", _host(address))
        return connect_ex(self, address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        host = _host(address)
        if host not in (None, ""):
            _refuse_remote("create_connection", host)
        return create_connection(address, *args, **kwargs)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        name = _host(host)
        if name not in (None, ""):
            _refuse_remote("getaddrinfo", name)
        return getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@dataclass(frozen=True)
class FakeConfig:
    run: RunSection
    signal: SignalConfig


@dataclass(frozen=True)
class FakeRunContext:
    run_id: str
    run_dir: Path
    cache_dir: Path
    config: FakeConfig
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def record(
        self,
        stage: str,
        *,
        runtime_s: float,
        counts: dict[str, int],
        params: dict | None = None,
        field: str | None = None,
    ) -> None:
        from hq.runs import STAGES  # H4's registry: record() rejects any other stage name

        if stage not in {s.name for s in STAGES}:
            raise ValueError(f"unknown stage {stage!r}: RunContext.record would reject it")
        self.records[stage] = {
            "runtime_s": runtime_s,
            "counts": counts,
            "params": params,
            "field": field,
        }


def load_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def run_section() -> RunSection:
    return RunSection.model_validate(load_yaml("run.yaml"))


@pytest.fixture(scope="session")
def raw_signal_yaml() -> dict:
    return load_yaml("signal.yaml")


@pytest.fixture(scope="session")
def signal_cfg(raw_signal_yaml: dict) -> SignalConfig:
    return SignalConfig.model_validate(raw_signal_yaml)


@pytest.fixture()
def fake_ctx(tmp_path: Path, run_section: RunSection, signal_cfg: SignalConfig) -> FakeRunContext:
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return FakeRunContext(
        run_id="test-run",
        run_dir=run_dir,
        cache_dir=cache_dir,
        config=FakeConfig(run=run_section, signal=signal_cfg),
    )

"""Shared fixtures for H2 seismology tests: config loading, a RunContext stand-in, no network.

The stand-in mirrors the parts of H4's RunContext (docs/02 §4) that H2 stages use until RUN-01
lands: ``config.run``, ``config.seismology``, ``path(name)`` and ``record(...)``.
"""

import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig

SEISMIC_ROOT = Path(__file__).resolve().parents[2]
SHOWCASE_DIR = SEISMIC_ROOT / "configs" / "showcase"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that opens a network connection, even indirectly."""

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network access in an offline test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)  # DNS lookups leave the machine too


@dataclass(frozen=True)
class StubConfig:
    run: RunSection
    seismology: SeismologyConfig


@dataclass(frozen=True)  # docs/02 §4: RunContext is frozen; records is appended to, never rebound
class StubRunContext:
    run_id: str
    run_dir: Path
    cache_dir: Path
    config: StubConfig
    records: list[dict[str, Any]] = field(default_factory=list)

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
        self.records.append(
            {"stage": stage, "runtime_s": runtime_s, "counts": counts, "params": params}
        )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{path} is not a mapping"
    return data


@pytest.fixture
def run_section() -> RunSection:
    return RunSection.model_validate(load_yaml(SHOWCASE_DIR / "run.yaml"))


@pytest.fixture
def seismology_config() -> SeismologyConfig:
    return SeismologyConfig.model_validate(load_yaml(SHOWCASE_DIR / "seismology.yaml"))


@pytest.fixture
def comcat_quakeml() -> Path:
    """Real ComCat QuakeML for the showcase query, trimmed to 3 events (newest first, as served).

    Provenance: fixtures/comcat_uu_3events.provenance.txt.
    """
    return FIXTURES / "comcat_uu_3events.quakeml"


@pytest.fixture
def make_ctx(tmp_path: Path, seismology_config: SeismologyConfig) -> Any:
    """Build a StubRunContext for a given RunSection, with its run dir under ``tmp_path``."""

    def _make(run: RunSection, seismology: SeismologyConfig | None = None) -> StubRunContext:
        run_dir = tmp_path / "runs" / "test-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return StubRunContext(
            run_id="test-run",
            run_dir=run_dir,
            cache_dir=tmp_path / "cache",
            config=StubConfig(run=run, seismology=seismology or seismology_config),
        )

    return _make

"""Shared fixtures for the signal lane's tests.

``FakeRunContext`` is a local stand-in for H4's ``hq.runs.RunContext`` (docs/02 -> Stage API)
until RUN-01 lands; it records what a stage passes to ``record()`` instead of writing run.json.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from hq.config.run import RunSection
from hq.config.signal import SignalConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "showcase"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


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
    ) -> None:
        self.records[stage] = {"runtime_s": runtime_s, "counts": counts, "params": params}


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

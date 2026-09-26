"""Shared fixtures for H2 seismology tests: config loading, a RunContext stand-in, no network.

The stand-in mirrors the parts of H4's RunContext (docs/02 §4) that H2 stages use:
``config.run``, ``config.seismology``, ``path(name)``, ``record(...)`` and the params fields of
``read_run()``.

LOC-02 helpers (test station geometry, test config, exact picks) reach the tests as the session
fixture ``loc02``: under ``--import-mode=importlib`` a test module can't import a sibling module.
"""

from __future__ import annotations

import socket
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import yaml

from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig

if TYPE_CHECKING:  # the LOC-02 modules load lazily, so an import error there fails only LOC-02
    from hq.locate.locator import Locator, LocatorSetup
    from hq.locate.velocity import LayerModel

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

    def read_run(self) -> Any:
        """The ProcessingRun params fields as ``records`` leave them, merged one level deep as
        H4's ``RunContext.record`` merges them (docs/02 §4); only those fields are stubbed."""
        from hq.runs import PARAM_FIELDS, STAGE_PARAM_FIELDS

        fields: dict[str, dict[str, Any]] = {name: {} for name in PARAM_FIELDS}
        for rec in self.records:
            target = rec["field"] or STAGE_PARAM_FIELDS.get(rec["stage"])
            if rec["params"] is not None and target is not None:
                fields[target] = {**fields[target], **rec["params"]}
        return SimpleNamespace(**fields)

    def record(
        self,
        stage: str,
        *,
        runtime_s: float,
        counts: dict[str, int],
        params: dict[str, Any] | None = None,
        field: str | None = None,
    ) -> None:
        self.records.append(
            {"stage": stage, "runtime_s": runtime_s, "counts": counts, "params": params,
             "field": field}
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


# --- LOC-02: travel-time tables, locator, synthetic test ---------------------------------------


class Loc02Kit:
    """LOC-02 test helpers: a small made-up station geometry, a test config, exact picks.

    The geometry is made up inside the tests (tests may build small synthetic data): 12 stations
    within 7 km of the origin, 3 of them borehole sensors about 300 m down. The config is the
    showcase ``seismology.yaml`` with a smaller search volume and shorter tables, so the smoke
    tests stay fast; every other knob is the showcase value. All LOC-02 modules share one
    travel-time cache per session (``cache_dir``).
    """

    # id, e (m), n (m), surfaceElevM, sensorDepthM
    STATIONS: tuple[tuple[str, float, float, float, float], ...] = (
        ("T.S01", 1500.0, 0.0, 1650.0, 0.0),
        ("T.S02", -1200.0, 2500.0, 1700.0, 0.0),
        ("T.S03", -3000.0, -1500.0, 1560.0, 0.0),
        ("T.S04", 2500.0, -3500.0, 1880.0, 0.0),
        ("T.S05", 5500.0, 1000.0, 2100.0, 0.0),
        ("T.S06", -6000.0, 2500.0, 1520.0, 0.0),
        ("T.S07", 1000.0, 6000.0, 1740.0, 0.0),
        ("T.S08", -2500.0, -6000.0, 1600.0, 0.0),
        ("T.S09", 4500.0, -5000.0, 1950.0, 0.0),
        ("T.B01", 300.0, 500.0, 1690.0, 290.0),
        ("T.B02", -800.0, -300.0, 1650.0, 300.0),
        ("T.B03", 200.0, -900.0, 1700.0, 320.0),
    )

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    @staticmethod
    def showcase_raw() -> dict[str, Any]:
        """A fresh parse of the showcase seismology.yaml (callers may mutate it)."""
        return load_yaml(SHOWCASE_DIR / "seismology.yaml")

    def test_config(self, **synthetic: object) -> SeismologyConfig:
        raw = self.showcase_raw()
        raw["grids"] = {**raw["grids"], "rMaxM": 15000.0, "bottomElevM": -6000.0}
        raw["locator"] = {
            **raw["locator"],
            "volume": {"halfWidthM": 5000.0, "topElevM": None, "bottomElevM": -5000.0},
            "nWorkers": 1,
        }
        raw["synthetic"] = {**raw["synthetic"], **synthetic}
        return SeismologyConfig.model_validate(raw)

    def stations(self, origin_elev_m: float) -> pd.DataFrame:
        rows = []
        for sid, e, n, surface, depth in self.STATIONS:
            sensor = surface - depth
            rows.append(
                {
                    "id": sid,
                    "surfaceElevM": surface,
                    "sensorDepthM": depth,
                    "sensorElevM": sensor,
                    "kind": "borehole" if depth > 0 else "surface",
                    "enu_e": e,
                    "enu_n": n,
                    "enu_u": sensor - origin_elev_m,
                    "preprocessProfile": "borehole" if depth > 0 else "surface",
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def toy_model(tops: list[float], vp: list[float], vs: list[float]) -> LayerModel:
        from hq.locate.velocity import LayerModel, SourceRef

        return LayerModel(
            name="toy",
            datum="topElevM is m above mean sea level.",
            source=SourceRef(citation="toy", url="https://example.invalid", verified=False),
            top_elev_m=np.array(tops),
            vp_m_per_s=np.array(vp),
            vs_m_per_s=np.array(vs),
            source_file="https://example.invalid/toy.csv",
            license="CC0",
        )

    def exact_picks(
        self,
        locator: Locator,
        e: float,
        n: float,
        elev_m: float,
        t0: float,
        *,
        s_stations: Collection[str] | None = None,
        p_stations: Collection[str] | None = None,
        prob: float = 1.0,
    ) -> pd.DataFrame:
        """Noise-free picks from the exact layered times (receivers from ``STATIONS``).

        P on every station (or ``p_stations``), S on every station (or ``s_stations``).
        """
        from hq.locate.tt_grid import PHASES, layered_first_arrival

        rows = []
        for sid, se, sn, surface, depth in self.STATIONS:
            z_rec = surface - depth
            r = float(np.hypot(e - se, n - sn))
            for ph in PHASES:
                wanted = s_stations if ph == "S" else p_stations
                if wanted is not None and sid not in wanted:
                    continue
                tt = float(layered_first_arrival(locator.tables.model, ph, z_rec, r, elev_m))
                rows.append({"id": f"test:{sid}:{ph}", "stationId": sid, "phase": ph,
                             "t": t0 + tt, "prob": prob})
        return pd.DataFrame(rows)

    @staticmethod
    def run_section() -> RunSection:
        return RunSection.model_validate(load_yaml(SHOWCASE_DIR / "run.yaml"))

    def setup(
        self, cfg: SeismologyConfig | None = None, run: RunSection | None = None
    ) -> LocatorSetup:
        from hq.locate.locator import LocatorSetup
        from hq.locate.velocity import load_configured_model

        config = cfg if cfg is not None else self.test_config()
        section = run if run is not None else self.run_section()
        return LocatorSetup(
            stations=self.stations(section.origin.elevM),
            model=load_configured_model(config.velocity),
            config=config,
            run=section,
            cache_dir=self.cache_dir,
        )


@pytest.fixture(scope="session")
def loc02(tmp_path_factory: pytest.TempPathFactory) -> Loc02Kit:
    """LOC-02 helpers with one travel-time cache directory for the whole session."""
    return Loc02Kit(tmp_path_factory.mktemp("loc02-cache"))

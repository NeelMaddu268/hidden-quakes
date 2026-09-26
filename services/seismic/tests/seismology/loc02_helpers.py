"""LOC-02 test helpers: a small synthetic station geometry, a test config, exact picks.

No conftest: each LOC-02 test module builds its fixtures from these helpers, and every module
shares one travel-time cache per pytest session (``shared_cache``).

The geometry is made up inside the tests (docs: tests may build small synthetic data): 12 stations
within 7 km of the origin, 3 of them borehole sensors about 300 m down. The config is the
showcase ``seismology.yaml`` with a smaller search volume and shorter tables, so the smoke tests
stay fast; every other knob is the showcase value.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from hq.config.run import RunSection
from hq.config.seismology import SEISMIC_ROOT, SeismologyConfig
from hq.locate.locator import Locator, LocatorSetup
from hq.locate.tt_grid import PHASES, layered_first_arrival
from hq.locate.velocity import LayerModel, SourceRef, load_configured_model

SHOWCASE = SEISMIC_ROOT / "configs" / "showcase"

# id, e (m), n (m), surfaceElevM, sensorDepthM
TEST_STATIONS = (
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


def showcase_raw() -> dict:
    return yaml.safe_load((SHOWCASE / "seismology.yaml").read_text(encoding="utf-8"))


def make_test_config(**synthetic: object) -> SeismologyConfig:
    raw = showcase_raw()
    raw["grids"] = {**raw["grids"], "rMaxM": 15000.0, "bottomElevM": -6000.0}
    raw["locator"] = {
        **raw["locator"],
        "volume": {"halfWidthM": 5000.0, "topElevM": None, "bottomElevM": -5000.0},
        "nWorkers": 1,
    }
    raw["synthetic"] = {**raw["synthetic"], **synthetic}
    return SeismologyConfig.model_validate(raw)


def make_stations(origin_elev_m: float) -> pd.DataFrame:
    rows = []
    for sid, e, n, surface, depth in TEST_STATIONS:
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
            }
        )
    return pd.DataFrame(rows)


def toy_model(tops: list[float], vp: list[float], vs: list[float]) -> LayerModel:
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
    locator: Locator,
    e: float,
    n: float,
    elev_m: float,
    t0: float,
    *,
    s_stations: set[str] | None = None,
) -> pd.DataFrame:
    """Noise-free picks from the exact layered times: P on every station, S where listed."""
    rows = []
    for sid, se, sn, *_ in TEST_STATIONS:
        z_rec = locator.tables.receiver_elev_m[sid]
        r = float(np.hypot(e - se, n - sn))
        for ph in PHASES:
            if ph == "S" and s_stations is not None and sid not in s_stations:
                continue
            tt = float(layered_first_arrival(locator.tables.model, ph, z_rec, r, elev_m))
            rows.append({"id": f"test:{sid}:{ph}", "stationId": sid, "phase": ph,
                         "t": t0 + tt, "prob": 1.0})
    return pd.DataFrame(rows)


def load_run_section() -> RunSection:
    raw = yaml.safe_load((SHOWCASE / "run.yaml").read_text(encoding="utf-8"))
    return RunSection.model_validate(raw)


def shared_cache(factory: pytest.TempPathFactory) -> Path:
    """One cache directory per pytest session, shared by the LOC-02 test modules."""
    path = factory.getbasetemp() / "loc02-cache"
    path.mkdir(exist_ok=True)
    return path


def make_setup(cache_dir: Path, cfg: SeismologyConfig | None = None) -> LocatorSetup:
    config = cfg if cfg is not None else make_test_config()
    run = load_run_section()
    return LocatorSetup(
        stations=make_stations(run.origin.elevM),
        model=load_configured_model(config.velocity),
        config=config,
        run=run,
        cache_dir=cache_dir,
    )


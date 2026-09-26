"""LOC-02: the synthetic recovery test (small smoke version) and its synthetic.json writer."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from loc02_helpers import make_setup, make_test_config, shared_cache

from hq.locate.locator import Locator, LocatorSetup, build_locator
from hq.locate.synthetic import (
    SyntheticTest,
    draw_hypocentres,
    run_synthetic,
    synthetic_picks,
    write_synthetic_json,
)

pytestmark = pytest.mark.smoke

# docs/02 SyntheticTest, written out so a dropped or renamed field fails here.
DOCS02_SYNTHETIC_FIELDS = ("nEvents", "pickSigmaS", "medianHErrM", "medianVErrM", "p90VErrM",
                           "medianDepthBiasM")
N_SMOKE = 6


@pytest.fixture(scope="module")
def loc_setup(tmp_path_factory: pytest.TempPathFactory) -> LocatorSetup:
    return make_setup(shared_cache(tmp_path_factory))


@pytest.fixture(scope="module")
def locator(loc_setup: LocatorSetup) -> Locator:
    return build_locator(loc_setup)


@pytest.fixture(scope="module")
def smoke_setup(loc_setup: LocatorSetup) -> LocatorSetup:
    return dataclasses.replace(loc_setup, config=make_test_config(nEvents=N_SMOKE))


def test_test_geometry_has_boreholes(loc_setup: LocatorSetup) -> None:
    st = loc_setup.stations
    assert len(st) >= 10
    deep = st[st["sensorDepthM"] >= 100.0]
    assert 2 <= len(deep) <= 3 and set(deep["kind"]) == {"borehole"}
    np.testing.assert_allclose(st["sensorElevM"], st["surfaceElevM"] - st["sensorDepthM"])


def test_small_synthetic_run(smoke_setup: LocatorSetup, tmp_path: Path) -> None:
    result = run_synthetic(smoke_setup)
    report = result.report
    assert report.nEvents == N_SMOKE == len(result.events)
    assert report.pickSigmaS == {"P": 0.02, "S": 0.04}
    clean = result.params["noiseFree"]
    assert abs(clean["medianDepthBiasM"]) <= 50.0  # ticket: noise-free bias within +/-50 m
    assert clean["medianHErrM"] <= 50.0 and clean["medianVErrM"] <= 50.0
    assert 0.0 < report.medianHErrM < 500.0 and 0.0 < report.medianVErrM < 500.0
    assert report.p90VErrM >= report.medianVErrM
    noisy = result.params["noisy"]
    assert 0.0 <= noisy["fracHWithinHErrM"] <= 1.0 and 0.0 <= noisy["fracVWithinVErrM"] <= 1.0
    assert result.params["depthBiasSign"].startswith("elevM_true - elevM_located")
    ev = result.events
    zone = smoke_setup.config.synthetic.zone
    assert np.all(np.hypot(ev["e"] - zone.centerEM, ev["n"] - zone.centerNM) <= zone.radiusM)
    assert ev["elevM"].between(zone.bottomElevM, zone.topElevM).all()
    np.testing.assert_allclose(ev["noisyDepthBiasM"], ev["elevM"] - ev["noisyElevM"])

    path = write_synthetic_json(tmp_path / "run" / "synthetic.json", report)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(written) == DOCS02_SYNTHETIC_FIELDS
    assert set(written["pickSigmaS"]) == {"P", "S"}
    assert written["medianVErrM"] == report.medianVErrM
    assert not list(path.parent.glob(".*.part"))


def test_synthetic_test_stand_in_matches_docs02() -> None:
    assert tuple(f.name for f in dataclasses.fields(SyntheticTest)) == DOCS02_SYNTHETIC_FIELDS


def test_synthetic_draws_are_seeded(smoke_setup: LocatorSetup, locator: Locator) -> None:
    def draw() -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame]]:
        rng = np.random.default_rng(smoke_setup.config.synthetic.seed)
        truth = draw_hypocentres(smoke_setup, 4, rng)
        noisy, clean = synthetic_picks(smoke_setup, locator.tables.model, truth, rng)
        return truth, noisy, clean

    (t1, n1, c1), (t2, n2, c2) = draw(), draw()
    pd.testing.assert_frame_equal(t1, t2)
    for a, b in zip(n1 + c1, n2 + c2, strict=True):
        pd.testing.assert_frame_equal(a, b)
    run = smoke_setup.run
    assert t1["t0"].between(run.window_start_s, run.window_end_s).all()
    for noisy, clean in zip(n1, c1, strict=True):
        assert list(noisy["id"]) == list(clean["id"])  # same kept S picks with and without noise
        assert (noisy["phase"] == "P").sum() == len(smoke_setup.stations)
        resid = (noisy["t"] - clean["t"]).to_numpy()
        assert np.all(np.abs(resid) < 0.3) and np.any(resid != 0.0)

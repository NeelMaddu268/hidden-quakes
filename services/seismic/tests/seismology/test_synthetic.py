"""LOC-02: the synthetic recovery test (small smoke version) and its synthetic.json writer."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hq_contracts.models import SyntheticTest

from hq.config.seismology import PhaseSigma
from hq.locate.locator import Locator, LocatorSetup, build_locator
from hq.locate.synthetic import (
    draw_hypocentres,
    geometry_record,
    run_synthetic,
    synthetic_picks,
    write_synthetic_json,
)

# docs/02 SyntheticTest, written out so a dropped or renamed field fails here.
DOCS02_SYNTHETIC_FIELDS = ("nEvents", "pickSigmaS", "medianHErrM", "medianVErrM", "p90VErrM",
                           "medianDepthBiasM")
N_SMOKE = 6


@pytest.fixture(scope="module")
def loc_setup(loc02: Any) -> LocatorSetup:
    return loc02.setup()


@pytest.fixture(scope="module")
def locator(loc_setup: LocatorSetup) -> Locator:
    return build_locator(loc_setup)


@pytest.fixture(scope="module")
def smoke_setup(loc02: Any, loc_setup: LocatorSetup) -> LocatorSetup:
    # seismology.yaml leaves sKeepProb / pickProb null (stage locate measures them from the run);
    # a direct run_synthetic call needs numbers.
    config = loc02.test_config(nEvents=N_SMOKE, sKeepProb=0.6, pickProb=1.0)
    return dataclasses.replace(loc_setup, config=config)


@pytest.mark.smoke
def test_test_geometry_has_boreholes(loc_setup: LocatorSetup) -> None:
    st = loc_setup.stations
    assert len(st) >= 10
    deep = st[st["sensorDepthM"] >= 100.0]
    assert 2 <= len(deep) <= 3 and set(deep["kind"]) == {"borehole"}
    np.testing.assert_allclose(st["sensorElevM"], st["surfaceElevM"] - st["sensorDepthM"])


def test_small_synthetic_run(smoke_setup: LocatorSetup, tmp_path: Path) -> None:
    result = run_synthetic(smoke_setup, geometry_label="TEST")
    report = result.report
    assert report.nEvents == N_SMOKE == len(result.events)
    assert report.pickSigmaS == {"P": 0.02, "S": 0.04}
    clean = result.params["noiseFree"]
    assert abs(clean["medianDepthBiasM"]) <= 50.0  # ticket: noise-free bias within +/-50 m
    assert clean["medianHErrM"] <= 50.0 and clean["medianVErrM"] <= 50.0
    assert 0.0 < report.medianHErrM < 500.0 and 0.0 < report.medianVErrM < 500.0
    assert report.p90VErrM >= report.medianVErrM
    noisy = result.params["noisy"]
    assert noisy["nPdfTruncated"] == 0
    for summary, prefix in ((noisy, "noisy"), (clean, "clean")):  # zone well below the volume top
        assert summary["nMapOnVolumeTop"] == int(result.events[f"{prefix}MapOnVolumeTop"].sum())
        assert summary["nMapOnVolumeTop"] == summary["nMapOnVolumeBottom"] == 0
    assert 0.0 <= noisy["fracHWithinHErrM"] <= 1.0 and 0.0 <= noisy["fracVWithinVErrM"] <= 1.0
    params = result.params
    assert params["depthBiasSign"].startswith("elevM_true - elevM_located")
    assert params["pickStats"]["source"] == "seismology.yaml"
    assert params["pickStats"]["sKeepProb"] == 0.6 and params["pickStats"]["pickProb"] == 1.0
    assert "P pick" in params["pickStats"]["caveat"]
    geometry = params["stationGeometry"]
    stations = smoke_setup.stations
    assert geometry["label"] == "TEST" and geometry["stationIds"] == stations["id"].tolist()
    assert geometry["sensorElevM"] == stations["sensorElevM"].tolist()
    assert geometry == geometry_record(stations, "TEST") and len(geometry["sha256"]) == 64
    assert params["locator"]["stationIds"] == geometry["stationIds"]
    assert params["velocityModel"]["topExtension"] is not None
    json.dumps(params, allow_nan=False)  # the params can go into run.json as they are
    ev = result.events
    zone = smoke_setup.config.synthetic.zone
    assert np.all(np.hypot(ev["e"] - zone.centerEM, ev["n"] - zone.centerNM) <= zone.radiusM)
    assert ev["elevM"].between(zone.bottomElevM, zone.topElevM).all()
    np.testing.assert_allclose(ev["noisyDepthBiasM"], ev["elevM"] - ev["noisyElevM"])

    path = write_synthetic_json(tmp_path / "run" / "synthetic.json", report)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(written) == DOCS02_SYNTHETIC_FIELDS
    assert SyntheticTest.model_validate(written) == report  # the landed docs/02 contract model
    assert set(written["pickSigmaS"]) == {"P", "S"}
    assert written["medianVErrM"] == report.medianVErrM
    assert not list(path.parent.glob(".*.part"))


@pytest.mark.smoke
def test_null_pick_stats_need_numbers(loc02: Any, loc_setup: LocatorSetup) -> None:
    """seismology.yaml's null sKeepProb / pickProb: only stage locate may fill them."""
    null = dataclasses.replace(loc_setup, config=loc02.test_config(nEvents=N_SMOKE))
    assert null.config.synthetic.sKeepProb is None and null.config.synthetic.pickProb is None
    with pytest.raises(ValueError, match="measures them from the run"):
        run_synthetic(null)


@pytest.mark.smoke
def test_geometry_hash_changes_with_the_geometry(loc_setup: LocatorSetup) -> None:
    st = loc_setup.stations
    base = geometry_record(st, None)["sha256"]
    moved = st.copy()
    moved.loc[0, "sensorElevM"] += 1.0
    assert geometry_record(moved, None)["sha256"] != base
    assert geometry_record(st.copy(), "other label")["sha256"] == base


@pytest.mark.smoke
def test_synthetic_test_model_matches_docs02() -> None:
    assert tuple(SyntheticTest.model_fields) == DOCS02_SYNTHETIC_FIELDS


@pytest.mark.smoke
def test_synthetic_draws_are_seeded(smoke_setup: LocatorSetup, locator: Locator) -> None:
    def draw() -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame]]:
        rng = np.random.default_rng(smoke_setup.config.synthetic.seed)
        truth = draw_hypocentres(smoke_setup, 4, rng)
        noisy, clean = synthetic_picks(smoke_setup, locator, truth, rng)
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


@pytest.mark.smoke
def test_synthetic_noise_uses_the_locator_sigma_per_station(
    smoke_setup: LocatorSetup, locator: Locator
) -> None:
    # A per-profile sigma override must change the simulated noise, not only the fit's weights.
    cfg = smoke_setup.config
    loud = cfg.locator.model_copy(
        update={"profilePickSigmaS": {"borehole": PhaseSigma(P=0.5, S=0.5)}})
    loud_locator = Locator(
        smoke_setup.stations, locator.tables, loud, origin_elev_m=smoke_setup.run.origin.elevM,
        ref_surface_elev_m=smoke_setup.run.refSurfaceElevM,
    )
    rng = np.random.default_rng(cfg.synthetic.seed)
    truth = draw_hypocentres(smoke_setup, 20, rng)
    noisy, clean = synthetic_picks(smoke_setup, loud_locator, truth, rng)
    resid = pd.concat([(n.assign(r=n["t"] - c["t"])) for n, c in zip(noisy, clean, strict=True)])
    borehole = resid["stationId"].str.startswith("T.B")
    surface_p = resid[~borehole & (resid["phase"] == "P")]["r"].std()
    borehole_p = resid[borehole & (resid["phase"] == "P")]["r"].std()
    assert surface_p == pytest.approx(0.02, rel=0.3) and borehole_p == pytest.approx(0.5, rel=0.3)

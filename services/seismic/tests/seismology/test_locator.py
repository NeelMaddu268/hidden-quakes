"""LOC-02: grid-search locator, origin-time elimination, outliers, PDF uncertainty, depthOnEdge."""

import math

import numpy as np
import pandas as pd
import pytest
from loc02_helpers import TEST_STATIONS, exact_picks, make_setup, make_stations, shared_cache

from hq.config.run import RunSection
from hq.config.seismology import LocatorConfig, SeismologyConfig
from hq.locate.locator import (
    Locator,
    LocatorSetup,
    azimuthal_gap_deg,
    build_locator,
    l1_misfit,
    locate_many,
    make_volume,
    weighted_median,
)
from hq.locate.uncertainty import chi2_2, summarize_pdf

pytestmark = pytest.mark.smoke

T0 = 1757462400.0  # an epoch in the showcase window; absolute times must not matter
FINE = 25.0
# docs/02 LocationQuality fields
QUALITY_FIELDS = {"method", "statics", "nStations", "nP", "nS", "rmsS", "gapDeg", "minEpiDistM",
                  "hErrM", "vErrM", "depthOnEdge"}


@pytest.fixture(scope="module")
def loc_setup(tmp_path_factory: pytest.TempPathFactory) -> LocatorSetup:
    return make_setup(shared_cache(tmp_path_factory))


@pytest.fixture(scope="module")
def locator(loc_setup: LocatorSetup) -> Locator:
    return build_locator(loc_setup)


@pytest.fixture(scope="module")
def loc_run(loc_setup: LocatorSetup) -> RunSection:
    return loc_setup.run


@pytest.fixture(scope="module")
def loc_cfg(loc_setup: LocatorSetup) -> SeismologyConfig:
    return loc_setup.config


# --- misfit core ------------------------------------------------------------------------------


def _brute_l1(values: np.ndarray, weights: np.ndarray) -> float:
    # The L1 minimum lies at a data value, so checking every value is exhaustive.
    return float(min(np.sum(weights * np.abs(values - v)) for v in values))


def test_weighted_median_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    for size in (1, 2, 3, 4, 7, 10, 25):
        for _ in range(40):
            values = rng.normal(0.0, 1.0, size)
            weights = rng.uniform(0.1, 3.0, size)
            if size > 2 and rng.random() < 0.3:
                values[1] = values[0]  # ties
            t0 = weighted_median(values, weights)
            best = _brute_l1(values, weights)
            assert np.sum(weights * np.abs(values - t0)) == pytest.approx(best, rel=1e-12, abs=1e-12)
            assert t0 in values
            below = weights[values < t0].sum()
            above = weights[values > t0].sum()
            assert below <= 0.5 * weights.sum() + 1e-12 and above <= 0.5 * weights.sum() + 1e-12
    # Equal weights, even count: the lower median.
    assert weighted_median([4.0, 1.0, 3.0, 2.0], [1.0, 1.0, 1.0, 1.0]) == 2.0
    with pytest.raises(ValueError, match="positive"):
        weighted_median([1.0, 2.0], [1.0, 0.0])


def test_l1_misfit_rows_match_one_dimensional() -> None:
    rng = np.random.default_rng(5)
    d = rng.normal(0.0, 0.3, (200, 9))
    w = rng.uniform(5.0, 50.0, 9)
    misfit, t0 = l1_misfit(d, w)
    for i in range(0, 200, 17):
        assert t0[i] == weighted_median(d[i], w)
        assert misfit[i] == pytest.approx(_brute_l1(d[i], w), rel=1e-12)


def test_azimuthal_gap() -> None:
    assert azimuthal_gap_deg([1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]) == pytest.approx(90.0)
    assert azimuthal_gap_deg([1.0, 1.0], [1.0, 1.001]) == pytest.approx(360.0 - 0.0286, abs=1e-3)
    assert azimuthal_gap_deg([1.0], [0.0]) == 360.0
    assert azimuthal_gap_deg([0.0, 1.0, -1.0], [0.0, 0.0, 0.0]) == pytest.approx(180.0)


# --- PDF uncertainty ---------------------------------------------------------------------------


def _gaussian_box(sig: tuple[float, float, float], half: tuple[int, int, int], center=(0, 0, 0)):
    e = np.arange(-half[0], half[0] + 1) * FINE + center[0]
    n = np.arange(-half[1], half[1] + 1) * FINE + center[1]
    z = np.arange(-half[2], half[2] + 1) * FINE + center[2] - 2000.0
    zz, nn, ee = np.meshgrid(z, n, e, indexing="ij")
    misfit = 0.5 * (((ee - center[0]) / sig[0]) ** 2 + ((nn - center[1]) / sig[1]) ** 2
                    + ((zz - center[2] + 2000.0) / sig[2]) ** 2)
    return misfit, e, n, z


def test_pdf_of_a_gaussian_misfit_gives_the_expected_sigmas() -> None:
    sig = (120.0, 60.0, 200.0)
    misfit, e, n, z = _gaussian_box(sig, (7 * 5, 7 * 3, 7 * 8))
    pdf = summarize_pdf(misfit, e, n, z, spacing_h_m=FINE, spacing_z_m=FINE, top_face_level=None,
                        bottom_face_level=None, confidence=0.68, edge_fraction=0.05)
    # Sampling a Gaussian on a lattice adds (spacing^2 / 12) to the variance (Sheppard).
    assert pdf.v_err_m == pytest.approx(math.sqrt(sig[2] ** 2 + FINE**2 / 12), rel=2e-3)
    expected_h = math.sqrt(chi2_2(0.68) * (sig[0] ** 2 + FINE**2 / 12))
    assert pdf.h_err_m == pytest.approx(expected_h, rel=2e-3)
    assert (pdf.mean_e_m, pdf.mean_n_m) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert pdf.mean_elev_m == pytest.approx(-2000.0, abs=1e-6)
    assert not (pdf.h_err_floored or pdf.v_err_floored or pdf.depth_on_edge)
    assert chi2_2(0.68) == pytest.approx(-2.0 * math.log(0.32), rel=1e-12)  # 2-dof closed form


def test_pdf_rotated_ellipse_uses_the_largest_eigenvalue() -> None:
    e = n = np.arange(-40, 41) * FINE
    z = np.arange(-10, 11) * FINE
    zz, nn, ee = np.meshgrid(z, n, e, indexing="ij")
    a, b = (ee + nn) / math.sqrt(2), (ee - nn) / math.sqrt(2)  # axes at 45 degrees
    misfit = 0.5 * ((a / 150.0) ** 2 + (b / 50.0) ** 2 + (zz / 80.0) ** 2)
    pdf = summarize_pdf(misfit, e, n, z, spacing_h_m=FINE, spacing_z_m=FINE, top_face_level=None,
                        bottom_face_level=None, confidence=0.68, edge_fraction=0.05)
    assert pdf.h_err_m == pytest.approx(math.sqrt(chi2_2(0.68)) * 150.0, rel=0.01)
    assert pdf.cov[0, 1] == pytest.approx(0.5 * (150.0**2 - 50.0**2), rel=0.01)


def test_grid_limited_pdf_is_floored_and_flagged() -> None:
    misfit, e, n, z = _gaussian_box((1.0, 1.0, 1.0), (4, 4, 4))
    pdf = summarize_pdf(misfit, e, n, z, spacing_h_m=FINE, spacing_z_m=FINE, top_face_level=None,
                        bottom_face_level=None, confidence=0.68, edge_fraction=0.05)
    assert pdf.max_node_mass > 0.999
    assert pdf.v_err_m == pytest.approx(FINE / math.sqrt(12.0))
    assert pdf.h_err_m == pytest.approx(math.sqrt(chi2_2(0.68) * FINE**2 / 12.0))
    assert pdf.h_err_floored and pdf.v_err_floored


def test_face_mass_sets_depth_on_edge() -> None:
    misfit, e, n, z = _gaussian_box((80.0, 80.0, 60.0), (12, 12, 12))
    top = z.size - 1
    cut = 12  # the misfit minimum: a box clipped there has its peak on the top face
    pdf = summarize_pdf(misfit[: cut + 1], e, n, z[: cut + 1], spacing_h_m=FINE,
                        spacing_z_m=FINE, top_face_level=cut, bottom_face_level=0,
                        confidence=0.68, edge_fraction=0.05)
    assert pdf.top_face_mass > 0.05 and pdf.depth_on_edge  # peak sits on the top face
    assert pdf.bottom_face_mass < 1e-5  # 5 sigma below the peak
    unclipped = summarize_pdf(misfit, e, n, z, spacing_h_m=FINE, spacing_z_m=FINE,
                              top_face_level=top, bottom_face_level=0, confidence=0.68,
                              edge_fraction=0.05)
    assert not unclipped.depth_on_edge


# --- locator on the synthetic test geometry ---------------------------------------------------


def _err(loc, e: float, n: float, z: float) -> tuple[float, float]:
    return math.hypot(loc.e_m - e, loc.n_m - n), abs(loc.elev_m - z)


def test_noise_free_recovery(locator: Locator) -> None:
    truths = [(0.0, 0.0, -2000.0), (1300.0, -700.0, -1200.0), (-1500.0, 900.0, -4200.0),
              (400.0, 1800.0, -3100.0), (-900.0, -1600.0, -2600.0)]
    bias = []
    for k, (e, n, z) in enumerate(truths):
        s_on = {sid for i, (sid, *_) in enumerate(TEST_STATIONS) if (i + k) % 2 == 0}
        loc = locator.locate(exact_picks(locator, e, n, z, T0 + k, s_stations=s_on))
        h_err, v_err = _err(loc, e, n, z)
        assert h_err <= 2 * FINE and v_err <= 2 * FINE, (h_err, v_err)
        assert loc.t0 == pytest.approx(T0 + k, abs=0.01)
        assert loc.rms_s < 0.01 and not loc.dropped_pick_ids and not loc.depth_on_edge
        assert loc.h_err_m > 0 and loc.v_err_m > 0
        bias.append(z - loc.elev_m)
    assert abs(float(np.median(bias))) <= 50.0


def test_origin_time_invariance(locator: Locator) -> None:
    picks = exact_picks(locator, 700.0, -300.0, -2400.0, T0)
    rng = np.random.default_rng(11)
    picks["t"] += rng.normal(0.0, 0.02, len(picks))
    shift = 3600.123456
    a = locator.locate(picks)
    b = locator.locate(picks.assign(t=picks["t"] + shift))
    assert (a.e_m, a.n_m, a.elev_m) == (b.e_m, b.n_m, b.elev_m)
    assert b.t0 - a.t0 == pytest.approx(shift, abs=1e-6)
    np.testing.assert_allclose(a.arrivals["residualS"], b.arrivals["residualS"], atol=1e-6)
    assert a.pdf.h_err_m == pytest.approx(b.pdf.h_err_m, rel=1e-9)


def test_planted_outlier_is_dropped(locator: Locator) -> None:
    picks = exact_picks(locator, -500.0, 600.0, -2800.0, T0)
    bad = picks.index[(picks["stationId"] == "T.S04") & (picks["phase"] == "P")][0]
    picks.loc[bad, "t"] += 1.0
    loc = locator.locate(picks)
    assert loc.dropped_pick_ids == (picks.loc[bad, "id"],)
    assert loc.relocated and loc.outlier_note is None
    row = loc.arrivals.set_index("pickId").loc[picks.loc[bad, "id"]]
    assert not row["used"] and row["residualS"] == pytest.approx(1.0, abs=0.02)
    assert loc.arrivals["used"].sum() == len(picks) - 1
    h_err, v_err = _err(loc, -500.0, 600.0, -2800.0)
    assert h_err <= 2 * FINE and v_err <= 2 * FINE
    assert loc.n_p == len(TEST_STATIONS) - 1


def test_depth_on_edge_fires_at_the_volume_top(locator: Locator) -> None:
    top = locator.volume.top_elev_m
    loc = locator.locate(exact_picks(locator, 200.0, -400.0, top, T0))
    assert loc.search["clippedTop"] and loc.search["mapOnTopFace"]
    assert loc.pdf.top_face_mass > 0.05
    assert loc.depth_on_edge and loc.quality()["depthOnEdge"]
    deep = locator.locate(exact_picks(locator, 200.0, -400.0, -2500.0, T0))
    assert not deep.depth_on_edge and deep.pdf.top_face_mass < 1e-6


def test_statics_are_additive(locator: Locator) -> None:
    picks = exact_picks(locator, 300.0, 300.0, -2200.0, T0)
    statics = {("T.S02", "P"): 0.12, ("T.B01", "S"): -0.08}
    shifted = picks.copy()
    for (sid, ph), s in statics.items():
        shifted.loc[(shifted["stationId"] == sid) & (shifted["phase"] == ph), "t"] += s
    plain = locator.locate(picks)
    with_statics = locator.locate(shifted, statics=statics)
    assert (plain.e_m, plain.n_m, plain.elev_m) == (with_statics.e_m, with_statics.n_m,
                                                    with_statics.elev_m)
    assert with_statics.statics_applied and not plain.statics_applied
    np.testing.assert_allclose(plain.arrivals["residualS"], with_statics.arrivals["residualS"],
                               atol=1e-9)
    rows = with_statics.arrivals.set_index(["stationId", "phase"])
    assert rows.loc[("T.S02", "P"), "staticS"] == 0.12


def test_quality_fields_and_arrivals(locator: Locator) -> None:
    s_on = {"T.S01", "T.S02", "T.B01"}
    loc = locator.locate(exact_picks(locator, 0.0, 0.0, -2000.0, T0, s_stations=s_on))
    q = loc.quality()
    assert set(q) == QUALITY_FIELDS and q["method"] == "grid1d" and q["statics"] is False
    assert (q["nStations"], q["nP"], q["nS"]) == (len(TEST_STATIONS), len(TEST_STATIONS), 3)
    se = np.array([s[1] for s in TEST_STATIONS]) - loc.e_m
    sn = np.array([s[2] for s in TEST_STATIONS]) - loc.n_m
    assert q["minEpiDistM"] == pytest.approx(float(np.hypot(se, sn).min()))
    assert q["gapDeg"] == pytest.approx(azimuthal_gap_deg(se, sn))
    arr = loc.arrivals
    np.testing.assert_allclose(arr["tPred"], loc.t0 + arr["travelTimeS"] + arr["staticS"])
    np.testing.assert_allclose(arr["residualS"], arr["tObs"] - arr["tPred"])
    np.testing.assert_allclose(arr["weight"], 1.0 / arr["sigmaS"])
    assert loc.u_m == pytest.approx(loc.elev_m - locator.origin_elev_m)
    assert set(locator.to_record()) >= {"method", "config", "volume", "misfit", "uncertainty"}


def test_bad_picks_raise(locator: Locator) -> None:
    picks = exact_picks(locator, 0.0, 0.0, -2000.0, T0)
    with pytest.raises(ValueError, match="at least"):
        locator.locate(picks.iloc[:3])
    with pytest.raises(ValueError, match="more than one pick"):
        locator.locate(pd.concat([picks, picks.iloc[:1]]))
    with pytest.raises(ValueError, match="without coordinates"):
        locator.locate(picks.assign(stationId="X.NOPE"))
    with pytest.raises(ValueError, match="probabilities"):
        locator.locate(picks.assign(prob=0.0))


def test_outlier_pass_keeps_min_picks(
    loc_setup: LocatorSetup, locator: Locator, loc_run: RunSection
) -> None:
    picks = exact_picks(locator, 0.0, 0.0, -2000.0, T0)
    keep = ["T.S01", "T.S02", "T.S03", "T.S04", "T.B01"]
    picks = picks[picks["stationId"].isin(keep)].reset_index(drop=True)
    picks.loc[0, "t"] += 2.0  # T.S01 P
    loc = locator.locate(picks)  # 10 picks: the outlier goes, 9 >= minPicks remain
    assert loc.dropped_pick_ids == (picks.loc[0, "id"],)
    strict = Locator(
        loc_setup.stations, locator.tables,
        loc_setup.config.locator.model_copy(update={"minPicks": len(picks)}),
        origin_elev_m=loc_run.origin.elevM, ref_surface_elev_m=loc_run.refSurfaceElevM,
    )
    kept = strict.locate(picks)  # dropping one would leave fewer than minPicks
    assert kept.dropped_pick_ids == () and not kept.relocated
    assert kept.outlier_note is not None and "minPicks" in kept.outlier_note
    assert kept.arrivals["used"].all()


def test_station_out_of_table_reach_raises(
    loc_setup: LocatorSetup, locator: Locator, loc_run: RunSection
) -> None:
    far = make_stations(loc_run.origin.elevM)
    far.loc[0, "enu_e"] = 9500.0  # 15.3 km from the far corner of the volume > rMaxM 15 km
    with pytest.raises(ValueError, match="rMaxM"):
        Locator(far, locator.tables, loc_setup.config.locator,
                origin_elev_m=loc_run.origin.elevM,
                ref_surface_elev_m=loc_run.refSurfaceElevM)
    bad_u = make_stations(loc_run.origin.elevM)
    bad_u.loc[1, "enu_u"] += 5.0
    with pytest.raises(ValueError, match="enu_u"):
        Locator(bad_u, locator.tables, loc_setup.config.locator,
                origin_elev_m=loc_run.origin.elevM,
                ref_surface_elev_m=loc_run.refSurfaceElevM)


def test_volume_top_defaults_to_the_reference_surface(loc_cfg: SeismologyConfig) -> None:
    vol = make_volume(loc_cfg.locator, 1627.7)
    assert vol.top_elev_m == 1625.0 and vol.configured_top_elev_m == 1627.7
    assert vol.top_elev_m <= 1627.7 and vol.coarse_stride == 8
    explicit = loc_cfg.locator.model_copy(
        update={"volume": loc_cfg.locator.volume.model_copy(update={"topElevM": 1000.0})})
    assert make_volume(explicit, 1627.7).top_elev_m == 1000.0
    raw = loc_cfg.locator.model_dump()
    with pytest.raises(ValueError, match="multiple of fineSpacingM"):
        LocatorConfig.model_validate({**raw, "coarseSpacingM": 210.0})


def test_locate_many_matches_serial_with_workers(loc_setup: LocatorSetup,
                                                 locator: Locator) -> None:
    events = [exact_picks(locator, e, n, z, T0) for e, n, z in
              ((0.0, 0.0, -1500.0), (800.0, 400.0, -3300.0), (-600.0, -900.0, -2500.0))]
    serial = locate_many(loc_setup, events, locator=locator)
    cfg = loc_setup.config
    two = cfg.model_copy(update={"locator": cfg.locator.model_copy(update={"nWorkers": 2})})
    parallel = locate_many(LocatorSetup(loc_setup.stations, loc_setup.model, two, loc_setup.run,
                                        loc_setup.cache_dir), events)
    for a, b in zip(serial, parallel, strict=True):
        assert (a.e_m, a.n_m, a.elev_m, a.t0, a.h_err_m, a.v_err_m) == (
            b.e_m, b.n_m, b.elev_m, b.t0, b.h_err_m, b.v_err_m)

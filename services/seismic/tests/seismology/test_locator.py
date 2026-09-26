"""LOC-02: grid-search locator, origin-time elimination, outliers, PDF uncertainty, depthOnEdge."""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hq_contracts.models import LocationQuality

from hq.config.run import RunSection
from hq.config.seismology import LocatorConfig, PhaseSigma, SeismologyConfig
from hq.locate.locator import (
    ARRIVAL_COLUMNS,
    EventLocation,
    Locator,
    LocatorSetup,
    azimuthal_gap_deg,
    build_locator,
    l1_misfit,
    locate_many,
    make_volume,
    weighted_median,
)
from hq.locate.uncertainty import FACES, PdfSummary, chi2_2, summarize_pdf

pytestmark = pytest.mark.smoke

FINE = 25.0
# docs/02 LocationQuality fields
QUALITY_FIELDS = {"method", "statics", "nStations", "nP", "nS", "rmsS", "gapDeg", "minEpiDistM",
                  "hErrM", "vErrM", "depthOnEdge"}
# docs/02 arrivals.parquet columns that LOC-04 builds straight from EventLocation.arrivals
DOCS02_ARRIVAL_COLUMNS = {"stationId", "phase", "tPred", "tObs", "residualS", "pickId",
                          "usedInLocation"}
FloatArray = np.ndarray


@pytest.fixture(scope="module")
def loc_setup(loc02: Any) -> LocatorSetup:
    return loc02.setup()


@pytest.fixture(scope="module")
def locator(loc_setup: LocatorSetup) -> Locator:
    return build_locator(loc_setup)


@pytest.fixture(scope="module")
def loc_run(loc_setup: LocatorSetup) -> RunSection:
    return loc_setup.run


@pytest.fixture(scope="module")
def loc_cfg(loc_setup: LocatorSetup) -> SeismologyConfig:
    return loc_setup.config


@pytest.fixture(scope="module")
def t0(loc_run: RunSection) -> float:
    """An origin time in the showcase window; absolute times must not matter."""
    return loc_run.window_start_s + 3600.0


def _with_locator_cfg(loc_setup: LocatorSetup, locator: Locator, **update: Any) -> Locator:
    """A Locator on the same tables and stations with some locator knobs changed (validated)."""
    cfg = LocatorConfig.model_validate({**loc_setup.config.locator.model_dump(), **update})
    return Locator(
        loc_setup.stations, locator.tables, cfg,
        origin_elev_m=loc_setup.run.origin.elevM, ref_surface_elev_m=loc_setup.run.refSurfaceElevM,
    )


def _noisy(picks: pd.DataFrame, seed: int, sigma_p: float, sigma_s: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sigma = np.where(picks["phase"] == "P", sigma_p, sigma_s)
    return picks.assign(t=picks["t"] + rng.normal(0.0, 1.0, len(picks)) * sigma)


# --- misfit core ------------------------------------------------------------------------------


def _brute_l1(values: FloatArray, weights: FloatArray) -> float:
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


def _gaussian_box(
    sig: tuple[float, float, float], half: tuple[int, int, int]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    e = np.arange(-half[0], half[0] + 1) * FINE
    n = np.arange(-half[1], half[1] + 1) * FINE
    z = np.arange(-half[2], half[2] + 1) * FINE - 2000.0
    zz, nn, ee = np.meshgrid(z, n, e, indexing="ij")
    misfit = 0.5 * ((ee / sig[0]) ** 2 + (nn / sig[1]) ** 2 + ((zz + 2000.0) / sig[2]) ** 2)
    return misfit, e, n, z


def _summary(misfit: FloatArray, e: FloatArray, n: FloatArray, z: FloatArray, *,
             scale: float = 1.0, top: bool = False, bottom: bool = False) -> PdfSummary:
    return summarize_pdf(misfit, e, n, z, spacing_h_m=FINE, spacing_z_m=FINE, misfit_scale=scale,
                         top_is_volume_top=top, bottom_is_volume_bottom=bottom, confidence=0.68,
                         edge_fraction=0.05)


def test_pdf_of_a_gaussian_misfit_gives_the_expected_sigmas() -> None:
    sig = (120.0, 60.0, 200.0)
    misfit, e, n, z = _gaussian_box(sig, (7 * 5, 7 * 3, 7 * 8))
    pdf = _summary(misfit, e, n, z)
    # A Gaussian density sampled at nodes spaced well below sigma has a discrete variance of
    # sigma^2 to exponentially small terms (no spacing^2/12 term: the nodes are samples, not bins).
    assert pdf.v_err_m == pytest.approx(sig[2], rel=1e-6)
    assert pdf.h_err_m == pytest.approx(math.sqrt(chi2_2(0.68)) * sig[0], rel=1e-6)
    assert (pdf.mean_e_m, pdf.mean_n_m) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert pdf.mean_elev_m == pytest.approx(-2000.0, abs=1e-6)
    assert not (pdf.h_err_floored or pdf.v_err_floored or pdf.depth_on_edge)
    assert chi2_2(0.68) == pytest.approx(-2.0 * math.log(0.32), rel=1e-12)  # 2-dof closed form
    # pdfMisfitScale s: exp(-s * misfit) of a quadratic misfit has sd sigma / sqrt(s).
    scaled = _summary(misfit, e, n, z, scale=0.64)
    assert scaled.v_err_m == pytest.approx(sig[2] / 0.8, rel=1e-6)
    assert scaled.h_err_m == pytest.approx(math.sqrt(chi2_2(0.68)) * sig[0] / 0.8, rel=1e-6)


def test_pdf_rotated_ellipse_uses_the_largest_eigenvalue() -> None:
    e = n = np.arange(-40, 41) * FINE
    z = np.arange(-10, 11) * FINE
    zz, nn, ee = np.meshgrid(z, n, e, indexing="ij")
    a, b = (ee + nn) / math.sqrt(2), (ee - nn) / math.sqrt(2)  # axes at 45 degrees
    misfit = 0.5 * ((a / 150.0) ** 2 + (b / 50.0) ** 2 + (zz / 80.0) ** 2)
    pdf = _summary(misfit, e, n, z)
    assert pdf.h_err_m == pytest.approx(math.sqrt(chi2_2(0.68)) * 150.0, rel=0.01)
    assert pdf.cov[0, 1] == pytest.approx(0.5 * (150.0**2 - 50.0**2), rel=0.01)


def test_grid_limited_pdf_is_floored_and_flagged() -> None:
    misfit, e, n, z = _gaussian_box((1.0, 1.0, 1.0), (4, 4, 4))
    pdf = _summary(misfit, e, n, z)
    assert pdf.max_node_mass > 0.999
    assert pdf.v_err_m == pytest.approx(FINE / math.sqrt(12.0))
    assert pdf.h_err_m == pytest.approx(math.sqrt(chi2_2(0.68) * FINE**2 / 12.0))
    assert pdf.h_err_floored and pdf.v_err_floored


def test_face_masses_and_depth_on_edge() -> None:
    misfit, e, n, z = _gaussian_box((80.0, 80.0, 60.0), (12, 12, 12))
    cut = 12  # the misfit minimum: a box clipped there has its peak on the top face
    pdf = _summary(misfit[: cut + 1], e, n, z[: cut + 1], top=True, bottom=True)
    assert set(pdf.face_mass) == set(FACES)
    assert pdf.top_face_mass > 0.05 and pdf.depth_on_edge  # peak sits on the top face
    assert pdf.bottom_face_mass < 1e-5  # 5 sigma below the peak
    assert pdf.face_mass["east"] == pytest.approx(pdf.face_mass["west"], rel=1e-9)
    # Lateral face masses: a box cut off at the east peak holds a large mass on its east face.
    east = _summary(misfit[:, :, :13], e[:13], n, z)
    assert east.face_mass["east"] > 0.05 and east.face_mass["west"] < 1e-3  # 3.75 sigma away
    assert not east.depth_on_edge  # lateral faces never set depthOnEdge
    assert not _summary(misfit, e, n, z, top=True, bottom=True).depth_on_edge


def test_broad_pdf_map_boundary_is_separate_from_contracted_face_mass_flag() -> None:
    # Vertical sd 400 m peaked on the top row: the top row holds < 5% of the mass.
    misfit, e, n, z = _gaussian_box((80.0, 80.0, 400.0), (8, 8, 100))
    peak = 100  # z index of the minimum
    box = (misfit[: peak + 1], e, n, z[: peak + 1])
    pinned = _summary(*box, top=True)
    assert pinned.top_face_mass < 0.05 and pinned.map_on_volume_top and not pinned.depth_on_edge
    interior = _summary(*box, top=False)  # the same face inside the volume: no z = 0 collapse
    assert not interior.map_on_volume_top and not interior.depth_on_edge
    upside_down = _summary(misfit[peak:], e, n, z[peak:], bottom=True)
    assert upside_down.map_on_volume_bottom and not upside_down.depth_on_edge


# --- locator on the synthetic test geometry ---------------------------------------------------


def _err(loc: EventLocation, e: float, n: float, z: float) -> tuple[float, float]:
    return math.hypot(loc.e_m - e, loc.n_m - n), abs(loc.elev_m - z)


def test_noise_free_recovery(loc02: Any, locator: Locator, t0: float) -> None:
    truths = [(0.0, 0.0, -2000.0), (1300.0, -700.0, -1200.0), (-1500.0, 900.0, -4200.0),
              (400.0, 1800.0, -3100.0), (-900.0, -1600.0, -2600.0)]
    bias = []
    for k, (e, n, z) in enumerate(truths):
        s_on = {sid for i, (sid, *_) in enumerate(loc02.STATIONS) if (i + k) % 2 == 0}
        loc = locator.locate(loc02.exact_picks(locator, e, n, z, t0 + k, s_stations=s_on))
        h_err, v_err = _err(loc, e, n, z)
        assert h_err <= 2 * FINE and v_err <= 2 * FINE, (h_err, v_err)
        assert loc.t0 == pytest.approx(t0 + k, abs=0.01)
        assert loc.rms_s < 0.01 and not loc.dropped_pick_ids and not loc.depth_on_edge
        assert loc.h_err_m is not None and loc.v_err_m is not None
        assert loc.h_err_m > 0 and loc.v_err_m > 0 and not loc.pdf_truncated
        bias.append(z - loc.elev_m)
    assert abs(float(np.median(bias))) <= 50.0


def test_origin_time_invariance(loc02: Any, locator: Locator, t0: float) -> None:
    picks = _noisy(loc02.exact_picks(locator, 700.0, -300.0, -2400.0, t0), 11, 0.02, 0.02)
    shift = 3600.123456
    a = locator.locate(picks)
    b = locator.locate(picks.assign(t=picks["t"] + shift))
    assert (a.e_m, a.n_m, a.elev_m) == (b.e_m, b.n_m, b.elev_m)
    assert b.t0 - a.t0 == pytest.approx(shift, abs=1e-6)
    np.testing.assert_allclose(a.arrivals["residualS"], b.arrivals["residualS"], atol=1e-6)
    assert a.pdf.h_err_m == pytest.approx(b.pdf.h_err_m, rel=1e-9)


def test_planted_outlier_is_dropped(loc02: Any, locator: Locator, t0: float) -> None:
    picks = loc02.exact_picks(locator, -500.0, 600.0, -2800.0, t0)
    bad = picks.index[(picks["stationId"] == "T.S04") & (picks["phase"] == "P")][0]
    picks.loc[bad, "t"] += 1.0
    loc = locator.locate(picks)
    assert loc.dropped_pick_ids == (picks.loc[bad, "id"],)
    assert loc.relocated and loc.outlier_note is None
    row = loc.arrivals.set_index("pickId").loc[picks.loc[bad, "id"]]
    assert not row["usedInLocation"] and row["residualS"] == pytest.approx(1.0, abs=0.02)
    assert loc.arrivals["usedInLocation"].sum() == len(picks) - 1
    h_err, v_err = _err(loc, -500.0, 600.0, -2800.0)
    assert h_err <= 2 * FINE and v_err <= 2 * FINE
    assert loc.n_p == len(loc02.STATIONS) - 1


def test_mad_branch_sets_the_outlier_threshold(
    loc02: Any, locator: Locator, loc_cfg: SeismologyConfig, t0: float
) -> None:
    # Pick noise well above sigma: madK * MAD exceeds floorS, so the MAD term sets the threshold.
    picks = _noisy(loc02.exact_picks(locator, 400.0, -200.0, -2600.0, t0), 21, 0.1, 0.2)
    bad = picks.index[(picks["stationId"] == "T.S02") & (picks["phase"] == "P")][0]
    picks.loc[bad, "t"] += 1.5
    loc = locator.locate(picks)
    # First pass by hand: all picks, residuals at its MAP, MAD over both phases, unscaled.
    p = locator._prepare(picks, None)
    first = locator._search(p, np.ones(len(picks), dtype=bool))
    res = locator._residuals(p, first)
    mad = float(np.median(np.abs(res - np.median(res))))
    out = loc_cfg.locator.outlier
    assert out.madK * mad > out.floorS  # the MAD branch, not the floor
    assert loc.outlier_mad_s == pytest.approx(mad, abs=1e-12)
    assert loc.outlier_threshold_s == pytest.approx(out.madK * mad, abs=1e-12)
    expected = tuple(pid for pid, r in zip(picks["id"], res, strict=True)
                     if abs(r) > out.madK * mad)
    assert loc.dropped_pick_ids == expected and picks.loc[bad, "id"] in expected


def test_depth_on_edge_fires_at_the_volume_top(loc02: Any, locator: Locator, t0: float) -> None:
    top = locator.volume.top_elev_m
    loc = locator.locate(loc02.exact_picks(locator, 200.0, -400.0, top, t0))
    assert loc.search["atVolumeTop"] and loc.search["mapOnVolumeTop"]
    assert loc.pdf.top_face_mass > 0.05
    assert loc.depth_on_edge and loc.quality()["depthOnEdge"]
    deep = locator.locate(loc02.exact_picks(locator, 200.0, -400.0, -2500.0, t0))
    assert not deep.depth_on_edge and not deep.search["atVolumeTop"]
    assert deep.pdf.top_face_mass < 1e-4  # an interior face at the pdfCutoff contour


def test_broad_pdf_boundary_diagnostic_does_not_override_face_mass_flag(
    loc02: Any, locator: Locator, t0: float
) -> None:
    # Noisy, P-only, low-probability picks on 6 stations from a source 300 m above the volume top:
    # the MAP collapses onto the volume top with less than 5% of the mass on that face row.
    top = locator.volume.top_elev_m
    six = [s[0] for s in loc02.STATIONS[:6]]
    for seed in range(3):
        picks = loc02.exact_picks(locator, 200.0, -400.0, top + 300.0, t0, p_stations=six,
                                  s_stations=[], prob=0.15)
        loc = locator.locate(_noisy(picks, 300 + seed, 0.02, 0.04))
        assert loc.search["mapOnVolumeTop"] and loc.elev_m == top
        assert loc.search["faceMass"]["top"] < locator.cfg.depthOnEdgeMassFraction
        assert not loc.depth_on_edge


def test_pdf_region_grows_past_the_first_fine_box_and_matches_brute_force(
    loc02: Any, loc_setup: LocatorSetup, locator: Locator, t0: float
) -> None:
    # A first fine box of +/- 100 m cannot hold this PDF: the region must grow until it covers it.
    small = _with_locator_cfg(loc_setup, locator, fineHalfWidthM=100.0)
    picks = _noisy(loc02.exact_picks(small, 300.0, 200.0, -2500.0, t0), 5, 0.02, 0.04)
    loc = small.locate(picks)
    s = loc.search
    assert not loc.pdf_truncated and s["nStageGrow"] >= 1
    grown = [s["evaluatedEM"][0] < s["firstFineBoxEM"][0], s["evaluatedEM"][1] > s["firstFineBoxEM"][1],
             s["evaluatedElevM"][0] < s["firstFineBoxElevM"][0]]
    assert all(grown)
    # Brute force: the misfit over a +/- 700 m box around the MAP with every fine node evaluated.
    vol = small.volume
    p = small._prepare(picks, None)
    used = loc.arrivals["usedInLocation"].to_numpy()
    iz = np.arange(round((loc.elev_m - vol.bottom_elev_m) / FINE) - 28,
                   round((loc.elev_m - vol.bottom_elev_m) / FINE) + 29)
    i_n = np.arange(round((loc.n_m - vol.n_min_m) / FINE) - 28,
                    round((loc.n_m - vol.n_min_m) / FINE) + 29)
    ie = np.arange(round((loc.e_m - vol.e_min_m) / FINE) - 28,
                   round((loc.e_m - vol.e_min_m) / FINE) + 29)
    misfit, _ = small._box_misfit(p, used, ie, i_n, iz)
    brute = _summary(misfit, vol.e_at(ie), vol.n_at(i_n), vol.z_at(iz))
    assert max(brute.face_mass.values()) < 1e-6  # the brute-force box holds the whole PDF
    assert loc.h_err_m == pytest.approx(brute.h_err_m, rel=5e-3)
    assert loc.v_err_m == pytest.approx(brute.v_err_m, rel=5e-3)
    assert (loc.pdf.mean_e_m, loc.pdf.mean_n_m, loc.pdf.mean_elev_m) == pytest.approx(
        (brute.mean_e_m, brute.mean_n_m, brute.mean_elev_m), abs=1.0)
    assert loc.misfit == pytest.approx(float(misfit.min()), rel=1e-12)  # the same MAP misfit


def test_weak_event_pdf_is_truncated_not_capped(loc02: Any, locator: Locator, t0: float) -> None:
    # 5 distant stations, P only, prob 0.3: the PDF needs more than maxPdfNodes, so the formal
    # errors are None (docs/02 allows it) instead of a box-limited number.
    far5 = ["T.S05", "T.S06", "T.S07", "T.S08", "T.S09"]
    picks = loc02.exact_picks(locator, 200.0, -400.0, 1000.0, t0, p_stations=far5, s_stations=[],
                              prob=0.3)
    loc = locator.locate(_noisy(picks, 0, 0.02, 0.04))
    assert loc.pdf_truncated and loc.search["pdfTruncated"] and loc.search["nodeBudgetHit"]
    assert loc.search["truncatedFaces"] and loc.h_err_m is None and loc.v_err_m is None
    quality = LocationQuality.model_validate(loc.quality())
    assert quality.hErrM is None and quality.vErrM is None


def test_node_budget_truncates_and_lateral_volume_faces_count(
    loc02: Any, loc_setup: LocatorSetup, locator: Locator, t0: float
) -> None:
    picks = _noisy(loc02.exact_picks(locator, 0.0, 0.0, -2000.0, t0), 3, 0.02, 0.04)
    tight = _with_locator_cfg(loc_setup, locator, fineHalfWidthM=100.0, maxPdfNodes=2000)
    loc = tight.locate(picks)
    assert loc.pdf_truncated and loc.search["nodeBudgetHit"] and loc.h_err_m is None
    assert loc.search["nFine"] <= 2000
    assert abs(loc.elev_m + 2000.0) <= 100.0  # still located, on the fine lattice
    with pytest.raises(ValueError, match="first fine box"):
        _with_locator_cfg(loc_setup, locator, maxPdfNodes=2000)
    full = locator.locate(picks)
    assert not full.pdf_truncated and full.search["truncatedFaces"] == []
    # An event on the east face of the volume: its PDF is cut by a lateral volume face.
    east = locator.volume.e_max_m
    edge = locator.locate(loc02.exact_picks(locator, east, 0.0, -2000.0, t0))
    assert "east" in edge.search["volumeFaces"] and "east" in edge.search["truncatedFaces"]
    assert edge.pdf_truncated and edge.h_err_m is None and not edge.depth_on_edge


def test_statics_are_additive_and_validated(loc02: Any, locator: Locator, t0: float) -> None:
    picks = loc02.exact_picks(locator, 300.0, 300.0, -2200.0, t0)
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
    # An empty or all-zero statics mapping applies nothing, so LocationQuality.statics is false.
    assert not locator.locate(picks, statics={}).quality()["statics"]
    assert not locator.locate(picks, statics={("T.S02", "P"): 0.0}).statics_applied
    for bad in ({("T.S02", "p"): 0.1}, {("UU.NOPE", "P"): 0.1}, {"T.S02": 0.1}):
        with pytest.raises(ValueError, match="statics keys"):
            locator.locate(picks, statics=bad)  # type: ignore[arg-type]


def test_quality_fields_and_arrivals(loc02: Any, locator: Locator, t0: float) -> None:
    s_on = {"T.S01", "T.S02", "T.B01"}
    loc = locator.locate(loc02.exact_picks(locator, 0.0, 0.0, -2000.0, t0, s_stations=s_on))
    q = loc.quality()
    assert set(q) == QUALITY_FIELDS and q["method"] == "grid1d" and q["statics"] is False
    LocationQuality.model_validate(q)  # the landed docs/02 contract model
    assert (q["nStations"], q["nP"], q["nS"]) == (len(loc02.STATIONS), len(loc02.STATIONS), 3)
    se = np.array([s[1] for s in loc02.STATIONS]) - loc.e_m
    sn = np.array([s[2] for s in loc02.STATIONS]) - loc.n_m
    assert q["minEpiDistM"] == pytest.approx(float(np.hypot(se, sn).min()))
    assert q["gapDeg"] == pytest.approx(azimuthal_gap_deg(se, sn))
    arr = loc.arrivals
    assert DOCS02_ARRIVAL_COLUMNS <= set(arr.columns) and tuple(arr.columns) == ARRIVAL_COLUMNS
    np.testing.assert_allclose(arr["tPred"], loc.t0 + arr["travelTimeS"] + arr["staticS"])
    np.testing.assert_allclose(arr["residualS"], arr["tObs"] - arr["tPred"])
    np.testing.assert_allclose(arr["weight"], 1.0 / arr["sigmaS"])
    assert loc.u_m == pytest.approx(loc.elev_m - locator.origin_elev_m)
    record = locator.to_record()
    assert set(record) >= {"method", "config", "volume", "misfit", "uncertainty", "tables"}
    ext = record["tables"]["velocityModel"]["topExtension"]  # the extension reaches run.json
    assert ext is not None and ext["toElevM"] == locator.tables.grid.top_elev_m
    assert locator.velocity_model_record() == record["tables"]["velocityModel"]


def test_pick_probability_and_profile_sigma_set_the_weights(
    loc02: Any, loc_setup: LocatorSetup, locator: Locator, t0: float
) -> None:
    override = {"borehole": PhaseSigma(P=0.03, S=0.07)}
    prof = _with_locator_cfg(loc_setup, locator, profilePickSigmaS=override)
    assert prof.pick_sigma("T.B02", "S") == 0.07 and prof.pick_sigma("T.S02", "S") == 0.04
    picks = _noisy(loc02.exact_picks(prof, 100.0, -300.0, -2300.0, t0), 9, 0.02, 0.04)
    picks["prob"] = np.random.default_rng(4).uniform(0.2, 1.0, len(picks))
    loc = prof.locate(picks)
    arr = loc.arrivals.set_index("pickId").loc[picks["id"]]
    borehole = arr["stationId"].str.startswith("T.B")
    sigma = np.where(borehole, np.where(arr["phase"] == "P", 0.03, 0.07),
                     np.where(arr["phase"] == "P", 0.02, 0.04))
    np.testing.assert_allclose(arr["sigmaS"], sigma)
    np.testing.assert_allclose(arr["weight"], picks["prob"].to_numpy() / sigma)
    # The search minimised exactly sum(weight * |residual|) over the picks it used.
    used = arr["usedInLocation"].to_numpy()
    # (rel 1e-4: residuals are rebuilt from epoch times, good to ~1e-7 s)
    assert loc.misfit == pytest.approx(
        float((arr["weight"] * arr["residualS"].abs())[used].sum()), rel=1e-4)


def test_bad_picks_raise(loc02: Any, locator: Locator, t0: float) -> None:
    picks = loc02.exact_picks(locator, 0.0, 0.0, -2000.0, t0)
    with pytest.raises(ValueError, match="at least"):
        locator.locate(picks.iloc[:3])
    with pytest.raises(ValueError, match="more than one pick"):
        locator.locate(pd.concat([picks, picks.iloc[:1]]))
    with pytest.raises(ValueError, match="without coordinates"):
        locator.locate(picks.assign(stationId="X.NOPE"))
    with pytest.raises(ValueError, match="probabilities"):
        locator.locate(picks.assign(prob=0.0))


def test_outlier_pass_keeps_min_picks(
    loc02: Any, loc_setup: LocatorSetup, locator: Locator, t0: float
) -> None:
    picks = loc02.exact_picks(locator, 0.0, 0.0, -2000.0, t0)
    keep = ["T.S01", "T.S02", "T.S03", "T.S04", "T.B01"]
    picks = picks[picks["stationId"].isin(keep)].reset_index(drop=True)
    picks.loc[0, "t"] += 2.0  # T.S01 P
    loc = locator.locate(picks)  # 10 picks: the outlier goes, 9 >= minPicks remain
    assert loc.dropped_pick_ids == (picks.loc[0, "id"],)
    strict = _with_locator_cfg(loc_setup, locator, minPicks=len(picks))
    kept = strict.locate(picks)  # dropping one would leave fewer than minPicks
    assert kept.dropped_pick_ids == () and not kept.relocated
    assert kept.outlier_note is not None and "minPicks" in kept.outlier_note
    assert kept.arrivals["usedInLocation"].all()


def test_station_out_of_table_reach_raises(
    loc02: Any, loc_setup: LocatorSetup, locator: Locator, loc_run: RunSection
) -> None:
    far = loc02.stations(loc_run.origin.elevM)
    far.loc[0, "enu_e"] = 9500.0  # 15.3 km from the far corner of the volume > rMaxM 15 km
    with pytest.raises(ValueError, match="rMaxM"):
        Locator(far, locator.tables, loc_setup.config.locator,
                origin_elev_m=loc_run.origin.elevM,
                ref_surface_elev_m=loc_run.refSurfaceElevM)
    bad_u = loc02.stations(loc_run.origin.elevM)
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


def test_datums_origin_and_reference_surface_are_not_mixed(
    loc02: Any, loc_run: RunSection, t0: float
) -> None:
    # origin.elevM (the ENU u datum) and refSurfaceElevM (the volume top) differ here.
    run = loc_run.model_copy(update={
        "origin": loc_run.origin.model_copy(update={"elevM": 1500.0}), "refSurfaceElevM": 1700.0})
    loc_b = build_locator(loc02.setup(run=run))
    assert loc_b.volume.top_elev_m == 1700.0 and loc_b.origin_elev_m == 1500.0
    loc = loc_b.locate(loc02.exact_picks(loc_b, 500.0, -500.0, -2000.0, t0))
    assert abs(loc.elev_m + 2000.0) <= 2 * FINE
    assert loc.u_m == pytest.approx(loc.elev_m - 1500.0)
    top = loc_b.locate(loc02.exact_picks(loc_b, 500.0, -500.0, 1700.0, t0))
    assert top.elev_m == 1700.0 and top.search["mapOnVolumeTop"] and top.depth_on_edge


def test_locate_many_matches_serial_with_workers(
    loc02: Any, loc_setup: LocatorSetup, locator: Locator, t0: float
) -> None:
    events = [loc02.exact_picks(locator, e, n, z, t0) for e, n, z in
              ((0.0, 0.0, -1500.0), (800.0, 400.0, -3300.0), (-600.0, -900.0, -2500.0))]
    serial = locate_many(loc_setup, events, locator=locator)
    cfg = loc_setup.config
    two = cfg.model_copy(update={"locator": cfg.locator.model_copy(update={"nWorkers": 2})})
    parallel = locate_many(LocatorSetup(loc_setup.stations, loc_setup.model, two, loc_setup.run,
                                        loc_setup.cache_dir), events)
    for a, b in zip(serial, parallel, strict=True):
        assert (a.e_m, a.n_m, a.elev_m, a.t0, a.h_err_m, a.v_err_m) == (
            b.e_m, b.n_m, b.elev_m, b.t0, b.h_err_m, b.v_err_m)
    # A locator built from another config would be ignored by the workers: refuse it.
    other = _with_locator_cfg(loc_setup, locator, minPicks=5)
    with pytest.raises(ValueError, match="not built from setup"):
        locate_many(loc_setup, events, locator=other)

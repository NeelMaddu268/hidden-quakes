"""VAL-01 acceptance for the Gutenberg-Richter numerics: Aki-Utsu b with Shi-Bolt sigma on a
seeded synthetic sample of known b, Mc by maximum curvature on a hand-made histogram, the
cumulative curve, and every None path of ``gr_curve``."""

import logging
import math

import numpy as np
import pytest
from hq_contracts import models as m

from hq.config.validate import GRConfig
from hq.validate.gr import (
    LOG10_E,
    above_mc,
    b_aki_utsu,
    cumulative_counts,
    decimals_of,
    gr_allowed,
    gr_curve,
    mag_bins,
    mc_max_curvature,
    sigma_shi_bolt,
)

pytestmark = pytest.mark.smoke

BIN = 0.1
TRUE_B = 1.0
TRUE_MC = 1.0
N_SAMPLE = 4000
SEED = 5


def gr_sample(
    n: int = N_SAMPLE, b: float = TRUE_B, mc: float = TRUE_MC, seed: int = SEED
) -> np.ndarray:
    """Magnitudes from the G-R law with slope ``b`` (exponential with scale ``log10(e) / b``)
    starting at the lower edge of the bin centred on ``mc``, rounded to the bin width as a
    catalog would report them, so the ``mc`` bin is the first complete one."""
    rng = np.random.default_rng(seed)
    return np.round(mc - BIN / 2 + rng.exponential(LOG10_E / b, size=n), decimals_of(BIN))


def cfg(**overrides: object) -> GRConfig:
    return GRConfig(**{"magBinWidth": BIN, "mcOffsetMag": 0.2, "minEvents": 30, **overrides})  # type: ignore[arg-type]


def test_b_value_recovers_a_known_slope_with_a_sane_sigma() -> None:
    mags = gr_sample()
    mc = mc_max_curvature(mags, BIN)
    assert mc == TRUE_MC  # the fullest bin is the first one of a complete exponential sample
    mc_used = mc + 0.2
    b = b_aki_utsu(mags, mc_used, BIN)
    above = above_mc(mags, mc_used, BIN)
    sigma = sigma_shi_bolt(above, b)
    assert abs(b - TRUE_B) < 0.06
    assert 0.0 < sigma < 0.05  # roughly b / sqrt(n) for n in the thousands
    assert abs(b - TRUE_B) < 3.0 * sigma + 0.02
    # Same seed, same numbers; a steeper law gives a larger b.
    assert b == b_aki_utsu(gr_sample(), mc_used, BIN)
    assert b_aki_utsu(gr_sample(b=1.5), mc_used, BIN) > b


def test_max_curvature_on_a_hand_made_histogram() -> None:
    mags = np.array([0.8, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.2, 2.0])
    assert mc_max_curvature(mags, BIN) == 1.0
    assert mc_max_curvature(np.array([1.3, 1.3, 0.7, 0.7, 0.9]), BIN) == 0.7  # tie -> lowest
    assert mc_max_curvature(np.array([1.24, 1.26, 1.31, 1.34]), BIN) == 1.3  # nearest grid value
    assert mc_max_curvature(np.array([np.nan, 0.5, np.nan]), BIN) == 0.5  # nulls are dropped
    with pytest.raises(ValueError, match="at least one"):
        mc_max_curvature(np.array([]), BIN)
    assert (decimals_of(0.1), decimals_of(0.05), decimals_of(1.0), decimals_of(0.25)) == (
        1,
        2,
        0,
        2,
    )


def test_bins_and_cumulative_counts() -> None:
    mags = np.array([0.5, 0.7, 0.7, 1.2, np.nan])
    bins = mag_bins(mags, BIN)
    assert bins.tolist() == [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # exact decimals, no drift
    assert cumulative_counts(mags, bins) == [4, 3, 3, 1, 1, 1, 1, 1]
    assert above_mc(mags, 0.7, BIN).tolist() == [0.7, 0.7, 1.2]  # the Mc bin and up
    with pytest.raises(ValueError, match="no magnitude at or above"):
        b_aki_utsu(mags, 5.0, BIN)
    with pytest.raises(ValueError, match="at least two"):
        sigma_shi_bolt(np.array([1.0]), 1.0)


def test_gr_curve_fills_every_field_from_the_two_sets() -> None:
    recovered = gr_sample(n=400, mc=0.2)
    public = gr_sample(n=60, mc=1.0, seed=SEED + 1)
    curve = gr_curve(public, recovered, cfg())
    assert curve is not None
    m.GRCurve.model_validate_json(curve.model_dump_json())
    assert curve.magBins[0] == 0.2 and curve.magBins[-1] == max(recovered.max(), public.max())
    assert len(curve.magBins) == len(curve.publicCum) == len(curve.recoveredCum)
    assert curve.recoveredCum[0] == len(recovered) and curve.publicCum[0] == len(public)
    assert curve.recoveredCum == sorted(curve.recoveredCum, reverse=True)  # cumulative, so monotone
    assert curve.mcRecovered == pytest.approx(0.2 + 0.2)
    assert curve.mcPublic == pytest.approx(1.0 + 0.2)
    assert curve.bValue is not None and abs(curve.bValue - TRUE_B) < 0.2
    assert curve.bSigma is not None and 0.0 < curve.bSigma < 0.2
    # The public curve's counts are what a reader would compute from the catalog magnitudes.
    assert curve.publicCum == [int((public >= edge).sum()) for edge in curve.magBins]


def test_gr_curve_none_paths_and_the_magnitude_kill_switch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="hq.validate.gr")
    few = cfg(minEvents=30)
    assert gr_curve(np.array([]), np.array([]), few) is None
    assert gr_curve(gr_sample(n=10), gr_sample(n=19), few) is None  # 29 < 30 together
    assert any("fewer than minEvents" in r.getMessage() for r in caplog.records)
    # Enough public magnitudes, too few candidate ones: the curve exists, its recovered
    # statistics are null.
    caplog.clear()
    curve = gr_curve(gr_sample(n=40), gr_sample(n=10), few)
    assert curve is not None
    assert curve.mcPublic is not None
    assert (curve.mcRecovered, curve.bValue, curve.bSigma) == (None, None, None)
    # Enough candidates for Mc but too few above it: b and sigma are null, Mc is not.
    caplog.clear()
    recovered = np.concatenate([np.full(25, 0.5), np.full(10, 1.5)])  # Mc = 0.5 + 0.2; 10 above
    curve = gr_curve(np.array([]), recovered, cfg(minEvents=20))
    assert curve is not None and curve.mcRecovered == pytest.approx(0.7)
    assert (curve.bValue, curve.bSigma) == (None, None)
    assert any("b is null" in r.getMessage() for r in caplog.records)
    # The kill switch: looMae above the knob blocks G-R; no calibration or a good one does not.
    caplog.clear()
    limit = cfg(maxLooMae=0.4)
    good = m.MagCalibration(n=10, looMae=0.3, coefficients={"a": 1.0})
    bad = m.MagCalibration(n=10, looMae=0.5, coefficients={"a": 1.0})
    assert gr_allowed(None, limit) and gr_allowed(good, limit)
    assert not gr_allowed(bad, limit)
    assert any("kill switch" in r.getMessage() for r in caplog.records)
    assert math.isfinite(LOG10_E)

"""Gutenberg-Richter curve (VAL-01): pure functions on magnitude arrays.

Method (docs/lanes/H4): Mc by maximum curvature plus an offset, Aki-Utsu b with the Shi-Bolt
standard error. With ``w`` the bin width, magnitudes are binned on the grid of multiples of
``w`` (a magnitude goes to the nearest grid value; an exact half-way value goes to the even
multiple, numpy's rounding):

- ``mc_max_curvature``: the grid value whose bin holds the most magnitudes (the lowest one when
  several tie); the caller adds ``mcOffsetMag``.
- "above Mc" means ``m >= Mc - w / 2``: the whole bin centred on Mc and everything above it.
- ``b_aki_utsu``: ``log10(e) / (mean(m) - (Mc - w / 2))`` over the magnitudes above Mc.
- ``sigma_shi_bolt``: ``2.3 b^2 sqrt(sum((m_i - mean)^2) / (n (n - 1)))`` over the same set.
- ``cumulative_counts``: for each bin edge, how many magnitudes are ``>=`` it.

``gr_curve`` puts them together as a ``GRCurve``: ``magBins`` runs from the lowest to the highest
magnitude of both sets on the grid; ``publicCum`` counts the public regional catalog's
magnitudes and ``recoveredCum`` the candidate events'; ``mcPublic`` / ``mcRecovered`` and the
recovered set's ``bValue`` / ``bSigma`` are null when fewer than ``minEvents`` magnitudes
support them. Fewer than ``minEvents`` magnitudes in both sets together give no curve at all.
"""

import logging
import math
from decimal import Decimal

import numpy as np
from hq_contracts.models import GRCurve, MagCalibration

from hq.config.validate import GRConfig

log = logging.getLogger(__name__)

LOG10_E = math.log10(math.e)
SHI_BOLT_FACTOR = 2.3  # the constant of Shi & Bolt (1982): sigma_b = 2.3 b^2 sqrt(var / (n-1))
FloatArray = np.ndarray


def as_magnitudes(values: object) -> FloatArray:
    """A float64 array with every NaN and infinity dropped (a null magnitude is "no
    magnitude", never a number)."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def decimals_of(bin_width: float) -> int:
    """Decimal places of the bin width as written (0.1 -> 1, 0.05 -> 2, 1 -> 0), so bin edges
    print exactly and compare exactly with magnitudes written at that precision."""
    return max(0, -Decimal(str(bin_width)).normalize().as_tuple().exponent)


def grid_index(mags: FloatArray, bin_width: float) -> np.ndarray:
    """The grid multiple nearest each magnitude (``numpy.rint``: half-way values go to the even
    multiple)."""
    return np.rint(mags / bin_width).astype(np.int64)


def grid_value(index: int | np.ndarray, bin_width: float) -> np.ndarray:
    """Grid multiples back to magnitudes, at the bin width's own precision."""
    return np.round(np.asarray(index, dtype=np.float64) * bin_width, decimals_of(bin_width))


def mc_max_curvature(mags: FloatArray, bin_width: float) -> float:
    """The magnitude bin (grid value) with the most magnitudes; the lowest on a tie."""
    mags = as_magnitudes(mags)
    if len(mags) == 0:
        raise ValueError("mc_max_curvature needs at least one magnitude")
    index = grid_index(mags, bin_width)
    values, counts = np.unique(index, return_counts=True)  # values ascending
    return float(grid_value(int(values[np.argmax(counts)]), bin_width))


def above_mc(mags: FloatArray, mc: float, bin_width: float) -> FloatArray:
    """The magnitudes ``>= mc - bin_width / 2`` (the bin centred on Mc and up)."""
    mags = as_magnitudes(mags)
    return mags[mags >= mc - bin_width / 2.0]


def b_aki_utsu(mags: FloatArray, mc: float, bin_width: float) -> float:
    """Aki-Utsu maximum-likelihood b over the magnitudes above ``mc``."""
    above = above_mc(mags, mc, bin_width)
    if len(above) == 0:
        raise ValueError(f"no magnitude at or above Mc {mc} - {bin_width} / 2")
    denominator = float(above.mean()) - (mc - bin_width / 2.0)
    if denominator <= 0.0:
        raise ValueError(f"mean magnitude above Mc is not above Mc - w/2 ({denominator})")
    return LOG10_E / denominator


def sigma_shi_bolt(mags: FloatArray, b: float) -> float:
    """Shi-Bolt standard error of ``b`` from the magnitudes it was estimated on (two at least)."""
    mags = as_magnitudes(mags)
    n = len(mags)
    if n < 2:
        raise ValueError("sigma_shi_bolt needs at least two magnitudes")
    spread = float(np.sum((mags - mags.mean()) ** 2))
    return SHI_BOLT_FACTOR * b * b * math.sqrt(spread / (n * (n - 1)))


def cumulative_counts(mags: FloatArray, bins: FloatArray) -> list[int]:
    """For each bin edge, the number of magnitudes ``>=`` it."""
    mags = as_magnitudes(mags)
    return [int((mags >= edge).sum()) for edge in bins]


def mag_bins(mags: FloatArray, bin_width: float) -> FloatArray:
    """Grid values from the lowest to the highest magnitude, inclusive, ``bin_width`` apart."""
    mags = as_magnitudes(mags)
    if len(mags) == 0:
        raise ValueError("mag_bins needs at least one magnitude")
    lo, hi = int(np.floor(mags.min() / bin_width)), int(np.ceil(mags.max() / bin_width))
    return grid_value(np.arange(lo, hi + 1), bin_width)


def completeness(mags: FloatArray, cfg: GRConfig, what: str) -> float | None:
    """MaxC + offset, or ``None`` (logged) below ``minEvents`` magnitudes."""
    mags = as_magnitudes(mags)
    if len(mags) < cfg.minEvents:
        log.warning(
            "G-R: %d %s magnitudes, fewer than minEvents %d; Mc is null", len(mags), what,
            cfg.minEvents,
        )  # fmt: skip
        return None
    mc = mc_max_curvature(mags, cfg.magBinWidth) + cfg.mcOffsetMag
    return float(np.round(mc, decimals_of(cfg.magBinWidth)))


def gr_allowed(calibration: MagCalibration | None, cfg: GRConfig) -> bool:
    """docs/03 magnitude kill switch: G-R is built unless H2's calibration reports a
    leave-one-out MAE above ``maxLooMae``. No calibration file means nothing blocks it."""
    if calibration is None:
        return True
    if calibration.looMae > cfg.maxLooMae:
        log.warning(
            "G-R skipped (docs/03 magnitude kill switch): MagCalibration.looMae %.3f exceeds "
            "maxLooMae %.3f (n=%d); the calibration is still embedded",
            calibration.looMae,
            cfg.maxLooMae,
            calibration.n,
        )
        return False
    log.info(
        "G-R: MagCalibration.looMae %.3f within maxLooMae %.3f (n=%d)",
        calibration.looMae,
        cfg.maxLooMae,
        calibration.n,
    )
    return True


def gr_curve(public_mags: FloatArray, recovered_mags: FloatArray, cfg: GRConfig) -> GRCurve | None:
    """The ``GRCurve`` of the public regional catalog's magnitudes and the candidate events',
    or ``None`` (logged) when both sets together hold fewer than ``minEvents`` magnitudes."""
    public = as_magnitudes(public_mags)
    recovered = as_magnitudes(recovered_mags)
    total = len(public) + len(recovered)
    if total < cfg.minEvents:
        log.warning(
            "G-R: %d public + %d candidate magnitudes, fewer than minEvents %d; no curve",
            len(public),
            len(recovered),
            cfg.minEvents,
        )
        return None
    bins = mag_bins(np.concatenate([public, recovered]), cfg.magBinWidth)
    mc_public = completeness(public, cfg, "public")
    mc_recovered = completeness(recovered, cfg, "candidate")
    b_value: float | None = None
    b_sigma: float | None = None
    if mc_recovered is not None:
        above = above_mc(recovered, mc_recovered, cfg.magBinWidth)
        if len(above) < cfg.minEvents:
            log.warning(
                "G-R: %d candidate magnitudes at or above Mc %.2f, fewer than minEvents %d; "
                "b is null",
                len(above),
                mc_recovered,
                cfg.minEvents,
            )
        else:
            b_value = b_aki_utsu(recovered, mc_recovered, cfg.magBinWidth)
            b_sigma = sigma_shi_bolt(above, b_value)
    curve = GRCurve(
        magBins=[float(v) for v in bins],
        publicCum=cumulative_counts(public, bins),
        recoveredCum=cumulative_counts(recovered, bins),
        mcPublic=mc_public,
        mcRecovered=mc_recovered,
        bValue=b_value,
        bSigma=b_sigma,
    )
    log.info(
        "G-R: %d bins from %.2f to %.2f (width %g): %d public magnitudes (Mc %s), %d candidate "
        "magnitudes (Mc %s, b %s +/- %s)",
        len(bins),
        bins[0],
        bins[-1],
        cfg.magBinWidth,
        len(public),
        _fmt(mc_public),
        len(recovered),
        _fmt(mc_recovered),
        _fmt(b_value),
        _fmt(b_sigma),
    )
    return curve


def _fmt(value: float | None) -> str:
    return "null" if value is None else f"{value:.3f}"

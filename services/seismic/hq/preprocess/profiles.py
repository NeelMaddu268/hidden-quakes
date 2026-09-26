"""Preprocessing profiles: raw counts in, model-ready traces at ``targetRateHz`` out.

Every profile works on a copy, splits masked gaps into separate traces and processes each
gap-separated segment on its own. Gaps are never merged or filled: PhaseNet fires on the step a
zero-filled gap leaves behind. Counts stay counts (PhaseNet normalizes internally).

Methods (configured per profile in ``signal.yaml`` -> ``preprocess.profiles``):

- ``passthrough`` (``surface-100``): detrend, taper. Input must already be at the target rate.
- ``decimate`` (``surface-hi``, ``borehole-A``): detrend, taper, zero-phase Butterworth lowpass,
  then an explicit zero-phase Chebyshev II anti-alias lowpass whose stopband starts at the output
  Nyquist, then integer sample picking. A 4-corner Butterworth at 40 Hz run forward and backward
  is still only about -17 dB at 50 Hz and needs about 71 Hz to reach -40 dB, so without the second
  stage content just above the new Nyquist would fold into the 40-50 Hz band. Rates that are a
  rational multiple of the target (250 Hz = 100 Hz x 5/2) are zero-stuffed by the small factor
  first; the same anti-alias filter removes the images.
- ``stretch`` (``borehole-B``): detrend, taper, zero-phase Butterworth bandpass, then relabel the
  sample rate as the target rate without resampling, so the model sees a slowed-down waveform.
  ``TimeMap.to_real`` turns model-time picks back into real time.
"""

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from typing import overload

import numpy as np
import numpy.typing as npt
from obspy import Stream, Trace, UTCDateTime
from scipy import signal as sps

from hq.config.signal import (
    DecimateProfile,
    PassthroughProfile,
    PreprocessConfig,
    SignalConfig,
    StretchProfile,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
Profile = PassthroughProfile | DecimateProfile | StretchProfile

# Evidence display copies only. ``display_copy`` has a fixed docs/02 signature with no config, so
# the shape of its filter and taper lives here. These values never touch anything that is picked,
# associated or located; they only change how a snippet looks in the evidence drawer.
DISPLAY_DETREND = "linear"
DISPLAY_TAPER_TYPE = "hann"
DISPLAY_TAPER_MAX_PERCENTAGE = 0.05
DISPLAY_TAPER_MAX_LENGTH_S = 1.0
DISPLAY_BANDPASS_CORNERS = 4


@dataclass(frozen=True)
class TimeMap:
    """Maps model time (what the picker saw) to real time and back, both epoch seconds UTC.

    ``to_real(t) = anchor + (t - anchor) / factor`` and ``to_model`` is its inverse.
    ``factor`` is 1.0 (identity) for every method except ``stretch``, where it is the input
    rate divided by the target rate. One map covers a whole stream, gaps included, so
    gap-separated segments keep their order and never overlap in model time.
    """

    anchor: float
    factor: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.anchor):
            raise ValueError(f"TimeMap anchor must be finite, got {self.anchor}")
        if not (math.isfinite(self.factor) and self.factor > 0.0):
            raise ValueError(f"TimeMap factor must be finite and positive, got {self.factor}")

    @property
    def is_identity(self) -> bool:
        return self.factor == 1.0

    @overload
    def to_real(self, t: float) -> float: ...

    @overload
    def to_real(self, t: FloatArray) -> FloatArray: ...

    def to_real(self, t: float | FloatArray) -> float | FloatArray:
        """Model time -> real time."""
        if isinstance(t, np.ndarray):
            arr = np.array(t, dtype=np.float64)
            return arr if self.is_identity else self.anchor + (arr - self.anchor) / self.factor
        value = float(t)
        return value if self.is_identity else self.anchor + (value - self.anchor) / self.factor

    @overload
    def to_model(self, t: float) -> float: ...

    @overload
    def to_model(self, t: FloatArray) -> FloatArray: ...

    def to_model(self, t: float | FloatArray) -> float | FloatArray:
        """Real time -> model time."""
        if isinstance(t, np.ndarray):
            arr = np.array(t, dtype=np.float64)
            return arr if self.is_identity else self.anchor + (arr - self.anchor) * self.factor
        value = float(t)
        return value if self.is_identity else self.anchor + (value - self.anchor) * self.factor


def for_picking(st: Stream, profile: str, cfg: SignalConfig) -> tuple[Stream, TimeMap]:
    """Turn raw counts into model input at exactly ``targetRateHz``, float64, components Z/N/E.

    The input stream is never modified. Gap-separated segments come out as separate traces;
    segments shorter than ``minSegmentS`` are dropped and counted. Horizontal components listed
    in ``componentRename`` (borehole ``1``/``2``) are renamed on the copy.
    """
    t_begin = time.perf_counter()
    pcfg = cfg.preprocess
    prof = _profile(profile, pcfg)
    if len(st) == 0:
        raise ValueError(f"for_picking({profile}) got an empty stream")

    ratios: dict[float, tuple[int, int]] = {}
    for tr in st:
        rate = float(tr.stats.sampling_rate)
        _check_rate_range(tr.id, rate, profile, prof, pcfg)
        ratios[rate] = _resample_ratio(tr.id, rate, pcfg)
        if isinstance(prof, PassthroughProfile) and ratios[rate] != (1, 1):
            raise ValueError(f"{tr.id}: {profile} does not resample, got {rate} Hz")

    anchor = min(tr.stats.starttime for tr in st).timestamp
    if isinstance(prof, StretchProfile):
        if len(ratios) != 1:
            raise ValueError(
                f"{profile} maps one stream with one TimeMap and needs a single input rate, "
                f"got {sorted(ratios)} Hz"
            )
        down, up = next(iter(ratios.values()))
        tmap = TimeMap(anchor=anchor, factor=down / up)
    else:
        tmap = TimeMap(anchor=anchor, factor=1.0)

    segments = _segments(st)
    overlaps = _count_overlaps(segments)
    if overlaps:
        logger.warning("for_picking(%s): %d overlapping segments in the input", profile, overlaps)

    out: list[Trace] = []
    dropped = 0
    for seg in segments:
        if seg.stats.npts * seg.stats.delta < pcfg.minSegmentS:
            dropped += 1
            continue
        _detrend_taper(
            seg, pcfg.detrend, pcfg.taper.type, pcfg.taper.maxPercentage, pcfg.taper.maxLengthS
        )
        if isinstance(prof, PassthroughProfile):
            out.append(_new_trace(seg, seg.data, pcfg.targetRateHz, seg.stats.starttime))
        elif isinstance(prof, DecimateProfile):
            down, up = ratios[float(seg.stats.sampling_rate)]
            out.append(_decimate(seg, prof, pcfg, down, up))
        else:
            out.append(_stretch(seg, prof, pcfg, tmap))

    renamed = _rename_components(out, pcfg)
    runtime_s = time.perf_counter() - t_begin
    logger.info(
        "for_picking profile=%s traces_in=%d segments=%d traces_out=%d dropped_short=%d "
        "renamed=%d factor=%g runtime_s=%.3f",
        profile,
        len(st),
        len(segments),
        len(out),
        dropped,
        renamed,
        tmap.factor,
        runtime_s,
    )
    if dropped:
        logger.info(
            "for_picking(%s): dropped %d segments shorter than minSegmentS=%g s",
            profile,
            dropped,
            pcfg.minSegmentS,
        )
    if not out:
        logger.warning(
            "for_picking(%s): nothing left to pick after dropping short segments", profile
        )
    return Stream(traces=out), tmap


def display_copy(st: Stream, band_hz: tuple[float, float]) -> Stream:
    """Detrend, taper and zero-phase Butterworth bandpass a copy, for evidence snippets only.

    Gaps stay gaps (separate traces, never filled). Segments too short for the zero-phase filter's
    edge padding are dropped and counted in the log. Filter shape: ``DISPLAY_*`` constants.
    """
    t_begin = time.perf_counter()
    low, high = band_hz
    if not 0.0 < low < high:
        raise ValueError(f"display band must satisfy 0 < low < high, got {band_hz}")
    out: list[Trace] = []
    dropped = 0
    segments = _segments(st)
    for seg in segments:
        rate = float(seg.stats.sampling_rate)
        if high >= rate / 2.0:
            raise ValueError(f"{seg.id}: display band {band_hz} Hz reaches Nyquist of {rate} Hz")
        sos = sps.butter(
            DISPLAY_BANDPASS_CORNERS, [low, high], btype="bandpass", fs=rate, output="sos"
        )
        if seg.stats.npts <= _min_filter_samples(sos):
            dropped += 1
            continue
        _detrend_taper(
            seg,
            DISPLAY_DETREND,
            DISPLAY_TAPER_TYPE,
            DISPLAY_TAPER_MAX_PERCENTAGE,
            DISPLAY_TAPER_MAX_LENGTH_S,
        )
        seg.data = sps.sosfiltfilt(sos, seg.data)
        out.append(seg)
    level = logging.INFO if dropped else logging.DEBUG
    logger.log(
        level,
        "display_copy band=%s traces_in=%d segments=%d traces_out=%d dropped_short=%d "
        "runtime_s=%.3f",
        band_hz,
        len(st),
        len(segments),
        len(out),
        dropped,
        time.perf_counter() - t_begin,
    )
    return Stream(traces=out)


def design_antialias(fs_hz: float, cfg: PreprocessConfig) -> FloatArray:
    """Chebyshev II lowpass (SOS) at ``fs_hz`` with its stopband at the output Nyquist fraction.

    The order is the minimum meeting ``passbandLossDb`` at the passband edge and
    ``stopbandAttenuationDb`` from the stopband edge up, per pass.
    """
    aa = cfg.antiAlias
    nyquist_out = cfg.targetRateHz / 2.0
    passband_hz = aa.passbandEdgeFraction * nyquist_out
    stopband_hz = aa.stopbandEdgeFraction * nyquist_out
    if stopband_hz >= fs_hz / 2.0:
        raise ValueError(f"anti-alias stopband {stopband_hz} Hz is not below Nyquist of {fs_hz} Hz")
    order, natural_hz = sps.cheb2ord(
        passband_hz, stopband_hz, aa.passbandLossDb, aa.stopbandAttenuationDb, fs=fs_hz
    )
    logger.debug(
        "anti-alias at %g Hz: Chebyshev II order %d, natural %.3f Hz", fs_hz, order, natural_hz
    )
    sos: FloatArray = sps.cheby2(
        order, aa.stopbandAttenuationDb, natural_hz, btype="lowpass", fs=fs_hz, output="sos"
    )
    return sos


def _profile(name: str, pcfg: PreprocessConfig) -> Profile:
    if name not in pcfg.profiles:
        raise ValueError(f"unknown preprocess profile {name!r}; valid: {sorted(pcfg.profiles)}")
    return pcfg.profiles[name]


def _check_rate_range(
    trace_id: str, rate: float, name: str, prof: Profile, pcfg: PreprocessConfig
) -> None:
    tol = pcfg.rateRelTol
    if not prof.minRateHz * (1.0 - tol) <= rate <= prof.maxRateHz * (1.0 + tol):
        raise ValueError(
            f"{trace_id}: {rate} Hz is outside profile {name}'s range "
            f"[{prof.minRateHz}, {prof.maxRateHz}] Hz"
        )


def _resample_ratio(trace_id: str, rate: float, pcfg: PreprocessConfig) -> tuple[int, int]:
    """Return ``(down, up)`` with ``rate * up / down == targetRateHz`` within ``rateRelTol``."""
    exact = rate / pcfg.targetRateHz
    approx = Fraction(exact).limit_denominator(pcfg.maxUpsampleFactor)
    if (
        approx.numerator < approx.denominator
        or abs(float(approx) - exact) > pcfg.rateRelTol * exact
    ):
        raise ValueError(
            f"{trace_id}: {rate} Hz is not {pcfg.targetRateHz} Hz x M / L with integers M >= L "
            f"and L <= maxUpsampleFactor={pcfg.maxUpsampleFactor}"
        )
    return approx.numerator, approx.denominator


def _segments(st: Stream) -> list[Trace]:
    """Copy each trace and split masked gaps into separate contiguous traces (never filled)."""
    out: list[Trace] = []
    for tr in st:
        copy = tr.copy()
        if isinstance(copy.data, np.ma.MaskedArray):
            out.extend(copy.split())
        else:
            out.append(copy)
    return out


def _count_overlaps(segments: list[Trace]) -> int:
    by_id: dict[str, list[Trace]] = defaultdict(list)
    for seg in segments:
        by_id[seg.id].append(seg)
    overlaps = 0
    for group in by_id.values():
        group.sort(key=lambda tr: tr.stats.starttime)
        for prev, nxt in pairwise(group):
            if nxt.stats.starttime <= prev.stats.endtime:
                overlaps += 1
    return overlaps


def _detrend_taper(
    seg: Trace, detrend: str, taper_type: str, max_percentage: float, max_length_s: float
) -> None:
    """In place on a trace this module owns (a copy of the input)."""
    seg.data = np.asarray(seg.data, dtype=np.float64)
    seg.detrend(detrend)
    seg.taper(max_percentage=max_percentage, type=taper_type, max_length=max_length_s)


def _decimate(
    seg: Trace, prof: DecimateProfile, pcfg: PreprocessConfig, down: int, up: int
) -> Trace:
    rate = float(seg.stats.sampling_rate)
    lowpass = sps.butter(
        prof.lowpassCorners, prof.lowpassHz, btype="lowpass", fs=rate, output="sos"
    )
    data: FloatArray = sps.sosfiltfilt(lowpass, seg.data)
    if up > 1:
        # Zero-stuff to rate * up; the anti-alias lowpass below also removes the images. The last
        # sample stays the last real sample, so nothing is extrapolated past the segment end.
        stuffed = np.zeros((data.size - 1) * up + 1, dtype=np.float64)
        stuffed[::up] = data * up
        data = stuffed
    if down > 1:
        data = sps.sosfiltfilt(design_antialias(rate * up, pcfg), data)
        data = np.ascontiguousarray(data[::down])
    return _new_trace(seg, data, pcfg.targetRateHz, seg.stats.starttime)


def _stretch(seg: Trace, prof: StretchProfile, pcfg: PreprocessConfig, tmap: TimeMap) -> Trace:
    rate = float(seg.stats.sampling_rate)
    low, high = prof.bandpassHz
    if high >= rate / 2.0:
        raise ValueError(f"{seg.id}: bandpass {prof.bandpassHz} Hz reaches Nyquist of {rate} Hz")
    bandpass = sps.butter(
        prof.bandpassCorners, [low, high], btype="bandpass", fs=rate, output="sos"
    )
    data: FloatArray = sps.sosfiltfilt(bandpass, seg.data)
    start = UTCDateTime(tmap.to_model(seg.stats.starttime.timestamp))
    return _new_trace(seg, data, pcfg.targetRateHz, start)


def _new_trace(template: Trace, data: FloatArray, rate_hz: float, start: UTCDateTime) -> Trace:
    """Same ids and metadata as ``template``, new samples, rate and start time."""
    header = template.stats.copy()
    header.npts = len(data)  # Trace(header=...) keeps a header npts over len(data)
    header.sampling_rate = rate_hz
    header.starttime = start
    return Trace(data=np.asarray(data, dtype=np.float64), header=header)


def _rename_components(traces: list[Trace], pcfg: PreprocessConfig) -> int:
    """Rename component codes per ``componentRename``; every result must be a model component."""
    original_ids = {tr.id for tr in traces}
    renamed: dict[str, str] = {}
    count = 0
    for tr in traces:
        channel = tr.stats.channel
        target = pcfg.componentRename.get(channel[-1:])
        if target is None:
            continue
        old_id = tr.id
        tr.stats.channel = channel[:-1] + target
        if tr.id in original_ids:
            raise ValueError(f"renaming {old_id} to {tr.id} collides with an existing channel")
        renamed[old_id] = tr.id
        count += 1
    if count:
        logger.info(
            "renamed %d traces by componentRename (orientation of 1/2 horizontals is uncertain): %s",
            count,
            ", ".join(f"{src} -> {dst}" for src, dst in sorted(renamed.items())),
        )
    allowed = set(pcfg.modelComponents)
    bad = sorted({tr.id for tr in traces if tr.stats.channel[-1:] not in allowed})
    if bad:
        raise ValueError(f"components must be one of {pcfg.modelComponents!r}; got {bad}")
    return count


def _min_filter_samples(sos: FloatArray) -> int:
    """Default edge padding of ``scipy.signal.sosfiltfilt``; the input must be longer than this."""
    zeros = min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    return 3 * (2 * len(sos) + 1 - zeros)

"""Sonification: sped-up audio renderings of cached public waveforms (SEIS-09).

A clip is a sped-up rendering of recorded ground motion on one vertical channel, never "the sound
of" an event. Two clips, both configured in ``signal.yaml`` -> ``sonify``:

- ``hour``: the busiest epoch-aligned bin (UTC clock hour) of candidate events in the bundle's
  ``events.json`` (ties: earliest), at the station of ``stationKind`` with the most picks
  (``picks.parquet``, any phase) inside it (ties: station id order).
- ``hero``: ``preS`` + ``postS`` around the bundle's hero event (``meta.json`` ->
  ``scene.heroEventId``, origin time from ``events.json``), at the nearest station of
  ``stationKind`` (epicentral ENU distance, ties: id order) whose vertical channel covers at least
  ``minCoverageFraction`` of the window.

Rendering, per clip. The audio's samples are real-axis samples played ``speed`` times faster, so
the real-axis rate is ``r = audioRateHz / speed`` and the band must stay below ``r / 2``.

1. ``read_window`` (raw counts, ``padS`` beyond each window end), vertical channel only.
2. Each gap-separated segment on its own: detrend, taper, zero-phase Butterworth bandpass
   (``bandHz``, real Hz), then ``scipy.signal.resample_poly`` to ``r`` (rational up / down, its
   own anti-alias FIR). Segments too short for the bandpass's edge padding are dropped and counted.
3. Every segment is placed on a zero timeline of ``round(window * r)`` samples at its true time
   (nearest output sample). Gaps and missing data at the window ends are never touched, so they are
   exact digital silence: nothing is interpolated, bridged or zero-filled inside a segment.
4. Level: the ``levelPercentile`` of ``|x|`` over samples that hold data is the reference (0 dB).
5. Compression: a soft-knee downward compressor (``thresholdDb``, ``ratio``, ``kneeDb``) on a
   look-ahead envelope, a centred running maximum of ``|x|`` (``envelopeHoldMs``) smoothed by a
   centred Hann window (``envelopeSmoothMs``). The gain is smooth and starts to fall just before a
   loud onset, so onsets are not let through first and clamped later; small events come up
   relative to large ones and keep their shape.
6. A raised-cosine fade of ``edgeFadeMs`` at both edges of every placed segment (gaps and window
   ends do not click), then peak normalization to ``peakDbfs``.
7. Checks, raising on failure: no sample above the peak target, silence exactly zero, duration
   equal to ``window / speed`` within one audio sample.
8. Mono OGG (Vorbis) and MP3 (constant bitrate) through python-soundfile, which is not a project
   dependency: run with ``uv run --with soundfile``. The Ogg stream serial is rewritten to
   ``oggStreamSerial`` (libsndfile draws a random one) so re-encodes are byte-identical, the MP3
   frame headers are checked for the configured bitrate, both files are decoded back and checked
   for clipping, and each must stay under ``maxBytes``. A JSON manifest sits next to them.

CLI::

    uv run --with soundfile python -m hq.preprocess.sonify --run-dir <run> --bundle-dir <bundle>
        --config-dir configs/showcase --cache-dir <cache> --out-dir <dir> [--clip hour|hero|all]

It writes nothing but ``<fileStem>.{ogg,mp3,json}`` under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import struct
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from obspy import Stream
from scipy import ndimage
from scipy import signal as sps

from hq.config.signal import (
    MPEG1_L3_BITRATES_KBPS,
    SignalConfig,
    SonifyCompressor,
    SonifyConfig,
    SonifyRender,
    SonifyResample,
)
from hq.ingest.cache import CacheMissError, read_window, window_segments
from hq.preprocess.profiles import _detrend_taper, _min_filter_samples, _segments

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# Labels, not knobs: the attribution string is fixed by the ticket and H3's copy.
SOURCE = "EarthScope public waveforms"
GENERATOR = "hq.preprocess.sonify"
NOTE = (
    "A sped-up audio rendering of public seismic waveform data from one sensor's vertical "
    "channel. Gaps in the recording play as silence."
)
CLIP_HOUR = "hour"
CLIP_HERO = "hero"
CLIPS = (CLIP_HOUR, CLIP_HERO)
HOUR_RULE = (
    "Busiest epoch-aligned UTC bin of binS seconds by candidate events in the bundle's "
    "events.json (ties: earliest); within it, the station of stationKind used in the run with "
    "the most picks.parquet picks of any phase (ties: station id order); its vertical channel."
)
HERO_RULE = (
    "The bundle's hero event (meta.json scene.heroEventId); the nearest station of stationKind "
    "used in the run by epicentral distance (ties: station id order) whose vertical channel "
    "covers at least minCoverageFraction of the window; preS before and postS after the origin."
)
COMPRESSION_DESCRIPTION = (
    "Soft-knee downward compressor on a look-ahead envelope (centred running maximum of |x| "
    "over envelopeHoldMs, smoothed by a centred Hann window of envelopeSmoothMs, audio ms). "
    "Levels in dB relative to the levelPercentile of |x| over samples with data. Gain is "
    "(1/ratio - 1) x (level - thresholdDb) above the knee, quadratic inside it."
)

_DB = 20.0  # amplitude decibels: 20 log10
_MS_PER_S = 1000.0
# libsndfile maps soundfile's compression_level c onto an MPEG-1 constant bitrate as
# 320 - c * (320 - 32) kbps; the encoded frame headers are checked after writing.
_MP3_MAX_KBPS = MPEG1_L3_BITRATES_KBPS[-1]
_MP3_MIN_KBPS = MPEG1_L3_BITRATES_KBPS[0]
_MPEG1_L3_BITRATE_INDEX = (0, *MPEG1_L3_BITRATES_KBPS)  # frame header bitrate index -> kbps
_MPEG1_RATE_INDEX = (44100, 48000, 32000)  # frame header sample rate index -> Hz
_MPEG1_L3_SAMPLES_PER_FRAME = 1152
# Frames per libsndfile write call. Not a knob: it changes no output byte (tested), it only keeps
# each call small (see _encode).
WRITE_BLOCK_FRAMES = 4096


class SoundfileMissingError(RuntimeError):
    """python-soundfile (the OGG / MP3 encoder) is not importable."""


# --- selection --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HourSelection:
    startS: float
    endS: float
    nEvents: int
    stationId: str
    channel: str
    nPicks: int


@dataclass(frozen=True)
class HeroSelection:
    eventId: str
    originS: float
    startS: float
    endS: float
    stationId: str
    channel: str
    epiDistM: float
    coverageFraction: float


def busiest_bin(event_times: Sequence[float], bin_s: float) -> tuple[float, int]:
    """Start (epoch s) and event count of the busiest bin on multiples of ``bin_s``; ties: earliest."""
    if len(event_times) == 0:
        raise ValueError("no candidate events to choose a busiest bin from")
    bins = np.floor(np.asarray(event_times, dtype=np.float64) / bin_s).astype(np.int64)
    starts, counts = np.unique(bins, return_counts=True)  # ascending, so argmax picks the earliest
    k = int(np.argmax(counts))
    return float(starts[k]) * bin_s, int(counts[k])


def eligible_stations(stations: pd.DataFrame, cfg: SonifyConfig) -> pd.DataFrame:
    """Stations of ``stationKind`` (and ``usedInRun`` when configured), sorted by id."""
    mask = stations["kind"] == cfg.stationKind
    if cfg.usedInRunOnly:
        mask &= stations["usedInRun"].astype(bool)
    out = stations[mask].sort_values("id", kind="stable")
    if out.empty:
        raise ValueError(
            f"no {cfg.stationKind} stations{' used in the run' if cfg.usedInRunOnly else ''}"
        )
    return out


def vertical_channel(station_id: str, channels: Sequence[str], component: str) -> str:
    """The one channel of ``channels`` whose component letter is ``component``."""
    found = [c for c in channels if c[-1:] == component]
    if len(found) != 1:
        raise ValueError(f"{station_id}: expected one {component} channel in {list(channels)}")
    return found[0]


def most_picked_station(
    stations: pd.DataFrame, picks: pd.DataFrame, t0: float, t1: float, cfg: SonifyConfig
) -> tuple[str, int]:
    """Eligible station with the most picks (any phase) in ``[t0, t1)``; ties: station id order."""
    elig = eligible_stations(stations, cfg)
    inside = picks[(picks["t"] >= t0) & (picks["t"] < t1)]
    counts = inside.groupby("stationId").size()
    ranked = [(str(sid), int(counts.get(sid, 0))) for sid in elig["id"]]
    best = max(ranked, key=lambda item: item[1])  # max keeps the first of equals: id order
    if best[1] == 0:
        raise ValueError(f"no picks at any eligible station in [{t0}, {t1})")
    log.info("sonify: picks per eligible station in the bin: %s", dict(ranked))
    return best


def nearest_station_with_data(
    stations: pd.DataFrame,
    east_m: float,
    north_m: float,
    coverage: Callable[[str, str], float],
    min_coverage: float,
    cfg: SonifyConfig,
) -> tuple[str, str, float, float]:
    """(station id, channel, epicentral m, coverage) of the nearest eligible station with data.

    Ordered by epicentral ENU distance, ties by id. ``coverage(station_id, channel)`` returns the
    fraction of the window its vertical channel covers; the first at or above ``min_coverage`` wins.
    """
    elig = eligible_stations(stations, cfg)
    dist = np.hypot(elig["enu_e"].to_numpy() - east_m, elig["enu_n"].to_numpy() - north_m)
    order = sorted(zip(dist.tolist(), elig["id"].tolist(), elig["channels"].tolist(), strict=True))
    for dist_m, sid, channels in order:
        channel = vertical_channel(sid, list(channels), cfg.component)
        frac = coverage(sid, channel)
        log.info("sonify: %s %s at %.0f m covers %.3f of the window", sid, channel, dist_m, frac)
        if frac >= min_coverage:
            return sid, channel, dist_m, frac
    raise ValueError(f"no eligible station covers {min_coverage} of the window")


def coverage_fraction(
    station_id: str, channel: str, t0: float, t1: float, *, cache_dir: Path
) -> float:
    """Fraction of ``[t0, t1)`` covered by cached samples of one channel (headers only)."""
    try:
        segs = window_segments(station_id, t0, t1, cache_dir=cache_dir)
    except CacheMissError:
        return 0.0
    spans = sorted(
        (max(s.start, t0), min(s.end + s.delta, t1)) for s in segs if s.channel == channel
    )
    covered, reach = 0.0, t0
    for lo, hi in spans:  # union, so overlapping records never count twice
        lo = max(lo, reach)
        if hi > lo:
            covered += hi - lo
            reach = hi
    return covered / (t1 - t0)


# --- rendering (pure) -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Timeline:
    """Real-axis samples at ``realRateHz`` on ``[t0, t1)``; exactly zero where there is no data."""

    samples: FloatArray
    covered: BoolArray
    runs: tuple[tuple[int, int], ...]  # [lo, hi) sample ranges that hold data, ascending
    realRateHz: float
    sourceRateHz: float
    nSegments: int  # gap-separated segments read (after joining abutting pieces)
    nDropped: int  # too short for the bandpass's edge padding
    nOutside: int  # entirely in the pad, outside the window


@dataclass(frozen=True)
class Rendered:
    audio: FloatArray  # final samples, played at audioRateHz
    timeline: Timeline
    levelCounts: float  # robust level (counts after the bandpass) that maps to 0 dB


def resample_factors(source_hz: float, target_hz: float, cfg: SonifyResample) -> tuple[int, int]:
    """(up, down) with ``source_hz * up / down == target_hz`` within ``rateRelTol``."""
    exact = Fraction(target_hz) / Fraction(source_hz)
    approx = exact.limit_denominator(cfg.maxFactor)
    if max(approx.numerator, approx.denominator) > cfg.maxFactor or abs(
        float(approx) - float(exact)
    ) > cfg.rateRelTol * float(exact):
        raise ValueError(
            f"{source_hz} Hz -> {target_hz} Hz needs up / down factors above {cfg.maxFactor}"
        )
    return approx.numerator, approx.denominator


def n_window_samples(t0: float, t1: float, rate_hz: float) -> int:
    return round((t1 - t0) * rate_hz)


def build_timeline(
    st: Stream, t0: float, t1: float, render: SonifyRender, cfg: SonifyConfig
) -> Timeline:
    """Bandpassed, resampled segments placed at their true times on a zero timeline.

    ``st`` holds one channel's gap-separated traces (they may reach past the window). A segment
    lands on the nearest output sample of its first sample; samples outside ``[t0, t1)`` are cut.
    Segments never overlap in source time, so two can only touch within one output sample, where
    the later one starts after the earlier one ends.
    """
    rate_out = render.realRateHz
    n = n_window_samples(t0, t1, rate_out)
    out = np.zeros(n, dtype=np.float64)
    covered = np.zeros(n, dtype=np.bool_)
    runs: list[tuple[int, int]] = []
    if len(st) == 0:
        raise ValueError("no traces to render")
    ids = {tr.id for tr in st}
    if len(ids) != 1:
        raise ValueError(f"render one channel at a time, got {sorted(ids)}")
    rates = {float(tr.stats.sampling_rate) for tr in st}
    if len(rates) != 1:
        raise ValueError(f"one source rate expected, got {sorted(rates)} Hz")
    (source_hz,) = rates
    low, high = render.bandHz
    if high >= source_hz / 2.0:
        raise ValueError(f"band {render.bandHz} Hz reaches the source Nyquist of {source_hz} Hz")
    sos = sps.butter(cfg.bandpassCorners, [low, high], btype="bandpass", fs=source_hz, output="sos")
    up, down = resample_factors(source_hz, rate_out, cfg.resample)

    segments, _ = _segments(st, cfg.joinMisalignmentSamples)
    segments.sort(key=lambda tr: tr.stats.starttime)
    dropped = outside = 0
    reach = 0  # first timeline index not yet written
    for seg in segments:
        if seg.stats.npts <= _min_filter_samples(sos):
            dropped += 1
            continue
        _detrend_taper(
            seg, cfg.detrend, cfg.taper.type, cfg.taper.maxPercentage, cfg.taper.maxLengthS
        )
        filtered = sps.sosfiltfilt(sos, seg.data)
        resampled = sps.resample_poly(
            filtered, up, down, window=("kaiser", cfg.resample.kaiserBeta)
        )
        first = round((seg.stats.starttime.timestamp - t0) * rate_out)
        lo, hi = max(first, reach, 0), min(first + resampled.size, n)
        if hi <= lo:
            outside += 1
            continue
        out[lo:hi] = resampled[lo - first : hi - first]
        covered[lo:hi] = True
        runs.append((lo, hi))
        reach = hi
    log.info(
        "sonify timeline: %d segments, %d placed, %d dropped (too short), %d outside the window; "
        "%g Hz -> %g Hz (up %d / down %d); %d of %d samples hold data",
        len(segments),
        len(runs),
        dropped,
        outside,
        source_hz,
        rate_out,
        up,
        down,
        int(covered.sum()),
        n,
    )
    return Timeline(
        samples=out,
        covered=covered,
        runs=tuple(runs),
        realRateHz=rate_out,
        sourceRateHz=source_hz,
        nSegments=len(segments),
        nDropped=dropped,
        nOutside=outside,
    )


def robust_level(x: FloatArray, covered: BoolArray, percentile: float) -> float:
    """``percentile`` of ``|x|`` over the samples that hold data."""
    values = np.abs(x[covered])
    if values.size == 0:
        raise ValueError("no samples with data: nothing to level")
    level = float(np.percentile(values, percentile))
    if not level > 0.0:
        raise ValueError(f"robust level is {level}: the data is flat")
    return level


def ms_to_samples(ms: float, audio_rate_hz: int) -> int:
    return max(1, round(ms / _MS_PER_S * audio_rate_hz))


def envelope(x_abs: FloatArray, hold_n: int, smooth_n: int) -> FloatArray:
    """Centred running maximum over ``hold_n`` samples, smoothed by a centred Hann of ``smooth_n``."""
    held = ndimage.maximum_filter1d(x_abs, size=hold_n, mode="constant", cval=0.0)
    window = np.hanning(smooth_n + 2)[1:-1]  # no zero end points
    window /= window.sum()
    smoothed: FloatArray = sps.fftconvolve(held, window, mode="same")
    return np.maximum(smoothed, 0.0)  # FFT round-off can dip a hair below zero in silence


def compressor_gain_db(level_db: FloatArray, comp: SonifyCompressor) -> FloatArray:
    """Soft-knee downward compressor gain (dB, <= 0) for envelope levels in dB."""
    slope = 1.0 / comp.ratio - 1.0
    over = level_db - comp.thresholdDb
    half = comp.kneeDb / 2.0
    gain = np.zeros_like(level_db)
    above = over > half
    gain[above] = slope * over[above]
    if comp.kneeDb > 0.0:
        knee = np.abs(over) <= half
        gain[knee] = slope * (over[knee] + half) ** 2 / (2.0 * comp.kneeDb)
    return gain


def compress(x: FloatArray, level: float, comp: SonifyCompressor, audio_rate_hz: int) -> FloatArray:
    """``x / level`` times the compressor gain of its envelope; zero stays zero."""
    normalized = x / level
    env = envelope(
        np.abs(normalized),
        ms_to_samples(comp.envelopeHoldMs, audio_rate_hz),
        ms_to_samples(comp.envelopeSmoothMs, audio_rate_hz),
    )
    gain_db = np.zeros_like(env)
    live = env > 0.0
    gain_db[live] = compressor_gain_db(_DB * np.log10(env[live]), comp)
    out: FloatArray = normalized * np.power(10.0, gain_db / _DB)
    return out


def apply_edge_fades(y: FloatArray, runs: Sequence[tuple[int, int]], fade_n: int) -> None:
    """Raised-cosine fade in and out at both edges of every run, in place (edge samples -> 0)."""
    for lo, hi in runs:
        n = min(fade_n, (hi - lo + 1) // 2)
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / n))
        y[lo : lo + n] *= ramp
        y[hi - n : hi] *= ramp[::-1]


def peak_normalize(y: FloatArray, peak_dbfs: float) -> FloatArray:
    """Scale so the largest ``|y|`` is exactly the target; rounding can only land below it."""
    peak = float(np.max(np.abs(y)))
    if not peak > 0.0:
        raise ValueError("nothing to normalize: every sample is zero")
    target = 10.0 ** (peak_dbfs / _DB)
    out: FloatArray = (y / peak) * target
    return out


def check_rendered(
    audio: FloatArray, timeline: Timeline, t0: float, t1: float, render: SonifyRender
) -> None:
    """Raise unless: no sample above the peak target, silence exactly zero, duration exact."""
    target = 10.0 ** (render.peakDbfs / _DB)
    peak = float(np.max(np.abs(audio)))
    if peak > target:
        raise RuntimeError(f"peak {peak!r} exceeds the target {target!r}")
    if np.any(audio[~timeline.covered] != 0.0):
        raise RuntimeError("a sample with no data is not exactly zero")
    duration = audio.size / render.audioRateHz
    expected = (t1 - t0) / render.speed
    if abs(duration - expected) > 1.0 / render.audioRateHz:
        raise RuntimeError(f"duration {duration} s is not window / speed = {expected} s")


def render_clip(
    st: Stream, t0: float, t1: float, render: SonifyRender, cfg: SonifyConfig
) -> Rendered:
    """One channel's traces -> checked audio samples at ``render.audioRateHz``."""
    timeline = build_timeline(st, t0, t1, render, cfg)
    level = robust_level(timeline.samples, timeline.covered, render.levelPercentile)
    y = compress(timeline.samples, level, render.compressor, render.audioRateHz)
    apply_edge_fades(y, timeline.runs, ms_to_samples(render.edgeFadeMs, render.audioRateHz))
    audio = peak_normalize(y, render.peakDbfs)
    check_rendered(audio, timeline, t0, t1, render)
    return Rendered(audio=audio, timeline=timeline, levelCounts=level)


def signal_report(audio: FloatArray, covered: BoolArray) -> dict[str, float]:
    """Peak and RMS in dBFS and the fraction of samples with no data."""
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    rms_data = float(np.sqrt(np.mean(audio[covered] ** 2))) if covered.any() else 0.0
    return {
        "peakDbfs": _dbfs(peak),
        "rmsDbfs": _dbfs(rms),
        "rmsDataDbfs": _dbfs(rms_data),
        "silentFraction": float(1.0 - covered.mean()),
    }


def _dbfs(value: float) -> float:
    return _DB * math.log10(value) if value > 0.0 else -math.inf


# --- encoding ---------------------------------------------------------------------------------------


def _soundfile() -> ModuleType:
    try:
        import soundfile
    except ImportError as exc:
        raise SoundfileMissingError(
            "hq.preprocess.sonify encodes OGG and MP3 with python-soundfile, which is not a "
            "project dependency: run it with `uv run --with soundfile python -m "
            "hq.preprocess.sonify ...`"
        ) from exc
    return soundfile


def _ogg_crc_table() -> tuple[int, ...]:
    table = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            r = ((r << 1) ^ 0x04C11DB7) if r & 0x80000000 else (r << 1)
            r &= 0xFFFFFFFF
        table.append(r)
    return tuple(table)


_OGG_CRC = _ogg_crc_table()  # Ogg's CRC-32: polynomial 0x04C11DB7, no reflection, init 0


def _ogg_crc(data: bytes | bytearray) -> int:
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _OGG_CRC[((crc >> 24) ^ byte) & 0xFF]
    return crc


def set_ogg_serial(data: bytes, serial: int) -> bytes:
    """Rewrite every page's stream serial and recompute its CRC (single logical stream only)."""
    out = bytearray(data)
    pos = 0
    serials: set[int] = set()
    while pos < len(out):
        if out[pos : pos + 4] != b"OggS":
            raise ValueError(f"not an Ogg page at byte {pos}")
        n_seg = out[pos + 26]
        length = 27 + n_seg + sum(out[pos + 27 : pos + 27 + n_seg])
        page = out[pos : pos + length]
        stored = struct.unpack_from("<I", page, 22)[0]
        page[22:26] = b"\0\0\0\0"
        if _ogg_crc(page) != stored:
            raise ValueError(f"Ogg page at byte {pos} fails its CRC")
        serials.add(struct.unpack_from("<I", page, 14)[0])
        struct.pack_into("<I", page, 14, serial)
        struct.pack_into("<I", page, 22, _ogg_crc(page))
        out[pos : pos + length] = page
        pos += length
    if len(serials) != 1:
        raise ValueError(f"expected one logical Ogg stream, found serials {sorted(serials)}")
    return bytes(out)


def vorbis_nominal_bitrate(data: bytes) -> int:
    """Nominal bitrate (bit/s) from the Vorbis identification header."""
    at = data.find(b"\x01vorbis")
    if at < 0:
        raise ValueError("no Vorbis identification header")
    # version u32, channels u8, rate u32, bitrate max / nominal / min i32
    _, _, _, _, nominal, _ = struct.unpack_from("<IBIiii", data, at + 7)
    return int(nominal)


def mp3_frame_bitrates(data: bytes) -> list[int]:
    """kbps of every MPEG-1 Layer III frame, walking frame by frame after any ID3v2 tag."""
    pos = 0
    if data[:3] == b"ID3":
        size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
        pos = 10 + size
    rates: list[int] = []
    while pos + 4 <= len(data):
        b1, b2 = data[pos + 1], data[pos + 2]
        if data[pos] != 0xFF or (b1 & 0xFE) != 0xFA:  # sync, MPEG-1, Layer III (either CRC flag)
            if data[pos : pos + 3] == b"TAG":  # ID3v1 trailer
                break
            raise ValueError(f"no MPEG-1 Layer III frame header at byte {pos}")
        kbps = _MPEG1_L3_BITRATE_INDEX[b2 >> 4] if (b2 >> 4) < len(_MPEG1_L3_BITRATE_INDEX) else 0
        rate_index = (b2 >> 2) & 0x3
        if kbps == 0 or rate_index >= len(_MPEG1_RATE_INDEX):
            raise ValueError(f"bad MPEG frame header at byte {pos}")
        padding = (b2 >> 1) & 0x1
        rates.append(kbps)
        pos += _MPEG1_L3_SAMPLES_PER_FRAME // 8 * kbps * 1000 // _MPEG1_RATE_INDEX[rate_index]
        pos += padding
    return rates


def _encode(audio: FloatArray, rate_hz: int, block_frames: int, **options: str | float) -> bytes:
    """Mono file bytes, written ``block_frames`` at a time.

    One long write overflows the stack inside libsndfile's Vorbis writer on Windows. The encoded
    bytes do not depend on the block size (``tests/signal/test_sonify.py``, round trip).
    """
    sf = _soundfile()
    buf = io.BytesIO()
    with sf.SoundFile(buf, "w", samplerate=rate_hz, channels=1, **options) as fh:
        for start in range(0, audio.size, block_frames):
            fh.write(audio[start : start + block_frames])
    return buf.getvalue()


def encode_ogg(
    audio: FloatArray, render: SonifyRender, serial: int, block_frames: int = WRITE_BLOCK_FRAMES
) -> bytes:
    data = _encode(
        audio,
        render.audioRateHz,
        block_frames,
        format="OGG",
        subtype="VORBIS",
        compression_level=1.0 - render.oggQuality,  # libsndfile: Vorbis quality = 1 - level
    )
    return set_ogg_serial(data, serial)


def encode_mp3(
    audio: FloatArray, render: SonifyRender, block_frames: int = WRITE_BLOCK_FRAMES
) -> bytes:
    level = (_MP3_MAX_KBPS - render.mp3BitrateKbps) / (_MP3_MAX_KBPS - _MP3_MIN_KBPS)
    data = _encode(
        audio,
        render.audioRateHz,
        block_frames,
        format="MP3",
        subtype="MPEG_LAYER_III",
        compression_level=level,
        bitrate_mode="CONSTANT",
    )
    rates = set(mp3_frame_bitrates(data))
    if rates != {render.mp3BitrateKbps}:
        raise RuntimeError(
            f"MP3 frames at {sorted(rates)} kbps, configured {render.mp3BitrateKbps}"
        )
    return data


def decode_check(data: bytes, audio_rate_hz: int) -> dict[str, float | int]:
    """Decode an encoded file; raise if its rate is wrong or it would clip (|x| > full scale)."""
    sf = _soundfile()
    decoded, rate = sf.read(io.BytesIO(data), dtype="float64", always_2d=False)
    if rate != audio_rate_hz:
        raise RuntimeError(f"decoded rate {rate} Hz, expected {audio_rate_hz} Hz")
    peak = float(np.max(np.abs(decoded)))
    if peak > 1.0:  # full scale: a decoder or player clips above it
        raise RuntimeError(f"decoded peak {peak} is above full scale: lower peakDbfs")
    return {"frames": int(decoded.shape[0]), "peakDbfs": _dbfs(peak)}


# --- clips ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunInputs:
    runId: str
    stations: pd.DataFrame
    picks: pd.DataFrame
    events: list[Any]  # hq_contracts.models.SeismicEvent
    heroEventId: str | None


def load_inputs(run_dir: Path, bundle_dir: Path) -> RunInputs:
    """Run tables through ``hq_contracts.io`` and the bundle JSON through the contract models."""
    from hq_contracts.io import from_frame, read_table
    from hq_contracts.models import BundleMeta, Pick, SeismicEvent, Station

    stations = read_table(run_dir / "stations.parquet")
    picks = read_table(run_dir / "picks.parquet")
    from_frame(stations, Station)  # schema checks; raise on a mismatch
    if not set(picks.columns) >= set(Pick.model_fields):
        raise ValueError(f"picks.parquet lacks Pick columns: {set(Pick.model_fields) - set(picks)}")
    meta = BundleMeta.model_validate(_read_json(bundle_dir / "meta.json"))
    events = [SeismicEvent.model_validate(e) for e in _read_json(bundle_dir / "events.json")]
    run_id = run_dir.name
    if meta.scene.runId != run_id:
        raise ValueError(f"bundle is from run {meta.scene.runId}, run dir is {run_id}")
    foreign = {e.runId for e in events} - {run_id}
    if foreign:
        raise ValueError(f"events.json holds events of other runs: {sorted(foreign)}")
    return RunInputs(run_id, stations, picks, events, meta.scene.heroEventId)


def select_hour(inputs: RunInputs, cfg: SonifyConfig) -> HourSelection:
    bin_s = cfg.busiestHour.binS
    start, n_events = busiest_bin([e.t for e in inputs.events], bin_s)
    end = start + bin_s
    station_id, n_picks = most_picked_station(inputs.stations, inputs.picks, start, end, cfg)
    row = inputs.stations.set_index("id").loc[station_id]
    channel = vertical_channel(station_id, list(row["channels"]), cfg.component)
    return HourSelection(start, end, n_events, station_id, channel, n_picks)


def select_hero(inputs: RunInputs, cfg: SonifyConfig, cache_dir: Path) -> HeroSelection:
    if inputs.heroEventId is None:
        raise ValueError("meta.json scene.heroEventId is null: no hero clip")
    hero = [e for e in inputs.events if e.id == inputs.heroEventId]
    if len(hero) != 1:
        raise ValueError(f"hero event {inputs.heroEventId} is not in events.json exactly once")
    event = hero[0]
    start, end = event.t - cfg.hero.preS, event.t + cfg.hero.postS

    def coverage(station_id: str, channel: str) -> float:
        return coverage_fraction(station_id, channel, start, end, cache_dir=cache_dir)

    station_id, channel, dist_m, frac = nearest_station_with_data(
        inputs.stations, event.enu.e, event.enu.n, coverage, cfg.hero.minCoverageFraction, cfg
    )
    return HeroSelection(event.id, event.t, start, end, station_id, channel, dist_m, frac)


def read_channel(
    station_id: str, channel: str, t0: float, t1: float, pad_s: float, cache_dir: Path
) -> Stream:
    st = read_window(station_id, t0 - pad_s, t1 + pad_s, cache_dir=cache_dir).select(
        channel=channel
    )
    if len(st) == 0:
        raise ValueError(f"{station_id} {channel}: nothing cached in [{t0}, {t1}]")
    return st


def iso_utc(t: float) -> str:
    moment = datetime.fromtimestamp(t, UTC)
    spec = "seconds" if moment.microsecond == 0 else "milliseconds"
    return moment.isoformat(timespec=spec).replace("+00:00", "Z")


def _file_entry(name: str, data: bytes) -> dict[str, Any]:
    return {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def build_manifest(
    *,
    clip: str,
    run_id: str,
    station_id: str,
    channel: str,
    t0: float,
    t1: float,
    render: SonifyRender,
    cfg: SonifyConfig,
    rendered: Rendered,
    files: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """The keys H3 reads first (``stationId`` ... ``source``), then provenance and checks."""
    tl = rendered.timeline
    silent_s = float((~tl.covered).sum()) / tl.realRateHz
    return {
        "stationId": station_id,
        "channel": channel,
        "startUtc": iso_utc(t0),
        "endUtc": iso_utc(t1),
        "speed": render.speed,
        "sampleRateHz": render.audioRateHz,
        "filterHz": [render.bandHz[0], render.bandHz[1]],
        "source": SOURCE,
        "note": NOTE,
        "clip": clip,
        "runId": run_id,
        **extra,
        "durationS": rendered.audio.size / render.audioRateHz,
        "windowS": t1 - t0,
        "realRateHz": tl.realRateHz,
        "sourceSampleRateHz": tl.sourceRateHz,
        "gapsAsSilence": True,
        "silentSeconds": silent_s,
        "segments": {
            "read": tl.nSegments,
            "placed": len(tl.runs),
            "droppedTooShort": tl.nDropped,
        },
        "selection": selection,
        "processing": {
            "detrend": cfg.detrend,
            "taper": cfg.taper.model_dump(mode="json"),
            "bandpass": {"corners": cfg.bandpassCorners, "zeroPhase": True},
            "resample": {"method": "scipy.signal.resample_poly", **cfg.resample.model_dump()},
            "levelPercentile": render.levelPercentile,
            "compression": {
                "description": COMPRESSION_DESCRIPTION,
                **render.compressor.model_dump(mode="json"),
            },
            "edgeFadeMs": render.edgeFadeMs,
            "peakDbfs": render.peakDbfs,
        },
        "files": files,
        "generator": GENERATOR,
    }


def write_clip(
    *,
    clip: str,
    st: Stream,
    t0: float,
    t1: float,
    render: SonifyRender,
    cfg: SonifyConfig,
    run_id: str,
    station_id: str,
    channel: str,
    selection: dict[str, Any],
    extra: dict[str, Any],
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render, encode, check and write one clip; returns (manifest, signal report)."""
    started = perf_counter()
    rendered = render_clip(st, t0, t1, render, cfg)
    ogg = encode_ogg(rendered.audio, render, cfg.oggStreamSerial)
    mp3 = encode_mp3(rendered.audio, render)
    report: dict[str, Any] = signal_report(rendered.audio, rendered.timeline.covered)
    files: dict[str, dict[str, Any]] = {}
    for ext, data in (("ogg", ogg), ("mp3", mp3)):
        if len(data) >= render.maxBytes:
            raise RuntimeError(f"{render.fileStem}.{ext} is {len(data)} bytes, over maxBytes")
        report[f"{ext}Decoded"] = decode_check(data, render.audioRateHz)
        files[ext] = _file_entry(f"{render.fileStem}.{ext}", data)
    files["ogg"]["nominalBitrateBps"] = vorbis_nominal_bitrate(ogg)
    files["ogg"]["vorbisQuality"] = render.oggQuality
    files["mp3"]["bitrateKbps"] = render.mp3BitrateKbps
    manifest = build_manifest(
        clip=clip,
        run_id=run_id,
        station_id=station_id,
        channel=channel,
        t0=t0,
        t1=t1,
        render=render,
        cfg=cfg,
        rendered=rendered,
        files=files,
        selection=selection,
        extra=extra,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{render.fileStem}.ogg").write_bytes(ogg)
    (out_dir / f"{render.fileStem}.mp3").write_bytes(mp3)
    text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    (out_dir / f"{render.fileStem}.json").write_text(text, encoding="utf-8")
    report["levelCounts"] = rendered.levelCounts
    report["runtimeS"] = perf_counter() - started
    log.info(
        "sonify %s: %s %s, %d audio samples, ogg %d B, mp3 %d B, %.2f s",
        clip,
        station_id,
        channel,
        rendered.audio.size,
        len(ogg),
        len(mp3),
        report["runtimeS"],
    )
    return manifest, report


def run_hour(
    inputs: RunInputs, cfg: SonifyConfig, cache_dir: Path, out_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    sel = select_hour(inputs, cfg)
    log.info(
        "sonify hour: bin %s with %d candidate events; %s %s with %d picks",
        iso_utc(sel.startS),
        sel.nEvents,
        sel.stationId,
        sel.channel,
        sel.nPicks,
    )
    st = read_channel(sel.stationId, sel.channel, sel.startS, sel.endS, cfg.padS, cache_dir)
    selection = {
        "rule": HOUR_RULE,
        "binS": cfg.busiestHour.binS,
        "stationKind": cfg.stationKind,
        "usedInRunOnly": cfg.usedInRunOnly,
        "eventsInBin": sel.nEvents,
        "stationPicksInBin": sel.nPicks,
    }
    render = cfg.busiestHour.render
    return write_clip(
        clip=CLIP_HOUR,
        st=st,
        t0=sel.startS,
        t1=sel.endS,
        render=render,
        cfg=cfg,
        run_id=inputs.runId,
        station_id=sel.stationId,
        channel=sel.channel,
        selection=selection,
        extra={},
        out_dir=out_dir,
    )


def run_hero(
    inputs: RunInputs, cfg: SonifyConfig, cache_dir: Path, out_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    sel = select_hero(inputs, cfg, cache_dir)
    log.info(
        "sonify hero: %s at %s; %s %s at %.0f m, coverage %.3f",
        sel.eventId,
        iso_utc(sel.originS),
        sel.stationId,
        sel.channel,
        sel.epiDistM,
        sel.coverageFraction,
    )
    st = read_channel(sel.stationId, sel.channel, sel.startS, sel.endS, cfg.padS, cache_dir)
    selection = {
        "rule": HERO_RULE,
        "heroEventId": sel.eventId,
        "eventOriginUtc": iso_utc(sel.originS),
        "preS": cfg.hero.preS,
        "postS": cfg.hero.postS,
        "stationKind": cfg.stationKind,
        "usedInRunOnly": cfg.usedInRunOnly,
        "epiDistM": sel.epiDistM,
        "coverageFraction": sel.coverageFraction,
        "minCoverageFraction": cfg.hero.minCoverageFraction,
    }
    return write_clip(
        clip=CLIP_HERO,
        st=st,
        t0=sel.startS,
        t1=sel.endS,
        render=cfg.hero.render,
        cfg=cfg,
        run_id=inputs.runId,
        station_id=sel.stationId,
        channel=sel.channel,
        selection=selection,
        extra={"heroEventId": sel.eventId},
        out_dir=out_dir,
    )


# --- CLI --------------------------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_signal(config_dir: Path) -> SignalConfig:
    with (config_dir / "signal.yaml").open(encoding="utf-8") as fh:
        return SignalConfig.model_validate(yaml.safe_load(fh))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hq.preprocess.sonify",
        description="Sped-up audio renderings of cached public waveforms (SEIS-09). "
        "Needs python-soundfile: uv run --with soundfile python -m hq.preprocess.sonify ...",
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="runs/<id> (read only)")
    parser.add_argument(
        "--bundle-dir", type=Path, required=True, help="web bundle with events.json, meta.json"
    )
    parser.add_argument("--config-dir", type=Path, required=True, help="folder with signal.yaml")
    parser.add_argument("--cache-dir", type=Path, required=True, help="cache root (holds mseed/)")
    parser.add_argument("--out-dir", type=Path, required=True, help="where the clips are written")
    parser.add_argument("--clip", choices=[*CLIPS, "all"], default="all")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    started = perf_counter()
    _soundfile()  # fail before any work when the encoder is missing
    cfg = _load_signal(args.config_dir).sonify
    inputs = load_inputs(args.run_dir, args.bundle_dir)
    log.info(
        "sonify: run %s, %d stations, %d picks, %d candidate events",
        inputs.runId,
        len(inputs.stations),
        len(inputs.picks),
        len(inputs.events),
    )
    runners = {CLIP_HOUR: run_hour, CLIP_HERO: run_hero}
    clips = CLIPS if args.clip == "all" else (args.clip,)
    for clip in clips:
        manifest, report = runners[clip](inputs, cfg, args.cache_dir, args.out_dir)
        print(f"== {clip}: manifest")
        print(json.dumps(manifest, indent=2))
        print(f"== {clip}: signal report")
        print(json.dumps(report, indent=2))
    log.info("sonify: %d clips in %.2f s", len(clips), perf_counter() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())

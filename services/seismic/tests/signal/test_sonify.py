"""SEIS-09 sonification. Offline, seeded, synthetic data built inside each test.

The two tests that encode (``test_encoding_round_trip``, ``test_cli_end_to_end``) need
python-soundfile, which is not a project dependency: they are skipped (``pytest.importorskip``)
unless the suite runs under ``uv run --with soundfile==0.14.0 pytest ...``.
"""

import hashlib
import io
import json
import math
import re
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from hq_contracts import models as m
from hq_contracts.io import to_frame, write_table
from obspy import Stream, Trace, UTCDateTime
from pydantic import ValidationError

from hq.config.signal import SignalConfig, SonifyCompressor, SonifyConfig, SonifyRender
from hq.preprocess import sonify

T0 = 1_000_000_000.0  # synthetic window start, on every sample grid used here (2001-09-09 UTC)
SOURCE_HZ = 1000.0
SEED = 20260926
REQUIRED_KEYS = {
    "stationId",
    "channel",
    "startUtc",
    "endUtc",
    "speed",
    "sampleRateHz",
    "filterHz",
    "source",
}


@pytest.fixture()
def cfg(signal_cfg: SignalConfig) -> SonifyConfig:
    return signal_cfg.sonify


@pytest.fixture()
def hour_render(cfg: SonifyConfig) -> SonifyRender:
    return cfg.busiestHour.render


def noise_trace(start: float, seconds: float, rng: np.random.Generator) -> Trace:
    """Seeded broadband noise with a few louder bursts, as raw integer-like counts."""
    n = round(seconds * SOURCE_HZ)
    data = rng.normal(0.0, 100.0, n)
    for k in range(3):
        at = (k + 1) * n // 4
        data[at : at + 500] += rng.normal(0.0, 5000.0 * (k + 1), 500)
    tr = Trace(data=data.astype(np.float64))
    tr.stats.network, tr.stats.station, tr.stats.channel = "XX", "SYN", "GNZ"
    tr.stats.sampling_rate = SOURCE_HZ
    tr.stats.starttime = UTCDateTime(start)
    return tr


def stations_frame(rows: list[dict]) -> pd.DataFrame:
    base = {"channels": ["GNZ", "GN1", "GN2"], "enu_e": 0.0, "enu_n": 0.0, "sampleRateHz": 1000.0}
    return pd.DataFrame([{**base, **row} for row in rows])


# --- config ------------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_sonify_section_parses_and_forbids_unknown_keys(
    signal_cfg: SignalConfig, raw_signal_yaml: dict
) -> None:
    assert isinstance(signal_cfg.sonify, SonifyConfig)
    for render in (signal_cfg.sonify.busiestHour.render, signal_cfg.sonify.hero.render):
        assert render.bandHz[1] < render.audioRateHz / render.speed / 2.0
    raw = json.loads(json.dumps(raw_signal_yaml))
    raw["sonify"]["notAKnob"] = 1
    with pytest.raises(ValidationError):
        SignalConfig.model_validate(raw)
    raw = json.loads(json.dumps(raw_signal_yaml))
    raw["sonify"]["hero"]["render"]["compressor"]["attackMs"] = 1.0
    with pytest.raises(ValidationError):
        SignalConfig.model_validate(raw)


@pytest.mark.smoke
def test_band_must_stay_below_the_real_axis_nyquist(raw_signal_yaml: dict) -> None:
    raw = json.loads(json.dumps(raw_signal_yaml))
    render = raw["sonify"]["busiestHour"]["render"]
    render["bandHz"] = [10.0, render["audioRateHz"] / render["speed"] / 2.0]  # exactly Nyquist
    with pytest.raises(ValidationError, match="bandHz"):
        SignalConfig.model_validate(raw)
    raw = json.loads(json.dumps(raw_signal_yaml))
    raw["sonify"]["hero"]["render"]["mp3BitrateKbps"] = 100  # not an MPEG-1 Layer III bitrate
    with pytest.raises(ValidationError, match="mp3BitrateKbps"):
        SignalConfig.model_validate(raw)


@pytest.mark.smoke
def test_sonify_validators_reject_inconsistent_sections(raw_signal_yaml: dict) -> None:
    def broken(edit: Callable[[dict], None]) -> dict:
        raw = json.loads(json.dumps(raw_signal_yaml))
        edit(raw["sonify"])
        return raw

    def short_pad(s: dict) -> None:
        s["padS"] = s["taper"]["maxLengthS"] / 2.0

    def same_stem(s: dict) -> None:
        s["hero"]["render"]["fileStem"] = s["busiestHour"]["render"]["fileStem"]

    def ceiling_below_peak(s: dict) -> None:
        s["hero"]["render"]["maxTruePeakDbtp"] = s["hero"]["render"]["peakDbfs"] - 0.1

    for edit, match in (
        (short_pad, "padS"),
        (same_stem, "fileStem"),
        (ceiling_below_peak, "maxTruePeakDbtp"),
    ):
        with pytest.raises(ValidationError, match=match):
            SignalConfig.model_validate(broken(edit))


# --- selection ---------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_busiest_bin_ties_go_to_the_earliest_and_bins_are_half_open() -> None:
    hour = 3600.0
    h0 = 1_000_000_800.0  # a clock hour boundary: multiple of 3600
    assert h0 % hour == 0.0
    # Bins h0 and h0 + 1 h both hold two events; an event exactly at h0 + 2 h opens the next bin.
    times = [h0 + 5.0, h0 + 3599.9, h0 + hour, h0 + hour + 10.0, h0 + 2 * hour]
    assert sonify.busiest_bin(times, hour) == (h0, 2)
    assert sonify.busiest_bin(list(reversed(times)), hour) == (h0, 2)  # input order is irrelevant
    assert sonify.busiest_bin([*times, h0 + hour + 20.0], hour) == (h0 + hour, 3)
    with pytest.raises(ValueError):
        sonify.busiest_bin([], hour)


@pytest.mark.smoke
def test_station_choice_is_the_used_borehole_with_most_picks(cfg: SonifyConfig) -> None:
    stations = stations_frame(
        [
            {"id": "XX.SURF", "kind": "surface", "usedInRun": True},
            {"id": "XX.OFF", "kind": "borehole", "usedInRun": False},
            {"id": "XX.B2", "kind": "borehole", "usedInRun": True},
            {"id": "XX.B1", "kind": "borehole", "usedInRun": True},
            {"id": "XX.B3", "kind": "borehole", "usedInRun": True},
        ]
    )
    t0, t1 = T0, T0 + 3600.0
    rows = (
        [("XX.SURF", t0 + 1.0)] * 9  # most picks, but surface
        + [("XX.OFF", t0 + 1.0)] * 8  # more picks, but not used in the run
        + [("XX.B2", t0 + 2.0)] * 3
        + [("XX.B1", t0 + 3.0)] * 2
        + [("XX.B1", t1 - 0.001)]  # last instant inside the bin
        + [("XX.B3", t1)] * 5  # exactly at the end: the next bin, not this one
        + [("XX.B3", t0 - 0.001)] * 5  # just before the bin
    )
    picks = pd.DataFrame(rows, columns=["stationId", "t"])
    # B1 and B2 tie on 3 picks: station id order decides.
    assert sonify.most_picked_station(stations, picks, t0, t1, cfg) == ("XX.B1", 3)
    empty = picks[picks["stationId"] == "XX.SURF"]
    with pytest.raises(ValueError, match="no picks"):
        sonify.most_picked_station(stations, empty, t0, t1, cfg)


@pytest.mark.smoke
def test_hero_station_is_the_nearest_used_borehole_with_data(cfg: SonifyConfig) -> None:
    stations = stations_frame(
        [
            {"id": "XX.SURF", "kind": "surface", "usedInRun": True, "enu_e": 1.0},
            # Nearest borehole with full data, but a 200 Hz source cannot carry a 160 Hz band.
            {"id": "XX.SLOW", "kind": "borehole", "usedInRun": True, "sampleRateHz": 200.0},
            {"id": "XX.NODATA", "kind": "borehole", "usedInRun": True, "enu_e": 10.0},
            {"id": "XX.OFF", "kind": "borehole", "usedInRun": False, "enu_e": 5.0},
            {"id": "XX.SPARSE", "kind": "borehole", "usedInRun": True, "enu_n": 15.0},
            {"id": "XX.TIE2", "kind": "borehole", "usedInRun": True, "enu_n": -30.0},
            {"id": "XX.TIE1", "kind": "borehole", "usedInRun": True, "enu_e": 30.0},
            {"id": "XX.FAR", "kind": "borehole", "usedInRun": True, "enu_e": 500.0},
        ]
    )
    coverage = {
        "XX.SLOW": 1.0,
        "XX.NODATA": 0.0,
        "XX.SPARSE": 0.2,
        "XX.TIE1": 1.0,
        "XX.TIE2": 1.0,
        "XX.FAR": 1.0,
    }
    asked: list[str] = []

    def cov(station_id: str, channel: str) -> float:
        assert channel == "GNZ"
        asked.append(station_id)
        return coverage[station_id]

    got = sonify.nearest_station_with_data(stations, 0.0, 0.0, cov, 0.5, 160.0, cfg)
    assert got == ("XX.TIE1", "GNZ", 30.0, 1.0)  # TIE1 and TIE2 are both 30 m away: id order
    # Nearest first; surface, unused and too-slow stations are never asked for coverage.
    assert asked == ["XX.NODATA", "XX.SPARSE", "XX.TIE1"]
    # A band the 200 Hz source can carry: the nearest station wins.
    assert sonify.nearest_station_with_data(stations, 0.0, 0.0, cov, 0.5, 80.0, cfg)[0] == "XX.SLOW"
    with pytest.raises(ValueError, match="covers"):
        sonify.nearest_station_with_data(stations, 0.0, 0.0, lambda s, c: 0.0, 0.5, 160.0, cfg)


@pytest.mark.smoke
def test_coverage_fraction_reads_headers_and_counts_a_gap_once(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "mseed").mkdir(parents=True)
    pieces = []
    for start, seconds in ((T0, 10.0), (T0 + 20.0, 10.0)):
        tr = Trace(data=np.arange(round(seconds * 100), dtype=np.int32))
        tr.stats.network, tr.stats.station, tr.stats.channel = "XX", "SYN", "GNZ"
        tr.stats.sampling_rate = 100.0
        tr.stats.starttime = UTCDateTime(start)
        pieces.append(tr)
    Stream(pieces).write(str(cache / "mseed" / "XX.SYN..GNZ.20010909.mseed"), format="MSEED")
    frac = sonify.coverage_fraction("XX.SYN", "GNZ", T0, T0 + 40.0, cache_dir=cache)
    assert frac == pytest.approx(0.5)
    assert sonify.coverage_fraction("XX.SYN", "GN1", T0, T0 + 40.0, cache_dir=cache) == 0.0
    assert sonify.coverage_fraction("XX.NONE", "GNZ", T0, T0 + 40.0, cache_dir=cache) == 0.0


@pytest.mark.smoke
def test_vertical_channel_and_resample_factors(cfg: SonifyConfig) -> None:
    assert sonify.vertical_channel("XX.A", ["GN1", "GNZ", "GN2"], "Z") == "GNZ"
    with pytest.raises(ValueError):
        sonify.vertical_channel("XX.A", ["GN1", "GN2"], "Z")
    assert sonify.resample_factors(1000.0, 240.0, cfg.resample) == (6, 25)
    assert sonify.resample_factors(1000.0, 480.0, cfg.resample) == (12, 25)
    assert sonify.resample_factors(200.0, 240.0, cfg.resample) == (6, 5)
    with pytest.raises(ValueError, match="factors"):
        sonify.resample_factors(999.9999, 240.0, cfg.resample)


# --- rendering ---------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_gap_is_exact_silence_at_the_right_samples(
    cfg: SonifyConfig, hour_render: SonifyRender
) -> None:
    """Data from 2 s to 40 s and from 50 s to the end of a 100 s window at 240 Hz real."""
    rng = np.random.default_rng(SEED)
    st = Stream([noise_trace(T0 + 2.0, 38.0, rng), noise_trace(T0 + 50.0, 50.0, rng)])
    t1 = T0 + 100.0
    rate = hour_render.realRateHz
    assert rate == 240.0
    rendered = sonify.render_clip(st, T0, t1, hour_render, cfg)
    audio, tl = rendered.audio, rendered.timeline

    first, gap_lo, gap_hi, n = (
        round(2 * rate),
        round(40 * rate),
        round(50 * rate),
        round(100 * rate),
    )
    assert tl.runs == ((first, gap_lo), (gap_hi, n))
    expected = np.zeros(n, dtype=bool)
    expected[first:gap_lo] = True
    expected[gap_hi:n] = True
    np.testing.assert_array_equal(tl.covered, expected)
    # The gap and the missing start are exact zeros, before and after every processing step.
    assert np.all(tl.samples[~expected] == 0.0)
    assert np.all(audio[~expected] == 0.0)
    # Nothing is interpolated: inside each run every sample holds data (only the fade end points
    # reach zero), and no sample outside the runs does.
    assert np.all(audio[first + 1 : gap_lo - 1] != 0.0)
    assert np.all(audio[gap_hi + 1 : n - 1] != 0.0)
    assert audio[first] == audio[gap_lo - 1] == audio[gap_hi] == audio[n - 1] == 0.0
    assert int(np.count_nonzero(audio == 0.0)) == int((~expected).sum()) + 4
    silent_s = float((~tl.covered).sum()) / rate
    assert silent_s == pytest.approx(2.0 + 10.0)


@pytest.mark.smoke
def test_a_run_ends_at_the_last_real_sample(cfg: SonifyConfig) -> None:
    """resample_poly returns ceil(n * up / down) samples; the ones past the last input sample
    (read from the FIR's zero padding) are not data, so they stay silence."""
    rng = np.random.default_rng(SEED)
    n_in = 98_405  # 98405 * 6 / 25 = 23617.2: ceil gives one sample past the last input sample
    tr = noise_trace(T0, n_in / SOURCE_HZ, rng)
    assert tr.stats.npts == n_in
    last_s = (n_in - 1) / SOURCE_HZ
    for render in (cfg.busiestHour.render, cfg.hero.render):
        tl = sonify.build_timeline(Stream([tr.copy()]), T0, T0 + 200.0, render, cfg)
        rate = tl.realRateHz
        (run,) = tl.runs
        assert run == (0, math.floor(last_s * rate) + 1)
        assert (run[1] - 1) / rate <= last_s < run[1] / rate
        assert np.all(tl.samples[run[1] :] == 0.0)


@pytest.mark.smoke
def test_duration_is_window_over_speed(cfg: SonifyConfig) -> None:
    rng = np.random.default_rng(SEED)
    for render, window_s in ((cfg.busiestHour.render, 100.0), (cfg.hero.render, 60.0)):
        st = Stream([noise_trace(T0 - 5.0, window_s + 10.0, rng)])
        audio = sonify.render_clip(st, T0, T0 + window_s, render, cfg).audio
        assert abs(audio.size / render.audioRateHz - window_s / render.speed) <= (
            1.0 / render.audioRateHz
        )


def tone_amplitude(y: np.ndarray, rate_hz: float, freq_hz: float) -> float:
    t = np.arange(y.size) / rate_hz
    return float(2.0 / y.size * abs(np.sum(y * np.exp(-2j * np.pi * freq_hz * t))))


@pytest.mark.smoke
def test_out_of_band_tones_are_attenuated(cfg: SonifyConfig, hour_render: SonifyRender) -> None:
    """Band 10-80 Hz at 240 Hz real: 2 Hz is below it; 300 Hz would alias to 60 Hz if unfiltered."""
    assert hour_render.bandHz == (10.0, 80.0)
    t = np.arange(round(60.0 * SOURCE_HZ)) / SOURCE_HZ
    data = (
        np.sin(2 * np.pi * 2.0 * t) + np.sin(2 * np.pi * 40.0 * t) + np.sin(2 * np.pi * 300.0 * t)
    )
    tr = Trace(data=data * 1000.0)
    tr.stats.sampling_rate, tr.stats.channel = SOURCE_HZ, "GNZ"
    tr.stats.starttime = UTCDateTime(T0)
    tl = sonify.build_timeline(Stream([tr]), T0, T0 + 60.0, hour_render, cfg)
    rate = tl.realRateHz
    middle = tl.samples[round(10 * rate) : round(50 * rate)] / 1000.0  # away from the edges
    assert tone_amplitude(middle, rate, 40.0) == pytest.approx(1.0, abs=0.05)
    assert tone_amplitude(middle, rate, 2.0) < 0.01  # below -40 dB
    assert tone_amplitude(middle, rate, 60.0) < 0.01  # 300 Hz folded onto 60 Hz: filtered first


@pytest.mark.smoke
def test_no_sample_exceeds_the_peak_target(cfg: SonifyConfig, hour_render: SonifyRender) -> None:
    rng = np.random.default_rng(SEED)
    tr = noise_trace(T0, 100.0, rng)
    tr.data[50_000] = 1.0e9  # one enormous spike
    audio = sonify.render_clip(Stream([tr]), T0, T0 + 100.0, hour_render, cfg).audio
    target = 10.0 ** (hour_render.peakDbfs / 20.0)
    assert float(np.max(np.abs(audio))) <= target
    assert float(np.max(np.abs(audio))) == pytest.approx(target, rel=1e-12)
    for peak_dbfs in (-0.1, -1.0, -3.0, -6.02):
        y = sonify.peak_normalize(rng.normal(0.0, 7.3, 10_001), peak_dbfs)
        assert float(np.max(np.abs(y))) <= 10.0 ** (peak_dbfs / 20.0)


@pytest.mark.smoke
def test_check_rendered_refuses_sound_in_a_gap(
    cfg: SonifyConfig, hour_render: SonifyRender
) -> None:
    rng = np.random.default_rng(SEED)
    st = Stream([noise_trace(T0, 40.0, rng), noise_trace(T0 + 50.0, 50.0, rng)])
    rendered = sonify.render_clip(st, T0, T0 + 100.0, hour_render, cfg)
    bad = rendered.audio.copy()
    bad[round(45 * hour_render.realRateHz)] = 1e-6
    with pytest.raises(RuntimeError, match="not exactly zero"):
        sonify.check_rendered(bad, rendered.timeline, T0, T0 + 100.0, hour_render)


@pytest.mark.smoke
def test_compressor_is_monotonic_and_lifts_small_relative_to_large(
    hour_render: SonifyRender,
) -> None:
    comp: SonifyCompressor = hour_render.compressor
    levels = np.linspace(-60.0, 120.0, 3601)
    gain = sonify.compressor_gain_db(levels, comp)
    out = levels + gain
    assert np.all(np.diff(out) > 0.0)  # louder in, louder out
    assert np.all(np.diff(gain) <= 1e-12)  # gain never rises with level
    assert np.all(gain <= 0.0)
    assert np.all(gain[levels < comp.thresholdDb - comp.kneeDb / 2.0] == 0.0)
    # Far above the knee, 1 dB more in gives 1 / ratio dB more out.
    far = levels > comp.thresholdDb + comp.kneeDb
    assert np.allclose(np.diff(out[far]) / np.diff(levels[far]), 1.0 / comp.ratio)

    # On the waveform: a quiet and a loud steady tone, 60 dB apart.
    rate = hour_render.audioRateHz
    t = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * 1000.0 * t)
    x = np.concatenate([tone * 10.0, tone * 10_000.0])
    y = sonify.compress(x, 1.0, comp, rate)
    quiet = float(np.max(np.abs(y[rate // 4 : rate * 3 // 4])))
    loud = float(np.max(np.abs(y[rate + rate // 4 : rate + rate * 3 // 4])))
    assert loud > quiet  # order kept
    assert 20.0 * math.log10(loud / quiet) < 30.0  # at least half of the 60 dB gap removed


@pytest.mark.smoke
def test_envelope_looks_ahead_briefly_covers_x_and_releases_exponentially(
    hour_render: SonifyRender,
) -> None:
    """A loud burst in noise: nothing earlier than lookaheadMs before it changes, the envelope
    never falls below |x|, and it falls by 1/e per releaseMs after the burst."""
    comp = hour_render.compressor
    rate = hour_render.audioRateHz
    ahead = sonify.ms_to_samples(comp.lookaheadMs, rate)
    release = sonify.ms_to_samples(comp.releaseMs, rate)
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0.0, 1.0, rate)  # 1 s of audio
    onset, width, loud = rate // 2, 200, 1.0e4
    x = noise.copy()
    x[onset : onset + width] = loud
    env = sonify.envelope(np.abs(x), ahead, release)
    quiet_env = sonify.envelope(np.abs(noise), ahead, release)

    assert np.all(env >= np.abs(x) * (1.0 - 1e-12))  # onsets are never let through first
    np.testing.assert_allclose(env[: onset - ahead], quiet_env[: onset - ahead], rtol=1e-12)
    assert env[onset - ahead] > 10.0 * quiet_env[onset - ahead]  # the ramp starts right there
    last = onset + width - 1  # last loud sample
    for k in (1, 2, 3):
        n = last + ahead + k * release
        expected = loud * math.exp(-k) * np.mean(np.exp(-np.arange(ahead + 1) / release))
        assert env[n] == pytest.approx(expected, rel=0.02)

    # The compressor inherits it: the record before the onset keeps its level.
    y = sonify.compress(x, 1.0, comp, rate)
    y_quiet = sonify.compress(noise, 1.0, comp, rate)
    np.testing.assert_allclose(y[: onset - ahead], y_quiet[: onset - ahead], rtol=1e-12)


# --- manifest and encoding ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_manifest_keys_source_and_copy(cfg: SonifyConfig, hour_render: SonifyRender) -> None:
    rng = np.random.default_rng(SEED)
    st = Stream([noise_trace(T0, 100.0, rng)])
    rendered = sonify.render_clip(st, T0, T0 + 100.0, hour_render, cfg)
    files = {"ogg": {"name": "a.ogg"}, "mp3": {"name": "a.mp3"}}
    manifest = sonify.build_manifest(
        clip=sonify.CLIP_HOUR,
        run_id="test-run",
        station_id="XX.SYN",
        channel="GNZ",
        t0=T0,
        t1=T0 + 100.25,
        render=hour_render,
        cfg=cfg,
        rendered=rendered,
        files=files,
        selection={"rule": sonify.HOUR_RULE},
        extra={"heroEventId": "hq-test-000001"},
        encoder={"library": "python-soundfile", "soundfile": "0.0", "libsndfile": "0.0"},
    )
    assert REQUIRED_KEYS <= set(manifest)
    assert list(manifest)[: len(REQUIRED_KEYS)] == [
        "stationId",
        "channel",
        "startUtc",
        "endUtc",
        "speed",
        "sampleRateHz",
        "filterHz",
        "source",
    ]
    assert manifest["source"] == "EarthScope public waveforms"
    assert manifest["startUtc"] == "2001-09-09T01:46:40Z"
    assert manifest["endUtc"] == "2001-09-09T01:48:20.250Z"
    assert manifest["speed"] == hour_render.speed
    assert manifest["sampleRateHz"] == hour_render.audioRateHz
    assert manifest["filterHz"] == list(hour_render.bandHz)
    # filterHz is ground motion; audioBandHz is where it plays (x speed).
    assert manifest["filterDomain"] == sonify.FILTER_DOMAIN
    assert manifest["audioBandHz"] == [f * hour_render.speed for f in hour_render.bandHz]
    assert manifest["seedId"] == "XX.SYN..GNZ"
    assert manifest["gapsAsSilence"] is True
    assert manifest["heroEventId"] == "hq-test-000001"
    assert manifest["encoder"]["library"] == "python-soundfile"
    assert manifest["generator"] == "hq.preprocess.sonify"
    json.dumps(manifest, allow_nan=False)  # strict JSON
    # Copy text: no counts, and a rendering, never "the sound of an earthquake".
    for text in (sonify.NOTE, sonify.HOUR_RULE, sonify.HERO_RULE, sonify.FILTER_DOMAIN):
        assert not re.search(r"\d", text)
        assert "earthquake" not in text.lower()


def mp3_frame(bitrate_index: int, padding: int = 0) -> bytes:
    """One MPEG-1 Layer III frame at 48 kHz: header, then zeros to the frame length."""
    kbps = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)[bitrate_index]
    header = bytes([0xFF, 0xFB, (bitrate_index << 4) | (1 << 2) | (padding << 1), 0xC4])
    length = 144 * kbps * 1000 // 48000 + padding
    return header + bytes(length - 4)


@pytest.mark.smoke
def test_mp3_frame_walk_reads_bitrates_and_rejects_garbage() -> None:
    data = mp3_frame(11) + mp3_frame(11, padding=1) + mp3_frame(9)
    assert sonify.mp3_frame_bitrates(data) == [192, 192, 128]
    id3 = b"ID3" + bytes([4, 0, 0, 0, 0, 0, 10]) + bytes(10)
    assert sonify.mp3_frame_bitrates(id3 + mp3_frame(11)) == [192]
    with pytest.raises(ValueError):
        sonify.mp3_frame_bitrates(mp3_frame(11) + b"\x00" * 8)


@pytest.mark.smoke
def test_missing_soundfile_names_the_uv_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "soundfile", None)  # makes `import soundfile` raise
    command = "uv run --with soundfile==" + sonify.SOUNDFILE_PIN
    assert sonify.UV_COMMAND.startswith(command)
    with pytest.raises(sonify.SoundfileMissingError, match=re.escape(command)):
        sonify._soundfile()


@pytest.mark.smoke
def test_encoding_round_trip(cfg: SonifyConfig, hour_render: SonifyRender) -> None:
    """Needs python-soundfile (not a project dependency): skipped without it."""
    sf = pytest.importorskip("soundfile")
    rng = np.random.default_rng(SEED)
    st = Stream([noise_trace(T0, 40.0, rng), noise_trace(T0 + 50.0, 50.0, rng)])
    audio = sonify.render_clip(st, T0, T0 + 100.0, hour_render, cfg).audio

    ogg = sonify.encode_ogg(audio, hour_render, cfg.oggStreamSerial)
    mp3 = sonify.encode_mp3(audio, hour_render)
    assert ogg[:4] == b"OggS"
    assert struct.unpack_from("<I", ogg, 14)[0] == cfg.oggStreamSerial
    assert set(sonify.mp3_frame_bitrates(mp3)) == {hour_render.mp3BitrateKbps}
    assert sonify.vorbis_nominal_bitrate(ogg) > 0
    oversample = cfg.truePeakOversample
    for data in (ogg, mp3):
        decoded, rate = sf.read(io.BytesIO(data), dtype="float64")
        assert rate == hour_render.audioRateHz
        assert decoded.shape == audio.shape  # both decoders honour the exact length
        assert float(np.max(np.abs(decoded))) < 1.0
        report = sonify.decode_check(data, hour_render, audio.size, oversample)
        assert report["frames"] == audio.size
        assert report["peakDbfs"] <= report["truePeakDbtp"] <= hour_render.maxTruePeakDbtp
        # A length the renderer did not produce, or a ceiling the file does not meet, raises.
        with pytest.raises(RuntimeError, match="decoded"):
            sonify.decode_check(data, hour_render, audio.size + 1, oversample)
        strict = hour_render.model_copy(update={"maxTruePeakDbtp": report["truePeakDbtp"] - 0.01})
        with pytest.raises(RuntimeError, match="true peak"):
            sonify.decode_check(data, strict, audio.size, oversample)
    # Deterministic, and independent of how the samples are handed to libsndfile.
    assert sonify.encode_ogg(audio, hour_render, cfg.oggStreamSerial, block_frames=1000) == ogg
    assert sonify.encode_mp3(audio, hour_render, block_frames=1000) == mp3
    with pytest.raises(ValueError, match="CRC"):
        broken = bytearray(ogg)
        broken[40] ^= 0xFF
        sonify.set_ogg_serial(bytes(broken), 1)


# --- orchestration: run dir + bundle + cache -> selection -> files ----------------------------------

RUN_ID = "20010909-0100-abc1234"
OTHER_RUN_ID = "20010909-0200-def5678"
BIN_S = 60.0  # a short bin keeps the synthetic "hour" small
H0 = 999_999_960.0  # a multiple of BIN_S: 2001-09-09T01:46:00Z
HERO_T = H0 + 90.0
HERO_ID = f"hq-{RUN_ID}-000005"
EVENT_TIMES = (H0 + 5.0, H0 + 20.0, H0 + 40.0, H0 + 70.0, HERO_T)  # 3 in [H0, H0 + 60), 2 after


@dataclass(frozen=True)
class RunFiles:
    run_dir: Path
    bundle_dir: Path
    cache_dir: Path


def contract_station(
    sid: str, kind: str, rate_hz: float, channels: list[str], east_m: float
) -> m.Station:
    net, sta = sid.split(".")
    depth = 300.0 if kind == "borehole" else 0.0
    return m.Station(
        id=sid,
        network=net,
        station=sta,
        latitude=38.5,
        longitude=-112.9,
        surfaceElevM=1700.0,
        sensorDepthM=depth,
        sensorElevM=1700.0 - depth,
        kind=kind,
        channels=channels,
        sampleRateHz=rate_hz,
        enu=m.Enu(e=east_m, n=0.0, u=-depth),
        preprocessProfile="synthetic",
        usedInRun=True,
    )


def contract_event(i: int, t: float, run_id: str = RUN_ID) -> m.SeismicEvent:
    return m.SeismicEvent(
        id=f"hq-{run_id}-{i:06d}",
        runId=run_id,
        t=t,
        latitude=38.5,
        longitude=-112.9,
        elevM=-1500.0,
        depthKm=3.2,
        enu=m.Enu(e=0.0, n=0.0, u=-3200.0),
        quality=m.LocationQuality(
            method="grid1d",
            statics=False,
            nStations=4,
            nP=4,
            nS=2,
            rmsS=0.04,
            gapDeg=120.0,
            minEpiDistM=500.0,
            hErrM=None,
            vErrM=None,
            depthOnEdge=False,
        ),
        tier="C",
        tierReasons=["synthetic"],
        meanPickProb=0.5,
        revealOrder=-1,
        pickIds=[],
    )


def bundle_meta(run_id: str, hero_id: str | None) -> m.BundleMeta:
    run = m.ProcessingRun(
        id=run_id,
        mode="mock",
        createdAt="2001-09-09T02:00:00Z",
        gitSha="abc1234",
        windowStart=H0 - 3600.0,
        windowEnd=H0 + 3600.0,
        windowLabel="synthetic",
        bbox=(-113.2, 38.28, -112.6, 38.74),
        stationIds=[],
        pickerModel="synthetic",
        pickerWeights="synthetic",
        softwareVersions={},
        runtimeS={},
        picker={},
        associator={},
        velocityModel={},
        locator={},
        tiering={},
        matching={},
        isSynthetic=True,
    )
    summary = m.AnalysisSummary(
        runId=run_id,
        publicCatalogCount=0,
        recoveredCatalogCount=0,
        recall=0.0,
        unmatchedPublicIds=[],
        candidateCount=len(EVENT_TIMES),
        additionalCount=len(EVENT_TIMES),
        additional=m.TierCounts(A=0, B=0, C=len(EVENT_TIMES)),
        strictQualityCount=0,
        strictAdditionalCount=0,
        medianStations=4.0,
        medianRmsS=0.04,
    )
    scene = m.SceneMeta(
        runId=run_id,
        originLat=38.5,
        originLon=-112.9,
        originElevM=1700.0,
        refSurfaceElevM=1700.0,
        projection="synthetic",
        depthLabel="synthetic",
        heroEventId=hero_id,
        isSynthetic=True,
    )
    return m.BundleMeta(mode="mock", scene=scene, run=run, summary=summary)


def write_bundle(bundle_dir: Path, meta: m.BundleMeta, events: list[m.SeismicEvent]) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    body = json.dumps([e.model_dump(mode="json") for e in events])
    (bundle_dir / "events.json").write_text(body, encoding="utf-8")
    return bundle_dir


def write_mseed(cache_dir: Path, sid: str, channel: str, rate_hz: float) -> None:
    """160 s of seeded integer noise from H0 - 20 s, with a burst after every candidate event."""
    rng = np.random.default_rng(SEED)
    start = H0 - 20.0
    data = rng.normal(0.0, 50.0, round(160.0 * rate_hz))
    burst = round(0.3 * rate_hz)
    for t in EVENT_TIMES:
        at = round((t + 0.5 - start) * rate_hz)
        data[at : at + burst] += rng.normal(0.0, 5000.0, burst)
    net, sta = sid.split(".")
    tr = Trace(data=data.astype(np.int32))
    tr.stats.network, tr.stats.station, tr.stats.channel = net, sta, channel
    tr.stats.sampling_rate = rate_hz
    tr.stats.starttime = UTCDateTime(start)
    path = cache_dir / "mseed" / f"{net}.{sta}..{channel}.20010909.mseed"
    path.parent.mkdir(parents=True, exist_ok=True)
    tr.write(str(path), format="MSEED")


@pytest.fixture(scope="module")
def run_files(tmp_path_factory: pytest.TempPathFactory) -> RunFiles:
    """A synthetic run dir, bundle and cache.

    Stations: XX.NEAR (borehole, 200 Hz, 10 m from the hero epicentre, data cached), XX.MID
    (borehole, 1000 Hz, 50 m, nothing cached), XX.FAR (borehole, 1000 Hz, 500 m, data cached) and
    XX.SURF (surface, the most picks). The bin [H0, H0 + 60) holds three events, the next two.
    """
    root = tmp_path_factory.mktemp("sonify-run")
    run_dir = root / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    stations = [
        contract_station("XX.NEAR", "borehole", 200.0, ["HHZ", "HH1", "HH2"], 10.0),
        contract_station("XX.MID", "borehole", 1000.0, ["GNZ", "GN1", "GN2"], 50.0),
        contract_station("XX.FAR", "borehole", 1000.0, ["GNZ", "GN1", "GN2"], 500.0),
        contract_station("XX.SURF", "surface", 100.0, ["HHZ", "HHN", "HHE"], 1.0),
    ]
    write_table(to_frame(stations), run_dir / "stations.parquet", "Station")
    counts = [("XX.SURF", H0, 9), ("XX.FAR", H0, 6), ("XX.NEAR", H0, 5), ("XX.MID", H0, 2)]
    counts.append(("XX.MID", H0 + BIN_S, 20))  # the most picks, but in the next bin
    picks = [
        m.Pick(
            id=f"phasenet:{sid}:P:{t0 + 1.0 + k:.3f}",
            stationId=sid,
            phase="P",
            t=t0 + 1.0 + k,
            prob=0.9,
            picker="phasenet:synthetic",
        )
        for sid, t0, n in counts
        for k in range(n)
    ]
    write_table(to_frame(picks), run_dir / "picks.parquet", "Pick")
    events = [contract_event(i, t) for i, t in enumerate(EVENT_TIMES, start=1)]
    assert events[-1].id == HERO_ID
    bundle_dir = write_bundle(root / "bundle", bundle_meta(RUN_ID, HERO_ID), events)
    cache_dir = root / "cache"
    write_mseed(cache_dir, "XX.NEAR", "HHZ", 200.0)
    write_mseed(cache_dir, "XX.FAR", "GNZ", 1000.0)
    return RunFiles(run_dir, bundle_dir, cache_dir)


def short_bin(cfg: SonifyConfig) -> SonifyConfig:
    return cfg.model_copy(
        update={"busiestHour": cfg.busiestHour.model_copy(update={"binS": BIN_S})}
    )


@pytest.mark.smoke
def test_selection_from_run_bundle_and_cache(run_files: RunFiles, cfg: SonifyConfig) -> None:
    inputs = sonify.load_inputs(run_files.run_dir, run_files.bundle_dir)
    assert inputs.runId == RUN_ID
    assert inputs.heroEventId == HERO_ID
    assert len(inputs.events) == len(EVENT_TIMES)

    hour = sonify.select_hour(inputs, short_bin(cfg))
    # Surface has more picks and MID has more in the next bin; FAR leads the boreholes in this one.
    assert hour == sonify.HourSelection(H0, H0 + BIN_S, 3, "XX.FAR", "GNZ", 6)

    hero = sonify.select_hero(inputs, cfg, run_files.cache_dir)
    # NEAR is nearest and has data, but its 200 Hz source cannot carry the hero band; MID has
    # nothing cached; FAR covers the whole window.
    assert cfg.hero.render.bandHz[1] >= 200.0 / 2.0
    assert (hero.eventId, hero.stationId, hero.channel) == (HERO_ID, "XX.FAR", "GNZ")
    assert (hero.startS, hero.endS) == (HERO_T - cfg.hero.preS, HERO_T + cfg.hero.postS)
    assert hero.epiDistM == pytest.approx(500.0)
    assert hero.coverageFraction == pytest.approx(1.0)

    cache = run_files.cache_dir
    st = sonify.read_channel("XX.FAR", "GNZ", hero.startS, hero.endS, cfg.padS, cache)
    assert {tr.id for tr in st} == {"XX.FAR..GNZ"}
    with pytest.raises(ValueError, match="nothing cached"):
        sonify.read_channel("XX.NEAR", "GNZ", hero.startS, hero.endS, cfg.padS, cache)


@pytest.mark.smoke
def test_load_inputs_rejects_a_bundle_from_another_run(
    run_files: RunFiles, cfg: SonifyConfig, tmp_path: Path
) -> None:
    events = [contract_event(1, H0 + 5.0)]
    other = write_bundle(tmp_path / "other", bundle_meta(OTHER_RUN_ID, None), events)
    with pytest.raises(ValueError, match="bundle is from run"):
        sonify.load_inputs(run_files.run_dir, other)
    foreign = [*events, contract_event(2, H0 + 6.0, run_id=OTHER_RUN_ID)]
    mixed = write_bundle(tmp_path / "mixed", bundle_meta(RUN_ID, None), foreign)
    with pytest.raises(ValueError, match="other runs"):
        sonify.load_inputs(run_files.run_dir, mixed)
    no_hero = write_bundle(tmp_path / "no-hero", bundle_meta(RUN_ID, None), events)
    inputs = sonify.load_inputs(run_files.run_dir, no_hero)
    with pytest.raises(ValueError, match="heroEventId is null"):
        sonify.select_hero(inputs, cfg, run_files.cache_dir)


@pytest.mark.smoke
def test_cli_end_to_end(
    run_files: RunFiles,
    raw_signal_yaml: dict,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Needs python-soundfile (not a project dependency): skipped without it."""
    pytest.importorskip("soundfile")
    raw = json.loads(json.dumps(raw_signal_yaml))
    raw["sonify"]["busiestHour"]["binS"] = BIN_S
    # The synthetic bursts are white noise: more codec overshoot than seismic signal, so this test
    # (about orchestration, not level) renders with extra headroom under the true-peak ceiling.
    for clip in ("busiestHour", "hero"):
        raw["sonify"][clip]["render"]["peakDbfs"] = -3.0
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "signal.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = SignalConfig.model_validate(raw).sonify

    def run(out_dir: Path) -> dict[str, bytes]:
        argv = [
            *("--run-dir", str(run_files.run_dir), "--bundle-dir", str(run_files.bundle_dir)),
            *("--config-dir", str(config_dir), "--cache-dir", str(run_files.cache_dir)),
            *("--out-dir", str(out_dir), "--clip", "all"),
        ]
        assert sonify.main(argv) == 0
        return {p.name: p.read_bytes() for p in sorted(out_dir.iterdir())}

    first = run(tmp_path / "a")
    stems = {cfg.busiestHour.render.fileStem: "hour", cfg.hero.render.fileStem: "hero"}
    assert set(first) == {f"{stem}.{ext}" for stem in stems for ext in ("ogg", "mp3", "json")}
    for stem, clip in stems.items():
        manifest = json.loads(first[f"{stem}.json"])
        render = cfg.busiestHour.render if clip == "hour" else cfg.hero.render
        assert REQUIRED_KEYS <= set(manifest)
        assert manifest["source"] == "EarthScope public waveforms"
        assert manifest["clip"] == clip
        assert manifest["runId"] == RUN_ID
        assert manifest["stationId"] == "XX.FAR"
        assert manifest["seedId"] == "XX.FAR..GNZ"
        assert manifest["sampleRateHz"] == render.audioRateHz
        assert manifest["durationS"] == pytest.approx(manifest["windowS"] / render.speed)
        assert manifest["encoder"]["library"] == "python-soundfile"
        for ext in ("ogg", "mp3"):
            entry = manifest["files"][ext]
            data = first[f"{stem}.{ext}"]
            assert entry["name"] == f"{stem}.{ext}"
            assert entry["bytes"] == len(data) < render.maxBytes
            assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    hour_manifest = json.loads(first[f"{cfg.busiestHour.render.fileStem}.json"])
    assert hour_manifest["startUtc"] == "2001-09-09T01:46:00Z"
    assert hour_manifest["selection"]["eventsInBin"] == 3
    hero_manifest = json.loads(first[f"{cfg.hero.render.fileStem}.json"])
    assert hero_manifest["heroEventId"] == HERO_ID
    assert "signal report" in capsys.readouterr().out
    assert run(tmp_path / "b") == first  # deterministic, byte for byte

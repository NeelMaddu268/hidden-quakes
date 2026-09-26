"""SEIS-09 sonification. Offline, seeded, synthetic data built inside each test.

The encoding round trip needs python-soundfile, which is not a project dependency: it is skipped
(``pytest.importorskip``) unless the suite runs under ``uv run --with soundfile pytest ...``.
"""

import io
import json
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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
    base = {"channels": ["GNZ", "GN1", "GN2"], "enu_e": 0.0, "enu_n": 0.0}
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
            {"id": "XX.NODATA", "kind": "borehole", "usedInRun": True, "enu_e": 10.0},
            {"id": "XX.OFF", "kind": "borehole", "usedInRun": False, "enu_e": 5.0},
            {"id": "XX.SPARSE", "kind": "borehole", "usedInRun": True, "enu_n": 15.0},
            {"id": "XX.TIE2", "kind": "borehole", "usedInRun": True, "enu_n": -30.0},
            {"id": "XX.TIE1", "kind": "borehole", "usedInRun": True, "enu_e": 30.0},
            {"id": "XX.FAR", "kind": "borehole", "usedInRun": True, "enu_e": 500.0},
        ]
    )
    coverage = {"XX.NODATA": 0.0, "XX.SPARSE": 0.2, "XX.TIE1": 1.0, "XX.TIE2": 1.0, "XX.FAR": 1.0}
    asked: list[str] = []

    def cov(station_id: str, channel: str) -> float:
        assert channel == "GNZ"
        asked.append(station_id)
        return coverage[station_id]

    got = sonify.nearest_station_with_data(stations, 0.0, 0.0, cov, 0.5, cfg)
    assert got == ("XX.TIE1", "GNZ", 30.0, 1.0)  # TIE1 and TIE2 are both 30 m away: id order
    assert asked == ["XX.NODATA", "XX.SPARSE", "XX.TIE1"]  # nearest first; surface/unused skipped
    with pytest.raises(ValueError, match="covers"):
        sonify.nearest_station_with_data(stations, 0.0, 0.0, lambda s, c: 0.0, 0.5, cfg)


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
    assert manifest["gapsAsSilence"] is True
    assert manifest["heroEventId"] == "hq-test-000001"
    assert manifest["generator"] == "hq.preprocess.sonify"
    json.dumps(manifest, allow_nan=False)  # strict JSON
    # Copy text: no counts, and a rendering, never "the sound of an earthquake".
    for text in (sonify.NOTE, sonify.HOUR_RULE, sonify.HERO_RULE):
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
    with pytest.raises(sonify.SoundfileMissingError, match=r"uv run --with soundfile"):
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
    for data in (ogg, mp3):
        decoded, rate = sf.read(io.BytesIO(data), dtype="float64")
        assert rate == hour_render.audioRateHz
        assert decoded.shape == audio.shape  # both decoders honour the exact length
        assert float(np.max(np.abs(decoded))) < 1.0
        report = sonify.decode_check(data, hour_render.audioRateHz)
        assert report["frames"] == audio.size
        assert not math.isinf(report["peakDbfs"])
    # Deterministic, and independent of how the samples are handed to libsndfile.
    assert sonify.encode_ogg(audio, hour_render, cfg.oggStreamSerial, block_frames=1000) == ogg
    assert sonify.encode_mp3(audio, hour_render, block_frames=1000) == mp3
    with pytest.raises(ValueError, match="CRC"):
        broken = bytearray(ogg)
        broken[40] ^= 0xFF
        sonify.set_ogg_serial(bytes(broken), 1)

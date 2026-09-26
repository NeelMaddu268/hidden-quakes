"""SEIS-03 preprocessing profiles. Offline, seeded, synthetic signals built inside each test."""

import copy
import logging

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from pydantic import ValidationError
from scipy import signal as sps

from hq.config.signal import (
    DecimateProfile,
    PassthroughProfile,
    SignalConfig,
    StretchProfile,
)
from hq.preprocess import TimeMap, display_copy, for_picking
from hq.preprocess.profiles import design_antialias

LOGGER = "hq.preprocess.profiles"
T0 = UTCDateTime(1_800_000_000.1234)  # arbitrary synthetic start, deliberately off the 10 ms grid
SEG_S = 20.0  # synthetic segment length, longer than minSegmentS in signal.yaml
GAP_S = 5.0


def make_trace(
    data: np.ndarray,
    rate: float,
    *,
    channel: str = "DPZ",
    start: UTCDateTime = T0,
    station: str = "SYN1",
) -> Trace:
    header = {
        "network": "XX",
        "station": station,
        "location": "",
        "channel": channel,
        "sampling_rate": rate,
        "starttime": start,
    }
    return Trace(data=data, header=header)


def noise(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def times(tr: Trace) -> np.ndarray:
    return tr.stats.starttime.timestamp + np.arange(tr.stats.npts) / tr.stats.sampling_rate


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def interior(tr: Trace, edge_s: float) -> Trace:
    """The part of a trace at least ``edge_s`` from either end (clear of taper and filter edges)."""
    return tr.slice(tr.stats.starttime + edge_s, tr.stats.endtime - edge_s)


def edge_s(cfg: SignalConfig) -> float:
    return 2.0 * cfg.preprocess.taper.maxLengthS


def snapshot(st: Stream) -> list[tuple[dict, np.ndarray]]:
    return [(copy.deepcopy(dict(tr.stats)), tr.data.copy()) for tr in st]


def assert_unchanged(st: Stream, before: list[tuple[dict, np.ndarray]]) -> None:
    assert len(st) == len(before)
    for tr, (stats, data) in zip(st, before):
        assert dict(tr.stats) == stats
        assert type(tr.data) is type(data)
        assert tr.data.dtype == data.dtype
        np.testing.assert_array_equal(np.ma.getdata(tr.data), np.ma.getdata(data))
        np.testing.assert_array_equal(np.ma.getmaskarray(tr.data), np.ma.getmaskarray(data))


def gapped_stream(rate: float, *, seed: int, masked: bool, short_s: float | None = None) -> Stream:
    """Two SEG_S segments of one channel separated by GAP_S, optionally merged into one masked
    trace, optionally followed by a short segment after another gap."""
    n = int(SEG_S * rate)
    parts = [
        make_trace(noise(seed, n), rate),
        make_trace(noise(seed + 1, n), rate, start=T0 + SEG_S + GAP_S),
    ]
    if short_s is not None:
        parts.append(
            make_trace(noise(seed + 2, int(short_s * rate)), rate, start=T0 + 2 * (SEG_S + GAP_S))
        )
    st = Stream(traces=parts)
    if masked:
        st.merge()  # default method: gaps become masked samples, not values
        assert len(st) == 1 and np.ma.is_masked(st[0].data)
    return st


# --- a. aliasing (ACCEPTANCE) --------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("profile", "rate", "f_lo", "f_hi"),
    [
        ("borehole-A", 1000.0, 50.0, 450.0),
        ("borehole-A", 500.0, 50.0, 240.0),
        ("surface-hi", 250.0, 50.0, 120.0),
        ("surface-hi", 200.0, 50.0, 99.0),
    ],
)
def test_chirp_above_output_nyquist_is_rejected(
    signal_cfg: SignalConfig, profile: str, rate: float, f_lo: float, f_hi: float
) -> None:
    """Every input frequency is above the 50 Hz output Nyquist, so anything left would be alias."""
    t = np.arange(int(SEG_S * rate)) / rate
    chirp = sps.chirp(t, f0=f_lo, t1=SEG_S, f1=f_hi, method="linear")
    out, tmap = for_picking(Stream([make_trace(chirp, rate)]), profile, signal_cfg)

    assert len(out) == 1
    tr = out[0]
    assert tr.stats.sampling_rate == signal_cfg.preprocess.targetRateHz
    assert tr.data.dtype == np.float64
    assert tmap.is_identity
    level_db = 20.0 * np.log10(rms(interior(tr, edge_s(signal_cfg)).data) / rms(chirp))
    assert level_db < -40.0, f"{profile} at {rate} Hz leaves {level_db:.1f} dB above Nyquist"


@pytest.mark.smoke
def test_antialias_design_meets_spec_and_butterworth_alone_would_not(
    signal_cfg: SignalConfig,
) -> None:
    pcfg = signal_cfg.preprocess
    nyquist_out = pcfg.targetRateHz / 2.0
    stop_db = -2.0 * pcfg.antiAlias.stopbandAttenuationDb  # zero-phase: two passes
    for fs in (1000.0, 500.0, 200.0):  # borehole-A 1000 / 500 Hz, surface-hi 250 Hz x2, 200 Hz
        freqs = np.linspace(0.0, fs / 2.0, 8001)
        _, h = sps.sosfreqz(design_antialias(fs, pcfg), worN=freqs, fs=fs)
        two_pass_db = 20.0 * np.log10(np.maximum(np.abs(h) ** 2, 1e-300))
        stop = freqs >= pcfg.antiAlias.stopbandEdgeFraction * nyquist_out
        assert two_pass_db[stop].max() <= stop_db + 0.1
        assert two_pass_db[stop].max() < -40.0
        passband = freqs <= pcfg.antiAlias.passbandEdgeFraction * nyquist_out
        assert two_pass_db[passband].min() >= -2.0 * pcfg.antiAlias.passbandLossDb - 1e-6

    # Why the explicit stage exists: the 40 Hz, 4-corner zero-phase Butterworth by itself.
    prof = pcfg.profiles["borehole-A"]
    assert isinstance(prof, DecimateProfile)
    fs = 1000.0
    bw = sps.butter(prof.lowpassCorners, prof.lowpassHz, btype="lowpass", fs=fs, output="sos")
    _, h = sps.sosfreqz(bw, worN=np.array([nyquist_out, 60.0]), fs=fs)
    assert np.all(20.0 * np.log10(np.abs(h) ** 2) > -40.0)


# --- b. passband ---------------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("profile", "rate"),
    [
        ("surface-100", 100.0),
        ("surface-hi", 200.0),
        ("surface-hi", 250.0),
        ("borehole-A", 500.0),
        ("borehole-A", 1000.0),
    ],
)
def test_5hz_passes_with_unit_gain_and_zero_phase(
    signal_cfg: SignalConfig, profile: str, rate: float
) -> None:
    freq = 5.0
    t_rel = np.arange(int(SEG_S * rate)) / rate
    x = np.cos(2 * np.pi * freq * t_rel)
    trend = np.polyfit(t_rel, x, 1)  # what linear detrend removes
    out, _ = for_picking(Stream([make_trace(x, rate)]), profile, signal_cfg)

    mid = interior(out[0], edge_s(signal_cfg))
    tm = times(mid) - T0.timestamp
    expected = np.cos(2 * np.pi * freq * tm) - np.polyval(trend, tm)
    assert np.max(np.abs(mid.data - expected)) < 1e-3

    design = np.column_stack([np.cos(2 * np.pi * freq * tm), np.sin(2 * np.pi * freq * tm)])
    (a, b), *_ = np.linalg.lstsq(design, mid.data + np.polyval(trend, tm), rcond=None)
    assert abs(20.0 * np.log10(np.hypot(a, b))) < 1.0  # amplitude within 1 dB
    phase_shift_s = np.arctan2(-b, a) / (2 * np.pi * freq)
    assert abs(phase_shift_s) < 1e-5  # zero phase: under 10 us


# --- c. TimeMap (ACCEPTANCE) ---------------------------------------------------------------------


@pytest.mark.smoke
def test_borehole_b_onset_round_trips_within_1ms(signal_cfg: SignalConfig) -> None:
    rate = 1000.0
    n = int(SEG_S * rate)
    starts = [T0, T0 + SEG_S + GAP_S]
    onset_idx = [7321, 12345]
    traces = []
    true_onsets = []
    for k, (start, idx) in enumerate(zip(starts, onset_idx)):
        data = 1e-3 * noise(10 + k, n)
        data[idx] += 1.0  # impulsive onset
        traces.append(make_trace(data, rate, start=start))
        true_onsets.append(start.timestamp + idx / rate)
    st = Stream(traces=traces)

    out, tmap = for_picking(st, "borehole-B", signal_cfg)

    target = signal_cfg.preprocess.targetRateHz
    assert tmap.factor == pytest.approx(rate / target)
    assert tmap.anchor == pytest.approx(T0.timestamp, abs=1e-6)
    assert len(out) == 2
    for tr, start, t_true in zip(out, starts, true_onsets):
        assert tr.stats.sampling_rate == target
        assert tr.stats.npts == n  # relabelled, not resampled
        assert abs(tr.stats.starttime.timestamp - tmap.to_model(start.timestamp)) < 1e-6
        t_model = tr.stats.starttime.timestamp + np.argmax(np.abs(tr.data)) / target
        assert abs(tmap.to_real(t_model) - t_true) < 1e-3
    assert out[0].stats.endtime < out[1].stats.starttime  # gap kept, no overlap in model time


@pytest.mark.smoke
def test_timemap_round_trip_and_identity(signal_cfg: SignalConfig) -> None:
    anchor = T0.timestamp
    tmap = TimeMap(anchor=anchor, factor=10.0)
    t = anchor + np.array([-5.0, 0.0, 1.2345, 3600.0, 86400.0])
    np.testing.assert_allclose(tmap.to_real(tmap.to_model(t)), t, rtol=0, atol=1e-6)
    np.testing.assert_allclose(tmap.to_model(tmap.to_real(t)), t, rtol=0, atol=1e-6)
    assert tmap.to_model(anchor + 1.0) == pytest.approx(anchor + 10.0, abs=1e-6)
    assert isinstance(tmap.to_real(anchor + 1.0), float)

    rate = 1000.0
    st = Stream([make_trace(noise(3, int(SEG_S * rate)), rate)])
    _, ident = for_picking(st, "borehole-A", signal_cfg)
    assert ident.is_identity
    assert ident.to_real(anchor + 1.2345) == anchor + 1.2345
    np.testing.assert_array_equal(ident.to_real(t), t)

    with pytest.raises(ValueError):
        TimeMap(anchor=anchor, factor=0.0)
    with pytest.raises(ValueError):
        TimeMap(anchor=float("nan"), factor=1.0)


# --- d. gaps stay gaps (ACCEPTANCE) --------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("masked", [False, True], ids=["separate-traces", "masked-merged"])
@pytest.mark.parametrize("profile", ["borehole-A", "borehole-B"])
def test_gaps_stay_gaps_and_short_segments_drop(
    signal_cfg: SignalConfig,
    caplog: pytest.LogCaptureFixture,
    masked: bool,
    profile: str,
) -> None:
    rate = 1000.0
    short_s = signal_cfg.preprocess.minSegmentS / 2.0
    st = gapped_stream(rate, seed=20, masked=masked, short_s=short_s)
    before = snapshot(st)
    gaps = [(T0 + SEG_S, T0 + SEG_S + GAP_S), (T0 + 2 * SEG_S + GAP_S, T0 + 2 * (SEG_S + GAP_S))]

    caplog.set_level(logging.INFO, logger=LOGGER)
    out, tmap = for_picking(st, profile, signal_cfg)

    assert_unchanged(st, before)
    assert len(out) == 2  # two long segments; the short one is dropped
    assert "dropped 1 segments shorter than minSegmentS" in caplog.text
    for tr in out:
        assert not isinstance(tr.data, np.ma.MaskedArray)
        assert np.count_nonzero(tr.data == 0.0) == 0  # no zero-filled samples
        t_real = tmap.to_real(times(tr))
        for gap_start, gap_end in gaps:
            inside = (t_real > gap_start.timestamp) & (t_real < gap_end.timestamp)
            assert not inside.any(), "a sample landed inside a gap"
    first, second = sorted(out, key=lambda tr: tr.stats.starttime)
    assert tmap.to_real(first.stats.endtime.timestamp) <= gaps[0][0].timestamp + 1e-6
    assert tmap.to_real(second.stats.starttime.timestamp) >= gaps[0][1].timestamp - 1e-6


# --- e. channel renaming -------------------------------------------------------------------------


@pytest.mark.smoke
def test_horizontals_1_2_renamed_on_copy(
    signal_cfg: SignalConfig, caplog: pytest.LogCaptureFixture
) -> None:
    rate = 1000.0
    n = int(SEG_S * rate)
    st = Stream([make_trace(noise(30 + k, n), rate, channel=f"DP{c}") for k, c in enumerate("Z12")])
    before = snapshot(st)
    caplog.set_level(logging.INFO, logger=LOGGER)

    out, _ = for_picking(st, "borehole-A", signal_cfg)

    assert sorted(tr.stats.channel for tr in out) == ["DPE", "DPN", "DPZ"]
    assert [tr.stats.channel for tr in st] == ["DPZ", "DP1", "DP2"]
    assert_unchanged(st, before)
    assert "renamed 2 traces" in caplog.text


@pytest.mark.smoke
def test_rename_collision_and_non_model_component_raise(signal_cfg: SignalConfig) -> None:
    rate = 1000.0
    n = int(SEG_S * rate)
    clash = Stream([make_trace(noise(40, n), rate, channel=c) for c in ("DP1", "DPN")])
    with pytest.raises(ValueError, match="collides"):
        for_picking(clash, "borehole-A", signal_cfg)
    odd = Stream([make_trace(noise(41, n), rate, channel="DP3")])
    with pytest.raises(ValueError, match="components must be one of"):
        for_picking(odd, "borehole-A", signal_cfg)


# --- f. errors -----------------------------------------------------------------------------------


@pytest.mark.smoke
def test_errors_fail_loudly(signal_cfg: SignalConfig, raw_signal_yaml: dict) -> None:
    def one(rate: float) -> Stream:
        return Stream([make_trace(noise(50, int(SEG_S * rate)), rate)])

    with pytest.raises(ValueError, match="borehole-A"):  # names the valid profiles
        for_picking(one(1000.0), "borehole-C", signal_cfg)
    with pytest.raises(ValueError, match="M / L"):  # 233 Hz is not 100 Hz x M / L, L <= 2
        for_picking(one(233.0), "surface-hi", signal_cfg)
    for profile, rate in (("surface-100", 200.0), ("borehole-A", 250.0), ("borehole-B", 500.0)):
        with pytest.raises(ValueError, match="outside profile"):
            for_picking(one(rate), profile, signal_cfg)
    with pytest.raises(ValueError, match="empty"):
        for_picking(Stream(), "borehole-A", signal_cfg)

    raw = copy.deepcopy(raw_signal_yaml)
    raw["preprocess"]["profiles"]["borehole-B"]["maxRateHz"] = 2000.0
    wide = SignalConfig.model_validate(raw)
    mixed = one(1000.0) + Stream([make_trace(noise(51, 40000), 2000.0, channel="DP1")])
    with pytest.raises(ValueError, match="single input rate"):
        for_picking(mixed, "borehole-B", wide)


# --- g. display copy -----------------------------------------------------------------------------


@pytest.mark.smoke
def test_display_copy_keeps_gaps_and_returns_a_copy(caplog: pytest.LogCaptureFixture) -> None:
    rate = 100.0
    st = gapped_stream(rate, seed=60, masked=True)
    fragment = make_trace(noise(61, 20), rate, start=T0 + 2 * (SEG_S + GAP_S))
    st += Stream([fragment])  # too short for the zero-phase filter's edge padding
    before = snapshot(st)
    caplog.set_level(logging.INFO, logger=LOGGER)

    out = display_copy(st, (1.0, 20.0))

    assert_unchanged(st, before)
    assert len(out) == 2
    assert "dropped_short=1" in caplog.text
    for tr in out:
        assert tr.data.dtype == np.float64
        assert not any(tr is orig or tr.data is orig.data for orig in st)
        t = times(tr)
        assert not ((t > (T0 + SEG_S).timestamp) & (t < (T0 + SEG_S + GAP_S).timestamp)).any()

    freq = 5.0
    t_rel = np.arange(int(SEG_S * rate)) / rate
    sine = make_trace(np.cos(2 * np.pi * freq * t_rel), rate)
    mid = interior(display_copy(Stream([sine]), (1.0, 20.0))[0], 2.0)
    assert abs(20.0 * np.log10(rms(mid.data) / np.sqrt(0.5))) < 1.0

    with pytest.raises(ValueError, match="Nyquist"):
        display_copy(st, (1.0, 60.0))
    with pytest.raises(ValueError, match="low < high"):
        display_copy(st, (20.0, 1.0))


# --- h. config -----------------------------------------------------------------------------------


@pytest.mark.smoke
def test_preprocess_block_parses_with_the_lane_profiles(signal_cfg: SignalConfig) -> None:
    pcfg = signal_cfg.preprocess
    expected = {
        "surface-100": PassthroughProfile,
        "surface-hi": DecimateProfile,
        "borehole-A": DecimateProfile,
        "borehole-B": StretchProfile,
    }
    for name, kind in expected.items():
        assert isinstance(pcfg.profiles[name], kind), name
    assert set(pcfg.componentRename.values()) <= set(pcfg.modelComponents)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("antiAlias", "passbandEdgeFraction"), 1.0),
        (("profiles", "borehole-B", "bandpassHz"), [5.0, 600.0]),
        (("profiles", "surface-hi", "lowpassHz"), 60.0),
        (("profiles", "surface-100", "maxRateHz"), 200.0),
        (("profiles", "borehole-A", "notAKnob"), 1),
        (("profiles", "borehole-A", "method"), "resample"),
        (("componentRename",), {"1": "X"}),
        (("componentRename",), {"1": "N", "2": "N"}),
    ],
)
def test_bad_preprocess_config_is_rejected(
    raw_signal_yaml: dict, path: tuple[str, ...], value: object
) -> None:
    raw = copy.deepcopy(raw_signal_yaml)
    node = raw["preprocess"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValidationError):
        SignalConfig.model_validate(raw)

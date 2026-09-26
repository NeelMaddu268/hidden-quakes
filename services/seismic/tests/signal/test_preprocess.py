"""SEIS-03 preprocessing profiles. Offline, seeded, synthetic signals built inside each test."""

import copy
import logging
from typing import Any

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
from hq.preprocess import TimeMap, display_copy, for_picking, profiles
from hq.preprocess.profiles import design_antialias

LOGGER = "hq.preprocess.profiles"
T0 = UTCDateTime(1_800_000_000.1234)  # arbitrary synthetic start, deliberately off the 10 ms grid
ON_GRID = UTCDateTime(1_800_000_000)  # on every sample grid used here
SEG_S = 40.0  # synthetic segment length, longer than minSegmentModelS in signal.yaml
GAP_S = 5.0
PASS_THROUGH_SOS = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])  # a "filter" that changes nothing
DECIMATE_CASES = [
    ("borehole-A", 1000.0),
    ("borehole-A", 500.0),
    ("surface-hi", 250.0),  # 100 Hz x 5 / 2: zero-stuffed x2, then decimated by 5
    ("surface-hi", 200.0),
]
ALL_PROFILE_CASES = [
    ("surface-100", 100.0),
    ("surface-hi", 250.0),
    ("borehole-A", 1000.0),
    ("borehole-B", 1000.0),
]


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
    out: np.ndarray = np.random.default_rng(seed).standard_normal(n)
    return out


def times(tr: Trace) -> np.ndarray:
    out: np.ndarray = (
        tr.stats.starttime.timestamp + np.arange(tr.stats.npts) / tr.stats.sampling_rate
    )
    return out


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def interior(tr: Trace, edge_s: float) -> Trace:
    """The part of a trace at least ``edge_s`` from either end (clear of taper and filter edges)."""
    return tr.slice(tr.stats.starttime + edge_s, tr.stats.endtime - edge_s)


def edge_s(cfg: SignalConfig) -> float:
    return 2.0 * cfg.preprocess.taper.maxLengthS


def output_nyquist(cfg: SignalConfig) -> float:
    return cfg.preprocess.targetRateHz / 2.0


def snapshot(st: Stream) -> list[tuple[dict[str, Any], np.ndarray]]:
    return [(copy.deepcopy(dict(tr.stats)), tr.data.copy()) for tr in st]


def assert_unchanged(st: Stream, before: list[tuple[dict[str, Any], np.ndarray]]) -> None:
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


def v_chirp(rate: float, f_mid: float, f_edge: float, duration_s: float) -> np.ndarray:
    """Unit linear sweep f_edge -> f_mid -> f_edge. The band just above the output Nyquist, where
    folding is worst, sits mid-trace, clear of the taper and the filter edges."""
    t = np.arange(int(duration_s * rate)) / rate
    freq = f_mid + (f_edge - f_mid) * np.abs(2.0 * t / duration_s - 1.0)
    return np.cos(2.0 * np.pi * np.cumsum(freq) / rate)


def alias_spectrum_db(cfg: SignalConfig, profile: str, rate: float) -> float:
    """Worst 1 Hz bin of the output PSD, in dB relative to the input chirp's in-band PSD.

    Every input frequency lies above the output Nyquist, so every output bin is alias. PSDs per
    Hz are comparable across rates; the ratio bounds the chain's power response at each folded
    input frequency.
    """
    f_mid = 1.01 * output_nyquist(cfg)
    f_edge = 0.9 * rate / 2.0
    x = v_chirp(rate, f_mid, f_edge, SEG_S)
    out, tmap = for_picking(Stream([make_trace(x, rate)]), profile, cfg)
    assert len(out) == 1 and tmap.is_identity
    tr = out[0]
    assert tr.stats.sampling_rate == cfg.preprocess.targetRateHz
    assert tr.data.dtype == np.float64

    skip = int(edge_s(cfg) * rate)
    f_in, p_in = sps.welch(x[skip:-skip], fs=rate, nperseg=int(rate))
    in_band = (f_in > f_mid + 1.0) & (f_in < f_edge - 1.0)
    fs_out = tr.stats.sampling_rate
    _, p_out = sps.welch(interior(tr, edge_s(cfg)).data, fs=fs_out, nperseg=int(fs_out))
    return float(10.0 * np.log10(p_out.max() / np.median(p_in[in_band])))


def folded_tone_db(cfg: SignalConfig, profile: str, rate: float, f_in: float) -> float:
    """Amplitude of a unit tone's folded image in the output, dB, fitted at the alias frequency."""
    t = np.arange(int(SEG_S * rate)) / rate
    out, _ = for_picking(
        Stream([make_trace(np.cos(2 * np.pi * f_in * t + 0.3), rate)]), profile, cfg
    )
    tr = out[0]
    target = tr.stats.sampling_rate
    f_alias = abs(f_in - target * round(f_in / target))
    mid = interior(tr, edge_s(cfg))
    tm = times(mid) - T0.timestamp
    design = np.column_stack(
        [
            np.cos(2 * np.pi * f_alias * tm),
            np.sin(2 * np.pi * f_alias * tm),
            np.ones_like(tm),  # detrend leftovers are not alias
            tm,
        ]
    )
    coef, *_ = np.linalg.lstsq(design, mid.data, rcond=None)
    return float(20.0 * np.log10(np.hypot(coef[0], coef[1])))


# --- a. aliasing (ACCEPTANCE) --------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(("profile", "rate"), DECIMATE_CASES)
def test_chirp_spectrum_above_output_nyquist_stays_below_40db(
    signal_cfg: SignalConfig, profile: str, rate: float
) -> None:
    """Spectrum test: a chirp that lives entirely above the output Nyquist leaves < -40 dB in
    every output bin."""
    level_db = alias_spectrum_db(signal_cfg, profile, rate)
    assert level_db < -40.0, f"{profile} at {rate} Hz: worst output bin {level_db:.1f} dB"


@pytest.mark.smoke
@pytest.mark.parametrize(("profile", "rate"), DECIMATE_CASES)
def test_tones_just_above_output_nyquist_do_not_fold(
    signal_cfg: SignalConfig, profile: str, rate: float
) -> None:
    for above in (1.01, 1.1, 1.2):  # 50.5, 55, 60 Hz at a 100 Hz target
        f_in = above * output_nyquist(signal_cfg)
        level_db = folded_tone_db(signal_cfg, profile, rate, f_in)
        assert level_db < -40.0, f"{profile} at {rate} Hz: {f_in} Hz folds at {level_db:.1f} dB"


@pytest.mark.smoke
@pytest.mark.parametrize(("profile", "rate"), DECIMATE_CASES)
def test_alias_checks_fail_without_the_antialias_stage(
    signal_cfg: SignalConfig, monkeypatch: pytest.MonkeyPatch, profile: str, rate: float
) -> None:
    """The two measurements above can fail: with the explicit stage gone, both see folding."""
    monkeypatch.setattr(profiles, "design_antialias", lambda fs_hz, cfg: PASS_THROUGH_SOS)
    f_in = 1.01 * output_nyquist(signal_cfg)
    assert folded_tone_db(signal_cfg, profile, rate, f_in) > -40.0
    assert alias_spectrum_db(signal_cfg, profile, rate) > -40.0


@pytest.mark.smoke
def test_antialias_design_meets_spec_and_butterworth_alone_would_not(
    signal_cfg: SignalConfig,
) -> None:
    pcfg = signal_cfg.preprocess
    nyquist_out = output_nyquist(signal_cfg)
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
    _, h = sps.sosfreqz(bw, worN=np.array([nyquist_out, 1.2 * nyquist_out]), fs=fs)
    assert np.all(20.0 * np.log10(np.abs(h) ** 2) > -40.0)


# --- b. passband and timing ----------------------------------------------------------------------


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


@pytest.mark.smoke
@pytest.mark.parametrize(("profile", "rate"), DECIMATE_CASES)
def test_decimated_components_share_the_absolute_output_grid(
    signal_cfg: SignalConfig, profile: str, rate: float
) -> None:
    """SeisBench snaps each trace start to a multiple of 1/target since the epoch, so the kept
    samples must be the ones nearest that grid, with the same residual on every component."""
    period_ns = round(1e9 / signal_cfg.preprocess.targetRateHz)
    n = int(SEG_S * rate)
    for base, bound_ns in ((ON_GRID, 1), (T0, 0.5e9 / rate + 1)):
        # Z/N/E on one input sample grid but at different phases of the output grid.
        st = Stream(
            [
                make_trace(noise(70 + k, n), rate, channel=f"HH{c}", start=base + 3 * k / rate)
                for k, c in enumerate("ZNE")
            ]
        )
        out, _ = for_picking(st, profile, signal_cfg)
        assert len(out) == 3
        residual_ns = [
            (tr.stats.starttime.ns + period_ns // 2) % period_ns - period_ns // 2 for tr in out
        ]
        assert max(abs(r) for r in residual_ns) <= bound_ns, residual_ns
        assert max(residual_ns) - min(residual_ns) <= 1, residual_ns  # no skew between components
        for tr, orig in zip(sorted(out, key=lambda tr: tr.id), sorted(st, key=lambda tr: tr.id)):
            lead_s = tr.stats.starttime - orig.stats.starttime
            assert 0.0 <= lead_s < 1.0 / signal_cfg.preprocess.targetRateHz


# --- c. TimeMap (ACCEPTANCE) ---------------------------------------------------------------------


@pytest.mark.smoke
def test_borehole_b_onset_round_trips_within_1ms(signal_cfg: SignalConfig) -> None:
    rate = 1000.0
    n = int(SEG_S * rate)
    starts = [T0, T0 + SEG_S + GAP_S]
    onset_idx = [7321, 32345]
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
def test_timemap_round_trip_and_identity(
    signal_cfg: SignalConfig, raw_signal_yaml: dict[str, Any]
) -> None:
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

    # A stretch is a relabel, so any rate works: the factor is simply rate / target.
    raw = copy.deepcopy(raw_signal_yaml)
    raw["preprocess"]["profiles"]["borehole-B"].update(minRateHz=900.0, maxRateHz=1100.0)
    odd_rate = 1000.3
    st_odd = Stream([make_trace(noise(4, int(SEG_S * odd_rate)), odd_rate)])
    _, odd = for_picking(st_odd, "borehole-B", SignalConfig.model_validate(raw))
    assert odd.factor == pytest.approx(odd_rate / signal_cfg.preprocess.targetRateHz, rel=1e-12)

    with pytest.raises(ValueError):
        TimeMap(anchor=anchor, factor=0.0)
    with pytest.raises(ValueError):
        TimeMap(anchor=float("nan"), factor=1.0)


# --- d. gaps stay gaps (ACCEPTANCE) --------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("masked", [False, True], ids=["separate-traces", "masked-merged"])
@pytest.mark.parametrize(("profile", "rate"), ALL_PROFILE_CASES)
def test_gaps_stay_gaps_and_short_segments_drop(
    signal_cfg: SignalConfig,
    caplog: pytest.LogCaptureFixture,
    masked: bool,
    profile: str,
    rate: float,
) -> None:
    pcfg = signal_cfg.preprocess
    factor = rate / pcfg.targetRateHz if profile == "borehole-B" else 1.0
    short_s = pcfg.minSegmentModelS / factor / 2.0  # real seconds: half the model-time minimum
    st = gapped_stream(rate, seed=20, masked=masked, short_s=short_s)
    before = snapshot(st)
    segments = [
        (T0, T0 + SEG_S - 1.0 / rate),
        (T0 + SEG_S + GAP_S, T0 + 2 * SEG_S + GAP_S - 1 / rate),
    ]
    gaps = [(T0 + SEG_S, T0 + SEG_S + GAP_S), (T0 + 2 * SEG_S + GAP_S, T0 + 2 * (SEG_S + GAP_S))]

    caplog.set_level(logging.INFO, logger=LOGGER)
    out, tmap = for_picking(st, profile, signal_cfg)

    assert_unchanged(st, before)
    assert len(out) == 2  # two long segments; the short one is dropped
    assert "dropped 1 segments shorter than minSegmentModelS" in caplog.text
    step_s = 1.0 / pcfg.targetRateHz  # one output sample, in real time for every profile here
    for tr, (seg_start, seg_end) in zip(sorted(out, key=lambda tr: tr.stats.starttime), segments):
        assert not isinstance(tr.data, np.ma.MaskedArray)
        assert tr.stats.sampling_rate == pcfg.targetRateHz
        # No zero-filled samples: exact zeros only where the taper pins the two end samples.
        assert set(np.flatnonzero(tr.data == 0.0).tolist()) <= {0, tr.stats.npts - 1}
        t_real = tmap.to_real(times(tr))
        for gap_start, gap_end in gaps:
            inside = (t_real > gap_start.timestamp) & (t_real < gap_end.timestamp)
            assert not inside.any(), "a sample landed inside a gap"
        # The segment is covered edge to edge: nothing extended, nothing lost past one sample.
        assert -1e-6 <= t_real[0] - seg_start.timestamp < step_s
        assert -1e-6 <= seg_end.timestamp - t_real[-1] < step_s


@pytest.mark.smoke
def test_abutting_pieces_join_and_conflicting_overlaps_raise(
    signal_cfg: SignalConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """File or record boundaries must not become tapered fake gaps."""
    rate = 1000.0
    n = int(SEG_S * rate)
    x = np.round(1000.0 * noise(80, 2 * n)).astype(np.int32)
    whole = Stream([make_trace(x, rate)])
    pieces = Stream(
        [
            make_trace(x[:n], rate),
            make_trace(x[n:], rate, start=T0 + n / rate),  # abuts the first piece
            make_trace(x[n - 500 : n + 500], rate, start=T0 + (n - 500) / rate),  # repeated record
        ]
    )
    before = snapshot(pieces)
    expected, _ = for_picking(whole, "borehole-A", signal_cfg)
    caplog.set_level(logging.INFO, logger=LOGGER)

    joined, _ = for_picking(pieces, "borehole-A", signal_cfg)

    assert_unchanged(pieces, before)
    assert "joined=2" in caplog.text
    assert len(joined) == 1
    assert joined[0].stats.starttime == expected[0].stats.starttime
    np.testing.assert_array_equal(joined[0].data, expected[0].data)

    clash = pieces.copy()
    clash[2].data = clash[2].data + 1  # same times, different samples
    with pytest.raises(ValueError, match="overlapping pieces with different samples"):
        for_picking(clash, "borehole-A", signal_cfg)
    mixed = Stream(
        [make_trace(x[:n], rate), make_trace(x[: n // 2], rate / 2, start=T0 + SEG_S + GAP_S)]
    )
    with pytest.raises(ValueError, match="different rates"):
        for_picking(mixed, "borehole-A", signal_cfg)


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
def test_errors_fail_loudly(signal_cfg: SignalConfig, raw_signal_yaml: dict[str, Any]) -> None:
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
    mixed = one(1000.0) + Stream([make_trace(noise(51, 80000), 2000.0, channel="DP1")])
    with pytest.raises(ValueError, match="single input rate"):
        for_picking(mixed, "borehole-B", wide)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("profile", "rate"), [("surface-hi", 250.0), ("borehole-A", 1000.0), ("borehole-B", 1000.0)]
)
def test_segments_too_short_for_the_filters_are_dropped_not_crashed(
    raw_signal_yaml: dict[str, Any], caplog: pytest.LogCaptureFixture, profile: str, rate: float
) -> None:
    raw = copy.deepcopy(raw_signal_yaml)
    raw["preprocess"]["minSegmentModelS"] = 1e-3  # lets a tiny fragment past the length rule
    cfg = SignalConfig.model_validate(raw)
    st = Stream(
        [
            make_trace(noise(90, 12), rate),  # shorter than any zero-phase filter's edge padding
            make_trace(noise(91, int(SEG_S * rate)), rate, start=T0 + 10.0),
        ]
    )
    caplog.set_level(logging.INFO, logger=LOGGER)

    out, _ = for_picking(st, profile, cfg)

    assert len(out) == 1
    assert "dropped 1 segments" in caplog.text


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

    half = sine.stats.npts // 2
    abutting = Stream(
        [
            make_trace(sine.data[:half], rate),
            make_trace(sine.data[half:], rate, start=T0 + half / rate),
        ]
    )
    np.testing.assert_array_equal(
        display_copy(abutting, (1.0, 20.0))[0].data,
        display_copy(Stream([sine]), (1.0, 20.0))[0].data,
    )

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
        (("joinMisalignmentSamples",), 0.5),
        (("minSegmentModelS",), 0.0),
    ],
)
def test_bad_preprocess_config_is_rejected(
    raw_signal_yaml: dict[str, Any], path: tuple[str, ...], value: object
) -> None:
    raw = copy.deepcopy(raw_signal_yaml)
    node = raw["preprocess"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValidationError):
        SignalConfig.model_validate(raw)

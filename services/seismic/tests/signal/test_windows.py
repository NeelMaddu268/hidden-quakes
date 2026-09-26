"""SEIS-02 known-event windows: selection, window bounds, gap accounting, windows.json, PASS/FAIL.

Offline and seeded. ``read_window`` (SEIS-05) is replaced by a fake that returns segmented
streams the way the real cache does: gaps as separate traces, never zero-filled, trimmed to the
window. The download manifests are replaced by a fake that lists the fetched spans, or by a fake
``hq.ingest.cache`` module plus manifest files on disk. Tests that need ``hq_contracts.io``
(CONTRACT-01) skip until it lands.
"""

import json
import sys
import types
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
import pytest
from obspy.geodetics import gps2dist_azimuth
from pydantic import ValidationError

from hq.config.run import RunSection
from hq.config.signal import KnownEventsConfig, SignalConfig
from hq.ingest.windows import (
    GAP_COLUMNS,
    NO_DATA,
    NOT_CACHED,
    NOT_DOWNLOADED,
    Gap,
    KnownWindows,
    KnownWindowsDoc,
    assess_windows,
    build_windows,
    coverage_gaps,
    event_passes,
    format_report,
    gap_file_name,
    load_windows,
    overall_pass,
    select_known_events,
    split_missing,
    write_outputs,
)

SEED = 20260910
RATE_HZ = 100.0
EVENT_LAT, EVENT_LON = 38.50, -112.88

# Segment layout per station, relative to the window start: channel -> [(start_s, end_s)].
Layout = dict[str, list[tuple[float, float]]]


# --- helpers ----------------------------------------------------------------------------------------


def _catalog(rows: list[dict]) -> pd.DataFrame:
    base = {
        "source": "test",
        "depthKm": 5.0,
        "depthDatum": "sea level",
        "elevM": -5000.0,
        "magType": "ml",
        "enu_e": 0.0,
        "enu_n": 0.0,
        "enu_u": 0.0,
        "matchedEventId": None,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def _stations(
    rows: list[tuple[str, float, float, list[str]]], used: bool = True, location: str = ""
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": sid,
                "network": sid.split(".")[0],
                "station": sid.split(".")[1],
                "location": location,
                "latitude": lat,
                "longitude": lon,
                "channels": chans,
                "usedInRun": used,
            }
            for sid, lat, lon, chans in rows
        ]
    )


def _trace_n(
    station_id: str,
    channel: str,
    start: float,
    npts: int,
    seed: int,
    rate: float = RATE_HZ,
    location: str = "",
) -> obspy.Trace:
    net, sta = station_id.split(".")[:2]
    data = np.random.default_rng(seed).normal(size=npts).astype(np.float64)
    tr = obspy.Trace(data=data)
    tr.stats.network, tr.stats.station = net, sta
    tr.stats.location, tr.stats.channel = location, channel
    tr.stats.sampling_rate = rate
    tr.stats.starttime = obspy.UTCDateTime(start)
    return tr


def _trace(
    station_id: str, channel: str, start: float, end: float, seed: int, rate: float = RATE_HZ
) -> obspy.Trace:
    return _trace_n(station_id, channel, start, round((end - start) * rate), seed, rate)


def _net_sta(station_id: str) -> str:
    return ".".join(station_id.split(".")[:2])


class FakeCache:
    """Stand-in for ``hq.ingest.cache.read_window``; records every call."""

    def __init__(self, layouts: dict[str, Layout], missing: frozenset[str] = frozenset()):
        self.layouts = layouts
        self.missing = missing
        self.calls: list[tuple[str, float, float, Path]] = []

    def __call__(self, station_id: str, t0: float, t1: float, *, cache_dir: Path) -> obspy.Stream:
        self.calls.append((station_id, t0, t1, cache_dir))
        key = _net_sta(station_id)
        if key in self.missing:
            raise _NotCachedError(f"nothing cached for {station_id}")
        st = obspy.Stream()
        for k, (channel, segments) in enumerate(sorted(self.layouts.get(key, {}).items())):
            for j, (a, b) in enumerate(segments):
                st += _trace(key, channel, t0 + a, t0 + b, SEED + 100 * k + j)
        return st


class FakeManifests:
    """Stand-in for the download manifests: the whole window is fetched unless listed here."""

    def __init__(self, fetched: dict[str, list[tuple[float, float]]] | None = None):
        self.fetched = fetched or {}  # station id -> fetched spans relative to the window start

    def __call__(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        t0: float,
        t1: float,
        *,
        cache_dir: Path,
    ) -> list[tuple[float, float]]:
        spans = self.fetched.get(f"{network}.{station}", [(0.0, t1 - t0)])
        return [(t0 + a, t0 + b) for a, b in spans]


class _NotCachedError(LookupError):
    pass


def _cfg(signal_cfg: SignalConfig, **update: object) -> KnownEventsConfig:
    return signal_cfg.known.model_copy(update=update)


def _event(run: RunSection, cfg: KnownEventsConfig, offset_s: float = 36000.0) -> dict:
    t = run.window_start_s + offset_s
    return {
        "eventId": "ev1",
        "t": t,
        "latitude": EVENT_LAT,
        "longitude": EVENT_LON,
        "depthKm": 5.0,
        "mag": 2.1,
        "magType": "ml",
        "windowStart": t - cfg.preS,
        "windowEnd": t + cfg.postS,
    }


FULL = [(0.0, 600.0)]
ZNE = ["HHZ", "HHN", "HHE"]
STATIONS = [
    # id, lat, lon, channels  (placed at increasing distance from the event)
    ("XX.A01", 38.505, -112.88, ZNE),
    ("XX.A02", 38.52, -112.88, ZNE),
    ("XX.A03", 38.54, -112.88, ZNE),
    ("XX.A04", 38.56, -112.88, ZNE),
    ("XX.A05", 38.58, -112.88, ZNE),
    ("XX.B06", 38.60, -112.88, ["DPZ", "DP1", "DP2"]),
    ("XX.A07", 38.62, -112.88, ZNE),
    ("XX.A08", 38.64, -112.88, ZNE),
]
LAYOUTS: dict[str, Layout] = {
    # clean three-component data, with sub-sample jitter at the edges and between segments
    "XX.A01": {"HHZ": [(0.004, 600.0)], "HHN": [(0.0, 300.0), (300.003, 600.0)], "HHE": FULL},
    # a 90 s gap on N: gapFraction 0.15 <= 0.2, still usable
    "XX.A02": {"HHZ": FULL, "HHN": [(0.0, 100.0), (190.0, 600.0)], "HHE": FULL},
    # a 150 s gap on E plus 30 s missing at the end: gapFraction 0.30 > 0.2, not usable
    "XX.A03": {"HHZ": FULL, "HHN": FULL, "HHE": [(0.0, 200.0), (350.0, 570.0)]},
    # two components only: the absent E counts as gapFraction 1.0
    "XX.A04": {"HHZ": FULL, "HHN": FULL},
    # XX.A05: fetched, but no data at all (empty stream): an outage, not a cache miss
    # borehole 1/2 horizontals plus a stray strong-motion channel that is not selected
    "XX.B06": {"DPZ": FULL, "DP1": FULL, "DP2": FULL, "HNZ": FULL},
    # XX.A07: read_window raises the cache's not-cached error
    # the downloader has fetched only the first 300 s so far
    "XX.A08": {"HHZ": [(0.0, 300.0)], "HHN": [(0.0, 300.0)], "HHE": [(0.0, 300.0)]},
}
FETCHED = {"XX.A08": [(0.0, 300.0)]}


def _assess(signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path, **update: object):
    cfg = _cfg(signal_cfg, **update)
    cache = FakeCache(LAYOUTS, missing=frozenset({"XX.A07"}))
    result = assess_windows(
        [_event(run_section, cfg)],
        _stations(STATIONS),
        cfg,
        tmp_path,
        cache,
        fetched_spans=FakeManifests(FETCHED),
        not_cached_errors=(_NotCachedError,),
    )
    return cfg, cache, result


# --- config -----------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_known_block_is_a_ten_minute_window(signal_cfg: SignalConfig) -> None:
    cfg = signal_cfg.known
    assert cfg.preS + cfg.postS == pytest.approx(600.0)
    assert cfg.nEvents >= 1 and cfg.minStations >= 1 and cfg.minGapSamples >= 1.0


@pytest.mark.smoke
def test_known_block_rejects_unknown_keys(raw_signal_yaml: dict) -> None:
    raw = {**raw_signal_yaml, "known": {**raw_signal_yaml["known"], "notAKnob": 1}}
    with pytest.raises(ValidationError):
        SignalConfig.model_validate(raw)


# --- selection --------------------------------------------------------------------------------------


def _selection_catalog(run: RunSection) -> pd.DataFrame:
    t0, t1 = run.window_start_s, run.window_end_s
    lon, lat = run.origin.lon, run.origin.lat
    return _catalog(
        [
            {"id": "a", "t": t0 + 5000.0, "latitude": lat, "longitude": lon, "mag": 2.5},
            {"id": "b", "t": t0 + 9000.0, "latitude": lat, "longitude": lon, "mag": 3.0},
            {"id": "c", "t": t0 + 1000.0, "latitude": lat, "longitude": lon, "mag": 2.5},
            {"id": "d", "t": t0 + 2000.0, "latitude": lat, "longitude": lon, "mag": None},
            {"id": "e", "t": t0 - 10.0, "latitude": lat, "longitude": lon, "mag": 4.0},
            {"id": "f", "t": t1, "latitude": lat, "longitude": lon, "mag": 4.1},
            {
                "id": "g",
                "t": t0 + 3000.0,
                "latitude": lat,
                "longitude": run.bbox[2] + 0.1,
                "mag": 3.5,
            },
            {"id": "h", "t": t0 + 4000.0, "latitude": lat, "longitude": lon, "mag": 1.0},
        ]
    )


@pytest.mark.smoke
def test_select_top_n_by_magnitude_with_ties_and_missing(
    signal_cfg: SignalConfig, run_section: RunSection, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _cfg(signal_cfg, nEvents=3)
    caplog.set_level("INFO", logger="hq.ingest.windows")
    events = select_known_events(_selection_catalog(run_section), run_section, cfg)
    # b is largest; a and c tie at 2.5 and the earlier one (c) ranks first. d has no magnitude,
    # e/f are outside [windowStart, windowEnd), g is outside the bbox.
    assert [e["eventId"] for e in events] == ["b", "c", "a"]
    assert "2 outside the run window" in caplog.text
    assert "1 in window but outside the bbox" in caplog.text
    assert "1 without a magnitude" in caplog.text
    assert "other catalog event" not in caplog.text  # every window holds only its own event


@pytest.mark.smoke
def test_select_is_independent_of_row_order(
    signal_cfg: SignalConfig, run_section: RunSection
) -> None:
    cfg = signal_cfg.known
    catalog = _selection_catalog(run_section)
    shuffled = catalog.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    assert select_known_events(catalog, run_section, cfg) == select_known_events(
        shuffled, run_section, cfg
    )


@pytest.mark.smoke
def test_select_reports_fewer_events_than_asked(
    signal_cfg: SignalConfig, run_section: RunSection, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _cfg(signal_cfg, nEvents=10)
    events = select_known_events(_selection_catalog(run_section), run_section, cfg)
    assert [e["eventId"] for e in events] == ["b", "c", "a", "h"]
    assert "only 4 of nEvents 10" in caplog.text


@pytest.mark.smoke
def test_window_bounds_follow_config(signal_cfg: SignalConfig, run_section: RunSection) -> None:
    cfg = signal_cfg.known
    events = select_known_events(_selection_catalog(run_section), run_section, cfg)
    for e in events:
        assert e["windowStart"] == pytest.approx(e["t"] - cfg.preS)
        assert e["windowEnd"] == pytest.approx(e["t"] + cfg.postS)
        assert e["windowEnd"] - e["windowStart"] == pytest.approx(600.0)
        assert isinstance(e["t"], float) and isinstance(e["mag"], float)


@pytest.mark.smoke
def test_select_warns_about_other_events_inside_a_window(
    signal_cfg: SignalConfig, run_section: RunSection, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _cfg(signal_cfg, nEvents=2)
    t, lat, lon = (
        run_section.window_start_s + 5000.0,
        run_section.origin.lat,
        run_section.origin.lon,
    )
    catalog = _catalog(
        [
            {"id": "s0", "t": t, "latitude": lat, "longitude": lon, "mag": 2.0},
            {"id": "s1", "t": t + 60.0, "latitude": lat, "longitude": lon, "mag": 1.9},
            {"id": "s2", "t": t + 200.0, "latitude": lat, "longitude": lon, "mag": None},
            {"id": "far", "t": t + 50000.0, "latitude": lat, "longitude": lon, "mag": 1.0},
        ]
    )
    events = select_known_events(catalog, run_section, cfg)
    assert [e["eventId"] for e in events] == ["s0", "s1"]
    assert "event s0: 2 other catalog event(s) inside its window" in caplog.text
    assert "s1 (t+60.0 s, mag 1.90), s2 (t+200.0 s, mag none)" in caplog.text
    assert "event s1: 2 other catalog event(s) inside its window" in caplog.text


@pytest.mark.smoke
def test_select_warns_when_candidates_mix_magnitude_types(
    signal_cfg: SignalConfig, run_section: RunSection, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _cfg(signal_cfg, nEvents=2)
    t, lat, lon = run_section.window_start_s, run_section.origin.lat, run_section.origin.lon
    rows = [
        {
            "id": "x",
            "t": t + 1000.0,
            "latitude": lat,
            "longitude": lon,
            "mag": 2.0,
            "magType": "ML",
        },
        {
            "id": "y",
            "t": t + 3000.0,
            "latitude": lat,
            "longitude": lon,
            "mag": 1.9,
            "magType": "ml",
        },
        {
            "id": "z",
            "t": t + 5000.0,
            "latitude": lat,
            "longitude": lon,
            "mag": 1.8,
            "magType": "Ml",
        },
    ]
    select_known_events(_catalog(rows), run_section, cfg)
    assert "mix magnitude types" not in caplog.text  # case variants are one type
    caplog.clear()
    md = {
        "id": "w",
        "t": t + 7000.0,
        "latitude": lat,
        "longitude": lon,
        "mag": 1.7,
        "magType": "md",
    }
    events = select_known_events(_catalog([*rows, md]), run_section, cfg)
    assert "w" not in [e["eventId"] for e in events]  # the md event lost the cross-type ranking
    assert "candidates mix magnitude types {'md': 1, 'ml': 3}" in caplog.text


# --- coverage and gaps ------------------------------------------------------------------------------


@pytest.mark.smoke
def test_coverage_gaps_ignores_jitter_and_finds_real_gaps() -> None:
    t0, t1 = 1_000_000.0, 1_000_600.0
    traces = [
        _trace("XX.S", "HHZ", t0 + 0.004, t0 + 100.0, 1),  # sub-sample late start: not a gap
        _trace("XX.S", "HHZ", t0 + 100.003, t0 + 250.0, 2),  # 3 ms jitter: not a gap
        _trace("XX.S", "HHZ", t0 + 280.0, t0 + 590.0, 3),  # 30 s gap, then 10 s missing at the end
    ]
    gaps = coverage_gaps(traces, t0, t1, min_gap_samples=1.5)
    assert len(gaps) == 2
    assert gaps[0] == pytest.approx((t0 + 250.003, t0 + 280.0), abs=1e-6)
    assert gaps[1] == pytest.approx((t0 + 590.0, t1), abs=1e-6)
    assert coverage_gaps([], t0, t1, min_gap_samples=1.5) == [(t0, t1)]


@pytest.mark.smoke
def test_coverage_gaps_counts_samples_at_1khz() -> None:
    t0, t1, rate = 1_000_000.0, 1_000_060.0, 1000.0
    dt = 1.0 / rate
    # first sample 0.4 ms after t0 (a trimmed read), 30,000 samples, then 3 missing samples
    a = _trace_n("XX.B", "DPZ", t0 + 0.0004, 30_000, 1, rate)
    b = _trace_n("XX.B", "DPZ", t0 + 30.0004 + 3 * dt, 29_997, 2, rate)
    gaps = coverage_gaps([a, b], t0, t1, min_gap_samples=1.5)
    assert gaps == [pytest.approx((t0 + 30.0004, t0 + 30.0034), abs=1e-7)]

    # 0.3-sample misalignment between segments is jitter, not a gap
    c = _trace_n("XX.B", "DP1", t0, 30_000, 3, rate)
    d = _trace_n("XX.B", "DP1", t0 + 30.0 + 0.3 * dt, 30_000, 4, rate)
    assert coverage_gaps([c, d], t0, t1, min_gap_samples=1.5) == []

    # a single missing sample is a gap
    e = _trace_n("XX.B", "DP2", t0, 30_000, 5, rate)
    f = _trace_n("XX.B", "DP2", t0 + 30.0 + dt, 29_999, 6, rate)
    assert coverage_gaps([e, f], t0, t1, min_gap_samples=1.5) == [
        pytest.approx((t0 + 30.0, t0 + 30.0 + dt), abs=1e-7)
    ]


@pytest.mark.smoke
def test_coverage_gaps_one_missing_sample_at_20hz() -> None:
    t0, rate = 1_000_000.0, 20.0
    t1 = t0 + 60.0
    a = _trace_n("XX.L", "BHZ", t0, 600, 1, rate)
    b = _trace_n("XX.L", "BHZ", t0 + 30.05, 599, 2, rate)
    assert coverage_gaps([a, b], t0, t1, min_gap_samples=1.5) == [
        pytest.approx((t0 + 30.0, t0 + 30.05), abs=1e-7)
    ]


@pytest.mark.smoke
def test_coverage_gaps_overlaps_clipping_and_outside_traces() -> None:
    t0, t1 = 1_000_000.0, 1_000_600.0
    overlapping = [
        _trace("XX.O", "HHZ", t0, t0 + 300.0, 1),
        _trace("XX.O", "HHZ", t0 + 250.0, t0 + 400.0, 2),
        _trace("XX.O", "HHZ", t0 + 450.0, t0 + 600.0, 3),
    ]
    assert coverage_gaps(overlapping, t0, t1, 1.5) == [
        pytest.approx((t0 + 400.0, t0 + 450.0), abs=1e-6)
    ]
    beyond = [_trace("XX.O", "HHZ", t0 - 100.0, t1 + 100.0, 4)]
    assert coverage_gaps(beyond, t0, t1, 1.5) == []
    outside = [_trace("XX.O", "HHZ", t1 + 10.0, t1 + 100.0, 5)]
    assert coverage_gaps(outside, t0, t1, 1.5) == [(t0, t1)]


@pytest.mark.smoke
def test_split_missing_separates_data_gaps_from_unfetched_spans() -> None:
    missing = [(10.0, 20.0), (50.0, 80.0), (90.0, 100.0)]
    fetched = [(0.0, 30.0), (30.0, 60.0), (95.0, 100.0)]  # adjacent chunks merge
    data_gaps, never_fetched = split_missing(missing, fetched)
    assert data_gaps == [(10.0, 20.0), (50.0, 60.0), (95.0, 100.0)]
    assert never_fetched == [(60.0, 80.0), (90.0, 95.0)]
    assert split_missing(missing, []) == ([], missing)


@pytest.mark.smoke
def test_station_coverage_flags_and_reasons(
    signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path
) -> None:
    _, cache, result = _assess(signal_cfg, run_section, tmp_path)
    ev = result.doc["events"][0]
    by_id = {s["stationId"]: s for s in ev["stations"]}
    assert len(by_id) == len(STATIONS)

    assert by_id["XX.A01"] == {
        **by_id["XX.A01"],
        "components": 3,
        "gapFraction": 0.0,
        "usable": True,
        "reason": None,
    }
    assert by_id["XX.A02"]["gapFraction"] == pytest.approx(90.0 / 600.0, abs=1e-6)
    assert by_id["XX.A02"]["usable"] is True
    assert by_id["XX.A03"]["gapFraction"] == pytest.approx(180.0 / 600.0, abs=1e-6)
    assert by_id["XX.A03"]["usable"] is False
    assert by_id["XX.A03"]["reason"].startswith("gapFraction 0.300 > maxGapFraction")
    assert by_id["XX.A04"] == {
        **by_id["XX.A04"],
        "components": 2,
        "gapFraction": 1.0,  # the absent E channel
        "usable": False,
        "reason": "2 components (NZ); gapFraction 1.000 > maxGapFraction 0.200",
    }
    assert by_id["XX.A05"] == {
        **by_id["XX.A05"],
        "components": 0,
        "gapFraction": 1.0,
        "usable": False,
        "reason": NO_DATA,
    }
    assert by_id["XX.A07"] == {
        **by_id["XX.A07"],
        "components": 0,
        "gapFraction": 1.0,
        "usable": False,
        "reason": NOT_CACHED,
    }
    assert by_id["XX.A08"]["gapFraction"] == pytest.approx(0.5, abs=1e-6)
    assert by_id["XX.A08"]["usable"] is False
    assert by_id["XX.A08"]["reason"].startswith(
        f"{NOT_DOWNLOADED} (300.0 s of the window never fetched)"
    )
    assert by_id["XX.B06"]["components"] == 3 and by_id["XX.B06"]["usable"] is True

    # read_window gets NET.STA.LOC ids, the event window and the cache dir
    assert sorted(c[0] for c in cache.calls) == sorted(f"{sid}." for sid, *_ in STATIONS)
    assert {(c[1], c[2], c[3]) for c in cache.calls} == {
        (ev["windowStart"], ev["windowEnd"], tmp_path)
    }
    assert result.counts == {
        **result.counts,
        "stationWindows": len(STATIONS),
        "usable": 3,
        "notCached": 1,
        "notDownloaded": 1,
        "noData": 1,
        "droppedTraces": 1,  # the unselected HNZ on XX.B06
        "eventsPass": 0,
    }


@pytest.mark.smoke
def test_epicentral_distance_and_sort_order(
    signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path
) -> None:
    _, _, result = _assess(signal_cfg, run_section, tmp_path)
    stations = result.doc["events"][0]["stations"]
    dists = [s["epiDistM"] for s in stations]
    assert dists == sorted(dists)
    lat, lon = next((la, lo) for sid, la, lo, _ in STATIONS if sid == "XX.A03")
    expected = gps2dist_azimuth(EVENT_LAT, EVENT_LON, lat, lon)[0]
    got = next(s["epiDistM"] for s in stations if s["stationId"] == "XX.A03")
    assert got == pytest.approx(expected)


@pytest.mark.smoke
def test_gap_rows_hold_data_gaps_only(
    signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path
) -> None:
    _, _, result = _assess(signal_cfg, run_section, tmp_path)
    ev = result.doc["events"][0]
    w0, w1 = ev["windowStart"], ev["windowEnd"]
    rows: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for g in result.gaps["ev1"]:
        rows.setdefault((g.stationId, g.channel), []).append((g.gapStart - w0, g.gapEnd - w0))
    assert not any(sid == "XX.A01" for sid, _ in rows)  # jitter only
    assert rows[("XX.A02", "HHN")] == [pytest.approx((100.0, 190.0), abs=1e-6)]
    assert rows[("XX.A03", "HHE")] == [
        pytest.approx((200.0, 350.0), abs=1e-6),
        pytest.approx((570.0, 600.0), abs=1e-6),
    ]
    assert rows[("XX.A04", "HHE")] == [pytest.approx((0.0, w1 - w0))]
    assert {ch for s, ch in rows if s == "XX.A05"} == set(ZNE)  # fetched, no samples
    assert not any(sid in ("XX.A07", "XX.A08") for sid, _ in rows)  # cache state, not data
    assert not any(ch == "HNZ" for _, ch in rows)
    assert result.counts["gapRows"] == len(result.gaps["ev1"])


@pytest.mark.smoke
def test_other_location_codes_are_dropped_not_mixed(
    signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path
) -> None:
    cfg = signal_cfg.known
    calls: list[str] = []

    def two_locations(station_id: str, t0: float, t1: float, *, cache_dir: Path) -> obspy.Stream:
        calls.append(station_id)
        n = round((t1 - t0) * RATE_HZ)
        return obspy.Stream(
            [
                _trace_n("XX.A01", ch, t0, n, k, location=loc)
                for k, (ch, loc) in enumerate(
                    [("HHZ", "10"), ("HHN", "10"), ("HHE", "10"), ("HHN", "20")]
                )
            ]
        )

    result = assess_windows(
        [_event(run_section, cfg)],
        _stations(STATIONS[:1], location="10"),
        cfg,
        tmp_path,
        two_locations,
        fetched_spans=FakeManifests(),
    )
    entry = result.doc["events"][0]["stations"][0]
    assert calls == ["XX.A01.10"]
    assert entry["usable"] is True and entry["components"] == 3
    assert result.counts["droppedTraces"] == 1


@pytest.mark.smoke
def test_three_components_must_come_from_one_instrument(
    signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path
) -> None:
    cfg = signal_cfg.known
    rows = [
        ("XX.M1", 38.51, -112.88, ["HHZ", "HNN", "HNE"]),  # vertical and horizontals differ
        ("XX.M2", 38.52, -112.88, ["EHZ", "EHN", "EH1"]),  # N with 1 is not a pair
    ]
    layouts = {sid: {ch: FULL for ch in chans} for sid, _, _, chans in rows}
    doc = build_windows(
        [_event(run_section, cfg)],
        _stations(rows),
        cfg,
        tmp_path,
        FakeCache(layouts),
        fetched_spans=FakeManifests(),
    )
    for s in doc["events"][0]["stations"]:
        assert s["components"] == 3 and s["usable"] is False
        assert s["reason"].startswith("no Z + N/E or Z + 1/2 set from one instrument")


@pytest.mark.smoke
def test_used_in_run_filter(
    signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path
) -> None:
    cfg = signal_cfg.known
    stations = pd.concat([_stations(STATIONS[:2]), _stations(STATIONS[2:3], used=False)])
    doc = build_windows(
        [_event(run_section, cfg)],
        stations,
        cfg,
        tmp_path,
        FakeCache(LAYOUTS),
        fetched_spans=FakeManifests(),
    )
    ids = {s["stationId"] for s in doc["events"][0]["stations"]}
    assert ids == ({"XX.A01", "XX.A02"} if cfg.usedInRunOnly else {"XX.A01", "XX.A02", "XX.A03"})


# --- the default cache module (SEIS-05) -------------------------------------------------------------


class _CacheMissError(LookupError):
    pass


@dataclass(frozen=True)
class _ChannelDayKey:
    network: str
    station: str
    location: str
    channel: str
    day: str


def _manifest_path(cache_dir: Path, key: _ChannelDayKey) -> Path:
    stem = f"{key.network}.{key.station}.{key.location}.{key.channel}.{key.day}"
    return Path(cache_dir) / "mseed" / f"{stem}.json"


def _days_covering(t0: float, t1: float) -> list[str]:
    first = datetime.fromtimestamp(t0, UTC).date()
    last = datetime.fromtimestamp(t1, UTC).date()
    return [(first + timedelta(days=i)).strftime("%Y%m%d") for i in range((last - first).days + 1)]


def _fake_cache_module(layouts: dict[str, Layout], missing: frozenset[str]) -> types.ModuleType:
    """A stand-in ``hq.ingest.cache`` with SEIS-05's names: reader, miss error, manifest layout."""
    fake_reader = FakeCache(layouts)

    def read_window(station_id: str, t0: float, t1: float, *, cache_dir: Path) -> obspy.Stream:
        if _net_sta(station_id) in missing:
            raise _CacheMissError(f"nothing cached for station {station_id}")
        return fake_reader(station_id, t0, t1, cache_dir=cache_dir)

    module = types.ModuleType("hq.ingest.cache")
    module.read_window = read_window  # type: ignore[attr-defined]
    module.CacheMissError = _CacheMissError  # type: ignore[attr-defined]
    module.ChannelDayKey = _ChannelDayKey  # type: ignore[attr-defined]
    module.manifest_path = _manifest_path  # type: ignore[attr-defined]
    module.days_covering = _days_covering  # type: ignore[attr-defined]
    module.calls = fake_reader.calls  # type: ignore[attr-defined]
    return module


def _write_manifests(
    cache_dir: Path, station_id: str, chunks: list[dict], version: int = 1
) -> None:
    net, sta = station_id.split(".")
    day = _days_covering(chunks[0]["start"], chunks[0]["start"])[0]
    for cha in ZNE:
        path = _manifest_path(cache_dir, _ChannelDayKey(net, sta, "", cha, day))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": version, "chunks": chunks}), encoding="utf-8")


def _chunk(start: float, end: float, status: str = "ok") -> dict:
    return {"start": start, "end": end, "status": status, "samples": 1, "fetchedAt": end + 1e5}


@pytest.mark.smoke
def test_default_reader_and_manifests_come_from_the_cache_module(
    signal_cfg: SignalConfig,
    run_section: RunSection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = signal_cfg.known
    event = _event(run_section, cfg)
    t, w0, w1 = event["t"], event["windowStart"], event["windowEnd"]
    hour = 3600.0 * (w0 // 3600.0)
    layouts: dict[str, Layout] = {
        "XX.A01": {ch: FULL for ch in ZNE},
        "XX.A02": {ch: [(0.0, t - w0)] for ch in ZNE},  # fetched up to the origin time only
    }
    module = _fake_cache_module(layouts, missing=frozenset({"XX.A03"}))
    monkeypatch.setitem(sys.modules, "hq.ingest.cache", module)
    # hour chunks; the event window crosses the hour boundary
    _write_manifests(tmp_path, "XX.A01", [_chunk(hour, hour + 3600.0), _chunk(hour + 3600.0, w1)])
    _write_manifests(tmp_path, "XX.A02", [_chunk(hour, t)])  # the hour before the origin
    _write_manifests(
        tmp_path,
        "XX.A04",
        [_chunk(hour, hour + 3600.0, "nodata"), _chunk(hour + 3600.0, w1, "nodata")],
    )

    result = assess_windows([event], _stations(STATIONS[:4]), cfg, tmp_path)
    by_id = {s["stationId"]: s for s in result.doc["events"][0]["stations"]}
    assert by_id["XX.A01"]["usable"] is True
    assert by_id["XX.A02"]["reason"].startswith(
        f"{NOT_DOWNLOADED} ({w1 - t:.1f} s of the window never fetched)"
    )
    assert by_id["XX.A03"]["reason"] == NOT_CACHED
    assert by_id["XX.A04"]["reason"] == NO_DATA
    assert {g.stationId for g in result.gaps["ev1"]} == {"XX.A04"}
    assert sorted(c[0] for c in module.calls) == ["XX.A01.", "XX.A02.", "XX.A04."]


@pytest.mark.smoke
def test_unknown_manifest_version_fails_loudly(
    signal_cfg: SignalConfig,
    run_section: RunSection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = signal_cfg.known
    event = _event(run_section, cfg)
    monkeypatch.setitem(sys.modules, "hq.ingest.cache", _fake_cache_module({}, frozenset()))
    _write_manifests(tmp_path, "XX.A01", [_chunk(event["windowStart"], event["windowEnd"])], 2)
    with pytest.raises(ValueError, match="manifest version 2"):
        assess_windows([event], _stations(STATIONS[:1]), cfg, tmp_path)


@pytest.mark.smoke
def test_missing_cache_module_names_seis05(
    signal_cfg: SignalConfig,
    run_section: RunSection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = signal_cfg.known
    monkeypatch.setitem(sys.modules, "hq.ingest.cache", None)
    with pytest.raises(RuntimeError, match="SEIS-05"):
        build_windows([_event(run_section, cfg)], _stations(STATIONS[:1]), cfg, tmp_path)


# --- windows.json and gap CSVs ----------------------------------------------------------------------


@pytest.mark.smoke
def test_windows_json_schema_round_trip(
    signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path
) -> None:
    cfg, _, result = _assess(signal_cfg, run_section, tmp_path / "cache")
    known = tmp_path / "known"
    written = write_outputs(result, known)
    assert [p.name for p in written] == ["windows.json", "gaps_ev1.csv"]
    assert sorted(p.name for p in known.iterdir()) == ["gaps_ev1.csv", "windows.json"]

    raw = json.loads((known / "windows.json").read_text(encoding="utf-8"))
    assert set(raw) == {"events", "params"}
    assert raw["params"] == cfg.model_dump(mode="json")
    assert set(raw["events"][0]) == {
        "eventId",
        "t",
        "latitude",
        "longitude",
        "depthKm",
        "mag",
        "magType",
        "windowStart",
        "windowEnd",
        "stations",
    }
    assert set(raw["events"][0]["stations"][0]) == {
        "stationId",
        "components",
        "gapFraction",
        "epiDistM",
        "usable",
        "reason",
    }
    assert load_windows(known / "windows.json") == result.doc

    gaps = pd.read_csv(known / "gaps_ev1.csv")
    assert tuple(gaps.columns) == GAP_COLUMNS
    assert len(gaps) == len(result.gaps["ev1"])
    assert (gaps["gapEnd"] > gaps["gapStart"]).all()


@pytest.mark.smoke
def test_stale_gap_reports_are_removed(
    signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path
) -> None:
    _, _, result = _assess(signal_cfg, run_section, tmp_path / "cache")
    known = tmp_path / "known"
    known.mkdir()
    (known / "gaps_old.csv").write_text("stationId,channel,gapStart,gapEnd\n", encoding="utf-8")
    (known / "ab.csv").write_text("kept\n", encoding="utf-8")  # SEIS-04's file, not ours
    write_outputs(result, known)
    assert sorted(p.name for p in known.iterdir()) == ["ab.csv", "gaps_ev1.csv", "windows.json"]


@pytest.mark.smoke
def test_file_name_collision_writes_nothing(tmp_path: Path) -> None:
    doc = _doc_with_usable([[True], [True]])
    doc["events"][0]["eventId"], doc["events"][1]["eventId"] = "uu/1", "uu_1"
    result = KnownWindows(doc=doc, gaps={"uu/1": [Gap("XX.S00", "HHZ", 0.0, 1.0)]}, counts={})
    known = tmp_path / "known"
    with pytest.raises(ValueError, match="same file gaps_uu_1.csv"):
        write_outputs(result, known)
    assert not known.exists()


@pytest.mark.smoke
def test_windows_json_rejects_schema_drift(tmp_path: Path) -> None:
    bad = {"events": [{"eventId": "x"}], "params": {}}
    path = tmp_path / "windows.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_windows(path)
    with pytest.raises(ValidationError):
        KnownWindowsDoc.model_validate({"events": [], "params": {}, "extra": 1})


@pytest.mark.smoke
def test_gap_file_name_is_filesystem_safe() -> None:
    assert gap_file_name("uu60512345") == "gaps_uu60512345.csv"
    assert gap_file_name("quakeml:uu/60512345?x") == "gaps_quakeml_uu_60512345_x.csv"


# --- PASS / FAIL ------------------------------------------------------------------------------------


def _doc_with_usable(flags_per_event: list[list[bool]]) -> dict:
    events = []
    for i, flags in enumerate(flags_per_event):
        stations = [
            {
                "stationId": f"XX.S{j:02d}",
                "components": 3,
                "gapFraction": 0.0,
                "epiDistM": 1000.0 * j,
                "usable": ok,
                "reason": None if ok else "2 components (NZ)",
            }
            for j, ok in enumerate(flags)
        ]
        events.append(
            {
                "eventId": f"ev{i}",
                "t": 0.0,
                "latitude": 0.0,
                "longitude": 0.0,
                "depthKm": 1.0,
                "mag": 2.0,
                "magType": "ml",
                "windowStart": -120.0,
                "windowEnd": 480.0,
                "stations": stations,
            }
        )
    return {"events": events, "params": {}}


@pytest.mark.smoke
def test_pass_fail_against_min_stations(signal_cfg: SignalConfig) -> None:
    cfg = _cfg(signal_cfg, nEvents=2, minStations=8)
    doc = _doc_with_usable([[True] * 8 + [False] * 3, [True] * 7 + [False] * 5])
    assert event_passes(doc["events"][0], 8) is True
    assert event_passes(doc["events"][1], 8) is False
    assert overall_pass(doc, cfg) is False

    report = format_report(doc, cfg)
    assert "PASS ev0: 8 usable three-component stations (minStations 8)" in report
    assert "FAIL ev1: 7 usable three-component stations (minStations 8)" in report
    assert "OVERALL FAIL: 1 of 2 windows pass" in report
    assert "XX.S10" in report and "2 components (NZ)" in report

    passing = _doc_with_usable([[True] * 8, [True] * 9])
    assert overall_pass(passing, cfg) is True
    # every window passing is not enough when fewer than nEvents windows exist
    assert overall_pass(_doc_with_usable([[True] * 9]), cfg) is False


# --- end to end (needs CONTRACT-01's hq_contracts.io) -----------------------------------------------


@pytest.mark.smoke
def test_run_known_windows_end_to_end(fake_ctx, run_section: RunSection) -> None:
    io = pytest.importorskip("hq_contracts.io")
    from hq_contracts.models import CatalogEvent, Enu, Station

    from hq.ingest.windows import run_known_windows

    t = run_section.window_start_s + 36000.0
    zero = Enu(e=0.0, n=0.0, u=0.0)
    catalog = [
        CatalogEvent(
            id=f"ev{i}",
            source="test",
            t=t + 1000.0 * i,
            latitude=EVENT_LAT,
            longitude=EVENT_LON,
            depthKm=5.0,
            depthDatum="sea level",
            elevM=-5000.0,
            mag=2.0 + 0.1 * i,
            magType="ml",
            enu=zero,
        )
        for i in range(4)
    ]
    stations = [
        Station(
            id=sid,
            network="XX",
            station=sid.split(".")[1],
            latitude=lat,
            longitude=lon,
            surfaceElevM=1600.0,
            sensorDepthM=0.0,
            sensorElevM=1600.0,
            kind="surface",
            channels=chans,
            sampleRateHz=RATE_HZ,
            enu=zero,
            preprocessProfile="surface-100",
            usedInRun=True,
        )
        for sid, lat, lon, chans in STATIONS
    ]
    io.write_table(io.to_frame(catalog), fake_ctx.path("catalog.parquet"), "CatalogEvent")
    io.write_table(io.to_frame(stations), fake_ctx.path("stations.parquet"), "Station")

    doc = run_known_windows(
        fake_ctx,
        read_window=FakeCache(LAYOUTS, frozenset({"XX.A07"})),
        fetched_spans=FakeManifests(FETCHED),
        not_cached_errors=(_NotCachedError,),
    )
    cfg = fake_ctx.config.signal.known
    assert [e["eventId"] for e in doc["events"]] == ["ev3", "ev2", "ev1"][: cfg.nEvents]
    assert load_windows(fake_ctx.path("known") / "windows.json") == doc
    record = fake_ctx.records["known_windows"]
    assert record["counts"]["events"] == len(doc["events"])
    assert record["params"] == cfg.model_dump(mode="json")

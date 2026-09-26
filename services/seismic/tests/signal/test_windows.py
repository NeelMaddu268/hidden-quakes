"""SEIS-02 known-event windows: selection, window bounds, gap accounting, windows.json, PASS/FAIL.

Offline and seeded. ``read_window`` (SEIS-05) is replaced by a fake that returns segmented
streams the way the real cache does: gaps as separate traces, never zero-filled, trimmed to the
window. Tests that need ``hq_contracts.io`` (CONTRACT-01) skip until it lands.
"""

import json
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
    NOT_CACHED,
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


def _stations(rows: list[tuple[str, float, float, list[str]]], used: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": sid, "latitude": lat, "longitude": lon, "channels": chans, "usedInRun": used}
            for sid, lat, lon, chans in rows
        ]
    )


def _trace(station_id: str, channel: str, start: float, end: float, seed: int) -> obspy.Trace:
    net, sta = station_id.split(".")[:2]
    npts = round((end - start) * RATE_HZ)
    data = np.random.default_rng(seed).normal(size=npts).astype(np.float64)
    tr = obspy.Trace(data=data)
    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = net, sta, "", channel
    tr.stats.sampling_rate = RATE_HZ
    tr.stats.starttime = obspy.UTCDateTime(start)
    return tr


class FakeCache:
    """Stand-in for ``hq.ingest.cache.read_window``; records every call."""

    def __init__(self, layouts: dict[str, Layout], missing: frozenset[str] = frozenset()):
        self.layouts = layouts
        self.missing = missing
        self.calls: list[tuple[str, float, float, Path]] = []

    def __call__(self, station_id: str, t0: float, t1: float, *, cache_dir: Path) -> obspy.Stream:
        self.calls.append((station_id, t0, t1, cache_dir))
        if station_id in self.missing:
            raise _NotCachedError(f"nothing cached for {station_id}")
        st = obspy.Stream()
        for k, (channel, segments) in enumerate(sorted(self.layouts.get(station_id, {}).items())):
            for j, (a, b) in enumerate(segments):
                st += _trace(station_id, channel, t0 + a, t0 + b, SEED + 100 * k + j)
        return st


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
]
LAYOUTS: dict[str, Layout] = {
    # clean three-component data, with sub-sample jitter at the edges and between segments
    "XX.A01": {"HHZ": [(0.004, 600.0)], "HHN": [(0.0, 300.0), (300.003, 600.0)], "HHE": FULL},
    # a 90 s gap on N: gapFraction 0.15 <= 0.2, still usable
    "XX.A02": {"HHZ": FULL, "HHN": [(0.0, 100.0), (190.0, 600.0)], "HHE": FULL},
    # a 150 s gap on E plus 30 s missing at the end: gapFraction 0.30 > 0.2, not usable
    "XX.A03": {"HHZ": FULL, "HHN": FULL, "HHE": [(0.0, 200.0), (350.0, 570.0)]},
    # two components only
    "XX.A04": {"HHZ": FULL, "HHN": FULL},
    # XX.A05: no data at all (empty stream)
    # borehole 1/2 horizontals plus a stray strong-motion channel that is not selected
    "XX.B06": {"DPZ": FULL, "DP1": FULL, "DP2": FULL, "HNZ": FULL},
    # XX.A07: read_window raises the cache's not-cached error
}


def _assess(signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path, **update: object):
    cfg = _cfg(signal_cfg, **update)
    cache = FakeCache(LAYOUTS, missing=frozenset({"XX.A07"}))
    result = assess_windows(
        [_event(run_section, cfg)],
        _stations(STATIONS),
        cfg,
        tmp_path,
        cache,
        not_cached_errors=(_NotCachedError,),
    )
    return cfg, cache, result


# --- config -----------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_known_block_is_a_ten_minute_window(signal_cfg: SignalConfig) -> None:
    cfg = signal_cfg.known
    assert cfg.preS + cfg.postS == pytest.approx(600.0)
    assert cfg.nEvents >= 1 and cfg.minStations >= 1


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


# --- coverage and gaps ------------------------------------------------------------------------------


@pytest.mark.smoke
def test_coverage_gaps_ignores_jitter_and_finds_real_gaps() -> None:
    t0, t1 = 1_000_000.0, 1_000_600.0
    traces = [
        _trace("XX.S", "HHZ", t0 + 0.004, t0 + 100.0, 1),  # sub-sample late start: not a gap
        _trace("XX.S", "HHZ", t0 + 100.003, t0 + 250.0, 2),  # 3 ms jitter: not a gap
        _trace("XX.S", "HHZ", t0 + 280.0, t0 + 590.0, 3),  # 30 s gap, then 10 s missing at the end
    ]
    gaps = coverage_gaps(traces, t0, t1, min_gap_s=0.05)
    assert len(gaps) == 2
    assert gaps[0] == pytest.approx((t0 + 250.003, t0 + 280.0), abs=1e-6)
    assert gaps[1] == pytest.approx((t0 + 590.0, t1), abs=1e-6)
    assert coverage_gaps([], t0, t1, min_gap_s=0.05) == [(t0, t1)]


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
    assert by_id["XX.A04"]["components"] == 2 and by_id["XX.A04"]["usable"] is False
    assert by_id["XX.A04"]["reason"] == "2 components (NZ)"
    for sid in ("XX.A05", "XX.A07"):
        assert by_id[sid] == {
            **by_id[sid],
            "components": 0,
            "gapFraction": 1.0,
            "usable": False,
            "reason": NOT_CACHED,
        }
    assert by_id["XX.B06"]["components"] == 3 and by_id["XX.B06"]["usable"] is True

    # the window handed to read_window is the event window, and the cache dir is passed through
    assert {(c[1], c[2], c[3]) for c in cache.calls} == {
        (ev["windowStart"], ev["windowEnd"], tmp_path)
    }
    assert result.counts["notCached"] == 2
    assert result.counts["usable"] == 3
    assert result.counts["droppedTraces"] == 1  # the unselected HNZ on XX.B06


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
def test_gap_rows_cover_gaps_and_missing_channels(
    signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path
) -> None:
    _, _, result = _assess(signal_cfg, run_section, tmp_path)
    ev = result.doc["events"][0]
    w0, w1 = ev["windowStart"], ev["windowEnd"]
    rows = {(g.stationId, g.channel): [] for g in result.gaps["ev1"]}
    for g in result.gaps["ev1"]:
        rows[(g.stationId, g.channel)].append((g.gapStart - w0, g.gapEnd - w0))
    assert not any(sid == "XX.A01" for sid, _ in rows)  # jitter only
    assert rows[("XX.A02", "HHN")] == [pytest.approx((100.0, 190.0), abs=1e-6)]
    assert rows[("XX.A03", "HHE")] == [
        pytest.approx((200.0, 350.0), abs=1e-6),
        pytest.approx((570.0, 600.0), abs=1e-6),
    ]
    assert rows[("XX.A04", "HHE")] == [pytest.approx((0.0, w1 - w0))]
    for sid in ("XX.A05", "XX.A07"):
        assert {ch for s, ch in rows if s == sid} == set(ZNE)
    assert not any(ch == "HNZ" for _, ch in rows)


@pytest.mark.smoke
def test_mixed_location_codes_fail_loudly(
    signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path
) -> None:
    cfg = signal_cfg.known

    def two_locations(station_id: str, t0: float, t1: float, *, cache_dir: Path) -> obspy.Stream:
        a = _trace(station_id, "HHZ", t0, t1, 1)
        b = _trace(station_id, "HHN", t0, t1, 2)
        b.stats.location = "10"
        return obspy.Stream([a, b])

    with pytest.raises(ValueError, match="location codes"):
        build_windows(
            [_event(run_section, cfg)], _stations(STATIONS[:1]), cfg, tmp_path, two_locations
        )


@pytest.mark.smoke
def test_used_in_run_filter(
    signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path
) -> None:
    cfg = signal_cfg.known
    stations = pd.concat([_stations(STATIONS[:2]), _stations(STATIONS[2:3], used=False)])
    doc = build_windows([_event(run_section, cfg)], stations, cfg, tmp_path, FakeCache(LAYOUTS))
    ids = {s["stationId"] for s in doc["events"][0]["stations"]}
    assert ids == ({"XX.A01", "XX.A02"} if cfg.usedInRunOnly else {"XX.A01", "XX.A02", "XX.A03"})


# --- windows.json and gap CSVs ----------------------------------------------------------------------


@pytest.mark.smoke
def test_windows_json_schema_round_trip(
    signal_cfg: SignalConfig, run_section: RunSection, tmp_path: Path
) -> None:
    cfg, _, result = _assess(signal_cfg, run_section, tmp_path / "cache")
    known = tmp_path / "known"
    written = write_outputs(result, known)
    assert [p.name for p in written] == ["windows.json", "gaps_ev1.csv"]

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
        not_cached_errors=(_NotCachedError,),
    )
    cfg = fake_ctx.config.signal.known
    assert [e["eventId"] for e in doc["events"]] == ["ev3", "ev2", "ev1"][: cfg.nEvents]
    assert load_windows(fake_ctx.path("known") / "windows.json") == doc
    assert fake_ctx.records["known_windows"]["counts"]["events"] == len(doc["events"])

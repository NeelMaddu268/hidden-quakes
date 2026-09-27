"""SEIS-10 station-day helicorder. Offline, seeded, synthetic data built inside each test."""

import json
import re

import numpy as np
import pandas as pd
import pytest
from obspy import Stream, Trace, UTCDateTime
from pydantic import ValidationError

from hq.config.signal import HelicorderConfig, SignalConfig
from hq.ingest.cache import Segment
from hq.preprocess import helicorder as h

T0 = 999_993_600.0  # 2001-09-09 00:00 UTC: a synthetic day start on every sample grid used here
RATE_HZ = 200.0
SEED = 20260926
CONTRACT_KEYS = [
    "image",
    "widthPx",
    "heightPx",
    "title",
    "caption",
    "runId",
    "stationId",
    "seedId",
    "channel",
    "stationKind",
    "sensorDepthM",
    "dayUtc",
    "startUtc",
    "endUtc",
    "rowMinutes",
    "rows",
    "filterHz",
    "gapsFilled",
    "gapSeconds",
    "coverageFraction",
    "markerTime",
    "legend",
    "selection",
    "source",
    "generator",
]


@pytest.fixture()
def cfg(signal_cfg: SignalConfig) -> HelicorderConfig:
    return signal_cfg.helicorder


def stations_frame() -> pd.DataFrame:
    rows = [
        ("XX.B1", "borehole", True, ["GHZ", "GH1", "GH2"], 120.0),
        ("XX.B2", "borehole", True, ["GHZ", "GH1", "GH2"], 90.0),
        ("XX.B3", "borehole", False, ["GHZ", "GH1", "GH2"], 90.0),  # not used in the run
        ("XX.S1", "surface", True, ["HHZ", "HHN", "HHE"], 0.0),
        ("XX.B0", "borehole", True, ["HHZ", "HH1", "HH2"], None),
    ]
    return pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "network": [r[0].split(".")[0] for r in rows],
            "station": [r[0].split(".")[1] for r in rows],
            "location": ["" for _ in rows],
            "kind": [r[1] for r in rows],
            "usedInRun": [r[2] for r in rows],
            "channels": [r[3] for r in rows],
            "sensorDepthM": [r[4] for r in rows],
        }
    )


def picks_frame(counts: dict[tuple[str, str], int], t: float) -> pd.DataFrame:
    rows = [(sid, phase, t + k) for (sid, phase), n in counts.items() for k in range(n)]
    return pd.DataFrame(
        {
            "stationId": [r[0] for r in rows],
            "phase": [r[1] for r in rows],
            "t": [r[2] for r in rows],
        }
    )


@pytest.mark.smoke
def test_helicorder_section_parses_and_forbids_unknown_keys(
    signal_cfg: SignalConfig, raw_signal_yaml: dict
) -> None:
    cfg = signal_cfg.helicorder
    assert isinstance(cfg, HelicorderConfig)
    assert cfg.layout.widthPx == 1920  # the station-day contract
    assert cfg.scale.clipRows <= 1.5
    for path in ((), ("scale",), ("layout",), ("markers",), ("style",), ("style", "tierOpacity")):
        raw = json.loads(json.dumps(raw_signal_yaml))
        node = raw["helicorder"]
        for key in path:
            node = node[key]
        node["notAKnob"] = 1
        with pytest.raises(ValidationError):
            SignalConfig.model_validate(raw)
    raw = json.loads(json.dumps(raw_signal_yaml))
    del raw["helicorder"]["bandHz"]  # no Python defaults: every knob lives in YAML
    with pytest.raises(ValidationError):
        SignalConfig.model_validate(raw)
    raw = json.loads(json.dumps(raw_signal_yaml))
    raw["helicorder"]["rowMinutes"] = 45
    with pytest.raises(ValidationError):
        SignalConfig.model_validate(raw)
    raw = json.loads(json.dumps(raw_signal_yaml))
    raw["helicorder"]["style"]["candidate"] = "amber"
    with pytest.raises(ValidationError):
        SignalConfig.model_validate(raw)


@pytest.mark.smoke
def test_station_choice_counts_used_borehole_picks_with_id_ties(cfg: HelicorderConfig) -> None:
    stations = stations_frame()
    picks = picks_frame(
        {
            ("XX.B1", "P"): 3,
            ("XX.B1", "S"): 2,  # 5
            ("XX.B2", "P"): 5,  # 5: ties with B1, B1 wins by id order
            ("XX.B3", "P"): 50,  # not used in the run
            ("XX.S1", "P"): 40,  # surface
            ("XX.B0", "P"): 2,
            ("XX.B0", "X"): 30,  # a phase not counted
        },
        T0 + 10.0,
    )
    outside = picks_frame({("XX.B2", "P"): 9}, T0 + 86400.0)  # the next day
    choice = h.choose_station(stations, pd.concat([picks, outside]), T0, T0 + 86400.0, cfg)
    assert (choice.stationId, choice.pickCount) == ("XX.B1", 5)
    assert choice.channel == "GHZ"
    assert choice.seedId == "XX.B1..GHZ"
    assert choice.sensorDepthM == 120.0
    assert choice.runnersUp[: cfg.runnersUp] == (("XX.B2", 5), ("XX.B0", 2))[: cfg.runnersUp]
    with pytest.raises(ValueError, match="no picks"):
        h.choose_station(stations, picks.iloc[0:0], T0, T0 + 86400.0, cfg)


@pytest.mark.smoke
def test_day_window_and_row_positions() -> None:
    t0, t1, day = h.day_window(T0, T0 + 86400.0)
    assert day == "2001-09-09" and (t0, t1) == (T0, T0 + 86400.0)
    with pytest.raises(ValueError):
        h.day_window(T0 + 1.0, T0 + 86401.0)
    with pytest.raises(ValueError):
        h.day_window(T0, T0 + 3600.0)
    row_s = 1800.0
    assert h.n_rows(t0, t1, row_s) == 48
    times = np.array([T0, T0 + 1799.5, T0 + 1800.0, T0 + 86399.0])
    rows, into = h.row_position(times, T0, row_s)
    assert rows.tolist() == [0, 0, 1, 47]
    assert into.tolist() == [0.0, 1799.5, 0.0, 1799.0]


@pytest.mark.smoke
def test_markers_keep_the_day_and_split_by_tier() -> None:
    candidates = [
        (T0 - 1.0, "A"),  # the day before
        (T0, "A"),
        (T0 + 3600.0 + 90.0, "B"),
        (T0 + 86399.9, "C"),
        (T0 + 86400.0, "C"),  # the next day: half-open
        (T0 + 7200.0, "C"),
    ]
    public = [T0 - 5.0, T0 + 1800.0 + 30.0, T0 + 90000.0]
    m = h.select_markers(candidates, public, T0, T0 + 86400.0, 1800.0)
    assert {k: m.count(k) for k in h.LEGEND_KEYS} == {
        "tierA": 1,
        "tierB": 1,
        "tierC": 2,
        "public": 1,
    }
    assert m.rows["tierB"].tolist() == [2] and m.minutes["tierB"].tolist() == [1.5]
    assert m.rows["tierC"].tolist() == [4, 47]
    assert m.rows["public"].tolist() == [1] and m.minutes["public"].tolist() == [0.5]
    with pytest.raises(ValueError, match="tier"):
        h.select_markers([(T0, "D")], [], T0, T0 + 86400.0, 1800.0)


def noisy(n: int, rng: np.random.Generator, offset: float) -> np.ndarray:
    return rng.normal(0.0, 50.0, n) + offset


@pytest.mark.smoke
def test_gaps_stay_blank_and_segments_are_filtered_on_their_own(cfg: HelicorderConfig) -> None:
    rng = np.random.default_rng(SEED)
    row_s, ncols = 600.0, 600  # one column per second
    gap = (200.0, 320.0)
    first = Trace(noisy(round(gap[0] * RATE_HZ), rng, 1.0e6), header={"sampling_rate": RATE_HZ})
    first.stats.starttime = UTCDateTime(T0)
    first.stats.network, first.stats.station, first.stats.channel = "XX", "B1", "GHZ"
    second = first.copy()
    second.data = noisy(round((row_s - gap[1]) * RATE_HZ), rng, -1.0e6)
    second.stats.starttime = UTCDateTime(T0 + gap[1])
    st = Stream([first, second])

    result = h.row_envelope(st, T0, row_s, ncols, cfg)
    assert result.nSegments == 2 and result.nDropped == 0 and len(result.pieces) == 2
    drawn = np.concatenate([p.cols for p in result.pieces])
    in_gap = (drawn >= int(gap[0])) & (drawn < int(gap[1]))
    assert not in_gap.any(), "no column inside the gap may hold a sample"
    assert result.pieces[0].cols[-1] == int(gap[0]) - 1
    assert result.pieces[1].cols[0] == int(gap[1])

    # Each piece equals that segment processed alone: the opposite offsets never meet.
    for piece, tr in zip(result.pieces, (first, second), strict=True):
        alone = h.row_envelope(Stream([tr.copy()]), T0, row_s, ncols, cfg).pieces[0]
        np.testing.assert_array_equal(piece.cols, alone.cols)
        np.testing.assert_allclose(piece.hi, alone.hi)
        np.testing.assert_allclose(piece.lo, alone.lo)
        assert np.max(np.abs(piece.hi)) < 1.0e4  # detrended: the 1e6 offsets are gone

    segs = [
        Segment("GHZ", T0, T0 + gap[0] - 1 / RATE_HZ, 1 / RATE_HZ),
        Segment("GHZ", T0 + gap[1], T0 + row_s - 1 / RATE_HZ, 1 / RATE_HZ),
        Segment("GH1", T0, T0 + row_s, 1 / RATE_HZ),  # another channel never counts
    ]
    covered = h.covered_seconds(segs, "GHZ", T0, T0 + row_s)
    assert covered == pytest.approx(row_s - (gap[1] - gap[0]))
    assert h.covered_seconds(segs, "GHZ", T0 + 250.0, T0 + 300.0) == 0.0


@pytest.mark.smoke
def test_short_spikes_survive_the_column_envelope() -> None:
    data = np.zeros(round(60 * RATE_HZ))
    data[round(30.37 * RATE_HZ)] = 5.0  # one sample
    piece = h.column_envelope(T0, RATE_HZ, data, T0, 60.0, 10)
    assert piece is not None
    assert piece.cols.tolist() == list(range(10))
    assert piece.hi.max() == 5.0 and int(piece.cols[np.argmax(piece.hi)]) == 5
    assert h.column_envelope(T0 + 100.0, RATE_HZ, data, T0, 60.0, 10) is None


@pytest.mark.smoke
def test_scale_clips_and_keeps_quiet_columns_visible() -> None:
    piece = h.Piece(
        cols=np.arange(4),
        lo=np.array([-1.0, -0.0, -100.0, 0.0]),
        hi=np.array([1.0, 0.0, 100.0, 0.0]),
    )
    top, bottom = h.scaled_band(piece, 3, gain=0.1, clip_rows=1.2, min_rows=0.05)
    assert top[2] == pytest.approx(3 - 1.2) and bottom[2] == pytest.approx(3 + 1.2)
    assert bottom[1] - top[1] == pytest.approx(0.05)  # a flat column still draws
    ref = h.day_reference([h.RowResult(pieces=[piece])], 50.0)
    assert ref == pytest.approx(0.5)  # median of the column peaks 1, 0, 100, 0


def synthetic_manifest(cfg: HelicorderConfig) -> dict:
    choice = h.StationChoice("XX.B1", "GHZ", "XX.B1..GHZ", 120.0, 5, (("XX.B2", 5),))
    markers = h.select_markers(
        [(T0 + 5.0, "A"), (T0 + 9.0, "C")], [T0 + 5.5], T0, T0 + 86400.0, 1800.0
    )
    return h.build_manifest(
        cfg=cfg,
        run_id="synthetic-run",
        choice=choice,
        t0=T0,
        t1=T0 + 86400.0,
        day_utc="2001-09-09",
        rows=48,
        gap_s=120.0,
        coverage=1.0 - 120.0 / 86400.0,
        markers=markers,
        width_px=cfg.layout.widthPx,
        height_px=cfg.layout.heightPx,
    )


@pytest.mark.smoke
def test_manifest_follows_the_station_day_contract(cfg: HelicorderConfig) -> None:
    man = synthetic_manifest(cfg)
    assert list(man) == CONTRACT_KEYS
    assert man["image"] == f"{cfg.fileStem}.png" and man["widthPx"] == 1920
    assert man["startUtc"] == "2001-09-09T00:00:00Z" and man["endUtc"] == "2001-09-10T00:00:00Z"
    assert man["gapsFilled"] is False and man["markerTime"] == "origin"
    assert man["source"] == "EarthScope public waveforms"
    assert man["generator"] == "hq.preprocess.helicorder"
    assert man["filterHz"] == list(cfg.bandHz) and man["rowMinutes"] in (30, 60)
    assert [e["key"] for e in man["legend"]] == list(h.LEGEND_KEYS)
    for entry in man["legend"]:
        assert set(entry) == {"key", "label", "color", "opacity", "shape", "count"}
        assert re.fullmatch(r"#[0-9A-F]{6}", entry["color"])
    counts = {e["key"]: e["count"] for e in man["legend"]}
    assert counts == {"tierA": 1, "tierB": 0, "tierC": 1, "public": 1}
    shapes = {e["key"]: e["shape"] for e in man["legend"]}
    assert shapes == {"tierA": "tick", "tierB": "tick", "tierC": "tick", "public": "diamond"}
    opacity = {e["key"]: e["opacity"] for e in man["legend"]}
    assert opacity["tierA"] > opacity["tierB"] > opacity["tierC"]
    assert man["selection"] == {
        "rule": h.selection_rule(cfg),
        "pickCount": 5,
        "runnersUp": [{"stationId": "XX.B2", "pickCount": 5}],
    }
    rule = man["selection"]["rule"]  # reader-facing (the web panel shows it), built from config
    assert cfg.stationKind in rule and ".parquet" not in rule
    assert all(phase in rule for phase in cfg.pickPhases)
    caption = man["caption"]
    for part in ("XX.B1", "GHZ", "2001-09-09", "candidate events", "public-catalog events"):
        assert part in caption
    for banned in ("confirmed", "caused by", "predict"):
        assert banned not in caption.lower() and banned not in rule.lower()
    json.dumps(man, allow_nan=False)  # strict JSON


@pytest.mark.smoke
def test_png_is_exactly_the_configured_size(cfg: HelicorderConfig) -> None:
    rng = np.random.default_rng(SEED)
    ncols = h.plot_columns(cfg)
    rows = []
    for r in range(4):
        cols = np.arange(0, ncols, dtype=np.int64)
        cols = cols[(cols < 300) | (cols > 400)] if r == 1 else cols  # a gap in row 1
        x = rng.normal(0.0, 1.0, cols.size)
        rows.append(h.RowResult(pieces=[h.Piece(cols=cols, lo=-np.abs(x), hi=np.abs(x))]))
    choice = h.StationChoice("XX.B1", "GHZ", "XX.B1..GHZ", 120.0, 5, ())
    markers = h.select_markers([(T0 + 100.0, "A")], [T0 + 100.0], T0, T0 + 7200.0, 1800.0)
    png = h.render_png(
        cfg=cfg,
        rows=rows,
        ref=1.0,
        markers=markers,
        t0=T0,
        choice=choice,
        day_utc="2001-09-09",
        run_id="synthetic-run",
    )
    assert h.png_size(png) == (cfg.layout.widthPx, cfg.layout.heightPx)
    assert len(png) < cfg.layout.maxBytes

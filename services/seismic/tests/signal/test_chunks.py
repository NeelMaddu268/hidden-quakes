"""Shared hour-chunk iterator (hq.preprocess.chunks). Offline: a fake read_window over synthetic
traces built inside each test."""

import copy
import logging
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from pydantic import ValidationError

from hq.config.signal import SignalConfig
from hq.preprocess import TimeMap
from hq.preprocess.chunks import (
    ChunkStats,
    ModelChunk,
    data_edges,
    iter_model_chunks,
    pick_is_kept,
    plan_keeps,
)

EPOCH_T = 1_789_063_200.0  # a multiple of 3600 s and of 100 s; synthetic
LENGTH_S = 100.0  # short chunks keep the tests fast; the logic is the same as for hours
OVERLAP_S = 35.0
PROBE_S = 1.0


@dataclass
class FakeCache:
    """Stand-in for ``hq.ingest.cache.read_window`` over in-memory contiguous pieces."""

    pieces: list[Trace]
    calls: list[tuple[str, float, float]] = field(default_factory=list)

    def read_window(self, station_id: str, t0: float, t1: float, *, cache_dir: Path) -> Stream:
        self.calls.append((station_id, t0, t1))
        out = Stream()
        for tr in self.pieces:
            piece = tr.slice(UTCDateTime(t0), UTCDateTime(t1), nearest_sample=False)
            if piece.stats.npts:
                out.append(piece.copy())
        out.sort(keys=["channel", "starttime"])
        return out


def piece(channel: str, start: float, end: float, rate: float, *, seed: int) -> Trace:
    """Noise sampled at ``rate`` from ``start`` up to (not including) ``end``."""
    n = round((end - start) * rate)
    data = np.random.default_rng(seed).standard_normal(n)
    header = {
        "network": "XX",
        "station": "SYN",
        "location": "",
        "channel": channel,
        "sampling_rate": rate,
        "starttime": UTCDateTime(start),
    }
    return Trace(data=data, header=header)


def triplet(
    start: float, end: float, rate: float, *, codes: str = "HH", horizontals: str = "NE", seed: int
) -> list[Trace]:
    comps = "Z" + horizontals
    return [piece(codes + c, start, end, rate, seed=seed + i) for i, c in enumerate(comps)]


def cfg_with_chunks(raw_signal_yaml: dict[str, Any], **chunks: float) -> SignalConfig:
    raw = copy.deepcopy(raw_signal_yaml)
    raw["preprocess"]["chunks"].update(
        {"lengthS": LENGTH_S, "overlapS": OVERLAP_S, "edgeProbeS": PROBE_S, **chunks}
    )
    return SignalConfig.model_validate(raw)


@pytest.fixture()
def cfg(raw_signal_yaml: dict[str, Any]) -> SignalConfig:
    return cfg_with_chunks(raw_signal_yaml)


def chunks_of(
    cache: FakeCache,
    cfg: SignalConfig,
    t0: float,
    t1: float,
    *,
    channels: tuple[str, ...] = ("HHZ", "HHN", "HHE"),
    profile: str = "surface-100",
    stats: ChunkStats | None = None,
) -> list[ModelChunk]:
    return list(
        iter_model_chunks(
            "XX.SYN",
            channels,
            profile,
            t0,
            t1,
            cfg,
            cache_dir=Path("unused"),
            read_window=cache.read_window,
            stats=stats,
        )
    )


# --- tiling --------------------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("t0", "t1"),
    [
        (EPOCH_T, EPOCH_T + 86_400.0),  # the showcase day shape: 24 whole hours
        (EPOCH_T + 1234.5, EPOCH_T + 9_000.25),  # off-grid both ends
        (EPOCH_T + 10.0, EPOCH_T + 20.0),  # inside one interval
        (EPOCH_T - 0.001, EPOCH_T + 3600.0),  # a sliver before a boundary
    ],
)
def test_plan_keeps_tiles_the_window_exactly_once(t0: float, t1: float) -> None:
    keeps = plan_keeps(t0, t1, 3600.0)
    assert keeps[0][0] == t0 and keeps[-1][1] == t1
    for (_, a1), (b0, _) in pairwise(keeps):
        assert a1 == b0  # contiguous: no gap, no double coverage
    for a, b in keeps:
        assert 0.0 < b - a <= 3600.0
    for _, b in keeps[:-1]:
        assert b % 3600.0 == 0.0  # interior cuts on multiples of lengthS since the epoch
    if t0 == EPOCH_T and t1 == EPOCH_T + 86_400.0:
        assert len(keeps) == 24


@pytest.mark.smoke
def test_plan_keeps_rejects_empty_window() -> None:
    with pytest.raises(ValueError, match="t0 < t1"):
        plan_keeps(EPOCH_T, EPOCH_T, 3600.0)
    with pytest.raises(ValueError, match="positive"):
        plan_keeps(EPOCH_T, EPOCH_T + 1.0, 0.0)


@pytest.mark.smoke
def test_chunks_tile_keep_once_and_read_with_overlap(cfg: SignalConfig) -> None:
    # off grid: chunks [30,100) [100,200) [200,300) [300,330)
    t0, t1 = EPOCH_T + 30.0, EPOCH_T + 330.0
    cache = FakeCache(triplet(EPOCH_T - 300.0, EPOCH_T + 700.0, 100.0, seed=1))
    stats = ChunkStats()
    chunks = chunks_of(cache, cfg, t0, t1, stats=stats)

    keeps = [c.keep for c in chunks]
    assert keeps == plan_keeps(t0, t1, LENGTH_S)
    assert [k[0] - EPOCH_T for k in keeps] == [30.0, 100.0, 200.0, 300.0]
    reach = OVERLAP_S + PROBE_S
    assert [(a, b) for _, a, b in cache.calls] == [(k0 - reach, k1 + reach) for k0, k1 in keeps]
    for c in chunks:
        assert c.stationId == "XX.SYN" and c.profile == "surface-100"
        assert c.timemap.is_identity
        assert sorted(tr.stats.channel for tr in c.stream) == ["HHE", "HHN", "HHZ"]
        for tr in c.stream:  # trimmed to the read span: the probe samples never reach the picker
            assert tr.stats.starttime.timestamp == pytest.approx(c.keep[0] - OVERLAP_S, abs=1e-6)
            assert tr.stats.endtime.timestamp == pytest.approx(c.keep[1] + OVERLAP_S, abs=1e-6)
        assert c.dataEdges == ()  # the data runs on past every cut: no data edge anywhere
    assert stats.as_counts() == {
        "chunksPlanned": 4,
        "chunksEmpty": 0,
        "chunksNoSegments": 0,
        "chunksYielded": 4,
        "chunksOtherChannelTraces": 0,
    }


# --- data edges ----------------------------------------------------------------------------------


@pytest.mark.smoke
def test_data_edges_exclude_cuts_but_include_gaps_and_real_ends(cfg: SignalConfig) -> None:
    t0, t1 = EPOCH_T, EPOCH_T + 300.0  # chunks [0,100) [100,200) [200,300)
    data_start = t0 - OVERLAP_S  # the cache genuinely starts exactly at the first read-span start
    data_end = t0 + 250.0  # and genuinely stops inside the last chunk
    gap = (t0 + 140.0, t0 + 145.0)  # N only: a gap on any component is an edge for every pick
    pieces = [
        piece("HHZ", data_start, data_end, 100.0, seed=1),
        piece("HHN", data_start, gap[0], 100.0, seed=2),
        piece("HHN", gap[1], data_end, 100.0, seed=3),
        piece("HHE", data_start, data_end, 100.0, seed=4),
    ]
    chunks = chunks_of(FakeCache(pieces), cfg, t0, t1)
    assert len(chunks) == 3
    last_sample = data_end - 0.01
    n_last_before_gap = gap[0] - 0.01
    first, second, third = (c.dataEdges for c in chunks)
    assert first == pytest.approx((data_start,))  # real start; the cut at 100 + overlap is not
    assert second == pytest.approx((n_last_before_gap, gap[1]))
    assert third == pytest.approx((last_sample,))  # real end; the cut at 200 - overlap is not
    for c in chunks:  # no edge sits on an artificial read-span end
        read_ends = (c.keep[0] - OVERLAP_S, c.keep[1] + OVERLAP_S)
        assert all(abs(e - r) > 0.5 for e in c.dataEdges for r in read_ends if e != data_start)


@pytest.mark.smoke
def test_data_edges_probe_rule() -> None:
    rate = 100.0
    inside = Stream(traces=[piece("HHZ", EPOCH_T + 0.5, EPOCH_T + 9.0, rate, seed=1)])
    assert data_edges(inside, EPOCH_T, EPOCH_T + 10.0) == pytest.approx(
        (EPOCH_T + 0.5, EPOCH_T + 8.99)
    )
    cut = Stream(traces=[piece("HHZ", EPOCH_T, EPOCH_T + 10.0, rate, seed=1)])  # spans the probe
    assert data_edges(cut, EPOCH_T, EPOCH_T + 9.995) == ()


# --- keep rule -----------------------------------------------------------------------------------


@pytest.mark.smoke
def test_pick_is_kept_reasons() -> None:
    chunk = ModelChunk(
        stationId="XX.SYN",
        profile="surface-100",
        stream=Stream(),
        timemap=TimeMap(anchor=0.0, factor=1.0),
        keep=(100.0, 200.0),
        dataEdges=(50.0, 150.0, 199.8),
    )
    assert pick_is_kept(99.999, chunk, 0.5) == (False, "outside_keep")
    assert pick_is_kept(200.0, chunk, 0.5) == (False, "outside_keep")  # end is exclusive
    assert pick_is_kept(50.2, chunk, 0.5) == (False, "outside_keep")  # neighbour counts it
    assert pick_is_kept(100.0, chunk, 0.5) == (True, "ok")
    assert pick_is_kept(149.4, chunk, 0.5) == (True, "ok")
    assert pick_is_kept(149.5, chunk, 0.5) == (False, "near_gap_edge")  # within: <= gap_edge_s
    assert pick_is_kept(150.3, chunk, 0.5) == (False, "near_gap_edge")
    assert pick_is_kept(199.9, chunk, 0.5) == (False, "near_gap_edge")
    assert pick_is_kept(151.0, chunk, 0.5) == (True, "ok")
    assert pick_is_kept(150.0, chunk, 0.0) == (False, "near_gap_edge")
    with pytest.raises(ValueError, match="gap_edge_s"):
        pick_is_kept(150.0, chunk, -1.0)
    with pytest.raises(ValueError, match="sorted"):
        ModelChunk("XX.SYN", "p", Stream(), TimeMap(0.0, 1.0), (0.0, 1.0), (2.0, 1.0))


# --- borehole-B: model time inside, real time outside ----------------------------------------------


@pytest.mark.smoke
def test_borehole_b_chunk_keeps_real_time_outside_the_stream(cfg: SignalConfig) -> None:
    rate = 1000.0
    t0, t1 = EPOCH_T, EPOCH_T + 100.0
    gap = (t0 + 40.0, t0 + 42.0)
    pieces = [
        piece(f"DP{c}", t0 - 50.0, gap[0], rate, seed=10 + i) for i, c in enumerate("Z12")
    ] + [piece(f"DP{c}", gap[1], t1 + 50.0, rate, seed=20 + i) for i, c in enumerate("Z12")]
    (chunk,) = chunks_of(
        FakeCache(pieces), cfg, t0, t1, channels=("DPZ", "DP1", "DP2"), profile="borehole-B"
    )
    factor = rate / cfg.preprocess.targetRateHz
    assert chunk.timemap.factor == factor
    assert chunk.keep == (t0, t1)  # real time, untouched by the stretch
    assert chunk.dataEdges == pytest.approx((gap[0] - 1.0 / rate, gap[1]), abs=1e-9)
    by_start: dict[float, list[Trace]] = {}
    for tr in chunk.stream:
        assert tr.stats.sampling_rate == cfg.preprocess.targetRateHz
        by_start.setdefault(tr.stats.starttime.timestamp, []).append(tr)
    first, second = sorted(by_start)
    # the stream is in model time; to_real brings each segment start back exactly once
    assert chunk.timemap.to_real(first) == pytest.approx(t0 - OVERLAP_S, abs=1e-3)
    assert chunk.timemap.to_real(second) == pytest.approx(gap[1], abs=1e-3)
    assert second - first == pytest.approx((gap[1] - (t0 - OVERLAP_S)) * factor, abs=1e-2)
    assert sorted(tr.stats.channel for tr in by_start[first]) == ["DPE", "DPN", "DPZ"]


# --- counting and failing ------------------------------------------------------------------------


@pytest.mark.smoke
def test_empty_and_short_chunks_yield_nothing_but_are_counted(
    cfg: SignalConfig, caplog: pytest.LogCaptureFixture
) -> None:
    t0 = EPOCH_T
    pieces = [
        *triplet(t0 - 50.0, t0 + 150.0, 100.0, seed=1),  # chunks [0,100) and [100,200)
        piece("HNZ", t0 - 50.0, t0 + 150.0, 100.0, seed=9),  # not a selected channel
        *triplet(t0 + 340.0, t0 + 350.0, 100.0, seed=5),  # 10 s, read by [300,400) only: too short
    ]
    stats = ChunkStats()
    with caplog.at_level(logging.INFO, logger="hq.preprocess.chunks"):
        chunks = chunks_of(FakeCache(pieces), cfg, t0, t0 + 500.0, stats=stats)
    assert [c.keep[0] - t0 for c in chunks] == [0.0, 100.0]
    assert stats.planned == 5 and stats.yielded == 2
    assert stats.empty == 2 and stats.noSegments == 1  # [200,300) and [400,500) have no samples
    assert stats.otherChannelTraces == 2  # HNZ read for both non-empty chunk spans
    assert all(tr.stats.channel != "HNZ" for c in chunks for tr in c.stream)
    assert "planned=5 yielded=2 empty=2 no_segments=1" in caplog.text


@pytest.mark.smoke
def test_stats_are_counted_when_the_consumer_stops_early(cfg: SignalConfig) -> None:
    cache = FakeCache(triplet(EPOCH_T - 50.0, EPOCH_T + 450.0, 100.0, seed=1))
    stats = ChunkStats()
    gen = iter_model_chunks(
        "XX.SYN",
        ("HHZ", "HHN", "HHE"),
        "surface-100",
        EPOCH_T,
        EPOCH_T + 400.0,
        cfg,
        cache_dir=Path("unused"),
        read_window=cache.read_window,
        stats=stats,
    )
    next(gen)
    gen.close()
    assert stats.planned == 1 and stats.yielded == 1


@pytest.mark.smoke
def test_for_picking_and_read_errors_propagate(cfg: SignalConfig) -> None:
    cache = FakeCache(triplet(EPOCH_T - 50.0, EPOCH_T + 150.0, 100.0, seed=1))
    with pytest.raises(ValueError, match="outside profile borehole-A"):
        chunks_of(cache, cfg, EPOCH_T, EPOCH_T + 100.0, profile="borehole-A")

    def missing(station_id: str, t0: float, t1: float, *, cache_dir: Path) -> Stream:
        raise LookupError(f"nothing cached for {station_id}")

    with pytest.raises(LookupError, match="nothing cached"):
        list(
            iter_model_chunks(
                "XX.SYN",
                ("HHZ",),
                "surface-100",
                EPOCH_T,
                EPOCH_T + 100.0,
                cfg,
                cache_dir=Path("unused"),
                read_window=missing,
            )
        )
    with pytest.raises(ValueError, match="no channels"):
        chunks_of(cache, cfg, EPOCH_T, EPOCH_T + 100.0, channels=())


@pytest.mark.smoke
def test_probe_must_exceed_one_sample(raw_signal_yaml: dict[str, Any]) -> None:
    cfg = cfg_with_chunks(raw_signal_yaml, edgeProbeS=0.5)
    slow = FakeCache(triplet(EPOCH_T - 50.0, EPOCH_T + 150.0, 1.0, seed=1))  # 1 Hz: delta 1 s
    with pytest.raises(ValueError, match="edgeProbeS"):
        chunks_of(slow, cfg, EPOCH_T, EPOCH_T + 100.0)


@pytest.mark.smoke
def test_chunk_config_is_validated(
    raw_signal_yaml: dict[str, Any], signal_cfg: SignalConfig
) -> None:
    chunks = signal_cfg.preprocess.chunks
    assert chunks.overlapS >= chunks.minOverlapS
    with pytest.raises(ValidationError, match="minOverlapS"):
        cfg_with_chunks(raw_signal_yaml, overlapS=10.0, minOverlapS=30.0)
    with pytest.raises(ValidationError, match="edgeProbeS"):
        cfg_with_chunks(raw_signal_yaml, edgeProbeS=OVERLAP_S)
    raw = copy.deepcopy(raw_signal_yaml)
    raw["preprocess"]["chunks"]["notAKnob"] = 1
    with pytest.raises(ValidationError):
        SignalConfig.model_validate(raw)

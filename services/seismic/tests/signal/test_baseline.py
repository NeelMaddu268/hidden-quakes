"""STA/LTA baseline (SEIS-07). Offline and seeded: synthetic noise plus impulsive arrivals at known
times, served through a fake read_window. Stage tests run once against the real ``hq_contracts``
(skipped until CONTRACT-01 is on this branch) and once against a parquet stand-in for it built
here from docs/02 sections 1-2."""

import copy
import dataclasses
import importlib
import json
import logging
import sys
import types
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from obspy import Stream, Trace, UTCDateTime
from pydantic import BaseModel, ConfigDict, ValidationError

import hq.baseline as stage
import hq.baseline.score as sc
from hq.config.run import RunSection
from hq.config.signal import SignalConfig
from hq.ingest.cache import CacheMissError
from hq.preprocess.chunks import ModelChunk, iter_model_chunks, pick_is_kept
from hq.preprocess.profiles import TimeMap
from hq.validate.errors import ValidateError

bl = importlib.import_module("hq.baseline.run")  # the module; ``hq.baseline.run`` is the stage
chunks_module = importlib.import_module("hq.preprocess.chunks")

T = 1_789_063_200.0  # 2026-09-10T18:00:00Z, a multiple of every chunk length used here
RATE = 100.0
LENGTH_S = 100.0  # short chunks keep the tests fast; the logic is the same as for hours
OVERLAP_S = 40.0
TOL_S = 0.05  # "a few samples" at 100 Hz
PICK_FIELDS = ["id", "stationId", "phase", "t", "prob", "picker", "eventId", "residualS", "weight"]
# Test-owned: the shipped chosen/grid are checked only by bl.check_config(signal_cfg), so setting a
# new baseline.chosen (or grid) in signal.yaml never breaks a test that is about something else.
TEST_CHOSEN = {"pOn": 5.0, "pOff": 1.5, "sOn": 5.0, "sOff": 1.5}
TEST_GRID = {
    "pOn": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
    "sOn": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
    "offLevels": [1.0, 1.5, 2.0],
}


# --- synthetic data -----------------------------------------------------------------------------


def burst(t: np.ndarray, t0: float, amp: float, freq: float, tau: float) -> np.ndarray:
    """A decaying sinusoid that starts exactly at ``t0`` (zero before it)."""
    dt = t - t0
    wave = amp * np.sin(2 * np.pi * freq * dt) * np.exp(-np.clip(dt, 0.0, None) / tau)
    out: np.ndarray = np.where(dt >= 0.0, wave, 0.0)
    return out


@dataclass
class Arrivals:
    p: list[tuple[float, float]] = field(default_factory=list)  # (time, amplitude) on Z
    s: list[tuple[float, float]] = field(default_factory=list)  # (time, amplitude) on N and E


def station_pieces(
    start: float,
    end: float,
    arrivals: Arrivals,
    *,
    rate: float = RATE,
    codes: str = "HH",
    horizontals: str = "NE",
    station: str = "SYN",
    seed: int = 0,
    gaps: tuple[tuple[float, float], ...] = (),
    p_freq: float = 12.0,
    s_freq: float = 6.0,
) -> list[Trace]:
    """Three components of unit noise plus bursts, split around ``gaps`` (never filled)."""
    n = round((end - start) * rate)
    t = start + np.arange(n) / rate
    rng = np.random.default_rng(seed)
    comps = {c: rng.standard_normal(n) for c in "Z" + horizontals}
    for tp, amp in arrivals.p:
        comps["Z"] += burst(t, tp, amp, p_freq, 0.4)
    for ts, amp in arrivals.s:
        for i, c in enumerate(horizontals):
            comps[c] += burst(t, ts, amp * (1.0 - 0.3 * i), s_freq, 0.8)
    keep = np.ones(n, dtype=bool)
    for g0, g1 in gaps:
        keep &= ~((t >= g0) & (t < g1))
    out: list[Trace] = []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], keep.astype(np.int8), [0]))))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        for c, data in comps.items():
            header = {
                "network": "XX",
                "station": station,
                "location": "",
                "channel": codes + c,
                "sampling_rate": rate,
                "starttime": UTCDateTime(float(t[a])),
            }
            out.append(Trace(data=data[a:b].copy(), header=header))
    return out


@dataclass
class FakeCache:
    """Stand-in for ``hq.ingest.cache.read_window``: pieces per station id."""

    pieces: dict[str, list[Trace]]
    not_cached: frozenset[str] = frozenset()

    def read_window(self, station_id: str, t0: float, t1: float, *, cache_dir: Path) -> Stream:
        if station_id in self.not_cached:
            raise CacheMissError(f"nothing cached for station {station_id}")
        out = Stream()
        for tr in self.pieces.get(station_id, []):
            piece = tr.slice(UTCDateTime(t0), UTCDateTime(t1), nearest_sample=False)
            if piece.stats.npts:
                out.append(piece.copy())
        out.sort(keys=["channel", "starttime"])
        return out


def small_cfg(raw_signal_yaml: dict[str, Any], **baseline: Any) -> SignalConfig:
    raw = copy.deepcopy(raw_signal_yaml)
    raw["preprocess"]["chunks"].update({"lengthS": LENGTH_S, "overlapS": OVERLAP_S})
    # The test-owned thresholds and grid first, so ``baseline`` overrides still win.
    raw["baseline"]["chosen"] = dict(TEST_CHOSEN)
    raw["baseline"]["sweep"].update(copy.deepcopy(TEST_GRID))
    raw["baseline"].update({"maxWorkers": 1, **baseline})
    return SignalConfig.model_validate(raw)


def small_grid(
    p_on: list[float],
    s_on: list[float],
    off: float,
    chosen: tuple[float, float],
    *,
    mode: str = "all",
) -> dict[str, Any]:
    """``baseline`` overrides for a small sweep; ``chosen`` (pOn, sOn) must be on its grid."""
    sweep = {"pOn": p_on, "sOn": s_on, "offLevels": [off], "scoreMode": mode}
    return {
        "sweep": {**sweep, "scoreWorkers": 1, "maxPasses": 3},
        "chosen": {"pOn": chosen[0], "pOff": off, "sOn": chosen[1], "sOff": off},
    }


@pytest.fixture()
def cfg(raw_signal_yaml: dict[str, Any]) -> SignalConfig:
    return small_cfg(raw_signal_yaml)


def pick(
    cache: FakeCache,
    cfg: SignalConfig,
    t0: float,
    t1: float,
    *,
    station: str = "XX.SYN",
    channels: tuple[str, ...] = ("HHZ", "HHN", "HHE"),
    profile: str = "surface-100",
    grid: list[Any] | None = None,
) -> Any:
    row = bl.StationRow(id=station, channels=channels, profile=profile)
    return bl.pick_station(
        row,
        t0,
        t1,
        cfg,
        bl.sweep_grid(cfg.baseline) if grid is None else grid,
        cache_dir=Path("unused"),
        read_window=cache.read_window,
    )


def times(result: Any, phase: str) -> list[float]:
    return sorted(t for t, ph in result.picks if ph == phase)


def assert_close(got: list[float], want: list[float], tol: float = TOL_S) -> None:
    assert len(got) == len(want), (got, want)
    for g, w in zip(got, want, strict=True):
        assert 0.0 <= g - w <= tol, (g, w)  # an STA/LTA onset never precedes the true onset


def spy_on_sta_lta(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Record ``(nsta, nlta)`` of every ``recursive_sta_lta`` call the baseline makes."""
    calls: list[tuple[int, int]] = []
    real = bl.recursive_sta_lta

    def spy(a: np.ndarray, nsta: int, nlta: int) -> np.ndarray:
        calls.append((nsta, nlta))
        out: np.ndarray = real(a, nsta, nlta)
        return out

    monkeypatch.setattr(bl, "recursive_sta_lta", spy)
    return calls


# --- picking -------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_p_and_s_are_picked_within_a_few_samples_of_truth(cfg: SignalConfig) -> None:
    arr = Arrivals(p=[(T + 50.0, 30.0), (T + 150.3, 30.0)], s=[(T + 51.5, 60.0), (T + 152.0, 60.0)])
    cache = FakeCache({"XX.SYN": station_pieces(T - 60.0, T + 260.0, arr)})
    res = pick(cache, cfg, T, T + 200.0)
    assert_close(times(res, "P"), [T + 50.0, T + 150.3])
    assert_close(times(res, "S"), [T + 51.5, T + 152.0])
    assert res.chunks.planned == 2 and res.chunks.yielded == 2
    assert res.counts["segmentsP"] == 2 and res.counts["segmentsS"] == 4
    assert res.reason == ""


@pytest.mark.smoke
def test_borehole_b_timemap_is_applied_exactly_once(
    cfg: SignalConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows = spy_on_sta_lta(monkeypatch)
    rate = 1000.0
    tp, ts = T + 30.123, T + 30.9
    # Frequencies inside borehole-B's STA/LTA prefilter (20-300 Hz real) and well above the
    # profile's 5 Hz zero-phase highpass corner, whose symmetric ringing would put a precursor
    # ahead of a low-frequency onset (a property of the profile, not of the time mapping). The
    # wide band passes about ten times more of the white test noise than 2-30 Hz, hence amp 15.
    pieces = station_pieces(
        T - 45.0,
        T + 105.0,
        Arrivals(p=[(tp, 15.0)], s=[(ts, 15.0)]),
        rate=rate,
        codes="DP",
        horizontals="12",
        seed=3,
        p_freq=60.0,
        s_freq=40.0,
    )
    res = pick(
        FakeCache({"XX.SYN": pieces}),
        cfg,
        T,
        T + 60.0,
        channels=("DPZ", "DP1", "DP2"),
        profile="borehole-B",
    )
    b = cfg.baseline
    # Windows are real seconds at the real rate (1000 Hz), not model samples at 100 Hz.
    assert set(windows) == {
        (round(b.p.staS * rate), round(b.p.ltaS * rate)),
        (round(b.s.staS * rate), round(b.s.ltaS * rate)),
    }
    # Onsets come back in real time within a few model samples (10 real ms each). A factor
    # applied twice or not at all would put them tens of seconds off, outside the keep interval.
    assert_close(times(res, "P"), [tp], tol=0.03)
    assert_close(times(res, "S"), [ts], tol=0.03)


@pytest.mark.smoke
def test_s_needs_a_p_ahead_of_it_within_the_window(cfg: SignalConfig) -> None:
    b = cfg.baseline
    arr = Arrivals(
        p=[(T + 30.0, 30.0), (T + 140.0, 30.0)],
        s=[
            (T + 80.0, 60.0),  # no P before it: not an S
            (T + 30.0 + b.maxSMinusPS + 1.0, 60.0),  # too late after the first P
            (T + 141.0, 60.0),  # 1 s after the second P: an S
        ],
    )
    cache = FakeCache({"XX.SYN": station_pieces(T - 60.0, T + 260.0, arr, seed=5)})
    res = pick(cache, cfg, T, T + 200.0)
    assert_close(times(res, "P"), [T + 30.0, T + 140.0])
    assert_close(times(res, "S"), [T + 141.0])


@pytest.mark.smoke
def test_s_whose_p_is_in_the_previous_chunk_is_picked_once(cfg: SignalConfig) -> None:
    arr = Arrivals(p=[(T + 99.0, 30.0)], s=[(T + 100.5, 60.0)])  # P in [0,100), S in [100,200)
    cache = FakeCache({"XX.SYN": station_pieces(T - 60.0, T + 260.0, arr, seed=11)})
    res = pick(cache, cfg, T, T + 200.0)
    assert_close(times(res, "P"), [T + 99.0])
    assert_close(times(res, "S"), [T + 100.5])  # chunk 2 sees the P in its overlap


@pytest.mark.smoke
def test_gap_edge_drops_are_counted_and_neighbour_picks_discarded(cfg: SignalConfig) -> None:
    arr = Arrivals(
        p=[
            (T + 59.5, 30.0),  # 0.5 s before a gap: dropped and counted
            (T + 95.0, 30.0),  # chunk 1's keep, also inside chunk 2's read span
            (T + 120.0, 30.0),  # chunk 2's keep, also inside chunk 1's read span
            (T + 205.0, 30.0),  # after t1: in the last chunk's overlap only, nobody's pick
        ]
    )
    pieces = station_pieces(T - 60.0, T + 260.0, arr, seed=7, gaps=((T + 60.0, T + 70.0),))
    res = pick(FakeCache({"XX.SYN": pieces}), cfg, T, T + 200.0)
    assert_close(times(res, "P"), [T + 95.0, T + 120.0])  # each exactly once
    assert res.counts["droppedNearGapEdgeP"] == 1
    assert res.counts["droppedNearGapEdgeS"] == 0


@pytest.mark.smoke
def test_warm_up_and_p_near_edge_losses_are_counted(cfg: SignalConfig) -> None:
    arr = Arrivals(
        # 3 s after a gap on every component: inside the P warm-up (6 s) and the S warm-up
        # (12 s) of the segments that start at the gap end.
        p=[(T + 73.0, 30.0), (T + 135.5, 30.0)],
        s=[(T + 74.5, 60.0), (T + 139.5, 60.0)],
    )
    pieces = station_pieces(T - 60.0, T + 260.0, arr, seed=17, gaps=((T + 60.0, T + 70.0),))
    # A gap on N alone, [130, 135): the P at 135.5 on the continuous Z is within gapEdgeS of a
    # data edge (dropped); its S on the continuous E at 139.5 is clear of every edge but has no
    # P. On N the same S falls in the S warm-up of the segment that starts at 135 (after ObsPy's
    # own zeroed first ltaS, so it does trigger there).
    split: list[Trace] = []
    for tr in pieces:
        if (
            tr.stats.channel == "HHN"
            and tr.stats.starttime < UTCDateTime(T + 130.0) < tr.stats.endtime
        ):
            split.append(tr.slice(tr.stats.starttime, UTCDateTime(T + 130.0 - 0.5 / RATE)))
            split.append(tr.slice(UTCDateTime(T + 135.0), tr.stats.endtime))
        else:
            split.append(tr)
    res = pick(FakeCache({"XX.SYN": split}), cfg, T, T + 200.0)
    assert res.picks == []
    assert res.counts["droppedNearGapEdgeP"] == 1  # 135.5
    assert res.counts["sLostPNearGapEdge"] == 1  # 139.5 on E
    assert res.counts["suppressedWarmupP"] == 1  # 73
    # 74.5 on N and on E (both horizontals start at the gap end), 139.5 on N (starts at 135),
    # and a noise trigger on N at 139.0, where ObsPy's own zeroing of the first ltaS ends and
    # the unconverged LTA inflates the ratio: the artefact the warm-up exists to suppress.
    assert res.counts["suppressedWarmupS"] == 4
    assert res.reason == "every trigger near a data edge or inside a warm-up span"


@pytest.mark.smoke
def test_reason_for_no_picks_names_empty_and_too_short_chunks() -> None:
    r = bl.StationResult(stationId="XX.A", profile="surface-100")
    r.chunks.planned, r.chunks.empty = 2, 2
    assert r.reason == "no usable data: 2 of 2 chunks empty"
    r.chunks.empty, r.chunks.noSegments = 1, 1
    assert r.reason == (
        "no usable data: 1 of 2 chunks empty, 1 with every segment too short for for_picking"
    )
    r.chunks.empty = 0
    assert r.reason == "no usable data: 1 with every segment too short for for_picking"


@pytest.mark.smoke
def test_a_station_error_names_the_station(cfg: SignalConfig) -> None:
    # 200 Hz data on a station configured as surface-100: for_picking rejects the rate. The
    # stage fails loudly (no per-station fallback) and the error says which station.
    pieces = station_pieces(T - 60.0, T + 160.0, Arrivals(), rate=200.0, seed=1)
    with pytest.raises(ValueError) as err:
        pick(FakeCache({"XX.SYN": pieces}), cfg, T, T + 100.0)
    assert any("station XX.SYN (profile surface-100)" in n for n in err.value.__notes__)


@pytest.mark.smoke
def test_short_segments_are_skipped_and_counted(raw_signal_yaml: dict[str, Any]) -> None:
    # One chunk [0, 100) read over [-40, 140]; a gap leaves segments [-40, 5) (45 s) and
    # [50, 140] (90 s), both long enough for for_picking. A segment needs warmupS + margin: its
    # first warmupS can never trigger (P warm-up 6 s, S 12 s).
    pieces = station_pieces(T - 60.0, T + 260.0, Arrivals(), gaps=((T + 5.0, T + 50.0),))
    cache = FakeCache({"XX.SYN": pieces})
    expected = {  # margin -> (P skipped, S skipped): the 45 s segment fails once need > 45 s
        30.0: (0, 0),  # P 36 s, S 42 s
        34.0: (0, 2),  # P 40 s, S 46 s: both horizontals of the 45 s segment skipped
        40.0: (1, 2),  # P 46 s
    }
    for margin, (skip_p, skip_s) in expected.items():
        res = pick(cache, small_cfg(raw_signal_yaml, minSegmentMarginS=margin), T, T + 100.0)
        assert res.counts["segmentsSkippedShortP"] == skip_p, margin
        assert res.counts["segmentsSkippedShortS"] == skip_s, margin
        assert res.counts["segmentsP"] == 2 - skip_p and res.counts["segmentsS"] == 4 - skip_s
    res = pick(cache, small_cfg(raw_signal_yaml, minSegmentMarginS=100.0), T, T + 100.0)
    assert (
        res.picks == [] and res.reason == "every segment shorter than warmupS + minSegmentMarginS"
    )


# --- sweep ----------------------------------------------------------------------------------------


def graded_events() -> Arrivals:
    """Six events of rising amplitude, 30 s apart, clear of chunk cuts and window ends."""
    amps = [2.0, 3.0, 4.0, 6.0, 9.0, 14.0]
    return Arrivals(
        p=[(T + 15.0 + 30.0 * i, a) for i, a in enumerate(amps)],
        s=[(T + 16.5 + 30.0 * i, 1.5 * a) for i, a in enumerate(amps)],
    )


@pytest.mark.smoke
def test_sweep_computes_each_cf_once_and_counts_fall_with_the_on_threshold(
    cfg: SignalConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = spy_on_sta_lta(monkeypatch)
    cache = FakeCache({"XX.SYN": station_pieces(T - 60.0, T + 260.0, graded_events(), seed=2)})
    grid = bl.sweep_grid(cfg.baseline)
    assert len(grid) == 7 * 7 * 3
    res = pick(cache, cfg, T, T + 200.0, grid=grid)
    n_cf = len(calls)
    # once per segment and phase: 2 chunks x (1 vertical + 2 horizontals), whatever the grid size
    assert n_cf == res.counts["segmentsP"] + res.counts["segmentsS"] == 6
    calls.clear()
    pick(cache, cfg, T, T + 200.0, grid=grid[:1])
    assert len(calls) == n_cf

    sweep = bl.sweep_frame([res], grid, [None] * len(grid))
    df = pd.concat([pd.DataFrame(list(sweep["params"].map(json.loads))), sweep], axis=1)
    for _, g in df.groupby(["sOn", "pOff"]):
        n_p = g.sort_values("pOn")["nP"].tolist()
        assert n_p == sorted(n_p, reverse=True), n_p
    for _, g in df.groupby(["pOn", "pOff"]):
        n_s = g.sort_values("sOn")["nS"].tolist()
        assert n_s == sorted(n_s, reverse=True), n_s
    assert df["nP"].max() >= 6 > df["nP"].min()  # the thresholds actually separate the events
    assert df["nS"].max() > df["nS"].min()
    assert set(df["nStations"]) <= {0, 1}


@pytest.mark.smoke
def test_sweep_row_of_the_chosen_thresholds_matches_the_published_picks(
    cfg: SignalConfig,
) -> None:
    arr = graded_events()
    arr.p.append((T + 59.6, 30.0))  # near a gap edge: dropped on both paths
    pieces = station_pieces(T - 60.0, T + 260.0, arr, seed=4, gaps=((T + 60.0, T + 62.0),))
    grid = bl.sweep_grid(cfg.baseline)
    res = pick(FakeCache({"XX.SYN": pieces}), cfg, T, T + 200.0, grid=grid)
    k = grid.index(bl.chosen_point(cfg.baseline))
    assert res.counts["droppedNearGapEdgeP"] >= 1
    assert res.sweepP[k].tolist() == times(res, "P")  # vectorised rule == pick_is_kept
    assert res.sweepS[k].tolist() == times(res, "S")
    assert times(res, "P")


@pytest.mark.smoke
def test_keep_masks_match_pick_is_kept() -> None:
    chunk = ModelChunk(
        stationId="XX.SYN",
        profile="surface-100",
        stream=Stream(),
        timemap=TimeMap(anchor=0.0, factor=1.0),
        keep=(100.0, 200.0),
        dataEdges=(50.0, 150.0, 199.8),
    )
    rng = np.random.default_rng(0)
    t = np.concatenate([rng.uniform(40.0, 260.0, 3000), [100.0, 200.0, 149.0, 151.0, 198.8]])
    for edges in (chunk.dataEdges, ()):
        c = dataclasses.replace(chunk, dataEdges=edges)
        kept, near = bl.keep_masks(t, c, 1.0)
        for ti, k, n in zip(t.tolist(), kept.tolist(), near.tolist(), strict=True):
            ok, why = pick_is_kept(ti, c, 1.0)
            assert (k, n) == (ok, why == "near_gap_edge"), ti


def trig(*pairs: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """One component's triggers as (onset, end) arrays."""
    on = np.array([a for a, _ in pairs], dtype=float)
    end = np.array([b for _, b in pairs], dtype=float)
    return on, end


@pytest.mark.smoke
def test_s_after_p_searches_each_horizontal_past_its_own_p_trigger() -> None:
    p = np.array([10.0, 20.0, 21.0, 30.0, 40.0, 50.0])
    n = trig(
        (5.0, 5.5),
        (10.1, 10.3),
        (10.5, 11.0),
        (21.6, 22.0),
        (30.02, 31.0),
        (33.0, 33.5),
        (39.5, 40.6),
        (40.7, 41.0),
    )
    e = trig((12.0, 12.5), (30.25, 30.5), (30.8, 31.3))
    # P 10: N's 10.1 is the P on N (on within 0.3 s), N's search starts at its end -> 10.5;
    #       E's first is 12.0; the earlier of the two wins -> 10.5.
    # P 20 and P 21 both choose 21.6 (one S).
    # P 30: N carries the P until 31.0; E's own P trigger (an emergent P crossing 0.25 s late)
    #       ends at 30.5 and E's next onset, 30.8, is the S, although N is still triggered.
    # P 40: a trigger already on at the P (39.5-40.6) holds N's search until 40.6 -> 40.7.
    # P 50: nothing within maxSMinusPS.
    assert bl.s_after_p(p, (n, e), 0.3, 0.15, 5.0).tolist() == [10.5, 21.6, 30.8, 40.7]
    # Without the tolerance E's late P becomes the S of P 30.
    assert bl.s_after_p(p, (n, e), 0.0, 0.15, 5.0).tolist() == [10.5, 21.6, 30.25, 40.7]
    # One trigger list for both horizontals: N's P trigger would hide E's S until 31.0.
    on, end = (np.concatenate(x) for x in zip(n, e, strict=True))
    order = np.argsort(on, kind="stable")
    merged = (on[order], end[order])
    assert bl.s_after_p(p, (merged,), 0.3, 0.15, 5.0).tolist() == [10.5, 21.6, 33.0, 40.7]
    assert bl.s_after_p(p[:0], (n, e), 0.3, 0.15, 5.0).size == 0
    assert bl.s_after_p(p, (), 0.3, 0.15, 5.0).size == 0
    assert bl.s_after_p(p, (trig(), trig()), 0.3, 0.15, 5.0).size == 0


@pytest.mark.smoke
def test_p_energy_on_the_horizontals_is_not_an_s(raw_signal_yaml: dict[str, Any]) -> None:
    # P on all components, no S wave: N carries the P from the onset, E's P crosses the trigger
    # level 0.2 s later (an emergent P on that component).
    tp = T + 50.0
    pieces = station_pieces(T - 60.0, T + 260.0, Arrivals(p=[(tp, 30.0)]), seed=13)
    rate = RATE
    t = (T - 60.0) + np.arange(round(320.0 * rate)) / rate
    for tr in pieces:
        comp = tr.stats.channel[-1]
        if comp == "N":
            tr.data = tr.data + burst(t, tp, 12.0, 12.0, 0.6)
        elif comp == "E":
            tr.data = tr.data + burst(t, tp + 0.2, 12.0, 12.0, 0.6)
    cache = FakeCache({"XX.SYN": pieces})
    cfg = small_cfg(raw_signal_yaml)
    res = pick(cache, cfg, T, T + 100.0)
    assert_close(times(res, "P"), [tp])
    assert times(res, "S") == []
    # Both horizontals did trigger: N with the P, E 0.2 s later. The plain earliest-onset rule
    # (every trigger a point, no P tolerance) would have made E's late P an S.
    b = cfg.baseline
    chosen = bl.chosen_point(b)
    chunk = next(
        iter_model_chunks(
            "XX.SYN",
            ("HHZ", "HHN", "HHE"),
            "surface-100",
            T,
            T + 100.0,
            cfg,
            cache_dir=Path("unused"),
            read_window=cache.read_window,
        )
    )
    prefilter = b.prefilter["surface-100"]
    onsets = bl.chunk_onsets(
        chunk, b, prefilter, [chosen.p_pair], [chosen.s_pair], chosen, Counter()
    )
    (n_on, _), (e_on, _) = onsets.h[chosen.s_pair]  # s.components "NE"
    assert [round(x - tp, 1) for x in (*n_on.tolist(), *e_on.tolist())] == [0.0, 0.2]
    p = onsets.p[chosen.p_pair]
    points = ((n_on, n_on), (e_on, e_on))
    naive = bl.s_after_p(p, points, 0.0, b.minSMinusPS, b.maxSMinusPS)
    assert [round(x - tp, 1) for x in naive.tolist()] == [0.2]


# --- config ---------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_config_is_validated_and_cross_checked(
    raw_signal_yaml: dict[str, Any], signal_cfg: SignalConfig
) -> None:
    bl.check_config(signal_cfg)  # the shipped signal.yaml passes
    # Not 1.0: H2's locator weights picks by prob, so 1.0 would narrow the baseline's formal errors.
    assert 0.0 < signal_cfg.baseline.prob < 1.0
    assert bl.gap_edge_s(signal_cfg) == signal_cfg.picker.gapEdgeS  # one distance, both pickers

    def with_baseline(**over: Any) -> SignalConfig:
        return small_cfg(raw_signal_yaml, **over)

    # The bad cases start from the test-owned chosen/grid, so they stay invalid whatever ships.
    b = {
        **raw_signal_yaml["baseline"],
        "chosen": TEST_CHOSEN,
        "sweep": {**raw_signal_yaml["baseline"]["sweep"], **TEST_GRID},
    }
    bad: list[dict[str, Any]] = [
        {"p": {**b["p"], "staS": 3.0}},  # STA not below LTA
        {"s": {**b["s"], "warmupS": 1.0}},  # warm-up shorter than the LTA
        {"s": {**b["s"], "components": "ZE"}},  # P and S share Z
        {"chosen": {**b["chosen"], "pOff": 6.0}},  # off above on
        {"chosen": {**b["chosen"], "pOn": 7.0}},  # not a sweep grid point
        {"chosen": {**b["chosen"], "sOff": 1.0}},  # the grid has one off level for both phases
        {"sweep": {**b["sweep"], "pOn": [4.0, 3.0]}},  # not increasing
        {"sweep": {**b["sweep"], "offLevels": [1.0, 3.5]}},  # an off level above an on level
        {"sweep": {**b["sweep"], "scoreMode": "best"}},  # all, coordinate or none
        {"sweep": {**b["sweep"], "scoreWorkers": 0}},
        {"sweep": {**b["sweep"], "maxPasses": 0}},
        {"sweep": {**b["sweep"], "scoreWithH2": True}},  # folded into scoreMode
        {"minSMinusPS": 6.0},  # empty S window
        {"prefilter": {}},  # every profile needs a band
        {"prefilter": {**b["prefilter"], "borehole-B": None}},  # no unfiltered STA/LTA
        {"gapEdgeS": 1.0},  # the gap-edge distance is picker.gapEdgeS, shared with PhaseNet
        {"notAKnob": 1},
    ]
    for over in bad:
        with pytest.raises(ValidationError):
            with_baseline(**over)

    with pytest.raises(ValueError, match="not model components"):
        bl.check_config(with_baseline(p={**b["p"], "components": "X"}))
    no_b = {k: v for k, v in b["prefilter"].items() if k != "borehole-B"}
    with pytest.raises(ValueError, match=r"missing \['borehole-B'\]"):
        bl.check_config(with_baseline(prefilter=no_b))
    with pytest.raises(ValueError, match=r"not a profile \['borehole-C'\]"):
        bl.check_config(
            with_baseline(prefilter={**b["prefilter"], "borehole-C": no_b["borehole-A"]})
        )
    # Nyquist is the REAL one of each profile: 50 Hz at the 100 Hz model rate, 500 Hz for the
    # time-stretched borehole-B (1000 Hz input), whose shipped band reaches 300 Hz.
    band = {"lowHz": 2.0, "corners": 4}
    for profile, high, nyquist in (("surface-100", 50.0, 50.0), ("borehole-B", 500.0, 500.0)):
        wide = {**b["prefilter"], profile: {**band, "highHz": high}}
        with pytest.raises(ValueError, match=f"real Nyquist {nyquist} Hz of profile {profile}"):
            bl.check_config(with_baseline(prefilter=wide))
    assert signal_cfg.baseline.prefilter["borehole-B"].highHz > 50.0

    def with_chunks(raw: dict[str, Any], overlap: float) -> SignalConfig:
        raw["preprocess"]["chunks"].update({"overlapS": overlap, "minOverlapS": 10.0})
        return SignalConfig.model_validate(raw)

    raw = copy.deepcopy(raw_signal_yaml)
    with pytest.raises(ValueError, match=r"max\(s.warmupS"):
        bl.check_config(with_chunks(raw, 11.5))  # S warm-up is 12 s
    raw["baseline"]["s"]["warmupS"] = raw["baseline"]["s"]["ltaS"]  # 4 s
    with pytest.raises(ValueError, match=r"p.warmupS \+ maxSMinusPS"):
        bl.check_config(with_chunks(raw, 10.5))  # P warm-up 6 s + 5 s S window
    raw = copy.deepcopy(raw_signal_yaml)
    with pytest.raises(ValueError, match="settleLtaMultiple"):
        bl.check_config(with_chunks(raw, 39.0))  # 10 x S ltaS 4 s = 40 s
    bl.check_config(with_chunks(raw, 40.0))
    raw["picker"]["gapEdgeS"] = 40.0
    with pytest.raises(ValueError, match="picker.gapEdgeS"):
        bl.check_config(with_chunks(raw, 40.0))  # an edge that close could lie outside the read


# --- tables and the stage --------------------------------------------------------------------------


@pytest.mark.smoke
def test_pick_id_format() -> None:
    assert bl.pick_id("UU.FOR1", "P", 1_789_063_200.12345) == "stalta:UU.FOR1:P:1789063200.123"
    assert bl.pick_id("6K.CS01.00", "S", 1_789_063_200.0) == "stalta:6K.CS01.00:S:1789063200.000"


class _StandInPick(BaseModel):
    """docs/02 ``Pick``, for the stand-in table io."""

    model_config = ConfigDict(extra="forbid")

    id: str
    stationId: str
    phase: Literal["P", "S"]
    t: float
    prob: float
    picker: str
    eventId: str | None = None
    residualS: float | None = None
    weight: float | None = None


@dataclass
class TableIO:
    io: Any
    written: dict[str, str] | None  # file name -> model name, stand-in only


@pytest.fixture(params=["hq_contracts", "stand-in"])
def table_io(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> TableIO:
    if request.param == "hq_contracts":
        io = pytest.importorskip("hq_contracts.io")  # CONTRACT-01 (H4) is not on this branch yet
        pytest.importorskip("hq_contracts.models")
        return TableIO(io=io, written=None)
    written: dict[str, str] = {}
    io_stub = types.ModuleType("hq_contracts.io")
    models_stub = types.ModuleType("hq_contracts.models")

    def to_frame(models: list[BaseModel], model: type[BaseModel] | None = None) -> pd.DataFrame:
        cls = type(models[0]) if model is None else model
        rows = [m.model_dump() for m in models]
        return pd.DataFrame.from_records(rows, columns=list(cls.model_fields))

    def write_table(df: pd.DataFrame, path: Path, model_name: str) -> None:
        written[Path(path).name] = model_name
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def read_table(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        df.attrs["model"] = written[Path(path).name]
        return df

    io_stub.to_frame = to_frame  # type: ignore[attr-defined]
    io_stub.write_table = write_table  # type: ignore[attr-defined]
    io_stub.read_table = read_table  # type: ignore[attr-defined]
    models_stub.Pick = _StandInPick  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hq_contracts.io", io_stub)
    monkeypatch.setitem(sys.modules, "hq_contracts.models", models_stub)
    return TableIO(io=io_stub, written=written)


@dataclass(frozen=True)
class StageConfig:
    run: RunSection
    signal: SignalConfig
    seismology: Any = None


def stage_ctx(
    fake_ctx: Any, run_section: RunSection, cfg: SignalConfig, name: str, seismology: Any = None
) -> Any:
    """The conftest ``FakeRunContext`` with a [T, T + 200) run window and its own run dir."""
    run = run_section.model_copy(
        update={
            "windowStart": datetime.fromtimestamp(T, UTC),
            "windowEnd": datetime.fromtimestamp(T + 200.0, UTC),
        }
    )
    run_dir = fake_ctx.run_dir.parent / name
    run_dir.mkdir()
    return dataclasses.replace(
        fake_ctx, run_dir=run_dir, config=StageConfig(run, cfg, seismology), records={}
    )


STATION_ROWS = [  # the columns the stage reads, plus identity
    {"id": "XX.A", "channels": ["HHZ", "HHN", "HHE"], "preprocessProfile": "surface-100"},
    {"id": "XX.B", "channels": ["HHZ", "HHN", "HHE"], "preprocessProfile": "surface-100"},
    {"id": "XX.C", "channels": ["HHZ", "HHN", "HHE"], "preprocessProfile": "surface-100"},
    {"id": "XX.OFF", "channels": ["HHZ", "HHN", "HHE"], "preprocessProfile": "surface-100"},
    {"id": "XX.GONE", "channels": ["HHZ", "HHN", "HHE"], "preprocessProfile": "surface-100"},
]


def seed_stage(tio: TableIO, ctx: Any, monkeypatch: pytest.MonkeyPatch) -> FakeCache:
    rows = [
        {**r, "network": "XX", "station": r["id"][3:], "usedInRun": r["id"] != "XX.OFF"}
        for r in STATION_ROWS
    ]
    tio.io.write_table(pd.DataFrame(rows), ctx.path("stations.parquet"), "Station")
    # A shared origin at T + 40: the same P/S moveout on every station, so picks interleave.
    pieces: dict[str, list[Trace]] = {}
    for i, sid in enumerate(("XX.A", "XX.B", "XX.C", "XX.OFF")):
        delay = 0.7 * i
        arr = Arrivals(
            p=[(T + 40.0 + delay, 30.0), (T + 130.0 + delay, 20.0)],
            s=[(T + 41.2 + 1.5 * delay, 60.0), (T + 131.2 + 1.5 * delay, 40.0)],
        )
        pieces[sid] = station_pieces(T - 60.0, T + 260.0, arr, station=sid[3:], seed=20 + i)
    cache = FakeCache(pieces, not_cached=frozenset({"XX.GONE"}))
    monkeypatch.setattr(chunks_module, "cache_read_window", cache.read_window)
    return cache


def missing_h2(monkeypatch: pytest.MonkeyPatch) -> None:
    """H2's API as absent modules, whatever is merged on the branch the test runs on."""
    api = tuple((f"hq_absent_for_test_{attr}", attr) for _, attr in sc._H2_API)
    monkeypatch.setattr(sc, "_H2_API", api)


def scoring_cfg(
    raw_signal_yaml: dict[str, Any], mode: str = "all", **baseline: Any
) -> SignalConfig:
    raw = {**copy.deepcopy(raw_signal_yaml["baseline"]["sweep"]), **copy.deepcopy(TEST_GRID)}
    return small_cfg(raw_signal_yaml, sweep={**raw, "scoreMode": mode}, **baseline)


@pytest.mark.smoke
def test_stage_writes_picks_and_a_null_sweep_while_h2_is_missing(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_h2(monkeypatch)
    cfg = scoring_cfg(raw_signal_yaml, "coordinate")
    ctx = stage_ctx(fake_ctx, run_section, cfg, "h2-missing")
    seed_stage(table_io, ctx, monkeypatch)
    stale = ctx.path(sc.REFERENCE_FILE)
    stale.write_text("{}", encoding="utf-8")  # from an earlier scored sweep
    with caplog.at_level(logging.INFO):
        stage.run(ctx)  # the registry entry point

    picks = table_io.io.read_table(ctx.path(bl.PICKS_FILE))
    assert picks.attrs["model"] == "Pick"
    assert list(picks.columns) == PICK_FIELDS
    assert len(picks) == 3 * 4  # 3 used, cached stations x 2 events x (P + S)
    assert set(picks["stationId"]) == {"XX.A", "XX.B", "XX.C"}  # not XX.OFF, XX.GONE has nothing
    assert (picks["picker"] == "stalta").all() and (picks["prob"] == cfg.baseline.prob).all()
    assert picks[["eventId", "residualS", "weight"]].isna().all().all()
    assert picks["t"].dtype == np.float64
    for row in picks.itertuples(index=False):
        assert row.id == f"stalta:{row.stationId}:{row.phase}:{row.t:.3f}"
    order = list(zip(picks["t"], picks["stationId"], picks["phase"], strict=True))
    assert order == sorted(order)
    assert picks["stationId"].tolist() != sorted(picks["stationId"])  # interleaved by time

    sweep = table_io.io.read_table(ctx.path(bl.SWEEP_FILE))
    assert sweep.attrs["model"] == "BaselineSweep"
    assert list(sweep.columns) == list(bl.SWEEP_COLUMNS)
    assert bl.SWEEP_COLUMNS == (
        "params",
        "candidates",
        "recoveredPublic",
        "tierA",
        "tierB",
        "tierC",
        "medianRmsS",
        "medianStations",
        "nP",
        "nS",
        "nStations",
    )
    grid = bl.sweep_grid(cfg.baseline)
    assert len(sweep) == len(grid)
    assert [json.loads(p) for p in sweep["params"]] == [g.params() for g in grid]
    assert sweep[list(bl.SWEEP_SCORE_COLUMNS)].isna().all().all()
    k = grid.index(bl.chosen_point(cfg.baseline))
    assert sweep.loc[k, "nP"] == (picks["phase"] == "P").sum()
    assert sweep.loc[k, "nS"] == (picks["phase"] == "S").sum()
    assert sweep.loc[k, "nStations"] == 3
    assert not stale.exists()  # it described another sweep

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "H2 pipeline not merged; sweep has pick counts only" in r.getMessage() for r in warnings
    )
    assert any("XX.GONE: not cached" in r.getMessage() for r in warnings)
    rec = ctx.records["baseline"]
    # Nested under the stage key (H4, REQ-H1-2): the pick stage owns ProcessingRun.picker's
    # top level, so baseline keys such as maxWorkers never overwrite another stage's.
    assert list(rec["params"]) == ["baseline"]
    params = dict(rec["params"]["baseline"])
    assert 0.0 < params.pop("pickingRuntimeS") <= rec["runtime_s"]
    assert params.pop("scoringRuntimeS") is None  # nothing scored
    assert params == {
        **cfg.baseline.model_dump(mode="json"),
        "pickerGapEdgeS": cfg.picker.gapEdgeS,
        "preprocessChunks": cfg.preprocess.chunks.model_dump(mode="json"),
        "sweepBestTierA": None,
        "sweepScoring": None,
    }
    counts = rec["counts"]
    assert counts["stations"] == 4 and counts["stationsWithPicks"] == 3
    assert counts["stationsNotCached"] == 1
    assert counts["nP"] == 6 and counts["nS"] == 6
    assert counts["sweepPoints"] == len(grid) and counts["sweepScored"] == 0
    assert counts["sweepScoresKept"] == 0
    assert "sweepBestTierA" not in counts and "sweepObjectiveFlat" not in counts
    assert counts["chunksYielded"] == 3 * 2
    assert counts["chunksPlanned"] == 3 * 2 + 1  # XX.GONE fails on its first read
    for key in ("sLostPNearGapEdge", "suppressedWarmupP", "suppressedWarmupS"):
        assert counts[key] == 0, key  # gap-free data, arrivals clear of every warm-up span
    assert all(isinstance(v, int) for v in counts.values())
    assert rec["runtime_s"] > 0.0
    out = capsys.readouterr().out
    assert "XX.GONE" in out and "not cached" in out and "XX.OFF" not in out
    if table_io.written is not None:
        assert table_io.written[bl.PICKS_FILE] == "Pick"


@pytest.mark.smoke
# Stand-in io only: the real hq_contracts leg runs in the other stage tests.
@pytest.mark.parametrize("table_io", ["stand-in"], indirect=True)
def test_stage_output_is_identical_across_worker_counts(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_h2(monkeypatch)
    grid = small_grid([4.0, 8.0], [4.0, 8.0], 1.5, (8.0, 8.0))
    frames = []
    for name, workers in (("serial", 1), ("threads", 3), ("rerun", 3)):
        cfg = small_cfg(raw_signal_yaml, maxWorkers=workers, **grid)
        ctx = stage_ctx(fake_ctx, run_section, cfg, name)
        seed_stage(table_io, ctx, monkeypatch)
        stage.run(ctx)
        frames.append(
            (
                table_io.io.read_table(ctx.path(bl.PICKS_FILE)),
                table_io.io.read_table(ctx.path(bl.SWEEP_FILE)),
            )
        )
    for picks, sw in frames[1:]:
        pd.testing.assert_frame_equal(picks, frames[0][0])
        pd.testing.assert_frame_equal(sw, frames[0][1])
    assert len(frames[0][0]) == 12


# --- scoring the sweep on VAL-01's scale -----------------------------------------------------------

# The run's ProcessingRun.tiering as H2's tier stage records it (only what the reruns read), with
# one rmsS bar per tier so barsMet has something to count.
RUN_TIERING: dict[str, Any] = {
    "thresholds": {
        "nMatched": 12,
        "quantiles": {"A": 0.25, "B": 0.0},
        "A": {"rmsS": {"op": "<=", "value": 0.15}},
        "B": {"rmsS": {"op": "<=", "value": 1.0}},
    },
    "thresholdSource": "derived",
}
TIER_RULES = {
    "A": {
        "nearestStation": {"focalDepthBelow": "nearestUsedSensor"},
        "mapOnVolumeTop": {"applied": False},
    }
}
# H2's SeismologyConfig: only the fields the sweep reads itself.
SEISMOLOGY = types.SimpleNamespace(
    associator=types.SimpleNamespace(minPickProb=0.3),
    locator=types.SimpleNamespace(nWorkers=2),
)


@dataclass(frozen=True)
class _H2Result:
    """Stand-in for H2's frozen result dataclasses (docs/02 section 5)."""

    events: pd.DataFrame
    picks: pd.DataFrame | None = None
    arrivals: pd.DataFrame | None = None
    statics: pd.DataFrame | None = None
    matches: pd.DataFrame | None = None
    sensitivity: pd.DataFrame | None = None
    tiering: dict[str, Any] | None = None


@dataclass
class FakeSeismologyApi:
    """A ``SeismologyApi`` whose counts follow the picks: one candidate event per three P picks
    (rmsS 0.1, 0.2, ...), the first matched to the first public event, every event Tier A except
    the last (Tier C); with ``flat`` every event is Tier C."""

    associated: list[pd.DataFrame] = field(default_factory=list)
    tier_kwargs: list[dict[str, Any]] = field(default_factory=list)
    located_args: list[int] = field(default_factory=list)
    flat: bool = False

    def associate(self, picks: pd.DataFrame, stations: pd.DataFrame, cfg: Any, run: Any) -> Any:
        self.associated.append(picks)
        n = int((picks["phase"] == "P").sum()) // 3
        return _H2Result(events=pd.DataFrame({"assocId": range(n)}), picks=pd.DataFrame())

    def locate(self, assoc: Any, *args: Any) -> Any:
        self.located_args.append(len(args))
        n = len(assoc.events)
        events = pd.DataFrame(
            {
                "id": [f"ev{i}" for i in range(n)],
                "quality_rmsS": [0.1 * (i + 1) for i in range(n)],
                "quality_nStations": [3] * n,
            }
        )
        return _H2Result(events=events, arrivals=pd.DataFrame({"eventId": events["id"]}))

    def match(self, events: pd.DataFrame, catalog: pd.DataFrame, cfg: Any) -> Any:
        ids = [events["id"].iloc[0]] + [None] * (len(catalog) - 1)
        return _H2Result(
            events=events, matches=pd.DataFrame({"catalogId": catalog["id"], "eventId": ids})
        )

    def assign_tiers(
        self, events: pd.DataFrame, matches: pd.DataFrame, cfg: Any, **kwargs: Any
    ) -> Any:
        self.tier_kwargs.append(kwargs)
        tiers = np.array(["C"] * len(events) if self.flat else ["A"] * (len(events) - 1) + ["C"])
        matched = events["id"].isin(set(matches["eventId"].dropna())).to_numpy(dtype=bool)

        def count(mask: np.ndarray) -> dict[str, int]:
            return {t: int(((tiers == t) & mask).sum()) for t in "ABC"}

        counts = {
            "events": len(tiers),
            "all": count(np.ones(len(tiers), dtype=bool)),
            "matched": count(matched),
            "additional": count(~matched),
        }
        tiering = {"thresholdSource": "supplied", "rules": TIER_RULES, "counts": counts}
        return _H2Result(events=events.assign(tier=tiers.tolist()), tiering=tiering)


def reversed_runner(api: Any, batches: list[list[str]]) -> Any:
    """Scores each batch in this process, completing the jobs in reverse submission order."""

    def run(shared: Any, jobs: Any) -> Any:
        todo = list(jobs)
        batches.append([j.key for j in todo])
        for job in reversed(todo):
            yield sc.score_job(shared, job, api)

    return run


@dataclass(frozen=True)
class _RunRecord:
    tiering: dict[str, Any]


@dataclass(frozen=True)
class ReadRunContext:
    """The conftest context plus H4's ``read_run()`` (docs/02 section 4)."""

    inner: Any
    tiering: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def read_run(self) -> _RunRecord:
        return _RunRecord(self.tiering)


def with_config(ctx: Any, cfg: SignalConfig, seismology: Any = SEISMOLOGY) -> Any:
    """The same run directory under another signal config (fresh records)."""
    inner = ctx.inner if isinstance(ctx, ReadRunContext) else ctx
    replaced = dataclasses.replace(
        inner, config=StageConfig(inner.config.run, cfg, seismology), records={}
    )
    return ReadRunContext(replaced, RUN_TIERING)


PHASENET_OUTSIDE = 3  # reference picks outside the sweep's stations or window


def seed_scoring(
    tio: TableIO,
    ctx: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    phasenet_p: int = 9,
    outside: bool = True,
) -> None:
    """``seed_stage`` plus the public catalog and a PhaseNet picks.parquet (with
    ``PHASENET_OUTSIDE`` picks outside the sweep unless ``outside`` is false); H2 counts as
    merged."""
    from hq_contracts.io import to_frame
    from hq_contracts.models import Pick

    seed_stage(tio, ctx, monkeypatch)
    tio.io.write_table(
        pd.DataFrame({"id": ["pub-1", "pub-2"]}), ctx.path("catalog.parquet"), "CatalogEvent"
    )
    rows = [("XX.A", "XX.B", "XX.C")[i % 3] for i in range(phasenet_p)]
    models = [
        Pick(
            id=f"pn:{sid}:{i}",
            stationId=sid,
            phase="P",
            t=T + 10.0 * i,
            prob=0.9,
            picker="phasenet:test",
        )
        for i, sid in enumerate(rows)
    ]
    if outside:
        models += [  # not in the sweep: an unused station, and times outside [T, T + 200)
            Pick(
                id="pn:off",
                stationId="XX.OFF",
                phase="P",
                t=T + 5.0,
                prob=0.9,
                picker="phasenet:test",
            ),
            Pick(
                id="pn:early",
                stationId="XX.A",
                phase="P",
                t=T - 1.0,
                prob=0.9,
                picker="phasenet:test",
            ),
            Pick(
                id="pn:late",
                stationId="XX.A",
                phase="P",
                t=T + 200.0,
                prob=0.9,
                picker="phasenet:test",
            ),
        ]
    tio.io.write_table(to_frame(models, Pick), ctx.path(sc.PHASENET_PICKS_FILE), "Pick")
    monkeypatch.setattr(sc, "check_h2_merged", lambda: None)  # H2's modules are not imported


def sweep_cfg(
    raw_signal_yaml: dict[str, Any],
    mode: str,
    p_on: list[float],
    s_on: list[float],
    off: list[float],
    chosen: tuple[float, float, float],
    **sweep: Any,
) -> SignalConfig:
    return small_cfg(
        raw_signal_yaml,
        sweep={
            "pOn": p_on,
            "sOn": s_on,
            "offLevels": off,
            "scoreMode": mode,
            "scoreWorkers": 2,
            "maxPasses": 3,
            **sweep,
        },
        chosen={"pOn": chosen[0], "pOff": chosen[2], "sOn": chosen[1], "sOff": chosen[2]},
    )


USED_STATIONS = ["XX.A", "XX.B", "XX.C", "XX.GONE"]  # usedInRun, sorted: what the sweep scores


@pytest.mark.smoke
@pytest.mark.parametrize("table_io", ["hq_contracts"], indirect=True)
def test_all_mode_scores_every_point_on_the_runs_bars_in_grid_order(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # pOn 40 picks nothing, so it scores lowest; baseline.chosen sits there.
    cfg = sweep_cfg(raw_signal_yaml, "all", [4.0, 8.0, 40.0], [4.0], [1.5], (40.0, 4.0, 1.5))
    ctx = ReadRunContext(
        stage_ctx(fake_ctx, run_section, cfg, "score-all", seismology=SEISMOLOGY), RUN_TIERING
    )
    seed_scoring(table_io, ctx, monkeypatch)
    api, batches = FakeSeismologyApi(), []
    with caplog.at_level(logging.INFO):
        res = bl.run_baseline(ctx, runner=reversed_runner(api, batches))

    # The reference alone first (it warms H2's table cache), then every grid point in one batch,
    # completed in reverse order.
    assert batches == [["phasenet"], ["grid point 0", "grid point 1", "grid point 2"]]
    sweep = table_io.io.read_table(ctx.path(bl.SWEEP_FILE))
    assert [json.loads(p)["pOn"] for p in sweep["params"]] == [4.0, 8.0, 40.0]  # grid order
    assert sweep["nP"].tolist() == [6, 6, 0]
    assert sweep["candidates"].tolist() == [2, 2, 0]
    assert sweep["recoveredPublic"].tolist() == [1, 1, 0]
    assert sweep["tierA"].tolist() == [1, 1, 0]
    assert sweep["tierB"].tolist() == [0, 0, 0]
    assert sweep["tierC"].tolist() == [1, 1, 0]
    assert sweep["medianRmsS"].tolist() == pytest.approx([0.15, 0.15, 0.0])
    assert sweep["medianStations"].tolist() == [3.0, 3.0, 0.0]
    assert str(sweep["tierB"].dtype) == "Int64" and sweep["medianRmsS"].dtype == np.float64

    # VAL-01's rerun: the run's own bars, with arrivals and stations (REQ-H2-9), 5-arg locate.
    assert len(api.tier_kwargs) == 3  # the reference and two points; pOn 40 associates nothing
    for kwargs in api.tier_kwargs:
        assert kwargs["thresholds"] == RUN_TIERING
        assert set(kwargs) == {"thresholds", "arrivals", "stations"}
        assert list(kwargs["stations"]["id"]) == [r["id"] for r in STATION_ROWS]
    assert api.located_args == [4, 4, 4]
    # The reference is PhaseNet's picks.parquet, once, cut to the sweep's stations and window.
    reference = [p for p in api.associated if (p["picker"] != "stalta").any()]
    assert len(reference) == 1 and len(reference[0]) == 9
    assert set(reference[0]["id"]) == {f"pn:XX.{'ABC'[i % 3]}:{i}" for i in range(9)}
    for picks in api.associated:
        assert list(picks.columns) == PICK_FIELDS
        assert (picks["prob"] == 0.9).all() or (picks["prob"] == cfg.baseline.prob).all()

    best = {"pOn": 4.0, "pOff": 1.5, "sOn": 4.0, "sOff": 1.5}  # ties: pOn 4 before 8 in grid order
    doc = json.loads(ctx.path(sc.REFERENCE_FILE).read_text(encoding="utf-8"))
    assert doc["mode"] == "all" and doc["associationProfile"] == "full"
    assert doc["profileNote"] == sc.PROFILE_NOTE
    assert doc["pointsScored"] == 3 and doc["gridPoints"] == 3 and doc["scoreWorkers"] == 2
    assert doc["window"] == {"t0": T, "t1": T + 200.0} and doc["stationIds"] == USED_STATIONS
    assert doc["objective"] == {
        "metric": sc.OBJECTIVE,
        "maxTierA": 1,
        "tierAValues": [0, 1],
        "flat": False,
        "note": None,
    }
    ref = doc["phasenetReference"]
    assert ref["picks"] == 9 and ref["row"]["method"] == "phasenet"
    assert ref["row"]["tiers"] == {"A": 2, "B": 0, "C": 1}
    assert ref["tierCounts"]["matched"] == {"A": 1, "B": 0, "C": 0}
    assert ref["tierCounts"]["additional"] == {"A": 1, "B": 0, "C": 1}
    assert ref["barsMet"]["A"] == {"perBar": {"rmsS": 1}, "everyBar": 1}  # rmsS 0.1 only
    assert doc["best"]["params"] == best and doc["best"]["row"]["tiers"]["A"] == 1
    assert doc["best"]["row"]["method"] == "stalta"
    chosen = doc["chosenAtScoring"]
    assert chosen["params"]["pOn"] == 40.0 and chosen["row"]["candidates"] == 0
    assert doc["tierARatioToPhasenet"] == 0.5
    assert doc["search"] == {"scope": sc.SCOPE["all"], "converged": True, "path": []}
    points = doc["points"]
    assert [p["index"] for p in points] == [0, 1, 2] and points[0]["params"] == best
    assert points[0]["picks"] == int(sweep["nP"][0] + sweep["nS"][0])
    assert points[0]["row"]["tiers"]["A"] == 1
    assert points[0]["tierCounts"] == {
        "all": {"A": 1, "B": 0, "C": 1},
        "matched": {"A": 1, "B": 0, "C": 0},
        "additional": {"A": 0, "B": 0, "C": 1},
    }
    assert points[0]["barsMet"] == {
        "A": {"perBar": {"rmsS": 1}, "everyBar": 1},
        "B": {"perBar": {"rmsS": 2}, "everyBar": 2},
    }
    assert points[2]["tierCounts"] is None  # associated nothing: never tiered
    assert points[2]["barsMet"]["A"] == {"perBar": {"rmsS": 0}, "everyBar": 0}
    notes = doc["notes"]
    assert notes["reruns"] == 4 and notes["rerunsTiered"] == 3
    assert notes["thresholds"]["nMatched"] == 12
    assert notes["tieringRules"]["thresholdSource"] == "supplied"

    rec = ctx.records["baseline"]
    params = rec["params"]["baseline"]
    assert params["sweepBestTierA"] == best
    assert params["sweepScoring"] == sc.record_view(doc) and "points" not in params["sweepScoring"]
    assert params["scoringRuntimeS"] > 0.0 and params["pickingRuntimeS"] > 0.0
    assert rec["counts"]["sweepScored"] == 3 and rec["counts"]["sweepObjectiveFlat"] == 0
    assert rec["counts"]["sweepBestTierA"] == 1 and rec["counts"]["phasenetReferenceTierA"] == 2
    assert res.scores is not None and res.scores.best == 0 and not res.scores.flat
    assert any(
        r.levelno == logging.WARNING and "is not the best Tier A point" in r.getMessage()
        for r in caplog.records
    )
    per_point = [r.getMessage() for r in caplog.records if "median stations" in r.getMessage()]
    assert len(per_point) == 4
    assert per_point[0].startswith("baseline sweep PhaseNet reference")
    assert per_point[1].startswith("baseline sweep point 3/3")
    assert "A bars met: rmsS 1; every A bar 1" in per_point[0]


@pytest.mark.smoke
@pytest.mark.parametrize("table_io", ["hq_contracts"], indirect=True)
def test_coordinate_mode_follows_the_search_and_leaves_other_points_null(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Grid pOn [4, 40] x sOn [4, 8] x off [1.5, 2.0]; chosen (40, 8, 2.0) picks nothing.
    cfg = sweep_cfg(
        raw_signal_yaml, "coordinate", [4.0, 40.0], [4.0, 8.0], [1.5, 2.0], (40.0, 8.0, 2.0)
    )
    ctx = ReadRunContext(
        stage_ctx(fake_ctx, run_section, cfg, "score-cd", seismology=SEISMOLOGY), RUN_TIERING
    )
    seed_scoring(table_io, ctx, monkeypatch)
    api, batches = FakeSeismologyApi(), []
    bl.run_baseline(ctx, runner=reversed_runner(api, batches))

    grid = bl.sweep_grid(cfg.baseline)
    at = {g.coords(): k for k, g in enumerate(grid)}
    key = {c: f"grid point {k}" for c, k in at.items()}
    # The reference alone; pass 1: pOn at (sOn 8, off 2) moves to 4; sOn and off tie, so the
    # incumbent stays. Pass 2 only revisits scored points, moves nowhere, and the search stops.
    assert batches == [
        ["phasenet"],
        [key[(4.0, 8.0, 2.0)], key[(40.0, 8.0, 2.0)]],
        [key[(4.0, 4.0, 2.0)]],
        [key[(4.0, 8.0, 1.5)]],
    ]
    sweep = table_io.io.read_table(ctx.path(bl.SWEEP_FILE))
    scored = {at[(4.0, 8.0, 2.0)], at[(40.0, 8.0, 2.0)], at[(4.0, 4.0, 2.0)], at[(4.0, 8.0, 1.5)]}
    for k in range(len(grid)):
        assert sweep.loc[k, list(bl.SWEEP_SCORE_COLUMNS)].isna().all() == (k not in scored), k
    doc = json.loads(ctx.path(sc.REFERENCE_FILE).read_text(encoding="utf-8"))
    assert doc["best"]["params"] == {"pOn": 4.0, "pOff": 2.0, "sOn": 8.0, "sOff": 2.0}
    assert doc["search"]["scope"] == sc.SCOPE["coordinate"]
    path = doc["search"]["path"]
    assert [(s["pass"], s["axis"], s["moved"]) for s in path] == [
        (1, "pOn", True),
        (1, "sOn", False),
        (1, "off", False),
        (2, "pOn", False),
        (2, "sOn", False),
        (2, "off", False),
    ]
    assert [p["tierA"] for p in path[0]["points"]] == [1, 0]
    assert doc["search"]["converged"] is True and doc["pointsScored"] == 4
    assert [p["index"] for p in doc["points"]] == sorted(scored)
    assert len([p for p in api.associated if (p["picker"] != "stalta").any()]) == 1


@pytest.mark.smoke
@pytest.mark.parametrize("table_io", ["hq_contracts"], indirect=True)
def test_a_flat_objective_names_no_best_point_and_warns(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = sweep_cfg(raw_signal_yaml, "coordinate", [4.0, 8.0], [4.0, 8.0], [1.5], (8.0, 8.0, 1.5))
    ctx = ReadRunContext(
        stage_ctx(fake_ctx, run_section, cfg, "flat", seismology=SEISMOLOGY), RUN_TIERING
    )
    seed_scoring(table_io, ctx, monkeypatch)
    with caplog.at_level(logging.INFO):
        res = bl.run_baseline(ctx, runner=reversed_runner(FakeSeismologyApi(flat=True), []))

    assert res.scores is not None and res.scores.flat and res.scores.tier_a_values == [0]
    doc = json.loads(ctx.path(sc.REFERENCE_FILE).read_text(encoding="utf-8"))
    assert doc["objective"]["flat"] is True and doc["objective"]["note"] == sc.FLAT_NOTE
    assert doc["objective"]["maxTierA"] == 0 and doc["objective"]["tierAValues"] == [0]
    assert doc["best"] is None  # the tie-break winner is not presented as a best point
    assert doc["chosenAtScoring"]["params"] == {"pOn": 8.0, "pOff": 1.5, "sOn": 8.0, "sOff": 1.5}
    assert doc["tierARatioToPhasenet"] is None  # the reference has no Tier A event either
    # One pass: pOn and sOn lines tie, the off line is the incumbent alone.
    assert doc["search"]["converged"] is True and doc["pointsScored"] == 3
    rec = ctx.records["baseline"]
    assert rec["params"]["baseline"]["sweepBestTierA"] is None
    assert rec["counts"]["sweepObjectiveFlat"] == 1 and rec["counts"]["sweepBestTierA"] == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("The objective is flat" in m and "3 scored point(s) of 4" in m for m in warnings)
    assert not any("is not the best Tier A point" in m for m in warnings)


@pytest.mark.smoke
def test_coordinate_search_visits_axis_lines_and_keeps_the_incumbent_on_ties() -> None:
    coords = [(p, s, o) for p in (1.0, 2.0, 3.0) for s in (1.0, 2.0) for o in (0.5,)]
    landscape = {(1.0, 1.0): 2, (2.0, 1.0): 5, (3.0, 1.0): 5, (1.0, 2.0): 1,
                 (2.0, 2.0): 7, (3.0, 2.0): 7}  # fmt: skip
    asked: list[list[int]] = []
    seen: set[int] = set()

    def score(indices: Any) -> dict[int, int]:
        asked.append([k for k in indices if k not in seen])
        seen.update(indices)
        return {k: landscape[coords[k][:2]] for k in indices}

    start = coords.index((1.0, 1.0, 0.5))
    found = sc.coordinate_search(coords, start, 5, score)
    # pOn at sOn 1: 5 at pOn 2 and 3 -> 2 (grid order); sOn at pOn 2: 7 at sOn 2 -> move.
    # Pass 2: pOn at sOn 2 ties 2 and 3 at 7: the incumbent (pOn 2) stays. Converged.
    assert coords[found.best] == (2.0, 2.0, 0.5)
    assert found.converged
    assert [(s.passNo, s.axis, s.moved) for s in found.steps] == [
        (1, "pOn", True), (1, "sOn", True), (1, "off", False),
        (2, "pOn", False), (2, "sOn", False), (2, "off", False),
    ]  # fmt: skip
    assert asked[0] == [0, 2, 4] and asked[1] == [3] and asked[3] == [1, 5]
    assert sorted(seen) == [0, 1, 2, 3, 4, 5]
    # A limit of one pass stops while the pass still moved.
    limited = sc.coordinate_search(coords, start, 1, score)
    assert not limited.converged and coords[limited.best] == (2.0, 2.0, 0.5)
    assert sc.best_of([0, 1, 2], {0: 3, 1: 3, 2: 1}, 1) == 1  # incumbent first on ties
    assert sc.best_of([0, 1, 2], {0: 3, 1: 3, 2: 1}, 2) == 0  # then grid order
    assert sc.best_of([0, 1, 2], {0: 3, 1: 4, 2: 1}, 0) == 1


@pytest.mark.smoke
@pytest.mark.parametrize("table_io", ["hq_contracts"], indirect=True)
def test_scoring_without_the_runs_tier_bars_fails_before_picking(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = sweep_cfg(raw_signal_yaml, "coordinate", [4.0, 8.0], [4.0], [1.5], (8.0, 4.0, 1.5))
    base = stage_ctx(fake_ctx, run_section, cfg, "no-bars", seismology=SEISMOLOGY)
    seed_scoring(table_io, base, monkeypatch)

    def must_not_pick(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no picking before the scoring inputs are checked")

    monkeypatch.setattr(bl, "pick_station", must_not_pick)
    for tiering in ({}, {"thresholds": None}):  # stage tier has not run
        with pytest.raises(ValidateError, match="run stage tier first"):
            bl.run_baseline(ReadRunContext(base, tiering))
    with pytest.raises(ValidateError, match="H2 Seismology"):
        bl.run_baseline(base, tiering=lambda: None)
    with pytest.raises(TypeError, match="no read_run"):  # docs/02 stand-in: say what is missing
        bl.run_baseline(base)
    # Association must see every STA/LTA pick, and the locate budget must be known.
    strict = types.SimpleNamespace(
        associator=types.SimpleNamespace(minPickProb=0.99), locator=SEISMOLOGY.locator
    )
    with pytest.raises(ValueError, match="below seismology.associator.minPickProb 0.99"):
        bl.run_baseline(with_config(base, cfg, strict))
    no_locator = types.SimpleNamespace(associator=SEISMOLOGY.associator)
    with pytest.raises(ValueError, match="seismology.locator.nWorkers"):
        bl.run_baseline(with_config(base, cfg, no_locator))
    ctx = ReadRunContext(base, RUN_TIERING)
    ctx.path(sc.PHASENET_PICKS_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="stage pick"):
        bl.run_baseline(ctx)
    assert not ctx.path(bl.PICKS_FILE).exists()
    with pytest.raises(ValueError, match="score-only needs a score mode"):
        bl.run_baseline(ctx, score_mode="none", write_picks=False)
    with pytest.raises(ValueError, match="keep-scores keeps an earlier scoring"):
        bl.run_baseline(ctx, keep_scores=True)  # scoreMode coordinate here


@pytest.mark.smoke
@pytest.mark.parametrize("table_io", ["stand-in"], indirect=True)
def test_score_only_fails_while_h2_is_missing(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_h2(monkeypatch)
    cfg = sweep_cfg(raw_signal_yaml, "coordinate", [4.0, 8.0], [4.0], [1.5], (8.0, 4.0, 1.5))
    ctx = stage_ctx(fake_ctx, run_section, cfg, "score-only-h2", seismology=SEISMOLOGY)
    seed_stage(table_io, ctx, monkeypatch)
    # An explicit scoring request must not end in a null sweep that replaces a scored one.
    with pytest.raises(sc.H2PipelineMissingError, match="H2 pipeline not merged"):
        bl.run_baseline(ctx, write_picks=False)
    assert not ctx.path(bl.SWEEP_FILE).exists()


@pytest.mark.smoke
@pytest.mark.parametrize("table_io", ["hq_contracts"], indirect=True)
def test_default_runner_binds_the_runs_cache_and_id_and_score_only_keeps_the_picks(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = sweep_cfg(
        raw_signal_yaml, "none", [4.0, 8.0], [4.0], [1.5], (8.0, 4.0, 1.5), scoreWorkers=1
    )
    ctx = ReadRunContext(
        stage_ctx(fake_ctx, run_section, cfg, "default-runner", seismology=SEISMOLOGY),
        RUN_TIERING,
    )
    seed_scoring(table_io, ctx, monkeypatch)
    api = FakeSeismologyApi()
    bound: list[dict[str, Any]] = []

    def fake_real_api(**kwargs: Any) -> Any:
        bound.append(kwargs)
        return api

    monkeypatch.setattr(sc, "real_seismology_api", fake_real_api)
    with pytest.raises(FileNotFoundError, match="score only leaves"):
        bl.run_baseline(ctx, score_mode="all", write_picks=False)  # nothing published yet
    bl.run_baseline(ctx)  # publishes picks_stalta.parquet; scoreMode none scores nothing
    published = ctx.path(bl.PICKS_FILE).read_bytes()
    res = bl.run_baseline(ctx, score_mode="all", write_picks=False)  # the CLI's --score-only
    assert bound == [{"cache_dir": ctx.cache_dir, "run_id": ctx.run_id}]  # REQ-H2-8, once
    assert ctx.path(bl.PICKS_FILE).read_bytes() == published
    assert res.scores is not None and res.scores.scored == 2
    assert len(api.associated) == 3  # the reference and both points, in this process
    assert ctx.records["baseline"]["params"]["baseline"]["sweepScoring"]["mode"] == "all"

    # Published picks of other thresholds, or of another prob, are not what the sweep scores.
    moved = sweep_cfg(  # chosen pOn 40 picks nothing
        raw_signal_yaml, "none", [4.0, 8.0, 40.0], [4.0], [1.5], (40.0, 4.0, 1.5), scoreWorkers=1
    )
    with pytest.raises(ValueError, match="other picks than the published file"):
        bl.run_baseline(with_config(ctx, moved), score_mode="all", write_picks=False)
    other_prob = cfg.model_copy(update={"baseline": cfg.baseline.model_copy(update={"prob": 0.5})})
    with pytest.raises(ValueError, match="written with prob"):
        bl.run_baseline(with_config(ctx, other_prob), score_mode="all", write_picks=False)
    assert ctx.path(bl.PICKS_FILE).read_bytes() == published
    assert len(api.associated) == 3  # both failed before scoring


@pytest.mark.smoke
@pytest.mark.parametrize("table_io", ["hq_contracts"], indirect=True)
def test_keep_scores_adopts_a_new_chosen_without_losing_the_scoring(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = sweep_cfg(raw_signal_yaml, "all", [4.0, 8.0, 40.0], [4.0], [1.5], (40.0, 4.0, 1.5))
    ctx = ReadRunContext(
        stage_ctx(fake_ctx, run_section, cfg, "keep", seismology=SEISMOLOGY), RUN_TIERING
    )
    seed_scoring(table_io, ctx, monkeypatch)
    bl.run_baseline(ctx, runner=reversed_runner(FakeSeismologyApi(), []))
    sweep_bytes = ctx.path(bl.SWEEP_FILE).read_bytes()
    ref_text = ctx.path(sc.REFERENCE_FILE).read_text(encoding="utf-8")
    doc = json.loads(ref_text)
    assert doc["best"]["params"]["pOn"] == 4.0

    # The lead copies the best point into chosen; the grid stays. Only the picks are rewritten.
    adopted = with_config(
        ctx, sweep_cfg(raw_signal_yaml, "none", [4.0, 8.0, 40.0], [4.0], [1.5], (4.0, 4.0, 1.5))
    )
    caplog.clear()  # the scoring run above warned about the old chosen
    with caplog.at_level(logging.INFO):
        res = bl.run_baseline(adopted, keep_scores=True)
    picks = table_io.io.read_table(adopted.path(bl.PICKS_FILE))
    assert res.counts["nP"] == 6 and len(picks) == res.counts["nP"] + res.counts["nS"]
    assert adopted.path(bl.SWEEP_FILE).read_bytes() == sweep_bytes
    assert adopted.path(sc.REFERENCE_FILE).read_text(encoding="utf-8") == ref_text
    rec = adopted.records["baseline"]
    assert rec["counts"]["sweepScoresKept"] == 1 and rec["counts"]["sweepScored"] == 3
    assert rec["counts"]["sweepBestTierA"] == 1
    assert rec["params"]["baseline"]["sweepScoring"] == sc.record_view(doc)
    assert rec["params"]["baseline"]["sweepBestTierA"] == doc["best"]["params"]
    assert rec["params"]["baseline"]["scoringRuntimeS"] is None
    assert not any("is not the best Tier A point" in r.getMessage() for r in caplog.records)

    # Scores of other pick sets are never kept, and a refusal writes nothing.
    published = adopted.path(bl.PICKS_FILE).read_bytes()
    other_grid = sweep_cfg(raw_signal_yaml, "none", [4.0, 8.0], [4.0], [1.5], (4.0, 4.0, 1.5))
    with pytest.raises(ValueError, match="another grid"):
        bl.run_baseline(with_config(ctx, other_grid), keep_scores=True)
    with pytest.raises(ValueError, match="scored stationIds"):  # XX.GONE has no picks anyway
        bl.run_baseline(adopted, keep_scores=True, station_ids=["XX.A", "XX.B", "XX.C"])
    assert adopted.path(bl.PICKS_FILE).read_bytes() == published
    with pytest.raises(ValueError, match="needs score mode none"):
        bl.run_baseline(adopted, keep_scores=True, score_mode="all")
    earlier = {k: v for k, v in doc.items() if k != "objective"}  # an earlier scorer's file
    adopted.path(sc.REFERENCE_FILE).write_text(json.dumps(earlier), encoding="utf-8")
    with pytest.raises(ValueError, match=r"lacks \['objective'\]"):
        bl.run_baseline(adopted, keep_scores=True)
    adopted.path(sc.REFERENCE_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="keep-scores keeps an earlier scoring"):
        bl.run_baseline(adopted, keep_scores=True)


@pytest.mark.smoke
@pytest.mark.parametrize("table_io", ["hq_contracts"], indirect=True)
def test_sweep_rows_equal_val01_rows_on_the_same_picks(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hq.config.validate import BaselineConfig as ValidateBaselineConfig
    from hq.validate.baseline import baseline_reruns

    cfg = sweep_cfg(raw_signal_yaml, "all", [4.0, 8.0], [4.0], [1.5], (8.0, 4.0, 1.5))
    ctx = ReadRunContext(
        stage_ctx(fake_ctx, run_section, cfg, "val01", seismology=SEISMOLOGY), RUN_TIERING
    )
    seed_scoring(table_io, ctx, monkeypatch, outside=False)  # VAL-01 reads all of picks.parquet
    res = bl.run_baseline(ctx, runner=reversed_runner(FakeSeismologyApi(), []))
    read = table_io.io.read_table
    reruns = baseline_reruns(  # H4's VAL-01 on the published tables
        read(ctx.path(sc.PHASENET_PICKS_FILE)),
        read(ctx.path(bl.PICKS_FILE)),
        read(ctx.path("stations.parquet")),
        read(ctx.path(sc.CATALOG_FILE)),
        FakeSeismologyApi(),
        SEISMOLOGY,
        ctx.config.run,
        ValidateBaselineConfig(profiles=["full"]),
        thresholds=RUN_TIERING,
    )
    assert res.scores is not None
    chosen = res.scores.chosen
    assert [r.row for r in reruns] == [res.scores.reference.row, res.scores.rows[chosen]]
    assert [r.tiering for r in reruns] == [
        res.scores.reference.tiering,
        res.scores.results[chosen].tiering,
    ]
    assert reruns[1].row.candidates > 0  # a real comparison, not two empty rows


@pytest.mark.smoke
# Stand-in io only: the real hq_contracts leg runs in the other stage tests.
@pytest.mark.parametrize("table_io", ["stand-in"], indirect=True)
def test_sweep_scoring_can_be_switched_off(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def must_not_load() -> Any:
        raise AssertionError("H2 must not be looked up when sweep.scoreMode is none")

    monkeypatch.setattr(sc, "check_h2_merged", must_not_load)
    grid = small_grid([4.0, 8.0], [4.0], 1.5, (4.0, 4.0), mode="none")
    cfg = small_cfg(raw_signal_yaml, **grid)
    ctx = stage_ctx(fake_ctx, run_section, cfg, "no-score")
    seed_stage(table_io, ctx, monkeypatch)
    with caplog.at_level(logging.INFO):
        stage.run(ctx)  # no read_run on this context: never asked for when not scoring
    out = table_io.io.read_table(ctx.path(bl.SWEEP_FILE))
    assert out[list(bl.SWEEP_SCORE_COLUMNS)].isna().all().all()
    assert out["nP"].tolist() == [6, 6]
    assert ctx.records["baseline"]["counts"]["sweepScored"] == 0
    assert any("scoreMode none" in r.getMessage() for r in caplog.records)
    assert not any("H2 pipeline not merged" in r.getMessage() for r in caplog.records)
    assert not ctx.path(sc.REFERENCE_FILE).exists()


@pytest.mark.smoke
def test_config_score_modes_match_the_scorer(signal_cfg: SignalConfig) -> None:
    from typing import get_args

    from hq.config.signal import BaselineSweep

    annotation = BaselineSweep.model_fields["scoreMode"].annotation
    assert get_args(annotation) == get_args(sc.ScoreMode) == bl.SCORE_MODES
    # Pinned on purpose: stage tier runs after this one, so a shipped scoreMode other than none
    # would stop every fresh `hq run` here. Scoring is a CLI rerun (--score-mode, signal.yaml).
    assert signal_cfg.baseline.sweep.scoreMode == "none"
    assert signal_cfg.baseline.sweep.scoreWorkers >= 1
    assert signal_cfg.baseline.sweep.maxPasses >= 1


@pytest.mark.smoke
def test_bars_met_counts_each_bar_and_all_at_once() -> None:
    events = pd.DataFrame(
        {"quality_nStations": [20.0, 5.0, 25.0, None], "quality_rmsS": [0.01, 0.02, 0.5, 0.01]}
    )
    record = {
        "A": {"nStations": {"op": ">=", "value": 20.0}, "rmsS": {"op": "<=", "value": 0.03}},
        "B": {"nStations": {"op": ">=", "value": 5.0}},
    }
    assert sc.bars_met(events, record) == {  # a null never meets a bar
        "A": {"perBar": {"nStations": 2, "rmsS": 3}, "everyBar": 1},
        "B": {"perBar": {"nStations": 3}, "everyBar": 3},
    }
    assert sc.bars_met(events.iloc[:0], record)["A"] == {
        "perBar": {"nStations": 0, "rmsS": 0},
        "everyBar": 0,
    }
    with pytest.raises(ValueError, match="quality_gapDeg"):
        sc.bars_met(events, {"A": {"gapDeg": {"op": "<=", "value": 1.0}}, "B": {}})
    with pytest.raises(ValueError, match="op '<'"):
        sc.bars_met(events, {"A": {"rmsS": {"op": "<", "value": 1.0}}, "B": {}})
    with pytest.raises(TypeError, match="no 'B' record"):
        sc.bars_met(events, {"A": {}})
    assert sc.tier_counts(None) is None
    with pytest.raises(ValueError, match="without counts"):
        sc.tier_counts({"counts": {"all": {"A": 1}}})


@pytest.mark.smoke
def test_process_budget_warns_above_the_cpu_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sc.os, "cpu_count", lambda: 8)
    with caplog.at_level(logging.INFO):
        sc.log_process_budget(4, SEISMOLOGY)  # 4 x 2 = 8 locate processes
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    with caplog.at_level(logging.INFO):
        sc.log_process_budget(5, SEISMOLOGY)  # 10 > 8
    assert any(
        r.levelno == logging.WARNING and "exceed the 8 CPUs" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.smoke
def test_an_h2_error_names_the_pick_set(run_section: RunSection, tmp_path: Path) -> None:
    class Broken(FakeSeismologyApi):
        def associate(self, *args: Any) -> Any:
            raise ValueError("associator failed")

    shared = sc.ScoreShared(
        stations=pd.DataFrame({"id": ["XX.A"]}),
        catalog=pd.DataFrame({"id": ["pub-1"]}),
        seismology=None,
        run=run_section,
        thresholds=RUN_TIERING,
        cache_dir=tmp_path,
        run_id="test-run",
        api_factory=lambda cache_dir, run_id: Broken(),
    )
    picks = pd.DataFrame({c: [] for c in PICK_FIELDS})
    with pytest.raises(ValueError, match="associator failed") as err:
        list(sc.SerialRunner()(shared, [sc.ScoreJob("grid point 4", "stalta", picks, index=4)]))
    assert "baseline sweep: scoring grid point 4" in err.value.__notes__


@pytest.mark.smoke
def test_cli_score_flags_are_checked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "showcase"
    argv = [
        "--run-dir",
        str(tmp_path),
        "--config-dir",
        str(config_dir),
        "--cache-dir",
        str(tmp_path),
    ]
    with pytest.raises(SystemExit) as exc:
        bl.main([*argv, "--score-only"])  # the shipped scoreMode is none: nothing to score
    assert exc.value.code == 2
    assert "--score-only needs a score mode other than none" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        bl.main([*argv, "--score-mode", "best"])
    assert exc.value.code == 2
    assert "invalid choice: 'best'" in capsys.readouterr().err
    for extra in (["--score-mode", "coordinate"], ["--score-only", "--score-mode", "all"]):
        with pytest.raises(SystemExit) as exc:
            bl.main([*argv, "--keep-scores", *extra])
        assert exc.value.code == 2
        assert "--keep-scores keeps an earlier scoring" in capsys.readouterr().err


@pytest.mark.smoke
def test_cli_records_into_run_json_only_for_the_whole_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_section: RunSection,
    signal_cfg: SignalConfig,
) -> None:
    import argparse

    from hq import runs

    config = types.SimpleNamespace(run=run_section, signal=signal_cfg)  # H4's RunConfig shape

    def args(**over: Any) -> argparse.Namespace:
        base = {"run_dir": tmp_path, "config_dir": tmp_path, "cache_dir": tmp_path / "cache"}
        return argparse.Namespace(**{**base, "stations": None, "start": None, "end": None, **over})

    assert isinstance(bl._cli_context(args(), config), bl._CliContext)  # no run.json
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    run = types.SimpleNamespace(
        id="run-1",
        windowStart=run_section.window_start_s,
        windowEnd=run_section.window_end_s,
        bbox=list(run_section.bbox),
    )
    monkeypatch.setattr(runs, "read_run_json", lambda run_dir: run)
    ctx = bl._cli_context(args(), config)
    assert isinstance(ctx, runs.RunContext)
    assert ctx.run_id == "run-1" and ctx.cache_dir == tmp_path / "cache"
    # A subset must not overwrite the run's record, nor can a run + signal config record.
    for over in ({"stations": "XX.A"}, {"start": "2026-09-10T18:00:00"}, {"end": "2026-09-11"}):
        assert isinstance(bl._cli_context(args(**over), config), bl._CliContext)
    assert isinstance(
        bl._cli_context(args(), bl._CliConfig(run=run_section, signal=signal_cfg)), bl._CliContext
    )
    moved = types.SimpleNamespace(**{**vars(run), "windowEnd": run.windowEnd + 1.0})
    monkeypatch.setattr(runs, "read_run_json", lambda run_dir: moved)
    with pytest.raises(ValueError, match="does not match"):
        bl._cli_context(args(), config)


@pytest.mark.smoke
def test_h2_missing_error_is_specific(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc, "_H2_API", (("hq", "associate_absent"), ("hq_absent_mod", "locate")))
    with pytest.raises(sc.H2PipelineMissingError) as err:
        sc.check_h2_merged()
    assert "H2 pipeline not merged; sweep has pick counts only" in str(err.value)
    assert "hq.associate_absent" in str(err.value) and "hq_absent_mod" in str(err.value)
    # An H2 module that exists but cannot import (a missing dependency, a bug) is not "not merged".
    (tmp_path / "hq_broken_h2_for_test.py").write_text(
        "import hq_dependency_absent_for_test\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sc, "_H2_API", (("hq_broken_h2_for_test", "associate"),))
    with pytest.raises(ModuleNotFoundError, match="hq_dependency_absent_for_test"):
        sc.check_h2_merged()


# A SeismologyApi a spawned worker can import (the tests themselves are not importable there).
_WORKER_API_MODULE = """
import os
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Result:
    events: pd.DataFrame
    arrivals: pd.DataFrame | None = None
    matches: pd.DataFrame | None = None
    tiering: dict | None = None


@dataclass(frozen=True)
class Api:
    def associate(self, picks, stations, cfg, run):
        return Result(pd.DataFrame({"assocId": range(int((picks["phase"] == "P").sum()))}))

    def locate(self, assoc, picks, stations, cfg, run):
        n = len(assoc.events)
        ids = [f"ev{i}" for i in range(n)]
        events = pd.DataFrame({"id": ids, "quality_rmsS": [0.1] * n, "quality_nStations": [3] * n})
        return Result(events, arrivals=pd.DataFrame({"eventId": ids}))

    def match(self, events, catalog, cfg):
        return Result(events, matches=pd.DataFrame({"catalogId": ["pub-1"], "eventId": [None]}))

    def assign_tiers(self, events, matches, cfg, *, thresholds, arrivals, stations):
        rules = {"A": {"nearestStation": {"focalDepthBelow": "x"}, "mapOnVolumeTop": {"applied": False}}}
        tiering = {"thresholdSource": "supplied", "rules": rules, "pid": os.getpid()}
        return Result(events.assign(tier="A"), tiering=tiering)


def make_api(cache_dir, run_id):
    return Api()
"""


# Not a smoke test: spawning interpreters that import the pipeline takes several seconds.
def test_process_runner_scores_in_spawned_workers(
    tmp_path: Path, run_section: RunSection, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from hq_contracts.io import to_frame
    from hq_contracts.models import Pick

    (tmp_path / "hq_sweep_worker_api.py").write_text(_WORKER_API_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))  # spawned children inherit sys.path
    worker_api = importlib.import_module("hq_sweep_worker_api")
    shared = sc.ScoreShared(
        stations=pd.DataFrame({"id": ["XX.A"]}),
        catalog=pd.DataFrame({"id": ["pub-1"]}),
        seismology=None,
        run=run_section,
        thresholds=RUN_TIERING,
        cache_dir=tmp_path,
        run_id="test-run",
        api_factory=worker_api.make_api,
    )

    def picks(n: int) -> pd.DataFrame:
        models = [
            Pick(id=f"p{i}", stationId="XX.A", phase="P", t=T + i, prob=1.0, picker="stalta")
            for i in range(n)
        ]
        return to_frame(models, Pick)

    jobs = [sc.ScoreJob(f"grid point {k}", "stalta", picks(k), index=k) for k in range(3)]
    results = list(sc.ProcessRunner(2)(shared, jobs))
    assert sorted(r.index for r in results) == [0, 1, 2]
    by_index = {r.index: r for r in results}
    assert [by_index[k].row.tiers.A for k in range(3)] == [0, 1, 2]
    assert [by_index[k].barsMet["A"]["everyBar"] for k in range(3)] == [0, 1, 2]  # rmsS 0.1
    pids = {r.tiering["pid"] for r in results if r.tiering is not None}
    assert pids and os.getpid() not in pids


@pytest.mark.smoke
def test_station_rows_select_used_stations_and_fail_loudly() -> None:
    df = pd.DataFrame(
        [
            {**STATION_ROWS[1], "usedInRun": True},
            {**STATION_ROWS[0], "usedInRun": True},
            {**STATION_ROWS[3], "usedInRun": False},
        ]
    )
    rows = bl.station_rows(df, None)
    assert [r.id for r in rows] == ["XX.A", "XX.B"]
    assert rows[0].channels == ("HHZ", "HHN", "HHE") and rows[0].profile == "surface-100"
    assert [r.id for r in bl.station_rows(df, ["XX.B"])] == ["XX.B"]
    with pytest.raises(ValueError, match="not usedInRun"):
        bl.station_rows(df, ["XX.OFF"])
    with pytest.raises(ValueError, match="missing columns"):
        bl.station_rows(df.drop(columns=["channels"]), None)
    with pytest.raises(ValueError, match="repeats"):
        bl.station_rows(pd.concat([df, df]), None)

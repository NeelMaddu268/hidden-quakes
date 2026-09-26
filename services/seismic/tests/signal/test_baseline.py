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
from hq.config.run import RunSection
from hq.config.signal import SignalConfig
from hq.ingest.cache import CacheMissError
from hq.preprocess.chunks import ModelChunk, pick_is_kept
from hq.preprocess.profiles import TimeMap

bl = importlib.import_module("hq.baseline.run")  # the module; ``hq.baseline.run`` is the stage
chunks_module = importlib.import_module("hq.preprocess.chunks")

T = 1_789_063_200.0  # 2026-09-10T18:00:00Z, a multiple of every chunk length used here
RATE = 100.0
LENGTH_S = 100.0  # short chunks keep the tests fast; the logic is the same as for hours
OVERLAP_S = 40.0
TOL_S = 0.05  # "a few samples" at 100 Hz
PICK_FIELDS = ["id", "stationId", "phase", "t", "prob", "picker", "eventId", "residualS", "weight"]


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
    raw["baseline"].update({"maxWorkers": 1, **baseline})
    return SignalConfig.model_validate(raw)


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
    # Frequencies well above borehole-B's 5 Hz zero-phase highpass corner, whose symmetric ringing
    # would put a precursor ahead of a low-frequency onset (a property of the profile, not of the
    # time mapping under test).
    pieces = station_pieces(
        T - 45.0,
        T + 105.0,
        Arrivals(p=[(tp, 5.0)], s=[(ts, 5.0)]),
        rate=rate,
        codes="DP",
        horizontals="12",
        seed=3,
        p_freq=25.0,
        s_freq=15.0,
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
def test_short_segments_are_skipped_and_counted(raw_signal_yaml: dict[str, Any]) -> None:
    # One chunk [0, 100) read over [-40, 140]; a gap leaves segments [-40, 5) (45 s) and
    # [50, 140] (90 s), both long enough for for_picking. P needs ltaS 2 s + margin, S 4 s + margin.
    pieces = station_pieces(T - 60.0, T + 260.0, Arrivals(), gaps=((T + 5.0, T + 50.0),))
    cache = FakeCache({"XX.SYN": pieces})
    expected = {  # margin -> (P skipped, S skipped): the 45 s segment fails once need > 45 s
        40.0: (0, 0),  # P 42 s, S 44 s
        42.0: (0, 2),  # P 44 s, S 46 s: both horizontals of the 45 s segment skipped
        44.0: (1, 2),  # P 46 s
    }
    for margin, (skip_p, skip_s) in expected.items():
        res = pick(cache, small_cfg(raw_signal_yaml, minSegmentMarginS=margin), T, T + 100.0)
        assert res.counts["segmentsSkippedShortP"] == skip_p, margin
        assert res.counts["segmentsSkippedShortS"] == skip_s, margin
        assert res.counts["segmentsP"] == 2 - skip_p and res.counts["segmentsS"] == 4 - skip_s
    res = pick(cache, small_cfg(raw_signal_yaml, minSegmentMarginS=100.0), T, T + 100.0)
    assert res.picks == [] and res.reason == "every segment shorter than ltaS + minSegmentMarginS"


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


@pytest.mark.smoke
def test_s_after_p_takes_the_earliest_horizontal_onset_in_the_window() -> None:
    p = np.array([10.0, 20.0, 21.0, 30.0])
    h = np.array([5.0, 10.1, 10.5, 12.0, 21.6, 30.0])
    # P 10 -> [10.15, 15]: 10.5 (10.1 is too early); P 20 and P 21 both choose 21.6 (one S);
    # P 30 -> [30.15, 35]: nothing
    assert bl.s_after_p(p, h, 0.15, 5.0).tolist() == [10.5, 21.6]
    assert bl.s_after_p(p[:0], h, 0.15, 5.0).size == 0
    assert bl.s_after_p(p, h[:0], 0.15, 5.0).size == 0


# --- config ---------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_config_is_validated_and_cross_checked(
    raw_signal_yaml: dict[str, Any], signal_cfg: SignalConfig
) -> None:
    bl.check_config(signal_cfg)  # the shipped signal.yaml passes
    assert signal_cfg.baseline.prob == 1.0

    def with_baseline(**over: Any) -> SignalConfig:
        return small_cfg(raw_signal_yaml, **over)

    b = raw_signal_yaml["baseline"]
    bad: list[dict[str, Any]] = [
        {"p": {**b["p"], "staS": 3.0}},  # STA not below LTA
        {"s": {**b["s"], "warmupS": 1.0}},  # warm-up shorter than the LTA
        {"s": {**b["s"], "components": "ZE"}},  # P and S share Z
        {"chosen": {**b["chosen"], "pOff": 6.0}},  # off above on
        {"sweep": {**b["sweep"], "pOn": [4.0, 3.0]}},  # not increasing
        {"sweep": {**b["sweep"], "offLevels": [1.0, 3.5]}},  # an off level above an on level
        {"minSMinusPS": 6.0},  # empty S window
        {"notAKnob": 1},
    ]
    for over in bad:
        with pytest.raises(ValidationError):
            with_baseline(**over)

    with pytest.raises(ValueError, match="not model components"):
        bl.check_config(with_baseline(p={**b["p"], "components": "X"}))
    with pytest.raises(ValueError, match="Nyquist"):
        bl.check_config(with_baseline(prefilter={"lowHz": 2.0, "highHz": 50.0, "corners": 4}))
    raw = copy.deepcopy(raw_signal_yaml)
    raw["preprocess"]["chunks"].update({"overlapS": 11.5, "minOverlapS": 10.0})
    with pytest.raises(ValueError, match="overlapS"):
        bl.check_config(SignalConfig.model_validate(raw))  # S warm-up is 12 s
    raw["baseline"]["s"]["warmupS"] = raw["baseline"]["s"]["ltaS"]  # 4 s
    raw["preprocess"]["chunks"]["overlapS"] = 10.5
    with pytest.raises(ValueError, match="maxSMinusPS"):
        bl.check_config(SignalConfig.model_validate(raw))  # P warm-up 6 s + 5 s S window


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
    api = tuple((f"hq_absent_for_test_{attr}", attr) for _, attr in bl._H2_API)
    monkeypatch.setattr(bl, "_H2_API", api)


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
    cfg = small_cfg(raw_signal_yaml)
    ctx = stage_ctx(fake_ctx, run_section, cfg, "h2-missing")
    seed_stage(table_io, ctx, monkeypatch)
    with caplog.at_level(logging.INFO):
        stage.run(ctx)  # the registry entry point

    picks = table_io.io.read_table(ctx.path(bl.PICKS_FILE))
    assert picks.attrs["model"] == "Pick"
    assert list(picks.columns) == PICK_FIELDS
    assert len(picks) == 3 * 4  # 3 used, cached stations x 2 events x (P + S)
    assert set(picks["stationId"]) == {"XX.A", "XX.B", "XX.C"}  # not XX.OFF, XX.GONE has nothing
    assert (picks["picker"] == "stalta").all() and (picks["prob"] == 1.0).all()
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
        "nP",
        "nS",
        "nStations",
    )
    grid = bl.sweep_grid(cfg.baseline)
    assert len(sweep) == len(grid)
    assert [json.loads(p) for p in sweep["params"]] == [g.params() for g in grid]
    assert sweep[["candidates", "recoveredPublic", "tierA"]].isna().all().all()
    k = grid.index(bl.chosen_point(cfg.baseline))
    assert sweep.loc[k, "nP"] == (picks["phase"] == "P").sum()
    assert sweep.loc[k, "nS"] == (picks["phase"] == "S").sum()
    assert sweep.loc[k, "nStations"] == 3

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "H2 pipeline not merged; sweep has pick counts only" in r.getMessage() for r in warnings
    )
    assert any("XX.GONE: not cached" in r.getMessage() for r in warnings)
    rec = ctx.records["baseline"]
    assert rec["params"] == cfg.baseline.model_dump(mode="json")
    counts = rec["counts"]
    assert counts["stations"] == 4 and counts["stationsWithPicks"] == 3
    assert counts["stationsNotCached"] == 1
    assert counts["nP"] == 6 and counts["nS"] == 6
    assert counts["sweepPoints"] == len(grid) and counts["sweepScored"] == 0
    assert counts["chunksYielded"] == 3 * 2
    assert counts["chunksPlanned"] == 3 * 2 + 1  # XX.GONE fails on its first read
    assert all(isinstance(v, int) for v in counts.values())
    assert rec["runtime_s"] > 0.0
    out = capsys.readouterr().out
    assert "XX.GONE" in out and "not cached" in out and "XX.OFF" not in out
    if table_io.written is not None:
        assert table_io.written[bl.PICKS_FILE] == "Pick"


@pytest.mark.smoke
def test_stage_output_is_identical_across_worker_counts(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_h2(monkeypatch)
    sweep = {"pOn": [4.0, 8.0], "sOn": [4.0, 8.0], "offLevels": [1.5]}
    frames = []
    for name, workers in (("serial", 1), ("threads", 3), ("rerun", 3)):
        cfg = small_cfg(raw_signal_yaml, maxWorkers=workers, sweep=sweep)
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


@pytest.mark.smoke
def test_sweep_is_scored_through_h2_when_it_exists(
    table_io: TableIO,
    fake_ctx: Any,
    run_section: RunSection,
    raw_signal_yaml: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[pd.DataFrame] = []

    @dataclass(frozen=True)
    class Result:
        events: pd.DataFrame | None = None
        matches: pd.DataFrame | None = None

    def associate(picks: pd.DataFrame, stations: pd.DataFrame, cfg: Any, run: Any) -> Result:
        seen.append(picks)
        n = int((picks["phase"] == "P").sum()) // 3  # one event per three P picks
        return Result(events=pd.DataFrame({"assocId": range(n)}))

    def locate(assoc: Result, picks: Any, stations: Any, cfg: Any, run: Any) -> Result:
        assert assoc.events is not None
        return Result(events=assoc.events.assign(id=[f"ev{i}" for i in range(len(assoc.events))]))

    def match(events: pd.DataFrame, catalog: pd.DataFrame, cfg: Any) -> Result:
        ids = [events["id"].iloc[0] if len(events) else None, None]
        return Result(matches=pd.DataFrame({"catalogId": catalog["id"], "eventId": ids}))

    def assign_tiers(events: pd.DataFrame, matches: pd.DataFrame, cfg: Any) -> Result:
        return Result(
            events=events.assign(tier=["A" if i == 0 else "B" for i in range(len(events))])
        )

    pipeline = bl.H2Pipeline(associate, locate, match, assign_tiers)
    monkeypatch.setattr(bl, "load_h2_pipeline", lambda: pipeline)
    sweep = {"pOn": [4.0, 40.0], "sOn": [4.0], "offLevels": [1.5]}
    cfg = small_cfg(raw_signal_yaml, sweep=sweep)
    ctx = stage_ctx(fake_ctx, run_section, cfg, "h2", seismology=object())
    seed_stage(table_io, ctx, monkeypatch)
    catalog = pd.DataFrame({"id": ["pub-1", "pub-2"]})
    table_io.io.write_table(catalog, ctx.path("catalog.parquet"), "CatalogEvent")
    stage.run(ctx)

    out = table_io.io.read_table(ctx.path(bl.SWEEP_FILE))
    assert out["candidates"].tolist() == [2, 0]  # 6 P picks -> 2 events; pOn 40 picks nothing
    assert out["recoveredPublic"].tolist() == [1, 0]
    assert out["tierA"].tolist() == [1, 0]
    assert out["nP"].tolist() == [6, 0]
    assert ctx.records["baseline"]["counts"]["sweepScored"] == 2
    assert len(seen) == 2
    for frame in seen:  # the Pick table H2's associate() gets is the published schema
        assert list(frame.columns) == PICK_FIELDS
        assert (frame["picker"] == "stalta").all() and (frame["prob"] == 1.0).all()


@pytest.mark.smoke
def test_h2_missing_error_is_specific(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bl, "_H2_API", (("hq", "associate_absent"), ("hq_absent_mod", "locate")))
    with pytest.raises(bl.H2PipelineMissingError) as err:
        bl.load_h2_pipeline()
    assert "H2 pipeline not merged; sweep has pick counts only" in str(err.value)
    assert "hq.associate_absent" in str(err.value) and "hq_absent_mod" in str(err.value)
    # An H2 module that exists but cannot import (a missing dependency, a bug) is not "not merged".
    (tmp_path / "hq_broken_h2_for_test.py").write_text(
        "import hq_dependency_absent_for_test\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(bl, "_H2_API", (("hq_broken_h2_for_test", "associate"),))
    with pytest.raises(ModuleNotFoundError, match="hq_dependency_absent_for_test"):
        bl.load_h2_pipeline()


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

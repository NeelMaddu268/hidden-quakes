"""SEIS-06: full-window PhaseNet picking (hq.pick.run).

Offline and fast: a fake model with PhaseNet's ``classify`` interface (it honours ``blinding``)
stands in for seisbench, and chunks come either from a fake iterator (exact control over keep
intervals, data edges and the TimeMap) or from the real ``iter_model_chunks`` over a fake
``read_window``. Tests that write parquet need ``hq_contracts`` (H4's CONTRACT-01) and skip
without it.
"""

import copy
import json
import logging
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import obspy
import pytest
from pydantic import ValidationError

import hq.pick
from hq.config.signal import PickerConfig, SignalConfig
from hq.ingest.cache import CacheMissError
from hq.pick.ab import StationInfo
from hq.pick.run import (
    PickIO,
    StageIO,
    StationReport,
    StationTask,
    dedupe_and_sort,
    default_runner,
    pick_window,
    plan_tasks,
    run_in_process,
    run_picking,
    zero_pick_reason,
)
from hq.pick.run import run as stage_run
from hq.preprocess import TimeMap
from hq.preprocess.chunks import ChunkStats, ModelChunk, iter_model_chunks

SR = 100.0
T0 = 1_789_063_200.0  # a multiple of 3600 s and of 100 s; synthetic
PICK_FIELDS = {"id", "stationId", "phase", "t", "prob", "picker", "eventId", "residualS", "weight"}


# --- fakes ---------------------------------------------------------------------------------------


@dataclass
class FakeSbPick:
    trace_id: str
    phase: str
    peak_time: obspy.UTCDateTime
    peak_value: float


class FakeModel:
    """PhaseNet stand-in: returns the configured (phase, MODEL time, prob) picks of a station that
    fall inside the block, and, like seisbench, nothing within ``blinding`` samples of its edges."""

    sampling_rate = SR
    in_samples = 3001
    component_order = "ZNE"

    def __init__(self, picks: dict[str, list[tuple[str, float, float]]]) -> None:
        self._picks = picks
        self.calls: list[dict[str, Any]] = []

    def classify(self, stream: obspy.Stream, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        pre, post = kwargs["blinding"]
        t0 = max(tr.stats.starttime.timestamp for tr in stream) + pre / self.sampling_rate
        t1 = min(tr.stats.endtime.timestamp for tr in stream) - post / self.sampling_rate
        sta = stream[0].stats.station
        tid = f"{stream[0].stats.network}.{sta}."
        return SimpleNamespace(
            picks=[
                FakeSbPick(tid, phase, obspy.UTCDateTime(t), prob)
                for phase, t, prob in self._picks.get(sta, [])
                if t0 <= t <= t1
            ]
        )


@dataclass
class FakeLoader:
    model: FakeModel
    loads: list[tuple[str, int]] = field(default_factory=list)

    def __call__(self, weights: str, picker: PickerConfig) -> FakeModel:
        self.loads.append((weights, picker.torchThreads))
        return self.model


@dataclass
class FakeChunks:
    """``iter_model_chunks`` stand-in: yields prepared chunks and adds to ``stats`` like it."""

    chunks: dict[str, list[ModelChunk]]
    empty: dict[str, int] = field(default_factory=dict)
    noSegments: dict[str, int] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def __call__(
        self,
        station_id: str,
        channels: Sequence[str],
        profile: str,
        t0: float,
        t1: float,
        cfg: SignalConfig,
        *,
        cache_dir: Path,
        stats: ChunkStats | None = None,
    ) -> Iterator[ModelChunk]:
        self.calls.append(station_id)
        mine = self.chunks.get(station_id, [])
        empty = self.empty.get(station_id, 0)
        short = self.noSegments.get(station_id, 0)
        if stats is not None:
            stats.add(
                ChunkStats(
                    planned=len(mine) + empty + short,
                    empty=empty,
                    noSegments=short,
                    yielded=len(mine),
                )
            )
        yield from mine


def model_stream(
    sid: str, spans: Sequence[tuple[float, float]], comps: str = "ZNE", seed: int = 0
) -> obspy.Stream:
    """``for_picking``-like output: one 100 Hz trace per component per [start, end) model span."""
    net, sta = sid.split(".")
    st = obspy.Stream()
    for i, comp in enumerate(comps):
        for j, (start, end) in enumerate(spans):
            rng = np.random.default_rng(seed + 10 * i + j)
            n = round((end - start) * SR)
            header = {
                "network": net,
                "station": sta,
                "channel": f"HH{comp}",
                "sampling_rate": SR,
                "starttime": obspy.UTCDateTime(start),
            }
            st.append(obspy.Trace(data=rng.normal(size=n) + 5.0, header=header))
    return st


def chunk(
    sid: str,
    keep: tuple[float, float],
    spans: Sequence[tuple[float, float]],
    *,
    edges: Sequence[float] = (),
    tmap: TimeMap | None = None,
    profile: str = "surface-100",
    comps: str = "ZNE",
) -> ModelChunk:
    return ModelChunk(
        stationId=sid,
        profile=profile,
        stream=model_stream(sid, spans, comps),
        timemap=tmap if tmap is not None else TimeMap(anchor=spans[0][0], factor=1.0),
        keep=keep,
        dataEdges=tuple(sorted(edges)),
    )


def no_cache_check(station_id: str, cache_dir: Path) -> None:
    return None


def fake_io(chunks: Any, model: FakeModel, check: Any = no_cache_check) -> PickIO:
    return PickIO(iter_chunks=chunks, load_model=FakeLoader(model), check_cached=check)


def cfg_from(raw: dict, **picker: Any) -> SignalConfig:
    """signal.yaml with top-level ``picker`` keys replaced (``blinding`` and ``run`` merge)."""
    raw = copy.deepcopy(raw)
    blinding = picker.pop("blinding", None)
    if blinding is not None:
        raw["picker"]["seisbench"]["blinding"] = blinding
    run_over = picker.pop("run", None)
    if run_over is not None:
        raw["picker"]["run"].update(run_over)
    raw["picker"].update(picker)
    return SignalConfig.model_validate(raw)


def stations(*sids: str, profile: str = "surface-100", used: bool = True) -> dict:
    return {
        sid: StationInfo(
            id=sid, preprocessProfile=profile, usedInRun=used, channels=("HHZ", "HHN", "HHE")
        )
        for sid in sids
    }


def tasks_for(cfg: SignalConfig, *sids: str, profile: str = "surface-100") -> list[StationTask]:
    return plan_tasks(stations(*sids, profile=profile), cfg.picker)


def reversed_runner(fn: Any, tasks: Sequence[StationTask]) -> list:
    return [fn(task) for task in reversed(tasks)]


def summary(picks: list[dict]) -> list[tuple[str, str, float]]:
    return [(p["stationId"], p["phase"], round(p["t"] - T0, 3)) for p in picks]


# --- keep intervals, gap edges, TimeMap ----------------------------------------------------------


def two_chunks_with_a_gap(sid: str) -> list[ModelChunk]:
    """Window [T0, T0 + 200): chunk A keeps [T0, T0 + 100), chunk B keeps [T0 + 100, T0 + 200),
    each read 40 s beyond its keep interval. The data has one gap, [T0 + 150, T0 + 160)."""
    gap_edges = (T0 + 149.99, T0 + 160.0)  # last sample before, first sample after
    a = chunk(sid, (T0, T0 + 100.0), [(T0 - 40.0, T0 + 140.0)])
    b = chunk(
        sid,
        (T0 + 100.0, T0 + 200.0),
        [(T0 + 60.0, T0 + 150.0), (T0 + 160.0, T0 + 240.0)],
        edges=gap_edges,
    )
    return [a, b]


GAP_PICKS = [
    ("P", T0 + 50.0, 0.9),  # A keeps it
    ("S", T0 + 95.0, 0.8),  # A keeps it; B reads it in its overlap and discards it
    ("P", T0 + 120.0, 0.7),  # A reads it in its overlap and discards it; B keeps it
    ("S", T0 + 149.3, 0.9),  # 0.69 s before the gap starts: dropped, counted
    ("P", T0 + 160.8, 0.9),  # 0.8 s after the gap ends: dropped, counted
    ("P", T0 + 180.0, 0.6),  # kept
    ("S", T0 + 210.0, 0.9),  # after the window end: outside every keep interval, not a drop
]


@pytest.mark.smoke
def test_keep_interval_discard_and_gap_edge_drop(raw_signal_yaml: dict, tmp_path: Path) -> None:
    # Blinding 0.5 s (< gapEdgeS 1 s), so the gap-edge rule is what removes the gap picks.
    cfg = cfg_from(raw_signal_yaml, blinding=[50, 50])
    model = FakeModel({"A": GAP_PICKS})
    io = fake_io(FakeChunks({"XX.A": two_chunks_with_a_gap("XX.A")}), model)
    result = pick_window(
        tasks_for(cfg, "XX.A"), cfg, T0, T0 + 200.0, cache_dir=tmp_path, io=io, runner=None
    )
    assert summary(result.picks) == [
        ("XX.A", "P", 50.0),
        ("XX.A", "S", 95.0),
        ("XX.A", "P", 120.0),
        ("XX.A", "P", 180.0),
    ]
    (rep,) = result.reports
    assert (rep.nP, rep.nS) == (3, 1)
    assert (rep.droppedNearGapP, rep.droppedNearGapS, rep.droppedNearGap) == (1, 1, 2)
    assert rep.outsideKeep == 3  # S95 in B, P120 in A, S210 in B: owned elsewhere, not drops
    assert (rep.chunks, rep.blocks, rep.blocksPicked, rep.dataEdges) == (2, 3, 3, 2)
    # data the model saw inside the window: A [T0, T0+100) + B [T0+100, T0+149.99] + [T0+160, +200)
    assert rep.secondsPicked == pytest.approx(100.0 + 49.99 + 40.0)
    # blinding (0.5 s) inside the window only at the two gap edges
    assert rep.blindedS == pytest.approx(1.0)
    assert rep.blindingEdgeS == pytest.approx(0.5) and rep.edgeExclusionS == pytest.approx(1.0)
    assert result.counts()["droppedNearGap"] == 2 and rep.zeroPickReason is None
    # Every classify call got the config's thresholds, strict mode and blinding.
    kwargs = model.calls[0]
    assert (kwargs["P_threshold"], kwargs["S_threshold"], kwargs["strict"]) == (0.1, 0.1, True)
    assert tuple(kwargs["blinding"]) == (50, 50)


@pytest.mark.smoke
def test_default_blinding_leaves_nothing_near_a_gap_and_says_so(
    signal_cfg: SignalConfig, tmp_path: Path
) -> None:
    # Blinding 250 samples = 2.5 s at 100 Hz > gapEdgeS 1 s: the gap picks are never produced.
    model = FakeModel({"A": GAP_PICKS})
    io = fake_io(FakeChunks({"XX.A": two_chunks_with_a_gap("XX.A")}), model)
    result = pick_window(
        tasks_for(signal_cfg, "XX.A"), signal_cfg, T0, T0 + 200.0, cache_dir=tmp_path, io=io
    )
    (rep,) = result.reports
    assert rep.droppedNearGap == 0
    assert rep.blindingEdgeS == pytest.approx(2.5) and rep.edgeExclusionS == pytest.approx(2.5)
    assert rep.blindedS == pytest.approx(5.0)  # 2.5 s on each side of the one gap


@pytest.mark.smoke
def test_to_real_applied_once_with_factor_10(signal_cfg: SignalConfig, tmp_path: Path) -> None:
    anchor = T0
    tmap = TimeMap(anchor=anchor, factor=10.0)
    # Model time [A, A + 600) and [A + 700, A + 1200) = real [A, A + 60) and [A + 70, A + 120).
    ch = chunk(
        "XX.B",
        (anchor + 10.0, anchor + 110.0),
        [(anchor, anchor + 600.0), (anchor + 700.0, anchor + 1200.0)],
        edges=(anchor, anchor + 59.999, anchor + 70.0, anchor + 119.999),
        tmap=tmap,
        profile="borehole-B",
    )
    model = FakeModel(
        {
            "B": [
                ("P", anchor + 500.0, 0.9),  # real A + 50: kept
                ("S", anchor + 800.0, 0.8),  # real A + 80: kept
                ("P", anchor + 50.0, 0.9),  # real A + 5: before the keep interval
                ("S", anchor + 592.0, 0.7),  # real A + 59.2, 0.8 s before a gap: dropped
                ("P", anchor + 1150.0, 0.9),  # real A + 115: after the keep interval
            ]
        }
    )
    io = fake_io(FakeChunks({"XX.B": [ch]}), model)
    tasks = tasks_for(signal_cfg, "XX.B", profile="borehole-B")
    result = pick_window(
        tasks, signal_cfg, anchor + 10.0, anchor + 110.0, cache_dir=tmp_path, io=io
    )
    # Exactly A + 50 and A + 80: converted once (twice would give A + 5 and A + 8).
    assert [(p["phase"], p["t"]) for p in result.picks] == [
        ("P", anchor + 50.0),
        ("S", anchor + 80.0),
    ]
    (rep,) = result.reports
    assert (rep.droppedNearGapS, rep.outsideKeep) == (1, 2)
    # 250 model samples at 100 Hz x (1 real s / 10 model s) = 0.25 real s per block edge
    assert rep.blindingEdgeS == pytest.approx(0.25)
    assert rep.edgeExclusionS == pytest.approx(1.0)  # gapEdgeS dominates on a stretched profile
    assert rep.secondsPicked == pytest.approx(49.999 + 40.0)  # real seconds inside the keep
    # only the two gap-side block edges lie inside the keep interval
    assert rep.blindedS == pytest.approx(2 * 0.25)


@pytest.mark.smoke
def test_real_chunk_iterator_with_fake_cache(raw_signal_yaml: dict, tmp_path: Path) -> None:
    # 100 s keep intervals read 35 s beyond; 100 Hz data with one 10 s gap on every component.
    raw = copy.deepcopy(raw_signal_yaml)
    raw["preprocess"]["chunks"].update({"lengthS": 100.0, "overlapS": 35.0, "edgeProbeS": 1.0})
    raw["picker"]["seisbench"]["blinding"] = [50, 50]
    cfg = SignalConfig.model_validate(raw)
    pieces = model_stream("XX.C", [(T0 - 60.0, T0 + 150.0), (T0 + 160.0, T0 + 260.0)])
    reads: list[tuple[float, float]] = []

    def read_window(station_id: str, t0: float, t1: float, *, cache_dir: Path) -> obspy.Stream:
        reads.append((t0, t1))
        out = obspy.Stream()
        for tr in pieces:
            piece = tr.slice(obspy.UTCDateTime(t0), obspy.UTCDateTime(t1), nearest_sample=False)
            if piece.stats.npts:
                out.append(piece.copy())
        return out

    model = FakeModel({"C": GAP_PICKS})
    io = PickIO(
        iter_chunks=partial(iter_model_chunks, read_window=read_window),
        load_model=FakeLoader(model),
        check_cached=no_cache_check,
    )
    result = pick_window(tasks_for(cfg, "XX.C"), cfg, T0, T0 + 200.0, cache_dir=tmp_path, io=io)
    assert summary(result.picks) == [
        ("XX.C", "P", 50.0),
        ("XX.C", "S", 95.0),
        ("XX.C", "P", 120.0),
        ("XX.C", "P", 180.0),
    ]
    (rep,) = result.reports
    assert (rep.chunksPlanned, rep.chunks, rep.droppedNearGap, rep.outsideKeep) == (2, 2, 2, 3)
    assert len(reads) == 2


# --- ordering, duplicates, ids -------------------------------------------------------------------


def three_station_setup() -> tuple[FakeChunks, FakeModel]:
    sids = ["XX.C", "XX.A", "XX.B"]
    chunks = {sid: [chunk(sid, (T0, T0 + 100.0), [(T0 - 40.0, T0 + 140.0)])] for sid in sids}
    model = FakeModel(
        {
            "A": [("S", T0 + 30.0, 0.5), ("P", T0 + 30.0, 0.4), ("P", T0 + 10.0, 0.9)],
            "B": [("P", T0 + 30.0, 0.7), ("S", T0 + 60.0, 0.3)],
            "C": [("P", T0 + 10.0, 0.2), ("S", T0 + 30.0, 0.6)],
        }
    )
    return FakeChunks(chunks), model


@pytest.mark.smoke
def test_output_independent_of_completion_order(signal_cfg: SignalConfig, tmp_path: Path) -> None:
    chunks, model = three_station_setup()
    tasks = tasks_for(signal_cfg, "XX.C", "XX.A", "XX.B")
    assert [t.stationId for t in tasks] == ["XX.A", "XX.B", "XX.C"]

    def shuffled_runner(fn: Any, todo: Sequence[StationTask]) -> list:
        order = list(todo)
        random.Random(7).shuffle(order)
        return [fn(task) for task in order]

    runs = [
        pick_window(
            tasks,
            signal_cfg,
            T0,
            T0 + 100.0,
            cache_dir=tmp_path,
            io=fake_io(chunks, model),
            runner=runner,
        )
        for runner in (run_in_process, reversed_runner, shuffled_runner)
    ]
    assert chunks.calls[3:6] == ["XX.C", "XX.B", "XX.A"]  # really ran in another order
    for other in runs[1:]:
        assert other.picks == runs[0].picks
        assert [r.stationId for r in other.reports] == ["XX.A", "XX.B", "XX.C"]
    # sorted by (t, stationId, phase, prob desc)
    assert summary(runs[0].picks) == [
        ("XX.A", "P", 10.0),
        ("XX.C", "P", 10.0),
        ("XX.A", "P", 30.0),
        ("XX.A", "S", 30.0),
        ("XX.B", "P", 30.0),
        ("XX.C", "S", 30.0),
        ("XX.B", "S", 60.0),
    ]


@pytest.mark.smoke
def test_duplicate_ids_are_merged_and_counted(signal_cfg: SignalConfig, tmp_path: Path) -> None:
    # 0.2 ms apart: the same id at millisecond resolution. The higher prob survives.
    model = FakeModel({"A": [("P", T0 + 20.0001, 0.4), ("P", T0 + 20.0003, 0.8)]})
    io = fake_io(FakeChunks({"XX.A": [chunk("XX.A", (T0, T0 + 100.0), [(T0, T0 + 100.0)])]}), model)
    result = pick_window(
        tasks_for(signal_cfg, "XX.A"), signal_cfg, T0, T0 + 100.0, cache_dir=tmp_path, io=io
    )
    assert result.duplicates == 1 and result.counts()["duplicates"] == 1
    (pick,) = result.picks
    assert pick["prob"] == 0.8
    (rep,) = result.reports
    assert (rep.nP, rep.duplicates) == (1, 1)


@pytest.mark.smoke
def test_dedupe_and_sort_is_order_independent() -> None:
    def p(sid: str, phase: str, t: float, prob: float) -> dict:
        return {
            "id": f"x:{sid}:{phase}:{t:.3f}",
            "stationId": sid,
            "phase": phase,
            "t": t,
            "prob": prob,
        }

    rows = [
        p("B", "P", 1.0, 0.5),
        p("A", "S", 1.0, 0.5),
        p("A", "P", 1.0, 0.3),
        p("A", "P", 1.0002, 0.9),
        p("A", "P", 0.5, 0.1),
    ]
    first, n1 = dedupe_and_sort(rows)
    second, n2 = dedupe_and_sort(list(reversed(rows)))
    assert first == second and n1 == n2 == 1
    # the duplicate of x:A:P:1.000 kept the prob-0.9 row, which sits at t 1.0002
    assert [(r["stationId"], r["phase"], r["prob"]) for r in first] == [
        ("A", "P", 0.1),
        ("A", "S", 0.5),
        ("B", "P", 0.5),
        ("A", "P", 0.9),
    ]


@pytest.mark.smoke
def test_pick_fields_ids_and_weights_fallback(
    raw_signal_yaml: dict, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw = copy.deepcopy(raw_signal_yaml)
    raw["picker"]["weightsByProfile"] = {"surface-100": "stead"}
    raw["picker"]["defaultWeights"] = "original"
    cfg = SignalConfig.model_validate(raw)
    infos = {**stations("XX.A"), **stations("XX.B", profile="surface-hi")}
    with caplog.at_level(logging.WARNING, logger="hq.pick.run"):
        tasks = plan_tasks(infos, cfg.picker)
    assert [(t.stationId, t.weights, t.weightsSource) for t in tasks] == [
        ("XX.A", "stead", "weightsByProfile"),
        ("XX.B", "original", "defaultWeights"),
    ]
    assert "no picker.weightsByProfile entry" in caplog.text
    chunks = {
        "XX.A": [chunk("XX.A", (T0, T0 + 100.0), [(T0, T0 + 100.0)])],
        "XX.B": [chunk("XX.B", (T0, T0 + 100.0), [(T0, T0 + 100.0)], profile="surface-hi")],
    }
    model = FakeModel({"A": [("P", T0 + 12.3456, 0.9)], "B": [("S", T0 + 40.0, 0.5)]})
    loader = FakeLoader(model)
    io = PickIO(iter_chunks=FakeChunks(chunks), load_model=loader, check_cached=no_cache_check)
    result = pick_window(
        tasks, cfg, T0, T0 + 100.0, cache_dir=tmp_path, io=io, runner=run_in_process
    )
    assert all(set(p) == PICK_FIELDS for p in result.picks)
    a, b = result.picks
    assert a["id"] == f"phasenet:stead:XX.A:P:{a['t']:.3f}" and a["picker"] == "phasenet:stead"
    assert a["t"] == pytest.approx(T0 + 12.3456, abs=1e-6)
    assert b["id"] == f"phasenet:original:XX.B:S:{T0 + 40.0:.3f}"
    assert (a["eventId"], a["residualS"], a["weight"]) == (None, None, None)
    # each worker loads its weights with the per-worker torch thread count
    threads = cfg.picker.run.torchThreadsPerWorker
    assert loader.loads == [("stead", threads), ("original", threads)]


@pytest.mark.smoke
def test_plan_tasks_rejects_unknown_or_unused_stations(signal_cfg: SignalConfig) -> None:
    infos = {**stations("XX.A"), **stations("XX.Z", used=False)}
    assert [t.stationId for t in plan_tasks(infos, signal_cfg.picker)] == ["XX.A"]
    with pytest.raises(ValueError, match="not in"):
        plan_tasks(infos, signal_cfg.picker, ["XX.Q"])
    with pytest.raises(ValueError, match="usedInRun"):
        plan_tasks(infos, signal_cfg.picker, ["XX.Z"])


# --- zero picks, cache misses --------------------------------------------------------------------


@pytest.mark.smoke
def test_zero_pick_reasons(raw_signal_yaml: dict, tmp_path: Path) -> None:
    cfg = cfg_from(raw_signal_yaml, run={"onCacheMiss": "report"})
    keep = (T0, T0 + 100.0)
    chunks = FakeChunks(
        {
            "XX.A": [chunk("XX.A", keep, [(T0, T0 + 100.0)])],  # fine, but the model is silent
            "XX.B": [chunk("XX.B", keep, [(T0, T0 + 20.0), (T0 + 40.0, T0 + 60.0)])],  # 20 s blocks
            "XX.C": [chunk("XX.C", keep, [(T0, T0 + 100.0)], comps="ZN")],  # no E
            "XX.P": [chunk("XX.P", keep, [(T0, T0 + 100.0)])],  # has a pick
        },
        empty={"XX.D": 2},
        noSegments={"XX.E": 1, "XX.D": 0},
    )
    model = FakeModel({"A": [("P", T0 + 50.0, 0.05)], "P": [("S", T0 + 50.0, 0.3)]})

    def check(station_id: str, cache_dir: Path) -> None:
        if station_id == "XX.F":
            raise CacheMissError(f"nothing cached for station {station_id}")

    sids = ("XX.A", "XX.B", "XX.C", "XX.D", "XX.E", "XX.F", "XX.P")
    result = pick_window(
        tasks_for(cfg, *sids),
        cfg,
        *keep,
        cache_dir=tmp_path,
        io=fake_io(chunks, model, check),
        runner=run_in_process,
    )
    reasons = {r.stationId: r.zeroPickReason for r in result.reports}
    assert reasons == {
        "XX.A": "no picks above threshold",
        "XX.B": "all segments shorter than model window",
        "XX.C": "no three-component data (a component is missing)",
        "XX.D": "no data in window",
        "XX.E": "all segments too short for preprocessing (minSegmentModelS / filter padding)",
        "XX.F": "nothing cached for this station",
        "XX.P": None,
    }
    by_id = {r.stationId: r for r in result.reports}
    assert by_id["XX.A"].droppedBelowThreshold == 1
    assert (by_id["XX.B"].blocksTooShort, by_id["XX.B"].secondsTooShort) == (
        2,
        pytest.approx(39.98),
    )
    assert by_id["XX.F"].cacheMiss and "XX.F" not in chunks.calls
    counts = result.counts()
    assert (counts["zeroPickStations"], counts["cacheMissStations"], counts["picksS"]) == (6, 1, 1)
    near_gap = StationReport(
        "XX.G",
        "surface-100",
        "instance",
        "weightsByProfile",
        [],
        1.0,
        1.0,
        chunks=1,
        blocks=1,
        blocksPicked=1,
        droppedNearGapP=2,
    )
    assert zero_pick_reason(near_gap) == "every pick was within gapEdgeS of a data edge"


@pytest.mark.smoke
def test_cache_miss_is_an_error_before_any_picking(
    signal_cfg: SignalConfig, tmp_path: Path
) -> None:
    assert signal_cfg.picker.run.onCacheMiss == "error"
    chunks = FakeChunks({"XX.A": [chunk("XX.A", (T0, T0 + 100.0), [(T0, T0 + 100.0)])]})

    def check(station_id: str, cache_dir: Path) -> None:
        if station_id == "XX.B":
            raise CacheMissError("nothing cached for station XX.B")

    with pytest.raises(CacheMissError):
        pick_window(
            tasks_for(signal_cfg, "XX.A", "XX.B"),
            signal_cfg,
            T0,
            T0 + 100.0,
            cache_dir=tmp_path,
            io=fake_io(chunks, FakeModel({}), check),
        )
    assert chunks.calls == []


# --- stage, outputs, config ----------------------------------------------------------------------


@dataclass
class MemoryStageIO:
    infos: dict[str, StationInfo]
    written: dict[Path, list[dict]] = field(default_factory=dict)

    def io(self) -> StageIO:
        return StageIO(load_stations=lambda path: self.infos, write_picks=self.write)

    def write(self, picks: list[dict], path: Path) -> None:
        self.written[path] = list(picks)


@pytest.mark.smoke
def test_stage_writes_report_and_records(fake_ctx: Any, raw_signal_yaml: dict) -> None:
    cfg = cfg_from(raw_signal_yaml, blinding=[50, 50])
    ctx = replace(fake_ctx, config=type(fake_ctx.config)(run=fake_ctx.config.run, signal=cfg))
    chunks, model = three_station_setup()
    chunks.chunks["XX.A"] = two_chunks_with_a_gap("XX.A")
    model._picks["A"] = GAP_PICKS
    sio = MemoryStageIO(stations("XX.A", "XX.B", "XX.C", "XX.D"))
    result = run_picking(
        ctx,
        window=(T0, T0 + 200.0),
        io=fake_io(chunks, model),
        runner=reversed_runner,
        stage_io=sio.io(),
    )
    assert sio.written[ctx.path("picks.parquet")] == result.picks
    rec = ctx.records["pick"]
    assert rec["counts"] == result.counts()
    assert {k: rec["counts"][k] for k in ("stations", "picksP", "picksS", "droppedNearGap")} == {
        "stations": 4,
        "picksP": 5,
        "picksS": 3,
        "droppedNearGap": 2,
    }
    assert (rec["counts"]["duplicates"], rec["counts"]["zeroPickStations"]) == (0, 1)
    params = rec["params"]
    assert params["pThreshold"] == 0.1 and params["seisbench"]["blinding"] == [50, 50]
    assert params["run"]["workers"] == cfg.picker.run.workers
    assert params["chunks"] == cfg.preprocess.chunks.model_dump(mode="json")
    report = json.loads(ctx.path("pick_report.json").read_text(encoding="utf-8"))
    assert [s["stationId"] for s in report["stations"]] == ["XX.A", "XX.B", "XX.C", "XX.D"]
    first = report["stations"][0]
    for key in (
        "profile",
        "weights",
        "chunks",
        "blocks",
        "nP",
        "nS",
        "droppedNearGap",
        "blindedS",
        "secondsPicked",
        "runtimeS",
        "zeroPickReason",
        "edgeExclusionS",
    ):
        assert key in first
    assert report["stations"][3]["zeroPickReason"] == "no data in window"
    assert report["totals"]["droppedNearGap"] == 2 and report["gapEdgeS"] == 1.0


@pytest.mark.smoke
def test_stage_sets_picker_fields_when_the_context_can(fake_ctx: Any) -> None:
    updates: dict[str, Any] = {}

    @dataclass
    class WithUpdateRun:
        inner: Any
        cache_dir: Path
        config: Any

        def path(self, name: str) -> Path:
            return self.inner.path(name)

        def record(self, stage: str, **kwargs: Any) -> None:
            self.inner.record(stage, **kwargs)

        def update_run(self, **fields: Any) -> None:
            updates.update(fields)

    ctx = WithUpdateRun(fake_ctx, fake_ctx.cache_dir, fake_ctx.config)
    sio = MemoryStageIO(stations("XX.A"))
    run_picking(
        ctx, window=(T0, T0 + 100.0), io=fake_io(FakeChunks({}), FakeModel({})), stage_io=sio.io()
    )
    picker = fake_ctx.config.signal.picker
    assert updates == {"pickerModel": picker.model, "pickerWeights": picker.defaultWeights}


@pytest.mark.smoke
def test_empty_and_full_picks_tables_round_trip(fake_ctx: Any) -> None:
    pytest.importorskip("hq_contracts.io")
    from hq_contracts.io import from_frame, read_table
    from hq_contracts.models import Pick

    from hq.pick.ab import write_picks

    empty_io = StageIO(load_stations=lambda path: stations("XX.A"), write_picks=write_picks)
    run_picking(
        fake_ctx,
        window=(T0, T0 + 100.0),
        io=fake_io(FakeChunks({}), FakeModel({})),
        stage_io=empty_io,
    )
    df = read_table(fake_ctx.path("picks.parquet"))
    assert len(df) == 0 and set(df.columns) == PICK_FIELDS and df.attrs["model"] == "Pick"

    chunks, model = three_station_setup()
    io3 = StageIO(
        load_stations=lambda path: stations("XX.A", "XX.B", "XX.C"), write_picks=write_picks
    )
    result = run_picking(
        fake_ctx,
        window=(T0, T0 + 100.0),
        io=fake_io(chunks, model),
        runner=run_in_process,
        stage_io=io3,
    )
    assert len(result.picks) == 7
    back = [p.model_dump() for p in from_frame(read_table(fake_ctx.path("picks.parquet")), Pick)]
    assert back == result.picks


@pytest.mark.smoke
def test_stage_is_exposed_by_the_package() -> None:
    assert hq.pick.run is stage_run and callable(hq.pick.run)


@pytest.mark.smoke
def test_run_config_block_and_runner_choice(
    raw_signal_yaml: dict, signal_cfg: SignalConfig
) -> None:
    rc = signal_cfg.picker.run
    assert rc.workers >= 1 and rc.torchThreadsPerWorker >= 1
    assert rc.onCacheMiss in ("error", "report")
    assert default_runner(rc, 1) == (run_in_process, 1)
    _, workers = default_runner(rc, 100)
    assert workers == rc.workers
    for bad in (
        {"workers": 0},
        {"torchThreadsPerWorker": 0},
        {"onCacheMiss": "skip"},
        {"notAKnob": 1},
    ):
        with pytest.raises(ValidationError):
            cfg_from(raw_signal_yaml, run=bad)

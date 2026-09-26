"""SEIS-04: gap-safe PhaseNet picking, weight A/B and Check B.

Offline and fast: a fake model with PhaseNet's ``classify`` interface (including ``blinding``)
stands in for seisbench, and ``for_picking`` / ``read_window`` / ``display_copy`` are injected
(SEIS-03 and SEIS-05 live on other branches). One test runs a randomly initialised seisbench
PhaseNet (no weights, no network) to check the no-zero-fill claim against the real library.
"""

import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import obspy
import pytest
from pydantic import ValidationError

from hq.config.signal import ArrivalWindowConfig, CheckBConfig, SignalConfig
from hq.pick import ab, phasenet
from hq.pick.ab import (
    AbIO,
    ComboSummary,
    EventMetrics,
    KnownEvent,
    StationInfo,
    VariantComparison,
    adopt_variants,
    arrival_window,
    best_phases,
    choose_weights,
    evaluate_check_b,
    event_metrics,
    format_check_b,
    parse_known_windows,
    select_channels,
    spearman_rho,
    summarize,
)
from hq.pick.phasenet import make_pick, pick_prepared, pick_stream, prepare_station, split_blocks

SR = 100.0
T0 = 1_757_462_400.0  # 2025-09-10 00:00:00 UTC; any epoch works
PICK_FIELDS = {"id", "stationId", "phase", "t", "prob", "picker", "eventId", "residualS", "weight"}


# --- helpers -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeTimeMap:
    """Same maths as hq.preprocess.TimeMap (lane lead's interface)."""

    anchor: float
    factor: float

    def to_real(self, t: float) -> float:
        return self.anchor + (t - self.anchor) / self.factor

    def to_model(self, t: float) -> float:
        return self.anchor + (t - self.anchor) * self.factor


def make_trace(
    comp: str,
    start: float,
    npts: int,
    *,
    station: str = "S01",
    seed: int = 0,
    sr: float = SR,
    band: str = "HH",
) -> obspy.Trace:
    rng = np.random.default_rng(seed)
    tr = obspy.Trace(data=rng.normal(size=npts) + 5.0)  # offset: no sample is ever exactly 0
    tr.stats.network = "XX"
    tr.stats.station = station
    tr.stats.channel = f"{band}{comp}"
    tr.stats.sampling_rate = sr
    tr.stats.starttime = obspy.UTCDateTime(start)
    return tr


def three_comp(spans: dict[str, list[tuple[float, float]]], station: str = "S01") -> obspy.Stream:
    """Stream with, per component, one trace per (start, seconds) span."""
    st = obspy.Stream()
    for i, (comp, comp_spans) in enumerate(spans.items()):
        for j, (start, seconds) in enumerate(comp_spans):
            st.append(
                make_trace(comp, start, round(seconds * SR), station=station, seed=10 * i + j)
            )
    return st


def cfg_with(raw: dict, path: tuple[str, ...], value: Any) -> SignalConfig:
    """signal.yaml with ``picker.<path>`` replaced, validated."""
    raw = copy.deepcopy(raw)
    node = raw["picker"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return SignalConfig.model_validate(raw)


@dataclass
class FakeSbPick:
    trace_id: str
    phase: str
    peak_time: obspy.UTCDateTime
    peak_value: float


class FakeModel:
    """PhaseNet stand-in: returns the configured (phase, model time, prob) picks inside a block.

    Like seisbench, it produces nothing in the first / last ``blinding`` samples of the block.
    """

    sampling_rate = 100.0
    in_samples = 3001
    component_order = "ZNE"

    def __init__(self, picks: list[tuple[str, float, float]]) -> None:
        self._picks = picks
        self.calls: list[tuple[obspy.Stream, dict[str, Any]]] = []

    def classify(self, stream: obspy.Stream, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((stream.copy(), kwargs))
        pre, post = kwargs["blinding"]
        t0 = max(tr.stats.starttime.timestamp for tr in stream) + pre / self.sampling_rate
        t1 = min(tr.stats.endtime.timestamp for tr in stream) - post / self.sampling_rate
        tid = f"{stream[0].stats.network}.{stream[0].stats.station}."
        return SimpleNamespace(
            picks=[
                FakeSbPick(tid, phase, obspy.UTCDateTime(t), prob)
                for phase, t, prob in self._picks
                if t0 <= t <= t1
            ]
        )


def identity_for_picking(st: obspy.Stream, profile: str, cfg: SignalConfig):
    return st.copy(), FakeTimeMap(anchor=T0, factor=1.0)


# --- config --------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_picker_config_block(signal_cfg: SignalConfig) -> None:
    p = signal_cfg.picker
    assert p.model == "seisbench.PhaseNet"
    assert {"instance", "stead", "original", "scedc"} <= set(p.candidateWeights)
    assert p.pThreshold == 0.1 and p.sThreshold == 0.1
    assert p.gapEdgeS == 1.0
    assert not hasattr(p, "sampleRateHz")  # the model's own rate is the only source of truth
    assert "borehole-B" in p.ab.profileVariants["borehole-A"]
    assert (p.ab.checkB.minStationsPS, p.ab.checkB.minRho, p.ab.checkB.minEventsPass) == (8, 0.8, 3)
    assert set(p.weightsByProfile) >= {"surface-100", "surface-hi", "borehole-A", "borehole-B"}
    assert p.seisbench.overlap >= sum(p.seisbench.blinding)
    aw = p.ab.arrivalWindow
    assert aw.minVelocityMps > 0 and aw.preOriginS >= 0 and aw.postMarginS >= 0


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("defaultWeights",), "not-a-weight"),
        (("weightsByProfile", "surface-100"), "not-a-weight"),
        (("candidateWeights",), ["instance", "instance"]),
        (("seisbench", "stacking"), "median"),
        (("seisbench", "overlap"), 400),  # < blinding 250 + 250: NaN holes inside a block
        (("ab", "arrivalWindow", "minVelocityMps"), 0.0),
        (("sampleRateHz",), 100.0),  # removed knob stays removed
        (("notAKnob",), 1),
        (("ab", "checkB", "notAKnob"), 1),
    ],
)
def test_picker_config_rejects_bad_values(raw_signal_yaml: dict, path: tuple, value: Any) -> None:
    with pytest.raises(ValidationError):
        cfg_with(raw_signal_yaml, path, value)


# --- model loading -------------------------------------------------------------------------------


class _LoadedModel:
    sampling_rate = 100.0
    in_samples = 3001
    component_order = "ENZ"  # the "original" weights' order; any Z/N/E order is fine

    def __init__(self) -> None:
        self.training = True

    def eval(self) -> "_LoadedModel":
        self.training = False
        return self


@pytest.mark.smoke
def test_load_model_pins_version_seeds_and_caches(
    signal_cfg: SignalConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, list[Any]] = {"threads": [], "seed": [], "from_pretrained": []}
    fake_torch = ModuleType("torch")
    fake_torch.set_num_threads = calls["threads"].append  # type: ignore[attr-defined]
    fake_torch.manual_seed = calls["seed"].append  # type: ignore[attr-defined]

    def from_pretrained(name: str, version_str: str) -> _LoadedModel:
        calls["from_pretrained"].append((name, version_str))
        return _LoadedModel()

    fake_sbm = ModuleType("seisbench.models")
    fake_sbm.PhaseNet = SimpleNamespace(from_pretrained=from_pretrained)  # type: ignore[attr-defined]
    fake_sb = ModuleType("seisbench")
    fake_sb.models = fake_sbm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "seisbench", fake_sb)
    monkeypatch.setitem(sys.modules, "seisbench.models", fake_sbm)
    monkeypatch.setattr(phasenet, "_MODEL_CACHE", {})

    p = signal_cfg.picker
    m1 = phasenet.load_model("original", p)
    m2 = phasenet.load_model("original", p)
    assert m1 is m2 and not m1.training
    assert calls["from_pretrained"] == [("original", p.weightsVersion)]
    assert calls["threads"] == [p.torchThreads] * 2 and calls["seed"] == [p.seed] * 2
    with pytest.raises(ValueError, match="candidateWeights"):
        phasenet.load_model("ethz", p)


@pytest.mark.smoke
def test_check_model_rejects_overlap_at_window(raw_signal_yaml: dict) -> None:
    cfg = cfg_with(raw_signal_yaml, ("seisbench", "overlap"), 3001)
    with pytest.raises(ValueError, match="model window"):
        phasenet.check_model(FakeModel([]), cfg.picker)


# --- picks ---------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_make_pick_fields_and_id() -> None:
    p = make_pick("UU.FOR1", "S", 1757462412.34567, 0.42, "stead")
    assert set(p) == PICK_FIELDS
    assert p["id"] == "phasenet:stead:UU.FOR1:S:1757462412.346"
    assert p["picker"] == "phasenet:stead"
    assert (p["eventId"], p["residualS"], p["weight"]) == (None, None, None)
    with pytest.raises(ValueError):
        make_pick("UU.FOR1", "N", 0.0, 0.5, "stead")


@pytest.mark.smoke
def test_to_real_applied_with_factor_10(signal_cfg: SignalConfig) -> None:
    tmap = FakeTimeMap(anchor=T0, factor=10.0)

    def stretched_for_picking(st: obspy.Stream, profile: str, cfg: SignalConfig):
        assert profile == "borehole-B"
        # 12 real seconds become 120 model seconds at 100 Hz
        return three_comp({c: [(T0, 120.0)] for c in "ZNE"}), tmap

    model = FakeModel(
        [
            ("P", T0 + 50.0, 0.9),  # real T0 + 5.0
            ("S", T0 + 80.0, 0.7),  # real T0 + 8.0
            ("P", T0 + 5.0, 0.8),  # 5 model s (past blinding) but 0.5 real s from start -> dropped
        ]
    )
    picks, diag = pick_stream(
        obspy.Stream(),
        "XX.S01",
        "borehole-B",
        signal_cfg,
        model,
        "instance",
        for_picking=stretched_for_picking,
    )
    assert [(p["phase"], p["t"]) for p in picks] == [("P", T0 + 5.0), ("S", T0 + 8.0)]
    assert picks[0]["id"] == f"phasenet:instance:XX.S01:P:{T0 + 5.0:.3f}"
    assert all(set(p) == PICK_FIELDS for p in picks)
    assert diag.droppedNearEdgeP == 1 and diag.droppedNearEdge == 1
    assert diag.secondsPicked == pytest.approx(11.999, abs=1e-6)  # real seconds, not model seconds
    # blinding 250 + 250 model samples = 5 model s = 0.5 real s; gapEdgeS (1 s) dominates
    assert diag.secondsBlinded == pytest.approx(0.5, abs=1e-6)
    assert diag.edgeExclusionS == pytest.approx(1.0)
    kwargs = model.calls[0][1]
    assert kwargs["P_threshold"] == 0.1 and kwargs["S_threshold"] == 0.1
    assert kwargs["strict"] is True and tuple(kwargs["blinding"]) == (250, 250)


@pytest.mark.smoke
def test_gap_edge_picks_dropped_and_counted(raw_signal_yaml: dict) -> None:
    # Blinding 0.5 s (< gapEdgeS 1 s) so the gap-edge rule is what removes these picks.
    cfg = cfg_with(raw_signal_yaml, ("seisbench", "blinding"), [50, 50])
    # block 1: [T0, T0 + 59.99]; 10 s gap on every component; block 2: [T0 + 70, T0 + 129.99]
    st = three_comp({c: [(T0, 60.0), (T0 + 70.0, 60.0)] for c in "ZNE"})
    model = FakeModel(
        [
            ("P", T0 + 0.3, 0.9),  # inside blinding: never produced, so not counted
            ("P", T0 + 0.7, 0.9),  # data start -> dropped
            ("P", T0 + 30.0, 0.9),  # kept
            ("S", T0 + 59.2, 0.9),  # gap start -> dropped
            ("P", T0 + 70.8, 0.9),  # gap end -> dropped
            ("S", T0 + 100.0, 0.9),  # kept
            ("S", T0 + 129.2, 0.9),  # data end -> dropped
        ]
    )
    picks, diag = pick_stream(
        st, "XX.S01", "surface-100", cfg, model, "instance", for_picking=identity_for_picking
    )
    assert [(p["phase"], p["t"]) for p in picks] == [("P", T0 + 30.0), ("S", T0 + 100.0)]
    assert (diag.nBlocks, diag.nBlocksPicked, len(model.calls)) == (2, 2, 2)
    assert (diag.droppedNearEdgeP, diag.droppedNearEdgeS, diag.droppedNearEdge) == (2, 2, 4)
    assert (diag.picksP, diag.picksS) == (1, 1)
    assert diag.edgeExclusionS == pytest.approx(1.0)
    assert diag.secondsBlinded == pytest.approx(2.0)  # 2 blocks x (0.5 + 0.5) s


@pytest.mark.smoke
def test_default_blinding_exceeds_gap_edge_and_is_reported(signal_cfg: SignalConfig) -> None:
    # With blinding 250 samples at 100 Hz no pick exists within 2.5 s of a block edge, so the
    # 1 s gap-edge rule has nothing left to drop; the diagnostics say so instead of hiding it.
    st = three_comp({c: [(T0, 60.0)] for c in "ZNE"})
    model = FakeModel([("P", T0 + 0.7, 0.9), ("P", T0 + 2.0, 0.9), ("P", T0 + 30.0, 0.9)])
    picks, diag = pick_stream(
        st, "XX.S01", "surface-100", signal_cfg, model, "instance", for_picking=identity_for_picking
    )
    assert [p["t"] for p in picks] == [T0 + 30.0]
    assert diag.droppedNearEdge == 0
    assert diag.edgeExclusionS == pytest.approx(2.5)
    assert diag.secondsBlinded == pytest.approx(5.0)


@pytest.mark.smoke
def test_blocks_never_span_a_gap_and_are_never_filled(signal_cfg: SignalConfig) -> None:
    # Z gap [100, 110), N gap [105, 120): the only 3-component spans are [0, 100) and [120, 200)
    spans = {
        "Z": [(T0, 100.0), (T0 + 110.0, 90.0)],
        "N": [(T0, 105.0), (T0 + 120.0, 80.0)],
        "E": [(T0, 200.0)],
    }
    st = three_comp(spans)
    split = split_blocks(st, SR)
    blocks = split.blocks
    assert len(blocks) == 2 and split.nOverlaps == 0
    gaps = [(T0 + 100.0, T0 + 110.0), (T0 + 105.0, T0 + 120.0)]
    for block in blocks:
        assert sorted(tr.stats.channel[-1] for tr in block) == ["E", "N", "Z"]
        assert len({tr.stats.npts for tr in block}) == 1
        assert len({tr.stats.starttime.timestamp for tr in block}) == 1
        b0, b1 = block[0].stats.starttime.timestamp, block[0].stats.endtime.timestamp
        assert all(b1 < g0 or b0 >= g1 for g0, g1 in gaps)
        for tr in block:  # samples are the original samples, nothing inserted
            src = next(
                s
                for s in st.select(channel=tr.stats.channel)
                if s.stats.starttime <= tr.stats.starttime <= s.stats.endtime
            )
            i0 = round((tr.stats.starttime - src.stats.starttime) * SR)
            np.testing.assert_array_equal(tr.data, src.data[i0 : i0 + tr.stats.npts])
            assert not np.any(tr.data == 0.0)
    assert blocks[0][0].stats.starttime.timestamp == pytest.approx(T0)
    assert blocks[1][0].stats.starttime.timestamp == pytest.approx(T0 + 120.0)

    model = FakeModel([])
    pick_stream(
        st, "XX.S01", "surface-100", signal_cfg, model, "instance", for_picking=identity_for_picking
    )
    assert len(model.calls) == 2
    for call_st, _ in model.calls:
        assert len(call_st) == 3 and len({tr.stats.npts for tr in call_st}) == 1


@pytest.mark.smoke
def test_subsample_misaligned_components_share_one_grid() -> None:
    st = three_comp({c: [(T0, 60.0)] for c in "ZNE"})
    st.select(channel="HHN")[0].stats.starttime += 0.004  # 0.4 sample late
    st.select(channel="HHE")[0].stats.starttime += 0.006  # 0.6 sample late
    (block,) = split_blocks(st, SR).blocks
    grid = {round(tr.stats.starttime.timestamp * SR) for tr in block}
    assert len(grid) == 1  # seisbench snaps to this grid; one index -> nothing to zero-pad
    assert len({tr.stats.npts for tr in block}) == 1


@pytest.mark.smoke
def test_conflicting_overlap_is_treated_as_a_gap_and_counted(signal_cfg: SignalConfig) -> None:
    # Z: [0, 60) and [50, 110) with different samples; N, E continuous over [0, 110).
    st = three_comp({"Z": [(T0, 60.0), (T0 + 50.0, 60.0)], "N": [(T0, 110.0)], "E": [(T0, 110.0)]})
    split = split_blocks(st, SR)
    assert split.nOverlaps == 1
    ((lo, hi),) = split.overlapRegions
    assert (lo, hi) == (pytest.approx(T0 + 50.0), pytest.approx(T0 + 59.99))
    spans = [(b[0].stats.starttime.timestamp, b[0].stats.endtime.timestamp) for b in split.blocks]
    assert spans == [
        (pytest.approx(T0), pytest.approx(T0 + 49.99)),
        (pytest.approx(T0 + 60.0), pytest.approx(T0 + 109.99)),
    ]
    assert spans[0][1] < spans[1][0]  # disjoint: no sample is classified twice

    prepared = prepare_station(
        st, "XX.S01", "surface-100", signal_cfg, for_picking=identity_for_picking
    )
    _, diag = pick_prepared(prepared, signal_cfg, FakeModel([]), "instance")
    assert diag.nOverlaps == 1
    assert diag.secondsOverlapRemoved == pytest.approx(9.99)


@pytest.mark.smoke
def test_two_channels_for_one_component_is_an_error(signal_cfg: SignalConfig) -> None:
    st = three_comp({c: [(T0, 60.0)] for c in "ZNE"})
    st.append(make_trace("Z", T0, 6000, band="EH", seed=99))
    with pytest.raises(ValueError, match="several channels per component"):
        prepare_station(st, "XX.S01", "surface-100", signal_cfg, for_picking=identity_for_picking)


@pytest.mark.smoke
def test_short_block_skipped_and_counted(signal_cfg: SignalConfig) -> None:
    # 20 s block (2000 samples < 3001) then a 60 s block
    st = three_comp({c: [(T0, 20.0), (T0 + 30.0, 60.0)] for c in "ZNE"})
    model = FakeModel([("P", T0 + 10.0, 0.9), ("P", T0 + 60.0, 0.9)])
    picks, diag = pick_stream(
        st, "XX.S01", "surface-100", signal_cfg, model, "instance", for_picking=identity_for_picking
    )
    assert (diag.nBlocks, diag.nBlocksTooShort, diag.nBlocksPicked) == (2, 1, 1)
    assert diag.secondsTooShort == pytest.approx(19.99)
    assert len(model.calls) == 1 and model.calls[0][0][0].stats.npts == 6000
    assert [p["t"] for p in picks] == [T0 + 60.0]


@pytest.mark.smoke
def test_wrong_rate_is_an_error_not_a_resample(signal_cfg: SignalConfig) -> None:
    def fp_200(st: obspy.Stream, profile: str, cfg: SignalConfig):
        out = obspy.Stream([make_trace(c, T0, 12000, sr=200.0) for c in "ZNE"])
        return out, FakeTimeMap(T0, 1.0)

    model = FakeModel([])
    with pytest.raises(ValueError, match="never resample"):
        pick_stream(
            obspy.Stream(),
            "XX.S01",
            "surface-hi",
            signal_cfg,
            model,
            "instance",
            for_picking=fp_200,
        )
    assert model.calls == []


@pytest.mark.smoke
def test_missing_component_is_not_picked(signal_cfg: SignalConfig) -> None:
    st = three_comp({c: [(T0, 60.0)] for c in "ZN"})
    model = FakeModel([("P", T0 + 30.0, 0.9)])
    picks, diag = pick_stream(
        st, "XX.S01", "surface-100", signal_cfg, model, "instance", for_picking=identity_for_picking
    )
    assert picks == [] and model.calls == []
    assert diag.missingComponents == ["E"] and diag.nBlocks == 0


@pytest.mark.smoke
def test_real_seisbench_never_sees_a_zero_or_a_resample(
    signal_cfg: SignalConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Random-init seisbench PhaseNet (no weights, offline): one array per block, no zero cell."""
    import seisbench.models as sbm
    import torch

    torch.manual_seed(0)
    model = sbm.PhaseNet(sampling_rate=100.0)
    model.eval()
    unknown = [
        k
        for k in phasenet.classify_kwargs(signal_cfg.picker)
        if not any(
            k == pattern or (pattern.startswith("*") and k.endswith(pattern[1:]))
            for pattern in model._annotate_args
        )
    ]
    assert unknown == []  # seisbench only warns on unknown kwargs and ignores them

    shapes: list[tuple[int, ...]] = []
    zeros: list[int] = []
    resampled: list[str] = []
    real_to_array = sbm.PhaseNet.stream_to_array

    def spy_to_array(self: Any, traces: obspy.Stream, argdict: dict) -> Any:
        out = real_to_array(self, traces, argdict)
        shapes.append(tuple(out.data.shape))
        zeros.append(int(np.sum(out.data == 0.0)))
        return out

    real_resample = sbm.PhaseNet.resample  # a staticmethod

    def spy_resample(stream: obspy.Stream, sampling_rate: float, zerophase: bool = True) -> Any:
        resampled.extend(tr.id for tr in stream if tr.stats.sampling_rate != sampling_rate)
        return real_resample(stream, sampling_rate, zerophase=zerophase)

    monkeypatch.setattr(sbm.PhaseNet, "stream_to_array", spy_to_array)
    monkeypatch.setattr(sbm.PhaseNet, "resample", staticmethod(spy_resample))
    # two 40 s blocks around a 5 s gap on every component; N and E up to half a sample late
    st = three_comp({c: [(T0, 40.0), (T0 + 45.0, 40.0)] for c in "ZNE"})
    for tr in st.select(channel="HHN"):
        tr.stats.starttime += 0.004
    for tr in st.select(channel="HHE"):
        tr.stats.starttime += 0.005
    before = st.copy()
    prepared = prepare_station(
        st, "XX.S01", "surface-100", signal_cfg, for_picking=identity_for_picking
    )
    blocks_before = [b.stream.copy() for b in prepared.blocks]
    pick_prepared(prepared, signal_cfg, model, "random")
    assert len(shapes) == len(prepared.blocks) == 2
    assert zeros == [0, 0]
    # stream_to_array sizes the array with int() of a float span (base.py:2416, 2433), which can
    # truncate the last sample; it never pads one.
    assert all(s in {(3, b.npts), (3, b.npts - 1)} for s, b in zip(shapes, prepared.blocks))
    assert resampled == []
    assert st == before  # input untouched
    for b, kept in zip(prepared.blocks, blocks_before, strict=True):  # classify works on a copy
        assert b.stream == kept


# --- A/B metrics ---------------------------------------------------------------------------------


def _p(phase: str, t: float, prob: float, sid: str = "XX.S01") -> dict[str, Any]:
    return make_pick(sid, phase, t, prob, "instance")


@pytest.mark.smoke
def test_best_p_and_s_after_p_and_violation() -> None:
    picks = [_p("P", 10.0, 0.6), _p("P", 12.0, 0.9), _p("S", 11.0, 0.95), _p("S", 15.0, 0.7)]
    b = best_phases(picks)
    assert b.bestP["t"] == 12.0  # type: ignore[index]
    assert b.bestSRaw["t"] == 11.0  # type: ignore[index]
    assert b.bestS["t"] == 15.0  # type: ignore[index]
    assert b.violation and b.hasPS

    ok = best_phases([_p("P", 10.0, 0.5), _p("S", 14.0, 0.8)])
    assert not ok.violation and ok.hasPS
    s_only = best_phases([_p("S", 14.0, 0.8)])
    assert s_only.bestP is None and not s_only.hasPS and not s_only.violation

    best = {
        "A": b,
        "B": ok,
        "C": s_only,
        "D": best_phases([_p("P", 9.0, 0.3), _p("S", 8.0, 0.4)]),  # only S before P
    }
    m = event_metrics(best, {"A": 3000.0, "B": 1000.0, "C": 5000.0, "D": 500.0}, min_n_rho=3)
    assert (m.nStations, m.nP, m.nS, m.nPS, m.violations, m.nRho) == (4, 3, 4, 2, 2, 3)
    assert m.rho == pytest.approx(1.0)  # P times 9, 10, 12 at 0.5, 1, 3 km
    assert m.nRhoPS == 2 and math.isnan(m.rhoPS)  # only 2 P&S stations < min_n_rho


@pytest.mark.smoke
def test_arrival_window_limits_best_picks_to_the_event() -> None:
    event = KnownEvent("ev", T0, 38.5, -112.9, 4.0, None, None, T0 - 120.0, T0 + 480.0, ())
    cfg = ArrivalWindowConfig(preOriginS=2.0, minVelocityMps=2000.0, postMarginS=5.0)
    lo, hi = arrival_window(event, 3000.0, cfg)
    assert lo == pytest.approx(T0 - 2.0)
    assert hi == pytest.approx(T0 + 5000.0 / 2000.0 + 5.0)  # hypocentral 5 km (3 km epi, 4 km deep)

    picks = [
        _p("S", T0 - 60.0, 0.83),  # noise before the origin: would be a violation
        _p("P", T0 + 1.0, 0.85),
        _p("S", T0 + 2.0, 0.80),
        _p("P", T0 + 200.0, 0.93),  # another event: would steal best P
    ]
    unwindowed = best_phases(picks)
    assert unwindowed.bestP["t"] == T0 + 200.0 and unwindowed.violation  # type: ignore[index]
    b = best_phases(picks, (lo, hi))
    assert b.bestP["t"] == T0 + 1.0 and b.bestS["t"] == T0 + 2.0  # type: ignore[index]
    assert not b.violation and b.nOutsideWindow == 2


@pytest.mark.smoke
def test_spearman_monotonic_shuffled_and_undefined() -> None:
    dists = np.linspace(1_000.0, 30_000.0, 12)
    times = list(T0 + dists / 6_000.0)
    assert spearman_rho(times, list(dists), 3) == pytest.approx(1.0)
    shuffled = list(np.random.default_rng(0).permutation(times))
    assert spearman_rho(shuffled, list(dists), 3) < 0.5
    assert math.isnan(spearman_rho(times[:2], list(dists[:2]), 3))
    assert math.isnan(spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], 3))


def _m(nps: int, violations: int, rho: float, n: int = 10) -> EventMetrics:
    return EventMetrics(
        nStations=n,
        nWithPicks=n,
        nP=n,
        nS=n,
        nPS=nps,
        violations=violations,
        rho=rho,
        nRho=n,
        rhoPS=rho,
        nRhoPS=nps,
        nOutsideWindow=0,
    )


def _s(weights: str, profile: str, ps: int, rho: float, violations: int = 0) -> ComboSummary:
    """Three identical events with ``ps`` P&S stations in total."""
    per_event = [_m(ps // 3 + (1 if i < ps % 3 else 0), 0, rho) for i in range(3)]
    if violations:
        per_event[0] = _m(per_event[0].nPS, violations, rho)
    return summarize(weights, profile, per_event, min_rho=0.8)


@pytest.mark.smoke
def test_weight_selection_prefers_check_b_consistency_then_ps_then_rho() -> None:
    order = ["instance", "stead", "original", "scedc"]
    summaries = [
        _s("instance", "surface-100", 30, 0.55, violations=1),  # most P&S, fails rho / violation
        _s("stead", "surface-100", 27, 0.95),  # consistent on every event -> wins
        _s("instance", "borehole-A", 10, 0.85),
        _s("scedc", "borehole-A", 10, 0.9),  # consistent tie, P&S tie -> higher mean rho
        _s("original", "borehole-A", 12, math.nan),  # NaN rho: never consistent
        _s("stead", "surface-hi", 5, 0.6),
        _s("instance", "surface-hi", 5, 0.6),  # full tie -> candidate order
        _s("stead", "borehole-B", 9, 0.9),
        _s("instance", "borehole-B", 9, 0.9, violations=2),  # violation breaks consistency
    ]
    assert choose_weights(summaries, order) == {
        "surface-100": "stead",
        "borehole-A": "scedc",
        "surface-hi": "instance",
        "borehole-B": "stead",
    }
    # the reviewer's case: 10 P&S / 1 violation / rho 0.55 vs 9 P&S / clean / rho 0.95
    a = summarize("instance", "p", [_m(10, 1, 0.55)] * 3, 0.8)
    b = summarize("stead", "p", [_m(9, 0, 0.95)] * 3, 0.8)
    assert choose_weights([a, b], order) == {"p": "stead"}
    cfg = CheckBConfig(minStationsPS=8, minRho=0.8, minEventsPass=3)
    assert not evaluate_check_b({f"e{i}": _m(10, 1, 0.55) for i in range(3)}, cfg).passed
    assert evaluate_check_b({f"e{i}": _m(9, 0, 0.95) for i in range(3)}, cfg).passed
    # NaN-rho events count against a weight set rather than disappearing from the mean
    one = summarize("instance", "q", [_m(9, 0, 0.99), _m(9, 0, math.nan), _m(9, 0, math.nan)], 0.8)
    three = summarize("stead", "q", [_m(9, 0, 0.9)] * 3, 0.8)
    assert choose_weights([one, three], order) == {"q": "stead"}


def _cmp(base_rho: float, var_rho: float, n: int = 3) -> VariantComparison:
    return VariantComparison(
        base="borehole-A",
        variant="borehole-B",
        nStationWindows=n,
        baseSummary=summarize("instance", "borehole-A", [_m(2, 0, base_rho, 3)] * 3, 0.8),
        variantSummary=summarize("scedc", "borehole-B", [_m(2, 0, var_rho, 3)] * 3, 0.8),
    )


@pytest.mark.smoke
def test_variant_adopted_only_when_strictly_better_like_for_like() -> None:
    bases = ["borehole-A", "surface-100"]
    assert adopt_variants(bases, [_cmp(0.9, 0.95)]) == {
        "borehole-A": "borehole-B",
        "surface-100": "surface-100",
    }
    assert adopt_variants(bases, [_cmp(0.9, 0.9)])["borehole-A"] == "borehole-A"  # tie keeps A
    assert adopt_variants(bases, [_cmp(0.9, 0.5)])["borehole-A"] == "borehole-A"
    assert adopt_variants(bases, [_cmp(0.9, 0.95, n=0)])["borehole-A"] == "borehole-A"
    assert adopt_variants(bases, [])["borehole-A"] == "borehole-A"


@pytest.mark.smoke
def test_check_b_pass_fail_logic() -> None:
    cfg = CheckBConfig(minStationsPS=8, minRho=0.8, minEventsPass=3)
    res = evaluate_check_b(
        {
            "e1": _m(8, 0, 0.8),
            "e2": _m(7, 0, 0.95),
            "e3": _m(9, 1, 0.9),
            "e4": _m(9, 0, math.nan),
            "e5": _m(9, 0, 0.79),
        },
        cfg,
    )
    assert [e.passed for e in res.events] == [True, False, False, False, False]
    assert res.nPass == 1 and not res.passed
    assert "P&S stations 7 < 8" in res.events[1].failures
    table = format_check_b(res, cfg)
    assert "PASS" in table and "FAIL" in table and "overall: FAIL (1/5" in table
    assert "rhoPS" in table and "over stations with a best P" in table

    ok = evaluate_check_b({f"e{i}": _m(8, 0, 0.9) for i in range(3)}, cfg)
    assert ok.passed and ok.nPass == 3


def _windows_doc(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {"events": events, "params": {"preS": 120.0}}


def _event_doc(event_id: str, t: float, stations: list[tuple[str, float, bool]]) -> dict[str, Any]:
    return {
        "eventId": event_id,
        "t": t,
        "latitude": 38.5,
        "longitude": -112.9,
        "depthKm": 5.0,
        "mag": None,
        "magType": None,
        "windowStart": t - 30.0,
        "windowEnd": t + 90.0,
        "stations": [
            {
                "stationId": sid,
                "components": 3,
                "gapFraction": 0.0,
                "epiDistM": d,
                "usable": u,
                "reason": None if u else "two components",
            }
            for sid, d, u in stations
        ],
    }


@pytest.mark.smoke
def test_windows_json_parsing(tmp_path: Path) -> None:
    doc = _windows_doc(
        [_event_doc("ev1", T0 + 100.0, [("XX.S01", 5000.0, True), ("XX.S02", 9000.0, False)])]
    )
    path = tmp_path / "windows.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    kw = ab.load_known_windows(path)
    (ev,) = kw.events
    assert (ev.eventId, ev.t, ev.mag, ev.windowEnd - ev.windowStart) == (
        "ev1",
        T0 + 100.0,
        None,
        120.0,
    )
    assert [(s.stationId, s.usable, s.reason) for s in ev.stations] == [
        ("XX.S01", True, None),
        ("XX.S02", False, "two components"),
    ]
    assert kw.params == {"preS": 120.0}

    broken = copy.deepcopy(doc)
    del broken["events"][0]["stations"][0]["epiDistM"]
    with pytest.raises(ValueError, match="epiDistM"):
        parse_known_windows(broken)
    for where in ("doc", "event", "station"):
        extra = copy.deepcopy(doc)
        node = {
            "doc": extra,
            "event": extra["events"][0],
            "station": extra["events"][0]["stations"][0],
        }[where]
        node["epiDistKm"] = 5.0
        with pytest.raises(ValueError, match="unknown keys"):
            parse_known_windows(extra)


@pytest.mark.smoke
def test_select_channels_keeps_only_the_station_triplet() -> None:
    raw = three_comp({c: [(T0, 10.0)] for c in "ZNE"})
    raw += obspy.Stream([make_trace(c, T0, 1000, band="HN") for c in "ZNE"])
    kept, dropped = select_channels(raw, ["HHZ", "HHN", "HHE"])
    assert dropped == 3 and sorted(tr.stats.channel for tr in kept) == ["HHE", "HHN", "HHZ"]


# --- end to end with fakes -----------------------------------------------------------------------


class DistanceModel:
    """Fake PhaseNet whose picks follow a 6 km/s P and 3.5 km/s S moveout from each event.

    It also fires on noise before each origin and on a later, unrelated arrival; those picks sit
    outside the arrival window and must not change best P / best S.
    """

    sampling_rate = 100.0
    in_samples = 3001
    component_order = "ZNE"

    def __init__(self, mode: str, events: list[float], dist: dict[str, float]) -> None:
        self.mode, self.events, self.dist = mode, events, dist

    def classify(self, stream: obspy.Stream, **kwargs: Any) -> SimpleNamespace:
        t0 = stream[0].stats.starttime.timestamp
        t1 = stream[0].stats.endtime.timestamp
        sid = f"{stream[0].stats.network}.{stream[0].stats.station}"
        picks = []
        for te in self.events:
            if not t0 <= te <= t1:
                continue
            tp, ts = te + self.dist[sid] / 6000.0, te + self.dist[sid] / 3500.0
            if self.mode in ("full", "p_only", "s_first"):
                picks.append(FakeSbPick(sid, "P", obspy.UTCDateTime(tp), 0.8))
            if self.mode == "full":
                picks.append(FakeSbPick(sid, "S", obspy.UTCDateTime(ts), 0.7))
                picks.append(FakeSbPick(sid, "S", obspy.UTCDateTime(te - 20.0), 0.95))
                picks.append(FakeSbPick(sid, "P", obspy.UTCDateTime(te + 60.0), 0.99))
            if self.mode == "s_first":
                picks.append(FakeSbPick(sid, "S", obspy.UTCDateTime(tp - 0.5), 0.9))
        return SimpleNamespace(picks=picks)


@dataclass
class AbFixture:
    io: AbIO
    fp_calls: list[tuple[str, str]]
    written: dict[str, Any]
    known: Path
    event_times: list[float]


def _ab_fixture(fake_ctx: Any, modes: dict[str, str]) -> AbFixture:
    surface = [f"XX.S{i:02d}" for i in range(1, 8)]
    borehole = ["XX.B01", "XX.B02"]  # B02 records at 500 Hz: borehole-B rejects it
    dist = {sid: 4000.0 + 3500.0 * i for i, sid in enumerate(surface + borehole)}
    event_times = [T0 + 1000.0, T0 + 5000.0, T0 + 9000.0]
    doc = _windows_doc(
        [
            _event_doc(
                f"uu{i}", t, [(sid, dist[sid], True) for sid in dist] + [("XX.S99", 1.0, False)]
            )
            for i, t in enumerate(event_times)
        ]
    )
    known = fake_ctx.run_dir / "known"
    known.mkdir()
    (known / "windows.json").write_text(json.dumps(doc), encoding="utf-8")

    hh = ("HHZ", "HHN", "HHE")
    stations = {sid: StationInfo(sid, "surface-100", True, hh) for sid in surface}
    stations |= {sid: StationInfo(sid, "borehole-A", True, hh) for sid in borehole}
    fp_calls: list[tuple[str, str]] = []
    written: dict[str, Any] = {}

    def read_window(station_id: str, t0: float, t1: float, *, cache_dir: Path) -> obspy.Stream:
        assert cache_dir == fake_ctx.cache_dir
        n = round((t1 - t0) * SR) + 1
        seed = sorted(dist).index(station_id)
        sta = station_id.split(".")[1]
        st = obspy.Stream([make_trace(c, t0, n, station=sta, seed=seed) for c in "ZNE"])
        # a strong-motion triplet also sits in the cache; it must never reach for_picking
        st += obspy.Stream([make_trace(c, t0, n, station=sta, band="HN") for c in "ZNE"])
        return st

    def for_picking(st: obspy.Stream, profile: str, cfg: SignalConfig):
        sta = st[0].stats.station
        fp_calls.append((sta, profile))
        assert {tr.stats.channel for tr in st} == set(hh)
        if profile == "borehole-B" and sta == "B02":
            raise ValueError("XX.B02..HHZ: 500.0 Hz is outside profile borehole-B's range")
        return st.copy(), FakeTimeMap(st[0].stats.starttime.timestamp, 1.0)

    def write_picks(picks: list[dict[str, Any]], path: Path) -> None:
        written["picks"] = picks
        path.write_text(json.dumps(picks), encoding="utf-8")

    io = AbIO(
        read_window=read_window,
        for_picking=for_picking,
        display_copy=lambda st, band: st.copy(),
        load_model=lambda w, picker: DistanceModel(modes[w], event_times, dist),
        load_stations=lambda path: stations,
        write_picks=write_picks,
    )
    return AbFixture(io, fp_calls, written, known, event_times)


@pytest.mark.smoke
def test_run_ab_end_to_end_with_fakes(fake_ctx: Any) -> None:
    modes = {"instance": "full", "stead": "p_only", "original": "s_first", "scedc": "none"}
    fx = _ab_fixture(fake_ctx, modes)
    res = ab.run_ab(fake_ctx.run_dir, fake_ctx.cache_dir, fake_ctx.config.signal, io=fx.io)

    assert res.chosenWeights == {
        "surface-100": "instance",
        "borehole-A": "instance",
        "borehole-B": "instance",
    }
    assert res.adoptedProfiles == {"surface-100": "surface-100", "borehole-A": "borehole-A"}
    assert {p for s, p in fx.fp_calls if s.startswith("B")} == {"borehole-A", "borehole-B"}
    assert {p for s, p in fx.fp_calls if s.startswith("S")} == {"surface-100"}
    assert "S99" not in {s for s, _ in fx.fp_calls}  # unusable station never read

    # B02 at 500 Hz: borehole-B not applicable, counted, and compared only on B01
    assert sorted(res.notApplicable) == [(f"uu{i}", "XX.B02", "borehole-B") for i in range(3)]
    (cmp,) = res.comparisons
    assert (cmp.base, cmp.variant, cmp.nStationWindows) == ("borehole-A", "borehole-B", 3)
    assert cmp.baseSummary.key == cmp.variantSummary.key and not cmp.variantWins
    assert res.counts["variantNotApplicable"] == 3
    assert res.counts["tracesOutsideStationChannels"] > 0

    # noise and unrelated arrivals sit outside the arrival window: no violation, rho intact
    assert res.checkB.passed and res.checkB.nPass == 3
    assert all(
        e.nPS == 9 and e.violations == 0 and e.rho == pytest.approx(1.0) for e in res.checkB.events
    )
    assert res.counts["picksOutsideArrivalWindow"] == 3 * 9 * 2
    # 3 events x 4 weights x 3 profiles, 4 x 3 summary rows, 2 comparison rows
    assert (len(res.eventRows), len(res.summaryRows), len(res.comparisonRows)) == (36, 12, 2)
    s_first = [r for r in res.eventRows if r["weights"] == "original"]
    assert all(r["violations"] == r["nStations"] and r["nPS"] == 0 for r in s_first)
    b_rows = [r for r in res.eventRows if r["profile"] == "borehole-B"]
    assert all(r["nStations"] == 1 and r["nNotApplicable"] == 1 for r in b_rows)

    picks = fx.written["picks"]  # every pick is written, inside the arrival window or not
    assert len(picks) == 3 * 9 * 4 and all(p["picker"] == "phasenet:instance" for p in picks)
    assert all(set(p) == PICK_FIELDS and p["eventId"] is None for p in picks)

    import pandas as pd

    table = pd.read_csv(res.paths["abCsv"])
    assert set(table["rowType"]) == {"event", "summary", "variantCompare"}
    chosen_rows = table[(table["rowType"] == "summary") & table["chosen"].eq(True)]
    assert set(zip(chosen_rows["profile"], chosen_rows["weights"], strict=True)) == {
        ("surface-100", "instance"),
        ("borehole-A", "instance"),
        ("borehole-B", "instance"),
    }
    summary = json.loads(res.paths["abJson"].read_text(encoding="utf-8"))
    assert summary["chosenWeightsByProfile"]["surface-100"] == "instance"
    assert summary["checkB"]["passed"] is True
    assert len(summary["variantNotApplicable"]) == 3
    for i in range(3):
        png = fx.known / f"record_section_uu{i}.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.smoke
def test_run_ab_with_no_picks_still_writes_the_verdict(fake_ctx: Any) -> None:
    fx = _ab_fixture(fake_ctx, dict.fromkeys(("instance", "stead", "original", "scedc"), "none"))
    res = ab.run_ab(fake_ctx.run_dir, fake_ctx.cache_dir, fake_ctx.config.signal, io=fx.io)
    assert fx.written["picks"] == [] and res.picks == []
    assert not res.checkB.passed and res.checkB.nPass == 0
    verdict = json.loads(res.paths["abJson"].read_text(encoding="utf-8"))
    assert verdict["checkB"]["passed"] is False
    assert all(e["rho"] is None for e in verdict["checkB"]["events"])


@pytest.mark.smoke
def test_stage_run_returns_none_and_records(fake_ctx: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    modes = {"instance": "full", "stead": "p_only", "original": "s_first", "scedc": "none"}
    fx = _ab_fixture(fake_ctx, modes)
    monkeypatch.setattr(ab, "default_io", lambda: fx.io)
    monkeypatch.setattr(ab, "plot_record_section", lambda *args: 0)  # drawn by the test above
    assert ab.run(fake_ctx) is None
    record = fake_ctx.records["pick_known"]
    assert record["counts"]["checkBEventsPassed"] == 3 and record["runtime_s"] > 0


# --- contracts I/O (runs once CONTRACT-01 lands) --------------------------------------------------


@pytest.mark.smoke
def test_known_picks_round_trip_through_contracts(tmp_path: Path) -> None:
    pytest.importorskip("hq_contracts.io")
    from hq_contracts.io import from_frame, read_table
    from hq_contracts.models import Pick

    picks = [_p("P", T0 + 1.0, 0.8), _p("S", T0 + 2.5, 0.4)]
    path = tmp_path / "picks.parquet"
    ab.write_picks(picks, path)
    back = from_frame(read_table(path), Pick)
    assert [p.model_dump() for p in back] == picks

    empty = tmp_path / "empty.parquet"
    ab.write_picks([], empty)  # total failure must still produce a readable table
    frame = read_table(empty)
    assert len(frame) == 0 and set(frame.columns) == PICK_FIELDS

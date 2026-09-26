"""SEIS-04: gap-safe PhaseNet picking, weight A/B and Check B.

Offline and fast: a fake model with PhaseNet's ``classify`` interface stands in for seisbench,
and ``for_picking`` / ``read_window`` / ``display_copy`` are injected (SEIS-03 and SEIS-05 live on
other branches).
"""

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import obspy
import pytest
from pydantic import ValidationError

from hq.config.signal import CheckBConfig, SignalConfig
from hq.pick import ab
from hq.pick.ab import (
    AbIO,
    ComboSummary,
    EventMetrics,
    StationInfo,
    adopt_variants,
    best_phases,
    choose_weights,
    evaluate_check_b,
    event_metrics,
    format_check_b,
    parse_known_windows,
    spearman_rho,
)
from hq.pick.phasenet import make_pick, pick_stream, split_blocks

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
    comp: str, start: float, npts: int, *, station: str = "S01", seed: int = 0, sr: float = SR
) -> obspy.Trace:
    rng = np.random.default_rng(seed)
    tr = obspy.Trace(data=rng.normal(size=npts) + 5.0)  # offset: no sample is ever exactly 0
    tr.stats.network = "XX"
    tr.stats.station = station
    tr.stats.channel = f"HH{comp}"
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


@dataclass
class FakeSbPick:
    trace_id: str
    phase: str
    peak_time: obspy.UTCDateTime
    peak_value: float


class FakeModel:
    """PhaseNet stand-in: returns the configured (phase, model time, prob) picks inside a block."""

    sampling_rate = 100.0
    in_samples = 3001

    def __init__(self, picks: list[tuple[str, float, float]]) -> None:
        self._picks = picks
        self.calls: list[tuple[obspy.Stream, dict[str, Any]]] = []

    def classify(self, stream: obspy.Stream, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((stream.copy(), kwargs))
        t0 = max(tr.stats.starttime.timestamp for tr in stream)
        t1 = min(tr.stats.endtime.timestamp for tr in stream)
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
    assert p.sampleRateHz == 100.0
    assert "borehole-B" in p.ab.profileVariants["borehole-A"]
    assert (p.ab.checkB.minStationsPS, p.ab.checkB.minRho, p.ab.checkB.minEventsPass) == (8, 0.8, 3)
    assert set(p.weightsByProfile) >= {"surface-100", "surface-hi", "borehole-A", "borehole-B"}


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("defaultWeights",), "not-a-weight"),
        (("weightsByProfile", "surface-100"), "not-a-weight"),
        (("candidateWeights",), ["instance", "instance"]),
        (("seisbench", "stacking"), "median"),
        (("notAKnob",), 1),
        (("ab", "checkB", "notAKnob"), 1),
    ],
)
def test_picker_config_rejects_bad_values(raw_signal_yaml: dict, path: tuple, value: Any) -> None:
    raw = copy.deepcopy(raw_signal_yaml)
    node = raw["picker"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValidationError):
        SignalConfig.model_validate(raw)


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
            ("P", T0 + 5.0, 0.8),  # 5 model s but only 0.5 real s after data start -> dropped
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
    kwargs = model.calls[0][1]
    assert kwargs["P_threshold"] == 0.1 and kwargs["S_threshold"] == 0.1
    assert kwargs["strict"] is True and tuple(kwargs["blinding"]) == (250, 250)


@pytest.mark.smoke
def test_gap_edge_picks_dropped_and_counted(signal_cfg: SignalConfig) -> None:
    # block 1: [T0, T0 + 59.99]; 10 s gap on every component; block 2: [T0 + 70, T0 + 129.99]
    st = three_comp({c: [(T0, 60.0), (T0 + 70.0, 60.0)] for c in "ZNE"})
    model = FakeModel(
        [
            ("P", T0 + 0.5, 0.9),  # data start -> dropped
            ("P", T0 + 30.0, 0.9),  # kept
            ("S", T0 + 59.5, 0.9),  # gap start -> dropped
            ("P", T0 + 70.4, 0.9),  # gap end -> dropped
            ("S", T0 + 100.0, 0.9),  # kept
            ("S", T0 + 129.5, 0.9),  # data end -> dropped
        ]
    )
    picks, diag = pick_stream(
        st,
        "XX.S01",
        "surface-100",
        signal_cfg,
        model,
        "instance",
        for_picking=identity_for_picking,
    )
    assert [(p["phase"], p["t"]) for p in picks] == [("P", T0 + 30.0), ("S", T0 + 100.0)]
    assert (diag.nBlocks, diag.nBlocksPicked, len(model.calls)) == (2, 2, 2)
    assert (diag.droppedNearEdgeP, diag.droppedNearEdgeS, diag.droppedNearEdge) == (2, 2, 4)
    assert (diag.picksP, diag.picksS) == (1, 1)


@pytest.mark.smoke
def test_blocks_never_span_a_gap_and_are_never_filled(signal_cfg: SignalConfig) -> None:
    # Z gap [100, 110), N gap [105, 120): the only 3-component spans are [0, 100) and [120, 200)
    spans = {
        "Z": [(T0, 100.0), (T0 + 110.0, 90.0)],
        "N": [(T0, 105.0), (T0 + 120.0, 80.0)],
        "E": [(T0, 200.0)],
    }
    st = three_comp(spans)
    blocks = split_blocks(st, SR)
    assert len(blocks) == 2
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
    (block,) = split_blocks(st, SR)
    grid = {round(tr.stats.starttime.timestamp * SR) for tr in block}
    assert len(grid) == 1  # seisbench snaps to this grid; one index -> nothing to zero-pad
    assert len({tr.stats.npts for tr in block}) == 1


@pytest.mark.smoke
def test_short_block_skipped_and_counted(signal_cfg: SignalConfig) -> None:
    # 20 s block (2000 samples < 3001) then a 60 s block
    st = three_comp({c: [(T0, 20.0), (T0 + 30.0, 60.0)] for c in "ZNE"})
    model = FakeModel([("P", T0 + 10.0, 0.9), ("P", T0 + 60.0, 0.9)])
    picks, diag = pick_stream(
        st,
        "XX.S01",
        "surface-100",
        signal_cfg,
        model,
        "instance",
        for_picking=identity_for_picking,
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

    with pytest.raises(ValueError, match="never resample"):
        pick_stream(
            obspy.Stream(),
            "XX.S01",
            "surface-hi",
            signal_cfg,
            FakeModel([]),
            "instance",
            for_picking=fp_200,
        )


@pytest.mark.smoke
def test_missing_component_is_not_picked(signal_cfg: SignalConfig) -> None:
    st = three_comp({c: [(T0, 60.0)] for c in "ZN"})
    model = FakeModel([("P", T0 + 30.0, 0.9)])
    picks, diag = pick_stream(
        st, "XX.S01", "surface-100", signal_cfg, model, "instance", for_picking=identity_for_picking
    )
    assert picks == [] and model.calls == []
    assert diag.missingComponents == ["E"] and diag.nBlocks == 0


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


@pytest.mark.smoke
def test_spearman_monotonic_shuffled_and_undefined() -> None:
    dists = np.linspace(1_000.0, 30_000.0, 12)
    times = list(T0 + dists / 6_000.0)
    assert spearman_rho(times, list(dists), 3) == pytest.approx(1.0)
    shuffled = list(np.random.default_rng(0).permutation(times))
    assert spearman_rho(shuffled, list(dists), 3) < 0.5
    assert math.isnan(spearman_rho(times[:2], list(dists[:2]), 3))
    assert math.isnan(spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], 3))


def _s(weights: str, profile: str, ps: int, rho: float) -> ComboSummary:
    return ComboSummary(weights, profile, 3, ps, rho, rho, 3, 0)


@pytest.mark.smoke
def test_weight_selection_tiebreak_and_variant_adoption() -> None:
    order = ["instance", "stead", "original", "scedc"]
    summaries = [
        _s("instance", "surface-100", 20, 0.7),
        _s("stead", "surface-100", 22, 0.5),  # more P&S wins despite lower rho
        _s("instance", "borehole-A", 10, 0.8),
        _s("scedc", "borehole-A", 10, 0.9),  # P&S tie -> higher mean rho
        _s("original", "borehole-A", 10, math.nan),  # NaN rho never wins a tie
        _s("instance", "borehole-B", 10, 0.95),
        _s("stead", "surface-hi", 5, 0.6),
        _s("instance", "surface-hi", 5, 0.6),  # full tie -> candidate order
    ]
    chosen = choose_weights(summaries, order)
    assert chosen == {
        "surface-100": "stead",
        "borehole-A": "scedc",
        "borehole-B": "instance",
        "surface-hi": "instance",
    }
    variants = {"borehole-A": ["borehole-B"]}
    assert adopt_variants(summaries, chosen, variants)["borehole-A"] == "borehole-B"

    equal_b = [s for s in summaries if s.profile != "borehole-B"] + [
        _s("instance", "borehole-B", 10, 0.9)
    ]
    adopted = adopt_variants(equal_b, choose_weights(equal_b, order), variants)
    assert adopted == {
        "borehole-A": "borehole-A",
        "surface-100": "surface-100",
        "surface-hi": "surface-hi",
    }  # a tie keeps the base profile


def _m(nps: int, violations: int, rho: float) -> EventMetrics:
    return EventMetrics(10, 10, 10, 10, nps, violations, rho, 10)


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

    ok = evaluate_check_b({f"e{i}": _m(8, 0, 0.9) for i in range(3)}, cfg)
    assert ok.passed and ok.nPass == 3


def _windows_doc(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {"events": events, "params": {"windowS": 600}}


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
    assert kw.params == {"windowS": 600}

    broken = copy.deepcopy(doc)
    del broken["events"][0]["stations"][0]["epiDistM"]
    with pytest.raises(ValueError, match="epiDistM"):
        parse_known_windows(broken)


# --- end to end with fakes -----------------------------------------------------------------------


class DistanceModel:
    """Fake PhaseNet whose picks follow a 6 km/s P and 3.5 km/s S moveout from each event."""

    sampling_rate = 100.0
    in_samples = 3001

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
            if self.mode == "s_first":
                picks.append(FakeSbPick(sid, "S", obspy.UTCDateTime(tp - 0.5), 0.9))
        return SimpleNamespace(picks=picks)


@pytest.mark.smoke
def test_run_ab_end_to_end_with_fakes(fake_ctx: Any) -> None:
    surface = [f"XX.S{i:02d}" for i in range(1, 8)]
    borehole = ["XX.B01", "XX.B02"]
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

    stations = {sid: StationInfo(sid, "surface-100", True) for sid in surface}
    stations |= {sid: StationInfo(sid, "borehole-A", True) for sid in borehole}
    modes = {"instance": "full", "stead": "p_only", "original": "s_first", "scedc": "none"}
    fp_calls: list[tuple[str, str]] = []
    written: dict[str, Any] = {}

    def read_window(station_id: str, t0: float, t1: float, *, cache_dir: Path) -> obspy.Stream:
        assert cache_dir == fake_ctx.cache_dir
        n = round((t1 - t0) * SR) + 1
        seed = sorted(dist).index(station_id)
        return obspy.Stream(
            [make_trace(c, t0, n, station=station_id.split(".")[1], seed=seed) for c in "ZNE"]
        )

    def for_picking(st: obspy.Stream, profile: str, cfg: SignalConfig):
        fp_calls.append((st[0].stats.station, profile))
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
    res = ab.run(fake_ctx, io=io)

    assert res.chosenWeights == {
        "surface-100": "instance",
        "borehole-A": "instance",
        "borehole-B": "instance",
    }
    assert res.adoptedProfiles == {"surface-100": "surface-100", "borehole-A": "borehole-A"}
    assert {p for s, p in fp_calls if s.startswith("B")} == {"borehole-A", "borehole-B"}
    assert {p for s, p in fp_calls if s.startswith("S")} == {"surface-100"}
    assert "S99" not in {s for s, _ in fp_calls}  # unusable station never read

    assert res.checkB.passed and res.checkB.nPass == 3
    assert all(
        e.nPS == 9 and e.violations == 0 and e.rho == pytest.approx(1.0) for e in res.checkB.events
    )
    # 3 events x 4 weights x 3 profiles, plus 4 x 3 summary rows
    assert (len(res.eventRows), len(res.summaryRows)) == (36, 12)
    s_first = [r for r in res.eventRows if r["weights"] == "original"]
    assert all(r["violations"] == r["nStations"] and r["nPS"] == 0 for r in s_first)

    picks = written["picks"]
    assert len(picks) == 3 * 9 * 2 and all(p["picker"] == "phasenet:instance" for p in picks)
    assert all(set(p) == PICK_FIELDS and p["eventId"] is None for p in picks)

    import pandas as pd

    table = pd.read_csv(res.paths["abCsv"])
    assert set(table["rowType"]) == {"event", "summary"}
    chosen_rows = table[(table["rowType"] == "summary") & (table["chosen"])]
    assert set(zip(chosen_rows["profile"], chosen_rows["weights"])) == {
        ("surface-100", "instance"),
        ("borehole-A", "instance"),
        ("borehole-B", "instance"),
    }
    summary = json.loads(res.paths["abJson"].read_text(encoding="utf-8"))
    assert summary["chosenWeightsByProfile"]["surface-100"] == "instance"
    assert summary["checkB"]["passed"] is True
    for i in range(3):
        png = known / f"record_section_uu{i}.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert fake_ctx.records["pick_known"]["counts"]["checkBEventsPassed"] == 3


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

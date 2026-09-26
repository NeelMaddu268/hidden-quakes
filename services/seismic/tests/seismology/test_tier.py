"""LOC-06 quality tiers: bars from matched-set quantiles, rules, final tables, stage, sweep.

Every table is small synthetic data built inside the tests (seeded); nothing touches the locator.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hq_contracts.io import (
    columns_for,
    dtypes_for,
    from_frame,
    read_models,
    read_table,
    to_frame,
    write_table,
)
from hq_contracts.models import Pick, SeismicEvent, SweepPoint

from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig
from hq.match import MATCH_DTYPES
from hq.runs import resolve_stage, stage_spec
from hq.tier import (
    LOCATED_COLUMNS,
    METRICS,
    Thresholds,
    TierError,
    assign_tiers,
    derive_thresholds,
)
from hq.tier.picks import event_picks

SEED = 20260926
T0 = 1789000000.0  # epoch s; any fixed time
N_MATCHED = 40
N_UNMATCHED = 20
REF = 1627.7  # reference surface elevation (m) of the pinned run section below



# --- synthetic tables -------------------------------------------------------------------------


def quality(**over: Any) -> dict[str, Any]:
    """A passing-looking LocationQuality; ``over`` replaces fields."""
    q = {
        "method": "grid1d", "statics": False, "nStations": 12, "nP": 12, "nS": 6, "rmsS": 0.05,
        "gapDeg": 90.0, "minEpiDistM": 1000.0, "hErrM": 150.0, "vErrM": 250.0,
        "depthOnEdge": False,
    }
    return {**q, **over}


def event(k: int, *, depth_m: float = 3000.0, run_id: str = "test-run", **q: Any) -> SeismicEvent:
    elev = REF - depth_m
    return SeismicEvent.model_validate(
        {
            "id": f"hq-{run_id}-{k:06d}", "runId": run_id, "t": T0 + 60.0 * k,
            "latitude": 38.5, "longitude": -112.9, "elevM": elev,
            "depthKm": (REF - elev) / 1000.0,
            "enu": {"e": 10.0 * k, "n": -5.0 * k, "u": elev - REF},
            "quality": quality(**q), "tier": "C", "tierReasons": [], "meanPickProb": 0.6,
            "revealOrder": -1, "pickIds": [f"p:{k}:P", f"p:{k}:S"],
        }
    )


def located(models: list[SeismicEvent]) -> pd.DataFrame:
    """events_located rows: SeismicEvent columns without tier/tierReasons/catalogMatch/magnitude."""
    return to_frame(models, SeismicEvent)[list(LOCATED_COLUMNS)].reset_index(drop=True)


def matches_for(events: pd.DataFrame, matched_ids: list[str], n_unmatched_public: int = 2
                ) -> pd.DataFrame:
    rows = [
        {"catalogId": f"uu{i:04d}", "eventId": eid, "dtS": 0.1 * (i % 5) - 0.2,
         "distM": 100.0 + i, "reason": None}
        for i, eid in enumerate(matched_ids)
    ]
    rows += [
        {"catalogId": f"uu9{i:03d}", "eventId": None, "dtS": math.nan, "distM": math.nan,
         "reason": "no candidate within 2 s / 5 km (no located events)"}
        for i in range(n_unmatched_public)
    ]
    return pd.DataFrame(rows, columns=list(MATCH_DTYPES)).astype(MATCH_DTYPES)


def seeded_world(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """``N_MATCHED`` matched + ``N_UNMATCHED`` unmatched events with seeded, varied metrics."""
    rng = np.random.default_rng(seed)
    models = []
    for k in range(N_MATCHED + N_UNMATCHED):
        n_sta = int(rng.integers(5, 25))
        models.append(
            event(
                k, depth_m=float(rng.uniform(1500.0, 5000.0)), nStations=n_sta,
                nP=int(rng.integers(4, n_sta + 1)), nS=int(rng.integers(0, n_sta + 1)),
                rmsS=float(rng.uniform(0.01, 0.2)), gapDeg=float(rng.uniform(40.0, 300.0)),
                minEpiDistM=float(rng.uniform(200.0, 8000.0)),
                hErrM=float(rng.uniform(50.0, 900.0)), vErrM=float(rng.uniform(80.0, 1500.0)),
                depthOnEdge=bool(rng.random() < 0.1),
            )
        )
    events = located(models)
    matched_ids = list(events["id"].iloc[:N_MATCHED])
    return events, matches_for(events, matched_ids), matched_ids


def cfg_with(cfg: SeismologyConfig, **tiering: Any) -> SeismologyConfig:
    raw = cfg.model_dump(mode="json")
    raw["tiering"] = {**raw["tiering"], **tiering}
    return SeismologyConfig.model_validate(raw)


# The tiering knobs and reference surface these tests assume, pinned so tuning seismology.yaml or
# run.yaml does not turn them red.
PINNED_TIERING: dict[str, Any] = {
    "quantiles": {"A": 0.25, "B": 0.0},
    "strictNearestStationFactor": 2.0,
    "minMatched": 10,
    "sweep": {"enabled": False},
}


@pytest.fixture
def cfg(seismology_config: SeismologyConfig) -> SeismologyConfig:
    return cfg_with(seismology_config, **PINNED_TIERING)


@pytest.fixture
def run(run_section: RunSection) -> RunSection:
    return run_section.model_copy(update={"refSurfaceElevM": REF})


def worst_side_rank_value(values: np.ndarray, better: str, q: float) -> float:
    """Independent statement of the bar: the ceil((1 - q) n)-th best value."""
    order = np.sort(values) if better == "lower" else np.sort(values)[::-1]
    return float(order[math.ceil((1 - q) * len(values)) - 1])


def numpy_bar(values: np.ndarray, better: str, q: float) -> float:
    """The bar by numpy's definition: ``inverted_cdf`` at 1 - q for lower-is-better metrics, and
    its mirror for higher-is-better ones (the largest value at least 1 - q of M meet or beat)."""
    if better == "lower":
        return float(np.quantile(values, 1 - q, method="inverted_cdf"))
    return float(-np.quantile(-values, 1 - q, method="inverted_cdf"))


# --- bars -----------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_bars_are_quantiles_of_the_matched_set(cfg: SeismologyConfig) -> None:
    events, matches, matched_ids = seeded_world()
    result = assign_tiers(events, matches, cfg)
    th = result.tiering["thresholds"]
    matched = events[events["id"].isin(matched_ids)]
    q = cfg.tiering.quantiles
    assert th["nMatched"] == N_MATCHED == result.tiering["matchedSet"]["n"]
    # Every top-level key on every path: run.json merges the tiering record shallowly, so a key
    # only the zero-event path wrote would outlive a later non-empty tier run.
    assert result.tiering["note"] is None
    assert th["quantiles"] == {"A": q.A, "B": q.B}
    for metric in METRICS:
        values = matched[metric.column].to_numpy(dtype=np.float64)
        a, b = th["A"][metric.name], th["B"][metric.name]
        assert a["value"] == worst_side_rank_value(values, metric.better, q.A)
        assert a["value"] == numpy_bar(values, metric.better, q.A)  # an external definition
        worst = values.min() if metric.better == "higher" else values.max()
        assert b["value"] == worst and b["label"] == "worst of matched"
        assert a["n"] == b["n"] == N_MATCHED
        assert a["label"] == ("p25 of matched" if metric.better == "higher" else "p75 of matched")
        assert a["quantile"] == (0.25 if metric.better == "higher" else 0.75)
        assert a["op"] == metric.op
        assert a["nMeeting"] >= math.ceil(0.75 * N_MATCHED)  # three-quarters meet each A bar
        assert b["nMeeting"] == N_MATCHED
    # The joint share: matched events meeting every A (B) bar at once, counted independently.
    every = result.tiering["matchedSet"]["meetingEveryBar"]
    for tier in ("A", "B"):
        expected = sum(
            all((row[m.column] >= th[tier][m.name]["value"]) if m.better == "higher"
                else (row[m.column] <= th[tier][m.name]["value"]) for m in METRICS)
            for _, row in matched.iterrows()
        )
        assert every[tier] == expected
    assert every["B"] == N_MATCHED
    assert every["A"] < min(th["A"][m.name]["nMeeting"] for m in METRICS)  # not a per-metric share
    assert every["A"] >= result.tiering["counts"]["matched"]["A"]  # the A rules only remove events
    # Unmatched events never move the bars.
    shifted = events.copy()
    unmatched = ~shifted["id"].isin(matched_ids)
    shifted.loc[unmatched, "quality_rmsS"] = 9.0
    assert assign_tiers(shifted, matches, cfg).tiering["thresholds"] == th


@pytest.mark.smoke
def test_a_bar_on_the_metrics_bound_is_flagged(cfg: SeismologyConfig) -> None:
    """A quarter of M without S picks puts the A bar at nS >= 0: recorded as excluding nothing."""
    models = [event(k, nS=0 if k < 4 else 4) for k in range(12)]  # the 9th best of 12 is a 0
    events = located(models)
    bars = assign_tiers(events, matches_for(events, [m.id for m in models]), cfg).tiering[
        "thresholds"]["A"]
    assert bars["nS"]["value"] == 0 and bars["nS"]["excludesNothing"]
    assert not any(bars[m.name]["excludesNothing"] for m in METRICS if m.name != "nS")


@pytest.mark.smoke
def test_boundary_equality_passes(cfg: SeismologyConfig) -> None:
    """Every matched event identical: each bar equals every value, and all of them pass A."""
    models = [event(k) for k in range(12)]
    extra = [event(12), event(13, rmsS=0.0501)]  # unmatched: on the bar, and just past it
    events = located(models + extra)
    matches = matches_for(events, [m.id for m in models])
    out = assign_tiers(events, matches, cfg).events
    assert list(out["tier"]) == ["A"] * 13 + ["C"]  # past the worst matched event too
    assert out["tierReasons"].iloc[0][3] == "rmsS 0.050 <= 0.050 (A: p75 of matched, n=12)"
    reason = out["tierReasons"].iloc[13][3]
    assert reason.startswith("rmsS 0.0501 > 0.0500 (A: p75 of matched, n=12); > 0.0500 (B")


@pytest.mark.smoke
def test_reasons_never_read_as_false_inequalities(cfg: SeismologyConfig) -> None:
    """A value within one display unit of the B bar: value and bars share one precision."""
    gaps = [100.0] * 8 + [130.6, 150.0, 160.0, 188.9]  # A bar 130.6 (9th of 12), B bar 188.9
    models = [event(k, gapDeg=g) for k, g in enumerate(gaps)]
    events = located(models + [event(12, gapDeg=188.86)])
    out = assign_tiers(events, matches_for(events, [m.id for m in models]), cfg).events
    assert out["tier"].iloc[12] == "B"
    assert out["tierReasons"].iloc[12][6] == (
        "gapDeg 188.86 > 130.60 (A: p75 of matched, n=12); <= 188.90 (B: worst of matched)"
    )


@pytest.mark.smoke
def test_null_errors_fail_a_and_b(cfg: SeismologyConfig) -> None:
    models = [event(k) for k in range(12)]
    events = located(models + [event(12, hErrM=None), event(13, vErrM=None)])
    matches = matches_for(events, [m.id for m in models])
    out = assign_tiers(events, matches, cfg).events
    assert list(out["tier"].iloc[12:]) == ["C", "C"]
    assert "hErrM null (no formal error): fails A and B" in out["tierReasons"].iloc[12]
    assert "vErrM null (no formal error): fails A and B" in out["tierReasons"].iloc[13]


@pytest.mark.smoke
def test_a_null_error_in_the_matched_set_never_loosens_a_bar(cfg: SeismologyConfig) -> None:
    """One null hErrM among 12 matched: both bars come from the 11 matched values, so the null
    event fails, and so does an event worse than every matched value."""
    models = [event(k, hErrM=100.0 + 10 * k) for k in range(11)] + [event(11, hErrM=None)]
    events = located(models + [event(12, hErrM=5000.0)])
    matches = matches_for(events, [m.id for m in models])
    result = assign_tiers(events, matches, cfg)
    a, b = (result.tiering["thresholds"][t]["hErrM"] for t in ("A", "B"))
    assert a["value"] == 100.0 + 10 * 8  # rank ceil(0.75 * 11) = 9 of the 11 values
    assert b["value"] == 200.0  # the worst matched value, not "unbounded"
    assert (a["n"], a["nUsed"], a["nNull"]) == (b["n"], b["nUsed"], b["nNull"]) == (12, 11, 1)
    assert (a["nMeeting"], b["nMeeting"]) == (9, 11)  # the null itself never meets a bar
    assert result.tiering["thresholds"]["A"]["rmsS"]["nUsed"] == 12
    tiers = list(result.events["tier"])
    assert tiers[11] == "C" and tiers[12] == "C"
    assert result.events["tierReasons"][12][4] == (
        "hErrM 5000 > 180 (A: p75 of matched, n=11); > 200 (B: worst of matched)"
    )
    rebuilt = Thresholds.from_record(result.tiering["thresholds"])
    assert rebuilt.bars["B"]["hErrM"].passes(200.0) and not rebuilt.bars["B"]["hErrM"].passes(None)


@pytest.mark.smoke
def test_too_few_matched_values_on_one_metric_fail_loudly(cfg: SeismologyConfig) -> None:
    min_matched = cfg.tiering.minMatched
    models = [event(k, vErrM=None if k < 3 else 200.0) for k in range(min_matched + 2)]
    events = located(models)
    matches = matches_for(events, [m.id for m in models])
    with pytest.raises(TierError, match=f"only {min_matched - 1} of the {len(models)} matched "
                       "events have a vErrM value"):
        assign_tiers(events, matches, cfg)


# --- rules ----------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_depth_on_edge_and_map_on_top_exclude_tier_a(cfg: SeismologyConfig) -> None:
    models = [event(k) for k in range(12)]
    events = located(models + [event(12, depthOnEdge=True), event(13)])
    matches = matches_for(events, [m.id for m in models])
    flags = pd.DataFrame({"eventId": events["id"], "mapOnVolumeTop": [False] * 13 + [True]})
    without = assign_tiers(events, matches, cfg)
    assert list(without.events["tier"].iloc[12:]) == ["B", "A"]  # no flags: MAP rule not applied
    assert not without.tiering["rules"]["A"]["mapOnVolumeTop"]["applied"]
    with_flags = assign_tiers(events, matches, cfg, flags=flags)
    assert list(with_flags.events["tier"].iloc[12:]) == ["B", "B"]
    assert with_flags.tiering["rules"]["A"]["mapOnVolumeTop"]["applied"]
    reasons = with_flags.events["tierReasons"]
    assert reasons.iloc[12][-1].startswith("depthOnEdge") and reasons.iloc[12][-1].endswith("A")
    assert reasons.iloc[13][-1] == "MAP on the search-volume top: fails A"
    assert len(reasons.iloc[0]) == len(METRICS)  # a Tier A event: one string per metric only


@pytest.mark.smoke
def test_nearest_station_rule(cfg: SeismologyConfig) -> None:
    factor = cfg.tiering.strictNearestStationFactor
    models = [event(k, depth_m=3000.0, minEpiDistM=100.0) for k in range(12)]
    depth_m = event(12, depth_m=2000.0).depthKm * 1000.0  # focal depth as the rule computes it
    on_limit = event(12, depth_m=2000.0, minEpiDistM=factor * depth_m)
    past = event(13, depth_m=2000.0, minEpiDistM=factor * depth_m + 1.0)
    events = located(models + [on_limit, past])
    out = assign_tiers(events, matches_for(events, [m.id for m in models]), cfg)
    assert list(out.events["tier"].iloc[12:]) == ["A", "B"]
    assert out.events["tierReasons"].iloc[13][-1] == (
        f"nearest station 4001 m > {factor:g} x focal depth 2000 m (epicentral, depth below "
        "refSurfaceElevM): fails A"
    )
    focal = out.tiering["rules"]["A"]["nearestStation"]["focalDepth"]
    assert "refSurfaceElevM - elevM" in focal


@pytest.mark.smoke
def test_nearest_station_rule_measures_depth_below_the_nearest_used_sensor(
    cfg: SeismologyConfig,
) -> None:
    """With arrivals and stations the focal depth is the depth below the nearest used station's
    sensor, so a borehole sensor tightens the rule and a sensor above the reference loosens it."""
    factor = cfg.tiering.strictNearestStationFactor
    models = [event(k, depth_m=3000.0, minEpiDistM=100.0) for k in range(12)]
    borehole = REF - 500.0
    below_borehole = borehole - event(13, depth_m=2000.0).elevM  # as the rule computes it
    cases = [  # (depth below REF, minEpiDistM, nearest used sensor's elevation)
        (2000.0, 5000.0, REF + 600.0),  # 2600 m below a high sensor: passes (fails vs REF)
        (2000.0, factor * below_borehole, borehole),  # on the limit below a borehole: passes
        (2000.0, factor * below_borehole + 1.0, borehole),  # just past it (passes vs REF)
        (300.0, 100.0, borehole),  # above the borehole sensor: fails (passes vs REF)
    ]
    extra = [event(12 + i, depth_m=d, minEpiDistM=m) for i, (d, m, _) in enumerate(cases)]
    events = located(models + extra)
    matches = matches_for(events, [m.id for m in models])
    _, arrivals = picks_and_arrivals(events)
    stations = stations_for(events, {12 + i: elev for i, (_, _, elev) in enumerate(cases)})
    reference = assign_tiers(events, matches, cfg)
    out = assign_tiers(events, matches, cfg, arrivals=arrivals, stations=stations)
    assert list(reference.events["tier"].iloc[12:]) == ["B", "A", "A", "A"]
    assert list(out.events["tier"].iloc[:12]) == ["A"] * 12
    assert list(out.events["tier"].iloc[12:]) == ["A", "A", "B", "B"]
    assert out.events["tierReasons"].iloc[14][-1] == (
        f"nearest station {station_id(14, 0)} 3001 m > {factor:g} x focal depth 1500 m "
        "(epicentral, depth below its sensor): fails A"
    )
    assert out.events["tierReasons"].iloc[15][-1].endswith("focal depth -200 m (epicentral, "
                                                            "depth below its sensor): fails A")
    rule = out.tiering["rules"]["A"]["nearestStation"]
    assert rule["focalDepthBelow"] == "nearestUsedSensor"
    assert "sensorElevM - elevM" in rule["focalDepth"]
    assert reference.tiering["rules"]["A"]["nearestStation"]["focalDepthBelow"] == (
        "refSurfaceElevM"
    )
    moved = stations.copy()  # stored minEpiDistM no longer matches the station table
    moved.loc[moved["id"] == station_id(12, 0), "enu_e"] += 1.0
    with pytest.raises(TierError, match="minEpiDistM"):
        assign_tiers(events, matches, cfg, arrivals=arrivals, stations=moved)
    with pytest.raises(TierError, match="together"):
        assign_tiers(events, matches, cfg, arrivals=arrivals)
    lonely = arrivals[arrivals["eventId"] != events["id"].iloc[3]]
    with pytest.raises(TierError, match="no used arrival"):
        assign_tiers(events, matches, cfg, arrivals=lonely, stations=stations)


# --- the final table ------------------------------------------------------------------------------


@pytest.mark.smoke
def test_final_events_are_contract_shaped(cfg: SeismologyConfig,
                                          tmp_path: Path) -> None:
    events, matches, matched_ids = seeded_world()
    result = assign_tiers(events, matches, cfg)
    out = result.events
    assert list(out.columns) == columns_for(SeismicEvent)
    assert {c: str(d) for c, d in out.dtypes.items()} == {
        c: str(pd.Series([], dtype=d).dtype) for c, d in dtypes_for(SeismicEvent).items()
    }
    assert set(out["tier"]) <= {"A", "B", "C"} and out["tier"].notna().all()
    assert out["tierReasons"].map(len).min() >= len(METRICS)
    assert (out["revealOrder"] == -1).all() and out["magnitude_value"].isna().all()
    assert list(out["id"]) == list(events["id"])  # input order kept
    for col in LOCATED_COLUMNS:
        if col not in ("revealOrder",):
            assert out[col].astype(str).tolist() == events[col].astype(str).tolist(), col
    by_event = matches.dropna(subset=["eventId"]).set_index("eventId")
    is_matched = out["id"].isin(matched_ids)
    assert (out.loc[is_matched, "catalogMatch_catalogId"].to_numpy()
            == by_event.loc[out.loc[is_matched, "id"], "catalogId"].to_numpy()).all()
    assert np.allclose(out.loc[is_matched, "catalogMatch_dtS"],
                       by_event.loc[out.loc[is_matched, "id"], "dtS"])
    assert out.loc[~is_matched, "catalogMatch_catalogId"].isna().all()
    counts = result.tiering["counts"]
    assert sum(counts["all"].values()) == len(out)
    assert sum(counts["matched"].values()) == N_MATCHED
    assert sum(counts["additional"].values()) == N_UNMATCHED
    assert result.tiering["caveat"].startswith("Public regional catalog events are the larger")
    write_table(out, tmp_path / "events.parquet", "SeismicEvent")
    assert read_models(tmp_path / "events.parquet", SeismicEvent) == from_frame(out, SeismicEvent)


@pytest.mark.smoke
def test_zero_events_give_a_typed_empty_table(cfg: SeismologyConfig) -> None:
    events = located([])
    result = assign_tiers(events, matches_for(events, [], n_unmatched_public=3),
                          cfg)
    assert len(result.events) == 0
    assert list(result.events.columns) == columns_for(SeismicEvent)
    assert result.tiering["thresholds"] is None
    assert result.tiering["note"] == "no located events, so no bars were derived"
    non_empty = assign_tiers(*seeded_world()[:2], cfg).tiering
    assert set(result.tiering) == set(non_empty)  # same top-level keys: nothing stale survives
    assert result.tiering["counts"]["all"] == {"A": 0, "B": 0, "C": 0}


@pytest.mark.smoke
def test_too_few_matched_fails_loudly_and_supplied_bars_apply(
    cfg: SeismologyConfig,
) -> None:
    events, matches, matched_ids = seeded_world()
    main = assign_tiers(events, matches, cfg)
    min_matched = cfg.tiering.minMatched
    few = matches_for(events, matched_ids[: min_matched - 1])
    with pytest.raises(TierError, match=f"minMatched is {min_matched}"):
        assign_tiers(events, few, cfg)
    applied = assign_tiers(events, few, cfg, thresholds=main.tiering)
    assert applied.tiering["thresholdSource"] == "supplied"
    assert applied.tiering["thresholds"] == main.tiering["thresholds"]
    assert list(applied.events["tier"]) == list(main.events["tier"])  # same bars, same tiers
    other = cfg_with(cfg, quantiles={"A": 0.2, "B": 0.0})
    with pytest.raises(TierError, match="quantiles"):
        assign_tiers(events, few, other, thresholds=main.tiering)


@pytest.mark.smoke
def test_inconsistent_inputs_fail_loudly(cfg: SeismologyConfig) -> None:
    events, matches, _ = seeded_world()
    stray = matches.copy()
    stray.loc[0, "eventId"] = "hq-other-000001"
    with pytest.raises(TierError, match="missing from events_located"):
        assign_tiers(events, stray, cfg)
    twice = matches.copy()
    twice.loc[1, "eventId"] = twice.loc[0, "eventId"]
    with pytest.raises(TierError, match="one-to-one"):
        assign_tiers(events, twice, cfg)
    flags = pd.DataFrame({"eventId": events["id"].iloc[1:], "mapOnVolumeTop": False})
    with pytest.raises(TierError, match="do not cover"):
        assign_tiers(events, matches, cfg, flags=flags)
    final = assign_tiers(events, matches, cfg).events
    with pytest.raises(TierError, match="final-event columns"):
        assign_tiers(final, matches, cfg)
    with pytest.raises(TierError, match="lacks columns"):
        assign_tiers(events.drop(columns=["quality_nS"]), matches, cfg)
    no_rms = events.copy()
    no_rms.loc[len(events) - 1, "quality_rmsS"] = np.nan  # an unmatched event
    with pytest.raises(TierError, match="null quality_rmsS"):
        assign_tiers(no_rms, matches, cfg)


# --- event picks ----------------------------------------------------------------------------------


def station_id(i: int, j: int) -> str:
    """The station of event ``i``'s ``j``-th pick (``stations_for`` places it)."""
    return f"T.E{i:03d}{j}"


def picks_and_arrivals(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    picks, arrivals = [], []
    for i, (eid, pids) in enumerate(zip(events["id"], events["pickIds"], strict=True)):
        for j, pid in enumerate(pids):
            phase = pid.rsplit(":", 1)[1]
            picks.append(Pick(id=pid, stationId=station_id(i, j), phase=phase, t=T0, prob=0.7,
                              picker="phasenet:test"))
            arrivals.append({"eventId": eid, "stationId": station_id(i, j), "phase": phase,
                             "tPred": T0, "tObs": T0 + 0.01 * j, "residualS": 0.01 * j,
                             "pickId": pid, "usedInLocation": True})
        arrivals.append({"eventId": eid, "stationId": "T.S99", "phase": "P", "tPred": T0,
                         "tObs": T0, "residualS": 0.5, "pickId": f"dropped:{eid}",
                         "usedInLocation": False})  # outlier-dropped: not in pickIds
    picks.append(Pick(id="unassociated", stationId="T.S99", phase="P", t=T0, prob=0.2,
                      picker="phasenet:test"))
    return to_frame(picks, Pick), pd.DataFrame(arrivals)


def stations_for(events: pd.DataFrame, nearest_elev: dict[int, float] | None = None
                 ) -> pd.DataFrame:
    """Stations consistent with each event's ``quality.minEpiDistM``: its first pick's station
    exactly that far east of the epicentre (sensorElevM REF, or ``nearest_elev[i]``), later ones
    500 m farther apart with sensors 5 km up (they would pass the rule if wrongly taken as the
    nearest), and T.S99 (dropped arrivals only) on event 0's epicentre, 5 km up."""
    rows = []
    for i, (e, n, d, pids) in enumerate(zip(events["enu_e"], events["enu_n"],
                                            events["quality_minEpiDistM"], events["pickIds"],
                                            strict=True)):
        for j in range(len(pids)):
            elev = (nearest_elev or {}).get(i, REF) if j == 0 else REF + 5000.0
            rows.append({"id": station_id(i, j), "enu_e": e + d + 500.0 * j, "enu_n": n,
                         "sensorElevM": elev})
    rows.append({"id": "T.S99", "enu_e": 0.0, "enu_n": 0.0, "sensorElevM": REF + 5000.0})
    return pd.DataFrame(rows)


@pytest.mark.smoke
def test_event_picks_carry_event_and_residual(cfg: SeismologyConfig) -> None:
    events, matches, _ = seeded_world()
    final = assign_tiers(events, matches, cfg).events
    picks, arrivals = picks_and_arrivals(final)
    out = event_picks(final, arrivals, picks)
    assert list(out.columns) == columns_for(Pick)
    assert len(out) == int(final["pickIds"].map(len).sum())
    assert list(out["id"]) == [p for pids in final["pickIds"] for p in pids]
    assert list(out["eventId"]) == [e for e, pids in zip(final["id"], final["pickIds"],
                                                         strict=True) for _ in pids]
    assert np.allclose(out["residualS"], [0.0, 0.01] * len(final))
    with pytest.raises(TierError, match="no used arrival"):
        event_picks(final, arrivals[arrivals["pickId"] != final["pickIds"].iloc[0][0]], picks)
    with pytest.raises(TierError, match="not in the picks table"):
        event_picks(final, arrivals, picks.iloc[1:])
    doubled = final.copy()
    doubled["pickIds"] = [list(final["pickIds"].iloc[0]) if i == 1 else p
                          for i, p in enumerate(final["pickIds"])]
    with pytest.raises(TierError, match="more than one event"):
        event_picks(doubled, arrivals, picks)


# --- stage ----------------------------------------------------------------------------------------


def catalog_for(events: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Public events ``matches`` could have come from: each matched one ``dtS`` earlier and
    ``distM`` west of its event. Returns (catalog, matches with ``dtS`` / ``distM`` recomputed
    from the two tables as stage match computes them)."""
    ev = events.set_index("id")
    rows = []
    for r in matches.itertuples(index=False):
        if pd.isna(r.eventId):
            rows.append({"id": r.catalogId, "t": T0 - 3600.0, "enu_e": 0.0, "enu_n": 0.0})
            continue
        x = ev.loc[r.eventId]
        rows.append({"id": r.catalogId, "t": x["t"] - r.dtS, "enu_e": x["enu_e"] - r.distM,
                     "enu_n": x["enu_n"]})
    catalog = pd.DataFrame(rows)
    matched = matches["eventId"].notna().to_numpy()
    x = ev.loc[matches.loc[matched, "eventId"]]
    c = catalog.set_index("id").loc[matches.loc[matched, "catalogId"]]
    out = matches.copy()
    out.loc[matched, "dtS"] = x["t"].to_numpy() - c["t"].to_numpy()
    out.loc[matched, "distM"] = np.hypot(x["enu_e"].to_numpy() - c["enu_e"].to_numpy(),
                                         x["enu_n"].to_numpy() - c["enu_n"].to_numpy())
    return catalog, out


def write_run(ctx: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    events, matches, _ = seeded_world()
    catalog, matches = catalog_for(events, matches)
    final = assign_tiers(events, matches, ctx.config.seismology).events
    picks, arrivals = picks_and_arrivals(final)
    write_table(stations_for(events), ctx.path("stations.parquet"), "Station")
    write_table(events, ctx.path("events_located.parquet"), "LocatedEvent")
    write_table(matches, ctx.path("matches.parquet"), "Match")
    write_table(catalog, ctx.path("catalog.parquet"), "CatalogEvent")
    write_table(arrivals, ctx.path("arrivals.parquet"), "Arrival")
    write_table(picks, ctx.path(ctx.config.seismology.associator.picksTable), "Pick")
    write_table(pd.DataFrame({"eventId": events["id"], "mapOnVolumeTop": False}),
                ctx.path("locate_flags.parquet"), "LocateFlags")
    return events, matches


def tier_record(ctx: Any) -> dict[str, Any]:
    """The stage's ProcessingRun.tiering record (it also records the magnitude status)."""
    (rec,) = [r for r in ctx.records if r["field"] is None]
    return rec


@pytest.mark.smoke
def test_stage_writes_final_tables_and_record(
    make_ctx: Any, run: RunSection, cfg: SeismologyConfig
) -> None:
    stage = importlib.import_module("hq.tier.run")
    ctx = make_ctx(run, cfg)
    events, _ = write_run(ctx)
    ctx.path("sweep.parquet").write_bytes(b"from an earlier tier run")
    ctx.path("magnitude.json").write_text('{"n": 23}', encoding="utf-8")  # an earlier MAG-01 run
    stage.run(ctx)
    final = read_models(ctx.path("events.parquet"), SeismicEvent)
    assert [e.id for e in final] == list(events["id"])
    assert all(e.magnitude is None for e in final)
    picks = read_models(ctx.path("event_picks.parquet"), Pick)
    assert len(picks) == sum(len(e.pickIds) for e in final)
    assert not ctx.path("sweep.parquet").exists()
    # No calibration outlives the magnitudes it wrote; its run record is replaced too.
    assert not ctx.path("magnitude.json").exists()
    (mag,) = [r for r in ctx.records if r["field"] == "matching"]
    assert mag["stage"] == "tier" and mag["params"]["magnitude"]["removedMagnitudeJson"]
    assert "stage magnitude" in mag["params"]["magnitude"]["status"]
    assert not list(ctx.run_dir.glob("*.part"))
    rec = tier_record(ctx)
    assert rec["stage"] == "tier" and rec["params"]["removedMagnitudeJson"]
    counts, params = rec["counts"], rec["params"]
    assert counts["events"] == len(events) == counts["tierA"] + counts["tierB"] + counts["tierC"]
    assert counts["additional"] == N_UNMATCHED and counts["matched"] == N_MATCHED
    assert counts["eventPicks"] == len(picks) and counts["sweepPoints"] == 0
    assert params["thresholds"]["A"]["rmsS"]["label"] == "p75 of matched"
    assert params["thresholds"]["nMatched"] == N_MATCHED
    assert params["rules"]["A"]["mapOnVolumeTop"]["applied"]
    assert params["rules"]["A"]["nearestStation"]["focalDepthBelow"] == "nearestUsedSensor"
    assert params["input"]["stations"] == "stations.parquet"
    assert params["sweep"] == {"enabled": False, "removedEarlierSweep": True}
    assert params["config"]["quantiles"] == {"A": 0.25, "B": 0.0}
    assert params["input"]["flags"] == "locate_flags.parquet"
    # The code this stage ran with (run.json's gitSha / softwareVersions date from run creation).
    prov = params["provenance"]
    assert prov["gitSha"] and prov["softwareVersions"]["python"] and "pandas" in prov[
        "softwareVersions"]
    # Stage locate writes locate_flags.parquet with events_located: without it the stage fails
    # (assign_tiers keeps the no-flags fallback for the docs/02 three-argument call only).
    ctx.path("locate_flags.parquet").unlink()
    ctx.path("events.parquet").unlink()
    with pytest.raises(TierError, match="locate_flags.parquet is absent"):
        stage.run(ctx)
    assert not ctx.path("events.parquet").exists()


def test_stage_checks_depth_against_the_run_section(
    make_ctx: Any, run: RunSection, cfg: SeismologyConfig
) -> None:
    stage = importlib.import_module("hq.tier.run")
    tol_m = cfg.tiering.consistencyTolM
    close = make_ctx(run.model_copy(update={"refSurfaceElevM": REF + 0.5 * tol_m}), cfg)
    write_run(close)
    stage.run(close)  # within tiering.consistencyTolM: round-off from another writer passes
    other = make_ctx(run.model_copy(update={"refSurfaceElevM": REF + 10.0}), cfg)
    write_run(other)
    with pytest.raises(TierError, match="depthKm"):
        stage.run(other)


@pytest.mark.smoke
def test_stage_refuses_matches_written_for_other_locations(
    make_ctx: Any, run: RunSection, cfg: SeismologyConfig
) -> None:
    stage = importlib.import_module("hq.tier.run")
    ctx = make_ctx(run, cfg)
    events, matches = write_run(ctx)
    catalog = read_table(ctx.path("catalog.parquet"))
    tol_m = cfg.tiering.consistencyTolM
    stage.check_matches_current(events, matches, catalog, tol_m)  # as matched: passes
    first = events["id"] == matches["eventId"].dropna().iloc[0]
    # Relocated after the match (LOC-05 statics pass 2): a new origin time, or a new epicentre.
    for moved in (events.assign(t=events["t"].where(~first, events["t"] + 0.01)),
                  events.assign(enu_e=events["enu_e"].where(~first, events["enu_e"] + 50.0))):
        with pytest.raises(TierError, match="stale"):
            stage.check_matches_current(moved, matches, catalog, tol_m)
    write_table(moved, ctx.path("events_located.parquet"), "LocatedEvent")
    with pytest.raises(TierError, match="rerun stage match"):
        stage.run(ctx)
    assert not ctx.path("events.parquet").exists()
    with pytest.raises(TierError, match="missing from catalog.parquet"):
        stage.check_matches_current(events, matches, catalog.iloc[1:], tol_m)
    ctx.path("catalog.parquet").unlink()
    with pytest.raises(TierError, match="catalog.parquet is absent"):
        stage.run(ctx)


@pytest.mark.parametrize("first", [None, "hq.tier.run"])
@pytest.mark.smoke
def test_stage_resolves_to_the_stage_function(first: str | None) -> None:
    if first is not None:
        importlib.import_module(first)
    fn = resolve_stage(stage_spec("tier"))
    assert fn is importlib.import_module("hq.tier.run").run


# --- sweep ----------------------------------------------------------------------------------------


def test_sweep_counts_tier_a_with_the_configured_runs_bars(
    cfg: SeismologyConfig,
) -> None:
    from hq.associate.result import (
        EVENT_DTYPES,
        PICK_DTYPES,
        AssocResult,
        empty_result,
        typed_frame,
    )
    from hq.associate.sweep import SweepRow
    from hq.tier.sweep import SweepPipeline, score_sweep

    events, _, matched_ids = seeded_world()
    main = derive_thresholds(events[events["id"].isin(matched_ids)], cfg.tiering)

    def assoc(n: int) -> AssocResult:
        return AssocResult(
            typed_frame({"assocId": [f"a{i}" for i in range(n)], "t": [T0] * n,
                         "latitude": [38.5] * n, "longitude": [-112.9] * n,
                         "elevM": [0.0] * n, "nPicks": [8] * n, "nP": [5] * n,
                         "nS": [3] * n}, EVENT_DTYPES),
            typed_frame(None, PICK_DTYPES),
        )

    def fake_locate(a: AssocResult) -> tuple[pd.DataFrame, None, None]:
        return events.iloc[: len(a.events)].reset_index(drop=True), None, None

    def fake_match(ev: pd.DataFrame) -> pd.DataFrame:  # only 3 matches: far below minMatched
        return matches_for(ev, list(ev["id"].iloc[:3]))

    grid = [{"minPickProb": 0.3, "nSPicks": 1, "minStations": s} for s in (4, 5, 6)]

    def run_points(evaluate: Any) -> list[SweepRow]:
        sizes = [len(events), 30, 0]
        results = [assoc(n) if n else empty_result() for n in sizes]
        return [SweepRow(params=p, candidates=len(r.events), counts={}, score=evaluate(r),
                         runtime_s=0.5) for p, r in zip(grid, results, strict=True)]

    points, record = score_sweep(run_points, SweepPipeline(fake_locate, fake_match, None),
                                 cfg, main)
    expected_a = [0 if n == 0 else int((assign_tiers(
        events.iloc[:n], matches_for(events.iloc[:n], []), cfg, thresholds=main
    ).events["tier"] == "A").sum()) for n in (len(events), 30, 0)]
    assert [p.candidates for p in points] == [len(events), 30, 0]
    assert [p.recoveredPublic for p in points] == [3, 3, 0]
    assert [p.tierA for p in points] == expected_a and expected_a[0] > 0
    assert [r["runtimeS"] for r in record] == [0.5, 0.5, 0.5]
    frame = to_frame(points, SweepPoint)
    assert from_frame(frame, SweepPoint) == points


def test_stage_writes_sweep_parquet_when_enabled(
    make_ctx: Any, run: RunSection, cfg: SeismologyConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hq.associate.sweep import SweepRow, SweepScore

    stage = importlib.import_module("hq.tier.run")
    ctx = make_ctx(run, cfg_with(cfg, sweep={"enabled": True}))
    write_run(ctx)
    statics = pd.DataFrame({"stationId": ["T.E0000", "T.E0000"], "phase": ["P", "S"],
                            "staticS": [0.0, -0.12], "nEvents": [3, 3]})
    write_table(statics, ctx.path("statics.parquet"), "StationStatic")
    seen: list[Thresholds] = []
    pipeline_kwargs: list[dict[str, Any]] = []

    def fake_pipeline(*args: Any, **kwargs: Any) -> tuple[None, None]:
        pipeline_kwargs.append(kwargs)
        return None, None

    def fake_score(run_points: Any, pipeline: Any, cfg: Any, thresholds: Thresholds) -> Any:
        seen.append(thresholds)
        row = SweepRow(params={"minPickProb": 0.3, "nSPicks": 1, "minStations": 5},
                       candidates=4, counts={}, score=SweepScore(3, 2, 1), runtime_s=1.0)
        return [SweepPoint(params=row.params, candidates=3, recoveredPublic=2, tierA=1)], [
            {"params": row.params, "tierA": 1}]

    monkeypatch.setattr("hq.tier.sweep.real_pipeline", fake_pipeline)
    monkeypatch.setattr("hq.tier.sweep.score_sweep", fake_score)
    stage.run(ctx)
    assert read_models(ctx.path("sweep.parquet"), SweepPoint)[0].tierA == 1
    rec = tier_record(ctx)
    assert rec["counts"]["sweepPoints"] == 1 and rec["params"]["sweep"]["enabled"]
    assert seen[0].to_record() == rec["params"]["thresholds"]
    # Sweep points are located with the run's statics.parquet.
    assert pipeline_kwargs[0]["statics"] == {("T.E0000", "P"): 0.0, ("T.E0000", "S"): -0.12}
    record = rec["params"]["sweep"]["statics"]
    assert {k: record[k] for k in ("table", "stationPhases", "nonZero", "maxAbsS")} == {
        "table": "statics.parquet", "stationPhases": 2, "nonZero": 1, "maxAbsS": 0.12}
    # Reference-event terms come from the public events the points are matched to: in-sample.
    assert record["inSample"] is (cfg.statics.mode == "referenceEvents")
    ctx.path("statics.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        stage.run(ctx)

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
from hq_contracts.io import columns_for, dtypes_for, from_frame, read_models, to_frame, write_table
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
REF = 1627.7  # run.yaml refSurfaceElevM; the tests check it against the fixture below

pytestmark = pytest.mark.smoke


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


def worst_side_rank_value(values: np.ndarray, better: str, q: float) -> float:
    """Independent statement of the bar: the ceil((1 - q) n)-th best value."""
    order = np.sort(values) if better == "lower" else np.sort(values)[::-1]
    return float(order[math.ceil((1 - q) * len(values)) - 1])


# --- bars -----------------------------------------------------------------------------------------


def test_bars_are_quantiles_of_the_matched_set(seismology_config: SeismologyConfig) -> None:
    events, matches, matched_ids = seeded_world()
    result = assign_tiers(events, matches, seismology_config)
    th = result.tiering["thresholds"]
    matched = events[events["id"].isin(matched_ids)]
    q = seismology_config.tiering.quantiles
    assert th["nMatched"] == N_MATCHED == result.tiering["matchedSet"]["n"]
    assert th["quantiles"] == {"A": q.A, "B": q.B}
    for metric in METRICS:
        values = matched[metric.column].to_numpy(dtype=np.float64)
        a, b = th["A"][metric.name], th["B"][metric.name]
        assert a["value"] == worst_side_rank_value(values, metric.better, q.A)
        worst = values.min() if metric.better == "higher" else values.max()
        assert b["value"] == worst and b["label"] == "worst of matched"
        assert a["n"] == b["n"] == N_MATCHED
        assert a["label"] == ("p25 of matched" if metric.better == "higher" else "p75 of matched")
        assert a["quantile"] == (0.25 if metric.better == "higher" else 0.75)
        assert a["op"] == metric.op
        assert a["nMeeting"] >= math.ceil(0.75 * N_MATCHED)  # three-quarters meet each A bar
        assert b["nMeeting"] == N_MATCHED
    # Unmatched events never move the bars.
    shifted = events.copy()
    unmatched = ~shifted["id"].isin(matched_ids)
    shifted.loc[unmatched, "quality_rmsS"] = 9.0
    assert assign_tiers(shifted, matches, seismology_config).tiering["thresholds"] == th


def test_boundary_equality_passes(seismology_config: SeismologyConfig) -> None:
    """Every matched event identical: each bar equals every value, and all of them pass A."""
    models = [event(k) for k in range(12)]
    extra = [event(12), event(13, rmsS=0.0501)]  # unmatched: on the bar, and just past it
    events = located(models + extra)
    matches = matches_for(events, [m.id for m in models])
    out = assign_tiers(events, matches, seismology_config).events
    assert list(out["tier"]) == ["A"] * 13 + ["C"]  # past the worst matched event too
    assert out["tierReasons"].iloc[0][3] == "rmsS 0.050 <= 0.050 (A: p75 of matched, n=12)"
    reason = out["tierReasons"].iloc[13][3]
    assert reason.startswith("rmsS 0.0501 > 0.0500 (A: p75 of matched, n=12); > 0.0500 (B")


def test_null_errors_fail_a_and_b(seismology_config: SeismologyConfig) -> None:
    models = [event(k) for k in range(12)]
    events = located(models + [event(12, hErrM=None), event(13, vErrM=None)])
    matches = matches_for(events, [m.id for m in models])
    out = assign_tiers(events, matches, seismology_config).events
    assert list(out["tier"].iloc[12:]) == ["C", "C"]
    assert "hErrM null (no formal error): fails A and B" in out["tierReasons"].iloc[12]
    assert "vErrM null (no formal error): fails A and B" in out["tierReasons"].iloc[13]


def test_null_error_in_the_matched_set_counts_as_worst(
    seismology_config: SeismologyConfig,
) -> None:
    """One null hErrM among 12 matched: the A bar (p75) is finite, the B bar unbounded."""
    models = [event(k, hErrM=100.0 + 10 * k) for k in range(11)] + [event(11, hErrM=None)]
    events = located(models + [event(12, hErrM=5000.0)])
    matches = matches_for(events, [m.id for m in models])
    result = assign_tiers(events, matches, seismology_config)
    a, b = (result.tiering["thresholds"][t]["hErrM"] for t in ("A", "B"))
    assert a["value"] == 100.0 + 10 * 8  # rank ceil(0.75 * 12) = 9 of 12, null last
    assert b["value"] is None and b["unbounded"] and b["nNull"] == 1
    assert b["nMeeting"] == 11  # the null itself never meets a bar
    tiers = list(result.events["tier"])
    assert tiers[11] == "C" and tiers[12] == "B"  # a finite 5 km passes the unbounded B bar
    assert "B: worst of matched is null, any value meets it" in result.events["tierReasons"][12][4]
    rebuilt = Thresholds.from_record(result.tiering["thresholds"])
    assert rebuilt.bars["B"]["hErrM"].passes(1e9) and not rebuilt.bars["B"]["hErrM"].passes(None)


# --- rules ----------------------------------------------------------------------------------------


def test_depth_on_edge_and_map_on_top_exclude_tier_a(seismology_config: SeismologyConfig) -> None:
    models = [event(k) for k in range(12)]
    events = located(models + [event(12, depthOnEdge=True), event(13)])
    matches = matches_for(events, [m.id for m in models])
    flags = pd.DataFrame({"eventId": events["id"], "mapOnVolumeTop": [False] * 13 + [True]})
    without = assign_tiers(events, matches, seismology_config)
    assert list(without.events["tier"].iloc[12:]) == ["B", "A"]  # no flags: MAP rule not applied
    assert not without.tiering["rules"]["A"]["mapOnVolumeTop"]["applied"]
    with_flags = assign_tiers(events, matches, seismology_config, flags=flags)
    assert list(with_flags.events["tier"].iloc[12:]) == ["B", "B"]
    assert with_flags.tiering["rules"]["A"]["mapOnVolumeTop"]["applied"]
    reasons = with_flags.events["tierReasons"]
    assert reasons.iloc[12][-1].startswith("depthOnEdge") and reasons.iloc[12][-1].endswith("A")
    assert reasons.iloc[13][-1] == "MAP on the search-volume top: fails A"
    assert len(reasons.iloc[0]) == len(METRICS)  # a Tier A event: one string per metric only


def test_nearest_station_rule(seismology_config: SeismologyConfig) -> None:
    factor = seismology_config.tiering.strictNearestStationFactor
    models = [event(k, depth_m=3000.0, minEpiDistM=100.0) for k in range(12)]
    depth_m = event(12, depth_m=2000.0).depthKm * 1000.0  # focal depth as the rule computes it
    on_limit = event(12, depth_m=2000.0, minEpiDistM=factor * depth_m)
    past = event(13, depth_m=2000.0, minEpiDistM=factor * depth_m + 1.0)
    events = located(models + [on_limit, past])
    out = assign_tiers(events, matches_for(events, [m.id for m in models]), seismology_config)
    assert list(out.events["tier"].iloc[12:]) == ["A", "B"]
    assert out.events["tierReasons"].iloc[13][-1] == (
        f"nearest station 4001 m > {factor:g} x focal depth 2000 m (epicentral, depth below "
        "refSurfaceElevM): fails A"
    )
    focal = out.tiering["rules"]["A"]["nearestStation"]["focalDepth"]
    assert "refSurfaceElevM - elevM" in focal


# --- the final table ------------------------------------------------------------------------------


def test_final_events_are_contract_shaped(seismology_config: SeismologyConfig,
                                          tmp_path: Path) -> None:
    events, matches, matched_ids = seeded_world()
    result = assign_tiers(events, matches, seismology_config)
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


def test_zero_events_give_a_typed_empty_table(seismology_config: SeismologyConfig) -> None:
    events = located([])
    result = assign_tiers(events, matches_for(events, [], n_unmatched_public=3),
                          seismology_config)
    assert len(result.events) == 0
    assert list(result.events.columns) == columns_for(SeismicEvent)
    assert result.tiering["thresholds"] is None
    assert result.tiering["counts"]["all"] == {"A": 0, "B": 0, "C": 0}


def test_too_few_matched_fails_loudly_and_supplied_bars_apply(
    seismology_config: SeismologyConfig,
) -> None:
    events, matches, matched_ids = seeded_world()
    main = assign_tiers(events, matches, seismology_config)
    few = matches_for(events, matched_ids[:9])  # minMatched is 10
    with pytest.raises(TierError, match="minMatched is 10"):
        assign_tiers(events, few, seismology_config)
    applied = assign_tiers(events, few, seismology_config, thresholds=main.tiering)
    assert applied.tiering["thresholdSource"] == "supplied"
    assert applied.tiering["thresholds"] == main.tiering["thresholds"]
    assert list(applied.events["tier"]) == list(main.events["tier"])  # same bars, same tiers
    other = cfg_with(seismology_config, quantiles={"A": 0.2, "B": 0.0})
    with pytest.raises(TierError, match="quantiles"):
        assign_tiers(events, few, other, thresholds=main.tiering)


def test_inconsistent_inputs_fail_loudly(seismology_config: SeismologyConfig) -> None:
    events, matches, _ = seeded_world()
    stray = matches.copy()
    stray.loc[0, "eventId"] = "hq-other-000001"
    with pytest.raises(TierError, match="missing from events_located"):
        assign_tiers(events, stray, seismology_config)
    twice = matches.copy()
    twice.loc[1, "eventId"] = twice.loc[0, "eventId"]
    with pytest.raises(TierError, match="one-to-one"):
        assign_tiers(events, twice, seismology_config)
    flags = pd.DataFrame({"eventId": events["id"].iloc[1:], "mapOnVolumeTop": False})
    with pytest.raises(TierError, match="do not cover"):
        assign_tiers(events, matches, seismology_config, flags=flags)
    final = assign_tiers(events, matches, seismology_config).events
    with pytest.raises(TierError, match="final-event columns"):
        assign_tiers(final, matches, seismology_config)
    with pytest.raises(TierError, match="lacks columns"):
        assign_tiers(events.drop(columns=["quality_nS"]), matches, seismology_config)
    no_rms = events.copy()
    no_rms.loc[len(events) - 1, "quality_rmsS"] = np.nan  # an unmatched event
    with pytest.raises(TierError, match="null quality_rmsS"):
        assign_tiers(no_rms, matches, seismology_config)


# --- event picks ----------------------------------------------------------------------------------


def picks_and_arrivals(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    picks, arrivals = [], []
    for eid, pids in zip(events["id"], events["pickIds"], strict=True):
        for j, pid in enumerate(pids):
            phase = pid.rsplit(":", 1)[1]
            picks.append(Pick(id=pid, stationId=f"T.S{j:02d}", phase=phase, t=T0, prob=0.7,
                              picker="phasenet:test"))
            arrivals.append({"eventId": eid, "stationId": f"T.S{j:02d}", "phase": phase,
                             "tPred": T0, "tObs": T0 + 0.01 * j, "residualS": 0.01 * j,
                             "pickId": pid, "usedInLocation": True})
        arrivals.append({"eventId": eid, "stationId": "T.S99", "phase": "P", "tPred": T0,
                         "tObs": T0, "residualS": 0.5, "pickId": f"dropped:{eid}",
                         "usedInLocation": False})  # outlier-dropped: not in pickIds
    picks.append(Pick(id="unassociated", stationId="T.S00", phase="P", t=T0, prob=0.2,
                      picker="phasenet:test"))
    return to_frame(picks, Pick), pd.DataFrame(arrivals)


def test_event_picks_carry_event_and_residual(seismology_config: SeismologyConfig) -> None:
    events, matches, _ = seeded_world()
    final = assign_tiers(events, matches, seismology_config).events
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


def write_run(ctx: Any, *, flags: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    events, matches, _ = seeded_world()
    final = assign_tiers(events, matches, ctx.config.seismology).events
    picks, arrivals = picks_and_arrivals(final)
    write_table(events, ctx.path("events_located.parquet"), "LocatedEvent")
    write_table(matches, ctx.path("matches.parquet"), "Match")
    write_table(arrivals, ctx.path("arrivals.parquet"), "Arrival")
    write_table(picks, ctx.path(ctx.config.seismology.associator.picksTable), "Pick")
    if flags:
        write_table(pd.DataFrame({"eventId": events["id"], "mapOnVolumeTop": False}),
                    ctx.path("locate_flags.parquet"), "LocateFlags")
    return events, matches


def test_stage_writes_final_tables_and_record(make_ctx: Any, run_section: RunSection) -> None:
    assert run_section.refSurfaceElevM == REF
    stage = importlib.import_module("hq.tier.run")
    ctx = make_ctx(run_section)
    events, _ = write_run(ctx)
    ctx.path("sweep.parquet").write_bytes(b"from an earlier tier run")
    stage.run(ctx)
    final = read_models(ctx.path("events.parquet"), SeismicEvent)
    assert [e.id for e in final] == list(events["id"])
    picks = read_models(ctx.path("event_picks.parquet"), Pick)
    assert len(picks) == sum(len(e.pickIds) for e in final)
    assert not ctx.path("sweep.parquet").exists()
    assert not list(ctx.run_dir.glob("*.part"))
    (rec,) = ctx.records
    assert rec["stage"] == "tier"
    counts, params = rec["counts"], rec["params"]
    assert counts["events"] == len(events) == counts["tierA"] + counts["tierB"] + counts["tierC"]
    assert counts["additional"] == N_UNMATCHED and counts["matched"] == N_MATCHED
    assert counts["eventPicks"] == len(picks) and counts["sweepPoints"] == 0
    assert params["thresholds"]["A"]["rmsS"]["label"] == "p75 of matched"
    assert params["thresholds"]["nMatched"] == N_MATCHED
    assert params["rules"]["A"]["mapOnVolumeTop"]["applied"]
    assert params["sweep"] == {"enabled": False}
    assert params["config"]["quantiles"] == {"A": 0.25, "B": 0.0}


def test_stage_fails_on_a_depth_from_another_run_section(
    make_ctx: Any, run_section: RunSection
) -> None:
    stage = importlib.import_module("hq.tier.run")
    other = run_section.model_copy(update={"refSurfaceElevM": REF + 10.0})
    ctx = make_ctx(other)
    write_run(ctx, flags=False)
    with pytest.raises(TierError, match="depthKm"):
        stage.run(ctx)


@pytest.mark.parametrize("first", [None, "hq.tier.run"])
def test_stage_resolves_to_the_stage_function(first: str | None) -> None:
    if first is not None:
        importlib.import_module(first)
    fn = resolve_stage(stage_spec("tier"))
    assert fn is importlib.import_module("hq.tier.run").run


# --- sweep ----------------------------------------------------------------------------------------


def test_sweep_counts_tier_a_with_the_configured_runs_bars(
    seismology_config: SeismologyConfig,
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
    main = derive_thresholds(events[events["id"].isin(matched_ids)], seismology_config.tiering)

    def assoc(n: int) -> AssocResult:
        return AssocResult(
            typed_frame({"assocId": [f"a{i}" for i in range(n)], "t": [T0] * n,
                         "latitude": [38.5] * n, "longitude": [-112.9] * n,
                         "elevM": [0.0] * n, "nPicks": [8] * n, "nP": [5] * n,
                         "nS": [3] * n}, EVENT_DTYPES),
            typed_frame(None, PICK_DTYPES),
        )

    def fake_locate(a: AssocResult) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        return events.iloc[: len(a.events)].reset_index(drop=True), None

    def fake_match(ev: pd.DataFrame) -> pd.DataFrame:  # only 3 matches: far below minMatched
        return matches_for(ev, list(ev["id"].iloc[:3]))

    grid = [{"minPickProb": 0.3, "nSPicks": 1, "minStations": s} for s in (4, 5, 6)]

    def run_points(evaluate: Any) -> list[SweepRow]:
        sizes = [len(events), 30, 0]
        results = [assoc(n) if n else empty_result() for n in sizes]
        return [SweepRow(params=p, candidates=len(r.events), counts={}, score=evaluate(r),
                         runtime_s=0.5) for p, r in zip(grid, results, strict=True)]

    points, record = score_sweep(run_points, SweepPipeline(fake_locate, fake_match),
                                 seismology_config, main)
    expected_a = [0 if n == 0 else int((assign_tiers(
        events.iloc[:n], matches_for(events.iloc[:n], []), seismology_config, thresholds=main
    ).events["tier"] == "A").sum()) for n in (len(events), 30, 0)]
    assert [p.candidates for p in points] == [len(events), 30, 0]
    assert [p.recoveredPublic for p in points] == [3, 3, 0]
    assert [p.tierA for p in points] == expected_a and expected_a[0] > 0
    assert [r["runtimeS"] for r in record] == [0.5, 0.5, 0.5]
    frame = to_frame(points, SweepPoint)
    assert from_frame(frame, SweepPoint) == points


def test_sweep_without_loc04_names_it() -> None:
    locate_pkg = importlib.import_module("hq.locate")
    if hasattr(locate_pkg, "locate_detailed"):
        pytest.skip("LOC-04 is merged: the real sweep driver is importable")
    from hq.tier.sweep import real_pipeline

    with pytest.raises(TierError, match="LOC-04"):
        real_pipeline(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None,  # type: ignore[arg-type]
                      None, run_id="x", cache_dir=Path("."))  # type: ignore[arg-type]


def test_stage_writes_sweep_parquet_when_enabled(
    make_ctx: Any, run_section: RunSection, seismology_config: SeismologyConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hq.associate.sweep import SweepRow, SweepScore

    stage = importlib.import_module("hq.tier.run")
    ctx = make_ctx(run_section, cfg_with(seismology_config, sweep={"enabled": True}))
    write_run(ctx)
    for name, model in (("stations.parquet", "Station"), ("catalog.parquet", "CatalogEvent")):
        write_table(pd.DataFrame({"id": ["x"]}), ctx.path(name), model)
    seen: list[Thresholds] = []

    def fake_score(run_points: Any, pipeline: Any, cfg: Any, thresholds: Thresholds) -> Any:
        seen.append(thresholds)
        row = SweepRow(params={"minPickProb": 0.3, "nSPicks": 1, "minStations": 5},
                       candidates=4, counts={}, score=SweepScore(3, 2, 1), runtime_s=1.0)
        return [SweepPoint(params=row.params, candidates=3, recoveredPublic=2, tierA=1)], [
            {"params": row.params, "tierA": 1}]

    monkeypatch.setattr("hq.tier.sweep.real_pipeline", lambda *a, **k: (None, None))
    monkeypatch.setattr("hq.tier.sweep.score_sweep", fake_score)
    stage.run(ctx)
    assert read_models(ctx.path("sweep.parquet"), SweepPoint)[0].tierA == 1
    (rec,) = ctx.records
    assert rec["counts"]["sweepPoints"] == 1 and rec["params"]["sweep"]["enabled"]
    assert seen[0].to_record() == rec["params"]["thresholds"]

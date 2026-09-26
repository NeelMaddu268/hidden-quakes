"""LOC-10: pick harvest at predicted arrivals (hq.locate.harvest) and its config section.

Offline. The smoke tests work on small tables built here; the others locate seven events on the
LOC-02 test geometry whose picks carry planted station delays and whose S pick at one western
station is left out of the association.
"""

import dataclasses
import importlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hq_contracts.io import read_table

from hq.config.seismology import PhaseSigma, SeismologyConfig
from hq.locate.harvest import (
    FREE_COLUMNS,
    SLOT_COLUMNS,
    UNTRUSTED,
    analytic_chance,
    free_picks,
    select,
    slot_table,
    untrusted_flags,
)
from hq.locate.statics import check_same_association

# --- config (smoke) ----------------------------------------------------------------------------


def _with_harvest(raw: dict[str, Any], **harvest: Any) -> dict[str, Any]:
    return {**raw, "harvest": {**raw["harvest"], **harvest}}


@pytest.mark.smoke
def test_showcase_config_ships_harvest_off(seismology_config: SeismologyConfig) -> None:
    h = seismology_config.harvest
    assert h.enabled is False
    assert h.windowS.P <= seismology_config.locator.outlier.floorS
    assert h.windowS.S <= seismology_config.locator.outlier.floorS
    assert h.minProb >= seismology_config.associator.minPickProb


@pytest.mark.smoke
def test_harvest_config_rejects_bad_values(seismology_config: SeismologyConfig) -> None:
    raw = seismology_config.model_dump(mode="json")
    floor = raw["locator"]["outlier"]["floorS"]
    with pytest.raises(ValueError, match="must not exceed locator.outlier.floorS"):
        SeismologyConfig.model_validate(_with_harvest(raw, windowS={"P": 0.1, "S": floor + 0.01}))
    with pytest.raises(ValueError, match="must not exceed locator.outlier.floorS"):
        SeismologyConfig.model_validate(_with_harvest(raw, windowS={"P": floor + 0.01, "S": 0.1}))
    below = raw["associator"]["minPickProb"] - 0.05
    with pytest.raises(ValueError, match="must not be below associator.minPickProb"):
        SeismologyConfig.model_validate(_with_harvest(raw, minProb=below))
    for bad in ({"iterations": 1}, {"windowS": {"P": 0.0, "S": 0.1}}, {"minProb": 1.5}):
        with pytest.raises(ValueError):
            SeismologyConfig.model_validate(_with_harvest(raw, **bad))
    missing = {k: v for k, v in raw.items() if k != "harvest"}
    with pytest.raises(ValueError, match="harvest"):
        SeismologyConfig.model_validate(missing)
    # The boundary itself is allowed.
    ok = SeismologyConfig.model_validate(_with_harvest(raw, windowS={"P": floor, "S": floor}))
    assert ok.harvest.windowS.S == floor


# --- selection rules on small tables (smoke) --------------------------------------------------

WINDOW = PhaseSigma(P=0.10, S=0.15)


def _slots(*rows: tuple[int, str, str, float, bool, bool]) -> pd.DataFrame:
    """(event, stationId, phase, tPred, filled, eligible) rows."""
    return pd.DataFrame(list(rows), columns=list(SLOT_COLUMNS))


def _free(*rows: tuple[str, str, str, float, float]) -> pd.DataFrame:
    """(id, stationId, phase, t, prob) rows."""
    return pd.DataFrame(list(rows), columns=list(FREE_COLUMNS))


def _picked(chosen: pd.DataFrame) -> dict[tuple[int, str, str], str]:
    return {(int(r.event), r.stationId, r.phase): r.pickId for r in chosen.itertuples()}


@pytest.mark.smoke
def test_window_is_per_phase_and_inclusive() -> None:
    slots = _slots((0, "A", "P", 100.0, False, True), (0, "B", "P", 100.0, False, True),
                   (0, "C", "P", 100.0, False, True), (0, "A", "S", 102.0, False, True),
                   (0, "B", "S", 102.0, False, True), (0, "C", "S", 102.0, False, True))
    free = _free(("pA", "A", "P", 100.0 + 0.10, 0.9),  # on the P boundary: in
                 ("pB", "B", "P", 100.0 - 0.10 - 1e-6, 0.9),  # just outside
                 ("pC", "C", "P", 100.0 + 0.12, 0.9),  # inside the S width, outside P's
                 ("sA", "A", "S", 102.0 - 0.15, 0.9),  # on the S boundary: in
                 ("sB", "B", "S", 102.0 + 0.15 + 1e-6, 0.9),  # just outside
                 ("sC", "C", "S", 102.0 + 0.12, 0.9))  # S width: in
    chosen, counts = select(slots, free, WINDOW)
    assert _picked(chosen) == {(0, "A", "P"): "pA", (0, "A", "S"): "sA", (0, "C", "S"): "sC"}
    got = chosen.set_index("pickId")
    assert got.loc["sC", "offsetS"] == pytest.approx(0.12)
    assert got.loc["sA", "tPred"] == 102.0
    assert counts == {"ambiguousPicks": 0, "skippedUntrustedSlots": 0,
                      "skippedMultiCandidateSlots": 0}


@pytest.mark.smoke
def test_free_picks_keep_unassociated_picks_above_min_prob_on_locator_stations() -> None:
    picks = pd.DataFrame({
        "id": ["a1", "f1", "f2", "f3", "f4", "f5"],
        "stationId": ["A", "A", "A", "Z", "A", "A"],
        "phase": ["S", "S", "P", "S", "S", "X"],
        "t": [10.0, 11.0, 9.0, 11.0, 12.0, 13.0],
        "prob": [0.9, 0.9, 0.29, 0.9, 0.3, 0.9],
        "picker": "phasenet:test",
    })
    free = free_picks(picks, {"a1"}, ["A", "B"], 0.3)
    # a1: associated; f2: below minProb; f3: not a locator station; f5: unknown phase.
    assert free["id"].tolist() == ["f1", "f4"]
    assert list(free.columns) == list(FREE_COLUMNS)


@pytest.mark.smoke
def test_never_steals_an_associated_pick_or_refills_a_slot() -> None:
    # The associated pick a1 sits right at event 1's empty slot; it is not free, so it is never
    # a candidate. Event 0's A S slot holds its own pick (filled; an outlier-dropped associated
    # pick counts the same), so the free pick next to it is not added.
    picks = pd.DataFrame({"id": ["a1", "f1"], "stationId": ["A", "A"], "phase": ["S", "S"],
                          "t": [50.0, 20.02], "prob": [0.9, 0.9]})
    free = free_picks(picks, {"a1"}, ["A"], 0.3)
    slots = _slots((0, "A", "S", 20.0, True, True), (1, "A", "S", 50.0, False, True))
    chosen, _ = select(slots, free, WINDOW)
    assert chosen.empty


@pytest.mark.smoke
def test_a_slot_with_two_candidates_is_skipped_and_counted() -> None:
    slots = _slots((0, "A", "S", 20.0, False, True), (0, "B", "S", 21.0, False, True))
    free = _free(("f1", "A", "S", 19.95, 0.9), ("f2", "A", "S", 20.05, 0.8),
                 ("f3", "B", "S", 21.01, 0.9))
    chosen, counts = select(slots, free, WINDOW)
    assert _picked(chosen) == {(0, "B", "S"): "f3"}
    assert counts["skippedMultiCandidateSlots"] == 1


@pytest.mark.smoke
def test_a_pick_near_two_events_is_ambiguous_even_when_one_slot_is_filled() -> None:
    # f1 lies within the S window of events 0 (empty slot) and 1 (filled slot): likely event 1's
    # arrival picked twice, so neither gets it. f2 lies within two empty slots: nobody gets it.
    slots = _slots((0, "A", "S", 20.00, False, True), (1, "A", "S", 20.20, True, True),
                   (2, "A", "S", 40.00, False, True), (3, "A", "S", 40.25, False, True))
    free = _free(("f1", "A", "S", 20.10, 0.9), ("f2", "A", "S", 40.12, 0.9))
    chosen, counts = select(slots, free, WINDOW)
    assert chosen.empty
    assert counts["ambiguousPicks"] == 2
    # Without event 1's row, f1 is event 0's alone.
    alone, _ = select(slots[slots["event"] != 1], free, WINDOW)
    assert _picked(alone) == {(0, "A", "S"): "f1"}


class _FakeLocator:
    station_ids = ("A", "B")

    @staticmethod
    def travel_times(e_m: float, n_m: float, elev_m: float) -> pd.DataFrame:
        return pd.DataFrame({"stationId": ["A", "A", "B", "B"], "phase": ["P", "S", "P", "S"],
                             "travelTimeS": [1.0, 2.0, 1.5, 3.0]})


def _loc(t0: float, **flags: bool) -> SimpleNamespace:
    return SimpleNamespace(e_m=0.0, n_m=0.0, elev_m=0.0, t0=t0,
                           map_on_volume_top=flags.get("mapOnVolumeTop", False),
                           map_on_volume_bottom=flags.get("mapOnVolumeBottom", False),
                           pdf_truncated=flags.get("pdfTruncated", False),
                           depth_on_edge=flags.get("depthOnEdge", False))


@pytest.mark.smoke
def test_slot_table_predicts_with_each_events_statics_and_marks_untrusted_events() -> None:
    frames = [pd.DataFrame({"id": ["a"], "stationId": ["A"], "phase": ["P"], "t": [101.0],
                            "prob": [0.9]}),
              pd.DataFrame({"id": ["b"], "stationId": ["B"], "phase": ["S"], "t": [203.0],
                            "prob": [0.9]})]
    slots = slot_table(_FakeLocator(), [_loc(100.0), _loc(200.0, pdfTruncated=True)], frames,
                       [{("A", "S"): 0.5}, {}])
    s = slots.set_index(["event", "stationId", "phase"])
    assert s.loc[(0, "A", "S"), "tPred"] == 100.0 + 2.0 + 0.5
    assert s.loc[(1, "A", "S"), "tPred"] == 200.0 + 2.0
    assert bool(s.loc[(0, "A", "P"), "filled"]) and not bool(s.loc[(0, "A", "S"), "filled"])
    assert s.loc[0, "eligible"].all() and not s.loc[1, "eligible"].any()
    for flag in UNTRUSTED:
        assert untrusted_flags(_loc(0.0, **{flag: True})) == [flag]  # type: ignore[arg-type]
    assert untrusted_flags(_loc(0.0)) == []  # type: ignore[arg-type]


@pytest.mark.smoke
def test_untrusted_events_are_skipped_but_their_windows_still_count() -> None:
    slots = _slots((0, "A", "S", 20.00, False, False),  # untrusted, empty: skipped and counted
                   (1, "A", "S", 60.00, False, True), (2, "A", "S", 60.10, False, False))
    free = _free(("f1", "A", "S", 20.01, 0.9), ("f2", "A", "S", 60.05, 0.9))
    chosen, counts = select(slots, free, WINDOW)
    assert chosen.empty  # f2 is ambiguous: untrusted event 2's window counts
    assert counts["skippedUntrustedSlots"] == 1 and counts["ambiguousPicks"] == 1


@pytest.mark.smoke
def test_decoys_are_rejected_and_the_true_pick_found_in_any_input_order() -> None:
    rng = np.random.default_rng(10)
    slots = _slots(
        (0, "A", "S", 100.0, False, True),  # the true pick f_true
        (0, "A", "P", 99.0, False, True),  # decoy: a P-labelled pick sits at S tPred
        (1, "B", "S", 300.0, False, True), (2, "B", "S", 300.1, False, True),  # share f_amb
        (3, "C", "S", 500.0, False, False),  # a face event: its pick stays
        (4, "D", "S", 700.0, False, True),  # chance picks far from tPred only
    )
    free = _free(("f_true", "A", "S", 100.03, 0.8), ("f_lbl", "A", "P", 100.0, 0.9),
                 ("f_amb", "B", "S", 300.05, 0.9), ("f_face", "C", "S", 500.01, 0.9),
                 *[(f"n{k}", "D", "S", float(t), 0.5)
                   for k, t in enumerate(rng.uniform(701.0, 900.0, 20))])
    chosen, counts = select(slots, free, WINDOW)
    assert _picked(chosen) == {(0, "A", "S"): "f_true"}
    assert counts == {"ambiguousPicks": 1, "skippedUntrustedSlots": 1,
                      "skippedMultiCandidateSlots": 0}
    for seed in range(3):
        shuffled, again = select(slots.sample(frac=1.0, random_state=seed),
                                 free.sample(frac=1.0, random_state=seed + 7), WINDOW)
        pd.testing.assert_frame_equal(shuffled, chosen)
        assert again == counts


@pytest.mark.smoke
def test_analytic_chance_counts_open_slots_of_trusted_events_only() -> None:
    slots = _slots((0, "A", "S", 10.0, False, True), (1, "A", "S", 50.0, True, True),
                   (2, "A", "S", 90.0, False, False))
    free = _free(*[(f"f{k}", "A", "S", float(k), 0.9) for k in range(10)])
    # rate 10 picks / 100 s, one open trusted slot, window 2 x 0.15 s.
    assert analytic_chance(slots, free, WINDOW, 100.0) == pytest.approx(0.1 * 0.3)


@pytest.mark.smoke
def test_check_same_association_ignores_harvested_picks_but_not_a_regrouping() -> None:
    pairs = pd.DataFrame({"catalogId": ["c1", "c2"], "assocId": ["x0", "x1"],
                          "pickIds": [["p0", "p1", "h1"], ["p2", "h2"]]})  # h*: harvested
    links = pd.DataFrame({"assocId": ["x0", "x0", "x1"], "pickId": ["p0", "p1", "p2"]})
    check_same_association(pairs, links)
    with pytest.raises(ValueError, match="another association"):
        check_same_association(pairs, links.assign(assocId=["x0", "x1", "x1"]))  # p1 moved


# --- locating the LOC-02 test geometry with withheld picks (not smoke) --------------------------

SEED = 20260926
NOISE = {"P": 0.004, "S": 0.008}
# (e, n, elevM) of the reference events and one unmatched event (the last), as in test_statics.
HYPOS = ((300.0, -200.0, -2000.0), (-400.0, 500.0, -2600.0), (100.0, 700.0, -1800.0),
         (600.0, 300.0, -3000.0), (-200.0, -600.0, -2300.0), (0.0, 0.0, -2500.0),
         (-700.0, 100.0, -2100.0))


def _delay(e_m: float, phase: str) -> float:
    """Planted station delay (s): west of the origin late, east early; S twice P."""
    base = 0.12 if e_m < -1000.0 else -0.08 if e_m > 1000.0 else 0.0
    return base * (2.0 if phase == "S" else 1.0)


@pytest.fixture(scope="module")
def hworld(loc02: Any) -> dict[str, Any]:
    """Seven events on the LOC-02 geometry with planted station delays. Each event's S pick at
    one western station (delay +0.24 s, far outside the window without statics) is left out of
    the association but stays in the picks table; decoys: chance picks between events, a
    P-labelled pick at a withheld S arrival and a sub-threshold pick next to another."""
    from hq.locate.coords import from_enu
    from hq.locate.locator import build_locator

    run = loc02.run_section()
    raw = loc02.test_config().model_dump(mode="json")
    raw["diagnostics"]["datumCheck"]["elevM"] = [-2000.0]
    raw["synthetic"] = {**raw["synthetic"], "nEvents": 2, "sKeepProb": 0.8, "pickProb": 0.8}
    raw["statics"]["wellConstrained"]["minStations"] = 8
    off = SeismologyConfig.model_validate(raw)
    on = SeismologyConfig.model_validate(_with_harvest(raw, enabled=True))
    locator = build_locator(loc02.setup(off, run))
    rng = np.random.default_rng(SEED)
    east = {sid: e for sid, e, _, _, _ in loc02.STATIONS}
    west = sorted(sid for sid, e in east.items() if e < -1000.0)
    frames, rows, links, withheld, decoys = [], [], [], [], []
    for k, (e, n, z) in enumerate(HYPOS):
        t0 = run.window_start_s + 600.0 * (k + 1)
        p = loc02.exact_picks(locator, e, n, z, t0, prob=0.8)
        p["t"] = (p["t"] + [_delay(east[s], ph) for s, ph in zip(p["stationId"], p["phase"],
                                                                strict=True)]
                  + rng.normal(0.0, 1.0, len(p)) * p["phase"].map(NOISE))
        p["id"] = [f"phasenet:{s}:{ph}:{t:.3f}" for s, ph, t in
                   zip(p["stationId"], p["phase"], p["t"], strict=True)]
        keep = ~((p["stationId"] == west[k % len(west)]) & (p["phase"] == "S"))
        withheld += p.loc[~keep, "id"].tolist()
        aid = f"assoc-{k:06d}"
        lat, lon, _ = from_enu(e, n, z - run.origin.elevM, run.origin)
        rows.append({"assocId": aid, "t": t0, "latitude": float(lat), "longitude": float(lon),
                     "elevM": z, "nPicks": int(keep.sum()),
                     "nP": int((p.loc[keep, "phase"] == "P").sum()),
                     "nS": int((p.loc[keep, "phase"] == "S").sum())})
        links += [{"assocId": aid, "pickId": i} for i in p.loc[keep, "id"]]
        frames.append(p)
        s_true = p.loc[~keep].iloc[0]
        extra = [(f"decoy:label:{k}", s_true["stationId"], "P", s_true["t"] + 0.01, 0.9)]
        if k == 1:  # a pick below minProb 0.03 s from the true one: filtered, not a 2nd candidate
            extra.append((f"decoy:weak:{k}", s_true["stationId"], "S", s_true["t"] + 0.03, 0.2))
        extra += [(f"decoy:chance:{k}:{j}", sid, ph, t0 + float(rng.uniform(60.0, 500.0)), 0.9)
                  for j, (sid, ph) in enumerate([(west[0], "S"), (west[1], "S"), ("T.S01", "P")])]
        decoys += [d[0] for d in extra]
        frames.append(pd.DataFrame(extra, columns=list(FREE_COLUMNS)))
    picks = pd.concat(frames, ignore_index=True).assign(picker="phasenet:test", eventId=None)[
        ["id", "stationId", "phase", "t", "prob", "picker", "eventId"]]
    from hq.associate.result import EVENT_DTYPES, PICK_DTYPES, AssocResult, typed_frame

    assoc = AssocResult(
        typed_frame({c: [r[c] for r in rows] for c in EVENT_DTYPES}, EVENT_DTYPES),
        typed_frame({c: [r[c] for r in links] for c in PICK_DTYPES}, PICK_DTYPES),
    )
    st = loc02.stations(run.origin.elevM)
    lat, lon, _ = from_enu(st["enu_e"], st["enu_n"], st["enu_u"], run.origin)
    st = st.assign(latitude=lat, longitude=lon, usedInRun=True)
    planted = {(sid, ph): _delay(east[sid], ph) for sid in east for ph in ("P", "S")}
    return {"run": run, "off": off, "on": on, "picks": picks, "assoc": assoc, "stations": st,
            "planted": planted, "withheld": sorted(withheld), "decoys": decoys,
            "cache": loc02.cache_dir, "rows": rows}


def _call(w: dict[str, Any], cfg: SeismologyConfig, **kw: Any) -> Any:
    from hq.locate import locate_detailed

    return locate_detailed(w["assoc"], w["picks"], w["stations"], cfg, w["run"], run_id="t",
                           cache_dir=w["cache"], **kw)


@pytest.fixture(scope="module")
def harvested(hworld: dict[str, Any]) -> Any:
    return _call(hworld, hworld["on"], statics=hworld["planted"], harvest=True)


def _harvested_ids(events: pd.DataFrame, assoc: Any) -> set[str]:
    return {str(p) for ids in events["pickIds"] for p in ids} - set(
        assoc.picks["pickId"].astype(str))


def test_harvest_recovers_the_withheld_picks_with_statics(
    hworld: dict[str, Any], harvested: Any
) -> None:
    rep = harvested.harvest
    assert sorted(rep.added["pickId"]) == hworld["withheld"]  # every one, and nothing else
    assert rep.added["usedInLocation"].all()
    assert rep.added["offsetS"].abs().max() < 0.05
    assert rep.counts["events"] == len(HYPOS) and len(rep.before) == len(HYPOS)
    res = harvested.result
    assert len(res.events) == len(HYPOS)
    assert _harvested_ids(res.events, hworld["assoc"]) == set(hworld["withheld"])
    arr = res.arrivals.set_index("pickId", drop=False)
    assert arr.loc[hworld["withheld"], "usedInLocation"].all()
    assert arr.loc[hworld["withheld"], "residualS"].abs().max() < 0.05
    assert not set(hworld["decoys"]) & set(res.arrivals["pickId"].dropna())
    c = harvested.counts
    assert c["picksIn"] == len(hworld["assoc"].picks)  # the associated picks only
    assert c["picksHarvested"] == len(hworld["withheld"]) == c["picksHarvestedUsed"]
    assert c["picksUsed"] + c["picksDroppedAsOutliers"] == c["picksIn"] + c["picksHarvested"]
    assert (res.events["quality_nS"] == 12).all()  # every S back
    record = harvested.record["harvest"]
    json.dumps(record)
    assert {p["pickId"] for p in record["picks"]} == set(hworld["withheld"])
    assert set(record["counts"]) >= {"picks", "ambiguousPicks", "skippedUntrustedSlots"}
    assert record["chance"]["controlShiftsS"] == [-1.0, -0.6, 0.6, 1.0]
    # Each event's pre-harvest location lacked the pick: one S fewer.
    assert all(loc.n_s == 11 for loc in rep.before.values())


def test_harvest_does_not_depend_on_the_worker_count(
    hworld: dict[str, Any], harvested: Any
) -> None:
    cfg = hworld["on"]
    two = cfg.model_copy(update={"locator": cfg.locator.model_copy(update={"nWorkers": 2})})
    parallel = _call(hworld, two, statics=hworld["planted"], harvest=True)
    for name in ("events", "arrivals", "statics"):
        pd.testing.assert_frame_equal(getattr(parallel.result, name),
                                      getattr(harvested.result, name))
    pd.testing.assert_frame_equal(parallel.harvest.added, harvested.harvest.added)


def test_harvest_off_changes_nothing(hworld: dict[str, Any]) -> None:
    kw = {"statics": hworld["planted"]}
    flag_off = _call(hworld, hworld["off"], harvest=True, **kw)  # asked, but not enabled
    not_asked = _call(hworld, hworld["on"], harvest=False, **kw)  # enabled, but not asked
    assert flag_off.harvest is None and not_asked.harvest is None
    for name in ("events", "arrivals", "statics"):
        pd.testing.assert_frame_equal(getattr(flag_off.result, name),
                                      getattr(not_asked.result, name))
    pd.testing.assert_frame_equal(flag_off.flags, not_asked.flags)
    assert flag_off.counts == not_asked.counts and "picksHarvested" not in flag_off.counts
    # "tables" holds build/load counts and build time, which differ between calls anyway.
    def same(rec: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in rec.items() if k != "tables"}

    assert same(flag_off.record) == same(not_asked.record) and "harvest" not in flag_off.record
    assert not _harvested_ids(flag_off.result.events, hworld["assoc"])


def test_locate_api_harvests_only_with_a_statics_table(
    hworld: dict[str, Any], harvested: Any
) -> None:
    from hq.locate import locate

    table = pd.DataFrame([{"stationId": s, "phase": p, "staticS": v}
                          for (s, p), v in hworld["planted"].items()])
    args = (hworld["assoc"], hworld["picks"], hworld["stations"], hworld["on"], hworld["run"])
    res = locate(*args, run_id="t", cache_dir=hworld["cache"], statics=table)
    pd.testing.assert_frame_equal(res.events, harvested.result.events)
    # referenceEvents without a match pass: pass 1, no statics, never a harvest.
    plain = locate(*args, run_id="t", cache_dir=hworld["cache"])
    assert not _harvested_ids(plain.events, hworld["assoc"])


def test_self_consistent_harvests_on_the_last_iteration_only(
    hworld: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import hq.locate.harvest as harvest_mod
    from hq.locate.statics import locate_with_statics

    calls: list[int] = []
    real = harvest_mod.harvest_and_relocate

    def counting(*a: Any, **k: Any) -> Any:
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(harvest_mod, "harvest_and_relocate", counting)
    outs = {}
    for label in ("off", "on"):
        raw = hworld[label].model_dump(mode="json")
        raw["statics"] = {**raw["statics"], "mode": "selfConsistent", "minEvents": 3,
                          "iterations": 2}
        cfg = SeismologyConfig.model_validate(raw)
        outs[label] = locate_with_statics(hworld["assoc"], hworld["picks"], hworld["stations"],
                                          cfg, hworld["run"], cache_dir=hworld["cache"])
    assert len(calls) == 1  # the "on" run's last iteration
    on, off = outs["on"], outs["off"]
    pd.testing.assert_frame_equal(on.report.terms, off.report.terms)
    pd.testing.assert_frame_equal(on.details.result.statics, off.details.result.statics)
    assert on.details.harvest is not None and off.details.harvest is None


def test_stage_pass_2_harvests_and_keeps_the_terms(
    hworld: dict[str, Any], make_ctx: Any, tmp_path: Any
) -> None:
    import shutil

    from hq_contracts.io import to_frame, write_table
    from hq_contracts.models import CatalogEvent

    from hq.locate.coords import from_enu
    from hq.match import match

    stage = importlib.import_module("hq.locate.run")
    run = hworld["run"]
    cat = []
    for k, (e, n, z) in enumerate(HYPOS[:-1]):
        lat, lon, _ = from_enu(e, n, z - run.origin.elevM, run.origin)
        cat.append(CatalogEvent(
            id=f"cat{k}", source="test", t=hworld["rows"][k]["t"], latitude=float(lat),
            longitude=float(lon), depthKm=-z / 1000.0, depthDatum="test: km below sea level",
            elevM=z, mag=1.0, magType="ml", enu={"e": e, "n": n, "u": z - run.origin.elevM}))
    catalog = to_frame(cat, CatalogEvent)
    off = dataclasses.replace(make_ctx(run, hworld["off"]), cache_dir=hworld["cache"])
    write_table(hworld["picks"], off.path(hworld["off"].associator.picksTable), "Pick")
    write_table(hworld["stations"], off.path("stations.parquet"), "Station")
    write_table(hworld["assoc"].events, off.path("assoc_events.parquet"), "AssocEvent")
    write_table(hworld["assoc"].picks, off.path("assoc_picks.parquet"), "AssocPick")
    write_table(catalog, off.path("catalog.parquet"), "CatalogEvent")
    stage.run(off)  # pass 1 (the same with harvest on: it never harvests)
    first = read_table(off.path("events_located.parquet"))
    write_table(match(first, catalog, hworld["off"]).matches, off.path("matches.parquet"),
                "Match")
    on_dir = tmp_path / "on"
    shutil.copytree(off.run_dir, on_dir)
    on = dataclasses.replace(off, run_dir=on_dir, records=[],
                             config=dataclasses.replace(off.config, seismology=hworld["on"]))
    stage.run(off)  # pass 2, harvest off
    stage.run(on)  # pass 2, harvest on

    def raw(ctx: Any, name: str) -> bytes:
        return ctx.path(name).read_bytes()

    assert raw(on, "statics.parquet") == raw(off, "statics.parquet")
    assert raw(on, "synthetic.json") == raw(off, "synthetic.json")  # fixed sKeepProb/pickProb
    ev_on, ev_off = (read_table(c.path("events_located.parquet")) for c in (on, off))
    assert _harvested_ids(ev_on, hworld["assoc"]) == set(hworld["withheld"])
    assert not _harvested_ids(ev_off, hworld["assoc"])
    params_on, params_off = on.records[-1]["params"], off.records[-1]["params"]
    assert "harvest" in params_on and "harvest" not in params_off
    assert on.records[-1]["counts"]["picksHarvested"] == len(hworld["withheld"])
    assert "picksHarvested" not in off.records[-1]["counts"]
    ref = params_on["statics"]["reference"]
    assert all("afterNoHarvestHM" in r for r in ref)
    assert not any("afterNoHarvestHM" in r for r in params_off["statics"]["reference"])
    assert set(params_on["statics"]["crossValidatedOffsets"]) == {
        "before", "after", "afterNoHarvest", "inSample"}
    report_on, report_off = (c.path("diagnostics.md").read_text() for c in (on, off))
    assert "## Pick harvest (LOC-10)" in report_on and "Pick harvest" not in report_off
    n = len(hworld["withheld"])
    assert f"{n} of {n} harvested picks used" in report_on
    assert "held-out terms, before the harvest" in report_on
    assert "not independent evidence" in report_on and "Chance:" in report_on
    # A pass 2 rerun on the harvested events_located (after match reruns) must not call the
    # association stale: the harvested pick ids belong to no association event.
    write_table(match(ev_on, catalog, hworld["on"]).matches, on.path("matches.parquet"), "Match")
    stage.run(on)
    pd.testing.assert_frame_equal(read_table(on.path("events_located.parquet")), ev_on)

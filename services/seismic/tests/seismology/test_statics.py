"""LOC-05: station statics (hq.locate.statics), the stage's pass 2 and the diagnostics section.

Smoke tests work on small tables built here (planted station terms, seeded noise). The heavier
tests (not smoke) locate synthetic events on the LOC-02 test geometry, whose picks carry planted
station delays: western stations late and eastern stations early, as the showcase day's lateral
structure is. Offline and seeded.
"""

import dataclasses
import importlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hq_contracts.io import read_table, to_frame, write_table
from hq_contracts.models import CatalogEvent

from hq.associate.result import EVENT_DTYPES as ASSOC_EVENT_DTYPES
from hq.associate.result import PICK_DTYPES as ASSOC_PICK_DTYPES
from hq.associate.result import AssocResult, typed_frame
from hq.config.seismology import SeismologyConfig
from hq.locate import locate
from hq.locate.coords import from_enu
from hq.locate.statics import (
    check_same_association,
    explain_terms,
    fold_of,
    held_out_terms,
    locate_with_statics,
    polish_terms,
    reference_pairs,
    residual_sigma,
    self_consistent_terms,
)
from hq.tier import TierError

SEED = 20260926
NOISE = {"P": 0.004, "S": 0.008}
POLISH = {"iterations": 3, "cap_s": 1.0, "min_events": 3}


# --- small tables (smoke) --------------------------------------------------------------------


def _residuals(terms: dict[tuple[str, str], float], n_events: int, *, drop: dict | None = None,
               seed: int = SEED) -> pd.DataFrame:
    """d = t0_k + term + noise for every event k and station-phase (``drop``: pair -> events
    that lack it)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_events):
        t0 = rng.uniform(-1.0, 1.0)
        for (sid, ph), term in terms.items():
            if drop and k in drop.get((sid, ph), ()):
                continue
            rows.append({"assocId": f"a{k:02d}", "stationId": sid, "phase": ph,
                         "d": t0 + term + rng.normal(0.0, 0.003), "w": rng.uniform(10.0, 40.0)})
    return pd.DataFrame(rows)


PLANTED = {("A", "P"): 0.20, ("A", "S"): 0.45, ("B", "P"): -0.10, ("B", "S"): -0.25,
           ("C", "P"): 0.0, ("C", "S"): 0.05, ("D", "P"): 0.05, ("D", "S"): 0.10,
           ("E", "P"): -0.02, ("E", "S"): 1.60}


@pytest.mark.smoke
def test_polish_recovers_planted_terms_up_to_one_constant() -> None:
    res = _residuals({k: v for k, v in PLANTED.items() if k != ("E", "S")}, 12)
    terms = polish_terms(res, **POLISH).set_index(["stationId", "phase"])
    diff = np.array([terms.loc[k, "staticS"] - v for k, v in PLANTED.items() if k != ("E", "S")])
    # The origin time absorbs one constant shared by every station-phase; nothing else.
    assert np.ptp(diff) < 0.01
    assert (terms["nEvents"] == 12).all()
    assert (terms["madS"] < 0.01).all()


@pytest.mark.smoke
def test_polish_caps_and_zeroes_sparse_station_phases() -> None:
    res = _residuals(PLANTED, 8, drop={("D", "S"): set(range(6))})  # D S in 2 of 8 events
    terms = polish_terms(res, **POLISH).set_index(["stationId", "phase"])
    assert terms.loc[("E", "S"), "rawS"] > 1.0
    assert terms.loc[("E", "S"), "staticS"] == pytest.approx(1.0)  # capped at cap_s
    assert terms.loc[("D", "S"), "nEvents"] == 2
    assert terms.loc[("D", "S"), "staticS"] == 0.0  # below min_events
    assert terms.loc[("D", "S"), "rawS"] != 0.0  # the estimate is kept for the report


@pytest.mark.smoke
def test_polish_on_no_residuals_is_an_empty_table() -> None:
    assert polish_terms(pd.DataFrame(), **POLISH).empty


@pytest.mark.smoke
def test_folds_leave_one_out_and_round_robin() -> None:
    pairs = pd.DataFrame({"assocId": [f"a{k}" for k in range(5)]})
    assert fold_of(pairs, None).tolist() == [0, 1, 2, 3, 4]
    assert fold_of(pairs, 2).tolist() == [0, 1, 0, 1, 0]


@pytest.mark.smoke
def test_held_out_terms_never_use_the_held_out_event() -> None:
    # Station X is picked only in event a00: a00's own terms must not contain it.
    res = _residuals({("A", "P"): 0.1, ("B", "P"): -0.1, ("C", "S"): 0.2}, 4)
    extra = pd.DataFrame([{"assocId": "a00", "stationId": "X", "phase": "S", "d": 5.0,
                           "w": 20.0}])
    res = pd.concat([res, extra], ignore_index=True)
    folds = pd.Series(range(4), index=[f"a{k:02d}" for k in range(4)])
    held = held_out_terms(res, folds, iterations=2, cap_s=9.0, min_events=1)
    assert ("X", "S") not in held["a00"]
    assert held["a01"][("X", "S")] != 0.0  # the other events do use a00's residual
    # Two folds: a00 and a02 share one map, computed from a01 and a03 alone.
    two = held_out_terms(res, fold_of(pd.DataFrame({"assocId": folds.index}), 2), iterations=2,
                         cap_s=9.0, min_events=1)
    assert two["a00"] is two["a02"] and ("X", "S") not in two["a00"]
    only = polish_terms(res[res["assocId"].isin(["a01", "a03"])], iterations=2, cap_s=9.0,
                        min_events=1)
    assert two["a00"] == {(s, p): v for s, p, v in
                          zip(only["stationId"], only["phase"], only["staticS"], strict=True)}


def _match_tables(run: Any) -> tuple[pd.DataFrame, ...]:
    lat, lon, _ = from_enu([100.0, -300.0, 0.0], [50.0, 400.0, 0.0], [0.0, 0.0, 0.0], run.origin)
    catalog = pd.DataFrame({"id": ["c1", "c2", "c3"], "t": [100.0, 200.0, 300.0],
                            "latitude": lat, "longitude": lon,
                            "elevM": [-2000.0, -2500.0, -3000.0]})
    events = pd.DataFrame({"id": ["e0", "e1", "e2"], "t": [100.3, 199.8, 500.0],
                           "pickIds": [["p0", "p1"], ["p2"], ["p3"]]})
    flags = pd.DataFrame({"eventId": ["e0", "e1", "e2"], "assocId": ["x0", "x1", "x2"]})
    matches = pd.DataFrame({"catalogId": ["c1", "c2", "c3"], "eventId": ["e0", "e1", None],
                            "dtS": [100.3 - 100.0, 199.8 - 200.0, np.nan],
                            "distM": [1.0, 1.0, np.nan], "reason": [None, None, "no candidate"]})
    return matches, events, flags, catalog


@pytest.mark.smoke
def test_reference_pairs_map_matches_and_refuse_stale_ones(run_section: Any) -> None:
    matches, events, flags, catalog = _match_tables(run_section)
    pairs = reference_pairs(matches, events, flags, catalog, run_section)
    assert pairs["catalogId"].tolist() == ["c1", "c2"]  # matched only, by catalog time
    assert pairs["assocId"].tolist() == ["x0", "x1"]
    assert pairs["catalogE"].tolist() == pytest.approx([100.0, -300.0], abs=1e-6)
    assert pairs["catalogN"].tolist() == pytest.approx([50.0, 400.0], abs=1e-6)
    assert pairs["catalogElevM"].tolist() == [-2000.0, -2500.0]
    links = pd.DataFrame({"assocId": ["x0", "x0", "x0", "x1"], "pickId": ["p0", "p1", "p9", "p2"]})
    check_same_association(pairs, links)  # every located pick belongs to its association event
    with pytest.raises(ValueError, match="another association"):
        check_same_association(pairs, links.assign(assocId=["x0", "x1", "x0", "x1"]))
    moved = events.assign(t=events["t"] + 0.01)  # relocated since the match: stale
    with pytest.raises(ValueError, match="stale"):
        reference_pairs(matches, moved, flags, catalog, run_section)
    with pytest.raises(ValueError, match="rerun stage match"):
        reference_pairs(matches, events.iloc[1:], flags, catalog, run_section)


def _fake_details(n_events: int = 6) -> SimpleNamespace:
    events = pd.DataFrame({
        "id": [f"e{k}" for k in range(n_events)],
        "quality_nStations": [12] * (n_events - 1) + [4],  # the last is not well constrained
        "quality_nS": [5] * n_events, "quality_gapDeg": [90.0] * n_events,
        "quality_depthOnEdge": [False] * n_events, "quality_hErrM": [100.0] * n_events,
    })
    rows = []
    for k in range(n_events):
        for sid, ph, r in (("A", "P", 0.10), ("A", "S", 0.50), ("B", "P", -0.05)):
            rows.append({"eventId": f"e{k}", "stationId": sid, "phase": ph,
                         "residualS": r if k < n_events - 1 else 9.0, "usedInLocation": True})
    return SimpleNamespace(result=SimpleNamespace(events=events, arrivals=pd.DataFrame(rows)))


@pytest.mark.smoke
def test_self_consistent_terms_add_the_median_residual_of_well_constrained_events(
    seismology_config: SeismologyConfig,
) -> None:
    cfg = seismology_config.model_copy(update={"statics": seismology_config.statics.model_copy(
        update={"mode": "selfConsistent", "capS": 0.3, "minEvents": 5})})
    terms = self_consistent_terms(_fake_details(), {("B", "P"): 0.02}, cfg)
    t = terms.set_index(["stationId", "phase"])
    assert t.loc[("A", "P"), "staticS"] == pytest.approx(0.10)  # the 9.0 s event is left out
    assert t.loc[("A", "S"), "staticS"] == pytest.approx(0.30)  # capped at capS
    assert t.loc[("B", "P"), "staticS"] == pytest.approx(-0.03)  # current static + residual
    assert (t["nEvents"] == 5).all()
    few = self_consistent_terms(_fake_details(4), {}, cfg)  # 3 well-constrained < minEvents 5
    assert (few["staticS"] == 0.0).all()


@pytest.mark.smoke
def test_explain_terms_gives_every_flagged_term_a_verdict(
    loc02: Any, seismology_config: SeismologyConfig
) -> None:
    # Vp/Vs 2.0 in the top layer, where every sensor sits.
    model = loc02.toy_model([2500.0, 0.0], [2000.0, 5500.0], [1000.0, 3200.0])
    st = pd.DataFrame({
        "id": ["W1", "W2", "W3", "C1", "C2", "C3", "T1", "F1", "U1"],
        "enu_e": [-5000.0, -5500.0, -4800.0, 500.0, 0.0, -300.0, 900.0, 20000.0, 300.0],
        "enu_n": [0.0, 800.0, -700.0, 0.0, 600.0, -500.0, -900.0, 0.0, 300.0],
        "sensorElevM": [1500.0] * 9,
    })
    rows = [("W1", 0.30, 0.90), ("W2", 0.25, 0.80), ("W3", 0.28, 0.85),  # lateral (shared)
            ("C1", 0.20, 0.40), ("C2", 0.0, 0.0), ("C3", 0.0, 0.0),  # C1: path (S/P = Vp/Vs)
            ("T1", 0.18, 0.18),  # equal delays: timing
            ("F1", -0.30, 0.0),  # early, far away: far
            ("U1", -0.20, 0.30)]  # opposite signs, not shared: unexplained
    terms = pd.DataFrame([{"stationId": s, "phase": ph, "staticS": v, "rawS": v, "nEvents": 10,
                           "madS": 0.01} for s, p, sv in rows for ph, v in (("P", p), ("S", sv))])
    ex = explain_terms(terms, st, model, (0.0, 0.0), 0.15, 3, seismology_config.statics.explain)
    verdict = dict(zip(zip(ex["stationId"], ex["phase"]), ex["verdict"], strict=True))
    assert verdict[("W1", "S")] == "lateral" and verdict[("W2", "P")] == "lateral"
    assert verdict[("C1", "P")] == "path" and verdict[("C1", "S")] == "path"
    assert verdict[("T1", "P")] == "timing"
    assert verdict[("F1", "P")] == "far"
    assert verdict[("U1", "P")] == "unexplained" and verdict[("U1", "S")] == "unexplained"
    assert ("C2", "P") not in verdict  # below the flag: no row
    assert ex["explanation"].str.len().gt(0).all()
    assert "unexplained static" in ex.set_index(["stationId", "phase"]).loc[("U1", "S"),
                                                                             "explanation"]
    assert "consistent with lateral structure" in ex.set_index(["stationId", "phase"]).loc[
        ("W1", "S"), "explanation"]
    # S-dominated: near-station rock whose Vp/Vs differs from the model's; it moves P too.
    vp = terms.assign(staticS=np.where((terms["stationId"] == "C1") & (terms["phase"] == "S"),
                                       0.9, terms["staticS"]))
    ex2 = explain_terms(vp, st, model, (0.0, 0.0), 0.15, 3, seismology_config.statics.explain)
    rows2 = ex2.set_index(["stationId", "phase"])
    assert rows2.loc[("C1", "S"), "verdict"] == "vpvs" == rows2.loc[("C1", "P"), "verdict"]
    assert "higher Vp/Vs than the model's" in rows2.loc[("C1", "S"), "explanation"]
    assert "delays P and delays S more" in rows2.loc[("C1", "P"), "explanation"]
    # A late term at another distant station contradicts the far-station hypothesis.
    far2 = pd.concat([st, pd.DataFrame({"id": ["F2"], "enu_e": [-20000.0], "enu_n": [0.0],
                                        "sensorElevM": [1500.0]})], ignore_index=True)
    late = pd.concat([terms, pd.DataFrame([
        {"stationId": "F2", "phase": ph, "staticS": v, "rawS": v, "nEvents": 10, "madS": 0.01}
        for ph, v in (("P", 0.40), ("S", 0.0))])], ignore_index=True)
    ex3 = explain_terms(late, far2, model, (0.0, 0.0), 0.15, 3,
                        seismology_config.statics.explain).set_index(["stationId", "phase"])
    assert ex3.loc[("F1", "P"), "verdict"] == "unexplained"
    assert ex3.loc[("F1", "P"), "farContradictedBy"] == "F2"
    assert "contradicts that: F2 at 20.0 km" in ex3.loc[("F1", "P"), "explanation"]
    # Neighbours beyond neighbourMaxDistM don't count.
    tight = seismology_config.statics.explain.model_copy(update={"neighbourMaxDistM": 500.0})
    ex4 = explain_terms(terms, st, model, (0.0, 0.0), 0.15, 3, tight).set_index(
        ["stationId", "phase"])
    assert ex4.loc[("W1", "S"), "verdict"] != "lateral"
    assert "No other station with a S term lies within 0.5 km" in ex4.loc[("W1", "S"),
                                                                          "explanation"]


@pytest.mark.smoke
def test_residual_sigma_flags_a_spread_well_above_the_configured_sigma(
    seismology_config: SeismologyConfig,
) -> None:
    rng = np.random.default_rng(SEED)
    n = 400
    arr = pd.DataFrame({
        "eventId": np.repeat(["e0", "e1"], n), "phase": np.tile(["P", "S"], n),
        "residualS": np.where(np.tile([True, False], n), rng.normal(0, 0.05, 2 * n),
                              rng.normal(0, 0.03, 2 * n)),
        "usedInLocation": True,
    })
    sig = residual_sigma(arr, ["e0", "e1"], seismology_config, "test").set_index("phase")
    sigma_p = seismology_config.locator.pickSigmaS.P
    assert sig.loc["P", "robustSigmaS"] == pytest.approx(0.05, rel=0.15)
    assert bool(sig.loc["P", "wellAbove"]) is (0.05 / sigma_p > 1.5)
    assert not bool(sig.loc["S", "wellAbove"])
    assert sig.loc["P", "recommendedS"] == round(sig.loc["P", "robustSigmaS"], 3)


@pytest.mark.smoke
def test_statics_config_rejects_bad_values(seismology_config: SeismologyConfig) -> None:
    raw = seismology_config.model_dump(mode="json")
    for key, bad in (("mode", "both"), ("folds", 1), ("referenceCapS", 0.0)):
        broken = {**raw, "statics": {**raw["statics"], key: bad}}
        with pytest.raises(ValueError):
            SeismologyConfig.model_validate(broken)


# --- locating with planted station delays (not smoke) -------------------------------------------

# Planted station-phase delays (s): stations west of the origin late, east early.
def _delay(e_m: float, phase: str) -> float:
    base = 0.12 if e_m < -1000.0 else -0.08 if e_m > 1000.0 else 0.0
    return base * (2.0 if phase == "S" else 1.0)


# (e, n, elevM) of the reference events and one unmatched event (the last)
HYPOS = ((300.0, -200.0, -2000.0), (-400.0, 500.0, -2600.0), (100.0, 700.0, -1800.0),
         (600.0, 300.0, -3000.0), (-200.0, -600.0, -2300.0), (0.0, 0.0, -2500.0),
         (-700.0, 100.0, -2100.0))


@pytest.fixture(scope="module")
def world(loc02: Any) -> dict[str, Any]:
    from hq.locate.locator import build_locator

    run = loc02.run_section()
    raw = loc02.test_config().model_dump(mode="json")
    raw["diagnostics"]["datumCheck"]["elevM"] = [-2000.0]
    raw["synthetic"]["nEvents"] = 2
    raw["statics"]["wellConstrained"]["minStations"] = 8
    cfg = SeismologyConfig.model_validate(raw)
    locator = build_locator(loc02.setup(cfg, run))
    rng = np.random.default_rng(SEED)
    east = {sid: e for sid, e, _, _, _ in loc02.STATIONS}
    frames, rows, links = [], [], []
    for k, (e, n, z) in enumerate(HYPOS):
        t0 = run.window_start_s + 600.0 * (k + 1)
        p = loc02.exact_picks(locator, e, n, z, t0, prob=0.8)
        p["t"] = (p["t"] + [_delay(east[s], ph) for s, ph in zip(p["stationId"], p["phase"],
                                                                strict=True)]
                  + rng.normal(0.0, 1.0, len(p)) * p["phase"].map(NOISE))
        p["id"] = [f"phasenet:{s}:{ph}:{t:.3f}" for s, ph, t in
                   zip(p["stationId"], p["phase"], p["t"], strict=True)]
        aid = f"assoc-{k:06d}"
        lat, lon, _ = from_enu(e, n, z - run.origin.elevM, run.origin)
        rows.append({"assocId": aid, "t": t0, "latitude": float(lat), "longitude": float(lon),
                     "elevM": z, "nPicks": len(p), "nP": int((p["phase"] == "P").sum()),
                     "nS": int((p["phase"] == "S").sum())})
        links += [{"assocId": aid, "pickId": i} for i in p["id"]]
        frames.append(p.assign(truthT0=t0))
    picks = pd.concat(frames, ignore_index=True)
    picks = picks.assign(picker="phasenet:test", eventId=None)[
        ["id", "stationId", "phase", "t", "prob", "picker", "eventId"]]
    assoc = AssocResult(
        typed_frame({c: [r[c] for r in rows] for c in ASSOC_EVENT_DTYPES}, ASSOC_EVENT_DTYPES),
        typed_frame({c: [r[c] for r in links] for c in ASSOC_PICK_DTYPES}, ASSOC_PICK_DTYPES),
    )
    st = loc02.stations(run.origin.elevM)
    lat, lon, _ = from_enu(st["enu_e"], st["enu_n"], st["enu_u"], run.origin)
    st = st.assign(latitude=lat, longitude=lon, usedInRun=True)
    cat = []
    for k, (e, n, z) in enumerate(HYPOS[:-1]):
        lat, lon, _ = from_enu(e, n, z - run.origin.elevM, run.origin)
        cat.append(CatalogEvent(
            id=f"cat{k}", source="test", t=rows[k]["t"], latitude=float(lat),
            longitude=float(lon), depthKm=-z / 1000.0, depthDatum="test: km below sea level",
            elevM=z, mag=1.0 + k / 10, magType="ml",
            enu={"e": e, "n": n, "u": z - run.origin.elevM}))
    catalog = to_frame(cat, CatalogEvent)
    pairs = pd.DataFrame({
        "catalogId": [f"cat{k}" for k in range(len(HYPOS) - 1)],
        "eventId": [f"unused{k}" for k in range(len(HYPOS) - 1)],
        "assocId": [f"assoc-{k:06d}" for k in range(len(HYPOS) - 1)],
        "catalogT": [r["t"] for r in rows[:-1]], "catalogE": [h[0] for h in HYPOS[:-1]],
        "catalogN": [h[1] for h in HYPOS[:-1]], "catalogElevM": [h[2] for h in HYPOS[:-1]],
        "pickIds": [links_k for links_k in (
            [x["pickId"] for x in links if x["assocId"] == f"assoc-{k:06d}"]
            for k in range(len(HYPOS) - 1))],
    })
    return {"run": run, "cfg": cfg, "picks": picks, "assoc": assoc, "stations": st,
            "catalog": catalog, "pairs": pairs, "cache": loc02.cache_dir}


def test_reference_statics_recover_the_planted_delays_held_out(world: dict[str, Any]) -> None:
    out = locate_with_statics(world["assoc"], world["picks"], world["stations"], world["cfg"],
                              world["run"], run_id="t", cache_dir=world["cache"],
                              reference=world["pairs"])
    rep, res = out.report, out.details.result
    assert rep.mode == "referenceEvents" and rep.pass_number == 2
    east = dict(zip(world["stations"]["id"], world["stations"]["enu_e"], strict=True))
    terms = res.statics.set_index(["stationId", "phase"])
    got = np.array([terms.loc[(s, p), "staticS"] for s, p in terms.index])
    want = np.array([_delay(east[s], p) for s, p in terms.index])
    assert np.ptp(got - want) < 0.03  # planted delays, up to the origin-time constant
    assert (terms["nEvents"] == len(world["pairs"])).all()
    ref = rep.reference
    assert len(ref) == len(world["pairs"])
    summary = rep.offsets_summary()
    assert summary["after"]["medianHM"] < summary["before"]["medianHM"]
    assert summary["after"]["medianRmsS"] < summary["before"]["medianRmsS"]
    assert summary["after"]["medianHM"] < 100.0
    assert res.events["quality_statics"].all()
    # The unmatched event uses the all-reference terms (the statics table); a reference event
    # uses its held-out ones.
    table = dict(zip(zip(res.statics["stationId"], res.statics["phase"]),
                     res.statics["staticS"], strict=True))
    for aid, same in ((f"assoc-{len(HYPOS) - 1:06d}", True), ("assoc-000000", False)):
        a = out.details.locations[out.details.assoc_ids.index(aid)].arrivals
        want_s = [table[k] for k in zip(a["stationId"], a["phase"], strict=True)]
        assert (a["staticS"].tolist() == pytest.approx(want_s)) is same
    record = rep.to_record()
    json.dumps(record)  # the run record must serialize
    assert record["crossValidatedOffsets"]["after"]["medianHM"] == pytest.approx(
        summary["after"]["medianHM"])


def test_locate_api_applies_no_statics_in_reference_mode(world: dict[str, Any]) -> None:
    res = locate(world["assoc"], world["picks"], world["stations"], world["cfg"], world["run"],
                 cache_dir=world["cache"])
    assert not res.events["quality_statics"].any()
    assert (res.statics["staticS"] == 0.0).all()


def test_self_consistent_mode_lowers_rms_inside_locate(world: dict[str, Any]) -> None:
    raw = world["cfg"].model_dump(mode="json")
    raw["statics"]["mode"] = "selfConsistent"
    raw["statics"]["minEvents"] = 3
    cfg = SeismologyConfig.model_validate(raw)
    args = (world["assoc"], world["picks"], world["stations"])
    plain = locate(*args, world["cfg"], world["run"], cache_dir=world["cache"])
    out = locate_with_statics(*args, cfg, world["run"], cache_dir=world["cache"])
    res = out.details.result
    assert res.events["quality_rmsS"].median() < plain.events["quality_rmsS"].median()
    assert (res.statics["staticS"].abs() <= cfg.statics.capS).all()
    assert (res.statics["staticS"] != 0.0).any() and res.events["quality_statics"].any()
    assert len(out.report.history) == cfg.statics.iterations + 1
    via_api = locate(*args, cfg, world["run"], cache_dir=world["cache"])
    pd.testing.assert_frame_equal(via_api.statics, res.statics)


def test_stage_runs_pass_1_then_pass_2_after_a_match(
    world: dict[str, Any], make_ctx: Any
) -> None:
    from hq.match import match

    stage = importlib.import_module("hq.locate.run")
    ctx = dataclasses.replace(make_ctx(world["run"], world["cfg"]), cache_dir=world["cache"])
    write_table(world["picks"], ctx.path(world["cfg"].associator.picksTable), "Pick")
    write_table(world["stations"], ctx.path("stations.parquet"), "Station")
    write_table(world["assoc"].events, ctx.path("assoc_events.parquet"), "AssocEvent")
    write_table(world["assoc"].picks, ctx.path("assoc_picks.parquet"), "AssocPick")
    write_table(world["catalog"], ctx.path("catalog.parquet"), "CatalogEvent")
    stage.run(ctx)  # pass 1: no matches.parquet yet
    first = read_table(ctx.path("events_located.parquet"))
    assert not first["quality_statics"].any()
    assert ctx.records[-1]["counts"]["staticsPass"] == 1
    assert "the statics pass needs a match first" in ctx.path("diagnostics.md").read_text()

    matches = match(first, world["catalog"], world["cfg"]).matches
    write_table(matches, ctx.path("matches.parquet"), "Match")
    stage.run(ctx)  # pass 2
    second = read_table(ctx.path("events_located.parquet"))
    assert second["quality_statics"].all()
    counts = ctx.records[-1]["counts"]
    assert counts["staticsPass"] == 2
    assert counts["staticsReferenceEvents"] == int(matches["eventId"].notna().sum())
    statics = read_table(ctx.path("statics.parquet"))
    assert (statics["staticS"] != 0.0).any()
    report = ctx.path("diagnostics.md").read_text()
    assert "## Station statics (LOC-05)" in report and "pass 2" in report
    assert "held-out terms (leave-one-out)" in report
    assert "tied to the public regional catalog's (UUSS) frame" in report
    record = ctx.records[-1]["params"]["statics"]
    assert record["pass"] == 2 and record["previousMedianRmsS"] == pytest.approx(
        float(first["quality_rmsS"].median()))

    # matches.parquet now refers to the pass-1 locations: a third locate must refuse it, and so
    # must stage tier.
    with pytest.raises(ValueError, match="stale"):
        stage.run(ctx)
    tier_stage = importlib.import_module("hq.tier.run")
    with pytest.raises(TierError, match="stale"):
        tier_stage.check_matches_current(second, matches, world["catalog"],
                                         world["cfg"].tiering.consistencyTolM)
    tier_stage.check_matches_current(second, match(second, world["catalog"], world["cfg"]).matches,
                                     world["catalog"], world["cfg"].tiering.consistencyTolM)


def test_catalog_hypocentres_off_the_grid_are_left_out(world: dict[str, Any]) -> None:
    pairs = world["pairs"].copy()
    pairs.loc[0, "catalogElevM"] = -50000.0  # below the travel-time grid
    out = locate_with_statics(world["assoc"], world["picks"], world["stations"], world["cfg"],
                              world["run"], cache_dir=world["cache"], reference=pairs)
    assert out.report.skipped == ("cat0",)
    assert len(out.report.reference) == len(pairs) - 1
    with pytest.raises(ValueError, match="outside the travel-time grid"):
        locate_with_statics(world["assoc"], world["picks"], world["stations"], world["cfg"],
                            world["run"], cache_dir=world["cache"],
                            reference=pairs.assign(catalogElevM=-50000.0))

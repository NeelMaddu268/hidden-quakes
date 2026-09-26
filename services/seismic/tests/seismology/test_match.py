"""hq.match (MATCH-02): one-to-one matching, sensitivity, unmatched reasons and the stage.

All data here is small, synthetic and seeded; nothing touches the network.
"""

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hq_contracts.io import columns_for, read_table, to_frame, write_table
from hq_contracts.models import Enu, Pick, SeismicEvent, Station
from pydantic import ValidationError
from pyproj import Proj
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig
from hq.locate.velocity import load_configured_model
from hq.match import (
    MATCH_DTYPES,
    REASONS,
    SENSITIVITY_DTYPES,
    Tolerance,
    assign,
    match,
    pair_offsets,
)
from hq.match import catalog as catalog_stage
from hq.match.reasons import Evidence, SpeedBounds, code_of, explain_unmatched

pytestmark = pytest.mark.smoke

stage = importlib.import_module("hq.match.run")

T0 = 1_789_041_600.25  # 2026-09-10T12:00:00.25Z, inside the showcase window
KM = 1000.0
MATCH_ARROW = {
    "catalogId": pa.string(),
    "eventId": pa.string(),
    "dtS": pa.float64(),
    "distM": pa.float64(),
    "reason": pa.string(),
}
SENSITIVITY_ARROW = {"dtS": pa.float64(), "distM": pa.float64(), "recovered": pa.int64()}
# events_located.parquet (docs/02 §2): SeismicEvent fields except tier, tierReasons, catalogMatch
# and magnitude.
LOCATED_COLUMNS = [
    c
    for c in columns_for(SeismicEvent)
    if c not in ("tier", "tierReasons")
    and not c.startswith("catalogMatch_")
    and not c.startswith("magnitude_")
]


# ---------------------------------------------------------------- builders


def public(
    rows: list[tuple[str, float, float, float]], run: RunSection | None = None
) -> pd.DataFrame:
    """Typed CatalogEvent frame from (id, t, e, n); u = -4000 m, lat/lon at the run origin."""
    lat = run.origin.lat if run else 38.51
    lon = run.origin.lon if run else -112.9
    frame = pd.DataFrame(
        [
            {
                "id": cid,
                "source": "UU via USGS ComCat",
                "t": t,
                "latitude": lat,
                "longitude": lon,
                "depthKm": 2.4,
                "depthDatum": "synthetic test datum",
                "elevM": -2372.3,
                "mag": 1.0,
                "magType": "ml",
                "enu_e": e,
                "enu_n": n,
                "enu_u": -4000.0,
                "matchedEventId": np.nan,
            }
            for cid, t, e, n in rows
        ],
        columns=list(catalog_stage._FRAME_DTYPES),
    )
    return frame.astype(catalog_stage._FRAME_DTYPES)


def located(
    rows: list[tuple[str, float, float, float]], pick_ids: dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """events_located-shaped frame from (id, t, e, n), with pickIds per event."""
    picks = pick_ids or {}
    data: dict[str, list[Any]] = {c: [] for c in LOCATED_COLUMNS}
    for eid, t, e, n in rows:
        values: dict[str, Any] = {c: 0 for c in LOCATED_COLUMNS}
        values.update(
            {
                "id": eid,
                "runId": "test-run",
                "source": "hq-pipeline",
                "t": t,
                "latitude": 38.51,
                "longitude": -112.9,
                "elevM": -2372.3,
                "depthKm": 4.0,
                "enu_e": e,
                "enu_n": n,
                "enu_u": -4000.0,
                "quality_method": "grid1d",
                "quality_statics": False,
                "quality_rmsS": 0.05,
                "quality_gapDeg": 90.0,
                "quality_minEpiDistM": 1000.0,
                "quality_hErrM": 200.0,
                "quality_vErrM": 400.0,
                "quality_depthOnEdge": False,
                "meanPickProb": 0.8,
                "revealOrder": -1,
                "pickIds": picks.get(eid, []),
            }
        )
        for c in LOCATED_COLUMNS:
            data[c].append(values[c])
    frame = pd.DataFrame(data, columns=LOCATED_COLUMNS)
    return frame.astype({"t": "float64", "enu_e": "float64", "enu_n": "float64"})


def stations(n: int = 6, radius_m: float = 5000.0, unused: int = 1) -> pd.DataFrame:
    """n used stations on a ring around the origin at u = 0, plus ``unused`` unused ones."""
    rows = []
    for k in range(n + unused):
        az = 2 * np.pi * k / (n + unused)
        rows.append(
            Station(
                id=f"XX.S{k:02d}",
                network="XX",
                station=f"S{k:02d}",
                latitude=38.51,
                longitude=-112.9,
                surfaceElevM=1627.7,
                sensorDepthM=0.0,
                sensorElevM=1627.7,
                kind="surface",
                channels=["HHZ", "HHN", "HHE"],
                sampleRateHz=100.0,
                enu=Enu(e=radius_m * np.cos(az), n=radius_m * np.sin(az), u=0.0),
                preprocessProfile="surface",
                usedInRun=k < n,
            )
        )
    return to_frame(rows, Station)


def pick(sid: str, phase: str, t: float) -> Pick:
    return Pick(
        id=f"phasenet:{sid}:{phase}:{t:.3f}",
        stationId=sid,
        phase=phase,
        t=t,
        prob=0.9,
        picker="phasenet:test",
    )


def arrival_picks(
    sta: pd.DataFrame, t0: float, hypo: tuple[float, float, float], n: int, v_p: float = 4000.0
) -> list[Pick]:
    """P picks at t0 + R / v_p on the first ``n`` used stations (a speed inside the model range)."""
    out = []
    used = sta[sta["usedInRun"]]
    for _, row in used.head(n).iterrows():
        r = float(np.linalg.norm(np.array([row.enu_e, row.enu_n, row.enu_u]) - np.array(hypo)))
        out.append(pick(row.id, "P", t0 + r / v_p))
    return out


def picks_frame(picks: list[Pick]) -> pd.DataFrame:
    return to_frame(picks, Pick)


def gaps_frame(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["stationId", "channel", "gapStart", "gapEnd"]).astype(
        {"stationId": "str", "channel": "str", "gapStart": "float64", "gapEnd": "float64"}
    )


def assoc_frame(pick_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"assocId": ["a1"] * len(pick_ids), "pickId": pick_ids}).astype("str")


def arrow_types(df: pd.DataFrame, tmp_path: Path, name: str) -> dict[str, pa.DataType]:
    path = tmp_path / f"{name}.parquet"
    write_table(df, path, name)
    return {f.name: f.type for f in pq.read_schema(path)}


@pytest.fixture
def speeds(seismology_config: SeismologyConfig) -> SpeedBounds:
    return SpeedBounds.from_model(load_configured_model(seismology_config.velocity))


def _with_matching(cfg: SeismologyConfig, **update: Any) -> SeismologyConfig:
    return cfg.model_copy(update={"matching": cfg.matching.model_copy(update=update)})


# ---------------------------------------------------------------- config


def test_matching_config_parses_and_rejects_unknown_keys(
    seismology_config: SeismologyConfig,
) -> None:
    m = seismology_config.matching
    assert (m.dtScaleS, m.distScaleM, m.maxDtS, m.maxDistM) == (2.0, 5000.0, 2.0, 5000.0)
    assert [(p.dtS, p.distM) for p in m.sensitivity] == [
        (1.0, 3000.0),
        (2.0, 5000.0),
        (3.0, 8000.0),
    ]
    raw = seismology_config.model_dump(mode="json")
    for bad in (
        {**raw, "matching": {**raw["matching"], "unknown": 1}},
        {**raw, "matching": {**raw["matching"], "reasons": {**raw["matching"]["reasons"], "x": 1}}},
    ):
        with pytest.raises(ValidationError, match="extra"):
            SeismologyConfig.model_validate(bad)
    dupes = {**raw["matching"], "sensitivity": [{"dtS": 1.0, "distM": 3000.0}] * 2}
    with pytest.raises(ValidationError, match="unique"):
        SeismologyConfig.model_validate({**raw, "matching": dupes})


def test_grid_scale_factor_is_negligible_over_the_showcase_bbox(run_section: RunSection) -> None:
    """The module docstring's claim: UTM grid distance is within 1.7e-4 of ellipsoidal distance."""
    proj = Proj("EPSG:32612")
    min_lon, min_lat, max_lon, max_lat = run_section.bbox
    lons, lats = np.meshgrid(np.linspace(min_lon, max_lon, 7), np.linspace(min_lat, max_lat, 7))
    k = np.asarray(proj.get_factors(lons.ravel(), lats.ravel()).meridional_scale)
    assert np.abs(k - 1.0).max() < 1.7e-4  # under 1 m at 5 km


# ---------------------------------------------------------------- assignment


def _brute_force(dt: np.ndarray, dist: np.ndarray, tol: Tolerance) -> tuple[int, float]:
    """(most admissible one-to-one pairs, least total cost among those), by enumeration."""
    ok = tol.admissible(dt, dist)
    cost = tol.cost(dt, dist)
    n, m = dt.shape
    best = [0, 0.0]

    def rec(i: int, used: int, count: int, total: float) -> None:
        if i == n:
            if count > best[0] or (count == best[0] and total < best[1]):
                best[0], best[1] = count, total
            return
        rec(i + 1, used, count, total)
        for j in range(m):
            if ok[i, j] and not (used >> j) & 1:
                rec(i + 1, used | (1 << j), count + 1, total + float(cost[i, j]))

    rec(0, 0, 0, 0.0)
    return int(best[0]), float(best[1])


def _scenario(
    rng: np.random.Generator, n_pub: int, n_loc: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dense conflicts: times within a few seconds and positions within ~10 km of each other."""
    pub = public(
        [
            (f"uu{i:03d}", T0 + rng.uniform(0, 5), rng.uniform(-5, 5) * KM, rng.uniform(-5, 5) * KM)
            for i in range(n_pub)
        ]
    )
    loc = located(
        [
            (
                f"hq-{j:04d}",
                T0 + rng.uniform(-1, 6),
                rng.uniform(-6, 6) * KM,
                rng.uniform(-6, 6) * KM,
            )
            for j in range(n_loc)
        ]
    )
    return pub, loc


def _check_one_to_one(pub: pd.DataFrame, loc: pd.DataFrame, cfg: SeismologyConfig) -> pd.DataFrame:
    matches = match(loc, pub, cfg).matches
    assert sorted(matches["catalogId"]) == sorted(pub["id"])  # one row per public event
    matched = matches[matches["eventId"].notna()]
    assert matched["eventId"].is_unique and matched["catalogId"].is_unique
    assert set(matched["eventId"]) <= set(loc["id"])
    assert matches.loc[matches["eventId"].isna(), "reason"].notna().all()
    assert matched["reason"].isna().all()
    # Every reported pair is admissible, with dtS / distM recomputed from the inputs.
    tol = Tolerance.from_config(cfg.matching)
    lt = loc.set_index("id")
    pt = pub.set_index("id")
    for row in matched.itertuples(index=False):
        dt = lt.at[row.eventId, "t"] - pt.at[row.catalogId, "t"]
        dist = np.hypot(
            lt.at[row.eventId, "enu_e"] - pt.at[row.catalogId, "enu_e"],
            lt.at[row.eventId, "enu_n"] - pt.at[row.catalogId, "enu_n"],
        )
        assert row.dtS == dt and row.distM == dist
        assert tol.admissible(np.array(dt), np.array(dist))
    return matches


def test_one_to_one_property_over_random_scenarios(seismology_config: SeismologyConfig) -> None:
    """Never a located or public event twice; the count is the maximum possible (Hopcroft-Karp)."""
    rng = np.random.default_rng(20260926)
    tol = Tolerance.from_config(seismology_config.matching)
    for _ in range(300):
        pub, loc = _scenario(rng, int(rng.integers(0, 25)), int(rng.integers(0, 30)))
        matches = _check_one_to_one(pub, loc, seismology_config)
        pub_sorted = pub.sort_values(["t", "id"]).reset_index(drop=True)
        loc_sorted = loc.sort_values(["t", "id"]).reset_index(drop=True)
        off = pair_offsets(loc_sorted, pub_sorted)
        graph = csr_matrix(tol.admissible(off.dt, off.dist).astype(np.int8))
        most = int((maximum_bipartite_matching(graph, perm_type="column") >= 0).sum())
        assert int(matches["eventId"].notna().sum()) == most


def test_assignment_is_optimal_against_brute_force(seismology_config: SeismologyConfig) -> None:
    rng = np.random.default_rng(7)
    tol = Tolerance.from_config(seismology_config.matching)
    for _ in range(200):
        n_pub, n_loc = int(rng.integers(1, 6)), int(rng.integers(1, 7))
        pub, loc = _scenario(rng, n_pub, n_loc)
        off = pair_offsets(loc, pub)
        rows, cols = assign(off, tol)
        count, total = _brute_force(off.dt, off.dist, tol)
        assert rows.size == count
        assert float(tol.cost(off.dt[rows, cols], off.dist[rows, cols]).sum()) == pytest.approx(
            total, abs=1e-9
        )


def test_competition_prefers_more_pairs_over_one_cheap_pair(
    seismology_config: SeismologyConfig,
) -> None:
    """a-x is the cheapest pair, but taking it strands b; a-y plus b-x recovers both."""
    pub = public([("a", T0, 0.0, 0.0), ("b", T0 + 3.0, 0.0, 0.0)])
    loc = located([("x", T0 + 1.5, 0.0, 0.0), ("y", T0 - 1.8, 0.0, 0.0)])
    matches = match(loc, pub, seismology_config).matches.set_index("catalogId")
    assert matches.at["a", "eventId"] == "y" and matches.at["b", "eventId"] == "x"


def test_result_does_not_depend_on_row_order(seismology_config: SeismologyConfig) -> None:
    rng = np.random.default_rng(11)
    pub, loc = _scenario(rng, 15, 20)
    first = match(loc, pub, seismology_config)
    shuffled = match(
        loc.sample(frac=1.0, random_state=1),
        pub.sample(frac=1.0, random_state=2),
        seismology_config,
    )
    pd.testing.assert_frame_equal(first.matches, shuffled.matches)
    pd.testing.assert_frame_equal(first.sensitivity, shuffled.sensitivity)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_admissibility_edges(seismology_config: SeismologyConfig, sign: float) -> None:
    """Exactly 2 s and exactly 5 km (together) are admissible; the next step past either is not."""
    tiny_dt = 2.0**-20  # a few ulps of an epoch near 1.8e9 s; T0 + 2 s + tiny_dt is exact
    pub = public(
        [("edge", T0, 100.5, -200.25), ("late", T0 + 60, 0.0, 0.0), ("far", T0 + 120, 0, 0)]
    )
    loc = located(
        [
            ("at-edge", T0 + sign * 2.0, 100.5 + sign * 3000.0, -200.25 + sign * 4000.0),
            ("over-dt", T0 + 60 + sign * (2.0 + tiny_dt), 0.0, 0.0),
            ("over-dist", T0 + 120, sign * 5000.001, 0.0),
        ]
    )
    matches = match(loc, pub, seismology_config).matches.set_index("catalogId")
    assert matches.at["edge", "eventId"] == "at-edge"
    assert abs(matches.at["edge", "dtS"]) == 2.0 and matches.at["edge", "distM"] == 5000.0
    for cid in ("late", "far"):
        assert pd.isna(matches.at[cid, "eventId"])
        assert matches.at[cid, "reason"].startswith("no candidate within 2 s / 5 km (nearest: ")


def test_magnitude_is_ignored(seismology_config: SeismologyConfig) -> None:
    """Magnitudes that agree or disagree, or no magnitude columns at all, give the same result."""
    pub = public([("a", T0, 0.0, 0.0)])
    loc = located([("near-time", T0 + 0.1, 1500.0, 0.0), ("near-space", T0 + 1.5, 100.0, 0.0)])
    base = match(loc, pub, seismology_config)
    assert base.matches.at[0, "eventId"] == "near-time"  # cost 0.05 + 0.3 < 0.75 + 0.02
    for mags in ((3.0, 0.1, 3.0), (0.1, 3.0, 0.1)):
        pub_m = pub.assign(mag=mags[0])
        loc_m = loc.assign(magnitude_value=[mags[1], mags[2]], magnitude_type="ML_cal")
        again = match(loc_m, pub_m, seismology_config)
        pd.testing.assert_frame_equal(base.matches, again.matches)
    bare = match(
        loc[["id", "t", "enu_e", "enu_n"]], pub[["id", "t", "enu_e", "enu_n"]], seismology_config
    )
    pd.testing.assert_frame_equal(base.matches, bare.matches)


def test_depth_is_ignored(seismology_config: SeismologyConfig) -> None:
    pub = public([("a", T0, 0.0, 0.0)])
    loc = located([("x", T0, 4999.0, 0.0)]).assign(enu_u=-20000.0, elevM=-18000.0)
    assert match(loc, pub, seismology_config).matches.at[0, "distM"] == 4999.0


def test_sensitivity_rows_rerun_with_each_pair(seismology_config: SeismologyConfig) -> None:
    pub = public([("a", T0, 0, 0), ("b", T0 + 100, 0, 0), ("c", T0 + 200, 0, 0)])
    loc = located(
        [
            ("in-all", T0 + 0.5, 1000.0, 0.0),
            ("default-only", T0 + 101.5, 2000.0, 0.0),  # dt 1.5 s: over the 1 s pair
            ("wide-only", T0 + 202.5, 6000.0, 0.0),  # dt 2.5 s, 6 km: only (3 s, 8 km)
        ]
    )
    result = match(loc, pub, seismology_config)
    sens = result.sensitivity
    assert list(sens.columns) == list(SENSITIVITY_DTYPES)
    assert sens.to_dict("records") == [
        {"dtS": 1.0, "distM": 3000.0, "recovered": 1},
        {"dtS": 2.0, "distM": 5000.0, "recovered": 2},
        {"dtS": 3.0, "distM": 8000.0, "recovered": 3},
    ]
    # The default pair's row equals the headline count.
    assert int(result.matches["eventId"].notna().sum()) == 2
    # Each pair is its own limit: a single (1 s, 3 km) pair gives the first row's count.
    only = _with_matching(
        seismology_config, sensitivity=[seismology_config.matching.sensitivity[0]]
    )
    assert match(loc, pub, only).sensitivity["recovered"].tolist() == [1]


def test_zero_located_events_all_unmatched_with_types(
    seismology_config: SeismologyConfig, tmp_path: Path
) -> None:
    pub = public([("a", T0, 0, 0), ("b", T0 + 10, 0, 0)])
    result = match(located([]), pub, seismology_config)
    m = result.matches
    assert m["eventId"].isna().all() and m["dtS"].isna().all() and m["distM"].isna().all()
    assert set(m["reason"]) == {"no candidate within 2 s / 5 km (no located events)"}
    assert dict(m.dtypes) == MATCH_DTYPES
    assert arrow_types(m, tmp_path, "matches") == MATCH_ARROW
    assert result.sensitivity["recovered"].tolist() == [0, 0, 0]
    assert arrow_types(result.sensitivity, tmp_path, "sens") == SENSITIVITY_ARROW


def test_zero_catalog_events(seismology_config: SeismologyConfig, tmp_path: Path) -> None:
    result = match(located([("x", T0, 0, 0)]), public([]), seismology_config)
    assert len(result.matches) == 0 and dict(result.matches.dtypes) == MATCH_DTYPES
    assert arrow_types(result.matches, tmp_path, "matches") == MATCH_ARROW
    assert result.sensitivity["recovered"].tolist() == [0, 0, 0]


def test_all_matched_keeps_reason_typed(
    seismology_config: SeismologyConfig, tmp_path: Path
) -> None:
    result = match(located([("x", T0, 0, 0)]), public([("a", T0, 0, 0)]), seismology_config)
    assert result.matches["reason"].isna().all()
    assert arrow_types(result.matches, tmp_path, "matches") == MATCH_ARROW


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda df: df.drop(columns=["enu_n"]), "missing columns"),
        (lambda df: pd.concat([df, df]), "duplicate ids"),
        (lambda df: df.assign(t=np.nan), "non-finite t"),
    ],
)
def test_bad_inputs_fail_loudly(
    seismology_config: SeismologyConfig, mutate: Any, message: str
) -> None:
    pub = public([("a", T0, 0, 0)])
    loc = located([("x", T0, 0, 0)])
    with pytest.raises(ValueError, match=message):
        match(mutate(loc), pub, seismology_config)
    with pytest.raises(ValueError, match=message):
        match(loc, mutate(pub), seismology_config)


# ---------------------------------------------------------------- unmatched reasons


HYPO = (0.0, 0.0, -4000.0)  # public hypocentre ENU used by public()


def _explain(
    loc: pd.DataFrame,
    pub: pd.DataFrame,
    cfg: SeismologyConfig,
    run: RunSection,
    speeds: SpeedBounds,
    evidence: Evidence | None = None,
) -> tuple[dict[str, str], Any]:
    result = match(loc, pub, cfg)
    explained = explain_unmatched(result, loc, pub, evidence or Evidence(), cfg, run, speeds)
    reasons = {
        r.catalogId: r.reason
        for r in explained.matches.itertuples(index=False)
        if pd.isna(r.eventId)
    }
    return reasons, explained


def test_reason_lost_one_to_one(
    seismology_config: SeismologyConfig, run_section: RunSection, speeds: SpeedBounds
) -> None:
    """x is admissible to both; a is cheaper, so b loses the competition for it."""
    pub = public([("a", T0, 0, 0), ("b", T0 + 1.0, 0, 0)], run_section)
    loc = located([("x", T0 + 0.1, 0.0, 0.0)])
    sta = stations()
    evidence = Evidence(stations=sta, picks=picks_frame([]))  # evidence would say "too few picks"
    reasons, explained = _explain(loc, pub, seismology_config, run_section, speeds, evidence)
    assert reasons == {
        "b": "lost one-to-one (1 admissible candidate(s), each assigned to another public event; "
        "best x: dt -0.90 s, 0.00 km, assigned to a)"
    }
    assert explained.codes == {"b": "lostOneToOne"}


def test_reason_outside_window_and_bbox(
    seismology_config: SeismologyConfig, run_section: RunSection, speeds: SpeedBounds
) -> None:
    early = run_section.window_start_s - 0.5
    pub = public([("early", early, 0, 0), ("east", T0, 0, 0)], run_section)
    pub.loc[pub["id"] == "east", "longitude"] = run_section.bbox[2] + 0.01
    reasons, explained = _explain(located([]), pub, seismology_config, run_section, speeds)
    assert reasons["early"] == (
        "outside run window (origin 2026-09-09T23:59:59.500Z not in "
        "[2026-09-10T00:00:00.000Z, 2026-09-11T00:00:00.000Z))"
    )
    assert reasons["east"].startswith("outside run bbox (lat 38.5100, lon -112.5900 not in ")
    assert explained.codes == {"early": "outsideWindow", "east": "outsideBbox"}


def test_reason_no_evidence_keeps_nearest_candidate(
    seismology_config: SeismologyConfig, run_section: RunSection, speeds: SpeedBounds
) -> None:
    pub = public([("a", T0, 0, 0)], run_section)
    loc = located([("x", T0 + 3.4, 7900.0, 0.0), ("y", T0 + 500.0, 0.0, 0.0)])
    reasons, explained = _explain(loc, pub, seismology_config, run_section, speeds)
    assert reasons == {"a": "no candidate within 2 s / 5 km (nearest: dt +3.40 s, 7.90 km)"}
    assert explained.checks_run == ["window", "bbox", "oneToOne", "tolerance"]
    assert explained.checks_skipped == {
        "waveformData": "missing stations, gaps",
        "picks": "missing stations, picks",
        "association": "missing stations, picks, assoc_picks",
        "location": "missing stations, picks",
    }


def test_reason_no_waveform_data(
    seismology_config: SeismologyConfig, run_section: RunSection, speeds: SpeedBounds
) -> None:
    pub = public([("a", T0, 0, 0)], run_section)
    sta = stations()
    used = sta.loc[sta["usedInRun"], "id"].tolist()
    # Three stations fully gapped (two touching gaps on one channel), one partly gapped.
    rows = [(s, c, T0 - 100, T0 + 100) for s in used[:2] for c in ("HHZ", "HHN", "HHE")]
    rows += [(used[2], c, T0 - 100, T0 + 100) for c in ("HHZ", "HHN")]
    rows += [(used[2], "HHE", T0 - 100, T0 + 3.0), (used[2], "HHE", T0 + 3.0, T0 + 100)]
    rows += [(used[3], "HHZ", T0 - 100, T0 + 100)]  # other channels have data: counts as data
    evidence = Evidence(stations=sta, gaps=gaps_frame(rows))
    reasons, explained = _explain(
        located([]), pub, seismology_config, run_section, speeds, evidence
    )
    assert reasons == {
        "a": "no waveform data (data on 3 of 6 used stations in the expected arrival windows; "
        "need 4)"
    }
    assert "waveformData" in explained.checks_run
    # One fewer gapped station leaves 4 with data: the data check passes; picks were not given.
    evidence = Evidence(stations=sta, gaps=gaps_frame([r for r in rows if r[0] != used[0]]))
    reasons, _ = _explain(located([]), pub, seismology_config, run_section, speeds, evidence)
    assert reasons["a"].startswith("no candidate within")


def test_reason_too_few_picks_counts_only_used_stations_and_windows(
    seismology_config: SeismologyConfig, run_section: RunSection, speeds: SpeedBounds
) -> None:
    pub = public([("a", T0, 0, 0)], run_section)
    sta = stations()
    unused = sta.loc[~sta["usedInRun"], "id"].item()
    picks = arrival_picks(sta, T0, HYPO, n=3)
    picks += [pick(unused, "P", T0 + 1.6)]  # an unused station never counts
    s0 = sta.loc[sta["usedInRun"], "id"].iloc[4]
    picks += [pick(s0, "P", T0 - 30.0), pick(s0, "S", T0 + 40.0)]  # outside both windows
    evidence = Evidence(stations=sta, gaps=gaps_frame([]), picks=picks_frame(picks))
    reasons, _ = _explain(located([]), pub, seismology_config, run_section, speeds, evidence)
    assert reasons == {
        "a": "too few picks (picks on 3 of 6 used stations in the expected arrival windows; need 4)"
    }
    # An S pick inside the S window makes a fourth picked station.
    picks += [pick(s0, "S", T0 + 3.0)]
    evidence = Evidence(stations=sta, picks=picks_frame(picks))
    reasons, _ = _explain(located([]), pub, seismology_config, run_section, speeds, evidence)
    assert reasons["a"].startswith("no candidate within")


def test_reason_picks_not_associated(
    seismology_config: SeismologyConfig, run_section: RunSection, speeds: SpeedBounds
) -> None:
    pub = public([("a", T0, 0, 0)], run_section)
    sta = stations()
    picks = arrival_picks(sta, T0, HYPO, n=6)
    evidence = Evidence(
        stations=sta, picks=picks_frame(picks), assoc_picks=assoc_frame([p.id for p in picks[:2]])
    )
    reasons, explained = _explain(
        located([]), pub, seismology_config, run_section, speeds, evidence
    )
    assert reasons == {
        "a": "picks not associated (associated picks on 2 of the 6 stations with picks in the "
        "expected arrival windows; need 4)"
    }
    assert explained.codes == {"a": "picksNotAssociated"}


def test_reason_associated_not_located(
    seismology_config: SeismologyConfig, run_section: RunSection, speeds: SpeedBounds
) -> None:
    pub = public([("a", T0, 0, 0)], run_section)
    sta = stations()
    picks = arrival_picks(sta, T0, HYPO, n=5)
    loc = located([("x", T0 + 900.0, 0.0, 0.0)], {"x": ["phasenet:other:P:1.000"]})
    evidence = Evidence(
        stations=sta,
        gaps=gaps_frame([]),
        picks=picks_frame(picks),
        assoc_picks=assoc_frame([p.id for p in picks]),
    )
    reasons, explained = _explain(loc, pub, seismology_config, run_section, speeds, evidence)
    assert reasons == {
        "a": "associated, not located (associated picks on 5 stations; no located event carries "
        "picks from 4 of them)"
    }
    assert explained.checks_skipped == {}


def test_reason_located_out_of_tolerance(
    seismology_config: SeismologyConfig, run_section: RunSection, speeds: SpeedBounds
) -> None:
    pub = public([("a", T0, 0, 0)], run_section)
    sta = stations()
    picks = arrival_picks(sta, T0, HYPO, n=6)
    ids = [p.id for p in picks]
    loc = located(
        [
            ("carrier-far", T0 + 0.30, 7000.0, 0.0),  # carries 5 stations' picks, 7 km off
            ("carrier-few", T0 + 0.10, 5500.0, 0.0),  # closer, but only 3 stations' picks
            ("unrelated", T0 + 3.0, 0.0, 6000.0),
        ],
        {"carrier-far": ids[:5], "carrier-few": ids[3:6]},
    )
    evidence = Evidence(stations=sta, picks=picks_frame(picks), assoc_picks=assoc_frame(ids))
    reasons, explained = _explain(loc, pub, seismology_config, run_section, speeds, evidence)
    assert reasons == {
        "a": "located out of tolerance 2 s / 5 km (nearest associated candidate carrier-far: "
        "dt +0.30 s, 7.00 km; picks from 5 stations)"
    }
    # Without assoc_picks the chain goes picks -> location and still finds the carrier.
    evidence = Evidence(stations=sta, picks=picks_frame(picks))
    reasons, explained = _explain(loc, pub, seismology_config, run_section, speeds, evidence)
    assert reasons["a"].startswith("located out of tolerance 2 s / 5 km (nearest associated ")
    assert explained.checks_skipped == {
        "waveformData": "missing gaps",
        "association": "missing assoc_picks",
    }


def test_reason_no_carrier_without_association_keeps_base(
    seismology_config: SeismologyConfig, run_section: RunSection, speeds: SpeedBounds
) -> None:
    pub = public([("a", T0, 0, 0)], run_section)
    sta = stations()
    evidence = Evidence(stations=sta, picks=picks_frame(arrival_picks(sta, T0, HYPO, n=6)))
    reasons, _ = _explain(located([]), pub, seismology_config, run_section, speeds, evidence)
    assert reasons == {"a": "no candidate within 2 s / 5 km (no located events)"}


def test_every_reason_has_one_stable_prefix() -> None:
    prefixes = list(REASONS.values())
    for p in prefixes:
        assert sum(q.startswith(p) or p.startswith(q) for q in prefixes) == 1
    assert code_of("too few picks (picks on 0 of 0 used stations)") == "tooFewPicks"
    with pytest.raises(ValueError):
        code_of("something else")


def test_arrival_bracket_uses_model_extremes(speeds: SpeedBounds) -> None:
    model = speeds
    assert model.vp_min < model.vp_max and model.vs_min < model.vs_max
    assert model.vs_max < model.vp_max and model.vs_min < model.vp_min


# ---------------------------------------------------------------- stage


def _write_run(ctx: Any, pub: pd.DataFrame, loc: pd.DataFrame, **evidence: pd.DataFrame) -> None:
    catalog_stage._write_table(pub, ctx.path("catalog.parquet"))
    write_table(loc, ctx.path("events_located.parquet"), "SeismicEvent")
    names = {"stations": "Station", "gaps": "Gap", "picks": "Pick", "assoc_picks": "AssocPick"}
    for key, frame in evidence.items():
        write_table(frame, ctx.path(f"{key}.parquet"), names[key])


def test_stage_writes_tables_fills_matched_ids_and_records(
    make_ctx: Any,
    run_section: RunSection,
    seismology_config: SeismologyConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = make_ctx(run_section)
    pub = public([("uu1", T0, 0, 0), ("uu2", T0 + 100, 0, 0), ("uu3", T0 + 200, 0, 0)], run_section)
    sta = stations()
    picks = arrival_picks(sta, T0 + 200, HYPO, n=6)
    loc = located([("hq-1", T0 + 0.2, 300.0, 0.0), ("hq-2", T0 + 101.0, 1000.0, 1000.0)])
    _write_run(ctx, pub, loc, stations=sta, picks=picks_frame(picks))
    before = pq.read_table(ctx.path("catalog.parquet"))

    with caplog.at_level(logging.INFO, logger=stage.__name__):
        stage.run(ctx)

    matches = read_table(ctx.path("matches.parquet"))
    assert matches.attrs["model"] == stage.MATCHES_MODEL
    assert {f.name: f.type for f in pq.read_schema(ctx.path("matches.parquet"))} == MATCH_ARROW
    assert matches["eventId"].tolist()[:2] == ["hq-1", "hq-2"] and pd.isna(matches.at[2, "eventId"])
    # Picks exist on 6 stations, but no assoc_picks: no located event carries them, and without
    # the association check "associated, not located" cannot be claimed, so the base reason stays.
    assert (
        matches.at[2, "reason"] == "no candidate within 2 s / 5 km (nearest: dt -99.00 s, 1.41 km)"
    )
    sens = read_table(ctx.path("match_sensitivity.parquet"))
    assert sens["recovered"].tolist() == [2, 2, 2]

    after = pq.read_table(ctx.path("catalog.parquet"))
    assert after.schema == before.schema  # same Arrow schema and metadata
    assert after.column("matchedEventId").to_pylist() == ["hq-1", "hq-2", None]
    for name in before.column_names:
        if name != "matchedEventId":
            assert after.column(name).equals(before.column(name))
    assert not list(ctx.run_dir.glob("*.part"))

    (record,) = ctx.records
    assert record["stage"] == "match" and set(record["params"]) == {"match"}
    counts = record["counts"]
    assert counts["publicEvents"] == 3 and counts["recovered"] == 2 and counts["unmatched"] == 1
    assert counts["locatedEvents"] == 2
    assert sum(v for k, v in counts.items() if k.startswith("unmatched.")) == 1
    params = record["params"]["match"]
    assert params["config"] == seismology_config.matching.model_dump(mode="json")
    assert params["checksRun"] == ["window", "bbox", "oneToOne", "tolerance", "picks", "location"]
    assert set(params["checksSkipped"]) == {"waveformData", "association"}
    assert params["sensitivity"][1] == {"dtS": 2.0, "distM": 5000.0, "recovered": 2}
    assert set(params["unmatched"]) == {"uu3"}

    text = caplog.text
    assert "recovered 2 / 3 public catalog events in window [2026-09-10T00:00:00Z, " in text
    assert "unmatched uu3: " in text
    assert "dt 3 s, dist 8 km: recovered 2 / 3" in text


def test_stage_reasons_from_run_dir_evidence(
    make_ctx: Any, run_section: RunSection, seismology_config: SeismologyConfig
) -> None:
    ctx = make_ctx(run_section)
    pub = public([("uu1", T0, 0, 0)], run_section)
    sta = stations()
    picks = arrival_picks(sta, T0, HYPO, n=6)
    _write_run(
        ctx,
        pub,
        located([]),
        stations=sta,
        gaps=gaps_frame([]),
        picks=picks_frame(picks),
        assoc_picks=assoc_frame([p.id for p in picks[:1]]),
    )
    stage.run(ctx)
    matches = read_table(ctx.path("matches.parquet"))
    assert matches.at[0, "reason"].startswith("picks not associated (associated picks on 1 of ")
    (record,) = ctx.records
    assert record["counts"]["unmatched.picksNotAssociated"] == 1
    assert record["params"]["match"]["checksSkipped"] == {}
    assert read_table(ctx.path("catalog.parquet"))["matchedEventId"].isna().all()


def test_stage_rerun_overwrites_and_failure_leaves_previous_files(
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
    seismology_config: SeismologyConfig,
) -> None:
    ctx = make_ctx(run_section)
    pub = public([("uu1", T0, 0, 0)], run_section)
    _write_run(ctx, pub, located([("hq-1", T0, 0, 0)]))
    stage.run(ctx)
    assert read_table(ctx.path("catalog.parquet"))["matchedEventId"].tolist() == ["hq-1"]

    # Rerun with no located events: matchedEventId is cleared, not left stale.
    write_table(located([]), ctx.path("events_located.parquet"), "SeismicEvent")
    stage.run(ctx)
    assert read_table(ctx.path("catalog.parquet"))["matchedEventId"].isna().all()
    saved = {
        n: ctx.path(n).read_bytes()
        for n in ("matches.parquet", "catalog.parquet", "match_sensitivity.parquet")
    }

    # A failure on the last write leaves every previous file byte-identical and no .part files.
    write_table(located([("hq-2", T0, 0, 0)]), ctx.path("events_located.parquet"), "SeismicEvent")

    def boom(df: pd.DataFrame, path: Path) -> None:
        path.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(catalog_stage, "_write_table", boom)
    with pytest.raises(OSError, match="disk full"):
        stage.run(ctx)
    assert {n: ctx.path(n).read_bytes() for n in saved} == saved
    assert not list(ctx.run_dir.glob("*.part"))


def test_stage_requires_its_inputs(make_ctx: Any, run_section: RunSection) -> None:
    ctx = make_ctx(run_section)
    catalog_stage._write_table(public([], run_section), ctx.path("catalog.parquet"))
    with pytest.raises(FileNotFoundError):
        stage.run(ctx)

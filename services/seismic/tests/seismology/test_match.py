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
import yaml
from hq_contracts.io import columns_for, read_table, to_frame, write_table
from hq_contracts.models import SCHEMA_VERSION, Enu, Pick, SeismicEvent, Station
from pydantic import ValidationError
from pyproj import Proj
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from hq.config.run import Origin, RunSection
from hq.config.seismology import MatchingConfig, SeismologyConfig
from hq.locate.coords import from_enu
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
from hq.match.reasons import (
    ArrivalModel,
    Evidence,
    code_of,
    expected_windows,
    explain_unmatched,
)

pytestmark = pytest.mark.smoke

stage = importlib.import_module("hq.match.run")

RUN = RunSection.model_validate(
    yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs" / "showcase" / "run.yaml").read_text()
    )
)
T0 = 1_789_041_600.25  # 2026-09-10T12:00:00.25Z, inside the showcase window
KM = 1000.0
HYPO_U = -4000.0  # ENU u of every synthetic hypocentre
HYPO = (0.0, 0.0, HYPO_U)  # public hypocentre ENU used by the reason tests
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
NEED = "classifier minimum 4 stations"


# ---------------------------------------------------------------- builders


def _geo(rows: list[tuple[str, float, float, float]], u: float) -> list[tuple[float, ...]]:
    """(lat, lon, elevation m ASL) per (id, t, e, n) row at ENU u, in the showcase frame."""
    if not rows:
        return []
    e = np.array([r[2] for r in rows], dtype=np.float64)
    n = np.array([r[3] for r in rows], dtype=np.float64)
    lat, lon, elev = from_enu(e, n, np.full(e.size, u), RUN.origin)
    return list(zip(lat.tolist(), lon.tolist(), elev.tolist(), strict=True))


def public(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    """Typed CatalogEvent frame from (id, t, e, n); u = HYPO_U, lat/lon/elevM from the ENU."""
    records = []
    for (cid, t, e, n), (lat, lon, elev) in zip(rows, _geo(rows, HYPO_U), strict=True):
        records.append(
            {
                "id": cid,
                "source": "UU via USGS ComCat",
                "t": t,
                "latitude": lat,
                "longitude": lon,
                "depthKm": 2.4,
                "depthDatum": "synthetic test datum",
                "elevM": elev,
                "mag": 1.0,
                "magType": "ml",
                "enu_e": e,
                "enu_n": n,
                "enu_u": HYPO_U,
                "matchedEventId": np.nan,
            }
        )
    frame = pd.DataFrame(records, columns=list(catalog_stage._FRAME_DTYPES))
    return frame.astype(catalog_stage._FRAME_DTYPES)


def located(
    rows: list[tuple[str, float, float, float]], pick_ids: dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """events_located-shaped frame from (id, t, e, n), with pickIds per event."""
    picks = pick_ids or {}
    data: dict[str, list[Any]] = {c: [] for c in LOCATED_COLUMNS}
    for (eid, t, e, n), (lat, lon, elev) in zip(rows, _geo(rows, HYPO_U), strict=True):
        values: dict[str, Any] = {c: 0 for c in LOCATED_COLUMNS}
        values.update(
            {
                "id": eid,
                "runId": "test-run",
                "source": "hq-pipeline",
                "t": t,
                "latitude": lat,
                "longitude": lon,
                "elevM": elev,
                "depthKm": 4.0,
                "enu_e": e,
                "enu_n": n,
                "enu_u": HYPO_U,
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
    return frame.astype(
        {c: "float64" for c in ("t", "latitude", "longitude", "elevM", "enu_e", "enu_n", "enu_u")}
    )


def stations(n: int = 6, radius_m: float = 5000.0, unused: int = 1) -> pd.DataFrame:
    """n used stations on a ring around the origin at u = 0, plus ``unused`` unused ones."""
    rows = []
    for k in range(n + unused):
        az = 2 * np.pi * k / (n + unused)
        e, nn = radius_m * np.cos(az), radius_m * np.sin(az)
        ((lat, lon, elev),) = _geo([("", 0.0, e, nn)], 0.0)
        rows.append(
            Station(
                id=f"XX.S{k:02d}",
                network="XX",
                station=f"S{k:02d}",
                latitude=lat,
                longitude=lon,
                surfaceElevM=elev,
                sensorDepthM=0.0,
                sensorElevM=elev,
                kind="surface",
                channels=["HHZ", "HHN", "HHE"],
                sampleRateHz=100.0,
                enu=Enu(e=e, n=nn, u=0.0),
                preprocessProfile="surface",
                usedInRun=k < n,
            )
        )
    return to_frame(rows, Station)


def pick(sid: str, phase: str, t: float, prob: float = 0.9) -> Pick:
    return Pick(
        id=f"phasenet:{sid}:{phase}:{t:.3f}",
        stationId=sid,
        phase=phase,
        t=t,
        prob=prob,
        picker="phasenet:test",
    )


def _used(sta: pd.DataFrame) -> pd.DataFrame:
    return sta[sta["usedInRun"]]


def arrival_picks(
    sta: pd.DataFrame,
    t0: float,
    hypo: tuple[float, float, float],
    n: int,
    speed: float = 4000.0,
    phase: str = "P",
    prob: float = 0.9,
) -> list[Pick]:
    """Picks at t0 + R / speed on the first ``n`` used stations (a speed inside the model range)."""
    out = []
    for _, row in _used(sta).head(n).iterrows():
        r = float(np.linalg.norm(np.array([row.enu_e, row.enu_n, row.enu_u]) - np.array(hypo)))
        out.append(pick(row.id, phase, t0 + r / speed, prob))
    return out


def picks_frame(picks: list[Pick]) -> pd.DataFrame:
    return to_frame(picks, Pick)


def gaps_frame(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["stationId", "channel", "gapStart", "gapEnd"]).astype(
        {"stationId": "str", "channel": "str", "gapStart": "float64", "gapEnd": "float64"}
    )


def assoc_frame(groups: dict[str, list[str]]) -> pd.DataFrame:
    """assoc_picks rows from {assocId: pick ids}."""
    rows = [(aid, pid) for aid, pids in groups.items() for pid in pids]
    return pd.DataFrame(rows, columns=["assocId", "pickId"]).astype("str")


def arrow_types(df: pd.DataFrame, tmp_path: Path, name: str) -> dict[str, pa.DataType]:
    path = tmp_path / f"{name}.parquet"
    write_table(df, path, name)
    return {f.name: f.type for f in pq.read_schema(path)}


@pytest.fixture
def arrivals(seismology_config: SeismologyConfig) -> ArrivalModel:
    return ArrivalModel.from_model(load_configured_model(seismology_config.velocity))


def _with_matching(cfg: SeismologyConfig, **update: Any) -> SeismologyConfig:
    return cfg.model_copy(update={"matching": cfg.matching.model_copy(update=update)})


def _with_reasons(cfg: SeismologyConfig, **update: Any) -> SeismologyConfig:
    return _with_matching(cfg, reasons=cfg.matching.reasons.model_copy(update=update))


def _scaled(cfg: SeismologyConfig) -> SeismologyConfig:
    """Cost scales that differ from the admissibility limits (limits stay 2 s / 5 km)."""
    return _with_matching(cfg, dtScaleS=0.5, distScaleM=20000.0)


# ---------------------------------------------------------------- config


def test_matching_config_parses_and_rejects_bad_values(
    seismology_config: SeismologyConfig,
) -> None:
    m = seismology_config.matching
    assert (m.dtScaleS, m.distScaleM, m.maxDtS, m.maxDistM) == (2.0, 5000.0, 2.0, 5000.0)
    assert [(p.dtS, p.distM) for p in m.sensitivity] == [
        (1.0, 3000.0),
        (2.0, 5000.0),
        (3.0, 8000.0),
    ]
    assert m.enuConsistencyM == 1.0
    r = m.reasons
    assert (r.minStations, r.minPickProb, r.arrivalPadS, r.maxCandidateDtS) == (4, 0.3, 1.0, 10.0)
    raw = seismology_config.model_dump(mode="json")
    for bad in (
        {**raw, "matching": {**raw["matching"], "unknown": 1}},
        {**raw, "matching": {**raw["matching"], "reasons": {**raw["matching"]["reasons"], "x": 1}}},
    ):
        with pytest.raises(ValidationError, match="extra"):
            SeismologyConfig.model_validate(bad)
    dupes = {**raw["matching"], "sensitivity": [{"dtS": 2.0, "distM": 5000.0}] * 2}
    with pytest.raises(ValidationError, match="unique"):
        SeismologyConfig.model_validate({**raw, "matching": dupes})
    # The sensitivity table must contain the headline tolerance.
    for moved in ({"maxDtS": 2.5}, {"maxDistM": 6000.0}):
        with pytest.raises(ValidationError, match="headline"):
            MatchingConfig.model_validate({**raw["matching"], **moved})


def test_grid_scale_factor_is_negligible_over_the_showcase_bbox(run_section: RunSection) -> None:
    """The module docstring's claim: UTM grid distance is within 1.7e-4 of ellipsoidal distance."""
    proj = Proj("EPSG:32612")
    min_lon, min_lat, max_lon, max_lat = run_section.bbox
    lons, lats = np.meshgrid(np.linspace(min_lon, max_lon, 7), np.linspace(min_lat, max_lat, 7))
    k = np.asarray(proj.get_factors(lons.ravel(), lats.ravel()).meridional_scale)
    assert np.abs(k - 1.0).max() < 1.7e-4  # under 1 m at 5 km


# ---------------------------------------------------------------- assignment

# The reference checks below write the cost and admissibility out from the ticket's formula,
# independent of hq.match.Tolerance, so a wrong rule there cannot agree with itself.


def _reference(
    dt: np.ndarray, dist: np.ndarray, m: MatchingConfig
) -> tuple[np.ndarray, np.ndarray]:
    ok = (np.abs(dt) <= m.maxDtS) & (dist <= m.maxDistM)
    cost = np.abs(dt) / m.dtScaleS + dist / m.distScaleM
    return ok, cost


def _brute_force(dt: np.ndarray, dist: np.ndarray, m: MatchingConfig) -> tuple[int, float]:
    """(most admissible one-to-one pairs, least total cost among those), by enumeration."""
    ok, cost = _reference(dt, dist, m)
    n, n_loc = dt.shape
    best = [0, 0.0]

    def rec(i: int, used: int, count: int, total: float) -> None:
        if i == n:
            if count > best[0] or (count == best[0] and total < best[1]):
                best[0], best[1] = count, total
            return
        rec(i + 1, used, count, total)
        for j in range(n_loc):
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
    lt = loc.set_index("id")
    pt = pub.set_index("id")
    for row in matched.itertuples(index=False):
        dt = lt.at[row.eventId, "t"] - pt.at[row.catalogId, "t"]
        dist = np.hypot(
            lt.at[row.eventId, "enu_e"] - pt.at[row.catalogId, "enu_e"],
            lt.at[row.eventId, "enu_n"] - pt.at[row.catalogId, "enu_n"],
        )
        assert row.dtS == dt and row.distM == dist
        assert _reference(np.array(dt), np.array(dist), cfg.matching)[0]
    return matches


@pytest.mark.parametrize("scaled", [False, True])
def test_one_to_one_property_over_random_scenarios(
    seismology_config: SeismologyConfig, scaled: bool
) -> None:
    """Never a located or public event twice; the count is the maximum possible (Hopcroft-Karp)."""
    cfg = _scaled(seismology_config) if scaled else seismology_config
    rng = np.random.default_rng(20260926)
    for _ in range(150):
        pub, loc = _scenario(rng, int(rng.integers(0, 25)), int(rng.integers(0, 30)))
        matches = _check_one_to_one(pub, loc, cfg)
        pub_sorted = pub.sort_values(["t", "id"]).reset_index(drop=True)
        loc_sorted = loc.sort_values(["t", "id"]).reset_index(drop=True)
        off = pair_offsets(loc_sorted, pub_sorted)
        ok, _ = _reference(off.dt, off.dist, cfg.matching)
        most = int((maximum_bipartite_matching(csr_matrix(ok.astype(np.int8))) >= 0).sum())
        assert int(matches["eventId"].notna().sum()) == most


@pytest.mark.parametrize("scaled", [False, True])
def test_assignment_is_optimal_against_brute_force(
    seismology_config: SeismologyConfig, scaled: bool
) -> None:
    cfg = _scaled(seismology_config) if scaled else seismology_config
    rng = np.random.default_rng(7)
    tol = Tolerance.from_config(cfg.matching)
    for _ in range(100):
        n_pub, n_loc = int(rng.integers(1, 6)), int(rng.integers(1, 7))
        pub, loc = _scenario(rng, n_pub, n_loc)
        off = pair_offsets(loc, pub)
        rows, cols = assign(off, tol)
        count, total = _brute_force(off.dt, off.dist, cfg.matching)
        _, cost = _reference(off.dt, off.dist, cfg.matching)
        assert rows.size == count
        assert float(cost[rows, cols].sum()) == pytest.approx(total, abs=1e-9)


def test_competition_prefers_more_pairs_over_one_cheap_pair(
    seismology_config: SeismologyConfig,
) -> None:
    """a-x is the cheapest pair, but taking it strands b; a-y plus b-x recovers both."""
    pub = public([("a", T0, 0.0, 0.0), ("b", T0 + 3.0, 0.0, 0.0)])
    loc = located([("x", T0 + 1.5, 0.0, 0.0), ("y", T0 - 1.8, 0.0, 0.0)])
    matches = match(loc, pub, seismology_config).matches.set_index("catalogId")
    assert matches.at["a", "eventId"] == "y" and matches.at["b", "eventId"] == "x"


@pytest.mark.parametrize("scaled", [False, True])
def test_more_pairs_win_even_when_their_costs_are_near_the_bound(
    seismology_config: SeismologyConfig, scaled: bool
) -> None:
    """a-x costs 0; a-y and b-x each cost 1.95 s / dtScaleS (3.9 when scaled, near the 4.25
    bound). Taking a-x strands b, so both near-limit pairs must be chosen instead."""
    cfg = _scaled(seismology_config) if scaled else seismology_config
    pub = public([("a", T0, 0.0, 0.0), ("b", T0 + 1.95, 0.0, 0.0)])
    loc = located([("x", T0, 0.0, 0.0), ("y", T0 - 1.95, 0.0, 0.0)])
    matches = match(loc, pub, cfg).matches.set_index("catalogId")
    assert matches.at["a", "eventId"] == "y" and matches.at["b", "eventId"] == "x"


def test_distance_decides_between_equal_time_offsets(seismology_config: SeismologyConfig) -> None:
    """Equal |dt|: the closer event wins, although the farther one sorts first."""
    pub = public([("a", T0, 0.0, 0.0)])
    loc = located([("far", T0 - 1.0, 4000.0, 0.0), ("near", T0 + 1.0, 1000.0, 0.0)])
    assert match(loc, pub, seismology_config).matches.at[0, "eventId"] == "near"


def test_cost_uses_scales_and_admissibility_uses_limits(
    seismology_config: SeismologyConfig,
) -> None:
    pub = public([("a", T0, 0.0, 0.0), ("b", T0 + 100, 0.0, 0.0), ("c", T0 + 200, 0.0, 0.0)])
    loc = located(
        [
            ("x", T0 + 0.2, 4000.0, 0.0),  # default cost 0.9, scaled 0.6
            ("y", T0 + 0.6, 500.0, 0.0),  # default cost 0.4, scaled 1.225
            ("z", T0 + 101.5, 0.0, 0.0),  # |dt| 1.5: over dtScaleS 0.5, within maxDtS 2
            ("w", T0 + 200, 6000.0, 0.0),  # 6 km: within distScaleM 20 km, over maxDistM 5
        ]
    )
    default = match(loc, pub, seismology_config).matches.set_index("catalogId")
    scaled = match(loc, pub, _scaled(seismology_config)).matches.set_index("catalogId")
    assert default.at["a", "eventId"] == "y" and scaled.at["a", "eventId"] == "x"
    for m in (default, scaled):
        assert m.at["b", "eventId"] == "z" and pd.isna(m.at["c", "eventId"])


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


# ---------------------------------------------------------------- arrival windows


def test_straight_ray_time_against_hand_sums() -> None:
    """Two layers: 2000 m/s above 0 m ASL (extending upward), 4000 m/s below (P; S is half)."""
    model = ArrivalModel(
        "two-layer",
        tops=np.array([1000.0, 0.0]),
        vp=np.array([2000.0, 4000.0]),
        vs=np.array([1000.0, 2000.0]),
    )
    rcv = np.array([1500.0, 1500.0, -500.0, 700.0])
    r = np.array([2000.0, np.hypot(2000.0, 3000.0), 3000.0, 3000.0])
    src = -500.0
    vertical = 1500.0 / 2000.0 + 500.0 / 4000.0  # 1500 m above 0 (500 of them above the top)
    expected_p = np.array(
        [
            vertical,  # vertical ray, r = elevation span
            vertical * r[1] / 2000.0,  # same span, oblique: time scales with length
            3000.0 / 4000.0,  # horizontal in the lower layer
            (700.0 / 2000.0 + 500.0 / 4000.0) * r[3] / 1200.0,
        ]
    )
    assert np.allclose(model.straight_ray_s("P", src, rcv, r), expected_p, rtol=1e-12)
    assert np.allclose(model.straight_ray_s("S", src, rcv, r), 2 * expected_p, rtol=1e-12)
    # A horizontal ray above the model top uses the top layer.
    above = model.straight_ray_s("P", 1200.0, np.array([1200.0]), np.array([100.0]))
    assert above[0] == pytest.approx(100.0 / 2000.0, rel=1e-12)


def test_expected_windows_follow_the_formula(
    arrivals: ArrivalModel, seismology_config: SeismologyConfig
) -> None:
    sta = _used(stations())
    enu = sta[["enu_e", "enu_n", "enu_u"]].to_numpy(dtype=np.float64)
    pad = seismology_config.matching.reasons.arrivalPadS
    windows = expected_windows(T0, np.array(HYPO), enu, RUN.origin.elevM, arrivals, pad)
    r = np.sqrt(((enu - np.array(HYPO)) ** 2).sum(axis=1))
    for phase, v in (("P", arrivals.vp), ("S", arrivals.vs)):
        lo, hi = windows[phase]
        assert np.allclose(lo, T0 + r / v.max() - pad, rtol=0, atol=1e-6)
        ray = arrivals.straight_ray_s(
            phase, HYPO_U + RUN.origin.elevM, enu[:, 2] + RUN.origin.elevM, r
        )
        assert np.allclose(hi, T0 + ray + pad, rtol=0, atol=1e-6)
        # The unpadded bracket holds every speed of the model between vMin and vMax.
        assert (r / v.max() <= ray).all() and (ray <= r / v.min()).all()


# ---------------------------------------------------------------- unmatched reasons


def _explain(
    loc: pd.DataFrame,
    pub: pd.DataFrame,
    cfg: SeismologyConfig,
    arrivals: ArrivalModel | None,
    evidence: Evidence | None = None,
    run: RunSection = RUN,
) -> tuple[dict[str, str], Any]:
    result = match(loc, pub, cfg)
    explained = explain_unmatched(result, loc, pub, evidence or Evidence(), cfg, run, arrivals)
    reasons = {
        r.catalogId: r.reason
        for r in explained.matches.itertuples(index=False)
        if pd.isna(r.eventId)
    }
    return reasons, explained


def test_reason_lost_one_to_one(
    seismology_config: SeismologyConfig, arrivals: ArrivalModel
) -> None:
    """x is admissible to both; a is cheaper, so b loses the competition for it."""
    pub = public([("a", T0, 0, 0), ("b", T0 + 1.0, 0, 0)])
    loc = located([("x", T0 + 0.1, 0.0, 0.0)])
    evidence = Evidence(stations=stations(), picks=picks_frame([]))  # would say "too few picks"
    reasons, explained = _explain(loc, pub, seismology_config, arrivals, evidence)
    assert reasons == {
        "b": "lost one-to-one (1 admissible candidate(s), each assigned to another public event; "
        "best x: dt -0.90 s, 0.00 km, assigned to a)"
    }
    assert explained.codes == {"b": "lostOneToOne"}


def test_reason_outside_window_and_bbox(
    seismology_config: SeismologyConfig, arrivals: ArrivalModel
) -> None:
    early = RUN.window_start_s - 0.5
    pub = public([("early", early, 0, 0), ("east", T0, 0, 0)])
    pub.loc[pub["id"] == "east", "longitude"] = RUN.bbox[2] + 0.01
    reasons, explained = _explain(located([]), pub, seismology_config, arrivals)
    assert reasons["early"] == (
        "outside run window (origin 2026-09-09T23:59:59.500Z not in "
        "[2026-09-10T00:00:00.000Z, 2026-09-11T00:00:00.000Z))"
    )
    assert reasons["east"].startswith("outside run bbox (lat 38.5100, lon -112.5900 not in ")
    assert explained.codes == {"early": "outsideWindow", "east": "outsideBbox"}


def test_reason_no_evidence_keeps_nearest_candidate(
    seismology_config: SeismologyConfig, arrivals: ArrivalModel
) -> None:
    pub = public([("a", T0, 0, 0)])
    loc = located([("x", T0 + 3.4, 7900.0, 0.0), ("y", T0 + 500.0, 0.0, 0.0)])
    reasons, explained = _explain(loc, pub, seismology_config, None)
    assert reasons == {"a": "no candidate within 2 s / 5 km (nearest: dt +3.40 s, 7.90 km)"}
    assert explained.checks_run == ["window", "bbox", "oneToOne", "tolerance"]
    assert explained.checks_skipped == {
        "arrivalWindows": "missing stations",
        "waveformData": "missing stations, gaps",
        "picks": "missing stations, picks",
        "association": "missing stations, picks, assoc_picks",
        "location": "missing stations, picks",
    }


def test_reason_arrivals_outside_run_window(
    seismology_config: SeismologyConfig, arrivals: ArrivalModel
) -> None:
    """An origin 0.5 s before windowEnd: every arrival window lies past the processed window."""
    end = RUN.window_end_s
    pub = public([("late", end - 0.5, 0, 0), ("inside", end - 60.0, 0, 0)])
    sta = stations(radius_m=20000.0)
    used = _used(sta)["id"].tolist()
    # Gap rows cover only the downloaded window, as H1 writes them: here every channel is gapped
    # over the last 2 minutes, so the in-window event has no data either.
    gaps = gaps_frame([(s, c, end - 120.0, end) for s in used for c in ("HHZ", "HHN", "HHE")])
    for evidence in (
        Evidence(stations=sta),
        Evidence(stations=sta, gaps=gaps, picks=picks_frame([])),
    ):
        reasons, explained = _explain(located([]), pub, seismology_config, arrivals, evidence)
        assert reasons["late"] == (
            "arrivals outside run window (expected arrival windows of 0 of 6 used stations "
            f"overlap [2026-09-10T00:00:00.000Z, 2026-09-11T00:00:00.000Z); {NEED})"
        )
        assert explained.codes["late"] == "arrivalsOutsideWindow"
    assert reasons["inside"] == (
        f"no waveform data (data on 0 of 6 used stations in the expected arrival windows; {NEED})"
    )


def test_reason_no_waveform_data_and_partial_gaps(
    seismology_config: SeismologyConfig, arrivals: ArrivalModel
) -> None:
    pub = public([("a", T0, 0, 0)])
    sta = stations()
    used = _used(sta)
    ids = used["id"].tolist()
    enu = used[["enu_e", "enu_n", "enu_u"]].to_numpy(dtype=np.float64)
    pad = seismology_config.matching.reasons.arrivalPadS
    win = expected_windows(T0, np.array(HYPO), enu, RUN.origin.elevM, arrivals, pad)
    (p_lo, p_hi), (s_lo, s_hi) = win["P"], win["S"]
    channels = ("HHZ", "HHN", "HHE")
    rows = []
    # ids[0]: a gap over the whole S window, which covers only the later part of the (earlier,
    # overlapping) P window: the start of the P window is data.
    assert p_lo[0] < s_lo[0] < p_hi[0]
    rows += [(ids[0], c, s_lo[0], s_hi[0] + 1) for c in channels]
    # ids[1]: the P window gapped, the end of the S window not: data.
    assert s_hi[1] > p_hi[1] + 1e-3
    rows += [(ids[1], c, p_lo[1] - 1, p_hi[1] + 1e-3) for c in channels]
    # ids[2]: fully gapped, with two touching gaps on one channel.
    rows += [(ids[2], c, T0 - 100, T0 + 100) for c in ("HHZ", "HHN")]
    rows += [(ids[2], "HHE", T0 - 100, T0 + 3.0), (ids[2], "HHE", T0 + 3.0, T0 + 100)]
    # ids[3..5]: fully gapped.
    rows += [(s, c, T0 - 100, T0 + 100) for s in ids[3:] for c in channels]
    evidence = Evidence(stations=sta, gaps=gaps_frame(rows))
    reasons, explained = _explain(located([]), pub, seismology_config, arrivals, evidence)
    assert reasons == {
        "a": f"no waveform data (data on 2 of 6 used stations in the expected arrival windows; "
        f"{NEED})"
    }
    assert "waveformData" in explained.checks_run
    # Ungap two stations: 4 with data, so the check passes; picks were not given.
    evidence = Evidence(stations=sta, gaps=gaps_frame([r for r in rows if r[0] not in ids[4:]]))
    reasons, _ = _explain(located([]), pub, seismology_config, arrivals, evidence)
    assert reasons["a"].startswith("no candidate within")


def test_reason_too_few_picks_and_below_threshold(
    seismology_config: SeismologyConfig, arrivals: ArrivalModel
) -> None:
    pub = public([("a", T0, 0, 0)])
    sta = stations()
    unused = sta.loc[~sta["usedInRun"], "id"].item()
    used = _used(sta)["id"].tolist()
    picks = arrival_picks(sta, T0, HYPO, n=3)
    picks += [pick(unused, "P", T0 + 1.6)]  # an unused station never counts
    picks += [pick(used[4], "P", T0 - 30.0), pick(used[4], "S", T0 + 40.0)]  # outside windows
    picks += [pick(used[5], "S", T0 + 0.5)]  # an S pick in the P window, before the S window
    evidence = Evidence(stations=sta, gaps=gaps_frame([]), picks=picks_frame(picks))
    reasons, _ = _explain(located([]), pub, seismology_config, arrivals, evidence)
    assert reasons == {
        "a": "too few picks (picks in the expected arrival windows on 3 of 6 used stations, 3 of "
        f"them at prob >= 0.3; {NEED})"
    }
    # Two more stations picked, below minPickProb: 5 stations in all, 3 at the threshold.
    low = [pick(used[4], "S", T0 + 3.0, prob=0.2), pick(used[5], "P", T0 + 1.6, prob=0.25)]
    evidence = Evidence(stations=sta, picks=picks_frame(picks + low))
    reasons, explained = _explain(located([]), pub, seismology_config, arrivals, evidence)
    assert reasons == {
        "a": "picks below threshold (picks in the expected arrival windows on 5 of 6 used "
        f"stations, 3 of them at prob >= 0.3; {NEED})"
    }
    assert explained.codes == {"a": "picksBelowThreshold"}
    # The same picks at the threshold pass the check.
    at = [pick(used[4], "S", T0 + 3.0, prob=0.3)]
    evidence = Evidence(stations=sta, picks=picks_frame(picks + at))
    reasons, _ = _explain(located([]), pub, seismology_config, arrivals, evidence)
    assert reasons["a"].startswith("no candidate within")


def test_window_edges_and_pad(seismology_config: SeismologyConfig, arrivals: ArrivalModel) -> None:
    """Picks exactly on a window edge count; 1 ms outside do not; the pad knob moves the edges."""
    pub = public([("a", T0, 0, 0)])
    sta = stations()
    used = _used(sta)
    ids = used["id"].tolist()
    enu = used[["enu_e", "enu_n", "enu_u"]].to_numpy(dtype=np.float64)

    def edge_picks(pad: float, nudge: float) -> list[Pick]:
        win = expected_windows(T0, np.array(HYPO), enu, RUN.origin.elevM, arrivals, pad)
        (p_lo, p_hi), (s_lo, s_hi) = win["P"], win["S"]
        return [
            pick(ids[0], "P", p_lo[0] - nudge),
            pick(ids[1], "P", p_hi[1] + nudge),
            pick(ids[2], "S", s_lo[2] - nudge),
            pick(ids[3], "S", s_hi[3] + nudge),
        ]

    def reason(picks: list[Pick], cfg: SeismologyConfig) -> str:
        evidence = Evidence(stations=sta, picks=picks_frame(picks))
        return _explain(located([]), pub, cfg, arrivals, evidence)[0]["a"]

    pad = seismology_config.matching.reasons.arrivalPadS
    assert reason(edge_picks(pad, 0.0), seismology_config).startswith("no candidate within")
    assert reason(edge_picks(pad, 1e-3), seismology_config).startswith(
        "too few picks (picks in the expected arrival windows on 0 of 6 used stations"
    )
    # Without the pad, picks on the padded edges fall outside.
    no_pad = _with_reasons(seismology_config, arrivalPadS=0.0)
    assert reason(edge_picks(pad, 0.0), no_pad).startswith("too few picks (")
    assert reason(edge_picks(0.0, 0.0), no_pad).startswith("no candidate within")


def test_reason_picks_not_associated_counts_one_associated_event(
    seismology_config: SeismologyConfig, arrivals: ArrivalModel
) -> None:
    pub = public([("a", T0, 0, 0)])
    sta = stations()
    picks = arrival_picks(sta, T0, HYPO, n=6)
    ids = [p.id for p in picks]
    for groups, most in (({"a1": ids[:2]}, 2), ({"a1": ids[:3], "a2": ids[3:]}, 3)):
        evidence = Evidence(stations=sta, picks=picks_frame(picks), assoc_picks=assoc_frame(groups))
        reasons, explained = _explain(located([]), pub, seismology_config, arrivals, evidence)
        assert reasons == {
            "a": f"picks not associated (at most {most} of the 6 stations with picks in the "
            f"expected arrival windows share one associated event; {NEED})"
        }
        assert explained.codes == {"a": "picksNotAssociated"}


def test_reason_associated_not_located_and_candidate_time_limit(
    seismology_config: SeismologyConfig, arrivals: ArrivalModel
) -> None:
    pub = public([("a", T0, 0, 0)])
    sta = stations()
    picks = arrival_picks(sta, T0, HYPO, n=5)
    ids = [p.id for p in picks]
    evidence = Evidence(
        stations=sta,
        gaps=gaps_frame([]),
        picks=picks_frame(picks),
        assoc_picks=assoc_frame({"a1": ids}),
    )
    not_located = (
        "associated, not located (associated event a1 holds picks in the expected arrival "
        "windows from 5 stations; no located event within 10 s of the public origin carries such "
        "picks from 4 stations)"
    )
    # Unrelated located event; and one carrying these picks but 30 s off (beyond maxCandidateDtS).
    for loc in (
        located([("x", T0 + 900.0, 0.0, 0.0)], {"x": ["phasenet:other:P:1.000"]}),
        located([("late", T0 + 30.0, 0.0, 0.0)], {"late": ids}),
    ):
        reasons, explained = _explain(loc, pub, seismology_config, arrivals, evidence)
        assert reasons == {"a": not_located}
        assert explained.checks_skipped == {}
    # Within maxCandidateDtS, the carrier is the located candidate.
    loc = located([("near", T0 + 5.0, 0.0, 0.0)], {"near": ids})
    reasons, _ = _explain(loc, pub, seismology_config, arrivals, evidence)
    assert reasons["a"] == (
        "located out of tolerance 2 s / 5 km (nearest associated candidate near: dt +5.00 s, "
        "0.00 km; picks in the expected arrival windows from 5 stations)"
    )


def test_reason_located_out_of_tolerance(
    seismology_config: SeismologyConfig, arrivals: ArrivalModel
) -> None:
    pub = public([("a", T0, 0, 0)])
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
    evidence = Evidence(
        stations=sta, picks=picks_frame(picks), assoc_picks=assoc_frame({"a1": ids})
    )
    reasons, explained = _explain(loc, pub, seismology_config, arrivals, evidence)
    assert reasons == {
        "a": "located out of tolerance 2 s / 5 km (nearest associated candidate carrier-far: "
        "dt +0.30 s, 7.00 km; picks in the expected arrival windows from 5 stations)"
    }
    # Without assoc_picks the chain goes picks -> location and still finds the carrier.
    evidence = Evidence(stations=sta, picks=picks_frame(picks))
    reasons, explained = _explain(loc, pub, seismology_config, arrivals, evidence)
    assert reasons["a"].startswith("located out of tolerance 2 s / 5 km (nearest associated ")
    assert explained.checks_skipped == {
        "waveformData": "missing gaps",
        "association": "missing assoc_picks",
    }


@pytest.mark.parametrize("dt_s", [8.0, 3.0, -3.0])
def test_neighbouring_event_picks_are_not_credited(
    seismology_config: SeismologyConfig, arrivals: ArrivalModel, dt_s: float
) -> None:
    """Review reproduction: the public event has no picks; an uncatalogued event X at the same
    epicentre, dt_s later, is picked (P 5 km/s, S 3 km/s), associated and located on all 6
    stations of a 20 km ring. X's picks must not make the public event 'located out of
    tolerance': it was never picked."""
    pub = public([("a", T0, 0, 0)])
    sta = stations(radius_m=20000.0)
    picks = arrival_picks(sta, T0 + dt_s, HYPO, n=6, speed=5000.0, phase="P")
    picks += arrival_picks(sta, T0 + dt_s, HYPO, n=6, speed=3000.0, phase="S")
    ids = [p.id for p in picks]
    loc = located([("X", T0 + dt_s, 0.0, 0.0)], {"X": ids})
    evidence = Evidence(
        stations=sta,
        gaps=gaps_frame([]),
        picks=picks_frame(picks),
        assoc_picks=assoc_frame({"aX": ids}),
    )
    reasons, explained = _explain(loc, pub, seismology_config, arrivals, evidence)
    assert reasons == {
        "a": "too few picks (picks in the expected arrival windows on 0 of 6 used stations, 0 of "
        f"them at prob >= 0.3; {NEED})"
    }
    assert explained.codes == {"a": "tooFewPicks"}


def test_reason_no_carrier_without_association_keeps_base(
    seismology_config: SeismologyConfig, arrivals: ArrivalModel
) -> None:
    pub = public([("a", T0, 0, 0)])
    sta = stations()
    evidence = Evidence(stations=sta, picks=picks_frame(arrival_picks(sta, T0, HYPO, n=6)))
    reasons, _ = _explain(located([]), pub, seismology_config, arrivals, evidence)
    assert reasons == {"a": "no candidate within 2 s / 5 km (no located events)"}


def test_every_reason_has_one_stable_prefix() -> None:
    prefixes = list(REASONS.values())
    for p in prefixes:
        assert sum(q.startswith(p) or p.startswith(q) for q in prefixes) == 1
    assert code_of("too few picks (picks on 0 of 0 used stations)") == "tooFewPicks"
    assert code_of("arrivals outside run window (x)") == "arrivalsOutsideWindow"
    assert code_of("outside run window (x)") == "outsideWindow"
    with pytest.raises(ValueError):
        code_of("something else")


def _bad_evidence(case: str) -> tuple[pd.DataFrame, pd.DataFrame, Evidence, bool]:
    sta = stations()
    picks = picks_frame(arrival_picks(sta, T0, HYPO, n=6))
    pub = public([("a", T0, 0, 0)])
    loc = located([])
    with_arrivals = True
    if case == "duplicate station ids":
        sta = pd.concat([sta, sta.iloc[[0]]], ignore_index=True)
    elif case == "non-finite t":
        picks.loc[0, "t"] = np.nan
    elif case == "null pickIds":
        loc = located([("x", T0 + 900.0, 0.0, 0.0)])
        loc["pickIds"] = pd.Series([None], dtype=object)
    elif case == "non-finite enu_u":
        pub.loc[0, "enu_u"] = np.nan
    elif case == "null assocId":
        return (
            pub,
            loc,
            Evidence(
                stations=sta,
                picks=picks,
                assoc_picks=pd.DataFrame({"assocId": [None], "pickId": ["p"]}, dtype=object),
            ),
            True,
        )
    elif case == "without an arrival model":
        with_arrivals = False
    return pub, loc, Evidence(stations=sta, picks=picks), with_arrivals


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate station ids", "stations: duplicate ids \\['XX.S00'\\]"),
        ("non-finite t", "picks: non-finite t for"),
        ("null pickIds", "events_located: null pickIds for \\['x'\\]"),
        ("non-finite enu_u", "catalog: non-finite enu_u for \\['a'\\]"),
        ("null assocId", "assoc_picks: null assocId or pickId"),
        ("without an arrival model", "stations given without an arrival model"),
    ],
)
def test_bad_evidence_fails_loudly(
    seismology_config: SeismologyConfig, arrivals: ArrivalModel, case: str, message: str
) -> None:
    pub, loc, evidence, with_arrivals = _bad_evidence(case)
    result = match(loc, pub, seismology_config)
    with pytest.raises(ValueError, match=message):
        explain_unmatched(
            result,
            loc,
            pub,
            evidence,
            seismology_config,
            RUN,
            arrivals if with_arrivals else None,
        )


# ---------------------------------------------------------------- stage


def _write_run(ctx: Any, pub: pd.DataFrame, loc: pd.DataFrame, **evidence: pd.DataFrame) -> None:
    catalog_stage.write_catalog(pub, ctx.path("catalog.parquet"))
    write_table(loc, ctx.path("events_located.parquet"), "SeismicEvent")
    names = {"stations": "Station", "gaps": "Gap", "picks": "Pick", "assoc_picks": "AssocPick"}
    for key, frame in evidence.items():
        write_table(frame, ctx.path(f"{key}.parquet"), names[key])


def _assert_catalog_metadata(path: Path) -> None:
    metadata = pq.read_schema(path).metadata
    assert metadata[b"model"] == b"CatalogEvent"
    assert metadata[b"schemaVersion"] == SCHEMA_VERSION.encode()
    assert pq.read_schema(path).remove_metadata().equals(catalog_stage._CATALOG_SCHEMA)


def test_stage_writes_tables_fills_matched_ids_and_records(
    make_ctx: Any,
    run_section: RunSection,
    seismology_config: SeismologyConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = make_ctx(run_section)
    pub = public([("uu1", T0, 0, 0), ("uu2", T0 + 100, 0, 0), ("uu3", T0 + 200, 0, 0)])
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
    _assert_catalog_metadata(ctx.path("catalog.parquet"))
    # Same fields and types; the pandas metadata block may differ, the contract keys may not.
    assert after.schema.remove_metadata().equals(before.schema.remove_metadata())
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
    assert params["checksRun"] == [
        "window",
        "bbox",
        "oneToOne",
        "tolerance",
        "arrivalWindows",
        "picks",
        "location",
    ]
    assert set(params["checksSkipped"]) == {"waveformData", "association"}
    assert params["sensitivity"][1] == {"dtS": 2.0, "distM": 5000.0, "recovered": 2}
    assert set(params["unmatched"]) == {"uu3"}
    assert params["arrivalWindows"]["padS"] == 1.0
    chance = params["chanceTimeCoincidences"]
    assert chance["perPublicEvent"] == pytest.approx(2 * 2 * 2.0 / 86400.0)
    assert chance["total"] == pytest.approx(3 * chance["perPublicEvent"])

    text = caplog.text
    assert (
        "recovered 2 / 3 public regional catalog events in window [2026-09-10T00:00:00Z, " in text
    )
    assert "expected chance time coincidences within +/-2 s" in text
    assert "unmatched uu3: " in text
    assert "dt 3 s, dist 8 km: recovered 2 / 3" in text


def test_stage_reasons_from_run_dir_evidence(
    make_ctx: Any, run_section: RunSection, seismology_config: SeismologyConfig
) -> None:
    ctx = make_ctx(run_section)
    pub = public([("uu1", T0, 0, 0)])
    sta = stations()
    picks = arrival_picks(sta, T0, HYPO, n=6)
    _write_run(
        ctx,
        pub,
        located([]),
        stations=sta,
        gaps=gaps_frame([]),
        picks=picks_frame(picks),
        assoc_picks=assoc_frame({"a1": [p.id for p in picks[:1]]}),
    )
    stage.run(ctx)
    matches = read_table(ctx.path("matches.parquet"))
    assert matches.at[0, "reason"].startswith("picks not associated (at most 1 of the 6 ")
    (record,) = ctx.records
    assert record["counts"]["unmatched.picksNotAssociated"] == 1
    assert record["params"]["match"]["checksSkipped"] == {}
    assert read_table(ctx.path("catalog.parquet"))["matchedEventId"].isna().all()


def test_stage_without_stations_does_not_load_the_velocity_model(
    monkeypatch: pytest.MonkeyPatch, make_ctx: Any, run_section: RunSection
) -> None:
    ctx = make_ctx(run_section)
    _write_run(ctx, public([("uu1", T0, 0, 0)]), located([]))

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("velocity model loaded without a window check")

    monkeypatch.setattr(stage, "load_configured_model", refuse)
    stage.run(ctx)
    (record,) = ctx.records
    assert record["params"]["match"]["arrivalWindows"] is None


def test_stage_rerun_overwrites_and_write_failure_leaves_previous_files(
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
) -> None:
    ctx = make_ctx(run_section)
    pub = public([("uu1", T0, 0, 0)])
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

    monkeypatch.setattr(catalog_stage, "write_catalog", boom)
    with pytest.raises(OSError, match="disk full"):
        stage.run(ctx)
    assert {n: ctx.path(n).read_bytes() for n in saved} == saved
    assert not list(ctx.run_dir.glob("*.part"))


def test_stage_failure_between_moves_is_logged(
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = make_ctx(run_section)
    _write_run(ctx, public([("uu1", T0, 0, 0)]), located([]))
    stage.run(ctx)
    saved = {n: ctx.path(n).read_bytes() for n in ("catalog.parquet", "match_sensitivity.parquet")}
    old_matches = ctx.path("matches.parquet").read_bytes()
    write_table(located([("hq-1", T0, 0, 0)]), ctx.path("events_located.parquet"), "SeismicEvent")

    real_replace = stage.os.replace
    calls: list[str] = []

    def flaky(src: Path, dst: Path) -> None:
        calls.append(Path(dst).name)
        if len(calls) == 2:
            raise OSError("rename failed")
        real_replace(src, dst)

    monkeypatch.setattr(stage.os, "replace", flaky)
    with caplog.at_level(logging.ERROR, logger=stage.__name__), pytest.raises(OSError):
        stage.run(ctx)
    # matches.parquet was replaced; the rest are the previous files. catalog.parquet moves last.
    assert calls == ["matches.parquet", "match_sensitivity.parquet"]
    assert ctx.path("matches.parquet").read_bytes() != old_matches
    assert {n: ctx.path(n).read_bytes() for n in saved} == saved
    assert "mixes new and previous match outputs" in caplog.text
    assert "['matches.parquet']" in caplog.text
    assert not list(ctx.run_dir.glob("*.part"))


def test_stage_zero_rows_keep_types(make_ctx: Any, run_section: RunSection) -> None:
    ctx = make_ctx(run_section)
    _write_run(ctx, public([]), located([]))
    stage.run(ctx)
    assert {f.name: f.type for f in pq.read_schema(ctx.path("matches.parquet"))} == MATCH_ARROW
    sens = ctx.path("match_sensitivity.parquet")
    assert {f.name: f.type for f in pq.read_schema(sens)} == SENSITIVITY_ARROW
    assert read_table(sens)["recovered"].tolist() == [0, 0, 0]
    _assert_catalog_metadata(ctx.path("catalog.parquet"))
    assert read_table(ctx.path("catalog.parquet")).empty


def test_stage_requires_its_inputs(make_ctx: Any, run_section: RunSection) -> None:
    ctx = make_ctx(run_section)
    catalog_stage.write_catalog(public([]), ctx.path("catalog.parquet"))
    with pytest.raises(FileNotFoundError):
        stage.run(ctx)
    ctx.path("catalog.parquet").unlink()
    write_table(located([]), ctx.path("events_located.parquet"), "SeismicEvent")
    with pytest.raises(FileNotFoundError):
        stage.run(ctx)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("catalog model", "holds 'Pick', not CatalogEvent"),
        ("stations model", "holds 'Pick' rows, not Station"),
        ("picks model", "holds 'Station' rows, not Pick"),
        ("final events", "carries final-event columns \\['tier'\\]"),
        ("catalog frame", "catalog: stored ENU differs"),
        ("located frame", "events_located: stored ENU differs"),
        ("stations frame", "stations: stored ENU differs"),
        ("shifted origin", "catalog: stored ENU differs"),
    ],
)
def test_stage_rejects_malformed_inputs(
    make_ctx: Any, run_section: RunSection, case: str, message: str
) -> None:
    pub = public([("uu1", T0, 0, 0)])
    loc = located([("hq-1", T0, 0, 0)])
    sta = stations()
    run = run_section
    if case == "shifted origin":  # every table in the showcase frame, the run config elsewhere
        run = run_section.model_copy(update={"origin": Origin(lat=38.52, lon=-112.9, elevM=1627.7)})
    ctx = make_ctx(run)
    _write_run(ctx, pub, loc)
    if case == "catalog model":
        write_table(pub, ctx.path("catalog.parquet"), "Pick")
    elif case == "stations model":
        write_table(sta, ctx.path("stations.parquet"), "Pick")
    elif case == "picks model":
        write_table(picks_frame([]), ctx.path("picks.parquet"), "Station")
    elif case == "final events":
        write_table(loc.assign(tier="A"), ctx.path("events_located.parquet"), "SeismicEvent")
    elif case == "catalog frame":
        catalog_stage.write_catalog(pub.assign(enu_e=2.0), ctx.path("catalog.parquet"))
    elif case == "located frame":
        write_table(loc.assign(enu_n=-5.0), ctx.path("events_located.parquet"), "SeismicEvent")
    elif case == "stations frame":
        write_table(sta.assign(enu_u=10.0), ctx.path("stations.parquet"), "Station")
    with pytest.raises(ValueError, match=message):
        stage.run(ctx)
    assert not ctx.path("matches.parquet").exists()

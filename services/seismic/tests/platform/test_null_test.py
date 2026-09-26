"""VAL-02 acceptance: the null test reruns associate -> locate -> match -> assign_tiers on picks
whose stations were shifted independently, finds fewer chance events than the coherent picks,
is seeded and reproducible, and the ``validate`` stage writes ``null_test.json`` (always) and
``validation.json`` (when H2's synthetic test exists) and records counts. Offline: picks come
from a few planted synthetic events built here, and H2's four functions are a toy associator
that clusters P arrival times, injected as a ``SeismologyApi`` or as fake ``hq.*`` modules."""

import json
import logging
import math
import shutil
import sys
import types
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from hq_contracts import models as m
from hq_contracts.io import from_frame, read_table, to_frame, write_models

from hq import cli, runs
from hq.config.run import RunSection
from hq.config.validate import NullTestConfig, POnlyAssociatorConfig
from hq.validate import (
    NULL_TEST_JSON,
    VALIDATION_JSON,
    LaneSeismologyApi,
    ValidateError,
    real_seismology_api,
    run_null_test,
)
from hq.validate import run as validate_stage
from hq.validate.null_test import (
    null_shuffles,
    profile_config,
    rerun_pipeline,
    select_profile,
    shift_picks,
    station_shifts,
)

pytestmark = pytest.mark.smoke

SHOWCASE_DIR = Path(__file__).resolve().parents[2] / "configs" / "showcase"
NOW = datetime(2026, 9, 10, 6, 7, tzinfo=UTC)
SEED = 11
STAGE_TEST_SHUFFLES = 4  # reruns the stage tests configure (the showcase config uses more)
RUN_ID = "test-run"

# --- synthetic picks (test-local; CLAUDE.md rule 5 allows small synthetic data in tests) ------
N_STATIONS = 10
RING_RADIUS_M = 6_000.0
N_EVENTS = 12
EVENT_SPACING_S = 20.0  # dense enough that +/-30 s shifts mix picks of neighbouring events
EVENT_DEPTH_M = 2_500.0
VP_M_S, VS_M_S = 5_500.0, 3_200.0
PICK_SIGMA_S = {"P": 0.03, "S": 0.06}
PICKER = "phasenet:test"
CATALOG_EVENT_INDICES = (0, 4, 8)
M_PER_DEG_LAT = 111_320.0

# --- toy associator knobs -----------------------------------------------------------------------
WINDOW_S = 3.0  # P picks of distinct stations within this window form a cluster
S_WINDOW_S = 3.0  # S picks accepted up to this long after the P window
MIN_STATIONS = 4
TIER_A_MIN_STATIONS = 7
TIER_B_MIN_STATIONS = 5
MATCH_DT_S = 2.0

ASSOC_EVENT_COLUMNS = ("assocId", "t", "latitude", "longitude", "elevM", "nPicks", "nP", "nS")
ASSOC_PICK_COLUMNS = ("assocId", "pickId")
ARRIVAL_COLUMNS = (
    "eventId", "stationId", "phase", "tPred", "tObs", "residualS", "pickId", "usedInLocation",
)  # fmt: skip
STATICS_COLUMNS = ("stationId", "phase", "staticS", "nEvents")
MATCH_COLUMNS = ("catalogId", "eventId", "dtS", "distM", "reason")
SENSITIVITY_COLUMNS = ("dtS", "distM", "recovered")
LOCATED_DROPPED_PREFIXES = ("tier", "catalogMatch_", "magnitude_")


def install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


# --- fixtures: config, run context, synthetic tables -------------------------------------------


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A copy of every YAML in configs/showcase (seismology.yaml included: the reruns take the
    real ``SeismologyConfig``)."""
    target = tmp_path / "config"
    target.mkdir()
    for path in sorted(SHOWCASE_DIR.glob("*.yaml")):
        shutil.copy(path, target / path.name)
    # Fewer reruns than the showcase config: the stage tests exercise the wiring, not the
    # statistics, and the lane's smoke suite must stay under 30 s (CLAUDE.md rule 10).
    validate = yaml.safe_load((target / "validate.yaml").read_text(encoding="utf-8"))
    validate["nullTest"]["nShuffles"] = STAGE_TEST_SHUFFLES
    (target / "validate.yaml").write_text(yaml.safe_dump(validate), encoding="utf-8")
    return target


@pytest.fixture
def ctx(config_dir: Path, tmp_path: Path) -> runs.RunContext:
    return runs.create_run(config_dir, tmp_path / "data", now=NOW)


@pytest.fixture
def section(ctx: runs.RunContext) -> RunSection:
    return ctx.config.run


@dataclass(frozen=True)
class SyntheticPicks:
    stations: list[m.Station]
    picks: list[m.Pick]
    catalog: list[m.CatalogEvent]
    event_times: list[float]

    @property
    def stations_frame(self) -> pd.DataFrame:
        return to_frame(self.stations, m.Station)

    @property
    def picks_frame(self) -> pd.DataFrame:
        return to_frame(self.picks, m.Pick)

    @property
    def catalog_frame(self) -> pd.DataFrame:
        return to_frame(self.catalog, m.CatalogEvent)


def make_station(section: RunSection, index: int) -> m.Station:
    angle = 2.0 * math.pi * index / N_STATIONS
    e, n = RING_RADIUS_M * math.sin(angle), RING_RADIUS_M * math.cos(angle)
    lat = section.origin.lat + n / M_PER_DEG_LAT
    lon = section.origin.lon + e / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
    surface = section.refSurfaceElevM + 25.0 * index
    return m.Station(
        id=f"XT.S{index + 1:02d}",
        network="XT",
        station=f"S{index + 1:02d}",
        latitude=lat,
        longitude=lon,
        surfaceElevM=surface,
        sensorDepthM=0.0,
        sensorElevM=surface,
        kind="surface",
        channels=["HHZ", "HHN", "HHE"],
        sampleRateHz=100.0,
        enu=m.Enu(e=e, n=n, u=surface - section.origin.elevM),
        preprocessProfile="surface",
        usedInRun=True,
    )


def build_picks(section: RunSection, seed: int = SEED) -> SyntheticPicks:
    """Planted events at the origin, every station picking P and S with small noise."""
    rng = np.random.default_rng(seed)
    stations = [make_station(section, i) for i in range(N_STATIONS)]
    elev = section.refSurfaceElevM - EVENT_DEPTH_M
    picks: list[m.Pick] = []
    event_times: list[float] = []
    for k in range(N_EVENTS):
        t = section.window_start_s + 3600.0 + EVENT_SPACING_S * k + float(rng.uniform(0.0, 5.0))
        t = round(t, 3)
        event_times.append(t)
        for st in stations:
            r = math.sqrt(st.enu.e**2 + st.enu.n**2 + (st.sensorElevM - elev) ** 2)
            for phase, v in (("P", VP_M_S), ("S", VS_M_S)):
                t_pick = round(t + r / v + float(rng.normal(0.0, PICK_SIGMA_S[phase])), 3)
                picks.append(
                    m.Pick(
                        id=f"{PICKER}:{st.id}:{phase}:{t_pick:.3f}",
                        stationId=st.id,
                        phase=phase,
                        t=t_pick,
                        prob=round(float(rng.uniform(0.6, 0.99)), 3),
                        picker=PICKER,
                    )
                )
    catalog = [
        m.CatalogEvent(
            id=f"testpub{j + 1:03d}",
            source="synthetic public catalog (test)",
            t=round(event_times[idx] + float(rng.uniform(-0.2, 0.2)), 3),
            latitude=section.origin.lat,
            longitude=section.origin.lon,
            depthKm=-elev / 1000.0,
            depthDatum="sea level (test)",
            elevM=elev,
            mag=round(float(rng.uniform(0.5, 2.0)), 2),
            magType="ML",
            enu=m.Enu(e=0.0, n=0.0, u=elev - section.origin.elevM),
        )
        for j, idx in enumerate(CATALOG_EVENT_INDICES)
    ]
    return SyntheticPicks(stations, picks, catalog, event_times)


@pytest.fixture
def synthetic(section: RunSection) -> SyntheticPicks:
    return build_picks(section)


# --- the toy seismology API ---------------------------------------------------------------------


@dataclass(frozen=True)
class FakeAssocResult:
    events: pd.DataFrame
    picks: pd.DataFrame


@dataclass(frozen=True)
class FakeLocateResult:
    events: pd.DataFrame
    arrivals: pd.DataFrame
    statics: pd.DataFrame


@dataclass(frozen=True)
class FakeMatchResult:
    matches: pd.DataFrame
    sensitivity: pd.DataFrame


@dataclass(frozen=True)
class FakeTierResult:
    events: pd.DataFrame
    tiering: dict[str, Any]


@dataclass
class FakeSeismologyApi:
    """H2's four calls over a toy associator: P picks of at least ``MIN_STATIONS`` distinct
    stations within ``WINDOW_S`` form an event, so coherent picks give one event per planted
    event and time-scrambled picks give only chance coincidences. Frames carry the docs/02 §2
    columns (the final events are validated as ``SeismicEvent`` rows)."""

    run_id: str = RUN_ID
    fail_in: str | None = None  # name of the call that raises, to test error propagation
    calls: dict[str, int] = field(
        default_factory=lambda: {"associate": 0, "locate": 0, "match": 0, "assign_tiers": 0}
    )
    phases_seen: set[str] = field(default_factory=set)
    cfgs_seen: list[Any] = field(default_factory=list)  # the SeismologyConfig each associate got

    def _tick(self, name: str) -> None:
        self.calls[name] += 1
        if self.fail_in == name:
            raise RuntimeError(f"boom in {name}")

    def associate(
        self, picks: pd.DataFrame, stations: pd.DataFrame, cfg: Any, run: RunSection
    ) -> FakeAssocResult:
        self._tick("associate")
        self.phases_seen.update(picks["phase"].astype(str))
        self.cfgs_seen.append(cfg)
        p = picks[picks["phase"] == "P"].sort_values(["t", "id"]).reset_index(drop=True)
        s = picks[picks["phase"] == "S"]
        t = p["t"].to_numpy(dtype=np.float64)
        sid = p["stationId"].astype(str).to_numpy()
        pid = p["id"].astype(str).to_numpy()
        events: list[dict[str, Any]] = []
        assoc_picks: list[dict[str, Any]] = []
        i, n = 0, len(p)
        while i < n:
            t0 = float(t[i])
            members: dict[str, str] = {}
            j = i
            while j < n and t[j] - t0 <= WINDOW_S:
                members.setdefault(sid[j], pid[j])
                j += 1
            if len(members) < MIN_STATIONS:
                i += 1
                continue
            assoc_id = f"assoc-{len(events) + 1:04d}"
            s_members = s[
                s["stationId"].isin(members)
                & (s["t"] >= t0)
                & (s["t"] <= t0 + WINDOW_S + S_WINDOW_S)
            ]
            events.append(
                {
                    "assocId": assoc_id,
                    "t": t0,
                    "latitude": run.origin.lat,
                    "longitude": run.origin.lon,
                    "elevM": run.refSurfaceElevM - EVENT_DEPTH_M,
                    "nPicks": len(members) + len(s_members),
                    "nP": len(members),
                    "nS": len(s_members),
                }
            )
            assoc_picks += [{"assocId": assoc_id, "pickId": p} for p in members.values()]
            assoc_picks += [{"assocId": assoc_id, "pickId": p} for p in s_members["id"]]
            i = j
        return FakeAssocResult(
            pd.DataFrame(events, columns=list(ASSOC_EVENT_COLUMNS)),
            pd.DataFrame(assoc_picks, columns=list(ASSOC_PICK_COLUMNS)),
        )

    def locate(
        self,
        assoc: FakeAssocResult,
        picks: pd.DataFrame,
        stations: pd.DataFrame,
        cfg: Any,
        run: RunSection,
    ) -> FakeLocateResult:
        self._tick("locate")
        known = set(picks["id"].astype(str))
        events: list[m.SeismicEvent] = []
        for k, row in enumerate(assoc.events.itertuples(index=False)):
            pick_ids = assoc.picks.loc[assoc.picks["assocId"] == row.assocId, "pickId"].tolist()
            assert set(pick_ids) <= known  # the associator only hands back picks it was given
            events.append(
                m.SeismicEvent(
                    id=f"hq-{self.run_id}-{k + 1:06d}",
                    runId=self.run_id,
                    t=float(row.t),
                    latitude=float(row.latitude),
                    longitude=float(row.longitude),
                    elevM=float(row.elevM),
                    depthKm=(run.refSurfaceElevM - float(row.elevM)) / 1000.0,
                    enu=m.Enu(e=0.0, n=0.0, u=float(row.elevM) - run.origin.elevM),
                    quality=m.LocationQuality(
                        method="grid1d",
                        statics=False,
                        nStations=int(row.nP),
                        nP=int(row.nP),
                        nS=int(row.nS),
                        rmsS=0.05,
                        gapDeg=360.0 / max(int(row.nP), 1),
                        minEpiDistM=RING_RADIUS_M,
                        hErrM=200.0,
                        vErrM=400.0,
                        depthOnEdge=False,
                    ),
                    tier="C",  # placeholder; dropped below, assign_tiers sets it
                    tierReasons=[],
                    meanPickProb=0.9,
                    revealOrder=-1,
                    pickIds=pick_ids,
                )
            )
        frame = to_frame(events, m.SeismicEvent)
        dropped = [c for c in frame.columns if c.startswith(LOCATED_DROPPED_PREFIXES)]
        return FakeLocateResult(
            frame.drop(columns=dropped),
            pd.DataFrame(columns=list(ARRIVAL_COLUMNS)),
            pd.DataFrame(columns=list(STATICS_COLUMNS)),
        )

    def match(
        self, events_located: pd.DataFrame, catalog: pd.DataFrame, cfg: Any
    ) -> FakeMatchResult:
        self._tick("match")
        ev_t = events_located["t"].to_numpy(dtype=np.float64)
        ev_id = events_located["id"].astype(str).to_numpy()
        rows: list[dict[str, Any]] = []
        for c in catalog.itertuples(index=False):
            dt = ev_t - float(c.t)
            k = int(np.argmin(np.abs(dt)))
            if abs(dt[k]) <= MATCH_DT_S:
                rows.append(
                    {"catalogId": c.id, "eventId": ev_id[k], "dtS": float(dt[k]), "distM": 0.0,
                     "reason": None}
                )  # fmt: skip
            else:
                rows.append(
                    {"catalogId": c.id, "eventId": None, "dtS": None, "distM": None,
                     "reason": "no candidate within tolerance (test)"}
                )  # fmt: skip
        return FakeMatchResult(
            pd.DataFrame(rows, columns=list(MATCH_COLUMNS)),
            pd.DataFrame(columns=list(SENSITIVITY_COLUMNS)),
        )

    def assign_tiers(
        self, events_located: pd.DataFrame, matches: pd.DataFrame, cfg: Any
    ) -> FakeTierResult:
        self._tick("assign_tiers")
        events = events_located.copy()
        n = events["quality_nStations"].to_numpy(dtype=np.int64)
        events["tier"] = np.where(
            n >= TIER_A_MIN_STATIONS, "A", np.where(n >= TIER_B_MIN_STATIONS, "B", "C")
        )
        events["tierReasons"] = [[f"nStations {k} (test)"] for k in n]
        matched = matches.dropna(subset=["eventId"]).set_index("eventId")
        events["catalogMatch_catalogId"] = events["id"].map(matched["catalogId"])
        events["catalogMatch_dtS"] = events["id"].map(matched["dtS"])
        events["catalogMatch_distM"] = events["id"].map(matched["distM"])
        events["magnitude_value"] = None
        events["magnitude_type"] = None
        events["magnitude_sigma"] = None
        final = to_frame(from_frame(events, m.SeismicEvent), m.SeismicEvent)  # docs/02 §2 columns
        return FakeTierResult(
            final, {"tierA": {"nStations": TIER_A_MIN_STATIONS}, "tierB": {"nStations": 5}}
        )

    def as_lane_api(self) -> LaneSeismologyApi:
        return LaneSeismologyApi(self.associate, self.locate, self.match, self.assign_tiers)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stand in for H2's modules so ``real_seismology_api`` resolves to this toy."""
        install_module(monkeypatch, "hq.associate", associate=self.associate)
        install_module(monkeypatch, "hq.locate", locate=self.locate)
        install_module(monkeypatch, "hq.match", match=self.match)
        install_module(monkeypatch, "hq.tier", assign_tiers=self.assign_tiers)


def null_cfg(**overrides: Any) -> NullTestConfig:
    return NullTestConfig(**{"nShuffles": 8, "shiftS": 30.0, "seed": 3, **overrides})


# --- the null test on its own -----------------------------------------------------------------


def test_coherent_picks_give_one_event_per_planted_event(
    synthetic: SyntheticPicks, section: RunSection, ctx: runs.RunContext
) -> None:
    api = FakeSeismologyApi()
    n_events, n_strict = rerun_pipeline(
        synthetic.picks_frame,
        synthetic.stations_frame,
        synthetic.catalog_frame,
        api,
        ctx.config.seismology,
        section,
    )
    assert (n_events, n_strict) == (N_EVENTS, N_EVENTS)  # every station picks every event
    assert api.calls == {"associate": 1, "locate": 1, "match": 1, "assign_tiers": 1}


def test_null_test_finds_fewer_chance_events_than_the_coherent_picks(
    synthetic: SyntheticPicks, section: RunSection, ctx: runs.RunContext
) -> None:
    api = FakeSeismologyApi()
    cfg = null_cfg()
    result = run_null_test(
        synthetic.picks_frame,
        synthetic.stations_frame,
        synthetic.catalog_frame,
        api,
        ctx.config.seismology,
        section,
        cfg,
    )
    assert isinstance(result, m.NullTest)
    assert result.nShuffles == cfg.nShuffles and result.shiftRangeS == cfg.shiftS
    assert api.calls["associate"] == cfg.nShuffles  # one rerun per shuffle
    assert all(math.isfinite(v) for v in (result.meanChanceEvents, result.meanChanceStrict,
                                         result.stdChanceEvents))  # fmt: skip
    assert 0.0 <= result.meanChanceEvents < N_EVENTS
    assert 0.0 <= result.meanChanceStrict <= result.meanChanceEvents
    assert result.stdChanceEvents >= 0.0
    m.NullTest.model_validate_json(result.model_dump_json())  # serializable, no NaN


def test_shifts_are_shared_within_a_station_and_bounded(
    synthetic: SyntheticPicks, section: RunSection, ctx: runs.RunContext
) -> None:
    cfg = null_cfg(nShuffles=3)
    picks = synthetic.picks_frame
    outcomes = null_shuffles(
        picks,
        synthetic.stations_frame,
        synthetic.catalog_frame,
        FakeSeismologyApi(),
        ctx.config.seismology,
        section,
        cfg,
    )
    assert [o.index for o in outcomes] == [0, 1, 2]
    station_ids = sorted({s.id for s in synthetic.stations})
    for outcome in outcomes:
        assert list(outcome.shifts_s.index) == station_ids
        assert (outcome.shifts_s.abs() <= cfg.shiftS).all()
        assert outcome.shifts_s.nunique() == N_STATIONS  # every station drew its own
        shifted = shift_picks(picks, outcome.shifts_s)
        delta = shifted["t"] - picks["t"]
        per_station = delta.groupby(picks["stationId"]).agg(["min", "max"])
        assert np.allclose(per_station["min"], per_station["max"])  # P and S moved together
        assert np.allclose(per_station["min"], outcome.shifts_s.reindex(per_station.index))
        assert shifted["id"].tolist() == picks["id"].tolist()  # keys untouched
        assert (shifted["phase"] == picks["phase"]).all()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="unique"):
        station_shifts(["XT.S01", "XT.S01"], rng, 1.0)
    with pytest.raises(ValidateError, match="no shift drawn"):
        shift_picks(picks, outcomes[0].shifts_s.drop("XT.S03"))


def test_same_seed_reproduces_and_a_different_seed_differs(
    synthetic: SyntheticPicks, section: RunSection, ctx: runs.RunContext
) -> None:
    args = (synthetic.picks_frame, synthetic.stations_frame, synthetic.catalog_frame)

    def outcomes(cfg: NullTestConfig) -> list[Any]:
        return null_shuffles(*args, FakeSeismologyApi(), ctx.config.seismology, section, cfg)

    cfg = null_cfg(nShuffles=6)
    first, again = outcomes(cfg), outcomes(cfg)
    for a, b in zip(first, again, strict=True):
        pd.testing.assert_series_equal(a.shifts_s, b.shifts_s)
        assert (a.n_events, a.n_strict) == (b.n_events, b.n_strict)
    result_a = run_null_test(*args, FakeSeismologyApi(), ctx.config.seismology, section, cfg)
    result_b = run_null_test(*args, FakeSeismologyApi(), ctx.config.seismology, section, cfg)
    assert result_a == result_b
    other = outcomes(null_cfg(nShuffles=2, seed=4))
    assert not np.allclose(other[0].shifts_s.to_numpy(), first[0].shifts_s.to_numpy())
    # Rerun i depends only on (seed, i): a shorter run reproduces the first reruns exactly.
    for a, b in zip(outcomes(null_cfg(nShuffles=2)), first[:2], strict=True):
        pd.testing.assert_series_equal(a.shifts_s, b.shifts_s)


def test_p_only_profile_feeds_p_picks_only(
    synthetic: SyntheticPicks, section: RunSection, ctx: runs.RunContext
) -> None:
    picks = synthetic.picks_frame
    assert set(select_profile(picks, "full")["phase"]) == {"P", "S"}
    assert set(select_profile(picks, "p_only")["phase"]) == {"P"}
    api = FakeSeismologyApi()
    run_null_test(
        picks,
        synthetic.stations_frame,
        synthetic.catalog_frame,
        api,
        ctx.config.seismology,
        section,
        null_cfg(nShuffles=2, profile="p_only"),
    )
    assert api.phases_seen == {"P"}
    # REQ-H2-7: the p_only reruns carry the associator overrides; the run's config is untouched.
    run_assoc = ctx.config.seismology.associator
    assert api.cfgs_seen and all(c is not ctx.config.seismology for c in api.cfgs_seen)
    assert {(c.associator.nSPicks, c.associator.nPAndSPicks) for c in api.cfgs_seen} == {(0, 0)}
    assert {c.associator.nPPicks for c in api.cfgs_seen} == {run_assoc.nPPicks}
    assert ctx.config.seismology.associator is run_assoc
    full_api = FakeSeismologyApi()
    run_null_test(
        picks,
        synthetic.stations_frame,
        synthetic.catalog_frame,
        full_api,
        ctx.config.seismology,
        section,
        null_cfg(nShuffles=2, profile="full"),
    )
    assert all(c is ctx.config.seismology for c in full_api.cfgs_seen)
    with pytest.raises(ValidateError, match="associator"):
        profile_config(object(), "p_only", POnlyAssociatorConfig())


def test_api_errors_propagate_unchanged(
    synthetic: SyntheticPicks, section: RunSection, ctx: runs.RunContext
) -> None:
    args = (synthetic.picks_frame, synthetic.stations_frame, synthetic.catalog_frame)
    with pytest.raises(RuntimeError, match="boom in associate"):
        run_null_test(
            *args, FakeSeismologyApi(fail_in="associate"), ctx.config.seismology, section,
            null_cfg(nShuffles=2),
        )  # fmt: skip
    # A rerun that associates nothing never reaches the later steps; shiftS far beyond the
    # window spread makes every rerun empty, and the summary is all zeros, not a failure.
    api = FakeSeismologyApi(fail_in="locate")
    result = run_null_test(
        *args, api, ctx.config.seismology, section, null_cfg(nShuffles=2, shiftS=1.0e6)
    )
    assert (result.meanChanceEvents, result.meanChanceStrict, result.stdChanceEvents) == (0, 0, 0)
    assert api.calls == {"associate": 2, "locate": 0, "match": 0, "assign_tiers": 0}
    with pytest.raises(ValidateError, match="picks lacks columns"):
        run_null_test(
            args[0].drop(columns=["stationId"]), *args[1:], FakeSeismologyApi(),
            ctx.config.seismology, section, null_cfg(nShuffles=2),
        )  # fmt: skip
    with pytest.raises(ValidateError, match="missing from stations table"):
        run_null_test(
            args[0], args[1].iloc[:2], args[2], FakeSeismologyApi(), ctx.config.seismology,
            section, null_cfg(nShuffles=2),
        )  # fmt: skip


# --- the stage -----------------------------------------------------------------------------------


def write_run_tables(ctx: runs.RunContext, synthetic: SyntheticPicks) -> None:
    write_models(synthetic.stations, ctx.path("stations.parquet"))
    write_models(synthetic.picks, ctx.path("picks.parquet"))
    write_models(synthetic.catalog, ctx.path("catalog.parquet"))


def write_h2_validation_inputs(ctx: runs.RunContext) -> tuple[m.SyntheticTest, list[m.SweepPoint]]:
    synthetic_test = m.SyntheticTest(
        nEvents=50,
        pickSigmaS={"P": 0.02, "S": 0.04},
        medianHErrM=120.0,
        medianVErrM=260.0,
        p90VErrM=610.0,
        medianDepthBiasM=-15.0,
    )
    ctx.path("synthetic.json").write_text(synthetic_test.model_dump_json(indent=2))
    sweep = [
        m.SweepPoint(params={"minStations": k, "minS": 1}, candidates=30 - k, recoveredPublic=3,
                     tierA=5)
        for k in (4, 5, 6)
    ]  # fmt: skip
    write_models(sweep, ctx.path("sweep.parquet"))
    return synthetic_test, sweep


def test_stage_writes_validation_json_and_records_counts(
    ctx: runs.RunContext,
    synthetic: SyntheticPicks,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="hq.validate")
    write_run_tables(ctx, synthetic)
    synthetic_test, sweep = write_h2_validation_inputs(ctx)
    api = FakeSeismologyApi()
    api.install(monkeypatch)
    cfg = ctx.config.validate.nullTest

    runs.run_stage(ctx, "validate")

    assert api.calls["associate"] == cfg.nShuffles
    null_test = m.NullTest.model_validate_json(ctx.path(NULL_TEST_JSON).read_text())
    assert (null_test.nShuffles, null_test.shiftRangeS) == (cfg.nShuffles, cfg.shiftS)
    assert null_test.meanChanceEvents < N_EVENTS
    validation = m.Validation.model_validate_json(ctx.path(VALIDATION_JSON).read_text())
    assert validation.nullTest == null_test
    assert validation.synthetic == synthetic_test
    assert validation.sweep == sweep
    assert validation.baseline == [] and validation.gr is None and validation.magnitude is None
    stages = json.loads(ctx.path("stages.json").read_text())
    counts = stages["validate"]["counts"]
    assert counts["nullShuffles"] == cfg.nShuffles
    assert counts["nullChanceEvents"] == round(null_test.meanChanceEvents * cfg.nShuffles)
    assert counts["nullChanceStrict"] == round(null_test.meanChanceStrict * cfg.nShuffles)
    assert counts["sweepPoints"] == len(sweep) and counts["validationJson"] == 1
    assert ctx.read_run().runtimeS["validate"] >= 0.0
    assert not list(ctx.run_dir.glob("*.tmp"))
    messages = [r.getMessage() for r in caplog.records]
    assert sum("null test: rerun" in msg for msg in messages) == cfg.nShuffles
    assert any(msg.startswith("null test: chance events") for msg in messages)
    # Same seed, same file: the stage is reproducible byte for byte.
    before = ctx.path(NULL_TEST_JSON).read_bytes()
    validate_stage(ctx)
    assert ctx.path(NULL_TEST_JSON).read_bytes() == before


def test_stage_without_synthetic_json_writes_the_sidecar_and_names_h2(
    ctx: runs.RunContext,
    synthetic: SyntheticPicks,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="hq.validate")
    write_run_tables(ctx, synthetic)
    FakeSeismologyApi().install(monkeypatch)
    runs.run_stage(ctx, "validate")
    m.NullTest.model_validate_json(ctx.path(NULL_TEST_JSON).read_text())
    assert not ctx.path(VALIDATION_JSON).exists()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("synthetic.json" in w and "H2 Seismology" in w for w in warnings)
    counts = json.loads(ctx.path("stages.json").read_text())["validate"]["counts"]
    assert counts["validationJson"] == 0 and counts["sweepPoints"] == 0
    ctx.path("synthetic.json").write_text("{not json")
    with pytest.raises(ValidateError, match="not a valid SyntheticTest"):
        validate_stage(ctx)


def test_missing_run_table_names_its_writer(
    ctx: runs.RunContext, synthetic: SyntheticPicks, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSeismologyApi().install(monkeypatch)
    write_models(synthetic.stations, ctx.path("stations.parquet"))
    with pytest.raises(ValidateError, match=r"picks\.parquet.*'pick'.*H1 Signal"):
        runs.run_stage(ctx, "validate")
    write_models(synthetic.picks, ctx.path("picks.parquet"))
    with pytest.raises(ValidateError, match=r"catalog\.parquet.*'catalog'.*H2 Seismology"):
        runs.run_stage(ctx, "validate")
    write_models(synthetic.picks, ctx.path("catalog.parquet"))  # wrong model in the file
    with pytest.raises(ValidateError, match="holds Pick rows, not CatalogEvent"):
        runs.run_stage(ctx, "validate")
    assert read_table(ctx.path("picks.parquet")).attrs["model"] == "Pick"


def test_missing_h2_modules_fail_naming_h2(
    ctx: runs.RunContext, synthetic: SyntheticPicks, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_run_tables(ctx, synthetic)
    api = FakeSeismologyApi()
    api.install(monkeypatch)
    monkeypatch.setitem(sys.modules, "hq.associate", None)  # as if H2 had not merged it
    with pytest.raises(ValidateError, match=r"hq\.associate\.associate.*not merged.*H2 Seismology"):
        real_seismology_api()
    with pytest.raises(ValidateError, match="H2 Seismology"):
        runs.run_stage(ctx, "validate")
    assert api.calls["associate"] == 0
    api.install(monkeypatch)
    install_module(monkeypatch, "hq.tier")  # merged, but without assign_tiers
    with pytest.raises(ValidateError, match=r"hq\.tier\.assign_tiers.*does not expose.*H2"):
        real_seismology_api()

    def broken_import(name: str, package: str | None = None) -> types.ModuleType:
        raise ModuleNotFoundError("No module named 'pyocto_xyz'", name="pyocto_xyz")

    monkeypatch.setattr("hq.validate.lanes.importlib.import_module", broken_import)
    with pytest.raises(ModuleNotFoundError, match="pyocto_xyz"):  # H2's own import is broken
        real_seismology_api()


def test_stage_needs_the_seismology_section(
    ctx: runs.RunContext, synthetic: SyntheticPicks, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hq.config import ConfigError, RunConfig

    write_run_tables(ctx, synthetic)
    FakeSeismologyApi().install(monkeypatch)
    config = RunConfig(
        run=ctx.config.run, signal=None, seismology=None, export=ctx.config.export,
        validate=ctx.config.validate,
    )  # fmt: skip
    bare = runs.RunContext(ctx.run_id, ctx.run_dir, ctx.cache_dir, config)
    with pytest.raises(ConfigError, match="seismology.*H2 Seismology"):
        validate_stage(bare)


def test_cli_lists_validate_as_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["stages"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("validate")]
    assert len(lines) == 1
    assert "hq.validate" in lines[0] and lines[0].rstrip().endswith("implemented")

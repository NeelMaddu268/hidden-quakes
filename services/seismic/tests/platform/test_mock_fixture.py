"""FIX-01 acceptance: ``scripts/mock-fixture.py`` writes a bundle whose every file validates
against the contract models, whose summary recomputes from the data, whose tiers trace to the
thresholds recorded in ``meta.run.tiering``, and which is byte-identical for the same seed."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from hq_contracts import models as m

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "mock-fixture.py"
SEED = 13
OTHER_SEED = 14
MAX_EVIDENCE_BYTES = 60 * 1024
BUNDLE_FILES = (
    "meta.json",
    "stations.json",
    "catalog.json",
    "events.json",
    "features.json",
    "validation.json",
)


@pytest.fixture(scope="module")
def mock_fixture() -> ModuleType:
    """Import the generator from its hyphenated path as a module."""
    spec = importlib.util.spec_from_file_location("mock_fixture", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["mock_fixture"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory: pytest.TempPathFactory, mock_fixture: ModuleType) -> Path:
    out = tmp_path_factory.mktemp("mock")
    mock_fixture.generate(out, seed=SEED)
    return out


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


class Bundle:
    """Every bundle file parsed through its contract model."""

    def __init__(self, out_dir: Path) -> None:
        self.dir = out_dir
        self.meta = m.BundleMeta.model_validate(read_json(out_dir / "meta.json"))
        self.stations = [m.Station.model_validate(x) for x in read_json(out_dir / "stations.json")]
        self.catalog = [
            m.CatalogEvent.model_validate(x) for x in read_json(out_dir / "catalog.json")
        ]
        self.events = [m.SeismicEvent.model_validate(x) for x in read_json(out_dir / "events.json")]
        self.features = [
            m.GeoFeature.model_validate(x) for x in read_json(out_dir / "features.json")
        ]
        self.validation = m.Validation.model_validate(read_json(out_dir / "validation.json"))
        self.evidence = {
            p.stem: m.EventEvidence.model_validate(read_json(p))
            for p in sorted((out_dir / "evidence").glob("*.json"))
        }
        self.events_by_id = {e.id: e for e in self.events}


@pytest.fixture(scope="module")
def bundle(out_dir: Path) -> Bundle:
    return Bundle(out_dir)


# --- files and flags -------------------------------------------------------------------


def test_bundle_has_exactly_the_documented_files(out_dir: Path) -> None:
    top = sorted(p.name for p in out_dir.iterdir())
    assert top == sorted([*BUNDLE_FILES, "evidence"])
    assert all(p.suffix == ".json" for p in (out_dir / "evidence").iterdir())


def test_synthetic_flags_and_mode(bundle: Bundle) -> None:
    assert bundle.meta.mode == "mock"
    assert bundle.meta.schemaVersion == m.SCHEMA_VERSION
    assert bundle.meta.scene.isSynthetic is True
    assert bundle.meta.run.isSynthetic is True
    assert bundle.meta.run.mode == "mock"
    assert bundle.meta.run.id == bundle.meta.scene.runId == bundle.meta.summary.runId
    assert all(e.runId == bundle.meta.run.id for e in bundle.events)


def test_nothing_looks_real(bundle: Bundle) -> None:
    for s in bundle.stations:
        assert s.network == "XX" and not s.id.startswith("UU.")
    assert all("synthetic" in c.source for c in bundle.catalog)
    for f in bundle.features:
        assert f.source.verified is False
        assert "synthetic" in f.source.citation
    assert "FORGE" not in json.dumps([f.model_dump() for f in bundle.features])


# --- counts and geometry ----------------------------------------------------------------


def test_station_counts_and_kinds(bundle: Bundle, mock_fixture: ModuleType) -> None:
    surface = [s for s in bundle.stations if s.kind == "surface"]
    borehole = [s for s in bundle.stations if s.kind == "borehole"]
    assert len(bundle.stations) == 17 and len(surface) == 14 and len(borehole) == 3
    knobs = mock_fixture.Knobs()
    for s in surface:  # the relief never clips: no station sits on a bound
        assert knobs.surface_elev_min_m < s.surfaceElevM < knobs.surface_elev_max_m, s.id
    assert all(s.sensorDepthM == 0.0 for s in surface)
    assert all(s.sensorDepthM > 0.0 and s.sampleRateHz > 100.0 for s in borehole)
    assert all(s.channels[0].startswith("DP") for s in borehole)
    assert any(s.staticsS for s in bundle.stations)
    assert any(not s.usedInRun for s in bundle.stations)
    used = [s.id for s in bundle.stations if s.usedInRun]
    assert bundle.meta.run.stationIds == used
    origin = bundle.meta.scene.originElevM
    for s in bundle.stations:
        assert s.sensorElevM == pytest.approx(s.surfaceElevM - s.sensorDepthM, abs=1e-6)
        assert s.enu.u == pytest.approx(s.sensorElevM - origin, abs=1e-6)


def test_event_counts_and_conventions(bundle: Bundle) -> None:
    scene = bundle.meta.scene
    min_lon, min_lat, max_lon, max_lat = bundle.meta.run.bbox
    tiers = {t: sum(1 for e in bundle.events if e.tier == t) for t in ("A", "B", "C")}
    assert tiers == {"A": 150, "B": 200, "C": 150}
    assert len({e.id for e in bundle.events}) == len(bundle.events)
    for e in bundle.events:
        assert e.source == "hq-pipeline"
        assert e.depthKm == pytest.approx((scene.refSurfaceElevM - e.elevM) / 1000.0, abs=1e-6)
        assert e.enu.u == pytest.approx(e.elevM - scene.originElevM, abs=1e-6)
        assert min_lon <= e.longitude <= max_lon and min_lat <= e.latitude <= max_lat
        assert bundle.meta.run.windowStart <= e.t < bundle.meta.run.windowEnd
        assert e.quality.nP + e.quality.nS == len(e.pickIds)
        assert all(p.startswith("phasenet:") for p in e.pickIds)
        assert 0.0 < e.meanPickProb < 1.0
    tier_a = [e for e in bundle.events if e.tier == "A"]
    assert all(e.quality.vErrM is not None and not e.quality.depthOnEdge for e in tier_a)
    assert all(0.9 <= e.depthKm <= 3.2 for e in tier_a), "Tier A cluster sits 1-3 km down"
    tier_c = [e for e in bundle.events if e.tier == "C"]
    assert any(e.quality.vErrM is None for e in tier_c)
    assert any(e.quality.depthOnEdge for e in tier_c)
    # Nullable UI paths, each exercised at least once (and only where the tier allows it).
    assert any(e.quality.hErrM is None for e in tier_c)
    assert all(e.quality.hErrM is not None for e in bundle.events if e.tier != "C")
    assert any(e.magnitude is None for e in bundle.events)
    assert any(e.magnitude is not None and e.magnitude.sigma is None for e in bundle.events)
    assert any(e.magnitude is not None and e.magnitude.sigma is not None for e in bundle.events)
    assert all(e.magnitude.type == "ML_cal" for e in bundle.events if e.magnitude)


def test_catalog_matching_is_one_to_one(bundle: Bundle) -> None:
    assert len(bundle.catalog) == 43
    matched = [c for c in bundle.catalog if c.matchedEventId is not None]
    assert len(matched) == 38
    assert len({c.matchedEventId for c in matched}) == 38
    dist_max = bundle.meta.run.matching["distMaxM"]
    dt_max = bundle.meta.run.matching["dtMaxS"]
    for c in bundle.catalog:
        assert c.elevM == pytest.approx(-c.depthKm * 1000.0, abs=1e-6)
        assert bundle.meta.run.windowStart <= c.t < bundle.meta.run.windowEnd
        if c.matchedEventId is None:
            continue
        ev = bundle.events_by_id[c.matchedEventId]
        assert ev.catalogMatch is not None and ev.catalogMatch.catalogId == c.id
        assert ev.catalogMatch.dtS == pytest.approx(c.t - ev.t, abs=1e-3)
        assert abs(ev.catalogMatch.dtS) <= dt_max
        assert ev.catalogMatch.distM <= dist_max
        assert ev.catalogMatch.distM == pytest.approx(
            np.hypot(c.enu.e - ev.enu.e, c.enu.n - ev.enu.n), abs=0.2
        )
    with_match = {e.id for e in bundle.events if e.catalogMatch is not None}
    assert with_match == {c.matchedEventId for c in matched}
    assert any(c.mag is None for c in bundle.catalog)
    assert any(c.mag is not None and c.magType is not None for c in bundle.catalog)


# --- summary, tiers, reveal --------------------------------------------------------------


def test_summary_recomputes_from_events_and_catalog(bundle: Bundle) -> None:
    s = bundle.meta.summary
    events, catalog = bundle.events, bundle.catalog
    recovered = sum(1 for c in catalog if c.matchedEventId is not None)
    additional = [e for e in events if e.catalogMatch is None]
    assert s.publicCatalogCount == len(catalog)
    assert s.recoveredCatalogCount == recovered
    assert s.recall == pytest.approx(recovered / len(catalog), abs=1e-4)
    assert s.unmatchedPublicIds == [c.id for c in catalog if c.matchedEventId is None]
    assert s.candidateCount == len(events)
    assert s.additionalCount == len(additional)
    for tier in ("A", "B", "C"):
        assert getattr(s.additional, tier) == sum(1 for e in additional if e.tier == tier)
    assert s.strictQualityCount == sum(1 for e in events if e.tier == "A")
    assert s.strictAdditionalCount == sum(1 for e in additional if e.tier == "A")
    assert s.medianStations == float(np.median([e.quality.nStations for e in events]))
    assert s.medianRmsS == round(float(np.median([e.quality.rmsS for e in events])), 3)
    assert s.baseline is not None
    rows = {(r.method, r.associationProfile): r for r in bundle.validation.baseline}
    assert s.baseline.strictPhasenet == rows[("phasenet", "full")].tiers.A
    assert s.baseline.strictStalta == rows[("stalta", "full")].tiers.A
    assert s.baseline.gain == pytest.approx(
        s.baseline.strictPhasenet / s.baseline.strictStalta, abs=0.01
    )


def derive_tier(q: m.LocationQuality, thresholds: dict) -> str:
    """The tier rule as documented in meta.run.tiering, written out independently of the script."""
    a, b = thresholds["A"], thresholds["B"]
    is_a = (
        q.nStations >= a["minStations"]
        and q.rmsS <= a["maxRmsS"]
        and q.hErrM is not None
        and q.hErrM <= a["maxHErrM"]
        and q.vErrM is not None
        and q.vErrM <= a["maxVErrM"]
        and q.gapDeg <= a["maxGapDeg"]
        and not q.depthOnEdge
    )
    if is_a:
        return "A"
    is_b = (
        q.nStations >= b["minStations"]
        and q.rmsS <= b["maxRmsS"]
        and q.hErrM is not None
        and q.hErrM <= b["maxHErrM"]
    )
    return "B" if is_b else "C"


def test_tiers_trace_to_recorded_thresholds(bundle: Bundle) -> None:
    thresholds = bundle.meta.run.tiering["thresholds"]
    assert set(thresholds["A"]) == {"minStations", "maxRmsS", "maxHErrM", "maxVErrM", "maxGapDeg"}
    assert set(thresholds["B"]) == {"minStations", "maxRmsS", "maxHErrM"}
    for e in bundle.events:
        assert derive_tier(e.quality, thresholds) == e.tier, e.id
        assert e.tierReasons, "every event explains its tier"
        not_a = [r for r in e.tierReasons if r.startswith("not A:")]
        not_b = [r for r in e.tierReasons if r.startswith("not B:")]
        if e.tier == "A":
            assert not not_a and not not_b, e.id
        elif e.tier == "B":
            assert not_a and not not_b, e.id
        else:
            assert not_a and not_b, e.id
        # Every quoted limit is one of the recorded thresholds.
        limits = {str(v) for t in thresholds.values() for v in t.values()}
        limits |= {f"{float(v):.0f}" for t in thresholds.values() for v in t.values()}
        for reason in e.tierReasons:
            if "missing" in reason or "depthOnEdge" in reason:
                continue
            quoted = reason.split("(")[0].split()[-1]
            assert quoted in limits, (e.id, reason)


def test_hero_is_the_tier_a_event_with_most_stations(bundle: Bundle) -> None:
    hero_id = bundle.meta.scene.heroEventId
    assert hero_id is not None and hero_id in bundle.events_by_id
    hero = bundle.events_by_id[hero_id]
    assert hero.tier == "A"
    best = max(e.quality.nStations for e in bundle.events if e.tier == "A")
    assert hero.quality.nStations == best
    used = sum(1 for s in bundle.stations if s.usedInRun)
    assert used == 16 and hero.quality.nStations == used, "the hero exercises every used station"
    assert len(bundle.evidence[hero_id].traces) == 16, "H3 QA: drawer handles 16 traces"


def test_reveal_order_is_a_permutation_ordered_a_b_c_then_time(bundle: Bundle) -> None:
    orders = sorted(e.revealOrder for e in bundle.events)
    assert orders == list(range(len(bundle.events)))
    rank = {"A": 0, "B": 1, "C": 2}
    ordered = sorted(bundle.events, key=lambda e: e.revealOrder)
    keys = [(rank[e.tier], e.t) for e in ordered]
    assert keys == sorted(keys)


# --- evidence -----------------------------------------------------------------------------


def test_evidence_files_only_for_the_chosen_events(bundle: Bundle) -> None:
    assert len(bundle.evidence) == 20
    assert set(bundle.evidence) <= set(bundle.events_by_id)
    assert bundle.meta.scene.heroEventId in bundle.evidence
    tiers = {bundle.events_by_id[i].tier for i in bundle.evidence}
    assert tiers == {"A", "B", "C"}, "evidence exercises every tier"
    for event_id, ev in bundle.evidence.items():
        assert ev.eventId == event_id
        assert (bundle.dir / "evidence" / f"{event_id}.json").stat().st_size < MAX_EVIDENCE_BYTES
    counts = [len(ev.traces) for ev in bundle.evidence.values()]
    assert min(counts) == 4 and max(counts) == 16, "the drawer sees both a 4- and a 16-trace file"
    smallest = min(bundle.evidence.values(), key=lambda ev: len(ev.traces))
    assert bundle.events_by_id[smallest.eventId].tier == "C"


def test_evidence_traces_are_well_formed(bundle: Bundle) -> None:
    used = {s.id for s in bundle.stations if s.usedInRun}
    saw_missing_s = saw_missing_pred = False
    for ev in bundle.evidence.values():
        assert 1 <= len(ev.traces) <= 16
        assert ev.filterHz[0] < ev.filterHz[1]
        dists = [t.epiDistM for t in ev.traces]
        assert dists == sorted(dists) and len(set(dists)) == len(dists)
        event = bundle.events_by_id[ev.eventId]
        for t in ev.traces:
            assert t.stationId in used
            seconds = len(t.samples) * t.dt
            assert 4.0 <= seconds <= 8.0 + 1e-9
            assert min(t.samples) >= -1.0 and max(t.samples) <= 1.0
            assert max(abs(x) for x in t.samples) == pytest.approx(1.0, abs=1e-3)
            assert t.pickP is not None and t.probP is not None, "pickP/probP are always set"
            assert t.t0 <= t.pickP < t.t0 + seconds
            assert event.t < t.pickP
            if t.predP is None:
                saw_missing_pred = True
                assert t.predS is None, "predP and predS are blanked together"
            else:
                assert t.predS is not None and t.predP < t.predS
                assert event.t < t.predP and t.t0 <= t.predS < t.t0 + seconds
            if t.pickS is None:
                saw_missing_s = True
                assert t.probS is None
            else:
                assert t.probS is not None and t.t0 <= t.pickS < t.t0 + seconds
    assert saw_missing_s, "some traces have no S pick, so the drawer's null path is exercised"
    assert saw_missing_pred, "some far traces have no predicted arrivals (WEB-05 null path)"


# --- features and validation ------------------------------------------------------------


def test_features(bundle: Bundle) -> None:
    kinds = sorted(f.kind for f in bundle.features)
    assert kinds == ["boundary", "facility", "well"]
    well = next(f for f in bundle.features if f.kind == "well")
    assert well.name == "Synthetic well A" and well.source.verified is False
    assert len(well.path) > 2
    depths = [p.u for p in well.path]
    assert depths == sorted(depths, reverse=True), "the well path goes down from the surface"
    facility = next(f for f in bundle.features if f.kind == "facility")
    assert len(facility.path) == 1


def test_validation_is_filled_and_consistent(bundle: Bundle, mock_fixture: ModuleType) -> None:
    v = bundle.validation
    assert v.nullTest is not None and v.gr is not None and v.magnitude is not None
    keys = {(r.method, r.associationProfile) for r in v.baseline}
    assert keys == {(mth, prof) for mth in ("phasenet", "stalta") for prof in ("full", "p_only")}
    for r in v.baseline:  # every row is internally consistent
        assert r.tiers.A + r.tiers.B + r.tiers.C == r.candidates, (r.method, r.associationProfile)
        assert r.recoveredPublic <= r.candidates
    full = next(r for r in v.baseline if (r.method, r.associationProfile) == ("phasenet", "full"))
    assert full.candidates == len(bundle.events)
    assert full.recoveredPublic == bundle.meta.summary.recoveredCatalogCount
    assert full.tiers.A == bundle.meta.summary.strictQualityCount
    assert len(v.sweep) >= 3
    assert any(p.candidates == len(bundle.events) for p in v.sweep)
    recovered_mags = np.asarray([e.magnitude.value for e in bundle.events if e.magnitude])
    public_mags = np.asarray([c.mag for c in bundle.catalog if c.mag is not None])
    for i, b in enumerate(v.gr.magBins):
        assert v.gr.recoveredCum[i] == int(np.sum(recovered_mags >= b - 1e-9))
        assert v.gr.publicCum[i] == int(np.sum(public_mags >= b - 1e-9))
    generating_b = mock_fixture.Knobs().mag_b_value
    assert v.gr.bValue is not None and v.gr.bValue == pytest.approx(generating_b, abs=0.1)
    assert v.gr.bSigma is not None and 0.0 < v.gr.bSigma < 0.2
    assert v.magnitude.n > 0 and v.magnitude.looMae > 0
    assert v.synthetic.medianVErrM > 0 and set(v.synthetic.pickSigmaS) == {"P", "S"}


# --- determinism --------------------------------------------------------------------------


def test_same_seed_gives_identical_bytes(
    out_dir: Path, tmp_path: Path, mock_fixture: ModuleType
) -> None:
    again = tmp_path / "again"
    mock_fixture.generate(again, seed=SEED)
    first = sorted(p.relative_to(out_dir) for p in out_dir.rglob("*.json"))
    second = sorted(p.relative_to(again) for p in again.rglob("*.json"))
    assert first == second
    for rel in first:
        assert (out_dir / rel).read_bytes() == (again / rel).read_bytes(), rel


def test_different_seed_gives_different_events(
    out_dir: Path, tmp_path: Path, mock_fixture: ModuleType
) -> None:
    other = tmp_path / "other"
    bundle = mock_fixture.generate(other, seed=OTHER_SEED)
    assert bundle.meta.run.id == f"mock-{OTHER_SEED}"
    assert len(bundle.events) == 500
    assert bundle.meta.scene.isSynthetic is True
    assert (other / "events.json").read_bytes() != (out_dir / "events.json").read_bytes()
    assert (other / "stations.json").read_bytes() != (out_dir / "stations.json").read_bytes()

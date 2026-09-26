"""CONTRACT-01 acceptance: every model round-trips through JSON; every table model round-trips
through to_frame/from_frame and parquet. All data here is tiny and built inside the test."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from hq_contracts import io
from hq_contracts import models as m
from hq_contracts.schema import bundle_schema
from pydantic import BaseModel, ValidationError

pytestmark = pytest.mark.smoke

RUN_ID = "20260910-0000-abc1234"


def enu(i: float = 0.0) -> m.Enu:
    return m.Enu(e=100.0 * i, n=-50.0 * i, u=-1200.0 - i)


def station(i: int = 0, *, borehole: bool = False) -> m.Station:
    return m.Station(
        id=f"XX.S{i:02d}",
        network="XX",
        station=f"S{i:02d}",
        latitude=38.5 + 0.01 * i,
        longitude=-112.9 - 0.01 * i,
        surfaceElevM=1700.0,
        sensorDepthM=300.0 if borehole else 0.0,
        sensorElevM=1400.0 if borehole else 1700.0,
        kind="borehole" if borehole else "surface",
        channels=["DPZ", "DP1", "DP2"] if borehole else ["HHZ", "HHN", "HHE"],
        sampleRateHz=1000.0 if borehole else 100.0,
        enu=enu(i),
        preprocessProfile="borehole-A" if borehole else "surface",
        usedInRun=True,
        staticsS={"P": 0.01 * i} if i else {},
    )


def pick(i: int = 0, *, assigned: bool = False) -> m.Pick:
    return m.Pick(
        id=f"phasenet:XX.S00:P:{1000.0 + i:.3f}",
        stationId="XX.S00",
        phase="P" if i % 2 == 0 else "S",
        t=1000.0 + i,
        prob=0.9,
        picker="phasenet:original",
        eventId=f"hq-{RUN_ID}-000001" if assigned else None,
        residualS=0.02 if assigned else None,
        weight=1.0 if assigned else None,
    )


def quality() -> m.LocationQuality:
    return m.LocationQuality(
        method="grid1d",
        statics=True,
        nStations=9,
        nP=9,
        nS=7,
        rmsS=0.04,
        gapDeg=120.0,
        minEpiDistM=2500.0,
        hErrM=350.0,
        vErrM=None,
        depthOnEdge=False,
    )


def event(i: int = 0, *, matched: bool = False, mag: bool = False) -> m.SeismicEvent:
    return m.SeismicEvent(
        id=f"hq-{RUN_ID}-{i:06d}",
        runId=RUN_ID,
        t=1.0e9 + i,
        latitude=38.5,
        longitude=-112.9,
        elevM=-1500.0,
        depthKm=3.1277,
        enu=enu(i),
        quality=quality(),
        tier="A",
        tierReasons=["rmsS 0.040 <= 0.055 (p75 of matched)"],
        meanPickProb=0.87,
        magnitude=m.Magnitude(value=0.8, type="ML_cal", sigma=0.2) if mag else None,
        catalogMatch=m.CatalogMatch(catalogId="uu60000001", dtS=0.4, distM=900.0)
        if matched
        else None,
        revealOrder=-1,
        pickIds=[pick(0).id],
    )


def catalog_event(i: int = 0) -> m.CatalogEvent:
    return m.CatalogEvent(
        id=f"uu6000000{i}",
        source="UU via USGS ComCat",
        t=1.0e9 + i,
        latitude=38.5,
        longitude=-112.9,
        depthKm=2.0,
        depthDatum="sea level",
        elevM=-2000.0,
        mag=1.2 if i else None,
        magType="ml" if i else None,
        enu=enu(i),
        matchedEventId=event(i).id if i else None,
    )


def snippet(i: int = 0) -> m.WaveformSnippet:
    return m.WaveformSnippet(
        stationId=f"XX.S{i:02d}",
        channel="HHZ",
        epiDistM=1000.0 * (i + 1),
        t0=1.0e9,
        dt=0.01,
        samples=[0.0, 0.5, -0.5, 1.0],
        pickP=1.0e9 + 1.0,
        pickS=None,
        probP=0.9,
        probS=None,
        predP=1.0e9 + 1.05,
        predS=1.0e9 + 2.1,
    )


def source_ref() -> m.SourceRef:
    return m.SourceRef(
        citation="synthetic", url="https://example.invalid/synthetic", verified=False
    )


def processing_run() -> m.ProcessingRun:
    return m.ProcessingRun(
        id=RUN_ID,
        mode="mock",
        createdAt="2026-09-26T01:00:00Z",
        gitSha="abc1234",
        windowStart=1.0e9,
        windowEnd=1.0e9 + 86400,
        windowLabel="window",
        bbox=(-113.2, 38.28, -112.6, 38.74),
        stationIds=[station(0).id],
        pickerModel="seisbench.PhaseNet",
        pickerWeights="original",
        softwareVersions={"python": "3.11"},
        runtimeS={"pick": 12.5},
        picker={"threshold": 0.3},
        associator={"n_picks": 6},
        velocityModel={"name": "synthetic"},
        locator={"grid": "1d"},
        tiering={"rmsS_p75": 0.055},
        matching={"dtS": 2.0},
        isSynthetic=True,
    )


def tier_counts() -> m.TierCounts:
    return m.TierCounts(A=1, B=2, C=3)


def summary(*, baseline: bool = True) -> m.AnalysisSummary:
    return m.AnalysisSummary(
        runId=RUN_ID,
        publicCatalogCount=4,
        recoveredCatalogCount=3,
        recall=0.75,
        unmatchedPublicIds=["uu60000009"],
        candidateCount=10,
        additionalCount=7,
        additional=tier_counts(),
        strictQualityCount=2,
        strictAdditionalCount=1,
        medianStations=8.0,
        medianRmsS=0.05,
        baseline=(
            m.BaselineGain(associationProfile="full", strictPhasenet=4, strictStalta=2, gain=2.0)
            if baseline
            else None
        ),
    )


def scene_meta() -> m.SceneMeta:
    return m.SceneMeta(
        runId=RUN_ID,
        originLat=38.51,
        originLon=-112.9,
        originElevM=1627.7,
        refSurfaceElevM=1627.7,
        projection="EPSG:32612 minus origin",
        depthLabel="Depth below site surface",
        heroEventId=event(0).id,
        isSynthetic=True,
    )


def validation() -> m.Validation:
    return m.Validation(
        baseline=[
            m.BaselineRow(
                method="phasenet",
                associationProfile="full",
                candidates=10,
                recoveredPublic=3,
                tiers=tier_counts(),
                medianRmsS=0.05,
                medianStations=8.0,
            )
        ],
        sweep=[sweep_point()],
        nullTest=m.NullTest(
            nShuffles=20,
            shiftRangeS=30.0,
            meanChanceEvents=0.5,
            meanChanceStrict=0.0,
            stdChanceEvents=0.7,
        ),
        gr=m.GRCurve(
            magBins=[0.0, 1.0], publicCum=[4, 1], recoveredCum=[9, 2], mcPublic=1.0, bValue=1.0
        ),
        magnitude=m.MagCalibration(n=3, looMae=0.2, coefficients={"a": 1.0, "b": -0.5}),
        synthetic=m.SyntheticTest(
            nEvents=50,
            pickSigmaS={"P": 0.05, "S": 0.1},
            medianHErrM=300.0,
            medianVErrM=600.0,
            p90VErrM=1200.0,
            medianDepthBiasM=-50.0,
        ),
    )


def sweep_point() -> m.SweepPoint:
    return m.SweepPoint(
        params={"n_picks": 6, "profile": "full"}, candidates=10, recoveredPublic=3, tierA=2
    )


def one_of_each() -> list[BaseModel]:
    """One instance of every model in docs/02 §1, with optional fields both set and unset."""
    evidence = m.EventEvidence(
        eventId=event(0).id, filterHz=(2.0, 20.0), traces=[snippet(0), snippet(1)]
    )
    feature = m.GeoFeature(
        id="well-1", kind="well", name="synthetic well", path=[enu(0), enu(1)], source=source_ref()
    )
    live = m.LiveStatus(
        updatedAt=1.0e9, windowS=7200.0, latencyS=240.0, stationsOnline=12, events=[event(3)]
    )
    meta = m.BundleMeta(mode="mock", scene=scene_meta(), run=processing_run(), summary=summary())
    bundle = m.Bundle(
        meta=meta,
        stations=[station(0), station(1, borehole=True)],
        catalog=[catalog_event(0), catalog_event(1)],
        events=[event(0), event(1, matched=True, mag=True)],
        features=[feature],
        validation=validation(),
        evidence=evidence,
        live=live,
    )
    return [
        enu(),
        station(0),
        station(1, borehole=True),
        pick(0),
        pick(1, assigned=True),
        quality(),
        m.CatalogMatch(catalogId="x", dtS=0.1, distM=10.0),
        m.Magnitude(value=1.0, type="ml"),
        event(0),
        event(1, matched=True, mag=True),
        catalog_event(0),
        catalog_event(1),
        snippet(),
        evidence,
        source_ref(),
        feature,
        processing_run(),
        tier_counts(),
        m.BaselineGain(associationProfile="p_only", strictPhasenet=3, strictStalta=1, gain=3.0),
        summary(),
        summary(baseline=False),
        scene_meta(),
        validation().baseline[0],
        sweep_point(),
        validation().nullTest,
        validation().gr,
        validation().magnitude,
        validation().synthetic,
        validation(),
        live,
        meta,
        bundle,
    ]


def test_one_of_every_model_round_trips_through_json() -> None:
    instances = one_of_each()
    assert {type(i) for i in instances} == set(m.ALL_MODELS), "add a factory for every model"
    for instance in instances:
        text = instance.model_dump_json()
        parsed = type(instance).model_validate_json(text)
        assert parsed == instance
        assert json.loads(text) == json.loads(parsed.model_dump_json())


def test_unknown_keys_are_an_error() -> None:
    with pytest.raises(ValidationError):
        m.Enu.model_validate({"e": 0.0, "n": 0.0, "u": 0.0, "up": 1.0})


def test_bundle_meta_carries_schema_version() -> None:
    meta = m.BundleMeta(
        mode="showcase", scene=scene_meta(), run=processing_run(), summary=summary()
    )
    assert meta.schemaVersion == m.SCHEMA_VERSION == "1.0"


TABLE_ROWS: dict[type[BaseModel], list[BaseModel]] = {
    m.Station: [station(0), station(1, borehole=True)],
    m.Pick: [pick(0), pick(1, assigned=True)],
    m.CatalogEvent: [catalog_event(0), catalog_event(1)],
    m.SeismicEvent: [event(0), event(1, matched=True, mag=True), event(2, matched=True)],
    m.SweepPoint: [
        sweep_point(),
        m.SweepPoint(params={}, candidates=0, recoveredPublic=0, tierA=0),
    ],
}


def test_every_table_model_has_rows() -> None:
    assert set(TABLE_ROWS) == set(m.TABLE_MODELS)


@pytest.mark.parametrize("model", list(TABLE_ROWS), ids=lambda model: model.__name__)
def test_to_frame_from_frame_round_trip(model: type[BaseModel]) -> None:
    rows = TABLE_ROWS[model]
    df = io.to_frame(rows)
    assert len(df) == len(rows)
    assert list(df.columns) == io.columns_for(model)
    assert io.from_frame(df, model) == rows


def test_flattening_rule() -> None:
    df = io.to_frame([event(1, matched=True, mag=True), event(0)])
    assert {"enu_e", "quality_nStations", "catalogMatch_dtS", "magnitude_sigma"} <= set(df.columns)
    assert "enu" not in df.columns and "quality" not in df.columns
    assert df.loc[0, "catalogMatch_dtS"] == 0.4
    assert pd.isna(df.loc[1, "catalogMatch_dtS"])
    assert isinstance(df.loc[0, "pickIds"], list)
    assert df["t"].dtype == "float64"


def test_dict_fields_are_json_text() -> None:
    df = io.to_frame([station(1), station(0)])
    assert df.loc[0, "staticsS"] == '{"P": 0.01}'
    assert df.loc[1, "staticsS"] == "{}"
    back = io.from_frame(df, m.Station)
    assert back[0].staticsS == {"P": 0.01} and back[1].staticsS == {}


def test_empty_frame_needs_model() -> None:
    with pytest.raises(ValueError):
        io.to_frame([])
    df = io.to_frame([], model=m.Pick)
    assert list(df.columns) == io.columns_for(m.Pick) and len(df) == 0
    assert io.from_frame(df, m.Pick) == []


def test_from_frame_rejects_missing_columns() -> None:
    df = io.to_frame([pick(0)]).drop(columns=["prob"])
    with pytest.raises(io.TableSchemaError):
        io.from_frame(df, m.Pick)


@pytest.mark.parametrize("model", list(TABLE_ROWS), ids=lambda model: model.__name__)
def test_parquet_round_trip(tmp_path, model: type[BaseModel]) -> None:
    rows = TABLE_ROWS[model]
    path = tmp_path / f"{model.__name__}.parquet"
    io.write_table(io.to_frame(rows), path, model.__name__)
    df = io.read_table(path)
    assert df.attrs == {"schemaVersion": "1.0", "model": model.__name__}
    assert io.from_frame(df, model) == rows
    assert io.read_models(path, model) == rows


def test_parquet_empty_table_round_trip(tmp_path) -> None:
    path = tmp_path / "empty.parquet"
    io.write_models([], path, model=m.SeismicEvent)
    assert io.read_models(path, m.SeismicEvent) == []


def test_read_table_rejects_foreign_parquet(tmp_path) -> None:
    path = tmp_path / "foreign.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(path)
    with pytest.raises(io.TableSchemaError):
        io.read_table(path)


def test_read_models_checks_model_name(tmp_path) -> None:
    path = tmp_path / "picks.parquet"
    io.write_models([pick(0)], path)
    with pytest.raises(io.TableSchemaError):
        io.read_models(path, m.Station)


def test_schema_exports_every_model_by_name() -> None:
    schema = bundle_schema()
    names = set(schema["$defs"])
    assert {model.__name__ for model in m.ALL_MODELS if model is not m.Bundle} <= names
    assert {"Tier", "Phase", "DataMode"} <= names, "Literal aliases must export as named TS types"


REPO = Path(__file__).resolve().parents[4]


def test_committed_schema_json_is_current() -> None:
    """`make contracts` was run after the last model change."""
    committed = json.loads((REPO / "packages/contracts/schema.json").read_text())
    assert committed == bundle_schema(), "run `make contracts` and commit the result"


def test_provider_types_reexport_every_model() -> None:
    text = (REPO / "apps/web/src/providers/types.ts").read_text()
    block = re.search(r"export type \{([^}]*)\} from \"@hq/contracts\";", text)
    assert block is not None
    exported = {name.strip() for name in block.group(1).split(",") if name.strip()}
    expected = {model.__name__ for model in m.ALL_MODELS} | {"Tier", "Phase", "DataMode"}
    assert expected <= exported, f"missing from types.ts: {sorted(expected - exported)}"

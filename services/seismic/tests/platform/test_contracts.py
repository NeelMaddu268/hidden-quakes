"""CONTRACT-01 acceptance: every model round-trips through JSON; every table model round-trips
through to_frame/from_frame and parquet. CONTRACT-02 (REQ-H2-3): column dtypes come from the
model annotations, so empty and all-null tables carry the same pandas dtypes and Arrow types as
populated ones. All data here is tiny and built inside the test."""

import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


def confidence() -> m.Confidence:
    return m.Confidence(
        schema="hq.confidence/1",
        runId="r",
        model={"name": "gbm", "features": ["nStations", "rmsS"]},
        heldOutRocAuc=0.91,
        scores={event(0).id: 0.8, event(1).id: 0.2},
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
        confidence=confidence(),
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
        confidence(),
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


# --- CONTRACT-02: dtype-stable frames (REQ-H2-3) ------------------------------------------

# pandas dtype -> the Arrow type write_table must produce, even with zero rows or all nulls.
# pandas 3 stores its `string` dtype as Arrow large_string; both are string types, never null.
ARROW_TYPES: dict[str, tuple[str, ...]] = {
    "float64": ("double",),
    "int64": ("int64",),
    "Int64": ("int64",),
    "bool": ("bool",),
    "boolean": ("bool",),
    "string": ("string", "large_string"),
}

# The full dtype map for the two models every lane writes, spelled out so a change in the
# rules (or in pandas' inference) fails loudly here rather than in a neighbour's stage.
EXPECTED_DTYPES: dict[type[BaseModel], dict[str, str]] = {
    m.Pick: {
        "id": "string",
        "stationId": "string",
        "phase": "string",
        "t": "float64",
        "prob": "float64",
        "picker": "string",
        "eventId": "string",
        "residualS": "float64",
        "weight": "float64",
    },
    m.SeismicEvent: {
        "id": "string",
        "runId": "string",
        "source": "string",
        "t": "float64",
        "latitude": "float64",
        "longitude": "float64",
        "elevM": "float64",
        "depthKm": "float64",
        "enu_e": "float64",
        "enu_n": "float64",
        "enu_u": "float64",
        "quality_method": "string",
        "quality_statics": "bool",
        "quality_nStations": "int64",
        "quality_nP": "int64",
        "quality_nS": "int64",
        "quality_rmsS": "float64",
        "quality_gapDeg": "float64",
        "quality_minEpiDistM": "float64",
        "quality_hErrM": "float64",
        "quality_vErrM": "float64",
        "quality_depthOnEdge": "bool",
        "tier": "string",
        "tierReasons": "object",
        "meanPickProb": "float64",
        "magnitude_value": "float64",
        "magnitude_type": "string",
        "magnitude_sigma": "float64",
        "catalogMatch_catalogId": "string",
        "catalogMatch_dtS": "float64",
        "catalogMatch_distM": "float64",
        "revealOrder": "int64",
        "pickIds": "object",
    },
}

# Spot checks on the other table models: one column per rule they exercise.
EXPECTED_DTYPES_SPOT: dict[type[BaseModel], dict[str, str]] = {
    m.Station: {
        "kind": "string",
        "channels": "object",
        "usedInRun": "bool",
        "staticsS": "string",  # dict -> JSON text
        "sampleRateHz": "float64",
    },
    m.CatalogEvent: {
        "t": "float64",
        "mag": "float64",
        "magType": "string",
        "matchedEventId": "string",
    },
    m.SweepPoint: {"params": "string", "candidates": "int64", "tierA": "int64"},
}


def dtype_names(df: pd.DataFrame) -> dict[str, str]:
    return {str(col): str(dtype) for col, dtype in df.dtypes.items()}


def arrow_types(path: Path) -> dict[str, str]:
    return {field.name: str(field.type) for field in pq.read_schema(path)}


def assert_arrow_types_match(df: pd.DataFrame, path: Path) -> None:
    """Every typed pandas column wrote its real Arrow type; only `object` columns may be `null`."""
    actual = arrow_types(path)
    for col, dtype in dtype_names(df).items():
        if dtype == "object":
            continue
        assert actual[col] in ARROW_TYPES[dtype], f"{col}: {dtype} wrote Arrow {actual[col]}"


@pytest.mark.parametrize("model", list(TABLE_ROWS), ids=lambda model: model.__name__)
def test_zero_row_frame_has_model_dtypes(tmp_path, model: type[BaseModel]) -> None:
    empty = io.to_frame([], model=model)
    populated = io.to_frame(TABLE_ROWS[model])
    assert len(empty) == 0 and list(empty.columns) == io.columns_for(model)
    assert dtype_names(empty) == io.dtypes_for(model) == dtype_names(populated)
    expected = EXPECTED_DTYPES.get(model) or EXPECTED_DTYPES_SPOT[model]
    for col, dtype in expected.items():
        assert dtype_names(empty)[col] == dtype, col
    assert set(dtype_names(empty).values()) <= set(ARROW_TYPES) | {"object"}
    if "t" in empty.columns:
        assert empty["t"].dtype == "float64"

    path = tmp_path / f"{model.__name__}.parquet"
    io.write_table(empty, path, model.__name__)
    assert_arrow_types_match(empty, path)
    back = io.read_table(path)
    assert dtype_names(back) == dtype_names(empty)
    assert io.from_frame(back, model) == []


def test_expected_dtype_maps_are_complete() -> None:
    for model, expected in EXPECTED_DTYPES.items():
        assert list(expected) == io.columns_for(model)
    assert set(EXPECTED_DTYPES) | set(EXPECTED_DTYPES_SPOT) == set(m.TABLE_MODELS)


def test_all_none_optional_columns_keep_their_dtype(tmp_path) -> None:
    # Optional leaf (Pick.eventId, residualS, weight) and optional nested model
    # (SeismicEvent.magnitude, catalogMatch): all-None rows must match populated rows.
    cases: list[tuple[type[BaseModel], list[BaseModel], list[BaseModel]]] = [
        (m.Pick, [pick(0), pick(2)], [pick(1, assigned=True)]),
        (m.SeismicEvent, [event(0), event(1)], [event(2, matched=True, mag=True)]),
        (m.CatalogEvent, [catalog_event(0)], [catalog_event(1)]),
    ]
    for model, all_none, populated in cases:
        df = io.to_frame(all_none)
        assert dtype_names(df) == dtype_names(io.to_frame(populated)) == io.dtypes_for(model)
        assert io.from_frame(df, model) == all_none
        path = tmp_path / f"{model.__name__}.parquet"
        io.write_table(df, path, model.__name__)
        assert_arrow_types_match(df, path)
        back = io.read_table(path)
        assert dtype_names(back) == dtype_names(df)
        assert io.from_frame(back, model) == all_none

    picks = io.to_frame([pick(0)])
    assert picks["eventId"].dtype == "string" and picks["eventId"].isna().all()
    assert picks["residualS"].dtype == "float64" and np.isnan(picks.loc[0, "residualS"])
    events = io.to_frame([event(0)])
    assert events["magnitude_type"].dtype == "string"
    assert events["catalogMatch_catalogId"].dtype == "string"
    assert events["magnitude_value"].dtype == "float64"


def test_optional_nested_model_makes_int_and_bool_leaves_nullable(tmp_path) -> None:
    """int/bool inside an optional nested model use the nullable Int64/boolean dtypes."""

    class Inner(m.Model):
        n: int
        ok: bool
        label: Literal["x", "y"]

    class Outer(m.Model):
        id: str
        count: int
        flag: bool
        inner: Inner | None = None

    assert io.dtypes_for(Outer) == {
        "id": "string",
        "count": "int64",
        "flag": "bool",
        "inner_n": "Int64",
        "inner_ok": "boolean",
        "inner_label": "string",
    }
    rows = [
        Outer(id="a", count=1, flag=True, inner=Inner(n=3, ok=False, label="x")),
        Outer(id="b", count=2, flag=False),
    ]
    for subset in (rows, rows[1:], []):
        df = io.to_frame(subset, model=Outer)
        assert dtype_names(df) == io.dtypes_for(Outer)
        assert io.from_frame(df, Outer) == subset
        path = tmp_path / "outer.parquet"
        io.write_table(df, path, Outer.__name__)
        assert_arrow_types_match(df, path)
        back = io.read_table(path)
        assert dtype_names(back) == dtype_names(df)
        assert io.from_frame(back, Outer) == subset
    assert isinstance(io.from_frame(io.to_frame(rows), Outer)[0].inner.n, int)  # type: ignore[union-attr]


@pytest.mark.parametrize("model", list(TABLE_ROWS), ids=lambda model: model.__name__)
def test_t_is_always_float64(model: type[BaseModel]) -> None:
    if "t" not in io.columns_for(model):
        pytest.skip(f"{model.__name__} has no t column")
    assert io.to_frame([], model=model)["t"].dtype == "float64"
    assert io.to_frame(TABLE_ROWS[model])["t"].dtype == "float64"
    assert io.dtypes_for(model)["t"] == "float64"


def test_from_frame_accepts_na_nan_and_none_for_optional_fields() -> None:
    """Frames built elsewhere may hold pd.NA, NaN or None in optional columns; all mean None."""
    df = io.to_frame([pick(1, assigned=True)] * 3)
    df = df.astype({"eventId": "object", "residualS": "object", "weight": "object"})
    df.loc[0, ["eventId", "residualS", "weight"]] = [pd.NA, pd.NA, pd.NA]
    df.loc[1, ["eventId", "residualS", "weight"]] = [np.nan, np.nan, np.nan]
    df.loc[2, ["eventId", "residualS", "weight"]] = [None, None, None]
    assert io.from_frame(df, m.Pick) == [pick(1)] * 3

    typed = io.to_frame([pick(1, assigned=True), pick(1)])
    back = io.from_frame(typed, m.Pick)
    assert back == [pick(1, assigned=True), pick(1)]
    assert type(back[0].eventId) is str and type(back[0].t) is float


def test_columns_for_order_unchanged_by_dtypes() -> None:
    assert io.columns_for(m.SeismicEvent)[:12] == [
        "id",
        "runId",
        "source",
        "t",
        "latitude",
        "longitude",
        "elevM",
        "depthKm",
        "enu_e",
        "enu_n",
        "enu_u",
        "quality_method",
    ]
    assert list(io.dtypes_for(m.SeismicEvent)) == io.columns_for(m.SeismicEvent)
    assert pa.Table.from_pandas(io.to_frame([], model=m.Pick)).column_names == io.columns_for(
        m.Pick
    )


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

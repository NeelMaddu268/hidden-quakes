"""Hidden Quakes shared data contracts (docs/02 §1).

Python is the source of truth. ``packages/contracts/ts/src/index.ts`` is generated from
``Bundle.model_json_schema()`` by ``scripts/gen-contracts.sh``; never edit it by hand.

Conventions (docs/01): one vertical (``elevM``, m above sea level, up positive); times are epoch
seconds UTC as float64; horizontal ENU is UTM 12N meters minus the run origin; field names carry
unit suffixes (``M``, ``S``, ``Km``, ``Hz``, ``Deg``). Every model rejects unknown keys.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypeAliasType

SCHEMA_VERSION = "1.0"

# Named aliases (TypeAliasType) so the generated TS exports them as named union types.
Tier = TypeAliasType("Tier", Literal["A", "B", "C"])
Phase = TypeAliasType("Phase", Literal["P", "S"])
DataMode = TypeAliasType("DataMode", Literal["mock", "showcase", "live", "snapshot"])


class Model(BaseModel):
    """Base for every contract model: unknown keys are an error, never silently dropped, and
    NaN/inf floats are rejected (a missing value is ``None``, never NaN; REQ-H2-3)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Enu(Model):
    """Position relative to the run origin, in meters."""

    e: float  # m east of origin
    n: float  # m north of origin
    u: float  # m, elevM - originElevM


class Station(Model):
    id: str  # "UU.FOR1" (net.sta; append .loc only if two locations coexist)
    network: str
    station: str
    location: str = ""
    latitude: float
    longitude: float
    surfaceElevM: float  # site ground surface at the sensor (DEM-checked; wellhead for boreholes)
    sensorDepthM: float  # StationXML channel depth; 0 for surface sensors
    sensorElevM: float  # surfaceElevM - sensorDepthM
    kind: Literal["surface", "borehole", "strong_motion"]
    channels: list[str]  # e.g. ["HHZ", "HHN", "HHE"] or ["DPZ", "DP1", "DP2"]
    sampleRateHz: float
    enu: Enu  # sensor position, not wellhead
    preprocessProfile: str  # key into signal.yaml profiles
    usedInRun: bool
    staticsS: dict[Phase, float] = Field(default_factory=dict)


class Pick(Model):
    id: str  # stable: f"{picker}:{stationId}:{phase}:{t:.3f}"
    stationId: str
    phase: Phase
    t: float  # epoch s UTC
    prob: float  # 0-1
    picker: str  # "phasenet:<weights>" | "stalta"
    eventId: str | None = None
    residualS: float | None = None
    weight: float | None = None


class LocationQuality(Model):
    method: Literal["pyocto", "grid1d", "grid3d", "relative"]
    statics: bool
    nStations: int
    nP: int
    nS: int
    rmsS: float
    gapDeg: float
    minEpiDistM: float
    hErrM: float | None  # 68% horizontal semi-major axis
    vErrM: float | None  # 68% vertical
    depthOnEdge: bool  # >5% of PDF mass on the grid's top or bottom face


class CatalogMatch(Model):
    catalogId: str
    dtS: float
    distM: float


class Magnitude(Model):
    value: float
    type: str  # "ML_cal" (ours) or the catalog's type
    sigma: float | None = None


class SeismicEvent(Model):
    id: str  # "hq-<runId>-000123"
    runId: str
    source: str = "hq-pipeline"
    t: float  # origin time, epoch s UTC
    latitude: float
    longitude: float
    elevM: float  # canonical, m ASL
    depthKm: float  # derived: (refSurfaceElevM - elevM) / 1000
    enu: Enu
    quality: LocationQuality
    tier: Tier
    tierReasons: list[str]  # e.g. "rmsS 0.041 <= 0.055 (p75 of matched)"
    meanPickProb: float  # picker confidence, NOT a probability the event is real
    magnitude: Magnitude | None = None
    catalogMatch: CatalogMatch | None = None
    revealOrder: int  # H2 writes -1; the exporter assigns the real order
    pickIds: list[str]


class CatalogEvent(Model):
    id: str
    source: str  # e.g. "UU via USGS ComCat"
    t: float
    latitude: float
    longitude: float
    depthKm: float  # as published
    depthDatum: str  # what the published depth is relative to
    elevM: float  # converted with that datum
    mag: float | None = None
    magType: str | None = None
    enu: Enu
    matchedEventId: str | None = None


class WaveformSnippet(Model):
    stationId: str
    channel: str
    epiDistM: float
    t0: float  # epoch s of first sample
    dt: float  # s between display samples (~0.01)
    samples: list[float]  # bandpassed display copy, scaled to [-1, 1], 4-8 s long
    pickP: float | None = None  # epoch s
    pickS: float | None = None
    probP: float | None = None
    probS: float | None = None
    predP: float | None = None  # predicted from the final location
    predS: float | None = None


class EventEvidence(Model):
    eventId: str
    filterHz: tuple[float, float]
    traces: list[WaveformSnippet]  # sorted by epiDistM, at most 16


class SourceRef(Model):
    citation: str
    url: str
    verified: bool  # location taken from an authoritative source


class GeoFeature(Model):
    id: str
    kind: Literal["well", "facility", "boundary"]
    name: str
    path: list[Enu]  # one point for a facility; trajectory for a well
    source: SourceRef


class ProcessingRun(Model):
    id: str  # "YYYYMMDD-HHMM-<gitsha7>"
    mode: DataMode
    createdAt: str  # ISO 8601 UTC
    gitSha: str
    windowStart: float
    windowEnd: float
    windowLabel: str  # "2026-09-10 00:00-24:00 UTC"
    bbox: tuple[float, float, float, float]  # minLon, minLat, maxLon, maxLat
    stationIds: list[str]
    pickerModel: str  # "seisbench.PhaseNet"
    pickerWeights: str  # chosen default; per-profile overrides live in picker
    softwareVersions: dict[str, str]  # python, obspy, seisbench, pyocto, torch, scikit-fmm, ...
    runtimeS: dict[str, float]  # per stage
    picker: dict
    associator: dict  # every argument passed to PyOcto
    velocityModel: dict  # name, SourceRef, layers or grid file, datum
    locator: dict
    tiering: dict  # thresholds + the quantiles they came from
    matching: dict
    isSynthetic: bool = False


class TierCounts(Model):
    A: int
    B: int
    C: int


class BaselineGain(Model):
    associationProfile: Literal["full", "p_only"]
    strictPhasenet: int
    strictStalta: int
    gain: float


class AnalysisSummary(Model):
    runId: str
    publicCatalogCount: int
    recoveredCatalogCount: int
    recall: float
    unmatchedPublicIds: list[str]
    candidateCount: int  # all associated + located events
    additionalCount: int  # candidates not matched to the public catalog
    additional: TierCounts
    strictQualityCount: int  # Tier A, all
    strictAdditionalCount: int  # Tier A, additional only
    medianStations: float
    medianRmsS: float
    baseline: BaselineGain | None = None


class SceneMeta(Model):
    runId: str
    originLat: float
    originLon: float
    originElevM: float
    refSurfaceElevM: float
    projection: str  # "EPSG:32612 minus origin"
    verticalExaggeration: float = 1.0
    depthLabel: str
    heroEventId: str | None = None
    isSynthetic: bool = False


class BaselineRow(Model):
    method: Literal["phasenet", "stalta"]
    associationProfile: Literal["full", "p_only"]
    candidates: int
    recoveredPublic: int
    tiers: TierCounts
    medianRmsS: float
    medianStations: float


class SweepPoint(Model):
    params: dict
    candidates: int
    recoveredPublic: int
    tierA: int


class NullTest(Model):
    nShuffles: int
    shiftRangeS: float
    meanChanceEvents: float
    meanChanceStrict: float
    stdChanceEvents: float


class GRCurve(Model):
    magBins: list[float]
    publicCum: list[int]
    recoveredCum: list[int]
    mcPublic: float | None = None
    mcRecovered: float | None = None
    bValue: float | None = None
    bSigma: float | None = None


class MagCalibration(Model):
    n: int
    looMae: float
    coefficients: dict[str, float]


class SyntheticTest(Model):
    nEvents: int
    pickSigmaS: dict[Phase, float]
    medianHErrM: float
    medianVErrM: float
    p90VErrM: float
    medianDepthBiasM: float


class Validation(Model):
    baseline: list[BaselineRow]
    sweep: list[SweepPoint]
    nullTest: NullTest | None = None
    gr: GRCurve | None = None
    magnitude: MagCalibration | None = None
    synthetic: SyntheticTest


CONFIDENCE_SCHEMA = "hq.confidence/1"


class Confidence(Model):
    """ML-01 (H2): one score per candidate event from a classifier trained to tell recovered
    public events from the rest, with the model's metadata and its held-out ROC AUC. A sidecar
    of the run (``confidence.json``) and an optional bundle file next to ``validation.json``;
    absent means no score and no card row. Added Sat evening after the contract freeze as a new
    file that carries its own ``schema`` tag, so ``SCHEMA_VERSION`` and every frozen bundle stay
    as they are. ``scores`` are not probabilities that an event is real (docs/00)."""

    schema_: Literal["hq.confidence/1"] = Field(alias="schema")
    runId: str
    model: dict[str, Any]  # H2's model metadata (name, features, training set), verbatim
    heldOutRocAuc: float | None = None  # None when no held-out split could be scored
    scores: dict[str, float]  # event id -> score

    # ``schema`` is a BaseModel attribute, so the field is ``schema_`` with the alias on disk;
    # dumps write the alias (``model_dump`` / ``model_dump_json`` need no ``by_alias``).
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, populate_by_name=True, serialize_by_alias=True
    )


class LiveStatus(Model):
    updatedAt: float
    windowS: float
    latencyS: float  # data end -> results ready
    stationsOnline: int
    events: list[SeismicEvent]


class BundleMeta(Model):
    schemaVersion: str = SCHEMA_VERSION
    mode: DataMode
    scene: SceneMeta
    run: ProcessingRun
    summary: AnalysisSummary


class Bundle(Model):
    """Schema export only; on disk the bundle is separate files (docs/01 → Data bundle)."""

    meta: BundleMeta
    stations: list[Station]
    catalog: list[CatalogEvent]
    events: list[SeismicEvent]
    features: list[GeoFeature]
    validation: Validation
    confidence: Confidence
    evidence: EventEvidence
    live: LiveStatus


# Models that are rows of a run table (docs/02 §2). Every other model is JSON.
TABLE_MODELS: tuple[type[Model], ...] = (Station, Pick, CatalogEvent, SeismicEvent, SweepPoint)

# Every exported model, in dependency order, for round-trip tests and the schema.
ALL_MODELS: tuple[type[Model], ...] = (
    Enu,
    Station,
    Pick,
    LocationQuality,
    CatalogMatch,
    Magnitude,
    SeismicEvent,
    CatalogEvent,
    WaveformSnippet,
    EventEvidence,
    SourceRef,
    GeoFeature,
    ProcessingRun,
    TierCounts,
    BaselineGain,
    AnalysisSummary,
    SceneMeta,
    BaselineRow,
    SweepPoint,
    NullTest,
    GRCurve,
    MagCalibration,
    SyntheticTest,
    Validation,
    Confidence,
    LiveStatus,
    BundleMeta,
    Bundle,
)

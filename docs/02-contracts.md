# 02 · Contracts

Every interface one lane consumes from another is defined here. **Frozen at 8:30 PM Friday.** After that, a change is one PR to `main` that updates this file and `packages/contracts/` together, bumps `SCHEMA_VERSION`, and is approved by every lane it touches.

Treat everything below as a spec. H4's CONTRACT-01 agent implements the Python models and generates the TS; nobody pastes these blocks as code.

Implementation notes (CONTRACT-01): every model rejects unknown keys (`extra="forbid"`), so a stray JSON key or a misspelled field is an error, not a silent drop. Python always serializes defaulted fields, so the generated TS has no optional fields.

## 1. Data models (Python source of truth)

`packages/contracts/python/hq_contracts/models.py`. TS in `packages/contracts/ts/src/index.ts` is generated from these (`scripts/gen-contracts.sh`: `Bundle.model_json_schema()` → `json-schema-to-typescript`).

```python
from typing import Literal
from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"
Tier = Literal["A", "B", "C"]
Phase = Literal["P", "S"]
DataMode = Literal["mock", "showcase", "live", "snapshot"]

class Enu(BaseModel):
    e: float                      # m east of origin
    n: float                      # m north of origin
    u: float                      # m, elevM - originElevM

class Station(BaseModel):
    id: str                       # "UU.FOR1" (net.sta; append .loc only if two locations coexist)
    network: str
    station: str
    location: str = ""
    latitude: float
    longitude: float
    surfaceElevM: float           # site ground surface at the sensor (DEM-checked; wellhead for boreholes)
    sensorDepthM: float           # StationXML channel depth; 0 for surface sensors
    sensorElevM: float            # surfaceElevM - sensorDepthM
    kind: Literal["surface", "borehole", "strong_motion"]
    channels: list[str]           # e.g. ["HHZ","HHN","HHE"] or ["DPZ","DP1","DP2"]
    sampleRateHz: float
    enu: Enu                      # sensor position, not wellhead
    preprocessProfile: str        # key into signal.yaml profiles
    usedInRun: bool
    staticsS: dict[Phase, float] = Field(default_factory=dict)

class Pick(BaseModel):
    id: str                       # stable: f"{picker}:{stationId}:{phase}:{t:.3f}"
    stationId: str
    phase: Phase
    t: float                      # epoch s UTC
    prob: float                   # 0-1
    picker: str                   # "phasenet:<weights>" | "stalta"
    eventId: str | None = None
    residualS: float | None = None
    weight: float | None = None

class LocationQuality(BaseModel):
    method: Literal["pyocto", "grid1d", "grid3d", "relative"]
    statics: bool
    nStations: int
    nP: int
    nS: int
    rmsS: float
    gapDeg: float
    minEpiDistM: float
    hErrM: float | None           # 68% horizontal semi-major axis
    vErrM: float | None           # 68% vertical
    depthOnEdge: bool             # >5% of PDF mass on the grid's top or bottom face

class CatalogMatch(BaseModel):
    catalogId: str
    dtS: float
    distM: float

class Magnitude(BaseModel):
    value: float
    type: str                     # "ML_cal" (ours) or the catalog's type
    sigma: float | None = None

class SeismicEvent(BaseModel):
    id: str                       # "hq-<runId>-000123"
    runId: str
    source: str = "hq-pipeline"
    t: float                      # origin time, epoch s UTC
    latitude: float
    longitude: float
    elevM: float                  # canonical, m ASL
    depthKm: float                # derived: (refSurfaceElevM - elevM) / 1000
    enu: Enu
    quality: LocationQuality
    tier: Tier
    tierReasons: list[str]        # e.g. "rmsS 0.041 <= 0.055 (p75 of matched)"
    meanPickProb: float           # picker confidence, NOT a probability the event is real
    magnitude: Magnitude | None = None
    catalogMatch: CatalogMatch | None = None
    revealOrder: int              # H2 writes -1; the exporter assigns the real order
    pickIds: list[str]

class CatalogEvent(BaseModel):
    id: str
    source: str                   # e.g. "UU via USGS ComCat"
    t: float
    latitude: float
    longitude: float
    depthKm: float                # as published
    depthDatum: str               # what the published depth is relative to
    elevM: float                  # converted with that datum
    mag: float | None = None
    magType: str | None = None
    enu: Enu
    matchedEventId: str | None = None

class WaveformSnippet(BaseModel):
    stationId: str
    channel: str
    epiDistM: float
    t0: float                     # epoch s of first sample
    dt: float                     # s between display samples (~0.01)
    samples: list[float]          # bandpassed display copy, scaled to [-1, 1], 4-8 s long
    pickP: float | None = None    # epoch s
    pickS: float | None = None
    probP: float | None = None
    probS: float | None = None
    predP: float | None = None    # predicted from the final location
    predS: float | None = None

class EventEvidence(BaseModel):
    eventId: str
    filterHz: tuple[float, float]
    traces: list[WaveformSnippet] # sorted by epiDistM, at most 16

class SourceRef(BaseModel):
    citation: str
    url: str
    verified: bool                # location taken from an authoritative source

class GeoFeature(BaseModel):
    id: str
    kind: Literal["well", "facility", "boundary"]
    name: str
    path: list[Enu]               # one point for a facility; trajectory for a well
    source: SourceRef

class ProcessingRun(BaseModel):
    id: str                       # "YYYYMMDD-HHMM-<gitsha7>"
    mode: DataMode
    createdAt: str                # ISO 8601 UTC
    gitSha: str
    windowStart: float
    windowEnd: float
    windowLabel: str              # "2026-09-10 00:00-24:00 UTC"
    bbox: tuple[float, float, float, float]   # minLon, minLat, maxLon, maxLat
    stationIds: list[str]
    pickerModel: str              # "seisbench.PhaseNet"
    pickerWeights: str            # chosen default; per-profile overrides live in picker
    softwareVersions: dict[str, str]  # python, obspy, seisbench, pyocto, torch, scikit-fmm, ...
    runtimeS: dict[str, float]    # per stage
    picker: dict
    associator: dict              # every argument passed to PyOcto
    velocityModel: dict           # name, SourceRef, layers or grid file, datum
    locator: dict
    tiering: dict                 # thresholds + the quantiles they came from
    matching: dict
    isSynthetic: bool = False

class TierCounts(BaseModel):
    A: int
    B: int
    C: int

class BaselineGain(BaseModel):
    associationProfile: Literal["full", "p_only"]
    strictPhasenet: int
    strictStalta: int
    gain: float

class AnalysisSummary(BaseModel):
    runId: str
    publicCatalogCount: int
    recoveredCatalogCount: int
    recall: float
    unmatchedPublicIds: list[str]
    candidateCount: int           # all associated + located events
    additionalCount: int          # candidates not matched to the public catalog
    additional: TierCounts
    strictQualityCount: int       # Tier A, all
    strictAdditionalCount: int    # Tier A, additional only
    medianStations: float
    medianRmsS: float
    baseline: BaselineGain | None = None

class SceneMeta(BaseModel):
    runId: str
    originLat: float
    originLon: float
    originElevM: float
    refSurfaceElevM: float
    projection: str               # "EPSG:32612 minus origin"
    verticalExaggeration: float = 1.0
    depthLabel: str
    heroEventId: str | None = None
    isSynthetic: bool = False

class BaselineRow(BaseModel):
    method: Literal["phasenet", "stalta"]
    associationProfile: Literal["full", "p_only"]
    candidates: int
    recoveredPublic: int
    tiers: TierCounts
    medianRmsS: float
    medianStations: float

class SweepPoint(BaseModel):
    params: dict
    candidates: int
    recoveredPublic: int
    tierA: int

class NullTest(BaseModel):
    nShuffles: int
    shiftRangeS: float
    meanChanceEvents: float
    meanChanceStrict: float
    stdChanceEvents: float

class GRCurve(BaseModel):
    magBins: list[float]
    publicCum: list[int]
    recoveredCum: list[int]
    mcPublic: float | None = None
    mcRecovered: float | None = None
    bValue: float | None = None
    bSigma: float | None = None

class MagCalibration(BaseModel):
    n: int
    looMae: float
    coefficients: dict[str, float]

class SyntheticTest(BaseModel):
    nEvents: int
    pickSigmaS: dict[Phase, float]
    medianHErrM: float
    medianVErrM: float
    p90VErrM: float
    medianDepthBiasM: float

class Validation(BaseModel):
    baseline: list[BaselineRow]
    sweep: list[SweepPoint]
    nullTest: NullTest | None = None
    gr: GRCurve | None = None
    magnitude: MagCalibration | None = None
    synthetic: SyntheticTest

class LiveStatus(BaseModel):
    updatedAt: float
    windowS: float
    latencyS: float               # data end -> results ready
    stationsOnline: int
    events: list[SeismicEvent]

class BundleMeta(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    mode: DataMode
    scene: SceneMeta
    run: ProcessingRun
    summary: AnalysisSummary

class Bundle(BaseModel):          # schema export only; the bundle is separate files
    meta: BundleMeta
    stations: list[Station]
    catalog: list[CatalogEvent]
    events: list[SeismicEvent]
    features: list[GeoFeature]
    validation: Validation
    evidence: EventEvidence
    live: LiveStatus
```

## 2. Tables on disk (`hq_contracts.io`)

CONTRACT-01 also ships `packages/contracts/python/hq_contracts/io.py`:

- `to_frame(models: list[BaseModel]) -> pd.DataFrame` and `from_frame(df, Model) -> list[Model]`
- `write_table(df, path, model_name)` / `read_table(path) -> pd.DataFrame`: parquet with `schemaVersion` and `model` in the file metadata
- **Flattening rule:** nested models become prefixed columns joined by `_` (`enu_e`, `quality_nStations`, `catalogMatch_dtS`). Lists stay list columns. `None` stays null. Times stay float64 epoch seconds. `dict`-typed fields (`Station.staticsS`, `SweepPoint.params`) are one JSON-text column, because a parquet struct can't hold a row-dependent key set; build frames with `to_frame` and read them with `from_frame` and you never see it.
- **Dtype rule (CONTRACT-02, REQ-H2-3):** `to_frame` sets every column's dtype from the model annotation, so an empty table or an all-null column has the same type as a full one: `float` → `float64` (NaN for null), `int` → `int64` (`Int64` when optional or inside an optional nested model), `bool` → `bool` (`boolean` when optional), `str` / `Literal` / dict-as-JSON → `string`, lists → `object`. `dtypes_for(Model)` returns the map. Parquet then carries real Arrow types (`double`, `int64`, `bool`, `large_string`), never `null`. `Model` also sets `allow_inf_nan=False`, so a NaN in a required float fails at write time, not in the exporter.

Every lane reads and writes run tables only through these helpers, so a column rename can't silently break a neighbor.

| File in `runs/<runId>/` | Writer | Rows | Columns |
| --- | --- | --- | --- |
| `stations.parquet` | H1 | `Station` | model fields |
| `inventory_report.json` | H1 | one entry per station considered | per-station elevation decision with its numbers, coverage, dropped triplets, skipped sites, flags; diagnostic, never read by another stage |
| `gaps.parquet` | H1 | one per gap | `stationId, channel, gapStart, gapEnd` |
| `picks.parquet` | H1 | `Pick` (PhaseNet) | model fields; `eventId` null |
| `picks_stalta.parquet` | H1 | `Pick` (`picker="stalta"`) | model fields |
| `known/picks.parquet` | H1 | `Pick` for the 3 known-event windows | model fields |
| `catalog.parquet` | H2 | `CatalogEvent` | model fields; `matchedEventId` null until match |
| `assoc_events.parquet` | H2 | one per associated event | `assocId, t, latitude, longitude, elevM, nPicks, nP, nS` |
| `assoc_picks.parquet` | H2 | one per associated pick | `assocId, pickId` |
| `events_located.parquet` | H2 | located events | `SeismicEvent` fields except `tier, tierReasons, catalogMatch, magnitude`; `revealOrder = -1` |
| `arrivals.parquet` | H2 | one per event × station × phase | `eventId, stationId, phase, tPred, tObs, residualS, pickId, usedInLocation` (tObs/residualS/pickId null where no pick) |
| `statics.parquet` | H2 | one per station × phase | `stationId, phase, staticS, nEvents` |
| `matches.parquet` | H2 | one per public event | `catalogId, eventId, dtS, distM, reason` (`eventId` null + `reason` when unmatched) |
| `match_sensitivity.parquet` | H2 | one per tolerance pair | `dtS, distM, recovered` |
| `events.parquet` | H2 | `SeismicEvent` (final) | model fields; `revealOrder = -1` |
| `event_picks.parquet` | H2 | `Pick` for associated picks | model fields with `eventId` and `residualS` set |
| `sweep.parquet` | H2 | `SweepPoint` | model fields |
| `synthetic.json` | H2 | `SyntheticTest` | JSON |
| `diagnostics.md` | H2 | depth diagnostics table + conclusions | Markdown, human-readable |
| `magnitude.json` | H2 | `MagCalibration` | JSON (P1) |
| `null_test.json` | H4 | `NullTest` | JSON sidecar; always written by `validate`, also embedded in `validation.json` |
| `validation.json` | H4 | `Validation` | JSON; written once H2's `synthetic.json` exists |
| `run.json` | every stage via `ctx.record` | `ProcessingRun` | JSON |
| `stages.json` | every stage via `ctx.record` | `{stage: {runtimeS, counts}}` | JSON sidecar; `ProcessingRun` has no counts field |

## 3. Config files

One YAML file per lane in `services/seismic/configs/showcase/`, one Pydantic config model per lane in `hq/config/`:

| File | Owner | Model | Holds |
| --- | --- | --- | --- |
| `run.yaml` | H2 | `RunSection` (`hq/config/run.py`) | `name`, `windowStart`, `windowEnd` (ISO UTC), `bbox`, `origin {lat, lon, elevM}`, `refSurfaceElevM` |
| `signal.yaml` | H1 | `SignalConfig` (`hq/config/signal.py`) | station selection, download, preprocessing profiles, picker weights and thresholds, baseline |
| `seismology.yaml` | H2 | `SeismologyConfig` (`hq/config/seismology.py`) | catalog query, velocity model, grids, associator, locator, statics, tiering, matching, magnitude |
| `export.yaml` | H4 | `ExportConfig` (`hq/config/export.py`) | modes, hero rule, evidence window and band, max traces, display rate |

`hq.config.load_config(dir: Path) -> RunConfig`, where `RunConfig` has fields `run`, `signal`, `seismology`, `export`. Unknown keys are an error, not a warning.

## 4. Stage API (`hq/runs.py`, H4)

```python
@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path                 # data/showcase/runs/<run_id>
    cache_dir: Path               # data/cache
    config: RunConfig
    def path(self, name: str) -> Path: ...        # run_dir / name
    def read_run(self) -> ProcessingRun: ...      # the current run.json
    def record(self, stage: str, *, runtime_s: float,
               counts: dict[str, int], params: dict | None = None,
               field: str | None = None) -> None: ...
        # merges into run.json: runtimeS[stage], plus params into the matching ProcessingRun field
        # (pick → picker, associate → associator, locate → locator, tier → tiering,
        # match and catalog → matching). Other stages pass field= (one of picker, associator,
        # velocityModel, locator, tiering, matching) or omit params. counts go to stages.json.
    def update_run(self, **fields) -> None: ...
        # only stationIds, pickerModel, pickerWeights, softwareVersions; validated through the model

# every stage module exposes exactly this
def run(ctx: RunContext) -> None: ...
```

`hq run configs/showcase [--stages a,b,c]` creates a run and executes stages in order. `hq stage <name> --run <runId>` reruns one stage in place.

## 5. Library APIs (for reuse across lanes)

H1 provides, for H2 (magnitude) and H4 (evidence):

```python
# hq/ingest/cache.py
def read_window(station_id: str, t0: float, t1: float, *, cache_dir: Path) -> obspy.Stream
    # raw counts from the cache; gaps preserved as separate traces; never zero-filled
def read_inventory(station_id: str, *, cache_dir: Path) -> obspy.Inventory   # response included

# hq/preprocess/__init__.py
def display_copy(st: obspy.Stream, band_hz: tuple[float, float]) -> obspy.Stream
    # detrend, taper, zero-phase bandpass; for evidence snippets only
def for_picking(st: obspy.Stream, profile: str, cfg: SignalConfig) -> tuple[obspy.Stream, TimeMap]
    # TimeMap.to_real(t_model) converts model-time picks to real time (identity except borehole-B)
```

H2 provides, for H4 (validation reruns on STA/LTA and time-scrambled picks):

```python
# hq/associate/__init__.py
def associate(picks: pd.DataFrame, stations: pd.DataFrame, cfg: SeismologyConfig,
              run: RunSection) -> AssocResult          # .events, .picks  (assoc_* schemas)
# hq/locate/__init__.py
def locate(assoc: AssocResult, picks: pd.DataFrame, stations: pd.DataFrame,
           cfg: SeismologyConfig, run: RunSection) -> LocateResult   # .events, .arrivals, .statics
# hq/match/__init__.py
def match(events_located: pd.DataFrame, catalog: pd.DataFrame,
          cfg: SeismologyConfig) -> MatchResult        # .matches, .sensitivity
# hq/tier/__init__.py
def assign_tiers(events_located: pd.DataFrame, matches: pd.DataFrame,
                 cfg: SeismologyConfig) -> TierResult  # .events (final schema), .tiering (dict)
```

All results are frozen dataclasses of DataFrames in the table schemas above.

## 6. Web interfaces

### Providers (`apps/web/src/providers/`, H4)

```ts
export interface ModeInfo {
  mode: DataMode; label: string; runId: string; generatedAt: number; isSynthetic: boolean;
}

export interface SeismicDataProvider {
  info(): Promise<ModeInfo>;
  getMeta(): Promise<BundleMeta>;
  getStations(): Promise<Station[]>;
  getCatalog(): Promise<CatalogEvent[]>;
  getEvents(): Promise<SeismicEvent[]>;
  getFeatures(): Promise<GeoFeature[]>;
  getEventEvidence(id: string): Promise<EventEvidence>;
  getValidation(): Promise<Validation | null>;
}

// hooks.ts: the only way components get data
export type BundleState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; info: ModeInfo; meta: BundleMeta; stations: Station[];
      catalog: CatalogEvent[]; events: SeismicEvent[]; features: GeoFeature[] };

export function useBundle(): BundleState;
export function useEvidence(eventId: string | null):
  { status: "idle" | "loading" | "ready" | "error"; evidence?: EventEvidence; message?: string };
export function useValidation(): Validation | null;
export function useLiveStatus(): LiveStatus | null;   // null outside live mode
```

Mode comes from `?mode=` (default `showcase`). A `meta.schemaVersion` mismatch renders a visible error.

### Demo store (`apps/web/src/state/demo.ts`, H3; zustand)

```ts
export type DemoPhase = "public" | "revealing" | "revealed";
export type EventFilter = "public" | "all" | "strict";
export type CameraView = "oblique" | "side" | "plan";

export interface DemoState {
  phase: DemoPhase;
  revealProgress: number;           // 0..1, advanced by the scene during "revealing"
  filter: EventFilter;
  timeMode: boolean;
  tNow: number | null;              // epoch s; null shows every event
  playing: boolean;
  selectedEventId: string | null;
  view: CameraView;
  reveal(): void;                   // public -> revealing; scene sets revealed at progress 1
  reset(): void;                    // back to public, progress 0, filter "public", selection null
  setFilter(f: EventFilter): void;
  setTimeMode(on: boolean): void;
  setTNow(t: number | null): void;
  setPlaying(p: boolean): void;
  select(id: string | null): void;
  setView(v: CameraView): void;
}
export const useDemo: UseBoundStore<StoreApi<DemoState>>;
```

H3 owns the store and everything that animates from it. H4's shell calls the actions (buttons, keyboard) and reads state for counters and labels.

### Components (H3 exports; H4 mounts them in `page.tsx`)

```ts
export function Scene(): JSX.Element;           // apps/web/src/scene/index.ts; full-bleed canvas
export function EvidenceDrawer(): JSX.Element;  // apps/web/src/drawer/index.ts; reads selectedEventId
export function TimeScrubber(): JSX.Element;    // apps/web/src/scene/time/index.ts
```

### Design tokens (`packages/visualization/tokens.ts`, H3)

```ts
export const colors: { bg; surface; terrain; contour; text; textDim; public; recovered;
                       strictHalo; geo; pickP; pickS; station; alert: string };
export const fonts: { ui: string; mono: string };
export const motion: { micro: 150; state: 600; scene: 1200; ease: string };
```

### Keyboard (bound by H4, acting on the demo store)

| Key | Action |
| --- | --- |
| Space | Next beat: public → reveal → strict → time |
| R | `reset()` |
| S | `setFilter("strict")` (again → `"all"`) |
| T | toggle `setTimeMode` |
| E | `select(meta.scene.heroEventId)` |
| P | toggle `setView("plan")` / `"oblique"` |
| Esc | `select(null)` |

## 7. Live API (P1, `services/api`, H4)

| Route | Returns |
| --- | --- |
| `GET /api/live/meta` | `BundleMeta` for the latest processed window (`mode: "live"`) |
| `GET /api/live/events` | `SeismicEvent[]` |
| `GET /api/live/evidence/{id}` | `EventEvidence` |
| `GET /api/live/status` | `LiveStatus` without `events` |

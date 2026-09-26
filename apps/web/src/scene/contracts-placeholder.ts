// TEMPORARY placeholder types, copied from the docs/02 §1 spec until CONTRACT-01's generated
// `@hq/contracts` lands. Delete this file and point `scene/types.ts` at `@hq/contracts` then.
// Fields with Python defaults are optional here because json-schema-to-typescript makes them optional.

export interface Enu {
  e: number;
  n: number;
  u: number;
}

export interface Station {
  id: string;
  network: string;
  station: string;
  location?: string;
  latitude: number;
  longitude: number;
  surfaceElevM: number;
  sensorDepthM: number;
  sensorElevM: number;
  kind: "surface" | "borehole" | "strong_motion";
  channels: string[];
  sampleRateHz: number;
  enu: Enu;
  preprocessProfile: string;
  usedInRun: boolean;
  staticsS?: { [k: string]: number };
}

export interface LocationQuality {
  method: "pyocto" | "grid1d" | "grid3d" | "relative";
  statics: boolean;
  nStations: number;
  nP: number;
  nS: number;
  rmsS: number;
  gapDeg: number;
  minEpiDistM: number;
  hErrM: number | null;
  vErrM: number | null;
  depthOnEdge: boolean;
}

export interface CatalogMatch {
  catalogId: string;
  dtS: number;
  distM: number;
}

export interface Magnitude {
  value: number;
  type: string;
  sigma?: number | null;
}

export interface SeismicEvent {
  id: string;
  runId: string;
  source?: string;
  t: number;
  latitude: number;
  longitude: number;
  elevM: number;
  depthKm: number;
  enu: Enu;
  quality: LocationQuality;
  tier: "A" | "B" | "C";
  tierReasons: string[];
  meanPickProb: number;
  magnitude?: Magnitude | null;
  catalogMatch?: CatalogMatch | null;
  revealOrder: number;
  pickIds: string[];
}

export interface CatalogEvent {
  id: string;
  source: string;
  t: number;
  latitude: number;
  longitude: number;
  depthKm: number;
  depthDatum: string;
  elevM: number;
  mag?: number | null;
  magType?: string | null;
  enu: Enu;
  matchedEventId?: string | null;
}

export interface SourceRef {
  citation: string;
  url: string;
  verified: boolean;
}

export interface GeoFeature {
  id: string;
  kind: "well" | "facility" | "boundary";
  name: string;
  path: Enu[];
  source: SourceRef;
}

export interface SceneMeta {
  runId: string;
  originLat: number;
  originLon: number;
  originElevM: number;
  refSurfaceElevM: number;
  projection: string;
  verticalExaggeration?: number;
  depthLabel: string;
  heroEventId?: string | null;
  isSynthetic?: boolean;
}

export interface TierCounts {
  A: number;
  B: number;
  C: number;
}

export interface AnalysisSummary {
  runId: string;
  publicCatalogCount: number;
  recoveredCatalogCount: number;
  recall: number;
  unmatchedPublicIds: string[];
  candidateCount: number;
  additionalCount: number;
  additional: TierCounts;
  strictQualityCount: number;
  strictAdditionalCount: number;
  medianStations: number;
  medianRmsS: number;
}

export type DataMode = "mock" | "showcase" | "live" | "snapshot";

export interface BundleMeta {
  schemaVersion?: string;
  mode: DataMode;
  scene: SceneMeta;
  run: { id: string; windowStart: number; windowEnd: number; windowLabel: string; isSynthetic?: boolean };
  summary: AnalysisSummary;
}

// docs/02 §6 provider shapes (H4's apps/web/src/providers/types.ts replaces these).
export interface ModeInfo {
  mode: DataMode;
  label: string;
  runId: string;
  generatedAt: number;
  isSynthetic: boolean;
}

export type BundleState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | {
      status: "ready";
      info: ModeInfo;
      meta: BundleMeta;
      stations: Station[];
      catalog: CatalogEvent[];
      events: SeismicEvent[];
      features: GeoFeature[];
    };

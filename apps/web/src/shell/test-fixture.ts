/**
 * A tiny hand-made bundle for shell and page tests (docs/01 → Handoffs: "tiny hand-made JSON in
 * the test"). Every number is invented inside the test suite; nothing derives from a run. The
 * shape mirrors `src/providers/providers.test.tsx` so the same fake fetch serves both.
 */
import type {
  AnalysisSummary,
  BundleMeta,
  CatalogEvent,
  FetchLike,
  SceneMeta,
  SeismicEvent,
  Station,
  Validation,
} from "@/providers";
import { SCHEMA_VERSION } from "@/providers";

export const HERO_EVENT_ID = "ev-hero";

export const SUMMARY: AnalysisSummary = {
  runId: "fixture",
  publicCatalogCount: 12,
  recoveredCatalogCount: 11,
  recall: 11 / 12,
  unmatchedPublicIds: ["pub-9"],
  candidateCount: 140,
  additionalCount: 129,
  additional: { A: 47, B: 52, C: 30 },
  strictQualityCount: 58,
  strictAdditionalCount: 47,
  medianStations: 7,
  medianRmsS: 0.08,
  baseline: null,
};

export interface FixtureOptions {
  meta?: Partial<BundleMeta>;
  scene?: Partial<SceneMeta>;
  summary?: Partial<AnalysisSummary>;
  isSynthetic?: boolean;
  /** Serve this as `validation.json`; absent (the default) means the file 404s. */
  validation?: Validation;
}

export function makeMeta(runId: string, options: FixtureOptions = {}): BundleMeta {
  const isSynthetic = options.isSynthetic ?? true;
  return {
    schemaVersion: SCHEMA_VERSION,
    mode: "mock",
    scene: {
      runId,
      originLat: 0,
      originLon: 0,
      originElevM: 0,
      refSurfaceElevM: 0,
      projection: "EPSG:32612 minus origin",
      verticalExaggeration: 1,
      depthLabel: "Depth below site surface",
      heroEventId: HERO_EVENT_ID,
      isSynthetic,
      ...options.scene,
    },
    run: {
      id: runId,
      mode: "mock",
      createdAt: "2026-09-26T01:00:00Z",
      gitSha: "abc1234",
      windowStart: 0,
      windowEnd: 1,
      windowLabel: "window",
      bbox: [0, 0, 1, 1],
      stationIds: [],
      pickerModel: "x",
      pickerWeights: "y",
      softwareVersions: {},
      runtimeS: {},
      picker: {},
      associator: {},
      velocityModel: {},
      locator: {},
      tiering: {},
      matching: {},
      isSynthetic,
    },
    summary: { ...SUMMARY, runId, ...options.summary },
    ...options.meta,
  };
}

export function makeEvent(id: string, revealOrder: number): SeismicEvent {
  return {
    id,
    runId: "fixture",
    source: "hq-pipeline",
    t: 1,
    latitude: 0,
    longitude: 0,
    elevM: 0,
    depthKm: 0,
    enu: { e: 0, n: 0, u: 0 },
    quality: {
      method: "grid1d",
      statics: false,
      nStations: 5,
      nP: 5,
      nS: 3,
      rmsS: 0.1,
      gapDeg: 90,
      minEpiDistM: 100,
      hErrM: null,
      vErrM: null,
      depthOnEdge: false,
    },
    tier: "A",
    tierReasons: [],
    meanPickProb: 0.9,
    magnitude: null,
    catalogMatch: null,
    revealOrder,
    pickIds: [],
  };
}

const station: Station = {
  id: "XX.T01",
  network: "XX",
  station: "T01",
  location: "",
  latitude: 0,
  longitude: 0,
  surfaceElevM: 0,
  sensorDepthM: 0,
  sensorElevM: 0,
  kind: "surface",
  channels: ["HHZ"],
  sampleRateHz: 100,
  enu: { e: 0, n: 0, u: 0 },
  preprocessProfile: "surface",
  usedInRun: true,
  staticsS: {},
};

const catalogEvent: CatalogEvent = {
  id: "pub-1",
  source: "test",
  t: 1,
  latitude: 0,
  longitude: 0,
  depthKm: 1,
  depthDatum: "sea level",
  elevM: -1000,
  mag: null,
  magType: null,
  enu: { e: 0, n: 0, u: 0 },
  matchedEventId: "ev-1",
};

export type Files = Record<string, unknown>;

/** The files under `/data/<mode>/` for one bundle. Evidence is absent (404); so is
 *  `validation.json` unless `options.validation` supplies one. */
export function bundleFiles(runId: string, options: FixtureOptions = {}): Files {
  const files: Files = {
    "meta.json": makeMeta(runId, options),
    "stations.json": [station],
    "catalog.json": [catalogEvent],
    "events.json": [makeEvent("ev-1", 1), makeEvent(HERO_EVENT_ID, 0)],
    "features.json": [],
  };
  if (options.validation) files["validation.json"] = options.validation;
  return files;
}

/** A fake fetch serving `/data/<mode>/<file>` from in-memory bundles; anything else is a 404. */
export function fakeFetch(bundles: Record<string, Files>, delayMs = 0): FetchLike & { calls: string[] } {
  const calls: string[] = [];
  const impl = async (url: string) => {
    calls.push(url);
    if (delayMs) await new Promise((resolve) => setTimeout(resolve, delayMs));
    const match = /^\/data\/([^/]+)\/(.+)$/.exec(url);
    const files = match ? bundles[match[1]] : undefined;
    const body = files ? files[match![2]] : undefined;
    if (body === undefined) return { ok: false, status: 404, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => body };
  };
  return Object.assign(impl, { calls });
}

/** A fetch that never resolves: the bundle stays "loading" for the whole test. */
export const pendingFetch: FetchLike = () => new Promise(() => undefined);

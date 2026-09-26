/**
 * API-01 acceptance: mock ↔ showcase swap with no component changes; a schemaVersion mismatch
 * renders a visible error. The bundle here is a tiny hand-made one (docs/01 → Handoffs).
 */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import {
  SCHEMA_VERSION,
  type BundleMeta,
  type CatalogEvent,
  type EventEvidence,
  type SeismicEvent,
  type Station,
} from "@hq/contracts";
import type { FetchLike } from "./fetch";
import { useBundle, useEvidence, useValidation } from "./hooks";
import { liveLabel } from "./live";
import { parseMode } from "./mode";
import { ProviderRoot, preloadEvidence } from "./root";
import { StaticBundleProvider, modeLabel } from "./static";

afterEach(cleanup);

const RUN_ID = "test-run";

function meta(overrides: Partial<BundleMeta> = {}): BundleMeta {
  return {
    schemaVersion: SCHEMA_VERSION,
    mode: "mock",
    scene: {
      runId: RUN_ID,
      originLat: 0,
      originLon: 0,
      originElevM: 0,
      refSurfaceElevM: 0,
      projection: "EPSG:32612 minus origin",
      verticalExaggeration: 1,
      depthLabel: "depth",
      heroEventId: "ev-2",
      isSynthetic: true,
    },
    run: {
      id: RUN_ID,
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
      isSynthetic: true,
    },
    summary: {
      runId: RUN_ID,
      publicCatalogCount: 1,
      recoveredCatalogCount: 1,
      recall: 1,
      unmatchedPublicIds: [],
      candidateCount: 3,
      additionalCount: 2,
      additional: { A: 1, B: 1, C: 0 },
      strictQualityCount: 2,
      strictAdditionalCount: 1,
      medianStations: 5,
      medianRmsS: 0.1,
      baseline: null,
    },
    ...overrides,
  };
}

function event(id: string, revealOrder: number): SeismicEvent {
  return {
    id,
    runId: RUN_ID,
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

const evidence: EventEvidence = { eventId: "ev-2", filterHz: [2, 20], traces: [] };

type Files = Record<string, unknown>;

function bundleFiles(runId: string, metaOverrides: Partial<BundleMeta> = {}): Files {
  const m = meta(metaOverrides);
  m.run = { ...m.run, id: runId };
  return {
    "meta.json": m,
    "stations.json": [station],
    "catalog.json": [catalogEvent],
    "events.json": [event("ev-1", 2), event("ev-2", 0), event("ev-3", 1)],
    "features.json": [],
    "validation.json": undefined, // 404
    "evidence/ev-2.json": evidence,
  };
}

/** A fake fetch serving `/data/<mode>/<file>` from in-memory bundles and counting requests. */
function fakeFetch(bundles: Record<string, Files>): FetchLike & { calls: string[] } {
  const calls: string[] = [];
  const impl = async (url: string) => {
    calls.push(url);
    const match = /^\/data\/([^/]+)\/(.+)$/.exec(url);
    const files = match ? bundles[match[1]] : undefined;
    const body = files ? files[match![2]] : undefined;
    if (body === undefined) return { ok: false, status: 404, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => body };
  };
  return Object.assign(impl, { calls });
}

/** One component, no mode-specific code: the acceptance test's "no component changes". */
function Counter() {
  const bundle = useBundle();
  const validation = useValidation();
  if (bundle.status === "loading") return <p>loading</p>;
  if (bundle.status === "error") return <p role="alert">{bundle.message}</p>;
  return (
    <div>
      <p data-testid="label">{bundle.info.label}</p>
      <p data-testid="run">{bundle.info.runId}</p>
      <p data-testid="public">{bundle.meta.summary.publicCatalogCount}</p>
      <p data-testid="events">{bundle.events.length}</p>
      <p data-testid="synthetic">{String(bundle.info.isSynthetic)}</p>
      <p data-testid="validation">{validation === null ? "none" : "loaded"}</p>
    </div>
  );
}

function Evidence({ id }: { id: string | null }) {
  const state = useEvidence(id);
  return <p data-testid="evidence">{state.status === "ready" ? state.evidence?.eventId : state.status}</p>;
}

describe("StaticBundleProvider through the hooks", () => {
  it("serves mock and showcase (a copy of mock) to the same component", async () => {
    const fetchImpl = fakeFetch({ mock: bundleFiles("mock-run"), showcase: bundleFiles("show-run") });

    const mock = render(
      <ProviderRoot provider={new StaticBundleProvider("mock", { fetchImpl })}>
        <Counter />
      </ProviderRoot>,
    );
    await waitFor(() => expect(screen.getByTestId("run").textContent).toBe("mock-run"));
    expect(screen.getByTestId("label").textContent).toContain("Synthetic");
    expect(screen.getByTestId("events").textContent).toBe("3");
    expect(screen.getByTestId("synthetic").textContent).toBe("true");
    expect(screen.getByTestId("validation").textContent).toBe("none");
    mock.unmount();

    render(
      <ProviderRoot provider={new StaticBundleProvider("showcase", { fetchImpl })}>
        <Counter />
      </ProviderRoot>,
    );
    await waitFor(() => expect(screen.getByTestId("run").textContent).toBe("show-run"));
    expect(screen.getByTestId("label").textContent).toBe("Showcase · window · run show-run");
    expect(fetchImpl.calls.some((u) => u.startsWith("/data/showcase/"))).toBe(true);
  });

  it("renders a visible error on a schemaVersion mismatch", async () => {
    const fetchImpl = fakeFetch({
      showcase: bundleFiles("old", { schemaVersion: "0.9" }),
    });
    render(
      <ProviderRoot provider={new StaticBundleProvider("showcase", { fetchImpl })}>
        <Counter />
      </ProviderRoot>,
    );
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("0.9");
    expect(alert.textContent).toContain(SCHEMA_VERSION);
  });

  it("renders a visible error when a bundle file is missing", async () => {
    const files = bundleFiles("x");
    delete files["events.json"];
    const fetchImpl = fakeFetch({ showcase: files });
    render(
      <ProviderRoot provider={new StaticBundleProvider("showcase", { fetchImpl })}>
        <Counter />
      </ProviderRoot>,
    );
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("events.json");
  });

  it("memoizes fetches per file", async () => {
    const fetchImpl = fakeFetch({ mock: bundleFiles("m") });
    const provider = new StaticBundleProvider("mock", { fetchImpl });
    await Promise.all([provider.getEvents(), provider.getEvents(), provider.getMeta(), provider.info()]);
    expect(fetchImpl.calls.filter((u) => u.endsWith("events.json"))).toHaveLength(1);
    expect(fetchImpl.calls.filter((u) => u.endsWith("meta.json"))).toHaveLength(1);
  });

  it("loads evidence through useEvidence and preloads hero + first by revealOrder", async () => {
    const fetchImpl = fakeFetch({ mock: bundleFiles("m") });
    const provider = new StaticBundleProvider("mock", { fetchImpl });
    render(
      <ProviderRoot provider={provider}>
        <Evidence id="ev-2" />
      </ProviderRoot>,
    );
    await waitFor(() => expect(screen.getByTestId("evidence").textContent).toBe("ev-2"));
    const chosen = preloadEvidence(provider, "ev-2", [event("ev-1", 2), event("ev-2", 0), event("ev-3", 1)]);
    expect(chosen).toEqual(["ev-2", "ev-3", "ev-1"]);
    await waitFor(() =>
      expect(fetchImpl.calls.filter((u) => u.includes("/evidence/ev-2.json"))).toHaveLength(1),
    );
  });
});

describe("mode and labels", () => {
  it("parses ?mode= with a showcase default", () => {
    expect(parseMode("")).toBe("showcase");
    expect(parseMode("?mode=mock")).toBe("mock");
    expect(parseMode("?mode=live&x=1")).toBe("live");
    expect(parseMode("?mode=bogus")).toBe("showcase");
  });

  it("builds labels only from bundle fields", () => {
    const m = meta();
    expect(modeLabel("showcase", m)).toBe(`Showcase · ${m.run.windowLabel} · run ${m.run.id}`);
    expect(modeLabel("snapshot", m)).toContain("2026-09-26 01:00 UTC");
    expect(modeLabel("mock", m)).toContain(m.run.id);
    expect(
      liveLabel({ updatedAt: 1000, windowS: 7200, latencyS: 10, stationsOnline: 3 }, 1000 + 180),
    ).toBe("Live · last 2 h · updated 3 min ago");
  });
});

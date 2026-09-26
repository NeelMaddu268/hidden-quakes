/**
 * API-01 acceptance: mock ↔ showcase swap with no component changes; a schemaVersion mismatch
 * renders a visible error. The bundle here is a tiny hand-made one (docs/01 → Handoffs).
 */
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  SCHEMA_VERSION,
  type BundleMeta,
  type CatalogEvent,
  type EventEvidence,
  type SeismicEvent,
  type Station,
} from "@hq/contracts";
import type { FetchLike } from "./fetch";
import { useBundle, useEvidence, useLiveStatus, useValidation } from "./hooks";
import { LiveProvider, formatWindow, liveLabel } from "./live";
import { parseMode } from "./mode";
import { ProviderRoot, preloadEvidence } from "./root";
import { LIVE_POLL_MS } from "./config";
import { StaticBundleProvider, modeLabel } from "./static";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.useRealTimers();
});

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
function fakeFetch(
  bundles: Record<string, Files>,
  extra: Record<string, () => unknown> = {},
  delayMs = 0,
): FetchLike & { calls: string[] } {
  const calls: string[] = [];
  const impl = async (url: string) => {
    calls.push(url);
    if (delayMs) await new Promise((resolve) => setTimeout(resolve, delayMs));
    if (url in extra) return { ok: true, status: 200, json: async () => extra[url]() };
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

describe("ProviderRoot without injected provider", () => {
  it("reads ?mode= from the URL and fetches that bundle", async () => {
    const fetchImpl = fakeFetch({ mock: bundleFiles("from-url") });
    vi.stubGlobal("fetch", fetchImpl);
    window.history.replaceState({}, "", "?mode=mock");
    render(
      <ProviderRoot>
        <Counter />
      </ProviderRoot>,
    );
    await waitFor(() => expect(screen.getByTestId("run").textContent).toBe("from-url"));
    expect(fetchImpl.calls[0]).toMatch(/^\/data\/mock\//);
  });

  it("refuses mock mode in a production build with a visible error", async () => {
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("NEXT_PUBLIC_ALLOW_MOCK", "");
    render(
      <ProviderRoot mode="mock">
        <Counter />
      </ProviderRoot>,
    );
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("disabled");
  });

  it("never shows the old provider's data after a switch", async () => {
    const a = new StaticBundleProvider("mock", { fetchImpl: fakeFetch({ mock: bundleFiles("run-a") }) });
    const b = new StaticBundleProvider("showcase", {
      fetchImpl: fakeFetch({ showcase: bundleFiles("run-b") }, {}, 20),
    });
    const view = render(
      <ProviderRoot provider={a}>
        <Counter />
      </ProviderRoot>,
    );
    await waitFor(() => expect(screen.getByTestId("run").textContent).toBe("run-a"));
    view.rerender(
      <ProviderRoot provider={b}>
        <Counter />
      </ProviderRoot>,
    );
    expect(screen.getByText("loading")).toBeTruthy();
    await waitFor(() => expect(screen.getByTestId("run").textContent).toBe("run-b"));
  });

  it("a failing validation.json never demotes a ready bundle", async () => {
    const files = bundleFiles("v");
    const fetchImpl = fakeFetch({ showcase: files }, { "/data/showcase/validation.json": () => { throw new Error("boom"); } });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    render(
      <ProviderRoot provider={new StaticBundleProvider("showcase", { fetchImpl })}>
        <Counter />
      </ProviderRoot>,
    );
    await waitFor(() => expect(screen.getByTestId("run").textContent).toBe("v"));
    await waitFor(() => expect(warn).toHaveBeenCalled());
    expect(screen.getByTestId("validation").textContent).toBe("none");
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

function LiveView() {
  const status = useLiveStatus();
  const bundle = useBundle();
  return (
    <div>
      <p data-testid="online">{status === null ? "null" : String(status.stationsOnline)}</p>
      <p data-testid="label">{bundle.status === "ready" ? bundle.info.label : bundle.status}</p>
    </div>
  );
}

describe("LiveProvider through the hooks", () => {
  it("useLiveStatus is null in showcase mode", async () => {
    const fetchImpl = fakeFetch({ showcase: bundleFiles("s") });
    render(
      <ProviderRoot provider={new StaticBundleProvider("showcase", { fetchImpl })}>
        <LiveView />
      </ProviderRoot>,
    );
    await waitFor(() => expect(screen.getByTestId("label").textContent).toContain("Showcase"));
    expect(screen.getByTestId("online").textContent).toBe("null");
  });

  it("polls status, refreshes the label, and single-flights concurrent route calls", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let updatedAt = 1000;
    let nowS = 1000 + 120;
    const live = { ...bundleFiles("live-run") };
    const fetchImpl = fakeFetch(
      { snapshot: bundleFiles("snap") },
      {
        "/api/live/meta": () => live["meta.json"],
        "/api/live/events": () => live["events.json"],
        "/api/live/status": () => ({ updatedAt, windowS: 7200, latencyS: 30, stationsOnline: 9 }),
      },
    );
    const provider = new LiveProvider("/api/live", {
      fetchImpl,
      fallback: new StaticBundleProvider("snapshot", { fetchImpl }),
      now: () => nowS,
    });
    render(
      <ProviderRoot provider={provider}>
        <LiveView />
      </ProviderRoot>,
    );
    await waitFor(() => expect(screen.getByTestId("online").textContent).toBe("9"));
    expect(screen.getByTestId("label").textContent).toBe("Live · last 2 h · updated 2 min ago");
    // info() + getMeta() and info() + poll() share requests within a tick.
    expect(fetchImpl.calls.filter((u) => u === "/api/live/meta")).toHaveLength(1);
    const statusCalls = fetchImpl.calls.filter((u) => u === "/api/live/status").length;
    expect(statusCalls).toBe(1);

    updatedAt = 4000;
    nowS = 4000 + 60 * 5;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_POLL_MS + 10);
    });
    await waitFor(() =>
      expect(screen.getByTestId("label").textContent).toBe("Live · last 2 h · updated 5 min ago"),
    );
    expect(fetchImpl.calls.filter((u) => u === "/api/live/status").length).toBe(statusCalls + 1);
  });

  it("formats the window from data", () => {
    expect(formatWindow(7200)).toBe("2 h");
    expect(formatWindow(1800)).toBe("30 min");
    expect(formatWindow(5400)).toBe("1.5 h");
  });
});

describe("hooks outside ProviderRoot", () => {
  it("throw instead of loading forever", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    expect(() => render(<Counter />)).toThrow(/ProviderRoot/);
    spy.mockRestore();
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

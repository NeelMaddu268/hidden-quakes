/**
 * API-01 acceptance: mock ↔ showcase swap with no component changes; a schemaVersion mismatch
 * renders a visible error. API-05: live failover to the snapshot bundle and back. The bundle
 * here is a tiny hand-made one (docs/01 → Handoffs).
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
import { useBundle, useEvidence, useFailedOver, useLiveStatus, useValidation } from "./hooks";
import { LiveProvider, fetchWithTimeout, formatWindow, liveLabel } from "./live";
import { parseMode } from "./mode";
import { ProviderRoot, isFailoverError, preloadEvidence } from "./root";
import { DEFAULT_LIVE_API_BASE, LIVE_FETCH_TIMEOUT_MS, LIVE_HEARTBEAT_MS, LIVE_POLL_MS, liveApiBase } from "./config";
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

/** Thrown by an `extra` route to answer with that HTTP status instead of a body. */
class HttpStatus extends Error {
  constructor(readonly status: number) {
    super(`HTTP ${status}`);
  }
}
const unavailable = () => {
  throw new HttpStatus(503);
};
const refused = () => {
  throw new TypeError("Failed to fetch");
};
const hanging = () => new Promise<never>(() => undefined);

/** A fake fetch serving `/data/<mode>/<file>` from in-memory bundles and counting requests.
 *  An `extra` route returns a body, throws `HttpStatus` for an error status, throws anything
 *  else for a network failure, or returns a promise that never settles for a hanging worker. */
function fakeFetch(
  bundles: Record<string, Files>,
  extra: Record<string, () => unknown> = {},
  delayMs = 0,
): FetchLike & { calls: string[] } {
  const calls: string[] = [];
  const impl = async (url: string) => {
    calls.push(url);
    if (delayMs) await new Promise((resolve) => setTimeout(resolve, delayMs));
    if (url in extra) {
      let body: unknown;
      try {
        body = await extra[url]();
      } catch (error) {
        if (error instanceof HttpStatus) return { ok: false, status: error.status, json: async () => ({}) };
        throw error;
      }
      return { ok: true, status: 200, json: async () => body };
    }
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
    // Exactly one more poll (events are only polled), plus at most the heartbeats that fit in a
    // poll interval (a heartbeat landing on the poll's tick may be single-flighted with it).
    expect(fetchImpl.calls.filter((u) => u === "/api/live/events")).toHaveLength(2);
    const status = fetchImpl.calls.filter((u) => u === "/api/live/status").length;
    expect(status).toBeGreaterThanOrEqual(statusCalls + 1);
    expect(status).toBeLessThanOrEqual(statusCalls + 1 + Math.floor(LIVE_POLL_MS / LIVE_HEARTBEAT_MS));
  });

  it("formats the window from data", () => {
    expect(formatWindow(7200)).toBe("2 h");
    expect(formatWindow(1800)).toBe("30 min");
    expect(formatWindow(5400)).toBe("1.5 h");
  });
});

function FailoverView() {
  const bundle = useBundle();
  const failedOver = useFailedOver();
  const status = useLiveStatus();
  if (bundle.status === "error") return <p role="alert">{bundle.message}</p>;
  return (
    <div>
      <p data-testid="label">{bundle.status === "ready" ? bundle.info.label : bundle.status}</p>
      <p data-testid="mode">{bundle.status === "ready" ? bundle.info.mode : bundle.status}</p>
      <p data-testid="failed-over">{String(failedOver)}</p>
      <p data-testid="online">{status === null ? "null" : String(status.stationsOnline)}</p>
    </div>
  );
}

const SNAPSHOT_LABEL = "Snapshot · generated 2026-09-26 01:00 UTC by our pipeline · run snap";

/** A live worker whose health the test flips: `up` serves the window, `down` answers like a
 *  worker that was killed (refused), a 503 or a hang, per `failure`. */
function liveWorker(failure: () => unknown, options: { snapshot?: boolean } = {}) {
  const state = { up: true };
  const window = bundleFiles("live-run");
  const route = (body: () => unknown) => () => (state.up ? body() : failure());
  const fetchImpl = fakeFetch(options.snapshot === false ? {} : { snapshot: bundleFiles("snap") }, {
    "/api/live/meta": route(() => window["meta.json"]),
    "/api/live/events": route(() => window["events.json"]),
    "/api/live/status": route(() => ({ updatedAt: 1000, windowS: 7200, latencyS: 30, stationsOnline: 9 })),
  });
  const provider = new LiveProvider("/api/live", {
    fetchImpl,
    fallback: new StaticBundleProvider("snapshot", { fetchImpl }),
    now: () => 1000 + 120,
  });
  return { state, fetchImpl, provider };
}

function mountLive(provider: LiveProvider) {
  return render(
    <ProviderRoot mode="live" provider={provider}>
      <FailoverView />
    </ProviderRoot>,
  );
}

describe("live failover to the snapshot bundle (API-05)", () => {
  it("shows the live label while the worker is healthy and never fails over", async () => {
    const { provider } = liveWorker(unavailable);
    mountLive(provider);
    await waitFor(() => expect(screen.getByTestId("label").textContent).toBe("Live · last 2 h · updated 2 min ago"));
    expect(screen.getByTestId("mode").textContent).toBe("live");
    expect(screen.getByTestId("failed-over").textContent).toBe("false");
    expect(screen.getByTestId("online").textContent).toBe("9");
  });

  it("a 503 (no window yet) shows the snapshot label from the snapshot bundle's meta", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { state, provider, fetchImpl } = liveWorker(unavailable);
    state.up = false;
    mountLive(provider);
    await waitFor(() => expect(screen.getByTestId("label").textContent).toBe(SNAPSHOT_LABEL));
    expect(screen.getByTestId("mode").textContent).toBe("snapshot");
    expect(screen.getByTestId("failed-over").textContent).toBe("true");
    expect(screen.getByTestId("online").textContent).toBe("null");
    expect(fetchImpl.calls.some((u) => u === "/data/snapshot/events.json")).toBe(true);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(warn).toHaveBeenCalled();
  });

  it("warms the snapshot while live is healthy so a failover needs no request", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { state, provider, fetchImpl } = liveWorker(refused);
    mountLive(provider);
    await waitFor(() => expect(screen.getByTestId("label").textContent).toMatch(/^Live/));
    await waitFor(() => expect(fetchImpl.calls.filter((u) => u === "/data/snapshot/events.json")).toHaveLength(1));
    const before = fetchImpl.calls.filter((u) => u.startsWith("/data/snapshot/")).length;
    state.up = false;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_HEARTBEAT_MS + 10);
    });
    await waitFor(() => expect(screen.getByTestId("label").textContent).toBe(SNAPSHOT_LABEL));
    // Only the evidence preload of the snapshot is new; the bundle files came from the memo.
    const after = fetchImpl.calls.filter((u) => u.startsWith("/data/snapshot/") && !u.includes("/evidence/")).length;
    expect(after).toBe(before);
  });

  it("keeps the live bundle on screen until the snapshot is ready, never a loading flash", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { state, provider } = liveWorker(refused);
    const seen: string[] = [];
    function Recorder() {
      const bundle = useBundle();
      seen.push(bundle.status === "ready" ? bundle.info.mode : bundle.status);
      return null;
    }
    render(
      <ProviderRoot mode="live" provider={provider}>
        <FailoverView />
        <Recorder />
      </ProviderRoot>,
    );
    await waitFor(() => expect(screen.getByTestId("label").textContent).toMatch(/^Live/));
    state.up = false;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_HEARTBEAT_MS + 10);
    });
    await waitFor(() => expect(screen.getByTestId("mode").textContent).toBe("snapshot"));
    const afterLive = seen.slice(seen.indexOf("live"));
    expect(afterLive).not.toContain("loading");
    expect(afterLive).toContain("snapshot");
  });

  it("a refused connection fails over the same way", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { state, provider } = liveWorker(refused);
    state.up = false;
    mountLive(provider);
    await waitFor(() => expect(screen.getByTestId("label").textContent).toBe(SNAPSHOT_LABEL));
  });

  it("a hanging worker fails over after the request timeout", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { state, provider } = liveWorker(hanging);
    state.up = false;
    mountLive(provider);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_FETCH_TIMEOUT_MS / 2);
    });
    expect(screen.getByTestId("label").textContent).toBe("loading");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_FETCH_TIMEOUT_MS / 2 + 20); // past the deadline
    });
    await waitFor(() => expect(screen.getByTestId("label").textContent).toBe(SNAPSHOT_LABEL));
  });

  it("killing the worker after a healthy start flips the label within one heartbeat", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { state, provider } = liveWorker(refused);
    mountLive(provider);
    await waitFor(() => expect(screen.getByTestId("label").textContent).toMatch(/^Live/));
    state.up = false;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_HEARTBEAT_MS + 10);
    });
    await waitFor(() => expect(screen.getByTestId("label").textContent).toBe(SNAPSHOT_LABEL));
    expect(screen.getByTestId("failed-over").textContent).toBe("true");
  });

  it("switches back to live when a later heartbeat succeeds", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { state, provider } = liveWorker(unavailable);
    state.up = false;
    mountLive(provider);
    await waitFor(() => expect(screen.getByTestId("label").textContent).toBe(SNAPSHOT_LABEL));
    state.up = true;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_HEARTBEAT_MS + 10);
    });
    await waitFor(() => expect(screen.getByTestId("label").textContent).toBe("Live · last 2 h · updated 2 min ago"));
    expect(screen.getByTestId("failed-over").textContent).toBe("false");
    expect(screen.getByTestId("mode").textContent).toBe("live");
  });

  it("re-renders the bundle only when the live label actually changes", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const { provider } = liveWorker(unavailable);
    const identities: unknown[] = [];
    function Identity() {
      const bundle = useBundle();
      if (bundle.status === "ready" && identities[identities.length - 1] !== bundle) identities.push(bundle);
      return null;
    }
    render(
      <ProviderRoot mode="live" provider={provider}>
        <FailoverView />
        <Identity />
      </ProviderRoot>,
    );
    await waitFor(() => expect(screen.getByTestId("label").textContent).toMatch(/^Live/));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_HEARTBEAT_MS * 3 + 10);
    });
    // The same status every heartbeat: the ready bundle keeps its identity.
    expect(identities).toHaveLength(1);
  });

  it("a missing snapshot bundle is the visible error, not a silent fallback", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const { state, provider } = liveWorker(unavailable, { snapshot: false });
    state.up = false;
    mountLive(provider);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/\/data\/snapshot\/\w+\.json: HTTP 404/);
  });

  it("a missing snapshot file while the worker is healthy is the visible error, not a retry loop", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const { provider, fetchImpl } = liveWorker(unavailable, { snapshot: false });
    mountLive(provider);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/\/data\/snapshot\/\w+\.json: HTTP 404/);
    const liveCalls = fetchImpl.calls.filter((u) => u === "/api/live/meta").length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_HEARTBEAT_MS * 3 + 10);
    });
    expect(screen.getByRole("alert").textContent).toMatch(/\/data\/snapshot\/\w+\.json: HTTP 404/);
    expect(fetchImpl.calls.filter((u) => u === "/api/live/meta")).toHaveLength(liveCalls); // no reload loop
  });

  it("a live schemaVersion mismatch is a visible error, never a failover", async () => {
    const stale = bundleFiles("old", { schemaVersion: "0.9" });
    const fetchImpl = fakeFetch(
      { snapshot: bundleFiles("snap") },
      {
        "/api/live/meta": () => stale["meta.json"],
        "/api/live/events": () => stale["events.json"],
        "/api/live/status": () => ({ updatedAt: 1000, windowS: 7200, latencyS: 30, stationsOnline: 9 }),
      },
    );
    const provider = new LiveProvider("/api/live", {
      fetchImpl,
      fallback: new StaticBundleProvider("snapshot", { fetchImpl }),
    });
    mountLive(provider);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("0.9");
  });

  it("fetchWithTimeout aborts and rejects with a fetch error at the deadline", async () => {
    vi.useFakeTimers();
    let signal: AbortSignal | undefined;
    const impl: FetchLike = (_url, init) => {
      signal = init?.signal;
      return new Promise(() => undefined);
    };
    const pending = fetchWithTimeout(impl, "/api/live/status", 50);
    const outcome = pending.then(
      () => "resolved",
      (error: unknown) => error,
    );
    await vi.advanceTimersByTimeAsync(60);
    const error = await outcome;
    expect(isFailoverError(error, new LiveProvider("/api/live", { fetchImpl: impl }))).toBe(true);
    expect(isFailoverError(error, new LiveProvider("http://other/api/live", { fetchImpl: impl }))).toBe(false);
    expect(String(error)).toContain("50 ms");
    expect(signal?.aborted).toBe(true);
  });

  it("reads the live API base from NEXT_PUBLIC_LIVE_API_BASE with a same-origin default", () => {
    expect(liveApiBase(undefined)).toBe(DEFAULT_LIVE_API_BASE);
    expect(liveApiBase("")).toBe("/api/live");
    expect(liveApiBase("http://127.0.0.1:8765/api/live/")).toBe("http://127.0.0.1:8765/api/live");
    expect(new LiveProvider().apiBase).toBe(DEFAULT_LIVE_API_BASE);
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
    expect(modeLabel("snapshot", m)).toBe("Snapshot · generated 2026-09-26 01:00 UTC by our pipeline · run test-run");
    expect(modeLabel("mock", m)).toContain(m.run.id);
    expect(
      liveLabel({ updatedAt: 1000, windowS: 7200, latencyS: 10, stationsOnline: 3 }, 1000 + 180),
    ).toBe("Live · last 2 h · updated 3 min ago");
  });
});

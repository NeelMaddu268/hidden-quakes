// @vitest-environment jsdom
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EventEvidence } from "./types";

// Each test gets a fresh module, so the evidence and bundle caches start empty.
async function freshShim() {
  vi.resetModules();
  return await import("./bundle-shim");
}

// Test-local synthetic evidence (rule 5 allows small synthetic data inside tests).
function evidence(eventId: string): EventEvidence {
  return {
    eventId,
    filterHz: [2, 20],
    traces: [{ stationId: "XX.A", channel: "HHZ", epiDistM: 1000, t0: 10, dt: 0.01, samples: [0, 1, -1] }],
  };
}

type Routes = Record<string, unknown | number>;

/** A fetch stub serving JSON by path; a number is an HTTP error status. */
function stubFetch(routes: Routes) {
  const fn = vi.fn(async (url: string) => {
    const body = routes[url];
    if (body === undefined) return new Response("not found", { status: 404 });
    if (typeof body === "number") return new Response("err", { status: body });
    return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

beforeEach(() => {
  window.history.replaceState({}, "", "/?mode=mock");
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useEvidence (docs/02 §6 shape)", () => {
  it("is idle for no selection and fetches nothing", async () => {
    const fetch = stubFetch({});
    const { useEvidence } = await freshShim();
    const { result } = renderHook(() => useEvidence(null));
    expect(result.current).toEqual({ status: "idle" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("loads /data/<mode>/evidence/<id>.json: loading, then ready", async () => {
    const fetch = stubFetch({ "/data/mock/evidence/hq-1.json": evidence("hq-1") });
    const { useEvidence } = await freshShim();
    const { result } = renderHook(() => useEvidence("hq-1"));
    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.evidence?.eventId).toBe("hq-1");
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith("/data/mock/evidence/hq-1.json");
  });

  it("defaults to the showcase mode without ?mode= and URL-encodes the id", async () => {
    window.history.replaceState({}, "", "/");
    const fetch = stubFetch({ "/data/showcase/evidence/a%2Fb.json": evidence("a/b") });
    const { useEvidence } = await freshShim();
    const { result } = renderHook(() => useEvidence("a/b"));
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(fetch).toHaveBeenCalledWith("/data/showcase/evidence/a%2Fb.json");
  });

  it("is memoized: a cached event is ready on the first render, with no second fetch", async () => {
    const fetch = stubFetch({ "/data/mock/evidence/hq-1.json": evidence("hq-1") });
    const { useEvidence } = await freshShim();
    const first = renderHook(() => useEvidence("hq-1"));
    await waitFor(() => expect(first.result.current.status).toBe("ready"));
    const seen: string[] = [];
    const second = renderHook(() => {
      const s = useEvidence("hq-1");
      seen.push(s.status);
      return s;
    });
    expect(seen[0]).toBe("ready");
    expect(second.result.current).toBe(first.result.current); // same frozen object
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("switches between events and back to idle", async () => {
    stubFetch({
      "/data/mock/evidence/hq-1.json": evidence("hq-1"),
      "/data/mock/evidence/hq-2.json": evidence("hq-2"),
    });
    const { useEvidence } = await freshShim();
    const { result, rerender } = renderHook(({ id }: { id: string | null }) => useEvidence(id), {
      initialProps: { id: "hq-1" as string | null },
    });
    await waitFor(() => expect(result.current.evidence?.eventId).toBe("hq-1"));
    rerender({ id: "hq-2" });
    expect(result.current.status).toBe("loading"); // never shows hq-1's traces for hq-2
    await waitFor(() => expect(result.current.evidence?.eventId).toBe("hq-2"));
    rerender({ id: null });
    expect(result.current).toEqual({ status: "idle" });
  });

  it("reports HTTP errors with a message and retries when the event is selected again", async () => {
    const fetch = stubFetch({ "/data/mock/evidence/hq-9.json": 500 });
    const { useEvidence } = await freshShim();
    const { result, rerender } = renderHook(({ id }: { id: string | null }) => useEvidence(id), {
      initialProps: { id: "hq-9" as string | null },
    });
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.message).toContain("HTTP 500");
    expect(result.current.evidence).toBeUndefined();
    rerender({ id: null });
    rerender({ id: "hq-9" });
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  });

  it("never builds a fetch path from an invalid ?mode=", async () => {
    window.history.replaceState({}, "", "/?mode=..%2Fsecret");
    const fetch = stubFetch({});
    const { useEvidence } = await freshShim();
    const { result } = renderHook(() => useEvidence("hq-1"));
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.message).toContain("unknown mode");
    expect(fetch).not.toHaveBeenCalled();
  });

  it("rejects a file that belongs to another event", async () => {
    stubFetch({ "/data/mock/evidence/hq-1.json": evidence("hq-2") });
    const { useEvidence } = await freshShim();
    const { result } = renderHook(() => useEvidence("hq-1"));
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.message).toContain("hq-2");
  });
});

describe("checkEvidence", () => {
  it("accepts a well-formed file and rejects broken ones", async () => {
    const { checkEvidence } = await freshShim();
    expect(checkEvidence(evidence("x"), "x").eventId).toBe("x");
    expect(() => checkEvidence(null, "x")).toThrow(/not a JSON object/);
    expect(() => checkEvidence({ ...evidence("x"), traces: null }, "x")).toThrow(/traces/);
    expect(() => checkEvidence({ ...evidence("x"), filterHz: [1] }, "x")).toThrow(/filterHz/);
  });
});

describe("hero preload", () => {
  it("useBundle warms the hero's evidence, so selecting it renders ready immediately", async () => {
    const meta = {
      mode: "mock",
      scene: { runId: "r", heroEventId: "hq-hero", isSynthetic: true },
      run: { id: "r", windowStart: 0, windowEnd: 1, windowLabel: "w" },
      summary: {},
    };
    const fetch = stubFetch({
      "/data/mock/meta.json": meta,
      "/data/mock/stations.json": [],
      "/data/mock/catalog.json": [],
      "/data/mock/events.json": [],
      "/data/mock/features.json": [],
      "/data/mock/evidence/hq-hero.json": evidence("hq-hero"),
    });
    const { useBundle, useEvidence } = await freshShim();
    const bundle = renderHook(() => useBundle());
    await waitFor(() => expect(bundle.result.current.status).toBe("ready"));
    // The bundle loader requested the hero's evidence before any component asked for it.
    await waitFor(() => expect(fetch).toHaveBeenCalledWith("/data/mock/evidence/hq-hero.json"));
    const probe = renderHook(() => useEvidence("hq-hero"));
    await waitFor(() => expect(probe.result.current.status).toBe("ready"));
    expect(fetch).toHaveBeenCalledTimes(6); // five bundle files + one evidence file, no refetch
    // So the first render after "E" (select(heroEventId)) is already ready.
    const seen: string[] = [];
    renderHook(() => {
      const s = useEvidence("hq-hero");
      seen.push(s.status);
      return s;
    });
    expect(seen[0]).toBe("ready");
  });
});

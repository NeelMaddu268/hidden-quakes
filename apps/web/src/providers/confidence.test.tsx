/**
 * ML-01: `confidence.json` is optional and untrusted. `parseConfidence` keeps only what the file
 * can back; `useConfidence` fetches it once per bundle, and a missing, broken or foreign file is
 * `null`, never an error on screen. Every number here is invented inside the test (rule 5).
 */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { HERO_EVENT_ID, bundleFiles, fakeFetch, type Files } from "../shell/test-fixture";
import { CONFIDENCE_SCHEMA, parseConfidence, useConfidence } from "./confidence";
import type { FetchLike } from "./fetch";
import { useBundle } from "./hooks";
import { ProviderRoot } from "./root";
import { StaticBundleProvider } from "./static";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const RUN = "conf-run";

function doc(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema: CONFIDENCE_SCHEMA,
    runId: RUN,
    model: { type: "test", heldOut: { rocAuc: 0.8125, folds: 5 } },
    label: "Decoy test",
    description: "How much this event looks like real timing rather than a decoy",
    events: { [HERO_EVENT_ID]: 0.875, "ev-1": 0.25 },
    ...over,
  };
}

describe("parseConfidence", () => {
  it("reads label, description, held-out AUC and per-event scores", () => {
    const c = parseConfidence(doc(), RUN)!;
    expect(c.label).toBe("Decoy test");
    expect(c.description).toContain("real timing");
    expect(c.rocAuc).toBe(0.8125);
    expect(c.score(HERO_EVENT_ID)).toBe(0.875);
    expect(c.score("ev-1")).toBe(0.25);
    expect(c.score("not-scored")).toBeNull();
  });

  it("is null for a wrong schema, another run's file, a non-object or nothing usable", () => {
    expect(parseConfidence(doc({ schema: "hq.confidence/2" }), RUN)).toBeNull();
    expect(parseConfidence(doc({ runId: "other" }), RUN)).toBeNull();
    expect(parseConfidence([doc()], RUN)).toBeNull();
    expect(parseConfidence("x", RUN)).toBeNull();
    expect(parseConfidence(null, RUN)).toBeNull();
    expect(parseConfidence(doc({ events: {}, model: {} }), RUN)).toBeNull();
  });

  it("degrades field by field: missing runId, label, description, AUC; bad scores dropped", () => {
    const c = parseConfidence(
      doc({
        runId: undefined,
        label: "  ",
        description: 7,
        model: { heldOut: { rocAuc: 1.5 } },
        events: { a: 1.2, b: -0.1, c: "0.5", d: null, e: Number.NaN, f: 0, g: 1 },
      }),
      RUN,
    )!;
    expect(c.label).toBeNull();
    expect(c.description).toBeNull();
    expect(c.rocAuc).toBeNull();
    expect(["a", "b", "c", "d", "e"].map((id) => c.score(id))).toEqual([null, null, null, null, null]);
    expect(c.score("f")).toBe(0);
    expect(c.score("g")).toBe(1);
    // Only the AUC, no events: still worth a row on the Validation card.
    expect(parseConfidence(doc({ events: undefined }), RUN)!.rocAuc).toBe(0.8125);
    expect(parseConfidence(doc({ model: undefined }), RUN)!.rocAuc).toBeNull();
  });
});

function Probe({ id }: { id: string }) {
  const bundle = useBundle();
  const c = useConfidence();
  const score = c?.score(id);
  return (
    <p data-testid={`probe-${id}`}>
      {bundle.status}:{c === null ? "none" : `${c.label}=${score ?? "unscored"}`}
    </p>
  );
}

function mount(files: Files, fetchImpl: FetchLike & { calls?: string[] } = fakeFetch({ mock: files })) {
  const provider = new StaticBundleProvider("mock", { fetchImpl });
  render(
    <ProviderRoot mode="mock" provider={provider}>
      <Probe id={HERO_EVENT_ID} />
      <Probe id="ev-1" />
    </ProviderRoot>,
  );
  return fetchImpl;
}

describe("useConfidence", () => {
  it("serves every caller from one request once the bundle is ready", async () => {
    const fetchImpl = mount({ ...bundleFiles(RUN), "confidence.json": doc() });
    await waitFor(() => expect(screen.getByTestId(`probe-${HERO_EVENT_ID}`).textContent).toBe("ready:Decoy test=0.875"));
    expect(screen.getByTestId("probe-ev-1").textContent).toBe("ready:Decoy test=0.25");
    expect(fetchImpl.calls!.filter((u) => u === "/data/mock/confidence.json")).toHaveLength(1);
  });

  it("is null when the bundle has no confidence.json (404)", async () => {
    mount(bundleFiles(RUN));
    await waitFor(() => expect(screen.getByTestId("probe-ev-1").textContent).toBe("ready:none"));
  });

  it("is null for another run's file", async () => {
    mount({ ...bundleFiles(RUN), "confidence.json": doc({ runId: "stale" }) });
    await waitFor(() => expect(screen.getByTestId("probe-ev-1").textContent).toBe("ready:none"));
  });

  it("is null, with the bundle still ready, when the file fails to load", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const files = fakeFetch({ mock: bundleFiles(RUN) });
    const failing: FetchLike = (url, init) =>
      url.endsWith("confidence.json") ? Promise.resolve({ ok: false, status: 500, json: async () => ({}) }) : files(url, init);
    mount({}, failing);
    await waitFor(() => expect(screen.getByTestId("probe-ev-1").textContent).toBe("ready:none"));
    await waitFor(() => expect(warn).toHaveBeenCalledWith("confidence.json unavailable:", expect.stringContaining("HTTP 500")));
  });

  it("is null outside <ProviderRoot> instead of throwing", () => {
    function Bare() {
      return <p data-testid="bare">{useConfidence() === null ? "none" : "some"}</p>;
    }
    render(<Bare />);
    expect(screen.getByTestId("bare").textContent).toBe("none");
  });
});

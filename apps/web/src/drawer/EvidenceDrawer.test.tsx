// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BundleState, EventEvidence, EvidenceState, SeismicEvent, Station, WaveformSnippet } from "../scene/types";
import { useDemo } from "../state/demo";
import { EvidenceDrawer } from "./index";
import { pickDelayMs } from "./record";

// The drawer reads data only through scene/data.ts; the test swaps that module for controllable state.
const data = vi.hoisted(() => ({
  bundle: { status: "loading" } as BundleState,
  evidence: {} as Record<string, EvidenceState>,
}));
vi.mock("../scene/data", () => ({
  useBundle: () => data.bundle,
  useEvidence: (id: string | null) => (id === null ? { status: "idle" } : (data.evidence[id] ?? { status: "loading" })),
}));

// ---- Test-local synthetic records (rule 5) ----------------------------------------------------

const ORIGIN = Date.UTC(2026, 8, 10, 14, 3, 7, 210) / 1000;

function station(i: number, over: Partial<Station> = {}): Station {
  const a = (i / 16) * 2 * Math.PI;
  return {
    id: `XX.S${String(i).padStart(2, "0")}`,
    network: "XX",
    location: "",
    staticsS: {},
    station: `S${i}`,
    latitude: 0,
    longitude: 0,
    surfaceElevM: 1650,
    sensorDepthM: 0,
    sensorElevM: 1650,
    kind: "surface",
    channels: ["HHZ"],
    sampleRateHz: 100,
    enu: { e: 8000 * Math.cos(a), n: 8000 * Math.sin(a), u: 20 },
    preprocessProfile: "test",
    usedInRun: true,
    ...over,
  };
}

const STATIONS: Station[] = [
  ...Array.from({ length: 15 }, (_, i) => station(i)),
  station(15, { kind: "borehole", sensorDepthM: 1000, sensorElevM: 650, enu: { e: 900, n: -2800, u: -980 } }),
];

function event(id: string, over: Partial<SeismicEvent> = {}): SeismicEvent {
  return {
    id,
    runId: "test",
    source: "test",
    t: ORIGIN,
    latitude: 0,
    longitude: 0,
    elevM: -90,
    depthKm: 99,
    enu: { e: 1270, n: 1414, u: -1717 },
    quality: {
      method: "pyocto",
      statics: true,
      nStations: 11,
      nP: 8,
      nS: 6,
      rmsS: 0.078,
      gapDeg: 90,
      minEpiDistM: 1500,
      hErrM: 369.3,
      vErrM: 287.5,
      depthOnEdge: false,
    },
    tier: "A",
    tierReasons: ["rmsS 0.078 <= 0.090 (p75 of matched)"],
    meanPickProb: 0.7,
    magnitude: null,
    catalogMatch: null,
    revealOrder: 0,
    pickIds: [],
    ...over,
  };
}

function trace(i: number, over: Partial<WaveformSnippet> = {}): WaveformSnippet {
  const dist = 1500 + i * 700;
  return {
    stationId: STATIONS[i].id,
    channel: "HHZ",
    epiDistM: dist,
    t0: ORIGIN - 1,
    dt: 0.01,
    samples: Array.from({ length: 600 }, (_, k) => Math.sin(k / 3) * Math.exp(-(((k - 200) / 80) ** 2))),
    pickP: ORIGIN + dist / 5500,
    pickS: ORIGIN + dist / 3200,
    probP: 0.9,
    probS: 0.7,
    predP: ORIGIN + dist / 5600,
    predS: ORIGIN + dist / 3250,
    ...over,
  };
}

function evidence(eventId: string, n: number, over: (i: number) => Partial<WaveformSnippet> = () => ({})): EventEvidence {
  return { eventId, filterHz: [2, 20], traces: Array.from({ length: n }, (_, i) => trace(i, over(i))) };
}

const HERO = "hq-test-000001";
const EVENTS = [
  event(HERO),
  event("hq-test-000002", {
    tier: "B",
    tierReasons: [],
    magnitude: { value: 0.83, type: "ML_cal", sigma: 0.21 },
    catalogMatch: { catalogId: "uu60500001", dtS: 0.12, distM: 180 },
  }),
  event("hq-test-000003", {
    quality: { ...event("x").quality, hErrM: null, vErrM: null, depthOnEdge: true },
  }),
];

function readyBundle(isSynthetic = false): BundleState {
  return {
    status: "ready",
    info: { mode: "mock", label: "mock", runId: "test", generatedAt: 0, isSynthetic },
    meta: {
      schemaVersion: "1.0",
      mode: "mock",
      scene: {
        runId: "test",
        originLat: 38.5,
        originLon: -112.9,
        originElevM: 1627.7,
        refSurfaceElevM: 1627.7,
        projection: "EPSG:32612 minus origin",
        depthLabel: "Depth below site surface (test ref)",
        heroEventId: HERO,
        verticalExaggeration: 1,
        isSynthetic,
      },
      run: {
        id: "test", windowStart: ORIGIN - 3600, windowEnd: ORIGIN + 3600, windowLabel: "test",
        associator: {}, bbox: [0, 0, 1, 1], createdAt: "test", gitSha: "test", isSynthetic,
        locator: {}, matching: {}, mode: "mock", picker: {}, pickerModel: "test", pickerWeights: "test",
        runtimeS: {}, softwareVersions: {}, stationIds: STATIONS.map(s => s.id), tiering: {}, velocityModel: {},
      },
      summary: {
        runId: "test",
        publicCatalogCount: 1,
        recoveredCatalogCount: 1,
        recall: 1,
        unmatchedPublicIds: [],
        candidateCount: 3,
        additionalCount: 2,
        additional: { A: 1, B: 1, C: 0 },
        strictQualityCount: 1,
        strictAdditionalCount: 1,
        medianStations: 11,
        medianRmsS: 0.078,
        baseline: null,
      },
    },
    stations: STATIONS,
    catalog: [],
    events: EVENTS,
    features: [],
  };
}

// ---- Helpers ----------------------------------------------------------------------------------

const drawer = () => document.querySelector("aside.hqd") as HTMLElement;
const rows = () => Array.from(document.querySelectorAll(".hqd-record .hqd-row"));
const select = (id: string | null) => act(() => useDemo.getState().select(id));

/** No missing value may leak into text or attributes. */
function expectNoBadValues() {
  const html = drawer().outerHTML;
  expect(html).not.toMatch(/NaN|undefined|Infinity|null%/);
}

beforeEach(() => {
  data.bundle = readyBundle();
  data.evidence = {};
  useDemo.getState().reset();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

// ---- Tests ------------------------------------------------------------------------------------

describe("closed", () => {
  it("renders an empty, inert, off-screen panel that doesn't block the canvas", () => {
    render(<EvidenceDrawer />);
    expect(drawer().dataset.open).toBe("false");
    expect(drawer().getAttribute("aria-hidden")).toBe("true");
    expect(drawer().hasAttribute("inert")).toBe(true);
    expect(drawer().querySelector("header")).toBeNull();
    expect(getComputedStyle(drawer()).pointerEvents).toBe("none");
  });
});

describe("opening (select / E → hero)", () => {
  it("E: selecting the hero from meta opens the drawer with its header and traces", () => {
    data.evidence[HERO] = { status: "ready", evidence: evidence(HERO, 16) };
    render(<EvidenceDrawer />);
    const bundle = data.bundle as Extract<BundleState, { status: "ready" }>;
    select(bundle.meta.scene.heroEventId ?? null);

    expect(drawer().dataset.open).toBe("true");
    expect(drawer().hasAttribute("inert")).toBe(false);
    expect(getComputedStyle(drawer()).pointerEvents).toBe("auto");
    expect(screen.getByRole("heading").textContent).toBe("11 stations agreed");
    expect(screen.getByText("Candidate event")).toBeTruthy();
    expect(screen.getByText(HERO)).toBeTruthy();
    expect(screen.getByText("Tier A")).toBeTruthy();
    expect(screen.getByText("0.078 s")).toBeTruthy();
    expect(screen.getByText("±369 m")).toBeTruthy();
    expect(screen.getByText("±288 m")).toBeTruthy();
    expect(screen.getByText("1.72 km")).toBeTruthy();
    expect(screen.getByText("8 P · 6 S")).toBeTruthy();
    expect(screen.getByText("2026-09-10 14:03:07.21 UTC")).toBeTruthy();
    expect(screen.getByText("2026-09-10 08:03:07.21 MDT")).toBeTruthy();
    expect(screen.getByText("rmsS 0.078 <= 0.090 (p75 of matched)")).toBeTruthy();
    expect(rows()).toHaveLength(16);
    expectNoBadValues();
  });

  it("follows the docs/00 language rules in visible text and tooltips", () => {
    data.evidence["hq-test-000002"] = { status: "ready", evidence: evidence("hq-test-000002", 4) };
    render(<EvidenceDrawer />);
    select("hq-test-000002");
    const copy = drawer().cloneNode(true) as HTMLElement;
    copy.querySelectorAll("style").forEach((el) => el.remove());
    const titles = Array.from(copy.querySelectorAll("[title]"), (el) => el.getAttribute("title"));
    const labels = Array.from(copy.querySelectorAll("[aria-label]"), (el) => el.getAttribute("aria-label"));
    const text = [copy.textContent, ...titles, ...labels].join(" ");
    expect(text).toMatch(/Candidate event/);
    expect(text).toMatch(/public regional catalog/);
    expect(text).not.toMatch(/confirmed|official catalog|caused by|predict|earthquakes? (found|discovered)/i);
  });

  it("handles 4 traces and 16 traces", () => {
    data.evidence["hq-test-000002"] = { status: "ready", evidence: evidence("hq-test-000002", 4) };
    data.evidence[HERO] = { status: "ready", evidence: evidence(HERO, 16) };
    render(<EvidenceDrawer />);
    select("hq-test-000002");
    expect(rows()).toHaveLength(4);
    expect(document.querySelector(".hqd-caption")!.textContent).toMatch(/(^|\D)4 (of \d+ agreeing stations|stations) shown, closest first.* · 2–20 Hz bandpass · normalized per trace/);
    expect(document.querySelectorAll(".hqd-trace")).toHaveLength(4);
    select(HERO);
    expect(rows()).toHaveLength(16);
    expect(document.querySelectorAll(".hqd-trace")).toHaveLength(16);
    for (const pl of document.querySelectorAll(".hqd-trace")) {
      expect(pl.getAttribute("points")!.split(" ").length).toBe(600);
    }
    expectNoBadValues();
  });

  it("sorts rows by epicentral distance even if the file isn't", () => {
    const ev = evidence(HERO, 6);
    ev.traces.reverse();
    data.evidence[HERO] = { status: "ready", evidence: ev };
    render(<EvidenceDrawer />);
    select(HERO);
    expect(rows().map((r) => r.getAttribute("data-station"))).toEqual(STATIONS.slice(0, 6).map((s) => s.id));
    expect(rows()[0].textContent).toContain("1.5 km");
  });

  it("P and S ticks animate in on a 150 ms stagger per trace", () => {
    data.evidence[HERO] = { status: "ready", evidence: evidence(HERO, 5) };
    render(<EvidenceDrawer />);
    select(HERO);
    const p = Array.from(document.querySelectorAll<HTMLElement>(".hqd-pick.p"));
    const s = Array.from(document.querySelectorAll<HTMLElement>(".hqd-pick.s"));
    expect(p).toHaveLength(5);
    expect(s).toHaveLength(5);
    p.forEach((el, i) => expect(el.style.animationDelay).toBe(`${pickDelayMs(i, "P")}ms`));
    s.forEach((el, i) => expect(el.style.animationDelay).toBe(`${pickDelayMs(i, "S")}ms`));
    expect(pickDelayMs(1, "P") - pickDelayMs(0, "P")).toBe(150);
    // Predicted arrivals: faint dashed lines, no animation delay.
    expect(document.querySelectorAll(".hqd-pred.p")).toHaveLength(5);
    expect(document.querySelectorAll(".hqd-pred.s")).toHaveLength(5);
    // Every mark sits inside the track.
    for (const el of document.querySelectorAll<HTMLElement>(".hqd-mark")) {
      const left = parseFloat(el.style.left);
      expect(left).toBeGreaterThanOrEqual(0);
      expect(left).toBeLessThanOrEqual(100);
    }
  });
});

describe("missing values render gracefully, never as NaN", () => {
  it("every optional pick/prob/prediction null: no marks, no NaN/undefined anywhere", () => {
    data.evidence[HERO] = {
      status: "ready",
      evidence: evidence(HERO, 4, () => ({ pickP: null, pickS: null, probP: null, probS: null, predP: null, predS: null })),
    };
    render(<EvidenceDrawer />);
    select(HERO);
    expect(rows()).toHaveLength(4);
    expect(document.querySelectorAll(".hqd-mark")).toHaveLength(0);
    expectNoBadValues();
  });

  it("a mix: missing predP and pickS on some traces draws only what exists", () => {
    data.evidence[HERO] = {
      status: "ready",
      evidence: evidence(HERO, 4, (i) => (i % 2 ? { predP: null, pickS: null, probS: null } : {})),
    };
    render(<EvidenceDrawer />);
    select(HERO);
    expect(document.querySelectorAll(".hqd-pick.p")).toHaveLength(4);
    expect(document.querySelectorAll(".hqd-pick.s")).toHaveLength(2);
    expect(document.querySelectorAll(".hqd-pred.p")).toHaveLength(2);
    expect(document.querySelectorAll(".hqd-pred.s")).toHaveLength(4);
    expectNoBadValues();
  });

  it("null hErrM / vErrM show a dash; depth-on-edge is noted", () => {
    data.evidence["hq-test-000003"] = { status: "ready", evidence: evidence("hq-test-000003", 4) };
    render(<EvidenceDrawer />);
    select("hq-test-000003");
    expect(screen.getAllByText("—")).toHaveLength(2);
    expect(screen.getByText(/edge of the location grid/)).toBeTruthy();
    expect(document.querySelector("circle title")?.textContent ?? "").not.toMatch(/horizontal error/);
    expectNoBadValues();
  });

  it("a pick outside the drawn window isn't drawn", () => {
    data.evidence[HERO] = { status: "ready", evidence: evidence(HERO, 4, () => ({ pickS: ORIGIN + 60 })) };
    render(<EvidenceDrawer />);
    select(HERO);
    expect(document.querySelectorAll(".hqd-pick.s")).toHaveLength(0);
    expect(document.querySelectorAll(".hqd-pick.p")).toHaveLength(4);
  });

  it("invalid traces are left out and reported, not drawn as garbage", () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    data.evidence[HERO] = { status: "ready", evidence: evidence(HERO, 4, (i) => (i === 2 ? { dt: 0 } : {})) };
    render(<EvidenceDrawer />);
    select(HERO);
    expect(rows()).toHaveLength(3);
    expect(screen.getByText(/1 trace not drawn/)).toBeTruthy();
    expect(err).toHaveBeenCalled();
    expectNoBadValues();
    err.mockRestore();
  });
});

describe("header details from the record", () => {
  it("magnitude with its type and the matched public-catalog id when present", () => {
    data.evidence["hq-test-000002"] = { status: "ready", evidence: evidence("hq-test-000002", 4) };
    render(<EvidenceDrawer />);
    select("hq-test-000002");
    expect(screen.getByText("M 0.8 ML_cal ±0.2")).toBeTruthy();
    expect(screen.getByText("uu60500001")).toBeTruthy();
    expect(drawer().textContent).toMatch(/public regional catalog · Δt 0\.12 s · 180 m/);
    expect(screen.getByText("Tier B")).toBeTruthy();
  });

  it("no magnitude or match rows when absent", () => {
    data.evidence[HERO] = { status: "ready", evidence: evidence(HERO, 4) };
    render(<EvidenceDrawer />);
    select(HERO);
    expect(drawer().textContent).not.toMatch(/Magnitude|Matched/);
  });

  it("flags synthetic data", () => {
    data.bundle = readyBundle(true);
    render(<EvidenceDrawer />);
    select(HERO);
    expect(screen.getByText("Synthetic")).toBeTruthy();
  });
});

describe("figures", () => {
  it("mini map: a line from each evidence station to the epicenter, plus a scale bar", () => {
    data.evidence[HERO] = { status: "ready", evidence: evidence(HERO, 16) };
    render(<EvidenceDrawer />);
    select(HERO);
    const map = screen.getByRole("img", { name: /Plan view: epicenter and 16 stations/ });
    expect(map.querySelectorAll("line")).toHaveLength(16);
    expect(map.textContent).toMatch(/\d+(\.\d+)? k?m/);
  });

  it("depth section: borehole sensor at true depth and the depth label from meta", () => {
    data.evidence[HERO] = { status: "ready", evidence: evidence(HERO, 16) };
    render(<EvidenceDrawer />);
    select(HERO);
    const depth = screen.getByRole("img", { name: /Depth section: event at 1\.72 km/ });
    expect(depth.querySelector("rect title")?.textContent).toBe("XX.S15 borehole sensor, 0.98 km");
    expect(screen.getAllByText(/Depth below site surface \(test ref\)/)[0]).toBeTruthy();
  });

  it("lists evidence stations the bundle doesn't know", () => {
    data.evidence[HERO] = { status: "ready", evidence: evidence(HERO, 4, (i) => (i === 3 ? { stationId: "ZZ.GONE" } : {})) };
    render(<EvidenceDrawer />);
    select(HERO);
    expect(document.querySelector(".hqd-note")?.textContent).toMatch(/station list: ZZ\.GONE$/);
    expect(screen.getByRole("img", { name: /3 stations/ })).toBeTruthy();
  });
});

describe("public-catalog status", () => {
  it("marks an event with no public-catalog match, and only that", () => {
    data.bundle = { ...readyBundle(), events: [event(HERO, { catalogMatch: null }), EVENTS[1]] } as BundleState;
    render(<EvidenceDrawer />);
    select(HERO);
    expect(screen.getByTestId("not-in-catalog").textContent).toBe("Not in the public regional catalog");
    select("hq-test-000002"); // matched to a public-catalog event
    expect(screen.queryByTestId("not-in-catalog")).toBeNull();
    expect(screen.getByText("uu60500001")).toBeTruthy();
  });
});

describe("tier reasons", () => {
  it("clamp to two lines with a toggle only when they overflow", () => {
    const sh = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollHeight");
    const ch = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientHeight");
    let overflow = false;
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", { configurable: true, get() { return overflow && this.classList.contains("hqd-reasons") ? 120 : 0; } });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", { configurable: true, get() { return overflow && this.classList.contains("hqd-reasons") ? 30 : 0; } });
    try {
      data.evidence[HERO] = { status: "ready", evidence: evidence(HERO, 4) };
      const { unmount } = render(<EvidenceDrawer />);
      select(HERO);
      expect(document.querySelector(".hqd-reasons")!.getAttribute("data-clamped")).toBe("true");
      expect(screen.queryByRole("button", { name: "Show all reasons" })).toBeNull(); // fits: no toggle
      unmount();
      act(() => useDemo.getState().select(null));
      overflow = true;
      render(<EvidenceDrawer />);
      select(HERO);
      const more = screen.getByRole("button", { name: "Show all reasons" });
      fireEvent.click(more);
      expect(document.querySelector(".hqd-reasons")!.getAttribute("data-clamped")).toBe("false");
      expect(screen.getByRole("button", { name: "Show less" }).getAttribute("aria-expanded")).toBe("true");
    } finally {
      if (sh) Object.defineProperty(HTMLElement.prototype, "scrollHeight", sh);
      if (ch) Object.defineProperty(HTMLElement.prototype, "clientHeight", ch);
    }
  });
});

describe("loading and errors", () => {
  it("evidence loading: header at once, a placeholder for traces", () => {
    render(<EvidenceDrawer />);
    select(HERO);
    expect(screen.getByRole("heading").textContent).toBe("11 stations agreed");
    expect(document.querySelector('.hqd-record[aria-busy="true"]')).toBeTruthy();
    expect(rows()).toHaveLength(0);
    expectNoBadValues();
  });

  it("no evidence file (404): a plain note with no URL or digits; figures from the picking stations", () => {
    const info = vi.spyOn(console, "info").mockImplementation(() => {});
    const ids = STATIONS.slice(0, 3).map((s, i) => `phasenet:instance:${s.id}:P:${1789000000 + i}.5`);
    data.bundle = { ...readyBundle(), events: [event(HERO, { pickIds: [...ids, `phasenet:instance:${STATIONS[0].id}:S:1789000001.5`] })] } as BundleState;
    data.evidence[HERO] = { status: "error", message: "/data/mock/evidence/x.json: HTTP 404" };
    render(<EvidenceDrawer />);
    select(HERO);
    const note = screen.getByTestId("evidence-unavailable");
    expect(note.textContent).toBe("No waveform evidence was exported for this event.");
    expect(note.className).toBe("hqd-note"); // neutral, not the alert style
    expect(document.querySelector(".hqd")!.textContent).not.toMatch(/HTTP|\/data\/|\.json/);
    expect(rows()).toHaveLength(0);
    // The event's location and errors are known: the figures show, with lines to the 3 picking stations.
    expect(screen.getByRole("img", { name: /3 stations/ })).toBeTruthy();
    expect(screen.getByText(/picking station → epicenter lines/)).toBeTruthy();
    expect(info).toHaveBeenCalledWith(expect.stringMatching(/HTTP 404/));
    info.mockRestore();
    expectNoBadValues();
  });

  it("evidence failing to load for another reason: a plain error without the URL, logged loudly", () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    data.evidence[HERO] = { status: "error", message: "/data/mock/evidence/x.json: HTTP 500" };
    render(<EvidenceDrawer />);
    select(HERO);
    const note = screen.getByTestId("evidence-unavailable");
    expect(note.textContent).toBe("Waveform evidence could not be loaded.");
    expect(note.className).toBe("hqd-error");
    expect(error).toHaveBeenCalledWith(expect.stringMatching(/HTTP 500/));
    error.mockRestore();
  });

  it("bundle still loading: the id and a loading line", () => {
    data.bundle = { status: "loading" };
    render(<EvidenceDrawer />);
    select(HERO);
    expect(screen.getByText(HERO)).toBeTruthy();
    expect(screen.getByText(/Loading the data bundle/)).toBeTruthy();
  });

  it("an id that isn't in the bundle says so", () => {
    render(<EvidenceDrawer />);
    select("hq-nope");
    expect(screen.getByText(/not in the loaded bundle/)).toBeTruthy();
  });
});

describe("closing", () => {
  it("the close button calls select(null); contents unmount after the slide-out", () => {
    vi.useFakeTimers();
    data.evidence[HERO] = { status: "ready", evidence: evidence(HERO, 4) };
    render(<EvidenceDrawer />);
    select(HERO);
    fireEvent.click(screen.getByRole("button", { name: "Close evidence drawer" }));
    expect(useDemo.getState().selectedEventId).toBeNull();
    expect(drawer().dataset.open).toBe("false");
    // Still showing the last event while it slides out...
    expect(rows()).toHaveLength(4);
    act(() => {
      vi.advanceTimersByTime(600);
    });
    // ...then empty.
    expect(drawer().querySelector("header")).toBeNull();
  });

  it("switching events while open swaps the contents immediately", () => {
    data.evidence[HERO] = { status: "ready", evidence: evidence(HERO, 16) };
    data.evidence["hq-test-000002"] = { status: "ready", evidence: evidence("hq-test-000002", 4) };
    render(<EvidenceDrawer />);
    select(HERO);
    select("hq-test-000002");
    expect(screen.getByText("hq-test-000002")).toBeTruthy();
    expect(rows()).toHaveLength(4);
  });

  it("reset() closes it (selection null)", () => {
    render(<EvidenceDrawer />);
    select(HERO);
    act(() => useDemo.getState().reset());
    expect(drawer().dataset.open).toBe("false");
  });
});

import { describe, expect, it } from "vitest";
import { FILTER_LOOK, type FilterLook } from "../filters/fade";
import { appearTimeOf, TIMELINE } from "../reveal/timeline";
import { TIME_ALL } from "../time/clock";
import type { CatalogEvent, SeismicEvent, Station } from "../types";
import {
  buildSectionModel,
  drawSection,
  niceStep,
  sectionHit,
  sectionPlot,
  type Ctx2D,
  type SectionState,
  type SectionStyle,
} from "./section";

// Test-local records (CLAUDE.md rule 5 allows tiny synthetic data inside tests). Every record carries a
// deliberately WRONG depthKm, so any read of it would show up in the assertions.
const REF = 1600;
function ev(id: string, e: number, elevM: number, tier: "A" | "B" | "C", order: number, errs: [number | null, number | null] = [100, 200]): SeismicEvent {
  return {
    id, runId: "t", source: "hq-pipeline", t: 0, latitude: 0, longitude: 0, elevM, depthKm: 99,
    enu: { e, n: 0, u: elevM - REF },
    quality: { method: "pyocto", statics: false, nStations: 8, nP: 8, nS: 6, rmsS: 0.05, gapDeg: 90, minEpiDistM: 1000, hErrM: errs[0], vErrM: errs[1], depthOnEdge: false },
    tier, tierReasons: [], meanPickProb: 0.8, magnitude: null, catalogMatch: null, revealOrder: order, pickIds: [],
  };
}
const events = [
  ev("a", 0, REF - 2000, "A", 0),
  ev("b", 1000, REF - 3000, "B", 1),
  ev("c", -1000, REF - 4000, "C", 2),
  ev("d", 2000, REF - 2500, "A", 3, [null, null]),
];
const catalog: CatalogEvent[] = [
  { id: "p1", source: "t", t: 0, latitude: 0, longitude: 0, depthKm: 77, depthDatum: "sea level", elevM: REF - 2100, mag: null, magType: null, enu: { e: 50, n: 0, u: -2100 }, matchedEventId: "a" },
  { id: "p2", source: "t", t: 0, latitude: 0, longitude: 0, depthKm: 77, depthDatum: "sea level", elevM: REF - 5000, mag: null, magType: null, enu: { e: -3000, n: 0, u: -5000 }, matchedEventId: null },
];
function station(id: string, e: number, surface: number, depth: number, kind: Station["kind"]): Station {
  return { id, network: "XX", station: id, location: "", latitude: 0, longitude: 0, surfaceElevM: surface, sensorDepthM: depth, sensorElevM: surface - depth, kind, channels: ["HHZ"], sampleRateHz: 100, enu: { e, n: 0, u: surface - depth - REF }, preprocessProfile: "p", usedInRun: true, staticsS: {} };
}
const stations = [station("S1", -2000, REF + 50, 0, "surface"), station("B1", 500, REF + 20, 1000, "borehole")];
const model = buildSectionModel(events, catalog, stations, { refSurfaceElevM: REF }, 0);

/** Records every draw call; `arcs` groups arc centers by the fill color + alpha in effect. */
class RecordingCtx implements Ctx2D {
  globalAlpha = 1; fillStyle = ""; strokeStyle = ""; lineWidth = 1; font = ""; textAlign = ""; textBaseline = "";
  calls: { op: string; args: number[] }[] = [];
  texts: string[] = [];
  private path: { x: number; y: number; r: number }[] = [];
  private segs: number[][] = [];
  fills: { style: string; alpha: number; arcs: { x: number; y: number; r: number }[] }[] = [];
  strokes: { style: string; alpha: number; arcs: { x: number; y: number; r: number }[]; segs: number[][] }[] = [];
  setTransform(...a: number[]) { this.calls.push({ op: "setTransform", args: a }); }
  clearRect(...a: number[]) { this.calls.push({ op: "clearRect", args: a }); }
  beginPath() { this.path = []; this.segs = []; }
  moveTo(x: number, y: number) { this.calls.push({ op: "moveTo", args: [x, y] }); this.segs.push([x, y]); }
  lineTo(x: number, y: number) { this.calls.push({ op: "lineTo", args: [x, y] }); this.segs[this.segs.length - 1]?.push(x, y); }
  arc(x: number, y: number, r: number, a0: number, a1: number) { this.calls.push({ op: "arc", args: [x, y, r, a0, a1] }); this.path.push({ x, y, r }); }
  closePath() {}
  fill() { if (this.path.length) this.fills.push({ style: this.fillStyle, alpha: this.globalAlpha, arcs: this.path }); }
  stroke() { this.strokes.push({ style: this.strokeStyle, alpha: this.globalAlpha, arcs: this.path, segs: this.segs }); }
  fillText(t: string) { this.texts.push(t); }
  setLineDash() {}
}

const style: SectionStyle = { recovered: "amber", publicDot: "white", station: "grey", contour: "c", textDim: "t", halo: "halo", tickFont: "10px mono", dpr: 2 };
const W = 480, H = 260;
const plot = sectionPlot(model, W, H);

function state(over: Partial<SectionState> = {}, look: FilterLook = { ...FILTER_LOOK.all }): SectionState {
  return { phase: "revealed", filter: "all", revealElapsedS: TIMELINE.endS, timeNowRel: TIME_ALL, look, selectedIndex: -1, ...over };
}
function draw(s: SectionState) {
  const ctx = new RecordingCtx();
  drawSection(ctx, model, plot, s, style, W, H);
  return ctx;
}
const toDepthKm = (y: number) => (y - plot.oy) / plot.k;
const toEastKm = (x: number) => (x - plot.ox) / plot.k;

describe("depth section model", () => {
  it("projects every event onto grid east and site depth from elevM, never the published depthKm", () => {
    expect(Array.from(model.candidates.xy)).toEqual([0, 2, 1, 3, -1, 4, 2, 2.5]);
    expect(Array.from(model.publicEvents.xy)).toEqual([0.05, 2.1, -3, 5]);
  });

  it("puts borehole sensors at sensorElevM with their wellhead at surfaceElevM", () => {
    expect(model.sensors[3]).toBeCloseTo((REF - (REF + 20 - 1000)) / 1000, 12);
    expect(model.wellheads[3]).toBeCloseTo(-0.02, 12);
    expect(Array.from(model.borehole)).toEqual([0, 1]);
  });

  it("uses the renderer's appearance times and links public events to their matched candidate", () => {
    expect(model.candidateAppearAt[0]).toBe(appearTimeOf(0));
    expect(model.candidateAppearAt[3]).toBe(appearTimeOf(1));
    expect(model.publicTargets).toEqual(["a", null]);
  });
});

describe("sectionPlot", () => {
  it("is true scale (one px/km for both axes) and fits the whole population, outliers included", () => {
    for (const [e, d] of [[-3, 5], [2, 2.5], [-1, 4]]) {
      const x = plot.ox + e * plot.k, y = plot.oy + d * plot.k;
      expect(x).toBeGreaterThanOrEqual(plot.x0);
      expect(x).toBeLessThanOrEqual(plot.x0 + plot.w);
      expect(y).toBeGreaterThanOrEqual(plot.y0);
      expect(y).toBeLessThanOrEqual(plot.y0 + plot.h);
    }
  });

  it("labels ticks with derived round numbers only, including the surface at 0", () => {
    const labels = plot.depthTicks.map((t) => t.label);
    expect(labels).toContain("0");
    for (const t of [...plot.depthTicks, ...plot.eastTicks]) expect(t.label).toMatch(/^-?\d+(\.\d+)?$/);
  });

  it("niceStep picks 1/2/5 × 10^n", () => {
    expect(niceStep(10, 5)).toBe(2);
    expect(niceStep(0.9, 5)).toBe(0.2);
    expect(niceStep(37, 6)).toBe(5);
    expect(niceStep(0, 5)).toBe(1);
  });
});

describe("drawSection", () => {
  it("before the reveal shows the public catalog and stations only", () => {
    const ctx = draw(state({ phase: "public", filter: "public", revealElapsedS: 0 }, { ...FILTER_LOOK.public }));
    expect(ctx.fills.some((f) => f.style === "amber")).toBe(false);
    const pub = ctx.fills.find((f) => f.style === "white")!;
    expect(pub.arcs).toHaveLength(2);
    expect(toDepthKm(pub.arcs[0].y)).toBeCloseTo(2.1, 9);
    expect(toEastKm(pub.arcs[0].x)).toBeCloseTo(0.05, 9);
  });

  it("during the reveal draws only candidates whose appearance time has passed", () => {
    const t = (appearTimeOf(1 / 3) + appearTimeOf(2 / 3)) / 2; // after a, b; before c, d
    const ctx = draw(state({ phase: "revealing", revealElapsedS: t }));
    const drawn = ctx.fills.filter((f) => f.style === "amber").flatMap((f) => f.arcs.map((a) => +toEastKm(a.x).toFixed(9)));
    expect(drawn.sort()).toEqual([0, 1]);
  });

  it("under ALL draws every tier at its tier opacity; under STRICT B and C at the background weight", () => {
    const all = draw(state()).fills.filter((f) => f.style === "amber").map((f) => +f.alpha.toFixed(3)).sort();
    expect(all).toEqual([0.3, 0.6, 1]);
    const strict = draw(state({ filter: "strict" }, { ...FILTER_LOOK.strict })).fills.filter((f) => f.style === "amber");
    expect(strict.map((f) => +f.alpha.toFixed(3)).sort()).toEqual([0.05, 0.05, 1]);
  });

  it("under STRICT draws uncertainty arms only where errors exist (no invented arm)", () => {
    const ctx = draw(state({ filter: "strict" }, { ...FILTER_LOOK.strict }));
    const arms = ctx.strokes.filter((s) => s.style === "halo" && s.arcs.length === 0);
    expect(arms).toHaveLength(1);
    const segs = arms[0].segs.map((seg) => seg.map((v) => +v.toFixed(6)));
    const ax = plot.ox, ay = plot.oy + 2 * plot.k;
    // event "a": ±100 m east, ±200 m depth
    expect(segs).toContainEqual([ax - 0.1 * plot.k, ay, ax + 0.1 * plot.k, ay].map((v) => +v.toFixed(6)));
    expect(segs).toContainEqual([ax, ay - 0.2 * plot.k, ax, ay + 0.2 * plot.k].map((v) => +v.toFixed(6)));
    // tier A event "d" has no errors: no arm anywhere near it; B and C never get arms
    expect(segs).toHaveLength(2);
  });

  it("rings the selected event, and only once it is drawn", () => {
    const ring = (s: SectionState) => draw(s).strokes.filter((x) => x.style === "halo" && x.arcs.length === 1);
    expect(ring(state({ selectedIndex: 1 }))).toHaveLength(1);
    expect(ring(state({ selectedIndex: 1, phase: "public", filter: "public" }, { ...FILTER_LOOK.public }))).toHaveLength(0);
  });

  it("never passes NaN or Infinity to the context", () => {
    for (const s of [state(), state({ filter: "strict" }, { ...FILTER_LOOK.strict }), state({ phase: "public", filter: "public" }, { ...FILTER_LOOK.public })]) {
      for (const c of draw(s).calls) for (const a of c.args) expect(Number.isFinite(a)).toBe(true);
    }
  });
});

describe("sectionHit", () => {
  const at = (e: number, d: number) => [plot.ox + e * plot.k, plot.oy + d * plot.k] as const;

  it("selects the nearest visible candidate within the threshold", () => {
    const [x, y] = at(1, 3);
    expect(sectionHit(model, plot, { phase: "revealed", filter: "all", revealElapsedS: TIMELINE.endS }, x + 2, y, 10)).toBe("b");
  });

  it("uses the 3D picker's gates: nothing before the reveal, only Tier A under STRICT", () => {
    const [x, y] = at(1, 3);
    expect(sectionHit(model, plot, { phase: "public", filter: "public", revealElapsedS: 0 }, x, y, 10)).toBeNull();
    expect(sectionHit(model, plot, { phase: "revealed", filter: "strict", revealElapsedS: TIMELINE.endS }, x, y, 10)).toBeNull();
  });

  it("a public event selects its matched candidate; an unmatched one selects nothing", () => {
    const [x1, y1] = at(0.05, 2.1);
    const [x2, y2] = at(-3, 5);
    const s = { phase: "public" as const, filter: "public" as const, revealElapsedS: 0 };
    expect(sectionHit(model, plot, s, x1, y1, 3)).toBe("a");
    expect(sectionHit(model, plot, s, x2, y2, 10)).toBeNull();
  });
});

describe("time mode in the depth section (WEB-06)", () => {
  // The same records with distinct origin times (s since windowStart = 0): a 1000, b 2000, c 3000, d 4000; public p1 500, p2 3500.
  const timed = events.map((e, i) => ({ ...e, t: (i + 1) * 1000 }));
  const timedCatalog = catalog.map((c, i) => ({ ...c, t: i === 0 ? 500 : 3500 }));
  const tm = buildSectionModel(timed, timedCatalog, stations, { refSurfaceElevM: REF }, 0);
  const tp = sectionPlot(tm, W, H);
  const drawAt = (now: number) => {
    const ctx = new RecordingCtx();
    drawSection(ctx, tm, tp, state({ timeNowRel: now }), style, W, H);
    return ctx;
  };
  const east = (ctx: RecordingCtx, color: string) =>
    ctx.fills.filter((f) => f.style === color).flatMap((f) => f.arcs.map((a) => +((a.x - tp.ox) / tp.k).toFixed(6))).sort((a, b) => a - b);

  it("draws only events whose origin time has passed; TIME_ALL draws everything", () => {
    expect(east(drawAt(2500), "amber")).toEqual([0, 1]); // a, b
    expect(east(drawAt(2500), "white")).toEqual([0.05]); // p1
    expect(east(drawAt(0), "amber")).toEqual([]);
    expect(east(drawAt(0), "white")).toEqual([]);
    expect(east(drawAt(TIME_ALL), "amber")).toEqual([-1, 0, 1, 2]);
    expect(east(drawAt(TIME_ALL), "white")).toEqual([-3, 0.05]);
  });

  it("rings the selected event only once it is shown, and never picks a hidden one", () => {
    const ring = (now: number) => {
      const ctx = new RecordingCtx();
      drawSection(ctx, tm, tp, state({ timeNowRel: now, selectedIndex: 1 }), style, W, H);
      return ctx.strokes.filter((x) => x.style === "halo" && x.arcs.length === 1).length;
    };
    expect(ring(1500)).toBe(0);
    expect(ring(2000)).toBe(1);
    const x = tp.ox + 1 * tp.k;
    const y = tp.oy + 3 * tp.k;
    const s = { phase: "revealed" as const, filter: "all" as const, revealElapsedS: TIMELINE.endS };
    expect(sectionHit(tm, tp, { ...s, timeNowRel: 1500 }, x, y, 10)).toBeNull();
    expect(sectionHit(tm, tp, { ...s, timeNowRel: 2000 }, x, y, 10)).toBe("b");
    expect(sectionHit(tm, tp, s, x, y, 10)).toBe("b"); // no time state: time mode off
  });
});

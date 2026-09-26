import { describe, expect, it } from "vitest";
import { buildStripModel, drawStrip, FUTURE_ALPHA, recoveredSet, stripTime, stripX, type StripCtx, type StripStyle } from "./strip";

const START = Date.UTC(2026, 8, 10) / 1000;
const END = START + 4 * 3600; // a test-local 4 h window, 10-minute bins → 24 bins
const BIN = 600;

// Test-local synthetic records (built inside the test, as CLAUDE.md allows).
const events = [
  { t: START + 60, tier: "A" },
  { t: START + 120, tier: "C" },
  { t: START + 300, tier: "B" },
  { t: START + 700, tier: "A" },
  { t: START + 3 * 3600 + 5, tier: "C" },
];
const catalog = [{ t: START + 30 }, { t: START + 650 }, { t: START + 660 }, { t: START + 670 }];

class RecordingCtx implements StripCtx {
  fillStyle: unknown = "";
  strokeStyle: unknown = "";
  globalAlpha = 1;
  lineWidth = 1;
  rects: { style: unknown; alpha: number; x: number; y: number; w: number; h: number }[] = [];
  lines: { x: number; y0: number; y1: number; style: unknown }[] = [];
  private from: [number, number] = [0, 0];
  setTransform() {}
  clearRect() {}
  fillRect(x: number, y: number, w: number, h: number) {
    this.rects.push({ style: this.fillStyle, alpha: this.globalAlpha, x, y, w, h });
  }
  beginPath() {}
  moveTo(x: number, y: number) {
    this.from = [x, y];
  }
  lineTo(x: number, y: number) {
    this.lines.push({ x, y0: this.from[1], y1: y, style: this.strokeStyle });
  }
  stroke() {}
}

const style: StripStyle = { recovered: "amber", publicBar: "white", baseline: "base", playhead: "head", dpr: 2 };
const model = buildStripModel(events, catalog, START, END, BIN);
const W = 240; // 10 px per bin
const H = 50;

describe("buildStripModel", () => {
  it("bins recovered (all and Tier A) and public on one shared scale", () => {
    expect(model.bins).toBe(24);
    expect(Array.from(model.recoveredAll.slice(0, 2))).toEqual([3, 1]);
    expect(Array.from(model.recoveredStrict.slice(0, 2))).toEqual([1, 1]);
    expect(Array.from(model.publicBins.slice(0, 2))).toEqual([1, 3]);
    expect(model.recoveredAll[18]).toBe(1);
    expect(model.maxBin).toBe(3); // the larger of either layer's busiest bin
    expect(Array.from(model.sortedPublic)).toEqual(catalog.map((c) => c.t));
  });
});

describe("drawStrip", () => {
  const draw = (tNow: number | null, filter: "public" | "all" | "strict" = "all") => {
    const ctx = new RecordingCtx();
    drawStrip(ctx, model, tNow, filter, style, W, H);
    return ctx;
  };

  it("draws recovered up and public down from the baseline, heights in proportion to counts", () => {
    const ctx = draw(null);
    const mid = Math.round(H / 2);
    const half = mid - 1;
    const rec0 = ctx.rects.find((r) => r.style === "amber" && r.x === 0)!;
    expect(rec0.y + rec0.h).toBe(mid);
    expect(rec0.h).toBeCloseTo(half, 6); // 3 of max 3
    const pub1 = ctx.rects.find((r) => r.style === "white" && r.x === 10)!;
    expect(pub1.y).toBe(mid + 1);
    expect(pub1.h).toBeCloseTo(half, 6); // 3 of max 3
    const pub0 = ctx.rects.find((r) => r.style === "white" && r.x === 0)!;
    expect(pub0.h).toBeCloseTo(half / 3, 6);
    expect(ctx.rects.some((r) => r.style === "base" && r.y === mid && r.w === W)).toBe(true);
    // Empty bins draw nothing.
    expect(ctx.rects.filter((r) => r.x === 50).length).toBe(0);
  });

  it("dims bins after now and draws the playhead at now", () => {
    const now = START + 900; // inside bin 1
    const ctx = draw(now);
    for (const r of ctx.rects.filter((r) => r.style === "amber" || r.style === "white")) {
      const bin = Math.round(r.x / 10);
      expect(r.alpha, `bin ${bin}`).toBe(bin <= 1 ? 1 : FUTURE_ALPHA);
    }
    const head = ctx.lines.find((l) => l.style === "head")!;
    expect(head.x).toBeCloseTo(Math.round(stripX(model, now, W)) + 0.5, 6);
    expect([head.y0, head.y1]).toEqual([0, H]);
  });

  it("null now draws the whole window as shown, with no playhead", () => {
    const ctx = draw(null);
    expect(ctx.rects.every((r) => r.alpha === 1)).toBe(true);
    expect(ctx.lines.length).toBe(0);
  });

  it("follows the filter: STRICT bars count Tier A only, PUBLIC hides recovered bars", () => {
    const strict = draw(null, "strict");
    const rec0 = strict.rects.find((r) => r.style === "amber" && r.x === 0)!;
    expect(rec0.h).toBeCloseTo((Math.round(H / 2) - 1) / 3, 6); // 1 Tier A in bin 0, same scale
    expect(draw(null, "public").rects.some((r) => r.style === "amber")).toBe(false);
    expect(recoveredSet("all")).toBe("all");
    expect(recoveredSet("strict")).toBe("strict");
    expect(recoveredSet("public")).toBe("none");
  });
});

describe("stripX / stripTime", () => {
  it("map the window onto the strip and back, clamped", () => {
    expect(stripX(model, START, W)).toBe(0);
    expect(stripX(model, END, W)).toBe(W);
    expect(stripX(model, START - 999, W)).toBe(0);
    expect(stripX(model, END + 999, W)).toBe(W);
    for (const t of [START, START + 1234, END]) expect(stripTime(model, stripX(model, t, W), W)).toBeCloseTo(t, 6);
    expect(stripTime(model, -10, W)).toBe(START);
    expect(stripTime(model, W + 10, W)).toBe(END);
  });
});

import { describe, expect, it } from "vitest";
import type { WaveformSnippet } from "../scene/types";
import {
  recordCaption,
  evidenceUnavailable,
  pickedTraceCount,
  markPercent,
  niceStep,
  normalizationGain,
  pickDelayMs,
  PICK_STAGGER_MS,
  PICKS_START_MS,
  prepareTraces,
  recordDomain,
  timeTicks,
  TRACK_H,
  TRACK_W,
  tracePoints,
} from "./record";

const ORIGIN = 1_000;

// Test-local synthetic traces (rule 5).
function tr(stationId: string, epiDistM: number, over: Partial<WaveformSnippet> = {}): WaveformSnippet {
  return {
    pickP: null, pickS: null, probP: null, probS: null, predP: null, predS: null,
    stationId,
    channel: "HHZ",
    epiDistM,
    t0: ORIGIN - 1,
    dt: 0.01,
    samples: Array.from({ length: 601 }, (_, i) => Math.sin(i / 5) * 0.5),
    ...over,
  };
}

function parsePoints(points: string): [number, number][] {
  return points.split(" ").map((p) => p.split(",").map(Number) as [number, number]);
}

describe("prepareTraces", () => {
  it("sorts by epicentral distance, stably", () => {
    const { traces, skipped } = prepareTraces([tr("C", 3000), tr("A", 1000), tr("B1", 2000), tr("B2", 2000)]);
    expect(traces.map((t) => t.stationId)).toEqual(["A", "B1", "B2", "C"]);
    expect(skipped).toEqual([]);
  });

  it("reports traces it can't draw instead of drawing garbage", () => {
    const { traces, skipped } = prepareTraces([
      tr("OK", 1),
      tr("BADDT", 2, { dt: 0 }),
      tr("BADT0", 3, { t0: NaN }),
      tr("SHORT", 4, { samples: [1] }),
    ]);
    expect(traces.map((t) => t.stationId)).toEqual(["OK"]);
    expect(skipped).toEqual([
      "BADDT HHZ: dt is not a positive number",
      "BADT0 HHZ: t0 is not a number",
      "SHORT HHZ: fewer than 2 samples",
    ]);
  });
});

describe("time axis", () => {
  it("spans every trace's window, relative to the origin time", () => {
    const d = recordDomain([tr("A", 1), tr("B", 2, { t0: ORIGIN + 0.5 })], ORIGIN);
    expect(d?.start).toBeCloseTo(-1, 9);
    expect(d?.end).toBeCloseTo(0.5 + 6, 9); // 601 samples × 0.01 s = 6 s after t0
  });

  it("is null without traces", () => {
    expect(recordDomain([], ORIGIN)).toBeNull();
  });

  it("uses 1-2-5 steps and ticks on multiples of the step", () => {
    expect(niceStep(6.5, 7)).toBe(1);
    expect(niceStep(12, 7)).toBe(2);
    expect(niceStep(0.9, 7)).toBe(0.2);
    expect(timeTicks({ start: -1.3, end: 5.2 })).toEqual({ step: 1, ticks: [-1, 0, 1, 2, 3, 4, 5] });
  });
});

describe("marks (picks and predicted arrivals)", () => {
  const d = { start: -1, end: 5 };
  it("positions a mark as a percentage of the track", () => {
    expect(markPercent(ORIGIN + 2, ORIGIN, d)).toBe(50);
    expect(markPercent(ORIGIN - 1, ORIGIN, d)).toBe(0);
  });
  it("draws nothing for a missing or out-of-window mark", () => {
    expect(markPercent(null, ORIGIN, d)).toBeNull();
    expect(markPercent(undefined, ORIGIN, d)).toBeNull();
    expect(markPercent(NaN, ORIGIN, d)).toBeNull();
    expect(markPercent(ORIGIN + 9, ORIGIN, d)).toBeNull();
    expect(markPercent(ORIGIN - 2, ORIGIN, d)).toBeNull();
  });
});

describe("trace polylines", () => {
  it("normalizes each trace to its own peak", () => {
    expect(normalizationGain([0.25, -0.5, 0.1])).toBe(2);
    expect(normalizationGain([0, 0])).toBe(0);
    expect(normalizationGain([NaN, 0.5])).toBe(2);
  });

  it("maps samples onto the shared axis inside the row box", () => {
    const t = tr("A", 1, { samples: [0, 1, -1, 0], dt: 1, t0: ORIGIN });
    const d = { start: 0, end: 3 };
    const pts = parsePoints(tracePoints(t, ORIGIN, d));
    expect(pts.map((p) => p[0])).toEqual([0, TRACK_W / 3, (2 * TRACK_W) / 3, TRACK_W].map((v) => Math.round(v * 100) / 100));
    expect(pts[1][1]).toBeLessThan(TRACK_H / 2); // positive = up
    expect(pts[2][1]).toBeGreaterThan(TRACK_H / 2);
    for (const [x, y] of pts) {
      expect(Number.isFinite(x) && Number.isFinite(y)).toBe(true);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(TRACK_H);
    }
  });

  it("draws non-finite samples at zero and a flat trace flat, never NaN", () => {
    const t = tr("A", 1, { samples: [0, NaN, 0, 0], dt: 1, t0: ORIGIN });
    const s = tracePoints(t, ORIGIN, { start: 0, end: 3 });
    expect(s).not.toMatch(/NaN|Infinity|undefined/);
    for (const [, y] of parsePoints(s)) expect(y).toBe(TRACK_H / 2);
  });

  it("decimates dense traces to a min/max envelope that keeps the peaks", () => {
    const n = 20_000;
    const samples = Array.from({ length: n }, (_, i) => (i === 12_345 ? 1 : i === 777 ? -1 : 0.01 * Math.sin(i)));
    const t = tr("A", 1, { samples, dt: 0.001, t0: ORIGIN });
    const d = recordDomain([t], ORIGIN)!;
    const pts = parsePoints(tracePoints(t, ORIGIN, d));
    expect(pts.length).toBeLessThanOrEqual(2 * (TRACK_W + 1));
    const ys = pts.map((p) => p[1]);
    expect(Math.min(...ys)).toBeCloseTo(TRACK_H / 2 - 0.46 * TRACK_H, 1);
    expect(Math.max(...ys)).toBeCloseTo(TRACK_H / 2 + 0.46 * TRACK_H, 1);
    for (let i = 1; i < pts.length; i++) expect(pts[i][0]).toBeGreaterThanOrEqual(pts[i - 1][0]);
  });
});

describe("pick stagger", () => {
  it("staggers rows by 150 ms, S half a step after P", () => {
    expect(PICK_STAGGER_MS).toBe(150);
    expect(pickDelayMs(0, "P")).toBe(PICKS_START_MS);
    expect(pickDelayMs(3, "P")).toBe(PICKS_START_MS + 450);
    expect(pickDelayMs(3, "S")).toBe(PICKS_START_MS + 525);
  });
});

 it("reports malformed samples instead of showing them as quiet waveform evidence", () => {
   const prepared = prepareTraces([tr("BAD", 0, { samples: [0, NaN, 1] })]);
   expect(prepared.traces).toHaveLength(0);
   expect(prepared.skipped[0]).toContain("non-finite waveform sample");
 });

describe("evidence notices", () => {
  it("a 404 is an expected, neutral note; other failures are errors; neither repeats the URL or status", () => {
    const miss = evidenceUnavailable("/data/showcase/evidence/hq-x.json: HTTP 404");
    expect(miss).toEqual({ text: "No waveform evidence was exported for this event.", expected: true });
    const fail = evidenceUnavailable("/data/showcase/evidence/hq-x.json: HTTP 500");
    expect(fail.expected).toBe(false);
    for (const t of [miss.text, fail.text, evidenceUnavailable(undefined).text]) expect(t).not.toMatch(/\d|http|\/|json/i);
  });

  it("counts traces with a P or S pick", () => {
    expect(pickedTraceCount([{ pickP: 1 }, { pickS: 2 }, { pickP: null, pickS: null }, {}])).toBe(2);
    expect(pickedTraceCount([])).toBe(0);
  });
});

describe("recordCaption", () => {
  it("says 'k of N agreeing stations shown' only when every shown trace has a pick", () => {
    expect(recordCaption(16, 16, 23)).toBe("16 of 23 agreeing stations shown, closest first");
    expect(recordCaption(16, 11, 23)).toBe("16 stations shown, closest first; 11 of the 23 that agreed are among them");
    expect(recordCaption(1, 0, null)).toBe("1 station shown, closest first, 0 with a pick");
    for (const c of [recordCaption(16, 16, NaN), recordCaption(4, 2, undefined)]) expect(c).not.toMatch(/NaN|undefined|null/);
  });
});

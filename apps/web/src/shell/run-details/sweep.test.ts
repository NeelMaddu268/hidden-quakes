/** Sweep layout: the swept parameter, the ticks and the lines all derive from the points. */
import { describe, expect, it } from "vitest";
import type { SweepPoint } from "@/providers";
import { groupKey, niceTicks, sweepLayout, sweptParam, SWEEP_SERIES } from "./sweep";

function point(params: SweepPoint["params"], candidates: number): SweepPoint {
  return { params, candidates, recoveredPublic: Math.round(candidates / 10), tierA: Math.round(candidates / 3) };
}

const SWEEP: SweepPoint[] = [
  point({ nMin: 4, tolS: 1.5 }, 900),
  point({ nMin: 5, tolS: 1.5 }, 700),
  point({ nMin: 6, tolS: 1.5 }, 500),
  point({ nMin: 6, tolS: 1.0 }, 420),
  point({ nMin: 6, tolS: 2.0 }, 610),
];

describe("sweptParam", () => {
  it("picks the numeric parameter with the most distinct values", () => {
    expect(sweptParam(SWEEP)).toBe("nMin");
  });

  it("breaks a tie by key order and ignores non-numeric parameters", () => {
    expect(sweptParam([point({ a: 1, b: 1 }, 1), point({ a: 2, b: 2 }, 2)])).toBe("a");
    expect(sweptParam([point({ name: "x", n: 1 }, 1), point({ name: "y", n: 2 }, 2)])).toBe("n");
    expect(sweptParam([point({ name: "x" }, 1), point({ name: "y" }, 2)])).toBeNull();
    expect(sweptParam([])).toBeNull();
  });
});

describe("niceTicks", () => {
  it("steps by a round number from zero to at least the maximum", () => {
    expect(niceTicks(950, 4)).toEqual([0, 500, 1000]);
    expect(niceTicks(38, 4)).toEqual([0, 10, 20, 30, 40]);
    expect(niceTicks(2.4, 4)).toEqual([0, 1, 2, 3]);
    expect(niceTicks(0, 4)).toEqual([0]);
  });
});

describe("sweepLayout", () => {
  it("builds one series per count field with one mark per point", () => {
    const layout = sweepLayout(SWEEP, 4)!;
    expect(layout.xKey).toBe("nMin");
    expect(layout.xTicks).toEqual([4, 5, 6]);
    expect(layout.series.map((s) => s.field)).toEqual([...SWEEP_SERIES]);
    for (const series of layout.series) expect(series.marks).toHaveLength(SWEEP.length);
    expect(layout.yMax).toBeGreaterThanOrEqual(900);
    expect(layout.yTicks[layout.yTicks.length - 1]).toBe(layout.yMax);
  });

  it("draws a line only through points that share every other parameter", () => {
    const layout = sweepLayout(SWEEP, 4)!;
    const candidates = layout.series[0];
    expect(candidates.lines).toHaveLength(1);
    expect(candidates.lines[0].map((m) => m.index)).toEqual([0, 1, 2]);
    expect(candidates.lines[0].every((m) => m.group === groupKey(SWEEP[0].params, "nMin"))).toBe(true);
    expect(groupKey({ nMin: 6, tolS: 1.5 }, "nMin")).toBe("tolS=1.5");
  });

  it("is null for an empty sweep or one without a numeric parameter", () => {
    expect(sweepLayout([], 4)).toBeNull();
    expect(sweepLayout([point({ profile: "full" }, 10)], 4)).toBeNull();
  });
});

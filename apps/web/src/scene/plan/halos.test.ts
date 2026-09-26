import { describe, expect, it } from "vitest";
import { buildPlanHaloInstances, planHaloEligible } from "./halos";

const events = [
  { tier: "A" as const, quality: { hErrM: 200, vErrM: null } },
  { tier: "B" as const, quality: { hErrM: 300, vErrM: 400 } },
  { tier: "A" as const, quality: { hErrM: null, vErrM: 500 } },
  { tier: "A" as const, quality: { hErrM: 400, vErrM: 500 } },
];
const candidates = { count: 4, positions: new Float32Array([1, -2, -3, 4, -5, -6, 7, -8, -9, 10, -11, -12]), appearAt: new Float32Array([1, 3, 4, 6]) };

describe("plan uncertainty", () => {
  it("retains horizontal-only Tier A errors and skips missing horizontal errors", () => {
    const h = buildPlanHaloInstances(events, candidates);
    expect(h.count).toBe(2); expect(h.tierAWithoutHalo).toBe(1);
    expect([...h.eventIndex]).toEqual([0, 3]);
    expect([...h.positions]).toEqual([1, -2, -3, 10, -11, -12]);
    expect([...h.appearAt]).toEqual([1, 6]);
    expect(h.radii[0]).toBeCloseTo(.2); expect(h.radii[1]).toBeCloseTo(.4);
  });
  it("does not turn vertical exaggeration into horizontal uncertainty", () => {
    const exaggerated = { ...candidates, positions: candidates.positions.map((n, i) => i % 3 === 1 ? n * 5 : n) };
    const h = buildPlanHaloInstances(events, exaggerated);
    expect(h.radii).toEqual(buildPlanHaloInstances(events, candidates).radii);
    expect(h.positions[1]).toBe(-10);
  });
  it.each([null, 0, -1, NaN, Infinity])("does not fabricate a radius from %s", (hErrM) => {
    expect(planHaloEligible({ tier: "A", quality: { hErrM } })).toBe(false);
  });
  it("rejects drift between bundle records and renderer arrays", () => {
    expect(() => buildPlanHaloInstances(events.slice(1), candidates)).toThrow(/out of step/);
    expect(() => buildPlanHaloInstances(events, { ...candidates, appearAt: new Float32Array(1) })).toThrow(/out of step/);
    expect(() => buildPlanHaloInstances(events, { ...candidates, positions: new Float32Array(1) })).toThrow(/out of step/);
  });
  it("supports an empty bundle", () => {
    const h = buildPlanHaloInstances([], { count: 0, positions: new Float32Array(), appearAt: new Float32Array() });
    expect(h.count).toBe(0); expect(h.tierAWithoutHalo).toBe(0);
  });
});

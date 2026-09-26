import { describe, expect, it } from "vitest";
import { appearTimeOf } from "../reveal/timeline";
import type { CatalogEvent, SeismicEvent } from "../types";
import {
  buildCandidateInstances,
  buildPublicInstances,
  framingPositions,
  revealOrderIssues,
  revealSlots,
} from "./instances";

// Test-local synthetic records (rule 5 allows small synthetic data inside tests).
function ev(id: string, over: Partial<SeismicEvent> = {}): SeismicEvent {
  return {
    id,
    runId: "test",
    t: 1_000,
    latitude: 0,
    longitude: 0,
    elevM: 0,
    depthKm: 0,
    enu: { e: 0, n: 0, u: 0 },
    quality: {
      method: "pyocto",
      statics: false,
      nStations: 8,
      nP: 8,
      nS: 6,
      rmsS: 0.05,
      gapDeg: 90,
      minEpiDistM: 1000,
      hErrM: 100,
      vErrM: 200,
      depthOnEdge: false,
    },
    tier: "A",
    tierReasons: [],
    meanPickProb: 0.8,
    source: "hq-pipeline",
    magnitude: null,
    catalogMatch: null,
    revealOrder: 0,
    pickIds: [],
    ...over,
  };
}

function cat(id: string, over: Partial<CatalogEvent> = {}): CatalogEvent {
  return {
    id,
    source: "test",
    t: 1_000,
    latitude: 0,
    longitude: 0,
    depthKm: 0,
    depthDatum: "test",
    elevM: 0,
    mag: null,
    magType: null,
    matchedEventId: null,
    enu: { e: 0, n: 0, u: 0 },
    ...over,
  };
}

describe("revealSlots", () => {
  it("is revealOrder / (n − 1) for a proper permutation, in bundle positions", () => {
    const slots = revealSlots([2, 0, 3, 1, 4]);
    expect(Array.from(slots)).toEqual([0.5, 0, 0.75, 0.25, 1]);
  });

  it("never re-sorts by anything but revealOrder; ties keep bundle order", () => {
    expect(Array.from(revealSlots([5, 5, 1]))).toEqual([0.5, 1, 0]);
  });

  it("handles 0 and 1 events", () => {
    expect(revealSlots([]).length).toBe(0);
    expect(Array.from(revealSlots([7]))).toEqual([0]);
  });

  it("spans exactly [0, 1] so the last event appears at the end of the reveal window", () => {
    const n = 2000;
    const order = Array.from({ length: n }, (_, i) => n - 1 - i);
    const slots = revealSlots(order);
    expect(Math.min(...slots)).toBe(0);
    expect(Math.max(...slots)).toBe(1);
  });
});

describe("revealOrderIssues", () => {
  it("accepts a proper permutation", () => {
    expect(revealOrderIssues([1, 0, 2])).toEqual([]);
  });

  it("flags H2's −1 placeholder, out-of-range and duplicate orders", () => {
    expect(revealOrderIssues([-1, -1])).toEqual([
      "2 event(s) have a negative or non-integer revealOrder",
      "1 duplicate revealOrder value(s)",
    ]);
    expect(revealOrderIssues([0, 5])).toEqual(["1 event(s) have revealOrder >= event count (2)"]);
  });
});

describe("buildCandidateInstances", () => {
  const events = [
    ev("a", { enu: { e: 2500, n: 1200, u: -3400 }, tier: "A", revealOrder: 1, t: 1_060 }),
    ev("b", { enu: { e: -1000, n: 0, u: -2000 }, tier: "B", revealOrder: 0, t: 1_000 }),
    ev("c", { enu: { e: 0, n: -500, u: -1000 }, tier: "C", revealOrder: 2, t: 4_600 }),
  ];

  it("places every event through the ENU → scene mapping", () => {
    const inst = buildCandidateInstances(events, 1, 1_000);
    expect(inst.count).toBe(3);
    const p = Array.from(inst.positions).map((v) => Number(v.toFixed(6)));
    expect(p).toEqual([2.5, -3.4, -1.2, -1, -2, 0, 0, -1, 0.5]);
  });

  it("applies vertical exaggeration to y only", () => {
    const inst = buildCandidateInstances(events, 2, 1_000);
    expect(inst.positions[0]).toBeCloseTo(2.5, 6);
    expect(inst.positions[1]).toBeCloseTo(-6.8, 6);
    expect(inst.positions[2]).toBeCloseTo(-1.2, 6);
  });

  it("encodes tier, size, reveal slot and time since window start", () => {
    const inst = buildCandidateInstances(events, 1, 1_000);
    expect(Array.from(inst.tiers)).toEqual([0, 1, 2]);
    expect(inst.scales[0]).toBeGreaterThan(inst.scales[1]);
    expect(inst.scales[1]).toBeGreaterThan(inst.scales[2]);
    expect(Array.from(inst.revealAt)).toEqual([0.5, 0, 1]);
    // appearance times are the counter clock's exact inverse: first event at 1.0 s, last at 6.0 s
    expect(inst.appearAt[1]).toBe(1);
    expect(inst.appearAt[2]).toBe(6);
    expect(inst.appearAt[0]).toBeCloseTo(appearTimeOf(0.5), 5);
    expect(Array.from(inst.times)).toEqual([60, 0, 3600]);
  });

  it("maps ids both ways for picking and selection", () => {
    const inst = buildCandidateInstances(events, 1, 1_000);
    expect(inst.ids).toEqual(["a", "b", "c"]);
    expect(inst.indexById.get("c")).toBe(2);
  });

  it("fails loudly on duplicate ids or unknown tiers", () => {
    expect(() => buildCandidateInstances([ev("x"), ev("x")], 1, 0)).toThrow(/duplicate/);
    expect(() =>
      buildCandidateInstances([ev("y", { tier: "D" as unknown as "A" })], 1, 0),
    ).toThrow(/unknown tier/);
  });

  it("builds 2,000 instances (the performance-budget size) with correctly sized arrays", () => {
    const many = Array.from({ length: 2000 }, (_, i) =>
      ev(`e${i}`, { enu: { e: i, n: -i, u: -i }, revealOrder: i, tier: (["A", "B", "C"] as const)[i % 3] }),
    );
    const inst = buildCandidateInstances(many, 1, 0);
    expect(inst.count).toBe(2000);
    expect(inst.positions.length).toBe(6000);
    expect(inst.revealAt[1999]).toBe(1);
  });
});

describe("buildPublicInstances", () => {
  it("places catalog events and keeps them visible before the reveal", () => {
    const inst = buildPublicInstances(
      [cat("p1", { enu: { e: 1000, n: 2000, u: -3000 }, t: 1_030 })],
      1,
      1_000,
    );
    expect(Array.from(inst.positions)).toEqual([1, -3, -2]);
    expect(Array.from(inst.revealAt)).toEqual([-1]);
    expect(Array.from(inst.appearAt)).toEqual([-1]);
    expect(Array.from(inst.tiers)).toEqual([0]);
    expect(Array.from(inst.times)).toEqual([30]);
  });

  it("fails loudly on duplicate ids", () => {
    expect(() => buildPublicInstances([cat("p"), cat("p")], 1, 0)).toThrow(/duplicate/);
  });
});

describe("framingPositions", () => {
  const events = [
    ev("a", { enu: { e: 1000, n: 0, u: 0 }, tier: "A", revealOrder: 0 }),
    ev("b", { enu: { e: 2000, n: 0, u: 0 }, tier: "B", revealOrder: 1 }),
    ev("c", { enu: { e: 9000, n: 0, u: 0 }, tier: "C", revealOrder: 2 }),
  ];

  it("keeps tiers up to the cut and drops the rest", () => {
    const inst = buildCandidateInstances(events, 1, 0);
    expect(Array.from(framingPositions(inst, 1)).filter((_, i) => i % 3 === 0)).toEqual([1, 2]);
    expect(Array.from(framingPositions(inst, 0)).filter((_, i) => i % 3 === 0)).toEqual([1]);
  });

  it("falls back to every instance when nothing qualifies", () => {
    const inst = buildCandidateInstances([ev("c", { tier: "C" })], 1, 0);
    expect(framingPositions(inst, 1)).toBe(inst.positions);
  });
});

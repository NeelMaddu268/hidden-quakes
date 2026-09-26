import { describe, expect, it } from "vitest";
import type { SeismicEvent } from "../types";
import { buildHaloInstances, haloEligible } from "./halos";
import { buildCandidateInstances } from "./instances";

function ev(id: string, tier: "A" | "B" | "C", hErrM: number | null, vErrM: number | null, e = 0): SeismicEvent {
  return {
    id,
    runId: "t",
    t: 0,
    latitude: 0,
    longitude: 0,
    elevM: 0,
    depthKm: 0,
    enu: { e, n: 0, u: -2000 },
    quality: {
      method: "pyocto",
      statics: false,
      nStations: 8,
      nP: 8,
      nS: 6,
      rmsS: 0.05,
      gapDeg: 90,
      minEpiDistM: 1000,
      hErrM,
      vErrM,
      depthOnEdge: false,
    },
    tier,
    tierReasons: [],
    meanPickProb: 0.8,
    revealOrder: 0,
    pickIds: [],
  };
}

describe("haloEligible", () => {
  it("only Tier A with both 68% errors present and positive", () => {
    expect(haloEligible(ev("a", "A", 120, 300))).toBe(true);
    expect(haloEligible(ev("b", "B", 120, 300))).toBe(false);
    expect(haloEligible(ev("c", "A", null, 300))).toBe(false);
    expect(haloEligible(ev("d", "A", 120, null))).toBe(false);
    expect(haloEligible(ev("e", "A", 0, 300))).toBe(false);
    expect(haloEligible(ev("f", "A", Infinity, 300))).toBe(false);
    expect(haloEligible(ev("g", "A", 120, Number.NaN))).toBe(false);
  });
});

describe("buildHaloInstances", () => {
  const events = [
    ev("a", "A", 120, 300, 1000),
    ev("b", "B", 90, 200, 2000),
    ev("c", "A", null, 400, 3000),
    ev("d", "A", 250, 500, 4000),
  ].map((e, i) => ({ ...e, revealOrder: i }));

  it("sizes each halo from hErrM (horizontal) and vErrM (vertical) in km, at the event's position", () => {
    const cand = buildCandidateInstances(events, 1, 0);
    const halos = buildHaloInstances(events, cand, 1);
    expect(halos.count).toBe(2);
    expect(halos.tierAWithoutHalo).toBe(1); // "c" has no hErrM
    expect(Array.from(halos.eventIndex)).toEqual([0, 3]);
    const m0 = halos.matrices.slice(0, 16);
    expect(m0[0]).toBeCloseTo(0.12, 6);
    expect(m0[5]).toBeCloseTo(0.3, 6);
    expect(m0[10]).toBeCloseTo(0.12, 6);
    expect([m0[12], m0[13], m0[14]].map((v) => Number(v.toFixed(6)))).toEqual([1, -2, 0]);
    const m1 = halos.matrices.slice(16, 32);
    expect(m1[0]).toBeCloseTo(0.25, 6);
    expect(m1[5]).toBeCloseTo(0.5, 6);
    expect(m1[12]).toBeCloseTo(4, 6);
  });

  it("stretches only the vertical axis under vertical exaggeration", () => {
    const cand = buildCandidateInstances(events, 2, 0);
    const halos = buildHaloInstances(events, cand, 2);
    expect(halos.matrices[0]).toBeCloseTo(0.12, 6);
    expect(halos.matrices[5]).toBeCloseTo(0.6, 6);
    expect(halos.matrices[13]).toBeCloseTo(-4, 6);
  });

  it("carries each event's appearance time so a halo never shows before its event", () => {
    const cand = buildCandidateInstances(events, 1, 0);
    const halos = buildHaloInstances(events, cand, 1);
    expect(Array.from(halos.appearAt)).toEqual([cand.appearAt[0], cand.appearAt[3]]);
  });

  it("fails loudly if events and candidate instances are out of step", () => {
    const cand = buildCandidateInstances(events.slice(0, 2), 1, 0);
    expect(() => buildHaloInstances(events, cand, 1)).toThrow(/halos/);
  });
});

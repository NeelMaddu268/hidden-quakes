import { describe, expect, it } from "vitest";
import { candidateLayerOpacity, candidateRevealUniform, FILTER_TIER_OPACITY } from "./driver";

describe("FILTER_TIER_OPACITY", () => {
  it("ALL shows tiers at A 1.0, B 0.6, C 0.3", () => {
    expect(FILTER_TIER_OPACITY.all).toEqual([1, 0.6, 0.3]);
  });

  it("STRICT keeps A at full and fades B and C to 0.05", () => {
    expect(FILTER_TIER_OPACITY.strict).toEqual([1, 0.05, 0.05]);
  });
});

describe("candidateLayerOpacity", () => {
  it("hides the candidate layer only under PUBLIC", () => {
    expect(candidateLayerOpacity("public")).toBe(0);
    expect(candidateLayerOpacity("all")).toBe(1);
    expect(candidateLayerOpacity("strict")).toBe(1);
  });
});

describe("candidateRevealUniform", () => {
  it("hides every candidate before the reveal (all revealAt are >= 0)", () => {
    expect(candidateRevealUniform("public", 0.7, 0.05)).toBeLessThan(0);
  });

  it("follows the timeline slot while revealing", () => {
    expect(candidateRevealUniform("revealing", 0.42, 0.05)).toBe(0.42);
  });

  it("settles every instance past its pop once revealed", () => {
    expect(candidateRevealUniform("revealed", 0.3, 0.05)).toBeGreaterThanOrEqual(1 + 0.05);
  });
});

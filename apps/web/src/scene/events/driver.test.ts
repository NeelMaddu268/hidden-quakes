import { describe, expect, it } from "vitest";
import { TIMELINE } from "../reveal/timeline";
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
  it("hides every candidate before the reveal (every appearance time is >= 1 s)", () => {
    expect(candidateRevealUniform("public", 3)).toBeLessThan(TIMELINE.events.startS);
  });

  it("follows the reveal clock while revealing", () => {
    expect(candidateRevealUniform("revealing", 2.5)).toBe(2.5);
  });

  it("settles every instance past its pop once revealed", () => {
    expect(candidateRevealUniform("revealed", 0)).toBeGreaterThan(TIMELINE.endS + TIMELINE.popS);
  });
});

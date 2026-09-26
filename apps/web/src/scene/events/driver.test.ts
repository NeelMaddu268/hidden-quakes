import { describe, expect, it } from "vitest";
import { TIMELINE } from "../reveal/timeline";
import { candidateRevealUniform } from "./driver";

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

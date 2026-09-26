// Demo state → event-layer uniform values. Pure functions, called once per frame with scalars only.

import type { DemoPhase } from "../../state/demo";

/** Reveal-clock value for "revealed": far past every appearance and pop. */
export const REVEALED_ELAPSED_S = 1e4;

/**
 * `uRevealElapsed` for the candidate layer. Before the reveal nothing shows (−1 is before every
 * appearance time); while revealing it's the reveal clock; once revealed, every instance has settled.
 */
export function candidateRevealUniform(phase: DemoPhase, elapsedS: number): number {
  if (phase === "public") return -1;
  if (phase === "revealed") return REVEALED_ELAPSED_S;
  return elapsedS;
}

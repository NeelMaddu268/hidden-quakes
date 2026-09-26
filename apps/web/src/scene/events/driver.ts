// Demo state → event-layer uniform values. Pure functions, called once per frame with scalars only.

import { strictFadeOpacity, tierStyle } from "@hq/visualization";
import type { DemoPhase, EventFilter } from "../../state/demo";

/** Tier A/B/C opacity for each filter (docs/lanes/H3: A 1.0, B 0.6, C 0.3; STRICT fades B and C). */
export const FILTER_TIER_OPACITY: Readonly<Record<EventFilter, readonly [number, number, number]>> = {
  public: [tierStyle.A.opacity, tierStyle.B.opacity, tierStyle.C.opacity],
  all: [tierStyle.A.opacity, tierStyle.B.opacity, tierStyle.C.opacity],
  strict: [tierStyle.A.opacity, strictFadeOpacity, strictFadeOpacity],
};

/** The candidate (amber) layer is hidden under PUBLIC; public-catalog points always show. */
export function candidateLayerOpacity(filter: EventFilter): number {
  return filter === "public" ? 0 : 1;
}

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

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

/**
 * The `uReveal` threshold for the candidate layer. Before the reveal nothing shows; after it every
 * instance is past its pop. During "revealing", `eventSlot` is where the reveal timeline is in
 * revealAt units (0 = first event appears, 1 = last event appears).
 */
export function candidateRevealUniform(phase: DemoPhase, eventSlot: number, popWidth: number): number {
  if (phase === "public") return -1;
  if (phase === "revealed") return 1 + popWidth;
  return eventSlot;
}

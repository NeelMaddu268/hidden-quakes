// What can be picked right now, and what selecting it means. Kept in step with what the event layers
// draw (scene/events/driver.ts + the event shader), so a click can only land on a visible glyph.

import type { DemoPhase, EventFilter } from "../../state/demo";
import { candidateRevealUniform } from "../events/driver";
import { TIER_INDEX } from "../events/instances";
import type { CatalogEvent } from "../types";

/**
 * Whether candidate instance (tier index, appearance time) is pickable in this demo state:
 * - never before the reveal or under PUBLIC (the candidate layer is hidden);
 * - under STRICT only Tier A (B and C are faded to 0.05, which reads as hidden);
 * - while revealing, only once the instance has appeared on the reveal clock.
 * Time mode (WEB-06) gates separately: callers also require `shownAt(time, sceneFx.timeNowRel)`.
 */
export function candidatePickable(
  phase: DemoPhase,
  filter: EventFilter,
  revealElapsedS: number,
  tier: number,
  appearAt: number,
): boolean {
  if (phase === "public" || filter === "public") return false;
  if (filter === "strict" && tier !== TIER_INDEX.A) return false;
  if (phase === "revealing") return candidateRevealUniform(phase, revealElapsedS) >= appearAt;
  return true;
}

/**
 * For each public-catalog instance (catalog order), the candidate event that clicking it selects after
 * the reveal (before it, public points select nothing, so the drawer can't show a candidate early): its
 * `matchedEventId` when that event is in the bundle, else null. A null point has nothing to open, so it
 * is skipped by the pick and a click near it falls through to the nearest selectable glyph.
 */
export function publicSelectTargets(
  catalog: readonly CatalogEvent[],
  candidateIds: ReadonlyMap<string, number>,
): (string | null)[] {
  return catalog.map((c) => (c.matchedEventId && candidateIds.has(c.matchedEventId) ? c.matchedEventId : null));
}

/** Instance index of the selected event in a layer, or −1 (the shader's `uSelected`). */
export function selectedInstanceIndex(
  selectedEventId: string | null,
  indexById: ReadonlyMap<string, number>,
): number {
  if (selectedEventId === null) return -1;
  return indexById.get(selectedEventId) ?? -1;
}

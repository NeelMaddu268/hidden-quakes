// Uncertainty halos for Tier A under STRICT (lane doc → WEB-04): one ellipsoid per event whose
// semi-axes are the 68% location errors, hErrM horizontally and vErrM vertically (× vertical
// exaggeration). An event without an error estimate gets no halo; we never invent one.

import { METERS_PER_UNIT } from "../coords";
import type { SeismicEvent } from "../types";
import type { EventInstances } from "./instances";

export interface HaloInstances {
  count: number;
  /** Candidate instance index each halo belongs to (selection, picking). */
  eventIndex: Int32Array;
  /** Column-major 4×4 per halo: scale (h, v·VE, h) in km, then translation to the event. */
  matrices: Float32Array;
  /** The event's reveal slot, so a halo never shows before its event has appeared. */
  revealAt: Float32Array;
}

/** Tier A events with both 68% errors present and positive. */
export function haloEligible(ev: Pick<SeismicEvent, "tier" | "quality">): boolean {
  const { hErrM, vErrM } = ev.quality;
  return ev.tier === "A" && hErrM != null && vErrM != null && hErrM > 0 && vErrM > 0;
}

/**
 * Builds halo instances aligned with the candidate layer's instances (same order, same positions and
 * reveal slots), so halos and glyphs can never disagree about where an event is.
 */
export function buildHaloInstances(
  events: readonly Pick<SeismicEvent, "tier" | "quality">[],
  candidates: Pick<EventInstances, "positions" | "revealAt" | "count">,
  verticalExaggeration: number,
): HaloInstances {
  if (events.length !== candidates.count) {
    throw new Error(`halos: ${events.length} events but ${candidates.count} candidate instances`);
  }
  let count = 0;
  for (const ev of events) if (haloEligible(ev)) count++;
  const eventIndex = new Int32Array(count);
  const matrices = new Float32Array(count * 16);
  const revealAt = new Float32Array(count);
  let j = 0;
  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    if (!haloEligible(ev)) continue;
    const h = (ev.quality.hErrM as number) / METERS_PER_UNIT;
    const v = ((ev.quality.vErrM as number) / METERS_PER_UNIT) * verticalExaggeration;
    const m = j * 16;
    matrices[m] = h;
    matrices[m + 5] = v;
    matrices[m + 10] = h;
    matrices[m + 15] = 1;
    matrices[m + 12] = candidates.positions[i * 3];
    matrices[m + 13] = candidates.positions[i * 3 + 1];
    matrices[m + 14] = candidates.positions[i * 3 + 2];
    eventIndex[j] = i;
    revealAt[j] = candidates.revealAt[i];
    j++;
  }
  return { count, eventIndex, matrices, revealAt };
}

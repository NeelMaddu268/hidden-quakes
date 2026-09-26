// Horizontal uncertainty for the plan camera. Unlike a 3D ellipsoid this needs no vertical estimate.
import type { EventInstances } from "../events/instances";
import type { SeismicEvent } from "../types";
import { usableErrorKm } from "./geometry";

type PlanEvent = Pick<SeismicEvent, "tier"> & {
  quality: Pick<SeismicEvent["quality"], "hErrM">;
};

export interface PlanHaloInstances {
  count: number;
  tierAWithoutHalo: number;
  eventIndex: Int32Array;
  /** Actual scene positions, including the event elevation; no snapping to terrain. */
  positions: Float32Array;
  /** Horizontal radius in km. The renderer must draw a circle in the east/north plane. */
  radii: Float32Array;
  appearAt: Float32Array;
}

export function planHaloEligible(event: PlanEvent): boolean {
  return event.tier === "A" && usableErrorKm(event.quality.hErrM) > 0;
}

/** Build once per bundle; retain candidate ordering and the renderer's appearance clock. */
export function buildPlanHaloInstances(
  events: readonly PlanEvent[],
  candidates: Pick<EventInstances, "count" | "positions" | "appearAt">,
): PlanHaloInstances {
  if (events.length !== candidates.count || candidates.positions.length !== events.length * 3 || candidates.appearAt.length !== events.length) {
    throw new Error("plan halos: events and candidate instance arrays are out of step");
  }
  let count = 0, tierAWithoutHalo = 0;
  for (const event of events) {
    if (planHaloEligible(event)) count++;
    else if (event.tier === "A") tierAWithoutHalo++;
  }
  const eventIndex = new Int32Array(count);
  const positions = new Float32Array(count * 3);
  const radii = new Float32Array(count);
  const appearAt = new Float32Array(count);
  let j = 0;
  for (let i = 0; i < events.length; i++) {
    if (!planHaloEligible(events[i])) continue;
    eventIndex[j] = i;
    radii[j] = usableErrorKm(events[i].quality.hErrM);
    appearAt[j] = candidates.appearAt[i];
    for (let axis = 0; axis < 3; axis++) positions[j * 3 + axis] = candidates.positions[i * 3 + axis];
    j++;
  }
  return { count, tierAWithoutHalo, eventIndex, positions, radii, appearAt };
}

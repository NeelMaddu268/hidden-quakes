// Filter transitions (lane doc → Strict): switching PUBLIC / ALL / STRICT eases tier opacities, the
// candidate layer, the public layer and the Tier A halos to their targets over `motion.state` with
// cubic-out. Time-based and allocation-free, so it's deterministic at any frame rate.

import { easeOutCubic, motion, strictFadeOpacity, tierStyle } from "@hq/visualization";
import type { EventFilter } from "../../state/demo";

/** Everything a filter controls, as plain numbers. */
export interface FilterLook {
  tierA: number;
  tierB: number;
  tierC: number;
  /** Candidate (amber) layer opacity: 0 under PUBLIC. */
  candidates: number;
  /** Public-catalog layer weight: full, except under STRICT where it steps back behind Tier A. */
  publicLayer: number;
  /** Tier A uncertainty halos: only under STRICT. */
  halos: number;
}

/** Public-catalog weight under STRICT: still there for reference, but Tier A carries the frame. */
export const STRICT_PUBLIC_WEIGHT = 0.4;

export const FILTER_LOOK: Readonly<Record<EventFilter, Readonly<FilterLook>>> = Object.freeze({
  public: Object.freeze({
    tierA: tierStyle.A.opacity,
    tierB: tierStyle.B.opacity,
    tierC: tierStyle.C.opacity,
    candidates: 0,
    publicLayer: 1,
    halos: 0,
  }),
  all: Object.freeze({
    tierA: tierStyle.A.opacity,
    tierB: tierStyle.B.opacity,
    tierC: tierStyle.C.opacity,
    candidates: 1,
    publicLayer: 1,
    halos: 0,
  }),
  strict: Object.freeze({
    tierA: tierStyle.A.opacity,
    tierB: strictFadeOpacity,
    tierC: strictFadeOpacity,
    candidates: 1,
    publicLayer: STRICT_PUBLIC_WEIGHT,
    halos: 1,
  }),
});

const KEYS = ["tierA", "tierB", "tierC", "candidates", "publicLayer", "halos"] as const;

export interface FilterFade {
  /** The current, eased values; read every frame. */
  readonly current: FilterLook;
  /** Advance toward `filter`'s look by `deltaS`. A new target restarts the ease from `current`. */
  step(filter: EventFilter, deltaS: number): void;
  /** Jump straight to `filter`'s look (start frame, reset). */
  snap(filter: EventFilter): void;
}

export function createFilterFade(initial: EventFilter, durationS: number = motion.state / 1000): FilterFade {
  const current: FilterLook = { ...FILTER_LOOK[initial] };
  const from: FilterLook = { ...current };
  let target: EventFilter = initial;
  let t = durationS;

  return {
    current,
    step(filter, deltaS) {
      if (filter !== target) {
        target = filter;
        t = 0;
        for (const k of KEYS) from[k] = current[k];
      }
      if (t >= durationS) return;
      t += deltaS > 0 ? deltaS : 0;
      const to = FILTER_LOOK[target];
      if (t >= durationS || durationS <= 0) {
        for (const k of KEYS) current[k] = to[k];
        t = durationS;
        return;
      }
      const k = easeOutCubic(t / durationS);
      for (const key of KEYS) current[key] = from[key] + (to[key] - from[key]) * k;
    },
    snap(filter) {
      target = filter;
      t = durationS;
      const to = FILTER_LOOK[filter];
      for (const k of KEYS) current[k] = to[k];
    },
  };
}

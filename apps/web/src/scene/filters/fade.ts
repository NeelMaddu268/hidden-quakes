// Filter transitions (lane doc → Strict): switching PUBLIC / ALL / STRICT eases tier opacities, the
// candidate layer, the public layer and the Tier A halos to their targets over `motion.state` with
// cubic-out. Time-based and allocation-free, so it's deterministic at any frame rate.

import { easeOutCubic, motion, strictFadeOpacity, tierStyle } from "@hq/visualization";
import type { DemoPhase, EventFilter } from "../../state/demo";
import { LOOK } from "../look";

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

/** Public-catalog weight under STRICT (scene/look.ts): still there for reference, Tier A leads. */
export const STRICT_PUBLIC_WEIGHT = LOOK.strictPublicWeight;

/**
 * The look actually applied. Before the reveal the start frame never changes: a STRICT (or ALL)
 * pressed early is remembered by the store and takes effect when the reveal starts.
 */
export function effectiveFilter(phase: DemoPhase, filter: EventFilter): EventFilter {
  return phase === "public" ? "public" : filter;
}

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

  const assign = (dst: FilterLook, src: Readonly<FilterLook>) => {
    for (let i = 0; i < KEYS.length; i++) dst[KEYS[i]] = src[KEYS[i]];
  };

  return {
    current,
    step(filter, deltaS) {
      if (filter !== target) {
        target = filter;
        t = 0;
        assign(from, current);
      }
      const to = FILTER_LOOK[target];
      if (durationS <= 0) {
        assign(current, to); // reduced motion: switch instantly
        return;
      }
      if (t >= durationS) return;
      t += deltaS > 0 ? deltaS : 0;
      if (t >= durationS) {
        assign(current, to);
        t = durationS;
        return;
      }
      const k = easeOutCubic(t / durationS);
      for (let i = 0; i < KEYS.length; i++) {
        const key = KEYS[i];
        current[key] = from[key] + (to[key] - from[key]) * k;
      }
    },
    snap(filter) {
      target = filter;
      t = durationS;
      assign(current, FILTER_LOOK[filter]);
    },
  };
}

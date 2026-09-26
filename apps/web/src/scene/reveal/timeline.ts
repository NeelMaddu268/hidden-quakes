// The ~7-second reveal (docs/lanes/H3-visualization.md → The reveal), as pure functions of the seconds
// elapsed since reveal(). Every layer derives its state from the same clock, so the choreography is
// identical on every run and at any frame rate.
//
//   0.0        reveal(); phase → "revealing"
//   0.0–1.2    terrain opacity 1 → 0.12 (contours stay)
//   0.4–2.0    camera dollies from the oblique surface view to the low side view
//   1.0–6.0    candidate events appear in revealOrder, each popping (scale 2 → 1, flash, settle);
//              revealProgress drives the shell's counter publicCatalogCount → candidateCount
//   6.0–7.0    settle; phase → "revealed"

import { easeOutCubic } from "@hq/visualization";

export const TIMELINE = Object.freeze({
  terrainFade: { startS: 0.0, endS: 1.2, to: 0.12 },
  dolly: { startS: 0.4, endS: 2.0 },
  events: {
    startS: 1.0,
    endS: 6.0,
    /**
     * Appearance curve: fraction of events shown = u^exponent for u = normalized time in the events
     * window. >1 starts slow (the first Tier A events pop one by one while structure forms) and
     * accelerates as the cloud fills in.
     */
    exponent: 1.6,
  },
  /** How long each event's pop (scale 2 → 1, brightness spike) lasts. */
  popS: 0.45,
  endS: 7.0,
});

const clamp01 = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x);

/** Normalized position in [a, b], clamped. */
function span(t: number, a: number, b: number): number {
  return clamp01((t - a) / (b - a));
}

/** Terrain surface opacity at `t` (1 before, 0.12 after the fade, contours unaffected). */
export function terrainOpacityAt(t: number): number {
  const { startS, endS, to } = TIMELINE.terrainFade;
  return 1 + (to - 1) * easeOutCubic(span(t, startS, endS));
}

/** Smooth start and stop for the camera move (ease-in-out cubic). */
export function easeInOutCubic(x: number): number {
  const c = clamp01(x);
  return c < 0.5 ? 4 * c * c * c : 1 - Math.pow(-2 * c + 2, 3) / 2;
}

/** Camera dolly blend at `t`: 0 = oblique start pose, 1 = side view. */
export function dollyAt(t: number): number {
  const { startS, endS } = TIMELINE.dolly;
  return easeInOutCubic(span(t, startS, endS));
}

/**
 * Fraction of candidate events visible at `t`, i.e. `revealProgress` for the shell's counter.
 * Exactly 0 up to 1.0 s and exactly 1 from 6.0 s.
 */
export function revealProgressAt(t: number): number {
  const { startS, endS, exponent } = TIMELINE.events;
  const u = span(t, startS, endS);
  if (u >= 1) return 1;
  return Math.pow(u, exponent);
}

/**
 * The time (s since reveal) at which the instance with normalized reveal slot `slot` ∈ [0, 1]
 * appears; the exact inverse of `revealProgressAt` on the events window. The event shader runs the
 * same formula per instance.
 */
export function appearTimeOf(slot: number): number {
  const { startS, endS, exponent } = TIMELINE.events;
  return startS + (endS - startS) * Math.pow(clamp01(slot), 1 / exponent);
}

/** Bloom multiplier: a soft swell while events pop, back to 1 by the end of the settle. */
export function bloomBoostAt(t: number): number {
  const { startS, endS } = TIMELINE.events;
  if (t <= startS || t >= TIMELINE.endS) return 1;
  const rise = span(t, startS, startS + 0.8);
  const fall = 1 - span(t, endS, TIMELINE.endS);
  return 1 + 0.35 * Math.min(rise, fall);
}

/** True once the whole choreography (including the settle) has played. */
export function isRevealDone(t: number): boolean {
  return t >= TIMELINE.endS;
}

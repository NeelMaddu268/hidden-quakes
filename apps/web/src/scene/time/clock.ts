// Time mode (WEB-06): the pure rules the scene, the picker, the depth section and the scrubber share,
// so what's drawn, what's clickable and what's counted can never disagree about "now".
//
// Semantics (docs/02 §6): `tNow` is epoch seconds; null shows every event. Time mode takes effect only
// after the reveal (docs/01: the scrubber "appears after the reveal"), so a stray T on the pre-reveal
// frame changes nothing. Entering it replays the window from its start (README: "T turns on the time
// scrubber, which replays the window"). An event is shown at tNow when its origin time t <= tNow.

import type { DemoPhase } from "../../state/demo";

/**
 * The scene-relative "now" (seconds since windowStart) that shows every event and lights none: far
 * beyond any window, still exact in float32 for the shaders.
 */
export const TIME_ALL = 1e9;

export interface TimeStateLike {
  phase: DemoPhase;
  timeMode: boolean;
  tNow: number | null;
}

/** Time mode is on, the reveal has finished, and the playhead is set. */
export function timeActive(s: TimeStateLike): boolean {
  return s.timeMode && s.phase === "revealed" && s.tNow !== null;
}

/**
 * "Now" relative to windowStart, in seconds, for the shaders and CPU gates: TIME_ALL when time mode
 * isn't active. Events with `t - windowStart <= timeNowRel` are shown.
 */
export function timeNowRel(s: TimeStateLike, windowStart: number): number {
  return timeActive(s) ? (s.tNow as number) - windowStart : TIME_ALL;
}

/** Whether an event at `timeRel` (seconds since windowStart) is shown at `nowRel`. */
export function shownAt(timeRel: number, nowRel: number): boolean {
  return timeRel <= nowRel;
}

/**
 * Recent-event emphasis in [0, 1]: 1 for an event at exactly now, falling linearly to 0 at
 * `glowWindowS` ago; 0 for events not yet shown and whenever time mode is off (nowRel = TIME_ALL).
 * Mirrors the event shader, so the tests pin the look the audience sees.
 */
export function recency(timeRel: number, nowRel: number, glowWindowS: number): number {
  const age = nowRel - timeRel;
  if (age < 0 || !(glowWindowS > 0)) return 0;
  const k = 1 - age / glowWindowS;
  return k > 0 ? k : 0;
}

/** Whether the scene should start the replay now: time mode just took effect and no playhead is set. */
export function shouldStartReplay(s: TimeStateLike): boolean {
  return s.timeMode && s.phase === "revealed" && s.tNow === null;
}

/**
 * One playback step: `deltaS` real seconds at `rate` data seconds per second, clamped to the window
 * end. A long frame (tab in the background) advances at most `maxDeltaS`, so playback never jumps.
 */
export function advancePlayback(
  tNow: number,
  deltaS: number,
  rate: number,
  windowEnd: number,
  maxDeltaS = 0.1,
): { tNow: number; done: boolean } {
  const dt = deltaS > 0 ? Math.min(deltaS, maxDeltaS) : 0;
  const next = tNow + dt * rate;
  return next >= windowEnd ? { tNow: windowEnd, done: true } : { tNow: next, done: false };
}

/**
 * What the time driver does this frame, or null for nothing: start the replay when time mode has just
 * taken effect (playhead at windowStart, playing), otherwise advance a playing replay and stop it at the
 * window end. Pure, so the whole playback rule is tested without a canvas.
 */
export function stepTime(
  s: TimeStateLike & { playing: boolean },
  deltaS: number,
  windowStart: number,
  windowEnd: number,
  rate: number,
): { tNow: number; playing: boolean } | null {
  if (shouldStartReplay(s)) return { tNow: windowStart, playing: true };
  if (!s.playing || !timeActive(s)) return null;
  const step = advancePlayback(s.tNow as number, deltaS, rate, windowEnd);
  return { tNow: step.tNow, playing: !step.done };
}

/** Number of `binS`-wide bins covering [windowStart, windowEnd] (at least 1). */
export function binCount(windowStart: number, windowEnd: number, binS: number): number {
  if (!(windowEnd > windowStart) || !(binS > 0)) return 1;
  return Math.max(1, Math.ceil((windowEnd - windowStart) / binS));
}

/**
 * Event counts per bin over the window. Bin k holds windowStart + k·binS <= t < windowStart +
 * (k + 1)·binS; the window end (and anything later) falls in the last bin, anything before the start in
 * the first, so every event is counted exactly once and the bins sum to the input length.
 */
export function histogram(times: ArrayLike<number>, windowStart: number, windowEnd: number, binS: number): Uint32Array {
  const n = binCount(windowStart, windowEnd, binS);
  const out = new Uint32Array(n);
  for (let i = 0; i < times.length; i++) {
    const k = Math.floor((times[i] - windowStart) / binS);
    out[k < 0 ? 0 : k >= n ? n - 1 : k]++;
  }
  return out;
}

/** Ascending copy of `times` for `countUpTo`. */
export function sortedTimes(times: ArrayLike<number>): Float64Array {
  return Float64Array.from(times).sort();
}

/** How many of the (ascending) `sorted` times are <= t: the events shown at t. Binary search. */
export function countUpTo(sorted: ArrayLike<number>, t: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * Epoch seconds → "HH:MM" in UTC (the scrubber's clock and axis labels). With `windowStart`, a UTC
 * midnight after the window's start reads "24:00" (the end of a day window, as in its windowLabel).
 */
export function fmtClockUtc(epochS: number, windowStart?: number): string {
  const d = new Date(Math.floor(epochS / 60) * 60_000);
  const pad = (v: number) => String(v).padStart(2, "0");
  const hh = d.getUTCHours();
  const mm = d.getUTCMinutes();
  if (hh === 0 && mm === 0 && windowStart !== undefined && epochS > windowStart) return "24:00";
  return `${pad(hh)}:${pad(mm)}`;
}

/** Axis ticks at whole-hour steps (1, 2, 3, 6 or 12 h) giving about `target` intervals over the window. */
export function hourTicks(windowStart: number, windowEnd: number, target = 4): number[] {
  const span = windowEnd - windowStart;
  if (!(span > 0)) return [windowStart];
  const steps = [1, 2, 3, 6, 12].map((h) => h * 3600);
  const step = steps.find((s) => span / s <= target) ?? 24 * 3600;
  const first = Math.ceil(windowStart / step) * step;
  const out: number[] = [];
  for (let t = first; t <= windowEnd + 1e-6; t += step) out.push(t);
  return out;
}

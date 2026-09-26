// The time scrubber's histogram strip (WEB-06): events per bin over the run window, recovered above the
// baseline and the public regional catalog below it, on one shared scale; bins after "now" are dimmed,
// and a playhead marks tNow. Pure Canvas2D over a minimal context, so the tests pin what's drawn.

import type { EventFilter } from "../../state/demo";
import { binCount, histogram, sortedTimes } from "./clock";

export interface StripModel {
  windowStart: number;
  windowEnd: number;
  binS: number;
  bins: number;
  /** Counts per bin: every candidate, Tier A only (STRICT), and the public regional catalog. */
  recoveredAll: Uint32Array;
  recoveredStrict: Uint32Array;
  publicBins: Uint32Array;
  /** The shared bar scale: the largest bin of either layer (so switching the filter never rescales). */
  maxBin: number;
  /** Ascending origin times (epoch s) for the "so far" counts. */
  sortedRecoveredAll: Float64Array;
  sortedRecoveredStrict: Float64Array;
  sortedPublic: Float64Array;
}

/** Built once per bundle from the provider's events and catalog. */
export function buildStripModel(
  events: readonly { t: number; tier: string }[],
  catalog: readonly { t: number }[],
  windowStart: number,
  windowEnd: number,
  binS: number,
): StripModel {
  const all = events.map((e) => e.t);
  const strict = events.filter((e) => e.tier === "A").map((e) => e.t);
  const pub = catalog.map((c) => c.t);
  const recoveredAll = histogram(all, windowStart, windowEnd, binS);
  const recoveredStrict = histogram(strict, windowStart, windowEnd, binS);
  const publicBins = histogram(pub, windowStart, windowEnd, binS);
  let maxBin = 0;
  for (let k = 0; k < recoveredAll.length; k++) maxBin = Math.max(maxBin, recoveredAll[k], publicBins[k]);
  return {
    windowStart,
    windowEnd,
    binS,
    bins: binCount(windowStart, windowEnd, binS),
    recoveredAll,
    recoveredStrict,
    publicBins,
    maxBin,
    sortedRecoveredAll: sortedTimes(all),
    sortedRecoveredStrict: sortedTimes(strict),
    sortedPublic: sortedTimes(pub),
  };
}

/** Which recovered set the strip and the counts follow: Tier A under STRICT, none under PUBLIC. */
export function recoveredSet(filter: EventFilter): "all" | "strict" | "none" {
  return filter === "strict" ? "strict" : filter === "public" ? "none" : "all";
}

export interface StripCtx {
  fillStyle: unknown;
  strokeStyle: unknown;
  globalAlpha: number;
  lineWidth: number;
  setTransform(a: number, b: number, c: number, d: number, e: number, f: number): void;
  clearRect(x: number, y: number, w: number, h: number): void;
  fillRect(x: number, y: number, w: number, h: number): void;
  beginPath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  stroke(): void;
}

export interface StripStyle {
  recovered: string;
  publicBar: string;
  baseline: string;
  playhead: string;
  dpr: number;
}

/** Opacity of bins after "now": still readable as what's coming, clearly not yet shown. */
export const FUTURE_ALPHA = 0.22;
/** Gap between neighbouring bars, CSS px (dropped when bars get narrower than 2 px). */
const BAR_GAP_PX = 1;

/** x (CSS px) of epoch time `t` on a strip `width` wide, clamped to the strip. */
export function stripX(model: Pick<StripModel, "windowStart" | "windowEnd">, t: number, width: number): number {
  const span = model.windowEnd - model.windowStart;
  if (!(span > 0)) return 0;
  const k = (t - model.windowStart) / span;
  return (k <= 0 ? 0 : k >= 1 ? 1 : k) * width;
}

/** Epoch time at x (CSS px) on a strip `width` wide, clamped to the window. */
export function stripTime(model: Pick<StripModel, "windowStart" | "windowEnd">, x: number, width: number): number {
  const k = width > 0 ? x / width : 0;
  return model.windowStart + (k <= 0 ? 0 : k >= 1 ? 1 : k) * (model.windowEnd - model.windowStart);
}

/**
 * Draws the strip for "now" = `tNow` (epoch s; null draws the whole window as shown). Recovered bars
 * grow up from the middle baseline, public bars down, both on `model.maxBin`. Allocates nothing.
 */
export function drawStrip(
  ctx: StripCtx,
  model: StripModel,
  tNow: number | null,
  filter: EventFilter,
  style: StripStyle,
  width: number,
  height: number,
): void {
  ctx.setTransform(style.dpr, 0, 0, style.dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const mid = Math.round(height / 2);
  const half = mid - 1;
  const bw = width / model.bins;
  const gap = bw >= 2 + BAR_GAP_PX ? BAR_GAP_PX : 0;
  const now = tNow ?? model.windowEnd;
  const set = recoveredSet(filter);
  const rec = set === "strict" ? model.recoveredStrict : model.recoveredAll;
  const scale = model.maxBin > 0 ? half / model.maxBin : 0;

  for (let k = 0; k < model.bins; k++) {
    const binStart = model.windowStart + k * model.binS;
    const alpha = binStart <= now ? 1 : FUTURE_ALPHA;
    const x = k * bw;
    const r = set === "none" ? 0 : rec[k] * scale;
    if (r > 0) {
      ctx.globalAlpha = alpha;
      ctx.fillStyle = style.recovered;
      ctx.fillRect(x, mid - r, bw - gap, r);
    }
    const p = model.publicBins[k] * scale;
    if (p > 0) {
      ctx.globalAlpha = alpha;
      ctx.fillStyle = style.publicBar;
      ctx.fillRect(x, mid + 1, bw - gap, p);
    }
  }

  ctx.globalAlpha = 1;
  ctx.fillStyle = style.baseline;
  ctx.fillRect(0, mid, width, 1);

  if (tNow !== null) {
    const x = Math.round(stripX(model, tNow, width)) + 0.5;
    ctx.strokeStyle = style.playhead;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
}

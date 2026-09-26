// Bundle records → per-instance typed arrays for the event InstancedMeshes. Built once per bundle,
// never per frame. Everything the shaders animate on (reveal order, tier, time) lives here as floats.

import { tierStyle, type TierToken } from "@hq/visualization";
import { writeEnuToScene } from "../coords";
import { appearTimeOf } from "../reveal/timeline";
import type { CatalogEvent, SeismicEvent } from "../types";

export const TIER_INDEX: Readonly<Record<TierToken, number>> = { A: 0, B: 1, C: 2 };

export interface EventInstances {
  count: number;
  /** Instance index → record id (picking, selection). */
  ids: string[];
  indexById: ReadonlyMap<string, number>;
  /** xyz scene coordinates (km), 3 floats per instance. */
  positions: Float32Array;
  /** 0 = A, 1 = B, 2 = C. Public-catalog instances are 0 (they are drawn at full weight). */
  tiers: Float32Array;
  /** Relative size multiplier from `tierStyle`. */
  scales: Float32Array;
  /**
   * Normalized reveal slot in [0, 1] from the exporter's `revealOrder` (0 = first to appear).
   * −1 means "always visible" (public-catalog events are on screen before the reveal).
   */
  revealAt: Float32Array;
  /**
   * Seconds after reveal() at which each instance appears: `appearTimeOf(revealAt)`, the exact inverse of
   * the counter clock. −1 for always-visible (public) instances.
   */
  appearAt: Float32Array;
  /** Seconds since `ProcessingRun.windowStart` (float32 keeps ~5 ms resolution over a day). */
  times: Float32Array;
}

/**
 * Reveal slots from `revealOrder`, which the exporter assigns (Tier A first, time-ordered within, then
 * B, then C). The browser never re-sorts by anything else: rank = position in `revealOrder`, ties
 * broken by bundle order, normalized to [0, 1]. For a proper 0..n−1 permutation this is exactly
 * revealOrder / (n − 1).
 */
export function revealSlots(revealOrder: readonly number[]): Float32Array {
  const n = revealOrder.length;
  const out = new Float32Array(n);
  if (n <= 1) return out;
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => revealOrder[a] - revealOrder[b] || a - b);
  for (let rank = 0; rank < n; rank++) out[idx[rank]] = rank / (n - 1);
  return out;
}

/** Problems with `revealOrder` that mean the exporter didn't assign it (e.g. H2's −1 placeholder). */
export function revealOrderIssues(revealOrder: readonly number[]): string[] {
  const issues: string[] = [];
  const n = revealOrder.length;
  const seen = new Set<number>();
  let negatives = 0;
  let outOfRange = 0;
  let duplicates = 0;
  for (const o of revealOrder) {
    if (!Number.isInteger(o) || o < 0) negatives++;
    else if (o >= n) outOfRange++;
    if (seen.has(o)) duplicates++;
    seen.add(o);
  }
  if (negatives) issues.push(`${negatives} event(s) have a negative or non-integer revealOrder`);
  if (outOfRange) issues.push(`${outOfRange} event(s) have revealOrder >= event count (${n})`);
  if (duplicates) issues.push(`${duplicates} duplicate revealOrder value(s)`);
  return issues;
}

function tierOf(tier: string, id: string): TierToken {
  if (tier === "A" || tier === "B" || tier === "C") return tier;
  throw new Error(`event ${id}: unknown tier ${JSON.stringify(tier)}`);
}

/** Pipeline candidate events (the amber layer). */
export function buildCandidateInstances(
  events: readonly SeismicEvent[],
  verticalExaggeration: number,
  windowStart: number,
): EventInstances {
  const count = events.length;
  const positions = new Float32Array(count * 3);
  const tiers = new Float32Array(count);
  const scales = new Float32Array(count);
  const times = new Float32Array(count);
  const ids: string[] = new Array(count);
  const indexById = new Map<string, number>();
  for (let i = 0; i < count; i++) {
    const ev = events[i];
    const tier = tierOf(ev.tier, ev.id);
    ids[i] = ev.id;
    if (indexById.has(ev.id)) throw new Error(`duplicate SeismicEvent id ${ev.id}`);
    indexById.set(ev.id, i);
    writeEnuToScene(ev.enu, verticalExaggeration, positions, i * 3);
    tiers[i] = TIER_INDEX[tier];
    scales[i] = tierStyle[tier].size;
    times[i] = ev.t - windowStart;
  }
  const revealAt = revealSlots(events.map((e) => e.revealOrder));
  const appearAt = new Float32Array(count);
  for (let i = 0; i < count; i++) appearAt[i] = appearTimeOf(revealAt[i]);
  return { count, ids, indexById, positions, tiers, scales, revealAt, appearAt, times };
}

/** Public regional catalog events (the cool-white layer), visible from the first frame. */
export function buildPublicInstances(
  catalog: readonly CatalogEvent[],
  verticalExaggeration: number,
  windowStart: number,
): EventInstances {
  const count = catalog.length;
  const positions = new Float32Array(count * 3);
  const tiers = new Float32Array(count); // all 0: full weight
  const scales = new Float32Array(count).fill(tierStyle.A.size);
  const revealAt = new Float32Array(count).fill(-1);
  const appearAt = new Float32Array(count).fill(-1);
  const times = new Float32Array(count);
  const ids: string[] = new Array(count);
  const indexById = new Map<string, number>();
  for (let i = 0; i < count; i++) {
    const ev = catalog[i];
    ids[i] = ev.id;
    if (indexById.has(ev.id)) throw new Error(`duplicate CatalogEvent id ${ev.id}`);
    indexById.set(ev.id, i);
    writeEnuToScene(ev.enu, verticalExaggeration, positions, i * 3);
    times[i] = ev.t - windowStart;
  }
  return { count, ids, indexById, positions, tiers, scales, revealAt, appearAt, times };
}

/**
 * Positions of instances at or above a tier (0 = A only, 1 = A and B), for camera framing. Falls back
 * to every instance when no event qualifies, so a run without Tier A/B still gets framed.
 */
export function framingPositions(inst: EventInstances, maxTier: number): Float32Array {
  let n = 0;
  for (let i = 0; i < inst.count; i++) if (inst.tiers[i] <= maxTier) n++;
  if (n === 0) return inst.positions;
  const out = new Float32Array(n * 3);
  let j = 0;
  for (let i = 0; i < inst.count; i++) {
    if (inst.tiers[i] > maxTier) continue;
    out[j++] = inst.positions[i * 3];
    out[j++] = inst.positions[i * 3 + 1];
    out[j++] = inst.positions[i * 3 + 2];
  }
  return out;
}

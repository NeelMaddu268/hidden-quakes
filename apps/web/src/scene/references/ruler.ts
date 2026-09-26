// Depth ruler and depth slices: 0–6 km below the site surface with 1 km ticks. Tick heights come from
// depthKmToSceneY (depth is measured from SceneMeta.refSurfaceElevM, docs/01), and the ruler's title is
// SceneMeta.depthLabel verbatim, never a string of our own.

import type { SceneBounds } from "../camera/bounds";
import { depthKmToSceneY, METERS_PER_UNIT } from "../coords";
import type { EnuBoundsM } from "../terrain/meta";
import type { SceneMeta } from "../types";

export const RULER_MAX_DEPTH_KM = 6;
export const RULER_TICK_KM = 1;
/** Tick length (km, scene units), drawn outward (west) from the spine. */
export const RULER_TICK_LENGTH_KM = 0.35;
/** Gap between the framed data and the ruler, as a fraction of the framing radius. */
export const RULER_GAP_FRACTION = 0.12;

export type Vec3 = [number, number, number];

export interface RulerTick {
  depthKm: number;
  y: number;
  label: string;
  /** Where the tick label is pinned (the tick's outer end). */
  labelAt: Vec3;
}

export interface RulerLayout {
  /** SceneMeta.depthLabel, exactly. */
  title: string;
  /** Where the title is pinned (top of the spine). */
  titleAt: Vec3;
  ticks: RulerTick[];
  /** Line segments (pairs of points): the spine, then one per tick. */
  segments: Vec3[];
}

/** Depths of the ticks, 0 … max inclusive. */
export function rulerDepthsKm(maxDepthKm: number = RULER_MAX_DEPTH_KM, stepKm: number = RULER_TICK_KM): number[] {
  if (!(stepKm > 0) || !(maxDepthKm >= 0)) throw new Error(`bad ruler range 0..${maxDepthKm} step ${stepKm}`);
  const n = Math.round(maxDepthKm / stepKm);
  return Array.from({ length: n + 1 }, (_, i) => i * stepKm);
}

/** Tick label for a depth (km): "0 km", "1 km", … Numbers come from the loop, never from copy. */
export function tickLabel(depthKm: number): string {
  return `${Number(depthKm.toFixed(2))} km`;
}

/**
 * Where the ruler stands: just west of the framed data, level with its centre north–south. In the side
 * view (from the south) that's the left edge; in plan view the left edge; in the oblique view the far
 * left. Keeping it in the framed box's middle plane keeps its perspective size close to the data's.
 */
export function rulerAnchor(bounds: SceneBounds): { x: number; z: number } {
  return { x: bounds.min[0] - RULER_GAP_FRACTION * bounds.radius, z: bounds.center[2] };
}

export function rulerLayout(
  scene: Pick<SceneMeta, "depthLabel" | "originElevM" | "refSurfaceElevM" | "verticalExaggeration">,
  anchor: { x: number; z: number },
  maxDepthKm: number = RULER_MAX_DEPTH_KM,
  stepKm: number = RULER_TICK_KM,
): RulerLayout {
  const { x, z } = anchor;
  const depths = rulerDepthsKm(maxDepthKm, stepKm);
  const ticks: RulerTick[] = depths.map((depthKm) => {
    const y = depthKmToSceneY(depthKm, scene);
    return { depthKm, y, label: tickLabel(depthKm), labelAt: [x - RULER_TICK_LENGTH_KM, y, z] };
  });
  const top = ticks[0].y;
  const bottom = ticks[ticks.length - 1].y;
  const segments: Vec3[] = [
    [x, top, z],
    [x, bottom, z],
  ];
  for (const t of ticks) segments.push([x, t.y, z], [x - RULER_TICK_LENGTH_KM, t.y, z]);
  return { title: scene.depthLabel, titleAt: [x, top, z], ticks, segments };
}

/**
 * Which labels to show, top to bottom, given their on-screen anchors (px). Labels are boxes about
 * `boxW` × `boxH`: two overlap when |dx| < boxW and |dy| < boxH (e.g. in plan view, where the ruler is
 * seen end-on). Greedy from the top, keeping both ends: the first (0 km) label always shows, and the
 * last (deepest) one shows too, displacing the nearest intermediate label if it has to. Writes into
 * `out` (1 = visible) so the per-frame caller doesn't allocate.
 */
function labelsClear(xs: ArrayLike<number>, ys: ArrayLike<number>, w: number, h: number, i: number, j: number): boolean {
  return Math.abs(xs[i] - xs[j]) >= w || Math.abs(ys[i] - ys[j]) >= h;
}

export function declutterLabels(
  screenX: ArrayLike<number>,
  screenY: ArrayLike<number>,
  boxW: number,
  boxH: number,
  out: Uint8Array,
): Uint8Array {
  const n = screenY.length;
  let last = -1;
  for (let i = 0; i < n; i++) {
    out[i] = last < 0 || labelsClear(screenX, screenY, boxW, boxH, i, last) ? 1 : 0;
    if (out[i]) last = i;
  }
  const end = n - 1;
  if (n >= 2 && !out[end]) {
    // Make room for the deepest label by dropping intermediate labels from the bottom up.
    while (last > 0 && !labelsClear(screenX, screenY, boxW, boxH, end, last)) {
      out[last] = 0;
      do last--;
      while (last > 0 && !out[last]);
    }
    out[end] = labelsClear(screenX, screenY, boxW, boxH, end, last) ? 1 : 0;
  }
  return out;
}

/**
 * Where along the spine (t = 0 at the surface, 1 at the bottom) the ruler's title should sit so it
 * stays on screen: 0 while the top of the spine is below `ndcTop`, otherwise the point where the spine
 * enters the viewport. Takes the clip-space y and w of both ends, because a 3D segment stays linear in
 * clip space, so the crossing is exact: (y0 + t·dy) / (w0 + t·dw) = ndcTop. Clamped to [0, 1].
 */
export function stickyTitleT(y0: number, w0: number, y1: number, w1: number, ndcTop: number): number {
  if (w0 > 0 && y0 / w0 <= ndcTop) return 0;
  const denom = y1 - y0 - ndcTop * (w1 - w0);
  if (denom === 0) return 0;
  const t = (ndcTop * w0 - y0) / denom;
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

export const SLICE_DEPTHS_KM: readonly number[] = rulerDepthsKm().filter((d) => d > 0);

/**
 * Faint horizontal square outlines, one per slice depth, over the surface extent (ENU metres).
 * Returned as line-segment pairs (4 edges per slice).
 */
export function sliceSegments(
  extent: EnuBoundsM,
  scene: Pick<SceneMeta, "originElevM" | "refSurfaceElevM" | "verticalExaggeration">,
  depthsKm: readonly number[] = SLICE_DEPTHS_KM,
): Vec3[] {
  const x0 = extent.eMin / METERS_PER_UNIT;
  const x1 = extent.eMax / METERS_PER_UNIT;
  const z0 = -extent.nMax / METERS_PER_UNIT; // north edge
  const z1 = -extent.nMin / METERS_PER_UNIT; // south edge
  const out: Vec3[] = [];
  for (const d of depthsKm) {
    const y = depthKmToSceneY(d, scene);
    out.push([x0, y, z0], [x1, y, z0]);
    out.push([x1, y, z0], [x1, y, z1]);
    out.push([x1, y, z1], [x0, y, z1]);
    out.push([x0, y, z1], [x0, y, z0]);
  }
  return out;
}

/**
 * The depth ruler's opacity from the terrain's (sceneFx.terrainOpacity, which the reveal fades 1 → 0.12
 * and reset() brings back): hidden on the pre-reveal frame (docs/00's 5-second test: only terrain, the
 * geothermal reference, PUBLIC and REVEAL), fully shown once the ground has faded, in step with it.
 */
export function rulerRevealOpacity(terrainOpacity: number, fadedTo: number): number {
  const span = 1 - fadedTo;
  if (!(span > 0)) return 1;
  const k = (1 - terrainOpacity) / span;
  return k <= 0 ? 0 : k >= 1 ? 1 : k;
}

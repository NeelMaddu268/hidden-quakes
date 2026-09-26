// Screen-space picking over instanced event glyphs (WEB-05). Events are drawn as camera-facing discs
// of a few pixels, so a ray-vs-sphere raycast would make them nearly unclickable. Instead every
// candidate instance is projected to the screen and the nearest one within a pixel threshold wins.
// Pure functions with no three.js import: the caller hands in the view-projection matrix.

/** Default click radius around an event's center, CSS px. */
export const PICK_THRESHOLD_PX = 12;
/** Two hits whose screen distances differ by less than this are a tie; the one nearer the camera wins. */
export const PICK_TIE_EPS_PX = 0.5;

/** A click is a pointer that went down and up without moving more than this (CSS px)... */
export const CLICK_MAX_MOVE_PX = 4;
/** ...within this long (ms). Anything else is an orbit drag and never selects. */
export const CLICK_MAX_MS = 300;

export interface PickQuery {
  /** Column-major 4×4 view-projection matrix (three.js `Matrix4.elements` of projection × view). */
  viewProj: ArrayLike<number>;
  /** Viewport size, CSS px. */
  width: number;
  height: number;
  /** Pointer position relative to the viewport's top-left corner, CSS px. */
  x: number;
  y: number;
  /** Click radius, CSS px. A glyph drawn larger than this is hit anywhere on its disc. */
  thresholdPx: number;
  /**
   * Pixels per scene unit at clip w = 1 (projection[1][1] × height / 2). With `radius`, widens the hit
   * area to the glyph as drawn when the camera is close. Omit to use `thresholdPx` alone.
   */
  focalPx?: number;
  /** World-space glyph radius of instance i (scene units). */
  radius?: (i: number) => number;
  /** CSS pixel cap matching the rendered glyph clamp (including the reveal pop). */
  maxRadiusPx?: number;
  /** Whether instance i is drawn right now. Hidden instances are never picked. */
  visible?: (i: number) => boolean;
  tieEpsPx?: number;
}

export interface PickHit {
  index: number;
  /** Screen distance from the pointer to the instance center, CSS px. */
  distPx: number;
  /** Clip-space w: distance along the view axis (smaller = nearer the camera). */
  depth: number;
}

/**
 * True when `a` should win over `b`: nearer on screen, ties (within `tieEpsPx`) broken by nearer the
 * camera, then by lower index so the result never depends on iteration order.
 */
export function betterHit(a: PickHit, b: PickHit | null, tieEpsPx: number = PICK_TIE_EPS_PX): boolean {
  if (b === null) return true;
  if (Math.abs(a.distPx - b.distPx) > tieEpsPx) return a.distPx < b.distPx;
  if (a.depth !== b.depth) return a.depth < b.depth;
  return a.index < b.index;
}

/**
 * The visible instance nearest the pointer on screen, within the click radius, or null. Points behind
 * the camera or outside the near/far planes are skipped.
 */
export function pickNearest(positions: ArrayLike<number>, q: PickQuery): PickHit | null {
  const m = q.viewProj;
  const tie = q.tieEpsPx ?? PICK_TIE_EPS_PX;
  const count = Math.floor(positions.length / 3);
  let best: PickHit | null = null;
  for (let i = 0; i < count; i++) {
    if (q.visible && !q.visible(i)) continue;
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (!(cw > 0)) continue; // behind the camera (perspective) or degenerate
    const cz = m[2] * x + m[6] * y + m[10] * z + m[14];
    const ndcZ = cz / cw;
    if (ndcZ < -1 || ndcZ > 1) continue; // clipped by the near or far plane
    const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    const sx = ((cx / cw + 1) / 2) * q.width;
    const sy = ((1 - cy / cw) / 2) * q.height;
    const dx = sx - q.x;
    const dy = sy - q.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    let reach = q.thresholdPx;
    if (q.radius && q.focalPx) reach = Math.max(reach, Math.min(q.maxRadiusPx ?? Infinity, (q.radius(i) * q.focalPx) / cw));
    if (dist > reach) continue;
    const hit: PickHit = { index: i, distPx: dist, depth: cw };
    if (betterHit(hit, best, tie)) best = hit;
  }
  return best;
}

export interface PointerSample {
  x: number;
  y: number;
  /** ms, e.g. `PointerEvent.timeStamp`. */
  t: number;
}

/** A click, not an orbit drag: little movement, short press. */
export function isClick(
  down: PointerSample,
  up: PointerSample,
  maxMovePx: number = CLICK_MAX_MOVE_PX,
  maxMs: number = CLICK_MAX_MS,
): boolean {
  const dx = up.x - down.x;
  const dy = up.y - down.y;
  return dx * dx + dy * dy < maxMovePx * maxMovePx && up.t - down.t < maxMs;
}

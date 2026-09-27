// Screen-space placement for the geo-feature labels (drei <Html>, CSS px). Each label sits just right of
// its anchor (wellhead, facility, boundary vertex) by default. When that box would cover another
// label, a fixed label (the depth ruler's title and tick labels), another feature's anchor, the plan
// view's depth-section panel, or run off the viewport, the label takes the next free slot: the
// anchor's left side, then a line up or down (with a thin leader back to its anchor). If nothing is
// free it takes the slot with the least overlap; a well or facility label is never hidden, because the
// features are the scene's geothermal reference. Among the slots with no overlap, the one that the
// feature lines (boundaries, well tracks: `lines`) and its own leader cross least wins, weighed against
// how far it moves (a dashed line through a label's text was the post-reveal side view's clutter).
// Pure and allocation-free: the per-frame caller runs it every frame.

/** Horizontal gap between the anchor and the label's near edge (the old `translate(10px, -50%)`). */
export const LABEL_GAP_PX = 10;
/** Vertical gap between stacked label lines. */
export const LABEL_LINE_GAP_PX = 3;
/** Half-size of the square kept clear around every other feature's anchor (its marker or wellhead). */
export const ANCHOR_CLEAR_PX = 6;
/**
 * Clear space kept between a label and any other label or fixed box (CSS px), so neighbouring texts
 * never butt into one run (e.g. "… · approximate" against the ruler's "0 km").
 */
export const LABEL_CLEARANCE_PX = 6;

export interface LabelSlot {
  /** 1 = right of the anchor, −1 = left. */
  side: 1 | -1;
  /** Whole label lines up (−) or down (+) from the anchor's line. */
  lines: number;
}

/** Tried in this order: same line (right, then left), then one to five lines away. */
export const LABEL_SLOTS: readonly LabelSlot[] = Object.freeze(
  [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5].flatMap((lines) => [
    { side: 1 as const, lines },
    { side: -1 as const, lines },
  ]),
);

/** A flat list of rects, [left, top, width, height] per rect, filled in place each frame. */
export interface RectList {
  rects: Float32Array;
  count: number;
}

export function makeRectList(capacity: number): RectList {
  return { rects: new Float32Array(Math.max(0, capacity) * 4), count: 0 };
}

/** Appends a rect; silently drops it when the list is full (capacity is sized by the caller). */
export function pushRect(list: RectList, left: number, top: number, width: number, height: number): void {
  const i = list.count * 4;
  if (i + 4 > list.rects.length) return;
  list.rects[i] = left;
  list.rects[i + 1] = top;
  list.rects[i + 2] = width;
  list.rects[i + 3] = height;
  list.count++;
}

export function clearRects(list: RectList): void {
  list.count = 0;
}

/** Appends every rect of `from` to `to` (up to `to`'s capacity). */
export function appendRects(to: RectList, from: RectList): void {
  const r = from.rects;
  for (let k = 0; k < from.count; k++) pushRect(to, r[k * 4], r[k * 4 + 1], r[k * 4 + 2], r[k * 4 + 3]);
}

/** Screen segments (CSS px), [x0, y0, x1, y1] per segment, filled in place each frame. */
export interface SegmentList {
  xy: Float32Array;
  count: number;
}

export function makeSegmentList(capacity: number): SegmentList {
  return { xy: new Float32Array(Math.max(0, capacity) * 4), count: 0 };
}

/** Appends a segment; silently drops it when the list is full (capacity is sized by the caller). */
export function pushSegment(list: SegmentList, x0: number, y0: number, x1: number, y1: number): void {
  const i = list.count * 4;
  if (i + 4 > list.xy.length) return;
  list.xy[i] = x0;
  list.xy[i + 1] = y0;
  list.xy[i + 2] = x1;
  list.xy[i + 3] = y1;
  list.count++;
}

export function clearSegments(list: SegmentList): void {
  list.count = 0;
}

/**
 * Length (px) of the segment (x0, y0)–(x1, y1) inside the rect [l, l + w] × [t, t + h]: Liang–Barsky
 * clipping, inlined (no closures: it runs per slot per segment each frame). A bounding-box test first.
 */
export function segmentLengthInRect(
  x0: number, y0: number, x1: number, y1: number,
  l: number, t: number, w: number, h: number,
): number {
  const r = l + w;
  const b = t + h;
  if ((x0 < l && x1 < l) || (x0 > r && x1 > r) || (y0 < t && y1 < t) || (y0 > b && y1 > b)) return 0;
  const dx = x1 - x0;
  const dy = y1 - y0;
  let u0 = 0;
  let u1 = 1;
  // Four edges as (p, q) pairs: −dx·u ≤ x0 − l, dx·u ≤ r − x0, −dy·u ≤ y0 − t, dy·u ≤ b − y0.
  for (let e = 0; e < 4; e++) {
    const pe = e === 0 ? -dx : e === 1 ? dx : e === 2 ? -dy : dy;
    const qe = e === 0 ? x0 - l : e === 1 ? r - x0 : e === 2 ? y0 - t : b - y0;
    if (pe === 0) {
      if (qe < 0) return 0;
      continue;
    }
    const ratio = qe / pe;
    if (pe < 0) {
      if (ratio > u1) return 0;
      if (ratio > u0) u0 = ratio;
    } else {
      if (ratio < u0) return 0;
      if (ratio < u1) u1 = ratio;
    }
  }
  return u1 > u0 ? (u1 - u0) * Math.hypot(dx, dy) : 0;
}

/**
 * World-space segments ([x0, y0, z0, x1, y1, z1] each) of polylines, each thinned to at most
 * `maxPerLine` segments (a uniform vertex stride, endpoints kept): label placement only needs the
 * lines' course on screen, and a well trajectory has hundreds of vertices.
 */
export function thinnedSegments(lines: readonly (readonly (readonly number[])[])[], maxPerLine: number): Float32Array {
  const out: number[] = [];
  for (const pts of lines) {
    if (pts.length < 2) continue;
    const stride = Math.max(1, Math.ceil((pts.length - 1) / maxPerLine));
    let prev = pts[0]!;
    for (let k = stride; ; k += stride) {
      const cur = pts[Math.min(k, pts.length - 1)]!;
      out.push(prev[0]!, prev[1]!, prev[2]!, cur[0]!, cur[1]!, cur[2]!);
      prev = cur;
      if (k >= pts.length - 1) break;
    }
  }
  return new Float32Array(out);
}

/** Segments per feature line used for label placement (thinnedSegments). */
export const PLACEMENT_SEGMENTS_PER_LINE = 48;

/**
 * Soft costs among overlap-free slots (px-equivalents): each px of feature line inside a label's box
 * (grown by LINE_CLEARANCE_PX) costs LINE_CROSS_COST; each px of its leader inside another text costs
 * LEADER_CROSS_COST; each line moved off the anchor's line costs LINE_SHIFT_COST and the left side
 * SIDE_COST. So a label steps a line away from a dashed line running along its text, but not from a
 * line merely clipping a corner.
 */
export const LINE_CLEARANCE_PX = 2;
export const LINE_CROSS_COST = 1;
export const LEADER_CROSS_COST = 2;
export const LINE_SHIFT_COST = 24;
export const SIDE_COST = 6;

/** Offset (CSS px) from the anchor to the label box's top-left corner for a slot. */
export function slotOffsetX(slot: LabelSlot, width: number): number {
  return slot.side > 0 ? LABEL_GAP_PX : -LABEL_GAP_PX - width;
}

export function slotOffsetY(slot: LabelSlot, height: number): number {
  return -height / 2 + slot.lines * (height + LABEL_LINE_GAP_PX);
}

function overlapArea(
  l: number, t: number, w: number, h: number,
  l2: number, t2: number, w2: number, h2: number,
): number {
  const x = Math.min(l + w, l2 + w2) - Math.max(l, l2);
  if (x <= 0) return 0;
  const y = Math.min(t + h, t2 + h2) - Math.max(t, t2);
  return y <= 0 ? 0 : x * y;
}

/** Per-frame inputs and outputs, preallocated for `n` labels (see makeLabelPlacement). */
export interface LabelPlacement {
  n: number;
  /** Anchor on screen (CSS px), in placement (priority) order. */
  ax: Float32Array;
  ay: Float32Array;
  /** Measured label size (CSS px); 0 = not measured yet (the label is skipped and keeps its CSS default). */
  w: Float32Array;
  h: Float32Array;
  /** 1 = the anchor is in front of the camera (drei hides the label otherwise). */
  active: Uint8Array;
  /** Slot index into LABEL_SLOTS chosen last frame and this frame; −1 = none (inactive, or dropped). */
  prev: Int8Array;
  slot: Int8Array;
  /**
   * 1 = the label may be dropped when no slot is free (boundary labels: their anchor is an arbitrary
   * vertex); 0 = always placed, least-overlap if need be (wells and facilities: the geothermal reference).
   */
  droppable: Uint8Array;
  /** Scratch: the labels placed so far this frame. */
  placed: RectList;
}

export function makeLabelPlacement(n: number): LabelPlacement {
  return {
    n,
    ax: new Float32Array(n),
    ay: new Float32Array(n),
    w: new Float32Array(n),
    h: new Float32Array(n),
    active: new Uint8Array(n),
    prev: new Int8Array(n).fill(-1),
    slot: new Int8Array(n).fill(-1),
    droppable: new Uint8Array(n),
    placed: makeRectList(n),
  };
}

function slotCost(
  p: LabelPlacement,
  i: number,
  s: number,
  fixed: RectList,
  viewportW: number,
  viewportH: number,
): number {
  const slot = LABEL_SLOTS[s];
  const w = p.w[i];
  const h = p.h[i];
  const l = p.ax[i] + slotOffsetX(slot, w);
  const t = p.ay[i] + slotOffsetY(slot, h);
  // Area outside the viewport counts like an overlap.
  let cost = w * h - overlapArea(l, t, w, h, 0, 0, viewportW, viewportH);
  // Against other texts, the label's box grows by the clearance on every side.
  const m = LABEL_CLEARANCE_PX;
  const fr = fixed.rects;
  for (let k = 0; k < fixed.count; k++) {
    cost += overlapArea(l - m, t - m, w + 2 * m, h + 2 * m, fr[k * 4], fr[k * 4 + 1], fr[k * 4 + 2], fr[k * 4 + 3]);
  }
  const pr = p.placed.rects;
  for (let k = 0; k < p.placed.count; k++) {
    cost += overlapArea(l - m, t - m, w + 2 * m, h + 2 * m, pr[k * 4], pr[k * 4 + 1], pr[k * 4 + 2], pr[k * 4 + 3]);
  }
  const c = ANCHOR_CLEAR_PX;
  for (let j = 0; j < p.n; j++) {
    if (j === i || !p.active[j]) continue;
    cost += overlapArea(l, t, w, h, p.ax[j] - c, p.ay[j] - c, 2 * c, 2 * c);
  }
  return cost;
}

/** Clipped length (px) of every feature line inside the box. */
function linesInBox(lines: SegmentList | undefined, l: number, t: number, w: number, h: number): number {
  if (!lines) return 0;
  let len = 0;
  const xy = lines.xy;
  for (let k = 0; k < lines.count; k++) {
    len += segmentLengthInRect(xy[k * 4], xy[k * 4 + 1], xy[k * 4 + 2], xy[k * 4 + 3], l, t, w, h);
  }
  return len;
}

/** Cost of what crosses label `i` in slot `s`: feature lines through its text, its leader through others. */
function slotCrossings(p: LabelPlacement, i: number, s: number, fixed: RectList, lines: SegmentList | undefined): number {
  const slot = LABEL_SLOTS[s];
  const w = p.w[i];
  const h = p.h[i];
  const l = p.ax[i] + slotOffsetX(slot, w);
  const t = p.ay[i] + slotOffsetY(slot, h);
  const c = LINE_CLEARANCE_PX;
  let cost = LINE_CROSS_COST * linesInBox(lines, l - c, t - c, w + 2 * c, h + 2 * c);
  if (slot.lines !== 0) {
    // The leader runs from the anchor to the label's near edge at mid-height.
    const x0 = p.ax[i];
    const y0 = p.ay[i];
    const x1 = slot.side > 0 ? l : l + w;
    const y1 = t + h / 2;
    let through = 0;
    const fr = fixed.rects;
    for (let k = 0; k < fixed.count; k++) {
      through += segmentLengthInRect(x0, y0, x1, y1, fr[k * 4], fr[k * 4 + 1], fr[k * 4 + 2], fr[k * 4 + 3]);
    }
    const pr = p.placed.rects;
    for (let k = 0; k < p.placed.count; k++) {
      through += segmentLengthInRect(x0, y0, x1, y1, pr[k * 4], pr[k * 4 + 1], pr[k * 4 + 2], pr[k * 4 + 3]);
    }
    cost += LEADER_CROSS_COST * through;
  }
  return cost;
}

/** What moving off the default slot costs: lines away from the anchor's line, and the left side. */
function slotMoveCost(s: number): number {
  const slot = LABEL_SLOTS[s];
  return LINE_SHIFT_COST * Math.abs(slot.lines) + (slot.side < 0 ? SIDE_COST : 0);
}

/**
 * Chooses a slot for every active, measured label, in order (earlier labels have priority), writing
 * `p.slot` (−1 for a droppable label with no free slot) and then copying it to `p.prev`. Per label: the
 * default slot when nothing overlaps it and no feature line crosses it; otherwise, with `holdPrevious`
 * (the camera is moving), last frame's slot when that's still as clean, so labels don't hop between
 * slots mid-move; otherwise the slot with the least overlap and, among equals, the least soft cost
 * (crossings plus the move; earliest on ties). A droppable label with no overlap-free slot is dropped; a line
 * crossing alone never drops a label. Without `holdPrevious` the result depends only on this frame, so
 * a still camera always gets the same layout whatever path led there. `lines`: the feature lines on
 * screen (optional; without them only overlaps and the move count).
 */
export function placeLabels(
  p: LabelPlacement,
  fixed: RectList,
  viewportW: number,
  viewportH: number,
  holdPrevious = false,
  lines?: SegmentList,
): Int8Array {
  p.placed.count = 0;
  for (let i = 0; i < p.n; i++) {
    p.slot[i] = -1;
    if (!p.active[i] || !(p.w[i] > 0) || !(p.h[i] > 0)) continue;
    // Clean: nothing overlaps it and nothing crosses it.
    const clean = (s: number) =>
      slotCost(p, i, s, fixed, viewportW, viewportH) === 0 && slotCrossings(p, i, s, fixed, lines) === 0;
    let chosen = -1;
    if (clean(0)) chosen = 0;
    else if (holdPrevious && p.prev[i] > 0 && clean(p.prev[i])) chosen = p.prev[i];
    else {
      let bestHard = Infinity;
      let bestSoft = Infinity;
      for (let s = 0; s < LABEL_SLOTS.length; s++) {
        const hard = slotCost(p, i, s, fixed, viewportW, viewportH);
        if (hard > bestHard) continue;
        const soft = slotCrossings(p, i, s, fixed, lines) + slotMoveCost(s);
        if (hard < bestHard || soft < bestSoft) {
          bestHard = hard;
          bestSoft = soft;
          chosen = s;
        }
      }
      if (bestHard > 0 && p.droppable[i]) {
        p.slot[i] = -1; // no clean spot: a low-priority label steps aside rather than collide
        continue;
      }
    }
    p.slot[i] = chosen;
    const slot = LABEL_SLOTS[chosen];
    pushRect(p.placed, p.ax[i] + slotOffsetX(slot, p.w[i]), p.ay[i] + slotOffsetY(slot, p.h[i]), p.w[i], p.h[i]);
  }
  p.prev.set(p.slot);
  return p.slot;
}

/**
 * The leader from the anchor to a displaced label's near edge, at the label's mid-height: length and
 * angle (radians, screen y down) for a 1 px CSS line rotated about its start. Zero length on the
 * anchor's own line, where the label already sits beside its anchor.
 */
export function leaderOf(slot: LabelSlot, height: number): { length: number; angle: number } {
  if (slot.lines === 0) return { length: 0, angle: 0 };
  const x = slot.side * LABEL_GAP_PX;
  const y = slot.lines * (height + LABEL_LINE_GAP_PX);
  return { length: Math.hypot(x, y), angle: Math.atan2(y, x) };
}

/** Storage for cameraMoved: world matrix, projection matrix, viewport width and height. */
export function makeCameraTrack(): Float64Array {
  return new Float64Array(34).fill(Number.NaN);
}

/**
 * Copies this frame's camera (world and projection matrices, viewport size in CSS px) into `last`;
 * true when anything changed since the previous call (always true on the first). No allocation.
 */
export function cameraMoved(
  last: Float64Array,
  world: ArrayLike<number>,
  projection: ArrayLike<number>,
  viewportW: number,
  viewportH: number,
): boolean {
  let moved = false;
  for (let k = 0; k < 34; k++) {
    const x = k < 16 ? world[k] : k < 32 ? projection[k - 16] : k === 32 ? viewportW : viewportH;
    if (last[k] !== x) {
      last[k] = x;
      moved = true;
    }
  }
  return moved;
}

/** Placement priority by feature kind: point-like features first, boundaries (arbitrary vertex) last. */
export function labelPriority(kind: string): number {
  return kind === "facility" ? 0 : kind === "well" ? 1 : 2;
}

/**
 * Depth-ruler label boxes as fixed obstacles (CSS px), mirroring DepthRuler's CSS transforms: the title
 * sits above its anchor, left-aligned with a small inset; tick labels sit left of the tick's end,
 * vertically centred.
 */
export const RULER_TITLE_INSET_PX = 4;
export const RULER_TITLE_RISE_PX = 10;
export const RULER_TICK_GAP_PX = 5;

export function pushRulerTitleRect(list: RectList, x: number, y: number, w: number, h: number): void {
  pushRect(list, x - RULER_TITLE_INSET_PX, y - RULER_TITLE_RISE_PX - h, w, h);
}

export function pushRulerTickRect(list: RectList, x: number, y: number, w: number, h: number): void {
  pushRect(list, x - RULER_TICK_GAP_PX - w, y - h / 2, w, h);
}

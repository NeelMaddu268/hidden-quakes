// Screen-space placement for the geo-feature labels (drei <Html>, CSS px). Each label sits just right of
// its anchor (wellhead, facility, boundary vertex) by default. When that box would cover another
// label, a fixed label (the depth ruler's title and tick labels), another feature's anchor, the plan
// view's depth-section panel, or run off the viewport, the label takes the next free slot: the
// anchor's left side, then a line up or down (with a thin leader back to its anchor). If nothing is
// free it takes the slot with the least overlap; a label is never hidden, because the features are
// the scene's geothermal reference. Pure and allocation-free: the per-frame caller runs it every frame.

/** Horizontal gap between the anchor and the label's near edge (the old `translate(10px, -50%)`). */
export const LABEL_GAP_PX = 10;
/** Vertical gap between stacked label lines. */
export const LABEL_LINE_GAP_PX = 3;
/** Half-size of the square kept clear around every other feature's anchor (its marker or wellhead). */
export const ANCHOR_CLEAR_PX = 6;

export interface LabelSlot {
  /** 1 = right of the anchor, −1 = left. */
  side: 1 | -1;
  /** Whole label lines up (−) or down (+) from the anchor's line. */
  lines: number;
}

/** Tried in this order: same line (right, then left), then one, two and three lines away. */
export const LABEL_SLOTS: readonly LabelSlot[] = Object.freeze(
  [0, -1, 1, -2, 2, -3, 3].flatMap((lines) => [
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
  /** Slot index into LABEL_SLOTS chosen last frame and this frame; −1 = none. */
  prev: Int8Array;
  slot: Int8Array;
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
  const fr = fixed.rects;
  for (let k = 0; k < fixed.count; k++) {
    cost += overlapArea(l, t, w, h, fr[k * 4], fr[k * 4 + 1], fr[k * 4 + 2], fr[k * 4 + 3]);
  }
  const pr = p.placed.rects;
  for (let k = 0; k < p.placed.count; k++) {
    cost += overlapArea(l, t, w, h, pr[k * 4], pr[k * 4 + 1], pr[k * 4 + 2], pr[k * 4 + 3]);
  }
  const c = ANCHOR_CLEAR_PX;
  for (let j = 0; j < p.n; j++) {
    if (j === i || !p.active[j]) continue;
    cost += overlapArea(l, t, w, h, p.ax[j] - c, p.ay[j] - c, 2 * c, 2 * c);
  }
  return cost;
}

/**
 * Chooses a slot for every active, measured label, in order (earlier labels have priority), writing
 * `p.slot` and then copying it to `p.prev`. Per label: the default slot when it's free; otherwise, with
 * `holdPrevious` (the camera is moving), last frame's slot when that's still free, so labels don't hop
 * between slots mid-move; otherwise the first free slot; otherwise the least-overlapping one (earliest
 * on ties). Without `holdPrevious` the result depends only on this frame, so a still camera always
 * gets the same layout whatever path led there.
 */
export function placeLabels(
  p: LabelPlacement,
  fixed: RectList,
  viewportW: number,
  viewportH: number,
  holdPrevious = false,
): Int8Array {
  p.placed.count = 0;
  for (let i = 0; i < p.n; i++) {
    p.slot[i] = -1;
    if (!p.active[i] || !(p.w[i] > 0) || !(p.h[i] > 0)) continue;
    let chosen = -1;
    if (slotCost(p, i, 0, fixed, viewportW, viewportH) === 0) chosen = 0;
    else if (holdPrevious && p.prev[i] > 0 && slotCost(p, i, p.prev[i], fixed, viewportW, viewportH) === 0) {
      chosen = p.prev[i];
    }
    else {
      let best = Infinity;
      for (let s = 0; s < LABEL_SLOTS.length; s++) {
        const cost = slotCost(p, i, s, fixed, viewportW, viewportH);
        if (cost < best) {
          best = cost;
          chosen = s;
          if (cost === 0) break;
        }
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

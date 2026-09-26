import { describe, expect, it } from "vitest";
import {
  ANCHOR_CLEAR_PX,
  appendRects,
  cameraMoved,
  clearRects,
  LABEL_GAP_PX,
  LABEL_LINE_GAP_PX,
  LABEL_SLOTS,
  labelPriority,
  leaderOf,
  makeCameraTrack,
  makeLabelPlacement,
  makeRectList,
  placeLabels,
  pushRect,
  pushRulerTickRect,
  pushRulerTitleRect,
  RULER_TICK_GAP_PX,
  RULER_TITLE_INSET_PX,
  RULER_TITLE_RISE_PX,
  slotOffsetX,
  slotOffsetY,
  type LabelPlacement,
  type RectList,
} from "./labelPlacement";

type Rect = { l: number; t: number; w: number; h: number };
const hit = (a: Rect, b: Rect) => a.l < b.l + b.w && b.l < a.l + a.w && a.t < b.t + b.h && b.t < a.t + a.h;
const W = 1280;
const H = 720;

interface Label {
  x: number;
  y: number;
  w?: number;
  h?: number;
  active?: boolean;
}

function setup(labels: Label[]): LabelPlacement {
  const p = makeLabelPlacement(labels.length);
  labels.forEach((l, i) => {
    p.ax[i] = l.x;
    p.ay[i] = l.y;
    p.w[i] = l.w ?? 150;
    p.h[i] = l.h ?? 14;
    p.active[i] = l.active === false ? 0 : 1;
  });
  return p;
}

function rectsOf(p: LabelPlacement): (Rect | null)[] {
  return Array.from({ length: p.n }, (_, i) => {
    const s = p.slot[i];
    if (s < 0) return null;
    const slot = LABEL_SLOTS[s];
    return { l: p.ax[i] + slotOffsetX(slot, p.w[i]), t: p.ay[i] + slotOffsetY(slot, p.h[i]), w: p.w[i], h: p.h[i] };
  });
}

function fixedOf(list: RectList): Rect[] {
  return Array.from({ length: list.count }, (_, k) => ({
    l: list.rects[k * 4],
    t: list.rects[k * 4 + 1],
    w: list.rects[k * 4 + 2],
    h: list.rects[k * 4 + 3],
  }));
}

/** Every placed label is on screen and clear of other labels, the fixed rects and other anchors. */
function expectClear(p: LabelPlacement, fixed: RectList) {
  const rects = rectsOf(p);
  const c = ANCHOR_CLEAR_PX;
  rects.forEach((a, i) => {
    if (!a) return;
    expect(a.l >= 0 && a.t >= 0 && a.l + a.w <= W && a.t + a.h <= H, `label ${i} on screen`).toBe(true);
    rects.forEach((b, j) => {
      if (b && j !== i) expect(hit(a, b), `labels ${i} and ${j} overlap`).toBe(false);
      if (j !== i && p.active[j]) {
        expect(hit(a, { l: p.ax[j] - c, t: p.ay[j] - c, w: 2 * c, h: 2 * c }), `label ${i} covers anchor ${j}`).toBe(false);
      }
    });
    for (const f of fixedOf(fixed)) expect(hit(a, f), `label ${i} covers a fixed label`).toBe(false);
  });
}

const none = () => makeRectList(0);

describe("label slots", () => {
  it("default slot is the old CSS: 10 px right of the anchor, vertically centred", () => {
    expect(LABEL_SLOTS[0]).toEqual({ side: 1, lines: 0 });
    expect(slotOffsetX(LABEL_SLOTS[0], 150)).toBe(LABEL_GAP_PX);
    expect(LABEL_GAP_PX).toBe(10);
    expect(slotOffsetY(LABEL_SLOTS[0], 14)).toBe(-7);
  });

  it("tries the anchor's left side next, then whole lines up and down, each slot once", () => {
    expect(LABEL_SLOTS[1]).toEqual({ side: -1, lines: 0 });
    expect(slotOffsetX(LABEL_SLOTS[1], 150)).toBe(-LABEL_GAP_PX - 150);
    const keys = LABEL_SLOTS.map((s) => `${s.side}:${s.lines}`);
    expect(new Set(keys).size).toBe(keys.length);
    const lines = LABEL_SLOTS.map((s) => Math.abs(s.lines));
    expect([...lines].sort((a, b) => a - b)).toEqual(lines);
    expect(slotOffsetY({ side: 1, lines: -1 }, 14)).toBe(-7 - (14 + LABEL_LINE_GAP_PX));
  });
});

describe("placeLabels", () => {
  it("leaves a lone label in its default slot", () => {
    const p = setup([{ x: 400, y: 300 }]);
    expect(Array.from(placeLabels(p, none(), W, H))).toEqual([0]);
  });

  it("separates two labels whose default boxes collide (the mock's pad and well A in the side view)", () => {
    // Anchors 60 px apart on one line: the first label's box runs over the second's anchor and label.
    const p = setup([
      { x: 560, y: 200, w: 175 },
      { x: 600, y: 196, w: 190 },
    ]);
    placeLabels(p, none(), W, H);
    expect(p.slot[0]).not.toBe(0);
    expect(p.slot[1]).toBe(0);
    expectClear(p, none());
  });

  it("gives earlier labels priority for their default slot", () => {
    const p = setup([
      { x: 600, y: 196, w: 190 },
      { x: 560, y: 200, w: 175 },
    ]);
    placeLabels(p, none(), W, H);
    expect(p.slot[0]).toBe(0);
    expect(p.slot[1]).not.toBe(0);
    expectClear(p, none());
  });

  it("keeps clear of fixed labels (the depth ruler's title and ticks)", () => {
    const fixed = makeRectList(4);
    pushRect(fixed, 400, 290, 260, 16); // right where the default box would go
    const p = setup([{ x: 390, y: 300 }]);
    placeLabels(p, fixed, W, H);
    expect(p.slot[0]).toBeGreaterThan(0);
    expectClear(p, fixed);
  });

  it("flips to the left near the right edge instead of running off screen", () => {
    const p = setup([{ x: W - 60, y: 300 }]);
    placeLabels(p, none(), W, H);
    expect(LABEL_SLOTS[p.slot[0]]).toEqual({ side: -1, lines: 0 });
    expectClear(p, none());
  });

  it("resolves a dense row of labels without any overlap", () => {
    const labels = [0, 1, 2, 3, 4].map((k) => ({ x: 420 + 22 * k, y: 360, w: 140 }));
    const p = setup(labels);
    placeLabels(p, none(), W, H);
    expect(Array.from(p.slot).every((s) => s >= 0)).toBe(true);
    expectClear(p, none());
  });

  it("never hides a label: with no free slot it takes the least-overlapping one", () => {
    const fixed = makeRectList(1);
    pushRect(fixed, 0, 0, W, H); // everything is covered
    const p = setup([{ x: 400, y: 300 }]);
    placeLabels(p, fixed, W, H);
    expect(p.slot[0]).toBeGreaterThanOrEqual(0);
  });

  it("skips unmeasured and inactive labels: no box of their own, and an inactive anchor blocks nothing", () => {
    const p = setup([
      // Unmeasured: its box would cover label 2's default slot, but it has none yet (its anchor stays clear).
      { x: 300, y: 200, w: 0 },
      // Inactive (behind the camera): its anchor sits inside label 2's default box but isn't on screen.
      { x: 500, y: 200, active: false },
      { x: 400, y: 200 },
    ]);
    placeLabels(p, none(), W, H);
    expect(p.slot[0]).toBe(-1);
    expect(p.slot[1]).toBe(-1);
    expect(p.slot[2]).toBe(0);
  });

  it("keeps clear of an unmeasured label's anchor (its marker is drawn)", () => {
    const p = setup([
      { x: 480, y: 200, w: 0 },
      { x: 400, y: 200 },
    ]);
    placeLabels(p, none(), W, H);
    expect(p.slot[1]).not.toBe(0);
    expectClear(p, none());
  });

  it("while the camera moves, holds a displaced label's slot while it stays free, and returns home when it can", () => {
    const fixed = makeRectList(1);
    pushRect(fixed, 395, 290, 200, 20); // blocks the default slot
    const p = setup([{ x: 390, y: 300 }]);
    placeLabels(p, fixed, W, H);
    const first = p.slot[0];
    expect(first).toBeGreaterThan(0);
    // Force a later free slot as last frame's choice: it's kept even though an earlier one is free.
    const later = LABEL_SLOTS.findIndex((s, i) => i > first && s.side === 1 && s.lines === 2);
    p.prev[0] = later;
    placeLabels(p, fixed, W, H, true);
    expect(p.slot[0]).toBe(later);
    // The obstacle goes away: back to the default slot.
    clearRects(fixed);
    placeLabels(p, fixed, W, H, true);
    expect(p.slot[0]).toBe(0);
  });

  it("with the camera still, ignores last frame's slot: the layout depends only on this frame", () => {
    const fixed = makeRectList(1);
    pushRect(fixed, 395, 290, 200, 20);
    const p = setup([{ x: 390, y: 300 }]);
    placeLabels(p, fixed, W, H);
    const canonical = p.slot[0];
    p.prev[0] = LABEL_SLOTS.findIndex((s, i) => i > canonical && s.side === 1 && s.lines === 2);
    placeLabels(p, fixed, W, H, false);
    expect(p.slot[0]).toBe(canonical);
  });

  it("is deterministic", () => {
    const labels = [0, 1, 2, 3].map((k) => ({ x: 500 + 30 * k, y: 300 + 3 * k, w: 130 + 10 * k }));
    const a = setup(labels);
    const b = setup(labels);
    expect(Array.from(placeLabels(a, none(), W, H))).toEqual(Array.from(placeLabels(b, none(), W, H)));
  });
});

describe("leaderOf", () => {
  it("draws nothing on the anchor's line", () => {
    expect(leaderOf({ side: 1, lines: 0 }, 14).length).toBe(0);
    expect(leaderOf({ side: -1, lines: 0 }, 14).length).toBe(0);
  });

  it("ends at the displaced label's near edge, mid-height", () => {
    for (const slot of LABEL_SLOTS.filter((s) => s.lines !== 0)) {
      const { length, angle } = leaderOf(slot, 14);
      const endX = length * Math.cos(angle);
      const endY = length * Math.sin(angle);
      const nearEdge = slot.side > 0 ? slotOffsetX(slot, 150) : slotOffsetX(slot, 150) + 150;
      expect(endX).toBeCloseTo(nearEdge, 6);
      expect(endY).toBeCloseTo(slotOffsetY(slot, 14) + 7, 6);
    }
  });
});

describe("cameraMoved", () => {
  it("is true on the first frame and on any change to the pose, projection or viewport; false when still", () => {
    const track = makeCameraTrack();
    const world = Array.from({ length: 16 }, (_, k) => (k % 5 === 0 ? 1 : 0));
    const proj = Array.from({ length: 16 }, (_, k) => k / 10);
    expect(cameraMoved(track, world, proj, 1280, 720)).toBe(true);
    expect(cameraMoved(track, world, proj, 1280, 720)).toBe(false);
    world[12] += 1e-9; // a damped orbit's last creep still counts as moving
    expect(cameraMoved(track, world, proj, 1280, 720)).toBe(true);
    expect(cameraMoved(track, world, proj, 1280, 720)).toBe(false);
    proj[5] *= 1.01; // zoom
    expect(cameraMoved(track, world, proj, 1280, 720)).toBe(true);
    expect(cameraMoved(track, world, proj, 1920, 1080)).toBe(true);
    expect(cameraMoved(track, world, proj, 1920, 1080)).toBe(false);
  });
});

describe("labelPriority", () => {
  it("places facilities, then wells, then boundaries", () => {
    expect(labelPriority("facility")).toBeLessThan(labelPriority("well"));
    expect(labelPriority("well")).toBeLessThan(labelPriority("boundary"));
  });
});

describe("rect lists", () => {
  it("drops rects past capacity, appends and clears in place", () => {
    const a = makeRectList(2);
    pushRect(a, 1, 2, 3, 4);
    pushRect(a, 5, 6, 7, 8);
    pushRect(a, 9, 9, 9, 9);
    expect(a.count).toBe(2);
    const b = makeRectList(3);
    pushRect(b, 0, 0, 1, 1);
    appendRects(b, a);
    expect(fixedOf(b)).toEqual([
      { l: 0, t: 0, w: 1, h: 1 },
      { l: 1, t: 2, w: 3, h: 4 },
      { l: 5, t: 6, w: 7, h: 8 },
    ]);
    clearRects(b);
    expect(b.count).toBe(0);
  });

  it("ruler obstacle boxes mirror DepthRuler's CSS transforms", () => {
    const list = makeRectList(2);
    // Title: translate(-4px, calc(-100% - 10px)) from its anchor.
    pushRulerTitleRect(list, 300, 200, 240, 14);
    // Tick label: translate(calc(-100% - 5px), -50%) from the tick's end.
    pushRulerTickRect(list, 300, 260, 36, 14);
    const [title, tick] = fixedOf(list);
    expect(title).toEqual({ l: 300 - RULER_TITLE_INSET_PX, t: 200 - RULER_TITLE_RISE_PX - 14, w: 240, h: 14 });
    expect(tick).toEqual({ l: 300 - RULER_TICK_GAP_PX - 36, t: 253, w: 36, h: 14 });
    expect([RULER_TITLE_INSET_PX, RULER_TITLE_RISE_PX, RULER_TICK_GAP_PX]).toEqual([4, 10, 5]);
  });
});

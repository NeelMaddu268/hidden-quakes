import { describe, expect, it } from "vitest";
import { scrubberRect } from "../time/layout";
import {
  drawerWidthPx,
  NO_EDGES,
  planReservePx,
  SECTION_LAYOUT,
  sectionDock,
  sectionPanelRect,
  type ShellEdges,
} from "./layout";

type Rect = { left: number; top: number; width: number; height: number };
const overlaps = (a: Rect, b: Rect) =>
  a.left < b.left + b.width && b.left < a.left + a.width && a.top < b.top + b.height && b.top < a.top + a.height;

/** The shell/drawer regions the panel must never cover, approximated generously from their CSS. */
function reserved(W: number, H: number): Record<string, Rect> {
  const inset = SECTION_LAYOUT.inset;
  return {
    drawer: { left: W - drawerWidthPx(W), top: 0, width: drawerWidthPx(W), height: H },
    titleBlock: { left: inset, top: 0, width: 420, height: 120 },
    modePills: { left: inset, top: H - inset - 40, width: 420, height: 40 },
    // shell/validation: left inset, bottom inset + 2.75rem, ≤ 22rem wide, ~270px tall with the baseline,
    // null-test and depth notes (1280 wide)
    validationPanel: { left: inset, top: H - (inset + 44) - 270, width: 352, height: 270 },
    runDetailsButton: { left: inset, top: 56 + 70, width: 200, height: 1 },
    revealButton: { left: W / 2 - W * 0.2, top: H - 0.14 * H - 90, width: W * 0.4, height: 90 },
    countersAndFilters: { left: W - 360, top: 0, width: 360, height: 200 },
  };
}

describe("sectionPanelRect (the left-column dock alone)", () => {
  for (const [W, H] of [
    [1440, 900],
    [1920, 1080],
    [2560, 1440],
    [3840, 2160],
  ]) {
    it(`at ${W}×${H} clears the drawer, title, mode pills, REVEAL and counters`, () => {
      const rect = sectionPanelRect(W, H);
      expect(rect).not.toBeNull();
      for (const [name, region] of Object.entries(reserved(W, H))) {
        expect(overlaps(rect!, region), `overlaps ${name}`).toBe(false);
      }
      expect(rect!.width).toBeGreaterThanOrEqual(SECTION_LAYOUT.minWidth);
      expect(rect!.height).toBeGreaterThanOrEqual(SECTION_LAYOUT.minHeight);
    });
  }

  it("grows with the viewport but stays capped for 4K", () => {
    const small = sectionPanelRect(1440, 900)!;
    const large = sectionPanelRect(3840, 2160)!;
    expect(large.width).toBeGreaterThan(small.width);
    expect(large.width).toBeLessThanOrEqual(SECTION_LAYOUT.maxWidth);
    expect(large.height).toBeLessThanOrEqual(SECTION_LAYOUT.maxHeight);
  });

  it("stops above a measured card, and is null when the left column is too short", () => {
    const fitted = sectionPanelRect(1920, 1080, 680.6)!;
    expect(fitted.top + fitted.height).toBeLessThanOrEqual(680.6 - SECTION_LAYOUT.cardGap);
    expect(sectionPanelRect(1280, 720, 320.6)).toBeNull();
    expect(sectionPanelRect(700, 500)).toBeNull();
    expect(sectionPanelRect(0, 0)).toBeNull();
  });
});

// The shell's edges measured on the final build (hidden-quakes.vercel.app, after the reveal): title block
// bottom, validation card top, counters + pills bottom, legend top.
const EDGES: Record<string, ShellEdges> = {
  "1280x720": { headerBottom: 144.4, cardTop: 320.6, topRightBottom: 154.4, cornerTop: 651 },
  "1440x900": { headerBottom: 144.4, cardTop: 500.6, topRightBottom: 156, cornerTop: 831 },
  "1920x1080": { headerBottom: 144.4, cardTop: 680.6, topRightBottom: 156, cornerTop: 1011 },
  "3840x2160": { headerBottom: 144.4, cardTop: 1760.6, topRightBottom: 156, cornerTop: 2091 },
};
const size = (key: string) => key.split("x").map(Number) as [number, number];
const block = (e: ShellEdges, W: number, H: number): Record<string, Rect> => ({
  titleBlock: { left: 0, top: 0, width: 600, height: e.headerBottom! },
  card: { left: 24, top: e.cardTop!, width: 352, height: H - 68 - e.cardTop! },
  countersAndPills: { left: W - 400, top: 0, width: 400, height: e.topRightBottom! },
  cornerStack: { left: W - 300, top: e.cornerTop!, width: 300, height: H - e.cornerTop! },
});

describe("sectionDock (left column, else right column)", () => {
  it("docks left where the left column has room, clear of every measured block", () => {
    for (const key of ["1440x900", "1920x1080", "3840x2160"]) {
      const [W, H] = size(key);
      const d = sectionDock(W, H, EDGES[key])!;
      expect(d.side, key).toBe("left");
      expect(d.left).toBe(SECTION_LAYOUT.inset);
      for (const [name, r] of Object.entries(block(EDGES[key]!, W, H))) expect(overlaps(d, r), `${key} ${name}`).toBe(false);
    }
  });

  it("docks right at 1280×720, where the taller card leaves the left column too short", () => {
    const e = EDGES["1280x720"]!;
    const d = sectionDock(1280, 720, e)!;
    expect(d.side).toBe("right");
    expect(d.left + d.width).toBe(1280 - SECTION_LAYOUT.inset);
    expect(d.height).toBeGreaterThanOrEqual(SECTION_LAYOUT.minHeight);
    for (const [name, r] of Object.entries(block(e, 1280, 720))) expect(overlaps(d, r), name).toBe(false);
  });

  it("never overlaps the time scrubber, drawer closed (both can be up at once)", () => {
    for (const key of Object.keys(EDGES)) {
      const [W, H] = size(key);
      const d = sectionDock(W, H, EDGES[key])!;
      const scrubber = scrubberRect(W, H, false);
      if (scrubber) expect(overlaps(d, scrubber), key).toBe(false);
    }
  });

  it("is sized the same whatever the drawer does (the camera frames around it once)", () => {
    for (const key of Object.keys(EDGES)) {
      const [W] = size(key);
      const d = sectionDock(W, size(key)[1], EDGES[key])!;
      expect(d.width).toBeLessThanOrEqual(W - drawerWidthPx(W) - SECTION_LAYOUT.drawerGap - SECTION_LAYOUT.inset);
    }
  });

  it("before the reveal (no card yet) plans for where the card will be", () => {
    // The title block is shorter before the reveal (no download row yet).
    expect(sectionDock(1280, 720, { ...EDGES["1280x720"]!, headerBottom: 114.4, cardTop: null })!.side).toBe("right");
    expect(sectionDock(1920, 1080, { ...EDGES["1920x1080"]!, headerBottom: 114.4, cardTop: null })!.side).toBe("left");
  });

  it("is null when neither column has room, and falls back to fixed edges when nothing is measured", () => {
    const short: ShellEdges = { headerBottom: 144.4, cardTop: 180, topRightBottom: 154.4, cornerTop: 331 };
    expect(sectionDock(1280, 400, short)).toBeNull();
    expect(sectionDock(700, 500)).toBeNull();
    expect(sectionDock(1920, 1080, NO_EDGES)!.side).toBe("left");
  });
});

describe("planReservePx", () => {
  it("reserves the panel's side plus a gap, or nothing without a panel", () => {
    const left = sectionDock(1920, 1080, EDGES["1920x1080"])!;
    expect(planReservePx(left, 1920)).toEqual({ left: left.left + left.width + SECTION_LAYOUT.inset, right: 0 });
    const right = sectionDock(1280, 720, EDGES["1280x720"])!;
    // Docked right, the card's column on the left stays clear too.
    expect(planReservePx(right, 1280)).toEqual({
      left: SECTION_LAYOUT.cardColumn,
      right: 1280 - right.left + SECTION_LAYOUT.inset,
    });
    expect(SECTION_LAYOUT.cardColumn).toBeGreaterThanOrEqual(376 + SECTION_LAYOUT.inset); // the card's right edge, measured
    expect(planReservePx(null, 1280)).toEqual({ left: 0, right: 0 });
  });
});

describe("drawerWidthPx", () => {
  it("matches the drawer's clamp(420px, 40vw, 720px)", () => {
    expect(drawerWidthPx(800)).toBe(420);
    expect(drawerWidthPx(1280)).toBe(512);
    expect(drawerWidthPx(3840)).toBe(720);
  });
});

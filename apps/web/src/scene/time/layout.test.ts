import { describe, expect, it } from "vitest";
import { drawerWidthPx } from "../plan/layout";
import { SCRUBBER_LAYOUT, scrubberRect } from "./layout";

type Rect = { left: number; top: number; width: number; height: number };
const overlaps = (a: Rect, b: Rect) =>
  a.left < b.left + b.width && b.left < a.left + a.width && a.top < b.top + b.height && b.top < a.top + a.height;

/** The shell regions near the bottom after the reveal, approximated generously from their CSS. */
function reserved(W: number, H: number): Record<string, Rect> {
  const inset = SCRUBBER_LAYOUT.inset;
  return {
    // shell/validation: left inset, bottom inset + 2.75rem, at most 22rem wide, ~210 px tall
    validationPanel: { left: inset, top: H - (inset + 44) - 210, width: 352, height: 210 },
    modePills: { left: inset, top: H - inset - 40, width: 300, height: 40 },
    // references/CornerNote: right-aligned at W − 24, two stacked slots of 28 px, the longest note ~280 px
    cornerNotes: { left: W - 24 - 290, top: H - 24 - 2 * 28, width: 290, height: 2 * 28 },
    countersAndFilters: { left: W - 360, top: 0, width: 360, height: 200 },
  };
}

describe("scrubberRect", () => {
  for (const [W, H] of [
    [1280, 720],
    [1440, 900],
    [1920, 1080],
    [2560, 1440],
    [3840, 2160],
  ]) {
    it(`at ${W}×${H} docks along the bottom, clear of the validation card, mode pills and corner notes`, () => {
      const rect = scrubberRect(W, H, false)!;
      expect(rect).not.toBeNull();
      for (const [name, region] of Object.entries(reserved(W, H))) {
        expect(overlaps(rect, region), `overlaps ${name}`).toBe(false);
      }
      expect(rect.top + rect.height).toBe(H - SCRUBBER_LAYOUT.inset);
      expect(rect.width).toBeGreaterThanOrEqual(SCRUBBER_LAYOUT.minWidth);
      expect(rect.width).toBeLessThanOrEqual(SCRUBBER_LAYOUT.maxWidth);
    });

    it(`at ${W}×${H} with the drawer open, stays left of it (or hides)`, () => {
      const rect = scrubberRect(W, H, true);
      if (rect === null) return;
      const drawer = { left: W - drawerWidthPx(W), top: 0, width: drawerWidthPx(W), height: H };
      expect(overlaps(rect, drawer)).toBe(false);
      for (const [name, region] of Object.entries(reserved(W, H))) expect(overlaps(rect, region), `overlaps ${name}`).toBe(false);
    });
  }

  it("is centered on the viewport when the band allows", () => {
    const rect = scrubberRect(1920, 1080, false)!;
    expect(rect.left + rect.width / 2).toBeCloseTo(960, 0);
    expect(rect.width).toBe(SCRUBBER_LAYOUT.maxWidth);
  });

  it("hides rather than covering the shell when the band is too narrow", () => {
    expect(scrubberRect(1280, 720, true)).toBeNull(); // the drawer takes the room
    expect(scrubberRect(800, 600, false)).toBeNull();
    expect(scrubberRect(0, 0, false)).toBeNull();
    expect(scrubberRect(1920, 100, false)).toBeNull();
  });
});

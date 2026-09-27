import { describe, expect, it } from "vitest";
import { drawerWidthPx, planReserveLeftPx, SECTION_LAYOUT, sectionPanelRect } from "./layout";

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

describe("sectionPanelRect", () => {
  for (const [W, H] of [
    [1280, 720],
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
    const small = sectionPanelRect(1280, 720)!;
    const large = sectionPanelRect(3840, 2160)!;
    expect(large.width).toBeGreaterThan(small.width);
    expect(large.width).toBeLessThanOrEqual(SECTION_LAYOUT.maxWidth);
    expect(large.height).toBeLessThanOrEqual(SECTION_LAYOUT.maxHeight);
  });

  it("hides the panel rather than covering the shell on a tiny viewport", () => {
    expect(sectionPanelRect(700, 500)).toBeNull();
    expect(sectionPanelRect(0, 0)).toBeNull();
  });
});

describe("sectionPanelRect with the validation card's measured top", () => {
  it("stops above a taller card, and hides when there's no room left", () => {
    const free = sectionPanelRect(1280, 720)!;
    const card = { left: 24, top: 386, width: 352, height: 266 }; // measured on the real bundle at 1280×720
    const fitted = sectionPanelRect(1280, 720, card.top)!;
    expect(fitted.top + fitted.height).toBeLessThanOrEqual(card.top - SECTION_LAYOUT.cardGap);
    expect(overlaps(fitted, card)).toBe(false);
    expect(fitted.width).toBe(free.width);
    expect(sectionPanelRect(1280, 720, SECTION_LAYOUT.top + 100)).toBeNull(); // under minHeight
    expect(sectionPanelRect(1280, 720, null)).toEqual(free);
    expect(sectionPanelRect(1280, 720, NaN)).toEqual(free);
  });
});

describe("planReserveLeftPx", () => {
  it("reserves the panel plus a gap, or nothing when the panel is hidden", () => {
    const rect = sectionPanelRect(1280, 720)!;
    expect(planReserveLeftPx(1280, 720)).toBe(rect.left + rect.width + SECTION_LAYOUT.inset);
    expect(planReserveLeftPx(700, 500)).toBe(0);
  });
});

describe("drawerWidthPx", () => {
  it("matches the drawer's clamp(420px, 40vw, 720px)", () => {
    expect(drawerWidthPx(800)).toBe(420);
    expect(drawerWidthPx(1280)).toBe(512);
    expect(drawerWidthPx(3840)).toBe(720);
  });
});

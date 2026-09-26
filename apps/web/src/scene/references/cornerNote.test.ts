import { describe, expect, it } from "vitest";
import { drawerWidthPx, sectionPanelRect } from "../plan/layout";
import { CORNER_INSET_PX, CORNER_SLOT_PX, cornerNoteAnchor } from "./CornerNote";

type Rect = { left: number; top: number; width: number; height: number };
const overlaps = (a: Rect, b: Rect) =>
  a.left < b.left + b.width && b.left < a.left + a.width && a.top < b.top + b.height && b.top < a.top + a.height;

/** A generous box for a "Vertical ×N" note anchored at its bottom-right corner. */
function noteBox(W: number, H: number, slot: number): Rect {
  const a = cornerNoteAnchor(W, H, slot);
  return { left: a.right - 160, top: a.bottom - 24, width: 160, height: 24 };
}

describe("corner notes (the permanent Vertical ×N badge)", () => {
  for (const [W, H] of [
    [1280, 720],
    [1920, 1080],
    [3840, 2160],
  ]) {
    it(`at ${W}×${H} sit bottom-right, clear of the shell's bottom-left stack, REVEAL and the plan panel`, () => {
      const inset = 24;
      const note = noteBox(W, H, 0);
      const regions: Record<string, Rect> = {
        modePills: { left: inset, top: H - inset - 40, width: 420, height: 40 },
        validationPanel: { left: inset, top: H - (inset + 44) - 210, width: 352, height: 210 },
        revealButton: { left: W / 2 - W * 0.2, top: H - 0.14 * H - 90, width: W * 0.4, height: 90 },
        countersAndFilters: { left: W - 360, top: 0, width: 360, height: 200 },
      };
      const panel = sectionPanelRect(W, H);
      if (panel) regions.depthSection = panel;
      for (const [name, r] of Object.entries(regions)) expect(overlaps(note, r), name).toBe(false);
      expect(note.left + note.width).toBe(W - CORNER_INSET_PX);
      expect(note.top + note.height).toBe(H - CORNER_INSET_PX);
      // (The open evidence drawer covers this corner; it covers the whole right of the canvas then.)
      expect(note.left).toBeGreaterThan(W - drawerWidthPx(W) - 1);
    });
  }

  it("stacks further notes upward by one slot", () => {
    expect(cornerNoteAnchor(1280, 720, 1).bottom).toBe(cornerNoteAnchor(1280, 720, 0).bottom - CORNER_SLOT_PX);
  });
});

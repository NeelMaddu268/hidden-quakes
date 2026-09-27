import { describe, expect, it } from "vitest";
import { CAPTION_LAYOUT, captionBox, type ObstacleRect } from "./layout";

// Rects measured on the real bundle at 1280×720 (WEB-08 QA): validation card, mode pills, legend,
// REVEAL, the time scrubber, the open drawer (clamp(420px, 40vw, 720px)).
const CARD: ObstacleRect = { left: 24, top: 386, width: 352, height: 266 };
const MODES: ObstacleRect = { left: 24, top: 663, width: 92, height: 33 };
const LEGEND: ObstacleRect = { left: 1010, top: 648, width: 246, height: 48 };
const REVEAL: ObstacleRect = { left: 382, top: 553, width: 516, height: 66 };
const SCRUBBER: ObstacleRect = { left: 400, top: 584, width: 550, height: 112 };
const DRAWER: ObstacleRect = { left: 1280 - 512, top: 0, width: 512, height: 720 };
const HEADER: ObstacleRect = { left: 24, top: 24, width: 450, height: 90 };

const L = CAPTION_LAYOUT;
const W = 1280;
const H = 720;
const h = 60;

const top = (b: { bottom: number }) => H - b.bottom - h;
const overlaps = (b: { left: number; bottom: number; width: number }, o: ObstacleRect) =>
  b.left < o.left + o.width && o.left < b.left + b.width && top(b) < o.top + o.height && o.top < top(b) + h;

describe("captionBox", () => {
  it("sits bottom-center, max width, with nothing on screen", () => {
    const b = captionBox(W, H, h, []);
    expect(b).toEqual({ left: (W - L.maxWidth) / 2, bottom: L.inset, width: L.maxWidth });
  });

  it("stays centered between the mode pills and the legend on the revealed frame", () => {
    const b = captionBox(W, H, h, [HEADER, MODES, LEGEND]);
    expect(b.bottom).toBe(L.inset);
    expect(b.left + b.width / 2).toBe(W / 2);
    for (const o of [MODES, LEGEND]) expect(overlaps(b, o)).toBe(false);
  });

  it("fits under REVEAL on the start frame when there's room, else steps up above it", () => {
    const b = captionBox(W, H, h, [HEADER, MODES, LEGEND, REVEAL]);
    expect(b.bottom).toBe(L.inset);
    for (const o of [MODES, LEGEND, REVEAL]) expect(overlaps(b, o)).toBe(false);
    const tall = 90;
    const up = captionBox(W, H, tall, [HEADER, MODES, LEGEND, REVEAL]);
    expect(H - up.bottom).toBe(REVEAL.top - L.gap);
    expect(H - up.bottom - tall).toBeGreaterThan(HEADER.top + HEADER.height);
  });

  it("steps up above the scrubber in time mode and keeps clear of the card beside it", () => {
    const b = captionBox(W, H, h, [HEADER, CARD, MODES, LEGEND, SCRUBBER]);
    expect(H - b.bottom).toBe(SCRUBBER.top - L.gap);
    expect(b.left).toBeGreaterThanOrEqual(CARD.left + CARD.width + L.gap);
    for (const o of [CARD, MODES, LEGEND, SCRUBBER]) expect(overlaps(b, o)).toBe(false);
  });

  it("keeps left of the open drawer, as close to center as the free span allows", () => {
    const b = captionBox(W, H, h, [HEADER, CARD, MODES, LEGEND, DRAWER]);
    expect(b.left + b.width).toBeLessThanOrEqual(DRAWER.left - L.gap);
    expect(b.left).toBeGreaterThanOrEqual(CARD.left + CARD.width + L.gap);
    for (const o of [CARD, MODES, LEGEND, DRAWER]) expect(overlaps(b, o)).toBe(false);
  });

  it("stays centered and capped on a 4K screen", () => {
    const b = captionBox(3840, 2160, h, [MODES]);
    expect(b.width).toBe(L.maxWidth);
    expect(b.left + b.width / 2).toBe(1920);
  });

  it("climbs above a full-width overlay, and pins to the top when there's no room anywhere", () => {
    const wall: ObstacleRect = { left: 0, top: 200, width: W, height: 520 };
    expect(H - captionBox(W, H, h, [wall]).bottom).toBe(wall.top - L.gap);
    const full: ObstacleRect = { left: 0, top: 0, width: W, height: H };
    const b = captionBox(W, H, h, [full]);
    expect(H - b.bottom - h).toBe(L.inset);
    expect(b.left + b.width / 2).toBe(W / 2);
  });

  it("ignores zero-size overlays (unmounted panels)", () => {
    expect(captionBox(W, H, h, [{ left: 300, top: 600, width: 0, height: 0 }])).toEqual(captionBox(W, H, h, []));
  });
});

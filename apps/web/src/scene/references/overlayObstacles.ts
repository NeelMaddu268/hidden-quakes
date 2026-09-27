// The DOM overlays the scene's feature labels keep clear of (WEB-08): H4's shell blocks (SYNTHETIC
// banner, title block, counters and filter pills, REVEAL, validation card, mode pills: the direct
// children of the shell root, found through the shell's `mode-label` test id) and H3's own panels
// (legend and corner notes, depth section, time scrubber, open evidence drawer, the tour's caption).
// Read-only: it measures rects and never touches another lane's DOM. On real data a wellhead label was
// placed under the PUBLIC counter; this is why.

import { clearRects, makeRectList, pushRect, type RectList } from "./labelPlacement";

/** H3 panels that float over the canvas. */
export const H3_OVERLAY_SELECTORS: readonly string[] = Object.freeze([
  '[data-testid="scene-legend"]',
  '[data-testid="vertical-badge"]',
  '[data-testid="abstract-surface-note"]',
  '[data-testid="depth-section"]',
  '[data-testid="time-scrubber"]',
  '.hqd[data-open="true"]',
  '[data-testid="tour-caption"]',
]);

/** The shell root: the element whose children are the shell's overlay blocks (header → shell). */
export function shellRoot(doc: Document): Element | null {
  return doc.querySelector('[data-testid="mode-label"]')?.parentElement?.parentElement ?? null;
}

/**
 * Writes the rects (CSS px, relative to `origin`, the canvas's top-left) of every visible overlay into
 * `out`, replacing its contents. Zero-size blocks (unmounted, display: none) are skipped.
 */
export function measureOverlayRects(doc: Document, origin: { left: number; top: number }, out: RectList): void {
  clearRects(out);
  const push = (el: Element) => {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) pushRect(out, r.left - origin.left, r.top - origin.top, r.width, r.height);
  };
  const shell = shellRoot(doc);
  if (shell) for (const child of Array.from(shell.children)) push(child);
  for (const selector of H3_OVERLAY_SELECTORS) for (const el of Array.from(doc.querySelectorAll(selector))) push(el);
}

/** Overlay rects kept current by periodic measurement (the overlays are fixed-position DOM). */
export class OverlayObstacles {
  readonly list: RectList;

  constructor(capacity = 32) {
    this.list = makeRectList(capacity);
  }

  measure(doc: Document, canvas: Element): void {
    const c = canvas.getBoundingClientRect();
    measureOverlayRects(doc, { left: c.left, top: c.top }, this.list);
  }
}

/** How often the overlay rects are re-measured (ms): they change only with phase, panels and resizes. */
export const OVERLAY_MEASURE_MS = 250;

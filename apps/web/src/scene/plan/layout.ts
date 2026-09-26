// Where the depth-section panel sits in plan view, as a pure function of the viewport (CSS px), so the
// panel, the plan camera's framing and the tests all agree. The shell (H4) owns the overlay: title at
// top-left, counters and filter pills top-right, mode pills bottom-left, REVEAL centered low (14vh),
// modal panels centered; the evidence drawer (H3) is the right edge at clamp(420px, 40vw, 720px). The
// left edge between the title and the mode pills is free, so the panel docks there.

export const SECTION_LAYOUT = Object.freeze({
  /** The shell's --shell-inset. */
  inset: 24,
  /** Clears the title + mode label block (and the SYNTHETIC banner's extra 2rem). */
  top: 136,
  /** Clears the mode pills row at the bottom-left. */
  bottomReserve: 88,
  /** Gap kept between the panel and the evidence drawer. */
  drawerGap: 24,
  widthFrac: 0.34,
  minWidth: 300,
  maxWidth: 760,
  heightFrac: 0.4,
  minHeight: 180,
  maxHeight: 480,
  /** Horizontal half-width of the REVEAL button region the panel must not cover (fraction of vw). */
  revealHalfWidthFrac: 0.2,
});

export interface PanelRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** The evidence drawer's width at this viewport width (drawer/styles.ts: clamp(420px, 40vw, 720px)). */
export function drawerWidthPx(viewportWidth: number): number {
  return Math.min(720, Math.max(420, 0.4 * viewportWidth));
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/**
 * The panel rect for a viewport, or null when the viewport is too small to show it without covering
 * the shell or the drawer (the panel is then simply not shown; the plan map still works).
 */
export function sectionPanelRect(viewportWidth: number, viewportHeight: number): PanelRect | null {
  const L = SECTION_LAYOUT;
  if (!(viewportWidth > 0) || !(viewportHeight > 0)) return null;
  // Never reaches under the open drawer, nor the centered REVEAL button.
  const maxByDrawer = viewportWidth - drawerWidthPx(viewportWidth) - L.drawerGap - L.inset;
  const maxByReveal = viewportWidth * (0.5 - L.revealHalfWidthFrac) - L.inset;
  const width = Math.min(clamp(viewportWidth * L.widthFrac, L.minWidth, L.maxWidth), maxByDrawer, maxByReveal);
  const maxHeight = viewportHeight - L.top - L.bottomReserve;
  const height = Math.min(clamp(viewportHeight * L.heightFrac, L.minHeight, L.maxHeight), maxHeight);
  if (width < L.minWidth || height < L.minHeight) return null;
  return { left: L.inset, top: L.top, width: Math.floor(width), height: Math.floor(height) };
}

/** CSS px on the left that the plan camera keeps clear of data (panel plus a gap), 0 without a panel. */
export function planReserveLeftPx(viewportWidth: number, viewportHeight: number): number {
  const rect = sectionPanelRect(viewportWidth, viewportHeight);
  return rect ? rect.left + rect.width + SECTION_LAYOUT.inset : 0;
}

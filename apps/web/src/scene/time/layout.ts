// Where the time scrubber sits, as a pure function of the viewport (CSS px) and whether the evidence
// drawer is open, so the component and the tests agree. It docks along the bottom in the band the shell
// leaves free after the reveal: right of the validation card and mode pills (bottom-left, H4, at most
// 22rem wide), left of the permanent corner notes (bottom-right: VE badge, abstract-surface note) and
// of the open drawer (right edge, clamp(420px, 40vw, 720px)). REVEAL is unmounted after the reveal.

import { drawerWidthPx } from "../plan/layout";

export const SCRUBBER_LAYOUT = Object.freeze({
  /** The shell's --shell-inset. */
  inset: 24,
  /** Validation card (max 22rem) plus the inset on both sides of it. */
  leftReserve: 24 + 352 + 24,
  /** Room for the widest corner note ("Abstract surface (no terrain model)") plus a gap. */
  rightReserve: 24 + 290 + 16,
  /** Gap kept from the open drawer. */
  drawerGap: 24,
  height: 112,
  minWidth: 360,
  maxWidth: 920,
});

export interface ScrubberRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** The scrubber's rect, centered in the free band, or null when the band is too narrow to show it. */
export function scrubberRect(viewportWidth: number, viewportHeight: number, drawerOpen: boolean): ScrubberRect | null {
  const L = SCRUBBER_LAYOUT;
  if (!(viewportWidth > 0) || !(viewportHeight > L.height + 2 * L.inset)) return null;
  const bandLeft = L.leftReserve;
  let bandRight = viewportWidth - L.rightReserve;
  if (drawerOpen) bandRight = Math.min(bandRight, viewportWidth - drawerWidthPx(viewportWidth) - L.drawerGap);
  const band = bandRight - bandLeft;
  if (band < L.minWidth) return null;
  const width = Math.min(L.maxWidth, band);
  // Centered on the viewport when the band allows it, else as close to center as the band permits.
  const centered = (viewportWidth - width) / 2;
  const left = Math.min(Math.max(centered, bandLeft), bandRight - width);
  return {
    left: Math.round(left),
    top: Math.round(viewportHeight - L.inset - L.height),
    width: Math.floor(width),
    height: L.height,
  };
}

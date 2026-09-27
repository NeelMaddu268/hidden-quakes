// Shared look for the small DOM labels the reference layers pin into the canvas (drei <Html>).
// Labels never take pointer events and stay well below the shell's overlays. A soft halo in the page
// background colour keeps them legible where they cross bright events (the cartographic text halo); it
// is not a decorative glow: it only darkens what sits right behind the letters.

import { colors, fonts, numeric } from "@hq/visualization";
import type { CSSProperties } from "react";

/** drei <Html> z-index range: keeps labels inside the canvas's own stacking context, low. */
export const LABEL_Z_RANGE: [number, number] = [20, 0];

/**
 * The legibility halo, all in the background colour: a crisp 1 px edge on four sides (a blurred shadow
 * alone is too faint behind thin glyphs) plus a soft falloff. Symmetric, so it never reads as a drop shadow.
 */
export const LABEL_HALO = [
  `1px 0 0 ${colors.bg}`,
  `-1px 0 0 ${colors.bg}`,
  `0 1px 0 ${colors.bg}`,
  `0 -1px 0 ${colors.bg}`,
  `0 0 3px ${colors.bg}`,
  `0 0 5px ${colors.bg}`,
].join(", ");

export const labelStyle: CSSProperties = {
  fontFamily: fonts.ui,
  textShadow: LABEL_HALO,
  fontSize: 11,
  lineHeight: 1.25,
  letterSpacing: "0.01em",
  color: colors.textDim,
  whiteSpace: "nowrap",
  pointerEvents: "none",
  userSelect: "none",
};

/**
 * A faint plate in the background colour behind the labels that sit in the scene (feature names, the
 * depth ruler's title and ticks). Placement keeps them off the feature lines where it can
 * (labelPlacement); where it can't, a dashed line passes behind the plate instead of through the text.
 */
export const LABEL_PLATE: CSSProperties = {
  background: `${colors.bg}B8`,
  padding: "0 4px",
  borderRadius: 3,
};

export const numericLabelStyle: CSSProperties = {
  ...labelStyle,
  fontFamily: fonts.mono,
  ...numeric,
};

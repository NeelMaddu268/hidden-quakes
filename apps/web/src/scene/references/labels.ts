// Shared look for the small DOM labels the reference layers pin into the canvas (drei <Html>).
// Labels never take pointer events and stay well below the shell's overlays.

import { colors, fonts, numeric } from "@hq/visualization";
import type { CSSProperties } from "react";

/** drei <Html> z-index range: keeps labels inside the canvas's own stacking context, low. */
export const LABEL_Z_RANGE: [number, number] = [20, 0];

export const labelStyle: CSSProperties = {
  fontFamily: fonts.ui,
  fontSize: 11,
  lineHeight: 1.25,
  letterSpacing: "0.01em",
  color: colors.textDim,
  whiteSpace: "nowrap",
  pointerEvents: "none",
  userSelect: "none",
};

export const numericLabelStyle: CSSProperties = {
  ...labelStyle,
  fontFamily: fonts.mono,
  ...numeric,
};

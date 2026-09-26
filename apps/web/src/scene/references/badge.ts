// The permanent "Vertical ×N" badge (docs/01: any vertical exaggeration other than 1.0 shows it).
// N comes from SceneMeta.verticalExaggeration, formatted without trailing zeros.

import { verticalExaggerationOf } from "../coords";
import type { SceneMeta } from "../types";

/** 2 → "2", 1.5 → "1.5", 2.25 → "2.25", 1.3333 → "1.33". Never rounds a real exaggeration to "1". */
export function formatExaggeration(ve: number): string {
  const short = String(Number(ve.toFixed(2)));
  if (short !== "1" || ve === 1) return short;
  return String(Number(ve.toPrecision(6)));
}

/** Badge text, or null when there's no exaggeration (nothing is shown). */
export function verticalBadgeText(scene: Pick<SceneMeta, "verticalExaggeration">): string | null {
  const ve = verticalExaggerationOf(scene);
  return ve === 1 ? null : `Vertical ×${formatExaggeration(ve)}`;
}

"use client";

import type { SceneMeta } from "../types";
import { verticalBadgeText } from "./badge";
import { CornerNote } from "./CornerNote";
import { LEGEND_SLOTS } from "./SceneLegend";

/**
 * Permanent "Vertical ×N" badge in the canvas's bottom-right corner whenever the scene is vertically
 * exaggerated (docs/01). Nothing at all when the exaggeration is 1.
 */
export function VerticalBadge({ scene }: { scene: SceneMeta }) {
  const text = verticalBadgeText(scene);
  if (!text) return null;
  return (
    <CornerNote slot={LEGEND_SLOTS} testId="vertical-badge">
      {text}
    </CornerNote>
  );
}

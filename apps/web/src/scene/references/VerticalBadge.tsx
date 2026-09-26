"use client";

import type { SceneMeta } from "../types";
import { verticalBadgeText } from "./badge";
import { CornerNote } from "./CornerNote";

/**
 * Permanent "Vertical ×N" badge in the canvas's bottom-left corner whenever the scene is vertically
 * exaggerated (docs/01). Nothing at all when the exaggeration is 1.
 */
export function VerticalBadge({ scene }: { scene: SceneMeta }) {
  const text = verticalBadgeText(scene);
  if (!text) return null;
  return (
    <CornerNote slot={0} testId="vertical-badge">
      {text}
    </CornerNote>
  );
}

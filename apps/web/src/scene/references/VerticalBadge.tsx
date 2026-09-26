"use client";

import { Html } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import { colors } from "@hq/visualization";
import type { SceneMeta } from "../types";
import { verticalBadgeText } from "./badge";
import { LABEL_Z_RANGE, numericLabelStyle } from "./labels";

const CANVAS_ORIGIN_PX: [number, number] = [0, 0];
const pinToCanvasOrigin = () => CANVAS_ORIGIN_PX;
/** Inset from the canvas's bottom-left corner (CSS px). */
const INSET_PX = 16;

/**
 * Permanent "Vertical ×N" badge, pinned to the canvas's bottom-left corner, whenever the scene is
 * vertically exaggerated (docs/01). Nothing at all when the exaggeration is 1.
 */
export function VerticalBadge({ scene }: { scene: SceneMeta }) {
  const height = useThree((s) => s.size.height);
  const text = verticalBadgeText(scene);
  if (!text) return null;
  return (
    <Html calculatePosition={pinToCanvasOrigin} zIndexRange={LABEL_Z_RANGE} pointerEvents="none">
      <div
        data-testid="vertical-badge"
        style={{
          ...numericLabelStyle,
          position: "absolute",
          left: INSET_PX,
          top: height - INSET_PX,
          transform: "translateY(-100%)",
          color: colors.text,
          padding: "3px 7px",
          border: `1px solid ${colors.contour}`,
          borderRadius: 3,
          background: colors.bg,
        }}
      >
        {text}
      </div>
    </Html>
  );
}

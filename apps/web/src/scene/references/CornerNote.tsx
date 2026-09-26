"use client";

import { SceneHtml as Html } from "./SceneHtml";
import { useThree } from "@react-three/fiber";
import { colors } from "@hq/visualization";
import type { ReactNode } from "react";
import { LABEL_Z_RANGE, numericLabelStyle } from "./labels";

const CANVAS_ORIGIN_PX: [number, number] = [0, 0];
const pinToCanvasOrigin = () => CANVAS_ORIGIN_PX;
/** Inset from the canvas's bottom-left corner, and the height of one stacked note (CSS px). */
const INSET_PX = 16;
const SLOT_PX = 28;

/**
 * A small permanent note pinned to the canvas's bottom-left corner, whatever the camera does.
 * Notes stack upward by `slot` (0 = bottom), so the scene's notes never overlap:
 * slot 0 = "Vertical ×N" (references), slot 1 = the abstract-surface note (terrain).
 */
export function CornerNote({ slot, testId, children }: { slot: number; testId: string; children: ReactNode }) {
  const height = useThree((s) => s.size.height);
  return (
    <Html calculatePosition={pinToCanvasOrigin} zIndexRange={LABEL_Z_RANGE} pointerEvents="none">
      <div
        data-testid={testId}
        style={{
          ...numericLabelStyle,
          position: "absolute",
          left: INSET_PX,
          top: height - INSET_PX - slot * SLOT_PX,
          transform: "translateY(-100%)",
          color: colors.text,
          padding: "3px 7px",
          border: `1px solid ${colors.contour}`,
          borderRadius: 3,
          background: colors.bg,
        }}
      >
        {children}
      </div>
    </Html>
  );
}

"use client";

import { SceneHtml as Html } from "./SceneHtml";
import { useThree } from "@react-three/fiber";
import { colors } from "@hq/visualization";
import type { ReactNode } from "react";
import { LABEL_Z_RANGE, numericLabelStyle } from "./labels";

const CANVAS_ORIGIN_PX: [number, number] = [0, 0];
const pinToCanvasOrigin = () => CANVAS_ORIGIN_PX;
/**
 * Inset from the canvas's bottom-RIGHT corner (the shell's --shell-inset), and the height of one stacked
 * note (CSS px). Bottom-right because H4's shell owns bottom-left (mode pills, validation panel); the
 * evidence drawer covers this corner only while it is open, when it covers the canvas anyway.
 */
export const CORNER_INSET_PX = 24;
export const CORNER_SLOT_PX = 28;

/** The note's anchor (its bottom-right corner) in canvas CSS px, for a canvas of `width × height`. */
export function cornerNoteAnchor(width: number, height: number, slot: number): { right: number; bottom: number } {
  return { right: width - CORNER_INSET_PX, bottom: height - CORNER_INSET_PX - slot * CORNER_SLOT_PX };
}

/**
 * A small permanent note pinned to the canvas's bottom-right corner, whatever the camera does.
 * Notes stack upward by `slot` (0 = bottom), so the scene's notes never overlap: the legend takes the
 * bottom LEGEND_SLOTS (SceneLegend), then "Vertical ×N" (references), then the abstract-surface note.
 */
export function CornerNote({ slot, testId, children }: { slot: number; testId: string; children: ReactNode }) {
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  const anchor = cornerNoteAnchor(width, height, slot);
  return (
    <Html calculatePosition={pinToCanvasOrigin} zIndexRange={LABEL_Z_RANGE} pointerEvents="none">
      <div
        data-testid={testId}
        style={{
          ...numericLabelStyle,
          position: "absolute",
          left: anchor.right,
          top: anchor.bottom,
          transform: "translate(-100%, -100%)",
          whiteSpace: "nowrap",
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

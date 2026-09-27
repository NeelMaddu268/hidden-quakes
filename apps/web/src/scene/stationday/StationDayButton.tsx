"use client";

import { colors, fonts } from "@hq/visualization";
import { useThree } from "@react-three/fiber";
import type { CSSProperties } from "react";
import { useDemo } from "../../state/demo";
import { cornerNoteAnchor } from "../references/CornerNote";
import { LABEL_Z_RANGE } from "../references/labels";
import { SceneHtml as Html } from "../references/SceneHtml";
import { useTour } from "../tour/store";
import { fittingManifest, openStationDay, useStationDay } from "./store";

/** The button's accessible name and label (a label, not data). */
export const STATION_DAY_BUTTON_LABEL = "Station day";
export const STATION_DAY_DIALOG_ID = "station-day-dialog";

const CANVAS_ORIGIN_PX: [number, number] = [0, 0];
const pinToCanvasOrigin = () => CANVAS_ORIGIN_PX;

const buttonStyle: CSSProperties = {
  pointerEvents: "auto",
  whiteSpace: "nowrap",
  padding: "2px 9px",
  border: `1px solid ${colors.contour}`,
  borderRadius: 3,
  background: colors.bg,
  color: colors.text,
  fontFamily: fonts.ui,
  fontSize: 13,
  lineHeight: 1.35,
  cursor: "pointer",
};

/**
 * Opens the station-day panel. Nothing at all without a valid manifest or when another run is loaded
 * (the picture is one run's day), and nothing before the reveal:
 * the picture carries the candidate-event ticks the reveal introduces. Hidden (space kept) while the
 * guided tour plays, so a recording shows the scene and the captions only.
 */
export function StationDayButton({ style }: { style?: CSSProperties }) {
  const hasManifest = useStationDay((s) => fittingManifest(s) !== null);
  const open = useStationDay((s) => s.open);
  const revealed = useDemo((s) => s.phase !== "public");
  const touring = useTour((s) => s.running);
  if (!hasManifest || !revealed) return null;
  return (
    <button
      type="button"
      data-testid="station-day-button"
      aria-haspopup="dialog"
      aria-expanded={open}
      aria-controls={open ? STATION_DAY_DIALOG_ID : undefined}
      style={{ ...buttonStyle, ...style, visibility: touring ? "hidden" : "visible" }}
      onClick={(event) => openStationDay(event.currentTarget)}
    >
      {STATION_DAY_BUTTON_LABEL}
    </button>
  );
}

/**
 * The button pinned to the canvas's bottom-right corner stack, one slot above the scene's corner notes
 * (legend, "Vertical ×N", abstract-surface note), whatever the camera does. The stack is H3's own; the
 * shell owns the other three corners and the scrubber reserves room for this column (time/layout.ts).
 */
export function StationDayCorner({ slot }: { slot: number }) {
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  const anchor = cornerNoteAnchor(width, height, slot);
  return (
    <Html calculatePosition={pinToCanvasOrigin} zIndexRange={LABEL_Z_RANGE} pointerEvents="none">
      <StationDayButton
        style={{ position: "absolute", left: anchor.right, top: anchor.bottom, transform: "translate(-100%, -100%)" }}
      />
    </Html>
  );
}

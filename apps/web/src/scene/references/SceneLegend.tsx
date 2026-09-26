"use client";

import { colors, fonts } from "@hq/visualization";
import { useThree } from "@react-three/fiber";
import { publicSwatch } from "../look";
import { CORNER_INSET_PX } from "./CornerNote";
import { LABEL_Z_RANGE, labelStyle } from "./labels";
import { SceneHtml as Html } from "./SceneHtml";

/**
 * Corner-note slots (CornerNote's CORNER_SLOT_PX each) the legend takes at the bottom of the
 * bottom-right stack; the VE badge and the abstract-surface note stack above it.
 */
export const LEGEND_SLOTS = 2;

const CANVAS_ORIGIN_PX: [number, number] = [0, 0];
const pinToCanvasOrigin = () => CANVAS_ORIGIN_PX;

/** The on-screen colors of the two event layers (public glyphs render dimmer than their token). */
const PUBLIC_DOT = publicSwatch(colors.public);

function Dot({ color, glow }: { color: string; glow?: boolean }) {
  return (
    <span
      aria-hidden="true"
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: color,
        boxShadow: glow ? `0 0 6px ${color}` : undefined,
        marginRight: 7,
        verticalAlign: "0px",
      }}
    />
  );
}

/**
 * What the two event colors mean, from the first frame (mock-judging request): the public regional
 * catalog in its cool white, candidate events in amber, the strict tier drawn brightest. Pinned to the
 * canvas's bottom-right corner under the corner notes, whatever the camera does.
 */
export function SceneLegend() {
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  return (
    <Html calculatePosition={pinToCanvasOrigin} zIndexRange={LABEL_Z_RANGE} pointerEvents="none">
      <div
        data-testid="scene-legend"
        role="note"
        aria-label="Legend"
        style={{
          ...labelStyle,
          fontFamily: fonts.ui,
          position: "absolute",
          left: width - CORNER_INSET_PX,
          top: height - CORNER_INSET_PX,
          transform: "translate(-100%, -100%)",
          display: "grid",
          rowGap: 3,
          padding: "5px 9px",
          border: `1px solid ${colors.contour}`,
          borderRadius: 3,
          background: colors.bg,
          color: colors.textDim,
          fontSize: 12,
        }}
      >
        <span>
          <Dot color={PUBLIC_DOT} />
          public regional catalog
        </span>
        <span>
          <Dot color={colors.recovered} glow />
          candidate events, bright = strict
        </span>
      </div>
    </Html>
  );
}

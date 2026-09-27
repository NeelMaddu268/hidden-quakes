"use client";

import { Line } from "@react-three/drei";
import { colors } from "@hq/visualization";
import { useMemo } from "react";
import type { EnuBoundsM } from "../terrain/meta";
import type { SceneMeta } from "../types";
import { RENDER_ORDER } from "../terrain/renderOrder";
import { sliceDepthsKm, sliceSegments } from "./ruler";

/** Opacity of the slice outlines: context, not content. */
const SLICE_OPACITY = 0.55;

/**
 * Faint square outlines every 1 km of depth over the surface extent, down to the ruler's depth (the
 * deepest displayed event). Underground: covered by the terrain until it fades.
 */
export function DepthSlices({ scene, extent, maxDepthKm }: { scene: SceneMeta; extent: EnuBoundsM; maxDepthKm: number }) {
  const segments = useMemo(() => sliceSegments(extent, scene, sliceDepthsKm(maxDepthKm)), [extent, scene, maxDepthKm]);
  return (
    <Line
      name="depth-slices"
      points={segments}
      segments
      color={colors.contour}
      lineWidth={1}
      transparent
      opacity={SLICE_OPACITY}
      depthWrite={false}
      renderOrder={RENDER_ORDER.underground}
    />
  );
}

"use client";

import { Line } from "@react-three/drei";
import { colors } from "@hq/visualization";
import { useMemo } from "react";
import type { EnuBoundsM } from "../terrain/meta";
import type { SceneMeta } from "../types";
import { sliceSegments } from "./ruler";

/** Opacity of the slice outlines: context, not content. */
const SLICE_OPACITY = 0.55;

/** Faint square outlines every 1 km of depth over the surface extent. Occluded by a solid terrain. */
export function DepthSlices({ scene, extent }: { scene: SceneMeta; extent: EnuBoundsM }) {
  const segments = useMemo(() => sliceSegments(extent, scene), [extent, scene]);
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
      renderOrder={1}
    />
  );
}

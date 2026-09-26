"use client";

import { useMemo } from "react";
import type { SceneBounds } from "../camera/bounds";
import { surfaceExtentM } from "../terrain/grid";
import { ABSTRACT_SURFACE_LABEL, useSurfaceChoice } from "../terrain/surface";
import { verticalBadgeText } from "./badge";
import { makeRectList } from "./labelPlacement";
import { rulerDepthsKm } from "./ruler";
import { CornerNote } from "./CornerNote";
import type { GeoFeature, SceneMeta, Station } from "../types";
import { DepthRuler } from "./DepthRuler";
import { DepthSlices } from "./DepthSlices";
import { FeaturesLayer } from "./FeaturesLayer";
import { StationsLayer } from "./StationsLayer";
import { VerticalBadge } from "./VerticalBadge";

export interface ReferencesProps {
  bundle: { meta: { scene: SceneMeta }; stations: Station[]; features: GeoFeature[] };
  /** The camera's framed bounds (the ruler stands just outside them). */
  bounds: SceneBounds;
  /**
   * Plan view (WEB-07): looking straight down, the vertical ruler collapses to a point and the depth
   * slices stack into one outline, so both are left out; the plan's depth section panel carries depth.
   */
  planView?: boolean;
}

/** Everything that gives the events scale and place: ruler, slices, stations, features, VE badge. */
export function References({ bundle, bounds, planView = false }: ReferencesProps) {
  const { scene } = bundle.meta;
  const { asset, choice } = useSurfaceChoice(scene);
  const terrainMeta = choice === "terrain" && asset.status === "ready" ? asset.meta : null;
  // Slices span the surface extent, so they wait until the terrain has settled (no resize flash).
  const extent = useMemo(() => surfaceExtentM(terrainMeta, bounds), [terrainMeta, bounds]);
  // The ruler writes its label boxes (title + one per tick) here each frame (priority −1); the feature
  // labels, placed later in the frame, keep clear of them.
  const labelObstacles = useMemo(() => makeRectList(1 + rulerDepthsKm().length), []);
  return (
    <group name="references">
      {!planView && <DepthRuler scene={scene} bounds={bounds} obstacles={labelObstacles} />}
      {!planView && choice !== "none" && <DepthSlices scene={scene} extent={extent} />}
      <StationsLayer stations={bundle.stations} scene={scene} />
      <FeaturesLayer features={bundle.features} scene={scene} obstacles={labelObstacles} planView={planView} />
      <VerticalBadge scene={scene} />
      {choice === "slab" && (
        // Permanent while the slab stands in for the terrain: before the reveal, in every view, in plan.
        <CornerNote slot={verticalBadgeText(scene) ? 1 : 0} testId="abstract-surface-note">
          {ABSTRACT_SURFACE_LABEL}
        </CornerNote>
      )}
    </group>
  );
}

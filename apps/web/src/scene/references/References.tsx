"use client";

import { useMemo } from "react";
import type { SceneBounds } from "../camera/bounds";
import { surfaceExtentM } from "../terrain/grid";
import { useTerrainAsset } from "../terrain/load";
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
}

/** Everything that gives the events scale and place: ruler, slices, stations, features, VE badge. */
export function References({ bundle, bounds }: ReferencesProps) {
  const { scene } = bundle.meta;
  const terrain = useTerrainAsset();
  const terrainMeta = terrain.status === "ready" ? terrain.meta : null;
  // Slices span the surface extent, so they wait until the terrain has settled (no resize flash).
  const extent = useMemo(() => surfaceExtentM(terrainMeta, bounds), [terrainMeta, bounds]);
  return (
    <group name="references">
      <DepthRuler scene={scene} bounds={bounds} />
      {terrain.status !== "loading" && <DepthSlices scene={scene} extent={extent} />}
      <StationsLayer stations={bundle.stations} scene={scene} />
      <FeaturesLayer features={bundle.features} scene={scene} />
      <VerticalBadge scene={scene} />
    </group>
  );
}

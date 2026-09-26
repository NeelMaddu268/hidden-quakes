"use client";
// What the scene shows at the ground surface, decided in one place so the terrain layer and the depth
// ruler's note ("Abstract surface …") can never disagree.

import { useState } from "react";
import type { SceneMeta } from "../types";
import { chooseSurface, type SurfaceChoice } from "./grid";
import { urlForcesSlab, useTerrainAsset, type TerrainAsset } from "./load";
import { terrainMismatch } from "./meta";

/** Flip to true to ship the abstract slab instead of the DEM (docs/03 kill switch "Terrain"). */
export const FORCE_ABSTRACT_SLAB = false;

/** Shown under the depth ruler's title whenever the abstract slab stands in for the terrain. */
export const ABSTRACT_SURFACE_LABEL = "Abstract surface (no terrain model)";

export interface SurfaceState {
  asset: TerrainAsset;
  choice: SurfaceChoice;
  /** Why the loaded terrain can't be used with this bundle (origin/projection), or null. */
  mismatch: string | null;
}

/**
 * The surface decision: the baked terrain, the labelled abstract slab (forced by FORCE_ABSTRACT_SLAB
 * or `?terrain=slab`, or because the terrain failed to load or belongs to another origin), or nothing
 * yet while loading. Both callers share the one terrain load.
 */
export function useSurfaceChoice(scene: Pick<SceneMeta, "originLat" | "originLon" | "projection">): SurfaceState {
  const asset = useTerrainAsset();
  const [urlForced] = useState(urlForcesSlab);
  const mismatch = asset.status === "ready" ? terrainMismatch(asset.meta, scene) : null;
  const choice = chooseSurface(asset.status, { forced: FORCE_ABSTRACT_SLAB || urlForced, mismatch });
  return { asset, choice, mismatch };
}

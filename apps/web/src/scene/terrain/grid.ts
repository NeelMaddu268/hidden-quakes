// Terrain grid → scene-space mesh arrays, built once per terrain (never per frame).
// Vertex (r, c) is pixel (r, c) of the baked grid: e = eMin + c·Δe (west → east), n = nMax − r·Δn
// (row 0 = north), u = elevM − originElevM, then the one ENU → scene mapping in coords.ts.

import type { SceneBounds } from "../camera/bounds";
import { METERS_PER_UNIT, writeEnuToScene, verticalExaggerationOf } from "../coords";
import type { Enu, SceneMeta } from "../types";
import type { EnuBoundsM, TerrainMeta } from "./meta";

export interface TerrainGrid {
  width: number;
  height: number;
  /** xyz scene coordinates (km), 3 floats per vertex, row-major from the north-west corner. */
  positions: Float32Array;
  /** Two triangles per cell, counter-clockwise seen from above (normals point up). */
  index: Uint32Array;
}

/** Triangle indices for a width × height vertex grid, row-major, row 0 = north. */
export function gridIndex(width: number, height: number): Uint32Array {
  if (width < 2 || height < 2) throw new Error(`grid must be at least 2 × 2, got ${width} × ${height}`);
  const index = new Uint32Array((width - 1) * (height - 1) * 6);
  let k = 0;
  for (let r = 0; r < height - 1; r++) {
    for (let c = 0; c < width - 1; c++) {
      const a = r * width + c; // north-west
      const b = a + width; // south-west
      const d = a + 1; // north-east
      const e = b + 1; // south-east
      index[k++] = a;
      index[k++] = b;
      index[k++] = d;
      index[k++] = b;
      index[k++] = e;
      index[k++] = d;
    }
  }
  return index;
}

/** Scene positions for an elevation grid laid over `bounds` (pixel centres, metres). */
export function buildTerrainGrid(
  elevM: ArrayLike<number>,
  meta: { sizePx: readonly [number, number]; enuBounds: EnuBoundsM },
  scene: Pick<SceneMeta, "originElevM" | "verticalExaggeration">,
): TerrainGrid {
  const [width, height] = meta.sizePx;
  if (elevM.length !== width * height) {
    throw new Error(`terrain has ${elevM.length} heights for a ${width} × ${height} grid`);
  }
  const ve = verticalExaggerationOf(scene);
  const { eMin, eMax, nMin, nMax } = meta.enuBounds;
  const de = (eMax - eMin) / (width - 1);
  const dn = (nMax - nMin) / (height - 1);
  const positions = new Float32Array(width * height * 3);
  const enu: Enu = { e: 0, n: 0, u: 0 };
  for (let r = 0; r < height; r++) {
    enu.n = nMax - r * dn;
    for (let c = 0; c < width; c++) {
      const i = r * width + c;
      enu.e = eMin + c * de;
      enu.u = elevM[i] - scene.originElevM;
      writeEnuToScene(enu, ve, positions, i * 3);
    }
  }
  return { width, height, positions, index: gridIndex(width, height) };
}

/** Half-width of the fallback surface when there's no terrain to size it (the bake's default). */
export const FALLBACK_HALF_WIDTH_M = 8000;
/** Margin the fallback surface keeps around the framed data (m). */
export const FALLBACK_MARGIN_M = 1000;

/**
 * Horizontal extent of the surface (terrain or abstract slab) in ENU metres. The baked terrain's own
 * bounds when it's loaded; otherwise an origin-centred square that covers the framed data.
 */
export function surfaceExtentM(meta: Pick<TerrainMeta, "enuBounds"> | null, bounds: SceneBounds): EnuBoundsM {
  if (meta) {
    const { eMin, eMax, nMin, nMax } = meta.enuBounds;
    return { eMin, eMax, nMin, nMax };
  }
  // Scene x = e / 1000 and z = −n / 1000, so the framed box's reach from the origin is max |x|, |z|.
  const reachM =
    Math.max(Math.abs(bounds.min[0]), Math.abs(bounds.max[0]), Math.abs(bounds.min[2]), Math.abs(bounds.max[2])) *
    METERS_PER_UNIT;
  const half = Math.max(FALLBACK_HALF_WIDTH_M, reachM + FALLBACK_MARGIN_M);
  return { eMin: -half, eMax: half, nMin: -half, nMax: half };
}

/** A flat 2 × 2 grid at the site surface (refSurfaceElevM): the abstract-slab fallback. */
export function buildSlabGrid(
  extent: EnuBoundsM,
  scene: Pick<SceneMeta, "originElevM" | "refSurfaceElevM" | "verticalExaggeration">,
): TerrainGrid {
  const flat = new Float32Array(4).fill(scene.refSurfaceElevM);
  return buildTerrainGrid(flat, { sizePx: [2, 2], enuBounds: extent }, scene);
}

export type SurfaceChoice = "none" | "terrain" | "slab";

/**
 * What to draw at the surface. Nothing while the terrain is still loading (no flash of the slab
 * label); the baked terrain when it loaded and lines up with the bundle; the labelled abstract slab
 * when it's forced, failed to load, or belongs to a different origin.
 */
export function chooseSurface(
  status: "loading" | "ready" | "unavailable",
  opts: { forced: boolean; mismatch: string | null },
): SurfaceChoice {
  if (opts.forced) return "slab";
  if (status === "loading") return "none";
  if (status === "ready" && opts.mismatch === null) return "terrain";
  return "slab";
}

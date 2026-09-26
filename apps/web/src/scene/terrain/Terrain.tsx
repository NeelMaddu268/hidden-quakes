"use client";

import { Html } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import { DoubleSide, type ShaderMaterial } from "three";
import type { SceneBounds } from "../camera/bounds";
import { enuToScene, depthKmToSceneY, verticalExaggerationOf } from "../coords";
import { sceneFx } from "../fx";
import { LABEL_Z_RANGE, labelStyle } from "../references/labels";
import type { SceneMeta } from "../types";
import { buildSlabGrid, buildTerrainGrid, chooseSurface, surfaceExtentM, type TerrainGrid } from "./grid";
import { urlForcesSlab, useTerrainAsset } from "./load";
import {
  createTerrainUniforms,
  TERRAIN_FRAGMENT_SHADER,
  TERRAIN_VERTEX_SHADER,
  terrainDepthWrite,
  terrainOpacity,
  type TerrainUniforms,
} from "./material";
import { terrainMismatch, type EnuBoundsM } from "./meta";

/** Flip to true to ship the abstract slab instead of the DEM (docs/03 kill switch "Terrain"). */
export const FORCE_ABSTRACT_SLAB = false;

export const ABSTRACT_SURFACE_LABEL = "Abstract surface (terrain unavailable)";

/** Contour half-width in CSS pixels (scaled by DPR each frame). */
const CONTOUR_HALF_WIDTH_PX = 0.6;

interface SurfaceMeshProps {
  name: string;
  grid: TerrainGrid;
  /** One hillshade byte per vertex. */
  shade: Uint8Array;
  flatShadeValue: number;
  contours: boolean;
  originElevM: number;
  verticalExaggeration: number;
}

/**
 * The surface mesh, terrain or slab. Geometry is built once per terrain; every frame only writes the
 * opacity (from sceneFx, which the reveal drives), the DPR-scaled line width and the depth-write flag.
 */
function SurfaceMesh({ name, grid, shade, flatShadeValue, contours, originElevM, verticalExaggeration }: SurfaceMeshProps) {
  const material = useRef<ShaderMaterial>(null);
  const uniforms = useMemo(
    () => createTerrainUniforms({ flatShadeValue, contours, originElevM, verticalExaggeration }),
    [flatShadeValue, contours, originElevM, verticalExaggeration],
  );

  useFrame((state) => {
    const m = material.current;
    if (!m) return;
    const u = m.uniforms as TerrainUniforms;
    const opacity = terrainOpacity(sceneFx.terrainOpacity);
    u.uOpacity.value = opacity;
    u.uContourWidthPx.value = CONTOUR_HALF_WIDTH_PX * state.viewport.dpr;
    m.depthWrite = terrainDepthWrite(opacity);
  });

  return (
    <mesh name={name} renderOrder={0}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[grid.positions, 3]} />
        <bufferAttribute attach="attributes-aShade" args={[shade, 1, true]} />
        <bufferAttribute attach="index" args={[grid.index, 1]} />
      </bufferGeometry>
      <shaderMaterial
        ref={material}
        uniforms={uniforms}
        vertexShader={TERRAIN_VERTEX_SHADER}
        fragmentShader={TERRAIN_FRAGMENT_SHADER}
        transparent
        side={DoubleSide}
      />
    </mesh>
  );
}

const SLAB_SHADE = new Uint8Array([255, 255, 255, 255]);

function AbstractSlab({ extent, scene }: { extent: EnuBoundsM; scene: SceneMeta }) {
  const grid = useMemo(() => buildSlabGrid(extent, scene), [extent, scene]);
  const ve = verticalExaggerationOf(scene);
  // Label at the slab's south-west corner, on the surface.
  const corner = useMemo(
    (): [number, number, number] => {
      const [x, , z] = enuToScene({ e: extent.eMin, n: extent.nMin, u: 0 }, ve);
      return [x, depthKmToSceneY(0, scene), z];
    },
    [extent, scene, ve],
  );
  return (
    <group name="abstract-slab">
      <SurfaceMesh
        name="abstract-slab-surface"
        grid={grid}
        shade={SLAB_SHADE}
        flatShadeValue={255}
        contours={false}
        originElevM={scene.originElevM}
        verticalExaggeration={ve}
      />
      <Html position={corner} zIndexRange={LABEL_Z_RANGE} pointerEvents="none">
        <div style={{ ...labelStyle, transform: "translate(6px, -100%)" }}>{ABSTRACT_SURFACE_LABEL}</div>
      </Html>
    </group>
  );
}

export interface TerrainProps {
  scene: SceneMeta;
  /** Framed data bounds; size the abstract slab when there's no terrain to size it. */
  bounds: SceneBounds;
  /** Draw the labelled abstract slab even when the baked terrain is available. */
  forceSlab?: boolean;
}

/**
 * The ground surface: the baked DEM from public/terrain/, or the labelled abstract slab when the
 * terrain is forced off (`forceSlab`, `FORCE_ABSTRACT_SLAB`, `?terrain=slab`), fails to load, or was
 * baked for a different origin.
 */
export function Terrain({ scene, bounds, forceSlab = false }: TerrainProps) {
  const asset = useTerrainAsset();
  const [urlForced] = useState(urlForcesSlab);
  const ready = asset.status === "ready" ? asset : null;
  const mismatch = ready ? terrainMismatch(ready.meta, scene) : null;
  const choice = chooseSurface(asset.status, { forced: forceSlab || FORCE_ABSTRACT_SLAB || urlForced, mismatch });

  useEffect(() => {
    if (mismatch) console.error(`[terrain] ${mismatch}; showing the abstract surface instead`);
  }, [mismatch]);

  const ve = verticalExaggerationOf(scene);
  const grid = useMemo(
    () => (ready && choice === "terrain" ? buildTerrainGrid(ready.elevM, ready.meta, scene) : null),
    [ready, choice, scene],
  );
  const extent = useMemo(() => surfaceExtentM(ready?.meta ?? null, bounds), [ready, bounds]);

  if (choice === "terrain" && grid && ready) {
    return (
      <SurfaceMesh
        name="terrain"
        grid={grid}
        shade={ready.shade}
        flatShadeValue={ready.meta.hillshade.flatValue}
        contours
        originElevM={scene.originElevM}
        verticalExaggeration={ve}
      />
    );
  }
  if (choice === "slab") return <AbstractSlab extent={extent} scene={scene} />;
  return null;
}

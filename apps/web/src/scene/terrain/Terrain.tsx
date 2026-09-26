"use client";

import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { BufferAttribute, BufferGeometry, DoubleSide, type ShaderMaterial } from "three";
import type { SceneBounds } from "../camera/bounds";
import { verticalExaggerationOf } from "../coords";
import { sceneFx } from "../fx";
import type { SceneMeta } from "../types";
import { buildSlabGrid, buildTerrainGrid, surfaceExtentM, type TerrainGrid } from "./grid";
import {
  CONTOUR_HALF_WIDTH_PX,
  createTerrainUniforms,
  TERRAIN_DEPTH_FRAGMENT_SHADER,
  TERRAIN_FRAGMENT_SHADER,
  TERRAIN_VERTEX_SHADER,
  terrainOpacity,
  type TerrainUniforms,
} from "./material";
import type { EnuBoundsM } from "./meta";
import { RENDER_ORDER } from "./renderOrder";
import { useSurfaceChoice } from "./surface";

/** Culling for the terrain passes. DoubleSide: the side view looks at the surface from below. */
export const TERRAIN_SIDE = DoubleSide;

function buildGeometry(grid: TerrainGrid, shade: Uint8Array): BufferGeometry {
  const g = new BufferGeometry();
  g.setAttribute("position", new BufferAttribute(grid.positions, 3));
  g.setAttribute("aShade", new BufferAttribute(shade, 1, true));
  g.setIndex(new BufferAttribute(grid.index, 1));
  g.computeBoundingSphere();
  return g;
}

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
 * The surface, terrain or slab: a depth-only pre-pass, then the colour pass (normal alpha blending,
 * depth test on, no depth writes), both drawn after every underground layer; renderOrder.ts explains
 * why. Both passes share one geometry (one GPU upload). Geometry is built once per terrain; every
 * frame only writes the opacity (from sceneFx, which the reveal drives) and the DPR-scaled line width.
 */
function SurfaceMesh({ name, grid, shade, flatShadeValue, contours, originElevM, verticalExaggeration }: SurfaceMeshProps) {
  const material = useRef<ShaderMaterial>(null);
  const uniforms = useMemo(
    () => createTerrainUniforms({ flatShadeValue, contours, originElevM, verticalExaggeration }),
    [flatShadeValue, contours, originElevM, verticalExaggeration],
  );
  const geometry = useMemo(() => buildGeometry(grid, shade), [grid, shade]);
  // Passed as a prop (not JSX), so R3F doesn't own it: dispose it ourselves.
  useEffect(() => () => geometry.dispose(), [geometry]);

  useFrame((state) => {
    const m = material.current;
    if (!m) return;
    const u = m.uniforms as TerrainUniforms;
    u.uOpacity.value = terrainOpacity(sceneFx.terrainOpacity);
    u.uContourWidthPx.value = CONTOUR_HALF_WIDTH_PX * state.viewport.dpr;
  });

  return (
    <group name={name}>
      {/* Depth pre-pass: the same vertex shader (identical depths), no colour, pushed back a hair so the
          colour pass wins its own depth test. Transparent, so it sorts after the underground layers. */}
      <mesh name={`${name}-depth`} geometry={geometry} renderOrder={RENDER_ORDER.terrainDepth}>
        <shaderMaterial
          uniforms={uniforms}
          vertexShader={TERRAIN_VERTEX_SHADER}
          fragmentShader={TERRAIN_DEPTH_FRAGMENT_SHADER}
          transparent
          colorWrite={false}
          depthWrite
          polygonOffset
          polygonOffsetFactor={1}
          polygonOffsetUnits={1}
          side={TERRAIN_SIDE}
        />
      </mesh>
      <mesh name={`${name}-surface`} geometry={geometry} renderOrder={RENDER_ORDER.terrain}>
        <shaderMaterial
          ref={material}
          uniforms={uniforms}
          vertexShader={TERRAIN_VERTEX_SHADER}
          fragmentShader={TERRAIN_FRAGMENT_SHADER}
          transparent
          depthWrite={false}
          side={TERRAIN_SIDE}
        />
      </mesh>
    </group>
  );
}

const SLAB_SHADE = new Uint8Array([255, 255, 255, 255]);

/** The flat abstract slab; its note sits under the depth ruler's title (references/DepthRuler). */
function AbstractSlab({ extent, scene }: { extent: EnuBoundsM; scene: SceneMeta }) {
  const grid = useMemo(() => buildSlabGrid(extent, scene), [extent, scene]);
  return (
    <SurfaceMesh
      name="abstract-slab"
      grid={grid}
      shade={SLAB_SHADE}
      flatShadeValue={255}
      contours={false}
      originElevM={scene.originElevM}
      verticalExaggeration={verticalExaggerationOf(scene)}
    />
  );
}

export interface TerrainProps {
  scene: SceneMeta;
  /** Framed data bounds; size the abstract slab when there's no terrain to size it. */
  bounds: SceneBounds;
}

/**
 * The ground surface: the baked DEM from public/terrain/, or the abstract slab when the terrain is
 * forced off (FORCE_ABSTRACT_SLAB in surface.ts, `?terrain=slab`), fails to load, or was baked for a
 * different origin. The depth ruler shows the matching "Abstract surface" note.
 */
export function Terrain({ scene, bounds }: TerrainProps) {
  const { asset, choice, mismatch } = useSurfaceChoice(scene);
  const ready = asset.status === "ready" ? asset : null;

  useEffect(() => {
    if (mismatch) console.error(`[terrain] ${mismatch}; showing the abstract surface instead`);
  }, [mismatch]);

  const ve = verticalExaggerationOf(scene);
  const grid = useMemo(
    () => (ready && choice === "terrain" ? buildTerrainGrid(ready.elevM, ready.meta, scene) : null),
    [ready, choice, scene],
  );
  const extent = useMemo(() => surfaceExtentM(choice === "terrain" ? ready?.meta ?? null : null, bounds), [ready, choice, bounds]);

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

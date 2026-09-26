"use client";

import { colors } from "@hq/visualization";
import { useFrame } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef } from "react";
import { CustomBlending, DoubleSide, MaxEquation, type InstancedMesh, type ShaderMaterial } from "three";
import { LOOK } from "../look";
import { createHaloUniforms, HALO_FRAGMENT_SHADER, HALO_VERTEX_SHADER, type HaloUniforms } from "./haloMaterial";
import type { HaloInstances } from "./halos";

const NO_RAYCAST = () => undefined;

export interface HalosLayerProps {
  halos: HaloInstances;
  /** Scene y of the site surface and fog density per scene unit (the same fog as the event glyphs). */
  surfaceY: number;
  depthFog: number;
  /** Called every frame with the halo uniforms; must not allocate. */
  drive: (uniforms: HaloUniforms) => void;
}

/**
 * Tier A uncertainty ellipsoids (WEB-04). Overlaps combine with MAX blending (never summing), both
 * faces draw so a camera inside a halo still sees it, and the mesh is skipped entirely (no vertex
 * work) whenever the filter look has halos at 0, i.e. outside STRICT.
 */
export function HalosLayer({ halos, surfaceY, depthFog, drive }: HalosLayerProps) {
  const mesh = useRef<InstancedMesh>(null);
  const material = useRef<ShaderMaterial>(null);
  const uniforms = useMemo(() => createHaloUniforms(colors.strictHalo), []);

  useLayoutEffect(() => {
    const m = mesh.current;
    if (!m) return;
    (m.instanceMatrix.array as Float32Array).set(halos.matrices);
    m.instanceMatrix.needsUpdate = true;
    m.computeBoundingSphere();
  }, [halos]);

  useFrame(() => {
    const mat = material.current;
    const m = mesh.current;
    if (!mat || !m) return;
    const u = mat.uniforms as HaloUniforms;
    u.uSurfaceY.value = surfaceY;
    u.uDepthFog.value = depthFog;
    drive(u);
    m.visible = u.uOpacity.value > 0.001;
  });

  if (halos.count === 0) return null;
  return (
    <instancedMesh
      key={halos.count}
      ref={mesh}
      name="tier-a-halos"
      args={[undefined, undefined, halos.count]}
      frustumCulled={false}
      raycast={NO_RAYCAST}
      renderOrder={0}
      visible={false}
    >
      <icosahedronGeometry args={[1, LOOK.halos.sphereDetail]}>
        <instancedBufferAttribute attach="attributes-aAppearAt" args={[halos.appearAt, 1]} />
        <instancedBufferAttribute attach="attributes-aTime" args={[halos.times, 1]} />
      </icosahedronGeometry>
      <shaderMaterial
        ref={material}
        uniforms={uniforms}
        vertexShader={HALO_VERTEX_SHADER}
        fragmentShader={HALO_FRAGMENT_SHADER}
        transparent
        depthWrite={false}
        side={DoubleSide}
        blending={CustomBlending}
        blendEquation={MaxEquation}
      />
    </instancedMesh>
  );
}

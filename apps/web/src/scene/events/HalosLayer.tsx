"use client";

import { colors } from "@hq/visualization";
import { useFrame } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef } from "react";
import { AdditiveBlending, type InstancedMesh, type ShaderMaterial } from "three";
import { createHaloUniforms, HALO_FRAGMENT_SHADER, HALO_VERTEX_SHADER, type HaloUniforms } from "./haloMaterial";
import type { HaloInstances } from "./halos";

/** Icosphere detail: 2 → 320 triangles per halo, smooth enough for a soft rim. */
const HALO_SPHERE_DETAIL = 2;
const NO_RAYCAST = () => undefined;

export interface HalosLayerProps {
  halos: HaloInstances;
  /** Called every frame with the halo uniforms; must not allocate. */
  drive: (uniforms: HaloUniforms) => void;
}

/** Tier A uncertainty ellipsoids (WEB-04). Invisible (and fragment-free) unless the filter is STRICT. */
export function HalosLayer({ halos, drive }: HalosLayerProps) {
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
    if (mat) drive(mat.uniforms as HaloUniforms);
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
    >
      <icosahedronGeometry args={[1, HALO_SPHERE_DETAIL]}>
        <instancedBufferAttribute attach="attributes-aAppearAt" args={[halos.appearAt, 1]} />
      </icosahedronGeometry>
      <shaderMaterial
        ref={material}
        uniforms={uniforms}
        vertexShader={HALO_VERTEX_SHADER}
        fragmentShader={HALO_FRAGMENT_SHADER}
        transparent
        depthWrite={false}
        blending={AdditiveBlending}
      />
    </instancedMesh>
  );
}

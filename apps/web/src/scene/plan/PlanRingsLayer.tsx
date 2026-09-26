"use client";

import { colors } from "@hq/visualization";
import { useFrame } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef } from "react";
import { CustomBlending, DoubleSide, MaxEquation, type InstancedMesh, type ShaderMaterial } from "three";
import type { PlanHaloInstances } from "./halos";
import {
  createRingUniforms,
  RING_FRAGMENT_SHADER,
  RING_VERTEX_SHADER,
  writeRingMatrices,
  type RingUniforms,
} from "./ringMaterial";

const NO_RAYCAST = () => undefined;

/**
 * Horizontal uncertainty rings for the plan view (WEB-07). Unlike the 3D ellipsoids they need only
 * hErrM, so a Tier A event without a vertical error still shows its horizontal uncertainty; an event
 * without a usable hErrM gets no ring. Skipped entirely (no vertex work) while their opacity is 0.
 */
export function PlanRingsLayer({ rings, drive }: { rings: PlanHaloInstances; drive: (u: RingUniforms) => void }) {
  const mesh = useRef<InstancedMesh>(null);
  const material = useRef<ShaderMaterial>(null);
  const uniforms = useMemo(() => createRingUniforms(colors.strictHalo), []);

  useLayoutEffect(() => {
    const m = mesh.current;
    if (!m) return;
    writeRingMatrices(rings.positions, rings.radii, m.instanceMatrix.array as Float32Array);
    m.instanceMatrix.needsUpdate = true;
    m.computeBoundingSphere();
  }, [rings]);

  useFrame(() => {
    const mat = material.current;
    const m = mesh.current;
    if (!mat || !m) return;
    const u = mat.uniforms as RingUniforms;
    drive(u);
    m.visible = u.uOpacity.value > 0.001;
  });

  if (rings.count === 0) return null;
  return (
    <instancedMesh
      key={rings.count}
      ref={mesh}
      name="plan-uncertainty-rings"
      args={[undefined, undefined, rings.count]}
      frustumCulled={false}
      raycast={NO_RAYCAST}
      renderOrder={0}
    >
      <planeGeometry args={[1, 1]}>
        <instancedBufferAttribute attach="attributes-aAppearAt" args={[rings.appearAt, 1]} />
      </planeGeometry>
      <shaderMaterial
        ref={material}
        uniforms={uniforms}
        vertexShader={RING_VERTEX_SHADER}
        fragmentShader={RING_FRAGMENT_SHADER}
        transparent
        depthWrite={false}
        side={DoubleSide}
        blending={CustomBlending}
        blendEquation={MaxEquation}
      />
    </instancedMesh>
  );
}

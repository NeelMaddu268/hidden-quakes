"use client";

import { useFrame } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef } from "react";
import { AdditiveBlending, type InstancedMesh, type ShaderMaterial } from "three";
import type { EventInstances } from "./instances";
import {
  createEventUniforms,
  EVENT_FRAGMENT_SHADER,
  EVENT_VERTEX_SHADER,
  writeInstanceMatrices,
  type EventUniforms,
} from "./material";

/**
 * The default InstancedMesh raycast would test 1 km unit quads, not the billboards the shader draws.
 * Picking is done in screen space instead (WEB-05), so the mesh opts out of raycasting.
 */
const NO_RAYCAST = () => undefined;

export interface EventsLayerProps {
  instances: EventInstances;
  color: string;
  /** Base radius in km before tier scaling. */
  size: number;
  /** Minimum on-screen radius, CSS pixels (scaled by DPR internally). */
  minPx: number;
  /** Maximum on-screen radius, CSS pixels (scaled by DPR internally). */
  maxPx: number;
  /** Called every frame with this layer's uniforms; must not allocate. */
  drive: (uniforms: EventUniforms, deltaS: number) => void;
  renderOrder?: number;
  name?: string;
}

/**
 * One InstancedMesh of event glyphs. The per-instance attributes are the bundle's typed arrays, set
 * once; every frame only writes a few uniforms. R3F disposes the geometry and material on unmount.
 */
export function EventsLayer({ instances, color, size, minPx, maxPx, drive, renderOrder, name }: EventsLayerProps) {
  const mesh = useRef<InstancedMesh>(null);
  const material = useRef<ShaderMaterial>(null);
  const uniforms = useMemo(() => createEventUniforms({ color, size, minPx, maxPx }), [color, size, minPx, maxPx]);
  // three.js caches a material's uniforms object when it compiles the program, so new uniforms need a
  // new material: key it on everything that rebuilds them.
  const materialKey = `${color}|${size}|${minPx}|${maxPx}`;

  useLayoutEffect(() => {
    const m = mesh.current;
    if (!m) return;
    writeInstanceMatrices(instances.positions, m.instanceMatrix.array as Float32Array);
    m.instanceMatrix.needsUpdate = true;
    m.computeBoundingSphere();
  }, [instances]);

  useFrame((state, delta) => {
    const mat = material.current;
    if (!mat) return;
    const u = mat.uniforms as EventUniforms;
    u.uViewportHeight.value = state.size.height * state.viewport.dpr;
    u.uMinPx.value = minPx * state.viewport.dpr;
    u.uMaxPx.value = maxPx * state.viewport.dpr;
    drive(u, delta);
  });

  return (
    <instancedMesh
      key={instances.count}
      ref={mesh}
      name={name}
      args={[undefined, undefined, instances.count]}
      frustumCulled={false}
      renderOrder={renderOrder}
      raycast={NO_RAYCAST}
    >
      <planeGeometry args={[1, 1]}>
        <instancedBufferAttribute attach="attributes-aTier" args={[instances.tiers, 1]} />
        <instancedBufferAttribute attach="attributes-aScale" args={[instances.scales, 1]} />
        <instancedBufferAttribute attach="attributes-aRevealAt" args={[instances.revealAt, 1]} />
        <instancedBufferAttribute attach="attributes-aTime" args={[instances.times, 1]} />
      </planeGeometry>
      <shaderMaterial
        key={materialKey}
        ref={material}
        uniforms={uniforms}
        vertexShader={EVENT_VERTEX_SHADER}
        fragmentShader={EVENT_FRAGMENT_SHADER}
        transparent
        depthWrite={false}
        blending={AdditiveBlending}
      />
    </instancedMesh>
  );
}

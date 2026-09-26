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

export interface EventsLayerProps {
  instances: EventInstances;
  color: string;
  /** Base radius in km before tier scaling. */
  size: number;
  /** Minimum on-screen radius, CSS pixels (scaled by DPR internally). */
  minPx: number;
  /** Intensity multiplier (cores above 1.0 bloom). */
  glow?: number;
  /** Scene y of the site surface, for depth fog. */
  surfaceY?: number;
  /** Depth fog density per km below the surface (0 = off). */
  depthFog?: number;
  /** Called every frame with this layer's uniforms; must not allocate. */
  drive: (uniforms: EventUniforms, deltaS: number) => void;
  renderOrder?: number;
  name?: string;
}

/**
 * One InstancedMesh of event glyphs. The per-instance attributes are the bundle's typed arrays, set
 * once; every frame only writes a few uniforms. R3F disposes the geometry and material on unmount.
 */
export function EventsLayer({
  instances,
  color,
  size,
  minPx,
  glow = 1,
  surfaceY = 0,
  depthFog = 0,
  drive,
  renderOrder,
  name,
}: EventsLayerProps) {
  const mesh = useRef<InstancedMesh>(null);
  const material = useRef<ShaderMaterial>(null);
  const uniforms = useMemo(
    () => createEventUniforms({ color, size, minPx, glow, depthFog, surfaceY }),
    [color, size, minPx, glow, depthFog, surfaceY],
  );

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
    >
      <planeGeometry args={[1, 1]}>
        <instancedBufferAttribute attach="attributes-aTier" args={[instances.tiers, 1]} />
        <instancedBufferAttribute attach="attributes-aScale" args={[instances.scales, 1]} />
        <instancedBufferAttribute attach="attributes-aRevealAt" args={[instances.revealAt, 1]} />
        <instancedBufferAttribute attach="attributes-aTime" args={[instances.times, 1]} />
      </planeGeometry>
      <shaderMaterial
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

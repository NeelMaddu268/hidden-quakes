"use client";

import { Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { colors } from "@hq/visualization";
import { useEffect, useMemo, useRef } from "react";
import { Color, type ShaderMaterial } from "three";
import type { SceneMeta, Station } from "../types";
import { buildStationGlyphs, stationIssues } from "./stations";

/** Glyph size in CSS px (scaled by DPR each frame). */
const STATION_SIZE_PX = 11;
/** Glyphs are nudged this far toward the camera (km) so a sensor on a 30 m DEM isn't swallowed by it. */
const SURFACE_BIAS_KM = 0.04;

const STATION_VERTEX_SHADER = /* glsl */ `
  attribute float aShape;
  attribute float aAlpha;

  uniform float uSizePx;
  uniform float uBiasKm;

  varying float vShape;
  varying float vAlpha;

  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    mv.xyz += normalize(-mv.xyz) * uBiasKm;
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uSizePx * (aShape > 1.5 ? 0.8 : 1.0);
    vShape = aShape;
    vAlpha = aAlpha;
  }
`;

// Shapes in point-sprite space (y down): 0 inverted triangle (surface station), 1 diamond (borehole
// sensor at depth), 2 ring (borehole wellhead). Signed distances, antialiased with fwidth.
const STATION_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;

  varying float vShape;
  varying float vAlpha;

  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d;
    if (vShape < 0.5) {
      float top = -0.62 - p.y;
      float side = dot(vec2(abs(p.x), p.y) - vec2(0.82, -0.62), normalize(vec2(1.47, 0.82)));
      d = max(top, side);
    } else if (vShape < 1.5) {
      d = (abs(p.x) + abs(p.y)) * 0.7071 - 0.5;
    } else {
      d = abs(length(p) - 0.55) - 0.16;
    }
    float w = max(fwidth(d), 1e-4);
    float a = (1.0 - smoothstep(-w, w, d)) * vAlpha;
    if (a < 0.01) discard;
    gl_FragColor = vec4(uColor, a);
    #include <colorspace_fragment>
  }
`;

/** Surface stations as inverted triangles; borehole sensors at true depth with a line to the wellhead. */
export function StationsLayer({ stations, scene }: { stations: Station[]; scene: SceneMeta }) {
  const glyphs = useMemo(() => buildStationGlyphs(stations, scene), [stations, scene]);
  const material = useRef<ShaderMaterial>(null);
  const uniforms = useMemo(
    () => ({
      uColor: { value: new Color(colors.station) },
      uSizePx: { value: STATION_SIZE_PX },
      uBiasKm: { value: SURFACE_BIAS_KM },
    }),
    [],
  );

  useEffect(() => {
    const issues = stationIssues(stations, scene);
    if (issues.length) console.warn(`[stations] inconsistent sensor elevations: ${issues.join("; ")}`);
  }, [stations, scene]);

  useFrame((state) => {
    const m = material.current;
    if (m) m.uniforms.uSizePx.value = STATION_SIZE_PX * state.viewport.dpr;
  });

  if (glyphs.count === 0) return null;
  return (
    <group name="stations">
      <points key={glyphs.count} frustumCulled={false} renderOrder={3}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[glyphs.positions, 3]} />
          <bufferAttribute attach="attributes-aShape" args={[glyphs.shapes, 1]} />
          <bufferAttribute attach="attributes-aAlpha" args={[glyphs.alphas, 1]} />
        </bufferGeometry>
        <shaderMaterial
          ref={material}
          uniforms={uniforms}
          vertexShader={STATION_VERTEX_SHADER}
          fragmentShader={STATION_FRAGMENT_SHADER}
          transparent
          depthWrite={false}
        />
      </points>
      {glyphs.boreholeSegments.length > 0 && (
        <Line
          name="borehole-lines"
          points={glyphs.boreholeSegments}
          segments
          color={colors.station}
          lineWidth={1.25}
          transparent
          opacity={0.85}
          depthWrite={false}
          renderOrder={3}
        />
      )}
    </group>
  );
}

"use client";

import { Line } from "@react-three/drei";
import { SceneHtml as Html } from "./SceneHtml";
import { useFrame } from "@react-three/fiber";
import { colors } from "@hq/visualization";
import { useEffect, useMemo, useRef } from "react";
import { AdditiveBlending, Color, type ShaderMaterial } from "three";
import { verticalExaggerationOf } from "../coords";
import type { GeoFeature, SceneMeta } from "../types";
import { RENDER_ORDER } from "../terrain/renderOrder";
import { featureGeometry, featureIssue, featureLabelAnchor, featureStyle, type FeatureGeometry, type FeatureStyle } from "./features";
import { LABEL_Z_RANGE, labelStyle } from "./labels";

// The geothermal reference reads in every view, over the terrain and through it: an annotation
// layer (no depth test), drawn after the terrain and the reference lines.
const LINE_WIDTH_PX = 1.75;
const GLOW_WIDTH_PX = 7;
const GLOW_OPACITY = 0.18;
/** Dash and gap lengths for unverified lines, in scene km. */
const DASH_KM = 0.14;
const GAP_KM = 0.09;
const MARKER_SIZE_PX = 20;

const MARKER_VERTEX_SHADER = /* glsl */ `
  attribute float aDashed;
  uniform float uSizePx;
  varying float vDashed;
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uSizePx;
    vDashed = aDashed;
  }
`;

// Facility marker: solid core + ring + soft halo when verified; a dashed hollow ring when approximate.
const MARKER_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;
  varying float vDashed;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r = length(p);
    if (r > 1.0) discard;
    float aa = max(fwidth(r), 1e-4);
    float ring = 1.0 - smoothstep(0.07 - aa, 0.07 + aa, abs(r - 0.52));
    float core = 1.0 - smoothstep(0.22 - aa, 0.22 + aa, r);
    float halo = (1.0 - r) * (1.0 - r) * 0.5;
    if (vDashed > 0.5) {
      float ang = atan(p.y, p.x);
      ring *= step(0.0, sin(ang * 6.0));
      core = 0.0;
      halo *= 0.5;
    }
    float a = clamp(core + ring + halo, 0.0, 1.0);
    gl_FragColor = vec4(uColor * (1.0 + 0.35 * core), a);
    #include <colorspace_fragment>
  }
`;

interface Drawable {
  feature: GeoFeature;
  style: FeatureStyle;
  geom: FeatureGeometry;
  anchor: [number, number, number];
}

function FeatureLine({ points, style }: { points: [number, number, number][]; style: FeatureStyle }) {
  return (
    <>
      {style.glow && (
        <Line
          points={points}
          color={colors.geo}
          lineWidth={GLOW_WIDTH_PX}
          dashed={style.dashed}
          dashSize={DASH_KM}
          gapSize={GAP_KM}
          transparent
          opacity={GLOW_OPACITY}
          blending={AdditiveBlending}
          depthTest={false}
          depthWrite={false}
          renderOrder={RENDER_ORDER.featureGlow}
        />
      )}
      <Line
        points={points}
        color={colors.geo}
        lineWidth={LINE_WIDTH_PX}
        dashed={style.dashed}
        dashSize={DASH_KM}
        gapSize={GAP_KM}
        transparent
        depthTest={false}
        depthWrite={false}
        renderOrder={RENDER_ORDER.feature}
      />
    </>
  );
}

function FacilityMarkers({ points }: { points: { at: [number, number, number]; dashed: boolean }[] }) {
  const material = useRef<ShaderMaterial>(null);
  const { positions, dashed } = useMemo(() => {
    const positions = new Float32Array(points.length * 3);
    const dashed = new Float32Array(points.length);
    points.forEach((p, i) => {
      positions.set(p.at, i * 3);
      dashed[i] = p.dashed ? 1 : 0;
    });
    return { positions, dashed };
  }, [points]);
  const uniforms = useMemo(() => ({ uColor: { value: new Color(colors.geo) }, uSizePx: { value: MARKER_SIZE_PX } }), []);
  useFrame((state) => {
    const m = material.current;
    if (m) m.uniforms.uSizePx.value = MARKER_SIZE_PX * state.viewport.dpr;
  });
  return (
    <points key={points.length} frustumCulled={false} renderOrder={RENDER_ORDER.feature}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-aDashed" args={[dashed, 1]} />
      </bufferGeometry>
      <shaderMaterial
        ref={material}
        uniforms={uniforms}
        vertexShader={MARKER_VERTEX_SHADER}
        fragmentShader={MARKER_FRAGMENT_SHADER}
        transparent
        depthTest={false}
        depthWrite={false}
      />
    </points>
  );
}

/**
 * Wells (polylines through `path`), facilities (point markers) and boundaries (closed polylines) from
 * features.json, in `colors.geo`. Verified locations are solid; unverified ones are dashed and
 * labelled "approximate". Labels are each feature's own name.
 */
export function FeaturesLayer({ features, scene }: { features: GeoFeature[]; scene: SceneMeta }) {
  const ve = verticalExaggerationOf(scene);
  const drawables = useMemo(() => {
    const out: Drawable[] = [];
    for (const feature of features) {
      const geom = featureGeometry(feature, ve);
      if (!geom) continue;
      out.push({ feature, style: featureStyle(feature), geom, anchor: featureLabelAnchor(geom) });
    }
    return out;
  }, [features, ve]);
  const markers = useMemo(
    () =>
      drawables.flatMap((d) => (d.geom.type === "point" ? [{ at: d.geom.at, dashed: d.style.dashed }] : [])),
    [drawables],
  );

  useEffect(() => {
    const issues = features.map(featureIssue).filter((m): m is string => m !== null);
    if (issues.length) console.error(`[features] not drawn: ${issues.join("; ")}`);
  }, [features]);

  if (drawables.length === 0) return null;
  return (
    <group name="features">
      {drawables.map((d) =>
        d.geom.type === "line" ? <FeatureLine key={d.feature.id} points={d.geom.points} style={d.style} /> : null,
      )}
      {markers.length > 0 && <FacilityMarkers points={markers} />}
      {drawables.map((d) => (
        <Html key={`label-${d.feature.id}`} position={d.anchor} zIndexRange={LABEL_Z_RANGE} pointerEvents="none">
          <div
            data-testid="feature-label"
            style={{ ...labelStyle, color: colors.geo, opacity: 0.9, transform: "translate(10px, -50%)" }}
          >
            {d.style.label}
          </div>
        </Html>
      ))}
    </group>
  );
}

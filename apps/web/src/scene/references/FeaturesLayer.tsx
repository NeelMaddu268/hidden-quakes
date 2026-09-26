"use client";

import { Line } from "@react-three/drei";
import { SceneHtml as Html } from "./SceneHtml";
import { useFrame, useThree } from "@react-three/fiber";
import { colors } from "@hq/visualization";
import { useEffect, useMemo, useRef } from "react";
import { AdditiveBlending, Color, Vector3, type ShaderMaterial } from "three";
import { verticalExaggerationOf } from "../coords";
import type { GeoFeature, SceneMeta } from "../types";
import { sectionPanelRect } from "../plan/layout";
import { RENDER_ORDER } from "../terrain/renderOrder";
import { featureGeometry, featureIssue, featureLabelAnchor, featureStyle, type FeatureGeometry, type FeatureStyle } from "./features";
import {
  appendRects,
  cameraMoved,
  clearRects,
  LABEL_GAP_PX,
  LABEL_SLOTS,
  labelPriority,
  leaderOf,
  makeCameraTrack,
  makeLabelPlacement,
  makeRectList,
  placeLabels,
  pushRect,
  slotOffsetX,
  slotOffsetY,
  type LabelPlacement,
  type RectList,
} from "./labelPlacement";
import { LABEL_Z_RANGE, labelStyle } from "./labels";
import { useLabelElements } from "./useLabelElements";

// The geothermal reference reads in every view, over the terrain and through it: an annotation
// layer (no depth test), drawn after the terrain and the reference lines.
const LINE_WIDTH_PX = 1.75;
const GLOW_WIDTH_PX = 7;
const GLOW_OPACITY = 0.18;
/** Dash and gap lengths for unverified lines, in scene km. */
const DASH_KM = 0.14;
const GAP_KM = 0.09;
const MARKER_SIZE_PX = 20;
/** Leader from an anchor to a label moved off its line (labelPlacement), fainter than the label. */
const LEADER_OPACITY = 0.55;

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

/** Per-frame scratch for FeatureLabels (lives in a ref; rebuilt when the label count changes). */
interface LabelFrameState {
  placement: LabelPlacement;
  fixed: RectList;
  v: Vector3;
  /** Slot and size last written to each label's style (−1 = never). */
  written: Int8Array;
  writtenW: Float32Array;
  writtenH: Float32Array;
  /** Sum of both LabelElements versions at the last write (a remount forces a rewrite). */
  version: number;
  /** Last frame's camera (world + projection matrices, viewport size), to tell a moving camera. */
  lastCamera: Float64Array;
}

/** `obstacleCapacity`: rects the ruler can publish; one more for the plan panel. */
function makeLabelFrameState(n: number, obstacleCapacity: number): LabelFrameState {
  return {
    placement: makeLabelPlacement(n),
    fixed: makeRectList(obstacleCapacity + 1),
    v: new Vector3(),
    written: new Int8Array(n).fill(-1),
    writtenW: new Float32Array(n),
    writtenH: new Float32Array(n),
    version: -1,
    lastCamera: makeCameraTrack(),
  };
}

interface FeatureLabelsProps {
  labels: { id: string; text: string; anchor: [number, number, number] }[];
  /** Screen boxes (CSS px) the labels keep clear of: the depth ruler's labels, written earlier each frame. */
  obstacles?: RectList;
  /** Plan view: the labels also keep clear of the depth-section panel. */
  planView: boolean;
}

/**
 * The feature names, placed each frame so they never sit on each other, on the depth ruler's labels,
 * on another feature's anchor, under the plan panel, or off screen (labelPlacement). A label moved off
 * its anchor's line gets a thin leader back to it. Style is written only when a label's slot or size
 * changes, so a still camera writes nothing.
 */
function FeatureLabels({ labels, obstacles, planView }: FeatureLabelsProps) {
  const n = labels.length;
  const els = useLabelElements();
  const leaders = useLabelElements();
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  const panel = useMemo(() => (planView ? sectionPanelRect(width, height) : null), [planView, width, height]);
  const frame = useRef<LabelFrameState | null>(null);

  useFrame(({ camera, size }) => {
    if (!frame.current || frame.current.placement.n !== n) {
      frame.current = makeLabelFrameState(n, obstacles ? obstacles.rects.length / 4 : 0);
    }
    const st = frame.current;
    const { placement: p, fixed, v } = st;
    clearRects(fixed);
    if (obstacles) appendRects(fixed, obstacles);
    if (panel) pushRect(fixed, panel.left, panel.top, panel.width, panel.height);
    for (let i = 0; i < n; i++) {
      const a = labels[i].anchor;
      v.set(a[0], a[1], a[2]).project(camera);
      p.ax[i] = (v.x * 0.5 + 0.5) * size.width;
      p.ay[i] = (-v.y * 0.5 + 0.5) * size.height;
      p.active[i] = v.z < 1 && Number.isFinite(v.x) && Number.isFinite(v.y) ? 1 : 0;
      p.w[i] = els.width(i);
      p.h[i] = els.height(i);
    }
    // Hold slots only mid-move (no hopping); a still camera gets the canonical layout.
    const moving = cameraMoved(st.lastCamera, camera.matrixWorld.elements, camera.projectionMatrix.elements, size.width, size.height);
    placeLabels(p, fixed, size.width, size.height, moving);

    const version = els.version + leaders.version;
    const remount = version !== st.version;
    st.version = version;
    for (let i = 0; i < n; i++) {
      const s = p.slot[i];
      if (s < 0) continue;
      if (!remount && s === st.written[i] && p.w[i] === st.writtenW[i] && p.h[i] === st.writtenH[i]) continue;
      st.written[i] = s;
      st.writtenW[i] = p.w[i];
      st.writtenH[i] = p.h[i];
      const slot = LABEL_SLOTS[s];
      const el = els.el(i);
      if (el) el.style.transform = `translate(${slotOffsetX(slot, p.w[i])}px, ${slotOffsetY(slot, p.h[i])}px)`;
      const leader = leaders.el(i);
      if (leader) {
        const { length, angle } = leaderOf(slot, p.h[i]);
        leader.style.display = length > 0 ? "block" : "none";
        leader.style.width = `${length}px`;
        leader.style.transform = `rotate(${angle}rad)`;
      }
    }
  });

  return (
    <>
      {labels.map((l, i) => (
        <Html key={`label-${l.id}`} position={l.anchor} zIndexRange={LABEL_Z_RANGE} pointerEvents="none">
          <div
            ref={leaders.ref(i)}
            data-testid="feature-label-leader"
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              height: 1,
              width: 0,
              display: "none",
              background: colors.geo,
              opacity: LEADER_OPACITY,
              transformOrigin: "0 0",
              pointerEvents: "none",
            }}
          />
          <div
            ref={els.ref(i)}
            data-testid="feature-label"
            style={{ ...labelStyle, color: colors.geo, opacity: 0.9, transform: `translate(${LABEL_GAP_PX}px, -50%)` }}
          >
            {l.text}
          </div>
        </Html>
      ))}
    </>
  );
}

/**
 * Wells (polylines through `path`), facilities (point markers) and boundaries (closed polylines) from
 * features.json, in `colors.geo`. Verified locations are solid; unverified ones are dashed and
 * labelled "approximate". Labels are each feature's own name, placed clear of each other and of the
 * depth ruler's labels (`obstacles`).
 */
export function FeaturesLayer({
  features,
  scene,
  obstacles,
  planView = false,
}: {
  features: GeoFeature[];
  scene: SceneMeta;
  obstacles?: RectList;
  planView?: boolean;
}) {
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

  // Placement priority: facilities and wells keep their default spot before boundaries (stable sort).
  const labels = useMemo(
    () =>
      drawables
        .map((d, i) => ({ i, id: d.feature.id, text: d.style.label, anchor: d.anchor, rank: labelPriority(d.feature.kind) }))
        .sort((a, b) => a.rank - b.rank || a.i - b.i)
        .map(({ id, text, anchor }) => ({ id, text, anchor })),
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
      <FeatureLabels labels={labels} obstacles={obstacles} planView={planView} />
    </group>
  );
}

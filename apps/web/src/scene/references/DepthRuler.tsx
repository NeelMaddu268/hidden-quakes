"use client";

import { Line } from "@react-three/drei";
import { SceneHtml as Html } from "./SceneHtml";
import { useFrame } from "@react-three/fiber";
import { colors } from "@hq/visualization";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { Matrix4, Vector3, Vector4, type Group } from "three";
import type { SceneBounds } from "../camera/bounds";
import type { SceneMeta } from "../types";
import { RENDER_ORDER } from "../terrain/renderOrder";
import { sceneFx } from "../fx";
import { TIMELINE } from "../reveal/timeline";
import {
  clearRects,
  pushRulerTickRect,
  pushRulerTitleRect,
  RULER_TICK_GAP_PX,
  RULER_TITLE_INSET_PX,
  RULER_TITLE_RISE_PX,
  type RectList,
} from "./labelPlacement";
import { LABEL_PLATE, LABEL_Z_RANGE, labelStyle, numericLabelStyle } from "./labels";
import { useLabelElements } from "./useLabelElements";
import { declutterLabels, rulerAnchor, rulerLayout, rulerRevealOpacity, stickyTitleT } from "./ruler";

/** Approximate tick-label box (CSS px): labels whose boxes would overlap are hidden (plan view). */
const LABEL_BOX_W_PX = 40;
const LABEL_BOX_H_PX = 15;
/** The title never rides higher than this below the canvas top (CSS px); it sits above its anchor. */
const TITLE_TOP_MARGIN_PX = 34;

interface Scratch {
  v: Vector3;
  clipTop: Vector4;
  clipBottom: Vector4;
  viewProj: Matrix4;
  xs: Float32Array;
  ys: Float32Array;
  mask: Uint8Array;
}

function makeScratch(n: number): Scratch {
  return {
    v: new Vector3(),
    clipTop: new Vector4(),
    clipBottom: new Vector4(),
    viewProj: new Matrix4(),
    xs: new Float32Array(n),
    ys: new Float32Array(n),
    mask: new Uint8Array(n),
  };
}

/**
 * From the site surface down to the deepest displayed event (1 km ticks), just west of the framed data. Drawn on top of the
 * terrain (no depth test) so it reads in every camera preset. The title is SceneMeta.depthLabel; it
 * sits at the top of the ruler and slides down the spine when the surface is out of frame, so the
 * label stays visible whenever the ruler is. Overlapping tick labels are hidden (plan view). Each frame
 * the shown title and tick-label boxes are written to `obstacles`, so the feature labels (placed after
 * this hook) keep clear of them; the list is empty while the ruler is hidden.
 */
export function DepthRuler({
  scene,
  bounds,
  maxDepthKm,
  obstacles,
}: {
  scene: SceneMeta;
  bounds: SceneBounds;
  /** The deepest displayed event, rounded up to a tick (ruler.ts → rulerMaxDepthKm). */
  maxDepthKm: number;
  obstacles?: RectList;
}) {
  const layout = useMemo(() => rulerLayout(scene, rulerAnchor(bounds), maxDepthKm), [scene, bounds, maxDepthKm]);
  // DOM labels: [0] is the title, [1 + i] is tick i. Sizes feed the obstacle boxes.
  const labels = useLabelElements();
  const title = useRef<Group>(null);
  const ruler = useRef<Group>(null);
  const lastOpacity = useRef(-1);
  const lastLabelsVersion = useRef(-1);
  const scratch = useRef<Scratch | null>(null);
  useLayoutEffect(() => {
    scratch.current = makeScratch(layout.ticks.length);
  }, [layout.ticks.length]);

  // Leaving (plan view, unmount) clears this ruler's obstacles.
  useEffect(
    () => () => {
      if (obstacles) clearRects(obstacles);
    },
    [obstacles],
  );

  // Priority −1: runs before drei's <Html> frame hooks, so labels follow the camera without a lag.
  useFrame(({ camera, size }) => {
    const n = layout.ticks.length;
    if (!scratch.current) return;

    // Fade with the reveal (hidden on the pre-reveal frame). DOM and material are touched only when the
    // quantized opacity changes (or a label element (re)mounts), so a settled scene writes nothing.
    const alpha = Math.round(rulerRevealOpacity(sceneFx.terrainOpacity, TIMELINE.terrainFade.to) * 100) / 100;
    if (alpha !== lastOpacity.current || labels.version !== lastLabelsVersion.current) {
      lastOpacity.current = alpha;
      lastLabelsVersion.current = labels.version;
      const g = ruler.current;
      if (g) {
        g.visible = alpha > 0;
        g.traverse((o) => {
          const m = (o as unknown as { material?: { opacity: number; transparent: boolean } }).material;
          if (m) {
            m.transparent = true;
            m.opacity = alpha;
          }
        });
      }
      const css = alpha > 0 ? String(alpha) : "0";
      for (let i = 0; i <= n; i++) {
        const el = labels.el(i);
        if (el) el.style.opacity = css;
      }
    }
    if (obstacles) clearRects(obstacles);
    if (alpha === 0) return;
    const { v, clipTop, clipBottom, viewProj, xs, ys, mask } = scratch.current;

    // Sticky title: find where the spine enters the viewport (exact, in clip space).
    const top = layout.segments[0];
    const bottom = layout.segments[1];
    camera.updateMatrixWorld();
    viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    clipTop.set(top[0], top[1], top[2], 1).applyMatrix4(viewProj);
    clipBottom.set(bottom[0], bottom[1], bottom[2], 1).applyMatrix4(viewProj);
    const ndcTop = 1 - (2 * TITLE_TOP_MARGIN_PX) / Math.max(size.height, 1);
    const t = stickyTitleT(clipTop.y, clipTop.w, clipBottom.y, clipBottom.w, ndcTop);
    const titleY = top[1] + t * (bottom[1] - top[1]);
    title.current?.position.set(top[0], titleY, top[2]);
    if (obstacles && labels.width(0) > 0) {
      v.set(top[0], titleY, top[2]).project(camera);
      const x = (v.x * 0.5 + 0.5) * size.width;
      pushRulerTitleRect(obstacles, x, (-v.y * 0.5 + 0.5) * size.height, labels.width(0), labels.height(0));
    }

    // Tick-label declutter against projected screen positions (no allocation).
    for (let i = 0; i < n; i++) {
      const p = layout.ticks[i].labelAt;
      v.set(p[0], p[1], p[2]).project(camera);
      xs[i] = (v.x * 0.5 + 0.5) * size.width;
      ys[i] = (-v.y * 0.5 + 0.5) * size.height;
    }
    declutterLabels(xs, ys, LABEL_BOX_W_PX, LABEL_BOX_H_PX, mask);
    for (let i = 0; i < n; i++) {
      const el = labels.el(i + 1);
      if (!el) continue;
      const vis = mask[i] ? "visible" : "hidden";
      if (el.style.visibility !== vis) el.style.visibility = vis;
      if (obstacles && mask[i] && labels.width(i + 1) > 0) {
        pushRulerTickRect(obstacles, xs[i], ys[i], labels.width(i + 1), labels.height(i + 1));
      }
    }
  }, -1);

  return (
    <group name="depth-ruler">
      <group ref={ruler}>
        <Line
          points={layout.segments}
          segments
          color={colors.contour}
          lineWidth={1.5}
          transparent
          depthTest={false}
          depthWrite={false}
          renderOrder={RENDER_ORDER.ruler}
        />
      </group>
      <group ref={title} position={layout.titleAt}>
        <Html zIndexRange={LABEL_Z_RANGE} pointerEvents="none">
          <div
            ref={labels.ref(0)}
            data-testid="depth-ruler-title"
            style={{
              ...labelStyle,
              ...LABEL_PLATE,
              opacity: 0,
              transform: `translate(-${RULER_TITLE_INSET_PX}px, calc(-100% - ${RULER_TITLE_RISE_PX}px))`,
            }}
          >
            <div>{layout.title}</div>
          </div>
        </Html>
      </group>
      {layout.ticks.map((t, i) => (
        <Html key={t.depthKm} position={t.labelAt} zIndexRange={LABEL_Z_RANGE} pointerEvents="none">
          <div
            ref={labels.ref(i + 1)}
            data-testid="depth-ruler-tick"
            style={{ ...numericLabelStyle, ...LABEL_PLATE, opacity: 0, transform: `translate(calc(-100% - ${RULER_TICK_GAP_PX}px), -50%)` }}
          >
            {t.label}
          </div>
        </Html>
      ))}
    </group>
  );
}

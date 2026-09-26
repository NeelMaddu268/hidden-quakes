"use client";

import { Html, Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { colors } from "@hq/visualization";
import { useLayoutEffect, useMemo, useRef } from "react";
import { Matrix4, Vector3, Vector4, type Group } from "three";
import type { SceneBounds } from "../camera/bounds";
import type { SceneMeta } from "../types";
import { RENDER_ORDER } from "../terrain/renderOrder";
import { ABSTRACT_SURFACE_LABEL } from "../terrain/surface";
import { LABEL_Z_RANGE, labelStyle, numericLabelStyle } from "./labels";
import { declutterLabels, rulerAnchor, rulerLayout, stickyTitleT } from "./ruler";

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
 * 0–6 km below the site surface with 1 km ticks, just west of the framed data. Drawn on top of the
 * terrain (no depth test) so it reads in every camera preset. The title is SceneMeta.depthLabel; it
 * sits at the top of the ruler and slides down the spine when the surface is out of frame, so the
 * label stays visible whenever the ruler is. Overlapping tick labels are hidden (plan view).
 */
export function DepthRuler({ scene, bounds, abstractSurface = false }: { scene: SceneMeta; bounds: SceneBounds; abstractSurface?: boolean }) {
  const layout = useMemo(() => rulerLayout(scene, rulerAnchor(bounds)), [scene, bounds]);
  const labelEls = useRef<(HTMLDivElement | null)[]>([]);
  const title = useRef<Group>(null);
  const scratch = useRef<Scratch | null>(null);
  useLayoutEffect(() => {
    scratch.current = makeScratch(layout.ticks.length);
  }, [layout.ticks.length]);

  // Priority −1: runs before drei's <Html> frame hooks, so labels follow the camera without a lag.
  useFrame(({ camera, size }) => {
    const n = layout.ticks.length;
    if (!scratch.current) return;
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
    title.current?.position.set(top[0], top[1] + t * (bottom[1] - top[1]), top[2]);

    // Tick-label declutter against projected screen positions (no allocation).
    for (let i = 0; i < n; i++) {
      const p = layout.ticks[i].labelAt;
      v.set(p[0], p[1], p[2]).project(camera);
      xs[i] = (v.x * 0.5 + 0.5) * size.width;
      ys[i] = (-v.y * 0.5 + 0.5) * size.height;
    }
    declutterLabels(xs, ys, LABEL_BOX_W_PX, LABEL_BOX_H_PX, mask);
    for (let i = 0; i < n; i++) {
      const el = labelEls.current[i];
      if (!el) continue;
      const vis = mask[i] ? "visible" : "hidden";
      if (el.style.visibility !== vis) el.style.visibility = vis;
    }
  }, -1);

  return (
    <group name="depth-ruler">
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
      <group ref={title} position={layout.titleAt}>
        <Html zIndexRange={LABEL_Z_RANGE} pointerEvents="none">
          <div data-testid="depth-ruler-title" style={{ ...labelStyle, transform: "translate(-4px, calc(-100% - 10px))" }}>
            <div>{layout.title}</div>
            {abstractSurface && <div data-testid="abstract-surface-note">{ABSTRACT_SURFACE_LABEL}</div>}
          </div>
        </Html>
      </group>
      {layout.ticks.map((t, i) => (
        <Html key={t.depthKm} position={t.labelAt} zIndexRange={LABEL_Z_RANGE} pointerEvents="none">
          <div
            ref={(el) => {
              labelEls.current[i] = el;
            }}
            style={{ ...numericLabelStyle, transform: "translate(calc(-100% - 5px), -50%)" }}
          >
            {t.label}
          </div>
        </Html>
      ))}
    </group>
  );
}

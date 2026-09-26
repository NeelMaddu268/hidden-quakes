"use client";

import { useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import { Matrix4 } from "three";
import { useDemo } from "../../state/demo";
import type { EventInstances } from "../events/instances";
import { POP_SCALE } from "../events/material";
import { LOOK } from "../look";
import { sceneFx } from "../fx";
import type { CatalogEvent } from "../types";
import { betterHit, isClick, pickNearest, PICK_THRESHOLD_PX, type PickHit, type PickQuery } from "./pick";
import { candidatePickable, publicSelectTargets } from "./selection";

export interface PickerProps {
  candidates: EventInstances;
  publicEvents: EventInstances;
  /** The bundle's catalog, in the same order as `publicEvents` (for `matchedEventId`). */
  catalog: readonly CatalogEvent[];
  /** Base glyph radius of each layer in km (the EventsLayer `size`), so close-up glyphs are hit anywhere. */
  sizeKm: { candidate: number; public: number };
}

/**
 * Click-to-select on the event layers (WEB-05). Listens on the canvas element: a pointer that goes down
 * and up without dragging picks the nearest visible glyph within PICK_THRESHOLD_PX and calls
 * `select(id)`. A public-catalog point selects its matched candidate. Empty space does nothing (Esc
 * deselects). Work happens only on click; nothing runs per frame.
 */
export function Picker({ candidates, publicEvents, catalog, sizeKm }: PickerProps) {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const publicTargets = useMemo(
    () => publicSelectTargets(catalog, candidates.indexById),
    [catalog, candidates],
  );

  useEffect(() => {
    const el = gl.domElement;
    const viewProj = new Matrix4();
    let down: { x: number; y: number; t: number; id: number } | null = null;

    const onDown = (e: PointerEvent) => {
      down = e.button === 0 && e.isPrimary ? { x: e.clientX, y: e.clientY, t: e.timeStamp, id: e.pointerId } : null;
    };
    const onCancel = () => {
      down = null;
    };
    /** The candidate event a pointer at (clientX, clientY) would select, or null for empty space. */
    const pickAt = (clientX: number, clientY: number): string | null => {
      const rect = el.getBoundingClientRect();
      camera.updateMatrixWorld();
      viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      const base: PickQuery = {
        viewProj: viewProj.elements,
        width: rect.width,
        height: rect.height,
        x: clientX - rect.left,
        y: clientY - rect.top,
        thresholdPx: PICK_THRESHOLD_PX,
        maxRadiusPx: LOOK.maxGlyphPx * POP_SCALE,
        focalPx: (camera.projectionMatrix.elements[5] * rect.height) / 2,
      };
      const { phase, filter } = useDemo.getState();
      const elapsed = sceneFx.revealElapsedS;
      const cand = pickNearest(candidates.positions, {
        ...base,
        radius: (i) => sizeKm.candidate * candidates.scales[i],
        visible: (i) => candidatePickable(phase, filter, elapsed, candidates.tiers[i], candidates.appearAt[i]),
      });
      const pub = pickNearest(publicEvents.positions, {
        ...base,
        radius: (i) => sizeKm.public * publicEvents.scales[i],
        visible: (i) => publicTargets[i] !== null,
      });
      const best: PickHit | null = pub && betterHit(pub, cand) ? pub : cand;
      if (!best) return null;
      return best === pub ? publicTargets[best.index] : candidates.ids[best.index];
    };

    const onUp = (e: PointerEvent) => {
      const start = down;
      down = null;
      if (!start || e.pointerId !== start.id) return;
      if (!isClick(start, { x: e.clientX, y: e.clientY, t: e.timeStamp })) return;
      const id = pickAt(e.clientX, e.clientY);
      if (id) useDemo.getState().select(id); // empty space keeps the current selection
    };

    // Hover affordance: a pointer cursor over anything a click would select. At most one pick per
    // animation frame, only while the pointer moves with no button held (orbiting keeps its cursor).
    let hoverRaf = 0;
    let hoverX = 0;
    let hoverY = 0;
    const onMove = (e: PointerEvent) => {
      if (e.buttons !== 0 || e.pointerType === "touch") return;
      hoverX = e.clientX;
      hoverY = e.clientY;
      if (hoverRaf) return;
      hoverRaf = requestAnimationFrame(() => {
        hoverRaf = 0;
        el.style.cursor = pickAt(hoverX, hoverY) ? "pointer" : "";
      });
    };
    const onLeave = () => {
      el.style.cursor = "";
    };

    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onCancel);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerleave", onLeave);
    return () => {
      cancelAnimationFrame(hoverRaf);
      el.style.cursor = "";
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onCancel);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerleave", onLeave);
    };
  }, [gl, camera, candidates, publicEvents, publicTargets, sizeKm.candidate, sizeKm.public]);

  return null;
}

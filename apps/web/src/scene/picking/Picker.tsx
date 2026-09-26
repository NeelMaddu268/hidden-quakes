"use client";

import { useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import { Matrix4 } from "three";
import { useDemo } from "../../state/demo";
import type { EventInstances } from "../events/instances";
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
    const onUp = (e: PointerEvent) => {
      const start = down;
      down = null;
      if (!start || e.pointerId !== start.id) return;
      if (!isClick(start, { x: e.clientX, y: e.clientY, t: e.timeStamp })) return;

      const rect = el.getBoundingClientRect();
      camera.updateMatrixWorld();
      viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      const base: PickQuery = {
        viewProj: viewProj.elements,
        width: rect.width,
        height: rect.height,
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        thresholdPx: PICK_THRESHOLD_PX,
        focalPx: (camera.projectionMatrix.elements[5] * rect.height) / 2,
      };
      const { phase, filter, select } = useDemo.getState();
      const elapsed = sceneFx.revealElapsedS;
      const cand = pickNearest(candidates.positions, {
        ...base,
        radius: (i) => sizeKm.candidate * candidates.scales[i],
        visible: (i) => candidatePickable(phase, filter, elapsed, candidates.tiers[i], candidates.revealAt[i]),
      });
      const pub = pickNearest(publicEvents.positions, {
        ...base,
        radius: (i) => sizeKm.public * publicEvents.scales[i],
        visible: (i) => publicTargets[i] !== null,
      });
      const best: PickHit | null = pub && betterHit(pub, cand) ? pub : cand;
      if (!best) return; // empty space: keep the current selection
      const id = best === pub ? publicTargets[best.index] : candidates.ids[best.index];
      if (id) select(id);
    };

    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onCancel);
    return () => {
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onCancel);
    };
  }, [gl, camera, candidates, publicEvents, publicTargets, sizeKm.candidate, sizeKm.public]);

  return null;
}

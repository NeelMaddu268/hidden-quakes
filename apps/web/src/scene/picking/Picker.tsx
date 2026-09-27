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
import { shownAt } from "../time/clock";
import { betterHit, isClick, pickNearest, PICK_THRESHOLD_PX, type PickQuery } from "./pick";
import { setHover } from "./hover";
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
 * deselects). Hovering publishes the event under the pointer for HoverTooltip (any drawn glyph,
 * including public points before the reveal, which a click doesn't select). Work happens only on
 * click and on pointer moves (at most once per frame); nothing runs per frame otherwise.
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
    type Hit = { kind: "candidate" | "public"; index: number };
    /**
     * The glyph nearest a pointer at (clientX, clientY), or null for empty space. "select" sees what a
     * click may select; "hover" also sees public points a click skips (before the reveal, or with no
     * matched candidate): the tooltip describes whatever is drawn.
     */
    const hitAt = (clientX: number, clientY: number, mode: "select" | "hover"): Hit | null => {
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
      const now = sceneFx.timeNowRel; // time mode: only events already shown at tNow (WEB-06)
      const cand = pickNearest(candidates.positions, {
        ...base,
        radius: (i) => sizeKm.candidate * candidates.scales[i],
        visible: (i) =>
          candidatePickable(phase, filter, elapsed, candidates.tiers[i], candidates.appearAt[i]) &&
          shownAt(candidates.times[i], now),
      });
      const pub = pickNearest(publicEvents.positions, {
        ...base,
        radius: (i) => sizeKm.public * publicEvents.scales[i],
        visible: (i) =>
          (mode === "hover" || (phase !== "public" && publicTargets[i] !== null)) && shownAt(publicEvents.times[i], now),
      });
      if (pub && betterHit(pub, cand)) return { kind: "public", index: pub.index };
      return cand ? { kind: "candidate", index: cand.index } : null;
    };
    /** The candidate event a click at (clientX, clientY) would select, or null for empty space. */
    const pickAt = (clientX: number, clientY: number): string | null => {
      const hit = hitAt(clientX, clientY, "select");
      if (!hit) return null;
      return hit.kind === "public" ? publicTargets[hit.index] : candidates.ids[hit.index];
    };

    const onUp = (e: PointerEvent) => {
      const start = down;
      down = null;
      if (!start || e.pointerId !== start.id) return;
      if (!isClick(start, { x: e.clientX, y: e.clientY, t: e.timeStamp })) return;
      const id = pickAt(e.clientX, e.clientY);
      if (id) useDemo.getState().select(id); // empty space keeps the current selection
    };

    // Hover: a pointer cursor over anything a click would select, and the tooltip's event. At most one
    // pick per animation frame, only while the pointer moves with no button held (orbiting keeps its
    // cursor and hides the tooltip).
    let hoverRaf = 0;
    let hoverX = 0;
    let hoverY = 0;
    const onMove = (e: PointerEvent) => {
      if (e.buttons !== 0 || e.pointerType === "touch") {
        setHover(null);
        return;
      }
      hoverX = e.clientX;
      hoverY = e.clientY;
      if (hoverRaf) return;
      hoverRaf = requestAnimationFrame(() => {
        hoverRaf = 0;
        el.style.cursor = pickAt(hoverX, hoverY) ? "pointer" : "";
        const hit = hitAt(hoverX, hoverY, "hover");
        setHover(
          hit === null
            ? null
            : {
                kind: hit.kind,
                id: hit.kind === "public" ? catalog[hit.index]!.id : candidates.ids[hit.index]!,
                x: hoverX,
                y: hoverY,
              },
        );
      });
    };
    const onLeave = () => {
      el.style.cursor = "";
      setHover(null);
    };
    // Anything that moves the glyphs under a still pointer ends the hover: a drag or zoom starting, and
    // the scene changing state (reveal, filter, time mode, view, reset).
    const onGesture = () => setHover(null);
    const offStore = useDemo.subscribe((s, prev) => {
      if (s.phase !== prev.phase || s.filter !== prev.filter || s.timeMode !== prev.timeMode || s.view !== prev.view) {
        setHover(null);
      }
    });

    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onCancel);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerleave", onLeave);
    el.addEventListener("pointerdown", onGesture);
    el.addEventListener("wheel", onGesture, { passive: true });
    return () => {
      cancelAnimationFrame(hoverRaf);
      offStore();
      setHover(null);
      el.style.cursor = "";
      el.removeEventListener("pointerdown", onGesture);
      el.removeEventListener("wheel", onGesture);
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onCancel);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerleave", onLeave);
    };
  }, [gl, camera, candidates, publicEvents, publicTargets, catalog, sizeKm.candidate, sizeKm.public]);

  return null;
}

"use client";

import { useFrame } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import { onDemoReset, useDemo } from "../../state/demo";
import { sceneFx } from "../fx";
import { createFilterFade, type FilterFade } from "./fade";

/** Runs after the reveal clock (−2) and before every layer that reads `sceneFx.filterLook` (0). */
export const FILTER_FRAME_PRIORITY = -1;

/**
 * Eases the PUBLIC / ALL / STRICT look toward the store's filter over `motion.state` and publishes it
 * as `sceneFx.filterLook` for the event, public and halo layers. reset() snaps straight to the start
 * look so the start frame is exact.
 */
export function FilterDriver() {
  const fade = useRef<FilterFade | null>(null);

  useEffect(() => {
    const f = createFilterFade(useDemo.getState().filter);
    fade.current = f;
    Object.assign(sceneFx.filterLook, f.current);
    const off = onDemoReset(() => {
      f.snap(useDemo.getState().filter);
      Object.assign(sceneFx.filterLook, f.current);
    });
    return () => {
      off();
      fade.current = null;
    };
  }, []);

  useFrame((_, delta) => {
    const f = fade.current;
    if (!f) return;
    f.step(useDemo.getState().filter, delta);
    const look = sceneFx.filterLook;
    const cur = f.current;
    look.tierA = cur.tierA;
    look.tierB = cur.tierB;
    look.tierC = cur.tierC;
    look.candidates = cur.candidates;
    look.publicLayer = cur.publicLayer;
    look.halos = cur.halos;
  }, FILTER_FRAME_PRIORITY);

  return null;
}

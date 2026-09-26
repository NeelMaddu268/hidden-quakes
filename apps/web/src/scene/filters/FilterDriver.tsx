"use client";

import { useFrame } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import { onDemoReset } from "../../state/demo";
import { createFilterLookDriver, type FilterLookDriver } from "./driver";

/** Runs after the reveal clock (−2) and before every layer that reads `sceneFx.filterLook` (0). */
export const FILTER_FRAME_PRIORITY = -1;

/** Mounts the filter-look driver (./driver.ts) on the frame loop and on reset(). */
export function FilterDriver() {
  const driver = useRef<FilterLookDriver | null>(null);

  useEffect(() => {
    const d = createFilterLookDriver();
    driver.current = d;
    const off = onDemoReset(() => d.onReset());
    return () => {
      off();
      driver.current = null;
    };
  }, []);

  useFrame((_, delta) => {
    driver.current?.step(delta);
  }, FILTER_FRAME_PRIORITY);

  return null;
}

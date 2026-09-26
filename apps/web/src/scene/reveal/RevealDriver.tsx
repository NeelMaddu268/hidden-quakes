"use client";

import { useFrame } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import { useDemo } from "../../state/demo";
import { createRevealMachine, type RevealMachine } from "./machine";

/**
 * Runs the reveal clock (scene/reveal/machine.ts). Store transitions arrive through a synchronous
 * subscription, so `reset(); reveal()` in one tick restarts cleanly; frames advance the clock before
 * any layer reads `sceneFx` (negative priority runs first).
 */
export const REVEAL_FRAME_PRIORITY = -2;

export function RevealDriver() {
  const machine = useRef<RevealMachine | null>(null);

  useEffect(() => {
    const m = createRevealMachine();
    machine.current = m;
    const { phase } = useDemo.getState();
    m.onPhase(phase, phase);
    const unsubscribe = useDemo.subscribe((s, prev) => {
      if (s.phase !== prev.phase) m.onPhase(prev.phase, s.phase);
    });
    return () => {
      unsubscribe();
      machine.current = null;
    };
  }, []);

  useFrame((_, delta) => {
    machine.current?.step(delta);
  }, REVEAL_FRAME_PRIORITY);

  return null;
}

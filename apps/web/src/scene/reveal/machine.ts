// The reveal clock as a small state machine, kept free of React and WebGL so it's testable against the
// real store. RevealDriver.tsx feeds it store transitions and frame deltas; it writes the shared
// per-frame values (scene/fx.ts) and advances the store's counter clock.

import { easeOutCubic, motion } from "@hq/visualization";
import { finishReveal, setRevealProgress, type DemoPhase } from "../../state/demo";
import { INITIAL_SCENE_FX, sceneFx } from "../fx";
import {
  appearTimeOf,
  bloomBoostAt,
  isRevealDone,
  revealProgressAt,
  terrainOpacityAt,
  TIMELINE,
} from "./timeline";

/**
 * Longest step the clock takes per frame. A hitch (tab switch, GC pause) slows the reveal down for a
 * frame instead of skipping a chunk of it, so every run shows the same choreography.
 */
export const MAX_STEP_S = 0.1;

type Mode = "idle" | "revealing" | "revealed" | "resetting";

export interface RevealMachine {
  readonly mode: Mode;
  /** Seconds on the reveal clock (0 when idle). */
  readonly elapsedS: number;
  /**
   * Call once on mount with the store's current phase and progress. Mounting mid-reveal resumes the
   * clock where the counter is (appearTimeOf is the exact inverse), so the counter never runs backwards.
   */
  sync(phase: DemoPhase, revealProgress: number): void;
  /** Call on every store phase change. */
  onPhase(prev: DemoPhase, next: DemoPhase): void;
  /** Advance by one frame. Never allocates. */
  step(deltaS: number): void;
}

export function createRevealMachine(): RevealMachine {
  let mode: Mode = "idle";
  let elapsed = 0;
  let resetT = 0;
  let resetFrom = 1;
  /** Terrain opacity when this reveal started (1 from a settled start frame; less if reset was mid-fade). */
  let revealFrom = 1;

  const applyRevealAt = (t: number) => {
    sceneFx.revealElapsedS = t;
    sceneFx.terrainOpacity = terrainOpacityAt(t, revealFrom);
    sceneFx.bloomBoost = bloomBoostAt(t);
  };

  const startRevealAt = (t: number) => {
    mode = "revealing";
    elapsed = t;
    revealFrom = t > 0 ? INITIAL_SCENE_FX.terrainOpacity : sceneFx.terrainOpacity;
    applyRevealAt(t);
  };

  const settleRevealed = () => {
    mode = "revealed";
    sceneFx.revealElapsedS = TIMELINE.endS;
    sceneFx.terrainOpacity = TIMELINE.terrainFade.to;
    sceneFx.bloomBoost = INITIAL_SCENE_FX.bloomBoost;
  };

  const settleIdle = () => {
    mode = "idle";
    elapsed = 0;
    sceneFx.revealElapsedS = INITIAL_SCENE_FX.revealElapsedS;
    sceneFx.terrainOpacity = INITIAL_SCENE_FX.terrainOpacity;
    sceneFx.bloomBoost = INITIAL_SCENE_FX.bloomBoost;
  };

  return {
    get mode() {
      return mode;
    },
    get elapsedS() {
      return elapsed;
    },

    sync(phase, revealProgress) {
      if (phase === "revealing") startRevealAt(revealProgress > 0 ? appearTimeOf(revealProgress) : 0);
      else if (phase === "revealed") settleRevealed();
      else settleIdle();
    },

    onPhase(prev, next) {
      if (next === "revealing" && (prev !== "revealing" || mode !== "revealing")) {
        startRevealAt(0);
        return;
      }
      if (next === "revealed" && mode !== "revealed") {
        settleRevealed();
        return;
      }
      if (next === "public" && prev !== "public") {
        // reset(): fade the terrain back up alongside the camera's return (motion.scene), then hold
        // the exact start values.
        mode = "resetting";
        elapsed = 0;
        resetT = 0;
        resetFrom = sceneFx.terrainOpacity;
        sceneFx.revealElapsedS = INITIAL_SCENE_FX.revealElapsedS;
        sceneFx.bloomBoost = INITIAL_SCENE_FX.bloomBoost;
        return;
      }
      if (next === "public" && mode !== "resetting") settleIdle();
    },

    step(deltaS) {
      const dt = deltaS > MAX_STEP_S ? MAX_STEP_S : deltaS < 0 ? 0 : deltaS;
      if (mode === "revealing") {
        elapsed += dt;
        applyRevealAt(elapsed);
        setRevealProgress(revealProgressAt(elapsed));
        if (isRevealDone(elapsed)) {
          settleRevealed();
          finishReveal();
        }
      } else if (mode === "resetting") {
        resetT += dt;
        const k = easeOutCubic(resetT / (motion.scene / 1000));
        sceneFx.terrainOpacity = resetFrom + (INITIAL_SCENE_FX.terrainOpacity - resetFrom) * k;
        if (k >= 1) settleIdle();
      }
    },
  };
}

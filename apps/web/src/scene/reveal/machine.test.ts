import { beforeEach, describe, expect, it } from "vitest";
import { initialDemoState, useDemo, type DemoPhase } from "../../state/demo";
import { INITIAL_SCENE_FX, resetSceneFx, sceneFx } from "../fx";
import { createRevealMachine, MAX_STEP_S, type RevealMachine } from "./machine";
import { TIMELINE } from "./timeline";

const store = () => useDemo.getState();

/** Wires the machine to the store the way RevealDriver does (synchronous subscription). */
function wired(): { m: RevealMachine; unsubscribe: () => void } {
  const m = createRevealMachine();
  m.onPhase(store().phase, store().phase);
  const unsubscribe = useDemo.subscribe((s, prev) => {
    if (s.phase !== prev.phase) m.onPhase(prev.phase, s.phase);
  });
  return { m, unsubscribe };
}

function runFrames(m: RevealMachine, dt: number, until: () => boolean, max = 100_000): number {
  let frames = 0;
  while (!until() && frames < max) {
    m.step(dt);
    frames++;
  }
  return frames;
}

beforeEach(() => {
  store().reset();
  resetSceneFx();
});

describe("reveal machine", () => {
  it("revealProgress hits exactly 1 and phase becomes revealed after the ~7 s choreography", () => {
    const { m, unsubscribe } = wired();
    store().reveal();
    const frames = runFrames(m, 1 / 60, () => store().phase === "revealed");
    unsubscribe();
    expect(store().revealProgress).toBe(1);
    expect(store().phase).toBe("revealed");
    expect(frames).toBeGreaterThanOrEqual(Math.floor(TIMELINE.endS * 60) - 1);
    expect(frames).toBeLessThanOrEqual(Math.ceil(TIMELINE.endS * 60) + 1);
    expect(sceneFx.terrainOpacity).toBe(TIMELINE.terrainFade.to);
    expect(sceneFx.bloomBoost).toBe(1);
  });

  it("keeps the counter at 0 until events start, and at 1 through the settle before revealed", () => {
    const { m, unsubscribe } = wired();
    store().reveal();
    runFrames(m, 1 / 60, () => m.elapsedS >= 0.95);
    expect(store().revealProgress).toBe(0);
    runFrames(m, 1 / 60, () => m.elapsedS >= 6.05);
    expect(store().revealProgress).toBe(1);
    expect(store().phase).toBe("revealing"); // still settling
    unsubscribe();
  });

  it("is identical on every run (same frame sequence → same values)", () => {
    const trace = () => {
      store().reset();
      resetSceneFx();
      const { m, unsubscribe } = wired();
      store().reveal();
      const out: number[] = [];
      runFrames(m, 1 / 60, () => {
        out.push(store().revealProgress, sceneFx.terrainOpacity, sceneFx.bloomBoost);
        return store().phase === "revealed";
      });
      unsubscribe();
      return out;
    };
    expect(trace()).toEqual(trace());
  });

  it("is time-based: 30 fps and 120 fps agree at the same elapsed time", () => {
    const at = (fps: number, seconds: number) => {
      store().reset();
      resetSceneFx();
      const { m, unsubscribe } = wired();
      store().reveal();
      for (let i = 0; i < Math.round(seconds * fps); i++) m.step(1 / fps);
      unsubscribe();
      return [store().revealProgress, sceneFx.terrainOpacity];
    };
    const a = at(30, 3);
    const b = at(120, 3);
    expect(a[0]).toBeCloseTo(b[0], 9);
    expect(a[1]).toBeCloseTo(b[1], 9);
  });

  it("caps a frame hitch at MAX_STEP_S instead of skipping the choreography", () => {
    const { m, unsubscribe } = wired();
    store().reveal();
    m.step(5);
    unsubscribe();
    expect(m.elapsedS).toBe(MAX_STEP_S);
    expect(store().phase).toBe("revealing");
  });

  it("reset() fades back to the exact start values (pixel-identical start frame)", () => {
    const { m, unsubscribe } = wired();
    store().reveal();
    runFrames(m, 1 / 60, () => store().phase === "revealed");
    store().reset();
    expect(m.mode).toBe("resetting");
    runFrames(m, 1 / 60, () => m.mode === "idle");
    unsubscribe();
    expect({ ...sceneFx }).toEqual({ ...INITIAL_SCENE_FX });
    const { phase, revealProgress, filter, view } = store();
    expect({ phase, revealProgress, filter, view }).toEqual({
      phase: initialDemoState.phase,
      revealProgress: initialDemoState.revealProgress,
      filter: initialDemoState.filter,
      view: initialDemoState.view,
    });
  });

  it("reset mid-reveal: late frames never resurrect the reveal", () => {
    const { m, unsubscribe } = wired();
    store().reveal();
    runFrames(m, 1 / 60, () => m.elapsedS > 3);
    store().reset();
    for (let i = 0; i < 600; i++) m.step(1 / 60);
    unsubscribe();
    expect(store().phase).toBe("public");
    expect(store().revealProgress).toBe(0);
    expect(m.mode).toBe("idle");
  });

  it("reset(); reveal() in the same tick restarts the clock from 0", () => {
    const { m, unsubscribe } = wired();
    store().reveal();
    runFrames(m, 1 / 60, () => m.elapsedS > 4);
    store().reset();
    store().reveal();
    expect(m.mode).toBe("revealing");
    expect(m.elapsedS).toBe(0);
    expect(sceneFx.terrainOpacity).toBe(1);
    m.step(1 / 60);
    unsubscribe();
    expect(store().revealProgress).toBe(0);
  });

  it("mounting mid-state syncs: revealed shows the settled scene, public the start frame", () => {
    for (const phase of ["revealed", "public"] as DemoPhase[]) {
      resetSceneFx();
      const m = createRevealMachine();
      m.onPhase(phase, phase);
      if (phase === "revealed") expect(sceneFx.terrainOpacity).toBe(TIMELINE.terrainFade.to);
      else expect({ ...sceneFx }).toEqual({ ...INITIAL_SCENE_FX });
    }
  });
});

// The demo store (docs/02 §6). One zustand store that the scene (H3) animates from and the shell (H4)
// drives with buttons and keys. `DemoState` matches docs/02 exactly; the module-level helpers below it
// are for the scene's per-frame loop and for tests, and never change the contract.

import { create, type StoreApi, type UseBoundStore } from "zustand";

export type DemoPhase = "public" | "revealing" | "revealed";
export type EventFilter = "public" | "all" | "strict";
export type CameraView = "oblique" | "side" | "plan";

export interface DemoState {
  phase: DemoPhase;
  revealProgress: number; // 0..1, advanced by the scene during "revealing"
  filter: EventFilter;
  timeMode: boolean;
  tNow: number | null; // epoch s; null shows every event
  playing: boolean;
  selectedEventId: string | null;
  view: CameraView;
  reveal(): void; // public -> revealing; scene sets revealed at progress 1
  reset(): void; // back to public, progress 0, filter "public", selection null
  setFilter(f: EventFilter): void;
  setTimeMode(on: boolean): void;
  setTNow(t: number | null): void;
  setPlaying(p: boolean): void;
  select(id: string | null): void;
  setView(v: CameraView): void;
}

/** Data fields only. The start frame of the demo, and exactly what `reset()` returns to. */
export type DemoData = Omit<
  DemoState,
  "reveal" | "reset" | "setFilter" | "setTimeMode" | "setTNow" | "setPlaying" | "select" | "setView"
>;

export const initialDemoState: Readonly<DemoData> = Object.freeze({
  phase: "public",
  revealProgress: 0,
  filter: "public",
  timeMode: false,
  tNow: null,
  playing: false,
  selectedEventId: null,
  view: "oblique",
});

const resetListeners = new Set<() => void>();

export const useDemo: UseBoundStore<StoreApi<DemoState>> = create<DemoState>()((set, get) => ({
  ...initialDemoState,

  // Only from "public": pressing REVEAL (or Space) mid-reveal or after it never restarts the animation.
  // The reveal is the moment every candidate event appears, so a "public" filter becomes "all" (an S
  // pressed beforehand keeps "strict"). The reveal dollies the camera into the low side view, so `view`
  // says "side", except from plan view, where the reveal plays top-down (WEB-07).
  reveal() {
    const { phase, filter, view } = get();
    if (phase !== "public") return;
    set({
      phase: "revealing",
      revealProgress: 0,
      filter: filter === "public" ? "all" : filter,
      view: view === "plan" ? "plan" : "side",
    });
  },

  // Back to the start frame: docs/02 names phase, progress, filter and selection; time mode and the
  // camera view reset too, because the reveal must restart from a pixel-identical frame (WEB-03).
  reset() {
    set({ ...initialDemoState });
    for (const listener of resetListeners) listener();
  },

  setFilter(f) {
    set({ filter: f });
  },

  // Leaving time mode stops playback and shows every event again.
  setTimeMode(on) {
    set(on ? { timeMode: true } : { timeMode: false, playing: false, tNow: null });
  },

  setTNow(t) {
    set({ tNow: t });
  },

  // Playback only means something in time mode, so starting it turns time mode on.
  setPlaying(p) {
    set(p ? { playing: true, timeMode: true } : { playing: false });
  },

  select(id) {
    set({ selectedEventId: id });
  },

  setView(v) {
    set({ view: v });
  },
}));

// ---- Scene-only helpers (not part of DemoState) -------------------------------------------------

/**
 * Scene-only: run `listener` after every reset(), even when reset() changes no field (e.g. R pressed
 * before the reveal after orbiting): the camera still has to return to the start pose. Returns the
 * unsubscribe function.
 */
export function onDemoReset(listener: () => void): () => void {
  resetListeners.add(listener);
  return () => {
    resetListeners.delete(listener);
  };
}
//
// Reveal timeline (docs/lanes/H3 → The reveal): revealProgress is the *counter* clock. It is 0 from
// reveal() until events start appearing (~1.0 s), climbs to exactly 1 as the last event appears
// (~6.0 s), and the shell's counter reads publicCatalogCount + (candidateCount − publicCatalogCount) ×
// revealProgress. The phase becomes "revealed" only after the settle (~7.0 s), via finishReveal().
// The scene calls setRevealProgress once per frame for those ~5 s; each call is one zustand set, which
// is how docs/02 routes progress to the shell. The scene itself reads the store with getState(), never
// a hook, so its per-frame work doesn't re-render React.

/** Scene-only: advance the counter clock. Clamped to [0, 1]; ignored outside "revealing". */
export function setRevealProgress(progress: number): void {
  const state = useDemo.getState();
  if (state.phase !== "revealing") return;
  if (Number.isNaN(progress)) throw new Error("setRevealProgress: progress is NaN");
  const clamped = progress <= 0 ? 0 : progress >= 1 ? 1 : progress;
  if (clamped !== state.revealProgress) useDemo.setState({ revealProgress: clamped });
}

/**
 * Scene-only: end the reveal after the settle. Progress is pinned to exactly 1 and the phase becomes
 * "revealed". Ignored outside "revealing", so a late frame after reset() can't resurrect a reveal.
 */
export function finishReveal(): void {
  if (useDemo.getState().phase !== "revealing") return;
  useDemo.setState({ revealProgress: 1, phase: "revealed" });
}

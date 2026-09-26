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

export const useDemo: UseBoundStore<StoreApi<DemoState>> = create<DemoState>()((set, get) => ({
  ...initialDemoState,

  // Only from "public": pressing REVEAL (or Space) mid-reveal or after it never restarts the animation.
  // The filter switches to "all" because the reveal *is* the moment every candidate event appears;
  // leaving it on "public" would hide exactly what is being revealed.
  reveal() {
    if (get().phase !== "public") return;
    set({ phase: "revealing", revealProgress: 0, filter: "all" });
  },

  // Back to the start frame: docs/02 names phase, progress, filter and selection; time mode and the
  // camera view reset too, because the reveal must restart from a pixel-identical frame (WEB-03).
  reset() {
    set({ ...initialDemoState });
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

/**
 * Scene-only: advance the reveal. Clamped to [0, 1]; at 1 the phase becomes "revealed" and progress is
 * exactly 1. Ignored outside "revealing", so a frame that lands after `reset()` can't resurrect a reveal.
 */
export function setRevealProgress(progress: number): void {
  const state = useDemo.getState();
  if (state.phase !== "revealing") return;
  if (Number.isNaN(progress)) throw new Error("setRevealProgress: progress is NaN");
  if (progress >= 1) {
    useDemo.setState({ revealProgress: 1, phase: "revealed" });
    return;
  }
  const clamped = progress <= 0 ? 0 : progress;
  if (clamped !== state.revealProgress) useDemo.setState({ revealProgress: clamped });
}

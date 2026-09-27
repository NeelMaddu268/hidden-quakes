// The Tour's own state (WEB-09), apart from the demo store (docs/02 §6 stays exactly as frozen). The
// button (mounted by H4) and the overlay (mounted by the scene) share it; the overlay owns the run.

import { create, type StoreApi, type UseBoundStore } from "zustand";
import type { TourStepId } from "./script";

export interface TourState {
  running: boolean;
  /** The step on screen, or null when not running. */
  step: TourStepId | null;
  /** Counts starts, so starting again always begins a fresh run from the first step. */
  runs: number;
}

export const useTour: UseBoundStore<StoreApi<TourState>> = create<TourState>()(() => ({
  running: false,
  step: null,
  runs: 0,
}));

/** Start (or restart) the tour from the public view. */
export function startTour(): void {
  useTour.setState((s) => ({ running: true, step: null, runs: s.runs + 1 }));
}

/** Stop the tour where it is; the scene stays as the tour left it. */
export function stopTour(): void {
  if (useTour.getState().running) useTour.setState({ running: false, step: null });
}

export function toggleTour(): void {
  if (useTour.getState().running) stopTour();
  else startTour();
}

/** Scene-only: the step the overlay is showing (written only on change). */
export function setTourStep(step: TourStepId | null): void {
  if (useTour.getState().step !== step) useTour.setState({ step });
}

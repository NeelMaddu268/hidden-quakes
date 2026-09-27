// Station-day state (WEB-10): the loaded manifest, the loaded bundle's run and whether the panel is open.
// The button (pinned into the canvas's corner stack, a separate React root under drei <Html>) and the
// panel (a DOM sibling of the canvas) share it, like the tour's store.

import { create, type StoreApi, type UseBoundStore } from "zustand";
import type { StationDayManifest } from "./manifest";

export interface StationDayState {
  /** Null until loaded, and for good when the manifest is missing or invalid (then no button at all). */
  manifest: StationDayManifest | null;
  /** The loaded bundle's run id (StationDayRunSync), null until a bundle is ready. */
  runId: string | null;
  open: boolean;
}

export const useStationDay: UseBoundStore<StoreApi<StationDayState>> = create<StationDayState>()(() => ({
  manifest: null,
  runId: null,
  open: false,
}));

/**
 * The manifest when it belongs to the loaded run, else null. The picture is one run's day (its ticks are
 * that run's candidate events), so another bundle (Today) gets no button and no panel.
 */
export function fittingManifest(s: StationDayState): StationDayManifest | null {
  return s.manifest !== null && s.runId !== null && s.manifest.runId === s.runId ? s.manifest : null;
}

/** The element that opened the panel; focus returns to it on close. Not state: nothing renders from it. */
let opener: HTMLElement | null = null;

export function setStationDayManifest(manifest: StationDayManifest | null): void {
  useStationDay.setState(manifest === null ? { manifest: null, open: false } : { manifest });
}

/** Records the loaded run; switching to a run the picture doesn't belong to closes the panel. */
export function setStationDayRun(runId: string | null): void {
  const s = useStationDay.getState();
  if (s.runId === runId) return;
  if (fittingManifest({ ...s, runId }) !== null) {
    useStationDay.setState({ runId });
    return;
  }
  opener = null;
  useStationDay.setState({ runId, open: false });
}

export function openStationDay(from: HTMLElement | null): void {
  if (fittingManifest(useStationDay.getState()) === null) return;
  opener = from;
  useStationDay.setState({ open: true });
}

export function closeStationDay(): void {
  if (!useStationDay.getState().open) return;
  useStationDay.setState({ open: false });
  const el = opener;
  opener = null;
  if (el?.isConnected) el.focus();
}

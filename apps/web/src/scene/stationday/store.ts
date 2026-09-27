// Station-day state (WEB-10): the loaded manifest and whether the panel is open. The button (pinned into
// the canvas's corner stack, a separate React root under drei <Html>) and the panel (a DOM sibling of the
// canvas) share it, like the tour's store.

import { create, type StoreApi, type UseBoundStore } from "zustand";
import type { StationDayManifest } from "./manifest";

export interface StationDayState {
  /** Null until loaded, and for good when the manifest is missing or invalid (then no button at all). */
  manifest: StationDayManifest | null;
  open: boolean;
}

export const useStationDay: UseBoundStore<StoreApi<StationDayState>> = create<StationDayState>()(() => ({
  manifest: null,
  open: false,
}));

/** The element that opened the panel; focus returns to it on close. Not state: nothing renders from it. */
let opener: HTMLElement | null = null;

export function setStationDayManifest(manifest: StationDayManifest | null): void {
  useStationDay.setState(manifest === null ? { manifest: null, open: false } : { manifest });
}

export function openStationDay(from: HTMLElement | null): void {
  if (useStationDay.getState().manifest === null) return;
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

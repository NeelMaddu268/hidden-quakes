"use client";
/**
 * Expo attract mode (DEMO-05 item 1). With `?kiosk=1` the guided tour (H3's, `@/scene`) starts
 * as soon as the bundle is ready and loops: when a run ends by itself the next one starts after a
 * short gap, so people walking past always see the reveal. Any key, click or scroll stops the tour
 * (the tour's own listener) and hands control back; the page stays fully interactive. After
 * `IDLE_MS` without input the tour resumes.
 *
 * Without `?kiosk=1` exactly one thing changes: after `IDLE_MS` without input on the start frame
 * (phase "public"), the tour starts once, as an attract loop for an unattended laptop. It never
 * starts over a revealed scene someone left mid-exploration.
 */
import { useEffect } from "react";
import { startTour, useTour } from "@/scene";
import { useDemo } from "@/state/demo";

export const KIOSK_PARAM = "kiosk";
/** Input-free time before the tour starts (again). */
export const IDLE_MS = 60_000;
/** Pause between two kiosk runs, on the tour's last frame. */
export const LOOP_GAP_MS = 2_500;
/** A tour that stops this soon after an input was stopped by a person, not by reaching its end. */
export const USER_STOP_WINDOW_MS = 1_000;
/** How often the idle clock is checked. */
const TICK_MS = 1_000;

const ACTIVITY_EVENTS = ["keydown", "pointerdown", "pointermove", "wheel", "touchstart"] as const;

export function kioskRequested(search: string): boolean {
  const value = new URLSearchParams(search).get(KIOSK_PARAM);
  return value === "1" || value === "true";
}

export interface AttractOptions {
  kiosk: boolean;
  ready: boolean;
  now?: () => number;
}

/**
 * Mounted once by the shell. Tracks the last input; starts the tour when idle (kiosk: any phase;
 * otherwise: only on the start frame); in kiosk mode restarts a run that ended by itself.
 */
export function useAttractMode({ kiosk, ready, now = () => Date.now() }: AttractOptions): void {
  useEffect(() => {
    if (!ready || typeof window === "undefined") return;
    let lastInput = now();
    const onActivity = () => {
      lastInput = now();
    };
    for (const type of ACTIVITY_EVENTS) window.addEventListener(type, onActivity, { capture: true, passive: true });

    let restart: ReturnType<typeof setTimeout> | null = null;
    // Kiosk: begin at once, before anyone has touched the page.
    if (kiosk && !useTour.getState().running) startTour();

    // A run that ended with no input just before it ended its own way: loop it (kiosk only).
    const unsubscribe = useTour.subscribe((next, prev) => {
      if (!kiosk || !prev.running || next.running) return;
      if (now() - lastInput < USER_STOP_WINDOW_MS) return; // a person stopped it: hand control back
      if (restart !== null) clearTimeout(restart);
      restart = setTimeout(() => {
        restart = null;
        if (!useTour.getState().running && now() - lastInput >= LOOP_GAP_MS) startTour();
      }, LOOP_GAP_MS);
    });

    const tick = setInterval(() => {
      if (useTour.getState().running) return;
      if (now() - lastInput < IDLE_MS) return;
      if (!kiosk && useDemo.getState().phase !== "public") return;
      startTour();
      // Idle is measured from this start, so a run that ends by itself (non-kiosk) doesn't restart
      // at once; in kiosk mode the loop above takes over.
      lastInput = now();
    }, TICK_MS);

    return () => {
      for (const type of ACTIVITY_EVENTS) window.removeEventListener(type, onActivity, { capture: true });
      unsubscribe();
      clearInterval(tick);
      if (restart !== null) clearTimeout(restart);
    };
    // `now` is a clock injection for tests; a new function identity must not restart the effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kiosk, ready]);
}

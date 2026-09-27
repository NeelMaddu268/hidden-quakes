/**
 * DEMO-05 item 1: attract mode. The real tour store and demo store; the tour's own run (the scene)
 * is not mounted, so a run "ends by itself" when the test calls stopTour() with no input before it.
 */
import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { stopTour, useTour } from "@/scene";
import { finishReveal, useDemo } from "@/state/demo";
import { IDLE_MS, LOOP_GAP_MS, kioskRequested, useAttractMode } from "./kiosk";

let clock = 0;
const now = () => clock;
const advance = (ms: number) => {
  clock += ms;
  vi.advanceTimersByTime(ms);
};
const running = () => useTour.getState().running;
const input = (type = "keydown") => window.dispatchEvent(new Event(type));

beforeEach(() => {
  vi.useFakeTimers();
  clock = 0;
  useDemo.getState().reset();
  stopTour();
});

afterEach(() => {
  stopTour();
  vi.useRealTimers();
});

describe("kioskRequested()", () => {
  it("reads ?kiosk=1 (or true) only", () => {
    expect(kioskRequested("?kiosk=1")).toBe(true);
    expect(kioskRequested("?mode=showcase&kiosk=true")).toBe(true);
    expect(kioskRequested("?kiosk=0")).toBe(false);
    expect(kioskRequested("")).toBe(false);
  });
});

describe("kiosk mode (?kiosk=1)", () => {
  it("starts the tour at once, loops a run that ends by itself, and hands control back on input", () => {
    const { unmount } = renderHook(() => useAttractMode({ kiosk: true, ready: true, now }));
    expect(running()).toBe(true);

    // The run ends by itself (no input for a while): the next one starts after the gap.
    advance(30_000);
    stopTour();
    expect(running()).toBe(false);
    advance(LOOP_GAP_MS);
    expect(running()).toBe(true);

    // A person presses a key: the tour's listener stops it; kiosk does not restart it.
    advance(10_000);
    input();
    stopTour();
    advance(LOOP_GAP_MS * 4);
    expect(running()).toBe(false);

    // Still interacting: no resume before the idle time has passed since the last input.
    advance(IDLE_MS / 2);
    input("pointermove");
    advance(IDLE_MS - 1_000);
    expect(running()).toBe(false);
    advance(2_000);
    expect(running()).toBe(true);
    unmount();
  });

  it("resumes after idle even over a revealed scene", () => {
    renderHook(() => useAttractMode({ kiosk: true, ready: true, now }));
    input();
    stopTour();
    useDemo.getState().reveal();
    finishReveal();
    advance(IDLE_MS + 1_000);
    expect(running()).toBe(true);
  });

  it("does nothing before the bundle is ready", () => {
    renderHook(() => useAttractMode({ kiosk: true, ready: false, now }));
    advance(IDLE_MS * 2);
    expect(running()).toBe(false);
  });
});

describe("normal mode (no ?kiosk)", () => {
  it("never starts on load, starts after idle on the start frame, and does not loop", () => {
    renderHook(() => useAttractMode({ kiosk: false, ready: true, now }));
    expect(running()).toBe(false);
    advance(IDLE_MS - 1_000);
    expect(running()).toBe(false);
    advance(2_000);
    expect(running()).toBe(true);
    // The run ends by itself: no immediate restart outside kiosk mode.
    stopTour();
    advance(LOOP_GAP_MS * 4);
    expect(running()).toBe(false);
  });

  it("never starts over a scene someone revealed and left", () => {
    renderHook(() => useAttractMode({ kiosk: false, ready: true, now }));
    useDemo.getState().reveal();
    finishReveal();
    advance(IDLE_MS * 3);
    expect(running()).toBe(false);
  });
});

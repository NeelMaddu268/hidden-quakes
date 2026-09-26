import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { initialDemoState, useDemo } from "../../state/demo";
import type { BundleState } from "../types";
import { TimeScrubber } from "./index";

// The scrubber reads data only through scene/data.ts; the test swaps that module for controllable state.
const data = vi.hoisted(() => ({ bundle: { status: "loading" } as BundleState }));
vi.mock("../data", () => ({ useBundle: () => data.bundle }));

// ---- Test-local synthetic records (rule 5): only the fields the scrubber reads ------------------
const START = Date.UTC(2026, 8, 10) / 1000;
const END = START + 86_400;
const EVENTS = [
  { id: "e1", t: START + 60, tier: "A" },
  { id: "e2", t: START + 700, tier: "C" },
  { id: "e3", t: START + 3600, tier: "A" },
  { id: "e4", t: START + 7200, tier: "B" },
];
const CATALOG = [{ id: "p1", t: START + 30 }, { id: "p2", t: START + 5000 }];

function readyBundle(): BundleState {
  return {
    status: "ready",
    meta: { run: { windowStart: START, windowEnd: END, windowLabel: "test window" } },
    events: EVENTS,
    catalog: CATALOG,
  } as unknown as BundleState;
}

function setViewport(width: number, height: number) {
  Object.defineProperty(window, "innerWidth", { configurable: true, value: width });
  Object.defineProperty(window, "innerHeight", { configurable: true, value: height });
}

const flush = () => act(() => vi.advanceTimersByTime(40));
const text = (id: string) => screen.getByTestId(id).textContent;

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => setTimeout(() => cb(performance.now()), 16));
  vi.stubGlobal("cancelAnimationFrame", (id: number) => clearTimeout(id));
  setViewport(1920, 1080);
  data.bundle = readyBundle();
  useDemo.setState({ ...initialDemoState });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function enterTimeMode(tNow: number | null = START) {
  act(() => useDemo.setState({ phase: "revealed", filter: "all", timeMode: true, tNow, playing: false }));
}

describe("TimeScrubber visibility (docs/01: appears after the reveal)", () => {
  it("renders nothing outside time mode, before the reveal ends, or without data", () => {
    render(<TimeScrubber />);
    expect(screen.queryByTestId("time-scrubber")).toBeNull();
    act(() => useDemo.setState({ timeMode: true, phase: "revealing" }));
    expect(screen.queryByTestId("time-scrubber")).toBeNull();
    act(() => useDemo.setState({ phase: "revealed" }));
    expect(screen.getByTestId("time-scrubber")).toBeTruthy();
    act(() => useDemo.setState({ timeMode: false }));
    expect(screen.queryByTestId("time-scrubber")).toBeNull();
  });

  it("stays out of the open drawer's way (hides when the band is too narrow)", () => {
    setViewport(1280, 720);
    render(<TimeScrubber />);
    enterTimeMode();
    expect(screen.getByTestId("time-scrubber")).toBeTruthy();
    act(() => useDemo.setState({ selectedEventId: "e1" }));
    expect(screen.queryByTestId("time-scrubber")).toBeNull();
  });

  it("renders nothing while the bundle loads", () => {
    data.bundle = { status: "loading" };
    render(<TimeScrubber />);
    enterTimeMode();
    expect(screen.queryByTestId("time-scrubber")).toBeNull();
  });
});

describe("TimeScrubber readouts (every number counted from the bundle)", () => {
  it("shows the UTC clock and how many events are on screen so far, following the filter", () => {
    render(<TimeScrubber />);
    enterTimeMode(START + 800);
    flush();
    expect(text("time-clock")).toBe("00:13 UTC");
    expect(text("time-counts")).toBe("1 public2 recovered");
    act(() => useDemo.setState({ tNow: START + 7200 }));
    flush();
    expect(text("time-clock")).toBe("02:00 UTC");
    expect(text("time-counts")).toBe("2 public4 recovered");
    act(() => useDemo.setState({ filter: "strict" }));
    flush();
    expect(text("time-counts")).toBe("2 public2 strict");
    act(() => useDemo.setState({ tNow: END }));
    flush();
    expect(text("time-clock")).toBe("24:00 UTC");
    expect(text("time-counts")).toBe("2 public2 strict");
  });

  it("labels the bins and the shared scale", () => {
    render(<TimeScrubber />);
    enterTimeMode();
    expect(text("time-scale")).toBe("10-min bins · max 1");
  });

  it("exposes the playhead as an accessible slider", () => {
    render(<TimeScrubber />);
    enterTimeMode(START + 3600);
    flush();
    const slider = screen.getByRole("slider", { name: "Time within the run window" });
    expect(slider.getAttribute("aria-valuemin")).toBe(String(START));
    expect(slider.getAttribute("aria-valuemax")).toBe(String(END));
    expect(slider.getAttribute("aria-valuenow")).toBe(String(START + 3600));
    expect(slider.getAttribute("aria-valuetext")).toBe("01:00 UTC");
  });
});

describe("TimeScrubber controls", () => {
  it("play / pause toggles playback; at the end, play restarts the replay from the window start", () => {
    render(<TimeScrubber />);
    enterTimeMode(START + 100);
    const play = screen.getByTestId("time-play");
    expect(play.getAttribute("aria-label")).toBe("Play replay");
    fireEvent.click(play);
    expect(useDemo.getState().playing).toBe(true);
    expect(useDemo.getState().tNow).toBe(START + 100);
    expect(screen.getByTestId("time-play").getAttribute("aria-label")).toBe("Pause replay");
    fireEvent.click(screen.getByTestId("time-play"));
    expect(useDemo.getState().playing).toBe(false);
    act(() => useDemo.setState({ tNow: END }));
    fireEvent.click(screen.getByTestId("time-play"));
    expect(useDemo.getState()).toMatchObject({ tNow: START, playing: true });
  });

  it("arrow keys step one bin (Shift: an hour), Home / End jump, all clamped, and pause the replay", () => {
    render(<TimeScrubber />);
    enterTimeMode(START + 3600);
    act(() => useDemo.setState({ playing: true }));
    const slider = screen.getByTestId("time-strip");
    fireEvent.keyDown(slider, { key: "ArrowRight" });
    expect(useDemo.getState()).toMatchObject({ tNow: START + 4200, playing: false });
    fireEvent.keyDown(slider, { key: "ArrowLeft", shiftKey: true });
    expect(useDemo.getState().tNow).toBe(START + 600);
    fireEvent.keyDown(slider, { key: "ArrowLeft", shiftKey: true });
    expect(useDemo.getState().tNow).toBe(START);
    fireEvent.keyDown(slider, { key: "End" });
    expect(useDemo.getState().tNow).toBe(END);
    fireEvent.keyDown(slider, { key: "ArrowRight" });
    expect(useDemo.getState().tNow).toBe(END);
    fireEvent.keyDown(slider, { key: "Home" });
    expect(useDemo.getState().tNow).toBe(START);
  });

  it("presenter keys pass through the strip: a non-seek key is left for the shell", () => {
    render(<TimeScrubber />);
    enterTimeMode(START + 3600);
    const onWindowKey = vi.fn();
    window.addEventListener("keydown", onWindowKey);
    fireEvent.keyDown(screen.getByTestId("time-strip"), { key: "s" });
    fireEvent.keyDown(screen.getByTestId("time-strip"), { key: "ArrowRight" });
    window.removeEventListener("keydown", onWindowKey);
    expect(onWindowKey).toHaveBeenCalledTimes(1); // "s" bubbled; the seek key did not
  });

  it("pressing on the strip pauses and seeks to that time; dragging follows the pointer", () => {
    render(<TimeScrubber />);
    enterTimeMode(START);
    act(() => useDemo.setState({ playing: true }));
    const strip = screen.getByTestId("time-strip");
    const canvas = strip.querySelector("canvas")!;
    canvas.getBoundingClientRect = () => ({ left: 100, top: 0, width: 400, height: 50, right: 500, bottom: 50, x: 100, y: 0, toJSON() {} }) as DOMRect;
    strip.setPointerCapture = vi.fn();
    strip.hasPointerCapture = vi.fn(() => true);
    strip.releasePointerCapture = vi.fn();
    fireEvent.pointerDown(strip, { button: 0, pointerId: 1, clientX: 300 });
    expect(useDemo.getState()).toMatchObject({ playing: false, tNow: START + 43_200 });
    fireEvent.pointerMove(strip, { pointerId: 1, clientX: 400 });
    expect(useDemo.getState().tNow).toBe(START + 64_800);
    fireEvent.pointerUp(strip, { pointerId: 1, clientX: 400 });
    fireEvent.pointerMove(strip, { pointerId: 1, clientX: 100 });
    expect(useDemo.getState().tNow).toBe(START + 64_800); // released: moves no longer seek
    expect(strip.releasePointerCapture).toHaveBeenCalledWith(1);
  });
});

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { parseListenManifest, type ListenManifest } from "../../audio/manifest";
import { initialDemoState, useDemo } from "../../state/demo";
import type { BundleState } from "../types";
import { TimeScrubber } from "./index";
import { listenTNow } from "./listen";

// The scrubber reads data only through scene/data.ts; the test swaps that module for controllable state.
// The Listen clip's manifest (WEB-10) likewise: null (no manifest) unless a test sets one.
const data = vi.hoisted(() => ({ bundle: { status: "loading" } as BundleState, listen: null as ListenManifest | null }));
vi.mock("../data", () => ({ useBundle: () => data.bundle }));
vi.mock("../../audio/manifest", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../audio/manifest")>()),
  useListenManifest: () => data.listen,
}));

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
    meta: { run: { windowStart: START, windowEnd: END, windowLabel: "test window" }, scene: { runId: "test-run" } },
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
  data.listen = null;
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
    expect(text("time-counts")).toBe("1 public · 2 recovered");
    act(() => useDemo.setState({ tNow: START + 7200 }));
    flush();
    expect(text("time-clock")).toBe("02:00 UTC");
    expect(text("time-counts")).toBe("2 public · 4 recovered");
    act(() => useDemo.setState({ filter: "strict" }));
    flush();
    expect(text("time-counts")).toBe("2 public · 2 strict");
    act(() => useDemo.setState({ tNow: END }));
    flush();
    expect(text("time-clock")).toBe("24:00 UTC");
    expect(text("time-counts")).toBe("2 public · 2 strict");
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

  it("a focused play button keeps Space / Enter to itself but passes every presenter key to the shell", () => {
    render(<TimeScrubber />);
    enterTimeMode(START);
    const onWindowKey = vi.fn();
    window.addEventListener("keydown", onWindowKey);
    const play = screen.getByTestId("time-play");
    for (const key of [" ", "Enter"]) fireEvent.keyDown(play, { key });
    expect(onWindowKey).not.toHaveBeenCalled();
    for (const key of ["p", "s", "r", "t", "e", "Escape"]) fireEvent.keyDown(play, { key });
    window.removeEventListener("keydown", onWindowKey);
    expect(onWindowKey.mock.calls.map(([e]) => (e as KeyboardEvent).key)).toEqual(["p", "s", "r", "t", "e", "Escape"]);
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

describe("Listen (WEB-10): the busiest-hour clip drives the replay", () => {
  // A synthetic clip: one hour, two hours into the test window, at 240× (15 s of audio).
  const CLIP_START = START + 7200;
  function clip(over: Record<string, unknown> = {}): ListenManifest {
    return parseListenManifest({
      stationId: "XX.S01",
      channel: "HHZ",
      startUtc: new Date(CLIP_START * 1000).toISOString(),
      endUtc: new Date((CLIP_START + 3600) * 1000).toISOString(),
      speed: 240,
      durationS: 15,
      note: "Synthetic clip.",
      runId: "test-run",
      files: { ogg: { name: "t.ogg" }, mp3: { name: "t.mp3" } },
      ...over,
    })!;
  }

  let play: ReturnType<typeof vi.fn>;
  let pause: ReturnType<typeof vi.fn>;
  beforeEach(() => {
    play = vi.fn(() => Promise.resolve());
    pause = vi.fn();
    vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(play as () => Promise<void>);
    vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(pause as () => void);
  });
  afterEach(() => {
    cleanup(); // before the media mocks go: a test may end mid-clip, and unmounting pauses it
    vi.restoreAllMocks();
  });

  /** Renders the scrubber in time mode with the clip; returns the audio element with a settable clock. */
  function setup(tNow = START + 100) {
    data.listen = clip();
    render(<TimeScrubber />);
    enterTimeMode(tNow);
    const audio = screen.getByTestId("time-listen-audio") as HTMLAudioElement;
    Object.defineProperty(audio, "currentTime", { configurable: true, writable: true, value: 0 });
    return audio;
  }
  const button = () => screen.getByTestId("time-listen");

  it("renders only with a manifest that fits this run", () => {
    render(<TimeScrubber />);
    enterTimeMode();
    expect(screen.queryByTestId("time-listen")).toBeNull();
    cleanup();
    data.listen = clip({ runId: "another-run" });
    render(<TimeScrubber />);
    expect(screen.queryByTestId("time-listen")).toBeNull();
    cleanup();
    data.listen = clip({ endUtc: new Date((END + 60) * 1000).toISOString() });
    render(<TimeScrubber />);
    expect(screen.queryByTestId("time-listen")).toBeNull();
  });

  it("labels the button from the manifest, with channel, UTC window and note on hover, and both sources", () => {
    setup();
    expect(button().textContent).toBe("Listen: XX.S01, sped up 240×");
    expect(button().getAttribute("aria-pressed")).toBe("false");
    expect(button().getAttribute("title")).toBe("XX.S01 channel HHZ · 2026-09-10 02:00:00–03:00:00 UTC. Synthetic clip.");
    const sources = [...screen.getByTestId("time-listen-audio").querySelectorAll("source")];
    expect(sources.map((s) => [s.getAttribute("src"), s.getAttribute("type")])).toEqual([
      ["/audio/t.ogg", "audio/ogg"],
      ["/audio/t.mp3", "audio/mpeg"],
    ]);
  });

  it("plays the clip from its start and hands the playhead to the audio", () => {
    const audio = setup();
    audio.currentTime = 7;
    fireEvent.click(button());
    expect(play).toHaveBeenCalledTimes(1);
    expect(audio.currentTime).toBe(0);
    expect(useDemo.getState()).toMatchObject({ timeMode: true, playing: true, tNow: CLIP_START });
    expect(button().getAttribute("aria-pressed")).toBe("true");
    expect(listenTNow()).toBe(CLIP_START);
    audio.currentTime = 1.5;
    expect(listenTNow()).toBe(CLIP_START + 360);
    audio.currentTime = 20;
    expect(listenTNow()).toBe(CLIP_START + 3600); // clamped to the clip window
  });

  it("pressing it again stops the clip and leaves the clock where the audio was", () => {
    const audio = setup();
    fireEvent.click(button());
    audio.currentTime = 2.5;
    fireEvent.click(button());
    expect(pause).toHaveBeenCalled();
    expect(useDemo.getState()).toMatchObject({ playing: false, tNow: CLIP_START + 600 });
    expect(button().getAttribute("aria-pressed")).toBe("false");
    expect(listenTNow()).toBeNull();
  });

  it("the clip ending stops the replay at the clip's end", () => {
    const audio = setup();
    fireEvent.click(button());
    audio.currentTime = 15;
    fireEvent.ended(audio);
    expect(useDemo.getState()).toMatchObject({ playing: false, tNow: CLIP_START + 3600 });
    expect(listenTNow()).toBeNull();
    expect(button().getAttribute("aria-pressed")).toBe("false");
  });

  it("scrubbing, the replay's pause, leaving time mode or a reset stop the clip and keep the user's clock", () => {
    const audio = setup();
    fireEvent.click(button());
    audio.currentTime = 1;
    fireEvent.keyDown(screen.getByTestId("time-strip"), { key: "Home" });
    expect(pause).toHaveBeenCalledTimes(1);
    expect(useDemo.getState()).toMatchObject({ playing: false, tNow: START });
    expect(listenTNow()).toBeNull();

    fireEvent.click(button());
    fireEvent.click(screen.getByTestId("time-play")); // "Pause replay"
    expect(pause).toHaveBeenCalledTimes(2);
    expect(useDemo.getState()).toMatchObject({ playing: false, tNow: CLIP_START });
    expect(listenTNow()).toBeNull();

    fireEvent.click(button());
    act(() => useDemo.getState().reset());
    expect(pause).toHaveBeenCalledTimes(3);
    expect(listenTNow()).toBeNull();
  });

  it("a play() the browser refuses restores the clock and says so, without throwing", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    play.mockImplementation(() => Promise.reject(Object.assign(new Error("user gesture required"), { name: "NotAllowedError" })));
    setup(START + 100);
    fireEvent.click(button());
    await act(async () => {
      await Promise.resolve();
    });
    expect(useDemo.getState()).toMatchObject({ playing: false, tNow: START + 100 });
    expect(listenTNow()).toBeNull();
    expect(button().getAttribute("aria-pressed")).toBe("false");
    expect(button().getAttribute("data-failed")).toBe("true");
    expect(button().getAttribute("title")).toMatch(/^Audio did not start \(NotAllowedError: user gesture required\)/);
    expect(warn).toHaveBeenCalled();
  });

  it("unmounting mid-clip (time mode off) stops the audio", () => {
    setup();
    fireEvent.click(button());
    act(() => useDemo.getState().setTimeMode(false));
    expect(screen.queryByTestId("time-scrubber")).toBeNull();
    expect(pause).toHaveBeenCalled();
    expect(useDemo.getState().playing).toBe(false);
    expect(listenTNow()).toBeNull();
  });
});

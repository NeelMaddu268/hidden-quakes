import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { initialDemoState, useDemo } from "../../state/demo";
import type { BundleState } from "../types";
import { drawerWidthPx } from "../plan/layout";
import { CAPTION_LAYOUT } from "./layout";
import { TOUR_TIMING } from "./script";
import { useTour } from "./store";
import { Tour, TourButton } from "./Tour";

// The overlay reads data only through scene/data.ts; the test swaps that module for controllable state.
const data = vi.hoisted(() => ({ bundle: { status: "loading" } as BundleState }));
vi.mock("../data", () => ({ useBundle: () => data.bundle }));

// ---- Test-local synthetic records (rule 5): only the fields the tour reads -----------------------
const SUMMARY = {
  publicCatalogCount: 43,
  recoveredCatalogCount: 41,
  candidateCount: 1654,
  additionalCount: 1613,
  strictQualityCount: 32,
  strictAdditionalCount: 14,
};
const q = (nStations: number, rmsS = 0.03) => ({ nStations, rmsS });
const EVENTS = [
  { id: "matched", tier: "A", catalogMatch: { publicId: "p1" }, revealOrder: 0, quality: q(30) },
  { id: "hidden", tier: "A", catalogMatch: null, revealOrder: 1, quality: q(23) },
  { id: "weak", tier: "C", catalogMatch: null, revealOrder: 2, quality: q(40) },
];

function readyBundle(): BundleState {
  return { status: "ready", meta: { summary: SUMMARY }, events: EVENTS, catalog: [] } as unknown as BundleState;
}

const caption = () => screen.queryByTestId("tour-caption");
const advance = (ms: number) => act(() => vi.advanceTimersByTime(ms));
const key = (k: string, init: KeyboardEventInit = {}) => {
  const event = new KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true, ...init });
  act(() => {
    window.dispatchEvent(event);
  });
  return event;
};

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => setTimeout(() => cb(performance.now()), 16));
  vi.stubGlobal("cancelAnimationFrame", (id: number) => clearTimeout(id));
  data.bundle = readyBundle();
  useDemo.setState({ ...initialDemoState });
  useTour.setState({ running: false, step: null, runs: 0 });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Tour overlay", () => {
  it("renders nothing until started, and nothing without data", () => {
    data.bundle = { status: "loading" } as BundleState;
    const { rerender } = render(<Tour />);
    expect(caption()).toBeNull();
    key("g");
    data.bundle = readyBundle();
    rerender(<Tour />);
    expect(caption()).toBeNull();
  });

  it("G starts at the public view with the caption's numbers from the bundle", () => {
    render(<Tour />);
    const g = key("g");
    expect(g.defaultPrevented).toBe(true);
    expect(useTour.getState()).toMatchObject({ running: true, step: "public" });
    expect(useDemo.getState().phase).toBe("public");
    expect(caption()?.dataset.step).toBe("public");
    expect(caption()?.textContent).toBe("The public regional catalog lists 43 events here in this window.");
    expect(caption()?.getAttribute("role")).toBe("status");
    // Values are marked, so a check can match them against the counters.
    expect(caption()?.querySelector('[data-placeholder="publicCatalogCount"]')?.textContent).toBe("43");
  });

  it("advances through the reveal and opens the hidden strict event", () => {
    render(<Tour />);
    key("g");
    advance(TOUR_TIMING.publicHoldMs + 50);
    expect(useTour.getState().step).toBe("reveal");
    expect(useDemo.getState().phase).toBe("revealing");
    expect(caption()?.textContent).toContain("we recovered 41 of 43, and associated 1,613 more candidate events");
    act(() => useDemo.setState({ phase: "revealed", revealProgress: 1 }));
    advance(TOUR_TIMING.revealHoldMs + 50);
    expect(useTour.getState().step).toBe("strict");
    expect(useDemo.getState().filter).toBe("strict");
    advance(TOUR_TIMING.strictHoldMs + 50);
    expect(useTour.getState().step).toBe("hidden");
    expect(useDemo.getState().selectedEventId).toBe("hidden");
    expect(caption()?.textContent).toContain("14 strict events are not in the public regional catalog");
    expect(caption()?.textContent).toContain("23 stations agree");
    // Laid out left of where the drawer is going, not where its slide-in has got to (jsdom: 1024 wide).
    const cap = caption()!;
    const right = parseFloat(cap.style.left) + parseFloat(cap.style.width);
    expect(right).toBeLessThanOrEqual(window.innerWidth - drawerWidthPx(window.innerWidth) - CAPTION_LAYOUT.gap);
  });

  it("outlines the Validation card in its step, and skips the step without one", () => {
    const card = document.createElement("section");
    card.dataset.testid = "validation-panel";
    document.body.appendChild(card);
    card.getBoundingClientRect = () => ({ left: 24, top: 386, width: 352, height: 266 }) as DOMRect;
    render(<Tour />);
    key("g");
    advance(TOUR_TIMING.publicHoldMs + 50);
    act(() => useDemo.setState({ phase: "revealed", revealProgress: 1 }));
    advance(TOUR_TIMING.revealHoldMs + TOUR_TIMING.strictHoldMs + TOUR_TIMING.hiddenHoldMs + 150);
    expect(useTour.getState().step).toBe("validation");
    const ring = screen.getByTestId("tour-spotlight");
    expect(ring.style.left).toBe("18px");
    expect(ring.style.width).toBe("364px");
    card.remove();
  });

  it("any key stops it, and that key does nothing else", () => {
    render(<Tour />);
    const bubbled = vi.fn();
    window.addEventListener("keydown", bubbled);
    key("g");
    const space = key(" ");
    expect(space.defaultPrevented).toBe(true);
    expect(bubbled).toHaveBeenCalledTimes(1); // only the G that started it
    expect(useTour.getState()).toMatchObject({ running: false, step: null });
    expect(caption()).toBeNull();
    // The scene stays where the tour left it (the start frame here); nothing was revealed.
    expect(useDemo.getState().phase).toBe("public");
    window.removeEventListener("keydown", bubbled);
  });

  it("modifier keys and shortcuts (a screen recorder's) don't stop it", () => {
    render(<Tour />);
    key("g");
    for (const k of ["Shift", "Meta", "Control", "Alt"]) key(k);
    key("5", { metaKey: true, shiftKey: true });
    expect(useTour.getState().running).toBe(true);
  });

  it("a click or a scroll anywhere stops it", () => {
    render(<Tour />);
    key("g");
    act(() => {
      fireEvent.pointerDown(document.body);
    });
    expect(useTour.getState().running).toBe(false);
    key("g");
    act(() => {
      fireEvent.wheel(document.body);
    });
    expect(useTour.getState().running).toBe(false);
  });

  it("G typed into a text field doesn't start it", () => {
    render(<Tour />);
    const input = document.createElement("input");
    document.body.appendChild(input);
    act(() => {
      fireEvent.keyDown(input, { key: "g" });
    });
    expect(useTour.getState().running).toBe(false);
    input.remove();
  });

  it("ends by itself after the full view and leaves every event on screen", () => {
    render(<Tour />);
    key("g");
    advance(TOUR_TIMING.publicHoldMs + 50);
    act(() => useDemo.setState({ phase: "revealed", revealProgress: 1 }));
    advance(TOUR_TIMING.revealHoldMs + TOUR_TIMING.strictHoldMs + TOUR_TIMING.hiddenHoldMs + 150);
    // No Validation card in this DOM: straight to the replay, which the scene would play.
    expect(useTour.getState().step).toBe("replay");
    act(() => useDemo.setState({ tNow: 100, playing: false }));
    advance(TOUR_TIMING.replayHoldMs + 50);
    expect(useTour.getState().step).toBe("full");
    expect(useDemo.getState()).toMatchObject({ filter: "all", timeMode: false, selectedEventId: null });
    expect(caption()?.textContent).toBe("1,654 candidate events from public waveforms, against 43 in the public regional catalog.");
    advance(TOUR_TIMING.fullHoldMs + 50);
    expect(useTour.getState().running).toBe(false);
    expect(caption()).toBeNull();
  });
});

describe("TourButton", () => {
  it("toggles the tour, hides while it plays, and its own click isn't a stop click", () => {
    render(
      <>
        <Tour />
        <TourButton />
      </>,
    );
    const button = screen.getByTestId("tour-button");
    expect(button.textContent).toBe("Tour");
    expect(button.getAttribute("aria-keyshortcuts")).toBe("G");
    act(() => {
      fireEvent.pointerDown(button);
      fireEvent.click(button);
    });
    expect(useTour.getState().running).toBe(true);
    expect(button.style.visibility).toBe("hidden");
    expect(document.activeElement).not.toBe(button);
    act(() => {
      fireEvent.click(button);
    });
    expect(useTour.getState().running).toBe(false);
    expect(button.style.visibility).toBe("visible");
  });

  it("takes the shell's classes instead of its own look", () => {
    render(<TourButton className="pill modePill" />);
    const button = screen.getByTestId("tour-button");
    expect(button.className).toBe("pill modePill");
    expect(button.style.borderRadius).toBe("");
  });

  it("renders nothing until the bundle is ready", () => {
    data.bundle = { status: "loading" } as BundleState;
    render(<TourButton />);
    expect(screen.queryByTestId("tour-button")).toBeNull();
  });
});

import { act, cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { initialDemoState, useDemo } from "../state/demo";
import { HiddenHeroKey } from "./HiddenHeroKey";
import { useTour } from "./tour/store";
import { Tour } from "./tour/Tour";
import type { BundleState } from "./types";

const data = vi.hoisted(() => ({ bundle: { status: "loading" } as BundleState }));
vi.mock("./data", () => ({ useBundle: () => data.bundle }));

// Test-local synthetic records (rule 5): only the fields the hidden-hero rule reads.
const q = (nStations: number, rmsS = 0.03) => ({ nStations, rmsS });
const EVENTS = [
  { id: "matched", tier: "A", catalogMatch: { catalogId: "p1" }, revealOrder: 0, quality: q(30) },
  { id: "hidden", tier: "A", catalogMatch: null, revealOrder: 1, quality: q(23) },
  { id: "weak", tier: "C", catalogMatch: null, revealOrder: 2, quality: q(40) },
];
const ready = () =>
  ({
    status: "ready",
    meta: { summary: { publicCatalogCount: 1, recoveredCatalogCount: 1, candidateCount: 3, additionalCount: 2, strictQualityCount: 2, strictAdditionalCount: 1 } },
    events: EVENTS,
    catalog: [],
  }) as unknown as BundleState;

const press = (key: string, init: KeyboardEventInit = {}, target: EventTarget = window) => {
  const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...init });
  act(() => {
    target.dispatchEvent(event);
  });
  return event;
};

beforeEach(() => {
  data.bundle = ready();
  useDemo.setState({ ...initialDemoState, phase: "revealed", filter: "all" });
  useTour.setState({ running: false, step: null, runs: 0 });
});

afterEach(cleanup);

describe("H: the hidden hero", () => {
  it("opens the Tier A event with no public-catalog match that most stations agreed on", () => {
    render(<HiddenHeroKey />);
    const e = press("h");
    expect(useDemo.getState().selectedEventId).toBe("hidden");
    expect(e.defaultPrevented).toBe(true);
    useDemo.getState().select(null);
    press("H", { shiftKey: true });
    expect(useDemo.getState().selectedEventId).toBe("hidden");
  });

  it("does nothing before the reveal, and works while the reveal plays", () => {
    render(<HiddenHeroKey />);
    act(() => useDemo.setState({ phase: "public" }));
    expect(press("h").defaultPrevented).toBe(false);
    expect(useDemo.getState().selectedEventId).toBeNull();
    act(() => useDemo.setState({ phase: "revealing" }));
    press("h");
    expect(useDemo.getState().selectedEventId).toBe("hidden");
  });

  it("leaves shortcuts, repeats and text fields alone", () => {
    render(<HiddenHeroKey />);
    press("h", { metaKey: true });
    press("h", { ctrlKey: true });
    press("h", { repeat: true });
    const input = document.createElement("input");
    document.body.appendChild(input);
    act(() => {
      fireEvent.keyDown(input, { key: "h" });
    });
    input.remove();
    expect(useDemo.getState().selectedEventId).toBeNull();
  });

  it("does nothing without data, or when the bundle has no such event", () => {
    data.bundle = { status: "loading" } as BundleState;
    const { rerender } = render(<HiddenHeroKey />);
    press("h");
    expect(useDemo.getState().selectedEventId).toBeNull();
    data.bundle = { ...ready(), events: EVENTS.filter((e) => e.id !== "hidden") } as unknown as BundleState;
    rerender(<HiddenHeroKey />);
    expect(press("h").defaultPrevented).toBe(false);
    expect(useDemo.getState().selectedEventId).toBeNull();
  });

  it("while the tour plays, H only stops the tour", () => {
    vi.useFakeTimers();
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => setTimeout(() => cb(performance.now()), 16));
    vi.stubGlobal("cancelAnimationFrame", (id: number) => clearTimeout(id));
    try {
      render(
        <>
          <Tour />
          <HiddenHeroKey />
        </>,
      );
      press("g");
      expect(useTour.getState().running).toBe(true);
      press("h");
      expect(useTour.getState().running).toBe(false);
      expect(useDemo.getState().selectedEventId).toBeNull();
    } finally {
      cleanup();
      vi.useRealTimers();
      vi.unstubAllGlobals();
    }
  });
});

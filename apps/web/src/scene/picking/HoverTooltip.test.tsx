import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { initialDemoState, useDemo } from "../../state/demo";
import { useTour } from "../tour/store";
import type { BundleState } from "../types";
import { useHover } from "./hover";
import { HoverTooltip } from "./HoverTooltip";

const data = vi.hoisted(() => ({ bundle: { status: "loading" } as BundleState }));
vi.mock("../data", () => ({ useBundle: () => data.bundle }));

// Test-local synthetic records (rule 5): only the fields the tooltip reads.
const T = Date.UTC(2026, 8, 10, 9, 15, 28) / 1000;
function readyBundle(): BundleState {
  return {
    status: "ready",
    meta: { scene: { refSurfaceElevM: 1000 } },
    events: [
      { id: "e1", tier: "A", t: T, elevM: -3000, magnitude: null, catalogMatch: null, quality: { nStations: 12 } },
      { id: "e2", tier: "B", t: T, elevM: -2000, magnitude: null, catalogMatch: { catalogId: "p1" }, quality: { nStations: 9 } },
    ],
    catalog: [{ id: "p1", t: T, elevM: -2100, mag: 1.1, magType: "ml", matchedEventId: "e2" }],
  } as unknown as BundleState;
}

const tip = () => screen.queryByTestId("hover-tooltip");
const hover = (kind: "candidate" | "public", id: string, x = 100, y = 100) =>
  act(() => useHover.setState({ target: { kind, id, x, y } }));

function setViewport(width: number, height: number) {
  Object.defineProperty(window, "innerWidth", { configurable: true, value: width });
  Object.defineProperty(window, "innerHeight", { configurable: true, value: height });
}

beforeEach(() => {
  data.bundle = readyBundle();
  setViewport(1280, 720);
  useDemo.setState({ ...initialDemoState, phase: "revealed", filter: "all" });
  useHover.setState({ target: null });
  useTour.setState({ running: false, step: null, runs: 0 });
});

afterEach(cleanup);

describe("HoverTooltip", () => {
  it("shows nothing without a hovered event or without data", () => {
    render(<HoverTooltip />);
    expect(tip()).toBeNull();
    hover("candidate", "missing");
    expect(tip()).toBeNull();
  });

  it("describes a candidate event next to the pointer", () => {
    render(<HoverTooltip />);
    hover("candidate", "e1", 200, 150);
    expect(tip()?.getAttribute("role")).toBe("tooltip");
    expect(tip()?.textContent).toContain("Candidate event · Tier A (strict)");
    expect(tip()?.textContent).toContain("4.00 km below site surface");
    expect(tip()?.textContent).toContain("12 stations agreed");
    expect(tip()?.textContent).toContain("Not in the public regional catalog");
    expect(tip()?.style.left).toBe("214px");
    expect(tip()?.style.top).toBe("164px");
  });

  it("describes a public point, with its recovery only after the reveal", () => {
    render(<HoverTooltip />);
    hover("public", "p1");
    expect(tip()?.textContent).toContain("Recovered by our pipeline · Tier B");
    act(() => useDemo.setState({ phase: "public" }));
    expect(tip()?.textContent).toContain("Public regional catalog event");
    expect(tip()?.textContent).not.toContain("Recovered");
  });

  it("flips to the pointer's other side at the viewport edges and before the open drawer", () => {
    render(<HoverTooltip />);
    const el = () => tip()!;
    // jsdom has no layout: give the card a size (restored below).
    const ow = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "offsetWidth")!;
    const oh = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "offsetHeight")!;
    Object.defineProperty(HTMLElement.prototype, "offsetWidth", { configurable: true, get: () => 220 });
    Object.defineProperty(HTMLElement.prototype, "offsetHeight", { configurable: true, get: () => 100 });
    hover("candidate", "e1", 1200, 700);
    expect(el().style.left).toBe(`${1200 - 14 - 220}px`);
    expect(el().style.top).toBe(`${700 - 14 - 100}px`);
    act(() => useDemo.setState({ selectedEventId: "e2" }));
    hover("candidate", "e1", 700, 100);
    // The drawer is 512 px wide at 1280: 700 + 14 + 220 would run under it.
    expect(el().style.left).toBe(`${700 - 14 - 220}px`);
    Object.defineProperty(HTMLElement.prototype, "offsetWidth", ow);
    Object.defineProperty(HTMLElement.prototype, "offsetHeight", oh);
  });

  it("stays hidden while the guided tour plays", () => {
    render(<HoverTooltip />);
    hover("candidate", "e1");
    expect(tip()).not.toBeNull();
    act(() => useTour.setState({ running: true, step: "public" }));
    expect(tip()).toBeNull();
  });
});

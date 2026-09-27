import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FetchLike } from "../../providers/fetch";
import { initialDemoState, useDemo } from "../../state/demo";
import type { BundleState } from "../types";
import { useTour } from "../tour/store";
import { StationDayButton } from "./StationDayButton";
import { StationDayPanel, StationDayRunSync } from "./StationDayPanel";
import { openStationDay, setStationDayRun, useStationDay } from "./store";
import { syntheticManifest } from "./test-fixture";

// The run sync reads the bundle only through scene/data.ts; the test swaps that module for controllable state.
const data = vi.hoisted(() => ({ bundle: { status: "loading" } as BundleState }));
vi.mock("../data", () => ({ useBundle: () => data.bundle }));

/** The synthetic manifest's own run: the button belongs to it. */
const RUN = syntheticManifest().runId as string;

// The corner wrapper (drei <Html> in the canvas) needs WebGL; the button and the panel are plain DOM.

const serve =
  (status: number, body: unknown = null): FetchLike =>
  async () => ({ ok: status >= 200 && status < 300, status, json: async () => body });

/** Mounts the panel (which loads the manifest) and the button, as the scene does, and lets the load land. */
async function mount(fetchImpl: FetchLike) {
  const utils = render(
    <>
      <StationDayButton />
      <StationDayPanel fetchImpl={fetchImpl} />
    </>,
  );
  await act(async () => {});
  return utils;
}

const button = () => screen.queryByTestId("station-day-button");
const dialog = () => screen.queryByRole("dialog");

function key(k: string, init: KeyboardEventInit = {}, target: EventTarget = document.activeElement ?? document.body) {
  const event = new KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true, ...init });
  act(() => {
    target.dispatchEvent(event);
  });
  return event;
}

/** Stands in for the shell's presenter keys (window, bubble phase), counting what reaches it. */
function shellKeys() {
  const seen: string[] = [];
  const on = (e: KeyboardEvent) => seen.push(e.key);
  window.addEventListener("keydown", on);
  return { seen, off: () => window.removeEventListener("keydown", on) };
}

beforeEach(() => {
  useDemo.setState({ ...initialDemoState, phase: "revealed", revealProgress: 1 });
  useTour.setState({ running: false, step: null, runs: 0 });
  useStationDay.setState({ manifest: null, runId: RUN, open: false });
  data.bundle = { status: "loading" } as BundleState;
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("station-day button", () => {
  it("is absent when the manifest is missing", async () => {
    await mount(serve(404));
    expect(button()).toBeNull();
    expect(dialog()).toBeNull();
  });

  it("is absent when the manifest is invalid", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    await mount(serve(200, { ...syntheticManifest(), legend: "nope" }));
    expect(button()).toBeNull();
  });

  it("appears with a valid manifest, after the reveal only", async () => {
    useDemo.setState({ ...initialDemoState });
    await mount(serve(200, syntheticManifest()));
    expect(button()).toBeNull();
    act(() => useDemo.setState({ phase: "revealing" }));
    expect(button()?.textContent).toBe("Station day");
    expect(button()?.getAttribute("aria-haspopup")).toBe("dialog");
    expect(button()?.getAttribute("aria-expanded")).toBe("false");
  });

  it("keeps its place but hides while the guided tour plays", async () => {
    await mount(serve(200, syntheticManifest()));
    act(() => useTour.setState({ running: true }));
    expect(button()?.style.visibility).toBe("hidden");
  });
});

describe("station-day panel", () => {
  it("does not request the image until opened", async () => {
    await mount(serve(200, syntheticManifest()));
    expect(document.querySelector("img")).toBeNull();
    fireEvent.click(button()!);
    const img = screen.getByTestId("station-day-image") as HTMLImageElement;
    expect(img.getAttribute("src")).toBe("/helicorder/station-day.png");
    expect(img.getAttribute("width")).toBe("1920");
    expect(img.getAttribute("height")).toBe("1350");
  });

  it("opens as a labelled modal dialog with the manifest's title, caption and legend", async () => {
    await mount(serve(200, syntheticManifest()));
    fireEvent.click(button()!);
    const d = dialog()!;
    expect(d.getAttribute("aria-modal")).toBe("true");
    const title = document.getElementById(d.getAttribute("aria-labelledby")!)!;
    expect(title.textContent).toBe("Synthetic station day");
    expect(screen.getByRole("dialog", { name: "Synthetic station day" })).toBe(d);
    expect(screen.getByTestId("station-day-caption").textContent).toBe(
      "Synthetic caption for XX.SYN01 channel ZZZ; gaps are left blank.",
    );
    expect(screen.getByTestId("station-day-meta").textContent).toBe("XX.SYN01.00.ZZZ · 2001-02-03 UTC · 2–20 Hz");
    expect(document.activeElement).toBe(d);
    expect(button()?.getAttribute("aria-expanded")).toBe("true");

    const items = Array.from(screen.getByTestId("station-day-legend").querySelectorAll("li"));
    expect(items.map((li) => li.dataset.key)).toEqual(["tierA", "tierB", "tierC", "public"]);
    expect(items.map((li) => li.textContent)).toEqual([
      "synthetic tier A7",
      "synthetic tier B1,234",
      "synthetic tier C0",
      "synthetic public3",
    ]);
    const swatch = (i: number) => items[i].querySelector<HTMLElement>("[data-shape] > span")!;
    expect(items[1].querySelector("[data-shape]")?.getAttribute("data-shape")).toBe("tick");
    expect(swatch(1).style.opacity).toBe("0.6");
    expect(items[3].querySelector("[data-shape]")?.getAttribute("data-shape")).toBe("diamond");
    expect(swatch(3).style.border).toContain("1.5px solid");
    expect(d.textContent).toContain("synthetic selection rule");
  });

  it("Esc closes it, returns focus to the button, and reaches nothing else", async () => {
    await mount(serve(200, syntheticManifest()));
    const shell = shellKeys();
    useDemo.setState({ selectedEventId: "some-event" });
    button()!.focus();
    fireEvent.click(button()!);
    const esc = key("Escape");
    expect(esc.defaultPrevented).toBe(true);
    expect(dialog()).toBeNull();
    expect(document.activeElement).toBe(button());
    expect(shell.seen).toEqual([]);
    expect(useDemo.getState().selectedEventId).toBe("some-event");
    // Closed, Esc is the shell's again.
    key("Escape");
    expect(shell.seen).toEqual(["Escape"]);
    shell.off();
  });

  it("presenter keys do not reach the scene or the shell while it is open", async () => {
    await mount(serve(200, syntheticManifest()));
    const shell = shellKeys();
    fireEvent.click(button()!);
    for (const k of [" ", "e", "s", "g", "h", "d"]) key(k);
    expect(shell.seen).toEqual([]);
    expect(dialog()).not.toBeNull();
    // Browser shortcuts pass through.
    key("r", { ctrlKey: true });
    expect(shell.seen).toEqual(["r"]);
    shell.off();
  });

  it("the Close button and a backdrop click close it; a click inside does not", async () => {
    await mount(serve(200, syntheticManifest()));
    fireEvent.click(button()!);
    fireEvent.click(screen.getByTestId("station-day-close"));
    expect(dialog()).toBeNull();
    expect(document.activeElement).toBe(button());

    fireEvent.click(button()!);
    fireEvent.pointerDown(screen.getByTestId("station-day-caption"));
    fireEvent.click(screen.getByTestId("station-day-caption"));
    expect(dialog()).not.toBeNull();

    const backdrop = screen.getByTestId("station-day-backdrop");
    fireEvent.pointerDown(backdrop);
    fireEvent.click(backdrop);
    expect(dialog()).toBeNull();
  });

  it("toggles the image between fit-to-screen and full size", async () => {
    await mount(serve(200, syntheticManifest()));
    fireEvent.click(button()!);
    const zoom = screen.getByTestId("station-day-zoom");
    expect(zoom.getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(zoom);
    expect(zoom.getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByTestId("station-day-image").parentElement?.hasAttribute("data-full")).toBe(true);
  });

  it("Tab stays inside the dialog", async () => {
    await mount(serve(200, syntheticManifest()));
    fireEvent.click(button()!);
    const d = dialog()!;
    const controls = Array.from(d.querySelectorAll<HTMLElement>("button, [href]"));
    controls[controls.length - 1].focus();
    expect(key("Tab").defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(controls[0]);
    expect(key("Tab", { shiftKey: true }).defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(controls[controls.length - 1]);
  });
});

describe("station-day run fit", () => {
  it("is absent when another run is loaded (Today), and cannot be opened there", async () => {
    useStationDay.setState({ runId: "another-run" });
    await mount(serve(200, syntheticManifest()));
    expect(button()).toBeNull();
    act(() => openStationDay(null));
    expect(dialog()).toBeNull();
  });

  it("is absent until a bundle is loaded", async () => {
    useStationDay.setState({ runId: null });
    await mount(serve(200, syntheticManifest()));
    expect(button()).toBeNull();
  });

  it("switching to another run closes the panel and hides the button; switching back shows the button", async () => {
    await mount(serve(200, syntheticManifest()));
    fireEvent.click(button()!);
    expect(dialog()).not.toBeNull();
    act(() => setStationDayRun("another-run"));
    expect(dialog()).toBeNull();
    expect(button()).toBeNull();
    const shell = shellKeys();
    key("e", {}, document.body);
    shell.off();
    expect(shell.seen).toEqual(["e"]);
    act(() => setStationDayRun(RUN));
    expect(button()).not.toBeNull();
    expect(dialog()).toBeNull();
  });

  it("StationDayRunSync records the loaded bundle's run, and null while none is ready", () => {
    useStationDay.setState({ runId: null });
    data.bundle = { status: "ready", meta: { scene: { runId: RUN } } } as unknown as BundleState;
    const { rerender } = render(<StationDayRunSync />);
    expect(useStationDay.getState().runId).toBe(RUN);
    data.bundle = { status: "ready", meta: { scene: { runId: "another-run" } } } as unknown as BundleState;
    rerender(<StationDayRunSync />);
    expect(useStationDay.getState().runId).toBe("another-run");
    data.bundle = { status: "loading" } as BundleState;
    rerender(<StationDayRunSync />);
    expect(useStationDay.getState().runId).toBeNull();
  });
});

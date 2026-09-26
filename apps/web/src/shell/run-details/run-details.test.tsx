/**
 * The Run details overlay: opens from its button or D, closes from Close or Esc, prints every
 * top-level `ProcessingRun` key verbatim, and draws one sweep mark per `SweepPoint`. The run and
 * the sweep are the mock bundle's own (`apps/web/public/data/mock/`).
 */
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ProviderRoot, StaticBundleProvider, type FetchLike, type ProcessingRun, type Validation } from "@/providers";
import { useDemo } from "@/state/demo";
import mockMeta from "../../../public/data/mock/meta.json";
import mockValidation from "../../../public/data/mock/validation.json";
import { Shell } from "../Shell";
import { bundleFiles, fakeFetch, pendingFetch, type FixtureOptions } from "../test-fixture";
import { sweptParam } from "./sweep";

const validation = mockValidation as unknown as Validation;
const run = mockMeta.run as unknown as ProcessingRun;

beforeEach(() => {
  useDemo.getState().reset();
});

afterEach(() => {
  cleanup();
});

function mount(options: FixtureOptions, fetchImpl?: FetchLike) {
  const provider = new StaticBundleProvider("mock", {
    fetchImpl: fetchImpl ?? fakeFetch({ mock: bundleFiles("mock-run", options) }),
  });
  render(
    <ProviderRoot mode="mock" provider={provider}>
      <Shell />
    </ProviderRoot>,
  );
}

async function mountReady(options: FixtureOptions = { validation, meta: { run } }) {
  mount(options);
  await screen.findByTestId("counter-public");
  await act(async () => {});
}

function press(key: string, init: KeyboardEventInit = {}): KeyboardEvent {
  const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...init });
  act(() => {
    window.dispatchEvent(event);
  });
  return event;
}

const button = () => screen.getByRole("button", { name: "Run details" });

describe("run details panel", () => {
  it("is closed until its button is pressed, and Close hides it again", async () => {
    await mountReady();
    expect(screen.queryByTestId("run-details")).toBeNull();
    expect(button().getAttribute("aria-expanded")).toBe("false");
    act(() => {
      fireEvent.click(button());
    });
    expect(screen.getByRole("dialog", { name: "Run details" })).toBeTruthy();
    expect(button().getAttribute("aria-expanded")).toBe("true");
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Close" }));
    });
    expect(screen.queryByTestId("run-details")).toBeNull();
  });

  it("D toggles it and Esc closes it; D does nothing while loading or with modifiers", async () => {
    mount({}, pendingFetch);
    press("d");
    expect(screen.queryByTestId("run-details")).toBeNull();
    cleanup();

    await mountReady();
    const event = press("d");
    expect(event.defaultPrevented).toBe(true);
    expect(screen.getByTestId("run-details")).toBeTruthy();
    press("D");
    expect(screen.queryByTestId("run-details")).toBeNull();
    press("d", { metaKey: true });
    expect(screen.queryByTestId("run-details")).toBeNull();
    press("d");
    expect(screen.getByTestId("run-details")).toBeTruthy();
    press("Escape");
    expect(screen.queryByTestId("run-details")).toBeNull();
    // Esc also cleared the selection (docs/02 §6); both are "dismiss".
    expect(useDemo.getState().selectedEventId).toBeNull();
  });

  it("prints every top-level ProcessingRun key and its values verbatim", async () => {
    await mountReady();
    press("d");
    const tree = screen.getByTestId("run-json");
    for (const key of Object.keys(run)) {
      expect(within(tree).getAllByText(key, { exact: true }).length).toBeGreaterThan(0);
    }
    expect(screen.getByTestId("run-details-id").textContent).toBe(run.id);
    // Leaves are JSON.stringify'd: strings keep their quotes, numbers their full precision.
    expect(tree.textContent).toContain(JSON.stringify(run.gitSha));
    expect(tree.textContent).toContain(JSON.stringify(run.windowLabel));
    expect(tree.textContent).toContain(JSON.stringify(run.windowStart));
    // Nested dicts print their own keys and values too.
    for (const [key, value] of Object.entries(run.runtimeS)) {
      expect(within(tree).getAllByText(key, { exact: true }).length).toBeGreaterThan(0);
      expect(tree.textContent).toContain(JSON.stringify(value));
    }
    for (const id of run.stationIds) expect(tree.textContent).toContain(JSON.stringify(id));
  });

  it("draws the sweep with one mark per SweepPoint per series and the swept key on the x axis", async () => {
    await mountReady();
    press("d");
    const plot = screen.getByTestId("sweep-plot");
    const svg = within(plot).getByRole("img", { name: "Association sweep" });
    for (const field of ["candidates", "tierA", "recoveredPublic"]) {
      const marks = svg.querySelectorAll(`[data-series="${field}"] [data-mark]`);
      expect(marks).toHaveLength(validation.sweep.length);
    }
    expect(screen.getByTestId("sweep-x-label").textContent).toBe(sweptParam(validation.sweep));
    expect(plot.textContent).toContain("candidates");
    expect(plot.textContent).toContain("tierA");
    expect(plot.textContent).toContain("recoveredPublic");
  });

  it("hides the sweep block when the sweep is empty or validation.json is missing", async () => {
    await mountReady({ validation: { ...validation, sweep: [] }, meta: { run } });
    press("d");
    expect(screen.getByTestId("run-json")).toBeTruthy();
    expect(screen.queryByTestId("sweep-plot")).toBeNull();
    cleanup();

    await mountReady({ meta: { run } });
    press("d");
    expect(screen.getByTestId("run-json")).toBeTruthy();
    expect(screen.queryByTestId("sweep-plot")).toBeNull();
  });
});

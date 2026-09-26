/**
 * DEMO-01 acceptance: keys drive the store; counters read only `AnalysisSummary` and
 * `revealProgress`; the SYNTHETIC banner, the mode label, the loading line and the error panel
 * all come from the provider. The bundle is the tiny hand-made one in `./test-fixture.ts`.
 */
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ProviderRoot, SCHEMA_VERSION, StaticBundleProvider, type FetchLike, type StaticMode } from "@/providers";
import { useDemo } from "@/state/demo";
import { formatCount } from "./counter-values";
import { Shell } from "./Shell";
import { HERO_EVENT_ID, SUMMARY, bundleFiles, fakeFetch, pendingFetch, type FixtureOptions } from "./test-fixture";

const state = () => useDemo.getState();

beforeEach(() => {
  state().reset();
});

afterEach(() => {
  cleanup();
  vi.unstubAllEnvs();
});

function mount(mode: StaticMode = "mock", options: FixtureOptions = {}, fetchImpl?: FetchLike) {
  const provider = new StaticBundleProvider(mode, {
    fetchImpl: fetchImpl ?? fakeFetch({ [mode]: bundleFiles(`${mode}-run`, options) }),
  });
  render(
    <ProviderRoot mode={mode} provider={provider}>
      <Shell />
    </ProviderRoot>,
  );
  return provider;
}

/** Mount and wait until the bundle is on screen (the PUBLIC counter is the first thing to render). */
async function mountReady(mode: StaticMode = "mock", options: FixtureOptions = {}) {
  const provider = mount(mode, options);
  await screen.findByTestId("counter-public");
  // Flush the passive effects of the "ready" render, so the keyboard listener already knows the
  // hero id and that the bundle is loaded (in the browser this happens within the same frame).
  await act(async () => {});
  return provider;
}

function press(key: string, init: KeyboardEventInit = {}): KeyboardEvent {
  const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...init });
  act(() => {
    window.dispatchEvent(event);
  });
  return event;
}

/** What the scene does at the end of the settle (docs/lanes/H3 → Demo store semantics). */
function sceneFinishesReveal() {
  act(() => {
    useDemo.setState({ phase: "revealed", revealProgress: 1 });
  });
}

describe("keyboard drives the store", () => {
  it("Space walks the beats: reveal, strict, time; gated on phase, not on progress", async () => {
    await mountReady();
    expect(state().phase).toBe("public");

    const space = press(" ");
    expect(space.defaultPrevented).toBe(true);
    expect(state().phase).toBe("revealing");
    expect(state().filter).toBe("all");

    // Mid-reveal, even at full progress, Space does nothing: the scene hasn't settled.
    act(() => useDemo.setState({ revealProgress: 1 }));
    press(" ");
    expect(state().phase).toBe("revealing");
    expect(state().filter).toBe("all");

    sceneFinishesReveal();
    press(" ");
    expect(state().filter).toBe("strict");
    expect(state().timeMode).toBe(false);

    press(" ");
    expect(state().timeMode).toBe(true);

    press(" ");
    expect(state()).toMatchObject({ phase: "revealed", filter: "strict", timeMode: true });
  });

  it("Space never reveals before the bundle is ready", () => {
    mount("mock", {}, pendingFetch);
    press(" ");
    expect(state().phase).toBe("public");
  });

  it("R resets to the start frame", async () => {
    await mountReady();
    press(" ");
    sceneFinishesReveal();
    press("s");
    press("t");
    press("p");
    press("r");
    expect(state()).toMatchObject({
      phase: "public",
      revealProgress: 0,
      filter: "public",
      timeMode: false,
      selectedEventId: null,
      view: "oblique",
    });
  });

  it("S toggles strict and back to all", async () => {
    await mountReady();
    press("s");
    expect(state().filter).toBe("strict");
    press("S");
    expect(state().filter).toBe("all");
  });

  it("T toggles time mode", async () => {
    await mountReady();
    press("t");
    expect(state().timeMode).toBe(true);
    press("t");
    expect(state().timeMode).toBe(false);
  });

  it("E selects the hero event from meta.scene and Esc clears it", async () => {
    await mountReady();
    press("e");
    expect(state().selectedEventId).toBe(HERO_EVENT_ID);
    press("Escape");
    expect(state().selectedEventId).toBeNull();
  });

  it("E is a no-op when the bundle has no hero event", async () => {
    await mountReady("mock", { scene: { heroEventId: null } });
    press("e");
    expect(state().selectedEventId).toBeNull();
  });

  it("P toggles plan and oblique", async () => {
    await mountReady();
    press("p");
    expect(state().view).toBe("plan");
    press("p");
    expect(state().view).toBe("oblique");
  });

  it("ignores repeats, browser shortcuts, handled events and text fields", async () => {
    await mountReady();
    press("s", { repeat: true });
    expect(state().filter).toBe("public");
    press("s", { metaKey: true });
    press("s", { ctrlKey: true });
    press("s", { altKey: true });
    expect(state().filter).toBe("public");

    const handled = new KeyboardEvent("keydown", { key: "s", bubbles: true, cancelable: true });
    handled.preventDefault();
    act(() => {
      window.dispatchEvent(handled);
    });
    expect(state().filter).toBe("public");

    const input = document.createElement("input");
    document.body.appendChild(input);
    act(() => {
      fireEvent.keyDown(input, { key: "s" });
    });
    expect(state().filter).toBe("public");
    input.remove();

    const unknown = press("x");
    expect(unknown.defaultPrevented).toBe(false);
  });
});

describe("counters read AnalysisSummary and revealProgress only", () => {
  const publicCount = SUMMARY.publicCatalogCount;
  const candidates = SUMMARY.candidateCount;

  it("shows PUBLIC and dims the other two before the reveal", async () => {
    await mountReady();
    expect(screen.getByTestId("counter-public").textContent).toBe(formatCount(publicCount));
    expect(screen.getByTestId("counter-recovered").textContent).toBe("—");
    expect(screen.getByTestId("counter-strict").textContent).toBe("—");
    expect(screen.getByTestId("counter-recovered").closest("[data-dim]")).not.toBeNull();
    expect(screen.getByTestId("counter-public").closest("[data-dim]")).toBeNull();
  });

  it("interpolates RECOVERED with the reveal clock and keeps STRICT hidden", async () => {
    await mountReady();
    act(() => useDemo.setState({ phase: "revealing", revealProgress: 0.5 }));
    const expected = Math.round(publicCount + (candidates - publicCount) * 0.5);
    expect(expected).toBe(76);
    expect(screen.getByTestId("counter-recovered").textContent).toBe(formatCount(expected));
    expect(screen.getByTestId("counter-strict").textContent).toBe("—");
    expect(screen.getByTestId("counter-recovered").closest("[data-dim]")).toBeNull();

    act(() => useDemo.setState({ revealProgress: 0.25 }));
    expect(screen.getByTestId("counter-recovered").textContent).toBe(
      formatCount(Math.round(publicCount + (candidates - publicCount) * 0.25)),
    );
  });

  it("settles on candidateCount and strictQualityCount once revealed", async () => {
    await mountReady();
    act(() => useDemo.setState({ phase: "revealed", revealProgress: 1 }));
    expect(screen.getByTestId("counter-recovered").textContent).toBe(formatCount(candidates));
    expect(screen.getByTestId("counter-strict").textContent).toBe(formatCount(SUMMARY.strictQualityCount));
    expect(screen.getByTestId("counter-public").textContent).toBe(formatCount(publicCount));
  });

  it("groups digits in large counts", async () => {
    await mountReady("mock", { summary: { candidateCount: 2048 } });
    act(() => useDemo.setState({ phase: "revealed", revealProgress: 1 }));
    expect(screen.getByTestId("counter-recovered").textContent).toBe("2,048");
  });
});

describe("labels come from the provider", () => {
  it("shows the SYNTHETIC banner and the provider's mode label for a synthetic bundle", async () => {
    const provider = await mountReady("mock", { isSynthetic: true });
    const banner = screen.getByRole("status");
    expect(banner.textContent).toMatch(/synthetic data/i);
    expect(banner.textContent).toMatch(/not a catalog/i);
    const info = await provider.info();
    expect(screen.getByTestId("mode-label").textContent).toBe(info.label);
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Hidden Quakes");
  });

  it("shows no banner for a real bundle", async () => {
    const provider = await mountReady("showcase", { isSynthetic: false });
    expect(screen.queryByRole("status")).toBeNull();
    const info = await provider.info();
    expect(screen.getByTestId("mode-label").textContent).toBe(info.label);
  });

  it("names the mode while loading and renders no counters or button", () => {
    mount("showcase", {}, pendingFetch);
    expect(screen.getByTestId("mode-label").textContent).toBe("Loading showcase…");
    expect(screen.queryByTestId("counter-public")).toBeNull();
    expect(screen.queryByRole("button", { name: /reveal hidden signal/i })).toBeNull();
  });
});

describe("error state", () => {
  it("renders the provider's schemaVersion message as an alert naming both versions", async () => {
    mount("showcase", { meta: { schemaVersion: "0.9" } });
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("0.9");
    expect(alert.textContent).toContain(SCHEMA_VERSION);
    expect(screen.queryByTestId("counter-public")).toBeNull();
    expect(screen.queryByRole("button", { name: /reveal hidden signal/i })).toBeNull();
    expect(screen.getByTestId("mode-label").textContent).toBe("showcase");
  });
});

describe("controls", () => {
  it("REVEAL calls reveal() and then leaves the screen", async () => {
    await mountReady();
    const button = screen.getByRole("button", { name: /reveal hidden signal/i });
    act(() => {
      fireEvent.click(button);
    });
    expect(state().phase).toBe("revealing");
    expect(screen.queryByRole("button", { name: /reveal hidden signal/i })).toBeNull();
  });

  it("filter pills appear once the reveal starts and call setFilter", async () => {
    await mountReady();
    expect(screen.queryByRole("group", { name: "Event filter" })).toBeNull();
    act(() => state().reveal());
    const group = screen.getByRole("group", { name: "Event filter" });
    const pills = Array.from(group.querySelectorAll("button"));
    expect(pills.map((b) => b.textContent)).toEqual(["Public", "All", "Strict"]);
    expect(pills.map((b) => b.getAttribute("aria-pressed"))).toEqual(["false", "true", "false"]);

    act(() => {
      fireEvent.click(pills[2]);
    });
    expect(state().filter).toBe("strict");
    await waitFor(() => expect(pills[2].getAttribute("aria-pressed")).toBe("true"));

    act(() => {
      fireEvent.click(pills[0]);
    });
    expect(state().filter).toBe("public");
  });
});

describe("download candidate catalog (overnight feature 2)", () => {
  it("appears once the reveal starts and saves CSV and GeoJSON built from the bundle", async () => {
    await mountReady();
    expect(screen.queryByTestId("download-catalog")).toBeNull();
    press(" ");
    const group = screen.getByTestId("download-catalog");
    expect(group.getAttribute("aria-label")).toBe("Download candidate catalog");
    // jsdom has no Blob URLs or downloads: capture the anchor the button clicks.
    const saved: { name: string; href: string }[] = [];
    vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: () => "blob:test", revokeObjectURL: () => {} }));
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      saved.push({ name: this.download, href: this.href });
    });
    fireEvent.click(screen.getByTestId("download-csv"));
    fireEvent.click(screen.getByTestId("download-geojson"));
    click.mockRestore();
    expect(saved.map((s) => s.name)).toEqual([
      "synthetic-hidden-quakes-candidates-mock-run.csv",
      "synthetic-hidden-quakes-candidates-mock-run.geojson",
    ]);
    expect(saved.every((s) => s.href === "blob:test")).toBe(true);
  });
});

describe("share links (overnight feature 3)", () => {
  afterEach(() => {
    window.history.replaceState(null, "", "/");
  });

  it("?event=<id> for a bundled event starts the reveal and opens its drawer once revealed", async () => {
    window.history.replaceState(null, "", "/?mode=mock&event=ev-1");
    await mountReady();
    expect(state().phase).toBe("revealing");
    expect(state().selectedEventId).toBeNull();
    sceneFinishesReveal();
    expect(state().selectedEventId).toBe("ev-1");
    expect(window.location.search).toBe("?mode=mock&event=ev-1");
  });

  it("ignores an id the bundle does not have", async () => {
    window.history.replaceState(null, "", "/?event=not-in-this-run");
    await mountReady();
    expect(state().phase).toBe("public");
    expect(state().selectedEventId).toBeNull();
  });

  it("mirrors the selection into the address bar and removes it when the drawer closes", async () => {
    window.history.replaceState(null, "", "/?mode=mock");
    await mountReady();
    press(" ");
    sceneFinishesReveal();
    press("e");
    expect(state().selectedEventId).toBe(HERO_EVENT_ID);
    expect(window.location.search).toBe(`?mode=mock&event=${HERO_EVENT_ID}`);
    press("Escape");
    expect(window.location.search).toBe("?mode=mock");
  });
});

/**
 * Mode pills: SHOWCASE always, LIVE only behind the API-04 flag, MOCK only as the current mode,
 * never SNAPSHOT; a switch is a full navigation through `navigateToMode`.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ProviderRoot, StaticBundleProvider, navigateToMode, type StaticMode } from "@/providers";
import { useDemo } from "@/state/demo";
import { Shell } from "./Shell";
import { bundleFiles, fakeFetch, type FixtureOptions } from "./test-fixture";

vi.mock("@/providers", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/providers")>();
  return { ...actual, navigateToMode: vi.fn() };
});

beforeEach(() => {
  useDemo.getState().reset();
  vi.mocked(navigateToMode).mockClear();
});

afterEach(() => {
  cleanup();
  vi.unstubAllEnvs();
});

async function mountReady(mode: StaticMode, options: FixtureOptions = {}) {
  const provider = new StaticBundleProvider(mode, {
    fetchImpl: fakeFetch({ [mode]: bundleFiles(`${mode}-run`, options) }),
  });
  render(
    <ProviderRoot mode={mode} provider={provider}>
      <Shell />
    </ProviderRoot>,
  );
  await screen.findByTestId("counter-public");
  return Array.from(screen.getByRole("group", { name: "Data mode" }).querySelectorAll("button"));
}

describe("mode pills", () => {
  it("shows MOCK as current in mock mode, SHOWCASE beside it, and no LIVE without the flag", async () => {
    const pills = await mountReady("mock");
    expect(pills.map((b) => b.textContent)).toEqual(["mock", "showcase"]);
    expect(pills.map((b) => b.getAttribute("aria-pressed"))).toEqual(["true", "false"]);
  });

  it("shows only SHOWCASE in showcase mode", async () => {
    const pills = await mountReady("showcase", { isSynthetic: false });
    expect(pills.map((b) => b.textContent)).toEqual(["showcase"]);
    expect(pills[0].getAttribute("aria-pressed")).toBe("true");
  });

  it("adds LIVE when NEXT_PUBLIC_LIVE_ENABLED is set", async () => {
    vi.stubEnv("NEXT_PUBLIC_LIVE_ENABLED", "1");
    const pills = await mountReady("showcase", { isSynthetic: false });
    expect(pills.map((b) => b.textContent)).toEqual(["showcase", "live"]);
  });

  it("never offers SNAPSHOT: in snapshot mode nothing is current and the label alone says so", async () => {
    const pills = await mountReady("snapshot", { isSynthetic: false });
    expect(pills.map((b) => b.textContent)).toEqual(["showcase"]);
    expect(pills[0].getAttribute("aria-pressed")).toBe("false");
    expect(screen.getByTestId("mode-label").textContent).toMatch(/^Snapshot/);
  });

  it("navigates on a switch and does nothing on the current mode", async () => {
    const pills = await mountReady("mock");
    fireEvent.click(pills[0]);
    expect(navigateToMode).not.toHaveBeenCalled();
    fireEvent.click(pills[1]);
    expect(navigateToMode).toHaveBeenCalledWith("showcase");
  });
});

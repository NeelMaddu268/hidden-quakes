/**
 * Mode pills: SHOWCASE always, TODAY (snapshot) behind the build's snapshot flag or as the
 * current mode (WEB-12), LIVE only behind the API-04 flag, MOCK only as the current mode; a
 * switch is a full navigation through `navigateToMode`. API-05: during live failover the label
 * says Snapshot while LIVE stays the pressed pill.
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LiveProvider, ProviderRoot, StaticBundleProvider, navigateToMode, type StaticMode } from "@/providers";
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

  it("offers TODAY only when the build carries a snapshot, and it switches to ?mode=snapshot", async () => {
    vi.stubEnv("NEXT_PUBLIC_SNAPSHOT_AVAILABLE", "1");
    const pills = await mountReady("showcase", { isSynthetic: false });
    expect(pills.map((b) => b.textContent)).toEqual(["showcase", "today"]);
    expect(pills.map((b) => b.getAttribute("aria-pressed"))).toEqual(["true", "false"]);
    fireEvent.click(pills[1]);
    expect(navigateToMode).toHaveBeenCalledWith("snapshot");
  });

  it("presses TODAY in snapshot mode, SHOWCASE goes back, and the label is the snapshot's window and run", async () => {
    const pills = await mountReady("snapshot", { isSynthetic: false });
    expect(pills.map((b) => b.textContent)).toEqual(["showcase", "today"]);
    expect(pills.map((b) => b.getAttribute("aria-pressed"))).toEqual(["false", "true"]);
    expect(screen.getByTestId("mode-label").textContent).toBe("Snapshot · window · run snapshot-run");
    fireEvent.click(pills[0]);
    expect(navigateToMode).toHaveBeenCalledWith("showcase");
  });

  it("LIVE navigates to ?mode=live only when NEXT_PUBLIC_LIVE_ENABLED=1", async () => {
    vi.stubEnv("NEXT_PUBLIC_LIVE_ENABLED", "1");
    const pills = await mountReady("showcase", { isSynthetic: false });
    fireEvent.click(pills[1]);
    expect(navigateToMode).toHaveBeenCalledWith("live");
    cleanup();
    vi.stubEnv("NEXT_PUBLIC_LIVE_ENABLED", "");
    const without = await mountReady("showcase", { isSynthetic: false });
    expect(without.map((b) => b.textContent)).toEqual(["showcase"]);
    expect(vi.mocked(navigateToMode)).toHaveBeenCalledTimes(1);
  });

  it("keeps LIVE pressed and shows the snapshot label while live is failed over", async () => {
    vi.stubEnv("NEXT_PUBLIC_LIVE_ENABLED", "1");
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    // Every `/api/live/*` request 404s (the fixture serves only `/data/<mode>/`): a failed
    // fetch, exactly like a killed worker, so the root fails over to the snapshot bundle.
    const fetchImpl = fakeFetch({ snapshot: bundleFiles("snapshot-run", { isSynthetic: false }) });
    const provider = new LiveProvider("/api/live", {
      fetchImpl,
      fallback: new StaticBundleProvider("snapshot", { fetchImpl }),
    });
    render(
      <ProviderRoot mode="live" provider={provider}>
        <Shell />
      </ProviderRoot>,
    );
    await screen.findByTestId("counter-public");
    await waitFor(() => expect(screen.getByTestId("mode-label").textContent).toMatch(/^Snapshot/));
    expect(screen.getByTestId("mode-label").textContent).toContain("snapshot-run");
    const pills = Array.from(screen.getByRole("group", { name: "Data mode" }).querySelectorAll("button"));
    expect(pills.map((b) => b.textContent)).toEqual(["showcase", "live"]);
    expect(pills.map((b) => b.getAttribute("aria-pressed"))).toEqual(["false", "true"]);
  });

  it("navigates on a switch and does nothing on the current mode", async () => {
    const pills = await mountReady("mock");
    fireEvent.click(pills[0]);
    expect(navigateToMode).not.toHaveBeenCalled();
    fireEvent.click(pills[1]);
    expect(navigateToMode).toHaveBeenCalledWith("showcase");
  });
});

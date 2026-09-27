/**
 * Page composition: the scene, the shell and the placeholders mount under one ProviderRoot that
 * reads `?mode=` from the URL. The scene is stubbed (no WebGL in jsdom); the bundle is served by
 * a fake fetch so the counters render real provider data.
 */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useDemo } from "@/state/demo";
import { formatCount } from "@/shell";
import { SUMMARY, bundleFiles, fakeFetch } from "@/shell/test-fixture";
import Page from "./page";

vi.mock("@/scene", () => ({
  Scene: () => <div data-testid="scene-stub" />,
  // The shell mounts H3's tour button (REQ-H3-15); the stub keeps WebGL out of jsdom.
  TourButton: () => <button type="button" data-testid="tour-button-stub" />,
  // The shell's attract mode (DEMO-05) reads the tour store; a real zustand store stands in.
  startTour: () => undefined,
  useTour: Object.assign(() => false, { getState: () => ({ running: false }), subscribe: () => () => undefined }),
}));

beforeEach(() => {
  useDemo.getState().reset();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.history.replaceState({}, "", "/");
});

describe("page", () => {
  it("composes the scene and the shell over the default showcase bundle", async () => {
    vi.stubGlobal(
      "fetch",
      fakeFetch({ showcase: bundleFiles("show", { isSynthetic: false, meta: { mode: "showcase" } }) }),
    );
    render(<Page />);
    expect(screen.getByTestId("scene-stub")).toBeTruthy();
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Hidden Quakes");
    await screen.findByRole("button", { name: /reveal hidden signal/i });
    expect(screen.getByTestId("counter-public").textContent).toBe(formatCount(SUMMARY.publicCatalogCount));
    expect(screen.getByTestId("counter-recovered").textContent).toBe("—");
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.getByTestId("mode-label").textContent).toMatch(/^Showcase/);
  });

  it("?mode=mock loads the mock bundle and shows the SYNTHETIC banner", async () => {
    vi.stubGlobal("fetch", fakeFetch({ mock: bundleFiles("mock") }));
    window.history.replaceState({}, "", "/?mode=mock");
    render(<Page />);
    await screen.findByRole("status");
    expect(screen.getByRole("status").textContent).toMatch(/synthetic data/i);
    await screen.findByTestId("counter-public");
  });
});

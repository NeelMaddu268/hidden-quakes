// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { parseListenManifest, type ListenManifest } from "../audio/manifest";
import { HeroListen } from "./HeroListen";

// The hero clip's manifest, controllable per test (null: no manifest).
const data = vi.hoisted(() => ({ hero: null as ListenManifest | null }));
vi.mock("../audio/manifest", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../audio/manifest")>()),
  useListenManifest: () => data.hero,
}));

// ---- Synthetic hero manifest (rule 5) -----------------------------------------------------------
const HERO_ID = "hq-test-000042";
function heroClip(): ListenManifest {
  return parseListenManifest({
    stationId: "XX.B01",
    channel: "GHZ",
    startUtc: "2030-05-06T07:08:00.500Z",
    endUtc: "2030-05-06T07:09:00.500Z",
    speed: 120,
    durationS: 0.5,
    note: "Synthetic hero clip.",
    heroEventId: HERO_ID,
    selection: { eventOriginUtc: "2030-05-06T07:08:30.500Z" },
    files: { ogg: { name: "h.ogg" }, mp3: { name: "h.mp3" } },
  })!;
}

let play: ReturnType<typeof vi.fn>;
let pause: ReturnType<typeof vi.fn>;
beforeEach(() => {
  data.hero = heroClip();
  play = vi.fn(() => Promise.resolve());
  pause = vi.fn();
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(play as () => Promise<void>);
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(pause as () => void);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("HeroListen", () => {
  it("renders only for the manifest's hero event", () => {
    const { rerender } = render(<HeroListen eventId="hq-test-000041" />);
    expect(screen.queryByTestId("hero-listen")).toBeNull();
    rerender(<HeroListen eventId={HERO_ID} />);
    expect(screen.getByTestId("hero-listen")).toBeTruthy();
    rerender(<HeroListen eventId="hq-test-000043" />);
    expect(screen.queryByTestId("hero-listen")).toBeNull();
  });

  it("renders nothing without a manifest", () => {
    data.hero = null;
    render(<HeroListen eventId={HERO_ID} />);
    expect(screen.queryByTestId("hero-listen")).toBeNull();
  });

  it("labels the clip from the manifest and plays it from the start", () => {
    render(<HeroListen eventId={HERO_ID} />);
    const button = screen.getByTestId("hero-listen-button");
    expect(button.textContent).toBe("Listen: XX.B01, sped up 120×");
    expect(button.getAttribute("title")).toBe(
      "XX.B01 channel GHZ · 2030-05-06 07:08:00.500–07:09:00.500 UTC · event origin 07:08:30.500 UTC. Synthetic hero clip.",
    );
    const audio = screen.getByTestId("hero-listen-audio") as HTMLAudioElement;
    Object.defineProperty(audio, "currentTime", { configurable: true, writable: true, value: 0.3 });
    fireEvent.click(button);
    expect(play).toHaveBeenCalledTimes(1);
    expect(audio.currentTime).toBe(0);
    expect(button.getAttribute("aria-pressed")).toBe("true");
    fireEvent.ended(audio);
    expect(button.getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(button);
    fireEvent.click(button);
    expect(pause).toHaveBeenCalled();
  });

  it("switching to another event mid-clip stops it, and the hero's button comes back ready to play", () => {
    const { rerender } = render(<HeroListen eventId={HERO_ID} />);
    fireEvent.click(screen.getByTestId("hero-listen-button"));
    expect(screen.getByTestId("hero-listen-button").getAttribute("aria-pressed")).toBe("true");
    rerender(<HeroListen eventId="hq-test-000043" />);
    expect(pause).toHaveBeenCalled();
    rerender(<HeroListen eventId={HERO_ID} />);
    expect(screen.getByTestId("hero-listen-button").getAttribute("aria-pressed")).toBe("false");
  });

  it("a refused play() resets the button and explains on hover", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    play.mockImplementation(() => Promise.reject(Object.assign(new Error("blocked"), { name: "NotAllowedError" })));
    render(<HeroListen eventId={HERO_ID} />);
    const button = screen.getByTestId("hero-listen-button");
    fireEvent.click(button);
    await act(async () => {
      await Promise.resolve();
    });
    expect(button.getAttribute("aria-pressed")).toBe("false");
    expect(button.getAttribute("title")).toMatch(/^Audio did not start \(NotAllowedError: blocked\)/);
  });
});

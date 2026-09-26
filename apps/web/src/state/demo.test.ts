import { beforeEach, describe, expect, it, vi } from "vitest";
import { initialDemoState, setRevealProgress, useDemo, type DemoState } from "./demo";

const s = () => useDemo.getState();

/** Only the data fields, for comparing against `initialDemoState`. */
function data(state: DemoState) {
  const { phase, revealProgress, filter, timeMode, tNow, playing, selectedEventId, view } = state;
  return { phase, revealProgress, filter, timeMode, tNow, playing, selectedEventId, view };
}

beforeEach(() => {
  s().reset();
});

describe("initial state (docs/02 §6)", () => {
  it("starts on the public frame", () => {
    expect(data(s())).toEqual({
      phase: "public",
      revealProgress: 0,
      filter: "public",
      timeMode: false,
      tNow: null,
      playing: false,
      selectedEventId: null,
      view: "oblique",
    });
    expect(data(s())).toEqual(initialDemoState);
  });

  it("exposes exactly the docs/02 actions", () => {
    const actions = Object.entries(s())
      .filter(([, v]) => typeof v === "function")
      .map(([k]) => k)
      .sort();
    expect(actions).toEqual(
      ["reset", "reveal", "select", "setFilter", "setPlaying", "setTNow", "setTimeMode", "setView"].sort(),
    );
  });

  it("freezes the initial snapshot so nobody can mutate the start frame", () => {
    expect(Object.isFrozen(initialDemoState)).toBe(true);
  });
});

describe("reveal()", () => {
  it("moves public -> revealing with progress 0 and shows all events", () => {
    s().reveal();
    expect(s().phase).toBe("revealing");
    expect(s().revealProgress).toBe(0);
    expect(s().filter).toBe("all");
  });

  it("is a no-op while revealing or after revealed (Space can't restart it)", () => {
    s().reveal();
    setRevealProgress(0.4);
    s().reveal();
    expect(s().phase).toBe("revealing");
    expect(s().revealProgress).toBe(0.4);

    setRevealProgress(1);
    s().setFilter("strict");
    s().reveal();
    expect(s().phase).toBe("revealed");
    expect(s().filter).toBe("strict");
  });

  it("keeps the current camera view (the reveal also starts from plan view)", () => {
    s().setView("plan");
    s().reveal();
    expect(s().view).toBe("plan");
  });
});

describe("setRevealProgress() (scene-only helper)", () => {
  it("advances progress during revealing and marks revealed at exactly 1", () => {
    s().reveal();
    setRevealProgress(0.25);
    expect(s().revealProgress).toBe(0.25);
    expect(s().phase).toBe("revealing");
    setRevealProgress(1.0000001);
    expect(s().revealProgress).toBe(1);
    expect(s().phase).toBe("revealed");
  });

  it("clamps negatives to 0", () => {
    s().reveal();
    setRevealProgress(-2);
    expect(s().revealProgress).toBe(0);
  });

  it("is ignored outside revealing, so a late frame after reset() can't resurrect the reveal", () => {
    setRevealProgress(0.5);
    expect(data(s())).toEqual(initialDemoState);

    s().reveal();
    setRevealProgress(1);
    setRevealProgress(0.3);
    expect(s().revealProgress).toBe(1);
    expect(s().phase).toBe("revealed");
  });

  it("does not notify subscribers when progress is unchanged", () => {
    s().reveal();
    setRevealProgress(0.5);
    const listener = vi.fn();
    const unsubscribe = useDemo.subscribe(listener);
    setRevealProgress(0.5);
    unsubscribe();
    expect(listener).not.toHaveBeenCalled();
  });

  it("fails loudly on NaN instead of freezing the reveal", () => {
    s().reveal();
    expect(() => setRevealProgress(Number.NaN)).toThrow(/NaN/);
  });
});

describe("reset()", () => {
  it("returns every field to the start frame from any state", () => {
    s().reveal();
    setRevealProgress(0.7);
    s().setFilter("strict");
    s().setTimeMode(true);
    s().setTNow(1_000);
    s().setPlaying(true);
    s().select("evt-1");
    s().setView("side");
    s().reset();
    expect(data(s())).toEqual(initialDemoState);
  });

  it("covers everything docs/02 names: public, progress 0, filter public, selection null", () => {
    s().reveal();
    setRevealProgress(1);
    s().select("evt-2");
    s().reset();
    expect(s().phase).toBe("public");
    expect(s().revealProgress).toBe(0);
    expect(s().filter).toBe("public");
    expect(s().selectedEventId).toBeNull();
  });

  it("allows a fresh reveal afterwards", () => {
    s().reveal();
    setRevealProgress(1);
    s().reset();
    s().reveal();
    expect(s().phase).toBe("revealing");
    expect(s().revealProgress).toBe(0);
  });
});

describe("filter, time, selection, view", () => {
  it("setFilter sets any filter", () => {
    for (const f of ["all", "strict", "public"] as const) {
      s().setFilter(f);
      expect(s().filter).toBe(f);
    }
  });

  it("setTimeMode(false) stops playback and shows every event", () => {
    s().setTimeMode(true);
    s().setTNow(42);
    s().setPlaying(true);
    s().setTimeMode(false);
    expect(s().timeMode).toBe(false);
    expect(s().playing).toBe(false);
    expect(s().tNow).toBeNull();
  });

  it("setTimeMode(true) keeps tNow so the scrubber decides where to start", () => {
    s().setTNow(7);
    s().setTimeMode(true);
    expect(s().timeMode).toBe(true);
    expect(s().tNow).toBe(7);
  });

  it("setTNow accepts null to show every event", () => {
    s().setTNow(123.456);
    expect(s().tNow).toBe(123.456);
    s().setTNow(null);
    expect(s().tNow).toBeNull();
  });

  it("setPlaying(true) turns time mode on; setPlaying(false) leaves it on", () => {
    s().setPlaying(true);
    expect(s().playing).toBe(true);
    expect(s().timeMode).toBe(true);
    s().setPlaying(false);
    expect(s().playing).toBe(false);
    expect(s().timeMode).toBe(true);
  });

  it("select and Esc (select(null))", () => {
    s().select("hq-run-000001");
    expect(s().selectedEventId).toBe("hq-run-000001");
    s().select(null);
    expect(s().selectedEventId).toBeNull();
  });

  it("setView cycles the presets", () => {
    for (const v of ["side", "plan", "oblique"] as const) {
      s().setView(v);
      expect(s().view).toBe(v);
    }
  });
});

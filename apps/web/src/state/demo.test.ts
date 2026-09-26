import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  finishReveal,
  initialDemoState,
  onDemoReset,
  setRevealProgress,
  useDemo,
  type DemoState,
} from "./demo";

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

  it("has exactly the docs/02 fields and actions, nothing more", () => {
    expect(Object.keys(s()).sort()).toEqual(
      [
        "phase",
        "revealProgress",
        "filter",
        "timeMode",
        "tNow",
        "playing",
        "selectedEventId",
        "view",
        "reveal",
        "reset",
        "setFilter",
        "setTimeMode",
        "setTNow",
        "setPlaying",
        "select",
        "setView",
      ].sort(),
    );
    expect(Object.keys(initialDemoState).sort()).toEqual(
      ["phase", "revealProgress", "filter", "timeMode", "tNow", "playing", "selectedEventId", "view"].sort(),
    );
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

  it("keeps STRICT if it was chosen before the reveal", () => {
    s().setFilter("strict");
    s().reveal();
    expect(s().filter).toBe("strict");
  });

  it("is a no-op while revealing or after revealed (Space can't restart it)", () => {
    s().reveal();
    setRevealProgress(0.4);
    s().reveal();
    expect(s().phase).toBe("revealing");
    expect(s().revealProgress).toBe(0.4);

    finishReveal();
    s().setFilter("strict");
    s().reveal();
    expect(s().phase).toBe("revealed");
    expect(s().filter).toBe("strict");
  });

  it("reports the side view the reveal dollies into", () => {
    s().reveal();
    expect(s().view).toBe("side");
  });

  it("keeps plan view (the reveal also plays from plan view)", () => {
    s().setView("plan");
    s().reveal();
    expect(s().view).toBe("plan");
  });
});

describe("setRevealProgress() and finishReveal() (scene-only helpers)", () => {
  it("advances the counter clock during revealing without ending the reveal", () => {
    s().reveal();
    setRevealProgress(0.25);
    expect(s().revealProgress).toBe(0.25);
    setRevealProgress(1.0000001);
    expect(s().revealProgress).toBe(1);
    expect(s().phase).toBe("revealing"); // the settle still runs
  });

  it("finishReveal pins progress to exactly 1 and marks revealed", () => {
    s().reveal();
    setRevealProgress(0.97);
    finishReveal();
    expect(s().revealProgress).toBe(1);
    expect(s().phase).toBe("revealed");
  });

  it("clamps negatives to 0", () => {
    s().reveal();
    setRevealProgress(-2);
    expect(s().revealProgress).toBe(0);
  });

  it("are ignored outside revealing, so a late frame after reset() can't resurrect the reveal", () => {
    setRevealProgress(0.5);
    finishReveal();
    expect(data(s())).toEqual(initialDemoState);

    s().reveal();
    finishReveal();
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
    finishReveal();
    s().select("evt-2");
    s().reset();
    expect(s().phase).toBe("public");
    expect(s().revealProgress).toBe(0);
    expect(s().filter).toBe("public");
    expect(s().selectedEventId).toBeNull();
  });

  it("allows a fresh reveal afterwards", () => {
    s().reveal();
    finishReveal();
    s().reset();
    s().reveal();
    expect(s().phase).toBe("revealing");
    expect(s().revealProgress).toBe(0);
  });
});

describe("onDemoReset() (scene-only)", () => {
  it("fires on every reset(), including one that changes no field", () => {
    const listener = vi.fn();
    const off = onDemoReset(listener);
    s().reset(); // already at the start frame
    expect(listener).toHaveBeenCalledTimes(1);
    s().reveal();
    s().reset();
    expect(listener).toHaveBeenCalledTimes(2);
    off();
    s().reset();
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it("fires after the state is back at the start frame", () => {
    let seen: unknown = null;
    const off = onDemoReset(() => {
      seen = data(s());
    });
    s().reveal();
    s().reset();
    off();
    expect(seen).toEqual(initialDemoState);
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

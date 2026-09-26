import { beforeEach, describe, expect, it, vi } from "vitest";
import { finishReveal, initialDemoState, useDemo } from "../../state/demo";
import { TOUR_STEPS, TOUR_TIMING, TourRun, type TourContext } from "./script";

// The tour drives the real demo store (docs/02 §6) with a stepped clock; the scene's part (the reveal's
// settle, the replay reaching the window end) is played by hand.

const T = TOUR_TIMING;

function context(over: Partial<TourContext> = {}) {
  const ctx = {
    openHidden: vi.fn(() => {
      useDemo.getState().select("hidden-1");
      return true;
    }),
    hasValidationCard: vi.fn(() => true),
    ...over,
  };
  return ctx;
}

function makeRun(ctx: TourContext, warn = vi.fn()) {
  return { run: new TourRun(TOUR_STEPS, useDemo.getState, ctx, warn), warn };
}

beforeEach(() => {
  useDemo.setState({ ...initialDemoState, phase: "revealed", filter: "strict", timeMode: true, tNow: 5, selectedEventId: "x", view: "plan" });
});

describe("TourRun: the judge sequence", () => {
  it("plays public → reveal → strict → hidden → validation → replay → full, then ends", () => {
    const ctx = context();
    const { run, warn } = makeRun(ctx);
    let now = 1000;

    // Start frame: reset() from wherever the presenter left the scene.
    expect(run.start(now)?.id).toBe("public");
    expect(useDemo.getState()).toMatchObject({ phase: "public", filter: "public", timeMode: false, selectedEventId: null, view: "oblique" });
    expect(run.tick((now += T.publicHoldMs - 1))?.id).toBe("public");

    // Reveal: the hold starts only once the scene settles.
    expect(run.tick((now += 1))?.id).toBe("reveal");
    expect(useDemo.getState().phase).toBe("revealing");
    expect(run.tick((now += 7000))?.id).toBe("reveal");
    finishReveal();
    expect(run.tick((now += 16))?.id).toBe("reveal");
    expect(run.tick((now += T.revealHoldMs - 1))?.id).toBe("reveal");

    expect(run.tick((now += 1))?.id).toBe("strict");
    expect(useDemo.getState().filter).toBe("strict");

    expect(run.tick((now += T.strictHoldMs))?.id).toBe("hidden");
    expect(ctx.openHidden).toHaveBeenCalledTimes(1);
    expect(useDemo.getState().selectedEventId).toBe("hidden-1");

    // Leaving the evidence closes the drawer before the card is outlined.
    expect(run.tick((now += T.hiddenHoldMs))?.id).toBe("validation");
    expect(useDemo.getState().selectedEventId).toBeNull();

    expect(run.tick((now += T.validationHoldMs))?.id).toBe("replay");
    expect(useDemo.getState().timeMode).toBe(true);
    // Not ready until the scene has set the playhead and the replay has stopped at the window end.
    expect(run.tick((now += 20_000))?.id).toBe("replay");
    useDemo.getState().setTNow(10);
    useDemo.getState().setPlaying(true);
    expect(run.tick((now += 16))?.id).toBe("replay");
    useDemo.getState().setPlaying(false);
    expect(run.tick((now += 16))?.id).toBe("replay");
    expect(run.tick((now += T.replayHoldMs))?.id).toBe("full");
    expect(useDemo.getState()).toMatchObject({ filter: "all", timeMode: false, tNow: null, playing: false, selectedEventId: null, phase: "revealed" });

    expect(run.tick((now += T.fullHoldMs - 1))?.id).toBe("full");
    expect(run.tick((now += 1))).toBeNull();
    expect(run.finished).toBe(true);
    expect(run.tick((now += 1000))).toBeNull();
    expect(warn).not.toHaveBeenCalled();
  });

  it("skips the evidence step when the bundle has no hidden event, and the card step without a card", () => {
    const ctx = context({ openHidden: vi.fn(() => false), hasValidationCard: vi.fn(() => false) });
    const { run } = makeRun(ctx);
    let now = 0;
    run.start(now);
    run.tick((now += T.publicHoldMs));
    finishReveal();
    run.tick((now += 1));
    run.tick((now += T.revealHoldMs));
    expect(run.step?.id).toBe("strict");
    expect(run.tick((now += T.strictHoldMs))?.id).toBe("replay");
    expect(ctx.openHidden).toHaveBeenCalledTimes(1);
    expect(ctx.hasValidationCard).toHaveBeenCalledTimes(1);
  });

  it("moves on (with a warning) when the scene never gets there", () => {
    const { run, warn } = makeRun(context());
    let now = 0;
    run.start(now);
    run.tick((now += T.publicHoldMs));
    expect(run.step?.id).toBe("reveal");
    // The reveal never settles (e.g. the canvas stopped rendering).
    run.tick((now += T.revealMaxWaitMs));
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('"reveal"'));
    expect(run.tick((now += T.revealHoldMs))?.id).toBe("strict");
  });

  it("restarts from the first step on start()", () => {
    const { run } = makeRun(context());
    run.start(0);
    run.tick(T.publicHoldMs);
    expect(run.step?.id).toBe("reveal");
    expect(run.start(99_000)?.id).toBe("public");
    expect(useDemo.getState().phase).toBe("public");
  });

  it("reaches the reveal from plan view the way Space does (top-down), because reset returns to oblique first", () => {
    const { run } = makeRun(context());
    run.start(0);
    run.tick(T.publicHoldMs);
    expect(useDemo.getState().view).toBe("side");
  });
});

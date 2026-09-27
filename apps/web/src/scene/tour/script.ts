// The Tour (WEB-09): the judge sequence played by itself, for a presenter who wants hands off and for
// recording the video. public view → reveal → STRICT → a hidden strict event's evidence → the
// Validation card → time replay → back to the full view. Each step drives the demo store exactly as the
// presenter's keys would (docs/02 §6), then holds; a step that waits for the scene (the reveal's settle,
// the replay reaching the window end) holds only once it has happened. Pure: the clock is passed in, so
// the tests step it; Tour.tsx ticks it once per animation frame.

import type { DemoState } from "../../state/demo";
import type { TourCopyKey } from "./copy";

export type TourStepId = "public" | "reveal" | "strict" | "hidden" | "validation" | "replay" | "full";

/** What a step needs from outside the store. */
export interface TourContext {
  /** Open the evidence drawer on the hidden hero; false when the bundle has none (the step is skipped). */
  openHidden(): boolean;
  /** Whether the Validation card is on screen (the step is skipped without it). */
  hasValidationCard(): boolean;
}

export interface TourStep {
  id: TourStepId;
  /** The copy file's key for this step's caption. */
  caption: TourCopyKey;
  /** Drive the store into this step. Returns false to skip the step. */
  enter(store: DemoState, ctx: TourContext): boolean;
  /** When set, the hold starts only once this is true (or after `maxWaitMs`). */
  ready?(store: DemoState): boolean;
  maxWaitMs?: number;
  /** How long the step stays on screen once ready. */
  holdMs: number;
  /** Undo what the next step shouldn't inherit (e.g. close the drawer). */
  leave?(store: DemoState): void;
}

/**
 * The tour's pacing, in ms. The reveal and the replay take as long as the scene takes (≈7 s and the
 * window at LOOK.time.playbackRate); the holds are reading time for each caption.
 */
export const TOUR_TIMING = Object.freeze({
  publicHoldMs: 4500,
  revealMaxWaitMs: 15_000,
  revealHoldMs: 3000,
  strictHoldMs: 6500,
  hiddenHoldMs: 8000,
  validationHoldMs: 6000,
  replayMaxWaitMs: 60_000,
  replayHoldMs: 1500,
  fullHoldMs: 6500,
});

export const TOUR_STEPS: readonly TourStep[] = Object.freeze([
  {
    id: "public",
    caption: "public",
    // The start frame, pixel-identical to a fresh load (reset() also closes the drawer and time mode).
    enter(s) {
      s.reset();
      return true;
    },
    holdMs: TOUR_TIMING.publicHoldMs,
  },
  {
    id: "reveal",
    caption: "reveal",
    enter(s) {
      s.reveal();
      return true;
    },
    ready: (s) => s.phase === "revealed",
    maxWaitMs: TOUR_TIMING.revealMaxWaitMs,
    holdMs: TOUR_TIMING.revealHoldMs,
  },
  {
    id: "strict",
    caption: "strict",
    enter(s) {
      s.setFilter("strict");
      return true;
    },
    holdMs: TOUR_TIMING.strictHoldMs,
  },
  {
    id: "hidden",
    caption: "hidden",
    enter: (_s, ctx) => ctx.openHidden(),
    holdMs: TOUR_TIMING.hiddenHoldMs,
    leave(s) {
      s.select(null);
    },
  },
  {
    id: "validation",
    caption: "validation",
    enter: (_s, ctx) => ctx.hasValidationCard(),
    holdMs: TOUR_TIMING.validationHoldMs,
  },
  {
    id: "replay",
    caption: "replay",
    // Time mode replays the window from its start (the scene's TimeDriver); it stops at the window end.
    enter(s) {
      s.setTimeMode(true);
      return true;
    },
    ready: (s) => s.timeMode && s.tNow !== null && !s.playing,
    maxWaitMs: TOUR_TIMING.replayMaxWaitMs,
    holdMs: TOUR_TIMING.replayHoldMs,
  },
  {
    id: "full",
    caption: "full",
    enter(s) {
      s.select(null);
      s.setTimeMode(false);
      s.setFilter("all");
      return true;
    },
    holdMs: TOUR_TIMING.fullHoldMs,
  },
]);

/**
 * One play-through. `start(now)` enters the first step; `tick(now)` advances it and returns the active
 * step, or null once the last step's hold has ended. Stopping is just dropping the run: the scene stays
 * wherever the tour left it, for the presenter to take over.
 */
export class TourRun {
  private index = -1;
  private enteredAt = 0;
  private readyAt: number | null = null;

  constructor(
    private readonly steps: readonly TourStep[],
    private readonly store: () => DemoState,
    private readonly ctx: TourContext,
    private readonly warn: (message: string) => void = (m) => console.warn(m),
  ) {}

  /** The active step, or null before start() and after the end. */
  get step(): TourStep | null {
    return this.index >= 0 && this.index < this.steps.length ? this.steps[this.index]! : null;
  }

  get finished(): boolean {
    return this.index >= this.steps.length;
  }

  start(now: number): TourStep | null {
    this.index = -1;
    this.next(now);
    return this.step;
  }

  tick(now: number): TourStep | null {
    const step = this.step;
    if (step === null) return null;
    if (this.readyAt === null) {
      if (step.ready?.(this.store()) ?? true) {
        this.readyAt = now;
      } else if (now - this.enteredAt >= (step.maxWaitMs ?? 0)) {
        this.warn(`Tour: step "${step.id}" still not ready after ${step.maxWaitMs} ms; moving on`);
        this.readyAt = now;
      }
    }
    if (this.readyAt !== null && now - this.readyAt >= step.holdMs) this.next(now);
    return this.step;
  }

  private next(now: number): void {
    this.step?.leave?.(this.store());
    this.index++;
    while (this.index < this.steps.length && !this.steps[this.index]!.enter(this.store(), this.ctx)) this.index++;
    this.enteredAt = now;
    const step = this.step;
    this.readyAt = step && !step.ready ? now : null;
  }
}

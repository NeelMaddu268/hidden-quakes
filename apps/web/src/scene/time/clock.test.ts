import { describe, expect, it } from "vitest";
import {
  advancePlayback,
  binCount,
  countUpTo,
  fmtClockUtc,
  histogram,
  hourTicks,
  recency,
  shouldStartReplay,
  shownAt,
  sortedTimes,
  stepTime,
  type TimeStateLike,
  type TimeStep,
  TIME_ALL,
  timeActive,
  timeNowRel,
} from "./clock";

// A test-local day window (UTC midnight to midnight).
const START = Date.UTC(2026, 8, 10) / 1000;
const END = START + 86_400;
const RATE = 3600;

describe("time mode takes effect only after the reveal, with a playhead", () => {
  it("timeActive / timeNowRel", () => {
    expect(timeActive({ phase: "revealed", timeMode: true, tNow: START + 60 })).toBe(true);
    expect(timeNowRel({ phase: "revealed", timeMode: true, tNow: START + 60 }, START)).toBe(60);
    // Off, before the reveal, mid-reveal, or no playhead: every event shows (TIME_ALL).
    for (const s of [
      { phase: "revealed" as const, timeMode: false, tNow: START + 60 },
      { phase: "public" as const, timeMode: true, tNow: START + 60 },
      { phase: "revealing" as const, timeMode: true, tNow: START + 60 },
      { phase: "revealed" as const, timeMode: true, tNow: null },
    ]) {
      expect(timeActive(s)).toBe(false);
      expect(timeNowRel(s, START)).toBe(TIME_ALL);
    }
  });

  it("TIME_ALL shows every event of any window and is exact in float32", () => {
    expect(shownAt(86_400 * 30, TIME_ALL)).toBe(true);
    expect(Math.fround(TIME_ALL)).toBe(TIME_ALL);
  });
});

describe("shownAt / recency (mirrors the event shader)", () => {
  it("shows an event once now reaches its origin time", () => {
    expect(shownAt(100, 99.9)).toBe(false);
    expect(shownAt(100, 100)).toBe(true);
    expect(shownAt(100, 5000)).toBe(true);
  });

  it("glows fully at its time, decays linearly over the glow window, and never before it's shown", () => {
    expect(recency(100, 100, 1800)).toBe(1);
    expect(recency(100, 1000, 1800)).toBeCloseTo(0.5, 10);
    expect(recency(100, 1900, 1800)).toBe(0);
    expect(recency(100, 5000, 1800)).toBe(0);
    expect(recency(100, 99, 1800)).toBe(0);
    expect(recency(100, TIME_ALL, 1800)).toBe(0); // time mode off lights nothing
    expect(recency(100, 100, 0)).toBe(0);
  });
});

describe("playback", () => {
  it("advances at the rate, clamps to the window end, and caps a long frame", () => {
    expect(advancePlayback(START, 0.05, RATE, END)).toEqual({ tNow: START + 180, done: false });
    expect(advancePlayback(END - 10, 0.1, RATE, END)).toEqual({ tNow: END, done: true });
    // A 5 s stall (background tab) moves at most maxDeltaS of playback, never a jump across the window.
    expect(advancePlayback(START, 5, RATE, END, 0.1).tNow).toBe(START + 360);
    expect(advancePlayback(START, -1, RATE, END).tNow).toBe(START);
  });

  it("a day replays in 24 s at 1 h/s", () => {
    let t = START;
    let frames = 0;
    for (;;) {
      const s = advancePlayback(t, 1 / 60, RATE, END);
      t = s.tNow;
      frames++;
      if (s.done) break;
    }
    expect(t).toBe(END);
    expect(frames / 60).toBeCloseTo(24, 1);
  });
});

/** stepTime writes into a caller-owned step; the tests compare a copy (or null for "nothing to do"). */
function step(...args: [TimeStateLike & { playing: boolean }, number, number, number, number]): TimeStep | null {
  const out: TimeStep = { tNow: NaN, playing: false };
  return stepTime(...args, out) ? { ...out } : null;
}

describe("stepTime (the TimeDriver's per-frame rule)", () => {
  const revealed = { phase: "revealed" as const, timeMode: true, tNow: null, playing: false };

  it("entering time mode after the reveal replays the window from its start", () => {
    expect(shouldStartReplay(revealed)).toBe(true);
    expect(step(revealed, 1 / 60, START, END, RATE)).toEqual({ tNow: START, playing: true });
  });

  it("T before or during the reveal waits: nothing happens until the reveal has finished", () => {
    for (const phase of ["public", "revealing"] as const) {
      expect(step({ ...revealed, phase }, 1 / 60, START, END, RATE)).toBeNull();
      expect(step({ ...revealed, phase, playing: true }, 1 / 60, START, END, RATE)).toBeNull();
    }
  });

  it("advances while playing, stops at the end, and leaves a paused or finished replay alone", () => {
    const playing = { ...revealed, tNow: START + 100, playing: true };
    expect(step(playing, 0.1, START, END, RATE)).toEqual({ tNow: START + 460, playing: true });
    expect(step({ ...playing, tNow: END - 1 }, 0.1, START, END, RATE)).toEqual({ tNow: END, playing: false });
    expect(step({ ...playing, playing: false }, 0.1, START, END, RATE)).toBeNull();
    expect(step({ ...revealed, tNow: END }, 0.1, START, END, RATE)).toBeNull();
  });

  it("does nothing with time mode off", () => {
    expect(step({ ...revealed, timeMode: false }, 0.1, START, END, RATE)).toBeNull();
  });
});

describe("histogram", () => {
  it("covers the window in whole bins", () => {
    expect(binCount(START, END, 600)).toBe(144);
    expect(binCount(START, START + 601, 600)).toBe(2);
    expect(binCount(START, START, 600)).toBe(1);
    expect(binCount(START, END, 0)).toBe(1);
  });

  it("puts each event in exactly one bin (edges, window end and strays included)", () => {
    const times = [START, START + 599.9, START + 600, END - 1, END, END + 50, START - 5];
    const h = histogram(times, START, END, 600);
    expect(h.length).toBe(144);
    expect(h[0]).toBe(3); // START, START+599.9, and the stray before the start
    expect(h[1]).toBe(1); // START+600 opens bin 1
    expect(h[143]).toBe(3); // END−1, END and the stray after the end
    expect(h.reduce((a, b) => a + b, 0)).toBe(times.length);
  });
});

describe("countUpTo", () => {
  it("counts events with t <= now (what's on screen)", () => {
    const sorted = sortedTimes([30, 10, 20, 20, 40]);
    expect(Array.from(sorted)).toEqual([10, 20, 20, 30, 40]);
    expect(countUpTo(sorted, 5)).toBe(0);
    expect(countUpTo(sorted, 10)).toBe(1);
    expect(countUpTo(sorted, 20)).toBe(3);
    expect(countUpTo(sorted, 39.9)).toBe(4);
    expect(countUpTo(sorted, 1e12)).toBe(5);
    expect(countUpTo(new Float64Array(0), 5)).toBe(0);
  });
});

describe("clock labels", () => {
  it("formats UTC HH:MM; the end of a day window reads 24:00", () => {
    expect(fmtClockUtc(START)).toBe("00:00");
    expect(fmtClockUtc(START + 12 * 3600 + 34 * 60 + 59)).toBe("12:34");
    expect(fmtClockUtc(END, START)).toBe("24:00");
    expect(fmtClockUtc(START, START)).toBe("00:00");
  });

  it("puts ticks on whole hours at a readable step", () => {
    expect(hourTicks(START, END).map((t) => (t - START) / 3600)).toEqual([0, 6, 12, 18, 24]);
    expect(hourTicks(START + 1800, START + 1800 + 4 * 3600).map((t) => (t - START) / 3600)).toEqual([1, 2, 3, 4]);
    expect(hourTicks(START, START)).toEqual([START]);
  });
});

describe("stepTime allocation", () => {
  it("writes into the caller's step and reuses it (no per-frame object)", () => {
    const out: TimeStep = { tNow: 0, playing: false };
    const s = { phase: "revealed" as const, timeMode: true, tNow: START + 100, playing: true };
    expect(stepTime(s, 0.05, START, END, RATE, out)).toBe(true);
    expect(out).toEqual({ tNow: START + 280, playing: true });
    expect(stepTime({ ...s, playing: false }, 0.05, START, END, RATE, out)).toBe(false);
    expect(out.tNow).toBe(START + 280); // untouched when there's nothing to do
  });
});

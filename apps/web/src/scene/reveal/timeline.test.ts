import { describe, expect, it } from "vitest";
import {
  appearTimeOf,
  bloomBoostAt,
  dollyAt,
  easeInOutCubic,
  isRevealDone,
  revealProgressAt,
  terrainOpacityAt,
  TIMELINE,
} from "./timeline";

describe("TIMELINE matches the lane doc table", () => {
  it("uses the documented windows", () => {
    expect(TIMELINE.terrainFade).toEqual({ startS: 0, endS: 1.2, to: 0.12 });
    expect(TIMELINE.dolly).toEqual({ startS: 0.4, endS: 2.0 });
    expect(TIMELINE.events.startS).toBe(1.0);
    expect(TIMELINE.events.endS).toBe(6.0);
    expect(TIMELINE.endS).toBe(7.0);
  });

  it("is frozen all the way down (nothing can retime the reveal at runtime)", () => {
    expect(Object.isFrozen(TIMELINE)).toBe(true);
    for (const v of Object.values(TIMELINE)) if (typeof v === "object") expect(Object.isFrozen(v)).toBe(true);
  });
});

describe("terrainOpacityAt", () => {
  it("fades 1 → 0.12 over 0.0–1.2 s and holds", () => {
    expect(terrainOpacityAt(0)).toBe(1);
    expect(terrainOpacityAt(-5)).toBe(1);
    expect(terrainOpacityAt(1.2)).toBeCloseTo(0.12, 12);
    expect(terrainOpacityAt(99)).toBeCloseTo(0.12, 12);
    expect(terrainOpacityAt(0.6)).toBeLessThan(1);
    expect(terrainOpacityAt(0.6)).toBeGreaterThan(0.12);
  });

  it("starts from the current opacity when a reveal begins mid-fade (no jump)", () => {
    expect(terrainOpacityAt(0, 0.5)).toBe(0.5);
    expect(terrainOpacityAt(1.2, 0.5)).toBeCloseTo(0.12, 12);
  });
});

describe("dollyAt", () => {
  it("starts at 0.4 s, ends at 2.0 s, eases both ends", () => {
    expect(dollyAt(0.4)).toBe(0);
    expect(dollyAt(0.2)).toBe(0);
    expect(dollyAt(2.0)).toBe(1);
    expect(dollyAt(10)).toBe(1);
    expect(dollyAt(1.2)).toBeCloseTo(0.5, 12);
  });

  it("easeInOutCubic is symmetric and monotonic", () => {
    let prev = -1;
    for (let x = 0; x <= 1.0001; x += 0.01) {
      const v = easeInOutCubic(x);
      expect(v).toBeGreaterThanOrEqual(prev);
      expect(v + easeInOutCubic(1 - x)).toBeCloseTo(1, 10);
      prev = v;
    }
  });
});

describe("revealProgressAt (the shell's counter clock)", () => {
  it("is exactly 0 until events start and exactly 1 from 6.0 s", () => {
    expect(revealProgressAt(0)).toBe(0);
    expect(revealProgressAt(1.0)).toBe(0);
    expect(revealProgressAt(6.0)).toBe(1);
    expect(revealProgressAt(6.5)).toBe(1);
    expect(revealProgressAt(1e9)).toBe(1);
  });

  it("is monotonic non-decreasing and starts slow (structure forms first)", () => {
    let prev = 0;
    for (let t = 0; t <= 7; t += 1 / 60) {
      const p = revealProgressAt(t);
      expect(p).toBeGreaterThanOrEqual(prev);
      prev = p;
    }
    expect(revealProgressAt(3.5)).toBeLessThan(0.5); // halfway in time, under half the events
  });

  it("is identical for identical t (deterministic)", () => {
    for (const t of [0.5, 1.7, 3.3333, 5.99]) expect(revealProgressAt(t)).toBe(revealProgressAt(t));
  });
});

describe("appearTimeOf", () => {
  it("inverts revealProgressAt: an event with slot s appears when progress reaches s", () => {
    for (const s of [0, 0.001, 0.25, 0.5, 0.9, 1]) {
      const t = appearTimeOf(s);
      expect(t).toBeGreaterThanOrEqual(TIMELINE.events.startS);
      expect(t).toBeLessThanOrEqual(TIMELINE.events.endS);
      expect(revealProgressAt(t)).toBeCloseTo(s, 9);
    }
  });

  it("puts the first event at 1.0 s and the last at 6.0 s", () => {
    expect(appearTimeOf(0)).toBe(1);
    expect(appearTimeOf(1)).toBe(6);
  });

  it("counter and visible instances agree: #appeared(t) / n tracks revealProgressAt(t)", () => {
    const n = 500;
    const appear = Array.from({ length: n }, (_, rank) => appearTimeOf(rank / (n - 1)));
    for (const t of [1.5, 2.5, 4, 5.5]) {
      const shown = appear.filter((a) => a <= t).length;
      expect(Math.abs(shown / n - revealProgressAt(t))).toBeLessThanOrEqual(1 / (n - 1) + 1e-9);
    }
  });
});

describe("bloomBoostAt / isRevealDone", () => {
  it("is 1 outside the events window and back to 1 by the end", () => {
    expect(bloomBoostAt(0)).toBe(1);
    expect(bloomBoostAt(7)).toBe(1);
    expect(bloomBoostAt(3)).toBeGreaterThan(1);
  });

  it("marks the reveal done at exactly 7.0 s", () => {
    expect(isRevealDone(6.999)).toBe(false);
    expect(isRevealDone(7)).toBe(true);
  });
});

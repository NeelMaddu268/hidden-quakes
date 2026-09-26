import { Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { createTween, startTween, stepTween } from "./tween";

const to = { position: [10, 20, 30] as [number, number, number], target: [1, 2, 3] as [number, number, number] };

describe("pose tween", () => {
  it("ends exactly on the destination pose, whatever the frame timing", () => {
    for (const dt of [1 / 30, 1 / 60, 1 / 144, 0.37]) {
      const tw = createTween();
      startTween(tw, [0, 0, 0], [0, 0, 0], to, 1.2);
      const p = new Vector3();
      const t = new Vector3();
      let finished = false;
      for (let i = 0; i < 10_000 && !finished; i++) finished = stepTween(tw, dt, p, t);
      expect(finished).toBe(true);
      expect(p.toArray()).toEqual(to.position);
      expect(t.toArray()).toEqual(to.target);
      expect(tw.active).toBe(false);
    }
  });

  it("is time-based: the same elapsed time gives the same pose at any frame rate", () => {
    const a = createTween();
    const b = createTween();
    startTween(a, [0, 0, 0], [0, 0, 0], to, 1.2);
    startTween(b, [0, 0, 0], [0, 0, 0], to, 1.2);
    const pa = new Vector3();
    const pb = new Vector3();
    const ta = new Vector3();
    const tb = new Vector3();
    for (let i = 0; i < 30; i++) stepTween(a, 1 / 60, pa, ta);
    for (let i = 0; i < 15; i++) stepTween(b, 1 / 30, pb, tb);
    expect(pa.distanceTo(pb)).toBeLessThan(1e-9);
  });

  it("eases out: more than half way at the half-time mark", () => {
    const tw = createTween();
    startTween(tw, [0, 0, 0], [0, 0, 0], to, 1);
    const p = new Vector3();
    stepTween(tw, 0.5, p, new Vector3());
    expect(p.x).toBeGreaterThan(5);
    expect(p.x).toBeLessThan(10);
  });

  it("does nothing when inactive, and a zero duration snaps immediately", () => {
    const tw = createTween();
    const p = new Vector3(7, 7, 7);
    expect(stepTween(tw, 1, p, new Vector3())).toBe(false);
    expect(p.toArray()).toEqual([7, 7, 7]);
    startTween(tw, [0, 0, 0], [0, 0, 0], to, 0);
    expect(stepTween(tw, 0, p, new Vector3())).toBe(true);
    expect(p.toArray()).toEqual(to.position);
  });
});

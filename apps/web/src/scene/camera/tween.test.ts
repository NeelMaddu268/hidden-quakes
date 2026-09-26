import { Vector3 } from "three";
import { describe, expect, it } from "vitest";
import type { CameraPose } from "./presets";
import { createTween, poseAt, setupTween, startTween, stepTween } from "./tween";

const from: CameraPose = { position: [10, 8, 20], target: [0, -3, 0] };
const to: CameraPose = { position: [0, -2.5, 25], target: [0.5, -3.5, -1] };

function run(dt: number, duration = 1.2) {
  const tw = createTween();
  startTween(tw, from.position, from.target, to, duration);
  const p = new Vector3();
  const t = new Vector3();
  let finished = false;
  let frames = 0;
  while (!finished && frames < 10_000) {
    finished = stepTween(tw, dt, p, t);
    frames++;
  }
  return { tw, p, t, finished };
}

describe("self-timed pose tween", () => {
  it("ends exactly on the destination pose, whatever the frame timing", () => {
    for (const dt of [1 / 30, 1 / 60, 1 / 144, 0.37]) {
      const { tw, p, t, finished } = run(dt);
      expect(finished).toBe(true);
      expect(p.toArray()).toEqual(to.position);
      expect(t.toArray()).toEqual(to.target);
      expect(tw.active).toBe(false);
    }
  });

  it("is time-based: the same elapsed time gives the same pose at any frame rate", () => {
    const a = createTween();
    const b = createTween();
    startTween(a, from.position, from.target, to, 1.2);
    startTween(b, from.position, from.target, to, 1.2);
    const pa = new Vector3();
    const pb = new Vector3();
    for (let i = 0; i < 30; i++) stepTween(a, 1 / 60, pa, new Vector3());
    for (let i = 0; i < 15; i++) stepTween(b, 1 / 30, pb, new Vector3());
    expect(pa.distanceTo(pb)).toBeLessThan(1e-9);
  });

  it("does nothing when inactive, and a zero duration snaps immediately", () => {
    const tw = createTween();
    const p = new Vector3(7, 7, 7);
    expect(stepTween(tw, 1, p, new Vector3())).toBe(false);
    expect(p.toArray()).toEqual([7, 7, 7]);
    startTween(tw, from.position, from.target, to, 0);
    expect(stepTween(tw, 0, p, new Vector3())).toBe(true);
    expect(p.toArray()).toEqual(to.position);
  });
});

describe("poseAt (spherical interpolation)", () => {
  it("starts exactly at `from` and ends exactly at `to`", () => {
    const tw = createTween();
    setupTween(tw, from.position, from.target, to);
    const p = new Vector3();
    const t = new Vector3();
    poseAt(tw, 0, p, t);
    expect(p.distanceTo(new Vector3(...from.position))).toBeLessThan(1e-9);
    expect(t.toArray()).toEqual(from.target);
    poseAt(tw, 1, p, t);
    expect(p.toArray()).toEqual(to.position);
  });

  it("keeps the camera at an interpolated distance from the target (swings, never cuts through)", () => {
    const tw = createTween();
    setupTween(tw, from.position, from.target, to);
    const r0 = new Vector3(...from.position).distanceTo(new Vector3(...from.target));
    const r1 = new Vector3(...to.position).distanceTo(new Vector3(...to.target));
    const p = new Vector3();
    const t = new Vector3();
    for (let k = 0; k <= 1; k += 0.1) {
      poseAt(tw, k, p, t);
      expect(p.distanceTo(t)).toBeCloseTo(r0 + (r1 - r0) * k, 9);
    }
  });

  it("takes the short way around in azimuth", () => {
    const tw = createTween();
    // 170° east of south → 170° west of south: the short way passes due north, not south.
    const a = (170 * Math.PI) / 180;
    setupTween(tw, [Math.sin(a) * 10, 0, Math.cos(a) * 10], [0, 0, 0], {
      position: [-Math.sin(a) * 10, 0, Math.cos(a) * 10],
      target: [0, 0, 0],
    });
    const p = new Vector3();
    poseAt(tw, 0.5, p, new Vector3());
    expect(p.z).toBeLessThan(-9.9);
  });
});

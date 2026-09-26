import { Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { dollyAt, TIMELINE } from "../reveal/timeline";
import { computeBounds } from "./bounds";
import { createCameraDirector } from "./director";
import { presetPose, type CameraPose } from "./presets";

// Test-local framing: a 4 km cloud 3.5 km below a surface at y = 0.05.
const bounds = computeBounds([new Float32Array([-2, -2, -3, 2, -5, 1])], 0.05, 0);
const aspect = 16 / 9;
const P = {
  oblique: presetPose("oblique", bounds, aspect),
  side: presetPose("side", bounds, aspect),
  plan: presetPose("plan", bounds, aspect),
};
const MOVE_S = 1.2;
const DT = 1 / 60;

function camera(pose: CameraPose) {
  return { position: new Vector3(...pose.position), target: new Vector3(...pose.target) };
}

function expectAt(cam: { position: Vector3; target: Vector3 }, pose: CameraPose) {
  expect(cam.position.distanceTo(new Vector3(...pose.position))).toBeLessThan(1e-9);
  expect(cam.target.distanceTo(new Vector3(...pose.target))).toBeLessThan(1e-9);
}

/**
 * Runs frames with a reveal clock that starts at `clockStart` (null = no reveal running) and returns
 * the largest single-frame camera displacement, so tests can assert "no jump".
 */
function run(
  d: ReturnType<typeof createCameraDirector>,
  cam: { position: Vector3; target: Vector3 },
  seconds: number,
  clock: { t: number | null },
  dollyTo: () => CameraPose = () => P.side,
): number {
  let maxStep = 0;
  const prev = new Vector3();
  for (let i = 0; i < Math.round(seconds / DT); i++) {
    if (clock.t !== null) clock.t += DT;
    prev.copy(cam.position);
    d.step(DT, clock.t ?? 0, cam.position, cam.target, dollyTo);
    maxStep = Math.max(maxStep, prev.distanceTo(cam.position));
  }
  return maxStep;
}

/** The largest single-frame displacement an uninterrupted oblique→side dolly makes (the smoothness bar). */
function referenceDollyStep(): number {
  const d = createCameraDirector();
  const cam = camera(P.oblique);
  d.onReveal(false);
  return run(d, cam, TIMELINE.endS, { t: 0 });
}

describe("camera director", () => {
  it("reveal from rest: holds until 0.4 s, lands exactly on the side view at 2.0 s, then frees the orbit", () => {
    const d = createCameraDirector();
    const cam = camera(P.oblique);
    const clock = { t: 0 as number | null };
    d.onReveal(false);
    expect(d.busy).toBe(true);
    run(d, cam, TIMELINE.dolly.startS - DT, clock);
    expectAt(cam, P.oblique);
    run(d, cam, TIMELINE.dolly.endS - (clock.t as number) + DT, clock);
    expectAt(cam, P.side);
    expect(d.busy).toBe(false);
  });

  it("is identical on every run from the same start pose", () => {
    const trace = () => {
      const d = createCameraDirector();
      const cam = camera(P.oblique);
      d.onReveal(false);
      const out: number[] = [];
      const clock = { t: 0 };
      for (let i = 0; i < 150; i++) {
        clock.t += DT;
        d.step(DT, clock.t, cam.position, cam.target, () => P.side);
        out.push(...cam.position.toArray());
      }
      return out;
    };
    expect(trace()).toEqual(trace());
  });

  it("reveal from plan view while a move is running: the move finishes and the orbit comes back", () => {
    const d = createCameraDirector();
    const cam = camera(P.oblique);
    d.moveTo(cam.position, cam.target, P.plan, MOVE_S);
    run(d, cam, 0.3, { t: null });
    d.onReveal(true); // P then Space within 1.2 s
    run(d, cam, MOVE_S, { t: 0 });
    expectAt(cam, P.plan);
    expect(d.busy).toBe(false);
  });

  it("R then Space mid-return: the camera keeps moving (no freeze, no jump) and lands on the side view", () => {
    const d = createCameraDirector();
    const cam = camera(P.side);
    d.moveTo(cam.position, cam.target, P.oblique, MOVE_S); // reset() from the revealed side view
    run(d, cam, 0.3, { t: null });
    d.onReveal(false);
    const clock = { t: 0 as number | null };
    // during the dolly's 0.4 s hold, the reset move continues
    const before = cam.position.clone();
    run(d, cam, 0.2, clock);
    expect(cam.position.distanceTo(before)).toBeGreaterThan(0);
    const maxStep = run(d, cam, TIMELINE.endS, clock);
    expect(maxStep).toBeLessThan(referenceDollyStep() * 2);
    expectAt(cam, P.side);
    expect(d.busy).toBe(false);
  });

  it("reset(); reveal() in one tick: continuous, and it still ends on the side view", () => {
    const d = createCameraDirector();
    const cam = camera(P.side);
    d.moveTo(cam.position, cam.target, P.oblique, MOVE_S);
    d.onReveal(false);
    const maxStep = run(d, cam, TIMELINE.endS, { t: 0 });
    expect(maxStep).toBeLessThan(referenceDollyStep() * 2);
    expectAt(cam, P.side);
  });

  it("P mid-dolly cancels the dolly and lands on plan view with the orbit back", () => {
    const d = createCameraDirector();
    const cam = camera(P.oblique);
    const clock = { t: 0 as number | null };
    d.onReveal(false);
    run(d, cam, 1.0, clock);
    expect(d.mode).toBe("dolly");
    d.moveTo(cam.position, cam.target, P.plan, MOVE_S);
    run(d, cam, MOVE_S + DT, clock);
    expectAt(cam, P.plan);
    expect(d.busy).toBe(false);
  });

  it("reset mid-dolly returns exactly to the start pose", () => {
    const d = createCameraDirector();
    const cam = camera(P.oblique);
    const clock = { t: 0 as number | null };
    d.onReveal(false);
    run(d, cam, 1.2, clock);
    d.moveTo(cam.position, cam.target, P.oblique, MOVE_S);
    clock.t = null; // reset() stops the reveal clock
    run(d, cam, MOVE_S + DT, clock);
    expectAt(cam, P.oblique);
    expect(d.busy).toBe(false);
  });

  it("asks for the dolly destination at takeover, not at reveal() (a resize in between is honored)", () => {
    const d = createCameraDirector();
    const cam = camera(P.oblique);
    let asked = 0;
    const clock = { t: 0 as number | null };
    d.onReveal(false);
    run(d, cam, 0.3, clock, () => (asked++, P.side));
    expect(asked).toBe(0);
    run(d, cam, 0.2, clock, () => (asked++, P.side));
    expect(asked).toBe(1);
  });

  it("a reveal clock that jumps past the dolly window (a long hitch) lands straight on the destination", () => {
    const d = createCameraDirector();
    const cam = camera(P.oblique);
    d.onReveal(false);
    d.step(DT, 3.0, cam.position, cam.target, () => P.side);
    expect(dollyAt(3.0)).toBe(1);
    expectAt(cam, P.side);
    expect(d.busy).toBe(false);
  });

  it("cancel() releases the camera immediately", () => {
    const d = createCameraDirector();
    const cam = camera(P.oblique);
    d.moveTo(cam.position, cam.target, P.plan, MOVE_S);
    d.onReveal(false);
    d.cancel();
    expect(d.busy).toBe(false);
    expect(d.step(DT, 1, cam.position, cam.target, () => P.side)).toBe(false);
  });
});

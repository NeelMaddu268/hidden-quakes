// Who moves the camera, and when: a small state machine behind CameraRig, kept free of React and
// WebGL so every presenter sequence (P mid-dolly, R then Space within a second, reveal from plan view)
// is unit-tested.
//
// - A *move* is a self-timed tween (setView, reset): starts now, runs over its duration.
// - The reveal *dolly* follows the reveal clock (timeline 0.4–2.0 s). reveal() only arms it; the
//   dolly takes over from wherever the camera is when the clock reaches 0.4 s, so a move that was
//   still running (reset just before Space) carries on smoothly until then instead of freezing, and
//   the camera never jumps. Its destination is computed at takeover, with the current aspect.
// - From plan view the reveal plays top-down: no dolly, and any running move just finishes.
// - A new move (view change, reset) cancels an armed or running dolly.

import { dollyAt } from "../reveal/timeline";
import type { CameraPose } from "./presets";
import { createTween, poseAt, setupTween, startTween, stepTween } from "./tween";

interface Vec3Like {
  x: number;
  y: number;
  z: number;
  set(x: number, y: number, z: number): unknown;
}

export interface CameraDirector {
  /** True while the director owns the camera; orbit controls are disabled. */
  readonly busy: boolean;
  readonly mode: "idle" | "move" | "armed" | "dolly" | "move+armed";
  /** reveal() fired. `revealsTopDown` = the view is plan: no dolly. */
  onReveal(revealsTopDown: boolean): void;
  /** Start a self-timed move from the current pose to `to` (view change, reset). Cancels any dolly. */
  moveTo(position: Vec3Like, target: Vec3Like, to: CameraPose, durationS: number): void;
  /** Cancel everything (a snap put the camera exactly where it must be). */
  cancel(): void;
  /**
   * Advance one frame. `dollyTo` is asked for the dolly's destination only at takeover. Writes the
   * camera pose into `position` / `target` when the director owns the camera; returns whether it did.
   */
  step(deltaS: number, revealElapsedS: number, position: Vec3Like, target: Vec3Like, dollyTo: () => CameraPose): boolean;
}

export function createCameraDirector(): CameraDirector {
  const move = createTween();
  const dolly = createTween();
  let armed = false;
  const fromPos = [0, 0, 0];
  const fromTarget = [0, 0, 0];

  const capture = (position: Vec3Like, target: Vec3Like) => {
    fromPos[0] = position.x;
    fromPos[1] = position.y;
    fromPos[2] = position.z;
    fromTarget[0] = target.x;
    fromTarget[1] = target.y;
    fromTarget[2] = target.z;
  };

  return {
    get busy() {
      return move.active || dolly.active || armed;
    },
    get mode() {
      if (dolly.active) return "dolly";
      if (move.active && armed) return "move+armed";
      if (move.active) return "move";
      if (armed) return "armed";
      return "idle";
    },

    onReveal(revealsTopDown) {
      dolly.active = false;
      armed = !revealsTopDown;
    },

    moveTo(position, target, to, durationS) {
      armed = false;
      dolly.active = false;
      capture(position, target);
      startTween(move, fromPos, fromTarget, to, durationS);
    },

    cancel() {
      armed = false;
      dolly.active = false;
      move.active = false;
    },

    step(deltaS, revealElapsedS, position, target, dollyTo) {
      if (armed && dollyAt(revealElapsedS) > 0) {
        armed = false;
        move.active = false;
        capture(position, target);
        setupTween(dolly, fromPos, fromTarget, dollyTo());
      }
      if (dolly.active) {
        const k = dollyAt(revealElapsedS);
        poseAt(dolly, k, position, target);
        if (k >= 1) dolly.active = false;
        return true;
      }
      if (move.active) {
        stepTween(move, deltaS, position, target);
        return true;
      }
      return false;
    },
  };
}

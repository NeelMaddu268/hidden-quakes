// A time-based camera tween between two poses. Time-based (not frame-based) so a move takes the same
// wall time at 30 or 120 fps, and it ends by assigning the exact destination pose.

import { easeOutCubic } from "@hq/visualization";
import type { CameraPose, Vec3 } from "./presets";

export interface PoseTween {
  from: CameraPose;
  to: CameraPose;
  durationS: number;
  elapsedS: number;
  active: boolean;
}

export function createTween(): PoseTween {
  return {
    from: { position: [0, 0, 0], target: [0, 0, 0] },
    to: { position: [0, 0, 0], target: [0, 0, 0] },
    durationS: 0,
    elapsedS: 0,
    active: false,
  };
}

function copyVec(dst: Vec3, src: readonly number[]): void {
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
}

/** Starts (or restarts) the tween in place; no allocation. */
export function startTween(
  tween: PoseTween,
  fromPosition: readonly number[],
  fromTarget: readonly number[],
  to: CameraPose,
  durationS: number,
): void {
  copyVec(tween.from.position, fromPosition);
  copyVec(tween.from.target, fromTarget);
  copyVec(tween.to.position, to.position);
  copyVec(tween.to.target, to.target);
  tween.durationS = Math.max(durationS, 0);
  tween.elapsedS = 0;
  tween.active = true;
}

/**
 * Advances by `deltaS` and writes the eased pose into `outPosition` / `outTarget`.
 * Returns true on the frame the tween finishes (the output is then exactly `to`).
 */
export function stepTween(
  tween: PoseTween,
  deltaS: number,
  outPosition: { set(x: number, y: number, z: number): unknown },
  outTarget: { set(x: number, y: number, z: number): unknown },
): boolean {
  if (!tween.active) return false;
  tween.elapsedS += Math.max(deltaS, 0);
  const done = tween.durationS === 0 || tween.elapsedS >= tween.durationS;
  const k = done ? 1 : easeOutCubic(tween.elapsedS / tween.durationS);
  const { from, to } = tween;
  if (done) {
    outPosition.set(to.position[0], to.position[1], to.position[2]);
    outTarget.set(to.target[0], to.target[1], to.target[2]);
    tween.active = false;
    return true;
  }
  outPosition.set(
    from.position[0] + (to.position[0] - from.position[0]) * k,
    from.position[1] + (to.position[1] - from.position[1]) * k,
    from.position[2] + (to.position[2] - from.position[2]) * k,
  );
  outTarget.set(
    from.target[0] + (to.target[0] - from.target[0]) * k,
    from.target[1] + (to.target[1] - from.target[1]) * k,
    from.target[2] + (to.target[2] - from.target[2]) * k,
  );
  return false;
}

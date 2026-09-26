// Camera moves between two poses. Poses are interpolated in spherical coordinates around a moving
// target (radius, azimuth, elevation), so a move swings around the subject instead of cutting
// through it. Moves are driven either by their own time (view changes, reset) or by an external
// blend value (the reveal dolly, driven by the reveal clock); both end on the exact destination pose.

import { easeOutCubic } from "@hq/visualization";
import type { CameraPose, Vec3 } from "./presets";

interface Spherical {
  radius: number;
  /** Radians from +z (south) toward +x (east). */
  azimuth: number;
  /** Radians above the horizontal. */
  elevation: number;
}

export interface PoseTween {
  from: CameraPose;
  to: CameraPose;
  fromS: Spherical;
  toS: Spherical;
  durationS: number;
  elapsedS: number;
  active: boolean;
}

interface Settable {
  set(x: number, y: number, z: number): unknown;
}

const blankPose = (): CameraPose => ({ position: [0, 0, 0], target: [0, 0, 0] });
const blankSph = (): Spherical => ({ radius: 0, azimuth: 0, elevation: 0 });

export function createTween(): PoseTween {
  return {
    from: blankPose(),
    to: blankPose(),
    fromS: blankSph(),
    toS: blankSph(),
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

function toSpherical(pose: CameraPose, out: Spherical): void {
  const dx = pose.position[0] - pose.target[0];
  const dy = pose.position[1] - pose.target[1];
  const dz = pose.position[2] - pose.target[2];
  const r = Math.hypot(dx, dy, dz);
  out.radius = r;
  out.azimuth = Math.atan2(dx, dz);
  out.elevation = r > 0 ? Math.asin(Math.max(-1, Math.min(1, dy / r))) : 0;
}

/** Sets up `tween` to move from the given pose to `to`; no allocation. Drive it with `poseAt` or `stepTween`. */
export function setupTween(
  tween: PoseTween,
  fromPosition: readonly number[],
  fromTarget: readonly number[],
  to: CameraPose,
): void {
  copyVec(tween.from.position, fromPosition);
  copyVec(tween.from.target, fromTarget);
  copyVec(tween.to.position, to.position);
  copyVec(tween.to.target, to.target);
  toSpherical(tween.from, tween.fromS);
  toSpherical(tween.to, tween.toS);
  // Take the short way around.
  let dAz = tween.toS.azimuth - tween.fromS.azimuth;
  if (dAz > Math.PI) dAz -= 2 * Math.PI;
  if (dAz < -Math.PI) dAz += 2 * Math.PI;
  tween.toS.azimuth = tween.fromS.azimuth + dAz;
  tween.elapsedS = 0;
  tween.active = true;
}

/** Starts a self-timed move over `durationS` with cubic-out easing. */
export function startTween(
  tween: PoseTween,
  fromPosition: readonly number[],
  fromTarget: readonly number[],
  to: CameraPose,
  durationS: number,
): void {
  setupTween(tween, fromPosition, fromTarget, to);
  tween.durationS = Math.max(durationS, 0);
}

/** Writes the pose at blend `k` ∈ [0, 1]. k ≥ 1 writes the exact destination. */
export function poseAt(tween: PoseTween, k: number, outPosition: Settable, outTarget: Settable): void {
  const { from, to, fromS, toS } = tween;
  if (k >= 1) {
    outPosition.set(to.position[0], to.position[1], to.position[2]);
    outTarget.set(to.target[0], to.target[1], to.target[2]);
    return;
  }
  const c = k <= 0 ? 0 : k;
  const tx = from.target[0] + (to.target[0] - from.target[0]) * c;
  const ty = from.target[1] + (to.target[1] - from.target[1]) * c;
  const tz = from.target[2] + (to.target[2] - from.target[2]) * c;
  const r = fromS.radius + (toS.radius - fromS.radius) * c;
  const az = fromS.azimuth + (toS.azimuth - fromS.azimuth) * c;
  const el = fromS.elevation + (toS.elevation - fromS.elevation) * c;
  const h = r * Math.cos(el);
  outTarget.set(tx, ty, tz);
  outPosition.set(tx + h * Math.sin(az), ty + r * Math.sin(el), tz + h * Math.cos(az));
}

/**
 * Advances a self-timed move by `deltaS` and writes the eased pose. Returns true on the frame the move
 * finishes (the output is then exactly `to`).
 */
export function stepTween(tween: PoseTween, deltaS: number, outPosition: Settable, outTarget: Settable): boolean {
  if (!tween.active) return false;
  tween.elapsedS += Math.max(deltaS, 0);
  const done = tween.durationS === 0 || tween.elapsedS >= tween.durationS;
  poseAt(tween, done ? 1 : easeOutCubic(tween.elapsedS / tween.durationS), outPosition, outTarget);
  if (done) tween.active = false;
  return done;
}

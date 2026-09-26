// Pure helpers behind CameraRig, kept out of the component so they're unit-testable.

import type { SceneBounds } from "./bounds";

/** Orbit-control tuning. Damping is per update (R3F calls it once per frame). */
export const ORBIT = Object.freeze({
  dampingFactor: 0.08,
  /** Closest the orbit camera may get to its target (km). */
  minDistanceKm: 0.5,
  /** Farthest, as a multiple of the framed radius. */
  maxDistanceRadii: 20,
});

/** Aspect used before the canvas has a size (a 0-width canvas would make framing throw). */
export const FALLBACK_ASPECT = 16 / 9;

export function aspectOf(width: number, height: number): number {
  return width > 0 && height > 0 ? width / height : FALLBACK_ASPECT;
}

/**
 * A value that changes only when the framing actually changes, so a provider that returns fresh but
 * identical arrays (e.g. a refetch) never snaps the camera mid-orbit or mid-reveal.
 */
export function boundsSignature(b: SceneBounds): string {
  const r = (v: number) => v.toFixed(4);
  return [...b.min, ...b.max, b.radius, b.surfaceY].map(r).join(",");
}

interface Vec3Like {
  x: number;
  y: number;
  z: number;
  set(x: number, y: number, z: number): unknown;
}

interface OrbitLike {
  target: Vec3Like;
  enableDamping: boolean;
  update(): unknown;
}

const saved = { px: 0, py: 0, pz: 0, tx: 0, ty: 0, tz: 0 };

/**
 * Discards leftover orbit momentum without moving the camera. OrbitControls keeps the remaining
 * damped motion in private state that only `update()` clears, and only by applying it; so save the
 * pose, flush the momentum with damping off, then put the pose back.
 */
export function dropOrbitMomentum(controls: OrbitLike, cameraPosition: Vec3Like): void {
  saved.px = cameraPosition.x;
  saved.py = cameraPosition.y;
  saved.pz = cameraPosition.z;
  saved.tx = controls.target.x;
  saved.ty = controls.target.y;
  saved.tz = controls.target.z;
  const damping = controls.enableDamping;
  controls.enableDamping = false;
  controls.update(); // applies and clears the leftover motion
  cameraPosition.set(saved.px, saved.py, saved.pz);
  controls.target.set(saved.tx, saved.ty, saved.tz);
  controls.update(); // nothing left to apply: just re-aims at the target
  controls.enableDamping = damping;
}

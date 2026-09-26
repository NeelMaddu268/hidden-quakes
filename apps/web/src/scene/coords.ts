// ENU (meters, from the bundle) → three.js scene coordinates (docs/01 → Conventions).
//   x = e / 1000, y = u / 1000 × verticalExaggeration, z = −n / 1000; 1 scene unit = 1 km, y-up.
// Positions always come precomputed as `enu` in every record; the browser never reprojects lat/lon.
// Pure functions with no three.js import, so they're trivially testable and allocation-free.

import type { Enu, SceneMeta } from "./types";

export const METERS_PER_UNIT = 1000;

/** SceneMeta.verticalExaggeration, defaulting to 1.0 when the bundle omits it (docs/02 default). */
export function verticalExaggerationOf(scene: Pick<SceneMeta, "verticalExaggeration">): number {
  const ve = scene.verticalExaggeration ?? 1;
  if (!(ve > 0) || !Number.isFinite(ve)) {
    throw new Error(`SceneMeta.verticalExaggeration must be a positive number, got ${ve}`);
  }
  return ve;
}

/** Scene position of an ENU point as a new tuple. Use `writeEnuToScene` in hot paths. */
export function enuToScene(enu: Enu, verticalExaggeration = 1): [number, number, number] {
  return [
    enu.e / METERS_PER_UNIT,
    (enu.u / METERS_PER_UNIT) * verticalExaggeration,
    -enu.n / METERS_PER_UNIT,
  ];
}

/** Writes the scene position of `enu` into `out[offset..offset+2]` (typed-array instance attributes). */
export function writeEnuToScene(
  enu: Enu,
  verticalExaggeration: number,
  out: Float32Array | number[],
  offset: number,
): void {
  out[offset] = enu.e / METERS_PER_UNIT;
  out[offset + 1] = (enu.u / METERS_PER_UNIT) * verticalExaggeration;
  out[offset + 2] = -enu.n / METERS_PER_UNIT;
}

/** Inverse of `enuToScene` (picking, tests). */
export function sceneToEnu(x: number, y: number, z: number, verticalExaggeration = 1): Enu {
  return {
    e: x * METERS_PER_UNIT,
    n: -z * METERS_PER_UNIT,
    u: (y / verticalExaggeration) * METERS_PER_UNIT,
  };
}

/** Scene y of an absolute elevation (m ASL): u = elevM − originElevM. */
export function elevMToSceneY(
  elevM: number,
  scene: Pick<SceneMeta, "originElevM" | "verticalExaggeration">,
): number {
  return ((elevM - scene.originElevM) / METERS_PER_UNIT) * verticalExaggerationOf(scene);
}

/** Scene y of a display depth below the site surface: elevM = refSurfaceElevM − depthKm × 1000. */
export function depthKmToSceneY(
  depthKm: number,
  scene: Pick<SceneMeta, "originElevM" | "refSurfaceElevM" | "verticalExaggeration">,
): number {
  return elevMToSceneY(scene.refSurfaceElevM - depthKm * METERS_PER_UNIT, scene);
}

/** Display depth (km below the site surface) of a scene y; the inverse of `depthKmToSceneY`. */
export function sceneYToDepthKm(
  y: number,
  scene: Pick<SceneMeta, "originElevM" | "refSurfaceElevM" | "verticalExaggeration">,
): number {
  const elevM = (y / verticalExaggerationOf(scene)) * METERS_PER_UNIT + scene.originElevM;
  return (scene.refSurfaceElevM - elevM) / METERS_PER_UNIT;
}

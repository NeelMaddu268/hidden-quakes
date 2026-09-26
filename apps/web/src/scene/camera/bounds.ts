// Scene-space bounds of what the camera frames: the event clouds (candidates + public catalog).
// Presets frame the data rather than fixed numbers, so mock and showcase both land well framed.

export interface SceneBounds {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  /** Radius of the bounding sphere around `center` (km). */
  radius: number;
  /** Scene y of the site surface the oblique view looks across (km). */
  surfaceY: number;
}

/** Framing never gets tighter than this radius (km): a handful of events shouldn't fill the screen. */
export const MIN_FRAME_RADIUS_KM = 3;

/**
 * Fraction trimmed from each end per axis before framing, so a few scattered Tier C outliers don't
 * shrink the structure to a speck. Outliers stay rendered; they may sit just off the framed sphere.
 */
export const FRAME_TRIM = 0.02;

/** Quantile of a sorted array by linear interpolation. */
function quantileSorted(sorted: Float64Array, q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/**
 * Bounds over one or more xyz position arrays, trimmed per axis by `trim` at each end.
 * `surfaceY` is the scene y of the site surface. With no points at all, frames a MIN_FRAME_RADIUS_KM
 * sphere just below the surface at the origin.
 */
export function computeBounds(
  positionArrays: readonly Float32Array[],
  surfaceY: number,
  trim: number = FRAME_TRIM,
): SceneBounds {
  let n = 0;
  for (const arr of positionArrays) n += Math.floor(arr.length / 3);
  if (n === 0) {
    const r = MIN_FRAME_RADIUS_KM;
    return {
      min: [-r, surfaceY - 2 * r, -r],
      max: [r, surfaceY, r],
      center: [0, surfaceY - r, 0],
      radius: r,
      surfaceY,
    };
  }
  const axes = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
  let j = 0;
  for (const arr of positionArrays) {
    for (let i = 0; i + 2 < arr.length; i += 3, j++) {
      axes[0][j] = arr[i];
      axes[1][j] = arr[i + 1];
      axes[2][j] = arr[i + 2];
    }
  }
  const q = n >= 1 / Math.max(trim, 1e-9) ? trim : 0;
  const min: [number, number, number] = [0, 0, 0];
  const max: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    axes[k].sort();
    min[k] = quantileSorted(axes[k], q);
    max[k] = quantileSorted(axes[k], 1 - q);
  }
  const center: [number, number, number] = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const half = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2;
  return { min, max, center, radius: Math.max(half, MIN_FRAME_RADIUS_KM), surfaceY };
}

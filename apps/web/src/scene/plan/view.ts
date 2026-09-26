// The plan camera's pose for a viewport, from the tested planFrame. R3F sizes a default orthographic
// camera's frustum in CSS pixels, so `zoom` is pixels per scene km. The depth-section panel covers
// the left `reserveLeftPx`; the framed data is centered in the remaining region.

import type { SceneBounds } from "../camera/bounds";
import { planFrame } from "./geometry";

export interface PlanView {
  /** Pixels per km (orthographic camera zoom with a pixel-sized frustum). */
  zoom: number;
  /** Camera position; it looks straight down at (x, targetY, z) with grid north up. */
  x: number;
  y: number;
  z: number;
  targetY: number;
  near: number;
  far: number;
}

/** Near plane for the plan camera (km). Everything framed lies at least `clearance` below the camera. */
export const PLAN_NEAR_KM = 0.01;

export function planView(
  bounds: SceneBounds,
  clipBounds: Pick<SceneBounds, "min" | "max">,
  widthPx: number,
  heightPx: number,
  reserveLeftPx: number,
): PlanView {
  if (!(widthPx > 0) || !(heightPx > 0)) throw new Error("planView: invalid viewport");
  const reserve = reserveLeftPx > 0 && reserveLeftPx < widthPx * 0.6 ? reserveLeftPx : 0;
  const freeWidth = widthPx - reserve;
  const frame = planFrame(bounds, freeWidth / heightPx, clipBounds);
  const zoom = heightPx / frame.height;
  // Screen center sits `reserve / 2` px left of the free region's center: shift the camera left by that.
  const x = frame.centerX - reserve / 2 / zoom;
  return {
    zoom,
    x,
    y: frame.cameraY,
    z: frame.centerZ,
    targetY: frame.cameraY - 1,
    near: PLAN_NEAR_KM,
    far: frame.far,
  };
}

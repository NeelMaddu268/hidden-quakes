// Camera presets (docs/02 CameraView): oblique surface view, low side view below ground, plan view.
// Each is a pure function of the framed bounds and the viewport aspect, so they're deterministic
// (the reveal must be identical every run) and testable without WebGL.

import type { CameraView } from "../../state/demo";
import type { SceneBounds } from "./bounds";

export type Vec3 = [number, number, number];

export interface CameraPose {
  position: Vec3;
  target: Vec3;
}

/** Vertical field of view (degrees) shared by the Canvas camera and the framing math. */
export const CAMERA_FOV_DEG = 40;

/** Extra room around the framed sphere so points never touch the viewport edge. */
const FRAME_MARGIN = 1.08;

const DEG = Math.PI / 180;

/** Direction the camera looks *from*, relative to the target: azimuth from south toward east, elevation. */
const VIEW_ANGLES: Record<CameraView, { azimuthDeg: number; elevationDeg: number }> = {
  // Surface view from the south-southeast, looking across and into the ground.
  oblique: { azimuthDeg: 28, elevationDeg: 30 },
  // Low view from due south, level with the middle of the column from the site surface down to the
  // deepest framed event: depth reads as height on screen, with the surface (0 km) in frame.
  side: { azimuthDeg: 0, elevationDeg: 4 },
  // Straight down, north up. Not exactly 90° so orbit controls keep a well-defined azimuth.
  plan: { azimuthDeg: 0, elevationDeg: 89.9 },
};

/** Distance at which a sphere of `radius` fits the viewport for the given vertical fov and aspect. */
export function fitDistance(radius: number, aspect: number, fovDeg: number = CAMERA_FOV_DEG): number {
  if (!(radius > 0) || !(aspect > 0)) throw new Error(`fitDistance: bad radius ${radius} or aspect ${aspect}`);
  const halfV = (fovDeg * DEG) / 2;
  const halfH = Math.atan(Math.tan(halfV) * aspect);
  return (radius / Math.sin(Math.min(halfV, halfH))) * FRAME_MARGIN;
}

/**
 * Where each preset looks. Oblique looks between the surface and the cloud so both are in frame. Side
 * looks at the middle of the whole column, site surface to deepest framed event, so a compact deep
 * cluster (the real showcase: a column several km below the site) still shows the surface, the
 * ruler's 0 km and the wellheads, instead of zooming onto the cluster alone.
 */
function presetTarget(view: CameraView, b: SceneBounds): Vec3 {
  const [cx, cy, cz] = b.center;
  if (view === "oblique") return [cx, (b.surfaceY + cy) / 2, cz];
  if (view === "side") return [cx, (Math.max(b.surfaceY, b.max[1]) + b.min[1]) / 2, cz];
  return [cx, cy, cz];
}

export function presetPose(
  view: CameraView,
  bounds: SceneBounds,
  aspect: number,
  fovDeg: number = CAMERA_FOV_DEG,
): CameraPose {
  const target = presetTarget(view, bounds);
  // Frame a sphere around the target that holds every corner of the box; oblique and side extend the
  // box up to the surface so the ground above the cloud is in frame too.
  const yTop = view === "plan" ? bounds.max[1] : Math.max(bounds.max[1], bounds.surfaceY);
  let radius = bounds.radius;
  for (const x of [bounds.min[0], bounds.max[0]])
    for (const y of [bounds.min[1], yTop])
      for (const z of [bounds.min[2], bounds.max[2]])
        radius = Math.max(radius, Math.hypot(x - target[0], y - target[1], z - target[2]));
  const d = fitDistance(radius, aspect, fovDeg);
  const { azimuthDeg, elevationDeg } = VIEW_ANGLES[view];
  const az = azimuthDeg * DEG;
  const el = elevationDeg * DEG;
  const horizontal = d * Math.cos(el);
  return {
    position: [
      target[0] + horizontal * Math.sin(az), // east
      target[1] + d * Math.sin(el), // up
      target[2] + horizontal * Math.cos(az), // +z is south
    ],
    target,
  };
}

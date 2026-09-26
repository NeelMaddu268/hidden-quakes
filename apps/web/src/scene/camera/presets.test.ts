import { PerspectiveCamera, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { computeBounds, MIN_FRAME_RADIUS_KM } from "./bounds";
import { CAMERA_FOV_DEG, fitDistance, presetPose, type CameraPose } from "./presets";

/** A test-local cloud: 4 km wide, 3 km tall, centered 3.5 km below a surface at y = 0.05. */
function cloud(): Float32Array {
  const pts: number[] = [];
  for (let i = 0; i < 100; i++) {
    const a = (i / 100) * Math.PI * 2;
    pts.push(Math.cos(a) * 2, -2 - (i % 10) * (3 / 9), Math.sin(a) * 2 - 1);
  }
  return new Float32Array(pts);
}

function cameraAt(pose: CameraPose, aspect: number): PerspectiveCamera {
  const cam = new PerspectiveCamera(CAMERA_FOV_DEG, aspect, 0.05, 1000);
  cam.position.set(...pose.position);
  cam.lookAt(new Vector3(...pose.target));
  cam.updateMatrixWorld(true);
  return cam;
}

function ndc(cam: PerspectiveCamera, p: [number, number, number]): Vector3 {
  return new Vector3(...p).project(cam);
}

describe("computeBounds", () => {
  it("frames the cloud's trimmed extent and centers on it", () => {
    const b = computeBounds([cloud()], 0.05, 0);
    expect(b.min[0]).toBeCloseTo(-2, 5);
    expect(b.max[0]).toBeCloseTo(2, 5);
    expect(b.min[1]).toBeCloseTo(-5, 5);
    expect(b.max[1]).toBeCloseTo(-2, 5);
    expect(b.center[1]).toBeCloseTo(-3.5, 5);
    expect(b.radius).toBeGreaterThanOrEqual(MIN_FRAME_RADIUS_KM);
  });

  it("ignores a lone far outlier via the per-axis trim", () => {
    const withOutlier = new Float32Array([...cloud(), 60, -40, 60]);
    const trimmed = computeBounds([withOutlier], 0.05);
    const raw = computeBounds([withOutlier], 0.05, 0);
    expect(trimmed.max[0]).toBeLessThan(3);
    expect(raw.max[0]).toBe(60);
  });

  it("merges several arrays (candidates + public catalog)", () => {
    const b = computeBounds([new Float32Array([0, -1, 0]), new Float32Array([10, -9, -10])], 0, 0);
    expect(b.min).toEqual([0, -9, -10]);
    expect(b.max).toEqual([10, -1, 0]);
  });

  it("frames a default sphere under the origin when there's no data", () => {
    const b = computeBounds([], 0.05);
    expect(b.center).toEqual([0, 0.05 - MIN_FRAME_RADIUS_KM, 0]);
    expect(b.radius).toBe(MIN_FRAME_RADIUS_KM);
  });
});

describe("presetPose", () => {
  const b = computeBounds([cloud()], 0.05, 0);

  for (const aspect of [16 / 9, 1280 / 720, 4 / 3, 0.6]) {
    for (const view of ["oblique", "side", "plan"] as const) {
      it(`${view} @ aspect ${aspect.toFixed(2)} keeps every framed corner on screen`, () => {
        const cam = cameraAt(presetPose(view, b, aspect), aspect);
        for (const x of [b.min[0], b.max[0]])
          for (const y of [b.min[1], b.max[1]])
            for (const z of [b.min[2], b.max[2]]) {
              const p = ndc(cam, [x, y, z]);
              expect(Math.abs(p.x)).toBeLessThanOrEqual(1);
              expect(Math.abs(p.y)).toBeLessThanOrEqual(1);
              expect(p.z).toBeLessThan(1); // in front of the far plane, behind the camera never
            }
      });
    }
  }

  it("oblique looks down from above the surface, from the south-east", () => {
    const pose = presetPose("oblique", b, 16 / 9);
    expect(pose.position[1]).toBeGreaterThan(b.surfaceY);
    expect(pose.position[0]).toBeGreaterThan(pose.target[0]); // east of target
    expect(pose.position[2]).toBeGreaterThan(pose.target[2]); // south of target (+z)
    const cam = cameraAt(pose, 16 / 9);
    expect(Math.abs(ndc(cam, [b.center[0], b.surfaceY, b.center[2]]).y)).toBeLessThan(1);
  });

  it("side is a low view from due south at the cloud's depth", () => {
    const pose = presetPose("side", b, 16 / 9);
    expect(pose.position[0]).toBeCloseTo(pose.target[0], 9);
    expect(pose.position[2]).toBeGreaterThan(pose.target[2]);
    const d = Math.hypot(...pose.position.map((v, i) => v - pose.target[i]));
    expect(pose.position[1] - pose.target[1]).toBeLessThan(0.1 * d); // under ~6° elevation
    expect(pose.position[1]).toBeLessThan(b.surfaceY); // below ground
  });

  it("side frames the whole column from the site surface down, even for a compact deep cluster", () => {
    // A tight cluster 4–5 km below the surface: the side view must still show the surface (0 km).
    const deep: number[] = [];
    for (let i = 0; i < 50; i++) deep.push(((i % 5) - 2) * 0.1, -4 - (i % 10) * 0.1, ((i % 7) - 3) * 0.1);
    const db = computeBounds([new Float32Array(deep)], 0, 0);
    for (const aspect of [16 / 9, 4 / 3]) {
      const cam = cameraAt(presetPose("side", db, aspect), aspect);
      const surface = ndc(cam, [db.center[0], db.surfaceY, db.center[2]]);
      const bottom = ndc(cam, [db.center[0], db.min[1], db.center[2]]);
      expect(Math.abs(surface.y)).toBeLessThan(1);
      expect(Math.abs(bottom.y)).toBeLessThan(1);
      expect(surface.y).toBeGreaterThan(bottom.y);
    }
  });

  it("side view shows depth as screen height: deeper points draw lower", () => {
    const cam = cameraAt(presetPose("side", b, 16 / 9), 16 / 9);
    const shallow = ndc(cam, [b.center[0], -2, b.center[2]]);
    const deep = ndc(cam, [b.center[0], -5, b.center[2]]);
    expect(deep.y).toBeLessThan(shallow.y);
  });

  it("plan looks straight down with north up and east right", () => {
    const pose = presetPose("plan", b, 16 / 9);
    const cam = cameraAt(pose, 16 / 9);
    const [cx, cy, cz] = pose.target;
    const north = ndc(cam, [cx, cy, cz - 1]); // north is −z
    const east = ndc(cam, [cx + 1, cy, cz]);
    expect(north.y).toBeGreaterThan(0);
    expect(Math.abs(north.x)).toBeLessThan(1e-3);
    expect(east.x).toBeGreaterThan(0);
    expect(Math.abs(east.y)).toBeLessThan(1e-3);
  });

  it("is deterministic (the reveal must be identical every run)", () => {
    expect(presetPose("oblique", b, 1.5)).toEqual(presetPose("oblique", b, 1.5));
  });
});

describe("fitDistance", () => {
  it("backs off further for narrow viewports", () => {
    expect(fitDistance(5, 0.5)).toBeGreaterThan(fitDistance(5, 2));
  });

  it("rejects nonsense instead of placing the camera at NaN", () => {
    expect(() => fitDistance(0, 1)).toThrow();
    expect(() => fitDistance(1, 0)).toThrow();
    expect(() => fitDistance(Number.NaN, 1)).toThrow();
  });
});

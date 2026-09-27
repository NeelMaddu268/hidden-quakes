import { OrthographicCamera, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { computeBounds } from "../camera/bounds";
import { planView } from "./view";

// Test-local cloud: 6 km east-west, 3 km north-south, 2–5 km below a surface at y = 0.05.
const cloud = new Float32Array([-3, -2, -1.5, 3, -5, 1.5, 0.5, -3, 0, -2, -4, -1]);
const bounds = computeBounds([cloud], 0.05, 0);

/** An orthographic camera set up the way R3F sizes a default one: pixel frustum, zoom = px per unit. */
function camera(W: number, H: number, v: ReturnType<typeof planView>): OrthographicCamera {
  const cam = new OrthographicCamera(-W / 2, W / 2, H / 2, -H / 2, v.near, v.far);
  cam.zoom = v.zoom;
  cam.up.set(0, 0, -1);
  cam.position.set(v.x, v.y, v.z);
  cam.lookAt(v.x, v.targetY, v.z);
  cam.updateProjectionMatrix();
  cam.updateMatrixWorld(true);
  return cam;
}

/** Screen position in CSS px (origin top-left). */
function screen(cam: OrthographicCamera, W: number, H: number, p: [number, number, number]) {
  const ndc = new Vector3(...p).project(cam);
  return { x: ((ndc.x + 1) / 2) * W, y: ((1 - ndc.y) / 2) * H, z: ndc.z };
}

describe("planView", () => {
  for (const [W, H, reserve] of [
    [1280, 720, 0],
    [1280, 720, 459],
    [3840, 2160, 808],
    [1920, 1080, 677],
  ]) {
    it(`fits the framed box in the free region at ${W}×${H} (reserve ${reserve}px)`, () => {
      const v = planView(bounds, bounds, W, H, reserve);
      const cam = camera(W, H, v);
      for (const x of [bounds.min[0], bounds.max[0]])
        for (const y of [bounds.min[1], bounds.max[1]])
          for (const z of [bounds.min[2], bounds.max[2]]) {
            const s = screen(cam, W, H, [x, y, z]);
            expect(s.x).toBeGreaterThanOrEqual(reserve - 1e-6);
            expect(s.x).toBeLessThanOrEqual(W + 1e-6);
            expect(s.y).toBeGreaterThanOrEqual(-1e-6);
            expect(s.y).toBeLessThanOrEqual(H + 1e-6);
            expect(Math.abs(s.z)).toBeLessThan(1); // inside the near/far clip range
          }
    });
  }

  it("centers the data in the region right of the panel", () => {
    const W = 1280, H = 720, reserve = 459;
    const v = planView(bounds, bounds, W, H, reserve);
    const s = screen(camera(W, H, v), W, H, [bounds.center[0], bounds.center[1], bounds.center[2]]);
    expect(s.x).toBeCloseTo(reserve + (W - reserve) / 2, 6);
    expect(s.y).toBeCloseTo(H / 2, 6);
  });

  it("with the panel docked right, fits and centers the data in the region left of it", () => {
    const W = 1280, H = 720, right = 408;
    const v = planView(bounds, bounds, W, H, 0, right);
    const cam = camera(W, H, v);
    for (const x of [bounds.min[0], bounds.max[0]])
      for (const z of [bounds.min[2], bounds.max[2]]) {
        const s = screen(cam, W, H, [x, bounds.center[1], z]);
        expect(s.x).toBeGreaterThanOrEqual(-1e-6);
        expect(s.x).toBeLessThanOrEqual(W - right + 1e-6);
      }
    const c = screen(cam, W, H, [bounds.center[0], bounds.center[1], bounds.center[2]]);
    expect(c.x).toBeCloseTo((W - right) / 2, 6);
  });

  it("is grid north up and grid east right, at equal scale on both axes", () => {
    const W = 1280, H = 720;
    const v = planView(bounds, bounds, W, H, 0);
    const cam = camera(W, H, v);
    const c = screen(cam, W, H, [0, -3, 0]);
    const north = screen(cam, W, H, [0, -3, -1]); // north is −z
    const east = screen(cam, W, H, [1, -3, 0]);
    expect(north.y).toBeLessThan(c.y);
    expect(north.x).toBeCloseTo(c.x, 6);
    expect(east.x).toBeGreaterThan(c.x);
    expect(east.y).toBeCloseTo(c.y, 6);
    expect(east.x - c.x).toBeCloseTo(c.y - north.y, 6); // 1 km east = 1 km north on screen
    expect(east.x - c.x).toBeCloseTo(v.zoom, 6);
  });

  it("keeps deep events outside the framed subset inside the clip range", () => {
    const deep = computeBounds([new Float32Array([...cloud, 0, -12, 0])], 0.05, 0);
    const v = planView(bounds, deep, 1280, 720, 0);
    const s = screen(camera(1280, 720, v), 1280, 720, [0, -12, 0]);
    expect(Math.abs(s.z)).toBeLessThan(1);
  });

  it("ignores an unreasonable reserve instead of squeezing the map to nothing", () => {
    const a = planView(bounds, bounds, 1280, 720, 0);
    const b = planView(bounds, bounds, 1280, 720, 1000);
    expect(b).toEqual(a);
  });

  it("fails loudly on a zero-size viewport", () => {
    expect(() => planView(bounds, bounds, 0, 720, 0)).toThrow();
  });
});

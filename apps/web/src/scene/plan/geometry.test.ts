import { describe, expect, it } from "vitest";
import { OrthographicCamera, Vector3 } from "three";
import { MIN_SECTION_MARGIN_KM, planFrame, SECTION_FRAME_MARGIN, sectionPoints, sectionStructureFit, usableErrorKm } from "./geometry";
import type { SceneBounds } from "../camera/bounds";

const bounds: SceneBounds = { min: [-2, -8, -3], max: [4, 1, 5], center: [1, -3.5, 1], radius: 8, surfaceY: 2 };

describe("plan framing", () => {
  it.each([16 / 9, 9 / 16, 1, 4])("keeps east/north at equal scale at aspect %s", (aspect) => {
    const f = planFrame(bounds, aspect);
    const c = new OrthographicCamera(-f.width / 2, f.width / 2, f.height / 2, -f.height / 2, .01, f.far);
    c.position.set(f.centerX, f.cameraY, f.centerZ); c.up.set(0, 0, -1);
    c.lookAt(f.centerX, bounds.center[1], f.centerZ); c.updateMatrixWorld();
    const origin = new Vector3(1, 0, 1).project(c);
    const east = new Vector3(2, 0, 1).project(c);
    const north = new Vector3(1, 0, 0).project(c);
    expect(east.x).toBeGreaterThan(origin.x);
    expect(north.y).toBeGreaterThan(origin.y);
    expect((east.x - origin.x) * aspect).toBeCloseTo(north.y - origin.y);
    expect(new Vector3(1, -8, 1).project(c).z).toBeLessThan(1);
  });
  it("uses depth only for clipping, never horizontal framing", () => {
    const f = planFrame(bounds, 2);
    const deep = planFrame({ ...bounds, min: [-2, -800, -3] }, 2);
    expect(deep.width).toBe(f.width); expect(deep.height).toBe(f.height); expect(deep.far).toBeGreaterThan(f.far);
  });
  it("keeps unframed outliers inside the depth clip range without shrinking the map", () => {
    const f = planFrame(bounds, 2, { min: [-500, -800, -500], max: [500, 100, 500] });
    expect(f.width).toBe(planFrame(bounds, 2).width);
    expect(f.cameraY).toBeGreaterThan(100);
    expect(f.far).toBeGreaterThan(f.cameraY + 800);
    expect(() => planFrame({ ...bounds, surfaceY: NaN }, 2)).toThrow(/nonfinite/);
  });
  it.each([0, -1, NaN, Infinity])("rejects invalid aspect %s", (a) => expect(() => planFrame(bounds, a)).toThrow());
});

describe("all-event depth projection", () => {
  it("uses site elevation despite conflicting published depth and ENU up, and actual sensor elevation", () => {
    const record = { enu: { e: 2000, n: 3000, u: 999 }, elevM: -500, depthKm: 99 };
    const projected = sectionPoints([], [record], [{ enu: record.enu, surfaceElevM: 1600, sensorElevM: 1300 }], { refSurfaceElevM: 1500 });
    expect([...projected.publicEvents.xy]).toEqual([2, 2]);
    expect([...projected.sensors]).toEqual([2, .2]); expect([...projected.wellheads]).toEqual([2, -.1]);
  });
  it("retains all input rows and horizontal errors when vertical error is missing", () => {
    const projected = sectionPoints([
      { enu: { e: 0, n: 0, u: 0 }, elevM: 100, quality: { hErrM: 200, vErrM: null } },
      { enu: { e: 999000, n: 0, u: 0 }, elevM: -100, quality: { hErrM: null, vErrM: null } },
    ], [], [], { refSurfaceElevM: 1000 });
    expect([...projected.candidates.xy]).toEqual([0, .9, 999, 1.1]);
    expect([...projected.candidates.errors]).toEqual([.2, 0, 0, 0]);
  });
  it("frames the structure from the site surface down, with a margin, at true scale", () => {
    // A compact deep column (east −0.5…0.5 km, 3.5…5 km deep), like the real showcase's.
    const xy = new Float64Array([-0.5, 3.5, 0.5, 5, 0, 4.2]);
    const f = sectionStructureFit(xy, new Float64Array(), 300, 180, 16, 0);
    const m = Math.max(MIN_SECTION_MARGIN_KM, SECTION_FRAME_MARGIN * 5);
    expect(f.minEast).toBeCloseTo(-0.5 - m, 9);
    expect(f.maxEast).toBeCloseTo(0.5 + m, 9);
    expect(f.minDepth).toBeCloseTo(-m / 2, 9); // the surface (0) plus room above it
    expect(f.maxDepth).toBeCloseTo(5 + m, 9);
    for (const [e, d] of [[-0.5, 0], [0.5, 5]]) {
      expect(f.offsetX + e * f.pxPerKm).toBeGreaterThanOrEqual(16 - 1e-9);
      expect(f.offsetX + e * f.pxPerKm).toBeLessThanOrEqual(284 + 1e-9);
      expect(f.offsetY + d * f.pxPerKm).toBeGreaterThanOrEqual(16 - 1e-9);
      expect(f.offsetY + d * f.pxPerKm).toBeLessThanOrEqual(164 + 1e-9);
    }
  });
  it("trims far outliers like the camera's framing, and falls back to every event without a structure", () => {
    const pts: number[] = [];
    for (let i = 0; i < 99; i++) pts.push((i % 10) * 0.1, 3 + (i % 7) * 0.1);
    pts.push(40, 30); // one far outlier among 100 points
    const f = sectionStructureFit(new Float64Array(pts), new Float64Array(), 300, 180, 16, 0.02);
    expect(f.maxEast).toBeLessThan(3);
    expect(f.maxDepth).toBeLessThan(6);
    const fb = sectionStructureFit(new Float64Array(), new Float64Array([10, 2, 12, 4]), 300, 180, 16, 0);
    expect(fb.minEast).toBeLessThan(10);
    expect(fb.maxEast).toBeGreaterThan(12);
  });
  it("handles empty and coincident populations deterministically", () => {
    const f = sectionStructureFit(new Float64Array(), new Float64Array(), 300, 180, 16, 0);
    expect(f).toEqual(sectionStructureFit(new Float64Array(), new Float64Array(), 300, 180, 16, 0));
    expect(f.pxPerKm).toBeGreaterThan(0);
    expect(sectionStructureFit(new Float64Array([0, 0, 0, 0]), new Float64Array(), 300, 180, 16, 0).pxPerKm).toBeGreaterThan(0);
  });
  it("rejects nonfinite positions and invalid viewports loudly", () => {
    expect(() => sectionPoints([], [{ enu: { e: NaN, n: 0, u: 0 }, elevM: 0 }], [], { refSurfaceElevM: 0 })).toThrow();
    expect(() => sectionStructureFit(new Float64Array(), new Float64Array(), 30, 30, 16, 0)).toThrow();
    expect(() => sectionStructureFit(new Float64Array([0, NaN]), new Float64Array(), 300, 180, 16, 0)).toThrow();
  });
  it.each([null, undefined, 0, -1, NaN, Infinity])("does not invent an error for %s", (error) => expect(usableErrorKm(error)).toBe(0));
});

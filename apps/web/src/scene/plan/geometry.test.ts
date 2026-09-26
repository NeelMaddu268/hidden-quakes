import { describe, expect, it } from "vitest";
import { OrthographicCamera, Vector3 } from "three";
import { planFrame, sectionFit, sectionPoints, usableErrorKm } from "./geometry";
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
  it("fits every finite uncertainty extent and borehole endpoint at true scale", () => {
    const xy = new Float64Array([-4, 2, 8, 6]);
    const errors = new Float64Array([2, 3, 1, 4]);
    const f = sectionFit([{ xy, errors }], [new Float64Array([20, -2])], 300, 180, 16);
    expect([f.minEast, f.maxEast, f.minDepth, f.maxDepth]).toEqual([-6, 20, -2, 10]);
    for (const [e, d] of [[-6, -2], [20, 10]]) {
      expect(f.offsetX + e * f.pxPerKm).toBeGreaterThanOrEqual(16 - 1e-9);
      expect(f.offsetX + e * f.pxPerKm).toBeLessThanOrEqual(284 + 1e-9);
      expect(f.offsetY + d * f.pxPerKm).toBeGreaterThanOrEqual(16 - 1e-9);
      expect(f.offsetY + d * f.pxPerKm).toBeLessThanOrEqual(164 + 1e-9);
    }
  });
  it("handles empty and coincident populations deterministically", () => {
    const f = sectionFit([], [], 300, 180, 16);
    expect(f).toEqual(sectionFit([], [], 300, 180, 16)); expect(f.pxPerKm).toBeGreaterThan(0);
    expect(sectionFit([{ xy: new Float64Array([0, 0, 0, 0]), errors: new Float64Array(4) }], [], 300, 180, 16).pxPerKm).toBe(f.pxPerKm);
  });
  it("rejects nonfinite positions and invalid viewports loudly", () => {
    expect(() => sectionPoints([], [{ enu: { e: NaN, n: 0, u: 0 }, elevM: 0 }], [], { refSurfaceElevM: 0 })).toThrow();
    expect(() => sectionFit([], [], 30, 30, 16)).toThrow();
    expect(() => sectionFit([{ xy: new Float64Array([0, NaN]), errors: new Float64Array(2) }], [], 300, 180, 16)).toThrow();
  });
  it.each([null, undefined, 0, -1, NaN, Infinity])("does not invent an error for %s", (error) => expect(usableErrorKm(error)).toBe(0));
});

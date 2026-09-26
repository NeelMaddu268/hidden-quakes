import { Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { computeBounds } from "../camera/bounds";
import { depthKmToSceneY, elevMToSceneY } from "../coords";
import {
  buildSlabGrid,
  buildTerrainGrid,
  chooseSurface,
  FALLBACK_HALF_WIDTH_M,
  FALLBACK_MARGIN_M,
  gridIndex,
  surfaceExtentM,
} from "./grid";

const scene = { originElevM: 1627.7, refSurfaceElevM: 1627.7, verticalExaggeration: 1 };
const bounds3 = { eMin: -8000, eMax: 8000, nMin: -8000, nMax: 8000 };

function vertex(pos: Float32Array, i: number): Vector3 {
  return new Vector3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
}

describe("buildTerrainGrid", () => {
  // Row 0 = north, column 0 = west; elevations chosen so every vertex is distinguishable.
  const elev = Float32Array.from([1600, 1610, 1620, 1700, 1710, 1720, 1800, 1810, 1820]);
  const grid = buildTerrainGrid(elev, { sizePx: [3, 3], enuBounds: bounds3 }, scene);

  it("puts pixel (r, c) at e = eMin + c·Δe, n = nMax − r·Δn, y from elevM (ENU → scene)", () => {
    // North-west corner: e = −8 km → x = −8; n = +8 km → z = −8.
    expect(vertex(grid.positions, 0).toArray()).toEqual([-8, expect.closeTo(elevMToSceneY(1600, scene), 6), -8]);
    // North-east corner.
    expect(vertex(grid.positions, 2).x).toBe(8);
    expect(vertex(grid.positions, 2).z).toBe(-8);
    // Centre = origin.
    expect(vertex(grid.positions, 4).x).toBe(0);
    expect(vertex(grid.positions, 4).z).toBeCloseTo(0, 9);
    expect(vertex(grid.positions, 4).y).toBeCloseTo((1710 - 1627.7) / 1000, 6);
    // South-west corner: n = −8 km → z = +8 (south is +z).
    expect(vertex(grid.positions, 6).toArray()).toEqual([-8, expect.closeTo(0.1723, 4), 8]);
  });

  it("applies the vertical exaggeration to heights only", () => {
    const ve2 = buildTerrainGrid(elev, { sizePx: [3, 3], enuBounds: bounds3 }, { ...scene, verticalExaggeration: 2 });
    for (let i = 0; i < 9; i++) {
      expect(ve2.positions[i * 3]).toBe(grid.positions[i * 3]);
      expect(ve2.positions[i * 3 + 1]).toBeCloseTo(grid.positions[i * 3 + 1] * 2, 6);
      expect(ve2.positions[i * 3 + 2]).toBe(grid.positions[i * 3 + 2]);
    }
  });

  it("rejects a height count that doesn't match the grid", () => {
    expect(() => buildTerrainGrid(new Float32Array(8), { sizePx: [3, 3], enuBounds: bounds3 }, scene)).toThrow();
  });
});

describe("gridIndex", () => {
  it("makes two triangles per cell, all facing up (+y)", () => {
    const flat = buildTerrainGrid(new Float32Array(12).fill(1627.7), { sizePx: [4, 3], enuBounds: bounds3 }, scene);
    const idx = flat.index;
    expect(idx.length).toBe(3 * 2 * 6);
    const ab = new Vector3();
    const ac = new Vector3();
    for (let t = 0; t < idx.length; t += 3) {
      const a = vertex(flat.positions, idx[t]);
      ab.subVectors(vertex(flat.positions, idx[t + 1]), a);
      ac.subVectors(vertex(flat.positions, idx[t + 2]), a);
      expect(ab.cross(ac).y).toBeGreaterThan(0);
    }
    expect(Math.max(...idx)).toBe(11);
  });

  it("rejects degenerate grids", () => {
    expect(() => gridIndex(1, 5)).toThrow();
  });
});

describe("surfaceExtentM", () => {
  it("uses the baked terrain's bounds when there is one", () => {
    const b = computeBounds([], 0);
    const meta = { enuBounds: { eMin: -1, eMax: 2, nMin: -3, nMax: 4, at: "pixelCenters" as const } };
    expect(surfaceExtentM(meta, b)).toEqual({ eMin: -1, eMax: 2, nMin: -3, nMax: 4 });
  });

  it("falls back to an origin-centred square that covers the framed data", () => {
    const small = computeBounds([new Float32Array([1, -2, 1])], 0, 0);
    expect(surfaceExtentM(null, small).eMax).toBe(FALLBACK_HALF_WIDTH_M);
    const wide = computeBounds([new Float32Array([-11, -2, 4, 3, -3, 9.5])], 0, 0);
    const ext = surfaceExtentM(null, wide);
    expect(ext.eMax).toBe(11_000 + FALLBACK_MARGIN_M);
    expect(ext).toEqual({ eMin: -ext.eMax, eMax: ext.eMax, nMin: -ext.eMax, nMax: ext.eMax });
  });
});

describe("buildSlabGrid", () => {
  it("is a flat quad at the site surface (depth 0)", () => {
    const s = { ...scene, originElevM: 1600, verticalExaggeration: 3 };
    const slab = buildSlabGrid(bounds3, s);
    expect(slab.width).toBe(2);
    for (let i = 0; i < 4; i++) expect(slab.positions[i * 3 + 1]).toBeCloseTo(depthKmToSceneY(0, s), 6); // Float32 positions
    expect(slab.index.length).toBe(6);
  });
});

describe("chooseSurface", () => {
  it("waits, shows the terrain, or falls back to the labelled slab", () => {
    expect(chooseSurface("loading", { forced: false, mismatch: null })).toBe("none");
    expect(chooseSurface("ready", { forced: false, mismatch: null })).toBe("terrain");
    expect(chooseSurface("unavailable", { forced: false, mismatch: null })).toBe("slab");
    expect(chooseSurface("ready", { forced: false, mismatch: "origin differs" })).toBe("slab");
  });

  it("the force flag wins in every state", () => {
    for (const status of ["loading", "ready", "unavailable"] as const) {
      expect(chooseSurface(status, { forced: true, mismatch: null })).toBe("slab");
    }
  });
});

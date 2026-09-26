import { colors } from "@hq/visualization";
import { Color } from "three";
import { describe, expect, it } from "vitest";
import { elevMToSceneY } from "../coords";
import {
  CONTOUR_INTERVAL_M,
  createTerrainUniforms,
  OPAQUE_THRESHOLD,
  TERRAIN_FRAGMENT_SHADER,
  TERRAIN_VERTEX_SHADER,
  terrainDepthWrite,
  terrainOpacity,
} from "./material";

describe("terrain opacity and depth writes", () => {
  it("writes depth only while (almost) solid, so events show under a faded terrain", () => {
    expect(terrainDepthWrite(1)).toBe(true);
    expect(terrainDepthWrite(OPAQUE_THRESHOLD)).toBe(true);
    expect(terrainDepthWrite(0.98)).toBe(false);
    expect(terrainDepthWrite(0.12)).toBe(false);
    expect(terrainDepthWrite(0)).toBe(false);
  });

  it("clamps the shared fx value; NaN renders solid rather than vanishing", () => {
    expect(terrainOpacity(0.12)).toBe(0.12);
    expect(terrainOpacity(-1)).toBe(0);
    expect(terrainOpacity(3)).toBe(1);
    expect(terrainOpacity(Number.NaN)).toBe(1);
  });
});

describe("createTerrainUniforms", () => {
  const base = { originElevM: 1627.7, verticalExaggeration: 1.5, flatShadeValue: 180 };

  it("uses the design tokens (converted to linear) and 100 m contours", () => {
    const u = createTerrainUniforms({ ...base, contours: true });
    expect(u.uBase.value.equals(new Color(colors.terrain))).toBe(true);
    expect(u.uContour.value.equals(new Color(colors.contour))).toBe(true);
    expect(u.uContourIntervalM.value).toBe(CONTOUR_INTERVAL_M);
    expect(CONTOUR_INTERVAL_M).toBe(100);
    expect(u.uContourOpacity.value).toBeGreaterThan(0);
    expect(u.uOpacity.value).toBe(1);
    expect(u.uFlatShade.value).toBeCloseTo(180 / 255, 9);
  });

  it("turns contours off for the abstract slab", () => {
    expect(createTerrainUniforms({ ...base, contours: false }).uContourOpacity.value).toBe(0);
  });

  it("gives every uniform the shaders declare a value", () => {
    const u = createTerrainUniforms({ ...base, contours: true });
    const declared = [...(TERRAIN_VERTEX_SHADER + TERRAIN_FRAGMENT_SHADER).matchAll(/uniform \w+ (\w+);/g)].map(
      (m) => m[1],
    );
    expect(declared.length).toBeGreaterThan(5);
    for (const name of declared) expect(u, name).toHaveProperty(name);
  });

  it("the shader's elevation recovery inverts the ENU → scene mapping", () => {
    // Vertex shader: vElevM = position.y / uVerticalExaggeration * 1000.0 + uOriginElevM.
    const scene = { originElevM: base.originElevM, verticalExaggeration: base.verticalExaggeration };
    for (const elevM of [1489.02, 1627.7, 2527.03, -300]) {
      const y = elevMToSceneY(elevM, scene);
      expect((y / scene.verticalExaggeration) * 1000 + scene.originElevM).toBeCloseTo(elevM, 6);
    }
    expect(TERRAIN_VERTEX_SHADER).toContain("position.y / uVerticalExaggeration * 1000.0 + uOriginElevM");
  });
});

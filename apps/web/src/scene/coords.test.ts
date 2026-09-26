import { describe, expect, it } from "vitest";
import {
  depthKmToSceneY,
  eastMToSceneX,
  elevMToSceneY,
  enuToScene,
  northMToSceneZ,
  sceneToEnu,
  sceneXToEastM,
  sceneZToNorthM,
  upMToSceneY,
  sceneYToDepthKm,
  verticalExaggerationOf,
  writeEnuToScene,
} from "./coords";

// A made-up site: origin at 1,600 m ASL, site surface at 1,650 m ASL. Test-local numbers only.
const scene = { originElevM: 1600, refSurfaceElevM: 1650, verticalExaggeration: 1 };

describe("ENU → scene (docs/01 conventions)", () => {
  it("puts a known ENU point at the expected scene coordinate", () => {
    // 2.5 km east, 1.2 km north, 3.4 km below the origin.
    expect(enuToScene({ e: 2500, n: 1200, u: -3400 })).toEqual([2.5, -3.4, -1.2]);
  });

  it("maps east to +x, north to −z, up to +y (right-handed, y-up)", () => {
    expect(enuToScene({ e: 1000, n: 0, u: 0 })).toEqual([1, 0, -0]);
    expect(enuToScene({ e: 0, n: 1000, u: 0 })).toEqual([0, 0, -1]);
    expect(enuToScene({ e: 0, n: 0, u: 1000 })).toEqual([0, 1, -0]);
  });

  it("scales only the vertical by the exaggeration", () => {
    expect(enuToScene({ e: 2500, n: 1200, u: -3400 }, 2)).toEqual([2.5, -6.8, -1.2]);
  });

  it("writeEnuToScene writes the same numbers into a typed array at an offset", () => {
    const out = new Float32Array(6);
    writeEnuToScene({ e: 2500, n: 1200, u: -3400 }, 1, out, 3);
    expect(Array.from(out.slice(0, 3))).toEqual([0, 0, 0]);
    expect(out[3]).toBeCloseTo(2.5, 6);
    expect(out[4]).toBeCloseTo(-3.4, 6);
    expect(out[5]).toBeCloseTo(-1.2, 6);
  });

  it("sceneToEnu inverts enuToScene, with and without exaggeration", () => {
    for (const ve of [1, 2.5]) {
      const enu = { e: -8123.4, n: 4410.1, u: -2750.9 };
      const [x, y, z] = enuToScene(enu, ve);
      const back = sceneToEnu(x, y, z, ve);
      expect(back.e).toBeCloseTo(enu.e, 9);
      expect(back.n).toBeCloseTo(enu.n, 9);
      expect(back.u).toBeCloseTo(enu.u, 9);
    }
  });
});

describe("elevation and display depth", () => {
  it("elevMToSceneY uses u = elevM − originElevM", () => {
    expect(elevMToSceneY(1600, scene)).toBe(0);
    expect(elevMToSceneY(-1400, scene)).toBeCloseTo(-3, 12);
  });

  it("depth 0 sits at the site surface, not at the origin", () => {
    expect(depthKmToSceneY(0, scene)).toBeCloseTo(0.05, 12);
    expect(depthKmToSceneY(3, scene)).toBeCloseTo(-2.95, 12);
  });

  it("sceneYToDepthKm inverts depthKmToSceneY, including under exaggeration", () => {
    for (const ve of [1, 3]) {
      const s = { ...scene, verticalExaggeration: ve };
      for (const d of [0, 1, 2.75, 6]) expect(sceneYToDepthKm(depthKmToSceneY(d, s), s)).toBeCloseTo(d, 12);
    }
  });

  it("agrees with SeismicEvent.depthKm = (refSurfaceElevM − elevM) / 1000", () => {
    const elevM = -1234; // a hypocenter
    const depthKm = (scene.refSurfaceElevM - elevM) / 1000;
    expect(sceneYToDepthKm(elevMToSceneY(elevM, scene), scene)).toBeCloseTo(depthKm, 12);
  });
});

describe("verticalExaggerationOf", () => {
  it("defaults to 1.0 when the bundle omits it", () => {
    expect(verticalExaggerationOf({})).toBe(1);
    expect(verticalExaggerationOf({ verticalExaggeration: 2 })).toBe(2);
  });

  it("fails loudly on nonsense instead of flattening the scene", () => {
    expect(() => verticalExaggerationOf({ verticalExaggeration: 0 })).toThrow();
    expect(() => verticalExaggerationOf({ verticalExaggeration: -1 })).toThrow();
    expect(() => verticalExaggerationOf({ verticalExaggeration: Number.NaN })).toThrow();
    expect(() => verticalExaggerationOf({ verticalExaggeration: Infinity })).toThrow();
  });
});

describe("per-axis helpers (the single home of the ENU → scene mapping)", () => {
  it("agree with enuToScene axis by axis", () => {
    const enu = { e: -2750, n: 4120, u: -3310 };
    expect([eastMToSceneX(enu.e), upMToSceneY(enu.u, 1.5), northMToSceneZ(enu.n)]).toEqual(enuToScene(enu, 1.5));
  });

  it("invert cleanly", () => {
    for (const m of [-8000, -1, 0, 31.25, 12345]) {
      expect(sceneXToEastM(eastMToSceneX(m))).toBeCloseTo(m, 9);
      expect(sceneZToNorthM(northMToSceneZ(m))).toBeCloseTo(m, 9);
    }
    expect(northMToSceneZ(1000)).toBe(-1); // north is −z
  });
});

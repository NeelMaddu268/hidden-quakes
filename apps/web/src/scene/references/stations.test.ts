import { describe, expect, it } from "vitest";
import { elevMToSceneY } from "../coords";
import type { Station } from "../types";
import {
  buildStationGlyphs,
  sensorPosition,
  STATION_SHAPE,
  stationIssues,
  UNUSED_STATION_ALPHA,
  wellheadPosition,
} from "./stations";

const scene = { originElevM: 1627.7, verticalExaggeration: 1 };

/** Test-local station records (not real station codes). */
function station(over: Partial<Station> & Pick<Station, "id" | "kind">): Station {
  const surfaceElevM = over.surfaceElevM ?? 1650;
  const sensorDepthM = over.sensorDepthM ?? 0;
  const sensorElevM = over.sensorElevM ?? surfaceElevM - sensorDepthM;
  return {
    network: "XX",
    location: "",
    staticsS: {},
    station: over.id,
    latitude: 0,
    longitude: 0,
    channels: ["HHZ"],
    sampleRateHz: 100,
    preprocessProfile: "test",
    usedInRun: true,
    enu: { e: 1200, n: -3400, u: sensorElevM - scene.originElevM },
    ...over,
    surfaceElevM,
    sensorDepthM,
    sensorElevM,
  };
}

describe("borehole sensors (acceptance: markers sit at sensorElevM)", () => {
  const bh = station({ id: "BH1", kind: "borehole", surfaceElevM: 1702.5, sensorDepthM: 1000 });

  it("puts the marker at sensorElevM and the wellhead at surfaceElevM, both above enu e/n", () => {
    const sensor = sensorPosition(bh, scene);
    const head = wellheadPosition(bh, scene);
    expect(sensor).toEqual([1.2, elevMToSceneY(702.5, scene), 3.4]);
    expect(sensor[1]).toBe(elevMToSceneY(bh.sensorElevM, scene));
    expect(head).toEqual([1.2, elevMToSceneY(1702.5, scene), 3.4]);
    expect(head[1] - sensor[1]).toBeCloseTo(1.0, 9); // 1000 m of borehole = 1 km in the scene
  });

  it("scales the sensor depth with the vertical exaggeration", () => {
    const ve3 = { ...scene, verticalExaggeration: 3 };
    expect(wellheadPosition(bh, ve3)[1] - sensorPosition(bh, ve3)[1]).toBeCloseTo(3.0, 9);
  });

  it("uses sensorElevM even if a record's enu.u disagrees, and reports the disagreement", () => {
    const bad = { ...bh, enu: { ...bh.enu, u: 0 } };
    expect(sensorPosition(bad, scene)[1]).toBe(elevMToSceneY(bh.sensorElevM, scene));
    expect(stationIssues([bad], scene)[0]).toMatch(/BH1: enu\.u/);
    expect(stationIssues([bh], scene)).toEqual([]);
  });

  it("draws a marker at depth, a wellhead ring, and the line between them", () => {
    const g = buildStationGlyphs([bh], scene);
    expect(g.underground.count).toBe(1);
    expect(g.surface.count).toBe(1);
    expect([...g.underground.shapes]).toEqual([STATION_SHAPE.borehole]);
    expect([...g.surface.shapes]).toEqual([STATION_SHAPE.wellhead]);
    expect(g.underground.positions[1]).toBeCloseTo(elevMToSceneY(bh.sensorElevM, scene), 6);
    expect(g.surface.positions[1]).toBeCloseTo(elevMToSceneY(bh.surfaceElevM, scene), 6);
    expect(g.boreholeSegments).toEqual([wellheadPosition(bh, scene), sensorPosition(bh, scene)]);
  });
});

describe("buildStationGlyphs", () => {
  const stations = [
    station({ id: "S1", kind: "surface" }),
    station({ id: "SM", kind: "strong_motion", usedInRun: false }),
    station({ id: "B1", kind: "borehole", sensorDepthM: 500 }),
    station({ id: "B2", kind: "borehole", sensorDepthM: 1000 }),
  ];
  const g = buildStationGlyphs(stations, scene);

  it("draws one glyph per sensor plus one wellhead per borehole", () => {
    expect(g.surface.count).toBe(4);
    expect(g.underground.count).toBe(2);
    expect([...g.surface.shapes]).toEqual([0, 0, 2, 2]);
    expect([...g.underground.shapes]).toEqual([1, 1]);
    expect(g.boreholeSegments).toHaveLength(4);
  });

  it("puts surface sensors at their sensor elevation", () => {
    expect(g.surface.positions[1]).toBeCloseTo(elevMToSceneY(1650, scene), 6);
  });

  it("dims stations the run didn't use instead of hiding them", () => {
    expect([...g.surface.alphas]).toEqual([1, UNUSED_STATION_ALPHA, 1, 1].map((a) => Math.fround(a)));
    expect([...g.underground.alphas]).toEqual([1, 1]);
  });

  it("handles an empty station list", () => {
    const empty = buildStationGlyphs([], scene);
    expect(empty.surface.count).toBe(0);
    expect(empty.underground.count).toBe(0);
  });

  it("flags sensorElevM that doesn't equal surfaceElevM − sensorDepthM", () => {
    const off = { ...station({ id: "X", kind: "borehole", sensorDepthM: 1000 }), sensorElevM: 1000 };
    off.enu = { ...off.enu, u: off.sensorElevM - scene.originElevM };
    expect(stationIssues([off], scene)).toEqual([expect.stringMatching(/X: sensorElevM/)]);
  });
});

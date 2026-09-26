// Station glyph data: surface stations as small inverted triangles at their sensor; borehole sensors
// as a marker at their true depth (y from sensorElevM, docs/01) plus a thin line up to the wellhead
// (y from surfaceElevM). Horizontal position is the record's precomputed `enu`; never lat/lon.

import { elevMToSceneY, METERS_PER_UNIT } from "../coords";
import type { SceneMeta, Station } from "../types";

/** Glyph shapes the station shader draws (the `aShape` attribute). */
export const STATION_SHAPE = { surface: 0, borehole: 1, wellhead: 2 } as const;
export type StationShape = (typeof STATION_SHAPE)[keyof typeof STATION_SHAPE];

/** Opacity of stations the run didn't use (still drawn, so the geometry is honest). */
export const UNUSED_STATION_ALPHA = 0.35;

type Vec3 = [number, number, number];
type SceneScale = Pick<SceneMeta, "originElevM" | "verticalExaggeration">;

export function isBorehole(s: Pick<Station, "kind">): boolean {
  return s.kind === "borehole";
}

/** Scene position of a station's sensor: x/z from `enu`, y from `sensorElevM`. */
export function sensorPosition(s: Pick<Station, "enu" | "sensorElevM">, scene: SceneScale): Vec3 {
  return [s.enu.e / METERS_PER_UNIT, elevMToSceneY(s.sensorElevM, scene), -s.enu.n / METERS_PER_UNIT];
}

/** Scene position of a borehole's wellhead: straight above the sensor, y from `surfaceElevM`. */
export function wellheadPosition(s: Pick<Station, "enu" | "surfaceElevM">, scene: SceneScale): Vec3 {
  return [s.enu.e / METERS_PER_UNIT, elevMToSceneY(s.surfaceElevM, scene), -s.enu.n / METERS_PER_UNIT];
}

export interface StationGlyphs {
  count: number;
  /** xyz per glyph (sensors first, in station order, then one wellhead per borehole). */
  positions: Float32Array;
  shapes: Float32Array;
  alphas: Float32Array;
  /** Wellhead → sensor segments, one pair per borehole. */
  boreholeSegments: Vec3[];
}

export function buildStationGlyphs(stations: readonly Station[], scene: SceneScale): StationGlyphs {
  const boreholes = stations.filter(isBorehole);
  const count = stations.length + boreholes.length;
  const positions = new Float32Array(count * 3);
  const shapes = new Float32Array(count);
  const alphas = new Float32Array(count);
  const boreholeSegments: Vec3[] = [];
  let i = 0;
  const put = (p: Vec3, shape: StationShape, alpha: number) => {
    positions.set(p, i * 3);
    shapes[i] = shape;
    alphas[i] = alpha;
    i++;
  };
  for (const s of stations) {
    const alpha = s.usedInRun ? 1 : UNUSED_STATION_ALPHA;
    put(sensorPosition(s, scene), isBorehole(s) ? STATION_SHAPE.borehole : STATION_SHAPE.surface, alpha);
  }
  for (const s of boreholes) {
    const alpha = s.usedInRun ? 1 : UNUSED_STATION_ALPHA;
    const head = wellheadPosition(s, scene);
    put(head, STATION_SHAPE.wellhead, alpha);
    boreholeSegments.push(head, sensorPosition(s, scene));
  }
  return { count, positions, shapes, alphas, boreholeSegments };
}

/** Records whose `enu.u` disagrees with `sensorElevM − originElevM` (the sensor-position contract). */
export function stationIssues(
  stations: readonly Station[],
  scene: Pick<SceneMeta, "originElevM">,
  toleranceM = 1,
): string[] {
  const out: string[] = [];
  for (const s of stations) {
    const expectU = s.sensorElevM - scene.originElevM;
    if (Math.abs(s.enu.u - expectU) > toleranceM) {
      out.push(`${s.id}: enu.u ${s.enu.u.toFixed(1)} m but sensorElevM − originElevM = ${expectU.toFixed(1)} m`);
    }
    if (Math.abs(s.surfaceElevM - s.sensorDepthM - s.sensorElevM) > toleranceM) {
      out.push(`${s.id}: sensorElevM ${s.sensorElevM} != surfaceElevM ${s.surfaceElevM} − sensorDepthM ${s.sensorDepthM}`);
    }
  }
  return out;
}

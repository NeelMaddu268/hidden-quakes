// Station glyph data: surface stations as small inverted triangles at their sensor; borehole sensors
// as a marker at their true depth (y from sensorElevM, docs/01) plus a thin line up to the wellhead
// (y from surfaceElevM). Horizontal position is the record's precomputed `enu`; never lat/lon.
// Glyphs come in two groups because they draw on opposite sides of the terrain (terrain/renderOrder):
// borehole sensors are underground (covered by the terrain, fading in with the reveal); surface
// stations and wellheads sit on top of it.

import { eastMToSceneX, elevMToSceneY, northMToSceneZ } from "../coords";
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
  return [eastMToSceneX(s.enu.e), elevMToSceneY(s.sensorElevM, scene), northMToSceneZ(s.enu.n)];
}

/** Scene position of a borehole's wellhead: straight above the sensor, y from `surfaceElevM`. */
export function wellheadPosition(s: Pick<Station, "enu" | "surfaceElevM">, scene: SceneScale): Vec3 {
  return [eastMToSceneX(s.enu.e), elevMToSceneY(s.surfaceElevM, scene), northMToSceneZ(s.enu.n)];
}

/** One batch of point glyphs (one draw call). */
export interface GlyphBatch {
  count: number;
  /** xyz per glyph. */
  positions: Float32Array;
  shapes: Float32Array;
  alphas: Float32Array;
}

export interface StationGlyphs {
  /** Surface stations (and strong-motion sites) at their sensor, plus one wellhead ring per borehole. */
  surface: GlyphBatch;
  /** Borehole sensors at true depth. */
  underground: GlyphBatch;
  /** Wellhead → sensor segments, one pair per borehole (underground). */
  boreholeSegments: Vec3[];
}

interface Glyph {
  at: Vec3;
  shape: StationShape;
  alpha: number;
}

function batch(glyphs: readonly Glyph[]): GlyphBatch {
  const count = glyphs.length;
  const positions = new Float32Array(count * 3);
  const shapes = new Float32Array(count);
  const alphas = new Float32Array(count);
  glyphs.forEach((g, i) => {
    positions.set(g.at, i * 3);
    shapes[i] = g.shape;
    alphas[i] = g.alpha;
  });
  return { count, positions, shapes, alphas };
}

export function buildStationGlyphs(stations: readonly Station[], scene: SceneScale): StationGlyphs {
  const surface: Glyph[] = [];
  const underground: Glyph[] = [];
  const boreholeSegments: Vec3[] = [];
  for (const s of stations) {
    const alpha = s.usedInRun ? 1 : UNUSED_STATION_ALPHA;
    const sensor = sensorPosition(s, scene);
    if (isBorehole(s)) {
      const head = wellheadPosition(s, scene);
      underground.push({ at: sensor, shape: STATION_SHAPE.borehole, alpha });
      surface.push({ at: head, shape: STATION_SHAPE.wellhead, alpha });
      boreholeSegments.push(head, sensor);
    } else {
      surface.push({ at: sensor, shape: STATION_SHAPE.surface, alpha });
    }
  }
  return { surface: batch(surface), underground: batch(underground), boreholeSegments };
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

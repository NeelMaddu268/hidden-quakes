// WEB-07 projections. All coordinates remain in the bundle's UTM grid; no browser reprojection.
import type { SceneBounds } from "../camera/bounds";
import type { CatalogEvent, SceneMeta, SeismicEvent, Station } from "../types";

export const PLAN_MARGIN = 1.15;
export const MIN_PLAN_SPAN_KM = 6;
export const MIN_SECTION_SPAN_KM = 1;

export interface PlanFrame {
  centerX: number;
  centerZ: number;
  width: number;
  height: number;
  cameraY: number;
  far: number;
}

/**
 * Equal east/north scale. Depth only sets clipping. Pass untrimmed bounds as clipBounds when the
 * composition frames a subset: off-frame events may come into view after pan/zoom and must survive.
 */
export function planFrame(
  bounds: SceneBounds,
  aspect: number,
  clipBounds: Pick<SceneBounds, "min" | "max"> = bounds,
): PlanFrame {
  if (!(aspect > 0) || !Number.isFinite(aspect)) throw new Error("planFrame: invalid aspect");
  if (![...bounds.min, ...bounds.max, ...bounds.center, bounds.surfaceY, ...clipBounds.min, ...clipBounds.max].every(Number.isFinite)) {
    throw new Error("planFrame: nonfinite bounds");
  }
  const east = Math.max(MIN_PLAN_SPAN_KM, bounds.max[0] - bounds.min[0]) * PLAN_MARGIN;
  const north = Math.max(MIN_PLAN_SPAN_KM, bounds.max[2] - bounds.min[2]) * PLAN_MARGIN;
  const height = Math.max(north, east / aspect);
  const clearance = Math.max(east, north, MIN_PLAN_SPAN_KM);
  const cameraY = Math.max(bounds.surfaceY, bounds.max[1], clipBounds.max[1]) + clearance;
  return {
    centerX: bounds.center[0], centerZ: bounds.center[2],
    width: height * aspect, height, cameraY,
    far: Math.max(clearance * 2, cameraY - Math.min(bounds.min[1], clipBounds.min[1]) + clearance),
  };
}

/** Null/zero/invalid errors carry no extent, rather than an invented uncertainty. */
export function usableErrorKm(errorM: number | null | undefined): number {
  return errorM != null && Number.isFinite(errorM) && errorM > 0 ? errorM / 1000 : 0;
}

export interface SectionPoints {
  /** East km and depth below site surface km, interleaved; depth grows down. */
  xy: Float64Array;
  /** Horizontal and vertical error semi-axes, km; zero means unavailable. */
  errors: Float64Array;
}

type SectionEvent = Pick<SeismicEvent, "enu" | "elevM"> & {
  quality: Pick<SeismicEvent["quality"], "hErrM" | "vErrM">;
};
type SectionCatalog = Pick<CatalogEvent, "enu" | "elevM">;
type SectionStation = Pick<Station, "enu" | "sensorElevM" | "surfaceElevM">;

/** All input records in input order. Never reads published CatalogEvent.depthKm or ENU up as depth. */
export function sectionPoints(
  events: readonly SectionEvent[],
  catalog: readonly SectionCatalog[],
  stations: readonly SectionStation[],
  scene: Pick<SceneMeta, "refSurfaceElevM">,
): { candidates: SectionPoints; publicEvents: SectionPoints; sensors: Float64Array; wellheads: Float64Array } {
  const project = (records: readonly SectionCatalog[]): SectionPoints => {
    const xy = new Float64Array(records.length * 2);
    for (let i = 0; i < records.length; i++) {
      xy[i * 2] = records[i].enu.e / 1000;
      xy[i * 2 + 1] = (scene.refSurfaceElevM - records[i].elevM) / 1000;
      if (!Number.isFinite(xy[i * 2]) || !Number.isFinite(xy[i * 2 + 1])) {
        throw new Error(`sectionPoints: nonfinite position at record ${i}`);
      }
    }
    return { xy, errors: new Float64Array(xy.length) };
  };
  const candidates = project(events);
  for (let i = 0; i < events.length; i++) {
    candidates.errors[i * 2] = usableErrorKm(events[i].quality.hErrM);
    candidates.errors[i * 2 + 1] = usableErrorKm(events[i].quality.vErrM);
  }
  const sensors = project(stations.map((s) => ({ enu: s.enu, elevM: s.sensorElevM }))).xy;
  const wellheads = project(stations.map((s) => ({ enu: s.enu, elevM: s.surfaceElevM }))).xy;
  return { candidates, publicEvents: project(catalog), sensors, wellheads };
}

export interface SectionFit {
  pxPerKm: number;
  offsetX: number;
  offsetY: number;
  minEast: number;
  maxEast: number;
  minDepth: number;
  maxDepth: number;
}

/** True-scale full-population projection, including errors and both ends of boreholes; no trimming. */
export function sectionFit(
  layers: readonly SectionPoints[],
  stations: readonly Float64Array[],
  width: number,
  height: number,
  pad: number,
): SectionFit {
  if (![width, height, pad].every(Number.isFinite) || pad < 0 || width <= 2 * pad || height <= 2 * pad) {
    throw new Error("sectionFit: viewport must have a positive drawing area");
  }
  let minEast = Infinity, maxEast = -Infinity, minDepth = 0, maxDepth = 0;
  const include = (xy: Float64Array, errors?: Float64Array) => {
    if (xy.length % 2 || (errors && errors.length !== xy.length)) throw new Error("sectionFit: unpaired coordinates");
    for (let i = 0; i < xy.length; i += 2) {
      const e = xy[i], d = xy[i + 1], h = errors?.[i] ?? 0, v = errors?.[i + 1] ?? 0;
      if (![e, d, h, v].every(Number.isFinite) || h < 0 || v < 0) throw new Error("sectionFit: invalid extent");
      minEast = Math.min(minEast, e - h); maxEast = Math.max(maxEast, e + h);
      minDepth = Math.min(minDepth, d - v); maxDepth = Math.max(maxDepth, d + v);
    }
  };
  for (const l of layers) include(l.xy, l.errors);
  for (const s of stations) include(s);
  if (!Number.isFinite(minEast)) { minEast = -MIN_SECTION_SPAN_KM / 2; maxEast = MIN_SECTION_SPAN_KM / 2; }
  const spanE = Math.max(MIN_SECTION_SPAN_KM, maxEast - minEast);
  const spanD = Math.max(MIN_SECTION_SPAN_KM, maxDepth - minDepth);
  const pxPerKm = Math.min((width - 2 * pad) / spanE, (height - 2 * pad) / spanD);
  const offsetX = width / 2 - ((minEast + maxEast) / 2) * pxPerKm;
  const offsetY = height / 2 - ((minDepth + maxDepth) / 2) * pxPerKm;
  return { pxPerKm, offsetX, offsetY, minEast, maxEast, minDepth, maxDepth };
}

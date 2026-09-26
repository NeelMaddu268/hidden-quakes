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

/** Margin around the framed structure, as a fraction of its larger span (and at least this many km). */
export const SECTION_FRAME_MARGIN = 0.12;
export const MIN_SECTION_MARGIN_KM = 0.25;

function quantile(sorted: Float64Array, q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/**
 * True-scale fit framed on the structure (WEB-08), the way the plan camera frames it: the given points
 * (Tier A and B candidates), trimmed by `trim` per axis at each end like the camera's framing, always
 * reaching up to the site surface (depth 0), plus a margin. Equal scale on both axes, so one axis
 * shows more than the structure's extent. Events outside the resulting frame are clipped by the
 * drawing and counted (sectionOutside), never silently dropped. With no points it falls back to
 * `fallback` (every event), then to a MIN_SECTION_SPAN_KM square under the origin.
 */
export function sectionStructureFit(
  framed: Float64Array,
  fallback: Float64Array,
  width: number,
  height: number,
  pad: number,
  trim: number,
): SectionFit {
  if (![width, height, pad].every(Number.isFinite) || pad < 0 || width <= 2 * pad || height <= 2 * pad) {
    throw new Error("sectionStructureFit: viewport must have a positive drawing area");
  }
  const xy = framed.length >= 2 ? framed : fallback;
  if (xy.length % 2) throw new Error("sectionStructureFit: unpaired coordinates");
  const n = xy.length / 2;
  let minEast = -MIN_SECTION_SPAN_KM / 2, maxEast = MIN_SECTION_SPAN_KM / 2, minDepth = 0, maxDepth = MIN_SECTION_SPAN_KM;
  if (n > 0) {
    const east = new Float64Array(n);
    const depth = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      east[i] = xy[i * 2];
      depth[i] = xy[i * 2 + 1];
      if (!Number.isFinite(east[i]) || !Number.isFinite(depth[i])) throw new Error("sectionStructureFit: invalid point");
    }
    east.sort();
    depth.sort();
    const q = n >= 1 / Math.max(trim, 1e-9) ? trim : 0;
    minEast = quantile(east, q);
    maxEast = quantile(east, 1 - q);
    minDepth = Math.min(0, quantile(depth, q)); // the site surface is always in frame
    maxDepth = quantile(depth, 1 - q);
  }
  const margin = Math.max(MIN_SECTION_MARGIN_KM, SECTION_FRAME_MARGIN * Math.max(maxEast - minEast, maxDepth - minDepth));
  minEast -= margin;
  maxEast += margin;
  minDepth -= margin / 2; // room above the surface for station glyphs
  maxDepth += margin;
  const spanE = Math.max(MIN_SECTION_SPAN_KM, maxEast - minEast);
  const spanD = Math.max(MIN_SECTION_SPAN_KM, maxDepth - minDepth);
  const pxPerKm = Math.min((width - 2 * pad) / spanE, (height - 2 * pad) / spanD);
  const offsetX = width / 2 - ((minEast + maxEast) / 2) * pxPerKm;
  const offsetY = height / 2 - ((minDepth + maxDepth) / 2) * pxPerKm;
  return { pxPerKm, offsetX, offsetY, minEast, maxEast, minDepth, maxDepth };
}

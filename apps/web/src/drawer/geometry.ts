// Geometry for the drawer's two small figures: a plan-view mini map (north up, east right, equal
// scale) and a depth section (east vs depth below the site surface). Positions come from the bundle's
// precomputed ENU and elevations; nothing is reprojected from lat/lon. Pure functions.

import type { Enu, SceneMeta, SeismicEvent, Station, WaveformSnippet } from "../scene/types";
import { isNum } from "./format";

export interface Box {
  width: number;
  height: number;
  pad: number;
}

/** A round length (1, 2, 2.5 or 5 × 10^k) no longer than `maxLen`. */
export function roundLengthAtMost(maxLen: number): number {
  if (!(maxLen > 0) || !Number.isFinite(maxLen)) return 0;
  const pow = Math.pow(10, Math.floor(Math.log10(maxLen)));
  for (const m of [5, 2.5, 2, 1]) if (m * pow <= maxLen * (1 + 1e-9)) return m * pow;
  return pow;
}

/**
 * Stations that picked the event, from its pick ids, deduplicated in pick order: the drawer's station
 * geometry when the event has no evidence file (the exporter caps evidence at `maxEvents`). The contract
 * doesn't fix a pick id's format, so the station is the one colon-separated part that is a bundle station
 * id (real ids read `<picker>:<instance>:<NET.STA>:<phase>:<time>`); a pick with no such part is skipped.
 */
export function pickingStations(pickIds: readonly string[], stationsById: ReadonlyMap<string, Station>): Station[] {
  const seen = new Set<string>();
  const out: Station[] = [];
  for (const id of pickIds) {
    const st = id
      .split(":")
      .map((part) => stationsById.get(part))
      .find((s): s is Station => s !== undefined);
    if (!st || seen.has(st.id) || !isNum(st.enu?.e) || !isNum(st.enu?.n)) continue;
    seen.add(st.id);
    out.push(st);
  }
  return out;
}

/** Stations of the evidence traces, in trace order, deduplicated; ids missing from the bundle are listed. */
export function evidenceStations(
  traces: readonly WaveformSnippet[],
  stationsById: ReadonlyMap<string, Station>,
): { stations: Station[]; missing: string[] } {
  const seen = new Set<string>();
  const stations: Station[] = [];
  const missing: string[] = [];
  for (const tr of traces) {
    if (seen.has(tr.stationId)) continue;
    seen.add(tr.stationId);
    const st = stationsById.get(tr.stationId);
    if (st && isNum(st.enu?.e) && isNum(st.enu?.n)) stations.push(st);
    else missing.push(tr.stationId);
  }
  return { stations, missing };
}

// ---- Mini map ---------------------------------------------------------------------------------

export interface MapGeometry {
  /** SVG x of an east offset (m). */
  x: (e: number) => number;
  /** SVG y of a north offset (m); north is up. */
  y: (n: number) => number;
  /** Pixels (SVG units) per meter, same on both axes. */
  pxPerM: number;
  /** Scale bar length in meters (a round number), and its length in SVG units. */
  scaleBarM: number;
  scaleBarPx: number;
}

/** Smallest map extent (m), so a lone epicenter doesn't blow up to fill the box. */
export const MIN_MAP_EXTENT_M = 1000;

/**
 * Fits the epicenter, the stations and the horizontal error circle into the box at one scale, centered.
 * The scale bar is the longest round length up to ~40% of the box width.
 */
export function mapGeometry(epicenter: Enu, points: readonly Enu[], hErrM: number | null | undefined, box: Box): MapGeometry {
  const r = isNum(hErrM) && hErrM > 0 ? hErrM : 0;
  let minE = epicenter.e - r;
  let maxE = epicenter.e + r;
  let minN = epicenter.n - r;
  let maxN = epicenter.n + r;
  for (const p of points) {
    minE = Math.min(minE, p.e);
    maxE = Math.max(maxE, p.e);
    minN = Math.min(minN, p.n);
    maxN = Math.max(maxN, p.n);
  }
  const spanE = Math.max(maxE - minE, MIN_MAP_EXTENT_M);
  const spanN = Math.max(maxN - minN, MIN_MAP_EXTENT_M);
  const innerW = box.width - 2 * box.pad;
  const innerH = box.height - 2 * box.pad;
  const pxPerM = Math.min(innerW / spanE, innerH / spanN);
  const cE = (minE + maxE) / 2;
  const cN = (minN + maxN) / 2;
  const scaleBarM = roundLengthAtMost((0.4 * innerW) / pxPerM);
  return {
    x: (e) => box.width / 2 + (e - cE) * pxPerM,
    y: (n) => box.height / 2 - (n - cN) * pxPerM,
    pxPerM,
    scaleBarM,
    scaleBarPx: scaleBarM * pxPerM,
  };
}

// ---- Depth section ----------------------------------------------------------------------------

/** Display depth (km below the site surface) of an elevation (m ASL), as in docs/01. */
export function depthKmOfElev(elevM: number, scene: Pick<SceneMeta, "refSurfaceElevM">): number {
  return (scene.refSurfaceElevM - elevM) / 1000;
}

export interface DepthGeometry {
  /** SVG x of an east offset (m). */
  x: (e: number) => number;
  /** SVG y of a depth below the site surface (km); deeper is lower. */
  y: (depthKm: number) => number;
  /** Vertical scale ÷ horizontal scale; 1 is true scale. Anything else is labeled "Vertical ×N". */
  verticalExaggeration: number;
  /** Depth ticks (km) and the depth range drawn. */
  ticks: number[];
  top: number;
  bottom: number;
}

/**
 * Side view looking north: east across, depth down. True scale when the depth range fits; otherwise
 * the depth axis is stretched by a round factor (1-2-5 series) and the figure says so.
 */
export function depthGeometry(
  points: readonly { e: number; depthKm: number }[],
  eventErrKm: number,
  box: Box,
  maxTicks = 4,
): DepthGeometry {
  let minE = Infinity;
  let maxE = -Infinity;
  let top = 0; // the site surface is always in view
  let bottom = 0;
  for (const p of points) {
    minE = Math.min(minE, p.e);
    maxE = Math.max(maxE, p.e);
    top = Math.min(top, p.depthKm);
    bottom = Math.max(bottom, p.depthKm);
  }
  if (!isNum(minE)) {
    minE = -MIN_MAP_EXTENT_M / 2;
    maxE = MIN_MAP_EXTENT_M / 2;
  }
  // The event's error bar is the deepest/shallowest thing drawn around it; points[0] is the event.
  if (points.length && eventErrKm > 0) {
    top = Math.min(top, points[0].depthKm - eventErrKm);
    bottom = Math.max(bottom, points[0].depthKm + eventErrKm);
  }
  const depthSpan = Math.max(bottom - top, 0.5);
  bottom = top + depthSpan * 1.08; // a little room below the deepest mark
  const spanE = Math.max(maxE - minE, MIN_MAP_EXTENT_M);
  const innerW = box.width - 2 * box.pad;
  const innerH = box.height - 2 * box.pad;
  const pxPerM = innerW / spanE;
  const vPxPerKmTrue = pxPerM * 1000;
  const vPxPerKmFit = innerH / (bottom - top);
  let hPxPerM = pxPerM;
  let vPxPerKm = vPxPerKmFit;
  let ve = 1;
  if (vPxPerKmFit <= vPxPerKmTrue) {
    // Depth range is the tall side: true scale, narrow the horizontal instead.
    hPxPerM = vPxPerKmFit / 1000;
  } else {
    ve = floorNice(vPxPerKmFit / vPxPerKmTrue);
    vPxPerKm = vPxPerKmTrue * ve;
  }
  const cE = (minE + maxE) / 2;
  const usedH = (bottom - top) * vPxPerKm;
  const y0 = box.pad + (innerH - usedH) / 2;
  const step = niceKmStep(bottom - top, maxTicks);
  const ticks: number[] = [];
  for (let k = Math.ceil(top / step - 1e-9); k * step <= bottom + 1e-9; k++) ticks.push(Number((k * step).toFixed(6)));
  return {
    x: (e) => box.width / 2 + (e - cE) * hPxPerM,
    y: (d) => y0 + (d - top) * vPxPerKm,
    verticalExaggeration: ve,
    ticks,
    top,
    bottom,
  };
}

/** Largest 1-2-5 × 10^k value ≤ v (v ≥ 1). */
function floorNice(v: number): number {
  if (!(v >= 1)) return 1;
  const pow = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [5, 2, 1]) if (m * pow <= v * (1 + 1e-9)) return m * pow;
  return pow;
}

function niceKmStep(span: number, maxTicks: number): number {
  const raw = span / Math.max(1, maxTicks);
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (m * pow >= raw) return m * pow;
  return 10 * pow;
}

/** Event position for the depth section: east offset and display depth from the shared site datum. */
export function eventDepthPoint(ev: Pick<SeismicEvent, "enu" | "elevM">, scene: Pick<SceneMeta, "refSurfaceElevM">): { e: number; depthKm: number } {
  return { e: ev.enu.e, depthKm: depthKmOfElev(ev.elevM, scene) };
}

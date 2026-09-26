// The plan view's depth section (WEB-07): every event projected onto grid east versus depth below the
// site surface, drawn with Canvas2D. Model arrays are built once per bundle; drawing reads the shared
// reveal clock, time-mode "now" (WEB-06), eased filter look and selection each frame and allocates
// nothing of its own. Depth is
// always (refSurfaceElevM − elevM) / 1000 (sectionPoints); published catalog depths are never read.

import { tierStyle } from "@hq/visualization";
import type { DemoPhase, EventFilter } from "../../state/demo";
import { candidateRevealUniform } from "../events/driver";
import { buildCandidateInstances, TIER_INDEX } from "../events/instances";
import type { FilterLook } from "../filters/fade";
import { candidatePickable, publicSelectTargets } from "../picking/selection";
import { shownAt, TIME_ALL } from "../time/clock";
import type { CatalogEvent, SceneMeta, SeismicEvent, Station } from "../types";
import { sectionFit, sectionPoints, type SectionFit, type SectionPoints } from "./geometry";

export interface SectionModel {
  candidates: SectionPoints;
  publicEvents: SectionPoints;
  sensors: Float64Array;
  wellheads: Float64Array;
  /** 1 for a borehole sensor, 0 for a surface or strong-motion station (drawn at its own elevation). */
  borehole: Uint8Array;
  candidateIds: readonly string[];
  candidateTier: Float32Array;
  candidateAppearAt: Float32Array;
  /** Origin times, seconds since windowStart (time mode), for candidates and public events. */
  candidateTime: Float32Array;
  publicTime: Float32Array;
  /** For each public event, the candidate id clicking it selects (its matched event), or null. */
  publicTargets: readonly (string | null)[];
  indexById: ReadonlyMap<string, number>;
}

/** Built once per bundle. Uses the same instance builder as the 3D layers, so appearance times match. */
export function buildSectionModel(
  events: readonly SeismicEvent[],
  catalog: readonly CatalogEvent[],
  stations: readonly Station[],
  scene: Pick<SceneMeta, "refSurfaceElevM">,
  windowStart: number,
): SectionModel {
  const points = sectionPoints(events, catalog, stations, scene);
  const inst = buildCandidateInstances(events, 1, windowStart);
  const publicTime = new Float32Array(catalog.length);
  for (let i = 0; i < catalog.length; i++) publicTime[i] = catalog[i].t - windowStart;
  const borehole = new Uint8Array(stations.length);
  for (let i = 0; i < stations.length; i++) borehole[i] = stations[i].kind === "borehole" ? 1 : 0;
  return {
    candidates: points.candidates,
    publicEvents: points.publicEvents,
    sensors: points.sensors,
    wellheads: points.wellheads,
    borehole,
    candidateIds: inst.ids,
    candidateTier: inst.tiers,
    candidateAppearAt: inst.appearAt,
    candidateTime: inst.times,
    publicTime,
    publicTargets: publicSelectTargets(catalog, inst.indexById),
    indexById: inst.indexById,
  };
}

/** Plot area inside the panel canvas (CSS px): axis labels need room on the left and bottom. */
export const SECTION_PAD = Object.freeze({ left: 46, right: 12, top: 10, bottom: 30, inner: 6 });

export interface Tick {
  /** Canvas position along the axis, CSS px. */
  at: number;
  label: string;
}

export interface SectionPlot {
  fit: SectionFit;
  /** Plot area origin and size, CSS px. */
  x0: number;
  y0: number;
  w: number;
  h: number;
  /** Canvas px of east 0 / depth 0, and px per km (equal on both axes: true scale). */
  ox: number;
  oy: number;
  k: number;
  /** Axis ticks, computed once per layout (labels are derived numbers, never copy). */
  depthTicks: readonly Tick[];
  eastTicks: readonly Tick[];
}

const fmtKm = (v: number) => `${+v.toFixed(3)}`;

/** True-scale fit of the whole population (no trimming) into the plot area of a `width × height` canvas. */
export function sectionPlot(model: SectionModel, width: number, height: number): SectionPlot {
  const P = SECTION_PAD;
  const w = width - P.left - P.right;
  const h = height - P.top - P.bottom;
  const fit = sectionFit([model.candidates, model.publicEvents], [model.sensors, model.wellheads], w, h, P.inner);
  const ox = P.left + fit.offsetX;
  const oy = P.top + fit.offsetY;
  const k = fit.pxPerKm;
  const depthTicks: Tick[] = [];
  const dStep = niceStep(fit.maxDepth - fit.minDepth || 1, 5);
  for (let d = Math.ceil(fit.minDepth / dStep) * dStep; d <= fit.maxDepth + 1e-9; d += dStep) {
    const at = oy + d * k;
    if (at >= P.top - 0.5 && at <= P.top + h + 0.5) depthTicks.push({ at, label: fmtKm(d) });
  }
  const eastTicks: Tick[] = [];
  const eStep = niceStep(fit.maxEast - fit.minEast || 1, 6);
  for (let e = Math.ceil(fit.minEast / eStep) * eStep; e <= fit.maxEast + 1e-9; e += eStep) {
    const at = ox + e * k;
    if (at >= P.left - 0.5 && at <= P.left + w + 0.5) eastTicks.push({ at, label: fmtKm(e) });
  }
  return { fit, x0: P.left, y0: P.top, w, h, ox, oy, k, depthTicks, eastTicks };
}

/** A "nice" tick step (1, 2 or 5 × 10^n km) giving about `target` ticks over `span` km. */
export function niceStep(span: number, target = 5): number {
  if (!(span > 0) || !Number.isFinite(span)) return 1;
  const raw = span / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const n = raw / mag;
  return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag;
}

/** Minimal 2D-context surface the drawing uses (so tests can record calls without a DOM canvas). */
export interface Ctx2D {
  globalAlpha: number;
  /** Written with token colors only; typed loosely so a real CanvasRenderingContext2D satisfies it. */
  fillStyle: unknown;
  strokeStyle: unknown;
  lineWidth: number;
  font: string;
  textAlign: string;
  textBaseline: string;
  setTransform(a: number, b: number, c: number, d: number, e: number, f: number): void;
  clearRect(x: number, y: number, w: number, h: number): void;
  beginPath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  arc(x: number, y: number, r: number, a0: number, a1: number): void;
  closePath(): void;
  fill(): void;
  stroke(): void;
  fillText(text: string, x: number, y: number): void;
  setLineDash(segments: number[]): void;
}

export interface SectionState {
  phase: DemoPhase;
  filter: EventFilter;
  revealElapsedS: number;
  /** Time mode "now", seconds since windowStart (scene/time/clock → timeNowRel); TIME_ALL when off. */
  timeNowRel: number;
  look: Readonly<FilterLook>;
  selectedIndex: number;
}

export interface SectionStyle {
  recovered: string;
  publicDot: string;
  station: string;
  contour: string;
  textDim: string;
  halo: string;
  /** Full CSS font for tick labels, e.g. "10px <mono stack>" (built once, not per frame). */
  tickFont: string;
  dpr: number;
}

/** Glyph radii in CSS px; tier carries size as in 3D (tokens.tierStyle). */
export const SECTION_GLYPH = Object.freeze({ candidatePx: 2.4, publicPx: 3, selectedRingPx: 6, markerPx: 3.5 });

const TAU = Math.PI * 2;
const DASH: number[] = [3, 3];
const SOLID: number[] = [];

const tierOpacity = (look: Readonly<FilterLook>, tier: number) =>
  tier === TIER_INDEX.A ? look.tierA : tier === TIER_INDEX.B ? look.tierB : look.tierC;

const tierSize = (tier: number) =>
  tier === TIER_INDEX.A ? tierStyle.A.size : tier === TIER_INDEX.B ? tierStyle.B.size : tierStyle.C.size;

/** Draws one frame. Reads the model and state; allocates nothing of its own (the two dash arrays are shared). */
export function drawSection(
  ctx: Ctx2D,
  model: SectionModel,
  plot: SectionPlot,
  state: SectionState,
  style: SectionStyle,
  cssWidth: number,
  cssHeight: number,
): void {
  const { x0, y0, w, h, ox, oy, k } = plot;

  ctx.setTransform(style.dpr, 0, 0, style.dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.globalAlpha = 1;

  // Axes: depth ticks on the left (0 = site surface), grid-east ticks along the bottom.
  ctx.font = style.tickFont;
  ctx.fillStyle = style.textDim;
  ctx.strokeStyle = style.contour;
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i < plot.depthTicks.length; i++) {
    const t = plot.depthTicks[i];
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.moveTo(x0, t.at);
    ctx.lineTo(x0 + w, t.at);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillText(t.label, x0 - 6, t.at);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i < plot.eastTicks.length; i++) {
    const t = plot.eastTicks[i];
    ctx.fillText(t.label, t.at, y0 + h + 6);
  }

  // Site surface (depth 0), dashed.
  ctx.setLineDash(DASH);
  ctx.globalAlpha = 0.9;
  ctx.beginPath();
  ctx.moveTo(x0, oy);
  ctx.lineTo(x0 + w, oy);
  ctx.stroke();
  ctx.setLineDash(SOLID);

  // Stations: borehole sensors at their true depth with a line to the wellhead; surface stations as
  // small inverted triangles at their own elevation.
  ctx.strokeStyle = style.station;
  ctx.fillStyle = style.station;
  ctx.globalAlpha = 0.9;
  const n = model.borehole.length;
  const M = SECTION_GLYPH.markerPx;
  for (let i = 0; i < n; i++) {
    const sx = ox + model.sensors[i * 2] * k;
    const sy = oy + model.sensors[i * 2 + 1] * k;
    if (model.borehole[i]) {
      ctx.beginPath();
      ctx.moveTo(ox + model.wellheads[i * 2] * k, oy + model.wellheads[i * 2 + 1] * k);
      ctx.lineTo(sx, sy);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(sx, sy - M);
      ctx.lineTo(sx + M, sy);
      ctx.lineTo(sx, sy + M);
      ctx.lineTo(sx - M, sy);
      ctx.closePath();
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.moveTo(sx - M, sy - M);
      ctx.lineTo(sx + M, sy - M);
      ctx.lineTo(sx, sy + M * 0.4);
      ctx.closePath();
      ctx.fill();
    }
  }

  // Public regional catalog: visible from the first frame at the layer's eased weight (in time mode,
  // once tNow reaches each event).
  const pub = model.publicEvents.xy;
  const now = state.timeNowRel;
  if (state.look.publicLayer > 0.001) {
    ctx.globalAlpha = state.look.publicLayer;
    ctx.fillStyle = style.publicDot;
    ctx.beginPath();
    for (let i = 0; i < pub.length; i += 2) {
      if (!shownAt(model.publicTime[i >> 1], now)) continue;
      const x = ox + pub[i] * k;
      const y = oy + pub[i + 1] * k;
      ctx.moveTo(x + SECTION_GLYPH.publicPx, y);
      ctx.arc(x, y, SECTION_GLYPH.publicPx, 0, TAU);
    }
    ctx.fill();
  }

  // Candidates, one batch per tier (tier sets opacity and size, as in 3D). Only appeared instances.
  const clock = candidateRevealUniform(state.phase, state.revealElapsedS);
  const cand = model.candidates.xy;
  const count = model.candidateIds.length;
  if (state.look.candidates > 0.001 && state.phase !== "public") {
    ctx.fillStyle = style.recovered;
    for (let tier = 2; tier >= 0; tier--) {
      const alpha = state.look.candidates * tierOpacity(state.look, tier);
      if (alpha <= 0.001) continue;
      const r = SECTION_GLYPH.candidatePx * tierSize(tier);
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      for (let i = 0; i < count; i++) {
        if (model.candidateTier[i] !== tier || model.candidateAppearAt[i] > clock) continue;
        if (!shownAt(model.candidateTime[i], now)) continue;
        const x = ox + cand[i * 2] * k;
        const y = oy + cand[i * 2 + 1] * k;
        ctx.moveTo(x + r, y);
        ctx.arc(x, y, r, 0, TAU);
      }
      ctx.fill();
    }
    // Uncertainty crosses (±hErrM along east, ±vErrM in depth) for Tier A under STRICT; a missing
    // error draws no arm (never an invented one).
    if (state.look.halos > 0.001) {
      const err = model.candidates.errors;
      ctx.strokeStyle = style.halo;
      ctx.globalAlpha = 0.55 * state.look.halos;
      ctx.beginPath();
      for (let i = 0; i < count; i++) {
        if (model.candidateTier[i] !== TIER_INDEX.A || model.candidateAppearAt[i] > clock) continue;
        if (!shownAt(model.candidateTime[i], now)) continue;
        const x = ox + cand[i * 2] * k;
        const y = oy + cand[i * 2 + 1] * k;
        const hPx = err[i * 2] * k;
        const vPx = err[i * 2 + 1] * k;
        if (hPx > 0) {
          ctx.moveTo(x - hPx, y);
          ctx.lineTo(x + hPx, y);
        }
        if (vPx > 0) {
          ctx.moveTo(x, y - vPx);
          ctx.lineTo(x, y + vPx);
        }
      }
      ctx.stroke();
    }
  }

  // Selection ring (the drawer's event), shown whenever the event itself is drawn.
  const s = state.selectedIndex;
  if (s >= 0 && s < count && state.phase !== "public" && model.candidateAppearAt[s] <= clock && shownAt(model.candidateTime[s], now)) {
    ctx.globalAlpha = 1;
    ctx.strokeStyle = style.halo;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const x = ox + cand[s * 2] * k;
    const y = oy + cand[s * 2 + 1] * k;
    ctx.moveTo(x + SECTION_GLYPH.selectedRingPx, y);
    ctx.arc(x, y, SECTION_GLYPH.selectedRingPx, 0, TAU);
    ctx.stroke();
    ctx.lineWidth = 1;
  }
  ctx.globalAlpha = 1;
}

/**
 * The candidate id a click at (x, y) (CSS px in the canvas) selects: the nearest candidate the 3D
 * picker would also allow (same phase/filter/reveal gates), or a public event's matched candidate;
 * null for empty space. Ties go to the lower index so results never depend on draw order.
 */
export function sectionHit(
  model: SectionModel,
  plot: SectionPlot,
  state: Pick<SectionState, "phase" | "filter" | "revealElapsedS"> & Partial<Pick<SectionState, "timeNowRel">>,
  x: number,
  y: number,
  thresholdPx: number,
): string | null {
  const { ox, oy, k } = plot;
  const now = state.timeNowRel ?? TIME_ALL;
  let best: string | null = null;
  let bestD = thresholdPx * thresholdPx;
  const cand = model.candidates.xy;
  for (let i = 0; i < model.candidateIds.length; i++) {
    if (!candidatePickable(state.phase, state.filter, state.revealElapsedS, model.candidateTier[i], model.candidateAppearAt[i])) continue;
    if (!shownAt(model.candidateTime[i], now)) continue;
    const dx = ox + cand[i * 2] * k - x;
    const dy = oy + cand[i * 2 + 1] * k - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = model.candidateIds[i];
    }
  }
  const pub = model.publicEvents.xy;
  for (let i = 0; i < model.publicTargets.length; i++) {
    const target = model.publicTargets[i];
    if (!target || !shownAt(model.publicTime[i], now)) continue;
    const dx = ox + pub[i * 2] * k - x;
    const dy = oy + pub[i * 2 + 1] * k - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = target;
    }
  }
  return best;
}

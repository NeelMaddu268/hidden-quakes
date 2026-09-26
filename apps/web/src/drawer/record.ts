// Record-section math for the evidence drawer: which traces can be drawn, the shared time axis
// (seconds after the origin time), per-trace normalization, SVG polyline points and the positions of
// pick and predicted-arrival marks. Pure functions; every output is finite or explicitly null.

import { motion } from "@hq/visualization";
import type { WaveformSnippet } from "../scene/types";
import { isNum } from "./format";

/** SVG viewBox width of one trace row. Rows stretch horizontally (preserveAspectRatio="none"). */
export const TRACK_W = 1000;
/** SVG viewBox height of one trace row. */
export const TRACK_H = 100;
/** Peak excursion of a normalized trace, as a fraction of the row height either side of center. */
export const TRACE_AMPLITUDE = 0.46;

/** Why a trace can't be drawn, or null when it can. */
export function traceProblem(tr: WaveformSnippet): string | null {
  if (!isNum(tr.t0)) return "t0 is not a number";
  if (!isNum(tr.dt) || tr.dt <= 0) return "dt is not a positive number";
  if (!Array.isArray(tr.samples) || tr.samples.length < 2) return "fewer than 2 samples";
  if (tr.samples.some(s => !isNum(s))) return "non-finite waveform sample";
  if (!isNum(tr.epiDistM)) return "epiDistM is not a number";
  return null;
}

/**
 * What the drawer says when an event's evidence can't be shown: never the raw provider message (a URL
 * and status code). A 404 is expected (the exporter writes evidence for the first `maxEvents` events in
 * reveal order only), so it reads as a neutral note; anything else is a real load failure.
 */
export function evidenceUnavailable(message: string | undefined): { text: string; expected: boolean } {
  return /\bHTTP 404\b/.test(message ?? "")
    ? { text: "No waveform evidence was exported for this event.", expected: true }
    : { text: "Waveform evidence could not be loaded.", expected: false };
}

/** How many traces carry a P or S pick (the rest are the closest stations' records without one). */
export function pickedTraceCount(traces: readonly { pickP?: number | null; pickS?: number | null }[]): number {
  let n = 0;
  for (const t of traces) if (t.pickP != null || t.pickS != null) n++;
  return n;
}

export interface PreparedTraces {
  /** Drawable traces, sorted by epicentral distance (stable), as the contract orders them. */
  traces: WaveformSnippet[];
  /** One line per trace that was left out, e.g. "XX.S03 HHZ: dt is not a positive number". */
  skipped: string[];
}

/** Drops traces that can't be drawn (reported, never silently) and sorts the rest by epiDistM. */
export function prepareTraces(traces: readonly WaveformSnippet[]): PreparedTraces {
  const ok: { tr: WaveformSnippet; i: number }[] = [];
  const skipped: string[] = [];
  traces.forEach((tr, i) => {
    const problem = traceProblem(tr);
    if (problem) skipped.push(`${tr.stationId ?? "?"} ${tr.channel ?? ""}: ${problem}`.replace(/\s+:/, ":"));
    else ok.push({ tr, i });
  });
  ok.sort((a, b) => a.tr.epiDistM - b.tr.epiDistM || a.i - b.i);
  return { traces: ok.map((o) => o.tr), skipped };
}

/** Seconds after the origin time, [start, end]. */
export interface TimeDomain {
  start: number;
  end: number;
}

/** Time of the last sample of a trace, epoch s. */
export function traceEnd(tr: WaveformSnippet): number {
  return tr.t0 + (tr.samples.length - 1) * tr.dt;
}

/** The union of every trace's window, relative to the origin time; null with no drawable trace. */
export function recordDomain(traces: readonly WaveformSnippet[], originT: number): TimeDomain | null {
  let start = Infinity;
  let end = -Infinity;
  for (const tr of traces) {
    start = Math.min(start, tr.t0 - originT);
    end = Math.max(end, traceEnd(tr) - originT);
  }
  if (!isNum(start) || !isNum(end) || !(end > start)) return null;
  return { start, end };
}

/** A 1-2-5 × 10^k step giving at most `maxTicks` ticks over `span`. */
export function niceStep(span: number, maxTicks: number): number {
  if (!(span > 0) || !(maxTicks >= 1)) return 1;
  const raw = span / maxTicks;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (m * pow >= raw) return m * pow;
  return 10 * pow;
}

/** Tick values (s after origin) on multiples of a nice step inside the domain. */
export function timeTicks(domain: TimeDomain, maxTicks = 7): { step: number; ticks: number[] } {
  const step = niceStep(domain.end - domain.start, maxTicks);
  const ticks: number[] = [];
  const first = Math.ceil(domain.start / step - 1e-9);
  for (let k = first; k * step <= domain.end + 1e-9; k++) ticks.push(Number((k * step).toFixed(6)));
  return { step, ticks };
}

/** Horizontal position of a time (s after origin) as a fraction [0, 1] of the axis. */
export function axisFraction(tRel: number, domain: TimeDomain): number {
  return (tRel - domain.start) / (domain.end - domain.start);
}

/**
 * Where a pick or predicted arrival (epoch s, possibly null/absent) sits on the axis, as a percentage
 * of the track width; null when it's missing or outside the drawn window, so nothing is drawn.
 */
export function markPercent(
  tAbs: number | null | undefined,
  originT: number,
  domain: TimeDomain,
): number | null {
  if (!isNum(tAbs) || !isNum(originT)) return null;
  const f = axisFraction(tAbs - originT, domain);
  if (!(f >= 0 && f <= 1)) return null;
  return Math.round(f * 1e5) / 1e3; // 0.001 % resolution keeps attributes short
}

/** 1 / peak |sample|, so every trace fills its row; 0 for an all-zero or non-finite trace. */
export function normalizationGain(samples: readonly number[]): number {
  let peak = 0;
  for (const s of samples) {
    const a = Math.abs(s);
    if (a > peak && Number.isFinite(a)) peak = a;
  }
  return peak > 0 ? 1 / peak : 0;
}

const r2 = (v: number) => Math.round(v * 100) / 100;

/**
 * SVG polyline `points` for a trace in a TRACK_W × TRACK_H box on the shared axis, normalized to its
 * own peak. Non-finite samples are drawn at zero. With more than two samples per horizontal unit it
 * keeps the min and max of each unit column (order preserved), so the envelope survives decimation.
 */
export function tracePoints(tr: WaveformSnippet, originT: number, domain: TimeDomain): string {
  const n = tr.samples.length;
  const gain = normalizationGain(tr.samples);
  const span = domain.end - domain.start;
  const x0 = ((tr.t0 - originT - domain.start) / span) * TRACK_W;
  const dx = (tr.dt / span) * TRACK_W;
  const mid = TRACK_H / 2;
  const amp = TRACE_AMPLITUDE * TRACK_H;
  const y = (i: number) => {
    const s = tr.samples[i];
    return mid - (Number.isFinite(s) ? s : 0) * gain * amp;
  };
  const out: string[] = [];
  if (dx >= 0.5) {
    for (let i = 0; i < n; i++) out.push(`${r2(x0 + i * dx)},${r2(y(i))}`);
    return out.join(" ");
  }
  // Decimate: one min and one max per unit column, in sample order.
  let col = Math.floor(x0);
  let lo = 0;
  let hi = 0;
  let loI = -1;
  let hiI = -1;
  const flush = () => {
    if (loI < 0) return;
    const [a, b] = loI < hiI ? [loI, hiI] : [hiI, loI];
    out.push(`${r2(x0 + a * dx)},${r2(y(a))}`);
    if (b !== a) out.push(`${r2(x0 + b * dx)},${r2(y(b))}`);
  };
  for (let i = 0; i < n; i++) {
    const c = Math.floor(x0 + i * dx);
    if (c !== col) {
      flush();
      col = c;
      loI = -1;
      hiI = -1;
    }
    const v = y(i);
    if (loI < 0 || v < lo) {
      lo = v;
      loI = i;
    }
    if (hiI < 0 || v > hi) {
      hi = v;
      hiI = i;
    }
  }
  flush();
  return out.join(" ");
}

/** Pick ticks appear this long after the drawer starts opening (it's mostly in by then). */
export const PICKS_START_MS = motion.state / 2;
/** Each trace's ticks appear this long after the previous trace's (docs/lanes/H3 → Drawer: 150 ms). */
export const PICK_STAGGER_MS = motion.micro;

/** Animation delay of a pick tick: rows stagger by 150 ms; S follows P within the row by half a step. */
export function pickDelayMs(row: number, phase: "P" | "S"): number {
  return PICKS_START_MS + row * PICK_STAGGER_MS + (phase === "S" ? PICK_STAGGER_MS / 2 : 0);
}

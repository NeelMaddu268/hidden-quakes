/**
 * Layout for the association-sweep plot: pure functions of `SweepPoint[]`. The swept parameter,
 * the tick values and the series all come from the data; nothing here is a fixed number.
 */
import type { SweepPoint } from "@/providers";
import { isFiniteNumber } from "../validation/format";

/** The count fields of `SweepPoint`, in legend order. */
export const SWEEP_SERIES = ["candidates", "tierA", "recoveredPublic"] as const;
export type SweepField = (typeof SWEEP_SERIES)[number];

export interface SweepMark {
  /** Index into the sweep, so one mark maps back to one `SweepPoint`. */
  index: number;
  x: number;
  y: number;
  /** The point's other parameters, `key=value` joined; points sharing it lie on one line. */
  group: string;
}

export interface SweepSeries {
  field: SweepField;
  marks: SweepMark[];
  /** Runs of marks with identical other parameters, sorted by x; only runs of two or more. */
  lines: SweepMark[][];
}

export interface SweepLayout {
  xKey: string;
  xTicks: number[];
  yTicks: number[];
  yMax: number;
  series: SweepSeries[];
}

/**
 * The parameter to put on the x axis: the numeric key of `params` with the most distinct values
 * across the sweep (first key wins a tie). Null when no key is numeric on every point.
 */
export function sweptParam(sweep: readonly SweepPoint[]): string | null {
  const keys = new Set<string>();
  for (const point of sweep) for (const key of Object.keys(point.params)) keys.add(key);
  let best: { key: string; distinct: number } | null = null;
  for (const key of keys) {
    const values = sweep.map((point) => point.params[key]);
    if (!values.every(isFiniteNumber)) continue;
    const distinct = new Set(values).size;
    if (!best || distinct > best.distinct) best = { key, distinct };
  }
  return best?.key ?? null;
}

/** The other parameters of a point, as `key=value` pairs in key order. */
export function groupKey(params: SweepPoint["params"], xKey: string): string {
  return Object.keys(params)
    .filter((key) => key !== xKey)
    .sort()
    .map((key) => `${key}=${String(params[key])}`)
    .join(" · ");
}

/** Round-number ticks from zero to at least `max`, about `target` of them (a 1-2-5 step). */
export function niceTicks(max: number, target: number): number[] {
  if (!(max > 0)) return [0];
  const raw = max / Math.max(1, target);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  const factor = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  const step = factor * magnitude;
  const ticks: number[] = [];
  const count = Math.ceil(max / step - Number.EPSILON);
  for (let i = 0; i <= count; i += 1) ticks.push(Number((i * step).toPrecision(12)));
  return ticks;
}

/** Everything the SVG needs, or null when there is nothing plottable. */
export function sweepLayout(sweep: readonly SweepPoint[], yTickTarget: number): SweepLayout | null {
  if (sweep.length === 0) return null;
  const xKey = sweptParam(sweep);
  if (xKey === null) return null;
  const xs = sweep.map((point) => point.params[xKey] as number);
  const xTicks = [...new Set(xs)].sort((a, b) => a - b);

  let yMax = 0;
  const series: SweepSeries[] = SWEEP_SERIES.map((field) => {
    const marks: SweepMark[] = [];
    sweep.forEach((point, index) => {
      const y = point[field];
      if (!isFiniteNumber(y)) return;
      yMax = Math.max(yMax, y);
      marks.push({ index, x: xs[index], y, group: groupKey(point.params, xKey) });
    });
    const groups = new Map<string, SweepMark[]>();
    for (const mark of marks) {
      const run = groups.get(mark.group);
      if (run) run.push(mark);
      else groups.set(mark.group, [mark]);
    }
    const lines = [...groups.values()]
      .map((run) => [...run].sort((a, b) => a.x - b.x))
      .filter((run) => run.length > 1);
    return { field, marks, lines };
  });

  const yTicks = niceTicks(yMax, yTickTarget);
  return { xKey, xTicks, yTicks, yMax: yTicks[yTicks.length - 1], series };
}

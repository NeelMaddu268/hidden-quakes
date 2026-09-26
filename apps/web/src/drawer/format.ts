// Text formatting for the evidence drawer. Every function takes values straight from the bundle and
// returns display text; a missing or non-finite value renders as an em dash, never "NaN" or
// "undefined". Pure and locale-independent (except the Utah local time zone, from the ICU database).

import type { Magnitude } from "../scene/types";

/** Shown for any missing number. */
export const DASH = "—";

/** A real number we can display (not null/undefined/NaN/±Infinity). */
export function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

/** Replaces the ASCII hyphen of a negative number with a true minus sign. */
function minus(s: string): string {
  return s.startsWith("-") ? `−${s.slice(1)}` : s;
}

/** Fixed decimals, or the dash. `-0.00` renders as `0.00`. */
export function fmtFixed(v: number | null | undefined, digits: number): string {
  if (!isNum(v)) return DASH;
  const s = v.toFixed(digits);
  return /^-0\.?0*$/.test(s) ? s.slice(1) : minus(s);
}

/** A number followed by a unit (`0.078 s`), or the dash alone. */
export function fmtUnit(v: number | null | undefined, digits: number, unit: string): string {
  return isNum(v) ? `${fmtFixed(v, digits)} ${unit}` : DASH;
}

/** A 68% error in meters as `±369 m`, or the dash. */
export function fmtPlusMinusM(v: number | null | undefined): string {
  return isNum(v) ? `±${fmtFixed(v, 0)} m` : DASH;
}

/** Meters as kilometers with one decimal (`3.2 km`). */
export function fmtKmFromM(m: number | null | undefined, digits = 1): string {
  return isNum(m) ? `${fmtFixed(m / 1000, digits)} km` : DASH;
}

/** "11 stations agreed" / "1 station agreed" (docs/lanes/H3 → Drawer). */
export function fmtStationsAgreed(n: number | null | undefined): string {
  if (!isNum(n)) return `${DASH} stations agreed`;
  return `${fmtFixed(n, 0)} ${n === 1 ? "station" : "stations"} agreed`;
}

const pad2 = (n: number) => String(n).padStart(2, "0");

/** Epoch seconds rounded to centiseconds, as a Date plus the centisecond digits. */
function centis(epochS: number): { date: Date; cs: string } {
  const c = Math.round(epochS * 100);
  return { date: new Date(c * 10), cs: pad2(((c % 100) + 100) % 100) };
}

/** Origin time in UTC: `2026-09-10 14:03:07.21 UTC`. */
export function fmtUtc(epochS: number | null | undefined): string {
  if (!isNum(epochS)) return DASH;
  const { date: d, cs } = centis(epochS);
  return (
    `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())} ` +
    `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}:${pad2(d.getUTCSeconds())}.${cs} UTC`
  );
}

/** Utah's zone. September is MDT (UTC−6); the ICU database handles the MST months too. */
export const UTAH_TIME_ZONE = "America/Denver";

let utahFormat: Intl.DateTimeFormat | null = null;

/** Origin time in Utah local time: `2026-09-10 08:03:07.21 MDT`. */
export function fmtUtahLocal(epochS: number | null | undefined): string {
  if (!isNum(epochS)) return DASH;
  const { date, cs } = centis(epochS);
  utahFormat ??= new Intl.DateTimeFormat("en-US", {
    timeZone: UTAH_TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
    timeZoneName: "short",
  });
  const p: Record<string, string> = {};
  for (const part of utahFormat.formatToParts(date)) p[part.type] = part.value;
  return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}:${p.second}.${cs} ${p.timeZoneName}`;
}

/** `M 1.2 ML_cal ±0.2`; the type always travels with the value (docs/01 → Magnitude). Null if absent. */
export function fmtMagnitude(m: Magnitude | null | undefined): string | null {
  if (!m || !isNum(m.value)) return null;
  const sigma = isNum(m.sigma) ? ` ±${fmtFixed(m.sigma, 1)}` : "";
  return `M ${fmtFixed(m.value, 1)} ${m.type}${sigma}`;
}

/** Seconds relative to the origin time, for axis labels (`−1`, `0`, `2.5`). */
export function fmtSeconds(s: number, step: number): string {
  if (!isNum(s)) return DASH;
  const digits = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  return fmtFixed(s, digits);
}

/** A scale-bar length: `500 m`, `2.5 km`. */
export function fmtLength(m: number): string {
  if (!isNum(m)) return DASH;
  return m >= 1000 ? `${Number((m / 1000).toFixed(3))} km` : `${Number(m.toFixed(1))} m`;
}

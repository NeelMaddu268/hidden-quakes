// The station-day (helicorder) manifest (SEIS-10 → WEB-10): one PNG of a day of ground motion at one
// borehole station, plus the JSON that describes it. The generator (hq.preprocess.helicorder) writes both
// under apps/web/public/helicorder/; this module fetches and validates the JSON. Every word and number
// the panel shows comes from here, so anything malformed means no panel at all (null), never a guess.

import { defaultFetch, fetchJson, type FetchLike } from "../../providers/fetch";

/** Site-relative folder of the asset pair. */
export const STATION_DAY_BASE_URL = "/helicorder";
export const STATION_DAY_MANIFEST_URL = `${STATION_DAY_BASE_URL}/station-day.json`;

export type StationDayLegendKey = "tierA" | "tierB" | "tierC" | "public";
export type StationDayMarkerShape = "tick" | "diamond";

export interface StationDayLegendEntry {
  key: StationDayLegendKey;
  label: string;
  /** "#RRGGBB". */
  color: string;
  opacity: number;
  shape: StationDayMarkerShape;
  count: number;
}

export interface StationDayRunnerUp {
  stationId: string;
  pickCount: number;
}

export interface StationDayManifest {
  image: string;
  widthPx: number;
  heightPx: number;
  title: string;
  caption: string;
  runId: string;
  /** "NET.STA". */
  stationId: string;
  /** "NET.STA.LOC.CHA". */
  seedId: string;
  channel: string;
  stationKind: string;
  sensorDepthM: number | null;
  /** "YYYY-MM-DD". */
  dayUtc: string;
  startUtc: string;
  endUtc: string;
  rowMinutes: 30 | 60;
  rows: number;
  filterHz: [number, number];
  gapsFilled: false;
  gapSeconds: number;
  coverageFraction: number;
  markerTime: "origin";
  legend: StationDayLegendEntry[];
  selection: { rule: string; pickCount: number; runnersUp: StationDayRunnerUp[] };
  source: string;
  generator: string;
}

export type StationDayParse = { ok: true; manifest: StationDayManifest } | { ok: false; errors: string[] };

const LEGEND_KEYS: ReadonlySet<string> = new Set(["tierA", "tierB", "tierC", "public"]);
const SHAPES: ReadonlySet<string> = new Set(["tick", "diamond"]);
/** A plain PNG file name next to the manifest: no folders, no "..", nothing to escape in a URL. */
const IMAGE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*\.png$/i;
const HEX_COLOR = /^#[0-9A-Fa-f]{6}$/;
const DAY = /^\d{4}-\d{2}-\d{2}$/;
const ISO_Z = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?Z$/;
const STATION_ID = /^[^.\s]+\.[^.\s]+$/;
const SEED_ID = /^[^.\s]+\.[^.\s]+\.[^.\s]*\.[^.\s]+$/;
/** Generous upper bound on the PNG's pixel size (the contract's width is 1920). */
const MAX_PX = 16384;

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const isText = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isInt = (v: unknown, min: number, max = Number.MAX_SAFE_INTEGER): v is number =>
  Number.isInteger(v) && (v as number) >= min && (v as number) <= max;
/** A finite number in [0, 1] (opacity, coverage). */
const isUnit = (v: unknown): v is number => isNum(v) && !(v < 0 || v > 1);

/**
 * Checks one parsed JSON value against the station-day contract. Unknown extra fields are ignored; the
 * result is a fresh object with only the contract's fields. Errors name the field, for the console.
 */
export function parseStationDayManifest(raw: unknown): StationDayParse {
  const errors: string[] = [];
  if (!isObj(raw)) return { ok: false, errors: ["manifest is not a JSON object"] };
  const m = raw;
  const need = (ok: boolean, field: string, what: string) => {
    if (!ok) errors.push(`${field}: expected ${what}`);
  };

  need(typeof m.image === "string" && IMAGE_NAME.test(m.image), "image", "a .png file name");
  need(isInt(m.widthPx, 1, MAX_PX), "widthPx", "a positive integer");
  need(isInt(m.heightPx, 1, MAX_PX), "heightPx", "a positive integer");
  for (const field of ["title", "caption", "runId", "channel", "stationKind", "source", "generator"]) {
    need(isText(m[field]), field, "a non-empty string");
  }
  need(typeof m.stationId === "string" && STATION_ID.test(m.stationId), "stationId", '"NET.STA"');
  need(typeof m.seedId === "string" && SEED_ID.test(m.seedId), "seedId", '"NET.STA.LOC.CHA"');
  need(m.sensorDepthM === null || isNum(m.sensorDepthM), "sensorDepthM", "a number or null");
  need(typeof m.dayUtc === "string" && DAY.test(m.dayUtc), "dayUtc", '"YYYY-MM-DD"');
  const start = typeof m.startUtc === "string" && ISO_Z.test(m.startUtc) ? Date.parse(m.startUtc) : NaN;
  const end = typeof m.endUtc === "string" && ISO_Z.test(m.endUtc) ? Date.parse(m.endUtc) : NaN;
  need(Number.isFinite(start), "startUtc", "an ISO UTC time ending in Z");
  need(Number.isFinite(end), "endUtc", "an ISO UTC time ending in Z");
  if (Number.isFinite(start) && Number.isFinite(end)) need(end > start, "endUtc", "a time after startUtc");
  need(m.rowMinutes === 30 || m.rowMinutes === 60, "rowMinutes", "30 or 60");
  need(isInt(m.rows, 1), "rows", "a positive integer");
  const f = m.filterHz;
  need(
    Array.isArray(f) && f.length === 2 && isNum(f[0]) && isNum(f[1]) && f[0] > 0 && f[1] > f[0],
    "filterHz",
    "[lo, hi] with 0 < lo < hi",
  );
  need(m.gapsFilled === false, "gapsFilled", "false");
  need(isNum(m.gapSeconds) && m.gapSeconds >= 0, "gapSeconds", "a number >= 0");
  need(isUnit(m.coverageFraction), "coverageFraction", "a number in [0, 1]");
  need(m.markerTime === "origin", "markerTime", '"origin"');

  const legend: StationDayLegendEntry[] = [];
  if (!Array.isArray(m.legend)) errors.push("legend: expected an array");
  else
    m.legend.forEach((entry: unknown, i) => {
      const at = `legend[${i}]`;
      if (!isObj(entry)) {
        errors.push(`${at}: expected an object`);
        return;
      }
      const before = errors.length;
      need(typeof entry.key === "string" && LEGEND_KEYS.has(entry.key), `${at}.key`, "tierA, tierB, tierC or public");
      need(isText(entry.label), `${at}.label`, "a non-empty string");
      need(typeof entry.color === "string" && HEX_COLOR.test(entry.color), `${at}.color`, '"#RRGGBB"');
      need(isUnit(entry.opacity), `${at}.opacity`, "a number in [0, 1]");
      need(typeof entry.shape === "string" && SHAPES.has(entry.shape), `${at}.shape`, '"tick" or "diamond"');
      need(isInt(entry.count, 0), `${at}.count`, "an integer >= 0");
      if (errors.length === before) {
        legend.push({
          key: entry.key as StationDayLegendKey,
          label: entry.label as string,
          color: entry.color as string,
          opacity: entry.opacity as number,
          shape: entry.shape as StationDayMarkerShape,
          count: entry.count as number,
        });
      }
    });

  const runnersUp: StationDayRunnerUp[] = [];
  const s = m.selection;
  if (!isObj(s)) errors.push("selection: expected an object");
  else {
    need(isText(s.rule), "selection.rule", "a non-empty string");
    need(isInt(s.pickCount, 0), "selection.pickCount", "an integer >= 0");
    if (!Array.isArray(s.runnersUp)) errors.push("selection.runnersUp: expected an array");
    else
      s.runnersUp.forEach((r: unknown, i) => {
        const ok = isObj(r) && typeof r.stationId === "string" && STATION_ID.test(r.stationId) && isInt(r.pickCount, 0);
        if (ok) runnersUp.push({ stationId: r.stationId as string, pickCount: r.pickCount as number });
        else errors.push(`selection.runnersUp[${i}]: expected {stationId: "NET.STA", pickCount: integer >= 0}`);
      });
  }

  if (errors.length > 0) return { ok: false, errors };
  const sel = s as Obj;
  return {
    ok: true,
    manifest: {
      image: m.image as string,
      widthPx: m.widthPx as number,
      heightPx: m.heightPx as number,
      title: m.title as string,
      caption: m.caption as string,
      runId: m.runId as string,
      stationId: m.stationId as string,
      seedId: m.seedId as string,
      channel: m.channel as string,
      stationKind: m.stationKind as string,
      sensorDepthM: m.sensorDepthM as number | null,
      dayUtc: m.dayUtc as string,
      startUtc: m.startUtc as string,
      endUtc: m.endUtc as string,
      rowMinutes: m.rowMinutes as 30 | 60,
      rows: m.rows as number,
      filterHz: [(f as number[])[0], (f as number[])[1]],
      gapsFilled: false,
      gapSeconds: m.gapSeconds as number,
      coverageFraction: m.coverageFraction as number,
      markerTime: "origin",
      legend,
      selection: { rule: sel.rule as string, pickCount: sel.pickCount as number, runnersUp },
      source: m.source as string,
      generator: m.generator as string,
    },
  };
}

/** The PNG's site-relative URL. */
export function stationDayImageUrl(manifest: Pick<StationDayManifest, "image">): string {
  return `${STATION_DAY_BASE_URL}/${manifest.image}`;
}

/**
 * Fetches and validates the manifest. Null when it is absent (404: no station day was generated), when
 * the request fails, or when it breaks the contract (logged, so a generator bug is visible in the console).
 */
export async function loadStationDayManifest(fetchImpl: FetchLike = defaultFetch): Promise<StationDayManifest | null> {
  let raw: unknown;
  try {
    raw = await fetchJson<unknown>(fetchImpl, STATION_DAY_MANIFEST_URL, { notFoundAsNull: true });
  } catch {
    return null;
  }
  if (raw === null) return null;
  const parsed = parseStationDayManifest(raw);
  if (!parsed.ok) {
    console.warn(`[stationday] ${STATION_DAY_MANIFEST_URL} ignored: ${parsed.errors.join("; ")}`);
    return null;
  }
  return parsed.manifest;
}

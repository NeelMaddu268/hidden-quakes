// Terrain asset metadata (apps/web/public/terrain/meta.json, written by scripts/bake-dem.py) and the
// pure decoders for its two PNGs. The browser turns the PNGs into RGBA pixels (load.ts); everything
// after that is here, so it is testable without a canvas.
//
// height.png is "rg16": 8-bit RGB where v = R * 256 + G (B = 0) and
//   elevM = elevMinM + v / 65535 * (elevMaxM − elevMinM).
// Browsers truncate true 16-bit PNGs to 8 bits in canvas and WebGL, hence the split across two bytes.
// Row 0 of both PNGs is the northernmost row (n = nMax), column 0 the westernmost (e = eMin), and
// enuBounds are pixel centres, so pixel (r, c) is exactly mesh vertex (r, c).

import type { SceneMeta } from "../types";

export const RG16_MAX = 65535;

export interface EnuBoundsM {
  eMin: number;
  eMax: number;
  nMin: number;
  nMax: number;
}

export interface TerrainMeta {
  version: number;
  encoding: "rg16";
  elevMinM: number;
  elevMaxM: number;
  /** [width, height] in pixels; the mesh has (width − 1) × (height − 1) cells. */
  sizePx: [number, number];
  enuBounds: EnuBoundsM & { at: "pixelCenters" };
  origin: { lat: number; lon: number; elevM: number };
  projection: string;
  hillshade: { azimuthDeg: number; altitudeDeg: number; flatValue: number };
  checksums: { heightSumV: number; hillshadeSum: number };
  files: { height: string; hillshade: string };
  source: { name: string; url: string; attribution: string; zoom: number };
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function num(o: Record<string, unknown>, key: string, problems: string[], path: string): number {
  const v = o[key];
  if (typeof v !== "number" || !Number.isFinite(v)) {
    problems.push(`${path}${key} must be a finite number, got ${JSON.stringify(v)}`);
    return Number.NaN;
  }
  return v;
}

function str(o: Record<string, unknown>, key: string, problems: string[], path: string): string {
  const v = o[key];
  if (typeof v !== "string" || v.length === 0) {
    problems.push(`${path}${key} must be a non-empty string, got ${JSON.stringify(v)}`);
    return "";
  }
  return v;
}

function obj(o: Record<string, unknown>, key: string, problems: string[]): Record<string, unknown> {
  const v = o[key];
  if (!isObject(v)) {
    problems.push(`${key} must be an object`);
    return {};
  }
  return v;
}

/** Validates meta.json. Throws one error listing every problem (never renders a half-valid terrain). */
export function parseTerrainMeta(json: unknown): TerrainMeta {
  const problems: string[] = [];
  if (!isObject(json)) throw new Error("terrain meta.json is not an object");
  const encoding = json.encoding;
  if (encoding !== "rg16") problems.push(`encoding must be "rg16", got ${JSON.stringify(encoding)}`);

  const size = json.sizePx;
  const sizeOk =
    Array.isArray(size) &&
    size.length === 2 &&
    size.every((s) => Number.isInteger(s) && (s as number) >= 2);
  if (!sizeOk) problems.push(`sizePx must be [w, h] integers >= 2, got ${JSON.stringify(size)}`);

  const b = obj(json, "enuBounds", problems);
  const eMin = num(b, "eMin", problems, "enuBounds.");
  const eMax = num(b, "eMax", problems, "enuBounds.");
  const nMin = num(b, "nMin", problems, "enuBounds.");
  const nMax = num(b, "nMax", problems, "enuBounds.");
  if (b.at !== "pixelCenters") problems.push(`enuBounds.at must be "pixelCenters", got ${JSON.stringify(b.at)}`);
  if (!(eMax > eMin) || !(nMax > nMin)) problems.push("enuBounds must have eMax > eMin and nMax > nMin");

  const elevMinM = num(json, "elevMinM", problems, "");
  const elevMaxM = num(json, "elevMaxM", problems, "");
  if (!(elevMaxM > elevMinM)) problems.push("elevMaxM must be greater than elevMinM");

  const o = obj(json, "origin", problems);
  const hs = obj(json, "hillshade", problems);
  const cs = obj(json, "checksums", problems);
  const files = obj(json, "files", problems);
  const src = obj(json, "source", problems);

  const meta: TerrainMeta = {
    version: num(json, "version", problems, ""),
    encoding: "rg16",
    elevMinM,
    elevMaxM,
    sizePx: sizeOk ? [size[0] as number, size[1] as number] : [0, 0],
    enuBounds: { eMin, eMax, nMin, nMax, at: "pixelCenters" },
    origin: {
      lat: num(o, "lat", problems, "origin."),
      lon: num(o, "lon", problems, "origin."),
      elevM: num(o, "elevM", problems, "origin."),
    },
    projection: str(json, "projection", problems, ""),
    hillshade: {
      azimuthDeg: num(hs, "azimuthDeg", problems, "hillshade."),
      altitudeDeg: num(hs, "altitudeDeg", problems, "hillshade."),
      flatValue: num(hs, "flatValue", problems, "hillshade."),
    },
    checksums: {
      heightSumV: num(cs, "heightSumV", problems, "checksums."),
      hillshadeSum: num(cs, "hillshadeSum", problems, "checksums."),
    },
    files: {
      height: str(files, "height", problems, "files."),
      hillshade: str(files, "hillshade", problems, "files."),
    },
    source: {
      name: str(src, "name", problems, "source."),
      url: str(src, "url", problems, "source."),
      attribution: str(src, "attribution", problems, "source."),
      zoom: num(src, "zoom", problems, "source."),
    },
  };
  if (meta.hillshade.flatValue <= 0 || meta.hillshade.flatValue > 255) {
    problems.push(`hillshade.flatValue must be in (0, 255], got ${meta.hillshade.flatValue}`);
  }
  if (problems.length) throw new Error(`terrain meta.json is invalid: ${problems.join("; ")}`);
  return meta;
}

/**
 * Why this terrain can't sit under this bundle, or null if it can. The terrain's ENU grid is relative
 * to its own origin, so it only lines up with the bundle's ENU records when the origins match.
 * Heights are absolute (m ASL), so a different origin elevation is fine.
 */
export function terrainMismatch(
  meta: Pick<TerrainMeta, "origin" | "projection">,
  scene: Pick<SceneMeta, "originLat" | "originLon" | "projection">,
): string | null {
  const TOL_DEG = 1e-6; // ~0.1 m
  if (meta.projection !== scene.projection) {
    return `terrain projection "${meta.projection}" differs from the bundle's "${scene.projection}"`;
  }
  if (Math.abs(meta.origin.lat - scene.originLat) > TOL_DEG || Math.abs(meta.origin.lon - scene.originLon) > TOL_DEG) {
    return (
      `terrain origin (${meta.origin.lat}, ${meta.origin.lon}) differs from the bundle origin ` +
      `(${scene.originLat}, ${scene.originLon}); rebake with scripts/bake-dem.py`
    );
  }
  return null;
}

export interface DecodedHeights {
  /** Elevation in m ASL per pixel, row-major, row 0 = north. */
  elevM: Float32Array;
  /** Σ v over every pixel; must equal meta.checksums.heightSumV for a bit-exact decode. */
  sumV: number;
  /** Pixels whose B byte isn't 0 (the encoder always writes 0): a sign the browser altered pixels. */
  nonZeroBlue: number;
}

/** RGBA pixels of height.png → elevations. */
export function decodeRg16(
  rgba: ArrayLike<number>,
  meta: Pick<TerrainMeta, "sizePx" | "elevMinM" | "elevMaxM">,
): DecodedHeights {
  const [w, h] = meta.sizePx;
  const n = w * h;
  if (rgba.length !== n * 4) throw new Error(`height.png has ${rgba.length / 4} pixels, meta says ${w} × ${h}`);
  const elevM = new Float32Array(n);
  const scale = (meta.elevMaxM - meta.elevMinM) / RG16_MAX;
  let sumV = 0;
  let nonZeroBlue = 0;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const v = rgba[p] * 256 + rgba[p + 1];
    sumV += v;
    if (rgba[p + 2] !== 0) nonZeroBlue++;
    elevM[i] = meta.elevMinM + v * scale;
  }
  return { elevM, sumV, nonZeroBlue };
}

/** RGBA pixels of the grayscale hillshade.png → one byte per pixel (the red channel), plus its sum. */
export function decodeHillshade(
  rgba: ArrayLike<number>,
  meta: Pick<TerrainMeta, "sizePx">,
): { shade: Uint8Array; sum: number } {
  const [w, h] = meta.sizePx;
  const n = w * h;
  if (rgba.length !== n * 4) throw new Error(`hillshade.png has ${rgba.length / 4} pixels, meta says ${w} × ${h}`);
  const shade = new Uint8Array(n);
  let sum = 0;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    shade[i] = rgba[p];
    sum += rgba[p];
  }
  return { shade, sum };
}

/** Problems that mean the browser didn't decode the PNGs bit-exactly (empty when all is well). */
export function decodeIssues(
  meta: Pick<TerrainMeta, "checksums">,
  heights: Pick<DecodedHeights, "sumV" | "nonZeroBlue">,
  hillshadeSum: number,
): string[] {
  const issues: string[] = [];
  if (heights.sumV !== meta.checksums.heightSumV) {
    issues.push(`height.png checksum ${heights.sumV} != ${meta.checksums.heightSumV} (pixels altered on decode)`);
  }
  if (heights.nonZeroBlue > 0) issues.push(`height.png has ${heights.nonZeroBlue} pixels with B != 0`);
  if (hillshadeSum !== meta.checksums.hillshadeSum) {
    issues.push(`hillshade.png checksum ${hillshadeSum} != ${meta.checksums.hillshadeSum}`);
  }
  return issues;
}

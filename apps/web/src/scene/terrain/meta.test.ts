import { readFileSync } from "node:fs";
import { join } from "node:path";
import { inflateSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  decodeHillshade,
  decodeIssues,
  decodeRg16,
  normalizeProjection,
  parseTerrainMeta,
  RG16_MAX,
  terrainMismatch,
  type TerrainMeta,
} from "./meta";

const TERRAIN_DIR = join(__dirname, "../../../public/terrain");

function committedMeta(): TerrainMeta {
  return parseTerrainMeta(JSON.parse(readFileSync(join(TERRAIN_DIR, "meta.json"), "utf8")));
}

/**
 * Test-only PNG decoder (8-bit, non-interlaced, gray or RGB) → RGBA, exactly what a canvas gives the
 * browser loader for these files. Lets the committed assets be checked end to end in Node.
 */
function decodePngToRgba(buf: Buffer): { width: number; height: number; rgba: Uint8Array } {
  expect(buf.subarray(0, 8)).toEqual(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  let off = 8;
  let width = 0;
  let height = 0;
  let channels = 0;
  const idat: Buffer[] = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off);
    const type = buf.toString("ascii", off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      expect(data[8]).toBe(8); // bit depth
      expect(data[12]).toBe(0); // no interlace
      channels = { 0: 1, 2: 3 }[data[9]] ?? 0;
      expect(channels).toBeGreaterThan(0);
    } else if (type === "IDAT") idat.push(data);
    off += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const px = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? px[y * stride + x - channels] : 0;
      const b = y > 0 ? px[(y - 1) * stride + x] : 0;
      const c = x >= channels && y > 0 ? px[(y - 1) * stride + x - channels] : 0;
      let pred = 0;
      if (filter === 1) pred = a;
      else if (filter === 2) pred = b;
      else if (filter === 3) pred = (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      px[y * stride + x] = (line[x] + pred) & 0xff;
    }
  }
  const rgba = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    for (let k = 0; k < 3; k++) rgba[i * 4 + k] = px[i * channels + (channels === 1 ? 0 : k)];
    rgba[i * 4 + 3] = 255;
  }
  return { width, height, rgba };
}

describe("parseTerrainMeta", () => {
  it("accepts the committed meta.json from scripts/bake-dem.py", () => {
    const meta = committedMeta();
    expect(meta.encoding).toBe("rg16");
    expect(meta.enuBounds.at).toBe("pixelCenters");
    expect(meta.projection).toBe("EPSG:32612 minus origin");
    expect(meta.elevMaxM).toBeGreaterThan(meta.elevMinM);
    expect(meta.source.url).toContain("terrarium");
    expect(meta.source.attribution.length).toBeGreaterThan(20);
  });

  it("rejects bad metadata loudly, listing every problem", () => {
    const good = JSON.parse(readFileSync(join(TERRAIN_DIR, "meta.json"), "utf8"));
    expect(() => parseTerrainMeta(null)).toThrow(/not an object/);
    expect(() => parseTerrainMeta({ ...good, encoding: "png16" })).toThrow(/encoding must be "rg16"/);
    expect(() => parseTerrainMeta({ ...good, sizePx: [513] })).toThrow(/sizePx/);
    expect(() => parseTerrainMeta({ ...good, elevMaxM: good.elevMinM })).toThrow(/elevMaxM/);
    expect(() => parseTerrainMeta({ ...good, enuBounds: { ...good.enuBounds, at: "edges" } })).toThrow(
      /pixelCenters/,
    );
    const both = () => parseTerrainMeta({ ...good, origin: { lat: "38" }, checksums: {} });
    expect(both).toThrow(/origin\.lat.*origin\.lon.*checksums\.heightSumV/);
  });
});

describe("the committed terrain assets", () => {
  // End to end: the PNGs on disk decode, with the browser's own decoder, to exactly the checksums the
  // bake recorded. A rebake that updates one file but not the others fails here.
  const meta = committedMeta();
  const height = decodePngToRgba(readFileSync(join(TERRAIN_DIR, meta.files.height)));
  const shade = decodePngToRgba(readFileSync(join(TERRAIN_DIR, meta.files.hillshade)));

  it("match meta.json's size and decode bit-exact", () => {
    expect([height.width, height.height]).toEqual(meta.sizePx);
    expect([shade.width, shade.height]).toEqual(meta.sizePx);
    const heights = decodeRg16(height.rgba, meta);
    const hs = decodeHillshade(shade.rgba, meta);
    expect(decodeIssues(meta, heights, hs.sum)).toEqual([]);
  });

  it("span elevMinM..elevMaxM, and are plausible for the site", () => {
    const { elevM } = decodeRg16(height.rgba, meta);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of elevM) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    const step = (meta.elevMaxM - meta.elevMinM) / RG16_MAX;
    expect(lo).toBeGreaterThanOrEqual(meta.elevMinM - 1e-3);
    expect(hi).toBeLessThanOrEqual(meta.elevMaxM + 1e-3);
    expect(lo - meta.elevMinM).toBeLessThan(0.01 + step);
    expect(meta.elevMaxM - hi).toBeLessThan(0.01 + step);
    // The centre pixel is the origin: the DEM there agrees with run.yaml's surveyed origin elevation.
    const [w, h] = meta.sizePx;
    const centre = elevM[((h - 1) / 2) * w + (w - 1) / 2];
    expect(Math.abs(centre - meta.origin.elevM)).toBeLessThan(15);
  });
});

describe("decodeRg16", () => {
  const meta = { sizePx: [2, 2] as [number, number], elevMinM: 1000, elevMaxM: 1000 + RG16_MAX / 100 };

  it("maps R*256 + G linearly onto elevMinM..elevMaxM", () => {
    const vs = [0, 1, 256 * 7 + 3, RG16_MAX];
    const rgba = new Uint8Array(16);
    vs.forEach((v, i) => {
      rgba[i * 4] = v >> 8;
      rgba[i * 4 + 1] = v & 0xff;
      rgba[i * 4 + 3] = 255;
    });
    const out = decodeRg16(rgba, meta);
    expect(out.elevM[0]).toBe(1000);
    expect(out.elevM[1]).toBeCloseTo(1000.01, 4);
    expect(out.elevM[2]).toBeCloseTo(1000 + (256 * 7 + 3) / 100, 3);
    expect(out.elevM[3]).toBeCloseTo(meta.elevMaxM, 3);
    expect(out.sumV).toBe(vs.reduce((a, b) => a + b, 0));
    expect(out.nonZeroBlue).toBe(0);
  });

  it("counts altered pixels and rejects a size mismatch", () => {
    const rgba = new Uint8Array(16);
    rgba[2] = 9;
    expect(decodeRg16(rgba, meta).nonZeroBlue).toBe(1);
    expect(() => decodeRg16(new Uint8Array(12), meta)).toThrow(/pixels/);
  });

  it("decodeIssues flags checksum drift and non-zero blue", () => {
    const m = { checksums: { heightSumV: 10, hillshadeSum: 5 } };
    expect(decodeIssues(m, { sumV: 10, nonZeroBlue: 0 }, 5)).toEqual([]);
    expect(decodeIssues(m, { sumV: 11, nonZeroBlue: 2 }, 6)).toHaveLength(3);
  });
});

describe("terrainMismatch", () => {
  const meta = { origin: { lat: 38.51, lon: -112.9, elevM: 1627.7 }, projection: "EPSG:32612 minus origin" };
  const scene = { originLat: 38.51, originLon: -112.9, projection: "EPSG:32612 minus origin" };

  it("accepts the same origin and projection (a different origin elevation is fine)", () => {
    expect(terrainMismatch(meta, scene)).toBeNull();
    expect(terrainMismatch({ ...meta, origin: { ...meta.origin, elevM: 1500 } }, scene)).toBeNull();
  });

  it("compares projections ignoring case and whitespace", () => {
    expect(normalizeProjection("  EPSG:32612   Minus\tOrigin ")).toBe("epsg:32612 minus origin");
    expect(terrainMismatch(meta, { ...scene, projection: "epsg:32612  MINUS origin " })).toBeNull();
  });

  it("rejects a different origin or projection", () => {
    expect(terrainMismatch(meta, { ...scene, originLat: 38.52 })).toMatch(/origin/);
    expect(terrainMismatch(meta, { ...scene, originLon: -112.91 })).toMatch(/origin/);
    expect(terrainMismatch(meta, { ...scene, projection: "EPSG:32611 minus origin" })).toMatch(/projection/);
  });
});

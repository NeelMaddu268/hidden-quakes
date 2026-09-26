"use client";
// Loads the baked terrain once per page (meta.json + height.png + hillshade.png from public/terrain/)
// and decodes it on the CPU. One cached promise, shared by every caller, so the terrain surface and
// the reference layers (which need its extent) never fetch twice. Failure is loud (console.error) and
// turns into the abstract-slab fallback, labelled on screen, never a silent change of shape.

import { useEffect, useState } from "react";
import { decodeHillshade, decodeIssues, decodeRg16, parseTerrainMeta, type TerrainMeta } from "./meta";

export const TERRAIN_BASE_URL = "/terrain";

export type TerrainAsset =
  | { status: "loading" }
  | { status: "ready"; meta: TerrainMeta; elevM: Float32Array; shade: Uint8Array }
  | { status: "unavailable"; reason: string };

async function fetchOk(url: string): Promise<Response> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res;
}

/** Exact RGBA bytes of a PNG: no colour-space conversion, no premultiplication. */
async function readPixels(url: string): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  const blob = await (await fetchOk(url)).blob();
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  try {
    const { width, height } = bitmap;
    const canvas =
      typeof OffscreenCanvas !== "undefined"
        ? new OffscreenCanvas(width, height)
        : Object.assign(document.createElement("canvas"), { width, height });
    const ctx = canvas.getContext("2d", { willReadFrequently: true }) as
      | CanvasRenderingContext2D
      | OffscreenCanvasRenderingContext2D
      | null;
    if (!ctx) throw new Error("2D canvas unavailable for decoding terrain PNGs");
    ctx.drawImage(bitmap, 0, 0);
    return { data: ctx.getImageData(0, 0, width, height).data, width, height };
  } finally {
    bitmap.close();
  }
}

async function loadFrom(base: string): Promise<TerrainAsset> {
  const t0 = performance.now();
  const meta = parseTerrainMeta(await (await fetchOk(`${base}/meta.json`)).json());
  const [heightPx, shadePx] = await Promise.all([
    readPixels(`${base}/${meta.files.height}`),
    readPixels(`${base}/${meta.files.hillshade}`),
  ]);
  for (const [name, px] of [
    ["height", heightPx],
    ["hillshade", shadePx],
  ] as const) {
    if (px.width !== meta.sizePx[0] || px.height !== meta.sizePx[1]) {
      throw new Error(`${name}.png is ${px.width} × ${px.height}, meta.json says ${meta.sizePx.join(" × ")}`);
    }
  }
  const heights = decodeRg16(heightPx.data, meta);
  const hillshade = decodeHillshade(shadePx.data, meta);
  const issues = decodeIssues(meta, heights, hillshade.sum);
  if (issues.length) throw new Error(issues.join("; "));
  console.info(
    `[terrain] ${meta.sizePx.join(" × ")} grid, ${meta.elevMinM}–${meta.elevMaxM} m ASL, decoded bit-exact in ` +
      `${Math.round(performance.now() - t0)} ms (${meta.source.name})`,
  );
  return { status: "ready", meta, elevM: heights.elevM, shade: hillshade.shade };
}

let pending: Promise<TerrainAsset> | null = null;
let settled: TerrainAsset | null = null;

/** The terrain, loaded at most once per page. Never rejects: failures resolve to "unavailable". */
export function loadTerrain(base: string = TERRAIN_BASE_URL): Promise<TerrainAsset> {
  if (!pending) {
    pending = loadFrom(base).then(
      (asset) => (settled = asset),
      (err: unknown) => {
        const reason = err instanceof Error ? err.message : String(err);
        console.error(`[terrain] unavailable, showing the abstract surface instead: ${reason}`);
        return (settled = { status: "unavailable", reason });
      },
    );
  }
  return pending;
}

/** React view of `loadTerrain()`. */
export function useTerrainAsset(): TerrainAsset {
  const [asset, setAsset] = useState<TerrainAsset>(() => settled ?? { status: "loading" });
  useEffect(() => {
    let live = true;
    loadTerrain().then((a) => {
      if (live) setAsset(a);
    });
    return () => {
      live = false;
    };
  }, []);
  return asset;
}

/** `?terrain=slab` forces the abstract slab (QA and the "Terrain" kill switch in docs/03). */
export function urlForcesSlab(): boolean {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("terrain") === "slab";
}

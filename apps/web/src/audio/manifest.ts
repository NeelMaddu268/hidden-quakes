"use client";
// Listen (WEB-10): the sonified clips under public/audio/ and their JSON manifests (written by
// hq.preprocess.sonify). Every label, station, channel, window and speed the UI shows comes from a
// manifest at runtime; nothing about a clip is hard-coded here. A missing or malformed manifest
// resolves to null and its button simply doesn't render.

import { useEffect, useState } from "react";
import { defaultFetch, fetchJson, type FetchLike } from "../providers/fetch";

/** Where the clips live, relative to the site root (mirrors DATA_BASE_URL and TERRAIN_BASE_URL). */
export const AUDIO_BASE_URL = "/audio";

/** Manifest base names under AUDIO_BASE_URL (`<name>.json`). */
export const LISTEN_CLIPS = Object.freeze({
  busiestHour: "hidden-quakes-busiest-hour",
  hero: "hidden-quakes-hero",
});

export interface ListenSource {
  src: string;
  type: string;
}

export interface ListenManifest {
  stationId: string;
  channel: string;
  /** Epoch seconds of the first and last sample the clip renders. */
  startUtc: number;
  endUtc: number;
  /** Data seconds per audio second. */
  speed: number;
  /** Clip length in audio seconds. */
  durationS: number;
  note: string;
  source: string | null;
  runId: string | null;
  heroEventId: string | null;
  /** Epoch seconds of the hero event's origin (hero clip only). */
  eventOriginUtc: number | null;
  /** Playable files, Ogg first (smaller), MP3 as the fallback. */
  sources: ListenSource[];
}

const FILE_NAME = /^[A-Za-z0-9._-]+$/;
const FORMATS = [
  ["ogg", "audio/ogg"],
  ["mp3", "audio/mpeg"],
] as const;

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function nonEmpty(v: unknown): string | null {
  return typeof v === "string" && v.trim() !== "" ? v : null;
}

function epochS(v: unknown): number | null {
  if (typeof v !== "string") return null;
  const ms = Date.parse(v);
  return Number.isFinite(ms) ? ms / 1000 : null;
}

function positive(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null;
}

/** A validated manifest, or null when anything the player needs is missing or malformed. */
export function parseListenManifest(raw: unknown, base: string = AUDIO_BASE_URL): ListenManifest | null {
  if (!isRecord(raw)) return null;
  const stationId = nonEmpty(raw.stationId);
  const channel = nonEmpty(raw.channel);
  const startUtc = epochS(raw.startUtc);
  const endUtc = epochS(raw.endUtc);
  const speed = positive(raw.speed);
  const durationS = positive(raw.durationS);
  if (!stationId || !channel || startUtc === null || endUtc === null || !(endUtc > startUtc)) return null;
  if (speed === null || durationS === null) return null;

  const files = isRecord(raw.files) ? raw.files : {};
  const sources: ListenSource[] = [];
  for (const [key, type] of FORMATS) {
    const entry = files[key];
    const name = isRecord(entry) ? entry.name : undefined;
    if (typeof name === "string" && FILE_NAME.test(name)) sources.push({ src: `${base}/${name}`, type });
  }
  if (sources.length === 0) return null;

  const selection = isRecord(raw.selection) ? raw.selection : {};
  return {
    stationId,
    channel,
    startUtc,
    endUtc,
    speed,
    durationS,
    note: typeof raw.note === "string" ? raw.note.trim() : "",
    source: nonEmpty(raw.source),
    runId: nonEmpty(raw.runId),
    heroEventId: nonEmpty(raw.heroEventId),
    eventOriginUtc: epochS(selection.eventOriginUtc),
    sources,
  };
}

/** The clip fits the loaded run: same run (when the manifest names one) and inside its window. */
export function listenFitsRun(
  m: ListenManifest,
  run: { windowStart: number; windowEnd: number },
  runId: string | null | undefined,
): boolean {
  if (m.runId !== null && runId != null && m.runId !== runId) return false;
  return m.startUtc >= run.windowStart && m.endUtc <= run.windowEnd;
}

/** Where the scene clock is (epoch s) when the clip has played `audioS` seconds: clamped to the clip window. */
export function audioToTNow(m: Pick<ListenManifest, "startUtc" | "endUtc" | "speed">, audioS: number): number {
  const t = m.startUtc + (Number.isFinite(audioS) && audioS > 0 ? audioS : 0) * m.speed;
  return t >= m.endUtc ? m.endUtc : t;
}

function fmtSpeed(speed: number): string {
  return speed.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

/** Button label, e.g. "Listen: <station>, sped up <speed>×". */
export function listenLabel(m: ListenManifest): string {
  return `Listen: ${m.stationId}, sped up ${fmtSpeed(m.speed)}×`;
}

function utcParts(epoch: number): { date: string; time: string } {
  const iso = new Date(Math.round(epoch * 1000)).toISOString();
  const time = iso.slice(11, 23);
  return { date: iso.slice(0, 10), time: time.endsWith(".000") ? time.slice(0, 8) : time };
}

/** "YYYY-MM-DD HH:MM:SS–HH:MM:SS UTC" (the end repeats the date only when it differs). */
export function fmtUtcWindow(start: number, end: number): string {
  const a = utcParts(start);
  const b = utcParts(end);
  return a.date === b.date ? `${a.date} ${a.time}–${b.time} UTC` : `${a.date} ${a.time} – ${b.date} ${b.time} UTC`;
}

/** Hover text: channel, UTC window, the event origin (hero clip), the manifest's note and source. */
export function listenTitle(m: ListenManifest): string {
  const parts = [`${m.stationId} channel ${m.channel}`, fmtUtcWindow(m.startUtc, m.endUtc)];
  if (m.eventOriginUtc !== null) {
    const o = utcParts(m.eventOriginUtc);
    parts.push(`event origin ${o.time} UTC`);
  }
  let text = parts.join(" · ");
  if (m.note) text += `. ${m.note}`;
  if (m.source) text += ` Source: ${m.source}.`;
  return text;
}

// ---- Loading -------------------------------------------------------------------------------------

const pending = new Map<string, Promise<ListenManifest | null>>();
const settled = new Map<string, ListenManifest | null>();

/** A clip's manifest, fetched at most once per page. Never rejects: failures resolve to null. */
export function loadListenManifest(
  name: string,
  fetchImpl: FetchLike = defaultFetch,
  base: string = AUDIO_BASE_URL,
): Promise<ListenManifest | null> {
  const key = `${base}/${name}`;
  let p = pending.get(key);
  if (!p) {
    p = fetchJson<unknown>(fetchImpl, `${key}.json`, { notFoundAsNull: true })
      .then((raw) => {
        if (raw === null) return null;
        const m = parseListenManifest(raw, base);
        if (!m) console.warn(`[listen] ${key}.json is not a usable audio manifest; the Listen button stays hidden`);
        return m;
      })
      .catch((err: unknown) => {
        console.warn(`[listen] no audio manifest: ${err instanceof Error ? err.message : String(err)}`);
        return null;
      })
      .then((m) => {
        settled.set(key, m);
        return m;
      });
    pending.set(key, p);
  }
  return p;
}

/** Test-only: forget every loaded manifest. */
export function resetListenManifests(): void {
  pending.clear();
  settled.clear();
}

/** React view of `loadListenManifest(name)`: null until (and unless) a valid manifest loads. */
export function useListenManifest(name: string): ListenManifest | null {
  const [manifest, setManifest] = useState<ListenManifest | null>(() => settled.get(`${AUDIO_BASE_URL}/${name}`) ?? null);
  useEffect(() => {
    let live = true;
    void loadListenManifest(name).then((m) => {
      if (live) setManifest(m);
    });
    return () => {
      live = false;
    };
  }, [name]);
  return manifest;
}

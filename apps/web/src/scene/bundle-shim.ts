// TEMPORARY `useBundle` / `useEvidence`-shaped shim (docs/02 §6) until H4's
// `apps/web/src/providers/hooks.ts` lands. It only reads the bundle files H4 generates under
// public/data/<mode>/; it never invents data. Delete this file and point `scene/data.ts` at
// `../providers/hooks` then.

import { useEffect, useState, useSyncExternalStore } from "react";
import type { BundleState, EventEvidence, EvidenceState } from "./types";

type Ready = Extract<BundleState, { status: "ready" }>;

/** docs/02 §6: mode comes from `?mode=`, default `showcase`. */
export const DEFAULT_MODE = "showcase";

/** The data mode for this page load (`?mode=`), or the default outside a browser. */
export function currentMode(): string {
  if (typeof window === "undefined") return DEFAULT_MODE;
  return new URLSearchParams(window.location.search).get("mode") ?? DEFAULT_MODE;
}

const cache = new Map<string, Promise<Ready>>();

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

function load(mode: string): Promise<Ready> {
  let p = cache.get(mode);
  if (!p) {
    const base = `/data/${mode}`;
    p = Promise.all([
      getJson<Ready["meta"]>(`${base}/meta.json`),
      getJson<Ready["stations"]>(`${base}/stations.json`),
      getJson<Ready["catalog"]>(`${base}/catalog.json`),
      getJson<Ready["events"]>(`${base}/events.json`),
      getJson<Ready["features"]>(`${base}/features.json`),
    ]).then(([meta, stations, catalog, events, features]) => {
      // docs/02 §6 / H4: the provider preloads the hero's evidence, so E opens the drawer from cache.
      const hero = meta.scene.heroEventId;
      if (hero) preloadEvidence(mode, hero);
      return {
        status: "ready" as const,
        info: {
          mode: meta.mode,
          label: mode,
          runId: meta.scene.runId,
          generatedAt: 0,
          isSynthetic: Boolean(meta.scene.isSynthetic),
        },
        meta,
        stations,
        catalog,
        events,
        features,
      };
    });
    cache.set(mode, p);
  }
  return p;
}

export function useBundle(): BundleState {
  const [state, setState] = useState<BundleState>({ status: "loading" });
  useEffect(() => {
    const mode = currentMode();
    let live = true;
    load(mode).then(
      (ready) => live && setState(ready),
      (err: unknown) => live && setState({ status: "error", message: String(err) }),
    );
    return () => {
      live = false;
    };
  }, []);
  return state;
}

// ---- Evidence --------------------------------------------------------------------------------
//
// One memoized entry per (mode, eventId), shared by every component. Components read it through
// useSyncExternalStore, so an entry that is already "ready" (preloaded, or opened before) renders on
// the very first render after select(): no loading frame, no effect round trip. A failed fetch stays
// "error" until that event is selected again, which retries it.

const IDLE: EvidenceState = Object.freeze({ status: "idle" });
const LOADING: EvidenceState = Object.freeze({ status: "loading" });

const evidenceEntries = new Map<string, EvidenceState>();
const evidenceListeners = new Set<() => void>();

function evidenceKey(mode: string, eventId: string): string {
  return `${mode}\n${eventId}`;
}

function setEvidenceEntry(key: string, state: EvidenceState): void {
  evidenceEntries.set(key, state);
  for (const l of evidenceListeners) l();
}

function subscribeEvidence(listener: () => void): () => void {
  evidenceListeners.add(listener);
  return () => {
    evidenceListeners.delete(listener);
  };
}

/** URL of one event's evidence file in a mode's bundle (docs/01 → Data bundle). */
export function evidenceUrl(mode: string, eventId: string): string {
  return `/data/${mode}/evidence/${encodeURIComponent(eventId)}.json`;
}

/**
 * Shape checks that keep a wrong or truncated file from reaching the drawer. Fails loudly: the drawer
 * shows the message instead of drawing another event's traces.
 */
export function checkEvidence(json: unknown, eventId: string): EventEvidence {
  if (typeof json !== "object" || json === null) throw new Error(`evidence ${eventId}: not a JSON object`);
  const ev = json as Partial<EventEvidence>;
  if (ev.eventId !== eventId) {
    throw new Error(`evidence ${eventId}: file is for ${JSON.stringify(ev.eventId)}`);
  }
  if (!Array.isArray(ev.traces)) throw new Error(`evidence ${eventId}: traces is not a list`);
  if (!Array.isArray(ev.filterHz) || ev.filterHz.length !== 2) {
    throw new Error(`evidence ${eventId}: filterHz is not a [low, high] pair`);
  }
  return ev as EventEvidence;
}

/**
 * Starts fetching one event's evidence unless it is already cached or in flight; a cached error is
 * retried. Safe to call any number of times.
 */
export function preloadEvidence(mode: string, eventId: string): void {
  const key = evidenceKey(mode, eventId);
  const current = evidenceEntries.get(key);
  if (current && current.status !== "error") return;
  setEvidenceEntry(key, LOADING);
  getJson<unknown>(evidenceUrl(mode, eventId))
    .then((json) => checkEvidence(json, eventId))
    .then(
      (evidence) => setEvidenceEntry(key, Object.freeze({ status: "ready" as const, evidence })),
      (err: unknown) =>
        setEvidenceEntry(
          key,
          Object.freeze({ status: "error" as const, message: err instanceof Error ? err.message : String(err) }),
        ),
    );
}

/** docs/02 §6 `useEvidence`: "idle" for null, else "loading" → "ready" | "error", memoized per event. */
export function useEvidence(eventId: string | null): EvidenceState {
  const mode = currentMode();
  const key = eventId === null ? null : evidenceKey(mode, eventId);
  const state = useSyncExternalStore(
    subscribeEvidence,
    () => (key === null ? IDLE : (evidenceEntries.get(key) ?? LOADING)),
    () => IDLE,
  );
  useEffect(() => {
    if (eventId !== null) preloadEvidence(mode, eventId);
  }, [mode, eventId]);
  return state;
}

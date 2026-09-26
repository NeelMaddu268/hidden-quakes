// TEMPORARY `useBundle`-shaped shim (docs/02 §6) until H4's `apps/web/src/providers/hooks.ts` lands.
// It only reads the bundle files H4 generates under public/data/<mode>/; it never invents data.
// Delete this file and point `scene/data.ts` at `../providers/hooks` then.

import { useEffect, useState } from "react";
import type { BundleState } from "./types";

type Ready = Extract<BundleState, { status: "ready" }>;

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
    ]).then(([meta, stations, catalog, events, features]) => ({
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
    }));
    cache.set(mode, p);
  }
  return p;
}

export function useBundle(): BundleState {
  const [state, setState] = useState<BundleState>({ status: "loading" });
  useEffect(() => {
    const mode = new URLSearchParams(window.location.search).get("mode") ?? "showcase";
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

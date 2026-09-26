/** Provider knobs. Not UI copy: nothing here is rendered. */

/** Evidence files fetched ahead of time once the bundle is ready: the hero plus this many by revealOrder. */
export const EVIDENCE_PRELOAD_COUNT = 20;

/** Live mode refreshes status and events (`/api/live/status` + `/events`) this often. */
export const LIVE_POLL_MS = 60_000;

/**
 * Live mode also probes `/api/live/status` this often, so a worker that goes away is noticed
 * within one heartbeat rather than one poll (API-05: "killing the API flips the label within
 * 5 s", which leaves the rest of that budget for the failed request and the snapshot load).
 * The response is a few fields; the cost is one small request per heartbeat.
 */
export const LIVE_HEARTBEAT_MS = 2_000;

/** A live request that has not answered after this long counts as failed and fails over. */
export const LIVE_FETCH_TIMEOUT_MS = 3_000;

/** Where static bundles live, relative to the site root (docs/01 → Data bundle). */
export const DATA_BASE_URL = "/data";

/** Same-origin live worker routes (docs/02 §7); the build flag below overrides it. */
export const DEFAULT_LIVE_API_BASE = "/api/live";

/**
 * Live worker routes. `NEXT_PUBLIC_LIVE_API_BASE` (inlined by Next at build time, like the other
 * `NEXT_PUBLIC_*` flags) points a static export at a cross-origin worker, e.g.
 * `http://127.0.0.1:8000/api/live`; the worker's `serve.corsOrigins` must then allow the site.
 * Unset or empty keeps the same-origin default. A trailing slash is dropped so routes join cleanly.
 */
export const LIVE_API_BASE = liveApiBase(process.env.NEXT_PUBLIC_LIVE_API_BASE);

export function liveApiBase(flag: string | undefined): string {
  const value = flag?.trim() ?? "";
  return value === "" ? DEFAULT_LIVE_API_BASE : value.replace(/\/+$/, "");
}

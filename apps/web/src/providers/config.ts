/** Provider knobs. Not UI copy: nothing here is rendered. */

/** Evidence files fetched ahead of time once the bundle is ready: the hero plus this many by revealOrder. */
export const EVIDENCE_PRELOAD_COUNT = 20;

/** Live mode polls `/api/live/status` this often. */
export const LIVE_POLL_MS = 60_000;

/** Where static bundles live, relative to the site root (docs/01 → Data bundle). */
export const DATA_BASE_URL = "/data";

/** Live worker routes (docs/02 §7). */
export const LIVE_API_BASE = "/api/live";

/** Build flags the shell reads. `NEXT_PUBLIC_*` values are inlined by Next at build time. */

/** API-04 flips this on once the live worker runs; until then there is no LIVE pill. */
export function liveEnabled(): boolean {
  return process.env.NEXT_PUBLIC_LIVE_ENABLED === "1";
}

/**
 * WEB-12: `next.config.ts` sets this when `public/data/snapshot/meta.json` exists at build time,
 * so the TONIGHT pill (`?mode=snapshot`) is only offered when the export actually carries a
 * snapshot bundle.
 */
export function snapshotAvailable(): boolean {
  return process.env.NEXT_PUBLIC_SNAPSHOT_AVAILABLE === "1";
}

/** Build flags the shell reads. `NEXT_PUBLIC_*` values are inlined by Next at build time. */

/** API-04 flips this on once the live worker runs; until then there is no LIVE pill. */
export function liveEnabled(): boolean {
  return process.env.NEXT_PUBLIC_LIVE_ENABLED === "1";
}

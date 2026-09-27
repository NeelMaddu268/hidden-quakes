/**
 * The event list's rows (DEMO-04 item 2): pure functions of the provider's `SeismicEvent[]`, so
 * sorting and filtering are tested without React. Every value comes from the event record.
 */
import type { SeismicEvent, Tier } from "@/providers";

export type SortKey = "time" | "depth" | "magnitude" | "tier" | "stations" | "catalog";
export type SortDir = "asc" | "desc";
export type TierFilter = Tier | "all";

export interface EventRow {
  id: string;
  t: number;
  timeUtc: string; // "HH:MM:SS"
  depthKm: number;
  magnitude: number | null;
  magnitudeType: string | null;
  tier: Tier;
  stations: number;
  inCatalog: boolean;
}

const TIER_RANK: Record<Tier, number> = { A: 0, B: 1, C: 2 };

export function toRow(event: SeismicEvent): EventRow {
  return {
    id: event.id,
    t: event.t,
    timeUtc: new Date(Math.round(event.t * 1000)).toISOString().slice(11, 19),
    depthKm: event.depthKm,
    magnitude: event.magnitude ? event.magnitude.value : null,
    magnitudeType: event.magnitude ? event.magnitude.type : null,
    tier: event.tier,
    stations: event.quality.nStations,
    inCatalog: event.catalogMatch !== null && event.catalogMatch !== undefined,
  };
}

/** The default order: tier (strict first), then most stations, then earliest, then id. */
function defaultCompare(a: EventRow, b: EventRow): number {
  return TIER_RANK[a.tier] - TIER_RANK[b.tier] || b.stations - a.stations || a.t - b.t || (a.id < b.id ? -1 : 1);
}

function keyValue(row: EventRow, key: SortKey): number {
  switch (key) {
    case "time":
      return row.t;
    case "depth":
      return row.depthKm;
    case "magnitude":
      return row.magnitude ?? Number.NEGATIVE_INFINITY;
    case "tier":
      return TIER_RANK[row.tier];
    case "stations":
      return row.stations;
    case "catalog":
      return row.inCatalog ? 0 : 1;
  }
}

/**
 * Filter by tier, then sort. `sort === null` is the default order; otherwise the chosen column,
 * with the default order breaking ties so the list never shuffles. Missing magnitudes sort last
 * in either direction.
 */
export function sortRows(rows: readonly EventRow[], sort: { key: SortKey; dir: SortDir } | null, tier: TierFilter): EventRow[] {
  const kept = tier === "all" ? rows.slice() : rows.filter((r) => r.tier === tier);
  if (sort === null) return kept.sort(defaultCompare);
  const sign = sort.dir === "asc" ? 1 : -1;
  return kept.sort((a, b) => {
    if (sort.key === "magnitude" && (a.magnitude === null) !== (b.magnitude === null)) {
      return a.magnitude === null ? 1 : -1;
    }
    const d = keyValue(a, sort.key) - keyValue(b, sort.key);
    return d !== 0 ? sign * d : defaultCompare(a, b);
  });
}

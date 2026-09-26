/**
 * "Download candidate catalog" (overnight feature 2, Sat): CSV and GeoJSON built in the browser
 * from the provider's `events` and `meta`, so the visitor leaves with the same records the scene
 * draws. No new data file is exported; every value comes from the bundle. Both formats open with
 * the run id and the caveat that these are candidate events, not a confirmed catalog.
 *
 * Pure functions here; `download.ts` does the Blob dance. Column order and field names are
 * stable so a downloaded file can be diffed across runs.
 */
import type { BundleMeta, SeismicEvent } from "@/providers";

export const CAVEAT =
  "Candidate events from public seismic waveforms, not a verified earthquake catalog: each needs consistent picks across stations, and only the STRICT tier meets every quality bar of the recovered public events.";

/** Column names, in file order. `depth_km` is below the site surface (`meta.scene.depthLabel`). */
export const COLUMNS = [
  "id",
  "time_utc",
  "latitude",
  "longitude",
  "depth_km",
  "elev_m_asl",
  "tier",
  "magnitude",
  "magnitude_type",
  "stations",
  "catalog_match",
] as const;

export type Column = (typeof COLUMNS)[number];

/** One event's cells, in `COLUMNS` order; a missing value is an empty string, never "null". */
export function catalogRow(event: SeismicEvent): Record<Column, string> {
  return {
    id: event.id,
    time_utc: isoUtc(event.t),
    latitude: num(event.latitude),
    longitude: num(event.longitude),
    depth_km: num(event.depthKm),
    elev_m_asl: num(event.elevM),
    tier: event.tier,
    magnitude: event.magnitude ? num(event.magnitude.value) : "",
    magnitude_type: event.magnitude ? event.magnitude.type : "",
    stations: num(event.quality.nStations),
    catalog_match: event.catalogMatch ? event.catalogMatch.catalogId : "",
  };
}

/** Epoch seconds → ISO 8601 UTC with millisecond precision. */
export function isoUtc(epochS: number): string {
  return new Date(Math.round(epochS * 1000)).toISOString();
}

function num(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? String(value) : "";
}

/** RFC 4180 quoting for any cell holding a comma, quote or newline. */
export function csvCell(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

/**
 * The CSV: `#` comment lines with the run id, the caveat and the depth convention, then the
 * header row and one row per event in reveal order (the bundle's order). Synthetic bundles are
 * marked in the first line so a mock download can never pass for data.
 */
export function catalogCsv(events: readonly SeismicEvent[], meta: BundleMeta): string {
  const lines = headerLines(meta).map((line) => `# ${line}`);
  lines.push(COLUMNS.join(","));
  for (const event of events) {
    const row = catalogRow(event);
    lines.push(COLUMNS.map((column) => csvCell(row[column])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

/**
 * The GeoJSON FeatureCollection: WGS84 points (longitude, latitude, elevation in metres above
 * sea level per the GeoJSON spec), the same properties as the CSV columns, and the run id and
 * caveat as foreign members of the collection (`metadata`).
 */
export function catalogGeoJson(events: readonly SeismicEvent[], meta: BundleMeta): string {
  const collection = {
    type: "FeatureCollection",
    metadata: {
      runId: meta.scene.runId,
      caveat: CAVEAT,
      depthLabel: meta.scene.depthLabel,
      isSynthetic: isSynthetic(meta),
      mode: meta.mode,
    },
    features: events.map((event) => {
      const row = catalogRow(event);
      return {
        type: "Feature",
        id: event.id,
        geometry: { type: "Point", coordinates: [event.longitude, event.latitude, event.elevM] },
        properties: {
          time_utc: row.time_utc,
          depth_km: event.depthKm,
          tier: event.tier,
          magnitude: event.magnitude ? event.magnitude.value : null,
          magnitude_type: event.magnitude ? event.magnitude.type : null,
          stations: event.quality.nStations,
          catalog_match: event.catalogMatch ? event.catalogMatch.catalogId : null,
        },
      };
    }),
  };
  return `${JSON.stringify(collection, null, 1)}\n`;
}

export function isSynthetic(meta: BundleMeta): boolean {
  return Boolean(meta.scene.isSynthetic || meta.run.isSynthetic);
}

/** The lines both formats open with: run id, caveat, depth convention, synthetic marker. */
export function headerLines(meta: BundleMeta): string[] {
  const lines = [`Hidden Quakes candidate events, run ${meta.scene.runId}`];
  if (isSynthetic(meta)) lines.unshift("SYNTHETIC DATA from the mock fixture: not a catalog, not for any use");
  lines.push(CAVEAT);
  lines.push(`depth_km is ${meta.scene.depthLabel}; elev_m_asl is metres above sea level.`);
  return lines;
}

/** The file name: the run id, and a synthetic marker so a mock download announces itself. */
export function catalogFileName(meta: BundleMeta, format: "csv" | "geojson"): string {
  const prefix = isSynthetic(meta) ? "synthetic-" : "";
  return `${prefix}hidden-quakes-candidates-${meta.scene.runId}.${format}`;
}

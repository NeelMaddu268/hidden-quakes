"use client";

import { useBundle } from "@/providers";
import { useDemo } from "@/state/demo";
import shell from "../Shell.module.css";
import styles from "./Download.module.css";
import { catalogCsv, catalogFileName, catalogGeoJson } from "./catalog";
import { saveTextFile } from "./download";

const FORMATS = [
  { format: "csv", label: "CSV", mime: "text/csv", build: catalogCsv },
  { format: "geojson", label: "GeoJSON", mime: "application/geo+json", build: catalogGeoJson },
] as const;

/**
 * "Download candidate catalog": two small pills, CSV and GeoJSON, built in the browser from the
 * bundle already loaded (no fetch). Mounted once the reveal has started, next to the mode label,
 * so the pre-reveal frame keeps its five-second essentials.
 */
export function DownloadButton() {
  const bundle = useBundle();
  const phase = useDemo((s) => s.phase);
  if (bundle.status !== "ready" || phase === "public") return null;
  const { events, meta } = bundle;
  return (
    <div className={styles.group} role="group" aria-label="Download candidate catalog" data-testid="download-catalog">
      <span className={styles.label}>Download candidate catalog</span>
      {FORMATS.map(({ format, label, mime, build }) => (
        <button
          key={format}
          type="button"
          className={`${shell.pill} ${shell.modePill}`}
          data-testid={`download-${format}`}
          onClick={(event) => {
            event.currentTarget.blur();
            saveTextFile(catalogFileName(meta, format), mime, build(events, meta));
          }}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

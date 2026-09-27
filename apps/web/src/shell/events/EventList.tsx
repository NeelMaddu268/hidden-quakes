"use client";

import { memo, useMemo, useState } from "react";
import { fonts, numeric } from "@hq/visualization";
import { useBundle } from "@/providers";
import { useDemo } from "@/state/demo";
import shell from "../Shell.module.css";
import { formatNumber } from "../validation/format";
import { openEventAfterReveal } from "../share";
import styles from "./EventList.module.css";
import { sortRows, toRow, type EventRow, type SortDir, type SortKey, type TierFilter } from "./rows";

const PANEL_ID = "event-list";

const COLUMNS: readonly { key: SortKey; label: string }[] = [
  { key: "time", label: "Time (UTC)" },
  { key: "depth", label: "Depth, km" },
  { key: "magnitude", label: "Mag" },
  { key: "tier", label: "Tier" },
  { key: "stations", label: "Stations" },
  { key: "catalog", label: "Public catalog" },
];

const TIERS: readonly { value: TierFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "A", label: "Strict" },
  { value: "B", label: "Tier B" },
  { value: "C", label: "Tier C" },
];

/** Display precision (decimals shown, fixed so the column lines up), not a data threshold. */
const DECIMALS = { depth: 2, magnitude: 1 } as const;
const fixed = new Map<number, Intl.NumberFormat>();
function formatFixed(value: number, digits: number): string {
  let f = fixed.get(digits);
  if (!f) {
    f = new Intl.NumberFormat("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
    fixed.set(digits, f);
  }
  return f.format(value);
}

const monoStyle = { fontFamily: fonts.mono, ...numeric };

export function EventListButton({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      className={`${shell.pill} ${shell.modePill}`}
      aria-expanded={open}
      aria-controls={PANEL_ID}
      data-testid="event-list-button"
      onClick={(event) => {
        event.currentTarget.blur();
        onToggle();
      }}
    >
      Event list
    </button>
  );
}

/**
 * The candidate events as a table (DEMO-04 item 2): sortable by any column (default tier, then
 * stations), filtered by tier; a click opens that event's evidence drawer, after the reveal if it
 * hasn't played (`openEventAfterReveal`, the share-link path). Rows are plain DOM, memoised, so a
 * selection change re-renders two rows, not the list.
 */
export function EventList({ open, onClose }: { open: boolean; onClose: () => void }) {
  const bundle = useBundle();
  const selected = useDemo((s) => s.selectedEventId);
  const [sort, setSort] = useState<{ key: SortKey; dir: SortDir } | null>(null);
  const [tier, setTier] = useState<TierFilter>("all");
  const events = bundle.status === "ready" ? bundle.events : null;
  const allRows = useMemo(() => (events ? events.map(toRow) : []), [events]);
  const rows = useMemo(() => sortRows(allRows, sort, tier), [allRows, sort, tier]);
  if (!open || bundle.status !== "ready") return null;
  const depthLabel = bundle.meta.scene.depthLabel;

  const onSort = (key: SortKey) => {
    setSort((current) => {
      if (current === null || current.key !== key) return { key, dir: key === "time" || key === "tier" ? "asc" : "desc" };
      if (current.dir === "desc") return { key, dir: "asc" };
      return null; // third click: back to the default order
    });
  };

  return (
    <section id={PANEL_ID} className={styles.panel} aria-label="Candidate events" data-testid="event-list">
      <div className={styles.head}>
        <h2 className={styles.title}>Candidate events</h2>
        <span className={styles.count} style={monoStyle} data-testid="event-list-count">
          {formatNumber(rows.length, 0)}
        </span>
        <button type="button" className={`${shell.pill} ${shell.modePill}`} onClick={onClose}>
          Close
        </button>
      </div>
      <div className={styles.filters} role="group" aria-label="Tier filter">
        {TIERS.map(({ value, label }) => (
          <button
            key={value}
            type="button"
            className={`${shell.pill} ${shell.modePill}`}
            aria-pressed={tier === value}
            onClick={() => setTier(value)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              {COLUMNS.map(({ key, label }) => (
                <th key={key} scope="col" aria-sort={sort?.key === key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}>
                  <button
                    type="button"
                    className={styles.sort}
                    title={key === "depth" ? depthLabel : undefined}
                    onClick={() => onSort(key)}
                  >
                    {label}
                    {sort?.key === key ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <Row key={row.id} row={row} selected={row.id === selected} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

const Row = memo(function Row({ row, selected }: { row: EventRow; selected: boolean }) {
  return (
    <tr
      className={styles.row}
      data-selected={selected || undefined}
      data-tier={row.tier}
      data-testid={`event-row-${row.id}`}
      tabIndex={0}
      aria-selected={selected}
      onClick={() => openEventAfterReveal(row.id)}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          openEventAfterReveal(row.id);
        }
      }}
    >
      <td style={monoStyle}>{row.timeUtc}</td>
      <td style={monoStyle}>{formatFixed(row.depthKm, DECIMALS.depth)}</td>
      <td style={monoStyle} title={row.magnitudeType ?? undefined}>
        {row.magnitude === null ? "–" : formatFixed(row.magnitude, DECIMALS.magnitude)}
      </td>
      <td>{row.tier === "A" ? "Strict" : row.tier}</td>
      <td style={monoStyle}>{formatNumber(row.stations, 0)}</td>
      <td>{row.inCatalog ? "Yes" : "No"}</td>
    </tr>
  );
});

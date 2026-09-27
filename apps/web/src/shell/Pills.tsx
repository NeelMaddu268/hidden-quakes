"use client";

import type { MouseEvent } from "react";
import { navigateToMode, type DataMode } from "@/providers";
import { useDemo, type EventFilter } from "@/state/demo";
import { liveEnabled, snapshotAvailable } from "./env";
import styles from "./Shell.module.css";

const FILTERS: readonly { value: EventFilter; label: string }[] = [
  { value: "public", label: "Public" },
  { value: "all", label: "All" },
  { value: "strict", label: "Strict" },
];

/** PUBLIC / ALL / STRICT, under the counters once the reveal has started. */
export function FilterPills() {
  const filter = useDemo((s) => s.filter);
  const onClick = (value: EventFilter) => (event: MouseEvent<HTMLButtonElement>) => {
    event.currentTarget.blur();
    useDemo.getState().setFilter(value);
  };
  return (
    <div className={styles.filters} role="group" aria-label="Event filter">
      {FILTERS.map(({ value, label }) => (
        <button
          key={value}
          type="button"
          className={styles.pill}
          aria-pressed={filter === value}
          onClick={onClick(value)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

/** What a mode pill says when it is not the mode's own name. */
const PILL_LABELS: Partial<Record<DataMode, string>> = { snapshot: "tonight" };

/**
 * Data-mode switch, bottom-left and small. SHOWCASE always; TONIGHT (`?mode=snapshot`, WEB-12)
 * when the build carries a snapshot bundle or snapshot is the current mode; LIVE only when the
 * build set `NEXT_PUBLIC_LIVE_ENABLED=1` (API-04); MOCK only while mock is the current mode.
 * During live failover LIVE stays pressed and the label alone says Snapshot (API-05).
 * A switch is a full navigation (`navigateToMode`), so the demo restarts from its start frame.
 */
export function ModePills({ current }: { current: DataMode | null }) {
  const modes: DataMode[] = [];
  if (current === "mock") modes.push("mock");
  modes.push("showcase");
  if (current === "snapshot" || snapshotAvailable()) modes.push("snapshot");
  if (liveEnabled()) modes.push("live");
  return (
    <div className={styles.modes} role="group" aria-label="Data mode">
      {modes.map((mode) => (
        <button
          key={mode}
          type="button"
          className={`${styles.pill} ${styles.modePill}`}
          aria-pressed={current === mode}
          onClick={() => {
            if (mode !== current) navigateToMode(mode);
          }}
        >
          {PILL_LABELS[mode] ?? mode}
        </button>
      ))}
    </div>
  );
}

"use client";

import { useEffect, type ReactNode } from "react";
import { useBundle, useValidation } from "@/providers";
import shell from "../Shell.module.css";
import styles from "./RunDetails.module.css";
import { SweepPlot } from "./SweepPlot";

/** The toggle key. Free in docs/02 §6's map (Space, R, S, T, E, P, Esc); Esc also closes. */
export const RUN_DETAILS_KEY = "d";
const PANEL_ID = "run-details";

/** Small text button under the mode label, top-left; the overlay opens from it or from D. */
export function RunDetailsButton({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      className={`${shell.pill} ${shell.modePill} ${styles.button}`}
      aria-expanded={open}
      aria-controls={PANEL_ID}
      onClick={(event) => {
        event.currentTarget.blur();
        onToggle();
      }}
    >
      Run details
    </button>
  );
}

function isTextInput(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

/**
 * D toggles the panel (only while the bundle is ready); Esc closes it. Esc is also the store's
 * `select(null)` in `useKeyboard`, which already prevented default, so it is honoured regardless.
 */
export function useRunDetailsKey(open: boolean, setOpen: (open: boolean) => void, enabled: boolean): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
      if (isTextInput(event.target)) return;
      const key = event.key.toLowerCase();
      if (key === RUN_DETAILS_KEY && !event.defaultPrevented) {
        if (!enabled) return;
        event.preventDefault();
        setOpen(!open);
      } else if (key === "escape" && open) {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, setOpen, enabled]);
}

/**
 * The overlay: `ProcessingRun` verbatim as a folded key/value tree, then the association sweep.
 * Uses only `useBundle()` and `useValidation()`; renders nothing while closed or before the
 * bundle is ready.
 */
export function RunDetailsPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const bundle = useBundle();
  const validation = useValidation();
  if (!open || bundle.status !== "ready") return null;
  const run = bundle.meta.run;
  const sweep = validation?.sweep ?? [];

  return (
    <section id={PANEL_ID} className={styles.panel} role="dialog" aria-label="Run details" data-testid="run-details">
      <header className={styles.panelHeader}>
        <div>
          <h2 className={styles.panelTitle}>Run details</h2>
          <p className={styles.panelSubtitle} data-testid="run-details-id">
            {run.id}
          </p>
        </div>
        <button type="button" className={`${shell.pill} ${shell.modePill}`} onClick={onClose}>
          Close
        </button>
      </header>

      <h3 className={styles.sectionTitle}>Processing run</h3>
      <div className={styles.tree} data-testid="run-json">
        {Object.keys(run).map((key) => (
          <JsonNode key={key} name={key} value={run[key as keyof typeof run]} depth={0} />
        ))}
      </div>

      {sweep.length > 0 && (
        <>
          <h3 className={styles.sectionTitle}>Association sweep</h3>
          <SweepPlot sweep={sweep} />
        </>
      )}
    </section>
  );
}

function isContainer(value: unknown): value is Record<string, unknown> | unknown[] {
  return typeof value === "object" && value !== null;
}

/** One key of the tree. Containers fold (top level open); leaves print `JSON.stringify` verbatim. */
export function JsonNode({ name, value, depth }: { name: string; value: unknown; depth: number }): ReactNode {
  if (isContainer(value)) {
    const entries = Array.isArray(value) ? value.map((item, i) => [String(i), item] as const) : Object.entries(value);
    if (entries.length > 0) {
      return (
        <details className={styles.node} open={depth === 0}>
          <summary className={styles.summary}>
            <span className={styles.key}>{name}</span>
            <span className={styles.meta}>{Array.isArray(value) ? `[${entries.length}]` : `{${entries.length}}`}</span>
          </summary>
          <div className={styles.children}>
            {entries.map(([childName, child]) => (
              <JsonNode key={childName} name={childName} value={child} depth={depth + 1} />
            ))}
          </div>
        </details>
      );
    }
  }
  return (
    <div className={styles.leaf}>
      <span className={styles.key}>{name}</span>
      <span className={styles.value}>{JSON.stringify(value)}</span>
    </div>
  );
}

"use client";

import { useEffect, useRef, type Dispatch, type ReactNode, type SetStateAction } from "react";
import { useBundle, useValidation } from "@/providers";
import shell from "../Shell.module.css";
import { isTextInput } from "../useKeyboard";
import styles from "./RunDetails.module.css";
import { SweepPlot } from "./SweepPlot";

/** The toggle key. Free in docs/02 §6's map (Space, R, S, T, E, P, Esc); Esc also closes. */
export const RUN_DETAILS_KEY = "d";
const PANEL_ID = "run-details";

/**
 * Small text button under the mode label, top-left; the overlay opens from it or from D. When
 * the overlay closes (Close button or Esc), focus returns here, so keyboard users land where
 * they left.
 */
export function RunDetailsButton({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const ref = useRef<HTMLButtonElement>(null);
  const wasOpen = useRef(open);
  useEffect(() => {
    if (wasOpen.current && !open) ref.current?.focus();
    wasOpen.current = open;
  }, [open]);
  return (
    <button
      ref={ref}
      type="button"
      className={`${shell.pill} ${shell.modePill} ${styles.button}`}
      aria-expanded={open}
      aria-controls={PANEL_ID}
      onClick={onToggle}
    >
      Run details
    </button>
  );
}

/**
 * D toggles the panel (only while the bundle is ready); Esc closes it. Esc is also the store's
 * `select(null)` in `useKeyboard`, which already prevented default, so it is honoured regardless.
 */
export function useRunDetailsKey(setOpen: Dispatch<SetStateAction<boolean>>, enabled: boolean): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
      if (isTextInput(event.target)) return;
      const key = event.key.toLowerCase();
      if (key === RUN_DETAILS_KEY && !event.defaultPrevented) {
        if (!enabled) return;
        event.preventDefault();
        setOpen((open) => !open);
      } else if (key === "escape") {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [setOpen, enabled]);
}

/**
 * The overlay: `ProcessingRun` verbatim as a folded key/value tree, then the association sweep.
 * Uses only `useBundle()` and `useValidation()`; renders nothing while closed or before the
 * bundle is ready. Takes focus when it opens (the dialog itself, so the tree is scrollable and
 * Tab reaches Close first).
 */
export function RunDetailsPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const bundle = useBundle();
  const validation = useValidation();
  const ref = useRef<HTMLElement>(null);
  const visible = open && bundle.status === "ready";
  useEffect(() => {
    if (visible) ref.current?.focus();
  }, [visible]);
  if (!visible) return null;
  const run = bundle.meta.run;
  const sweep = validation?.sweep ?? [];

  return (
    <section
      ref={ref}
      id={PANEL_ID}
      className={styles.panel}
      role="dialog"
      aria-label="Run details"
      tabIndex={-1}
      data-testid="run-details"
    >
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

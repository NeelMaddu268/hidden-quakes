"use client";

import { useEffect, useRef, type Dispatch, type SetStateAction } from "react";
import shell from "../Shell.module.css";
import { isTextInput } from "../useKeyboard";
import styles from "./Help.module.css";

/** The toggle key: "?" (Shift + / on most layouts). Esc also closes. */
export const HELP_KEY = "?";
const PANEL_ID = "how-it-works";

/** Three steps, one line each (DEMO-04 item 1). Words only; the language rules of docs/00 apply. */
export const HOW_IT_WORKS: readonly { title: string; line: string }[] = [
  { title: "Public seismometers", line: "Continuous waveforms from the public seismic network around Milford, nothing private." },
  { title: "A neural network marks arrivals", line: "A pretrained picker marks the P and S wave arrivals at each station." },
  {
    title: "Located and graded",
    line: "Arrivals that agree across stations become candidate events, located underground and graded against the public regional catalog.",
  },
];

/** The presenter keys, in the order a visitor meets them. */
export const KEYS: readonly { key: string; action: string }[] = [
  { key: "Space", action: "Reveal, then strict events only, then replay over time" },
  { key: "G", action: "Guided tour (any key, click or scroll stops it)" },
  { key: "H", action: "Evidence for a strict event the public catalog doesn't list" },
  { key: "E", action: "Evidence for the strict event most stations agreed on" },
  { key: "P", action: "Plan view and back" },
  { key: "R", action: "Reset to the start" },
  { key: "D", action: "Run details: every setting of this run" },
  { key: "?", action: "This help" },
  { key: "Esc", action: "Close" },
];

/** "?" toggles the overlay; Esc closes it. Text fields and browser shortcuts win. */
export function useHelpKey(setOpen: Dispatch<SetStateAction<boolean>>): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
      if (isTextInput(event.target)) return;
      if (event.key === HELP_KEY && !event.defaultPrevented) {
        event.preventDefault();
        setOpen((open) => !open);
      } else if (event.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [setOpen]);
}

export function HelpButton({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      className={`${shell.pill} ${shell.modePill}`}
      aria-expanded={open}
      aria-controls={PANEL_ID}
      aria-keyshortcuts="?"
      title="How it works and the keys"
      data-testid="help-button"
      onClick={(event) => {
        event.currentTarget.blur();
        onToggle();
      }}
    >
      ?
    </button>
  );
}

/** One screen: how the catalog is made, then the keys. Renders nothing while closed. */
export function HelpOverlay({ open, onClose }: { open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    if (open) ref.current?.focus();
  }, [open]);
  if (!open) return null;
  return (
    <section
      ref={ref}
      id={PANEL_ID}
      className={styles.panel}
      role="dialog"
      aria-modal="false"
      aria-label="How it works"
      tabIndex={-1}
      data-testid="help-overlay"
    >
      <div className={styles.head}>
        <h2 className={styles.title}>How it works</h2>
        <button type="button" className={`${shell.pill} ${shell.modePill}`} onClick={onClose}>
          Close
        </button>
      </div>
      <ol className={styles.steps}>
        {HOW_IT_WORKS.map((step) => (
          <li key={step.title} className={styles.step}>
            <span className={styles.stepTitle}>{step.title}</span>
            <span className={styles.stepLine}>{step.line}</span>
          </li>
        ))}
      </ol>
      <p className={styles.caveat}>
        Every dot is a candidate event, not a verified earthquake. Strict ones pass every quality bar set by the public
        events we recovered. Click any dot to see the waveforms behind it.
      </p>
      <h3 className={styles.keysTitle}>Keys</h3>
      <dl className={styles.keys}>
        {KEYS.map(({ key, action }) => (
          <div key={key} className={styles.keyRow}>
            <dt>
              <kbd className={styles.kbd}>{key}</kbd>
            </dt>
            <dd className={styles.action}>{action}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

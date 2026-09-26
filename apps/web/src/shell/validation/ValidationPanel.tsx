"use client";

import { fonts, numeric } from "@hq/visualization";
import { useBundle, useValidation } from "@/providers";
import { useDemo } from "@/state/demo";
import { rows } from "./rows";
import styles from "./ValidationPanel.module.css";

const valueStyle = { fontFamily: fonts.mono, ...numeric };

/**
 * The tiny validation card, bottom-left above the mode pills (the drawer takes the right edge,
 * the scrubber the bottom centre). Mounted only after the reveal starts: the pre-reveal frame
 * stays down to its five-second essentials, and the evidence beat follows the reveal. Reads only
 * `AnalysisSummary` and `Validation`; every row hides itself when its source field is missing.
 */
export function ValidationPanel() {
  const bundle = useBundle();
  const validation = useValidation();
  const phase = useDemo((s) => s.phase);
  if (bundle.status !== "ready" || phase === "public") return null;

  const list = rows(bundle.meta.summary, validation);
  if (list.length === 0) return null;

  return (
    <section className={styles.panel} aria-label="Validation" data-testid="validation-panel">
      <h2 className={styles.heading}>Validation</h2>
      <dl className={styles.rows}>
        {list.map((row) => (
          <div key={row.id} className={styles.row} data-testid={`validation-row-${row.id}`}>
            <dt className={styles.label}>{row.label}</dt>
            <dd className={styles.value} style={valueStyle}>
              {row.value}
            </dd>
            {row.note ? (
              <dd className={styles.note} data-testid={`validation-note-${row.id}`}>
                {row.note}
              </dd>
            ) : null}
          </div>
        ))}
      </dl>
    </section>
  );
}

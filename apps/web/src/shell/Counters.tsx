"use client";

import { fonts, numeric } from "@hq/visualization";
import type { AnalysisSummary } from "@/providers";
import { useDemo } from "@/state/demo";
import { counterValues, formatCount } from "./counter-values";
import styles from "./Shell.module.css";

type Tone = "public" | "recovered" | "strict";

// JetBrains Mono with tabular figures, so digits don't jitter while the reveal counts up.
const valueStyle = { fontFamily: fonts.mono, ...numeric };

/** PUBLIC / RECOVERED / STRICT, top-right. The last two are dashes until the reveal fills them. */
export function Counters({ summary }: { summary: AnalysisSummary }) {
  const phase = useDemo((s) => s.phase);
  const revealProgress = useDemo((s) => s.revealProgress);
  const values = counterValues(summary, phase, revealProgress);
  return (
    <dl className={styles.counters} aria-label="Event counts">
      <Counter tone="public" label="Public" value={values.public} />
      <Counter tone="recovered" label="Recovered" value={values.recovered} />
      <Counter tone="strict" label="Strict" value={values.strict} />
    </dl>
  );
}

function Counter({ tone, label, value }: { tone: Tone; label: string; value: number | null }) {
  return (
    <div className={styles.counter} data-tone={tone} data-dim={value === null || undefined}>
      <dt className={styles.counterLabel}>{label}</dt>
      <dd className={styles.counterValue} style={valueStyle} data-testid={`counter-${tone}`}>
        {value === null ? "—" : formatCount(value)}
      </dd>
    </div>
  );
}

import type { AnalysisSummary } from "@/providers";
import type { DemoPhase } from "@/state/demo";

/**
 * The three top-right counters. `null` renders as a dimmed dash: the value isn't known to the
 * viewer yet, not zero. Reads only `AnalysisSummary` plus the store's phase and reveal clock
 * (docs/lanes/H3 → Demo store semantics → Reveal timing for the counter).
 */
export interface CounterValues {
  public: number;
  recovered: number | null;
  strict: number | null;
}

export function counterValues(
  summary: Pick<AnalysisSummary, "publicCatalogCount" | "candidateCount" | "strictQualityCount">,
  phase: DemoPhase,
  revealProgress: number,
): CounterValues {
  const { publicCatalogCount, candidateCount, strictQualityCount } = summary;
  switch (phase) {
    case "public":
      return { public: publicCatalogCount, recovered: null, strict: null };
    case "revealing":
      return {
        public: publicCatalogCount,
        recovered: Math.round(publicCatalogCount + (candidateCount - publicCatalogCount) * revealProgress),
        strict: null,
      };
    case "revealed":
      return { public: publicCatalogCount, recovered: candidateCount, strict: strictQualityCount };
  }
}

const countFormat = new Intl.NumberFormat("en-US");

/** Digit grouping for counters; the text is always derived from provider data. */
export function formatCount(n: number): string {
  return countFormat.format(n);
}

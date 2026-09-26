// Tour captions (WEB-09): the copy file's templates with their {placeholders} filled from the bundle.
// Every number on screen comes from here, so a caption can never disagree with the counters.

import type { AnalysisSummary } from "../types";

/** The placeholders the copy file may use, each filled from provider data. */
export const TOUR_PLACEHOLDERS = [
  "publicCatalogCount",
  "recoveredCatalogCount",
  "candidateCount",
  "additionalCount",
  "strictQualityCount",
  "strictAdditionalCount",
  "heroStations",
  "replayRate",
] as const;

export type TourPlaceholder = (typeof TOUR_PLACEHOLDERS)[number];

/** Filled values; null when the bundle has no such value (a caption that needs one is skipped). */
export type TourValues = Readonly<Record<TourPlaceholder, string | null>>;

/** How a filled value is drawn: public counts in the public color, pipeline counts in the accent. */
export type ValueTone = "public" | "recovered" | "plain";

export const PLACEHOLDER_TONE: Readonly<Record<TourPlaceholder, ValueTone>> = Object.freeze({
  publicCatalogCount: "public",
  recoveredCatalogCount: "recovered",
  candidateCount: "recovered",
  additionalCount: "recovered",
  strictQualityCount: "recovered",
  strictAdditionalCount: "recovered",
  heroStations: "recovered",
  replayRate: "plain",
});

const countFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** A count as the counters print it (grouped, whole). */
export function formatCount(n: number): string {
  return countFormat.format(n);
}

/** The replay speed in words, e.g. 3600 data s per real s → "1 hour per second". */
export function formatReplayRate(dataSecondsPerSecond: number): string {
  if (!(dataSecondsPerSecond > 0)) throw new Error(`formatReplayRate: rate must be > 0, got ${dataSecondsPerSecond}`);
  const units: readonly [number, string][] = [
    [86400, "day"],
    [3600, "hour"],
    [60, "minute"],
  ];
  for (const [seconds, unit] of units) {
    const n = dataSecondsPerSecond / seconds;
    if (n >= 1) {
      const shown = Number.isInteger(n) ? formatCount(n) : n.toFixed(1);
      return `${shown} ${unit}${n === 1 ? "" : "s"} per second`;
    }
  }
  return `${formatCount(dataSecondsPerSecond)}× real time`;
}

/** Every placeholder's value from the bundle's summary, the tour's hidden event and the playback rate. */
export function tourValues(
  summary: Pick<
    AnalysisSummary,
    | "publicCatalogCount"
    | "recoveredCatalogCount"
    | "candidateCount"
    | "additionalCount"
    | "strictQualityCount"
    | "strictAdditionalCount"
  >,
  heroStations: number | null,
  playbackRate: number,
): TourValues {
  return {
    publicCatalogCount: formatCount(summary.publicCatalogCount),
    recoveredCatalogCount: formatCount(summary.recoveredCatalogCount),
    candidateCount: formatCount(summary.candidateCount),
    additionalCount: formatCount(summary.additionalCount),
    strictQualityCount: formatCount(summary.strictQualityCount),
    strictAdditionalCount: formatCount(summary.strictAdditionalCount),
    heroStations: heroStations !== null && Number.isFinite(heroStations) ? formatCount(heroStations) : null,
    replayRate: formatReplayRate(playbackRate),
  };
}

export type CaptionSegment =
  | { kind: "text"; text: string }
  | { kind: "value"; text: string; tone: ValueTone; name: TourPlaceholder };

const PLACEHOLDER = /\{([A-Za-z]+)\}/g;

function isPlaceholder(name: string): name is TourPlaceholder {
  return (TOUR_PLACEHOLDERS as readonly string[]).includes(name);
}

/** The placeholder names a template uses, in order. Throws on a name the tour can't fill. */
export function placeholdersIn(template: string): TourPlaceholder[] {
  const names: TourPlaceholder[] = [];
  for (const match of template.matchAll(PLACEHOLDER)) {
    const name = match[1]!;
    if (!isPlaceholder(name)) {
      throw new Error(`Tour copy: unknown placeholder {${name}} (known: ${TOUR_PLACEHOLDERS.join(", ")})`);
    }
    names.push(name);
  }
  return names;
}

/**
 * The template as text and value segments, or null when a placeholder it uses has no value in this
 * bundle (the step then plays without a caption rather than with a hole in it). Throws on an unknown
 * placeholder: the copy file is wrong, and copy.test.ts catches it before it ships.
 */
export function fillCaption(template: string, values: TourValues): CaptionSegment[] | null {
  const segments: CaptionSegment[] = [];
  let last = 0;
  for (const match of template.matchAll(PLACEHOLDER)) {
    const name = match[1]!;
    if (!isPlaceholder(name)) {
      throw new Error(`Tour copy: unknown placeholder {${name}} (known: ${TOUR_PLACEHOLDERS.join(", ")})`);
    }
    const value = values[name];
    if (value === null) return null;
    if (match.index > last) segments.push({ kind: "text", text: template.slice(last, match.index) });
    segments.push({ kind: "value", text: value, tone: PLACEHOLDER_TONE[name], name });
    last = match.index + match[0].length;
  }
  if (last < template.length) segments.push({ kind: "text", text: template.slice(last) });
  return segments;
}

/** The caption as plain text (for aria-live and tests). */
export function captionText(segments: readonly CaptionSegment[]): string {
  return segments.map((s) => s.text).join("");
}

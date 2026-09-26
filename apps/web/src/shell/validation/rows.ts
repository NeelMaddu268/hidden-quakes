/**
 * The validation panel's rows (docs/lanes/H4 → Validation panel). A pure function of
 * `AnalysisSummary` and `Validation`: a row exists only when its source field is a finite number,
 * so a missing file, a null field or a hand-edited bundle hides the row instead of rendering
 * "NaN". Labels are words; every digit in a value comes from the data.
 */
import type { AnalysisSummary, BaselineRow, Validation } from "@/providers";
import { formatNumber, isFiniteNumber } from "./format";

type Nullable<T> = { [K in keyof T]?: T[K] | null };

/** `AnalysisSummary` with any field possibly missing or null (the panel never trusts the shape). */
export type SummaryInput = Nullable<AnalysisSummary> | null | undefined;

/** `Validation` likewise; `synthetic` itself may be null or have null fields. */
export type ValidationInput =
  | (Omit<Nullable<Validation>, "synthetic"> & { synthetic?: Nullable<Validation["synthetic"]> | null })
  | null
  | undefined;

export type RowId = "recall" | "strict" | "stations" | "residual" | "depth" | "stalta" | "gain" | "chance";

export interface ValidationRow {
  id: RowId;
  label: string;
  value: string;
  /** One short line under the row, for a value that must not be read against another row unqualified. */
  note?: string;
}

/** Display precision per row (decimals shown), not a data threshold. */
const DECIMALS = { count: 0, stations: 1, residual: 3, depth: 0, gain: 2, chance: 1 } as const;

/** Both association profiles of the contract (`BaselineRow.associationProfile`). */
const ASSOCIATION_PROFILES = ["full", "p_only"] as const satisfies readonly BaselineRow["associationProfile"][];

/** The profile the STA/LTA row is quoted on (the one `BaselineGain` is quoted on too). */
const COMPARISON_PROFILE = "full" satisfies BaselineRow["associationProfile"];

/**
 * The STA/LTA baseline's strict (Tier A) count and its candidate count, from the `full` profile
 * row, when both are finite. The card shows "A of N candidates" (H2's card copy, Sat evening):
 * the same downstream code and tier bars, an energy-ratio picker instead of PhaseNet. It never
 * shows the rerun's PhaseNet strict count, which is not the STRICT counter (the rerun applies
 * one statics table to every event; the published run leaves each matched event out of its own).
 */
export function staltaStrict(
  rows: readonly Nullable<BaselineRow>[] | null | undefined,
): { strict: number; candidates: number } | null {
  if (!rows) return null;
  const row = rows.find((r) => r.method === "stalta" && r.associationProfile === COMPARISON_PROFILE);
  const strict = row?.tiers?.A;
  const candidates = row?.candidates;
  if (!isFiniteNumber(strict) || !isFiniteNumber(candidates)) return null;
  return { strict, candidates };
}

/**
 * "The baseline ran": a presence check only. The table has a PhaseNet row and an STA/LTA row for
 * both association profiles. Used by the gain row: the exporter writes `summary.baseline` only
 * then, and the UI never recomputes the ratio.
 */
export function baselineRan(rows: readonly Nullable<BaselineRow>[] | null | undefined): boolean {
  if (!rows) return false;
  return ASSOCIATION_PROFILES.every(
    (profile) =>
      rows.some((row) => row.method === "phasenet" && row.associationProfile === profile) &&
      rows.some((row) => row.method === "stalta" && row.associationProfile === profile),
  );
}

/** The rows to show, in the lane doc's order; absent sources yield no row. */
/**
 * The line under "Chance associations": the value is a mean over the null test's timing
 * scrambles, and how many of those reached the strict tier is the other half of the claim
 * (H2, Sat evening). Every number comes from `validation.nullTest`; a missing field drops its
 * clause rather than guessing.
 */
export function chanceNote(nullTest: Nullable<NonNullable<Validation["nullTest"]>> | null | undefined): string {
  const n = nullTest?.nShuffles;
  const strict = nullTest?.meanChanceStrict;
  const scrambles = isFiniteNumber(n) ? `Mean of ${formatNumber(n, DECIMALS.count)} timing scrambles` : "Mean over timing scrambles";
  if (!isFiniteNumber(strict)) return scrambles;
  if (strict === 0) return `${scrambles}; none reached the strict tier`;
  return `${scrambles}; about ${formatNumber(strict, DECIMALS.chance)} per scramble reached the strict tier`;
}

export function rows(summary: SummaryInput, validation: ValidationInput): ValidationRow[] {
  const s = summary ?? {};
  const v = validation ?? {};
  const out: ValidationRow[] = [];

  if (isFiniteNumber(s.recoveredCatalogCount) && isFiniteNumber(s.publicCatalogCount)) {
    out.push({
      id: "recall",
      label: "Catalog recall",
      value: `${formatNumber(s.recoveredCatalogCount, DECIMALS.count)} / ${formatNumber(s.publicCatalogCount, DECIMALS.count)}`,
    });
  }
  // "32 (14 not in public catalog)": the additional strict count rides along when it is present.
  if (isFiniteNumber(s.strictQualityCount)) {
    const strict = formatNumber(s.strictQualityCount, DECIMALS.count);
    const additional = isFiniteNumber(s.strictAdditionalCount)
      ? ` (${formatNumber(s.strictAdditionalCount, DECIMALS.count)} not in public catalog)`
      : "";
    out.push({ id: "strict", label: "Strict events", value: `${strict}${additional}` });
  }
  if (isFiniteNumber(s.medianStations)) {
    out.push({ id: "stations", label: "Median stations", value: formatNumber(s.medianStations, DECIMALS.stations) });
  }
  if (isFiniteNumber(s.medianRmsS)) {
    out.push({ id: "residual", label: "Median timing misfit", value: `${formatNumber(s.medianRmsS, DECIMALS.residual)} s` });
  }
  const medianVErrM = v.synthetic?.medianVErrM;
  if (isFiniteNumber(medianVErrM)) {
    // The synthetic test records every event on every station; a typical candidate has fewer.
    out.push({
      id: "depth",
      label: "Depth resolution (synthetic, all stations)",
      value: `±${formatNumber(medianVErrM, DECIMALS.depth)} m`,
    });
  }
  // The STA/LTA baseline from the table itself, so the row also shows when STA/LTA has no strict
  // event at all and no gain exists (REQ-H1-5).
  const stalta = staltaStrict(v.baseline);
  if (stalta) {
    out.push({
      id: "stalta",
      label: "STA/LTA strict events",
      value: `${formatNumber(stalta.strict, DECIMALS.count)} of ${formatNumber(stalta.candidates, DECIMALS.count)} candidates`,
    });
  }
  // Lane doc: "Baseline ran and gain > 1 in both profiles". The value is summary.baseline.gain;
  // "ran" is the presence of the four baseline rows; "both profiles" is the exporter's guarantee.
  const gain = s.baseline?.gain;
  if (isFiniteNumber(gain) && gain > 1 && baselineRan(v.baseline)) {
    out.push({ id: "gain", label: "PhaseNet vs STA/LTA gain", value: `${formatNumber(gain, DECIMALS.gain)}×` });
  }
  const meanChanceEvents = v.nullTest?.meanChanceEvents;
  if (isFiniteNumber(meanChanceEvents)) {
    out.push({
      id: "chance",
      label: "Chance associations",
      value: formatNumber(meanChanceEvents, DECIMALS.chance),
      note: chanceNote(v.nullTest),
    });
  }
  return out;
}

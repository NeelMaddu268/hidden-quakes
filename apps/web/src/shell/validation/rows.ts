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

export type RowId = "recall" | "strict" | "stations" | "residual" | "depth" | "strictCompare" | "gain" | "chance";

export interface ValidationRow {
  id: RowId;
  label: string;
  value: string;
}

/** Display precision per row (decimals shown), not a data threshold. */
const DECIMALS = { count: 0, stations: 1, residual: 3, depth: 0, gain: 2, chance: 1 } as const;

/** Both association profiles of the contract (`BaselineRow.associationProfile`). */
const ASSOCIATION_PROFILES = ["full", "p_only"] as const satisfies readonly BaselineRow["associationProfile"][];

/** The profile the strict-events comparison is quoted on (the one `BaselineGain` is quoted on too). */
const COMPARISON_PROFILE = "full" satisfies BaselineRow["associationProfile"];

/**
 * The strict (Tier A) counts of the `full` profile's PhaseNet and STA/LTA rows, when both rows
 * exist with a finite count; the comparison is shown from the table itself, so it survives when
 * no gain can be claimed (STA/LTA with no strict event at all, REQ-H1-5).
 */
export function strictComparison(
  rows: readonly Nullable<BaselineRow>[] | null | undefined,
): { phasenet: number; stalta: number } | null {
  if (!rows) return null;
  const strictOf = (method: BaselineRow["method"]): number | null => {
    const row = rows.find((r) => r.method === method && r.associationProfile === COMPARISON_PROFILE);
    const tierA = row?.tiers?.A;
    return isFiniteNumber(tierA) ? tierA : null;
  };
  const phasenet = strictOf("phasenet");
  const stalta = strictOf("stalta");
  if (phasenet === null || stalta === null) return null;
  return { phasenet, stalta };
}

/**
 * "The baseline ran": a presence check only. The table has a PhaseNet row and an STA/LTA row for
 * each association profile. Whether the gain holds in both profiles is VAL-01's call: the
 * exporter writes `summary.baseline` only then, and the UI never recomputes the ratio.
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
  if (isFiniteNumber(s.strictQualityCount)) {
    out.push({ id: "strict", label: "Strict events", value: formatNumber(s.strictQualityCount, DECIMALS.count) });
  }
  if (isFiniteNumber(s.medianStations)) {
    out.push({ id: "stations", label: "Median stations", value: formatNumber(s.medianStations, DECIMALS.stations) });
  }
  if (isFiniteNumber(s.medianRmsS)) {
    out.push({ id: "residual", label: "Median residual", value: `${formatNumber(s.medianRmsS, DECIMALS.residual)} s` });
  }
  const medianVErrM = v.synthetic?.medianVErrM;
  if (isFiniteNumber(medianVErrM)) {
    out.push({ id: "depth", label: "Depth resolution", value: `±${formatNumber(medianVErrM, DECIMALS.depth)} m` });
  }
  // Lane doc: the strict counts side by side whenever the full profile has both rows; the table
  // is the source, so the row also shows when STA/LTA's strict count is zero and no gain exists.
  const comparison = strictComparison(v.baseline);
  if (comparison) {
    out.push({
      id: "strictCompare",
      label: "Strict events, PhaseNet vs STA/LTA",
      value: `${formatNumber(comparison.phasenet, DECIMALS.count)} vs ${formatNumber(comparison.stalta, DECIMALS.count)}`,
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
    });
  }
  return out;
}

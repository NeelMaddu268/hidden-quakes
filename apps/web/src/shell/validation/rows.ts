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

export type RowId = "recall" | "strict" | "stations" | "residual" | "depth" | "gain" | "chance";

export interface ValidationRow {
  id: RowId;
  label: string;
  value: string;
}

/** Display precision per row (decimals shown), not a data threshold. */
const DECIMALS = { count: 0, stations: 1, residual: 3, depth: 0, gain: 2, chance: 1 } as const;

/** Both association profiles of the contract (`BaselineRow.associationProfile`). */
const ASSOCIATION_PROFILES = ["full", "p_only"] as const satisfies readonly BaselineRow["associationProfile"][];

/**
 * "Gain > 1 in both profiles", computed from the baseline table itself: for each association
 * profile, the PhaseNet row's Tier A count divided by the STA/LTA row's must exceed one. Either
 * row missing, or an STA/LTA count of zero (an unbounded ratio), means the claim is not supported.
 */
export function gainHoldsInBothProfiles(rows: readonly Nullable<BaselineRow>[] | null | undefined): boolean {
  if (!rows) return false;
  return ASSOCIATION_PROFILES.every((profile) => {
    const phasenet = rows.find((row) => row.method === "phasenet" && row.associationProfile === profile);
    const stalta = rows.find((row) => row.method === "stalta" && row.associationProfile === profile);
    const strictPhasenet = phasenet?.tiers?.A;
    const strictStalta = stalta?.tiers?.A;
    if (!isFiniteNumber(strictPhasenet) || !isFiniteNumber(strictStalta)) return false;
    return strictStalta > 0 && strictPhasenet / strictStalta > 1;
  });
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
  const gain = s.baseline?.gain;
  if (isFiniteNumber(gain) && gain > 1 && gainHoldsInBothProfiles(v.baseline)) {
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

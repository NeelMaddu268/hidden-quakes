/**
 * DEMO-02 acceptance: "every row hides itself when its source field is null". `rows()` is pure,
 * so each source field is nulled one at a time and the matching row, and only that row, goes.
 * Every number here is invented inside the test.
 */
import { describe, expect, it } from "vitest";
import type { AnalysisSummary, BaselineRow, Validation } from "@/providers";
import { gainHoldsInBothProfiles, rows, type RowId, type SummaryInput, type ValidationInput } from "./rows";

const SUMMARY: AnalysisSummary = {
  runId: "t",
  publicCatalogCount: 40,
  recoveredCatalogCount: 36,
  recall: 0.9,
  unmatchedPublicIds: [],
  candidateCount: 400,
  additionalCount: 364,
  additional: { A: 100, B: 150, C: 114 },
  strictQualityCount: 136,
  strictAdditionalCount: 100,
  medianStations: 7.5,
  medianRmsS: 0.0625,
  baseline: { associationProfile: "full", gain: 2.5, strictPhasenet: 100, strictStalta: 40 },
};

function baselineRow(method: BaselineRow["method"], profile: BaselineRow["associationProfile"], tierA: number): BaselineRow {
  return {
    method,
    associationProfile: profile,
    candidates: tierA * 3,
    recoveredPublic: 30,
    tiers: { A: tierA, B: tierA, C: tierA },
    medianRmsS: 0.1,
    medianStations: 6,
  };
}

const VALIDATION: Validation = {
  baseline: [
    baselineRow("phasenet", "full", 100),
    baselineRow("phasenet", "p_only", 70),
    baselineRow("stalta", "full", 40),
    baselineRow("stalta", "p_only", 35),
  ],
  sweep: [],
  nullTest: { nShuffles: 20, shiftRangeS: 30, meanChanceEvents: 1.25, meanChanceStrict: 0, stdChanceEvents: 0.5 },
  gr: null,
  magnitude: null,
  synthetic: { nEvents: 100, pickSigmaS: { P: 0.03, S: 0.06 }, medianHErrM: 150, medianVErrM: 240.4, p90VErrM: 500, medianDepthBiasM: -10 },
};

const ids = (summary: SummaryInput, validation: ValidationInput) => rows(summary, validation).map((r) => r.id);
const ALL: RowId[] = ["recall", "strict", "stations", "residual", "depth", "gain", "chance"];

describe("rows()", () => {
  it("renders every row, in the lane doc's order, from a complete summary and validation", () => {
    const list = rows(SUMMARY, VALIDATION);
    expect(list.map((r) => r.id)).toEqual(ALL);
    const byId = Object.fromEntries(list.map((r) => [r.id, r.value]));
    expect(byId.recall).toBe("36 / 40");
    expect(byId.strict).toBe("136");
    expect(byId.stations).toBe("7.5");
    expect(byId.residual).toBe("0.063 s");
    expect(byId.depth).toBe("±240 m");
    expect(byId.gain).toBe("2.5×");
    expect(byId.chance).toBe("1.3");
  });

  it("labels are words only", () => {
    for (const row of rows(SUMMARY, VALIDATION)) expect(row.label).not.toMatch(/\d/);
  });

  it.each<[keyof AnalysisSummary, RowId]>([
    ["recoveredCatalogCount", "recall"],
    ["publicCatalogCount", "recall"],
    ["strictQualityCount", "strict"],
    ["medianStations", "stations"],
    ["medianRmsS", "residual"],
    ["baseline", "gain"],
  ])("hides only its row when summary.%s is null", (field, rowId) => {
    const expected = ALL.filter((id) => id !== rowId);
    expect(ids({ ...SUMMARY, [field]: null }, VALIDATION)).toEqual(expected);
    const without = Object.fromEntries(Object.entries(SUMMARY).filter(([key]) => key !== field)) as SummaryInput;
    expect(ids(without, VALIDATION)).toEqual(expected);
  });

  it("treats NaN like a missing field", () => {
    expect(ids({ ...SUMMARY, medianRmsS: Number.NaN }, VALIDATION)).toEqual(ALL.filter((id) => id !== "residual"));
  });

  it("hides the depth row when synthetic.medianVErrM or synthetic itself is null", () => {
    const expected = ALL.filter((id) => id !== "depth");
    expect(ids(SUMMARY, { ...VALIDATION, synthetic: { ...VALIDATION.synthetic, medianVErrM: null } })).toEqual(expected);
    expect(ids(SUMMARY, { ...VALIDATION, synthetic: null })).toEqual(expected);
  });

  it("hides the chance row when the null test did not run", () => {
    expect(ids(SUMMARY, { ...VALIDATION, nullTest: null })).toEqual(ALL.filter((id) => id !== "chance"));
  });

  it("shows only the summary rows without validation.json", () => {
    expect(ids(SUMMARY, null)).toEqual(["recall", "strict", "stations", "residual"]);
    expect(ids(SUMMARY, undefined)).toEqual(["recall", "strict", "stations", "residual"]);
  });

  it("keeps only the validation-sourced rows without a summary, and none without either", () => {
    expect(ids(null, VALIDATION)).toEqual(["depth", "chance"]);
    expect(rows(undefined, null)).toEqual([]);
  });

  describe("the gain row needs gain > 1 in both profiles, computed from the baseline table", () => {
    const withoutGain = ALL.filter((id) => id !== "gain");

    it("hides when summary.baseline.gain is not above one", () => {
      expect(ids({ ...SUMMARY, baseline: { ...SUMMARY.baseline!, gain: 1 } }, VALIDATION)).toEqual(withoutGain);
      expect(ids({ ...SUMMARY, baseline: { ...SUMMARY.baseline!, gain: 0.8 } }, VALIDATION)).toEqual(withoutGain);
    });

    it("hides when the second profile is missing from the table", () => {
      const onlyFull = VALIDATION.baseline.filter((r) => r.associationProfile === "full");
      expect(ids(SUMMARY, { ...VALIDATION, baseline: onlyFull })).toEqual(withoutGain);
      expect(ids(SUMMARY, { ...VALIDATION, baseline: [] })).toEqual(withoutGain);
      expect(ids(SUMMARY, { ...VALIDATION, baseline: null })).toEqual(withoutGain);
    });

    it("hides when one profile's gain is not above one", () => {
      const tied = VALIDATION.baseline.map((r) =>
        r.method === "stalta" && r.associationProfile === "p_only" ? { ...r, tiers: { ...r.tiers, A: 70 } } : r,
      );
      expect(ids(SUMMARY, { ...VALIDATION, baseline: tied })).toEqual(withoutGain);
      const worse = VALIDATION.baseline.map((r) =>
        r.method === "phasenet" && r.associationProfile === "full" ? { ...r, tiers: { ...r.tiers, A: 10 } } : r,
      );
      expect(ids(SUMMARY, { ...VALIDATION, baseline: worse })).toEqual(withoutGain);
    });

    it("hides when the STA/LTA strict count is zero (an unbounded ratio is not a gain)", () => {
      const zero = VALIDATION.baseline.map((r) => (r.method === "stalta" ? { ...r, tiers: { ...r.tiers, A: 0 } } : r));
      expect(ids(SUMMARY, { ...VALIDATION, baseline: zero })).toEqual(withoutGain);
      expect(gainHoldsInBothProfiles(zero)).toBe(false);
    });

    it("holds on the complete table", () => {
      expect(gainHoldsInBothProfiles(VALIDATION.baseline)).toBe(true);
      expect(gainHoldsInBothProfiles(null)).toBe(false);
    });
  });
});

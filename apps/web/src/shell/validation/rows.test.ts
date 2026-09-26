/**
 * DEMO-02 acceptance: "every row hides itself when its source field is null". `rows()` is pure,
 * so each source field is nulled one at a time and the matching row, and only that row, goes.
 * Every number here is invented inside the test.
 */
import { describe, expect, it } from "vitest";
import type { AnalysisSummary, BaselineRow, Validation } from "@/providers";
import { baselineRan, chanceNote, rows, staltaStrict, type RowId, type SummaryInput, type ValidationInput } from "./rows";

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
const ALL: RowId[] = ["recall", "strict", "stations", "residual", "depth", "stalta", "gain", "chance"];

describe("rows()", () => {
  it("renders every row, in the lane doc's order, from a complete summary and validation", () => {
    const list = rows(SUMMARY, VALIDATION);
    expect(list.map((r) => r.id)).toEqual(ALL);
    const byId = Object.fromEntries(list.map((r) => [r.id, r.value]));
    expect(byId.recall).toBe("36 / 40");
    expect(byId.strict).toBe("136 (100 not in public catalog)");
    expect(byId.stations).toBe("7.5");
    expect(byId.residual).toBe("0.063 s");
    expect(byId.depth).toBe("±240 m");
    expect(byId.stalta).toBe("40 of 120 candidates");
    expect(byId.gain).toBe("2.5×");
    expect(byId.chance).toBe("1.3");
  });

  it("labels are words only", () => {
    for (const row of rows(SUMMARY, VALIDATION)) expect(row.label).not.toMatch(/\d/);
  });

  it("the strict row drops its parenthetical when the additional strict count is missing", () => {
    const value = (summary: SummaryInput) => rows(summary, VALIDATION).find((r) => r.id === "strict")?.value;
    expect(value({ ...SUMMARY, strictAdditionalCount: null })).toBe("136");
    expect(value({ ...SUMMARY, strictAdditionalCount: Number.NaN })).toBe("136");
    expect(value({ ...SUMMARY, strictQualityCount: null })).toBeUndefined();
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
    expect(ids(null, VALIDATION)).toEqual(["depth", "stalta", "chance"]);
    expect(rows(undefined, null)).toEqual([]);
  });

  describe("the gain row: summary.baseline.gain > 1 and the baseline ran (all four rows present)", () => {
    const withoutGain = ALL.filter((id) => id !== "gain");

    it("shows with summary.baseline and the full baseline table", () => {
      expect(ids(SUMMARY, VALIDATION)).toContain("gain");
      expect(baselineRan(VALIDATION.baseline)).toBe(true);
    });

    it("hides when summary.baseline.gain is not above one", () => {
      expect(ids({ ...SUMMARY, baseline: { ...SUMMARY.baseline!, gain: 1 } }, VALIDATION)).toEqual(withoutGain);
      expect(ids({ ...SUMMARY, baseline: { ...SUMMARY.baseline!, gain: 0.8 } }, VALIDATION)).toEqual(withoutGain);
    });

    it("hides when a profile's row is missing from the table, or validation is missing", () => {
      const onlyFull = VALIDATION.baseline.filter((r) => r.associationProfile === "full");
      expect(ids(SUMMARY, { ...VALIDATION, baseline: onlyFull })).toEqual(withoutGain);
      const noStaltaPOnly = VALIDATION.baseline.filter((r) => !(r.method === "stalta" && r.associationProfile === "p_only"));
      expect(ids(SUMMARY, { ...VALIDATION, baseline: noStaltaPOnly })).toEqual(withoutGain);
      const withoutTable = withoutGain.filter((id) => id !== "stalta");
      expect(ids(SUMMARY, { ...VALIDATION, baseline: [] })).toEqual(withoutTable);
      expect(ids(SUMMARY, { ...VALIDATION, baseline: null })).toEqual(withoutTable);
      expect(ids(SUMMARY, null)).not.toContain("gain");
      expect(baselineRan(null)).toBe(false);
    });

    it("never recomputes the ratio from the table: the exporter owns 'both profiles'", () => {
      const tied = VALIDATION.baseline.map((r) =>
        r.method === "stalta" && r.associationProfile === "p_only" ? { ...r, tiers: { ...r.tiers, A: 70 } } : r,
      );
      expect(ids(SUMMARY, { ...VALIDATION, baseline: tied })).toEqual(ALL);
      expect(rows(SUMMARY, { ...VALIDATION, baseline: tied }).find((r) => r.id === "gain")!.value).toBe("2.5×");
    });
  });

  describe("the STA/LTA row: the full-profile STA/LTA row of validation.baseline, right before the gain", () => {
    const value = (validation: ValidationInput) => rows(SUMMARY, validation).find((r) => r.id === "stalta")?.value;

    it("shows the strict count of the candidate count, both from the table row", () => {
      expect(value(VALIDATION)).toBe("40 of 120 candidates");
      expect(staltaStrict(VALIDATION.baseline)).toEqual({ strict: 40, candidates: 120 });
      const list = ids(SUMMARY, VALIDATION);
      expect(list.indexOf("stalta")).toBe(list.indexOf("gain") - 1);
    });

    it("stays when STA/LTA has no strict event and no gain is claimed (REQ-H1-5), with digit grouping", () => {
      const noStrictStalta = VALIDATION.baseline.map((r) =>
        r.method === "stalta" && r.associationProfile === "full" ? { ...r, candidates: 4120, tiers: { ...r.tiers, A: 0 } } : r,
      );
      const validation = { ...VALIDATION, baseline: noStrictStalta };
      expect(value(validation)).toBe("0 of 4,120 candidates");
      expect(ids({ ...SUMMARY, baseline: null }, validation)).toEqual(ALL.filter((id) => id !== "gain"));
    });

    it("needs only the full STA/LTA row: other rows may be missing, and the summary is not consulted", () => {
      const onlyStaltaFull = VALIDATION.baseline.filter((r) => r.method === "stalta" && r.associationProfile === "full");
      expect(value({ ...VALIDATION, baseline: onlyStaltaFull })).toBe("40 of 120 candidates");
      expect(rows(null, { ...VALIDATION, baseline: onlyStaltaFull }).map((r) => r.id)).toEqual(["depth", "stalta", "chance"]);
    });

    it("never shows the rerun's PhaseNet strict count", () => {
      const list = rows(SUMMARY, VALIDATION);
      expect(list.some((r) => r.value.includes(" vs "))).toBe(false);
      expect(list.some((r) => r.label.includes("PhaseNet vs STA/LTA") && r.id !== "gain")).toBe(false);
    });

    it("hides when the STA/LTA full row, its strict count, its candidate count, or the table is missing", () => {
      const noStaltaFull = VALIDATION.baseline.filter((r) => !(r.method === "stalta" && r.associationProfile === "full"));
      expect(value({ ...VALIDATION, baseline: noStaltaFull })).toBeUndefined();
      const nullCount = VALIDATION.baseline.map((r) =>
        r.method === "stalta" && r.associationProfile === "full" ? { ...r, tiers: { ...r.tiers, A: null } } : r,
      ) as unknown as BaselineRow[];
      expect(value({ ...VALIDATION, baseline: nullCount })).toBeUndefined();
      const nullCandidates = VALIDATION.baseline.map((r) =>
        r.method === "stalta" && r.associationProfile === "full" ? { ...r, candidates: null } : r,
      ) as unknown as BaselineRow[];
      expect(value({ ...VALIDATION, baseline: nullCandidates })).toBeUndefined();
      const nanCount = VALIDATION.baseline.map((r) =>
        r.method === "stalta" && r.associationProfile === "full" ? { ...r, tiers: { ...r.tiers, A: Number.NaN } } : r,
      );
      expect(value({ ...VALIDATION, baseline: nanCount })).toBeUndefined();
      expect(value({ ...VALIDATION, baseline: [] })).toBeUndefined();
      expect(value({ ...VALIDATION, baseline: null })).toBeUndefined();
      expect(value(null)).toBeUndefined();
      expect(staltaStrict(null)).toBeNull();
    });

    it("labels are words only, and no row carries a note but the chance row", () => {
      const list = rows(SUMMARY, VALIDATION);
      expect(list.find((r) => r.id === "stalta")!.label).not.toMatch(/\d/);
      expect(list.find((r) => r.id === "depth")!.label).toMatch(/synthetic, all stations/);
      expect(list.find((r) => r.id === "residual")!.label).toBe("Median timing misfit");
      for (const other of list) if (other.id !== "chance") expect(other.note).toBeUndefined();
    });
  });

  describe("the chance-associations note: every number from validation.nullTest", () => {
    const note = (nullTest: ValidationInput extends infer V ? (V extends { nullTest?: infer T } ? T : never) : never) =>
      rows(SUMMARY, { ...VALIDATION, nullTest }).find((r) => r.id === "chance")?.note;

    it("names the shuffle count and says none reached the strict tier when the strict mean is zero", () => {
      expect(note({ ...VALIDATION.nullTest!, meanChanceStrict: 0 })).toBe("Mean of 20 timing scrambles; none reached the strict tier");
      expect(chanceNote({ nShuffles: 20, meanChanceStrict: 0 })).toBe("Mean of 20 timing scrambles; none reached the strict tier");
    });

    it("reports a non-zero strict mean instead of claiming none", () => {
      expect(note({ ...VALIDATION.nullTest!, meanChanceStrict: 0.1 })).toBe("Mean of 20 timing scrambles; about 0.1 per scramble reached the strict tier");
    });

    it("drops a clause whose field is missing rather than guessing", () => {
      expect(chanceNote({ nShuffles: null, meanChanceStrict: 0 })).toBe("Mean over timing scrambles; none reached the strict tier");
      expect(chanceNote({ nShuffles: 20, meanChanceStrict: null })).toBe("Mean of 20 timing scrambles");
      expect(chanceNote({ nShuffles: Number.NaN, meanChanceStrict: Number.NaN })).toBe("Mean over timing scrambles");
      expect(chanceNote(null)).toBe("Mean over timing scrambles");
    });

    it("has no digit that did not come from the data", () => {
      expect(chanceNote({ nShuffles: null, meanChanceStrict: null })).not.toMatch(/\d/);
      expect(chanceNote({ nShuffles: null, meanChanceStrict: 0 })).not.toMatch(/\d/);
    });
  });
});

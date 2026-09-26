import { describe, expect, it } from "vitest";
import { filterCountIssues, isCandidateShown, shownCandidateCount, shownCandidateIds } from "./selectors";

// Test-local records: 5 A, 3 B, 2 C.
const tiers = ["A", "B", "A", "C", "A", "B", "A", "C", "B", "A"] as const;
const events = tiers.map((tier, i) => ({ id: `e${i}`, tier }));
const summary = { strictQualityCount: 5, candidateCount: 10 };

describe("filter selector", () => {
  it("STRICT shows exactly strictQualityCount instances (acceptance)", () => {
    expect(shownCandidateCount(events, "strict")).toBe(summary.strictQualityCount);
    expect(shownCandidateIds(events, "strict")).toEqual(["e0", "e2", "e4", "e6", "e9"]);
  });

  it("ALL shows every candidate, PUBLIC none", () => {
    expect(shownCandidateCount(events, "all")).toBe(summary.candidateCount);
    expect(shownCandidateCount(events, "public")).toBe(0);
  });

  it("isCandidateShown encodes the three filters per tier", () => {
    expect(["A", "B", "C"].map((t) => isCandidateShown(t as "A", "strict"))).toEqual([true, false, false]);
    expect(["A", "B", "C"].map((t) => isCandidateShown(t as "A", "all"))).toEqual([true, true, true]);
    expect(["A", "B", "C"].map((t) => isCandidateShown(t as "A", "public"))).toEqual([false, false, false]);
  });
});

describe("filterCountIssues", () => {
  it("is empty when the scene and the counters agree", () => {
    expect(filterCountIssues(events, summary)).toEqual([]);
  });

  it("names every mismatch with both numbers", () => {
    expect(filterCountIssues(events, { strictQualityCount: 6, candidateCount: 11 })).toEqual([
      "STRICT would show 5 events but summary.strictQualityCount is 6",
      "ALL would show 10 events but summary.candidateCount is 11",
    ]);
  });
});

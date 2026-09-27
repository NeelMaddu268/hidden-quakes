import { describe, expect, it } from "vitest";
import {
  captionText,
  fillCaption,
  formatCount,
  formatReplayRate,
  placeholdersIn,
  TOUR_PLACEHOLDERS,
  tourValues,
  type TourValues,
} from "./captions";

const SUMMARY = {
  publicCatalogCount: 43,
  recoveredCatalogCount: 41,
  candidateCount: 1654,
  additionalCount: 1613,
  strictQualityCount: 32,
  strictAdditionalCount: 14,
};

describe("formatCount / formatReplayRate", () => {
  it("prints counts as the counters do (grouped, whole)", () => {
    expect(formatCount(1654)).toBe("1,654");
    expect(formatCount(0)).toBe("0");
  });

  it("names the replay speed in the largest whole unit", () => {
    expect(formatReplayRate(3600)).toBe("1 hour per second");
    expect(formatReplayRate(7200)).toBe("2 hours per second");
    expect(formatReplayRate(86400)).toBe("1 day per second");
    expect(formatReplayRate(1800)).toBe("30 minutes per second");
    expect(formatReplayRate(5400)).toBe("1.5 hours per second");
    expect(formatReplayRate(30)).toBe("30× real time");
  });

  it("fails loudly on a non-positive rate", () => {
    expect(() => formatReplayRate(0)).toThrow(/rate/);
    expect(() => formatReplayRate(Number.NaN)).toThrow(/rate/);
  });
});

describe("tourValues", () => {
  it("fills every placeholder from the summary, the hidden event and the playback rate", () => {
    const v = tourValues(SUMMARY, 23, 3600);
    expect(Object.keys(v).sort()).toEqual([...TOUR_PLACEHOLDERS].sort());
    expect(v).toMatchObject({
      publicCatalogCount: "43",
      recoveredCatalogCount: "41",
      candidateCount: "1,654",
      additionalCount: "1,613",
      strictQualityCount: "32",
      strictAdditionalCount: "14",
      heroStations: "23",
      replayRate: "1 hour per second",
    });
  });

  it("has no hidden-event value when the bundle has no such event", () => {
    expect(tourValues(SUMMARY, null, 3600).heroStations).toBeNull();
    expect(tourValues(SUMMARY, Number.NaN, 3600).heroStations).toBeNull();
  });
});

describe("fillCaption", () => {
  const values: TourValues = tourValues(SUMMARY, 23, 3600);

  it("splits a template into text and toned value segments", () => {
    const s = fillCaption("We recovered {recoveredCatalogCount} of {publicCatalogCount}.", values)!;
    expect(s).toEqual([
      { kind: "text", text: "We recovered " },
      { kind: "value", text: "41", tone: "recovered", name: "recoveredCatalogCount" },
      { kind: "text", text: " of " },
      { kind: "value", text: "43", tone: "public", name: "publicCatalogCount" },
      { kind: "text", text: "." },
    ]);
    expect(captionText(s)).toBe("We recovered 41 of 43.");
  });

  it("handles templates that start or end with a placeholder, or have none", () => {
    expect(captionText(fillCaption("{candidateCount}", values)!)).toBe("1,654");
    expect(fillCaption("No numbers here.", values)).toEqual([{ kind: "text", text: "No numbers here." }]);
  });

  it("returns null when a value is missing, so no caption shows a hole", () => {
    expect(fillCaption("{heroStations} stations", { ...values, heroStations: null })).toBeNull();
  });

  it("throws on a placeholder the tour can't fill", () => {
    expect(() => fillCaption("{magnitude}", values)).toThrow(/unknown placeholder \{magnitude\}/);
    expect(() => placeholdersIn("{publicCount}")).toThrow(/unknown placeholder/);
    expect(placeholdersIn("{candidateCount} and {publicCatalogCount}")).toEqual(["candidateCount", "publicCatalogCount"]);
  });
});

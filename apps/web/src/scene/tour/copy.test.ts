import { describe, expect, it } from "vitest";
import { fillCaption, placeholdersIn, tourValues } from "./captions";
import { TOUR_COPY } from "./copy";
import { TOUR_STEPS } from "./script";

// The copy file is H4's to edit (REQ-H3-15); these checks keep every edit inside the honesty rules
// (CLAUDE.md 4 and 6, docs/00) before it can ship.

// The same phrases scripts/check-copy.sh rejects, plus docs/00's words about our own depths.
const FORBIDDEN = [
  /confirmed earthquake/i,
  /caused by/i,
  /predict/i,
  /\bofficial\b/i,
  /\b(induced|triggered|generated|created) by\b/i,
  /\b(due to|because of|attributed to|attributable to|blamed on)\b/i,
  /\bforge\b/i,
  /cape station/i,
  /\boperators?\b/i,
  /\bfractures?\b/i,
  /\bfaults?\b/i,
  /\bstructures?\b/i,
];

const SUMMARY = {
  publicCatalogCount: 43,
  recoveredCatalogCount: 41,
  candidateCount: 1654,
  additionalCount: 1613,
  strictQualityCount: 32,
  strictAdditionalCount: 14,
};

describe("tour copy (the file H4 edits)", () => {
  const entries = Object.entries(TOUR_COPY);

  it("carries no digits: every number comes from a placeholder", () => {
    for (const [key, text] of entries) expect({ key, digits: /\d/.test(text) }).toEqual({ key, digits: false });
  });

  it("uses none of the phrases docs/00 forbids", () => {
    for (const [key, text] of entries) {
      for (const phrase of FORBIDDEN) expect({ key, hit: phrase.test(text) }).toEqual({ key, hit: false });
    }
  });

  it("says candidate events and public regional catalog, never a bare 'catalog' or 'earthquakes'", () => {
    for (const [key, template] of entries) {
      const text = template.replace(/\{[A-Za-z]+\}/g, "#");
      expect({ key, earthquakes: /earthquake/i.test(text) }).toEqual({ key, earthquakes: false });
      for (const m of text.matchAll(/catalog/gi)) {
        expect({ key, qualified: text.slice(Math.max(0, m.index - 16), m.index).endsWith("public regional ") }).toEqual({
          key,
          qualified: true,
        });
      }
    }
  });

  it("uses only placeholders the tour can fill", () => {
    for (const text of Object.values(TOUR_COPY)) expect(() => placeholdersIn(text)).not.toThrow();
  });

  it("has a caption for every step, and each fills from bundle values", () => {
    const values = tourValues(SUMMARY, 23, 3600);
    for (const step of TOUR_STEPS) {
      const filled = fillCaption(TOUR_COPY[step.caption], values);
      expect(filled, step.id).not.toBeNull();
      expect(filled!.map((s) => s.text).join("")).not.toMatch(/[{}]/);
    }
  });

  it("keeps each caption short enough to read in its step", () => {
    for (const step of TOUR_STEPS) expect(TOUR_COPY[step.caption].length, step.id).toBeLessThanOrEqual(170);
  });
});

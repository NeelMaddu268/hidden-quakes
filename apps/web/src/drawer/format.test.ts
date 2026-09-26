import { describe, expect, it } from "vitest";
import {
  DASH,
  fmtFixed,
  fmtKmFromM,
  fmtLength,
  fmtMagnitude,
  fmtPlusMinusM,
  fmtSeconds,
  fmtStationsAgreed,
  fmtUnit,
  fmtUtahLocal,
  fmtUtc,
  isNum,
} from "./format";

// 2026-09-10 14:03:07.21 UTC as epoch seconds, computed from the calendar, not hard-coded.
const T = Date.UTC(2026, 8, 10, 14, 3, 7, 210) / 1000;

describe("missing values never render as NaN or undefined", () => {
  const missing = [null, undefined, NaN, Infinity, -Infinity];
  it("every formatter returns the dash", () => {
    for (const v of missing) {
      expect(isNum(v)).toBe(false);
      expect(fmtFixed(v, 2)).toBe(DASH);
      expect(fmtUnit(v, 3, "s")).toBe(DASH);
      expect(fmtPlusMinusM(v)).toBe(DASH);
      expect(fmtKmFromM(v)).toBe(DASH);
      expect(fmtUtc(v)).toBe(DASH);
      expect(fmtUtahLocal(v)).toBe(DASH);
      expect(fmtStationsAgreed(v)).not.toMatch(/NaN|undefined|Infinity/);
    }
  });
});

describe("numbers", () => {
  it("fixed decimals with a true minus sign and no negative zero", () => {
    expect(fmtFixed(0.0781, 3)).toBe("0.078");
    expect(fmtFixed(-1.5, 1)).toBe("−1.5");
    expect(fmtFixed(-0.0001, 2)).toBe("0.00");
  });
  it("units and errors", () => {
    expect(fmtUnit(0.078, 3, "s")).toBe("0.078 s");
    expect(fmtPlusMinusM(369.3)).toBe("±369 m");
    expect(fmtKmFromM(3240)).toBe("3.2 km");
    expect(fmtKmFromM(3240, 2)).toBe("3.24 km");
  });
  it("stations agreed, singular and plural", () => {
    expect(fmtStationsAgreed(11)).toBe("11 stations agreed");
    expect(fmtStationsAgreed(1)).toBe("1 station agreed");
  });
  it("axis seconds follow the tick step", () => {
    expect(fmtSeconds(2, 1)).toBe("2");
    expect(fmtSeconds(-1, 1)).toBe("−1");
    expect(fmtSeconds(2.5, 0.5)).toBe("2.5");
  });
});

describe("origin time", () => {
  it("UTC to the centisecond", () => {
    expect(fmtUtc(T)).toBe("2026-09-10 14:03:07.21 UTC");
  });
  it("rounds across a second boundary without printing 60", () => {
    const t = Date.UTC(2026, 8, 10, 23, 59, 59, 996) / 1000;
    expect(fmtUtc(t)).toBe("2026-09-11 00:00:00.00 UTC");
  });
  it("Utah local is MDT (UTC−6) in September", () => {
    expect(fmtUtahLocal(T)).toBe("2026-09-10 08:03:07.21 MDT");
  });
  it("crosses the local date line correctly", () => {
    const t = Date.UTC(2026, 8, 10, 2, 0, 0) / 1000;
    expect(fmtUtahLocal(t)).toBe("2026-09-09 20:00:00.00 MDT");
  });
  it("uses MST (UTC−7) in winter, from the time zone database", () => {
    const t = Date.UTC(2026, 0, 15, 12, 0, 0) / 1000;
    expect(fmtUtahLocal(t)).toBe("2026-01-15 05:00:00.00 MST");
  });
});

describe("magnitude", () => {
  it("always carries its type", () => {
    expect(fmtMagnitude({ value: 1.23, type: "ML_cal" })).toBe("M 1.2 ML_cal");
    expect(fmtMagnitude({ value: 0.4, type: "ml", sigma: 0.21 })).toBe("M 0.4 ml ±0.2");
    expect(fmtMagnitude({ value: 0.4, type: "ml", sigma: null })).toBe("M 0.4 ml");
  });
  it("is null when absent", () => {
    expect(fmtMagnitude(null)).toBeNull();
    expect(fmtMagnitude(undefined)).toBeNull();
    expect(fmtMagnitude({ value: NaN, type: "ml" })).toBeNull();
  });
});

describe("scale-bar lengths", () => {
  it("meters below a kilometer, kilometers above", () => {
    expect(fmtLength(500)).toBe("500 m");
    expect(fmtLength(2500)).toBe("2.5 km");
    expect(fmtLength(10000)).toBe("10 km");
    expect(fmtLength(NaN)).toBe(DASH);
  });
});

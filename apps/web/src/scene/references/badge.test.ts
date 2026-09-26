import { describe, expect, it } from "vitest";
import { formatExaggeration, verticalBadgeText } from "./badge";

describe("verticalBadgeText", () => {
  it("shows nothing without exaggeration (1.0)", () => {
    expect(verticalBadgeText({ verticalExaggeration: 1 })).toBeNull();
  });

  it("shows 'Vertical ×N' with N from the data", () => {
    expect(verticalBadgeText({ verticalExaggeration: 2 })).toBe("Vertical ×2");
    expect(verticalBadgeText({ verticalExaggeration: 1.5 })).toBe("Vertical ×1.5");
    expect(verticalBadgeText({ verticalExaggeration: 0.5 })).toBe("Vertical ×0.5");
  });

  it("rejects nonsense exaggeration instead of drawing a wrong badge", () => {
    expect(() => verticalBadgeText({ verticalExaggeration: 0 })).toThrow();
    expect(() => verticalBadgeText({ verticalExaggeration: -2 })).toThrow();
  });
});

describe("formatExaggeration", () => {
  it("drops trailing zeros and rounds to 2 decimals", () => {
    expect(formatExaggeration(2)).toBe("2");
    expect(formatExaggeration(2.0)).toBe("2");
    expect(formatExaggeration(2.5)).toBe("2.5");
    expect(formatExaggeration(2.25)).toBe("2.25");
    expect(formatExaggeration(10)).toBe("10");
    expect(formatExaggeration(4 / 3)).toBe("1.33");
  });

  it("never shows a real exaggeration as ×1", () => {
    expect(formatExaggeration(1.001)).toBe("1.001");
    expect(formatExaggeration(0.999)).toBe("0.999");
    expect(formatExaggeration(1)).toBe("1");
  });
});

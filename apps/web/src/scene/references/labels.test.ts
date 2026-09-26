import { colors } from "@hq/visualization";
import { describe, expect, it } from "vitest";
import { LABEL_HALO, labelStyle, numericLabelStyle } from "./labels";

describe("scene label style", () => {
  it("carries a legibility halo in the page background colour that is symmetric (not a drop shadow)", () => {
    expect(labelStyle.textShadow).toBe(LABEL_HALO);
    expect(numericLabelStyle.textShadow).toBe(LABEL_HALO);
    let sx = 0;
    let sy = 0;
    for (const shadow of LABEL_HALO.split(", ")) {
      expect(shadow.endsWith(colors.bg)).toBe(true);
      const [x, y] = shadow.split(" ").map((v) => Number.parseFloat(v));
      expect(Math.abs(x)).toBeLessThanOrEqual(1);
      expect(Math.abs(y)).toBeLessThanOrEqual(1);
      sx += x;
      sy += y;
    }
    expect([sx, sy]).toEqual([0, 0]);
  });

  it("never takes pointer events", () => {
    expect(labelStyle.pointerEvents).toBe("none");
  });
});

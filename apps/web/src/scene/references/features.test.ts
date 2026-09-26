import { describe, expect, it } from "vitest";
import type { GeoFeature } from "../types";
import { APPROXIMATE, featureGeometry, featureIssue, featureLabelAnchor, featureStyle } from "./features";

/** Test-local features with neutral names. */
function feature(over: Partial<GeoFeature> & Pick<GeoFeature, "kind">, verified = true): GeoFeature {
  return {
    id: "f1",
    name: "Test well",
    path: [
      { e: 0, n: 0, u: 20 },
      { e: 100, n: 50, u: -1000 },
      { e: 400, n: 200, u: -2500 },
    ],
    source: { citation: "test", url: "https://example.invalid", verified },
    ...over,
  };
}

describe("featureStyle (acceptance: unverified renders dashed with 'approximate')", () => {
  it("unverified → dashed, labelled approximate", () => {
    const s = featureStyle(feature({ kind: "well" }, false));
    expect(s.dashed).toBe(true);
    expect(s.approximate).toBe(true);
    expect(s.label).toBe(`Test well (${APPROXIMATE})`);
    expect(APPROXIMATE).toBe("approximate");
  });

  it("verified → solid, labelled with its own name only", () => {
    const s = featureStyle(feature({ kind: "well" }, true));
    expect(s.dashed).toBe(false);
    expect(s.approximate).toBe(false);
    expect(s.label).toBe("Test well");
  });

  it("applies to every kind, and treats a missing/odd verified flag as unverified", () => {
    for (const kind of ["well", "facility", "boundary"] as const) {
      expect(featureStyle(feature({ kind }, false)).dashed).toBe(true);
      expect(featureStyle(feature({ kind }, true)).dashed).toBe(false);
    }
    const odd = feature({ kind: "well" });
    (odd.source as { verified: unknown }).verified = "yes";
    expect(featureStyle(odd).dashed).toBe(true);
  });

  it("labels with the data's name verbatim (no invented operator or site names)", () => {
    expect(featureStyle(feature({ kind: "facility", name: "Plant 7" }, true)).label).toBe("Plant 7");
  });

  it("wells and facilities glow as the geothermal reference; boundaries don't", () => {
    expect(featureStyle(feature({ kind: "well" })).glow).toBe(true);
    expect(featureStyle(feature({ kind: "facility" })).glow).toBe(true);
    expect(featureStyle(feature({ kind: "boundary" })).glow).toBe(false);
  });
});

describe("featureGeometry", () => {
  it("draws a well as a polyline through its path, via the ENU → scene mapping", () => {
    const g = featureGeometry(feature({ kind: "well" }), 1);
    expect(g).toEqual({
      type: "line",
      closed: false,
      points: [
        [0, 0.02, -0],
        [0.1, -1, -0.05],
        [0.4, -2.5, -0.2],
      ],
    });
  });

  it("exaggerates heights only", () => {
    const g = featureGeometry(feature({ kind: "well" }), 2);
    expect(g?.type === "line" && g.points[2]).toEqual([0.4, -5, -0.2]);
  });

  it("draws a facility as a point at path[0], and closes boundaries", () => {
    expect(featureGeometry(feature({ kind: "facility" }), 1)).toEqual({ type: "point", at: [0, 0.02, -0] });
    const b = featureGeometry(feature({ kind: "boundary" }), 1);
    expect(b?.type).toBe("line");
    if (b?.type === "line") {
      expect(b.closed).toBe(true);
      expect(b.points).toHaveLength(4);
      expect(b.points[3]).toEqual(b.points[0]);
    }
  });

  it("skips what can't be drawn, and says why", () => {
    expect(featureGeometry(feature({ kind: "well", path: [] }), 1)).toBeNull();
    const tiny = feature({ kind: "boundary", path: [{ e: 0, n: 0, u: 0 }, { e: 1, n: 1, u: 0 }] });
    expect(featureGeometry(tiny, 1)).toBeNull();
    expect(featureIssue(tiny)).toMatch(/< 3/);
    expect(featureIssue(feature({ kind: "well", path: [] }))).toMatch(/empty path/);
    expect(featureIssue(feature({ kind: "well" }))).toBeNull();
  });

  it("pins the label at the feature's top", () => {
    const g = featureGeometry(feature({ kind: "well" }), 1);
    expect(g && featureLabelAnchor(g)).toEqual([0, 0.02, -0]);
  });
});

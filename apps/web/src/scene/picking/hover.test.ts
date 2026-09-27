import { beforeEach, describe, expect, it } from "vitest";
import { candidateTooltip, fmtUtcSeconds, publicTooltip, setHover, useHover } from "./hover";

// Test-local synthetic records (rule 5): only the fields the tooltip reads.
const SCENE = { refSurfaceElevM: 1627.7 };
const T = Date.UTC(2026, 8, 10, 9, 15, 28, 830) / 1000;

const candidate = (over: Record<string, unknown> = {}) => ({
  tier: "A" as const,
  t: T,
  elevM: 1627.7 - 3980,
  magnitude: { value: 0.93, type: "ML_cal", sigma: 0.1 },
  catalogMatch: null,
  quality: { nStations: 23 },
  ...over,
});

beforeEach(() => useHover.setState({ target: null }));

describe("candidateTooltip", () => {
  it("says tier, UTC time, depth below the site surface, magnitude, stations and catalog status", () => {
    expect(candidateTooltip(candidate(), SCENE)).toEqual({
      title: "Candidate event · Tier A (strict)",
      rows: [
        { text: "2026-09-10 09:15:28 UTC" },
        { text: "3.98 km below site surface" },
        { text: "M 0.9 ML_cal ±0.1" },
        { text: "23 stations agreed" },
        { text: "Not in the public regional catalog", tone: "accent" },
      ],
    });
  });

  it("says when the public regional catalog lists it, and handles missing values without NaN", () => {
    const c = candidateTooltip(
      candidate({ tier: "C", catalogMatch: { catalogId: "p1", distM: 100, dtS: 0.2 }, magnitude: null, quality: { nStations: Number.NaN } }),
      SCENE,
    );
    expect(c.title).toBe("Candidate event · Tier C");
    expect(c.rows.map((r) => r.text)).toEqual([
      "2026-09-10 09:15:28 UTC",
      "3.98 km below site surface",
      "No magnitude",
      "Stations unknown",
      "In the public regional catalog",
    ]);
    expect(candidateTooltip(candidate({ quality: { nStations: 1 } }), SCENE).rows[3]!.text).toBe("1 station agreed");
  });

  it("uses the drawer's depth rule: above the reference surface is a negative depth with a true minus", () => {
    expect(candidateTooltip(candidate({ elevM: 1627.7 + 120 }), SCENE).rows[1]!.text).toBe("−0.12 km below site surface");
  });
});

describe("publicTooltip", () => {
  const pub = { t: T, elevM: 1627.7 - 4200, mag: 1.24, magType: "ml", matchedEventId: "e1" };

  it("says what the catalog says, and nothing about recovery before the reveal", () => {
    expect(publicTooltip(pub, SCENE, false, "A")).toEqual({
      title: "Public regional catalog event",
      rows: [{ text: "2026-09-10 09:15:28 UTC" }, { text: "4.20 km below site surface" }, { text: "M 1.2 ml" }],
    });
  });

  it("after the reveal, says whether our pipeline recovered it", () => {
    expect(publicTooltip(pub, SCENE, true, "B").rows[3]).toEqual({ text: "Recovered by our pipeline · Tier B", tone: "dim" });
    expect(publicTooltip({ ...pub, matchedEventId: null }, SCENE, true, null).rows[3]).toEqual({
      text: "Not recovered by our pipeline",
      tone: "dim",
    });
  });

  it("handles a catalog event without a magnitude or type", () => {
    expect(publicTooltip({ ...pub, mag: null }, SCENE, false, null).rows[2]!.text).toBe("No magnitude");
    expect(publicTooltip({ ...pub, magType: null }, SCENE, false, null).rows[2]!.text).toBe("M 1.2");
  });
});

describe("fmtUtcSeconds / setHover", () => {
  it("drops the drawer's centiseconds", () => {
    expect(fmtUtcSeconds(T)).toBe("2026-09-10 09:15:28 UTC");
  });

  it("publishes only changes, so a still pointer re-renders nothing", () => {
    const a = { kind: "candidate" as const, id: "e1", x: 10, y: 20 };
    setHover(a);
    const first = useHover.getState().target;
    setHover({ ...a });
    expect(useHover.getState().target).toBe(first);
    setHover({ ...a, x: 11 });
    expect(useHover.getState().target).not.toBe(first);
    setHover(null);
    expect(useHover.getState().target).toBeNull();
  });
});

/**
 * Overnight feature 2: the candidate-catalog download is built from the bundle alone. Every
 * number here is invented inside the test; the fixture events come from `../test-fixture.ts`.
 */
import { describe, expect, it } from "vitest";
import { makeEvent, makeMeta } from "../test-fixture";
import { CAVEAT, COLUMNS, catalogCsv, catalogFileName, catalogGeoJson, catalogRow, csvCell, headerLines, isoUtc } from "./catalog";

const meta = makeMeta("run-x", { isSynthetic: false, scene: { depthLabel: "Depth below site surface (ref 1600 m ASL)" } });
const matched = {
  ...makeEvent("ev-a", 0),
  t: 1788998697.5424778,
  latitude: 38.51,
  longitude: -112.9,
  elevM: -2475,
  depthKm: 4.1,
  tier: "A" as const,
  magnitude: { value: 1.25, type: "ML_cal", sigma: 0.1 },
  catalogMatch: { catalogId: "uu80012345", dtS: 0.2, distM: 300 },
};
const bare = { ...makeEvent("ev,b", 1), tier: "C" as const };
const events = [matched, bare];

describe("catalogRow()", () => {
  it("maps the contract fields to the columns, blank for a missing magnitude or match", () => {
    expect(catalogRow(matched)).toEqual({
      id: "ev-a",
      time_utc: "2026-09-10T00:04:57.542Z",
      latitude: "38.51",
      longitude: "-112.9",
      depth_km: "4.1",
      elev_m_asl: "-2475",
      tier: "A",
      magnitude: "1.25",
      magnitude_type: "ML_cal",
      stations: "5",
      catalog_match: "uu80012345",
    });
    const row = catalogRow(bare);
    expect(row.magnitude).toBe("");
    expect(row.magnitude_type).toBe("");
    expect(row.catalog_match).toBe("");
    expect(Object.keys(row)).toEqual([...COLUMNS]);
  });

  it("formats time as ISO 8601 UTC", () => {
    expect(isoUtc(0)).toBe("1970-01-01T00:00:00.000Z");
    expect(isoUtc(1.5)).toBe("1970-01-01T00:00:01.500Z");
  });
});

describe("catalogCsv()", () => {
  const csv = catalogCsv(events, meta);
  const lines = csv.split("\n");

  it("opens with the run id, the caveat and the depth convention as comment lines", () => {
    expect(lines[0]).toBe("# Hidden Quakes candidate events, run run-x");
    expect(lines[1]).toBe(`# ${CAVEAT}`);
    expect(lines[2]).toBe("# depth_km is Depth below site surface (ref 1600 m ASL); elev_m_asl is metres above sea level.");
    expect(lines[3]).toBe(COLUMNS.join(","));
  });

  it("writes one row per event in bundle order, quoting cells that need it", () => {
    expect(lines[4]).toBe("ev-a,2026-09-10T00:04:57.542Z,38.51,-112.9,4.1,-2475,A,1.25,ML_cal,5,uu80012345");
    expect(lines[5]).toBe('"ev,b",1970-01-01T00:00:01.000Z,0,0,0,0,C,,,5,');
    expect(csv.endsWith("\n")).toBe(true);
    expect(lines.filter((l) => l !== "").length).toBe(4 + events.length);
  });

  it("never writes the word null", () => {
    expect(csv).not.toMatch(/null/);
  });

  it("quotes per RFC 4180", () => {
    expect(csvCell("plain")).toBe("plain");
    expect(csvCell("a,b")).toBe('"a,b"');
    expect(csvCell('say "hi"')).toBe('"say ""hi"""');
    expect(csvCell("two\nlines")).toBe('"two\nlines"');
  });

  it("marks a synthetic bundle in its first line and its file name", () => {
    const synthetic = makeMeta("mock-run", { isSynthetic: true });
    expect(headerLines(synthetic)[0]).toMatch(/^SYNTHETIC DATA/);
    expect(catalogFileName(synthetic, "csv")).toBe("synthetic-hidden-quakes-candidates-mock-run.csv");
    expect(catalogFileName(meta, "geojson")).toBe("hidden-quakes-candidates-run-x.geojson");
    expect(headerLines(meta)[0]).not.toMatch(/SYNTHETIC/);
  });
});

describe("catalogGeoJson()", () => {
  const collection = JSON.parse(catalogGeoJson(events, meta)) as {
    type: string;
    metadata: Record<string, unknown>;
    features: { type: string; id: string; geometry: { type: string; coordinates: number[] }; properties: Record<string, unknown> }[];
  };

  it("is a FeatureCollection of WGS84 points with the run id and caveat as metadata", () => {
    expect(collection.type).toBe("FeatureCollection");
    expect(collection.metadata).toEqual({
      runId: "run-x",
      caveat: CAVEAT,
      depthLabel: "Depth below site surface (ref 1600 m ASL)",
      isSynthetic: false,
      mode: "mock",
    });
    expect(collection.features).toHaveLength(2);
    const [first, second] = collection.features;
    expect(first.type).toBe("Feature");
    expect(first.id).toBe("ev-a");
    expect(first.geometry).toEqual({ type: "Point", coordinates: [-112.9, 38.51, -2475] });
    expect(first.properties).toEqual({
      time_utc: "2026-09-10T00:04:57.542Z",
      depth_km: 4.1,
      tier: "A",
      magnitude: 1.25,
      magnitude_type: "ML_cal",
      stations: 5,
      catalog_match: "uu80012345",
    });
    expect(second.properties.magnitude).toBeNull();
    expect(second.properties.catalog_match).toBeNull();
  });

  it("is deterministic for the same input", () => {
    expect(catalogGeoJson(events, meta)).toBe(catalogGeoJson(events, meta));
    expect(catalogCsv(events, meta)).toBe(catalogCsv(events, meta));
  });
});

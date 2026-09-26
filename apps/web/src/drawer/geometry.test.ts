import { describe, expect, it } from "vitest";
import type { Station, WaveformSnippet } from "../scene/types";
import { depthGeometry, depthKmOfElev, evidenceStations, mapGeometry, pickingStations, roundLengthAtMost } from "./geometry";

const BOX = { width: 240, height: 200, pad: 16 };

function station(id: string, e: number, n: number, over: Partial<Station> = {}): Station {
  return {
    id,
    network: "XX",
    location: "",
    staticsS: {},
    station: id,
    latitude: 0,
    longitude: 0,
    surfaceElevM: 1600,
    sensorDepthM: 0,
    sensorElevM: 1600,
    kind: "surface",
    channels: ["HHZ"],
    sampleRateHz: 100,
    enu: { e, n, u: 0 },
    preprocessProfile: "test",
    usedInRun: true,
    ...over,
  };
}

describe("roundLengthAtMost", () => {
  it("picks 1, 2, 2.5 or 5 × 10^k no longer than the limit", () => {
    expect(roundLengthAtMost(7300)).toBe(5000);
    expect(roundLengthAtMost(2600)).toBe(2500);
    expect(roundLengthAtMost(2400)).toBe(2000);
    expect(roundLengthAtMost(1000)).toBe(1000);
    expect(roundLengthAtMost(180)).toBe(100);
    expect(roundLengthAtMost(0)).toBe(0);
  });
});

describe("evidenceStations", () => {
  it("looks stations up by id, deduplicated, and lists the missing ones", () => {
    const byId = new Map([
      ["XX.A", station("XX.A", 0, 0)],
      ["XX.B", station("XX.B", 1, 1)],
    ]);
    const traces = ["XX.B", "XX.A", "XX.B", "XX.GONE"].map(
      (stationId) => ({ stationId, channel: "HHZ", epiDistM: 0, t0: 0, dt: 0.01, samples: [0, 0] }) as WaveformSnippet,
    );
    const { stations, missing } = evidenceStations(traces, byId);
    expect(stations.map((s) => s.id)).toEqual(["XX.B", "XX.A"]);
    expect(missing).toEqual(["XX.GONE"]);
  });
});

describe("mapGeometry (north up, east right, one scale)", () => {
  const epi = { e: 0, n: 0, u: -2000 };
  const pts = [
    { e: 10_000, n: 0, u: 0 },
    { e: -5_000, n: 4_000, u: 0 },
  ];
  const g = mapGeometry(epi, pts, 300, BOX);

  it("keeps every point inside the padded box", () => {
    for (const p of [epi, ...pts]) {
      expect(g.x(p.e)).toBeGreaterThanOrEqual(BOX.pad - 1e-9);
      expect(g.x(p.e)).toBeLessThanOrEqual(BOX.width - BOX.pad + 1e-9);
      expect(g.y(p.n)).toBeGreaterThanOrEqual(BOX.pad - 1e-9);
      expect(g.y(p.n)).toBeLessThanOrEqual(BOX.height - BOX.pad + 1e-9);
    }
  });

  it("east is right and north is up, at the same scale", () => {
    expect(g.x(1000) - g.x(0)).toBeCloseTo(1000 * g.pxPerM, 9);
    expect(g.y(0) - g.y(1000)).toBeCloseTo(1000 * g.pxPerM, 9);
  });

  it("the scale bar is a round length that fits", () => {
    expect([1, 2, 2.5, 5].map((m) => m * 1000)).toContain(g.scaleBarM);
    expect(g.scaleBarPx).toBeCloseTo(g.scaleBarM * g.pxPerM, 9);
    expect(g.scaleBarPx).toBeLessThanOrEqual(0.4 * (BOX.width - 2 * BOX.pad) + 1e-9);
  });

  it("a lone epicenter gets a minimum extent instead of an infinite scale", () => {
    const lone = mapGeometry(epi, [], null, BOX);
    expect(Number.isFinite(lone.pxPerM)).toBe(true);
    expect(lone.x(0)).toBeCloseTo(BOX.width / 2, 9);
  });
});

describe("depthGeometry", () => {
  it("depth from elevation uses the site-surface reference (docs/01)", () => {
    expect(depthKmOfElev(-400, { refSurfaceElevM: 1600 })).toBeCloseTo(2, 9);
    expect(depthKmOfElev(1650, { refSurfaceElevM: 1600 })).toBeCloseTo(-0.05, 9);
  });

  it("deeper is lower, the surface is in view, and ticks cover the range", () => {
    const pts = [
      { e: 0, depthKm: 2.5 },
      { e: -8000, depthKm: 0 },
      { e: 8000, depthKm: 1 },
    ];
    const g = depthGeometry(pts, 0.3, BOX);
    expect(g.y(2.5)).toBeGreaterThan(g.y(1));
    expect(g.top).toBeLessThanOrEqual(0);
    expect(g.bottom).toBeGreaterThanOrEqual(2.8);
    expect(g.ticks[0]).toBeLessThanOrEqual(0.0001);
    for (const p of pts) {
      expect(g.y(p.depthKm)).toBeGreaterThanOrEqual(BOX.pad - 1e-9);
      expect(g.y(p.depthKm)).toBeLessThanOrEqual(BOX.height - BOX.pad + 1e-9);
      expect(g.x(p.e)).toBeGreaterThanOrEqual(BOX.pad - 1e-9);
      expect(g.x(p.e)).toBeLessThanOrEqual(BOX.width - BOX.pad + 1e-9);
    }
  });

  it("stretches a shallow, wide section by a round factor and reports it", () => {
    const g = depthGeometry(
      [
        { e: 0, depthKm: 2 },
        { e: -10_000, depthKm: 0 },
        { e: 10_000, depthKm: 0 },
      ],
      0,
      BOX,
    );
    expect([1, 2, 5, 10]).toContain(g.verticalExaggeration);
    expect(g.verticalExaggeration).toBeGreaterThan(1);
    const hPerKm = g.x(1000) - g.x(0);
    const vPerKm = g.y(1) - g.y(0);
    expect(vPerKm / hPerKm).toBeCloseTo(g.verticalExaggeration, 6);
  });

  it("uses true scale when the depth range is the tall side", () => {
    const g = depthGeometry(
      [
        { e: 0, depthKm: 6 },
        { e: 500, depthKm: 0 },
      ],
      0,
      BOX,
    );
    expect(g.verticalExaggeration).toBe(1);
    expect(g.y(1) - g.y(0)).toBeCloseTo(g.x(1000) - g.x(0), 6);
  });
});

describe("pickingStations", () => {
  const st = (id: string, e = 0, n = 0) => ({ id, enu: { e, n, u: 0 } }) as unknown as Station;
  const byId = new Map([st("UU.FORK2"), st("6K.CS03", 10, 20), st("UU.NOPOS", NaN, 0)].map((s) => [s.id, s]));

  it("takes the part of each pick id that is a bundle station id, deduplicated in pick order", () => {
    const ids = [
      "phasenet:instance:6K.CS03:P:1789037720.490",
      "phasenet:instance:6K.CS03:S:1789037721.410",
      "phasenet:instance:UU.FORK2:P:1789037723.410",
    ];
    expect(pickingStations(ids, byId).map((s) => s.id)).toEqual(["6K.CS03", "UU.FORK2"]);
  });

  it("skips picks naming no known station and stations without a position", () => {
    expect(pickingStations(["x:y:XX.GONE:P:1", "a:b:UU.NOPOS:P:2", "garbage"], byId)).toEqual([]);
  });
});

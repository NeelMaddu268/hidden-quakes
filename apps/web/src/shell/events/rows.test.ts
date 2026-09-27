/** DEMO-04 item 2: the event list's rows, default order, column sorts and tier filter. Numbers are invented here. */
import { describe, expect, it } from "vitest";
import { makeEvent } from "../test-fixture";
import { sortRows, toRow } from "./rows";

function ev(id: string, tier: "A" | "B" | "C", stations: number, t: number, extra: Record<string, unknown> = {}) {
  const base = makeEvent(id, 0);
  return { ...base, tier, t, quality: { ...base.quality, nStations: stations }, ...extra };
}

const events = [
  ev("c-few", "C", 4, 30),
  ev("a-many", "A", 20, 50, { magnitude: { value: 1.2, type: "ML_cal", sigma: null }, catalogMatch: { catalogId: "uu1", dtS: 0, distM: 1 } }),
  ev("a-few", "A", 9, 10, { depthKm: 3.5 }),
  ev("b-mid", "B", 12, 20, { magnitude: { value: 0.4, type: "ML_cal", sigma: null } }),
];
const rows = events.map(toRow);
const ids = (r: { id: string }[]) => r.map((x) => x.id);

describe("toRow()", () => {
  it("takes every value from the event record", () => {
    const row = toRow(events[1]);
    expect(row).toMatchObject({ id: "a-many", tier: "A", stations: 20, magnitude: 1.2, magnitudeType: "ML_cal", inCatalog: true });
    expect(row.timeUtc).toBe("00:00:50");
    expect(toRow(events[0])).toMatchObject({ magnitude: null, magnitudeType: null, inCatalog: false });
  });
});

describe("sortRows()", () => {
  it("defaults to tier, then most stations", () => {
    expect(ids(sortRows(rows, null, "all"))).toEqual(["a-many", "a-few", "b-mid", "c-few"]);
  });

  it("sorts by a column in either direction, ties broken by the default order", () => {
    expect(ids(sortRows(rows, { key: "time", dir: "asc" }, "all"))).toEqual(["a-few", "b-mid", "c-few", "a-many"]);
    expect(ids(sortRows(rows, { key: "stations", dir: "desc" }, "all"))).toEqual(["a-many", "b-mid", "a-few", "c-few"]);
    expect(ids(sortRows(rows, { key: "catalog", dir: "asc" }, "all"))[0]).toBe("a-many");
  });

  it("puts missing magnitudes last in both directions", () => {
    expect(ids(sortRows(rows, { key: "magnitude", dir: "desc" }, "all")).slice(0, 2)).toEqual(["a-many", "b-mid"]);
    expect(ids(sortRows(rows, { key: "magnitude", dir: "asc" }, "all")).slice(0, 2)).toEqual(["b-mid", "a-many"]);
  });

  it("filters by tier without touching the input", () => {
    expect(ids(sortRows(rows, null, "A"))).toEqual(["a-many", "a-few"]);
    expect(sortRows(rows, null, "B")).toHaveLength(1);
    expect(ids(rows)).toEqual(["c-few", "a-many", "a-few", "b-mid"]);
  });

  it("sorts a list the size of the showcase run quickly", () => {
    const many = Array.from({ length: 700 }, (_, i) => toRow(ev(`e${i}`, (["A", "B", "C"] as const)[i % 3], i % 30, i)));
    const t0 = performance.now();
    for (let k = 0; k < 20; k++) sortRows(many, { key: "stations", dir: "desc" }, "all");
    expect(performance.now() - t0).toBeLessThan(500);
  });
});

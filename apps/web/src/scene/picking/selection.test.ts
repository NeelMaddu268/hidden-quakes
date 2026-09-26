import { describe, expect, it } from "vitest";
import { appearTimeOf, TIMELINE } from "../reveal/timeline";
import type { CatalogEvent } from "../types";
import { candidatePickable, publicSelectTargets, selectedInstanceIndex } from "./selection";

const A = 0;
const B = 1;
const C = 2;

describe("candidatePickable (matches what the candidate layer draws)", () => {
  it("nothing before the reveal or under PUBLIC", () => {
    expect(candidatePickable("public", "all", 0, A, 0)).toBe(false);
    expect(candidatePickable("public", "strict", 0, A, 0)).toBe(false);
    expect(candidatePickable("revealed", "public", 0, A, 0)).toBe(false);
  });

  it("every tier under ALL once revealed", () => {
    for (const t of [A, B, C]) expect(candidatePickable("revealed", "all", 0, t, 1)).toBe(true);
  });

  it("only Tier A under STRICT (B and C are faded out)", () => {
    expect(candidatePickable("revealed", "strict", 0, A, 0.5)).toBe(true);
    expect(candidatePickable("revealed", "strict", 0, B, 0.5)).toBe(false);
    expect(candidatePickable("revealed", "strict", 0, C, 0.5)).toBe(false);
  });

  it("while revealing, only instances that have appeared on the reveal clock", () => {
    const slot = 0.5;
    const at = appearTimeOf(slot);
    expect(at).toBeGreaterThan(TIMELINE.events.startS);
    expect(candidatePickable("revealing", "all", at - 0.01, A, at)).toBe(false);
    expect(candidatePickable("revealing", "all", at + 0.01, A, at)).toBe(true);
    expect(candidatePickable("revealing", "all", 0, A, appearTimeOf(0))).toBe(false); // first event appears at 1.0 s
  });
});

describe("publicSelectTargets", () => {
  const cat = (id: string, matchedEventId?: string | null): CatalogEvent => ({
    id,
    source: "test",
    t: 0,
    latitude: 0,
    longitude: 0,
    depthKm: 0,
    depthDatum: "test",
    elevM: 0,
    enu: { e: 0, n: 0, u: 0 },
    mag: null,
    magType: null,
    matchedEventId: matchedEventId ?? null,
  });

  it("maps each public event to its matched candidate, or null", () => {
    const ids = new Map([
      ["hq-1", 0],
      ["hq-2", 1],
    ]);
    const targets = publicSelectTargets([cat("p0", "hq-2"), cat("p1", null), cat("p2"), cat("p3", "hq-missing")], ids);
    expect(targets).toEqual(["hq-2", null, null, null]);
  });
});

describe("selectedInstanceIndex", () => {
  const ids = new Map([
    ["hq-1", 0],
    ["hq-2", 7],
  ]);
  it("is the instance index, or −1 for none / unknown", () => {
    expect(selectedInstanceIndex("hq-2", ids)).toBe(7);
    expect(selectedInstanceIndex(null, ids)).toBe(-1);
    expect(selectedInstanceIndex("dev-pub-0", ids)).toBe(-1);
  });
});

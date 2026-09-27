import { afterEach, describe, expect, it } from "vitest";
import { initialDemoState, useDemo } from "../state/demo";
import { hiddenHeroEventId, selectHiddenHero } from "./hiddenHero";

// Test-local records (only the fields the rule reads).
const ev = (id: string, tier: "A" | "B" | "C", nStations: number, rmsS: number, revealOrder: number, matched = false) => ({
  id,
  tier,
  revealOrder,
  catalogMatch: matched ? { catalogId: "p", dtS: 0.1, distM: 100 } : null,
  quality: { nStations, rmsS },
});

afterEach(() => useDemo.setState({ ...initialDemoState }));

describe("hiddenHeroEventId", () => {
  it("picks the unmatched Tier A event most stations agreed on", () => {
    const events = [
      ev("matched-a", "A", 30, 0.01, 0, true), // in the public catalog: never the hidden hero
      ev("b-unmatched", "B", 40, 0.01, 1), // not Tier A
      ev("a-19", "A", 19, 0.01, 2),
      ev("a-23", "A", 23, 0.03, 3),
    ];
    expect(hiddenHeroEventId(events)).toBe("a-23");
  });

  it("breaks ties on the lowest residual, then the earliest reveal slot", () => {
    expect(hiddenHeroEventId([ev("x", "A", 23, 0.029, 1), ev("y", "A", 23, 0.022, 5)])).toBe("y");
    expect(hiddenHeroEventId([ev("x", "A", 23, 0.02, 9), ev("y", "A", 23, 0.02, 4)])).toBe("y");
  });

  it("is null when every Tier A event is in the public catalog", () => {
    expect(hiddenHeroEventId([ev("m", "A", 23, 0.02, 0, true), ev("c", "C", 5, 0.1, 1)])).toBeNull();
    expect(hiddenHeroEventId([])).toBeNull();
  });
});

describe("selectHiddenHero", () => {
  it("opens the drawer on it, or leaves the selection alone when there is none", () => {
    expect(selectHiddenHero([ev("h", "A", 20, 0.02, 0)])).toBe("h");
    expect(useDemo.getState().selectedEventId).toBe("h");
    expect(selectHiddenHero([ev("m", "A", 20, 0.02, 0, true)])).toBeNull();
    expect(useDemo.getState().selectedEventId).toBe("h");
  });
});

import { motion } from "@hq/visualization";
import { beforeEach, describe, expect, it } from "vitest";
import { useDemo } from "../../state/demo";
import { resetSceneFx, sceneFx } from "../fx";
import { TIMELINE } from "../reveal/timeline";
import { createFilterLookDriver } from "./driver";
import { effectiveFilter, FILTER_LOOK } from "./fade";

const store = () => useDemo.getState();

beforeEach(() => {
  store().reset();
  resetSceneFx();
});

describe("effectiveFilter", () => {
  it("keeps the start look before the reveal, whatever was pressed", () => {
    expect(effectiveFilter("public", "strict")).toBe("public");
    expect(effectiveFilter("public", "all")).toBe("public");
    expect(effectiveFilter("revealing", "strict")).toBe("strict");
    expect(effectiveFilter("revealed", "all")).toBe("all");
  });
});

describe("filter look driver", () => {
  it("S before the reveal leaves the start frame untouched, then applies STRICT once the reveal starts", () => {
    const d = createFilterLookDriver();
    store().setFilter("strict");
    for (let i = 0; i < 60; i++) d.step(1 / 60);
    expect({ ...sceneFx.filterLook }).toEqual({ ...FILTER_LOOK.public });
    store().reveal();
    for (let i = 0; i < 60; i++) d.step(1 / 60);
    expect({ ...sceneFx.filterLook }).toEqual({ ...FILTER_LOOK.strict });
  });

  it("reset() mid-fade snaps to the exact start look, and the next step changes nothing", () => {
    const d = createFilterLookDriver();
    store().reveal();
    store().setFilter("strict");
    for (let i = 0; i < 10; i++) d.step(1 / 60); // mid-fade
    store().reset();
    d.onReset();
    expect({ ...sceneFx.filterLook }).toEqual({ ...FILTER_LOOK.public });
    d.step(1 / 60);
    expect({ ...sceneFx.filterLook }).toEqual({ ...FILTER_LOOK.public });
  });

  it("finishes the PUBLIC → ALL fade before the first event pops", () => {
    expect(TIMELINE.events.startS).toBeGreaterThanOrEqual(motion.state / 1000);
  });

  it("a zero duration (reduced motion) switches instantly", () => {
    const d = createFilterLookDriver(0);
    store().reveal();
    d.step(1 / 60);
    expect({ ...sceneFx.filterLook }).toEqual({ ...FILTER_LOOK.all });
  });
});

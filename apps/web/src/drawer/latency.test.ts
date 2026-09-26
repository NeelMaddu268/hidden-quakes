import { describe, expect, it } from "vitest";
import { formatOpenLatency, OPEN_BUDGET_MS } from "./latency";

describe("formatOpenLatency", () => {
  it("reports the three stages against the 200 ms budget", () => {
    expect(OPEN_BUDGET_MS).toBe(200);
    expect(formatOpenLatency("hq-1", 18.24, 21, 37.5)).toBe(
      "[drawer] hq-1: traces committed 18.2 ms · frame 21.0 ms · painted 37.5 ms after select() (budget 200 ms)",
    );
  });
  it("flags an open over budget", () => {
    expect(formatOpenLatency("hq-1", 20, 25, 250)).toMatch(/OVER BUDGET$/);
  });
});

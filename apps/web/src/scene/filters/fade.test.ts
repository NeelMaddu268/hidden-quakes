import { describe, expect, it } from "vitest";
import { createFilterFade, FILTER_LOOK, STRICT_PUBLIC_WEIGHT } from "./fade";

describe("FILTER_LOOK", () => {
  it("encodes the lane doc: tiers A 1.0 / B 0.6 / C 0.3, STRICT fades B and C to 0.05, halos only under STRICT", () => {
    expect([FILTER_LOOK.all.tierA, FILTER_LOOK.all.tierB, FILTER_LOOK.all.tierC]).toEqual([1, 0.6, 0.3]);
    expect([FILTER_LOOK.strict.tierA, FILTER_LOOK.strict.tierB, FILTER_LOOK.strict.tierC]).toEqual([1, 0.05, 0.05]);
    expect(FILTER_LOOK.public.candidates).toBe(0);
    expect(FILTER_LOOK.all.halos).toBe(0);
    expect(FILTER_LOOK.strict.halos).toBe(1);
    expect(FILTER_LOOK.strict.publicLayer).toBe(STRICT_PUBLIC_WEIGHT);
  });
});

describe("filter fade", () => {
  it("reaches STRICT exactly after 600 ms, whatever the frame rate", () => {
    for (const fps of [30, 60, 144]) {
      const fade = createFilterFade("all");
      for (let i = 0; i < Math.ceil(0.6 * fps) + 1; i++) fade.step("strict", 1 / fps);
      expect({ ...fade.current }).toEqual({ ...FILTER_LOOK.strict });
    }
  });

  it("is part-way at 300 ms (cubic-out: more than half)", () => {
    const fade = createFilterFade("all");
    for (let i = 0; i < 18; i++) fade.step("strict", 1 / 60);
    expect(fade.current.tierB).toBeLessThan(0.6);
    expect(fade.current.tierB).toBeGreaterThan(0.05);
    expect((0.6 - fade.current.tierB) / (0.6 - 0.05)).toBeGreaterThan(0.5);
  });

  it("retargeting mid-fade eases from where it is, with no jump", () => {
    const fade = createFilterFade("all");
    for (let i = 0; i < 12; i++) fade.step("strict", 1 / 60);
    const mid = fade.current.tierB;
    fade.step("all", 1 / 60);
    expect(Math.abs(fade.current.tierB - mid)).toBeLessThan(0.05);
    for (let i = 0; i < 60; i++) fade.step("all", 1 / 60);
    expect({ ...fade.current }).toEqual({ ...FILTER_LOOK.all });
  });

  it("snap jumps straight to a look (start frame)", () => {
    const fade = createFilterFade("strict");
    fade.snap("public");
    expect({ ...fade.current }).toEqual({ ...FILTER_LOOK.public });
  });

  it("does not allocate a new current object per step (the scene holds a reference)", () => {
    const fade = createFilterFade("all");
    const ref = fade.current;
    fade.step("strict", 0.1);
    expect(fade.current).toBe(ref);
  });
});

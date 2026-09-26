import { describe, expect, it } from "vitest";
import { depthFogPerSceneUnit, LOOK, msaaSamplesFor, publicSwatch } from "./look";

describe("look", () => {
  it("keeps fog per km of real depth under vertical exaggeration", () => {
    expect(depthFogPerSceneUnit(1)).toBe(LOOK.depthFogPerKm);
    expect(depthFogPerSceneUnit(2) * 2).toBeCloseTo(LOOK.depthFogPerKm, 12);
  });

  it("drops MSAA only for large drawing buffers (a DPR-1 4K screen included)", () => {
    expect(msaaSamplesFor(1280 * 720 * 4)).toBe(LOOK.msaa.samples);
    expect(msaaSamplesFor(3840 * 2160)).toBe(LOOK.msaa.largeBufferSamples);
  });

  it("gives legends the displayed public color, darker than the raw token at glow < 1", () => {
    const swatch = publicSwatch("#DCE6F2");
    expect(swatch).toMatch(/^#[0-9A-F]{6}$/);
    expect(parseInt(swatch.slice(1, 3), 16)).toBeLessThan(0xdc);
  });

  it("is frozen", () => {
    expect(Object.isFrozen(LOOK)).toBe(true);
    for (const v of Object.values(LOOK)) if (typeof v === "object") expect(Object.isFrozen(v)).toBe(true);
  });
});

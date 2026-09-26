import { describe, expect, it } from "vitest";
import { createHaloUniforms, HALO_FRAGMENT_SHADER, HALO_VERTEX_SHADER } from "./haloMaterial";
import { TIME_ALL } from "../time/clock";

describe("halo shaders", () => {
  it("declare exactly the uniforms the factory provides", () => {
    const uniforms = createHaloUniforms("#FFD08A");
    const src = HALO_VERTEX_SHADER + HALO_FRAGMENT_SHADER;
    const declared = [...new Set([...src.matchAll(/uniform\s+\w+\s+(\w+);/g)].map((m) => m[1]))].sort();
    expect(declared).toEqual(Object.keys(uniforms).sort());
    expect(src).not.toMatch(/\$\{|undefined|NaN/);
  });

  it("clamp the fresnel base so pow() never sees a negative (NaN would smear through bloom)", () => {
    expect(HALO_FRAGMENT_SHADER).toMatch(/clamp\(abs\(dot\(/);
  });

  it("start hidden: zero opacity and a reveal clock before every appearance", () => {
    const u = createHaloUniforms("#FFD08A");
    expect(u.uOpacity.value).toBe(0);
    expect(u.uRevealElapsed.value).toBeLessThan(0);
  });
});

describe("time mode (WEB-06)", () => {
  it("shows a halo only once time mode's now reaches its event, and defaults to off", () => {
    expect(HALO_VERTEX_SHADER).toContain("attribute float aTime;");
    expect(HALO_VERTEX_SHADER).toContain("step(aTime, uTimeNow)");
    expect(createHaloUniforms("#FFD08A").uTimeNow.value).toBe(TIME_ALL);
  });
});

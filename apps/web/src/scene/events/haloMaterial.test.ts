import { describe, expect, it } from "vitest";
import { createHaloUniforms, HALO_FRAGMENT_SHADER, HALO_VERTEX_SHADER } from "./haloMaterial";

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

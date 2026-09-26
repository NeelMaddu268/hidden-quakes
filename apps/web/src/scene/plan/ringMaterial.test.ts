import { InstancedMesh, Matrix4, PlaneGeometry, Quaternion, ShaderMaterial, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { LOOK } from "../look";
import { createRingUniforms, RING_FRAGMENT_SHADER, RING_VERTEX_SHADER, writeRingMatrices } from "./ringMaterial";
import { TIME_ALL } from "../time/clock";

describe("plan ring shaders", () => {
  it("declare exactly the uniforms the factory provides", () => {
    const u = createRingUniforms("#FFD08A");
    const src = RING_VERTEX_SHADER + RING_FRAGMENT_SHADER;
    const declared = [...new Set([...src.matchAll(/uniform\s+\w+\s+(\w+);/g)].map((m) => m[1]))].sort();
    expect(declared).toEqual(Object.keys(u).sort());
    expect(src).not.toMatch(/\$\{|undefined|NaN/);
  });

  it("lay the quad flat in the east/north plane (y stays at the event)", () => {
    expect(RING_VERTEX_SHADER).toMatch(/vec3\(position\.x, 0\.0, -position\.y\)/);
  });

  it("start hidden", () => {
    const u = createRingUniforms("#FFD08A");
    expect(u.uOpacity.value).toBe(0);
    expect(u.uRevealElapsed.value).toBeLessThan(0);
  });

  it("peak below the bloom threshold, like the 3D halos (strictHalo luminance ≈ 0.68)", () => {
    expect(0.682 * LOOK.halos.rimAlpha).toBeLessThan(LOOK.bloom.luminanceThreshold);
  });
});

describe("writeRingMatrices", () => {
  it("places each ring at its event and scales x/z to the diameter, y untouched", () => {
    const positions = new Float32Array([1, -2, -3, -0.5, -4, 0.25]);
    const radii = new Float32Array([0.2, 0.05]);
    const mesh = new InstancedMesh(new PlaneGeometry(1, 1), new ShaderMaterial(), 2);
    writeRingMatrices(positions, radii, mesh.instanceMatrix.array as Float32Array);
    const m = new Matrix4();
    const p = new Vector3();
    const s = new Vector3();
    mesh.getMatrixAt(0, m);
    m.decompose(p, new Quaternion(), s);
    expect(p.toArray().map((v) => Number(v.toFixed(6)))).toEqual([1, -2, -3]);
    expect(Number(s.x.toFixed(6))).toBe(0.4);
    expect(s.y).toBe(1);
    expect(Number(s.z.toFixed(6))).toBe(0.4);
  });

  it("fails loudly when arrays disagree", () => {
    expect(() => writeRingMatrices(new Float32Array(3), new Float32Array(2), new Float32Array(32))).toThrow();
  });
});

describe("time mode (WEB-06)", () => {
  it("shows a ring only once time mode's now reaches its event, and defaults to off", () => {
    expect(RING_VERTEX_SHADER).toContain("attribute float aTime;");
    expect(RING_VERTEX_SHADER).toContain("step(aTime, uTimeNow)");
    expect(createRingUniforms("#FFD08A").uTimeNow.value).toBe(TIME_ALL);
  });
});

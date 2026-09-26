import { InstancedMesh, Matrix4, PlaneGeometry, ShaderMaterial, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { enuToScene } from "../coords";
import { LOOK } from "../look";
import { TIME_ALL } from "../time/clock";
import { createEventUniforms, EVENT_FRAGMENT_SHADER, EVENT_VERTEX_SHADER, writeInstanceMatrices } from "./material";

describe("writeInstanceMatrices", () => {
  it("places each instance at its scene position (what the shader reads as the glyph center)", () => {
    const enu = [
      { e: 2500, n: 1200, u: -3400 },
      { e: -800, n: -400, u: -1200 },
    ];
    const positions = new Float32Array(enu.flatMap((p) => enuToScene(p)));
    const mesh = new InstancedMesh(new PlaneGeometry(1, 1), new ShaderMaterial(), enu.length);
    writeInstanceMatrices(positions, mesh.instanceMatrix.array as Float32Array);
    const m = new Matrix4();
    const p = new Vector3();
    enu.forEach((point, i) => {
      mesh.getMatrixAt(i, m);
      p.setFromMatrixPosition(m);
      const [x, y, z] = enuToScene(point);
      expect(p.x).toBeCloseTo(x, 5);
      expect(p.y).toBeCloseTo(y, 5);
      expect(p.z).toBeCloseTo(z, 5);
      // translation only: the upper 3×3 is identity
      const e = m.elements;
      expect([e[0], e[1], e[2], e[4], e[5], e[6], e[8], e[9], e[10], e[15]]).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1, 1]);
    });
  });

  it("refuses an array too small for the instances instead of writing out of bounds", () => {
    expect(() => writeInstanceMatrices(new Float32Array(6), new Float32Array(16))).toThrow();
  });
});

describe("event shaders", () => {
  it("declare every uniform the uniforms factory provides, and nothing undefined", () => {
    const uniforms = createEventUniforms({ color: "#FFB547", size: 0.06, minPx: 2, maxPx: 18 });
    const src = EVENT_VERTEX_SHADER + EVENT_FRAGMENT_SHADER;
    for (const name of Object.keys(uniforms)) expect(src).toContain(name);
    const declared = [...src.matchAll(/uniform\s+\w+\s+(\w+);/g)].map((m) => m[1]);
    for (const name of declared) expect(Object.keys(uniforms)).toContain(name);
    expect(src).not.toMatch(/\$\{|undefined|NaN/);
  });
});

describe("createEventUniforms", () => {
  it("rejects an empty or inverted pixel clamp (GLSL clamp is undefined when lo > hi)", () => {
    expect(() => createEventUniforms({ color: "#FFFFFF", size: 0.06, minPx: 20, maxPx: 18 })).toThrow(/minPx/);
    expect(() => createEventUniforms({ color: "#FFFFFF", size: 0.06, minPx: 0, maxPx: 18 })).toThrow(/minPx/);
  });
});

describe("time mode in the event shader (WEB-06)", () => {
  it("hides an instance until tNow reaches its time and lights recent ones in the token hue (no whitening)", () => {
    expect(EVENT_VERTEX_SHADER).toContain("attribute float aTime;");
    expect(EVENT_VERTEX_SHADER).toContain("float tShown = step(0.0, tAge);");
    expect(EVENT_VERTEX_SHADER).toMatch(/vAlpha = [^;]*\* tShown \*/);
    // The whitening mix is the reveal pop's alone: a recent amber event never turns white (public).
    const whiten = EVENT_FRAGMENT_SHADER.match(/mix\(uColor, vec3\(1\.0\), ([^;]*)\);/);
    expect(whiten).not.toBeNull();
    expect(whiten![1]).not.toContain("vWarm");
    expect(EVENT_FRAGMENT_SHADER).toContain("(1.0 + vBoost + vWarm)");
  });

  it("defaults to time mode off: every instance shown, none lit", () => {
    const u = createEventUniforms({ color: "#FFB547", size: 0.06, minPx: 2, maxPx: 18 });
    expect(u.uTimeNow.value).toBe(TIME_ALL);
    expect(u.uTimeGlowS.value).toBe(LOOK.time.glowWindowS);
  });
});

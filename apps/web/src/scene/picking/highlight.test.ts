import { colors } from "@hq/visualization";
import { Color } from "three";
import { describe, expect, it } from "vitest";
import { createEventUniforms, EVENT_FRAGMENT_SHADER, EVENT_VERTEX_SHADER } from "../events/material";

// The selected-instance highlight (WEB-05) lives in the event material. WebGL isn't available in the
// test runner, so these check the uniform defaults and the shader wiring; the look is verified in the
// browser.

describe("selection highlight uniforms", () => {
  const u = createEventUniforms({ color: colors.recovered, size: 0.06, minPx: 2, maxPx: 18 });

  it("starts with nothing selected", () => {
    expect(u.uSelected.value).toBe(-1);
  });

  it("rings the selected event in the strict-halo color", () => {
    expect(u.uRingColor.value.equals(new Color(colors.strictHalo))).toBe(true);
  });

  it("the shaders read the selection per instance and draw the ring", () => {
    expect(EVENT_VERTEX_SHADER).toContain("uniform float uSelected;");
    expect(EVENT_VERTEX_SHADER).toContain("gl_InstanceID");
    expect(EVENT_FRAGMENT_SHADER).toContain("uniform vec3 uRingColor;");
    // The template-interpolated size constants must land as GLSL float literals.
    expect(EVENT_VERTEX_SHADER).not.toMatch(/\$\{|undefined|NaN/);
    expect(EVENT_VERTEX_SHADER).toMatch(/radius \* \d+\.\d+/);
  });
});

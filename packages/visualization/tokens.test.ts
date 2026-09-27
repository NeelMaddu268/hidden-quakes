import { describe, expect, it } from "vitest";
import {
  colors,
  cssVariables,
  easeOutCubic,
  fonts,
  motion,
  strictFadeOpacity,
  tierStyle,
} from "./tokens";

/** y(x) for a CSS cubic-bezier(x1, y1, x2, y2), solved by bisection on x(t). */
function cubicBezierAt(x: number, x1: number, y1: number, x2: number, y2: number): number {
  const bez = (t: number, p1: number, p2: number) =>
    3 * (1 - t) * (1 - t) * t * p1 + 3 * (1 - t) * t * t * p2 + t * t * t;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (bez(mid, x1, x2) < x) lo = mid;
    else hi = mid;
  }
  return bez((lo + hi) / 2, y1, y2);
}

describe("colors", () => {
  it("has exactly the 14 palette tokens", () => {
    expect(Object.keys(colors)).toHaveLength(14);
  });

  it("are all 6-digit hex strings", () => {
    for (const value of Object.values(colors)) expect(value).toMatch(/^#[0-9A-F]{6}$/);
  });
});

describe("fonts", () => {
  it("name Inter for UI and JetBrains Mono for numbers, with system fallbacks", () => {
    expect(fonts.ui).toContain("Inter");
    expect(fonts.ui).toMatch(/sans-serif$/);
    expect(fonts.mono).toContain("JetBrains Mono");
    expect(fonts.mono).toMatch(/monospace$/);
  });

  it("give every CSS var() a fallback so an undefined variable never invalidates the declaration", () => {
    for (const family of [fonts.ui, fonts.mono]) {
      for (const match of family.matchAll(/var\(([^)]*)\)/g)) expect(match[1]).toContain(",");
    }
  });
});

describe("motion", () => {
  it("uses the docs/02 durations", () => {
    expect(motion.micro).toBe(150);
    expect(motion.state).toBe(600);
    expect(motion.scene).toBe(1200);
  });

  it("CSS ease and JS easeOutCubic describe the same curve", () => {
    const m = /^cubic-bezier\(([^)]+)\)$/.exec(motion.ease);
    expect(m).not.toBeNull();
    const [x1, y1, x2, y2] = m![1].split(",").map(Number);
    for (let x = 0; x <= 1.0001; x += 0.05) {
      expect(Math.abs(cubicBezierAt(x, x1, y1, x2, y2) - easeOutCubic(x))).toBeLessThan(0.02);
    }
  });
});

describe("easeOutCubic", () => {
  it("hits its endpoints exactly and clamps outside [0, 1]", () => {
    expect(easeOutCubic(0)).toBe(0);
    expect(easeOutCubic(1)).toBe(1);
    expect(easeOutCubic(-3)).toBe(0);
    expect(easeOutCubic(7)).toBe(1);
  });

  it("is monotonic and front-loaded (cubic-out)", () => {
    let prev = -Infinity;
    for (let t = 0; t <= 1; t += 0.01) {
      const v = easeOutCubic(t);
      expect(v).toBeGreaterThanOrEqual(prev);
      prev = v;
    }
    expect(easeOutCubic(0.5)).toBeCloseTo(0.875, 10);
  });
});

describe("tierStyle", () => {
  it("sets opacity A 1.0, B 0.6, C 0.3 per the lane doc", () => {
    expect(tierStyle.A.opacity).toBe(1.0);
    expect(tierStyle.B.opacity).toBe(0.6);
    expect(tierStyle.C.opacity).toBe(0.3);
  });

  it("carries tier in size too, strictly decreasing A > B > C", () => {
    expect(tierStyle.A.size).toBeGreaterThan(tierStyle.B.size);
    expect(tierStyle.B.size).toBeGreaterThan(tierStyle.C.size);
  });

  it("fades B and C under STRICT to the lane doc's 0.05", () => {
    expect(strictFadeOpacity).toBe(0.05);
  });
});

describe("cssVariables", () => {
  it("exposes every color as a kebab-case --hq-* property", () => {
    const vars = cssVariables();
    expect(vars["--hq-bg"]).toBe(colors.bg);
    expect(vars["--hq-text-dim"]).toBe(colors.textDim);
    expect(vars["--hq-strict-halo"]).toBe(colors.strictHalo);
    expect(vars["--hq-pick-p"]).toBe(colors.pickP);
    expect(vars["--hq-motion-scene"]).toBe("1200ms");
    const colorVars = Object.keys(vars).filter((k) => Object.values(colors).includes(vars[k]));
    expect(colorVars).toHaveLength(Object.keys(colors).length);
  });
});

// Event glyph material: camera-facing discs with a hot core, drawn from one InstancedMesh per layer.
// Everything that animates (reveal, tier/filter opacity, layer opacity) is a uniform, so per-frame
// updates are a handful of float writes and never touch per-instance data.

import { colors } from "@hq/visualization";
import { Color, Vector3, type IUniform } from "three";
import { LOOK } from "../look";
import { TIMELINE } from "../reveal/timeline";
import { TIME_ALL } from "../time/clock";

export interface EventUniforms {
  [name: string]: IUniform;
  uColor: IUniform<Color>;
  /** Base glyph radius in scene units (km) before tier scaling. */
  uSize: IUniform<number>;
  /** Minimum on-screen base radius in device pixels (before tier scaling; see GLYPH_FLOOR_PX). */
  uMinPx: IUniform<number>;
  /** Maximum on-screen radius in device pixels, so a close-up never fills the screen with quads. */
  uMaxPx: IUniform<number>;
  /** Drawing-buffer height in device pixels (set on resize). */
  uViewportHeight: IUniform<number>;
  /**
   * Seconds since reveal() on the reveal clock. Each instance appears when this passes its
   * `aAppearAt` (computed on the CPU with scene/reveal/timeline.ts → appearTimeOf, so the pops and the
   * shell's counter can never disagree). −1 hides every candidate.
   */
  uRevealElapsed: IUniform<number>;
  /** Seconds each instance's pop (scale 2 → 1, brightness spike, settle) lasts after it appears. */
  uPopS: IUniform<number>;
  /** Scene y of the site surface; fog increases with depth below it. */
  uSurfaceY: IUniform<number>;
  /** Exponential depth-fog density per scene unit below the surface (per km ÷ vertical exaggeration). */
  uDepthFog: IUniform<number>;
  /** Opacity for tiers A, B, C (the filter drives B and C). */
  uTierOpacity: IUniform<Vector3>;
  /** Whole-layer opacity (the PUBLIC filter fades the candidate layer out). */
  uLayerOpacity: IUniform<number>;
  /** Core intensity: 1 draws the token color exactly; above 1 renders HDR cores for bloom to catch. */
  uGlow: IUniform<number>;
  /** Instance index of the selected event in this layer (WEB-05), or −1 for none. */
  uSelected: IUniform<number>;
  /** Color of the thin ring around the selected event. */
  uRingColor: IUniform<Color>;
  /**
   * Time mode (WEB-06): "now" in seconds since windowStart. Instances whose `aTime` is later are hidden;
   * those from the last `uTimeGlowS` glow warmer and larger, decaying to their settled look. TIME_ALL
   * (time mode off) shows every instance and lights none.
   */
  uTimeNow: IUniform<number>;
  uTimeGlowS: IUniform<number>;
}

/** Room around the clamped glyph for the selection ring. */
export const SELECTED_QUAD_SCALE = 2.6;
export const SELECTED_MIN_PX_FACTOR = 5;

// Glyph-shape constants live here (not scene/look.ts) because they're baked into the shader source.
/** Absolute smallest glyph radius in device pixels after tier scaling, so Tier C never shimmers out. */
export const GLYPH_FLOOR_PX = 1.25;
/** A popping instance starts at this multiple of its size and settles to 1. */
export const POP_SCALE = 2.0;
/** Extra brightness at the start of a pop (0.8 = 1.8× the settled intensity), decaying to 0. */
export const POP_FLASH = 0.8;
/** How far toward white a pop starts (a white-hot spark that settles into the token color). */
export const POP_WHITEN = 0.4;


export const EVENT_VERTEX_SHADER = /* glsl */ `
  attribute float aTier;
  attribute float aScale;
  attribute float aAppearAt;
  attribute float aTime;

  uniform float uSize;
  uniform float uMinPx;
  uniform float uMaxPx;
  uniform float uViewportHeight;
  uniform float uRevealElapsed;
  uniform float uPopS;
  uniform float uSurfaceY;
  uniform float uDepthFog;
  uniform vec3 uTierOpacity;
  uniform float uLayerOpacity;
  uniform float uSelected;
  uniform float uTimeNow;
  uniform float uTimeGlowS;

  varying vec2 vUv;
  varying float vAlpha;
  varying float vBoost;
  varying float vWarm;
  varying float vSelected;
  varying float vCoreFrac;

  void main() {
    vec4 worldCenter = modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
    vec4 mvCenter = viewMatrix * worldCenter;

    // aAppearAt < 0 marks always-visible instances (public catalog): settled, whatever the clock says.
    float age = aAppearAt < 0.0 ? 1.0e4 : uRevealElapsed - aAppearAt;
    float shown = step(0.0, age);
    float pop = clamp(age / max(uPopS, 1e-6), 0.0, 1.0);
    float settle = 1.0 - pow(1.0 - pop, 3.0);
    vBoost = (1.0 - settle) * ${POP_FLASH.toFixed(3)};

    // Time mode: shown once tNow reaches the event's origin time; the last uTimeGlowS of data time glow.
    float tAge = uTimeNow - aTime;
    float tShown = step(0.0, tAge);
    float recent = tShown * clamp(1.0 - tAge / max(uTimeGlowS, 1e-6), 0.0, 1.0);
    vWarm = recent * ${LOOK.time.glowBoost.toFixed(3)};

    float tierOpacity = aTier < 0.5 ? uTierOpacity.x : (aTier < 1.5 ? uTierOpacity.y : uTierOpacity.z);
    float fog = exp(-uDepthFog * max(0.0, uSurfaceY - worldCenter.y));
    vAlpha = tierOpacity * uLayerOpacity * shown * tShown * fog;
    // Selection identifies the event without overriding its filter opacity.
    vSelected = abs(float(gl_InstanceID) - uSelected) < 0.5 ? 1.0 : 0.0;

    // Pixels per scene unit at this depth; perspective when projectionMatrix[2][3] == -1.
    float pxPerUnit = projectionMatrix[1][1] * uViewportHeight * 0.5;
    if (projectionMatrix[2][3] < -0.5) pxPerUnit /= max(-mvCenter.z, 1e-4);
    // Clamp the base size to [uMinPx, uMaxPx] on screen first, then scale by tier, so tier still reads
    // as size at every zoom.
    float radius = clamp(uSize, uMinPx / pxPerUnit, uMaxPx / pxPerUnit) * aScale;
    radius = max(radius, ${GLYPH_FLOOR_PX.toFixed(3)} / pxPerUnit);

    float quad = mix(radius,
      max(radius * ${SELECTED_QUAD_SCALE.toFixed(2)}, uMinPx * ${SELECTED_MIN_PX_FACTOR.toFixed(2)} / pxPerUnit),
      vSelected);
    vCoreFrac = radius / quad;
    radius = quad;
    // Hidden instances collapse to a point: no fragments, no overdraw.
    radius *= mix(${POP_SCALE.toFixed(3)}, 1.0, settle) * (1.0 + ${LOOK.time.glowGrow.toFixed(3)} * recent) * step(0.001, vAlpha);

    mvCenter.xy += position.xy * 2.0 * radius;
    gl_Position = projectionMatrix * mvCenter;
    vUv = uv;
  }
`;

export const EVENT_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;
  uniform float uGlow;
  uniform vec3 uRingColor;

  varying vec2 vUv;
  varying float vAlpha;
  varying float vBoost;
  varying float vWarm;
  varying float vSelected;
  varying float vCoreFrac;

  void main() {
    vec2 p = vUv * 2.0 - 1.0;
    float rq = length(p);
    float px = fwidth(rq);
    if (rq > 1.0) discard;
    vec2 g = p / vCoreFrac;
    float r2 = dot(g, g);
    float core = 1.0 - smoothstep(0.0, 0.3, r2);
    float falloff = max(0.0, 1.0 - r2);
    float halo = falloff * falloff * 0.35;
    // Additive blending multiplies rgb by alpha, so the disc profile lives in alpha only: the center of
    // a settled glyph at uGlow = 1 is exactly the token color.
    float shape = min(core + halo, 1.0);
    vec3 color = mix(uColor, vec3(1.0), vBoost * ${(POP_WHITEN / POP_FLASH).toFixed(4)});
    // The time-mode glow brightens in the token hue only (no whitening): recent amber stays amber.
    vec3 rgb = color * uGlow * (1.0 + vBoost + vWarm);
    float alpha = vAlpha * shape;
    if (vSelected > 0.5) {
      float ring = 1.0 - smoothstep(0.0, 1.5 * px, abs(rq - (1.0 - 2.0 * px)));
      rgb = mix(rgb, uRingColor, ring);
      alpha = max(alpha, ring * vAlpha);
    }
    if (alpha <= 0.0) discard;
    gl_FragColor = vec4(rgb, alpha);

    #include <colorspace_fragment>
  }
`;

export interface EventMaterialOptions {
  color: string;
  size: number;
  minPx: number;
  maxPx: number;
  glow?: number;
  depthFog?: number;
  surfaceY?: number;
}

/** Fresh uniforms for one layer. The layer's ShaderMaterial keeps this object; frames write `.value`s. */
export function createEventUniforms(opts: EventMaterialOptions): EventUniforms {
  if (!(opts.minPx > 0 && opts.minPx <= opts.maxPx)) {
    throw new Error(`event glyph pixel clamp needs 0 < minPx <= maxPx, got ${opts.minPx}..${opts.maxPx}`);
  }
  return {
    uColor: { value: new Color(opts.color) }, // sRGB hex → linear working space
    uSize: { value: opts.size },
    uMinPx: { value: opts.minPx },
    uMaxPx: { value: opts.maxPx },
    uViewportHeight: { value: 1 },
    uRevealElapsed: { value: -1 },
    uPopS: { value: TIMELINE.popS },
    uSurfaceY: { value: opts.surfaceY ?? 0 },
    uDepthFog: { value: opts.depthFog ?? 0 },
    uTierOpacity: { value: new Vector3(1, 1, 1) },
    uLayerOpacity: { value: 1 },
    uGlow: { value: opts.glow ?? 1 },
    uSelected: { value: -1 },
    uRingColor: { value: new Color(colors.strictHalo) },
    uTimeNow: { value: TIME_ALL },
    uTimeGlowS: { value: LOOK.time.glowWindowS },
  };
}

/** Writes translation-only instance matrices straight into the InstancedMesh's matrix array. */
export function writeInstanceMatrices(positions: Float32Array, matrixArray: Float32Array): void {
  const count = positions.length / 3;
  if (matrixArray.length < count * 16) throw new Error("instance matrix array too small");
  for (let i = 0; i < count; i++) {
    const m = i * 16;
    matrixArray.fill(0, m, m + 16);
    matrixArray[m] = 1;
    matrixArray[m + 5] = 1;
    matrixArray[m + 10] = 1;
    matrixArray[m + 15] = 1;
    matrixArray[m + 12] = positions[i * 3];
    matrixArray[m + 13] = positions[i * 3 + 1];
    matrixArray[m + 14] = positions[i * 3 + 2];
  }
}

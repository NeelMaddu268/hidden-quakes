// Event glyph material: camera-facing discs with a hot core, drawn from one InstancedMesh per layer.
// Everything that animates (reveal, tier/filter opacity, layer opacity) is a uniform, so per-frame
// updates are a handful of float writes and never touch per-instance data.

import { Color, Vector3, type IUniform } from "three";

export interface EventUniforms {
  [name: string]: IUniform;
  uColor: IUniform<Color>;
  /** Base glyph radius in scene units (km) before tier scaling. */
  uSize: IUniform<number>;
  /** Minimum on-screen radius in device pixels, so distant events never vanish. */
  uMinPx: IUniform<number>;
  /** Drawing-buffer height in device pixels (set on resize). */
  uViewportHeight: IUniform<number>;
  /** Instances with revealAt <= uReveal are shown. −1 hides every candidate; public (−1) always shows. */
  uReveal: IUniform<number>;
  /** Width, in revealAt units, of the pop (scale 2 → 1, brightness spike, settle) after appearing. */
  uPopWidth: IUniform<number>;
  /** Opacity for tiers A, B, C (the filter drives B and C). */
  uTierOpacity: IUniform<Vector3>;
  /** Whole-layer opacity (the PUBLIC filter fades the candidate layer out). */
  uLayerOpacity: IUniform<number>;
  /** Intensity multiplier; values above 1 push cores past the bloom threshold. */
  uGlow: IUniform<number>;
}

export const EVENT_VERTEX_SHADER = /* glsl */ `
  attribute float aTier;
  attribute float aScale;
  attribute float aRevealAt;

  uniform float uSize;
  uniform float uMinPx;
  uniform float uViewportHeight;
  uniform float uReveal;
  uniform float uPopWidth;
  uniform vec3 uTierOpacity;
  uniform float uLayerOpacity;

  varying vec2 vUv;
  varying float vAlpha;
  varying float vBoost;

  void main() {
    vec4 mvCenter = modelViewMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);

    float age = uReveal - aRevealAt;
    float shown = aRevealAt < 0.0 ? 1.0 : step(0.0, age);
    float pop = aRevealAt < 0.0 ? 1.0 : clamp(age / max(uPopWidth, 1e-6), 0.0, 1.0);
    float settle = 1.0 - pow(1.0 - pop, 3.0);
    vBoost = (1.0 - settle) * 1.5;

    float tierOpacity = aTier < 0.5 ? uTierOpacity.x : (aTier < 1.5 ? uTierOpacity.y : uTierOpacity.z);
    vAlpha = tierOpacity * uLayerOpacity * shown;

    // Pixels per scene unit at this depth; perspective when projectionMatrix[2][3] == -1.
    float pxPerUnit = projectionMatrix[1][1] * uViewportHeight * 0.5;
    if (projectionMatrix[2][3] < -0.5) pxPerUnit /= max(-mvCenter.z, 1e-4);
    float radius = max(uSize * aScale, uMinPx / pxPerUnit);
    // Hidden instances collapse to a point: no fragments, no overdraw.
    radius *= mix(2.0, 1.0, settle) * step(0.001, vAlpha);

    mvCenter.xy += position.xy * 2.0 * radius;
    gl_Position = projectionMatrix * mvCenter;
    vUv = uv;
  }
`;

export const EVENT_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;
  uniform float uGlow;

  varying vec2 vUv;
  varying float vAlpha;
  varying float vBoost;

  void main() {
    vec2 p = vUv * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;
    float core = 1.0 - smoothstep(0.0, 0.3, r2);
    float falloff = 1.0 - r2;
    float halo = falloff * falloff * 0.35;
    float shape = core + halo;
    gl_FragColor = vec4(uColor * shape * uGlow * (1.0 + vBoost), vAlpha * min(shape, 1.0));
    #include <colorspace_fragment>
  }
`;

export interface EventMaterialOptions {
  color: string;
  size: number;
  minPx: number;
  glow?: number;
}

/** Fresh uniforms for one layer. The layer's ShaderMaterial keeps this object; frames write `.value`s. */
export function createEventUniforms(opts: EventMaterialOptions): EventUniforms {
  return {
    uColor: { value: new Color(opts.color) }, // sRGB hex → linear working space
    uSize: { value: opts.size },
    uMinPx: { value: opts.minPx },
    uViewportHeight: { value: 1 },
    uReveal: { value: -1 },
    uPopWidth: { value: 0.05 },
    uTierOpacity: { value: new Vector3(1, 1, 1) },
    uLayerOpacity: { value: 1 },
    uGlow: { value: opts.glow ?? 1 },
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

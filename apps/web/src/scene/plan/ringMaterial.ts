// Plan-view uncertainty rings: a flat disc in the east/north plane per Tier A event, radius hErrM (the
// 68% horizontal semi-major axis, drawn as a circle: the contract carries no orientation). A thin
// constant-width outline plus a faint fill, MAX-blended like the 3D halos so overlaps never add up
// into something event-bright, and kept under the bloom threshold (scene/look.ts → halos).

import { Color, type IUniform } from "three";
import { LOOK } from "../look";
import { TIME_ALL } from "../time/clock";

export interface RingUniforms {
  [name: string]: IUniform;
  uColor: IUniform<Color>;
  /** Filter-driven opacity (0 outside STRICT). */
  uOpacity: IUniform<number>;
  /** Reveal clock (s): a ring shows once it passes its event's appearance time. */
  uRevealElapsed: IUniform<number>;
  /** Time mode "now" (s since windowStart): a ring shows only once its event does. TIME_ALL = off. */
  uTimeNow: IUniform<number>;
}

/** Outline width in device pixels, and the faint fill's weight relative to the outline. */
export const RING_LINE_PX = 1.5;
export const RING_FILL = 0.18;

export const RING_VERTEX_SHADER = /* glsl */ `
  attribute float aAppearAt;
  attribute float aTime;
  uniform float uRevealElapsed;
  uniform float uTimeNow;
  uniform float uOpacity;
  varying vec2 vUv;
  varying float vAlpha;

  void main() {
    vAlpha = uOpacity * step(aAppearAt, uRevealElapsed) * step(aTime, uTimeNow);
    vUv = uv;
    // The unit quad is authored in XY; lay it flat in the east/north (XZ) plane. Hidden rings
    // collapse onto their center: degenerate, but inside the clip volume.
    vec3 flatPos = vec3(position.x, 0.0, -position.y) * step(0.001, vAlpha);
    vec4 p = modelViewMatrix * instanceMatrix * vec4(flatPos, 1.0);
    gl_Position = projectionMatrix * p;
  }
`;

export const RING_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;
  varying vec2 vUv;
  varying float vAlpha;

  void main() {
    float d = length(vUv * 2.0 - 1.0);
    if (d > 1.0) discard;
    float aa = max(fwidth(d), 1e-5);
    float line = 1.0 - smoothstep(0.0, ${RING_LINE_PX.toFixed(2)} * aa, abs(d - (1.0 - ${RING_LINE_PX.toFixed(2)} * aa)));
    float a = vAlpha * ${LOOK.halos.rimAlpha.toFixed(3)} * max(line, ${RING_FILL.toFixed(3)});
    // MAX blending ignores blend factors, so premultiply here.
    gl_FragColor = vec4(uColor * a, a);
    #include <colorspace_fragment>
  }
`;

export function createRingUniforms(color: string): RingUniforms {
  return {
    uColor: { value: new Color(color) },
    uOpacity: { value: 0 },
    uRevealElapsed: { value: -1 },
    uTimeNow: { value: TIME_ALL },
  };
}

/**
 * Instance matrices for the rings: the unit quad (laid into the east/north plane by the vertex shader)
 * scaled to the ring diameter on x and z and placed at the event's scene position.
 */
export function writeRingMatrices(positions: Float32Array, radiiKm: Float32Array, out: Float32Array): void {
  const n = radiiKm.length;
  if (positions.length !== n * 3 || out.length < n * 16) throw new Error("plan rings: array sizes disagree");
  for (let i = 0; i < n; i++) {
    const m = i * 16;
    const d = radiiKm[i] * 2;
    out.fill(0, m, m + 16);
    out[m] = d; // x scale
    out[m + 5] = 1; // y (the quad is flat after rotation)
    out[m + 10] = d; // z scale
    out[m + 15] = 1;
    out[m + 12] = positions[i * 3];
    out[m + 13] = positions[i * 3 + 1];
    out[m + 14] = positions[i * 3 + 2];
  }
}

// Halo shader: a unit sphere scaled per instance to the 68% error ellipsoid, drawn as a soft rim
// (fresnel) so the outline reads as "the event is somewhere in here" without hiding the glyph.

import { Color, type IUniform } from "three";
import { TIMELINE } from "../reveal/timeline";

export interface HaloUniforms {
  [name: string]: IUniform;
  uColor: IUniform<Color>;
  /** Filter-driven halo opacity (0 outside STRICT). */
  uOpacity: IUniform<number>;
  uRevealElapsed: IUniform<number>;
  uEventsStart: IUniform<number>;
  uEventsDuration: IUniform<number>;
  uEventsExponent: IUniform<number>;
}

/** Peak rim alpha at full halo opacity: a halo is context, never brighter than its event. */
export const HALO_RIM_ALPHA = 0.55;
/** Fresnel exponent: higher = thinner rim. */
export const HALO_RIM_POWER = 2.2;

export const HALO_VERTEX_SHADER = /* glsl */ `
  attribute float aRevealAt;

  uniform float uRevealElapsed;
  uniform float uEventsStart;
  uniform float uEventsDuration;
  uniform float uEventsExponent;
  uniform float uOpacity;

  varying float vRim;
  varying float vAlpha;

  void main() {
    // Same appearance time as the event glyph (scene/reveal/timeline.ts → appearTimeOf).
    float appearT = uEventsStart + uEventsDuration * pow(clamp(aRevealAt, 0.0, 1.0), 1.0 / uEventsExponent);
    float shown = step(appearT, uRevealElapsed);
    vAlpha = uOpacity * shown;

    // Ellipsoid normal: the unit sphere's normal divided by the per-axis scale.
    vec3 scale = vec3(length(instanceMatrix[0].xyz), length(instanceMatrix[1].xyz), length(instanceMatrix[2].xyz));
    vec3 n = normalize(normal / max(scale, vec3(1e-6)));
    vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
    vec3 viewNormal = normalize(mat3(viewMatrix) * mat3(modelMatrix) * n);
    vec3 viewDir = normalize(-mvPosition.xyz);
    vRim = pow(1.0 - abs(dot(viewNormal, viewDir)), ${HALO_RIM_POWER.toFixed(3)});

    // Collapse when invisible: no fragments.
    gl_Position = projectionMatrix * mvPosition * step(0.001, vAlpha);
  }
`;

export const HALO_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;
  varying float vRim;
  varying float vAlpha;

  void main() {
    gl_FragColor = vec4(uColor, vAlpha * vRim * ${HALO_RIM_ALPHA.toFixed(3)});
    #include <colorspace_fragment>
  }
`;

export function createHaloUniforms(color: string): HaloUniforms {
  return {
    uColor: { value: new Color(color) },
    uOpacity: { value: 0 },
    uRevealElapsed: { value: -1 },
    uEventsStart: { value: TIMELINE.events.startS },
    uEventsDuration: { value: TIMELINE.events.endS - TIMELINE.events.startS },
    uEventsExponent: { value: TIMELINE.events.exponent },
  };
}

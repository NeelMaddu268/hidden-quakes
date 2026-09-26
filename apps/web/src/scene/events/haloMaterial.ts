// Halo shader: a unit sphere scaled per instance to the 68% error ellipsoid, drawn as a soft fresnel
// rim so the outline reads as "the event is somewhere in here" without hiding the glyph. Overlapping
// halos combine with MAX blending (see HalosLayer), so no pile-up can outshine an event or bloom.

import { Color, type IUniform } from "three";
import { LOOK } from "../look";
import { TIMELINE } from "../reveal/timeline";

export interface HaloUniforms {
  [name: string]: IUniform;
  uColor: IUniform<Color>;
  /** Filter-driven halo opacity (0 outside STRICT). */
  uOpacity: IUniform<number>;
  /** Reveal clock (s); a halo fades in with its event's pop once the clock passes its appearance time. */
  uRevealElapsed: IUniform<number>;
  uPopS: IUniform<number>;
  /** Scene y of the site surface and fog density per scene unit, matching the event glyphs. */
  uSurfaceY: IUniform<number>;
  uDepthFog: IUniform<number>;
}

export const HALO_VERTEX_SHADER = /* glsl */ `
  attribute float aAppearAt;

  uniform float uRevealElapsed;
  uniform float uPopS;
  uniform float uOpacity;
  uniform float uSurfaceY;
  uniform float uDepthFog;

  varying vec3 vViewNormal;
  varying vec3 vViewPosition;
  varying float vAlpha;

  void main() {
    // Same appearance time as the glyph (computed on the CPU); fades in over the glyph's pop.
    float age = uRevealElapsed - aAppearAt;
    float pop = clamp(age / max(uPopS, 1e-6), 0.0, 1.0) * step(0.0, age);
    vec4 center = modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
    float fog = exp(-uDepthFog * max(0.0, uSurfaceY - center.y));
    vAlpha = uOpacity * pop * fog;

    // Ellipsoid normal: the unit sphere's normal divided by the per-axis instance scale.
    vec3 scale = vec3(length(instanceMatrix[0].xyz), length(instanceMatrix[1].xyz), length(instanceMatrix[2].xyz));
    vViewNormal = normalize(normalMatrix * normalize(normal / max(scale, vec3(1e-6))));

    // Hidden halos collapse onto their center: degenerate, but inside the clip volume.
    vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position * step(0.001, vAlpha), 1.0);
    vViewPosition = mvPosition.xyz;
    gl_Position = projectionMatrix * mvPosition;
  }
`;

export const HALO_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;

  varying vec3 vViewNormal;
  varying vec3 vViewPosition;
  varying float vAlpha;

  void main() {
    // Per-fragment fresnel (smooth at any orientation); the base is clamped so float error never
    // takes pow() out of its domain.
    float facing = clamp(abs(dot(normalize(vViewNormal), normalize(-vViewPosition))), 0.0, 1.0);
    float rim = pow(1.0 - facing, ${LOOK.halos.rimPower.toFixed(3)});
    float a = vAlpha * rim * ${LOOK.halos.rimAlpha.toFixed(3)};
    // MAX blending ignores blend factors, so premultiply here.
    gl_FragColor = vec4(uColor * a, a);
    #include <colorspace_fragment>
  }
`;

export function createHaloUniforms(color: string): HaloUniforms {
  return {
    uColor: { value: new Color(color) },
    uOpacity: { value: 0 },
    uRevealElapsed: { value: -1 },
    uPopS: { value: TIMELINE.popS },
    uSurfaceY: { value: 0 },
    uDepthFog: { value: 0 },
  };
}

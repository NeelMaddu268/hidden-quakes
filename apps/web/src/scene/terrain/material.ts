// Terrain surface material: the dark base colour modulated by the baked hillshade, thin antialiased
// contour lines every 100 m of elevation, and one opacity uniform the reveal fades (sceneFx). Contours
// are composited over the fill, so they stay visible when the fill fades ("contours stay").

import { colors } from "@hq/visualization";
import { Color, type IUniform } from "three";

/** Contour interval (m of elevation) and how often an index contour is drawn a little heavier. */
export const CONTOUR_INTERVAL_M = 100;
export const CONTOUR_INDEX_EVERY = 5;

/** Contour half-width in CSS pixels; the frame loop scales it by the device pixel ratio. */
export const CONTOUR_HALF_WIDTH_PX = 0.6;

export interface TerrainUniforms {
  [name: string]: IUniform;
  uBase: IUniform<Color>;
  uContour: IUniform<Color>;
  /** Fill opacity (sceneFx.terrainOpacity, read every frame). */
  uOpacity: IUniform<number>;
  /** Contour line opacity, independent of the fill. 0 turns contours off (abstract slab). */
  uContourOpacity: IUniform<number>;
  uContourIntervalM: IUniform<number>;
  uContourIndexEvery: IUniform<number>;
  /** Line half-width in device pixels (CONTOUR_HALF_WIDTH_PX × DPR, written every frame). */
  uContourWidthPx: IUniform<number>;
  /** Hillshade byte of flat ground / 255, so flat ground renders exactly `colors.terrain`. */
  uFlatShade: IUniform<number>;
  /** 0 = ignore the hillshade, 1 = full modulation. */
  uShadeStrength: IUniform<number>;
  /** Inverse of the ENU → scene mapping for y, to recover elevation per fragment. */
  uOriginElevM: IUniform<number>;
  uVerticalExaggeration: IUniform<number>;
}

export const TERRAIN_VERTEX_SHADER = /* glsl */ `
  attribute float aShade;

  uniform float uOriginElevM;
  uniform float uVerticalExaggeration;

  varying float vShade;
  varying float vElevM;

  void main() {
    vShade = aShade;
    // Scene y = (elevM − originElevM) / 1000 × verticalExaggeration (coords.ts), inverted.
    vElevM = position.y / uVerticalExaggeration * 1000.0 + uOriginElevM;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const TERRAIN_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uBase;
  uniform vec3 uContour;
  uniform float uOpacity;
  uniform float uContourOpacity;
  uniform float uContourIntervalM;
  uniform float uContourIndexEvery;
  uniform float uContourWidthPx;
  uniform float uFlatShade;
  uniform float uShadeStrength;

  varying float vShade;
  varying float vElevM;

  void main() {
    vec3 fill = uBase * mix(1.0, vShade / uFlatShade, uShadeStrength);

    // Distance to the nearest contour in pixels, from the screen-space rate of change of elevation.
    float h = vElevM / uContourIntervalM;
    float w = max(fwidth(h), 1e-6);
    float distPx = abs(fract(h + 0.5) - 0.5) / w;
    float isIndex = step(abs(mod(floor(h + 0.5), uContourIndexEvery)), 0.5);
    float halfWidth = uContourWidthPx * (1.0 + 0.6 * isIndex);
    float line = 1.0 - smoothstep(halfWidth - 0.5, halfWidth + 0.5, distPx);
    // Where contours crowd closer than a few pixels (grazing views, cliffs), fade them out.
    line *= 1.0 - smoothstep(0.2, 0.45, w);
    float lineA = line * uContourOpacity * mix(0.75, 1.0, isIndex);

    // Contour composited over the fill ("over"), so contours survive a faded fill.
    float outA = lineA + uOpacity * (1.0 - lineA);
    vec3 outC = (uContour * lineA + fill * uOpacity * (1.0 - lineA)) / max(outA, 1e-5);
    gl_FragColor = vec4(outC, outA);
    #include <colorspace_fragment>
  }
`;

/** Depth pre-pass: colour writes are off, so the fragment only has to exist. */
export const TERRAIN_DEPTH_FRAGMENT_SHADER = /* glsl */ `
  void main() {
    gl_FragColor = vec4(0.0);
  }
`;

export interface TerrainMaterialOptions {
  originElevM: number;
  verticalExaggeration: number;
  /** Hillshade byte of flat ground (meta.hillshade.flatValue). */
  flatShadeValue: number;
  /** false for the abstract slab. */
  contours: boolean;
}

/** Contour opacity when contours are on; the contour colour is already subtle. */
export const CONTOUR_OPACITY = 0.9;
export const SHADE_STRENGTH = 0.8;

export function createTerrainUniforms(opts: TerrainMaterialOptions): TerrainUniforms {
  return {
    uBase: { value: new Color(colors.terrain) }, // sRGB hex → linear working space
    uContour: { value: new Color(colors.contour) },
    uOpacity: { value: 1 },
    uContourOpacity: { value: opts.contours ? CONTOUR_OPACITY : 0 },
    uContourIntervalM: { value: CONTOUR_INTERVAL_M },
    uContourIndexEvery: { value: CONTOUR_INDEX_EVERY },
    uContourWidthPx: { value: CONTOUR_HALF_WIDTH_PX },
    uFlatShade: { value: opts.flatShadeValue / 255 },
    uShadeStrength: { value: SHADE_STRENGTH },
    uOriginElevM: { value: opts.originElevM },
    uVerticalExaggeration: { value: opts.verticalExaggeration },
  };
}

/** Clamps sceneFx.terrainOpacity into [0, 1] (a NaN from upstream renders as solid, never invisible). */
export function terrainOpacity(raw: number): number {
  if (Number.isNaN(raw)) return 1;
  return raw < 0 ? 0 : raw > 1 ? 1 : raw;
}

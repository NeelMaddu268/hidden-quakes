// The scene's visual tuning in one place (glyph sizes, glow, depth fog, bloom, vignette), so every
// layer (events, halos, terrain, references) reads the same values and nothing is tuned twice.
//
// Bloom is designed around the token colors in linear light at glow 1, with the threshold set so:
//   recovered amber (#FFB547, luminance ~0.55) glows at every depth: "the only accent";
//   public white (#DCE6F2, ~0.78) is drawn at glow 0.6 (~0.47), so it reads cool and quieter than amber;
//   Tier B (alpha 0.6) glows faintly, Tier C (0.3) doesn't; terrain (~0.01), contours (~0.03) and
//   station glyphs (~0.22) never do. The geothermal reference (#7FE0CF, ~0.62) glows.
// Settled glyph centers never exceed 1.0, so nothing clips away from its token hue; only the brief
// pop flash goes white-hot on purpose.

export const LOOK = Object.freeze({
  candidates: Object.freeze({ sizeKm: 0.06, minPx: 1.8, glow: 1.0 }),
  publicCatalog: Object.freeze({ sizeKm: 0.08, minPx: 2.6, glow: 0.6 }),
  /** No glyph grows past this on-screen radius (CSS px), however close the camera gets. */
  maxGlyphPx: 18,
  /** Exponential fog per km of depth below the site surface: deeper events read slightly dimmer. */
  depthFogPerKm: 0.05,
  bloom: Object.freeze({ intensity: 1.35, luminanceThreshold: 0.3, luminanceSmoothing: 0.12, radius: 0.72 }),
  vignette: Object.freeze({ offset: 0.3, darkness: 0.45 }),
  /** MSAA samples: 4 at DPR ≤ 1.5; 2 above that, where the half-float buffer gets large. */
  msaa: Object.freeze({ lowDpr: 4, highDpr: 2, highDprFrom: 1.5 }),
});

/**
 * Fog density per scene unit of height. Scene y is exaggerated by VE, so density per km of real depth
 * is divided by VE to keep the fog's meaning (per km) independent of the exaggeration.
 */
export function depthFogPerSceneUnit(verticalExaggeration: number): number {
  return LOOK.depthFogPerKm / verticalExaggeration;
}

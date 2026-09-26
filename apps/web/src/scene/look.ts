// The scene's visual tuning in one place (glyph sizes, glow, depth fog, bloom, vignette), so every
// layer (events, halos, terrain, references) reads the same values and nothing is tuned twice.
//
// Bloom is designed around the token colors in linear light at glow 1, with the threshold set so:
//   recovered amber (#FFB547, luminance ~0.55) glows at every depth: "the only accent";
//   public white (#DCE6F2, ~0.78) is drawn at glow 0.6 (~0.47), so it reads cool and quieter than amber;
//   Tier B (alpha 0.6) glows faintly, Tier C (0.3) doesn't; terrain (~0.01), contours (~0.03) and
//   station glyphs (~0.22) never do.
//   The geothermal reference (#7FE0CF, ~0.62) glows ON PURPOSE: docs/00's 5-second test is "dark terrain,
//   one glowing geothermal reference, PUBLIC {N}". It is a location reference in its own color, never
//   drawn in the event language (amber), and nothing on screen attributes events to it (CLAUDE.md 6).
//   Tier A halos (#FFD08A at rim alpha ≤ 0.55) bloom only faintly. Labels are DOM overlays and never bloom.
// A single settled glyph center never exceeds 1.0, so it keeps its token hue; dense clusters sum
// additively past 1 and warm toward yellow-white, and the brief pop flash goes white-hot on purpose.
// Public glyphs at glow 0.6 display darker than the #DCE6F2 token; a legend swatch should use
// `publicSwatch()` below, not the raw token.

export const LOOK = Object.freeze({
  candidates: Object.freeze({ sizeKm: 0.06, minPx: 1.8, glow: 1.0 }),
  publicCatalog: Object.freeze({ sizeKm: 0.08, minPx: 2.6, glow: 0.6 }),
  /** No glyph grows past this on-screen radius (CSS px), however close the camera gets. */
  maxGlyphPx: 18,
  /** Exponential fog per km of depth below the site surface: deeper events read slightly dimmer. */
  depthFogPerKm: 0.05,
  bloom: Object.freeze({ intensity: 1.35, luminanceThreshold: 0.3, luminanceSmoothing: 0.12, radius: 0.72 }),
  vignette: Object.freeze({ offset: 0.3, darkness: 0.45 }),
  /**
   * Tier A uncertainty halos (WEB-04). Overlapping halos combine with MAX blending, so a dense cluster
   * is never brighter than one rim; the rim peak (rimAlpha × strictHalo luminance ~0.68 ≈ 0.27) stays
   * under the bloom threshold, so halos never glow and never outshine their events.
   */
  halos: Object.freeze({ rimAlpha: 0.4, rimPower: 2.2, sphereDetail: 2 }),
  /** Public-catalog weight under STRICT (after the reveal): still there for reference, Tier A leads. */
  strictPublicWeight: 0.4,
  /**
   * Time mode (WEB-06). Playback runs at `playbackRate` data seconds per real second (1 h/s replays a
   * day in 24 s). Events from the last `glowWindowS` of data time before tNow glow brighter and a little
   * larger, decaying linearly to their settled look; the glow is warm (no whitening), so a recent amber
   * event never reads as a white public one. The histogram bins are `binS` wide.
   */
  time: Object.freeze({ playbackRate: 3600, glowWindowS: 1800, glowBoost: 1.1, glowGrow: 0.6, binS: 600 }),
  /** MSAA samples: 4, dropping to 2 once the drawing buffer exceeds ~2560×1440 (half-float memory). */
  msaa: Object.freeze({ samples: 4, largeBufferSamples: 2, largeBufferPixels: 2560 * 1440 }),
});

/**
 * Fog density per scene unit of height. Scene y is exaggerated by VE, so density per km of real depth
 * is divided by VE to keep the fog's meaning (per km) independent of the exaggeration.
 */
export function depthFogPerSceneUnit(verticalExaggeration: number): number {
  return LOOK.depthFogPerKm / verticalExaggeration;
}

/** MSAA sample count for a drawing buffer of `pixels` device pixels. */
export function msaaSamplesFor(pixels: number): number {
  return pixels > LOOK.msaa.largeBufferPixels ? LOOK.msaa.largeBufferSamples : LOOK.msaa.samples;
}

/** The on-screen color of a settled public glyph center (token × glow, in sRGB), for legend swatches. */
export function publicSwatch(tokenHex: string): string {
  const toLin = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  const toSrgb = (c: number) => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
  const n = parseInt(tokenHex.slice(1), 16);
  const ch = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const out = toSrgb(Math.min(1, toLin(v / 255) * LOOK.publicCatalog.glow));
    return Math.round(out * 255).toString(16).padStart(2, "0");
  });
  return `#${ch.join("").toUpperCase()}`;
}

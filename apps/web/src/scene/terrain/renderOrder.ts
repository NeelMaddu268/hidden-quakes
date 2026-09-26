// Draw order of the scene's layers. Everything here is a transparent object, and three.js draws
// transparent objects sorted by renderOrder first (then by distance), so these numbers are the
// layering contract between the terrain and every other layer.
//
// Why the terrain sits in the middle:
// 1. Everything underground draws first: halos (WEB-04, 0), candidate events (1), public events (2),
//    depth slices and borehole sensors (UNDERGROUND). Nothing depth-tests against the terrain yet, so
//    they are all in the frame buffer before the terrain covers them.
// 2. The terrain then draws over them with normal alpha blending and no depth writes in its colour
//    pass. At opacity 1 it hides everything beneath (the 5-second test: terrain, the geothermal
//    reference and the shell's PUBLIC count only). As the reveal fades it, what's underneath fades in
//    continuously, with no pop, because nothing underground was ever depth-rejected by the terrain.
//    A depth-only pre-pass (TERRAIN_DEPTH) runs just before the colour pass so the terrain's own folds
//    occlude each other correctly from any orbit angle (otherwise its triangles would draw in index
//    order and far ridges could paint over near ones). It comes after the underground layers, so it
//    never hides them.
// 3. Surface glyphs (station triangles, wellhead rings) draw over the terrain, depth-tested against
//    that pre-pass, so a hill can hide a station behind it.
// 4. The geothermal reference and the depth ruler are annotation overlays (no depth test), last.

export const RENDER_ORDER = Object.freeze({
  /** The event layers' values, for reference (scene/Canvas.tsx, WEB-04 halos use 0). */
  eventsMax: 2,
  /** Depth slices, borehole sensors and their lines. */
  underground: 2,
  terrainDepth: 3,
  terrain: 4,
  /** Surface stations and wellheads. */
  surface: 5,
  featureGlow: 6,
  feature: 7,
  ruler: 8,
});

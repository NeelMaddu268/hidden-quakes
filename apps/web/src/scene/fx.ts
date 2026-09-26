// Per-frame values shared between scene layers, so independent layers stay decoupled: the reveal
// driver (scene/reveal, WEB-03) writes them, other layers (terrain WEB-02, post WEB-03) read them in
// their own useFrame. Plain mutable numbers: no React state, no allocation, nothing re-renders.
// There is exactly one <Scene/> on the page, so one module-level object is enough.

export interface SceneFx {
  /** Seconds since reveal() was called; 0 before it and after reset(). */
  revealElapsedS: number;
  /** Terrain surface opacity: 1 before the reveal, fading to the revealed value (lane doc: 0.12). */
  terrainOpacity: number;
  /** Multiplier on bloom intensity (the reveal can flare it while events pop). */
  bloomBoost: number;
}

export const INITIAL_SCENE_FX: Readonly<SceneFx> = Object.freeze({
  revealElapsedS: 0,
  terrainOpacity: 1,
  bloomBoost: 1,
});

export const sceneFx: SceneFx = { ...INITIAL_SCENE_FX };

/** Back to the start-frame values (tests, and the reveal driver on reset()). */
export function resetSceneFx(): void {
  sceneFx.revealElapsedS = INITIAL_SCENE_FX.revealElapsedS;
  sceneFx.terrainOpacity = INITIAL_SCENE_FX.terrainOpacity;
  sceneFx.bloomBoost = INITIAL_SCENE_FX.bloomBoost;
}

import { describe, expect, it } from "vitest";
import { INITIAL_SCENE_FX, resetSceneFx, sceneFx } from "./fx";

describe("sceneFx", () => {
  it("starts at the start-frame values and resetSceneFx restores them", () => {
    expect({ ...sceneFx }).toEqual({ ...INITIAL_SCENE_FX });
    sceneFx.revealElapsedS = 3.2;
    sceneFx.terrainOpacity = 0.12;
    sceneFx.bloomBoost = 1.3;
    resetSceneFx();
    expect({ ...sceneFx }).toEqual({ ...INITIAL_SCENE_FX });
  });

  it("keeps the initial values frozen", () => {
    expect(Object.isFrozen(INITIAL_SCENE_FX)).toBe(true);
  });
});

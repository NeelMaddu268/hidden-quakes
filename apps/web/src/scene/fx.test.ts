import { describe, expect, it } from "vitest";
import { INITIAL_SCENE_FX, resetSceneFx, sceneFx } from "./fx";

describe("sceneFx", () => {
  it("starts at the start-frame values and resetSceneFx restores them", () => {
    expect({ ...sceneFx }).toEqual({ ...INITIAL_SCENE_FX });
    sceneFx.revealElapsedS = 3.2;
    sceneFx.terrainOpacity = 0.12;
    sceneFx.bloomBoost = 1.3;
    sceneFx.filterLook.tierB = 0.05;
    const look = sceneFx.filterLook;
    resetSceneFx();
    expect({ ...sceneFx }).toEqual({ ...INITIAL_SCENE_FX });
    expect(sceneFx.filterLook).toBe(look); // reset in place: readers may hold the object
  });

  it("keeps the initial values frozen, and the live look is a separate object", () => {
    expect(Object.isFrozen(INITIAL_SCENE_FX)).toBe(true);
    expect(sceneFx.filterLook).not.toBe(INITIAL_SCENE_FX.filterLook);
  });
});

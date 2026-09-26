import { PerspectiveCamera, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { computeBounds } from "./bounds";
import { aspectOf, boundsSignature, dropOrbitMomentum, FALLBACK_ASPECT } from "./rig";

describe("aspectOf", () => {
  it("divides width by height, and falls back before the canvas has a size", () => {
    expect(aspectOf(1280, 720)).toBeCloseTo(16 / 9, 12);
    expect(aspectOf(0, 720)).toBe(FALLBACK_ASPECT);
    expect(aspectOf(1280, 0)).toBe(FALLBACK_ASPECT);
  });
});

describe("boundsSignature", () => {
  it("is equal for identical framing built from fresh arrays", () => {
    const a = computeBounds([new Float32Array([0, -1, 0, 4, -3, -2])], 0.05, 0);
    const b = computeBounds([new Float32Array([0, -1, 0, 4, -3, -2])], 0.05, 0);
    expect(a).not.toBe(b);
    expect(boundsSignature(a)).toBe(boundsSignature(b));
  });

  it("changes when the framing changes", () => {
    const a = computeBounds([new Float32Array([0, -1, 0, 4, -3, -2])], 0.05, 0);
    const b = computeBounds([new Float32Array([0, -1, 0, 5, -3, -2])], 0.05, 0);
    expect(boundsSignature(a)).not.toBe(boundsSignature(b));
  });
});

describe("dropOrbitMomentum", () => {
  /**
   * A stand-in with three-stdlib OrbitControls' damping semantics (controls/OrbitControls.js,
   * update()): with damping on, a fraction of the pending motion is applied per update and the rest
   * kept; with damping off, all of it is applied and then cleared. The real class isn't importable
   * from apps/web under pnpm (it's drei's transitive dependency).
   */
  function rig() {
    const camera = new PerspectiveCamera(40, 16 / 9, 0.01, 1000);
    camera.position.set(0, 5, 10);
    const pending = new Vector3();
    const controls = {
      target: new Vector3(0, 0, 0),
      enableDamping: true,
      dampingFactor: 0.08,
      rotateLeft(amount: number) {
        pending.x += amount;
      },
      update() {
        const k = this.enableDamping ? this.dampingFactor : 1;
        camera.position.x += pending.x * k;
        if (this.enableDamping) pending.multiplyScalar(1 - this.dampingFactor);
        else pending.set(0, 0, 0);
        camera.lookAt(this.target);
      },
    };
    return { camera, controls };
  }

  it("leaves the pose untouched and kills leftover motion", () => {
    const { camera, controls } = rig();
    controls.rotateLeft(0.5); // a drag's leftover velocity
    const before = camera.position.clone();
    dropOrbitMomentum(controls, camera.position);
    expect(camera.position.distanceTo(before)).toBeLessThan(1e-12);
    expect(controls.target.toArray()).toEqual([0, 0, 0]);
    for (let i = 0; i < 30; i++) controls.update();
    expect(camera.position.distanceTo(before)).toBeLessThan(1e-12);
  });

  it("control: without it, the leftover motion keeps moving the camera", () => {
    const { camera, controls } = rig();
    controls.rotateLeft(0.5);
    const before = camera.position.clone();
    for (let i = 0; i < 30; i++) controls.update();
    expect(camera.position.distanceTo(before)).toBeGreaterThan(0.1);
  });

  it("restores damping afterwards", () => {
    const { camera, controls } = rig();
    dropOrbitMomentum(controls, camera.position);
    expect(controls.enableDamping).toBe(true);
  });
});

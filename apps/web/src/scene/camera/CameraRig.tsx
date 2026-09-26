"use client";

import { OrbitControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { motion } from "@hq/visualization";
import { useEffect, useLayoutEffect, useRef, type ComponentRef } from "react";
import type { Camera } from "three";
import { onDemoReset, useDemo, type CameraView } from "../../state/demo";
import { sceneFx } from "../fx";
import type { SceneBounds } from "./bounds";
import { createCameraDirector, type CameraDirector } from "./director";
import { presetPose, type CameraPose } from "./presets";
import { aspectOf, boundsSignature, dropOrbitMomentum, ORBIT } from "./rig";

type Controls = ComponentRef<typeof OrbitControls>;

/** Puts camera and orbit target exactly on `pose` and releases the camera. */
function snapTo(camera: Camera, c: Controls, pose: CameraPose, director: CameraDirector): void {
  director.cancel();
  dropOrbitMomentum(c, camera.position);
  camera.position.set(pose.position[0], pose.position[1], pose.position[2]);
  c.target.set(pose.target[0], pose.target[1], pose.target[2]);
  c.update();
  c.enabled = true;
}

/**
 * Orbit camera with the docs/02 presets. The decisions (which move runs when, the reveal dolly locked
 * to the reveal clock, reveal from plan view, presenter sequences like R then Space) live in the tested
 * camera director (./director.ts); this component wires it to the store, the frame loop and the orbit
 * controls, which stay disabled while the director owns the camera.
 */
export function CameraRig({ bounds }: { bounds: SceneBounds }) {
  const camera = useThree((s) => s.camera);
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  const controls = useRef<Controls>(null);
  const director = useRef<CameraDirector>(createCameraDirector());
  const aspect = useRef(aspectOf(width, height));
  const framed = useRef<string | null>(null);
  const bounds$ = useRef(bounds);

  useLayoutEffect(() => {
    aspect.current = aspectOf(width, height);
  }, [width, height]);

  useLayoutEffect(() => {
    bounds$.current = bounds;
  }, [bounds]);

  // A different default camera (e.g. an orthographic plan camera) must be framed afresh.
  useLayoutEffect(() => {
    framed.current = null;
  }, [camera]);

  // New framing: snap to the current view's preset, but only when the framing really changed and the
  // demo is at its start (a refetch returning equal data never yanks the camera mid-orbit or mid-reveal).
  useLayoutEffect(() => {
    const c = controls.current;
    if (!c) return;
    const signature = boundsSignature(bounds);
    const first = framed.current === null;
    if (signature === framed.current) return;
    framed.current = signature;
    const { view, phase } = useDemo.getState();
    if (first || phase === "public") snapTo(camera, c, presetPose(view, bounds, aspect.current), director.current);
  }, [bounds, camera]);

  useEffect(() => {
    const d = director.current;
    const moveTo = (view: CameraView) => {
      const c = controls.current;
      if (!c) return;
      dropOrbitMomentum(c, camera.position);
      d.moveTo(camera.position, c.target, presetPose(view, bounds$.current, aspect.current), motion.scene / 1000);
      c.enabled = false;
    };
    const offStore = useDemo.subscribe((s, prev) => {
      const c = controls.current;
      if (!c) return;
      if (prev.phase === "public" && s.phase === "revealing") {
        dropOrbitMomentum(c, camera.position);
        d.onReveal(s.view === "plan");
        c.enabled = !d.busy;
        return;
      }
      // reset() is handled by onDemoReset (including its view change back to the start view).
      if (s.phase === "public" && prev.phase !== "public") return;
      if (s.view !== prev.view) moveTo(s.view);
    });
    const offReset = onDemoReset(() => moveTo(useDemo.getState().view));
    return () => {
      offStore();
      offReset();
    };
  }, [camera]);

  useFrame((_, delta) => {
    const c = controls.current;
    if (!c) return;
    const d = director.current;
    if (!d.busy) return;
    d.step(delta, sceneFx.revealElapsedS, camera.position, c.target, () =>
      presetPose(useDemo.getState().view, bounds$.current, aspect.current),
    );
    c.update();
    if (!d.busy) c.enabled = true;
  });

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={ORBIT.dampingFactor}
      minDistance={ORBIT.minDistanceKm}
      maxDistance={bounds.radius * ORBIT.maxDistanceRadii}
    />
  );
}

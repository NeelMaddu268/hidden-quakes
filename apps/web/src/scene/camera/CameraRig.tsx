"use client";

import { OrbitControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { motion } from "@hq/visualization";
import { useEffect, useLayoutEffect, useRef, type ComponentRef } from "react";
import type { Camera } from "three";
import { onDemoReset, useDemo } from "../../state/demo";
import type { SceneBounds } from "./bounds";
import { presetPose, type CameraPose } from "./presets";
import { aspectOf, boundsSignature, dropOrbitMomentum, ORBIT } from "./rig";
import { createTween, startTween, stepTween, type PoseTween } from "./tween";

type Controls = ComponentRef<typeof OrbitControls>;

/** Puts camera and orbit target exactly on `pose` and cancels any running move. */
function snapTo(camera: Camera, c: Controls, pose: CameraPose, tween: PoseTween): void {
  tween.active = false;
  dropOrbitMomentum(c, camera.position);
  camera.position.set(pose.position[0], pose.position[1], pose.position[2]);
  c.target.set(pose.target[0], pose.target[1], pose.target[2]);
  c.update();
  c.enabled = true;
}

/**
 * Orbit camera with the docs/02 presets. `setView` tweens to a preset over `motion.scene`; `reset()`
 * tweens back to the view's preset (even when no store field changed, e.g. R after orbiting). User
 * orbiting works between moves and is disabled while one runs.
 */
export function CameraRig({ bounds }: { bounds: SceneBounds }) {
  const camera = useThree((s) => s.camera);
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  const controls = useRef<Controls>(null);
  const tween = useRef<PoseTween>(createTween());
  const aspect = useRef(aspectOf(width, height));
  const framed = useRef<string | null>(null);

  useLayoutEffect(() => {
    aspect.current = aspectOf(width, height);
  }, [width, height]);

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
    if (first || phase === "public") snapTo(camera, c, presetPose(view, bounds, aspect.current), tween.current);
  }, [bounds, camera]);

  useEffect(() => {
    const moveTo = (pose: CameraPose) => {
      const c = controls.current;
      if (!c) return;
      dropOrbitMomentum(c, camera.position);
      startTween(tween.current, camera.position.toArray(), c.target.toArray(), pose, motion.scene / 1000);
      c.enabled = false;
    };
    const offView = useDemo.subscribe((s, prev) => {
      if (s.view !== prev.view) moveTo(presetPose(s.view, bounds, aspect.current));
    });
    const offReset = onDemoReset(() => moveTo(presetPose(useDemo.getState().view, bounds, aspect.current)));
    return () => {
      offView();
      offReset();
    };
  }, [bounds, camera]);

  useFrame((_, delta) => {
    const c = controls.current;
    if (!c || !tween.current.active) return;
    const done = stepTween(tween.current, delta, camera.position, c.target);
    c.update();
    if (done) c.enabled = true;
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

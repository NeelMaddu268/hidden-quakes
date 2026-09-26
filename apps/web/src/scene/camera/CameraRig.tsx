"use client";

import { OrbitControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { motion } from "@hq/visualization";
import { useEffect, useLayoutEffect, useRef, type ComponentRef } from "react";
import type { Camera } from "three";
import { useDemo } from "../../state/demo";
import type { SceneBounds } from "./bounds";
import { presetPose, type CameraPose } from "./presets";
import { createTween, startTween, stepTween, type PoseTween } from "./tween";

type Controls = ComponentRef<typeof OrbitControls>;

/** Drops any leftover damping velocity so a programmatic pose isn't nudged afterwards. */
function settleControls(c: Controls): void {
  const damping = c.enableDamping;
  c.enableDamping = false;
  c.update();
  c.enableDamping = damping;
}

/** Puts camera and orbit target exactly on `pose` and cancels any running move. */
function snapTo(camera: Camera, c: Controls, pose: CameraPose, tween: PoseTween): void {
  tween.active = false;
  camera.position.set(pose.position[0], pose.position[1], pose.position[2]);
  c.target.set(pose.target[0], pose.target[1], pose.target[2]);
  c.enabled = true;
  settleControls(c);
}

/**
 * Orbit camera with the docs/02 presets. `setView` tweens to a preset over `motion.scene`; `reset()`
 * (phase back to "public") tweens back to the view's preset. User orbiting works between moves and
 * is disabled while one runs.
 */
export function CameraRig({ bounds }: { bounds: SceneBounds }) {
  const camera = useThree((s) => s.camera);
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  const controls = useRef<Controls>(null);
  const tween = useRef<PoseTween>(createTween());
  const aspect = useRef(1);

  useLayoutEffect(() => {
    aspect.current = width / Math.max(height, 1);
  }, [width, height]);

  // New framing (a bundle loaded or changed): snap to the current view's preset.
  useLayoutEffect(() => {
    const c = controls.current;
    if (!c) return;
    snapTo(camera, c, presetPose(useDemo.getState().view, bounds, aspect.current), tween.current);
  }, [bounds, camera]);

  useEffect(
    () =>
      useDemo.subscribe((s, prev) => {
        const viewChanged = s.view !== prev.view;
        const wasReset = s.phase === "public" && prev.phase !== "public";
        const c = controls.current;
        if (!c || !(viewChanged || wasReset)) return;
        const pose = presetPose(s.view, bounds, aspect.current);
        startTween(tween.current, camera.position.toArray(), c.target.toArray(), pose, motion.scene / 1000);
        c.enabled = false;
      }),
    [bounds, camera],
  );

  useFrame((_, delta) => {
    const c = controls.current;
    if (!c || !tween.current.active) return;
    const done = stepTween(tween.current, delta, camera.position, c.target);
    settleControls(c);
    if (done) c.enabled = true;
  });

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      minDistance={0.3}
      maxDistance={bounds.radius * 20}
    />
  );
}

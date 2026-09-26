"use client";

import { MapControls, OrthographicCamera } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import { useLayoutEffect, useRef, type ComponentRef } from "react";
import type { SceneBounds } from "../camera/bounds";
import { planReserveLeftPx } from "./layout";
import { planView } from "./view";

type Cam = ComponentRef<typeof OrthographicCamera>;
type Controls = ComponentRef<typeof MapControls>;

/** How far the presenter can zoom out / in from the framed plan (multiples of the framed zoom). */
export const PLAN_ZOOM_RANGE = Object.freeze({ out: 0.25, in: 24 });

/**
 * The plan view's camera (WEB-07): orthographic, looking straight down, grid north up and grid east
 * right at equal scale (docs/01: scene axes are the UTM grid; nothing is rotated to true north).
 * While mounted it is the default camera and MapControls the default controls: pan and zoom, no
 * rotation. Canvas mounts it instead of the perspective CameraRig, so the rig's presets, reveal dolly
 * and orbit momentum can't touch the plan; unmounting (P again, reset) hands the perspective camera
 * back (drei restores the previous default camera and controls).
 */
export function PlanCamera({ bounds, clipBounds }: { bounds: SceneBounds; clipBounds: Pick<SceneBounds, "min" | "max"> }) {
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  const camera = useRef<Cam>(null);
  const controls = useRef<Controls>(null);
  const size = useRef({ width, height });

  useLayoutEffect(() => {
    size.current = { width, height };
  }, [width, height]);

  // Frame on entry and whenever the framed data changes. Pan/zoom are the presenter's afterwards; a
  // resize keeps them (R3F resizes the pixel frustum, the zoom keeps the map scale).
  useLayoutEffect(() => {
    const cam = camera.current;
    if (!cam) return;
    const { width: W, height: H } = size.current;
    const v = planView(bounds, clipBounds, W, H, planReserveLeftPx(W, H));
    cam.up.set(0, 0, -1);
    cam.position.set(v.x, v.y, v.z);
    cam.near = v.near;
    cam.far = v.far;
    cam.zoom = v.zoom;
    cam.lookAt(v.x, v.targetY, v.z);
    cam.updateProjectionMatrix();
    const c = controls.current;
    if (c) {
      c.target.set(v.x, v.targetY, v.z);
      c.minZoom = v.zoom * PLAN_ZOOM_RANGE.out;
      c.maxZoom = v.zoom * PLAN_ZOOM_RANGE.in;
      c.update();
    }
  }, [bounds, clipBounds]);

  return (
    <>
      <OrthographicCamera ref={camera} makeDefault />
      <MapControls
        ref={controls}
        makeDefault
        enableRotate={false}
        screenSpacePanning
        enableDamping
        dampingFactor={0.12}
        zoomToCursor
      />
    </>
  );
}

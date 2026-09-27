"use client";

import { MapControls, OrthographicCamera } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { memo, useMemo, useRef } from "react";
import { Vector3 } from "three";
import type { SceneBounds } from "../camera/bounds";
import { usePlanDock } from "./dock";
import { planReservePx } from "./layout";
import { planView } from "./view";

/** How far the presenter can zoom out / in from the framed plan (multiples of the framed zoom). */
export const PLAN_ZOOM_RANGE = Object.freeze({ out: 0.25, in: 24 });
/** Grid north (−z) is screen-up in plan view. */
const NORTH_UP: [number, number, number] = [0, 0, -1];

/** Frames after entering plan view on which the invariant below is checked (the pose settles by then). */
const CHECK_FRAMES = [3, 12];

/**
 * Fails loudly (console.error) if the live default camera in plan view is not the orthographic plan
 * camera looking straight down with grid north up: e.g. if controls bound to another camera re-aim it.
 * Runs on two frames after entry, then costs nothing.
 */
function PlanInvariant() {
  const camera = useThree((s) => s.camera);
  const frame = useRef(0);
  const dir = useRef(new Vector3());
  const up = useRef(new Vector3());
  useFrame(() => {
    const n = ++frame.current;
    if (n > CHECK_FRAMES[CHECK_FRAMES.length - 1] || !CHECK_FRAMES.includes(n)) return;
    const d = camera.getWorldDirection(dir.current);
    const u = up.current.set(0, 1, 0).applyQuaternion(camera.quaternion);
    const ok = "isOrthographicCamera" in camera && Math.abs(d.y + 1) < 1e-6 && u.z < -1 + 1e-6;
    if (!ok) {
      console.error(
        `[plan] camera is not a straight-down, grid-north-up orthographic view: dir (${d.x.toFixed(4)}, ` +
          `${d.y.toFixed(4)}, ${d.z.toFixed(4)}), screen-up (${u.x.toFixed(4)}, ${u.y.toFixed(4)}, ${u.z.toFixed(4)})`,
      );
    }
  });
  return null;
}

interface PlanCameraProps {
  bounds: SceneBounds;
  clipBounds: Pick<SceneBounds, "min" | "max">;
}

/**
 * The plan view's camera (WEB-07): orthographic, looking straight down, grid north up and grid east
 * right at equal scale (docs/01: scene axes are the UTM grid; nothing is rotated to true north).
 * While mounted it is the default camera and MapControls the default controls: pan and zoom, no
 * rotation. Canvas mounts it instead of the perspective CameraRig, so the rig's presets, reveal dolly
 * and orbit momentum can't touch the plan; unmounting (P again, reset) hands the perspective camera
 * and orbit controls back (drei restores the previous defaults).
 *
 * The pose is declarative (props, not a one-shot effect): drei builds the controls from whichever
 * camera is default at render time, so the controls bound to this orthographic camera only exist
 * after it becomes default; props reach that instance too. R3F diffs array props by value, so the
 * presenter's pan/zoom survive re-renders; the view is re-framed only when the framed data or the
 * viewport size changes.
 */
export const PlanCamera = memo(function PlanCamera({ bounds, clipBounds }: PlanCameraProps) {
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  // Framed clear of the depth section wherever it docks (left or right; never re-framed for the drawer).
  const dock = usePlanDock((s) => s.dock);
  const v = useMemo(() => {
    const reserve = planReservePx(dock, width);
    return planView(bounds, clipBounds, width, height, reserve.left, reserve.right);
  }, [bounds, clipBounds, width, height, dock]);
  const position = useMemo((): [number, number, number] => [v.x, v.y, v.z], [v]);
  const target = useMemo((): [number, number, number] => [v.x, v.targetY, v.z], [v]);

  return (
    <>
      <OrthographicCamera makeDefault position={position} up={NORTH_UP} zoom={v.zoom} near={v.near} far={v.far} />
      <MapControls
        makeDefault
        target={target}
        minZoom={v.zoom * PLAN_ZOOM_RANGE.out}
        maxZoom={v.zoom * PLAN_ZOOM_RANGE.in}
        enableRotate={false}
        screenSpacePanning
        enableDamping
        dampingFactor={0.12}
        zoomToCursor
      />
      <PlanInvariant />
    </>
  );
});

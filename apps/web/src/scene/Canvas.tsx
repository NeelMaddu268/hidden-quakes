"use client";

import { colors } from "@hq/visualization";
import { Canvas } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import { useDemo } from "../state/demo";
import { computeBounds } from "./camera/bounds";
import { CameraRig } from "./camera/CameraRig";
import { CAMERA_FOV_DEG } from "./camera/presets";
import { depthKmToSceneY, verticalExaggerationOf } from "./coords";
import { useBundle } from "./data";
import { candidateLayerOpacity, candidateRevealUniform, FILTER_TIER_OPACITY } from "./events/driver";
import { EventsLayer } from "./events/EventsLayer";
import {
  buildCandidateInstances,
  buildPublicInstances,
  framingPositions,
  revealOrderIssues,
  TIER_INDEX,
} from "./events/instances";
import type { EventUniforms } from "./events/material";
import { References } from "./references";
import { Terrain } from "./terrain";
import type { BundleState } from "./types";

type ReadyBundle = Extract<BundleState, { status: "ready" }>;

// Glyph sizes (scene km) and on-screen minimums (CSS px). Public points read slightly larger so the
// sparse public catalog is legible on its own before the reveal.
const CANDIDATE_SIZE_KM = 0.06;
const CANDIDATE_MIN_PX = 1.8;
const PUBLIC_SIZE_KM = 0.08;
const PUBLIC_MIN_PX = 2.6;

/** Candidate (amber) layer: follows the reveal and the filter. Reads the store without re-rendering. */
function driveCandidates(u: EventUniforms): void {
  const s = useDemo.getState();
  u.uReveal.value = candidateRevealUniform(s.phase, s.revealProgress, u.uPopWidth.value);
  const [a, b, c] = FILTER_TIER_OPACITY[s.filter];
  u.uTierOpacity.value.set(a, b, c);
  u.uLayerOpacity.value = candidateLayerOpacity(s.filter);
}

/** Public-catalog (cool white) layer: on screen from the first frame, at full weight. */
function drivePublic(u: EventUniforms): void {
  u.uReveal.value = 1;
  u.uTierOpacity.value.set(1, 1, 1);
  u.uLayerOpacity.value = 1;
}

function BundleScene({ bundle }: { bundle: ReadyBundle }) {
  const { meta, events, catalog } = bundle;
  const ve = verticalExaggerationOf(meta.scene);
  const windowStart = meta.run.windowStart;

  const candidates = useMemo(() => buildCandidateInstances(events, ve, windowStart), [events, ve, windowStart]);
  const publicEvents = useMemo(() => buildPublicInstances(catalog, ve, windowStart), [catalog, ve, windowStart]);
  // Frame the structure: Tier A and B candidates plus the public catalog. Scattered Tier C events
  // stay rendered but don't widen the shot.
  const bounds = useMemo(
    () =>
      computeBounds(
        [framingPositions(candidates, TIER_INDEX.B), publicEvents.positions],
        depthKmToSceneY(0, meta.scene),
      ),
    [candidates, publicEvents, meta.scene],
  );

  useEffect(() => {
    const issues = revealOrderIssues(events.map((e) => e.revealOrder));
    if (issues.length) console.error(`[scene] revealOrder was not assigned by the exporter: ${issues.join("; ")}`);
  }, [events]);

  return (
    <>
      <Terrain scene={meta.scene} bounds={bounds} />
      <References bundle={bundle} bounds={bounds} />
      <EventsLayer
        name="public-events"
        instances={publicEvents}
        color={colors.public}
        size={PUBLIC_SIZE_KM}
        minPx={PUBLIC_MIN_PX}
        drive={drivePublic}
        renderOrder={2}
      />
      <EventsLayer
        name="candidate-events"
        instances={candidates}
        color={colors.recovered}
        size={CANDIDATE_SIZE_KM}
        minPx={CANDIDATE_MIN_PX}
        drive={driveCandidates}
        renderOrder={1}
      />
      <CameraRig bounds={bounds} />
    </>
  );
}

function SceneContents() {
  const bundle = useBundle();
  // Loading and error states are the shell's to show; the canvas stays an empty dark stage.
  if (bundle.status !== "ready") return null;
  return <BundleScene bundle={bundle} />;
}

/** The full-bleed 3D canvas (docs/02 §6). H4's page mounts it beneath the shell. */
export function Scene() {
  return (
    <Canvas
      dpr={[1, 2]}
      flat
      gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
      camera={{ fov: CAMERA_FOV_DEG, near: 0.02, far: 1000, position: [0, 10, 20] }}
      style={{ position: "fixed", inset: 0, background: colors.bg }}
    >
      <color attach="background" args={[colors.bg]} />
      <SceneContents />
    </Canvas>
  );
}

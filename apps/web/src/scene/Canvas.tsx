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
import {
  candidateLayerOpacity,
  candidateRevealUniform,
  FILTER_TIER_OPACITY,
  REVEALED_ELAPSED_S,
} from "./events/driver";
import { EventsLayer } from "./events/EventsLayer";
import {
  buildCandidateInstances,
  buildPublicInstances,
  framingPositions,
  revealOrderIssues,
  TIER_INDEX,
} from "./events/instances";
import type { EventUniforms } from "./events/material";
import { sceneFx } from "./fx";
import { depthFogPerSceneUnit, LOOK } from "./look";
import { Post } from "./post/Post";
import { RevealDriver } from "./reveal/RevealDriver";
import { References } from "./references";
import { Terrain } from "./terrain";
import type { BundleState } from "./types";

type ReadyBundle = Extract<BundleState, { status: "ready" }>;

/** Candidate (amber) layer: follows the reveal and the filter. Reads the store without re-rendering. */
function driveCandidates(u: EventUniforms): void {
  const s = useDemo.getState();
  u.uRevealElapsed.value = candidateRevealUniform(s.phase, sceneFx.revealElapsedS);
  const tiers = FILTER_TIER_OPACITY[s.filter];
  u.uTierOpacity.value.set(tiers[0], tiers[1], tiers[2]);
  u.uLayerOpacity.value = candidateLayerOpacity(s.filter);
}

/** Public-catalog (cool white) layer: on screen from the first frame, at full weight. */
function drivePublic(u: EventUniforms): void {
  u.uRevealElapsed.value = REVEALED_ELAPSED_S;
  u.uTierOpacity.value.set(1, 1, 1);
  u.uLayerOpacity.value = 1;
}

function BundleScene({ bundle }: { bundle: ReadyBundle }) {
  const { meta, events, catalog } = bundle;
  const ve = verticalExaggerationOf(meta.scene);
  const windowStart = meta.run.windowStart;

  const candidates = useMemo(() => buildCandidateInstances(events, ve, windowStart), [events, ve, windowStart]);
  const publicEvents = useMemo(() => buildPublicInstances(catalog, ve, windowStart), [catalog, ve, windowStart]);
  // Frame the structure: Tier A and B candidates. Scattered Tier C events and the public regional
  // catalog (which spans the whole run bbox, tens of km) stay rendered but don't widen the shot. With
  // no candidates at all, the public catalog is framed instead.
  const surfaceY = depthKmToSceneY(0, meta.scene);
  const bounds = useMemo(
    () =>
      computeBounds(
        [candidates.count > 0 ? framingPositions(candidates, TIER_INDEX.B) : publicEvents.positions],
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
        size={LOOK.publicCatalog.sizeKm}
        minPx={LOOK.publicCatalog.minPx}
        maxPx={LOOK.maxGlyphPx}
        drive={drivePublic}
        glow={LOOK.publicCatalog.glow}
        surfaceY={surfaceY}
        depthFog={depthFogPerSceneUnit(ve)}
        renderOrder={2}
      />
      <EventsLayer
        name="candidate-events"
        instances={candidates}
        color={colors.recovered}
        size={LOOK.candidates.sizeKm}
        minPx={LOOK.candidates.minPx}
        maxPx={LOOK.maxGlyphPx}
        drive={driveCandidates}
        glow={LOOK.candidates.glow}
        surfaceY={surfaceY}
        depthFog={depthFogPerSceneUnit(ve)}
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
      gl={{ antialias: false, alpha: false, powerPreference: "high-performance" }}
      camera={{ fov: CAMERA_FOV_DEG, near: 0.02, far: 1000, position: [0, 10, 20] }}
      style={{ position: "fixed", inset: 0, background: colors.bg }}
    >
      <color attach="background" args={[colors.bg]} />
      <RevealDriver />
      <SceneContents />
      <Post />
    </Canvas>
  );
}

"use client";

import { colors } from "@hq/visualization";
import { Canvas } from "@react-three/fiber";
import { useCallback, useEffect, useMemo } from "react";
import { useDemo } from "../state/demo";
import { computeBounds } from "./camera/bounds";
import { CameraRig } from "./camera/CameraRig";
import { CAMERA_FOV_DEG } from "./camera/presets";
import { depthKmToSceneY, verticalExaggerationOf } from "./coords";
import { useBundle } from "./data";
import { candidateRevealUniform, REVEALED_ELAPSED_S } from "./events/driver";
import { EventsLayer } from "./events/EventsLayer";
import type { HaloUniforms } from "./events/haloMaterial";
import { buildHaloInstances } from "./events/halos";
import { HalosLayer } from "./events/HalosLayer";
import {
  buildCandidateInstances,
  buildPublicInstances,
  framingPositions,
  revealOrderIssues,
  TIER_INDEX,
} from "./events/instances";
import type { EventUniforms } from "./events/material";
import { FilterDriver } from "./filters/FilterDriver";
import { filterCountIssues } from "./filters/selectors";
import { sceneFx } from "./fx";
import { depthFogPerSceneUnit, LOOK } from "./look";

import { Picker } from "./picking/Picker";
import { DepthSection } from "./plan/DepthSection";
import { HoverTooltip } from "./picking/HoverTooltip";
import { Tour } from "./tour/Tour";
import { buildPlanHaloInstances } from "./plan/halos";
import { PlanCamera } from "./plan/PlanCamera";
import { PlanRingsLayer } from "./plan/PlanRingsLayer";
import type { RingUniforms } from "./plan/ringMaterial";
import { selectedInstanceIndex } from "./picking/selection";
import { Post } from "./post/Post";
import { RevealDriver } from "./reveal/RevealDriver";
import { References } from "./references";
import { rulerMaxDepthKm } from "./references/ruler";
import { Terrain } from "./terrain";
import { TimeDriver } from "./time/TimeDriver";
import type { BundleState } from "./types";

type ReadyBundle = Extract<BundleState, { status: "ready" }>;

/**
 * Candidate (amber) layer: follows the reveal clock, the eased filter look (scene/filters) and time
 * mode's "now" (scene/time).
 */
function driveCandidates(u: EventUniforms, indexById: ReadonlyMap<string, number>): void {
  const look = sceneFx.filterLook;
  u.uSelected.value = selectedInstanceIndex(useDemo.getState().selectedEventId, indexById);
  u.uRevealElapsed.value = candidateRevealUniform(useDemo.getState().phase, sceneFx.revealElapsedS);
  u.uTierOpacity.value.set(look.tierA, look.tierB, look.tierC);
  u.uLayerOpacity.value = look.candidates;
  u.uTimeNow.value = sceneFx.timeNowRel;
}

/** Public-catalog (cool white) layer: on screen from the first frame; steps back under STRICT. */
function drivePublic(u: EventUniforms): void {
  u.uRevealElapsed.value = REVEALED_ELAPSED_S;
  u.uTierOpacity.value.set(1, 1, 1);
  u.uLayerOpacity.value = sceneFx.filterLook.publicLayer;
  u.uTimeNow.value = sceneFx.timeNowRel;
}

/** Tier A ellipsoid halos: appear with their events, visible only under STRICT, and only in 3D views. */
function driveHalos(u: HaloUniforms): void {
  const s = useDemo.getState();
  u.uRevealElapsed.value = candidateRevealUniform(s.phase, sceneFx.revealElapsedS);
  u.uOpacity.value = s.view === "plan" ? 0 : sceneFx.filterLook.halos;
  u.uTimeNow.value = sceneFx.timeNowRel;
}

/** Plan-view horizontal uncertainty rings: same reveal, STRICT and time rules as the 3D halos. */
function driveRings(u: RingUniforms): void {
  u.uRevealElapsed.value = candidateRevealUniform(useDemo.getState().phase, sceneFx.revealElapsedS);
  u.uOpacity.value = sceneFx.filterLook.halos;
  u.uTimeNow.value = sceneFx.timeNowRel;
}

function BundleScene({ bundle }: { bundle: ReadyBundle }) {
  const { meta, events, catalog } = bundle;
  const ve = verticalExaggerationOf(meta.scene);
  const windowStart = meta.run.windowStart;

  const candidates = useMemo(() => buildCandidateInstances(events, ve, windowStart), [events, ve, windowStart]);
  const publicEvents = useMemo(() => buildPublicInstances(catalog, ve, windowStart), [catalog, ve, windowStart]);
  const drive = useCallback((u: EventUniforms) => driveCandidates(u, candidates.indexById), [candidates]);
  const surfaceY = depthKmToSceneY(0, meta.scene);
  // Frame Tier A and B candidates. Scattered Tier C events and the public regional catalog (which spans
  // the whole run bbox, tens of km) stay rendered but don't widen the shot. With no candidates at all,
  // the public catalog is framed instead.
  const bounds = useMemo(
    () =>
      computeBounds(
        [candidates.count > 0 ? framingPositions(candidates, TIER_INDEX.B) : publicEvents.positions],
        depthKmToSceneY(0, meta.scene),
      ),
    [candidates, publicEvents, meta.scene],
  );

  // The plan camera frames the same structure, but clips against every event so pan/zoom never loses
  // deep or distant ones.
  const clipBounds = useMemo(
    () => computeBounds([candidates.positions, publicEvents.positions], depthKmToSceneY(0, meta.scene), 0),
    [candidates, publicEvents, meta.scene],
  );
  const view = useDemo((s) => s.view);
  // The depth ruler and slices reach the deepest displayed event (candidates and public catalog).
  const rulerDepthKm = useMemo(
    () => rulerMaxDepthKm([...events.map((e) => e.elevM), ...catalog.map((c) => c.elevM)], meta.scene),
    [events, catalog, meta.scene],
  );

  const halos = useMemo(() => buildHaloInstances(events, candidates, ve), [events, candidates, ve]);
  const planRings = useMemo(() => buildPlanHaloInstances(events, candidates), [events, candidates]);

  useEffect(() => {
    const issues = revealOrderIssues(events.map((e) => e.revealOrder));
    if (issues.length) console.error(`[scene] revealOrder was not assigned by the exporter: ${issues.join("; ")}`);
  }, [events]);

  useEffect(() => {
    console.info(
      `[scene] ${events.length} candidate events, ${halos.count} Tier A halos` +
        (halos.tierAWithoutHalo ? `, ${halos.tierAWithoutHalo} Tier A without a 68% error (no halo)` : ""),
    );
  }, [events.length, halos]);

  useEffect(() => {
    const issues = filterCountIssues(events, meta.summary);
    if (issues.length) console.error(`[scene] filter counts disagree with the summary: ${issues.join("; ")}`);
  }, [events, meta.summary]);

  return (
    <>
      <TimeDriver windowStart={windowStart} windowEnd={meta.run.windowEnd} />
      <Terrain scene={meta.scene} bounds={bounds} />
      <References bundle={bundle} bounds={bounds} rulerDepthKm={rulerDepthKm} planView={view === "plan"} />
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
        drive={drive}
        glow={LOOK.candidates.glow}

        surfaceY={surfaceY}
        depthFog={depthFogPerSceneUnit(ve)}
        renderOrder={1}
      />
      <HalosLayer halos={halos} surfaceY={surfaceY} depthFog={depthFogPerSceneUnit(ve)} drive={driveHalos} />
      {view === "plan" ? (
        <>
          <PlanCamera bounds={bounds} clipBounds={clipBounds} />
          <PlanRingsLayer rings={planRings} drive={driveRings} />
        </>
      ) : (
        <CameraRig bounds={bounds} />
      )}
      <Picker candidates={candidates} publicEvents={publicEvents} catalog={catalog} sizeKm={{ candidate: LOOK.candidates.sizeKm, public: LOOK.publicCatalog.sizeKm }} />
    </>
  );
}

function SceneContents() {
  const bundle = useBundle();
  // Loading and error states are the shell's to show; the canvas stays an empty dark stage.
  if (bundle.status !== "ready") return null;
  return <BundleScene bundle={bundle} />;
}

/**
 * The full-bleed 3D canvas (docs/02 §6), plus the plan view's docked depth section (a DOM panel, shown
 * only in plan view), the hover tooltip, and the guided tour's captions (WEB-09, shown only while it
 * plays). H4's page mounts it beneath the shell.
 */
export function Scene() {
  return (
    <>
    <Canvas
      dpr={[1, 2]}
      flat
      gl={{ antialias: false, alpha: false, powerPreference: "high-performance" }}
      camera={{ fov: CAMERA_FOV_DEG, near: 0.02, far: 1000, position: [0, 10, 20] }}
      style={{ position: "fixed", inset: 0, background: colors.bg }}
    >
      <color attach="background" args={[colors.bg]} />
      <RevealDriver />
      <FilterDriver />
      <SceneContents />
      <Post />
    </Canvas>
    <DepthSection />
    <HoverTooltip />
    <Tour />
    </>
  );
}

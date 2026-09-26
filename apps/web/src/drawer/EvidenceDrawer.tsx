"use client";

import { motion } from "@hq/visualization";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useBundle, useEvidence } from "../scene/data";
import type { SeismicEvent, Station } from "../scene/types";
import { useDemo } from "../state/demo";
import { Figures } from "./Figures";
import { evidenceStations, pickingStations } from "./geometry";
import { Header } from "./Header";
import { formatOpenLatency } from "./latency";
import { RecordSection } from "./RecordSection";
import { DRAWER_CSS } from "./styles";

/**
 * The evidence drawer (docs/02 §6; WEB-05). Opens when `selectedEventId` is set (a click in the scene,
 * or E → the hero event), slides in over `motion.state`, and closes on `select(null)` (Esc in the
 * shell, or the close button). While closed it is off-screen, hidden, inert and ignores the pointer, so
 * it never blocks the canvas; its contents unmount once the slide-out finishes.
 *
 * Data comes only from the scene's data hooks (`scene/data.ts`): the event record and stations from
 * the bundle, the traces from the event's evidence file. The header renders from the bundle at once;
 * the traces follow when the evidence is ready (immediately when it was preloaded).
 */
export function EvidenceDrawer() {
  const selectedId = useDemo((s) => s.selectedEventId);
  // The last opened event stays on screen while the drawer slides out.
  const [lastId, setLastId] = useState<string | null>(() => useDemo.getState().selectedEventId);
  // performance.now() of the select() that opened the drawer, for the open-latency log.
  const openedAt = useRef<number | null>(null);

  useEffect(
    () =>
      useDemo.subscribe((s, prev) => {
        if (s.selectedEventId === prev.selectedEventId || s.selectedEventId === null) return;
        openedAt.current = performance.now();
        setLastId(s.selectedEventId);
      }),
    [],
  );

  const open = selectedId !== null;
  const shownId = selectedId ?? lastId;

  useEffect(() => {
    if (open || lastId === null) return;
    const t = window.setTimeout(() => setLastId(null), motion.state);
    return () => window.clearTimeout(t);
  }, [open, lastId]);

  const bundle = useBundle();
  const evidence = useEvidence(shownId);
  const ready = bundle.status === "ready" ? bundle : null;

  const eventById = useMemo(() => (ready ? new Map(ready.events.map((e) => [e.id, e] as const)) : null), [ready]);
  const stationById = useMemo(() => (ready ? new Map(ready.stations.map((s) => [s.id, s] as const)) : null), [ready]);
  const event: SeismicEvent | null = (shownId && eventById?.get(shownId)) || null;

  // Station geometry for the figures: the evidence traces' stations; without an evidence file (the
  // exporter caps evidence at maxEvents) the stations that picked the event, so location and errors show.
  const evStations = useMemo<{ stations: Station[] | null; missing: string[]; source: "evidence" | "picks" }>(() => {
    if (!stationById) return { stations: null, missing: [], source: "evidence" };
    if (evidence.status === "ready" && evidence.evidence) {
      return { ...evidenceStations(evidence.evidence.traces, stationById), source: "evidence" };
    }
    if (evidence.status === "error" && event) {
      return { stations: pickingStations(event.pickIds, stationById), missing: [], source: "picks" };
    }
    return { stations: null, missing: [], source: "evidence" };
  }, [evidence, stationById, event]);

  // Open latency, logged per open, in three stages after select(): traces committed to the DOM (layout
  // effect), the frame that paints them starts (rAF), and the frame after it starts (double rAF, so the
  // paint has happened). A slow third number with fast first two means the compositor, not the drawer.
  const tracesOnScreen = open && event !== null && evidence.status === "ready";
  useLayoutEffect(() => {
    const start = openedAt.current;
    if (!tracesOnScreen || start === null || typeof requestAnimationFrame !== "function") return;
    openedAt.current = null;
    const committed = performance.now() - start;
    let frame = 0;
    let inner = 0;
    const outer = requestAnimationFrame(() => {
      frame = performance.now() - start;
      inner = requestAnimationFrame(() => {
        console.info(formatOpenLatency(shownId ?? "", committed, frame, performance.now() - start));
      });
    });
    return () => {
      cancelAnimationFrame(outer);
      cancelAnimationFrame(inner);
    };
  }, [tracesOnScreen, shownId]);

  const close = () => useDemo.getState().select(null);

  return (
    <aside
      className="hqd"
      data-open={open}
      aria-hidden={!open}
      inert={!open}
      aria-label={shownId ? `Evidence for candidate event ${shownId}` : "Evidence drawer"}
    >
      <style>{DRAWER_CSS}</style>
      {shownId !== null && (
        <>
          <Header eventId={shownId} event={event} meta={ready?.meta ?? null} onClose={close} />
          {bundle.status === "loading" && <div className="hqd-section hqd-dim">Loading the data bundle…</div>}
          {bundle.status === "error" && (
            <div className="hqd-section hqd-error" role="status">
              <b>Data bundle unavailable.</b> {bundle.message}
            </div>
          )}
          {ready && !event && (
            <div className="hqd-section hqd-error" role="status">
              This event is not in the loaded bundle.
            </div>
          )}
          {ready && event && (
            <>
              <RecordSection
                key={event.id}
                status={evidence.status}
                evidence={evidence.status === "ready" ? evidence.evidence : undefined}
                message={evidence.status === "error" ? evidence.message : undefined}
                originT={event.t}
                nStations={event.quality.nStations}
              />
              <Figures
                event={event}
                meta={ready.meta}
                stations={evStations.stations}
                missing={evStations.missing}
                source={evStations.source}
              />
            </>
          )}
        </>
      )}
    </aside>
  );
}

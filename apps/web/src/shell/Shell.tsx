"use client";

import { useMemo, useState } from "react";
import { useBundle, useMode } from "@/providers";
import { TourButton } from "@/scene";
import { useDemo } from "@/state/demo";
import { Counters } from "./Counters";
import { DownloadButton } from "./download/DownloadButton";
import { EventList, EventListButton } from "./events/EventList";
import { HelpButton, HelpOverlay, useHelpKey } from "./help/HelpOverlay";
import { FilterPills, ModePills } from "./Pills";
import { RevealButton } from "./RevealButton";
import { RunDetailsButton, RunDetailsPanel, useRunDetailsKey } from "./run-details/RunDetailsPanel";
import { kioskRequested, useAttractMode } from "./kiosk";
import { useShareLink } from "./share";
import styles from "./Shell.module.css";
import { useKeyboard } from "./useKeyboard";
import { ValidationPanel } from "./validation/ValidationPanel";

/**
 * The overlay on top of the scene (docs/01 → Web app). Pointer-transparent, so the canvas keeps
 * orbit and drag; only the controls accept input. Every word here is a label; every number comes
 * from the provider's `AnalysisSummary` through `<Counters/>`.
 */
export function Shell() {
  const bundle = useBundle();
  // The mode is known from `?mode=` before the bundle is; the loading line names it.
  const mode = useMode();
  const phase = useDemo((s) => s.phase);
  const ready = bundle.status === "ready";
  useKeyboard({ heroEventId: ready ? bundle.meta.scene.heroEventId : null, ready });
  // DEMO-02: the Run details overlay, toggled by its button under the mode label or by D.
  const [detailsOpen, setDetailsOpen] = useState(false);
  useRunDetailsKey(setDetailsOpen, ready);
  // DEMO-04: "How it works" (? button and key) and the candidate event list.
  const [helpOpen, setHelpOpen] = useState(false);
  useHelpKey(setHelpOpen);
  const [listOpen, setListOpen] = useState(false);
  // Share links: `?event=<id>` opens that drawer after the reveal; the address bar follows selection.
  const events = ready ? bundle.events : null;
  const eventIds = useMemo(() => (events ? new Set(events.map((e) => e.id)) : null), [events]);
  useShareLink(eventIds);
  // DEMO-05: expo attract mode (`?kiosk=1` loops the tour; otherwise idle on the start frame starts it).
  const [kiosk] = useState(() => typeof window !== "undefined" && kioskRequested(window.location.search));
  useAttractMode({ kiosk, ready });

  const synthetic = ready && bundle.info.isSynthetic;

  return (
    <div className={styles.shell} data-banner={synthetic || undefined}>
      {synthetic && <SyntheticBanner />}

      <header className={styles.header}>
        <h1 className={styles.title}>Hidden Quakes</h1>
        {/* REQ-H3-13: what the first frame shows, in one plain line (H2's wording). */}
        <p className={styles.subtitle}>Seismic events beneath Utah&apos;s geothermal field near Milford, from public data only</p>
        <p className={styles.phoneNote}>Best on a laptop: the view is three-dimensional and keyboard driven.</p>
        <p className={styles.modeLabel} data-testid="mode-label">
          {bundle.status === "ready" && bundle.info.label}
          {bundle.status === "loading" && (mode ? `Loading ${mode}…` : "Loading…")}
          {bundle.status === "error" && mode}
        </p>
        <div className={styles.headerButtons}>
          {ready && <RunDetailsButton open={detailsOpen} onToggle={() => setDetailsOpen((open) => !open)} />}
          {/* REQ-H3-15: the guided tour (G); hides itself while the tour plays. */}
          {ready && <TourButton className={`${styles.pill} ${styles.modePill}`} />}
          {ready && <EventListButton open={listOpen} onToggle={() => setListOpen((open) => !open)} />}
          <HelpButton open={helpOpen} onToggle={() => setHelpOpen((open) => !open)} />
        </div>
        <DownloadButton />
      </header>

      <div className={styles.topRight}>
        {ready && <Counters summary={bundle.meta.summary} />}
        {ready && phase !== "public" && <FilterPills />}
      </div>

      {ready && phase === "public" && <RevealButton />}
      {bundle.status === "error" && <ErrorPanel message={bundle.message} />}

      {/* DEMO-02 panels: validation (bottom-left, after the reveal) and the Run details overlay. */}
      <ValidationPanel />
      {ready && <RunDetailsPanel open={detailsOpen} onClose={() => setDetailsOpen(false)} />}
      <EventList open={listOpen} onClose={() => setListOpen(false)} />
      <HelpOverlay open={helpOpen} onClose={() => setHelpOpen(false)} />

      {/* The pressed pill is the mode chosen in the URL: during live failover the bundle says
          "snapshot" while LIVE is still what the visitor picked, and the label carries the rest. */}
      <ModePills current={ready ? (mode ?? bundle.info.mode) : null} />
    </div>
  );
}

/** Mock data on screen is never mistaken for a catalog: full-width, alert red, words only. */
function SyntheticBanner() {
  return (
    <div className={styles.banner} role="status">
      Synthetic data · generated by the mock fixture · not a catalog
    </div>
  );
}

/** The provider's own message (a schemaVersion mismatch, a missing file, a disabled mode). */
function ErrorPanel({ message }: { message: string }) {
  return (
    <div className={styles.error} role="alert">
      <p className={styles.errorTitle}>Data unavailable</p>
      <p className={styles.errorMessage}>{message}</p>
    </div>
  );
}

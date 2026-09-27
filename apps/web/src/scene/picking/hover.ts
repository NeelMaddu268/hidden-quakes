// Hover tooltips (H2's overnight item 2): what the event under the pointer is, in one glance, without
// opening the drawer. The Picker writes the hovered event here (at most once per animation frame, only
// while the pointer moves over one); HoverTooltip reads it. Every value is the bundle's, formatted with
// the drawer's own functions so the tooltip and the drawer never disagree.

import { create, type StoreApi, type UseBoundStore } from "zustand";
import { fmtMagnitude, fmtUnit, fmtUtc } from "../../drawer/format";
import { depthKmOfElev } from "../../drawer/geometry";
import type { CatalogEvent, SceneMeta, SeismicEvent } from "../types";

export interface HoverTarget {
  kind: "candidate" | "public";
  /** SeismicEvent id (candidate) or CatalogEvent id (public). */
  id: string;
  /** Pointer position, viewport CSS px. */
  x: number;
  y: number;
}

export const useHover: UseBoundStore<StoreApi<{ target: HoverTarget | null }>> = create<{
  target: HoverTarget | null;
}>()(() => ({ target: null }));

/** Publish the hovered event (or null); a no-op when nothing changed, so a still pointer re-renders nothing. */
export function setHover(next: HoverTarget | null): void {
  const cur = useHover.getState().target;
  if (cur === next) return;
  if (cur && next && cur.kind === next.kind && cur.id === next.id && cur.x === next.x && cur.y === next.y) return;
  useHover.setState({ target: next });
}

export interface TooltipRow {
  text: string;
  /** Emphasised rows: the public-catalog status. */
  tone?: "accent" | "dim";
}

export interface TooltipContent {
  title: string;
  rows: TooltipRow[];
}

/** Origin time to the second (the drawer's `fmtUtc` without the centiseconds). */
export function fmtUtcSeconds(epochS: number): string {
  return fmtUtc(epochS).replace(/\.\d\d UTC$/, " UTC");
}

const TIER_TITLE: Readonly<Record<SeismicEvent["tier"], string>> = {
  A: "Candidate event · Tier A (strict)",
  B: "Candidate event · Tier B",
  C: "Candidate event · Tier C",
};

/**
 * A candidate event: tier, origin time (UTC), depth below the site surface, magnitude, stations that
 * agreed, and whether the public regional catalog lists it.
 */
export function candidateTooltip(
  e: Pick<SeismicEvent, "tier" | "t" | "elevM" | "magnitude" | "catalogMatch"> & {
    quality: Pick<SeismicEvent["quality"], "nStations">;
  },
  scene: Pick<SceneMeta, "refSurfaceElevM">,
): TooltipContent {
  const n = e.quality.nStations;
  return {
    title: TIER_TITLE[e.tier],
    rows: [
      { text: fmtUtcSeconds(e.t) },
      { text: `${fmtUnit(depthKmOfElev(e.elevM, scene), 2, "km")} below site surface` },
      { text: fmtMagnitude(e.magnitude) ?? "No magnitude" },
      { text: Number.isFinite(n) ? `${n} ${n === 1 ? "station" : "stations"} agreed` : "Stations unknown" },
      e.catalogMatch != null
        ? { text: "In the public regional catalog", tone: "dim" }
        : { text: "Not in the public regional catalog", tone: "accent" },
    ],
  };
}

/**
 * A public regional catalog event: origin time, depth below the site surface (from its elevation, as
 * the scene draws it), the catalog's magnitude, and after the reveal whether our pipeline recovered it
 * (before the reveal that would give the reveal away, so it isn't said).
 */
export function publicTooltip(
  c: Pick<CatalogEvent, "t" | "elevM" | "mag" | "magType" | "matchedEventId">,
  scene: Pick<SceneMeta, "refSurfaceElevM">,
  revealed: boolean,
  matchedTier: SeismicEvent["tier"] | null,
): TooltipContent {
  const mag =
    c.mag !== null && Number.isFinite(c.mag) ? fmtMagnitude({ value: c.mag, type: c.magType ?? "", sigma: null }) : null;
  const rows: TooltipRow[] = [
    { text: fmtUtcSeconds(c.t) },
    { text: `${fmtUnit(depthKmOfElev(c.elevM, scene), 2, "km")} below site surface` },
    { text: mag?.trimEnd() ?? "No magnitude" },
  ];
  if (revealed) {
    rows.push(
      matchedTier !== null
        ? { text: `Recovered by our pipeline · Tier ${matchedTier}`, tone: "dim" }
        : { text: "Not recovered by our pipeline", tone: "dim" },
    );
  }
  return { title: "Public regional catalog event", rows };
}

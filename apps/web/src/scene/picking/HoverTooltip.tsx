"use client";

import { colors, fonts, numeric } from "@hq/visualization";
import { useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useDemo } from "../../state/demo";
import { useBundle } from "../data";
import { drawerWidthPx } from "../plan/layout";
import { useTour } from "../tour/store";
import type { BundleState, CatalogEvent, SeismicEvent } from "../types";
import { candidateTooltip, publicTooltip, useHover, type TooltipContent } from "./hover";

type ReadyBundle = Extract<BundleState, { status: "ready" }>;

/** Distance from the pointer to the tooltip's corner (CSS px). */
const OFFSET_PX = 14;
/** Kept clear of the viewport edges and the open drawer (CSS px). */
const EDGE_PX = 8;

/**
 * The event under the pointer, described in one small card that follows it (H2's item 2): tier, origin
 * time (UTC), depth below the site surface, magnitude, stations and whether the public regional catalog
 * lists it; public points say what the catalog says. Hidden while the guided tour plays (a clean
 * recording), and never over the open drawer.
 */
export function HoverTooltip() {
  const bundle = useBundle();
  if (bundle.status !== "ready") return null;
  return <Tooltip bundle={bundle} />;
}

function Tooltip({ bundle }: { bundle: ReadyBundle }) {
  const target = useHover((s) => s.target);
  const touring = useTour((s) => s.running);
  const revealed = useDemo((s) => s.phase !== "public");
  const drawerOpen = useDemo((s) => s.selectedEventId !== null);
  const eventsById = useMemo(() => new Map<string, SeismicEvent>(bundle.events.map((e) => [e.id, e])), [bundle.events]);
  const catalogById = useMemo(
    () => new Map<string, CatalogEvent>(bundle.catalog.map((c) => [c.id, c])),
    [bundle.catalog],
  );
  const scene = bundle.meta.scene;

  const content: TooltipContent | null = useMemo(() => {
    if (target === null) return null;
    if (target.kind === "candidate") {
      const e = eventsById.get(target.id);
      return e ? candidateTooltip(e, scene) : null;
    }
    const c = catalogById.get(target.id);
    if (!c) return null;
    const matched = c.matchedEventId ? (eventsById.get(c.matchedEventId)?.tier ?? null) : null;
    return publicTooltip(c, scene, revealed, matched);
  }, [target, eventsById, catalogById, scene, revealed]);

  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const x = target?.x ?? 0;
  const y = target?.y ?? 0;
  // Below-right of the pointer; flipped to the other side of it when that would leave the viewport or
  // run under the open drawer. Measured before paint, so it never flashes in the wrong place.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    const right = window.innerWidth - (drawerOpen ? drawerWidthPx(window.innerWidth) : 0) - EDGE_PX;
    const bottom = window.innerHeight - EDGE_PX;
    let left = x + OFFSET_PX;
    if (left + w > right) left = x - OFFSET_PX - w;
    let top = y + OFFSET_PX;
    if (top + h > bottom) top = y - OFFSET_PX - h;
    const next = { left: Math.max(EDGE_PX, Math.round(left)), top: Math.max(EDGE_PX, Math.round(top)) };
    setPos((prev) => (prev && prev.left === next.left && prev.top === next.top ? prev : next));
  }, [x, y, content, drawerOpen]);

  if (content === null || touring) return null;
  return (
    <div
      ref={ref}
      role="tooltip"
      data-testid="hover-tooltip"
      style={{ ...cardStyle, left: pos?.left ?? x + OFFSET_PX, top: pos?.top ?? y + OFFSET_PX }}
    >
      <div style={titleStyle}>{content.title}</div>
      {content.rows.map((row, i) => (
        <div key={i} style={row.tone === "accent" ? accentRow : row.tone === "dim" ? dimRow : rowStyle}>
          {row.text}
        </div>
      ))}
    </div>
  );
}

const cardStyle: CSSProperties = {
  position: "fixed",
  zIndex: 26,
  boxSizing: "border-box",
  maxWidth: 300,
  padding: "7px 10px 8px",
  border: `1px solid ${colors.contour}`,
  borderRadius: 6,
  background: `${colors.surface}F2`,
  color: colors.text,
  fontFamily: fonts.ui,
  fontSize: 12.5,
  lineHeight: 1.45,
  pointerEvents: "none",
  whiteSpace: "nowrap",
  boxShadow: `0 6px 20px ${colors.bg}B3`,
};

const titleStyle: CSSProperties = {
  marginBottom: 2,
  color: colors.textDim,
  fontSize: 10.5,
  fontWeight: 600,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
};

const rowStyle: CSSProperties = { fontFamily: fonts.mono, ...numeric, fontSize: 12 };
const dimRow: CSSProperties = { color: colors.textDim };
const accentRow: CSSProperties = { color: colors.recovered, fontWeight: 600 };

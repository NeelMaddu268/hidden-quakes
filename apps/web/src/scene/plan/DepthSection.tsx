"use client";

import { colors, fonts, numeric } from "@hq/visualization";
import { useEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent as ReactMouseEvent } from "react";
import { useDemo } from "../../state/demo";
import { useBundle } from "../data";
import { sceneFx } from "../fx";
import { publicSwatch } from "../look";
import { selectedInstanceIndex } from "../picking/selection";
import type { BundleState } from "../types";
import { sectionPanelRect } from "./layout";
import {
  buildSectionModel,
  drawSection,
  sectionHit,
  sectionPlot,
  type SectionState,
  type SectionStyle,
} from "./section";

type ReadyBundle = Extract<BundleState, { status: "ready" }>;

/** Header (title + projection note) and footer (axis captions) heights inside the panel, CSS px. */
const HEADER_PX = 38;
const FOOTER_PX = 36;
/** Click radius in the section, CSS px (the 3D picker's default). */
const HIT_PX = 10;

/**
 * The plan view's depth section (WEB-07): a docked panel beside the plan map that projects every event
 * onto grid east against depth below the site surface, at true scale. Mounted only in plan view. It
 * shares the scene's reveal clock, filter look and selection (clicks select through the store, so the
 * drawer opens exactly as from the map), and it never covers the shell or the evidence drawer.
 */
export function DepthSection() {
  const bundle = useBundle();
  const view = useDemo((s) => s.view);
  if (bundle.status !== "ready" || view !== "plan") return null;
  return <SectionPanel bundle={bundle} />;
}

function useViewport(): { width: number; height: number } {
  const [vp, setVp] = useState(() => ({ width: window.innerWidth, height: window.innerHeight }));
  useEffect(() => {
    const on = () => setVp({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return vp;
}

function SectionPanel({ bundle }: { bundle: ReadyBundle }) {
  const { meta, events, catalog, stations } = bundle;
  const vp = useViewport();
  const rect = sectionPanelRect(vp.width, vp.height);
  const canvas = useRef<HTMLCanvasElement>(null);
  const headerPx = HEADER_PX;

  const model = useMemo(
    () => buildSectionModel(events, catalog, stations, meta.scene, meta.run.windowStart),
    [events, catalog, stations, meta.scene, meta.run.windowStart],
  );
  const cssW = rect ? rect.width - 2 : 0;
  const cssH = rect ? rect.height - headerPx - FOOTER_PX : 0;
  const plot = useMemo(() => (cssW > 0 && cssH > 0 ? sectionPlot(model, cssW, cssH) : null), [model, cssW, cssH]);

  // Draw loop: redraws only when something visible changed (reveal clock, filter look, selection,
  // size), so a settled section costs nothing per frame.
  useEffect(() => {
    const el = canvas.current;
    if (!el || !plot) return;
    const ctx = el.getContext("2d");
    if (!ctx) throw new Error("depth section: 2D canvas unavailable");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    el.width = Math.round(cssW * dpr);
    el.height = Math.round(cssH * dpr);
    const style: SectionStyle = {
      recovered: colors.recovered,
      publicDot: publicSwatch(colors.public),
      station: colors.station,
      contour: colors.contour,
      textDim: colors.textDim,
      halo: colors.strictHalo,
      tickFont: `10px ${fonts.mono}`,
      dpr,
    };
    const state: SectionState = {
      phase: "public",
      filter: "public",
      revealElapsedS: 0,
      timeNowRel: sceneFx.timeNowRel,
      look: { ...sceneFx.filterLook },
      selectedIndex: -1,
    };
    const last = { phase: "", filter: "", clock: NaN, now: NaN, a: NaN, b: NaN, c: NaN, cand: NaN, pub: NaN, halos: NaN, sel: NaN };
    let raf = 0;
    const frame = () => {
      raf = requestAnimationFrame(frame);
      const s = useDemo.getState();
      const look = sceneFx.filterLook;
      const sel = selectedInstanceIndex(s.selectedEventId, model.indexById);
      const clock = sceneFx.revealElapsedS;
      if (
        s.phase === last.phase && s.filter === last.filter && clock === last.clock && sel === last.sel &&
        sceneFx.timeNowRel === last.now &&
        look.tierA === last.a && look.tierB === last.b && look.tierC === last.c &&
        look.candidates === last.cand && look.publicLayer === last.pub && look.halos === last.halos
      ) return;
      last.phase = s.phase; last.filter = s.filter; last.clock = clock; last.sel = sel;
      last.now = sceneFx.timeNowRel;
      last.a = look.tierA; last.b = look.tierB; last.c = look.tierC;
      last.cand = look.candidates; last.pub = look.publicLayer; last.halos = look.halos;
      state.phase = s.phase;
      state.filter = s.filter;
      state.revealElapsedS = clock;
      state.timeNowRel = sceneFx.timeNowRel;
      state.selectedIndex = sel;
      Object.assign(state.look, look);
      drawSection(ctx, model, plot, state, style, cssW, cssH);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [model, plot, cssW, cssH]);

  if (!rect || !plot) return null;

  const onClick = (e: ReactMouseEvent<HTMLCanvasElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const s = useDemo.getState();
    const id = sectionHit(
      model,
      plot,
      { phase: s.phase, filter: s.filter, revealElapsedS: sceneFx.revealElapsedS, timeNowRel: sceneFx.timeNowRel },
      e.clientX - box.left,
      e.clientY - box.top,
      HIT_PX,
    );
    if (id) s.select(id); // empty space keeps the selection (Esc deselects), as on the map
  };

  const panel: CSSProperties = {
    position: "fixed",
    left: rect.left,
    top: rect.top,
    width: rect.width,
    height: rect.height,
    zIndex: 5,
    boxSizing: "border-box",
    background: `${colors.surface}E6`,
    border: `1px solid ${colors.contour}`,
    borderRadius: 8,
    color: colors.textDim,
    fontFamily: fonts.ui,
    overflow: "hidden",
  };
  return (
    <section style={panel} aria-label="Depth section" data-testid="depth-section">
      <header style={{ height: headerPx, padding: "7px 10px 0", boxSizing: "border-box", lineHeight: 1.25 }}>
        <div style={{ color: colors.text, fontSize: 11, fontWeight: 600, letterSpacing: "0.14em", textTransform: "uppercase" }}>
          Depth section
        </div>
        <div style={{ fontSize: 10.5 }}>Every event projected onto grid east · true scale (1 km = 1 km)</div>
      </header>
      <canvas
        ref={canvas}
        onClick={onClick}
        style={{ display: "block", width: cssW, height: cssH, cursor: "crosshair" }}
        role="img"
        aria-label={`Depth section: ${events.length} candidate events and ${catalog.length} public-catalog events projected onto grid east`}
      />
      <footer style={{ height: FOOTER_PX, padding: "2px 10px 0", boxSizing: "border-box", fontSize: 10, lineHeight: 1.35, ...numeric }}>
        <div style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>↓ {meta.scene.depthLabel}, km</div>
        <div style={{ textAlign: "right" }}>grid east of origin, km →</div>
      </footer>
    </section>
  );
}

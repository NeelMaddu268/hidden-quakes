"use client";

import { colors, fonts, numeric } from "@hq/visualization";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { useDemo } from "../../state/demo";
import { useBundle } from "../data";
import { LOOK, publicSwatch } from "../look";
import { countUpTo, fmtClockUtc, hourTicks } from "./clock";
import { scrubberRect } from "./layout";
import { buildStripModel, drawStrip, recoveredSet, stripTime, stripX, type StripModel, type StripStyle } from "./strip";

const PAD_X = 12;
const HEADER_PX = 30;
const STRIP_PX = 50;
const AXIS_PX = 18;

const PUBLIC_BAR = publicSwatch(colors.public);

function useViewport(): { width: number; height: number } {
  const [size, setSize] = useState(() =>
    typeof window === "undefined" ? { width: 0, height: 0 } : { width: window.innerWidth, height: window.innerHeight },
  );
  useEffect(() => {
    const onResize = () => setSize({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return size;
}

/** Play / pause glyph (inline SVG, token colors). */
function PlayIcon({ playing }: { playing: boolean }) {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      {playing ? (
        <>
          <rect x="2" y="1.5" width="3" height="9" fill="currentColor" />
          <rect x="7" y="1.5" width="3" height="9" fill="currentColor" />
        </>
      ) : (
        <path d="M3 1.5 L10.5 6 L3 10.5 Z" fill="currentColor" />
      )}
    </svg>
  );
}

/** One replay action: resume, or restart from the window start once the playhead is at the end. */
function togglePlay(model: StripModel): void {
  const s = useDemo.getState();
  if (s.playing) {
    s.setPlaying(false);
    return;
  }
  if (s.tNow === null || s.tNow >= model.windowEnd) s.setTNow(model.windowStart);
  s.setPlaying(true);
}

interface StripViewProps {
  model: StripModel;
  width: number;
}

/**
 * The strip, the clock and the "so far" counts. They change every frame during playback, so they are
 * drawn from a store subscription (coalesced to one redraw per animation frame) straight into the
 * canvas and a few text nodes: nothing here re-renders React while the replay runs.
 */
function StripView({ model, width }: StripViewProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const slider = useRef<HTMLDivElement>(null);
  const clock = useRef<HTMLSpanElement>(null);
  const pubCount = useRef<HTMLSpanElement>(null);
  const recCount = useRef<HTMLSpanElement>(null);
  const recLabel = useRef<HTMLSpanElement>(null);
  const recSep = useRef<HTMLSpanElement>(null);
  const playButton = useRef<HTMLButtonElement>(null);
  const dragging = useRef<number | null>(null);
  const playing = useDemo((s) => s.playing);
  const stripW = Math.max(1, width - 2 * PAD_X);
  const ticks = useMemo(() => hourTicks(model.windowStart, model.windowEnd), [model]);

  useEffect(() => {
    const c = canvas.current;
    // No 2D context (jsdom, a lost context): the strip stays blank, the clock and counts still update.
    const ctx = c?.getContext("2d") ?? null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (c) {
      c.width = Math.round(stripW * dpr);
      c.height = Math.round(STRIP_PX * dpr);
    }
    const style: StripStyle = {
      recovered: colors.recovered,
      publicBar: PUBLIC_BAR,
      baseline: colors.contour,
      playhead: colors.text,
      dpr,
    };
    let raf = 0;
    // What the text nodes last showed: during playback the strip redraws every frame, but the clock (to
    // the minute) and the counts change far less often, so strings are built only when they change.
    const shown = { minute: NaN, pub: -1, rec: -1, set: "" };
    const draw = () => {
      raf = 0;
      const s = useDemo.getState();
      if (ctx) drawStrip(ctx, model, s.tNow, s.filter, style, stripW, STRIP_PX);
      const now = s.tNow ?? model.windowEnd;
      const set = recoveredSet(s.filter);
      const sorted = set === "strict" ? model.sortedRecoveredStrict : model.sortedRecoveredAll;
      const minute = Math.floor(now / 60);
      const pub = countUpTo(model.sortedPublic, now);
      const rec = set === "none" ? 0 : countUpTo(sorted, now);
      if (minute !== shown.minute) {
        shown.minute = minute;
        const text = `${fmtClockUtc(now, model.windowStart)} UTC`;
        if (clock.current) clock.current.textContent = text;
        const el = slider.current;
        if (el) {
          el.setAttribute("aria-valuenow", String(Math.round(now)));
          el.setAttribute("aria-valuetext", text);
        }
      }
      if (pub !== shown.pub && pubCount.current) {
        shown.pub = pub;
        pubCount.current.textContent = pub.toLocaleString("en-US");
      }
      if ((rec !== shown.rec || set !== shown.set) && recCount.current) {
        shown.rec = rec;
        recCount.current.textContent = set === "none" ? "" : rec.toLocaleString("en-US");
      }
      if (set !== shown.set) {
        shown.set = set;
        if (recLabel.current) recLabel.current.textContent = set === "none" ? "" : set === "strict" ? " strict" : " recovered";
        if (recSep.current) recSep.current.textContent = set === "none" ? "" : " · ";
      }
    };
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(draw);
    };
    draw();
    const unsubscribe = useDemo.subscribe((s, prev) => {
      if (s.tNow !== prev.tNow || s.filter !== prev.filter) schedule();
    });
    return () => {
      unsubscribe();
      if (raf) cancelAnimationFrame(raf);
    };
  }, [model, stripW]);

  const seekTo = (clientX: number) => {
    const box = canvas.current?.getBoundingClientRect();
    if (!box) return;
    useDemo.getState().setTNow(stripTime(model, clientX - box.left, box.width));
  };
  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    dragging.current = e.pointerId;
    useDemo.getState().setPlaying(false);
    seekTo(e.clientX);
  };
  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (dragging.current === e.pointerId) seekTo(e.clientX);
  };
  const onPointerEnd = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (dragging.current !== e.pointerId) return;
    dragging.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  };
  // Slider keys (only while the strip has focus): one bin per arrow, an hour with Shift, Home / End.
  const onKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const s = useDemo.getState();
    const now = s.tNow ?? model.windowEnd;
    const step = e.shiftKey ? 3600 : model.binS;
    let next: number | null = null;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") next = now - step;
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") next = now + step;
    else if (e.key === "Home") next = model.windowStart;
    else if (e.key === "End") next = model.windowEnd;
    if (next === null) return;
    e.preventDefault();
    e.stopPropagation();
    s.setPlaying(false);
    s.setTNow(Math.min(model.windowEnd, Math.max(model.windowStart, next)));
  };

  const dim: CSSProperties = { color: colors.textDim };
  return (
    <>
      <header
        style={{ height: HEADER_PX, display: "flex", alignItems: "center", gap: 10, padding: `0 ${PAD_X}px`, boxSizing: "border-box" }}
      >
        <button
          ref={playButton}
          type="button"
          onClick={() => togglePlay(model)}
          // Space / Enter activate the button, so they must not also reach the shell's Space beat. Every
          // other key (P, S, R, T, E, Esc) still reaches the presenter keyboard while the button has focus.
          onKeyDown={(e) => {
            if (e.key === " " || e.key === "Enter") e.stopPropagation();
          }}
          aria-label={playing ? "Pause replay" : "Play replay"}
          data-testid="time-play"
          style={{
            width: 24,
            height: 24,
            display: "grid",
            placeItems: "center",
            padding: 0,
            border: `1px solid ${colors.contour}`,
            borderRadius: 999,
            background: "transparent",
            color: colors.text,
            cursor: "pointer",
          }}
        >
          <PlayIcon playing={playing} />
        </button>
        <span style={{ color: colors.text, fontSize: 11, fontWeight: 600, letterSpacing: "0.14em", textTransform: "uppercase" }}>
          Time
        </span>
        <span ref={clock} data-testid="time-clock" style={{ color: colors.text, fontFamily: fonts.mono, fontSize: 14, ...numeric }} />
        <span style={{ flex: 1 }} />
        <span data-testid="time-counts" style={{ fontFamily: fonts.mono, fontSize: 11, ...numeric, ...dim, whiteSpace: "nowrap" }}>
          <span ref={pubCount} style={{ color: PUBLIC_BAR }} /> public
          <span ref={recSep} />
          <span ref={recCount} style={{ color: colors.recovered }} />
          <span ref={recLabel} />
        </span>
      </header>
      <div
        ref={slider}
        role="slider"
        tabIndex={0}
        aria-label="Time within the run window"
        aria-valuemin={Math.round(model.windowStart)}
        aria-valuemax={Math.round(model.windowEnd)}
        // The draw loop keeps aria-valuenow / aria-valuetext current; this is the pre-draw value.
        aria-valuenow={Math.round(model.windowEnd)}
        data-testid="time-strip"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerEnd}
        onPointerCancel={onPointerEnd}
        onKeyDown={onKeyDown}
        style={{ padding: `0 ${PAD_X}px`, cursor: "ew-resize", touchAction: "none", outline: "none" }}
      >
        <canvas ref={canvas} style={{ display: "block", width: stripW, height: STRIP_PX }} />
      </div>
      <div style={{ position: "relative", height: AXIS_PX, margin: `0 ${PAD_X}px`, fontFamily: fonts.mono, fontSize: 10, ...numeric, ...dim }}>
        {ticks.map((t, i) => {
          const x = stripX(model, t, stripW);
          const edge = i === 0 && x < 20 ? "start" : i === ticks.length - 1 && x > stripW - 20 ? "end" : "mid";
          return (
            <span
              key={t}
              style={{
                position: "absolute",
                top: 3,
                left: x,
                transform: edge === "start" ? "none" : edge === "end" ? "translateX(-100%)" : "translateX(-50%)",
                whiteSpace: "nowrap",
              }}
            >
              {fmtClockUtc(t, model.windowStart)}
            </span>
          );
        })}
      </div>
    </>
  );
}

/**
 * The time scrubber (docs/02 §6, WEB-06): shown in time mode once the reveal has finished, docked along
 * the bottom (layout.ts). A per-10-minute histogram of the window (recovered above, the public regional
 * catalog below, one shared scale), a playhead at tNow, the clock in UTC, play / pause, and how many
 * events of each kind are on screen so far. Drag or click the strip to scrub (it pauses the replay);
 * with the strip focused, arrow keys step one bin (Shift: an hour), Home / End jump to the ends. Every
 * number is counted from the bundle.
 */
export function TimeScrubber() {
  const bundle = useBundle();
  const timeMode = useDemo((s) => s.timeMode);
  const phase = useDemo((s) => s.phase);
  const drawerOpen = useDemo((s) => s.selectedEventId !== null);
  const viewport = useViewport();
  const ready = bundle.status === "ready" ? bundle : null;
  const model = useMemo(
    () =>
      ready
        ? buildStripModel(ready.events, ready.catalog, ready.meta.run.windowStart, ready.meta.run.windowEnd, LOOK.time.binS)
        : null,
    [ready],
  );
  const rect = scrubberRect(viewport.width, viewport.height, drawerOpen);
  if (!model || !ready || !rect || !timeMode || phase !== "revealed") return null;

  const binMin = Math.round(LOOK.time.binS / 60);
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
    userSelect: "none",
  };
  return (
    <section style={panel} aria-label="Time scrubber" data-testid="time-scrubber">
      <StripView model={model} width={rect.width} />
      <div
        style={{
          position: "absolute",
          right: PAD_X,
          bottom: 4,
          fontSize: 9.5,
          color: colors.textDim,
          opacity: 0.8,
          fontFamily: fonts.mono,
          ...numeric,
        }}
        data-testid="time-scale"
      >
        {`${binMin}-min bins · max ${model.maxBin}`}
      </div>
    </section>
  );
}

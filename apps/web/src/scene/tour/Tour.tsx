"use client";

import { colors, fonts, motion, numeric } from "@hq/visualization";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useDemo } from "../../state/demo";
import { useBundle } from "../data";
import { hiddenHeroEventId, selectHiddenHero } from "../hiddenHero";
import { isPlainPress } from "../keys";
import { LOOK, publicSwatch } from "../look";
import { drawerWidthPx } from "../plan/layout";
import { H3_OVERLAY_SELECTORS, shellRoot } from "../references/overlayObstacles";
import type { BundleState } from "../types";
import { captionText, fillCaption, tourValues, type CaptionSegment, type ValueTone } from "./captions";
import { TOUR_COPY } from "./copy";
import { captionBox, type CaptionBox, type ObstacleRect } from "./layout";
import { TOUR_STEPS, TourRun, type TourContext, type TourStepId } from "./script";
import { setTourStep, startTour, stopTour, toggleTour, useTour } from "./store";

/** The key that starts the tour (no other presenter key uses it; docs/02 §6 → Keyboard). */
export const TOUR_KEY = "g";

/** The shell's Validation card, measured read-only (its `validation-panel` test id). */
const VALIDATION_CARD = '[data-testid="validation-panel"]';

/** Marks the tour's own controls: a click on them is theirs, not a "stop" click. */
const CONTROL_ATTR = "data-tour-control";

/** How often overlay rects are re-measured while the tour runs (ms). */
const MEASURE_MS = 250;

/** Keys that never stop the tour: bare modifiers (the first half of an OS or recorder shortcut). */
const MODIFIER_KEYS = new Set(["Shift", "Control", "Alt", "Meta", "CapsLock", "Fn", "FnLock", "Hyper", "Super"]);

type ReadyBundle = Extract<BundleState, { status: "ready" }>;

/**
 * The guided tour (WEB-09): G (or H4's Tour button) plays the judge sequence with a caption per step;
 * any key, click or scroll stops it and leaves the scene where it is. Mounted by the scene next to the
 * depth section; renders nothing until the bundle is ready, and only the caption while running.
 */
export function Tour() {
  const bundle = useBundle();
  if (bundle.status !== "ready") return null;
  return <TourOverlay bundle={bundle} />;
}

function TourOverlay({ bundle }: { bundle: ReadyBundle }) {
  const running = useTour((s) => s.running);
  const runs = useTour((s) => s.runs);
  const stepId = useTour((s) => s.step);
  const events = bundle.events;
  const heroStations = useMemo(() => {
    const id = hiddenHeroEventId(events);
    return id === null ? null : (events.find((e) => e.id === id)?.quality.nStations ?? null);
  }, [events]);
  const values = useMemo(
    () => tourValues(bundle.meta.summary, heroStations, LOOK.time.playbackRate),
    [bundle.meta.summary, heroStations],
  );

  useTourInput();

  // One run per start: the animation-frame clock ticks it; the store publishes the step on change.
  useEffect(() => {
    if (!running) return;
    const ctx: TourContext = {
      openHidden: () => selectHiddenHero(events) !== null,
      hasValidationCard: () => document.querySelector(VALIDATION_CARD) !== null,
    };
    const run = new TourRun(TOUR_STEPS, useDemo.getState, ctx);
    let raf = 0;
    const publish = (id: TourStepId | null) => {
      if (id === null) stopTour();
      else setTourStep(id);
      return id !== null;
    };
    const frame = () => {
      if (publish(run.tick(performance.now())?.id ?? null)) raf = requestAnimationFrame(frame);
    };
    if (publish(run.start(performance.now())?.id ?? null)) raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [running, runs, events]);

  const segments = useMemo(
    () => (stepId === null ? null : fillCaption(TOUR_COPY[TOUR_STEPS.find((s) => s.id === stepId)!.caption], values)),
    [stepId, values],
  );
  if (!running || stepId === null) return null;
  return (
    <>
      {stepId === "validation" && <Spotlight selector={VALIDATION_CARD} />}
      {segments !== null && <Caption segments={segments} stepId={stepId} />}
    </>
  );
}

/** G starts the tour; while it runs, any key (swallowed), click or scroll stops it. */
function useTourInput(): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (useTour.getState().running) {
        // OS and recorder shortcuts (Cmd/Ctrl/Alt combos, bare modifiers) must not end a recording.
        if (MODIFIER_KEYS.has(event.key) || event.metaKey || event.ctrlKey || event.altKey) return;
        // Captured on window before the shell's presenter keys, which skip prevented events: the key
        // that stops the tour does nothing else.
        event.preventDefault();
        event.stopImmediatePropagation();
        stopTour();
        return;
      }
      if (!isPlainPress(event)) return;
      if (event.key.toLowerCase() === TOUR_KEY) {
        event.preventDefault();
        startTour();
      }
    };
    const onPointer = (event: Event) => {
      if (!useTour.getState().running) return;
      if (event.target instanceof Element && event.target.closest(`[${CONTROL_ATTR}]`)) return;
      stopTour();
    };
    window.addEventListener("keydown", onKeyDown, { capture: true });
    window.addEventListener("pointerdown", onPointer, { capture: true });
    window.addEventListener("wheel", onPointer, { capture: true, passive: true });
    return () => {
      window.removeEventListener("keydown", onKeyDown, { capture: true });
      window.removeEventListener("pointerdown", onPointer, { capture: true });
      window.removeEventListener("wheel", onPointer, { capture: true });
    };
  }, []);
}

/** The evidence drawer's element (open, or still sliding out). */
const DRAWER = ".hqd";

/**
 * Every overlay on screen (shell blocks and H3 panels), except `exclude`, in viewport px. The drawer
 * counts where it is going, not where its slide has got to: at its final width from the moment an event
 * is selected (the caption never waits under a drawer sliding in), and at its measured rect while it
 * slides out.
 */
function measureObstacles(exclude: Element | null, drawerOpen: boolean): ObstacleRect[] {
  const out: ObstacleRect[] = [];
  const W = window.innerWidth;
  const H = window.innerHeight;
  const push = (el: Element) => {
    if (el === exclude || el.matches(DRAWER)) return;
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) out.push({ left: r.left, top: r.top, width: r.width, height: r.height });
  };
  const shell = shellRoot(document);
  if (shell) for (const child of Array.from(shell.children)) push(child);
  for (const selector of H3_OVERLAY_SELECTORS) for (const el of Array.from(document.querySelectorAll(selector))) push(el);
  if (drawerOpen) {
    const w = drawerWidthPx(W);
    out.push({ left: W - w, top: 0, width: w, height: H });
  }
  for (const el of Array.from(document.querySelectorAll(DRAWER))) {
    if (getComputedStyle(el).visibility === "hidden") continue;
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0 && r.left < W) out.push({ left: r.left, top: r.top, width: r.width, height: r.height });
  }
  return out;
}

const sameBox = (a: CaptionBox | null, b: CaptionBox) =>
  a !== null && a.left === b.left && a.bottom === b.bottom && a.width === b.width;

const TONE_COLOR: Readonly<Record<Exclude<ValueTone, "plain">, string>> = {
  public: publicSwatch(colors.public),
  recovered: colors.recovered,
};

const captionStyle: CSSProperties = {
  position: "fixed",
  zIndex: 25,
  boxSizing: "border-box",
  padding: "10px 18px 11px",
  border: `1px solid ${colors.contour}`,
  borderRadius: 8,
  background: `${colors.surface}EB`,
  color: colors.text,
  fontFamily: fonts.ui,
  fontSize: "clamp(16px, 0.9vw + 7px, 22px)",
  lineHeight: 1.4,
  textAlign: "center",
  textWrap: "balance",
  pointerEvents: "none",
  boxShadow: `0 8px 28px ${colors.bg}B3`,
};

/** Counts in the counters' mono face and tone; a plain value (the replay speed) reads as text. */
const valueStyle = (tone: ValueTone): CSSProperties =>
  tone === "plain"
    ? { fontWeight: 600 }
    : { fontFamily: fonts.mono, ...numeric, fontWeight: 600, color: TONE_COLOR[tone] };

/** The step's caption: a lower third that keeps clear of every overlay; fades in on each step. */
function Caption({ segments, stepId }: { segments: CaptionSegment[]; stepId: TourStepId }) {
  const ref = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState<CaptionBox | null>(null);
  const text = captionText(segments);

  const drawerOpen = useDemo((s) => s.selectedEventId !== null);
  // The shell mounts the Validation card when the reveal settles (phase "revealed"), mid-caption: it
  // renders in the same commit as this component, so a layout effect on the phase sees it before paint.
  const phase = useDemo((s) => s.phase);

  const measure = useRef(() => {
    const el = ref.current;
    if (!el) return;
    const obstacles = measureObstacles(el, useDemo.getState().selectedEventId !== null);
    const next = captionBox(window.innerWidth, window.innerHeight, el.offsetHeight, obstacles);
    setBox((prev) => (sameBox(prev, next) ? prev : next));
  });

  // Before paint on every new caption, width (the height depends on the wrap), drawer change and phase
  // change, then a few times a second: the overlays move with the steps (the scrubber appears, the
  // drawer slides out).
  useLayoutEffect(() => measure.current(), [text, box?.width, drawerOpen, phase]);
  useEffect(() => {
    const on = () => measure.current();
    const id = window.setInterval(on, MEASURE_MS);
    window.addEventListener("resize", on);
    return () => {
      window.clearInterval(id);
      window.removeEventListener("resize", on);
    };
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof el.animate !== "function") return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    el.animate([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], {
      duration: motion.state / 2,
      easing: motion.ease,
    });
  }, [stepId]);

  return (
    <div
      ref={ref}
      role="status"
      aria-live="polite"
      data-testid="tour-caption"
      data-step={stepId}
      style={{
        ...captionStyle,
        left: box?.left ?? 0,
        bottom: box?.bottom ?? 0,
        width: box?.width ?? "min(680px, calc(100vw - 48px))",
        visibility: box === null ? "hidden" : "visible",
      }}
    >
      {segments.map((s, i) =>
        s.kind === "text" ? (
          <span key={i}>{s.text}</span>
        ) : (
          <span key={i} style={valueStyle(s.tone)} data-placeholder={s.name}>
            {s.text}
          </span>
        ),
      )}
    </div>
  );
}

/** An accent outline around an element of the shell (read-only: measured, never touched). */
function Spotlight({ selector }: { selector: string }) {
  const [rect, setRect] = useState<ObstacleRect | null>(null);
  useEffect(() => {
    const measure = () => {
      const r = document.querySelector(selector)?.getBoundingClientRect();
      const next = r && r.width > 0 ? { left: r.left, top: r.top, width: r.width, height: r.height } : null;
      setRect((prev) =>
        prev !== null &&
        next !== null &&
        prev.left === next.left &&
        prev.top === next.top &&
        prev.width === next.width &&
        prev.height === next.height
          ? prev
          : next,
      );
    };
    measure();
    const id = window.setInterval(measure, MEASURE_MS);
    window.addEventListener("resize", measure);
    return () => {
      window.clearInterval(id);
      window.removeEventListener("resize", measure);
    };
  }, [selector]);
  if (rect === null) return null;
  const pad = 6;
  return (
    <div
      aria-hidden="true"
      data-testid="tour-spotlight"
      style={{
        position: "fixed",
        zIndex: 24,
        left: rect.left - pad,
        top: rect.top - pad,
        width: rect.width + 2 * pad,
        height: rect.height + 2 * pad,
        boxSizing: "border-box",
        border: `2px solid ${colors.recovered}`,
        borderRadius: 10,
        boxShadow: `0 0 18px ${colors.recovered}66`,
        pointerEvents: "none",
      }}
    />
  );
}

/**
 * The Tour button, for H4 to mount next to Run details (REQ-H3-15). Pass the shell's pill classes as
 * `className` to match its look; without one it draws a pill of its own. Hidden (space kept) while the
 * tour runs, so a recording shows the scene and the captions only.
 */
export function TourButton({ className }: { className?: string }) {
  const bundle = useBundle();
  const running = useTour((s) => s.running);
  if (bundle.status !== "ready") return null;
  return (
    <button
      type="button"
      className={className}
      style={{ ...(className ? null : fallbackButtonStyle), visibility: running ? "hidden" : "visible" }}
      title={TOUR_COPY.buttonTitle}
      aria-pressed={running}
      aria-keyshortcuts={TOUR_KEY.toUpperCase()}
      data-testid="tour-button"
      {...{ [CONTROL_ATTR]: "" }}
      onClick={(event) => {
        // Focus would stay on the button, and a Space that stops the tour would click it again.
        event.currentTarget.blur();
        toggleTour();
      }}
    >
      {TOUR_COPY.button}
    </button>
  );
}

const fallbackButtonStyle: CSSProperties = {
  pointerEvents: "auto",
  marginTop: "0.5rem",
  padding: "0.25rem 0.65rem",
  border: `1px solid ${colors.textDim}73`,
  borderRadius: 999,
  background: `${colors.surface}BF`,
  color: colors.textDim,
  fontFamily: fonts.ui,
  fontSize: "0.62rem",
  fontWeight: 600,
  letterSpacing: "0.14em",
  textTransform: "uppercase",
  cursor: "pointer",
};

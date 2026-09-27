"use client";

import { colors, fonts, motion } from "@hq/visualization";
import { useEffect, useRef, useState, type CSSProperties } from "react";
import type { FetchLike } from "../../providers/fetch";
import { useBundle } from "../data";
import { STATION_DAY_DIALOG_ID } from "./StationDayButton";
import { loadStationDayManifest, stationDayImageUrl, type StationDayLegendEntry, type StationDayManifest } from "./manifest";
import { closeStationDay, fittingManifest, setStationDayManifest, setStationDayRun, useStationDay } from "./store";

const TITLE_ID = "station-day-title";
const CAPTION_ID = "station-day-caption";

const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

const COUNT = new Intl.NumberFormat("en-US");

/**
 * The station-day panel (WEB-10), mounted by the scene next to the tour. Loads the manifest once (no
 * manifest, no button), then renders the dialog while the corner button has it open. While open, the
 * dialog owns the keyboard: Esc closes it and nothing else (not the drawer, not Run details), Tab stays
 * inside, and the presenter keys (Space, E, S, T, G, H, ...) do not act on the scene behind it.
 */
export function StationDayPanel({ fetchImpl }: { fetchImpl?: FetchLike }) {
  const manifest = useStationDay(fittingManifest);
  const open = useStationDay((s) => s.open);

  useEffect(() => {
    let live = true;
    void loadStationDayManifest(fetchImpl).then((m) => {
      if (live) setStationDayManifest(m);
    });
    return () => {
      live = false;
    };
  }, [fetchImpl]);

  useModalKeys();

  if (!open || manifest === null) return null;
  return <StationDayDialog manifest={manifest} />;
}

/**
 * Tells the station-day store which run the scene has loaded, so the button and panel appear only for the
 * picture's own run (not in Today, whose run the picture doesn't show). Mounted next to the panel.
 */
export function StationDayRunSync() {
  const bundle = useBundle();
  const runId = bundle.status === "ready" ? (bundle.meta.scene.runId ?? null) : null;
  useEffect(() => {
    setStationDayRun(runId);
  }, [runId]);
  return null;
}

/**
 * Registered on mount, on window in the capture phase: mounted before the tour's listener (Canvas.tsx),
 * it runs first, and while the panel is open it stops every key from reaching the scene and the shell.
 * Default actions still happen (Enter or Space on a focused button, arrow-key scrolling of the image).
 */
function useModalKeys(): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const state = useStationDay.getState();
      if (!state.open || fittingManifest(state) === null) return;
      const dialog = document.getElementById(STATION_DAY_DIALOG_ID);
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopImmediatePropagation();
        closeStationDay();
        return;
      }
      if (event.key === "Tab" && dialog) {
        trapTab(event, dialog);
        return;
      }
      // Browser and OS shortcuts pass; the scene's and the shell's presenter keys do not.
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      event.stopImmediatePropagation();
    };
    window.addEventListener("keydown", onKeyDown, { capture: true });
    return () => window.removeEventListener("keydown", onKeyDown, { capture: true });
  }, []);
}

/** Keeps Tab and Shift+Tab cycling through the dialog's own controls. */
function trapTab(event: KeyboardEvent, dialog: HTMLElement): void {
  const items = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => !el.hasAttribute("disabled"));
  if (items.length === 0) {
    event.preventDefault();
    dialog.focus();
    return;
  }
  const first = items[0];
  const last = items[items.length - 1];
  const active = document.activeElement;
  const inside = active instanceof Node && dialog.contains(active);
  if (event.shiftKey && (!inside || active === first || active === dialog)) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && (!inside || active === last)) {
    event.preventDefault();
    first.focus();
  }
}

function StationDayDialog({ manifest }: { manifest: StationDayManifest }) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const downOnBackdrop = useRef(false);
  const [fullSize, setFullSize] = useState(false);

  useEffect(() => {
    dialogRef.current?.focus();
  }, []);

  const src = stationDayImageUrl(manifest);
  const [lo, hi] = manifest.filterHz;
  return (
    <div
      className="hqsd-backdrop"
      data-testid="station-day-backdrop"
      onPointerDown={(event) => {
        downOnBackdrop.current = event.target === event.currentTarget;
      }}
      onClick={(event) => {
        // Only a click that starts and ends on the backdrop closes (not a drag out of the caption).
        if (event.target === event.currentTarget && downOnBackdrop.current) closeStationDay();
        downOnBackdrop.current = false;
      }}
    >
      <style>{PANEL_CSS}</style>
      <div
        ref={dialogRef}
        id={STATION_DAY_DIALOG_ID}
        className="hqsd"
        role="dialog"
        aria-modal="true"
        aria-labelledby={TITLE_ID}
        aria-describedby={CAPTION_ID}
        tabIndex={-1}
        data-testid="station-day-dialog"
      >
        <header className="hqsd-head">
          <div className="hqsd-headtext">
            <h2 id={TITLE_ID} className="hqsd-title">
              {manifest.title}
            </h2>
            <p className="hqsd-meta hqsd-num" data-testid="station-day-meta">
              {manifest.seedId} · {manifest.dayUtc} UTC · {lo}–{hi} Hz
            </p>
          </div>
          <button
            type="button"
            className="hqsd-btn"
            aria-pressed={fullSize}
            onClick={() => setFullSize((v) => !v)}
            data-testid="station-day-zoom"
          >
            {fullSize ? "Fit to screen" : "Full size"}
          </button>
          <button type="button" className="hqsd-btn" onClick={closeStationDay} data-testid="station-day-close">
            Close
          </button>
        </header>

        <div className="hqsd-figure" data-full={fullSize || undefined}>
          {/* The PNG is requested only now, when the panel opens. Its size comes from the manifest. */}
          {/* eslint-disable-next-line @next/next/no-img-element -- a static, pre-sized PNG; next/image adds nothing here */}
          <img
            className="hqsd-img"
            src={src}
            width={manifest.widthPx}
            height={manifest.heightPx}
            alt={`${manifest.title}: ${manifest.seedId}, ${manifest.dayUtc} UTC`}
            decoding="async"
            data-testid="station-day-image"
            onClick={() => setFullSize((v) => !v)}
          />
        </div>

        <p id={CAPTION_ID} className="hqsd-caption" data-testid="station-day-caption">
          {manifest.caption}
        </p>

        {manifest.legend.length > 0 && (
          <ul className="hqsd-legend" aria-label="Markers" data-testid="station-day-legend">
            {manifest.legend.map((entry) => (
              <li key={entry.key} className="hqsd-item" data-key={entry.key}>
                <Swatch entry={entry} />
                <span>{entry.label}</span>
                <span className="hqsd-num hqsd-count">{COUNT.format(entry.count)}</span>
              </li>
            ))}
          </ul>
        )}

        <p className="hqsd-foot">
          {manifest.selection.rule} ·{" "}
          <a className="hqsd-link" href={src} target="_blank" rel="noopener noreferrer">
            Open image
          </a>
        </p>
      </div>
    </div>
  );
}

/** The marker as drawn in the picture: a vertical tick or a hollow diamond, in its color and opacity. */
function Swatch({ entry }: { entry: StationDayLegendEntry }) {
  const style: CSSProperties =
    entry.shape === "tick"
      ? { width: 2, height: 14, background: entry.color, opacity: entry.opacity }
      : {
          width: 8,
          height: 8,
          border: `1.5px solid ${entry.color}`,
          transform: "rotate(45deg)",
          opacity: entry.opacity,
        };
  return (
    <span className="hqsd-swatch" aria-hidden="true" data-shape={entry.shape}>
      <span style={{ display: "block", boxSizing: "border-box", ...style }} />
    </span>
  );
}

/** Tokens only (packages/visualization/tokens.ts), in the drawer's idiom. Text is 13 px or larger. */
const PANEL_CSS = `
.hqsd-backdrop {
  position: fixed; inset: 0; z-index: 40; display: grid; place-items: center;
  padding: 16px; box-sizing: border-box; background: ${colors.bg}CC;
}
.hqsd {
  box-sizing: border-box; width: min(calc(1920px + 40px), 100%); max-height: 100%; overflow: auto;
  overscroll-behavior: contain; padding: 14px 20px 16px; border: 1px solid ${colors.contour}; border-radius: 8px;
  background: ${colors.surface}; color: ${colors.text}; font-family: ${fonts.ui}; font-size: 14px; line-height: 1.4;
  box-shadow: 0 12px 40px ${colors.bg}B3; outline: none; user-select: text;
}
.hqsd *, .hqsd *::before, .hqsd *::after { box-sizing: border-box; }
.hqsd-num { font-family: ${fonts.mono}; font-variant-numeric: tabular-nums; }
.hqsd-head { display: flex; align-items: flex-start; gap: 8px; }
.hqsd-headtext { flex: 1 1 auto; min-width: 0; }
.hqsd-title { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: -0.005em; }
.hqsd-meta { margin: 2px 0 0; font-size: 13px; color: ${colors.textDim}; }
.hqsd-btn {
  flex: none; padding: 4px 12px; border: 1px solid ${colors.contour}; border-radius: 999px; cursor: pointer;
  background: transparent; color: ${colors.textDim}; font: 600 13px/1.3 ${fonts.ui};
  transition: color ${motion.micro}ms ${motion.ease}, border-color ${motion.micro}ms ${motion.ease};
}
.hqsd-btn:hover { color: ${colors.text}; border-color: ${colors.textDim}; }
.hqsd-btn:focus-visible, .hqsd-link:focus-visible { outline: 2px solid ${colors.text}; outline-offset: 2px; }
.hqsd-figure {
  margin-top: 12px; max-height: max(240px, calc(100dvh - 250px)); overflow: auto;
  background: ${colors.bg}; border: 1px solid ${colors.contour}; border-radius: 4px;
}
.hqsd-img {
  display: block; margin: 0 auto; width: auto; height: auto; max-width: 100%;
  max-height: max(238px, calc(100dvh - 252px)); cursor: zoom-in;
}
.hqsd-figure[data-full] .hqsd-img { max-width: none; max-height: none; cursor: zoom-out; }
.hqsd-caption { margin: 12px 0 0; max-width: 110ch; font-size: 14px; color: ${colors.text}; }
.hqsd-legend {
  display: flex; flex-wrap: wrap; gap: 6px 20px; margin: 10px 0 0; padding: 0; list-style: none;
  font-size: 13px; color: ${colors.textDim};
}
.hqsd-item { display: inline-flex; align-items: center; gap: 7px; }
.hqsd-swatch { display: inline-grid; place-items: center; width: 14px; height: 14px; }
.hqsd-count { color: ${colors.text}; }
.hqsd-foot { margin: 10px 0 0; font-size: 13px; color: ${colors.textDim}; }
.hqsd-link { color: ${colors.textDim}; text-decoration: underline; text-underline-offset: 2px; }
.hqsd-link:hover { color: ${colors.text}; }
`;

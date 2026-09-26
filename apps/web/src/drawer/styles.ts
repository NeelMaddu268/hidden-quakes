// The drawer's stylesheet, built from the design tokens (packages/visualization/tokens.ts) so the panel
// and the canvas stay one palette. Class names are prefixed `hqd-`. One static string, rendered once
// inside the drawer; motion (slide, pick ticks) is CSS, so nothing re-renders per animation frame.

import { colors, fonts, motion } from "@hq/visualization";

/** Record-section columns: station + channel, distance, trace track. */
const RECORD_COLUMNS = "90px 58px minmax(0, 1fr)";

/** SVG viewBox of the two figures (plan view, depth section). */
export const FIGURE_W = 260;
export const FIGURE_H = 150;

/** Drawer width: 40% of the viewport, kept readable on small and very large screens. */
export const DRAWER_WIDTH_CSS = "clamp(420px, 40vw, 720px)";

export const DRAWER_CSS = `
.hqd {
  position: fixed; top: 0; right: 0; bottom: 0; z-index: 30;
  width: ${DRAWER_WIDTH_CSS}; max-width: 100vw; box-sizing: border-box;
  display: flex; flex-direction: column; overflow-y: auto; overscroll-behavior: contain;
  background: ${colors.surface}; color: ${colors.text}; border-left: 1px solid ${colors.contour};
  font-family: ${fonts.ui}; font-size: 13px; line-height: 1.35;
  transform: translateX(100%); visibility: hidden; pointer-events: none;
  transition: transform ${motion.state}ms ${motion.ease}, visibility 0s linear ${motion.state}ms;
}
.hqd[data-open="true"] {
  transform: none; visibility: visible; pointer-events: auto;
  transition: transform ${motion.state}ms ${motion.ease}, visibility 0s;
}
.hqd *, .hqd *::before, .hqd *::after { box-sizing: border-box; }
.hqd-num { font-family: ${fonts.mono}; font-variant-numeric: tabular-nums; }
.hqd-dim { color: ${colors.textDim}; }
.hqd-label {
  color: ${colors.textDim}; font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase;
}
.hqd-section { padding: 12px 20px; border-top: 1px solid ${colors.contour}; }

.hqd-head { padding: 12px 20px 12px; }
.hqd-topline { display: flex; align-items: center; gap: 10px; min-height: 28px; }
.hqd-id { color: ${colors.textDim}; font-size: 11.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hqd-synth {
  color: ${colors.alert}; border: 1px solid ${colors.alert}; border-radius: 3px;
  padding: 1px 5px; font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase;
}
.hqd-close {
  margin-left: auto; width: 28px; height: 28px; flex: none; border-radius: 4px; cursor: pointer;
  border: 1px solid transparent; background: transparent; color: ${colors.textDim};
  font: 18px/1 ${fonts.ui}; display: grid; place-items: center;
  transition: color ${motion.micro}ms ${motion.ease}, border-color ${motion.micro}ms ${motion.ease};
}
.hqd-close:hover { color: ${colors.text}; border-color: ${colors.contour}; }
.hqd-close:focus-visible { outline: 1px solid ${colors.textDim}; outline-offset: 1px; }
.hqd-title { margin: 4px 0 0; font-size: 21px; font-weight: 600; letter-spacing: -0.01em; }
.hqd-tierline { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 10px; margin-top: 6px; }
.hqd-tier {
  font-size: 11px; font-weight: 600; letter-spacing: 0.04em; padding: 1px 6px; border-radius: 3px;
  border: 1px solid ${colors.contour}; color: ${colors.text};
}
.hqd-tier[data-tier="A"] { border-color: ${colors.strictHalo}; color: ${colors.strictHalo}; }
.hqd-reasons { color: ${colors.textDim}; font-size: 11px; }
.hqd-reasons span + span::before { content: " · "; }
.hqd-origin { display: grid; grid-template-columns: 76px 1fr; gap: 1px 8px; margin-top: 10px; font-size: 12.5px; }
.hqd-depth { display: flex; flex-wrap: wrap; gap: 8px 24px; margin-top: 10px; }
.hqd-stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(62px, 1fr)); gap: 8px 10px;
  margin-top: 10px; padding-top: 10px; border-top: 1px solid ${colors.contour};
}
.hqd-stat { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.hqd-stat-value { font-size: 12.5px; white-space: nowrap; }
.hqd-extra { display: grid; grid-template-columns: 76px 1fr; gap: 4px 8px; margin-top: 10px; font-size: 12.5px; }
.hqd-note { margin-top: 8px; color: ${colors.textDim}; font-size: 11.5px; }

.hqd-sechead { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.hqd-caption { color: ${colors.textDim}; font-size: 11.5px; margin-top: 3px; }
.hqd-legend { display: flex; gap: 12px; color: ${colors.textDim}; font-size: 11px; align-items: center; }
.hqd-legend i { display: inline-block; width: 2px; height: 11px; margin-right: 5px; vertical-align: -1px; }
.hqd-legend i.p { background: ${colors.pickP}; }
.hqd-legend i.s { background: ${colors.pickS}; }
.hqd-legend i.pred { width: 0; border-left: 1px dashed ${colors.textDim}; }

.hqd-record {
  display: grid; grid-template-columns: ${RECORD_COLUMNS}; grid-auto-rows: minmax(12px, 1fr);
  height: clamp(160px, 29vh, 420px); margin: 8px 0 0; padding: 0; list-style: none; position: relative;
}
.hqd-row { display: contents; }
.hqd-sta, .hqd-dist {
  align-self: center; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.hqd-sta { grid-column: 1; color: ${colors.text}; }
.hqd-dist { grid-column: 2; color: ${colors.textDim}; text-align: right; padding-right: 8px; }
.hqd-track { position: relative; grid-column: 3; min-height: 0; }
.hqd-track svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
.hqd-trace { fill: none; stroke: ${colors.text}; stroke-opacity: 0.72; stroke-width: 1; stroke-linejoin: round; }
.hqd-grid { grid-column: 3; grid-row: 1 / -1; position: relative; pointer-events: none; }
.hqd-grid span { position: absolute; top: 0; bottom: 0; width: 0; border-left: 1px solid ${colors.contour}; }
.hqd-grid span.zero { border-left-color: ${colors.textDim}; opacity: 0.45; }
.hqd-mark { position: absolute; top: 6%; bottom: 6%; width: 0; }
.hqd-pred { top: 0; bottom: 0; border-left: 1px dashed; opacity: 0.5; }
.hqd-pred.p { border-color: ${colors.pickP}; }
.hqd-pred.s { border-color: ${colors.pickS}; }
.hqd-pick { width: 2px; margin-left: -1px; border-radius: 1px; transform-origin: 50% 50%;
  animation: hqd-pick-in ${motion.micro}ms ${motion.ease} both; }
.hqd-pick.p { background: ${colors.pickP}; }
.hqd-pick.s { background: ${colors.pickS}; }
@keyframes hqd-pick-in { from { opacity: 0; transform: scaleY(0.15); } to { opacity: 1; transform: none; } }
.hqd-axis { display: grid; grid-template-columns: ${RECORD_COLUMNS}; height: 16px; margin-top: 3px; }
.hqd-axis-track { grid-column: 3; position: relative; font-size: 10.5px; color: ${colors.textDim}; }
.hqd-axis-track span { position: absolute; top: 2px; transform: translateX(-50%); white-space: nowrap; }
.hqd-axis-title { text-align: right; color: ${colors.textDim}; font-size: 10.5px; margin-top: 1px; }
.hqd-placeholder { grid-column: 1 / -1; border-top: 1px solid ${colors.contour}; opacity: 0.6; }
.hqd-status { grid-column: 1 / -1; align-self: center; justify-self: center; color: ${colors.textDim}; font-size: 12px; }
.hqd-error { color: ${colors.textDim}; font-size: 12px; margin-top: 10px; }
.hqd-error b { color: ${colors.alert}; font-weight: 600; }

.hqd-figs { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.hqd-fig svg { display: block; width: 100%; height: auto; margin-top: 4px; overflow: visible; }
.hqd-fig-caption { color: ${colors.textDim}; font-size: 10.5px; margin-top: 4px; }
.hqd-fig-empty { aspect-ratio: ${FIGURE_W} / ${FIGURE_H}; margin-top: 4px; border: 1px dashed ${colors.contour}; border-radius: 3px; }
.hqd-svg-text { font-family: ${fonts.mono}; font-variant-numeric: tabular-nums; font-size: 10px; fill: ${colors.textDim}; }

@media (prefers-reduced-motion: reduce) {
  .hqd, .hqd[data-open="true"] { transition: none; }
  .hqd-pick { animation: none; }
}
`;

"use client";

import { useMemo } from "react";
import type { SweepPoint } from "@/providers";
import { formatNumber } from "../validation/format";
import styles from "./RunDetails.module.css";
import { sweepLayout, type SweepField, type SweepLayout } from "./sweep";

/** Series colour by meaning: candidates wear the recovered accent, Tier A the pick-P blue,
 *  recovered public events the public white. Legend and end labels carry identity too. */
const SERIES_COLOR: Record<SweepField, string> = {
  candidates: "var(--hq-recovered)",
  tierA: "var(--hq-pick-p)",
  recoveredPublic: "var(--hq-public)",
};

// Plot geometry in viewBox units; the SVG scales to the panel width.
const WIDTH = 600;
const HEIGHT = 280;
const MARGIN = { top: 16, right: 120, bottom: 48, left: 60 };
const INNER_W = WIDTH - MARGIN.left - MARGIN.right;
const INNER_H = HEIGHT - MARGIN.top - MARGIN.bottom;
const Y_TICK_TARGET = 4;
const MARK_RADIUS = 4;
const TICK_DECIMALS = 2;
const TICK_LENGTH = 5;
const LABEL_GAP = 8;

function scales(layout: SweepLayout) {
  const xMin = layout.xTicks[0];
  const xMax = layout.xTicks[layout.xTicks.length - 1];
  const sx = (x: number) => (xMax === xMin ? MARGIN.left + INNER_W / 2 : MARGIN.left + ((x - xMin) / (xMax - xMin)) * INNER_W);
  const sy = (y: number) => (layout.yMax > 0 ? MARGIN.top + INNER_H - (y / layout.yMax) * INNER_H : MARGIN.top + INNER_H);
  return { sx, sy };
}

/**
 * `validation.sweep` as an inline SVG: counts against the swept parameter, one mark per
 * `SweepPoint` and series, lines through points that share every other parameter. Renders
 * nothing when the sweep is empty or has no numeric parameter.
 */
export function SweepPlot({ sweep }: { sweep: readonly SweepPoint[] }) {
  const layout = useMemo(() => sweepLayout(sweep, Y_TICK_TARGET), [sweep]);
  if (!layout) return null;
  const { sx, sy } = scales(layout);
  const baseline = MARGIN.top + INNER_H;

  return (
    <figure className={styles.sweep} data-testid="sweep-plot">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Association sweep" className={styles.sweepSvg}>
        <g className={styles.grid}>
          {layout.yTicks.map((tick) => (
            <line key={tick} x1={MARGIN.left} x2={MARGIN.left + INNER_W} y1={sy(tick)} y2={sy(tick)} />
          ))}
        </g>
        <g className={styles.axis}>
          <line x1={MARGIN.left} x2={MARGIN.left + INNER_W} y1={baseline} y2={baseline} />
          {layout.xTicks.map((tick) => (
            <g key={tick} transform={`translate(${sx(tick)} ${baseline})`}>
              <line y2={TICK_LENGTH} />
              <text y={TICK_LENGTH + LABEL_GAP} dominantBaseline="hanging" textAnchor="middle">
                {formatNumber(tick, TICK_DECIMALS)}
              </text>
            </g>
          ))}
          {layout.yTicks.map((tick) => (
            <text key={tick} x={MARGIN.left - LABEL_GAP} y={sy(tick)} dominantBaseline="middle" textAnchor="end">
              {formatNumber(tick, TICK_DECIMALS)}
            </text>
          ))}
          <text
            className={styles.axisTitle}
            x={MARGIN.left + INNER_W / 2}
            y={HEIGHT - LABEL_GAP / 2}
            textAnchor="middle"
            data-testid="sweep-x-label"
          >
            {layout.xKey}
          </text>
          <text
            className={styles.axisTitle}
            transform={`translate(${LABEL_GAP * 1.5} ${MARGIN.top + INNER_H / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            {layout.series.map((s) => s.field).join(" · ")}
          </text>
        </g>
        {layout.series.map((series) => (
          <g key={series.field} data-series={series.field} style={{ color: SERIES_COLOR[series.field] }}>
            {series.lines.map((line) => (
              <polyline
                key={line.map((m) => m.index).join("-")}
                className={styles.line}
                points={line.map((m) => `${sx(m.x)},${sy(m.y)}`).join(" ")}
              />
            ))}
            {series.lines.map((line) => {
              const last = line[line.length - 1];
              return (
                <text
                  key={`label-${last.index}`}
                  className={styles.endLabel}
                  x={sx(last.x) + LABEL_GAP}
                  y={sy(last.y)}
                  dominantBaseline="middle"
                >
                  {series.field}
                </text>
              );
            })}
            {series.marks.map((mark) => (
              <circle key={mark.index} className={styles.mark} data-mark cx={sx(mark.x)} cy={sy(mark.y)} r={MARK_RADIUS}>
                <title>
                  {[`${layout.xKey}=${formatNumber(mark.x, TICK_DECIMALS)}`, mark.group, `${series.field}=${formatNumber(mark.y, 0)}`]
                    .filter(Boolean)
                    .join(" · ")}
                </title>
              </circle>
            ))}
          </g>
        ))}
      </svg>
      <figcaption className={styles.legend}>
        {layout.series.map((series) => (
          <span key={series.field} className={styles.legendItem}>
            <span className={styles.swatch} style={{ background: SERIES_COLOR[series.field] }} />
            {series.field}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}

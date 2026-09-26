import { colors } from "@hq/visualization";
import { useMemo } from "react";
import type { BundleMeta, SeismicEvent, Station } from "../scene/types";
import { fmtFixed, fmtLength, isNum } from "./format";
import { depthGeometry, depthKmOfElev, eventDepthPoint, mapGeometry, type Box } from "./geometry";
import { FIGURE_H, FIGURE_W } from "./styles";

const MAP_BOX: Box = { width: FIGURE_W, height: FIGURE_H, pad: 16 };
/** The depth section leaves room on the left for depth labels. */
const DEPTH_LABEL_W = 26;
const DEPTH_BOX: Box = { width: FIGURE_W - DEPTH_LABEL_W, height: FIGURE_H, pad: 12 };

const r1 = (v: number) => Math.round(v * 10) / 10;

/** Inverted triangle, the scene's station glyph. */
function triangle(x: number, y: number, s = 4): string {
  return `M${r1(x - s)} ${r1(y - s * 0.75)}L${r1(x + s)} ${r1(y - s * 0.75)}L${r1(x)} ${r1(y + s)}Z`;
}

function isSurface(st: Station): boolean {
  return st.kind !== "borehole";
}

export interface FiguresProps {
  event: SeismicEvent;
  meta: BundleMeta;
  /** Stations of the evidence traces (null while the evidence loads). */
  stations: Station[] | null;
  /** Evidence station ids the bundle has no station record for. */
  missing: string[];
  /** Where `stations` came from: the evidence traces, or the event's picks when it has no evidence file. */
  source?: "evidence" | "picks";
}

/** Plan view (north up) with station-to-epicenter lines, and the depth section beside it. */
export function Figures({ event, meta, stations, missing, source = "evidence" }: FiguresProps) {
  return (
    <section className="hqd-section" aria-label="Station geometry">
      <div className="hqd-figs">
        <div className="hqd-fig">
          <span className="hqd-label">Plan view</span>
          {stations ? <MiniMap event={event} stations={stations} /> : <div className="hqd-fig-empty" />}
          <div className="hqd-fig-caption">
            Grid north up · {source === "picks" ? "picking station" : "station"} → epicenter lines
          </div>
        </div>
        <div className="hqd-fig">
          <span className="hqd-label">Depth section · looking north</span>
          {stations ? <DepthSection event={event} meta={meta} stations={stations} /> : <div className="hqd-fig-empty" />}
          <div className="hqd-fig-caption">{meta.scene.depthLabel}</div>
        </div>
      </div>
      {missing.length > 0 && (
        <div className="hqd-note">
          Not in the bundle&apos;s station list: <span className="hqd-num">{missing.join(", ")}</span>
        </div>
      )}
    </section>
  );
}

function MiniMap({ event, stations }: { event: SeismicEvent; stations: Station[] }) {
  const hErrM = event.quality.hErrM;
  const g = useMemo(() => mapGeometry(event.enu, stations.map((s) => s.enu), hErrM, MAP_BOX), [event.enu, stations, hErrM]);
  const ex = g.x(event.enu.e);
  const ey = g.y(event.enu.n);
  const errR = isNum(hErrM) && hErrM > 0 ? hErrM * g.pxPerM : 0;
  const { width: W, height: H } = MAP_BOX;
  const barY = H - 6;
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Plan view: epicenter and ${stations.length} ${stations.length === 1 ? "station" : "stations"}`}
    >
      {stations.map((st) => (
        <line
          key={`l-${st.id}`}
          x1={r1(g.x(st.enu.e))}
          y1={r1(g.y(st.enu.n))}
          x2={r1(ex)}
          y2={r1(ey)}
          stroke={colors.textDim}
          strokeOpacity={0.4}
          strokeWidth={0.75}
        />
      ))}
      {errR > 0 && (
        <circle cx={r1(ex)} cy={r1(ey)} r={r1(errR)} fill="none" stroke={colors.recovered} strokeOpacity={0.55} strokeDasharray="2 2" strokeWidth={0.75}>
          <title>{`One-sigma horizontal error ±${fmtFixed(hErrM, 0)} m`}</title>
        </circle>
      )}
      {stations.map((st) => (
        <path
          key={`s-${st.id}`}
          d={triangle(g.x(st.enu.e), g.y(st.enu.n))}
          fill={isSurface(st) ? colors.station : colors.surface}
          stroke={colors.station}
          strokeWidth={1}
        >
          <title>{`${st.id} (${st.kind})`}</title>
        </path>
      ))}
      <circle cx={r1(ex)} cy={r1(ey)} r={6.5} fill="none" stroke={colors.strictHalo} strokeWidth={1} />
      <circle cx={r1(ex)} cy={r1(ey)} r={3.5} fill={colors.recovered}>
        <title>Epicenter</title>
      </circle>
      {/* North arrow */}
      <g transform={`translate(${W - 10} 14)`} aria-hidden="true">
        <path d="M0 -8L3.5 0L0 -2L-3.5 0Z" fill={colors.textDim} />
        <text className="hqd-svg-text" x={0} y={10} textAnchor="middle">
          N
        </text>
      </g>
      {/* Scale bar: a round length derived from the extent */}
      {g.scaleBarPx > 0 && (
        <g aria-hidden="true">
          <path
            d={`M2 ${barY - 3}V${barY}H${r1(2 + g.scaleBarPx)}V${barY - 3}`}
            fill="none"
            stroke={colors.textDim}
            strokeWidth={1}
          />
          <text className="hqd-svg-text" x={r1(2 + g.scaleBarPx + 5)} y={barY} dominantBaseline="middle">
            {fmtLength(g.scaleBarM)}
          </text>
        </g>
      )}
    </svg>
  );
}

function DepthSection({ event, meta, stations }: { event: SeismicEvent; meta: BundleMeta; stations: Station[] }) {
  const vErrM = event.quality.vErrM;
  const vErrKm = isNum(vErrM) && vErrM > 0 ? vErrM / 1000 : 0;
  const scene = meta.scene;
  const layout = useMemo(() => {
    const ev = eventDepthPoint(event, scene);
    const sensors = stations.map((st) => ({
      st,
      e: st.enu.e,
      depthKm: depthKmOfElev(st.sensorElevM, scene),
      wellheadKm: depthKmOfElev(st.surfaceElevM, scene),
    }));
    const pts = [ev, ...sensors.map((s) => ({ e: s.e, depthKm: s.depthKm })), ...sensors.map((s) => ({ e: s.e, depthKm: s.wellheadKm }))];
    return { ev, sensors, g: depthGeometry(pts, vErrKm, DEPTH_BOX) };
  }, [event, stations, scene, vErrKm]);
  const { ev, sensors, g } = layout;
  const { width: W, height: H, pad } = DEPTH_BOX;
  const ex = g.x(ev.e);
  const ey = g.y(ev.depthKm);
  const ve = g.verticalExaggeration;
  return (
    <svg
      viewBox={`0 0 ${W + DEPTH_LABEL_W} ${H}`}
      role="img"
      aria-label={`Depth section: event at ${fmtFixed(ev.depthKm, 2)} km below the site surface`}
    >
      <g aria-hidden="true">
        {g.ticks.map((d) => (
          <g key={d}>
            <line
              x1={DEPTH_LABEL_W}
              x2={DEPTH_LABEL_W + W - pad / 2}
              y1={r1(g.y(d))}
              y2={r1(g.y(d))}
              stroke={colors.contour}
              strokeWidth={1}
              strokeDasharray={d === 0 ? "3 3" : undefined}
            />
            <text className="hqd-svg-text" x={DEPTH_LABEL_W - 4} y={r1(g.y(d))} textAnchor="end" dominantBaseline="middle">
              {fmtFixed(d, d % 1 === 0 ? 0 : 1)}
            </text>
          </g>
        ))}
        <text className="hqd-svg-text" x={DEPTH_LABEL_W - 4} y={6} textAnchor="end">
          km
        </text>
        {ve !== 1 && (
          <text className="hqd-svg-text" x={DEPTH_LABEL_W + W - 2} y={6} textAnchor="end">
            {`Vertical ×${ve}`}
          </text>
        )}
      </g>
      <g transform={`translate(${DEPTH_LABEL_W} 0)`}>
        {sensors.map(({ st, e, depthKm, wellheadKm }) =>
          isSurface(st) ? (
            <path key={st.id} d={triangle(g.x(e), g.y(depthKm))} fill={colors.station}>
              <title>{`${st.id} (${st.kind})`}</title>
            </path>
          ) : (
            <g key={st.id}>
              <line x1={r1(g.x(e))} x2={r1(g.x(e))} y1={r1(g.y(wellheadKm))} y2={r1(g.y(depthKm))} stroke={colors.station} strokeWidth={0.75} />
              <rect x={r1(g.x(e) - 2.5)} y={r1(g.y(depthKm) - 2.5)} width={5} height={5} fill={colors.station}>
                <title>{`${st.id} borehole sensor, ${fmtFixed(depthKm, 2)} km`}</title>
              </rect>
            </g>
          ),
        )}
        {vErrKm > 0 && (
          <path
            d={`M${r1(ex - 3)} ${r1(g.y(ev.depthKm - vErrKm))}H${r1(ex + 3)}M${r1(ex)} ${r1(g.y(ev.depthKm - vErrKm))}V${r1(g.y(ev.depthKm + vErrKm))}M${r1(ex - 3)} ${r1(g.y(ev.depthKm + vErrKm))}H${r1(ex + 3)}`}
            stroke={colors.recovered}
            strokeOpacity={0.7}
            strokeWidth={1}
            fill="none"
          >
            <title>{`One-sigma vertical error ±${fmtFixed(vErrM, 0)} m`}</title>
          </path>
        )}
        <circle cx={r1(ex)} cy={r1(ey)} r={6.5} fill="none" stroke={colors.strictHalo} strokeWidth={1} />
        <circle cx={r1(ex)} cy={r1(ey)} r={3.5} fill={colors.recovered}>
          <title>{`Candidate event, ${fmtFixed(ev.depthKm, 2)} km`}</title>
        </circle>
      </g>
    </svg>
  );
}

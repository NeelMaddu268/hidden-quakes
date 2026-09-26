import type { Confidence } from "../providers/confidence";
import type { BundleMeta, SceneMeta, SeismicEvent } from "../scene/types";
import { depthKmOfElev } from "./geometry";
import { DASH, fmtFixed, fmtMagnitude, fmtPlusMinusM, fmtUnit, fmtUtahLocal, fmtUtc, isNum } from "./format";

export interface HeaderProps {
  /** Selected event id (shown even before the bundle is ready). */
  eventId: string;
  event: SeismicEvent | null;
  meta: BundleMeta | null;
  /** The bundle's `confidence.json` (ML-01), or null when it has none. */
  confidence?: Confidence | null;
  onClose: () => void;
}

/** Shown when `confidence.json` scores the event but carries no label of its own. */
const CONFIDENCE_FALLBACK_LABEL = "Decoy test score";

function Stat({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="hqd-stat" title={title}>
      <span className="hqd-label">{label}</span>
      <span className="hqd-stat-value hqd-num">{value}</span>
    </div>
  );
}

/**
 * Drawer header (docs/lanes/H3 → Drawer): "{nStations} stations agreed", tier, rms, ±h / ±v, depth,
 * origin time in UTC and Utah local, magnitude with its type, and the matched public-catalog id. Every
 * number comes from the event record; missing ones render as a dash.
 */
export function Header({ eventId, event, meta, confidence = null, onClose }: HeaderProps) {
  const synthetic = Boolean(meta?.scene.isSynthetic || meta?.run.isSynthetic);
  return (
    <header className="hqd-head">
      <div className="hqd-topline">
        <span className="hqd-label">Candidate event</span>
        <span className="hqd-id hqd-num" title={eventId}>
          {eventId}
        </span>
        {synthetic && <span className="hqd-synth">Synthetic</span>}
        <button type="button" className="hqd-close" onClick={onClose} aria-label="Close evidence drawer">
          ×
        </button>
      </div>
      {event && meta ? <EventSummary event={event} scene={meta.scene} confidence={confidence} /> : null}
    </header>
  );
}

function EventSummary({ event, scene, confidence }: { event: SeismicEvent; scene: SceneMeta; confidence: Confidence | null }) {
  const q = event.quality;
  // ML-01's decoy-test score, beside the depth so the record section does not move down. A score in
  // [0, 1], not a probability that the event is an earthquake; the file's own sentence is the tooltip.
  const score = confidence?.score(event.id) ?? null;
  const n = q.nStations;
  const mag = fmtMagnitude(event.magnitude);
  const match = event.catalogMatch;
  return (
    <>
      <h2 className="hqd-title">
        <span className="hqd-num">{isNum(n) ? fmtFixed(n, 0) : DASH}</span> {n === 1 ? "station" : "stations"} agreed
      </h2>
      <div className="hqd-tierline">
        <span className="hqd-tier" data-tier={event.tier}>
          Tier {event.tier}
        </span>
        {event.tierReasons.length > 0 && (
          <span className="hqd-reasons hqd-num">
            {event.tierReasons.map((r, i) => (
              <span key={i}>{r}</span>
            ))}
          </span>
        )}
      </div>
      <div className="hqd-origin">
        <span className="hqd-label">Origin</span>
        <span className="hqd-num">{fmtUtc(event.t)}</span>
        <span />
        <span className="hqd-num hqd-dim">{fmtUtahLocal(event.t)}</span>
      </div>
      <div className="hqd-depth">
        <Stat label={scene.depthLabel} value={fmtUnit(depthKmOfElev(event.elevM, scene), 2, "km")} />
        {confidence && score !== null && (
          <Stat
            label={confidence.label ?? CONFIDENCE_FALLBACK_LABEL}
            value={fmtFixed(score, 2)}
            title={confidence.description ?? undefined}
          />
        )}
      </div>
      <div className="hqd-stats">
        <Stat label="RMS" value={fmtUnit(q.rmsS, 3, "s")} />
        <Stat label="±h" value={fmtPlusMinusM(q.hErrM)} />
        <Stat label="±v" value={fmtPlusMinusM(q.vErrM)} />
        <Stat label="Picks" value={`${fmtFixed(q.nP, 0)} P · ${fmtFixed(q.nS, 0)} S`} />
        <Stat label="Gap" value={isNum(q.gapDeg) ? `${fmtFixed(q.gapDeg, 0)}°` : DASH} />
      </div>
      {(mag || match) && (
        <div className="hqd-extra">
          {mag && (
            <>
              <span className="hqd-label">Magnitude</span>
              <span className="hqd-num">{mag}</span>
            </>
          )}
          {match && (
            <>
              <span className="hqd-label">Matched</span>
              <span>
                <span className="hqd-num">{match.catalogId}</span>
                <span className="hqd-dim"> in the public regional catalog</span>
                <span className="hqd-dim hqd-num">
                  {" "}
                  · Δt {fmtUnit(match.dtS, 2, "s")} · {fmtUnit(match.distM, 0, "m")}
                </span>
              </span>
            </>
          )}
        </div>
      )}
      {q.depthOnEdge && <div className="hqd-note">Depth solution reaches the edge of the location grid.</div>}
    </>
  );
}

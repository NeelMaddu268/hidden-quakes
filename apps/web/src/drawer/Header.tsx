import type { BundleMeta, SeismicEvent } from "../scene/types";
import { DASH, fmtFixed, fmtMagnitude, fmtPlusMinusM, fmtUnit, fmtUtahLocal, fmtUtc, isNum } from "./format";

export interface HeaderProps {
  /** Selected event id (shown even before the bundle is ready). */
  eventId: string;
  event: SeismicEvent | null;
  meta: BundleMeta | null;
  onClose: () => void;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="hqd-stat">
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
export function Header({ eventId, event, meta, onClose }: HeaderProps) {
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
      {event ? <EventSummary event={event} /> : null}
    </header>
  );
}

function EventSummary({ event }: { event: SeismicEvent }) {
  const q = event.quality;
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
      <div className="hqd-stats">
        <Stat label="Depth" value={fmtUnit(event.depthKm, 2, "km")} />
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

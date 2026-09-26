import { useEffect, useMemo } from "react";
import type { EventEvidence, WaveformSnippet } from "../scene/types";
import { fmtFixed, fmtKmFromM, fmtSeconds, isNum } from "./format";
import {
  evidenceUnavailable,
  markPercent,
  pickDelayMs,
  pickedTraceCount,
  prepareTraces,
  recordDomain,
  timeTicks,
  tracePoints,
  TRACK_H,
  TRACK_W,
  type TimeDomain,
} from "./record";

/** Placeholder rows while the evidence file loads (no motion: nothing loops). */
const PLACEHOLDER_ROWS = 6;

export interface RecordSectionProps {
  status: "idle" | "loading" | "ready" | "error";
  evidence?: EventEvidence;
  message?: string;
  /** Event origin time, epoch s: the axis is seconds after it. */
  originT: number;
}

/** Tooltip for a mark: "P pick 1.23 s after origin · prob 0.87"; only finite numbers are printed. */
function markTitle(kind: string, t: number | null | undefined, originT: number, prob?: number | null): string {
  const when = isNum(t) ? `${fmtFixed(t - originT, 2)} s after origin` : "";
  const p = isNum(prob) ? ` · prob ${fmtFixed(prob, 2)}` : "";
  return `${kind} ${when}${p}`.trim();
}

function Mark({ className, left, delayMs, title }: { className: string; left: number | null; delayMs?: number; title: string }) {
  if (left === null) return null;
  return (
    <span
      className={`hqd-mark ${className}`}
      style={delayMs === undefined ? { left: `${left}%` } : { left: `${left}%`, animationDelay: `${delayMs}ms` }}
      title={title}
    />
  );
}

function TraceRow({ tr, row, originT, domain }: { tr: WaveformSnippet; row: number; originT: number; domain: TimeDomain }) {
  const points = useMemo(() => tracePoints(tr, originT, domain), [tr, originT, domain]);
  const pickP = markPercent(tr.pickP, originT, domain);
  const pickS = markPercent(tr.pickS, originT, domain);
  const predP = markPercent(tr.predP, originT, domain);
  const predS = markPercent(tr.predS, originT, domain);
  const dist = fmtKmFromM(tr.epiDistM);
  return (
    <li className="hqd-row" aria-label={`${tr.stationId} ${tr.channel}, ${dist}`} data-station={tr.stationId}>
      <span className="hqd-sta hqd-num" style={{ gridRow: row + 1 }} title={`${tr.stationId} ${tr.channel}`}>
        {tr.stationId} <span className="hqd-dim">{tr.channel}</span>
      </span>
      <span className="hqd-dist hqd-num" style={{ gridRow: row + 1 }}>
        {dist}
      </span>
      <div className="hqd-track" style={{ gridRow: row + 1 }}>
        <svg viewBox={`0 0 ${TRACK_W} ${TRACK_H}`} preserveAspectRatio="none" aria-hidden="true">
          <polyline className="hqd-trace" points={points} vectorEffect="non-scaling-stroke" />
        </svg>
        <Mark className="hqd-pred p" left={predP} title={markTitle("Modeled P arrival", tr.predP, originT)} />
        <Mark className="hqd-pred s" left={predS} title={markTitle("Modeled S arrival", tr.predS, originT)} />
        <Mark className="hqd-pick p" left={pickP} delayMs={pickDelayMs(row, "P")} title={markTitle("P pick", tr.pickP, originT, tr.probP)} />
        <Mark className="hqd-pick s" left={pickS} delayMs={pickDelayMs(row, "S")} title={markTitle("S pick", tr.pickS, originT, tr.probS)} />
      </div>
    </li>
  );
}

function Legend() {
  return (
    <div className="hqd-legend" aria-hidden="true">
      <span>
        <i className="p" />P pick
      </span>
      <span>
        <i className="s" />S pick
      </span>
      <span>
        <i className="pred" />
        modeled arrival
      </span>
    </div>
  );
}

/**
 * The record section: one row per trace sorted by epicentral distance, each normalized to its own peak
 * on a shared time axis (seconds after the origin time). P and S picks tick in on a 150 ms stagger per
 * row (CSS animation delays, so nothing re-renders while they play); the arrivals modeled from the
 * final location (`predP` / `predS`) are faint dashed lines. Missing picks or arrivals draw nothing.
 */
export function RecordSection({ status, evidence, message, originT }: RecordSectionProps) {
  const prepared = useMemo(() => (evidence ? prepareTraces(evidence.traces) : null), [evidence]);
  const domain = useMemo(
    () => (prepared && isNum(originT) ? recordDomain(prepared.traces, originT) : null),
    [prepared, originT],
  );
  const axis = useMemo(() => (domain ? timeTicks(domain) : null), [domain]);

  // The drawer never shows the provider's raw message (a URL and status code); it goes to the console:
  // a 404 is expected (evidence is capped at the exporter's maxEvents), anything else is a real failure.
  const unavailable = status === "error" ? evidenceUnavailable(message) : null;
  useEffect(() => {
    if (status !== "error") return;
    if (evidenceUnavailable(message).expected) console.info(`[drawer] no evidence file: ${message}`);
    else console.error(`[drawer] evidence failed to load: ${message}`);
  }, [status, message]);

  useEffect(() => {
    if (prepared?.skipped.length) {
      console.error(`[drawer] ${evidence?.eventId}: skipped ${prepared.skipped.length} trace(s): ${prepared.skipped.join("; ")}`);
    }
  }, [prepared, evidence]);

  const n = prepared?.traces.length ?? 0;
  const picked = evidence ? pickedTraceCount(evidence.traces) : 0;
  const [lo, hi] = evidence?.filterHz ?? [NaN, NaN];
  const band = isNum(lo) && isNum(hi) ? ` · ${fmtFixed(lo, lo < 1 ? 1 : 0)}–${fmtFixed(hi, hi < 1 ? 1 : 0)} Hz bandpass` : "";

  return (
    <section className="hqd-section" aria-label="Record section">
      <div className="hqd-sechead">
        <span className="hqd-label">Record section</span>
        <Legend />
      </div>
      <div className="hqd-caption">
        {status === "ready" && evidence ? (
          <>
            <span className="hqd-num">{n}</span> {n === 1 ? "trace" : "traces"} from the closest stations,{" "}
            <span className="hqd-num">{picked}</span> with a pick{band} · sorted by epicentral distance · normalized
            per trace
          </>
        ) : (
          "Waveforms from the stations closest to this event"
        )}
      </div>

      {unavailable ? (
        <div className={unavailable.expected ? "hqd-note" : "hqd-error"} role="status" data-testid="evidence-unavailable">
          {unavailable.text}
        </div>
      ) : status !== "ready" || !prepared ? (
        <ol className="hqd-record" data-status="loading" aria-busy="true">
          {Array.from({ length: PLACEHOLDER_ROWS }, (_, i) => (
            <li key={i} className="hqd-placeholder" />
          ))}
        </ol>
      ) : !domain || !axis || n === 0 ? (
        <div className="hqd-error" role="status">
          No drawable waveform snippets in this evidence file.
        </div>
      ) : (
        <>
          <ol className="hqd-record" data-status="ready" style={{ gridTemplateRows: `repeat(${n}, minmax(0, 1fr))` }}>
            <li className="hqd-grid" aria-hidden="true">
              {axis.ticks.map((t) => (
                <span key={t} className={t === 0 ? "zero" : undefined} style={{ left: `${markPercent(originT + t, originT, domain) ?? 0}%` }} />
              ))}
            </li>
            {prepared.traces.map((tr, i) => (
              <TraceRow key={`${tr.stationId}.${tr.channel}.${i}`} tr={tr} row={i} originT={originT} domain={domain} />
            ))}
          </ol>
          <div className="hqd-axis" aria-hidden="true">
            <div className="hqd-axis-track hqd-num">
              {axis.ticks.map((t) => (
                <span key={t} style={{ left: `${markPercent(originT + t, originT, domain) ?? 0}%` }}>
                  {fmtSeconds(t, axis.step)}
                </span>
              ))}
            </div>
          </div>
          <div className="hqd-axis-title">Seconds after origin</div>
        </>
      )}
      {prepared && prepared.skipped.length > 0 && (
        <div className="hqd-note">
          {prepared.skipped.length} {prepared.skipped.length === 1 ? "trace" : "traces"} not drawn (invalid data).
        </div>
      )}
    </section>
  );
}

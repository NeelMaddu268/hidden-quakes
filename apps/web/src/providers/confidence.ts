"use client";
/**
 * `confidence.json` (ML-01): the chance-association model's per-event scores, an optional file of
 * a static bundle (`/data/<mode>/confidence.json`, written by the exporter when the run has one).
 * The model is trained on decoy events made by scrambling station clocks; a score says how much an
 * event's timing looks like a real association rather than a scrambled-clock decoy. It is not a
 * probability that the event is an earthquake.
 *
 * The types are local to the web app (`@hq/contracts` is frozen). Nothing here trusts the file: a
 * wrong schema, another run's file or an unreadable one is `null`; a missing label, description or
 * held-out AUC is `null` in its field; a score that is not a number in [0, 1] is no score.
 */
import { useContext, useEffect, useMemo, useState } from "react";
import { ProviderContext, errorMessage } from "./root";
import { StaticBundleProvider } from "./static";

export const CONFIDENCE_SCHEMA = "hq.confidence/1";

/** The file as ML-01 writes it; every field may be missing at runtime. */
export interface ConfidenceFile {
  schema: string;
  runId?: string;
  model?: {
    type?: string;
    features?: string[];
    trainedOn?: { positives?: number; decoys?: number; shuffles?: number; shiftS?: number };
    heldOut?: { rocAuc?: number; averagePrecision?: number; folds?: number };
    createdAt?: string;
    gitSha?: string;
  };
  /** Short UI label. */
  label?: string;
  /** One sentence saying what the score means. */
  description?: string;
  /** eventId -> score in [0, 1]. */
  events?: Record<string, number>;
}

/** What the UI reads from a parsed `confidence.json`. */
export interface Confidence {
  label: string | null;
  description: string | null;
  /** `model.heldOut.rocAuc` when it is a number in [0, 1]. */
  rocAuc: number | null;
  /** The event's score in [0, 1], or `null` when the file does not score it. */
  score(eventId: string): number | null;
}

function isUnit(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value.trim() : null;
}

function object(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

/**
 * The usable part of a raw `confidence.json` for the bundle of run `runId`, or `null` when there is
 * none: not an `hq.confidence/1` object, a `runId` naming another run, or neither a score nor an AUC.
 */
export function parseConfidence(raw: unknown, runId: string): Confidence | null {
  const file = object(raw);
  if (!file || file.schema !== CONFIDENCE_SCHEMA) return null;
  if (file.runId !== undefined && file.runId !== runId) return null;
  const scores = new Map<string, number>();
  for (const [id, score] of Object.entries(object(file.events) ?? {})) {
    if (isUnit(score)) scores.set(id, score);
  }
  const rocAuc = object(object(file.model)?.heldOut)?.rocAuc;
  if (scores.size === 0 && !isUnit(rocAuc)) return null;
  return {
    label: text(file.label),
    description: text(file.description),
    rocAuc: isUnit(rocAuc) ? rocAuc : null,
    score: (eventId) => scores.get(eventId) ?? null,
  };
}

interface ConfidenceLoad {
  provider: StaticBundleProvider;
  raw: unknown;
}

/**
 * The active bundle's `confidence.json`, or `null`: while loading, when the file is missing (404),
 * unreadable or for another run, in live mode, and outside `<ProviderRoot>`. Never throws into the
 * UI. The provider memoizes the file, so every caller shares one request.
 */
export function useConfidence(): Confidence | null {
  const { provider, bundle } = useContext(ProviderContext);
  const source = provider instanceof StaticBundleProvider ? provider : null;
  const [load, setLoad] = useState<ConfidenceLoad | null>(null);

  useEffect(() => {
    if (!source) return;
    let cancelled = false;
    source.getConfidence().then(
      (raw) => {
        if (!cancelled) setLoad({ provider: source, raw });
      },
      (error: unknown) => {
        console.warn("confidence.json unavailable:", errorMessage(error));
        if (!cancelled) setLoad({ provider: source, raw: null });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [source]);

  const runId = bundle.status === "ready" ? bundle.meta.run.id : null;
  const raw = load && load.provider === source ? load.raw : null;
  return useMemo(() => (runId === null || raw === null ? null : parseConfidence(raw, runId)), [raw, runId]);
}

"use client";
/**
 * The only way components get data (docs/02 §6). Mount `<ProviderRoot>` once in `page.tsx`.
 */
import { useContext, useEffect, useState } from "react";
import type { Confidence, DataMode, LiveStatus, Validation } from "@hq/contracts";
import { ProviderContext, errorMessage } from "./root";
import type { BundleState, EvidenceState, SeismicDataProvider } from "./types";

function useProviderContext(hook: string) {
  const context = useContext(ProviderContext);
  if (!context.mounted) {
    throw new Error(`${hook} must be used inside <ProviderRoot> (mount it once in page.tsx)`);
  }
  return context;
}

export function useBundle(): BundleState {
  return useProviderContext("useBundle").bundle;
}

/** The mode chosen from `?mode=`; known before the bundle is, null until the root has parsed it. */
export function useMode(): DataMode | null {
  return useProviderContext("useMode").mode;
}

interface EvidenceLoad {
  provider: SeismicDataProvider;
  eventId: string;
  state: EvidenceState;
}

export function useEvidence(eventId: string | null): EvidenceState {
  const { provider } = useProviderContext("useEvidence");
  const [load, setLoad] = useState<EvidenceLoad | null>(null);

  useEffect(() => {
    if (!eventId || !provider) return;
    let cancelled = false;
    provider
      .getEventEvidence(eventId)
      .then((evidence) => {
        if (!cancelled) setLoad({ provider, eventId, state: { status: "ready", evidence } });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setLoad({ provider, eventId, state: { status: "error", message: errorMessage(error) } });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [eventId, provider]);

  if (!eventId || !provider) return { status: "idle" };
  if (load && load.provider === provider && load.eventId === eventId) return load.state;
  return { status: "loading" };
}

/** `null` until loaded, and `null` for bundles without `validation.json`. */
export function useValidation(): Validation | null {
  return useProviderContext("useValidation").validation;
}

/** `null` until loaded, and `null` for bundles without `confidence.json` (ML-01). */
export function useConfidence(): Confidence | null {
  return useProviderContext("useConfidence").confidence;
}

/** `null` outside live mode, and `null` while live mode is failed over to the snapshot. */
export function useLiveStatus(): LiveStatus | null {
  return useProviderContext("useLiveStatus").liveStatus;
}

/** True while `?mode=live` is showing the snapshot bundle because the worker is unreachable. */
export function useFailedOver(): boolean {
  return useProviderContext("useFailedOver").failedOver;
}

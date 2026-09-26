"use client";
/**
 * The only way components get data (docs/02 §6). Mount `<ProviderRoot>` once in `page.tsx`.
 */
import { useContext, useEffect, useState } from "react";
import type { LiveStatus, Validation } from "@hq/contracts";
import { ProviderContext, errorMessage } from "./root";
import type { BundleState, EvidenceState, SeismicDataProvider } from "./types";

export function useBundle(): BundleState {
  return useContext(ProviderContext).bundle;
}

interface EvidenceLoad {
  provider: SeismicDataProvider;
  eventId: string;
  state: EvidenceState;
}

export function useEvidence(eventId: string | null): EvidenceState {
  const { provider } = useContext(ProviderContext);
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
  return useContext(ProviderContext).validation;
}

/** `null` outside live mode. */
export function useLiveStatus(): LiveStatus | null {
  return useContext(ProviderContext).liveStatus;
}

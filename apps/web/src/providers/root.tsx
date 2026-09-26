"use client";
/**
 * `ProviderRoot` picks the provider from `?mode=` (or an explicit prop), loads the bundle once,
 * and hands the result to every hook in `./hooks.ts` through context. Components never fetch.
 *
 * State is keyed by provider and derived, never reset inside effects: a new provider means
 * "loading" until its own load resolves, so mode switches can't show stale data.
 */
import { createContext, useEffect, useMemo, useState, useSyncExternalStore, type ReactNode } from "react";
import type { DataMode, LiveStatus, Validation } from "@hq/contracts";
import { EVIDENCE_PRELOAD_COUNT, LIVE_POLL_MS } from "./config";
import { LiveProvider } from "./live";
import { mockAllowed, parseMode } from "./mode";
import { StaticBundleProvider } from "./static";
import type { BundleState, SeismicDataProvider } from "./types";

export interface ProviderContextValue {
  mode: DataMode | null;
  provider: SeismicDataProvider | null;
  bundle: BundleState;
  validation: Validation | null;
  liveStatus: LiveStatus | null;
}

export const ProviderContext = createContext<ProviderContextValue>({
  mode: null,
  provider: null,
  bundle: { status: "loading" },
  validation: null,
  liveStatus: null,
});

export class ModeDisabledError extends Error {
  constructor(mode: DataMode) {
    super(`${mode} mode is disabled in this build (set NEXT_PUBLIC_ALLOW_MOCK=1 to enable it)`);
    this.name = "ModeDisabledError";
  }
}

/** The provider for a mode; throws `ModeDisabledError` for mock in a production build. */
export function createProvider(mode: DataMode): SeismicDataProvider {
  if (mode === "live") return new LiveProvider();
  if (mode === "mock" && !mockAllowed()) throw new ModeDisabledError(mode);
  return new StaticBundleProvider(mode);
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** `window.location.search` as an external store: null while prerendering/hydrating. */
function subscribeToLocation(onChange: () => void): () => void {
  window.addEventListener("popstate", onChange);
  return () => window.removeEventListener("popstate", onChange);
}
const readSearch = () => window.location.search;
const readServerSearch = () => null;

export interface ProviderRootProps {
  /** Force a mode (tests, embeds). Otherwise read from `window.location.search` after hydration. */
  mode?: DataMode;
  /** Inject a provider (tests). Otherwise built from the mode. */
  provider?: SeismicDataProvider;
  children: ReactNode;
}

interface Loaded<T> {
  provider: SeismicDataProvider;
  value: T;
}

export function ProviderRoot({ mode, provider, children }: ProviderRootProps) {
  const search = useSyncExternalStore(subscribeToLocation, readSearch, readServerSearch);
  const resolvedMode: DataMode | null = mode ?? (search === null ? null : parseMode(search));

  const resolved = useMemo<{ provider: SeismicDataProvider | null; error: string | null }>(() => {
    if (resolvedMode === null) return { provider: null, error: null };
    if (provider) return { provider, error: null };
    try {
      return { provider: createProvider(resolvedMode), error: null };
    } catch (error) {
      return { provider: null, error: errorMessage(error) };
    }
  }, [resolvedMode, provider]);
  const activeProvider = resolved.provider;

  const [loadedBundle, setLoadedBundle] = useState<Loaded<BundleState> | null>(null);
  const [loadedValidation, setLoadedValidation] = useState<Loaded<Validation | null> | null>(null);
  const [loadedLive, setLoadedLive] = useState<Loaded<LiveStatus> | null>(null);

  useEffect(() => {
    if (!activeProvider) return;
    let cancelled = false;
    Promise.all([
      activeProvider.info(),
      activeProvider.getMeta(),
      activeProvider.getStations(),
      activeProvider.getCatalog(),
      activeProvider.getEvents(),
      activeProvider.getFeatures(),
    ])
      .then(([info, meta, stations, catalog, events, features]) => {
        if (cancelled) return;
        setLoadedBundle({
          provider: activeProvider,
          value: { status: "ready", info, meta, stations, catalog, events, features },
        });
        preloadEvidence(activeProvider, meta.scene.heroEventId, events);
        return activeProvider.getValidation().then((validation) => {
          if (!cancelled) setLoadedValidation({ provider: activeProvider, value: validation });
        });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setLoadedBundle({ provider: activeProvider, value: { status: "error", message: errorMessage(error) } });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [activeProvider]);

  useEffect(() => {
    if (!(activeProvider instanceof LiveProvider)) return;
    const live = activeProvider;
    let cancelled = false;
    const poll = () =>
      Promise.all([live.getStatus(), live.getEvents()])
        .then(([status, events]) => {
          if (!cancelled) setLoadedLive({ provider: live, value: { ...status, events } });
        })
        .catch(() => {
          // API-05 turns this into snapshot failover; until then the last status stays visible.
        });
    void poll();
    const timer = setInterval(() => void poll(), LIVE_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [activeProvider]);

  const value = useMemo<ProviderContextValue>(() => {
    const bundle: BundleState = resolved.error
      ? { status: "error", message: resolved.error }
      : loadedBundle && loadedBundle.provider === activeProvider
        ? loadedBundle.value
        : { status: "loading" };
    const validation =
      loadedValidation && loadedValidation.provider === activeProvider ? loadedValidation.value : null;
    const liveStatus = loadedLive && loadedLive.provider === activeProvider ? loadedLive.value : null;
    return { mode: resolvedMode, provider: activeProvider, bundle, validation, liveStatus };
  }, [resolvedMode, activeProvider, resolved.error, loadedBundle, loadedValidation, loadedLive]);

  return <ProviderContext.Provider value={value}>{children}</ProviderContext.Provider>;
}

/** Warm the provider's memo for the hero and the first events of the reveal. Errors are ignored
 *  here; the drawer reports them when it actually asks. Returns the ids requested, in order. */
export function preloadEvidence(
  provider: SeismicDataProvider,
  heroEventId: string | null,
  events: readonly { id: string; revealOrder: number }[],
): string[] {
  const ordered = [...events].sort((a, b) => a.revealOrder - b.revealOrder).map((e) => e.id);
  const ids = heroEventId ? [heroEventId, ...ordered.filter((id) => id !== heroEventId)] : ordered;
  const chosen = ids.slice(0, EVIDENCE_PRELOAD_COUNT + (heroEventId ? 1 : 0));
  for (const id of chosen) {
    provider.getEventEvidence(id).catch(() => undefined);
  }
  return chosen;
}

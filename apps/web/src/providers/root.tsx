"use client";
/**
 * `ProviderRoot` picks the provider from `?mode=` (or an explicit prop), loads the bundle once,
 * and hands the result to every hook in `./hooks.ts` through context. Components never fetch.
 *
 * State is keyed by provider and derived, never reset inside effects: a new provider means
 * "loading" until its own load resolves, so mode switches can't show stale data.
 *
 * The mode is read once per document (`window.location.search` after hydration). Switch modes
 * with `navigateToMode()` (a full navigation), not a soft route change.
 *
 * Live failover (API-05, docs/02 §6). In `live` mode the active provider is one of two:
 *
 *   live ──(any live request fails: refused, timed out, 503)──▶ snapshot (`live.fallback`)
 *   snapshot ──(a heartbeat or poll of `/api/live/status` succeeds)──▶ live
 *
 * The live routes are polled every `LIVE_POLL_MS` (status + events, refreshing "updated n min
 * ago") and probed every `LIVE_HEARTBEAT_MS` (status only), whichever provider is active, so a
 * worker that goes away is noticed within one heartbeat plus the request timeout, and one that
 * comes back is noticed the same way. Each switch loads the bundle from the new provider and
 * swaps it in whole, so the label is always the active provider's own (`ModeInfo.label`); the
 * bundle already on screen stays until then (no "loading" flash between live and snapshot, which
 * are two views of one document), and the snapshot's files are warmed while live is healthy, so
 * a failover needs no network at all. A live failure that is not a fetch failure (a schemaVersion
 * mismatch, a bug) is a visible error, never a silent failover; a snapshot that fails to load
 * after a failover is the visible error too.
 */
import { createContext, useEffect, useMemo, useState, useSyncExternalStore, type ReactNode } from "react";
import type { Confidence, DataMode, LiveStatus, Validation } from "@hq/contracts";
import { EVIDENCE_PRELOAD_COUNT, LIVE_HEARTBEAT_MS, LIVE_POLL_MS } from "./config";
import { BundleFetchError } from "./fetch";
import { LiveProvider, type LiveStatusSummary } from "./live";
import { mockAllowed, parseMode } from "./mode";
import { StaticBundleProvider } from "./static";
import type { BundleState, SeismicDataProvider } from "./types";

export interface ProviderContextValue {
  /** False outside `<ProviderRoot>`; the hooks throw instead of spinning forever. */
  mounted: boolean;
  mode: DataMode | null;
  provider: SeismicDataProvider | null;
  bundle: BundleState;
  validation: Validation | null;
  confidence: Confidence | null;
  liveStatus: LiveStatus | null;
  /** True while `live` mode is showing the snapshot bundle because the worker is unreachable. */
  failedOver: boolean;
}

export const ProviderContext = createContext<ProviderContextValue>({
  mounted: false,
  mode: null,
  provider: null,
  bundle: { status: "loading" },
  validation: null,
  confidence: null,
  liveStatus: null,
  failedOver: false,
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

/**
 * Only a failed request to the live API itself (refused, timed out, non-2xx such as 503) fails
 * over. Anything else surfaces as the visible error: a schemaVersion mismatch, a bug, or a
 * snapshot file the live provider borrows that is missing (live cannot work without it either,
 * so retrying live would only loop).
 */
export function isFailoverError(error: unknown, live: LiveProvider): boolean {
  return error instanceof BundleFetchError && error.url.startsWith(live.apiBase);
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
  /**
   * Inject a provider (tests). Must be referentially stable across renders (create it once with
   * `useState(() => new StaticBundleProvider(...))`); a new instance per render reloads forever.
   */
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

  // The live provider of this document, if any, and whether it is currently failed over. Keyed
  // by instance, so a provider switch never inherits the old one's failover.
  const live = resolved.provider instanceof LiveProvider ? resolved.provider : null;
  const [failedOverLive, setFailedOverLive] = useState<LiveProvider | null>(null);
  const failedOver = live !== null && failedOverLive === live;
  const activeProvider: SeismicDataProvider | null = failedOver ? live.fallback : resolved.provider;

  const [loadedBundle, setLoadedBundle] = useState<Loaded<BundleState> | null>(null);
  const [loadedValidation, setLoadedValidation] = useState<Loaded<Validation | null> | null>(null);
  const [loadedConfidence, setLoadedConfidence] = useState<Loaded<Confidence | null> | null>(null);
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
        // Static providers memoize, so warming them is free; the live API is polled instead.
        if (!(activeProvider instanceof LiveProvider)) {
          preloadEvidence(activeProvider, meta.scene.heroEventId, events);
        } else {
          warmFallback(activeProvider.fallback);
        }
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        if (activeProvider === live && isFailoverError(error, live)) {
          console.warn("live bundle unavailable; showing the snapshot bundle:", errorMessage(error));
          setFailedOverLive(live);
          return;
        }
        setLoadedBundle({ provider: activeProvider, value: { status: "error", message: errorMessage(error) } });
      });
    // validation.json is optional (P1): its failure never touches the bundle state.
    activeProvider
      .getValidation()
      .then((validation) => {
        if (!cancelled) setLoadedValidation({ provider: activeProvider, value: validation });
      })
      .catch((error: unknown) => {
        console.warn("validation.json unavailable:", errorMessage(error));
        if (!cancelled) setLoadedValidation({ provider: activeProvider, value: null });
      });
    // confidence.json (ML-01) is optional the same way: absent or failing means no scores.
    activeProvider
      .getConfidence()
      .then((confidence) => {
        if (!cancelled) setLoadedConfidence({ provider: activeProvider, value: confidence });
      })
      .catch((error: unknown) => {
        console.warn("confidence.json unavailable:", errorMessage(error));
        if (!cancelled) setLoadedConfidence({ provider: activeProvider, value: null });
      });
    return () => {
      cancelled = true;
    };
  }, [activeProvider, live]);

  // Poll and heartbeat the live worker for as long as this document is in live mode, failed over
  // or not: the heartbeat is what notices the worker leaving and coming back.
  useEffect(() => {
    if (!live) return;
    let cancelled = false;
    let announced = false; // the failover is logged once per outage
    const healthy = (status: LiveStatusSummary) => {
      announced = false;
      // Back to live if we were on the snapshot; the load effect reloads from the worker.
      setFailedOverLive((current) => (current === live ? null : current));
      // Keep "updated n min ago" honest. Same label, same bundle: nothing re-renders.
      setLoadedBundle((previous) => {
        if (!previous || previous.provider !== live || previous.value.status !== "ready") return previous;
        const label = live.labelFor(status);
        const { info } = previous.value;
        if (info.label === label && info.generatedAt === status.updatedAt) return previous;
        return { provider: live, value: { ...previous.value, info: { ...info, label, generatedAt: status.updatedAt } } };
      });
    };
    const unhealthy = (what: string) => (error: unknown) => {
      if (cancelled) return;
      if (!isFailoverError(error, live)) {
        console.warn(`live ${what} failed:`, errorMessage(error));
        return;
      }
      if (!announced) {
        announced = true;
        console.warn(`live ${what} failed; showing the snapshot bundle:`, errorMessage(error));
      }
      setFailedOverLive(live);
    };
    const poll = () =>
      Promise.all([live.getStatus(), live.getEvents()])
        .then(([status, events]) => {
          if (cancelled) return;
          setLoadedLive({ provider: live, value: { ...status, events } });
          healthy(status);
        })
        .catch(unhealthy("status poll"));
    const heartbeat = () =>
      live
        .getStatus()
        .then((status) => {
          if (!cancelled) healthy(status);
        })
        .catch(unhealthy("heartbeat"));
    void poll();
    const pollTimer = setInterval(() => void poll(), LIVE_POLL_MS);
    const heartbeatTimer = setInterval(() => void heartbeat(), LIVE_HEARTBEAT_MS);
    return () => {
      cancelled = true;
      clearInterval(pollTimer);
      clearInterval(heartbeatTimer);
    };
  }, [live]);

  const value = useMemo<ProviderContextValue>(() => {
    // Live and its snapshot are two views of one document: while switching between them the
    // bundle already loaded from the other one stays on screen until the new one is ready.
    const liveFamily = (p: SeismicDataProvider) => live !== null && (p === live || p === live.fallback);
    const bundle: BundleState = resolved.error
      ? { status: "error", message: resolved.error }
      : loadedBundle && loadedBundle.provider === activeProvider
        ? loadedBundle.value
        : loadedBundle && activeProvider && liveFamily(activeProvider) && liveFamily(loadedBundle.provider)
          ? loadedBundle.value
          : { status: "loading" };
    const validation =
      loadedValidation && loadedValidation.provider === activeProvider ? loadedValidation.value : null;
    const confidence =
      loadedConfidence && loadedConfidence.provider === activeProvider ? loadedConfidence.value : null;
    const liveStatus = loadedLive && loadedLive.provider === activeProvider ? loadedLive.value : null;
    return {
      mounted: true,
      mode: resolvedMode,
      provider: activeProvider,
      bundle,
      validation,
      confidence,
      liveStatus,
      failedOver,
    };
  }, [resolvedMode, activeProvider, live, resolved.error, loadedBundle, loadedValidation, loadedConfidence, loadedLive, failedOver]);

  return <ProviderContext.Provider value={value}>{children}</ProviderContext.Provider>;
}

/** Warm the snapshot's memo while live is healthy (the live load already fetched its meta,
 *  stations, catalog and features), so a failover swaps bundles without a request. */
function warmFallback(fallback: SeismicDataProvider): void {
  fallback.getEvents().catch(() => undefined);
  fallback.getValidation().catch(() => undefined);
  fallback.getConfidence().catch(() => undefined);
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

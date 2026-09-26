/**
 * `LiveProvider` (docs/02 §7): reads the live worker's routes for meta, events and evidence,
 * and borrows stations, catalog and features from a fallback static provider (the snapshot
 * bundle), because the live API doesn't serve them.
 *
 * Every live request is bounded by `LIVE_FETCH_TIMEOUT_MS`: one that has not answered by then
 * is aborted and rejects with a `BundleFetchError`, the same error a refused connection or a
 * 503 produces. `ProviderRoot` turns any `BundleFetchError` from this provider into snapshot
 * failover (API-05); the fallback it switches to is `fallback`, so live and failed-over views
 * read the same snapshot files.
 */
import {
  SCHEMA_VERSION,
  type BundleMeta,
  type CatalogEvent,
  type EventEvidence,
  type GeoFeature,
  type LiveStatus,
  type SeismicEvent,
  type Station,
  type Validation,
} from "@hq/contracts";
import { LIVE_API_BASE, LIVE_FETCH_TIMEOUT_MS } from "./config";
import { BundleFetchError, SchemaVersionError, defaultFetch, fetchJson, type FetchLike } from "./fetch";
import { StaticBundleProvider } from "./static";
import type { ModeInfo, SeismicDataProvider } from "./types";

/** `GET /api/live/status` returns `LiveStatus` without its `events`. */
export type LiveStatusSummary = Omit<LiveStatus, "events">;

export interface LiveProviderOptions {
  fetchImpl?: FetchLike;
  /** Source of stations, catalog and features; defaults to the snapshot bundle. */
  fallback?: SeismicDataProvider;
  /** Clock for "updated n min ago"; injectable for tests. Returns epoch seconds. */
  now?: () => number;
  /** Per-request deadline; defaults to `LIVE_FETCH_TIMEOUT_MS`. */
  timeoutMs?: number;
}

const SECONDS_PER_MINUTE = 60;
const SECONDS_PER_HOUR = 3600;

/** "2 h", "90 min", "1.5 h": whole hours stay whole, sub-hour windows show minutes. */
export function formatWindow(windowS: number): string {
  if (windowS < SECONDS_PER_HOUR) return `${Math.round(windowS / SECONDS_PER_MINUTE)} min`;
  const hours = windowS / SECONDS_PER_HOUR;
  return Number.isInteger(hours) ? `${hours} h` : `${hours.toFixed(1)} h`;
}

export function liveLabel(status: LiveStatusSummary, nowS: number): string {
  const minutesAgo = Math.max(0, Math.round((nowS - status.updatedAt) / SECONDS_PER_MINUTE));
  return `Live · last ${formatWindow(status.windowS)} · updated ${minutesAgo} min ago`;
}

/**
 * `fetchJson` with a deadline. On timeout the request is aborted (when the runtime has
 * `AbortController`) and the result is a `BundleFetchError` naming the deadline, so a hanging
 * worker fails over exactly like a refused connection.
 */
export function fetchWithTimeout<T>(fetchImpl: FetchLike, url: string, timeoutMs: number): Promise<T | null> {
  const controller = typeof AbortController === "function" ? new AbortController() : null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      controller?.abort();
      reject(new BundleFetchError(url, null, `no response within ${timeoutMs} ms`));
    }, timeoutMs);
  });
  return Promise.race([fetchJson<T>(fetchImpl, url, { signal: controller?.signal }), deadline]).finally(() =>
    clearTimeout(timer),
  );
}

export class LiveProvider implements SeismicDataProvider {
  readonly apiBase: string;
  /** Source of stations, catalog and features, and the failover target (the snapshot bundle). */
  readonly fallback: SeismicDataProvider;
  readonly timeoutMs: number;
  private readonly fetchImpl: FetchLike;
  private readonly now: () => number;
  /** Single-flight: concurrent calls for one route share a request; the next call refetches. */
  private readonly inflight = new Map<string, Promise<unknown>>();

  constructor(apiBase: string = LIVE_API_BASE, options: LiveProviderOptions = {}) {
    this.apiBase = apiBase;
    this.fetchImpl = options.fetchImpl ?? defaultFetch;
    this.fallback = options.fallback ?? new StaticBundleProvider("snapshot", { fetchImpl: this.fetchImpl });
    this.now = options.now ?? (() => Date.now() / 1000);
    this.timeoutMs = options.timeoutMs ?? LIVE_FETCH_TIMEOUT_MS;
  }

  private get<T>(route: string): Promise<T> {
    const hit = this.inflight.get(route);
    if (hit) return hit as Promise<T>;
    const url = `${this.apiBase}/${route}`;
    const promise = fetchWithTimeout<T>(this.fetchImpl, url, this.timeoutMs)
      .then((value) => {
        if (value === null) throw new BundleFetchError(url, 404);
        return value;
      })
      .finally(() => this.inflight.delete(route));
    this.inflight.set(route, promise);
    return promise;
  }

  /** The shell label for a status, using this provider's clock. */
  labelFor(status: LiveStatusSummary): string {
    return liveLabel(status, this.now());
  }

  async info(): Promise<ModeInfo> {
    // The fallback's meta runs the schemaVersion guard on the snapshot too, so a stale snapshot
    // can't be mixed into live data silently.
    const [meta, status] = await Promise.all([this.getMeta(), this.getStatus(), this.fallback.getMeta()]);
    return {
      mode: "live",
      label: this.labelFor(status),
      runId: meta.run.id,
      generatedAt: status.updatedAt,
      isSynthetic: meta.run.isSynthetic || meta.scene.isSynthetic,
    };
  }

  async getMeta(): Promise<BundleMeta> {
    const meta = await this.get<BundleMeta>("meta");
    if (meta.schemaVersion !== SCHEMA_VERSION) {
      throw new SchemaVersionError(SCHEMA_VERSION, String(meta.schemaVersion), `${this.apiBase}/meta`);
    }
    return meta;
  }

  getStatus(): Promise<LiveStatusSummary> {
    return this.get<LiveStatusSummary>("status");
  }

  getEvents(): Promise<SeismicEvent[]> {
    return this.get<SeismicEvent[]>("events");
  }

  getEventEvidence(id: string): Promise<EventEvidence> {
    return this.get<EventEvidence>(`evidence/${encodeURIComponent(id)}`);
  }

  getStations(): Promise<Station[]> {
    return this.fallback.getStations();
  }

  getCatalog(): Promise<CatalogEvent[]> {
    return this.fallback.getCatalog();
  }

  getFeatures(): Promise<GeoFeature[]> {
    return this.fallback.getFeatures();
  }

  /** Validation belongs to the frozen showcase run, not to a rolling window. */
  getValidation(): Promise<Validation | null> {
    return Promise.resolve(null);
  }
}

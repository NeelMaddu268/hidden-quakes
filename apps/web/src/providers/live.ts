/**
 * `LiveProvider` (P1 stub, docs/02 §7): reads the live worker's routes for meta, events and
 * evidence, and borrows stations, catalog and features from a fallback static provider (the
 * snapshot bundle), because the live API doesn't serve them. API-05 adds the 5 s failover.
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
import { LIVE_API_BASE } from "./config";
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

export class LiveProvider implements SeismicDataProvider {
  readonly apiBase: string;
  private readonly fetchImpl: FetchLike;
  private readonly fallback: SeismicDataProvider;
  private readonly now: () => number;
  /** Single-flight: concurrent calls for one route share a request; the next call refetches. */
  private readonly inflight = new Map<string, Promise<unknown>>();

  constructor(apiBase: string = LIVE_API_BASE, options: LiveProviderOptions = {}) {
    this.apiBase = apiBase;
    this.fetchImpl = options.fetchImpl ?? defaultFetch;
    this.fallback = options.fallback ?? new StaticBundleProvider("snapshot", { fetchImpl: this.fetchImpl });
    this.now = options.now ?? (() => Date.now() / 1000);
  }

  private get<T>(route: string): Promise<T> {
    const hit = this.inflight.get(route);
    if (hit) return hit as Promise<T>;
    const url = `${this.apiBase}/${route}`;
    const promise = fetchJson<T>(this.fetchImpl, url)
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

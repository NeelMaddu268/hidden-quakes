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

export function liveLabel(status: LiveStatusSummary, nowS: number): string {
  const hours = Math.round(status.windowS / SECONDS_PER_HOUR);
  const minutesAgo = Math.max(0, Math.round((nowS - status.updatedAt) / SECONDS_PER_MINUTE));
  return `Live · last ${hours} h · updated ${minutesAgo} min ago`;
}

export class LiveProvider implements SeismicDataProvider {
  readonly apiBase: string;
  private readonly fetchImpl: FetchLike;
  private readonly fallback: SeismicDataProvider;
  private readonly now: () => number;

  constructor(apiBase: string = LIVE_API_BASE, options: LiveProviderOptions = {}) {
    this.apiBase = apiBase;
    this.fetchImpl = options.fetchImpl ?? defaultFetch;
    this.fallback = options.fallback ?? new StaticBundleProvider("snapshot", { fetchImpl: this.fetchImpl });
    this.now = options.now ?? (() => Date.now() / 1000);
  }

  private async get<T>(route: string): Promise<T> {
    const value = await fetchJson<T>(this.fetchImpl, `${this.apiBase}/${route}`);
    if (value === null) throw new BundleFetchError(`${this.apiBase}/${route}`, 404);
    return value;
  }

  async info(): Promise<ModeInfo> {
    const [meta, status] = await Promise.all([this.getMeta(), this.getStatus()]);
    return {
      mode: "live",
      label: liveLabel(status, this.now()),
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

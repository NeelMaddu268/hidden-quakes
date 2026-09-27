/**
 * `StaticBundleProvider`: the same reader for mock, showcase and snapshot, pointed at a different
 * folder under `/data/<mode>/` (docs/01 → Data bundle). Every fetch is memoized per file, so
 * hooks can call it freely; the meta fetch also enforces `schemaVersion`.
 */
import {
  SCHEMA_VERSION,
  type BundleMeta,
  type CatalogEvent,
  type DataMode,
  type EventEvidence,
  type GeoFeature,
  type SeismicEvent,
  type Station,
  type Validation,
} from "@hq/contracts";
import { DATA_BASE_URL } from "./config";
import { BundleFetchError, SchemaVersionError, defaultFetch, fetchJson, type FetchLike } from "./fetch";
import type { ModeInfo, SeismicDataProvider } from "./types";

export type StaticMode = Exclude<DataMode, "live">;

export interface StaticBundleOptions {
  /** Site-relative folder holding `<mode>/`; defaults to `/data`. */
  baseUrl?: string;
  fetchImpl?: FetchLike;
}

/** Epoch seconds of an ISO 8601 timestamp; throws instead of returning NaN. */
export function epochSeconds(iso: string, url: string): number {
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) throw new BundleFetchError(url, null, `invalid ISO timestamp ${JSON.stringify(iso)}`);
  return ms / 1000;
}

/** Mode label for the shell (lane doc → Providers table). Every value comes from the bundle. */
export function modeLabel(mode: StaticMode, meta: BundleMeta): string {
  switch (mode) {
    case "showcase":
      return `Showcase · ${meta.run.windowLabel} · run ${meta.run.id}`;
    case "snapshot":
      return `Snapshot · ${meta.run.windowLabel} · run ${meta.run.id}`;
    case "mock":
      return `Synthetic · mock bundle · run ${meta.run.id}`;
  }
}

export class StaticBundleProvider implements SeismicDataProvider {
  readonly mode: StaticMode;
  readonly base: string;
  private readonly fetchImpl: FetchLike;
  private readonly cache = new Map<string, Promise<unknown>>();

  constructor(mode: StaticMode, options: StaticBundleOptions = {}) {
    this.mode = mode;
    this.base = `${options.baseUrl ?? DATA_BASE_URL}/${mode}`;
    this.fetchImpl = options.fetchImpl ?? defaultFetch;
  }

  url(name: string): string {
    return `${this.base}/${name}`;
  }

  /** Memoized by file; a failed fetch is forgotten so the next call retries. */
  private memo<T>(key: string, load: () => Promise<T>): Promise<T> {
    const hit = this.cache.get(key);
    if (hit) return hit as Promise<T>;
    const promise = load().catch((error: unknown) => {
      this.cache.delete(key);
      throw error;
    });
    this.cache.set(key, promise);
    return promise;
  }

  private file<T>(name: string): Promise<T> {
    return this.memo(name, async () => {
      const value = await fetchJson<T>(this.fetchImpl, this.url(name));
      return value as T;
    });
  }

  async info(): Promise<ModeInfo> {
    const meta = await this.getMeta();
    return {
      mode: this.mode,
      label: modeLabel(this.mode, meta),
      runId: meta.run.id,
      generatedAt: epochSeconds(meta.run.createdAt, this.url("meta.json")),
      isSynthetic: meta.run.isSynthetic || meta.scene.isSynthetic,
    };
  }

  getMeta(): Promise<BundleMeta> {
    return this.memo("meta", async () => {
      const meta = await fetchJson<BundleMeta>(this.fetchImpl, this.url("meta.json"));
      if (meta === null) throw new BundleFetchError(this.url("meta.json"), 404);
      if (meta.schemaVersion !== SCHEMA_VERSION) {
        throw new SchemaVersionError(SCHEMA_VERSION, String(meta.schemaVersion), this.url("meta.json"));
      }
      return meta;
    });
  }

  getStations(): Promise<Station[]> {
    return this.file<Station[]>("stations.json");
  }

  getCatalog(): Promise<CatalogEvent[]> {
    return this.file<CatalogEvent[]>("catalog.json");
  }

  getEvents(): Promise<SeismicEvent[]> {
    return this.file<SeismicEvent[]>("events.json");
  }

  getFeatures(): Promise<GeoFeature[]> {
    return this.file<GeoFeature[]>("features.json");
  }

  getEventEvidence(id: string): Promise<EventEvidence> {
    return this.file<EventEvidence>(`evidence/${encodeURIComponent(id)}.json`);
  }

  /** `confidence.json` (ML-01) is optional too; a missing file is `null`. `./confidence` parses it. */
  getConfidence(): Promise<unknown> {
    return this.memo("confidence", () =>
      fetchJson<unknown>(this.fetchImpl, this.url("confidence.json"), { notFoundAsNull: true }),
    );
  }

  /** `validation.json` is optional in a bundle (P1); a missing file is `null`, not an error. */
  getValidation(): Promise<Validation | null> {
    return this.memo("validation", () =>
      fetchJson<Validation>(this.fetchImpl, this.url("validation.json"), { notFoundAsNull: true }),
    );
  }
}

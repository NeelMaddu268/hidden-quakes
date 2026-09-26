/**
 * Provider interfaces (docs/02 §6). Components never fetch; they call the hooks in `./hooks.ts`
 * (API-01), and every hook goes through a `SeismicDataProvider`.
 *
 * Contract types are re-exported here so app code imports one module: `@/providers/types`.
 */
import type {
  BundleMeta,
  CatalogEvent,
  DataMode,
  EventEvidence,
  GeoFeature,
  SeismicEvent,
  Station,
  Validation,
} from "@hq/contracts";

export type {
  AnalysisSummary,
  BaselineGain,
  BaselineRow,
  Bundle,
  BundleMeta,
  CatalogEvent,
  CatalogMatch,
  DataMode,
  Enu,
  EventEvidence,
  GRCurve,
  GeoFeature,
  LiveStatus,
  LocationQuality,
  MagCalibration,
  Magnitude,
  NullTest,
  Phase,
  Pick,
  ProcessingRun,
  SceneMeta,
  SeismicEvent,
  SourceRef,
  Station,
  SweepPoint,
  SyntheticTest,
  Tier,
  TierCounts,
  Validation,
  WaveformSnippet,
} from "@hq/contracts";
export { SCHEMA_VERSION } from "@hq/contracts";

export interface ModeInfo {
  mode: DataMode;
  label: string;
  runId: string;
  generatedAt: number;
  isSynthetic: boolean;
}

export interface SeismicDataProvider {
  info(): Promise<ModeInfo>;
  getMeta(): Promise<BundleMeta>;
  getStations(): Promise<Station[]>;
  getCatalog(): Promise<CatalogEvent[]>;
  getEvents(): Promise<SeismicEvent[]>;
  getFeatures(): Promise<GeoFeature[]>;
  getEventEvidence(id: string): Promise<EventEvidence>;
  getValidation(): Promise<Validation | null>;
}

export type BundleState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | {
      status: "ready";
      info: ModeInfo;
      meta: BundleMeta;
      stations: Station[];
      catalog: CatalogEvent[];
      events: SeismicEvent[];
      features: GeoFeature[];
    };

export type EvidenceState = {
  status: "idle" | "loading" | "ready" | "error";
  evidence?: EventEvidence;
  message?: string;
};

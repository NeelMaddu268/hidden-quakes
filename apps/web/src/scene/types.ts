// The one place the scene imports contract types from: data models are generated from the Python
// contracts (CONTRACT-01, `@hq/contracts`), and the bundle state shape is H4's provider type (docs/02 §6).
export type {
  AnalysisSummary,
  BundleMeta,
  CatalogEvent,
  CatalogMatch,
  Enu,
  EventEvidence,
  GeoFeature,
  LocationQuality,
  Magnitude,
  SceneMeta,
  SeismicEvent,
  Station,
  WaveformSnippet,
} from "@hq/contracts";
export type { BundleState, EvidenceState } from "../providers/types";

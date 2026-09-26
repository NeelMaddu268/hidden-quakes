// The one place the scene imports contract types from: data models are generated from the Python
// contracts (CONTRACT-01, `@hq/contracts`), and the bundle state shape is H4's provider type (docs/02 §6).
export type {
  AnalysisSummary,
  BundleMeta,
  CatalogEvent,
  Enu,
  GeoFeature,
  SceneMeta,
  SeismicEvent,
  Station,
} from "@hq/contracts";
export type { BundleState } from "../providers/types";

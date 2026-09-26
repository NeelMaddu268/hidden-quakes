"""File names inside a bundle directory (docs/01 -> Data bundle), shared by the writer and the
checker so neither imports the other for a string."""

META_JSON = "meta.json"
STATIONS_JSON = "stations.json"
CATALOG_JSON = "catalog.json"
EVENTS_JSON = "events.json"
FEATURES_JSON = "features.json"
VALIDATION_JSON = "validation.json"
CONFIDENCE_JSON = "confidence.json"  # ML-01 (H2) sidecar, copied when present
EVIDENCE_DIR = "evidence"

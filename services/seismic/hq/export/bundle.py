"""``export_bundle``: run tables -> one static bundle directory (docs/01 -> Data bundle).

Files, all key-sorted JSON so the same run gives byte-identical output::

    meta.json                BundleMeta   schemaVersion, mode, SceneMeta, ProcessingRun, summary
    stations.json            Station[]
    catalog.json             CatalogEvent[]   matchedEventId filled from matches.parquet
    events.json              SeismicEvent[]   revealOrder assigned, sorted by (t, id)
    features.json            GeoFeature[]     from hq.export.features (FEAT-01)
    validation.json          Validation       omitted when the run has none yet (VAL-01)
    confidence.json          Confidence       omitted when the run has none (ML-01, H2)
    evidence/{eventId}.json  EventEvidence    hero first, then reveal order, capped by maxEvents

The bundle is built in a temporary sibling directory, checked there with ``check_bundle`` and
renamed into place only when the check passes, so a failure leaves the previous bundle untouched
and never a half-written or invalid one.
"""

import importlib
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hq_contracts.models import BundleMeta, GeoFeature, SeismicEvent, Station

from hq.config.export import ExportConfig, ExportMode
from hq.config.run import RunSection
from hq.config.validate import BaselineConfig
from hq.export.check import check_bundle
from hq.export.errors import ExportError
from hq.export.evidence import build_evidence
from hq.export.files import (
    CATALOG_JSON,
    CONFIDENCE_JSON,
    EVENTS_JSON,
    EVIDENCE_DIR,
    FEATURES_JSON,
    META_JSON,
    STATIONS_JSON,
    VALIDATION_JSON,
)
from hq.export.summary import (
    analysis_summary,
    apply_matches,
    assign_reveal_order,
    choose_hero,
    scene_meta,
)
from hq.export.tables import RunTables
from hq.export.waveforms import WaveformSource

log = logging.getLogger(__name__)

FEATURES_MODULE, FEATURES_NAME = "hq.export.features", "load_features"
FEAT_OWNER = "FEAT-01 (H4 Platform)"
PRETTY_INDENT = 2  # small files are readable in git diffs; events and evidence are compact
COMPACT_SEPARATORS = (",", ":")
TMP_PREFIX = "."  # temp dirs beside the bundle start with a dot so nothing serves them
_MIB = 1024.0 * 1024.0  # log unit


@dataclass(frozen=True)
class ExportResult:
    """What one ``export_bundle`` call wrote."""

    out_dir: Path
    mode: str
    counts: dict[str, int]
    sizes: dict[str, int] = field(repr=False)  # relative path -> bytes

    @property
    def total_bytes(self) -> int:
        return sum(self.sizes.values())


FeaturesLoader = Callable[[ExportConfig, RunSection], list[GeoFeature]]


def load_features_lazily(cfg: ExportConfig, section: RunSection) -> list[GeoFeature]:
    """``hq.export.features.load_features`` (FEAT-01), imported now so this package works
    before it is merged. Without the module, an empty ``features`` config gives an empty
    ``features.json`` and a warning naming FEAT-01 (like a missing VAL-01 output); a configured
    feature list that nothing can load is an error."""
    try:
        module = importlib.import_module(FEATURES_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != FEATURES_MODULE:
            raise
        if cfg.features:
            raise ExportError(
                f"export.yaml lists features but {FEATURES_MODULE}.{FEATURES_NAME} is not merged "
                f"yet (owner: {FEAT_OWNER})"
            ) from exc
        log.warning(
            "export: %s.%s is not merged yet (%s) and export.yaml lists no features; "
            "features.json is empty",
            FEATURES_MODULE,
            FEATURES_NAME,
            FEAT_OWNER,
        )
        return []
    fn = getattr(module, FEATURES_NAME, None)
    if not callable(fn):
        raise ExportError(
            f"{FEATURES_MODULE} has no {FEATURES_NAME}(cfg, run) (owner: {FEAT_OWNER})"
        )
    features = fn(cfg, section)
    bad = [f for f in features if not isinstance(f, GeoFeature)]
    if bad:
        raise ExportError(f"{FEATURES_MODULE}.{FEATURES_NAME} returned non-GeoFeature items")
    ids = [f.id for f in features]
    if len(set(ids)) != len(ids):
        raise ExportError(f"{FEATURES_MODULE}.{FEATURES_NAME} returned duplicate feature ids")
    return list(features)


def dump_json(path: Path, data: Any, *, pretty: bool) -> int:
    """Write key-sorted JSON (indented or compact) and return the bytes written."""
    text = json.dumps(
        data,
        sort_keys=True,
        indent=PRETTY_INDENT if pretty else None,
        separators=None if pretty else COMPACT_SEPARATORS,
        allow_nan=False,
    )
    payload = (text + "\n").encode("utf-8")
    path.write_bytes(payload)
    return len(payload)


def _write_evidence(
    tables: RunTables,
    events: list[SeismicEvent],
    hero_id: str | None,
    cfg: ExportConfig,
    source: WaveformSource,
    cache_dir: Path,
    evidence_dir: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    """Evidence files for the hero, then reveal order, capped by ``evidence.maxEvents``."""
    ordered = sorted(events, key=lambda e: e.revealOrder)
    if hero_id is not None:
        ordered = [e for e in ordered if e.id == hero_id] + [e for e in ordered if e.id != hero_id]
    if cfg.evidence.maxEvents is not None:
        ordered = ordered[: cfg.evidence.maxEvents]
    stations = {s.id: s for s in tables.stations}
    by_event = {
        str(event_id): group for event_id, group in tables.arrivals.groupby("eventId", sort=False)
    }
    sizes: dict[str, int] = {}
    counts = {
        "evidenceFiles": 0,
        "evidenceNoArrivals": 0,
        "evidenceNoData": 0,
        "evidenceTracesDropped": 0,
        "evidenceTracesOverBudget": 0,
        "evidenceTraces": 0,
        "evidenceStationsSkippedNoData": 0,
    }
    for event in ordered:
        arrivals = by_event.get(event.id)
        if arrivals is None or arrivals.empty:
            counts["evidenceNoArrivals"] += 1
            log.info("evidence %s: no arrivals; no file", event.id)
            continue
        built = build_evidence(
            event,
            arrivals,
            stations,
            tables.picks,
            cfg.evidence,
            cfg.rounding,
            source,
            cache_dir,
            no_data_stations=tables.stations_without_data,
        )
        if built is None:
            counts["evidenceNoData"] += 1
            continue
        evidence, data, trace_counts = built
        rel = f"{EVIDENCE_DIR}/{event.id}.json"
        (evidence_dir / f"{event.id}.json").write_bytes(data)
        sizes[rel] = len(data)
        counts["evidenceFiles"] += 1
        counts["evidenceTraces"] += len(evidence.traces)
        counts["evidenceTracesDropped"] += trace_counts["dropped"]
        counts["evidenceTracesOverBudget"] += trace_counts["overBudget"]
        counts["evidenceStationsSkippedNoData"] += trace_counts.get("skippedNoData", 0)
        if event.id == hero_id and len(evidence.traces) < cfg.evidence.maxTraces:
            log.info(
                "evidence %s (hero): %d trace(s), fewer than maxTraces %d",
                event.id,
                len(evidence.traces),
                cfg.evidence.maxTraces,
            )
    if hero_id is not None and f"{EVIDENCE_DIR}/{hero_id}.json" not in sizes:
        log.warning("export: the hero %s has no evidence file; the E key opens nothing", hero_id)
    return sizes, counts


def _build(
    tmp: Path,
    tables: RunTables,
    cfg: ExportConfig,
    section: RunSection,
    mode: ExportMode,
    source: WaveformSource,
    cache_dir: Path,
    features_loader: FeaturesLoader,
    baseline_cfg: BaselineConfig,
) -> tuple[dict[str, int], dict[str, int]]:
    events, catalog = apply_matches(tables.events, tables.catalog, tables.matches)
    events = assign_reveal_order(events)
    hero_id = choose_hero(events, cfg.heroRule)
    summary = analysis_summary(
        tables.run.id, events, catalog, tables.validation, cfg.rounding, baseline_cfg
    )
    meta = BundleMeta(
        mode=mode,
        scene=scene_meta(tables.run.id, section, cfg, hero_id, tables.run.isSynthetic),
        run=tables.run,
        summary=summary,
    )
    features = features_loader(cfg, section)
    stations: list[Station] = list(tables.stations)  # stations.parquet order (H1's)

    sizes: dict[str, int] = {}
    sizes[META_JSON] = dump_json(tmp / META_JSON, meta.model_dump(mode="json"), pretty=True)
    sizes[STATIONS_JSON] = dump_json(
        tmp / STATIONS_JSON, [s.model_dump(mode="json") for s in stations], pretty=True
    )
    sizes[CATALOG_JSON] = dump_json(
        tmp / CATALOG_JSON, [c.model_dump(mode="json") for c in catalog], pretty=True
    )
    sizes[EVENTS_JSON] = dump_json(
        tmp / EVENTS_JSON, [e.model_dump(mode="json") for e in events], pretty=False
    )
    sizes[FEATURES_JSON] = dump_json(
        tmp / FEATURES_JSON, [f.model_dump(mode="json") for f in features], pretty=True
    )
    if tables.validation is not None:
        sizes[VALIDATION_JSON] = dump_json(
            tmp / VALIDATION_JSON, tables.validation.model_dump(mode="json"), pretty=True
        )
    if tables.confidence is not None:
        sizes[CONFIDENCE_JSON] = dump_json(
            tmp / CONFIDENCE_JSON,
            tables.confidence.model_dump(mode="json", by_alias=True),
            pretty=True,
        )
    evidence_dir = tmp / EVIDENCE_DIR
    evidence_dir.mkdir()
    evidence_sizes, counts = _write_evidence(
        tables, events, hero_id, cfg, source, cache_dir, evidence_dir
    )
    sizes.update(evidence_sizes)
    counts = {
        "events": len(events),
        "stations": len(stations),
        "catalogEvents": len(catalog),
        "recovered": summary.recoveredCatalogCount,
        "strict": summary.strictQualityCount,
        "features": len(features),
        "hasValidation": int(tables.validation is not None),
        "hasConfidence": int(tables.confidence is not None),
        "hasHero": int(hero_id is not None),
        **counts,
        "bytes": sum(sizes.values()),
    }
    log.info(
        "export: %s bundle is %.1f MB of the %.1f MB maxBundleBytes budget",
        mode,
        counts["bytes"] / _MIB,
        cfg.maxBundleBytes / _MIB,
    )
    if counts["bytes"] > cfg.maxBundleBytes:
        raise ExportError(
            f"{mode} bundle is {counts['bytes']} bytes > maxBundleBytes {cfg.maxBundleBytes}; "
            "lower evidence.maxEvents (docs/01 budgets the committed bundle at < 30 MB)"
        )
    # Swap in only a bundle that passes the same check anyone can run on the directory.
    check_bundle(
        tmp,
        rounding=cfg.rounding,
        max_evidence_bytes=cfg.evidence.maxFileBytes,
        max_bundle_bytes=cfg.maxBundleBytes,
        mode=mode,
    )
    return sizes, counts


def replace_directory(build: Callable[[Path], Any], out_dir: Path) -> Any:
    """Run ``build(tmp)`` in a fresh sibling of ``out_dir`` and swap it in atomically.

    On success the old ``out_dir`` (if any) is removed after the swap. On any failure the temp
    directory is removed and ``out_dir`` is exactly what it was before.
    """
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{TMP_PREFIX}{out_dir.name}-", dir=out_dir.parent))
    old: Path | None = None
    try:
        result = build(tmp)
        if out_dir.exists():
            old = Path(
                tempfile.mkdtemp(prefix=f"{TMP_PREFIX}{out_dir.name}-old-", dir=out_dir.parent)
            )
            old.rmdir()  # only the unique name is needed; rename wants a free target
            os.rename(out_dir, old)
        os.rename(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        if old is not None and old.exists() and not out_dir.exists():
            os.rename(old, out_dir)  # the swap failed halfway: put the previous bundle back
        raise
    if old is not None:
        shutil.rmtree(old)
    return result


def export_bundle(
    tables: RunTables,
    cfg: ExportConfig,
    section: RunSection,
    mode: ExportMode,
    source: WaveformSource,
    *,
    cache_dir: Path,
    out_dir: Path,
    baseline_cfg: BaselineConfig,
    features_loader: FeaturesLoader = load_features_lazily,
) -> ExportResult:
    """Write the bundle for ``mode`` to ``out_dir`` (replaced atomically) and report counts.
    ``baseline_cfg`` (the run's ``validate.yaml`` baseline section) is the rule for
    ``AnalysisSummary.baseline``."""
    started = time.perf_counter()
    if mode not in cfg.modes:
        raise ExportError(f"mode {mode!r} is not in export.yaml modes {cfg.modes}")
    if tables.run.isSynthetic:
        raise ExportError(
            f"run {tables.run.id} is synthetic; synthetic bundles come only from "
            "scripts/mock-fixture.py (CLAUDE.md rule 5)"
        )
    out_dir = Path(out_dir)
    log.info("export: writing %s bundle for run %s to %s", mode, tables.run.id, out_dir)
    sizes, counts = replace_directory(
        lambda tmp: _build(
            tmp, tables, cfg, section, mode, source, cache_dir, features_loader, baseline_cfg
        ),
        out_dir,
    )
    runtime_s = time.perf_counter() - started
    log.info(
        "export: %s bundle: %d events, %d evidence files (%d traces), %.1f KB total in %.1f s "
        "-> %s",
        mode,
        counts["events"],
        counts["evidenceFiles"],
        counts["evidenceTraces"],
        counts["bytes"] / 1024,
        runtime_s,
        out_dir,
    )
    return ExportResult(out_dir=out_dir, mode=mode, counts=counts, sizes=sizes)

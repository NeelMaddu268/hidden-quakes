"""``check_bundle``: re-validate a written bundle against the contracts and the exporter's own
invariants (ticket API-02 acceptance). It reads only the bundle directory, so it also runs on
the mock bundle and on a bundle fetched from another laptop.

    uv run python -m hq.export apps/web/public/data/showcase [--config-dir configs/showcase]

Checks: every file parses with its Pydantic model and ``schemaVersion`` matches; run ids agree
across meta, events and summary; ``revealOrder`` is a permutation of ``0..n-1`` in Tier A -> B
-> C, time order; every ``AnalysisSummary`` count recomputes from ``events.json`` and
``catalog.json``; catalog matches are one-to-one and agree with ``SeismicEvent.catalogMatch``;
the hero is a Tier A event with the most stations and has evidence; every evidence file is under
the byte cap, names an event in ``events.json``, has at most 16 traces sorted by ``epiDistM``
with samples in ``[-1, 1]``; the whole directory stays under the bundle budget; ``meta.mode``
matches the directory name (or the mode passed in) and only a ``mock`` bundle is synthetic.
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_args

import numpy as np
from hq_contracts.models import (
    SCHEMA_VERSION,
    BaselineGain,
    BundleMeta,
    CatalogEvent,
    DataMode,
    EventEvidence,
    GeoFeature,
    SeismicEvent,
    Station,
    Validation,
)
from pydantic import BaseModel, ValidationError

from hq.config import load_config
from hq.config.export import MAX_EVIDENCE_TRACES, EvidenceConfig, ExportConfig, RoundingConfig
from hq.config.validate import BaselineConfig
from hq.export.files import (
    CATALOG_JSON,
    EVENTS_JSON,
    EVIDENCE_DIR,
    FEATURES_JSON,
    META_JSON,
    STATIONS_JSON,
    VALIDATION_JSON,
)
from hq.export.summary import STRICT_TIER, analysis_summary, reveal_key
from hq.validate.baseline import GAIN_PROFILE, PHASENET, STALTA, baseline_gain, index_rows
from hq.validate.errors import ValidateError

log = logging.getLogger(__name__)

REQUIRED_FILES: tuple[str, ...] = (
    META_JSON,
    STATIONS_JSON,
    CATALOG_JSON,
    EVENTS_JSON,
    FEATURES_JSON,
)
OPTIONAL_FILES: tuple[str, ...] = (VALIDATION_JSON,)
DATA_MODES: tuple[str, ...] = get_args(DataMode.__value__)
MAX_LISTED = 5
SAMPLE_LIMIT = 1.0  # WaveformSnippet.samples are scaled to [-1, 1]
SYNTHETIC_MODE = "mock"  # the only mode whose bundle may carry isSynthetic (CLAUDE.md rule 5)


class BundleCheckError(ValueError):
    """The bundle failed one or more checks; the message lists every one."""

    def __init__(self, bundle_dir: Path, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{bundle_dir}: {len(self.problems)} problem(s):\n{lines}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse(path: Path, model: type[BaseModel], problems: list[str]) -> Any:
    try:
        return model.model_validate(_read_json(path))
    except (ValidationError, json.JSONDecodeError, OSError) as exc:
        problems.append(f"{path.name}: not a valid {model.__name__}: {exc}")
        return None


def _parse_list(path: Path, model: type[BaseModel], problems: list[str]) -> list[Any] | None:
    try:
        raw = _read_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        problems.append(f"{path.name}: unreadable: {exc}")
        return None
    if not isinstance(raw, list):
        problems.append(f"{path.name}: expected a JSON array of {model.__name__}")
        return None
    out: list[Any] = []
    for i, item in enumerate(raw):
        try:
            out.append(model.model_validate(item))
        except ValidationError as exc:
            problems.append(f"{path.name}[{i}]: not a valid {model.__name__}: {exc}")
            return None
    return out


def _listed(items: Sequence[str]) -> str:
    shown = ", ".join(list(items)[:MAX_LISTED])
    return shown + (", ..." if len(items) > MAX_LISTED else "")


def _check_events(
    meta: BundleMeta, events: list[SeismicEvent], problems: list[str]
) -> dict[str, SeismicEvent]:
    ids = [e.id for e in events]
    if len(set(ids)) != len(ids):
        problems.append("events.json: duplicate event ids")
    foreign = [e.id for e in events if e.runId != meta.run.id]
    if foreign:
        problems.append(f"events.json: runId != meta.run.id for {_listed(foreign)}")
    orders = sorted(e.revealOrder for e in events)
    if orders != list(range(len(events))):
        problems.append("events.json: revealOrder is not a permutation of 0..n-1")
    else:
        by_order = sorted(events, key=lambda e: e.revealOrder)
        keys = [reveal_key(e)[:2] for e in by_order]
        if keys != sorted(keys):
            problems.append("events.json: revealOrder is not Tier A -> B -> C, time-ordered within")
    return {e.id: e for e in events}


def _check_catalog(
    catalog: list[CatalogEvent], events: dict[str, SeismicEvent], problems: list[str]
) -> None:
    ids = [c.id for c in catalog]
    if len(set(ids)) != len(ids):
        problems.append("catalog.json: duplicate catalog ids")
    matched = [c for c in catalog if c.matchedEventId is not None]
    unknown = [c.id for c in matched if c.matchedEventId not in events]
    if unknown:
        problems.append(f"catalog.json: matchedEventId not in events.json for {_listed(unknown)}")
    event_ids = [c.matchedEventId for c in matched]
    if len(set(event_ids)) != len(event_ids):
        problems.append("catalog.json: matching is not one-to-one (an event matches twice)")
    for c in matched:
        e = events.get(str(c.matchedEventId))
        if e is not None and (e.catalogMatch is None or e.catalogMatch.catalogId != c.id):
            problems.append(f"catalog {c.id} -> event {c.matchedEventId}, but the event disagrees")
    matched_ids = {c.id for c in matched}
    for e in events.values():
        if e.catalogMatch is not None and e.catalogMatch.catalogId not in matched_ids:
            problems.append(
                f"event {e.id} names catalog {e.catalogMatch.catalogId}, catalog disagrees"
            )


def _check_summary(
    meta: BundleMeta,
    events: list[SeismicEvent],
    catalog: list[CatalogEvent],
    validation: Validation | None,
    rounding: RoundingConfig,
    problems: list[str],
) -> None:
    got = meta.summary
    # No run config here: the claimed gain is checked against its own numbers below instead.
    want = analysis_summary(meta.run.id, events, catalog, validation, rounding, None)
    if got.runId != meta.run.id or meta.scene.runId != meta.run.id:
        problems.append("meta.json: runId differs between run, scene and summary")
    exact = (
        "publicCatalogCount",
        "recoveredCatalogCount",
        "candidateCount",
        "additionalCount",
        "additional",
        "strictQualityCount",
        "strictAdditionalCount",
        "medianStations",
    )
    for name in exact:
        if getattr(got, name) != getattr(want, name):
            problems.append(
                f"summary.{name} = {getattr(got, name)!r}, recomputed {getattr(want, name)!r}"
            )
    if sorted(got.unmatchedPublicIds) != sorted(want.unmatchedPublicIds):
        problems.append("summary.unmatchedPublicIds differ from the catalog's unmatched ids")
    for name, decimals in (("recall", rounding.recall), ("medianRmsS", rounding.rmsS)):
        tolerance = 0.5 * 10.0**-decimals + 1e-12
        if abs(getattr(got, name) - getattr(want, name)) > tolerance:
            problems.append(
                f"summary.{name} = {getattr(got, name)}, recomputed {getattr(want, name)}"
            )
    if got.baseline is not None:
        _check_baseline_claim(got.baseline, validation, rounding, problems)


def _check_baseline_claim(
    claim: BaselineGain,
    validation: Validation | None,
    rounding: RoundingConfig,
    problems: list[str],
) -> None:
    """A claimed ``summary.baseline`` must come from the bundle's own ``validation.baseline``
    rows: the two strict counts are the ``full`` rows' Tier A counts, and the shared rule
    (``hq.validate.baseline.baseline_gain`` with the default ``minGain``, the docs/lanes floor)
    must claim a gain too, so a claim with no or partial rows, mismatched counts or a
    ``p_only`` profile without gain is a problem."""
    rows = validation.baseline if validation is not None else []
    if not rows:
        problems.append("summary.baseline claims a gain but validation.json has no baseline rows")
        return
    if claim.associationProfile != GAIN_PROFILE:
        problems.append(
            f"summary.baseline.associationProfile = {claim.associationProfile!r}, the rule quotes "
            f"{GAIN_PROFILE!r}"
        )
    try:
        indexed = index_rows(rows)
    except ValidateError as exc:
        problems.append(f"validation.baseline: {exc}")
        return
    for method, field in ((PHASENET, "strictPhasenet"), (STALTA, "strictStalta")):
        row = indexed.get((method, GAIN_PROFILE))
        if row is None:
            problems.append(f"summary.baseline claims a gain but validation.baseline has no "
                            f"({method}, {GAIN_PROFILE}) row")  # fmt: skip
        elif row.tiers.A != getattr(claim, field):
            problems.append(
                f"summary.baseline.{field} = {getattr(claim, field)}, but the ({method}, "
                f"{GAIN_PROFILE}) row has {row.tiers.A} Tier {STRICT_TIER} events"
            )
    if baseline_gain(rows, BaselineConfig()) is None:
        problems.append(
            "summary.baseline claims a gain that validation.baseline does not support under the "
            "shared rule (gain must hold in both profiles with STA/LTA Tier A events in each)"
        )
    if claim.strictStalta > 0:
        tolerance = 0.5 * 10.0**-rounding.gain + 1e-12
        if abs(claim.strictPhasenet / claim.strictStalta - claim.gain) > tolerance:
            problems.append("summary.baseline.gain != strictPhasenet / strictStalta")


def _check_hero(
    meta: BundleMeta, events: dict[str, SeismicEvent], evidence_ids: set[str], problems: list[str]
) -> None:
    hero_id = meta.scene.heroEventId
    strict = [e for e in events.values() if e.tier == STRICT_TIER]
    if hero_id is None:
        if strict:
            problems.append("scene.heroEventId is null although Tier A events exist")
        return
    hero = events.get(hero_id)
    if hero is None:
        problems.append(f"scene.heroEventId {hero_id} is not in events.json")
        return
    if hero.tier != STRICT_TIER:
        problems.append(f"hero {hero_id} is Tier {hero.tier}, not {STRICT_TIER}")
    best = max(e.quality.nStations for e in strict) if strict else 0
    if hero.quality.nStations != best:
        problems.append(
            f"hero {hero_id} has {hero.quality.nStations} stations; a Tier A event has {best}"
        )
    if hero_id not in evidence_ids:
        problems.append(f"hero {hero_id} has no evidence file")


def _check_evidence(
    bundle_dir: Path,
    events: dict[str, SeismicEvent],
    stations: list[Station],
    max_bytes: int,
    problems: list[str],
) -> set[str]:
    evidence_dir = bundle_dir / EVIDENCE_DIR
    if not evidence_dir.is_dir():
        problems.append(f"{EVIDENCE_DIR}/ is missing")
        return set()
    station_ids = {s.id for s in stations}
    seen: set[str] = set()
    for path in sorted(evidence_dir.iterdir()):
        name = f"{EVIDENCE_DIR}/{path.name}"
        if path.suffix != ".json" or not path.is_file():
            problems.append(f"{name}: not an evidence file")
            continue
        size = path.stat().st_size
        if size >= max_bytes:
            problems.append(f"{name}: {size} bytes, not under {max_bytes}")
        evidence = _parse(path, EventEvidence, problems)
        if evidence is None:
            continue
        if evidence.eventId != path.stem:
            problems.append(f"{name}: eventId {evidence.eventId} != file name")
        if evidence.eventId not in events:
            problems.append(f"{name}: event {evidence.eventId} is not in events.json")
        seen.add(evidence.eventId)
        if not 1 <= len(evidence.traces) <= MAX_EVIDENCE_TRACES:
            problems.append(f"{name}: {len(evidence.traces)} traces, want 1..{MAX_EVIDENCE_TRACES}")
        dists = [t.epiDistM for t in evidence.traces]
        if dists != sorted(dists):
            problems.append(f"{name}: traces are not sorted by epiDistM")
        trace_stations = [t.stationId for t in evidence.traces]
        if len(set(trace_stations)) != len(trace_stations):
            problems.append(f"{name}: a station appears twice")
        for t in evidence.traces:
            if t.stationId not in station_ids:
                problems.append(f"{name}: station {t.stationId} is not in stations.json")
            if not t.samples:
                problems.append(f"{name}: {t.stationId} has no samples")
                continue
            arr = np.asarray(t.samples, dtype=np.float64)
            if not np.all(np.isfinite(arr)) or np.max(np.abs(arr)) > SAMPLE_LIMIT:
                problems.append(f"{name}: {t.stationId} samples leave [-1, 1] or are not finite")
            if t.dt <= 0.0:
                problems.append(f"{name}: {t.stationId} has dt {t.dt}")
    return seen


def check_bundle(
    bundle_dir: Path,
    *,
    rounding: RoundingConfig | None = None,
    max_evidence_bytes: int | None = None,
    max_bundle_bytes: int | None = None,
    mode: str | None = None,
) -> dict[str, int]:
    """Check every file in ``bundle_dir``; raise ``BundleCheckError`` listing every problem.

    ``rounding`` sets the tolerance for the summary's rounded floats, ``max_evidence_bytes``
    the per-file evidence cap and ``max_bundle_bytes`` the whole-directory cap; each defaults to
    the ``export.yaml`` default. ``mode`` is what ``meta.mode`` must be; by default the bundle
    directory's name. Returns counts of what was checked.
    """
    bundle_dir = Path(bundle_dir)
    rounding = rounding or RoundingConfig()
    max_bytes = (
        max_evidence_bytes if max_evidence_bytes is not None else EvidenceConfig().maxFileBytes
    )
    bundle_budget = (
        max_bundle_bytes if max_bundle_bytes is not None else ExportConfig().maxBundleBytes
    )
    expected_mode = mode if mode is not None else bundle_dir.name
    problems: list[str] = []
    if not bundle_dir.is_dir():
        raise BundleCheckError(bundle_dir, [f"{bundle_dir} is not a directory"])
    missing = [name for name in REQUIRED_FILES if not (bundle_dir / name).is_file()]
    if missing:
        raise BundleCheckError(bundle_dir, [f"missing files: {missing}"])
    extra = sorted(
        p.name
        for p in bundle_dir.iterdir()
        if p.name not in (*REQUIRED_FILES, *OPTIONAL_FILES, EVIDENCE_DIR)
        and not p.name.startswith(".")  # a Finder .DS_Store is not part of the bundle
    )
    if extra:
        problems.append(f"unexpected entries: {extra}")

    meta = _parse(bundle_dir / META_JSON, BundleMeta, problems)
    stations = _parse_list(bundle_dir / STATIONS_JSON, Station, problems)
    catalog = _parse_list(bundle_dir / CATALOG_JSON, CatalogEvent, problems)
    events = _parse_list(bundle_dir / EVENTS_JSON, SeismicEvent, problems)
    features = _parse_list(bundle_dir / FEATURES_JSON, GeoFeature, problems)
    validation: Validation | None = None
    if (bundle_dir / VALIDATION_JSON).is_file():
        validation = _parse(bundle_dir / VALIDATION_JSON, Validation, problems)
    if meta is None or stations is None or catalog is None or events is None or features is None:
        raise BundleCheckError(bundle_dir, problems)

    if meta.schemaVersion != SCHEMA_VERSION:
        problems.append(f"meta.schemaVersion {meta.schemaVersion!r} != {SCHEMA_VERSION!r}")
    if meta.mode not in DATA_MODES:
        problems.append(f"meta.mode {meta.mode!r} is not a DataMode")
    if meta.mode != expected_mode:
        problems.append(f"meta.mode {meta.mode!r} != {expected_mode!r} (the bundle directory)")
    if meta.mode != SYNTHETIC_MODE and (meta.run.isSynthetic or meta.scene.isSynthetic):
        problems.append(
            f"meta.mode {meta.mode!r} carries isSynthetic; only a {SYNTHETIC_MODE!r} bundle may"
        )
    total_bytes = sum(p.stat().st_size for p in bundle_dir.rglob("*") if p.is_file())
    if total_bytes > bundle_budget:
        problems.append(f"bundle is {total_bytes} bytes > maxBundleBytes {bundle_budget}")
    station_ids = [s.id for s in stations]
    if len(set(station_ids)) != len(station_ids):
        problems.append("stations.json: duplicate station ids")
    feature_ids = [f.id for f in features]
    if len(set(feature_ids)) != len(feature_ids):
        problems.append("features.json: duplicate feature ids")
    events_by_id = _check_events(meta, events, problems)
    _check_catalog(catalog, events_by_id, problems)
    _check_summary(meta, events, catalog, validation, rounding, problems)
    evidence_ids = _check_evidence(bundle_dir, events_by_id, stations, max_bytes, problems)
    _check_hero(meta, events_by_id, evidence_ids, problems)
    if problems:
        raise BundleCheckError(bundle_dir, problems)
    counts = {
        "stations": len(stations),
        "catalogEvents": len(catalog),
        "events": len(events),
        "features": len(features),
        "evidenceFiles": len(evidence_ids),
        "hasValidation": int(validation is not None),
        "bytes": total_bytes,
    }
    log.info(
        "check_bundle %s: ok, %d bytes of %d maxBundleBytes (%s)",
        bundle_dir,
        total_bytes,
        bundle_budget,
        counts,
    )
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Hidden Quakes data bundle directory.")
    parser.add_argument("bundle_dir", type=Path, help="e.g. apps/web/public/data/showcase")
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="config directory whose export.yaml sets the rounding and byte caps to check "
        "against (default: the ExportConfig defaults)",
    )
    args = parser.parse_args(argv)
    kwargs: dict[str, Any] = {}
    if args.config_dir is not None:
        cfg = load_config(args.config_dir).export  # a bad config is a loud error, not a default
        kwargs = {
            "rounding": cfg.rounding,
            "max_evidence_bytes": cfg.evidence.maxFileBytes,
            "max_bundle_bytes": cfg.maxBundleBytes,
        }
    try:
        counts = check_bundle(args.bundle_dir, **kwargs)
    except BundleCheckError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"{args.bundle_dir}: ok " + " ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

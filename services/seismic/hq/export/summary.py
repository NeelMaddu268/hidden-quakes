"""What the exporter derives from the run tables: reveal order, the hero, catalog matches,
``AnalysisSummary`` and ``SceneMeta`` (docs/02 §1). Every number here is computed from rows;
nothing is typed in.
"""

import logging
from typing import get_args

import numpy as np
import pandas as pd
from hq_contracts.models import (
    AnalysisSummary,
    BaselineGain,
    BaselineRow,
    CatalogEvent,
    CatalogMatch,
    SceneMeta,
    SeismicEvent,
    Tier,
    TierCounts,
    Validation,
)

from hq.config.export import ExportConfig, HeroRule, RoundingConfig
from hq.config.run import RunSection
from hq.export.errors import ExportError
from hq.export.tables import str_or_none

log = logging.getLogger(__name__)

TIERS: tuple[str, ...] = get_args(Tier.__value__)  # ("A", "B", "C"): reveal order of tiers
TIER_RANK: dict[str, int] = {tier: rank for rank, tier in enumerate(TIERS)}
STRICT_TIER = TIERS[0]  # "strict" in the summary and the UI means Tier A
# SceneMeta.projection: the ENU convention every lane shares (docs/01 -> Conventions, docs/02 §1).
PROJECTION = "EPSG:32612 minus origin"
BASELINE_PROFILES: tuple[str, ...] = ("full", "p_only")
GAIN_PROFILE = "full"  # the profile whose gain the summary quotes


def rnd(value: float, decimals: int) -> float:
    """Round to ``decimals`` and normalise ``-0.0`` to ``0.0``."""
    return round(float(value), decimals) + 0.0


def reveal_key(event: SeismicEvent) -> tuple[int, float, str]:
    """Tier A -> B -> C, time-ordered within a tier, id as the final tie-break."""
    return (TIER_RANK[event.tier], event.t, event.id)


def assign_reveal_order(events: list[SeismicEvent]) -> list[SeismicEvent]:
    """Events sorted by ``(t, id)`` with ``revealOrder`` 0..n-1 in ``reveal_key`` order."""
    order = {e.id: i for i, e in enumerate(sorted(events, key=reveal_key))}
    return [
        e.model_copy(update={"revealOrder": order[e.id]})
        for e in sorted(events, key=lambda e: (e.t, e.id))
    ]


def choose_hero(events: list[SeismicEvent], rule: HeroRule) -> str | None:
    """``SceneMeta.heroEventId`` by ``ExportConfig.heroRule``; None when no event qualifies."""
    if rule != "tierA_most_stations":  # pragma: no cover - the Literal admits one value
        raise ExportError(f"unknown heroRule {rule!r}")
    strict = [e for e in events if e.tier == STRICT_TIER]
    if not strict:
        log.warning("export: no Tier %s event; SceneMeta.heroEventId is null", STRICT_TIER)
        return None
    # Most stations; ties: lowest rmsS, then earliest, then id, so the choice is deterministic.
    hero = min(strict, key=lambda e: (-e.quality.nStations, e.quality.rmsS, e.t, e.id))
    log.info(
        "export: hero %s (Tier %s, %d stations, rmsS %.3f)",
        hero.id,
        hero.tier,
        hero.quality.nStations,
        hero.quality.rmsS,
    )
    return hero.id


def apply_matches(
    events: list[SeismicEvent], catalog: list[CatalogEvent], matches: pd.DataFrame
) -> tuple[list[SeismicEvent], list[CatalogEvent]]:
    """Fill ``CatalogEvent.matchedEventId`` and ``SeismicEvent.catalogMatch`` from
    ``matches.parquet`` (one row per public event), checking that the three agree.

    ``matches`` is the table of record. An event whose ``catalogMatch`` is null but whose id a
    match names gets it filled (counted and logged); one that names a different public event is
    an error, as is a match naming an unknown event or catalog id, or two rows per catalog id.
    """
    events_by_id = {e.id: e for e in events}
    catalog_ids = {c.id for c in catalog}
    matched: dict[str, tuple[str, float, float]] = {}  # catalogId -> (eventId, dtS, distM)
    for row in matches.to_dict("records"):
        catalog_id = str(row["catalogId"])
        if catalog_id not in catalog_ids:
            raise ExportError(f"matches.parquet names unknown catalog id {catalog_id!r}")
        if catalog_id in matched:
            raise ExportError(f"matches.parquet has several rows for catalog id {catalog_id!r}")
        event_id = str_or_none(row["eventId"])
        if event_id is None:
            continue
        if event_id not in events_by_id:
            raise ExportError(
                f"matches.parquet matches {catalog_id!r} to unknown event {event_id!r}"
            )
        matched[catalog_id] = (event_id, float(row["dtS"]), float(row["distM"]))
    by_event: dict[str, str] = {}
    for catalog_id, (event_id, _, _) in matched.items():
        if event_id in by_event:
            raise ExportError(
                f"matches.parquet is not one-to-one: event {event_id!r} matches both "
                f"{by_event[event_id]!r} and {catalog_id!r}"
            )
        by_event[event_id] = catalog_id

    filled = 0
    out_events: list[SeismicEvent] = []
    for e in events:
        catalog_id = by_event.get(e.id)
        if e.catalogMatch is None and catalog_id is not None:
            _, dt_s, dist_m = matched[catalog_id]
            e = e.model_copy(
                update={"catalogMatch": CatalogMatch(catalogId=catalog_id, dtS=dt_s, distM=dist_m)}
            )
            filled += 1
        elif e.catalogMatch is not None and e.catalogMatch.catalogId != catalog_id:
            raise ExportError(
                f"event {e.id} carries catalogMatch {e.catalogMatch.catalogId!r} but "
                f"matches.parquet says {catalog_id!r}"
            )
        out_events.append(e)
    if filled:
        log.warning(
            "export: filled catalogMatch on %d event(s) from matches.parquet (events.parquet "
            "had null; the tier stage should set it)",
            filled,
        )
    out_catalog = [
        c.model_copy(update={"matchedEventId": matched[c.id][0] if c.id in matched else None})
        for c in catalog
    ]
    log.info("export: %d of %d public events matched", len(matched), len(catalog))
    return out_events, out_catalog


def tier_counts(events: list[SeismicEvent]) -> TierCounts:
    counts = {tier: 0 for tier in TIERS}
    for e in events:
        counts[e.tier] += 1
    return TierCounts(**counts)


def baseline_gain(validation: Validation | None, rounding: RoundingConfig) -> BaselineGain | None:
    """``AnalysisSummary.baseline``: the strict-count gain of PhaseNet over STA/LTA, quoted
    only when both association profiles are in the table and the gain exceeds 1 in both."""
    if validation is None or not validation.baseline:
        return None
    rows: dict[tuple[str, str], BaselineRow] = {}
    for row in validation.baseline:
        key = (row.method, row.associationProfile)
        if key in rows:
            raise ExportError(f"validation.baseline has several rows for {key}")
        rows[key] = row
    needed = [(m, p) for p in BASELINE_PROFILES for m in ("phasenet", "stalta")]
    missing = [key for key in needed if key not in rows]
    if missing:
        log.warning("export: baseline table lacks %s; no gain claimed", missing)
        return None
    gains: dict[str, float] = {}
    for profile in BASELINE_PROFILES:
        strict_stalta = rows[("stalta", profile)].tiers.A
        strict_phasenet = rows[("phasenet", profile)].tiers.A
        if strict_stalta == 0:
            log.warning(
                "export: STA/LTA has no Tier A events in profile %s; no gain claimed", profile
            )
            return None
        gains[profile] = strict_phasenet / strict_stalta
    if not all(g > 1.0 for g in gains.values()):
        log.info("export: PhaseNet gain does not hold in every profile (%s); not claimed", gains)
        return None
    return BaselineGain(
        associationProfile=GAIN_PROFILE,
        strictPhasenet=rows[("phasenet", GAIN_PROFILE)].tiers.A,
        strictStalta=rows[("stalta", GAIN_PROFILE)].tiers.A,
        gain=rnd(gains[GAIN_PROFILE], rounding.gain),
    )


def analysis_summary(
    run_id: str,
    events: list[SeismicEvent],
    catalog: list[CatalogEvent],
    validation: Validation | None,
    rounding: RoundingConfig,
) -> AnalysisSummary:
    """Every count from the event and catalog lists (``check_bundle`` recomputes them)."""
    public = len(catalog)
    recovered = sum(1 for c in catalog if c.matchedEventId is not None)
    additional = [e for e in events if e.catalogMatch is None]
    if public == 0:
        log.warning("export: the public catalog is empty for this window; recall is 0")
    if not events:
        log.warning("export: no candidate events; medians are 0")
    return AnalysisSummary(
        runId=run_id,
        publicCatalogCount=public,
        recoveredCatalogCount=recovered,
        recall=rnd(recovered / public, rounding.recall) if public else 0.0,
        unmatchedPublicIds=[c.id for c in catalog if c.matchedEventId is None],
        candidateCount=len(events),
        additionalCount=len(additional),
        additional=tier_counts(additional),
        strictQualityCount=sum(1 for e in events if e.tier == STRICT_TIER),
        strictAdditionalCount=sum(1 for e in additional if e.tier == STRICT_TIER),
        medianStations=float(np.median([e.quality.nStations for e in events])) if events else 0.0,
        medianRmsS=(
            rnd(float(np.median([e.quality.rmsS for e in events])), rounding.rmsS)
            if events
            else 0.0
        ),
        baseline=baseline_gain(validation, rounding),
    )


def scene_meta(
    run_id: str, section: RunSection, cfg: ExportConfig, hero_id: str | None, is_synthetic: bool
) -> SceneMeta:
    """``SceneMeta`` from ``run.yaml`` (origin, reference surface) and ``export.yaml``."""
    return SceneMeta(
        runId=run_id,
        originLat=section.origin.lat,
        originLon=section.origin.lon,
        originElevM=section.origin.elevM,
        refSurfaceElevM=section.refSurfaceElevM,
        projection=PROJECTION,
        verticalExaggeration=cfg.scene.verticalExaggeration,
        depthLabel=cfg.scene.depthLabel.format(refSurfaceElevM=section.refSurfaceElevM),
        heroEventId=hero_id,
        isSynthetic=is_synthetic,
    )

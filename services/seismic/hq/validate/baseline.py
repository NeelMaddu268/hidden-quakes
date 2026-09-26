"""Baseline comparison (VAL-01): PhaseNet picks against STA/LTA picks, each through H2's
``associate -> locate -> match -> assign_tiers`` in every configured association profile, with
the run's own ``SeismologyConfig`` (docs/02 §5). One ``BaselineRow`` per picker x profile.

Every input is a stored table (``picks.parquet``, ``picks_stalta.parquet``, ``stations.parquet``,
``catalog.parquet``) and the stored config; nothing is drawn at random, so the table reproduces
byte for byte from the same run directory.

Row fields, exactly:

- ``candidates``: rows of the final events table ``assign_tiers`` returns (every associated and
  located candidate event);
- ``recoveredPublic``: rows of ``match``'s table (one per public event) whose ``eventId`` is not
  null and names one of those final events;
- ``tiers``: how many final events carry tier A, B and C;
- ``medianRmsS``: median of the final events' ``quality.rmsS`` (``quality_rmsS`` column);
- ``medianStations``: median of their ``quality.nStations``; both are ``0.0`` with no events,
  as ``AnalysisSummary`` does.

``baseline_gain`` is the one rule for ``AnalysisSummary.baseline`` (the exporter imports it):
the gain is quoted on the ``full`` profile and claimed only when, in both ``full`` and
``p_only``, STA/LTA has at least one Tier A event and PhaseNet's Tier A count divided by
STA/LTA's exceeds ``minGain``. The docs/03 baseline kill switch (STA/LTA within
``comparableFraction`` of PhaseNet's strict count) is logged, never decides.
"""

import logging
import time
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
from hq_contracts.models import BaselineGain, BaselineRow, TierCounts

from hq.config.run import RunSection
from hq.config.validate import ASSOCIATION_PROFILES, AssociationProfile, BaselineConfig
from hq.validate.errors import ValidateError
from hq.validate.lanes import SeismologyApi
from hq.validate.null_test import (
    PICK_COLUMNS,
    STATION_ID_COLUMN,
    STRICT_TIER,
    TIER_COLUMN,
    _require_columns,
    select_profile,
)

log = logging.getLogger(__name__)

Method = Literal["phasenet", "stalta"]
METHODS: tuple[Method, ...] = get_args(BaselineRow.model_fields["method"].annotation)
PHASENET: Method = "phasenet"
STALTA: Method = "stalta"
GAIN_PROFILE: AssociationProfile = "full"  # the profile whose gain the summary quotes
TIERS: tuple[str, ...] = tuple(TierCounts.model_fields)  # ("A", "B", "C")
RMS_COLUMN = "quality_rmsS"
STATIONS_COLUMN = "quality_nStations"
EVENT_ID_COLUMN = "id"
MATCH_EVENT_COLUMN = "eventId"
FINAL_EVENT_COLUMNS: tuple[str, ...] = (EVENT_ID_COLUMN, TIER_COLUMN, RMS_COLUMN, STATIONS_COLUMN)


def rerun_tables(
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    catalog: pd.DataFrame,
    api: SeismologyApi,
    seismology_cfg: Any,
    run: RunSection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``associate -> locate -> match -> assign_tiers`` on ``picks``; returns the final events
    table and the matches table. A step that yields no events ends the rerun with two empty
    frames, as ``null_test.rerun_pipeline`` does. API errors propagate unchanged."""
    empty = (
        pd.DataFrame(columns=list(FINAL_EVENT_COLUMNS)),
        pd.DataFrame(columns=["catalogId", MATCH_EVENT_COLUMN]),
    )
    assoc = api.associate(picks, stations, seismology_cfg, run)
    if len(assoc.events) == 0:
        return empty
    located = api.locate(assoc, picks, stations, seismology_cfg, run)
    if len(located.events) == 0:
        return empty
    matched = api.match(located.events, catalog, seismology_cfg)
    tiered = api.assign_tiers(located.events, matched.matches, seismology_cfg)
    _require_columns(tiered.events, FINAL_EVENT_COLUMNS, "assign_tiers events (docs/02 §2)")
    _require_columns(matched.matches, (MATCH_EVENT_COLUMN,), "match matches (docs/02 §2)")
    return tiered.events, matched.matches


def summarize_row(
    method: Method, profile: AssociationProfile, events: pd.DataFrame, matches: pd.DataFrame
) -> BaselineRow:
    """A ``BaselineRow`` from a rerun's final events and matches (definitions in the module
    docstring)."""
    tiers = events[TIER_COLUMN].astype(str) if len(events) else pd.Series([], dtype=str)
    unknown = sorted(set(tiers) - set(TIERS))
    if unknown:
        raise ValidateError(f"assign_tiers returned tiers {unknown}; docs/02 admits {TIERS}")
    event_ids = set(events[EVENT_ID_COLUMN].astype(str)) if len(events) else set()
    matched_ids = matches[MATCH_EVENT_COLUMN].dropna().astype(str)
    stray = sorted(set(matched_ids) - event_ids)
    if stray:
        log.warning(
            "baseline %s/%s: %d match(es) name events absent from the final table (%s); not "
            "counted as recovered",
            method,
            profile,
            len(stray),
            ", ".join(stray[:3]),
        )
    return BaselineRow(
        method=method,
        associationProfile=profile,
        candidates=len(events),
        recoveredPublic=int(matched_ids.isin(event_ids).sum()),
        tiers=TierCounts(**{tier: int((tiers == tier).sum()) for tier in TIERS}),
        medianRmsS=(
            float(np.median(events[RMS_COLUMN].to_numpy(dtype=np.float64))) if len(events) else 0.0
        ),
        medianStations=(
            float(np.median(events[STATIONS_COLUMN].to_numpy(dtype=np.float64)))
            if len(events)
            else 0.0
        ),
    )


def run_baseline(
    picks_phasenet: pd.DataFrame,
    picks_stalta: pd.DataFrame,
    stations: pd.DataFrame,
    catalog: pd.DataFrame,
    api: SeismologyApi,
    seismology_cfg: Any,
    run: RunSection,
    cfg: BaselineConfig,
) -> list[BaselineRow]:
    """The baseline table: for each method (PhaseNet, then STA/LTA) and each profile in
    ``cfg.profiles`` (in that order), the picks the profile selects go through the pipeline."""
    _require_columns(stations, (STATION_ID_COLUMN,), "stations")
    known = set(stations[STATION_ID_COLUMN].astype(str))
    for name, picks in ((PHASENET, picks_phasenet), (STALTA, picks_stalta)):
        _require_columns(picks, PICK_COLUMNS, f"{name} picks")
        unknown = sorted(set(picks["stationId"].astype(str)) - known)
        if unknown:
            raise ValidateError(
                f"{name} picks name stations missing from stations table: {unknown}"
            )
    log.info(
        "baseline: %d phasenet picks, %d stalta picks, %d stations, %d catalog events, profiles %s",
        len(picks_phasenet),
        len(picks_stalta),
        len(stations),
        len(catalog),
        list(cfg.profiles),
    )
    rows: list[BaselineRow] = []
    for method, picks in ((PHASENET, picks_phasenet), (STALTA, picks_stalta)):
        for profile in cfg.profiles:
            started = time.perf_counter()
            selected = select_profile(picks, profile)
            events, matches = rerun_tables(selected, stations, catalog, api, seismology_cfg, run)
            row = summarize_row(method, profile, events, matches)
            rows.append(row)
            log.info(
                "baseline %s/%s: %d picks -> %d candidate events, %d of %d public recovered, "
                "tiers A %d / B %d / C %d, median rmsS %.3f s, median stations %.1f, %.1f s",
                method,
                profile,
                len(selected),
                row.candidates,
                row.recoveredPublic,
                len(catalog),
                row.tiers.A,
                row.tiers.B,
                row.tiers.C,
                row.medianRmsS,
                row.medianStations,
                time.perf_counter() - started,
            )
    return rows


def index_rows(rows: list[BaselineRow]) -> dict[tuple[str, str], BaselineRow]:
    """Rows by ``(method, associationProfile)``; a repeated pair is an error."""
    indexed: dict[tuple[str, str], BaselineRow] = {}
    for row in rows:
        key = (row.method, row.associationProfile)
        if key in indexed:
            raise ValidateError(f"baseline table has several rows for {key}")
        indexed[key] = row
    return indexed


def baseline_gain(rows: list[BaselineRow], cfg: BaselineConfig) -> BaselineGain | None:
    """``AnalysisSummary.baseline`` from the table, or ``None`` when the claim does not hold
    (module docstring). The gain is not rounded here; the exporter rounds for display."""
    if not rows:
        return None
    indexed = index_rows(rows)
    needed = [(m, p) for p in ASSOCIATION_PROFILES for m in METHODS]
    missing = [key for key in needed if key not in indexed]
    if missing:
        log.warning(
            "baseline: table lacks %s; no gain claimed (both profiles %s are needed)",
            missing,
            ASSOCIATION_PROFILES,
        )
        return None
    gains: dict[str, float] = {}
    for profile in ASSOCIATION_PROFILES:
        phasenet, stalta = indexed[(PHASENET, profile)], indexed[(STALTA, profile)]
        strict_phasenet, strict_stalta = phasenet.tiers.A, stalta.tiers.A
        if strict_phasenet > 0 and (
            abs(strict_phasenet - strict_stalta) <= cfg.comparableFraction * strict_phasenet
        ):
            log.warning(
                "baseline kill switch (docs/03): in profile %s STA/LTA's Tier %s count (%d, "
                "median rmsS %.3f s) is within %.0f%% of PhaseNet's (%d, median rmsS %.3f s); "
                "drop the neural-advantage claim and keep the table as context",
                profile,
                STRICT_TIER,
                strict_stalta,
                stalta.medianRmsS,
                cfg.comparableFraction * 100.0,
                strict_phasenet,
                phasenet.medianRmsS,
            )
        if strict_stalta == 0:
            log.warning(
                "baseline: STA/LTA has no Tier %s event in profile %s; no gain claimed",
                STRICT_TIER,
                profile,
            )
            return None
        gains[profile] = strict_phasenet / strict_stalta
    if not all(g > cfg.minGain for g in gains.values()):
        log.info(
            "baseline: PhaseNet gain does not exceed %g in every profile (%s); not claimed",
            cfg.minGain,
            gains,
        )
        return None
    gain = BaselineGain(
        associationProfile=GAIN_PROFILE,
        strictPhasenet=indexed[(PHASENET, GAIN_PROFILE)].tiers.A,
        strictStalta=indexed[(STALTA, GAIN_PROFILE)].tiers.A,
        gain=gains[GAIN_PROFILE],
    )
    log.info(
        "baseline: PhaseNet gain %.2f on %s (Tier %s %d vs %d); holds in every profile (%s)",
        gain.gain,
        GAIN_PROFILE,
        STRICT_TIER,
        gain.strictPhasenet,
        gain.strictStalta,
        gains,
    )
    return gain

"""Null test (VAL-02): how many candidate events the pipeline finds on picks that carry no
inter-station coherence.

For each of ``nShuffles`` reruns, every station's picks are moved together by one draw from
``uniform(-shiftS, +shiftS)`` seconds (one draw per station per rerun, from
``numpy.random.default_rng([seed, i])``), so the P/S structure inside a station survives and
only the coherence between stations is destroyed. H2's ``associate -> locate -> match ->
assign_tiers`` then run on the shifted picks with the run's own ``SeismologyConfig``. The
number of final events and of Tier A events per rerun are the "chance" counts; ``NullTest``
carries their mean, the sample standard deviation (``n - 1``) of the event count, and the
configuration that produced them. Same seed, same numbers.

Pick ids are left as they are: they are the keys ``assoc_picks`` and ``arrivals`` refer to, and
a shifted pick is still the same pick.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from hq_contracts.models import NullTest

from hq.config.run import RunSection
from hq.config.validate import AssociationProfile, NullTestConfig
from hq.validate.errors import ValidateError
from hq.validate.lanes import SeismologyApi

log = logging.getLogger(__name__)

PICK_COLUMNS: tuple[str, ...] = ("id", "stationId", "phase", "t")  # what the shuffle touches
STATION_ID_COLUMN = "id"
TIER_COLUMN = "tier"
STRICT_TIER = "A"
PHASE_P = "P"


@dataclass(frozen=True)
class ShuffleOutcome:
    """One rerun: the per-station shifts it used and what the pipeline found on them."""

    index: int
    shifts_s: pd.Series  # station id -> shift, seconds
    n_events: int
    n_strict: int
    runtime_s: float


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValidateError(f"{what} lacks columns {missing}; docs/02 §2 lists {list(columns)}")


def select_profile(picks: pd.DataFrame, profile: AssociationProfile) -> pd.DataFrame:
    """The picks an association profile feeds to ``associate``: every pick for ``full``, P
    picks only for ``p_only``."""
    if profile == "full":
        return picks
    if profile == "p_only":
        # A fresh RangeIndex, so H2's code sees the same frame shape a real picks table has.
        return picks[picks["phase"].astype(str) == PHASE_P].reset_index(drop=True)
    raise ValidateError(f"unknown association profile {profile!r}")  # unreachable via config


def station_shifts(station_ids: list[str], rng: np.random.Generator, shift_s: float) -> pd.Series:
    """One ``uniform(-shift_s, +shift_s)`` draw per station, in the given station order."""
    if len(set(station_ids)) != len(station_ids):
        raise ValueError("station_ids must be unique")
    values = rng.uniform(-shift_s, shift_s, size=len(station_ids))
    return pd.Series(values, index=pd.Index(station_ids, name="stationId"), name="shiftS")


def shift_picks(picks: pd.DataFrame, shifts: pd.Series) -> pd.DataFrame:
    """A copy of ``picks`` with ``t`` moved by its station's shift; every station must have one."""
    ids = picks["stationId"].astype(str)
    unknown = sorted(set(ids) - set(shifts.index))
    if unknown:
        raise ValidateError(f"no shift drawn for station(s) {unknown}")
    shifted = picks.copy()
    shifted["t"] = picks["t"].to_numpy(dtype=np.float64) + shifts.reindex(ids).to_numpy(
        dtype=np.float64
    )
    return shifted


def _count_strict(events: pd.DataFrame) -> int:
    if TIER_COLUMN not in events.columns:
        raise ValidateError(
            f"assign_tiers returned events without a {TIER_COLUMN!r} column; docs/02 §2 "
            "events.parquet holds SeismicEvent rows (owner: H2 Seismology)"
        )
    return int((events[TIER_COLUMN].astype(str) == STRICT_TIER).sum())


def rerun_pipeline(
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    catalog: pd.DataFrame,
    api: SeismologyApi,
    seismology_cfg: Any,
    run: RunSection,
) -> tuple[int, int]:
    """``associate -> locate -> match -> assign_tiers`` on ``picks``; returns (events, Tier A).

    A step that yields no events ends the rerun with (0, 0): there is nothing to locate, match
    or tier, and the later steps are not asked to handle an empty frame. Any error the API
    raises propagates unchanged.
    """
    assoc = api.associate(picks, stations, seismology_cfg, run)
    if len(assoc.events) == 0:
        return 0, 0
    located = api.locate(assoc, picks, stations, seismology_cfg, run)
    if len(located.events) == 0:
        return 0, 0
    matched = api.match(located.events, catalog, seismology_cfg)
    tiered = api.assign_tiers(located.events, matched.matches, seismology_cfg)
    return len(tiered.events), _count_strict(tiered.events)


def null_shuffles(
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    catalog: pd.DataFrame,
    api: SeismologyApi,
    seismology_cfg: Any,
    run: RunSection,
    cfg: NullTestConfig,
) -> list[ShuffleOutcome]:
    """Every rerun of the null test, in order, each with its shifts and counts."""
    _require_columns(picks, PICK_COLUMNS, "picks")
    _require_columns(stations, (STATION_ID_COLUMN,), "stations")
    selected = select_profile(picks, cfg.profile)
    station_ids = sorted(set(selected["stationId"].astype(str)))
    known = set(stations[STATION_ID_COLUMN].astype(str))
    unknown = sorted(set(station_ids) - known)
    if unknown:
        raise ValidateError(f"picks name stations missing from stations table: {unknown}")
    log.info(
        "null test: %d reruns, shifts uniform(±%g s) per station, seed %d, profile %s: "
        "%d of %d picks on %d stations, %d catalog events",
        cfg.nShuffles,
        cfg.shiftS,
        cfg.seed,
        cfg.profile,
        len(selected),
        len(picks),
        len(station_ids),
        len(catalog),
    )
    outcomes: list[ShuffleOutcome] = []
    for i in range(cfg.nShuffles):
        started = time.perf_counter()
        rng = np.random.default_rng([cfg.seed, i])
        shifts = station_shifts(station_ids, rng, cfg.shiftS)
        shifted = shift_picks(selected, shifts)
        n_events, n_strict = rerun_pipeline(shifted, stations, catalog, api, seismology_cfg, run)
        runtime_s = time.perf_counter() - started
        log.info(
            "null test: rerun %d/%d: %d chance events, %d Tier A, %.1f s",
            i + 1,
            cfg.nShuffles,
            n_events,
            n_strict,
            runtime_s,
        )
        outcomes.append(ShuffleOutcome(i, shifts, n_events, n_strict, runtime_s))
    return outcomes


def summarize(outcomes: list[ShuffleOutcome], cfg: NullTestConfig) -> NullTest:
    """``NullTest`` from the reruns: means of both counts, sample std (``n - 1``) of the event
    count, and the configuration that produced them."""
    if len(outcomes) != cfg.nShuffles:
        raise ValidateError(f"expected {cfg.nShuffles} reruns, got {len(outcomes)}")
    events = np.array([o.n_events for o in outcomes], dtype=np.float64)
    strict = np.array([o.n_strict for o in outcomes], dtype=np.float64)
    result = NullTest(
        nShuffles=cfg.nShuffles,
        shiftRangeS=cfg.shiftS,
        meanChanceEvents=float(events.mean()),
        meanChanceStrict=float(strict.mean()),
        stdChanceEvents=float(events.std(ddof=1)),
    )
    if not all(
        math.isfinite(v)
        for v in (result.meanChanceEvents, result.meanChanceStrict, result.stdChanceEvents)
    ):
        raise ValidateError(f"null test produced non-finite statistics: {result}")
    log.info(
        "null test: chance events %.2f ± %.2f (std, n=%d), chance Tier A %.2f, %.1f s total",
        result.meanChanceEvents,
        result.stdChanceEvents,
        result.nShuffles,
        result.meanChanceStrict,
        sum(o.runtime_s for o in outcomes),
    )
    return result


def run_null_test(
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    catalog: pd.DataFrame,
    api: SeismologyApi,
    seismology_cfg: Any,
    run: RunSection,
    cfg: NullTestConfig,
) -> NullTest:
    """The null test end to end: ``nShuffles`` seeded reruns summarized as a ``NullTest``."""
    return summarize(null_shuffles(picks, stations, catalog, api, seismology_cfg, run, cfg), cfg)

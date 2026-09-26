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

Tier A is counted against supplied bars (``thresholds=``, REQ-H2-9): scrambled picks match
almost no public event, so H2's ``assign_tiers`` could not derive bars from a shuffle and never
invents them. The bars are the ones the stage hands in: the run's own ``ProcessingRun.tiering``
by default (every rerun locates with the run's ``statics.parquet``, bound into ``locate`` by
the stage, REQ-H1-5 option (a)), or with ``validate.yaml`` ``rerunBars: reference`` those H2
derived from the PhaseNet ``full`` rerun through this same path (``hq.validate.reference``,
option (b)). The reruns pass ``arrivals=`` and ``stations=`` too, so the nearest-station rule
measures focal depth as stage ``tier`` does. What a rerun did and could not do (statics applied,
no locate flags for the ``mapOnVolumeTop`` rule) is read back from the tiering record H2
returns and written to ``validation_notes.json`` (``hq.validate.notes``).

Pick ids are left as they are: they are the keys ``assoc_picks`` and ``arrivals`` refer to, and
a shifted pick is still the same pick.
"""

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from hq_contracts.models import NullTest

from hq.config.run import RunSection
from hq.config.validate import AssociationProfile, NullTestConfig, POnlyAssociatorConfig
from hq.validate.errors import ValidateError
from hq.validate.lanes import SeismologyApi

log = logging.getLogger(__name__)

PICK_COLUMNS: tuple[str, ...] = ("id", "stationId", "phase", "t")  # what the shuffle touches
STATION_ID_COLUMN = "id"
EVENT_ID_COLUMN = "id"
MATCH_EVENT_COLUMN = "eventId"
TIER_COLUMN = "tier"
STRICT_TIER = "A"
PHASE_P = "P"


@dataclass(frozen=True)
class Rerun:
    """What one ``associate -> locate -> match -> assign_tiers`` rerun produced: the final events
    table ``assign_tiers`` returned, ``match``'s table, and the tiering record H2 returned
    (``TierResult.tiering``; None when a step yielded no events and the rerun stopped there)."""

    events: pd.DataFrame  # events.parquet schema (empty, with the checked columns, when stopped)
    matches: pd.DataFrame  # matches.parquet schema
    tiering: dict[str, Any] | None = None

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def n_strict(self) -> int:
        return _count_strict(self.events) if len(self.events) else 0


@dataclass(frozen=True)
class ShuffleOutcome:
    """One rerun: the per-station shifts it used and what the pipeline found on them."""

    index: int
    shifts_s: pd.Series  # station id -> shift, seconds
    n_events: int
    n_strict: int
    runtime_s: float
    tiering: dict[str, Any] | None = field(default=None, compare=False)  # H2's TierResult.tiering


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValidateError(f"{what} lacks columns {missing}; docs/02 §2 lists {list(columns)}")


def profile_overrides(
    profiles: tuple[AssociationProfile, ...] | list[AssociationProfile],
    overrides: POnlyAssociatorConfig,
) -> dict[str, dict[str, int]]:
    """Per profile, the associator fields its reruns override (for the notes): ``p_only``'s
    values when it is among ``profiles``, nothing for ``full``."""
    return {"p_only": overrides.model_dump()} if "p_only" in profiles else {}


def profile_config(
    seismology_cfg: Any, profile: AssociationProfile, overrides: POnlyAssociatorConfig
) -> Any:
    """The ``SeismologyConfig`` a profile's rerun uses: the run's own for ``full``; for
    ``p_only`` a copy whose associator carries ``overrides`` (REQ-H2-7: the run's associator
    requires S picks, so P-only picks would associate nothing)."""
    if profile == "full":
        return seismology_cfg
    if profile == "p_only":
        associator = getattr(seismology_cfg, "associator", None)
        if associator is None or not hasattr(associator, "model_copy"):
            raise ValidateError(
                "the p_only profile needs SeismologyConfig.associator (H2 Seismology) to copy "
                f"with {overrides.model_dump()}; got {type(seismology_cfg).__name__}"
            )
        patched = associator.model_copy(update=overrides.model_dump())
        log.info(
            "profile p_only: associator overrides %s (run values nSPicks %s, nPAndSPicks %s)",
            overrides.model_dump(),
            getattr(associator, "nSPicks", None),
            getattr(associator, "nPAndSPicks", None),
        )
        return seismology_cfg.model_copy(update={"associator": patched})
    raise ValidateError(f"unknown association profile {profile!r}")  # unreachable via config


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


def require_thresholds(thresholds: Mapping[str, Any] | None, what: str) -> Mapping[str, Any]:
    """A ``ProcessingRun.tiering``-shaped dict with its ``thresholds`` record (the run's own, or
    the reference rerun's), or a loud error naming H2's tier stage: reruns tier against supplied
    bars and never invent them."""
    record = thresholds.get("thresholds") if isinstance(thresholds, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValidateError(
            f"{what} needs tier bars (a ProcessingRun.tiering['thresholds'] record: the run's "
            "own, written by H2's 'tier' stage, owner H2 Seismology, or the ones H2's "
            "assign_tiers derived from the reference rerun) to count Tier A with; run stage tier "
            "first, or leave validate.yaml rerunBars at 'reference'. Bars are never invented for "
            "a rerun (REQ-H2-9)"
        )
    return thresholds


def rerun_pipeline(
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    catalog: pd.DataFrame,
    api: SeismologyApi,
    seismology_cfg: Any,
    run: RunSection,
    *,
    thresholds: Mapping[str, Any] | None,
) -> Rerun:
    """``associate -> locate -> match -> assign_tiers`` on ``picks`` (docs/02 §5 calls, plus the
    REQ-H2-8/9 keywords): ``locate`` as bound by ``real_seismology_api`` and
    ``assign_tiers(located.events, matched.matches, cfg, thresholds=<tiering dict>,
    arrivals=located.arrivals, stations=stations)``, ``stations`` being the table the rerun
    located with. ``thresholds`` is a ``ProcessingRun.tiering``-shaped dict with the bars to
    apply; ``None`` (the reference rerun only, ``hq.validate.reference``) leaves the keyword out
    so H2 derives the bars from this rerun's own matched set, and H2's ``TierError`` (too few
    matched events) propagates for the caller to name.

    A step that yields no events ends the rerun with empty tables and no tiering record: there
    is nothing to locate, match or tier, and the later steps are not asked to handle an empty
    frame. Any error the API raises propagates unchanged.
    """
    if thresholds is not None:
        require_thresholds(thresholds, "rerun_pipeline")
    empty = Rerun(pd.DataFrame(columns=[EVENT_ID_COLUMN, TIER_COLUMN]),
                  pd.DataFrame(columns=[MATCH_EVENT_COLUMN]))  # fmt: skip
    assoc = api.associate(picks, stations, seismology_cfg, run)
    if len(assoc.events) == 0:
        return empty
    located = api.locate(assoc, picks, stations, seismology_cfg, run)
    if len(located.events) == 0:
        return empty
    matched = api.match(located.events, catalog, seismology_cfg)
    bars: dict[str, Any] = {} if thresholds is None else {"thresholds": thresholds}
    tiered = api.assign_tiers(
        located.events,
        matched.matches,
        seismology_cfg,
        **bars,
        arrivals=located.arrivals,
        stations=stations,
    )
    tiering = tiered.tiering
    if not isinstance(tiering, Mapping):
        raise ValidateError(
            f"assign_tiers returned tiering of type {type(tiering).__name__}, not the dict "
            "docs/02 §5 names (owner: H2 Seismology)"
        )
    return Rerun(tiered.events, matched.matches, dict(tiering))


def null_shuffles(
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    catalog: pd.DataFrame,
    api: SeismologyApi,
    seismology_cfg: Any,
    run: RunSection,
    cfg: NullTestConfig,
    p_only: POnlyAssociatorConfig | None = None,
    *,
    thresholds: Mapping[str, Any],
) -> list[ShuffleOutcome]:
    """Every rerun of the null test, in order, each with its shifts and counts. ``p_only`` holds
    the associator overrides the p_only profile reruns with (``ValidateConfig.pOnlyAssociator``;
    the config defaults when None); ``thresholds`` is a ``ProcessingRun.tiering``-shaped dict
    with the bars every rerun's Tier A is counted against (REQ-H2-9): the run's own by default,
    the reference rerun's with ``rerunBars: reference``. Never None: a shuffle cannot derive
    bars."""
    require_thresholds(thresholds, "the null test")
    _require_columns(picks, PICK_COLUMNS, "picks")
    _require_columns(stations, (STATION_ID_COLUMN,), "stations")
    selected = select_profile(picks, cfg.profile)
    seismology_cfg = profile_config(seismology_cfg, cfg.profile, p_only or POnlyAssociatorConfig())
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
        rerun = rerun_pipeline(
            shifted, stations, catalog, api, seismology_cfg, run, thresholds=thresholds
        )
        runtime_s = time.perf_counter() - started
        log.info(
            "null test: rerun %d/%d: %d chance events, %d Tier A, %.1f s",
            i + 1,
            cfg.nShuffles,
            rerun.n_events,
            rerun.n_strict,
            runtime_s,
        )
        outcomes.append(
            ShuffleOutcome(i, shifts, rerun.n_events, rerun.n_strict, runtime_s, rerun.tiering)
        )
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
    p_only: POnlyAssociatorConfig | None = None,
    *,
    thresholds: Mapping[str, Any],
) -> NullTest:
    """The null test end to end: ``nShuffles`` seeded reruns summarized as a ``NullTest``
    (the stage calls ``null_shuffles`` and ``summarize`` itself, to keep the reruns' tiering
    records for the notes)."""
    return summarize(
        null_shuffles(
            picks, stations, catalog, api, seismology_cfg, run, cfg, p_only, thresholds=thresholds
        ),
        cfg,
    )

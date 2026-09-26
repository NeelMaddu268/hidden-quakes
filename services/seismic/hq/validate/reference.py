"""The reference rerun (REQ-H1-5, option (b); ``validate.yaml`` ``rerunBars: reference``): an
alternative source for the validation reruns' tier bars.

The default (option (a), ``rerunBars: run``) tiers every rerun against the run's own bars, with
every rerun located through H2's ``locate(..., statics=<the run's statics.parquet>)`` so rerun
events sit on the run's scale. H1's alternative keeps the picker comparison on bars derived from
the rerun path itself: H2's ``assign_tiers`` derives them, without ``thresholds=``, from the
PhaseNet ``full``-profile rerun made through the very same calls every other rerun makes. That
rerun's matched set is the run's recovered public events, above ``tiering.minMatched``; nothing
fitted to PhaseNet's picks enters, so the scale is fair to STA/LTA; and the rerun doubles as the
baseline table's ``(phasenet, full)`` row.

``reference_rerun`` makes that one rerun with ``rerun_pipeline(..., thresholds=None)``, the only
call in the stage that leaves ``thresholds=`` out of ``assign_tiers`` so H2 derives the bars.
H2's ``TierError`` for too few matched events, an empty rerun, or a tiering record without bars
each fail loudly naming H2's rule: bars are never invented. The result carries the bars in the
``ProcessingRun.tiering`` shape (``{"thresholds": record}``) that every other rerun receives as
``thresholds=`` (the same dict object, so the notes and tests can tell it was that one).
"""

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from hq.config.run import RunSection
from hq.config.validate import AssociationProfile, POnlyAssociatorConfig
from hq.validate.errors import ValidateError
from hq.validate.lanes import SeismologyApi
from hq.validate.notes import THRESHOLDS_SOURCE_REFERENCE
from hq.validate.null_test import (
    PICK_COLUMNS,
    STATION_ID_COLUMN,
    STRICT_TIER,
    _require_columns,
    profile_config,
    rerun_pipeline,
    select_profile,
)

log = logging.getLogger(__name__)

H2 = "H2 Seismology"
REFERENCE_METHOD = "phasenet"  # BaselineRow.method of the rerun the bars come from
REFERENCE_PROFILE: AssociationProfile = "full"  # every pick, the run's SeismologyConfig unchanged
MIN_MATCHED_RULE = "seismology.yaml tiering.minMatched"  # H2's rule that refuses a small M
THRESHOLDS_KEY = "thresholds"
MATCH_EVENT_COLUMN = "eventId"


@dataclass(frozen=True)
class ReferenceRerun:
    """The PhaseNet ``full`` rerun and the bars H2 derived from it."""

    events: pd.DataFrame  # the final events table assign_tiers returned (events.parquet schema)
    matches: pd.DataFrame  # match's table (one row per public event)
    tiering: dict[str, Any]  # H2's TierResult.tiering of that rerun (thresholdSource "derived")
    thresholds: dict[str, Any]  # {"thresholds": record}: what every other rerun gets as thresholds=
    n_matched: int  # the matched set the bars were derived from
    source: str = THRESHOLDS_SOURCE_REFERENCE

    @property
    def record(self) -> Mapping[str, Any]:
        """H2's ``Thresholds.to_record`` of the derived bars."""
        return self.thresholds[THRESHOLDS_KEY]


def _too_few(exc: BaseException) -> ValidateError:
    return ValidateError(
        "the reference rerun (PhaseNet picks.parquet, profile full) "
        "could not give H2's assign_tiers a matched set it derives bars from: H2's rule "
        f"({MIN_MATCHED_RULE}, owner {H2}) refuses fewer matched events than minMatched, and "
        "bars are never invented for a rerun (REQ-H1-5, REQ-H2-9). Either the run recovers too "
        "few public events on this path, or set validate.yaml rerunBars: run (the default) to "
        f"apply the run's own bars. H2 said: {exc}"
    )


def reference_rerun(
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    catalog: pd.DataFrame,
    api: SeismologyApi,
    seismology_cfg: Any,
    run: RunSection,
) -> ReferenceRerun:
    """The PhaseNet ``full`` rerun through ``rerun_pipeline`` with ``thresholds=None``, so H2's
    ``assign_tiers(events, matches, cfg, arrivals=..., stations=...)`` derives the bars from
    the rerun's own matched set. Fails loudly, naming H2's ``minMatched`` rule, when it cannot."""
    _require_columns(picks, PICK_COLUMNS, "picks")
    _require_columns(stations, (STATION_ID_COLUMN,), "stations")
    selected = select_profile(picks, REFERENCE_PROFILE)
    # The full profile is the run's SeismologyConfig unchanged (no associator overrides).
    cfg = profile_config(seismology_cfg, REFERENCE_PROFILE, POnlyAssociatorConfig())
    log.info(
        "reference rerun (rerunBars reference): %s picks, profile %s (%d picks, %d stations, %d "
        "catalog events): assign_tiers derives the bars every validation rerun is tiered "
        "against (REQ-H1-5 b)",
        REFERENCE_METHOD,
        REFERENCE_PROFILE,
        len(selected),
        len(stations),
        len(catalog),
    )
    started = time.perf_counter()
    try:
        rerun = rerun_pipeline(selected, stations, catalog, api, cfg, run, thresholds=None)
    except ValueError as exc:  # H2's TierError subclasses ValueError (hq.tier.TierError)
        if type(exc).__name__ != "TierError" and "TierError" not in str(exc):
            raise
        raise _too_few(exc) from exc
    if rerun.tiering is None:
        raise _too_few(
            RuntimeError("the rerun associated or located no event, so nothing was tiered")
        )
    record = rerun.tiering.get(THRESHOLDS_KEY)
    if not isinstance(record, Mapping):
        raise ValidateError(
            "the reference rerun's assign_tiers returned a tiering record without a "
            f"'thresholds' bars record (owner: {H2}); nothing to tier the other reruns with"
        )
    if MATCH_EVENT_COLUMN not in rerun.matches.columns:
        raise ValidateError(
            f"match returned a table without {MATCH_EVENT_COLUMN!r} (docs/02 §2, owner: {H2})"
        )
    n_matched = int(rerun.matches[MATCH_EVENT_COLUMN].notna().sum())
    thresholds: dict[str, Any] = {THRESHOLDS_KEY: dict(record)}
    result = ReferenceRerun(rerun.events, rerun.matches, rerun.tiering, thresholds, n_matched)
    log.info(
        "reference rerun: %d candidate events, %d of %d public recovered, Tier %s %d; bars "
        "derived from %s matched events (thresholdSource %s, quantiles %s), %.1f s",
        rerun.n_events,
        n_matched,
        len(catalog),
        STRICT_TIER,
        rerun.n_strict,
        record.get("nMatched"),
        rerun.tiering.get("thresholdSource"),
        record.get("quantiles"),
        time.perf_counter() - started,
    )
    return result


__all__ = [
    "MIN_MATCHED_RULE",
    "REFERENCE_METHOD",
    "REFERENCE_PROFILE",
    "ReferenceRerun",
    "reference_rerun",
]

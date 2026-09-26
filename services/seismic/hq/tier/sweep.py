"""Association sweep scored through locate, match and tiers (LOC-06; writes ``sweep.parquet``).

At every point of ``associator.sweep`` (``hq.associate.sweep.run_sweep``: minStations x nSPicks x
minPickProb, every other knob as configured) the association result is located, matched to the
public regional catalog and tiered, and becomes one docs/02 ``SweepPoint``: ``candidates``
(located candidate events, as ``AnalysisSummary.candidateCount`` counts them), ``recoveredPublic``
(public events matched one-to-one) and ``tierA``. Tier A is counted with the bars derived from the
configured run's own matched set (``thresholds``), so points compare on one scale: a point's own
matched set would move the bars with the point.

The configured point is re-evaluated through the same driver (a fresh association, located with
the run's ``statics.parquet``, the statics ``events_located.parquet`` carries) so it compares with
the other points. The stage logs its Tier A next to the one in ``events.parquet``.

``locate`` is LOC-04's ``hq.locate.locate_detailed``, imported only when the sweep runs
(``real_pipeline``); a branch without it fails with a message naming LOC-04. Tests inject a
``SweepPipeline`` and the association step.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from hq_contracts.models import SweepPoint

from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig
from hq.tier import Thresholds, TierError, assign_tiers

if TYPE_CHECKING:
    from hq.associate.result import AssocResult
    from hq.associate.sweep import Evaluate, SweepRow, SweepScore

log = logging.getLogger(__name__)

STRICT = "A"

# assoc -> (events_located, locate flags or None, arrivals or None)
LocateFn = Callable[
    ["AssocResult"], tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]
]
MatchFn = Callable[[pd.DataFrame], pd.DataFrame]  # events_located -> matches
RunPoints = Callable[["Evaluate"], "list[SweepRow]"]  # evaluator -> one row per sweep point


@dataclass(frozen=True)
class SweepPipeline:
    """The steps after association, bound to one run's inputs. ``stations`` goes with the
    arrivals ``locate`` returns to ``assign_tiers`` (both or neither)."""

    locate: LocateFn
    match: MatchFn
    stations: pd.DataFrame | None


def make_evaluator(
    pipeline: SweepPipeline, cfg: SeismologyConfig, thresholds: Thresholds
) -> "Evaluate":
    """``AssocResult -> SweepScore`` through locate, match and ``assign_tiers(thresholds=...)``."""
    from hq.associate.sweep import SweepScore

    def evaluate(assoc: "AssocResult") -> "SweepScore":
        if len(assoc.events) == 0:
            return SweepScore(candidates=0, recovered_public=0, tier_a=0)
        events, flags, arrivals = pipeline.locate(assoc)
        matches = pipeline.match(events)
        tiered = assign_tiers(events, matches, cfg, flags=flags, thresholds=thresholds,
                              arrivals=arrivals, stations=pipeline.stations)
        recovered = int(matches["eventId"].notna().sum())
        tier_a = int((tiered.events["tier"].astype(str) == STRICT).sum())
        return SweepScore(candidates=len(tiered.events), recovered_public=recovered,
                          tier_a=tier_a)

    return evaluate


def score_sweep(
    run_points: RunPoints, pipeline: SweepPipeline, cfg: SeismologyConfig, thresholds: Thresholds
) -> tuple[list[SweepPoint], list[dict[str, Any]]]:
    """``SweepPoint`` rows and a per-point record (params, associated, scores, runtime)."""
    from hq.associate.sweep import sweep_points

    rows = run_points(make_evaluator(pipeline, cfg, thresholds))
    points = sweep_points(rows)
    record = [
        {
            "params": row.params,
            "associated": row.candidates,
            "candidates": point.candidates,
            "recoveredPublic": point.recoveredPublic,
            "tierA": point.tierA,
            "runtimeS": round(row.runtime_s, 3),
        }
        for row, point in zip(rows, points, strict=True)
    ]
    return points, record


def real_pipeline(
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    catalog: pd.DataFrame,
    cfg: SeismologyConfig,
    run: RunSection,
    *,
    run_id: str,
    cache_dir: Path,
    statics: Mapping[tuple[str, str], float],
) -> tuple[RunPoints, SweepPipeline]:
    """The real association sweep and LOC-04 / MATCH-02 steps for one run's tables; every point
    is located with ``statics`` ({(stationId, phase): s}, the run's ``statics.parquet``)."""
    try:
        from hq.locate import locate_detailed  # LOC-04
    except ImportError as exc:
        raise TierError(
            "the association sweep needs hq.locate.locate_detailed (LOC-04), which this branch "
            "does not have yet; set tiering.sweep.enabled false or merge LOC-04"
        ) from exc
    from hq.associate.core import prepared
    from hq.associate.sweep import run_sweep
    from hq.match import match

    def locate(
        assoc: "AssocResult",
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        details = locate_detailed(assoc, picks, stations, cfg, run, run_id=run_id,
                                  cache_dir=cache_dir, statics=dict(statics))
        return details.result.events, details.flags, details.result.arrivals

    def match_events(events: pd.DataFrame) -> pd.DataFrame:
        return match(events, catalog, cfg).matches

    def run_points(evaluate: "Evaluate") -> "list[SweepRow]":
        with prepared(stations, cfg, run, cache_dir=cache_dir) as setup:
            return run_sweep(picks, setup, cfg.associator, evaluate)

    return run_points, SweepPipeline(locate=locate, match=match_events, stations=stations)

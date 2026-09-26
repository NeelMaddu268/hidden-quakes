"""Association parameter sweep (LOC-03; completed with public recall and Tier A in LOC-06).

Runs every combination of ``associator.sweep`` (minStations x nSPicks x minPickProb), all other
knobs as configured. ``minStations`` is not a PyOcto argument (it filters after the merge), so
PyOcto runs once per (nSPicks, minPickProb) and each result is finished for every minStations.

``SweepPoint.recoveredPublic`` and ``tierA`` exist only after locate, match and tier (LOC-04,
MATCH-02, LOC-06), so the driver takes an ``evaluate`` callback that turns an ``AssocResult`` into
those numbers. Without it the driver still counts candidates per point, and no ``SweepPoint`` is
built: those fields are never filled with placeholders.
"""

import itertools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from hq_contracts.models import SweepPoint

from hq.associate.core import Setup, finish, run_pyocto
from hq.associate.result import AssocResult
from hq.config.seismology import AssociatorConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepScore:
    """What an evaluator reports for one association result."""

    recovered_public: int  # public catalog events matched one-to-one
    tier_a: int  # Tier A events


Evaluate = Callable[[AssocResult], SweepScore]


@dataclass(frozen=True)
class SweepRow:
    params: dict[str, Any]  # the knobs this point overrides
    candidates: int  # associated events at this point
    counts: dict[str, int]
    score: SweepScore | None  # None when no evaluator is wired


def point_config(acfg: AssociatorConfig, params: dict[str, Any]) -> AssociatorConfig:
    """``acfg`` with ``params`` overridden, re-validated."""
    return AssociatorConfig.model_validate({**acfg.model_dump(), **params})


def grid(acfg: AssociatorConfig) -> list[dict[str, Any]]:
    """Every sweep point's overrides, minStations varying fastest."""
    sweep = acfg.sweep
    return [
        {"minPickProb": prob, "nSPicks": n_s, "minStations": n_sta}
        for prob, n_s, n_sta in itertools.product(
            sweep.minPickProb, sweep.nSPicks, sweep.minStations
        )
    ]


def run_sweep(
    picks: pd.DataFrame,
    setup: Setup,
    acfg: AssociatorConfig,
    evaluate: Evaluate | None = None,
) -> list[SweepRow]:
    """Associate at every sweep point; score each result with ``evaluate`` when given."""
    started = time.perf_counter()
    rows: list[SweepRow] = []
    raw_cache: dict[tuple[float, int], Any] = {}
    for params in grid(acfg):
        cfg = point_config(acfg, params)
        key = (cfg.minPickProb, cfg.nSPicks)  # everything PyOcto sees that the sweep varies
        if key not in raw_cache:
            raw_cache[key] = run_pyocto(picks, setup, cfg)
        result, counts = finish(raw_cache[key], cfg, setup.origin)
        score = evaluate(result) if evaluate is not None else None
        rows.append(SweepRow(params=params, candidates=len(result.events), counts=counts,
                             score=score))
        log.info(
            "associate sweep: minPickProb %s, nSPicks %d, minStations %d -> %d candidate events%s",
            cfg.minPickProb, cfg.nSPicks, cfg.minStations, len(result.events),
            "" if score is None
            else f", {score.recovered_public} public recovered, {score.tier_a} Tier A",
        )
    log.info(
        "associate sweep: %d points, %d PyOcto runs in %.1f s",
        len(rows), len(raw_cache), time.perf_counter() - started,
    )
    return rows


def sweep_points(rows: list[SweepRow]) -> list[SweepPoint]:
    """``SweepPoint`` rows; raises if any row has no evaluator score."""
    points: list[SweepPoint] = []
    for row in rows:
        if row.score is None:
            raise ValueError(f"sweep point {row.params} has no recoveredPublic/tierA")
        points.append(
            SweepPoint(
                params=row.params,
                candidates=row.candidates,
                recoveredPublic=row.score.recovered_public,
                tierA=row.score.tier_a,
            )
        )
    return points

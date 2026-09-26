"""Stage ``tier`` (LOC-06): located events + matches -> ``events.parquet``, ``event_picks.parquet``.

Reads ``events_located.parquet``, ``matches.parquet``, ``arrivals.parquet``, the picks table the
association read (``associator.picksTable``) and, when present, LOC-04's ``locate_flags.parquet``
(without it the ``mapOnVolumeTop`` rule is not applied, logged and recorded). Checks every
event's ``depthKm`` against ``(run.refSurfaceElevM - elevM) / 1000``, runs ``hq.tier.assign_tiers``
and ``hq.tier.picks.event_picks``, and writes ``events.parquet`` (final ``SeismicEvent`` rows) and
``event_picks.parquet`` (docs/02 §2).

With ``tiering.sweep.enabled`` it also runs the association sweep (``hq.tier.sweep``, with the
bars just derived) and writes ``sweep.parquet``; disabled, it removes a ``sweep.parquet`` left by
an earlier tier run, whose Tier A counts used that run's bars. Every output is written under a
``.part`` name first and moved into place only after all of them were written.

``ctx.record`` gets the counts (tiers for all, additional and matched events, event picks, sweep
points) and ``ProcessingRun.tiering``: every bar with its source quantile and the size of the
matched set, the rules, the caveat, the inputs and the sweep record.

The package attribute ``hq.tier.run`` is this module's ``run`` function (the stage registry
resolves it there), so ``import hq.tier.run as m`` binds the function, not this module; reach the
module with ``importlib.import_module("hq.tier.run")``.
"""

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from hq_contracts.io import read_table, to_frame, write_table
from hq_contracts.models import Pick, SeismicEvent, SweepPoint

from hq.config.run import RunSection
from hq.tier import Thresholds, TierError, assign_tiers
from hq.tier.picks import event_picks

if TYPE_CHECKING:
    from hq.runs import RunContext

log = logging.getLogger(__name__)

STAGE = "tier"
EVENTS_LOCATED_TABLE = "events_located.parquet"
MATCHES_TABLE = "matches.parquet"
ARRIVALS_TABLE = "arrivals.parquet"
FLAGS_TABLE = "locate_flags.parquet"
STATIONS_TABLE = "stations.parquet"
CATALOG_TABLE = "catalog.parquet"
EVENTS_TABLE = "events.parquet"
EVENT_PICKS_TABLE = "event_picks.parquet"
SWEEP_TABLE = "sweep.parquet"
PART_SUFFIX = ".part"
# Model name each input's parquet metadata must carry (LOC-04's and MATCH-02's writers).
INPUT_MODELS = {
    "events_located": "LocatedEvent",
    "matches": "Match",
    "arrivals": "Arrival",
    "flags": "LocateFlags",
    "picks": "Pick",
    "stations": "Station",
    "catalog": "CatalogEvent",
}


def _part(path: Path) -> Path:
    return path.with_name(path.name + PART_SUFFIX)


def _read(path: Path, what: str) -> pd.DataFrame:
    frame = read_table(path)
    model = frame.attrs.get("model")
    if model != INPUT_MODELS[what]:
        raise TierError(f"{path} holds {model!r} rows; the {what} input must hold "
                        f"{INPUT_MODELS[what]!r} rows")
    return frame


def check_depth(events: pd.DataFrame, run: RunSection) -> None:
    """``depthKm`` must equal ``(refSurfaceElevM - elevM) / 1000`` (docs/02) for this run."""
    elev = events["elevM"].to_numpy(dtype=np.float64)
    expected = (run.refSurfaceElevM - elev) / 1000.0
    stored = events["depthKm"].to_numpy(dtype=np.float64)
    bad = np.flatnonzero(stored != expected)
    if bad.size:
        k = int(bad[0])
        raise TierError(
            f"{bad.size} located events have depthKm != (refSurfaceElevM {run.refSurfaceElevM} "
            f"- elevM) / 1000, e.g. {events['id'].iloc[k]}: {stored[k]} vs {expected[k]}; "
            "events_located.parquet was written with another run section"
        )


def _sweep(
    ctx: "RunContext", picks: pd.DataFrame, thresholds: Thresholds, tier_a: int
) -> tuple[list[SweepPoint], dict[str, Any]]:
    from hq.tier.sweep import real_pipeline, score_sweep

    cfg = ctx.config.seismology
    stations = _read(ctx.path(STATIONS_TABLE), "stations")
    catalog = _read(ctx.path(CATALOG_TABLE), "catalog")
    run_points, pipeline = real_pipeline(
        picks, stations, catalog, cfg, ctx.config.run, run_id=ctx.run_id, cache_dir=ctx.cache_dir
    )
    started = time.perf_counter()
    points, per_point = score_sweep(run_points, pipeline, cfg, thresholds)
    configured = {
        "minPickProb": cfg.associator.minPickProb,
        "nSPicks": cfg.associator.nSPicks,
        "minStations": cfg.associator.minStations,
    }
    at_configured = [p for p in per_point if p["params"] == configured]
    if at_configured:
        log.info(
            "tier sweep: configured point %s gives %d Tier A through the sweep driver; "
            "events.parquet has %d (they differ when the locate stage applied statics)",
            configured, at_configured[0]["tierA"], tier_a,
        )
    return points, {
        "enabled": True,
        "grid": "associator.sweep (minStations x nSPicks x minPickProb)",
        "tierABars": "the bars in 'thresholds' (derived from this run's matched set)",
        "configuredPoint": configured,
        "configuredPointTierAInEvents": tier_a,
        "note": "each point: associate -> locate (no station statics) -> match -> tiers; "
        "candidates = located candidate events",
        "points": per_point,
        "runtimeS": round(time.perf_counter() - started, 3),
    }


def run(ctx: "RunContext") -> None:
    """Stage ``tier`` (docs/02 §4); see the module docstring."""
    started = time.perf_counter()
    cfg = ctx.config.seismology
    events_located = _read(ctx.path(EVENTS_LOCATED_TABLE), "events_located")
    matches = _read(ctx.path(MATCHES_TABLE), "matches")
    arrivals = _read(ctx.path(ARRIVALS_TABLE), "arrivals")
    picks_path = ctx.path(cfg.associator.picksTable)
    picks = _read(picks_path, "picks")
    flags_path = ctx.path(FLAGS_TABLE)
    flags = _read(flags_path, "flags") if flags_path.is_file() else None
    if flags is None:
        log.warning("tier: no %s in %s; Tier A does not apply the mapOnVolumeTop rule",
                    FLAGS_TABLE, ctx.run_dir)
    log.info("tier: %d located candidate events, %d matches rows, %d arrivals, %d picks from %s",
             len(events_located), len(matches), len(arrivals), len(picks), picks_path)
    check_depth(events_located, ctx.config.run)

    result = assign_tiers(events_located, matches, cfg, flags=flags)
    picks_out = event_picks(result.events, arrivals, picks)
    tiering = result.tiering
    n_a = int((result.events["tier"].astype(str) == "A").sum())

    sweep_path = ctx.path(SWEEP_TABLE)
    sweep_points: list[SweepPoint] | None = None
    if cfg.tiering.sweep.enabled:
        if tiering["thresholds"] is None:
            raise TierError("the association sweep needs this run's bars, and it has no "
                            "located events to derive them from")
        sweep_points, sweep_record = _sweep(
            ctx, picks, Thresholds.from_record(tiering["thresholds"]), n_a
        )
    else:
        sweep_record = {"enabled": False}

    tables: dict[Path, tuple[pd.DataFrame, str]] = {
        ctx.path(EVENTS_TABLE): (result.events, SeismicEvent.__name__),
        ctx.path(EVENT_PICKS_TABLE): (picks_out, Pick.__name__),
    }
    if sweep_points is not None:
        tables[sweep_path] = (to_frame(sweep_points, SweepPoint), SweepPoint.__name__)
    try:
        for path, (frame, model_name) in tables.items():
            write_table(frame, _part(path), model_name)
        for path in tables:
            os.replace(_part(path), path)
    finally:
        for path in tables:
            _part(path).unlink(missing_ok=True)
    if sweep_points is None and sweep_path.exists():
        sweep_path.unlink()
        log.warning("tier: removed %s from an earlier tier run (tiering.sweep.enabled is false; "
                    "its Tier A counts used that run's bars)", sweep_path)

    counts = result.tiering["counts"]
    stage_counts = {
        "events": counts["events"],
        **{f"tier{t}": n for t, n in counts["all"].items()},
        "additional": sum(counts["additional"].values()),
        **{f"additionalTier{t}": n for t, n in counts["additional"].items()},
        "matched": sum(counts["matched"].values()),
        **{f"matchedTier{t}": n for t, n in counts["matched"].items()},
        "eventPicks": len(picks_out),
        "sweepPoints": 0 if sweep_points is None else len(sweep_points),
    }
    params: dict[str, Any] = {
        **tiering,
        "config": cfg.tiering.model_dump(mode="json"),
        "input": {
            "eventsLocated": EVENTS_LOCATED_TABLE,
            "matches": MATCHES_TABLE,
            "arrivals": ARRIVALS_TABLE,
            "picks": cfg.associator.picksTable,
            "flags": FLAGS_TABLE if flags is not None else None,
        },
        "outputs": [p.name for p in tables],
        "eventPicks": "one Pick per id in each final event's pickIds (picks used in the final "
        "location); eventId set, residualS from arrivals.parquet",
        "final": "SeismicEvent rows: located columns unchanged, tier, tierReasons, catalogMatch "
        "from matches.parquet, magnitude null (MAG-01), revealOrder -1",
        "sweep": sweep_record,
    }
    runtime_s = time.perf_counter() - started
    log.info(
        "tier: wrote %s: all A %d / B %d / C %d, additional A %d / B %d / C %d, %d event picks, "
        "%d sweep points in %.1f s",
        ", ".join(p.name for p in tables), *counts["all"].values(),
        *counts["additional"].values(), len(picks_out), stage_counts["sweepPoints"], runtime_s,
    )
    ctx.record(STAGE, runtime_s=runtime_s, counts=stage_counts, params=params)

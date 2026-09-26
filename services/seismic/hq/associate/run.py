"""Stage ``associate``: picks + stations -> ``assoc_events.parquet``, ``assoc_picks.parquet`` (LOC-03).

Reads ``associator.picksTable`` (``known/picks.parquet`` or ``picks.parquet``) and
``stations.parquet`` from the run dir, associates at the configured point, then runs the sweep
(``associator.sweep.enabled``). ``sweep.parquet`` is written only when an ``evaluate`` callback is
passed (locate + match + tier, LOC-06); without one the per-point candidate counts go to the log
and the record, and a ``sweep.parquet`` left by an earlier run is removed, since it would describe
another association. Outputs are written under ``.part`` names and moved into place together.
"""

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hq_contracts.io import read_table, write_models, write_table
from hq_contracts.models import SweepPoint

from hq.associate.core import associate_setup, prepared, record
from hq.associate.result import EVENTS_MODEL, PICKS_MODEL
from hq.associate.sweep import Evaluate, grid, run_sweep, sweep_points

if TYPE_CHECKING:
    from hq.runs import RunContext

log = logging.getLogger(__name__)

STAGE = "associate"
STATIONS_TABLE = "stations.parquet"
EVENTS_TABLE = "assoc_events.parquet"
PICKS_TABLE = "assoc_picks.parquet"
SWEEP_TABLE = "sweep.parquet"
PART_SUFFIX = ".part"


def _part(path: Path) -> Path:
    return path.with_name(path.name + PART_SUFFIX)


def run(ctx: "RunContext", *, evaluate: Evaluate | None = None) -> None:
    """Stage ``associate`` (docs/02 §4). ``evaluate`` wires the full sweep (LOC-06)."""
    started = time.perf_counter()
    seis = ctx.config.seismology
    acfg = seis.associator
    picks_path = ctx.path(acfg.picksTable)
    picks = read_table(picks_path)
    stations = read_table(ctx.path(STATIONS_TABLE))
    log.info("associate: %d picks from %s, %d stations", len(picks), picks_path, len(stations))

    with prepared(stations, seis, ctx.config.run, cache_dir=ctx.cache_dir) as setup:
        result, counts = associate_setup(picks, setup, acfg)
        params: dict[str, Any] = record(acfg, setup)
        rows = run_sweep(picks, setup, acfg, evaluate) if acfg.sweep.enabled else []

    outputs = {
        ctx.path(EVENTS_TABLE): (result.events, EVENTS_MODEL),
        ctx.path(PICKS_TABLE): (result.picks, PICKS_MODEL),
    }
    sweep_path = ctx.path(SWEEP_TABLE)
    parts = [_part(path) for path in [*outputs, sweep_path]]
    try:
        for path, (frame, model_name) in outputs.items():
            write_table(frame, _part(path), model_name)
        if evaluate is not None and rows:
            write_models(sweep_points(rows), _part(sweep_path), SweepPoint)
        for path in outputs:
            os.replace(_part(path), path)
        if evaluate is not None and rows:
            os.replace(_part(sweep_path), sweep_path)
        elif sweep_path.exists():
            sweep_path.unlink()
            log.warning(
                "associate: removed %s from an earlier run (sweep not scored in this run)",
                sweep_path,
            )
    finally:
        for part in parts:
            part.unlink(missing_ok=True)

    params["input"] = {"picksTable": acfg.picksTable, "stationsTable": STATIONS_TABLE}
    params["sweep"] = {
        "enabled": acfg.sweep.enabled,
        "grid": grid(acfg) if acfg.sweep.enabled else [],
        "points": [{"params": r.params, "candidates": r.candidates} for r in rows],
        "sweepTableWritten": evaluate is not None and bool(rows),
        "note": "recoveredPublic and tierA need locate, match and tier; the full sweep "
        "(sweep.parquet) is completed in LOC-06",
    }
    runtime_s = time.perf_counter() - started
    counts = {**counts, "sweepPoints": len(rows)}
    log.info(
        "associate: wrote %d events and %d associated picks; %d sweep points; %.1f s",
        counts["events"], counts["assocPicks"], len(rows), runtime_s,
    )
    ctx.record(STAGE, runtime_s=runtime_s, counts=counts, params=params)

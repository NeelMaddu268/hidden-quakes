"""Stage ``associate``: picks + stations -> ``assoc_events.parquet``, ``assoc_picks.parquet`` (LOC-03).

Reads ``associator.picksTable`` (``known/picks.parquet`` or ``picks.parquet``) and
``stations.parquet`` from the run dir and associates at the configured point. With
``associator.sweep.enabled`` it also associates at every sweep point and logs and records the
candidate count of each (the configured point's PyOcto run is reused). ``sweep.parquet`` needs
public recall and Tier A from locate, match and tier, so LOC-06's tier stage writes it with
``hq.associate.sweep.run_sweep`` and ``sweep_points``; this stage only removes a ``sweep.parquet``
left by an earlier run, since it would describe another association. For the same reason it
removes a ``matches.parquet`` and ``match_sensitivity.parquet`` left by an earlier match: they
name events located from the old association, and stage locate takes a ``matches.parquet`` as
the signal for its reference-statics pass 2 (LOC-05), which then refuses the new association.
Without them the next locate runs pass 1, as a fresh run does. Every removal is logged and
recorded (``removedStale``). Outputs are written under ``.part`` names and moved into place
together.

The package attribute ``hq.associate.run`` is this module's ``run`` function (the stage registry
resolves it there), so ``import hq.associate.run as m`` binds the function, not this module; reach
the module with ``importlib.import_module("hq.associate.run")``.
"""

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from hq_contracts.io import read_table, write_table

from hq.associate.core import finish, prepared, record, run_pyocto
from hq.associate.result import EVENTS_MODEL, PICKS_MODEL
from hq.associate.sweep import grid, run_sweep
from hq.locate.provenance import stage_provenance

if TYPE_CHECKING:
    from hq.runs import RunContext

log = logging.getLogger(__name__)

STAGE = "associate"
STATIONS_TABLE = "stations.parquet"
EVENTS_TABLE = "assoc_events.parquet"
PICKS_TABLE = "assoc_picks.parquet"
SWEEP_TABLE = "sweep.parquet"
# Outputs of stage match on events located from an earlier association (see the docstring).
MATCH_TABLES = ("matches.parquet", "match_sensitivity.parquet")
PART_SUFFIX = ".part"
INPUT_MODELS = {"picks": "Pick", "stations": "Station"}  # docs/02 §2 model names


def _part(path: Path) -> Path:
    return path.with_name(path.name + PART_SUFFIX)


def _read_input(path: Path, what: str) -> pd.DataFrame:
    frame = read_table(path)
    model = frame.attrs.get("model")
    if model != INPUT_MODELS[what]:
        raise ValueError(f"{path} holds {model!r} rows; the {what} input must hold "
                         f"{INPUT_MODELS[what]!r} rows (docs/02 §2)")
    return frame


def run(ctx: "RunContext") -> None:
    """Stage ``associate`` (docs/02 §4)."""
    started = time.perf_counter()
    seis = ctx.config.seismology
    acfg = seis.associator
    picks_path = ctx.path(acfg.picksTable)
    picks = _read_input(picks_path, "picks")
    stations = _read_input(ctx.path(STATIONS_TABLE), "stations")
    log.info("associate: %d picks from %s, %d stations", len(picks), picks_path, len(stations))

    with prepared(stations, seis, ctx.config.run, cache_dir=ctx.cache_dir) as setup:
        raw = run_pyocto(picks, setup, acfg)
        result, counts = finish(raw, acfg, setup)
        params: dict[str, Any] = record(acfg, setup, picks)
        rows = run_sweep(picks, setup, acfg, reuse=raw) if acfg.sweep.enabled else []

    outputs = {
        ctx.path(EVENTS_TABLE): (result.events, EVENTS_MODEL),
        ctx.path(PICKS_TABLE): (result.picks, PICKS_MODEL),
    }
    try:
        for path, (frame, model_name) in outputs.items():
            write_table(frame, _part(path), model_name)
        for path in outputs:
            os.replace(_part(path), path)
    finally:
        for path in outputs:
            _part(path).unlink(missing_ok=True)
    removed = []
    for name in (SWEEP_TABLE, *MATCH_TABLES):
        path = ctx.path(name)
        if path.exists():
            path.unlink()
            removed.append(name)
            log.warning("associate: removed %s from an earlier association (%s rewrites it)",
                        path, "LOC-06's stage tier" if name == SWEEP_TABLE else "stage match")

    params["input"] = {"picksTable": acfg.picksTable, "stationsTable": STATIONS_TABLE}
    params["provenance"] = stage_provenance()
    params["removedStale"] = removed
    params["sweep"] = {
        "enabled": acfg.sweep.enabled,
        "grid": grid(acfg) if acfg.sweep.enabled else [],
        "points": [{"params": r.params, "associated": r.candidates} for r in rows],
        "note": "associated = events after this stage's post-processing at that point, before "
        "location; sweep.parquet (SweepPoint with located candidates, recoveredPublic and tierA) "
        "is written by LOC-06",
    }
    runtime_s = time.perf_counter() - started
    counts = {**counts, "sweepPoints": len(rows)}
    log.info(
        "associate: wrote %d events and %d associated picks; %d sweep points; %.1f s",
        counts["events"], counts["assocPicks"], len(rows), runtime_s,
    )
    ctx.record(STAGE, runtime_s=runtime_s, counts=counts, params=params)

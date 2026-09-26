"""Stage ``locate`` (LOC-04): association -> located candidate events, arrivals, statics, report.

Reads ``assoc_events.parquet``, ``assoc_picks.parquet``, the picks table the association read
(``associator.picksTable``) and ``stations.parquet`` (its ``usedInRun`` stations), runs
``hq.locate.locate_detailed`` with the run's cache dir, and writes ``events_located.parquet``,
``arrivals.parquet``, ``statics.parquet`` (docs/02 §2), ``locate_flags.parquet`` (H2-internal, see
``hq.locate``) and ``diagnostics.md`` (``hq.locate.diagnostics``). The report also reads, when
present, ``catalog.parquet`` and ``catalog.quakeml`` (public regional catalog comparison) and
H1's ``known/windows.json`` (which public events are the known ones). Every output is written
under a ``.part`` name first and moved into place only after all of them were written.

``ctx.record`` gets the counts, the runtime, the locator record (``ProcessingRun.locator``, with
the stage's conventions under ``locate`` and the diagnostics knobs) and the top-extended velocity
model the tables were solved on (``ProcessingRun.velocityModel``).

The package attribute ``hq.locate.run`` is this module's ``run`` function (the stage registry
resolves it there), so ``import hq.locate.run as m`` binds the function, not this module; reach
the module with ``importlib.import_module("hq.locate.run")``.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from hq_contracts.io import read_table, write_table

from hq.locate import locate_detailed
from hq.locate.diagnostics import DiagnosticsInputs, build_diagnostics, catalog_uncertainties
from hq.locate.result import ARRIVALS_MODEL, EVENTS_MODEL, FLAGS_MODEL, STATICS_MODEL

if TYPE_CHECKING:
    from hq.runs import RunContext

log = logging.getLogger(__name__)

STAGE = "locate"
STATIONS_TABLE = "stations.parquet"
ASSOC_EVENTS_TABLE = "assoc_events.parquet"
ASSOC_PICKS_TABLE = "assoc_picks.parquet"
EVENTS_TABLE = "events_located.parquet"
ARRIVALS_TABLE = "arrivals.parquet"
STATICS_TABLE = "statics.parquet"
FLAGS_TABLE = "locate_flags.parquet"
DIAGNOSTICS = "diagnostics.md"
CATALOG_TABLE = "catalog.parquet"
CATALOG_QUAKEML = "catalog.quakeml"
KNOWN_WINDOWS = "known/windows.json"
PART_SUFFIX = ".part"
# Model name each input's parquet metadata must carry (docs/02 §2; the assoc tables are LOC-03's).
INPUT_MODELS = {"picks": "Pick", "stations": "Station", "assoc_events": "AssocEvent",
                "assoc_picks": "AssocPick", "catalog": "CatalogEvent"}


def _part(path: Path) -> Path:
    return path.with_name(path.name + PART_SUFFIX)


def _read(path: Path, what: str) -> pd.DataFrame:
    frame = read_table(path)
    model = frame.attrs.get("model")
    if model != INPUT_MODELS[what]:
        raise ValueError(f"{path} holds {model!r} rows; the {what} input must hold "
                         f"{INPUT_MODELS[what]!r} rows")
    return frame


@dataclass(frozen=True, eq=False)
class StoredAssociation:
    """``assoc_events`` / ``assoc_picks`` as stored in the run dir: what ``locate`` reads.

    Not ``hq.associate.AssocResult``, so this stage never imports PyOcto.
    """

    events: pd.DataFrame
    picks: pd.DataFrame


def _assoc(ctx: "RunContext") -> StoredAssociation:
    return StoredAssociation(
        events=_read(ctx.path(ASSOC_EVENTS_TABLE), "assoc_events"),
        picks=_read(ctx.path(ASSOC_PICKS_TABLE), "assoc_picks"),
    )


def known_ids(path: Path) -> tuple[str, ...] | None:
    """Event ids listed in H1's ``known/windows.json`` (``events[].eventId``), or None."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(ev["eventId"]) for ev in data.get("events", [])]
    return tuple(ids) if ids else None


def run(ctx: "RunContext") -> None:
    """Stage ``locate`` (docs/02 §4); see the module docstring."""
    started = time.perf_counter()
    cfg = ctx.config.seismology
    run_cfg = ctx.config.run
    picks_path = ctx.path(cfg.associator.picksTable)
    picks = _read(picks_path, "picks")
    stations = _read(ctx.path(STATIONS_TABLE), "stations")
    assoc = _assoc(ctx)
    log.info("locate: %d association events, %d picks from %s, %d stations",
             len(assoc.events), len(picks), picks_path, len(stations))

    details = locate_detailed(assoc, picks, stations, cfg, run_cfg, run_id=ctx.run_id,
                              cache_dir=ctx.cache_dir)
    catalog_path = ctx.path(CATALOG_TABLE)
    quakeml_path = ctx.path(CATALOG_QUAKEML)
    catalog = _read(catalog_path, "catalog") if catalog_path.is_file() else None
    report = build_diagnostics(
        DiagnosticsInputs(
            run_id=ctx.run_id,
            run=run_cfg,
            cfg=cfg,
            stations=stations,
            details=details,
            assoc_events=assoc.events,
            catalog=catalog,
            catalog_errors=catalog_uncertainties(quakeml_path)
            if catalog is not None and quakeml_path.is_file() else None,
            known_ids=known_ids(ctx.path(KNOWN_WINDOWS)),
        )
    )

    tables = {
        ctx.path(EVENTS_TABLE): (details.result.events, EVENTS_MODEL),
        ctx.path(ARRIVALS_TABLE): (details.result.arrivals, ARRIVALS_MODEL),
        ctx.path(STATICS_TABLE): (details.result.statics, STATICS_MODEL),
        ctx.path(FLAGS_TABLE): (details.flags, FLAGS_MODEL),
    }
    report_path = ctx.path(DIAGNOSTICS)
    targets = [*tables, report_path]
    try:
        for path, (frame, model_name) in tables.items():
            write_table(frame, _part(path), model_name)
        _part(report_path).write_text(report, encoding="utf-8")
        for path in targets:
            os.replace(_part(path), path)
    finally:
        for path in targets:
            _part(path).unlink(missing_ok=True)

    runtime_s = time.perf_counter() - started
    counts = details.counts
    params: dict[str, Any] = {
        **details.record,
        "input": {
            "picksTable": cfg.associator.picksTable,
            "stationsTable": STATIONS_TABLE,
            "assocTables": [ASSOC_EVENTS_TABLE, ASSOC_PICKS_TABLE],
        },
        "outputs": [p.name for p in targets],
        "diagnostics": cfg.diagnostics.model_dump(mode="json"),
        "locateRuntimeS": details.runtime_s,
    }
    log.info("locate: wrote %s in %.1f s", ", ".join(p.name for p in targets), runtime_s)
    ctx.record(STAGE, runtime_s=runtime_s, counts=counts, params=details.velocity_model,
               field="velocityModel")
    ctx.record(STAGE, runtime_s=runtime_s, counts=counts, params=params)

"""Stage ``locate`` (LOC-04, LOC-05): association -> located candidate events, arrivals, statics.

Reads ``assoc_events.parquet``, ``assoc_picks.parquet``, the picks table the association read
(``associator.picksTable``) and ``stations.parquet`` (its ``usedInRun`` stations), locates with
the configured statics (``hq.locate.statics.locate_with_statics``) and the run's cache dir, then
the synthetic recovery test
(``hq.locate.synthetic``) on the used stations that recorded picks (same tables), and writes
``events_located.parquet``, ``arrivals.parquet``, ``statics.parquet``, ``synthetic.json``
(docs/02 §2), ``locate_flags.parquet`` (H2-internal, see ``hq.locate``) and ``diagnostics.md``
(``hq.locate.diagnostics``). Null ``synthetic.sKeepProb`` / ``pickProb`` are measured from this
run's located events first (``measured_pick_stats``). The report also reads, when present,
``catalog.parquet`` and ``catalog.quakeml`` (public regional catalog comparison) and H1's
``known/windows.json`` (which public events are the known ones). Every output is written under a
``.part`` name first and moved into place only after all of them were written.

Statics (``statics.mode``). referenceEvents, the default, runs in two passes of this stage:
locate (pass 1, no statics) -> match -> locate (pass 2) -> match -> tier. When
``matches.parquet`` is in the run dir, this stage runs pass 2: the matched public events are the
reference events (mapped to their association events through the ``events_located.parquet`` and
``locate_flags.parquet`` the match read; a matches table written for other located events fails
the stage), and ``catalog.parquet`` gives their hypocentres. Without ``matches.parquet`` it runs
pass 1 and logs that the statics pass needs a match first. selfConsistent needs nothing more.
After pass 2 the run dir's ``matches.parquet`` belongs to the pass-1 locations (logged): stage
tier (``hq.tier.run.check_matches_current``) and a further locate refuse it until match reruns.

``ctx.record`` gets the counts, the runtime, the locator record (``ProcessingRun.locator``, with
the stage's conventions under ``locate``, the synthetic test's params under ``synthetic`` and the
diagnostics knobs) and the top-extended velocity model the tables were solved on
(``ProcessingRun.velocityModel``).

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

from hq.locate import LocateDetails
from hq.locate.diagnostics import (
    DiagnosticsInputs,
    PreviousLocation,
    build_diagnostics,
    catalog_uncertainties,
)
from hq.locate.locator import LocatorSetup
from hq.locate.result import ARRIVALS_MODEL, EVENTS_MODEL, FLAGS_MODEL, STATICS_MODEL
from hq.locate.statics import REFERENCE_EVENTS, locate_with_statics, reference_pairs
from hq.locate.synthetic import (
    SyntheticResult,
    measured_pick_stats,
    run_synthetic,
    with_pick_stats,
    write_synthetic_json,
)
from hq.locate.velocity import load_configured_model

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
SYNTHETIC_JSON = "synthetic.json"  # docs/02 SyntheticTest
# Synthetic params left out of the run record: ProcessingRun.locator and .velocityModel hold them.
SYNTHETIC_PARAMS_SKIPPED = ("locator", "velocityModel")
CATALOG_TABLE = "catalog.parquet"
CATALOG_QUAKEML = "catalog.quakeml"
MATCHES_TABLE = "matches.parquet"  # a prior match pass: the reference events (LOC-05)
KNOWN_WINDOWS = "known/windows.json"
PART_SUFFIX = ".part"
# Model name each input's parquet metadata must carry (docs/02 §2; the assoc tables are LOC-03's,
# matches MATCH-02's, events_located and locate_flags this stage's own).
INPUT_MODELS = {"picks": "Pick", "stations": "Station", "assoc_events": "AssocEvent",
                "assoc_picks": "AssocPick", "catalog": "CatalogEvent", "matches": "Match",
                "events_located": EVENTS_MODEL, "locate_flags": FLAGS_MODEL,
                "arrivals": ARRIVALS_MODEL}


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


def reference_input(ctx: "RunContext") -> pd.DataFrame | None:
    """Reference events for statics mode referenceEvents: ``reference_pairs`` from the run dir's
    matches, or None (logged) when ``matches.parquet`` is absent or the mode is another."""
    if ctx.config.seismology.statics.mode != REFERENCE_EVENTS:
        return None
    path = ctx.path(MATCHES_TABLE)
    if not path.is_file():
        log.info("locate: statics mode referenceEvents: no %s yet; this is pass 1 (no statics). "
                 "The statics pass needs a match first: rerun stage locate after stage match",
                 MATCHES_TABLE)
        return None
    needed = [ctx.path(n) for n in (EVENTS_TABLE, FLAGS_TABLE, CATALOG_TABLE)]
    absent = [p.name for p in needed if not p.is_file()]
    if absent:
        raise ValueError(f"{path} is present but {absent} are not: the reference statics pass "
                         "maps matched events through the tables the match read")
    return reference_pairs(
        _read(path, "matches"), _read(ctx.path(EVENTS_TABLE), "events_located"),
        _read(ctx.path(FLAGS_TABLE), "locate_flags"), _read(ctx.path(CATALOG_TABLE), "catalog"),
        ctx.config.run,
    )


def synthetic_stations(used: pd.DataFrame, picks: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The used stations that recorded at least one pick in this run's picks table.

    A usedInRun station that recorded no pick in this run (for whatever reason: no data served,
    a dead channel, a quiet window) would get perfect synthetic picks and flatter the recovery
    test, so it is left out and named in the run record (``synthetic.stationsWithoutPicks``) and
    in diagnostics.md.
    """
    with_picks = set(picks["stationId"].astype(str))
    keep = used["id"].astype(str).isin(with_picks).to_numpy()
    dropped = sorted(used.loc[~keep, "id"].astype(str))
    kept = used[keep].reset_index(drop=True)
    if kept.empty:
        raise ValueError("synthetic test: no used station recorded a pick in this run")
    return kept, dropped


def synthetic_test(
    ctx: "RunContext", details: LocateDetails, picks: pd.DataFrame, stations: pd.DataFrame
) -> SyntheticResult:
    """The synthetic recovery test on the run's used stations that recorded picks, with measured
    pick stats; ``params['stationsWithoutPicks']`` names the used stations left out."""
    cfg = ctx.config.seismology
    syn = cfg.synthetic
    stats = (
        measured_pick_stats(details.result.events, picks)
        if syn.sKeepProb is None or syn.pickProb is None else None
    )
    kind = stations.assign(id=stations["id"].astype(str)).set_index("id")["kind"].astype(str)
    used = details.stations.assign(kind=kind.reindex(details.stations["id"]).to_numpy())
    used, no_picks = synthetic_stations(used, picks)
    if no_picks:
        log.warning("locate: synthetic test leaves out %d used station(s) with no picks: %s",
                    len(no_picks), no_picks)
    setup = LocatorSetup(
        stations=used,
        model=load_configured_model(cfg.velocity),
        config=cfg if stats is None else with_pick_stats(cfg, stats),
        run=ctx.config.run,
        cache_dir=ctx.cache_dir,
    )
    result = run_synthetic(setup, geometry_label=ctx.run_id, pick_stats=stats)
    result.params["stationsWithoutPicks"] = no_picks
    return result


def previous_location(ctx: "RunContext", events: pd.DataFrame | None) -> PreviousLocation | None:
    """The run dir's events_located / locate_flags / arrivals from before this stage run (the
    1D vs 3D section of diagnostics.md compares against them), or None when any is absent."""
    paths = [ctx.path(n) for n in (FLAGS_TABLE, ARRIVALS_TABLE)]
    if events is None or not all(p.is_file() for p in paths):
        return None
    return PreviousLocation(events=events, flags=_read(paths[0], "locate_flags"),
                            arrivals=_read(paths[1], "arrivals"))


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

    reference = reference_input(ctx)
    previous = (_read(ctx.path(EVENTS_TABLE), "events_located")
                if ctx.path(EVENTS_TABLE).is_file() else None)
    earlier = previous_location(ctx, previous)
    outcome = locate_with_statics(assoc, picks, stations, cfg, run_cfg, run_id=ctx.run_id,
                                  cache_dir=ctx.cache_dir, reference=reference,
                                  previous_events=previous)
    details = outcome.details
    synthetic = synthetic_test(ctx, details, picks, stations)
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
            synthetic=synthetic,
            statics=outcome.report,
            previous=earlier,
        )
    )

    tables = {
        ctx.path(EVENTS_TABLE): (details.result.events, EVENTS_MODEL),
        ctx.path(ARRIVALS_TABLE): (details.result.arrivals, ARRIVALS_MODEL),
        ctx.path(STATICS_TABLE): (details.result.statics, STATICS_MODEL),
        ctx.path(FLAGS_TABLE): (details.flags, FLAGS_MODEL),
    }
    report_path = ctx.path(DIAGNOSTICS)
    synthetic_path = ctx.path(SYNTHETIC_JSON)
    targets = [*tables, synthetic_path, report_path]
    try:
        for path, (frame, model_name) in tables.items():
            write_table(frame, _part(path), model_name)
        write_synthetic_json(_part(synthetic_path), synthetic.report)
        _part(report_path).write_text(report, encoding="utf-8")
        for path in targets:
            os.replace(_part(path), path)
    finally:
        for path in targets:
            _part(path).unlink(missing_ok=True)

    runtime_s = time.perf_counter() - started
    rep = outcome.report
    counts = {
        **details.counts, "syntheticEvents": synthetic.report.nEvents,
        "syntheticStations": int(synthetic.params["nStations"]),
        "syntheticStationsWithoutPicks": len(synthetic.params["stationsWithoutPicks"]),
        "staticsPass": rep.pass_number,
        "staticsReferenceEvents": 0 if rep.reference is None else len(rep.reference),
        "staticsNonZero": int((rep.terms["staticS"] != 0.0).sum()),
        "staticsAboveFlag": len(rep.explanations),
        "staticsUnexplained": int((rep.explanations["verdict"] == "unexplained").sum()),
        "staticsFarHypothesis": int((rep.explanations["verdict"] == "far").sum()),
    }
    params: dict[str, Any] = {
        **details.record,
        "synthetic": {
            "report": synthetic.report.model_dump(mode="json"),
            **{k: v for k, v in synthetic.params.items() if k not in SYNTHETIC_PARAMS_SKIPPED},
        },
        "input": {
            "picksTable": cfg.associator.picksTable,
            "stationsTable": STATIONS_TABLE,
            "assocTables": [ASSOC_EVENTS_TABLE, ASSOC_PICKS_TABLE],
            "referenceMatches": MATCHES_TABLE if reference is not None else None,
        },
        "outputs": [p.name for p in targets],
        "diagnostics": cfg.diagnostics.model_dump(mode="json"),
        "statics": {"config": cfg.statics.model_dump(mode="json"), **rep.to_record()},
        "locateRuntimeS": details.runtime_s,
    }
    log.info("locate: wrote %s in %.1f s", ", ".join(p.name for p in targets), runtime_s)
    if reference is not None:
        log.warning("locate: pass 2 relocated every event; %s still holds the match of the "
                    "pass-1 locations: rerun stage match before tier (stage tier and a further "
                    "locate refuse it until then)", MATCHES_TABLE)
    ctx.record(STAGE, runtime_s=runtime_s, counts=counts, params=details.velocity_model,
               field="velocityModel")
    ctx.record(STAGE, runtime_s=runtime_s, counts=counts, params=params)

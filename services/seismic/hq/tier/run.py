"""Stage ``tier`` (LOC-06): located events + matches -> ``events.parquet``, ``event_picks.parquet``.

Reads ``events_located.parquet``, ``matches.parquet``, ``arrivals.parquet``, ``stations.parquet``,
the picks table the association read (``associator.picksTable``) and LOC-04's
``locate_flags.parquet``. Stage locate writes the flags with every ``events_located.parquet``, so
a run dir without them fails the stage: tiering without them would drop the ``mapOnVolumeTop``
rule and change tiers (``assign_tiers`` keeps that fallback for the docs/02 three-argument call
only). Checks every event's ``depthKm`` against ``(run.refSurfaceElevM - elevM) / 1000``
within ``tiering.consistencyTolM``, runs ``hq.tier.assign_tiers`` (with arrivals and stations, so
the nearest-station rule measures depth below the nearest used station's sensor) and
``hq.tier.picks.event_picks``, and writes ``events.parquet`` (final ``SeismicEvent`` rows) and
``event_picks.parquet`` (docs/02 §2).

``matches.parquet`` must have been written for this ``events_located.parquet``
(``check_matches_current`` against ``catalog.parquet``): stage ``locate``'s reference-statics pass
(LOC-05) relocates every event after a match, and the match from before it would put the
pre-statics offsets into ``catalogMatch``. A stale one fails the stage: rerun stage match first.

With ``tiering.sweep.enabled`` it also runs the association sweep (``hq.tier.sweep``, with the
bars just derived and the run's ``statics.parquet``; with ``statics.mode`` referenceEvents those
terms come from the very public events the points are matched to, so the points' recall is
in-sample: warned and recorded as ``sweep.statics.inSample``) and writes ``sweep.parquet``; disabled, it
removes a ``sweep.parquet`` left by an earlier tier run, whose Tier A counts used that run's bars
and rules, so ``Validation.sweep`` stays empty until a tier run with the sweep enabled (logged and
recorded). Every output is written under a ``.part`` name first and moved into place only after
all of them were written.

Run mode live (``hq.locate.calibration``): the bars are the calibration run's
``ProcessingRun.tiering`` thresholds, applied unchanged (``thresholdSource`` supplied), since a
short window has too few matched events; ``ProcessingRun.tiering["calibration"]`` records the run.

``events.parquet`` leaves this stage with every magnitude null; stage magnitude (MAG-01) fills
them. So a ``magnitude.json`` from an earlier magnitude run is removed before the new
``events.parquet`` moves into place (a crash between the two leaves no calibration next to
events without magnitudes), and ``ProcessingRun.matching["magnitude"]`` is replaced by a status
that says so, until stage magnitude writes its own record again (logged and recorded).

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
from hq.locate.calibration import load_calibration
from hq.locate.provenance import stage_provenance
from hq.tier import Thresholds, TierError, assign_tiers, matched_rows
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
STATICS_TABLE = "statics.parquet"
CATALOG_TABLE = "catalog.parquet"
EVENTS_TABLE = "events.parquet"
EVENT_PICKS_TABLE = "event_picks.parquet"
SWEEP_TABLE = "sweep.parquet"
MAGNITUDE_JSON = "magnitude.json"  # stage magnitude's MagCalibration (MAG-01)
# Stage magnitude records under ProcessingRun.matching["magnitude"] (FYI-H2-7).
MAGNITUDE_FIELD = "matching"
MAGNITUDE_KEY = "magnitude"
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
    "statics": "StationStatic",
}
STATIC_COLUMNS: tuple[str, ...] = ("stationId", "phase", "staticS")
# |t_event - t_catalog - matches.dtS| above this means matches.parquet was written for another
# events_located.parquet (parquet keeps float64 exactly; this only absorbs float round-off).
ROUNDOFF_S = 1e-6


def _part(path: Path) -> Path:
    return path.with_name(path.name + PART_SUFFIX)


def _read(path: Path, what: str) -> pd.DataFrame:
    frame = read_table(path)
    model = frame.attrs.get("model")
    if model != INPUT_MODELS[what]:
        raise TierError(f"{path} holds {model!r} rows; the {what} input must hold "
                        f"{INPUT_MODELS[what]!r} rows")
    return frame


def check_depth(events: pd.DataFrame, run: RunSection, tol_m: float) -> None:
    """``depthKm`` must equal ``(refSurfaceElevM - elevM) / 1000`` (docs/02) for this run, within
    ``tol_m`` metres (``tiering.consistencyTolM``)."""
    missing = [c for c in ("id", "elevM", "depthKm") if c not in events.columns]
    if missing:
        raise TierError(f"events_located lacks columns {missing}")
    elev = events["elevM"].to_numpy(dtype=np.float64)
    expected = (run.refSurfaceElevM - elev) / 1000.0
    stored = events["depthKm"].to_numpy(dtype=np.float64)
    bad = np.flatnonzero(~(np.abs(stored - expected) * 1000.0 <= tol_m))  # NaN fails too
    if bad.size:
        k = int(bad[0])
        raise TierError(
            f"{bad.size} located events have depthKm != (refSurfaceElevM {run.refSurfaceElevM} "
            f"- elevM) / 1000 within {tol_m} m, e.g. {events['id'].iloc[k]}: {stored[k]} vs "
            f"{expected[k]}; events_located.parquet was written with another run section"
        )


def check_matches_current(
    events_located: pd.DataFrame, matches: pd.DataFrame, catalog: pd.DataFrame, tol_m: float
) -> None:
    """Refuse a ``matches`` table written for other located events than ``events_located``.

    Every matched row's ``dtS`` must equal ``t_event - t_catalog`` within ``ROUNDOFF_S`` and its
    ``distM`` the ENU epicentral distance within ``tol_m`` (``tiering.consistencyTolM``), as stage
    match computes them (``hq.match``). Stage locate's reference-statics pass (LOC-05) moves
    every event after a match, so the match from before it fails here.
    """
    rows = matched_rows(matches, events_located["id"].astype(str))
    ev = events_located.set_index(events_located["id"].astype(str))
    if not ev.index.is_unique:
        raise TierError("events_located: duplicate ids")
    missing = [c for c in ("id", "t", "enu_e", "enu_n") if c not in catalog.columns]
    if missing:
        raise TierError(f"{CATALOG_TABLE} lacks columns {missing}")
    cat = catalog.set_index(catalog["id"].astype(str))
    unknown = sorted(set(rows["catalogId"]) - set(cat.index))
    if unknown:
        raise TierError(f"matches name public events missing from {CATALOG_TABLE}: {unknown[:5]}")
    e, c = ev.loc[rows["eventId"]], cat.loc[rows["catalogId"]]
    dt = e["t"].to_numpy(dtype=np.float64) - c["t"].to_numpy(dtype=np.float64)
    dist = np.hypot(e["enu_e"].to_numpy(dtype=np.float64) - c["enu_e"].to_numpy(dtype=np.float64),
                    e["enu_n"].to_numpy(dtype=np.float64) - c["enu_n"].to_numpy(dtype=np.float64))
    stored_dt = rows["dtS"].to_numpy(dtype=np.float64)
    stored_dist = rows["distM"].to_numpy(dtype=np.float64)
    bad = np.flatnonzero(~((np.abs(dt - stored_dt) <= ROUNDOFF_S)
                           & (np.abs(dist - stored_dist) <= tol_m)))
    if bad.size:
        k = int(bad[0])
        raise TierError(
            f"{MATCHES_TABLE} is stale: {bad.size} matched event(s) have another origin time or "
            f"epicentre in {EVENTS_LOCATED_TABLE} than when matched, e.g. {rows['eventId'].iloc[k]}"
            f": dtS {stored_dt[k]:+.6f} s stored vs {dt[k]:+.6f} s now, distM {stored_dist[k]:.3f} "
            f"vs {dist[k]:.3f} m (stage locate relocated them, e.g. the LOC-05 statics pass 2): "
            "rerun stage match, then tier"
        )


def run_statics(ctx: "RunContext") -> dict[tuple[str, str], float]:
    """The run's ``statics.parquet`` as ``{(stationId, phase): staticS}`` (LOC-04's statics hook),
    so sweep points are located with the statics ``events_located.parquet`` was located with."""
    frame = _read(ctx.path(STATICS_TABLE), "statics")
    missing = [c for c in STATIC_COLUMNS if c not in frame.columns]
    if missing:
        raise TierError(f"{STATICS_TABLE} lacks columns {missing}")
    keys = list(zip(frame["stationId"].astype(str), frame["phase"].astype(str), strict=True))
    if len(set(keys)) != len(keys):
        raise TierError(f"{STATICS_TABLE}: a station-phase appears twice")
    values = frame["staticS"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise TierError(f"{STATICS_TABLE}: non-finite staticS")
    return dict(zip(keys, values.tolist(), strict=True))


def _sweep(
    ctx: "RunContext",
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    thresholds: Thresholds,
    tier_a: int,
) -> tuple[list[SweepPoint], dict[str, Any]]:
    from hq.tier.sweep import real_pipeline, score_sweep

    cfg = ctx.config.seismology
    catalog = _read(ctx.path(CATALOG_TABLE), "catalog")
    statics = run_statics(ctx)
    # referenceEvents terms were estimated from the public events every point is matched to.
    in_sample = (cfg.statics.mode == "referenceEvents"
                 and any(v != 0.0 for v in statics.values()))
    if in_sample:
        log.warning(
            "tier sweep: %s holds reference-event terms estimated from the public events each "
            "point is matched to, so every point's recoveredPublic and matched-event Tier A are "
            "in-sample, not held out (events.parquet relocated each reference event with terms "
            "computed without it); recorded as sweep.statics.inSample", STATICS_TABLE)
    run_points, pipeline = real_pipeline(
        picks, stations, catalog, cfg, ctx.config.run, run_id=ctx.run_id, cache_dir=ctx.cache_dir,
        statics=statics,
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
            "events.parquet has %d (a fresh association of the same picks, every event located "
            "with %s%s)",
            configured, at_configured[0]["tierA"], tier_a, STATICS_TABLE,
            ", where events.parquet used held-out terms for the reference events" if in_sample
            else "",
        )
    return points, {
        "enabled": True,
        "grid": "associator.sweep (minStations x nSPicks x minPickProb)",
        "tierABars": "the bars in 'thresholds' (derived from this run's matched set)",
        "configuredPoint": configured,
        "configuredPointTierAInEvents": tier_a,
        "note": "each point: associate -> locate (with the run's statics.parquet) -> match -> "
        "tiers; candidates = located candidate events",
        "statics": {
            "table": STATICS_TABLE,
            "stationPhases": len(statics),
            "nonZero": sum(1 for v in statics.values() if v != 0.0),
            "maxAbsS": max((abs(v) for v in statics.values()), default=0.0),
            "inSample": in_sample,
            "note": ("statics.mode referenceEvents: these terms were estimated from the public "
                     "events every point is matched to, so recoveredPublic and the matched "
                     "events' tiers are in-sample, not held out" if in_sample else
                     "no reference-event terms: nothing in-sample"),
        },
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
    stations = _read(ctx.path(STATIONS_TABLE), "stations")
    flags_path = ctx.path(FLAGS_TABLE)
    if not flags_path.is_file():
        raise TierError(f"{flags_path} is absent: stage locate writes it with "
                        f"{EVENTS_LOCATED_TABLE}, and without it Tier A would skip the "
                        "mapOnVolumeTop rule; rerun stage locate, then match")
    flags = _read(flags_path, "flags")
    log.info("tier: %d located candidate events, %d matches rows, %d arrivals, %d picks from %s",
             len(events_located), len(matches), len(arrivals), len(picks), picks_path)
    check_depth(events_located, ctx.config.run, cfg.tiering.consistencyTolM)
    catalog_path = ctx.path(CATALOG_TABLE)
    if not catalog_path.is_file():
        raise TierError(f"{catalog_path} is absent: {MATCHES_TABLE} can't be checked against the "
                        "public events it was matched from; rerun stage catalog and match")
    check_matches_current(events_located, matches, _read(catalog_path, "catalog"),
                          cfg.tiering.consistencyTolM)

    # Live: bars from the calibration run (a short window has too few matched events to derive
    # them); every other mode derives them from this run's matched set.
    calibration = load_calibration(ctx)
    result = assign_tiers(
        events_located, matches, cfg, flags=flags, arrivals=arrivals, stations=stations,
        thresholds=None if calibration is None else calibration.tiering,
    )
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
            ctx, picks, stations, Thresholds.from_record(tiering["thresholds"]), n_a
        )
    else:
        sweep_record = {"enabled": False, "removedEarlierSweep": sweep_path.exists()}

    tables: dict[Path, tuple[pd.DataFrame, str]] = {
        ctx.path(EVENTS_TABLE): (result.events, SeismicEvent.__name__),
        ctx.path(EVENT_PICKS_TABLE): (picks_out, Pick.__name__),
    }
    if sweep_points is not None:
        tables[sweep_path] = (to_frame(sweep_points, SweepPoint), SweepPoint.__name__)
    magnitude_path = ctx.path(MAGNITUDE_JSON)
    removed_magnitude = magnitude_path.exists()
    try:
        for path, (frame, model_name) in tables.items():
            write_table(frame, _part(path), model_name)
        # The new events.parquet has no magnitudes: the old calibration goes first, so a crash
        # between the two leaves no magnitude.json next to it, never the old one.
        magnitude_path.unlink(missing_ok=True)
        for path in tables:
            os.replace(_part(path), path)
    finally:
        for path in tables:
            _part(path).unlink(missing_ok=True)
    if removed_magnitude:
        log.warning("tier: removed %s from an earlier magnitude run: %s now has every magnitude "
                    "null; rerun stage magnitude", magnitude_path, EVENTS_TABLE)
    if sweep_points is None and sweep_path.exists():
        sweep_path.unlink()
        log.warning("tier: removed %s from an earlier tier run: tiering.sweep.enabled is false and "
                    "its Tier A counts used that run's bars and rules. Validation.sweep stays "
                    "empty until a tier run with tiering.sweep.enabled true", sweep_path)

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
            "stations": STATIONS_TABLE,
            "flags": FLAGS_TABLE,
            "catalog": CATALOG_TABLE,
        },
        "matchesChecked": "every matched row's dtS / distM equals what events_located and "
        "catalog give now (a match from before stage locate's statics pass 2 is refused)",
        "outputs": [p.name for p in tables],
        "eventPicks": "one Pick per id in each final event's pickIds (picks used in the final "
        "location); eventId set, residualS from arrivals.parquet",
        "final": "SeismicEvent rows as this stage writes them: located columns unchanged, tier, "
        "tierReasons, catalogMatch from matches.parquet, magnitude null (stage magnitude, MAG-01, "
        "fills it afterwards), revealOrder -1",
        "removedMagnitudeJson": removed_magnitude,
        "sweep": sweep_record,
        "provenance": stage_provenance(),
    }
    if calibration is not None:  # live only, so a showcase record keeps its keys
        params["calibration"] = {
            **calibration.to_record(),
            "note": "bars are the calibration run's ProcessingRun.tiering thresholds (its "
            "matched set), applied unchanged; none derived from this window",
        }
    magnitude_status = {
        "status": f"none: stage tier rewrote {EVENTS_TABLE} with every magnitude null; stage "
        f"magnitude (MAG-01) fills them, writes {MAGNITUDE_JSON} and replaces this record",
        "removedMagnitudeJson": removed_magnitude,
    }
    runtime_s = time.perf_counter() - started
    log.info(
        "tier: wrote %s: all A %d / B %d / C %d, additional A %d / B %d / C %d, %d event picks, "
        "%d sweep points in %.1f s",
        ", ".join(p.name for p in tables), *counts["all"].values(),
        *counts["additional"].values(), len(picks_out), stage_counts["sweepPoints"], runtime_s,
    )
    ctx.record(STAGE, runtime_s=runtime_s, counts=stage_counts,
               params={MAGNITUDE_KEY: magnitude_status}, field=MAGNITUDE_FIELD)
    ctx.record(STAGE, runtime_s=runtime_s, counts=stage_counts, params=params)

"""Stage ``match`` (MATCH-02): located candidate events + public regional catalog -> matches.

Reads ``events_located.parquet`` and ``catalog.parquet`` (both required) and checks that their
ENU columns (and those of ``stations.parquet``, when present) agree with latitude, longitude and
elevation in the run's frame. Runs ``hq.match.match``, explains every unmatched public event from
whichever evidence tables exist in the run dir (``stations``, ``gaps``, the picks table the
association read (``associator.picksTable``), ``assoc_picks`` and ``statics``; see
``hq.match.reasons``), and writes ``matches.parquet``,
``match_sensitivity.parquet`` and ``catalog.parquet`` with ``matchedEventId`` filled (docs/02 §2:
null until match).

Writes: all three files are written under ``.part`` names first and moved into place only after
every one was written, ``catalog.parquet`` last. Each move is atomic, so a failure before the first
move leaves the previous files untouched. A failure between moves is logged as an error naming the
files already replaced (rerun the stage). Parameters go to ``ProcessingRun.matching`` under the
``match`` key, so they never overwrite the ``catalog`` stage's key there.
"""

import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from hq_contracts.io import read_table, write_table

from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig
from hq.locate.coords import to_enu
from hq.locate.provenance import stage_provenance
from hq.locate.velocity import load_configured_model
from hq.match import REASONS, STRING_DTYPE, Tolerance, match
from hq.match import catalog as catalog_stage
from hq.match.reasons import ArrivalModel, Evidence, Explained, explain_unmatched

if TYPE_CHECKING:
    from hq.runs import RunContext

log = logging.getLogger(__name__)

STAGE = "match"
EVENTS_NAME = "events_located.parquet"
MATCHES_NAME = "matches.parquet"
SENSITIVITY_NAME = "match_sensitivity.parquet"
# Model names in the parquet metadata of the two tables with no docs/02 model of their own.
MATCHES_MODEL = "Match"
SENSITIVITY_MODEL = "MatchSensitivity"
# Evidence tables in the run dir; "picks" is the associator's picksTable (evidence_files).
EVIDENCE_FILES = {
    "stations": "stations.parquet",
    "gaps": "gaps.parquet",
    "picks": "picks.parquet",
    "assoc_picks": "assoc_picks.parquet",
    "statics": "statics.parquet",
}
# Models the evidence tables must hold, where the table has one (gaps and assoc_picks have none;
# statics is LOC-04's StationStatic table).
EVIDENCE_MODELS = {"stations": "Station", "picks": "Pick", "statics": "StationStatic"}
# SeismicEvent fields that events_located.parquet leaves out (docs/02 §2): a table carrying any
# of them is a final events table, not located events.
FINAL_EVENT_FIELDS = ("tier", "tierReasons", "catalogMatch", "magnitude")
# Table -> the elevation column its ENU u is measured from (u = elevation - origin.elevM).
ENU_ELEVATION = {"catalog": "elevM", "events_located": "elevM", "stations": "sensorElevM"}
_PART_SUFFIX = ".part"


def _part(path: Path) -> Path:
    return path.with_name(path.name + _PART_SUFFIX)


def evidence_files(cfg: SeismologyConfig) -> dict[str, str]:
    """``EVIDENCE_FILES`` with the picks table the association read, as stages associate,
    locate and tier read it."""
    return {**EVIDENCE_FILES, "picks": cfg.associator.picksTable}


def load_evidence(ctx: "RunContext") -> Evidence:
    """The evidence tables present in the run dir; an absent file is ``None``."""
    tables: dict[str, pd.DataFrame | None] = {}
    for key, name in evidence_files(ctx.config.seismology).items():
        path = ctx.path(name)
        table = read_table(path) if path.is_file() else None
        expected = EVIDENCE_MODELS.get(key)
        if table is not None and expected is not None and table.attrs["model"] != expected:
            raise ValueError(f"{path} holds {table.attrs['model']!r} rows, not {expected}")
        tables[key] = table
    return Evidence(**tables)


def check_enu_frame(df: pd.DataFrame, name: str, run: RunSection, tol_m: float) -> None:
    """Fail unless stored ``enu_e/n/u`` match ``to_enu`` of latitude/longitude/elevation."""
    elev = ENU_ELEVATION[name]
    columns = ["latitude", "longitude", elev, "enu_e", "enu_n", "enu_u"]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing} for the ENU frame check")
    if df.empty:
        return
    values = {c: df[c].to_numpy(dtype=np.float64) for c in columns}
    for c in ("latitude", "longitude", elev):
        if not np.isfinite(values[c]).all():
            bad = sorted(df.loc[~np.isfinite(values[c]), "id"].astype(str))
            raise ValueError(f"{name}: non-finite {c} for {bad}")
    e, n, u = to_enu(values["latitude"], values["longitude"], values[elev], run.origin)
    diff = np.max(
        np.abs(np.stack([e - values["enu_e"], n - values["enu_n"], u - values["enu_u"]])), axis=0
    )
    if not np.isfinite(diff).all() or diff.max() > tol_m:
        worst = int(np.nanargmax(np.where(np.isfinite(diff), diff, np.inf)))
        raise ValueError(
            f"{name}: stored ENU differs from ENU of latitude/longitude/{elev} (EPSG:32612 minus "
            f"run origin) by up to {diff[worst]:.3f} m at id {df['id'].iloc[worst]}, over "
            f"enuConsistencyM {tol_m:g} m: another frame or origin?"
        )


def with_matched_ids(catalog: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """``catalog`` with ``matchedEventId`` set from ``matches`` (null where unmatched)."""
    by_public = matches.set_index("catalogId")["eventId"]
    ids = catalog["id"].astype(str)
    if not ids.isin(by_public.index).all() or len(by_public) != len(catalog):
        raise ValueError("matches do not hold exactly one row per catalog event")
    out = catalog.copy()
    out["matchedEventId"] = pd.Series(
        by_public.reindex(ids).to_numpy(dtype=object), index=catalog.index, dtype=STRING_DTYPE
    )
    return out


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")


def _read_inputs(ctx: "RunContext") -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    events_path = ctx.path(EVENTS_NAME)
    events = read_table(events_path)
    final = [c for c in events.columns if c.split("_")[0] in FINAL_EVENT_FIELDS]
    if final:
        raise ValueError(
            f"{events_path} carries final-event columns {final}; the match stage reads located "
            "events (docs/02 §2 events_located.parquet)"
        )
    catalog_path = ctx.path(catalog_stage.TABLE_NAME)
    catalog = read_table(catalog_path)
    if catalog.attrs.get("model") != "CatalogEvent":
        raise ValueError(f"{catalog_path} holds {catalog.attrs.get('model')!r}, not CatalogEvent")
    return events, catalog, catalog_path


def _write_outputs(
    targets: dict[Path, tuple[pd.DataFrame, str]], catalog_out: pd.DataFrame, catalog_path: Path
) -> None:
    """Write every output under ``.part``, then move them into place, catalog last."""
    order = [*targets, catalog_path]
    replaced: list[str] = []
    try:
        for path, (frame, model_name) in targets.items():
            write_table(frame, _part(path), model_name)
        catalog_stage.write_catalog(catalog_out, _part(catalog_path))
        for path in order:
            try:
                os.replace(_part(path), path)
            except OSError:
                if replaced:
                    log.error(
                        "match: moving %s into place failed after %s were replaced; the run dir "
                        "mixes new and previous match outputs: rerun the stage",
                        path.name,
                        replaced,
                    )
                raise
            replaced.append(path.name)
    finally:
        for path in order:
            _part(path).unlink(missing_ok=True)


def run(ctx: "RunContext") -> None:
    """Stage ``match``; see the module docstring."""
    started = time.perf_counter()
    cfg = ctx.config.seismology
    run_cfg = ctx.config.run
    events, catalog, catalog_path = _read_inputs(ctx)
    evidence = load_evidence(ctx)
    tol_m = cfg.matching.enuConsistencyM
    check_enu_frame(catalog, "catalog", run_cfg, tol_m)
    check_enu_frame(events, "events_located", run_cfg, tol_m)
    if evidence.stations is not None:
        check_enu_frame(evidence.stations, "stations", run_cfg, tol_m)

    result = match(events, catalog, cfg)
    # The window checks are the only users of the velocity model: load it only for them.
    arrivals = (
        ArrivalModel.from_model(load_configured_model(cfg.velocity))
        if evidence.stations is not None
        else None
    )
    explained = explain_unmatched(result, events, catalog, evidence, cfg, run_cfg, arrivals)
    matches = explained.matches
    _write_outputs(
        {
            ctx.path(MATCHES_NAME): (matches, MATCHES_MODEL),
            ctx.path(SENSITIVITY_NAME): (result.sensitivity, SENSITIVITY_MODEL),
        },
        with_matched_ids(catalog, matches),
        catalog_path,
    )

    runtime_s = time.perf_counter() - started
    tol = Tolerance.from_config(cfg.matching)
    n_public, recovered = len(matches), int(matches["eventId"].notna().sum())
    window = f"[{_iso(run_cfg.window_start_s)}, {_iso(run_cfg.window_end_s)})"
    chance = _chance(len(events), n_public, tol.max_dt_s, run_cfg)
    log.info(
        "match: recovered %d / %d public regional catalog events in window %s within %s (%d "
        "located candidate events) in %.1f s",
        recovered,
        n_public,
        window,
        tol.label(),
        len(events),
        runtime_s,
    )
    log.info(
        "match: expected chance time coincidences within +/-%g s (origin times uniform over the "
        "window, distance ignored): %.3f per public event, %.2f in total",
        tol.max_dt_s,
        chance["perPublicEvent"],
        chance["total"],
    )
    for row in matches.itertuples(index=False):
        if pd.isna(row.eventId):
            log.info("match: unmatched %s: %s", row.catalogId, row.reason)
    log.info("match: checks run %s; skipped %s", explained.checks_run, explained.checks_skipped)
    log.info("match: sensitivity (each pair is both the admissibility limit and the cost scale)")
    for s in result.sensitivity.itertuples(index=False):
        log.info(
            "match:   dt %g s, dist %g km: recovered %d / %d",
            s.dtS,
            s.distM / 1000.0,
            s.recovered,
            n_public,
        )

    by_code = {code: 0 for code in REASONS}
    for code in explained.codes.values():
        by_code[code] += 1
    counts = {
        "publicEvents": n_public,
        "locatedEvents": len(events),
        "recovered": recovered,
        "unmatched": n_public - recovered,
        **{f"unmatched.{code}": n for code, n in by_code.items()},
    }
    params = _params(cfg, explained, arrivals, result.sensitivity, window, chance)
    params["provenance"] = stage_provenance()
    ctx.record(STAGE, runtime_s=runtime_s, counts=counts, params={"match": params})


def _chance(n_located: int, n_public: int, max_dt_s: float, run: RunSection) -> dict[str, Any]:
    """Expected located events within ``+/-max_dt_s`` of a public origin by chance alone."""
    per_public = n_located * 2.0 * max_dt_s / (run.window_end_s - run.window_start_s)
    return {
        "perPublicEvent": per_public,
        "total": per_public * n_public,
        "rule": "nLocated * 2 * maxDtS / window length: located origin times uniform over the "
        "run window; the distance limit is ignored",
    }


def _params(
    cfg: SeismologyConfig,
    explained: Explained,
    arrivals: ArrivalModel | None,
    sensitivity: pd.DataFrame,
    window: str,
    chance: dict[str, Any],
) -> dict[str, Any]:
    """``ProcessingRun.matching.match``: every knob, the rules applied and what ran."""
    return {
        "config": cfg.matching.model_dump(mode="json"),  # every matching knob
        "cost": "|dt| / dtScaleS + epicentral distance / distScaleM (magnitude not used)",
        "admissible": "|dt| <= maxDtS and distance <= maxDistM (edges admissible)",
        "objective": "most admissible one-to-one pairs, then least total cost "
        "(scipy.optimize.linear_sum_assignment)",
        "dtSign": "located t minus public t",
        "distance": "horizontal ENU (EPSG:32612 minus run origin); grid scale factor negligible",
        "lowestCost": "in unmatched reasons, the located event (or associated candidate) with the "
        "lowest cost; it can be far from the public event in time or space",
        "sensitivityRule": "each pair reruns the assignment as both limit and cost scale",
        "sensitivity": [
            {"dtS": float(s.dtS), "distM": float(s.distM), "recovered": int(s.recovered)}
            for s in sensitivity.itertuples(index=False)
        ],
        "chanceTimeCoincidences": chance,
        "window": window,
        "checksRun": explained.checks_run,
        "checksSkipped": explained.checks_skipped,
        # Recorded only when the window checks ran (they are its only users).
        "arrivalWindows": None
        if arrivals is None
        else arrivals.to_record(cfg.matching.reasons.arrivalPadS),
        "evidenceFiles": evidence_files(cfg),
        "reasonPrefixes": dict(REASONS),
        "unmatched": {
            str(r.catalogId): str(r.reason)
            for r in explained.matches.itertuples(index=False)
            if pd.isna(r.eventId)
        },
    }

"""Stage ``match`` (MATCH-02): located events + public catalog -> matches and sensitivity.

Reads ``events_located.parquet`` and ``catalog.parquet`` (both required), runs ``hq.match.match``,
explains every unmatched public event from whichever evidence tables exist in the run dir
(``stations``, ``gaps``, ``picks``, ``assoc_picks``; see ``hq.match.reasons``), and writes
``matches.parquet``, ``match_sensitivity.parquet`` and ``catalog.parquet`` with
``matchedEventId`` filled (docs/02 §2: null until match). All three are written under ``.part``
names and moved into place only after every one of them was written, so a failed run leaves the
previous files untouched. Parameters go to ``ProcessingRun.matching`` under the ``match`` key, so
they never overwrite the ``catalog`` stage's key there.
"""

import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from hq_contracts.io import read_table, write_table

from hq.config.seismology import SeismologyConfig
from hq.locate.velocity import load_configured_model
from hq.match import REASONS, STRING_DTYPE, Tolerance, match
from hq.match import catalog as catalog_stage
from hq.match.reasons import Evidence, Explained, SpeedBounds, explain_unmatched

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
EVIDENCE_FILES = {
    "stations": "stations.parquet",
    "gaps": "gaps.parquet",
    "picks": "picks.parquet",
    "assoc_picks": "assoc_picks.parquet",
}
# docs/02 models the evidence tables must hold, where the table has one (gaps and assoc_picks
# have none in docs/02).
EVIDENCE_MODELS = {"stations": "Station", "picks": "Pick"}
_PART_SUFFIX = ".part"


def _part(path: Path) -> Path:
    return path.with_name(path.name + _PART_SUFFIX)


def load_evidence(ctx: "RunContext") -> Evidence:
    """The evidence tables present in the run dir; an absent file is ``None``."""
    tables: dict[str, pd.DataFrame | None] = {}
    for key, name in EVIDENCE_FILES.items():
        path = ctx.path(name)
        table = read_table(path) if path.is_file() else None
        expected = EVIDENCE_MODELS.get(key)
        if table is not None and expected is not None and table.attrs["model"] != expected:
            raise ValueError(f"{path} holds {table.attrs['model']!r} rows, not {expected}")
        tables[key] = table
    return Evidence(**tables)


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


def run(ctx: "RunContext") -> None:
    """Stage ``match``; see the module docstring."""
    started = time.perf_counter()
    cfg = ctx.config.seismology
    run_cfg = ctx.config.run
    events = read_table(ctx.path(EVENTS_NAME))
    catalog_path = ctx.path(catalog_stage.TABLE_NAME)
    catalog = read_table(catalog_path)
    if catalog.attrs.get("model") != "CatalogEvent":
        raise ValueError(f"{catalog_path} holds {catalog.attrs.get('model')!r}, not CatalogEvent")

    result = match(events, catalog, cfg)
    evidence = load_evidence(ctx)
    speeds = SpeedBounds.from_model(load_configured_model(cfg.velocity))
    explained = explain_unmatched(result, events, catalog, evidence, cfg, run_cfg, speeds)
    matches = explained.matches
    catalog_out = with_matched_ids(catalog, matches)

    targets = {
        ctx.path(MATCHES_NAME): (matches, MATCHES_MODEL),
        ctx.path(SENSITIVITY_NAME): (result.sensitivity, SENSITIVITY_MODEL),
    }
    parts = [_part(p) for p in (*targets, catalog_path)]
    try:
        for path, (frame, model_name) in targets.items():
            write_table(frame, _part(path), model_name)
        # The catalog stage's own writer, so the rewritten table keeps its exact Arrow schema.
        catalog_stage._write_table(catalog_out, _part(catalog_path))
        for path in (*targets, catalog_path):
            os.replace(_part(path), path)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)

    runtime_s = time.perf_counter() - started
    tol = Tolerance.from_config(cfg.matching)
    n_public, recovered = len(matches), int(matches["eventId"].notna().sum())
    window = f"[{_iso(run_cfg.window_start_s)}, {_iso(run_cfg.window_end_s)})"
    log.info(
        "match: recovered %d / %d public catalog events in window %s within %s (%d located "
        "candidate events) in %.1f s",
        recovered,
        n_public,
        window,
        tol.label(),
        len(events),
        runtime_s,
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
    params = _params(cfg, explained, speeds, result.sensitivity, window)
    ctx.record(STAGE, runtime_s=runtime_s, counts=counts, params={"match": params})


def _params(
    cfg: SeismologyConfig,
    explained: Explained,
    speeds: SpeedBounds,
    sensitivity: pd.DataFrame,
    window: str,
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
        "sensitivityRule": "each pair reruns the assignment as both limit and cost scale",
        "sensitivity": [
            {"dtS": float(s.dtS), "distM": float(s.distM), "recovered": int(s.recovered)}
            for s in sensitivity.itertuples(index=False)
        ],
        "window": window,
        "checksRun": explained.checks_run,
        "checksSkipped": explained.checks_skipped,
        "arrivalWindows": speeds.to_record(),
        "evidenceFiles": dict(EVIDENCE_FILES),
        "reasonPrefixes": dict(REASONS),
        "unmatched": {
            str(r.catalogId): str(r.reason)
            for r in explained.matches.itertuples(index=False)
            if pd.isna(r.eventId)
        },
    }

"""The run tables the exporter reads (docs/02 §2), loaded only through ``hq_contracts.io``.

``load_run_tables`` reads every input of the ``export`` stage from one run directory and checks
the cross-references the exporter relies on (event ids unique and belonging to the run, every
pick an event or an arrival names present in ``picks.parquet``). Tables that are not model rows
(``arrivals.parquet``, ``matches.parquet``) are checked column by column against docs/02.

``validation.json`` is written by VAL-01 and is optional until then: without it the bundle's
validation carries only what H2's stages already produced (``synthetic.json``,
``sweep.parquet``), or nothing at all, and the log names VAL-01.
"""

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from hq_contracts.io import from_frame, read_models, read_table
from hq_contracts.models import (
    CatalogEvent,
    Pick,
    ProcessingRun,
    SeismicEvent,
    Station,
    SweepPoint,
    SyntheticTest,
    Validation,
)
from pydantic import ValidationError

from hq.export.errors import ExportError
from hq.runs import read_run_json

log = logging.getLogger(__name__)

STATIONS_TABLE = "stations.parquet"
CATALOG_TABLE = "catalog.parquet"
EVENTS_TABLE = "events.parquet"
MATCHES_TABLE = "matches.parquet"
PICKS_TABLE = "picks.parquet"
ARRIVALS_TABLE = "arrivals.parquet"
SWEEP_TABLE = "sweep.parquet"
SYNTHETIC_JSON = "synthetic.json"
VALIDATION_JSON = "validation.json"

# docs/02 §2 column lists of the two non-model tables the exporter reads.
ARRIVAL_COLUMNS: tuple[str, ...] = (
    "eventId",
    "stationId",
    "phase",
    "tPred",
    "tObs",
    "residualS",
    "pickId",
    "usedInLocation",
)
MATCH_COLUMNS: tuple[str, ...] = ("catalogId", "eventId", "dtS", "distM", "reason")

MAX_LISTED_IDS = 5  # ids quoted in an error message before "..."


@dataclass(frozen=True)
class RunTables:
    """Everything the exporter reads from ``runs/<runId>/``."""

    run: ProcessingRun
    stations: list[Station]
    catalog: list[CatalogEvent]
    events: list[SeismicEvent]
    matches: pd.DataFrame  # MATCH_COLUMNS, one row per public event
    arrivals: pd.DataFrame  # ARRIVAL_COLUMNS, one row per event x station x phase
    picks: dict[str, Pick]  # every pick an event or an arrival names, by id
    validation: Validation | None
    validation_source: str  # where ``validation`` came from, for the log


def _listed(ids: Iterable[str]) -> str:
    items = sorted(ids)
    shown = ", ".join(items[:MAX_LISTED_IDS])
    return shown + (", ..." if len(items) > MAX_LISTED_IDS else "")


def _require(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    if not path.is_file():
        raise ExportError(f"{path} not found; the stage that writes it has not run (docs/01)")
    return path


def _read_columns(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    df = read_table(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ExportError(f"{path} lacks columns {missing}; docs/02 §2 lists {list(columns)}")
    return df


def str_or_none(value: object) -> str | None:
    """A string cell, or None for null (pandas reads a null string as None or NaN)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def select_picks(df: pd.DataFrame, ids: set[str], path: Path) -> dict[str, Pick]:
    """The ``Pick`` rows named in ``ids``; every id must be present, exactly once."""
    if not ids:
        return {}
    subset = df[df["id"].isin(ids)]
    dupes = subset["id"][subset["id"].duplicated()]
    if not dupes.empty:
        raise ExportError(f"{path} holds duplicate pick ids: {_listed(dupes)}")
    picks = {pick.id: pick for pick in from_frame(subset, Pick)}
    missing = ids - set(picks)
    if missing:
        raise ExportError(
            f"{len(missing)} pick id(s) named by events or arrivals are not in {path}: "
            f"{_listed(missing)}"
        )
    return picks


def _load_validation(run_dir: Path) -> tuple[Validation | None, str]:
    path = run_dir / VALIDATION_JSON
    if path.is_file():
        try:
            return Validation.model_validate_json(path.read_text(encoding="utf-8")), path.name
        except ValidationError as exc:
            raise ExportError(f"{path} is not a valid Validation:\n{exc}") from exc
    synthetic_path = run_dir / SYNTHETIC_JSON
    if not synthetic_path.is_file():
        log.warning(
            "export: neither %s (VAL-01, H4 Platform) nor %s (locate, H2 Seismology) exists in "
            "%s; the bundle gets no validation.json",
            VALIDATION_JSON,
            SYNTHETIC_JSON,
            run_dir,
        )
        return None, "none"
    try:
        synthetic = SyntheticTest.model_validate(
            json.loads(synthetic_path.read_text(encoding="utf-8"))
        )
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ExportError(f"{synthetic_path} is not a valid SyntheticTest:\n{exc}") from exc
    sweep: list[SweepPoint] = []
    sources = [SYNTHETIC_JSON]
    sweep_path = run_dir / SWEEP_TABLE
    if sweep_path.is_file():
        sweep = read_models(sweep_path, SweepPoint)
        sources.append(SWEEP_TABLE)
    log.warning(
        "export: %s not found in %s (VAL-01, H4 Platform, has not run); validation.json carries "
        "only %s, with baseline%s, nullTest, gr and magnitude empty",
        VALIDATION_JSON,
        run_dir,
        " + ".join(sources),
        "" if sweep else ", sweep",
    )
    validation = Validation(
        baseline=[], sweep=sweep, nullTest=None, gr=None, magnitude=None, synthetic=synthetic
    )
    return validation, " + ".join(sources)


def load_run_tables(run_dir: Path) -> RunTables:
    """Read and cross-check every exporter input in ``run_dir``."""
    run_dir = Path(run_dir)
    run = read_run_json(run_dir)
    stations = read_models(_require(run_dir, STATIONS_TABLE), Station)
    catalog = read_models(_require(run_dir, CATALOG_TABLE), CatalogEvent)
    events = read_models(_require(run_dir, EVENTS_TABLE), SeismicEvent)
    matches = _read_columns(_require(run_dir, MATCHES_TABLE), MATCH_COLUMNS)
    arrivals = _read_columns(_require(run_dir, ARRIVALS_TABLE), ARRIVAL_COLUMNS)
    picks_path = _require(run_dir, PICKS_TABLE)
    picks_df = read_table(picks_path)
    if picks_df.attrs["model"] != Pick.__name__:
        raise ExportError(f"{picks_path} holds {picks_df.attrs['model']} rows, not Pick")

    for name, items in (("station", stations), ("catalog", catalog), ("event", events)):
        ids = [item.id for item in items]
        if len(set(ids)) != len(ids):
            dupes = {i for i in ids if ids.count(i) > 1}
            raise ExportError(f"{run_dir}: duplicate {name} ids: {_listed(dupes)}")
    foreign = {e.id for e in events if e.runId != run.id}
    if foreign:
        raise ExportError(
            f"{run_dir / EVENTS_TABLE}: events with runId != {run.id!r}: {_listed(foreign)}"
        )
    station_ids = {s.id for s in stations}
    unknown_stations = set(arrivals["stationId"].astype(str)) - station_ids
    if unknown_stations:
        raise ExportError(
            f"{run_dir / ARRIVALS_TABLE} names stations missing from {STATIONS_TABLE}: "
            f"{_listed(unknown_stations)}"
        )
    bad_phase = set(arrivals["phase"].astype(str)) - {"P", "S"}
    if bad_phase:
        raise ExportError(f"{run_dir / ARRIVALS_TABLE}: phase must be P or S, got {bad_phase}")

    pick_ids = {pid for e in events for pid in e.pickIds}
    pick_ids |= {p for p in (str_or_none(v) for v in arrivals["pickId"]) if p is not None}
    picks = select_picks(picks_df, pick_ids, picks_path)
    validation, validation_source = _load_validation(run_dir)
    log.info(
        "export: read run %s: %d stations, %d catalog events, %d events, %d matches, "
        "%d arrivals, %d picks, validation from %s",
        run.id,
        len(stations),
        len(catalog),
        len(events),
        len(matches),
        len(arrivals),
        len(picks),
        validation_source,
    )
    return RunTables(
        run=run,
        stations=stations,
        catalog=catalog,
        events=events,
        matches=matches,
        arrivals=arrivals,
        picks=picks,
        validation=validation,
        validation_source=validation_source,
    )

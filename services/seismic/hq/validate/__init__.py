"""Stage ``validate`` (H4, tickets VAL-02 and VAL-01): validation reruns -> ``validation.json``.

``run(ctx)`` reads the run's picks, stations and public catalog, reruns H2's pipeline functions
through ``hq.validate.lanes`` and writes:

- ``null_test.json``: the ``NullTest`` (VAL-02), always, so the number is never lost;
- ``validation.json``: a ``Validation`` with ``nullTest`` filled, ``sweep`` from H2's
  ``sweep.parquet`` when present, ``synthetic`` from H2's ``synthetic.json``, and ``baseline``,
  ``gr`` and ``magnitude`` empty until VAL-01 fills them.

``Validation.synthetic`` is required by the contract and only H2's locate stage produces it, so
when ``synthetic.json`` is absent the stage keeps ``hq run`` going: it writes the sidecar,
skips ``validation.json`` with a warning naming H2, and records ``validationJson: 0``.

Every knob is in ``configs/showcase/validate.yaml`` (``hq.config.validate.ValidateConfig``).
H2's functions are imported lazily, so this package loads before they are merged and a missing
one fails naming H2.
"""

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from hq_contracts.io import read_models, read_table
from hq_contracts.models import (
    CatalogEvent,
    NullTest,
    Pick,
    Station,
    SweepPoint,
    SyntheticTest,
    Validation,
)
from pydantic import BaseModel, ValidationError

from hq.config.validate import ValidateConfig
from hq.runs import write_text_atomic
from hq.validate.errors import ValidateError
from hq.validate.lanes import LaneSeismologyApi, SeismologyApi, real_seismology_api
from hq.validate.null_test import run_null_test

if TYPE_CHECKING:
    from hq.runs import RunContext

log = logging.getLogger(__name__)

STAGE = "validate"
H2 = "H2 Seismology"
PICKS_TABLE = "picks.parquet"
STATIONS_TABLE = "stations.parquet"
CATALOG_TABLE = "catalog.parquet"
SWEEP_TABLE = "sweep.parquet"
SYNTHETIC_JSON = "synthetic.json"
NULL_TEST_JSON = "null_test.json"
VALIDATION_JSON = "validation.json"

# Which stage writes each input (docs/01 -> Pipeline), for the error when it is missing.
TABLE_WRITERS: dict[str, tuple[str, str]] = {
    PICKS_TABLE: ("pick", "H1 Signal"),
    STATIONS_TABLE: ("inventory", "H1 Signal"),
    CATALOG_TABLE: ("catalog", H2),
    SWEEP_TABLE: ("associate", H2),
    SYNTHETIC_JSON: ("locate", H2),
}


def _require(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    if not path.is_file():
        stage, owner = TABLE_WRITERS[name]
        raise ValidateError(
            f"{path} not found; the {stage!r} stage (owner: {owner}) has not run (docs/01)"
        )
    return path


def read_model_table(run_dir: Path, name: str, model: type[BaseModel]) -> pd.DataFrame:
    """A run table that holds ``model`` rows, as the DataFrame H2's API takes."""
    path = _require(run_dir, name)
    df = read_table(path)
    if df.attrs["model"] != model.__name__:
        raise ValidateError(f"{path} holds {df.attrs['model']} rows, not {model.__name__}")
    return df


def read_synthetic(run_dir: Path) -> SyntheticTest | None:
    """H2's ``synthetic.json`` when the locate stage wrote it, else None (logged, naming H2)."""
    path = run_dir / SYNTHETIC_JSON
    if not path.is_file():
        stage, owner = TABLE_WRITERS[SYNTHETIC_JSON]
        log.warning(
            "validate: %s not found in %s (the %r stage, owner %s, has not written it); "
            "Validation.synthetic is required, so %s is not written this time and %s carries "
            "the null test",
            SYNTHETIC_JSON,
            run_dir,
            stage,
            owner,
            VALIDATION_JSON,
            NULL_TEST_JSON,
        )
        return None
    try:
        return SyntheticTest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ValidateError(f"{path} is not a valid SyntheticTest:\n{exc}") from exc


def read_sweep(run_dir: Path) -> list[SweepPoint]:
    """H2's ``sweep.parquet`` when the associate stage wrote it, else an empty list."""
    path = run_dir / SWEEP_TABLE
    if not path.is_file():
        log.warning(
            "validate: no %s in %s (written by H2's associate stage); Validation.sweep stays "
            "empty and the sweep plot has nothing to show",
            SWEEP_TABLE,
            run_dir,
        )
        return []
    return read_models(path, SweepPoint)


def write_json_model(path: Path, model: BaseModel) -> None:
    write_text_atomic(path, model.model_dump_json(indent=2) + "\n")


def assemble_validation(
    null_test: NullTest, synthetic: SyntheticTest, sweep: list[SweepPoint]
) -> Validation:
    """The ``Validation`` this stage can fill today; VAL-01 adds baseline, gr and magnitude."""
    return Validation(
        baseline=[], sweep=sweep, nullTest=null_test, gr=None, magnitude=None, synthetic=synthetic
    )


def validate_run(
    ctx: "RunContext", api: SeismologyApi | None = None
) -> tuple[NullTest, Validation | None]:
    """Everything ``run`` does except the timing and the record; ``api`` defaults to H2's."""
    cfg: ValidateConfig = ctx.config.validate
    seismology_cfg = ctx.config.section("seismology")  # ConfigError naming H2 when not merged
    picks = read_model_table(ctx.run_dir, PICKS_TABLE, Pick)
    stations = read_model_table(ctx.run_dir, STATIONS_TABLE, Station)
    catalog = read_model_table(ctx.run_dir, CATALOG_TABLE, CatalogEvent)
    log.info(
        "validate: run %s: %d picks, %d stations, %d catalog events",
        ctx.run_id,
        len(picks),
        len(stations),
        len(catalog),
    )
    if api is None:
        api = real_seismology_api()
    null_test = run_null_test(
        picks, stations, catalog, api, seismology_cfg, ctx.config.run, cfg.nullTest
    )
    write_json_model(ctx.path(NULL_TEST_JSON), null_test)
    synthetic = read_synthetic(ctx.run_dir)
    if synthetic is None:
        return null_test, None
    validation = assemble_validation(null_test, synthetic, read_sweep(ctx.run_dir))
    write_json_model(ctx.path(VALIDATION_JSON), validation)
    log.info(
        "validate: wrote %s (nullTest, synthetic, %d sweep points; baseline, gr and magnitude "
        "wait for VAL-01)",
        ctx.path(VALIDATION_JSON),
        len(validation.sweep),
    )
    return null_test, validation


def run(ctx: "RunContext") -> None:
    """Stage entry: the null test, ``null_test.json`` and, when H2's synthetic test exists,
    ``validation.json``; counts and runtime go to the run record."""
    started = time.perf_counter()
    null_test, validation = validate_run(ctx)
    counts = {
        "nullShuffles": null_test.nShuffles,
        # Sums over reruns (counts must be ints); the means are in null_test.json.
        "nullChanceEvents": round(null_test.meanChanceEvents * null_test.nShuffles),
        "nullChanceStrict": round(null_test.meanChanceStrict * null_test.nShuffles),
        "sweepPoints": len(validation.sweep) if validation is not None else 0,
        "validationJson": int(validation is not None),
    }
    ctx.record(STAGE, runtime_s=time.perf_counter() - started, counts=counts)


__all__ = [
    "NULL_TEST_JSON",
    "STAGE",
    "VALIDATION_JSON",
    "LaneSeismologyApi",
    "SeismologyApi",
    "ValidateError",
    "assemble_validation",
    "read_model_table",
    "read_sweep",
    "read_synthetic",
    "real_seismology_api",
    "run",
    "run_null_test",
    "validate_run",
]

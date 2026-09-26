"""Stage ``validate`` (H4, tickets VAL-02 and VAL-01): validation reruns -> ``validation.json``.

``run(ctx)`` reads the run's picks, stations and public catalog, reruns H2's pipeline functions
through ``hq.validate.lanes`` and writes (``hq.validate.sidecars``):

- ``null_test.json``: the ``NullTest`` (VAL-02), always, so the number is never lost;
- ``baseline.json``: the ``BaselineRow`` table (VAL-01), always; empty when H1's
  ``picks_stalta.parquet`` is not there (logged, naming H1);
- ``gr.json``: the ``GRCurve`` (VAL-01) when magnitudes exist and H2's ``magnitude.json`` does
  not trip the docs/03 magnitude kill switch; a stale one from an earlier rerun is removed;
- ``validation.json``: the ``Validation`` with all of the above plus ``sweep`` from H2's
  ``sweep.parquet``, ``magnitude`` from H2's ``magnitude.json`` and ``synthetic`` from H2's
  ``synthetic.json``.

``Validation.synthetic`` is required by the contract and only H2's locate stage produces it, so
when ``synthetic.json`` is absent the stage keeps ``hq run`` going: it writes the sidecars, skips
``validation.json`` with a warning naming H2, and records ``validationJson: 0``. The exporter
then assembles the ``Validation`` from the sidecars once ``synthetic.json`` appears.

``AnalysisSummary.baseline`` is derived by the exporter from ``Validation.baseline`` with
``hq.validate.baseline.baseline_gain``; this stage logs the same verdict so the run log says
whether the claim holds.

Every knob is in ``configs/showcase/validate.yaml`` (``hq.config.validate.ValidateConfig``).
H2's functions are imported lazily, so this package loads before they are merged and a missing
one fails naming H2.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from hq_contracts.io import read_models, read_table
from hq_contracts.models import (
    BaselineGain,
    BaselineRow,
    CatalogEvent,
    GRCurve,
    MagCalibration,
    NullTest,
    Pick,
    SeismicEvent,
    Station,
    SweepPoint,
    SyntheticTest,
    Validation,
)
from pydantic import BaseModel

from hq.config.validate import ValidateConfig
from hq.runs import write_text_atomic
from hq.validate import sidecars
from hq.validate.baseline import baseline_gain, run_baseline
from hq.validate.errors import ValidateError
from hq.validate.gr import as_magnitudes, gr_allowed, gr_curve
from hq.validate.lanes import LaneSeismologyApi, SeismologyApi, real_seismology_api
from hq.validate.null_test import run_null_test
from hq.validate.sidecars import (
    BASELINE_JSON,
    GR_JSON,
    MAGNITUDE_JSON,
    NULL_TEST_JSON,
    SYNTHETIC_JSON,
    VALIDATION_JSON,
)

if TYPE_CHECKING:
    from hq.runs import RunContext

log = logging.getLogger(__name__)

STAGE = "validate"
H1 = "H1 Signal"
H2 = "H2 Seismology"
PICKS_TABLE = "picks.parquet"
PICKS_STALTA_TABLE = "picks_stalta.parquet"
STATIONS_TABLE = "stations.parquet"
CATALOG_TABLE = "catalog.parquet"
EVENTS_TABLE = "events.parquet"
SWEEP_TABLE = "sweep.parquet"
CATALOG_MAG_COLUMN = "mag"  # CatalogEvent.mag
CATALOG_MAG_TYPE_COLUMN = "magType"
EVENT_MAG_COLUMN = "magnitude_value"  # SeismicEvent.magnitude.value, flattened (docs/02 §2)
EVENT_MAG_TYPE_COLUMN = "magnitude_type"

# Which stage writes each input (docs/01 -> Pipeline), for the message when it is missing.
TABLE_WRITERS: dict[str, tuple[str, str]] = {
    PICKS_TABLE: ("pick", H1),
    PICKS_STALTA_TABLE: ("baseline", H1),
    STATIONS_TABLE: ("inventory", H1),
    CATALOG_TABLE: ("catalog", H2),
    EVENTS_TABLE: ("tier", H2),
    SWEEP_TABLE: ("tier", H2),  # FYI-H2-3: the sweep needs locate/match/tier, so tier writes it
    SYNTHETIC_JSON: ("locate", H2),
    MAGNITUDE_JSON: ("magnitude", H2),
}


@dataclass(frozen=True)
class ValidateOutcome:
    """Everything one run of the stage produced (the files are written as it goes)."""

    null_test: NullTest
    baseline: list[BaselineRow]
    gain: BaselineGain | None  # the exporter's verdict, computed here for the log and counts
    magnitude: MagCalibration | None
    gr: GRCurve | None
    public_magnitudes: int
    recovered_magnitudes: int
    validation: Validation | None


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


def read_optional_table(run_dir: Path, name: str, model: type[BaseModel]) -> pd.DataFrame | None:
    """``read_model_table`` for an input the stage can do without: ``None`` (logged, naming
    the writer) when the file is not there."""
    if not (run_dir / name).is_file():
        stage, owner = TABLE_WRITERS[name]
        log.warning(
            "validate: %s not found in %s (the %r stage, owner %s, has not written it)",
            name,
            run_dir,
            stage,
            owner,
        )
        return None
    return read_model_table(run_dir, name, model)


def read_synthetic(run_dir: Path) -> SyntheticTest | None:
    """H2's ``synthetic.json`` when the locate stage wrote it, else None (logged, naming H2)."""
    synthetic = sidecars.SYNTHETIC.read(run_dir, ValidateError)
    if synthetic is None:
        stage, owner = TABLE_WRITERS[SYNTHETIC_JSON]
        log.warning(
            "validate: %s not found in %s (the %r stage, owner %s, has not written it); "
            "Validation.synthetic is required, so %s is not written this time and the sidecars "
            "%s, %s and %s carry the results",
            SYNTHETIC_JSON,
            run_dir,
            stage,
            owner,
            VALIDATION_JSON,
            NULL_TEST_JSON,
            BASELINE_JSON,
            GR_JSON,
        )
    return synthetic


def read_magnitude(run_dir: Path) -> MagCalibration | None:
    """H2's ``magnitude.json`` when the magnitude stage wrote it, else None (logged)."""
    calibration = sidecars.MAGNITUDE.read(run_dir, ValidateError)
    if calibration is None:
        stage, owner = TABLE_WRITERS[MAGNITUDE_JSON]
        log.info(
            "validate: no %s in %s (the %r stage, owner %s, has not written it); "
            "Validation.magnitude is null",
            MAGNITUDE_JSON,
            run_dir,
            stage,
            owner,
        )
    return calibration


def read_sweep(run_dir: Path) -> list[SweepPoint]:
    """H2's ``sweep.parquet`` when the tier stage wrote it (FYI-H2-3), else an empty list."""
    path = run_dir / SWEEP_TABLE
    if not path.is_file():
        log.warning(
            "validate: no %s in %s (written by H2's tier stage, LOC-06); Validation.sweep stays "
            "empty and the sweep plot has nothing to show",
            SWEEP_TABLE,
            run_dir,
        )
        return []
    return read_models(path, SweepPoint)


def write_json_model(path: Path, model: BaseModel) -> None:
    """One model as pretty JSON, atomically (kept for VAL-02 callers; ``sidecars`` is the
    typed way)."""
    write_text_atomic(path, model.model_dump_json(indent=2) + "\n")


def _magnitude_types(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(str(v) for v in frame[column].dropna().unique())


def magnitudes(catalog: pd.DataFrame, events: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """The public regional catalog's magnitudes (``CatalogEvent.mag`` where not null) and the
    candidate events' (``SeismicEvent.magnitude.value`` where not null); types are logged,
    never compared."""
    public = (
        as_magnitudes(catalog[CATALOG_MAG_COLUMN].to_numpy(dtype=np.float64))
        if CATALOG_MAG_COLUMN in catalog.columns
        else np.array([], dtype=np.float64)
    )
    if events is None or EVENT_MAG_COLUMN not in events.columns:
        recovered = np.array([], dtype=np.float64)
    else:
        recovered = as_magnitudes(events[EVENT_MAG_COLUMN].to_numpy(dtype=np.float64))
    log.info(
        "validate: %d of %d public events carry a magnitude (types %s); %d of %d candidate "
        "events do (types %s)",
        len(public),
        len(catalog),
        _magnitude_types(catalog, CATALOG_MAG_TYPE_COLUMN),
        len(recovered),
        0 if events is None else len(events),
        [] if events is None else _magnitude_types(events, EVENT_MAG_TYPE_COLUMN),
    )
    return public, recovered


def assemble_validation(
    null_test: NullTest,
    synthetic: SyntheticTest,
    sweep: list[SweepPoint],
    baseline: Sequence[BaselineRow] = (),
    gr: GRCurve | None = None,
    magnitude: MagCalibration | None = None,
) -> Validation:
    """The ``Validation`` from the stage's results and H2's inputs."""
    return Validation(
        baseline=list(baseline),
        sweep=sweep,
        nullTest=null_test,
        gr=gr,
        magnitude=magnitude,
        synthetic=synthetic,
    )


def validate_run(ctx: "RunContext", api: SeismologyApi | None = None) -> ValidateOutcome:
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

    # VAL-02: chance associations.
    null_test = run_null_test(
        picks,
        stations,
        catalog,
        api,
        seismology_cfg,
        ctx.config.run,
        cfg.nullTest,
        cfg.pOnlyAssociator,
    )
    sidecars.NULL_TEST.write(ctx.run_dir, null_test)

    # VAL-01: the baseline table, when H1's STA/LTA picks exist.
    picks_stalta = read_optional_table(ctx.run_dir, PICKS_STALTA_TABLE, Pick)
    baseline: list[BaselineRow] = []
    if picks_stalta is None:
        log.warning(
            "validate: the baseline comparison needs %s (the 'baseline' stage, owner %s); "
            "Validation.baseline stays empty and no neural-advantage gain is claimed",
            PICKS_STALTA_TABLE,
            H1,
        )
    else:
        baseline = run_baseline(
            picks,
            picks_stalta,
            stations,
            catalog,
            api,
            seismology_cfg,
            ctx.config.run,
            cfg.baseline,
            cfg.pOnlyAssociator,
        )
    sidecars.BASELINE.write(ctx.run_dir, baseline)
    gain = baseline_gain(baseline, cfg.baseline)

    # VAL-01: the Gutenberg-Richter curve, when magnitudes exist and H2's calibration passes.
    magnitude = read_magnitude(ctx.run_dir)
    events = read_optional_table(ctx.run_dir, EVENTS_TABLE, SeismicEvent)
    public_mags, recovered_mags = magnitudes(catalog, events)
    gr: GRCurve | None = None
    if len(public_mags) + len(recovered_mags) == 0:
        log.warning(
            "validate: no magnitude in %s or %s (H2's magnitude stage fills them); Validation.gr "
            "is null",
            CATALOG_TABLE,
            EVENTS_TABLE,
        )
    elif gr_allowed(magnitude, cfg.gr):
        gr = gr_curve(public_mags, recovered_mags, cfg.gr)
    gr_path = sidecars.GR.path(ctx.run_dir)
    if gr is not None:
        sidecars.GR.write(ctx.run_dir, gr)
    elif gr_path.is_file():
        gr_path.unlink()
        log.warning("validate: removed stale %s from an earlier rerun", gr_path)

    synthetic = read_synthetic(ctx.run_dir)
    validation: Validation | None = None
    if synthetic is not None:
        validation = assemble_validation(
            null_test, synthetic, read_sweep(ctx.run_dir), baseline, gr, magnitude
        )
        sidecars.VALIDATION.write(ctx.run_dir, validation)
        log.info(
            "validate: wrote %s (nullTest, synthetic, %d sweep points, %d baseline rows, gr %s, "
            "magnitude %s)",
            ctx.path(VALIDATION_JSON),
            len(validation.sweep),
            len(validation.baseline),
            "set" if gr is not None else "null",
            "set" if magnitude is not None else "null",
        )
    return ValidateOutcome(
        null_test=null_test,
        baseline=baseline,
        gain=gain,
        magnitude=magnitude,
        gr=gr,
        public_magnitudes=len(public_mags),
        recovered_magnitudes=len(recovered_mags),
        validation=validation,
    )


def run(ctx: "RunContext") -> None:
    """Stage entry: the null test, the baseline table, the G-R curve, their sidecars and, when
    H2's synthetic test exists, ``validation.json``; counts and runtime go to the run record."""
    started = time.perf_counter()
    out = validate_run(ctx)
    null_test, validation = out.null_test, out.validation
    counts = {
        "nullShuffles": null_test.nShuffles,
        # Sums over reruns (counts must be ints); the means are in null_test.json.
        "nullChanceEvents": round(null_test.meanChanceEvents * null_test.nShuffles),
        "nullChanceStrict": round(null_test.meanChanceStrict * null_test.nShuffles),
        "sweepPoints": len(validation.sweep) if validation is not None else 0,
        "baselineRows": len(out.baseline),
        "baselineGain": int(out.gain is not None),
        "publicMagnitudes": out.public_magnitudes,
        "recoveredMagnitudes": out.recovered_magnitudes,
        "grBins": len(out.gr.magBins) if out.gr is not None else 0,
        "hasMagnitude": int(out.magnitude is not None),
        "validationJson": int(validation is not None),
    }
    ctx.record(STAGE, runtime_s=time.perf_counter() - started, counts=counts)


__all__ = [
    "BASELINE_JSON",
    "GR_JSON",
    "NULL_TEST_JSON",
    "STAGE",
    "VALIDATION_JSON",
    "LaneSeismologyApi",
    "SeismologyApi",
    "ValidateError",
    "ValidateOutcome",
    "assemble_validation",
    "baseline_gain",
    "gr_curve",
    "read_magnitude",
    "read_model_table",
    "read_sweep",
    "read_synthetic",
    "real_seismology_api",
    "run",
    "run_baseline",
    "run_null_test",
    "validate_run",
]

"""Stage ``validate`` (H4, tickets VAL-02 and VAL-01): validation reruns -> ``validation.json``.

``run(ctx)`` reads the run's picks, stations and public catalog, reruns H2's pipeline functions
through ``hq.validate.lanes`` and writes (``hq.validate.sidecars``):

- ``null_test.json``: the ``NullTest`` (VAL-02), always, so the number is never lost;
- ``baseline.json``: the ``BaselineRow`` table (VAL-01), always; empty when H1's
  ``picks_stalta.parquet`` is not there (logged, naming H1);
- ``gr.json``: the ``GRCurve`` (VAL-01) when magnitudes exist and H2's ``magnitude.json`` does
  not trip the docs/03 magnitude kill switch; a stale one from an earlier rerun is removed;
- ``validation_notes.json``: ``hq.validate.notes.ValidationNotes``, always: what the reruns
  could and could not do next to the numbers (the run's bars they tiered against, the Tier A
  rules H2 applied, no station statics, the p_only associator overrides) and the G-R curve's
  magnitude type, exclusions and gate;
- ``validation.json``: the ``Validation`` with all of the above plus ``sweep`` from H2's
  ``sweep.parquet``, ``magnitude`` from H2's ``magnitude.json`` and ``synthetic`` from H2's
  ``synthetic.json``.

The reruns call H2 as ``locate(assoc, picks, stations, cfg, run, cache_dir=ctx.cache_dir,
run_id=ctx.run_id)`` (bound in ``real_seismology_api``, REQ-H2-8) and ``assign_tiers(events,
matches, cfg, thresholds=<the run's ProcessingRun.tiering>, arrivals=located.arrivals,
stations=<the stations table>)`` (REQ-H2-9): the bars come from the run's own tier stage and
are never invented, so the stage fails naming H2's tier stage when ``run.json`` has none.

``Validation.synthetic`` is required by the contract and only H2's locate stage produces it, so
when ``synthetic.json`` is absent the stage keeps ``hq run`` going: it writes the sidecars, skips
``validation.json`` with a warning naming H2, and records ``validationJson: 0``. The exporter
then assembles the ``Validation`` from the sidecars once ``synthetic.json`` appears.

The G-R public curve is drawn on one magnitude scale (REQ-H2-13): ``CatalogEvent.mag`` of the
events whose ``magType`` equals the calibration type H2 recorded
(``ProcessingRun.matching["magnitude"]["calibrationMagType"]``), or ``validate.yaml``
``gr.publicMagType`` when that record is absent; with public magnitudes and neither, the stage
fails rather than mix scales. Public magnitudes of other types are counted per type in the
notes. The kill-switch gate is H2's recorded ``gate.maxLooMae`` when present (a differing
``validate.yaml`` value is a warning), and ``looMae`` is written to the notes next to H2's
``nullModelMae`` (FYI-H2-7) with a warning when the calibration shows no skill.

``AnalysisSummary.baseline`` is derived by the exporter from ``Validation.baseline`` with
``hq.validate.baseline.baseline_gain``; this stage logs the same verdict so the run log says
whether the claim holds.

Every knob is in ``configs/showcase/validate.yaml`` (``hq.config.validate.ValidateConfig``).
H2's functions are imported lazily, so this package loads before they are merged and a missing
one fails naming H2.
"""

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    ProcessingRun,
    SeismicEvent,
    Station,
    SweepPoint,
    SyntheticTest,
    Validation,
)
from pydantic import BaseModel

from hq.config.validate import GRConfig, ValidateConfig
from hq.runs import write_text_atomic
from hq.validate import sidecars
from hq.validate.baseline import baseline_gain, baseline_reruns, run_baseline
from hq.validate.errors import ValidateError
from hq.validate.gr import as_magnitudes, gr_allowed, gr_curve
from hq.validate.lanes import LaneSeismologyApi, SeismologyApi, real_seismology_api
from hq.validate.notes import GRNotes, RerunNotes, ValidationNotes, rerun_notes
from hq.validate.null_test import (
    null_shuffles,
    profile_overrides,
    require_thresholds,
    run_null_test,
    summarize,
)
from hq.validate.sidecars import (
    BASELINE_JSON,
    GR_JSON,
    MAGNITUDE_JSON,
    NOTES_JSON,
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
NO_MAG_TYPE = "null"  # key for public magnitudes without a magType in the notes
# Where H2's magnitude stage records its params (FYI-H2-7) and the keys the stage reads there.
MAGNITUDE_RECORD_KEY = "magnitude"  # ProcessingRun.matching["magnitude"]
CALIBRATION_TYPE_KEY = "calibrationMagType"
CALIBRATION_TYPE_RECORD = f"ProcessingRun.matching.{MAGNITUDE_RECORD_KEY}.{CALIBRATION_TYPE_KEY}"
PUBLIC_MAG_TYPE_KNOB = "validate.yaml gr.publicMagType"
GATE_RECORD = f"ProcessingRun.matching.{MAGNITUDE_RECORD_KEY}.gate.maxLooMae"
GATE_KNOB = "validate.yaml gr.maxLooMae"
NULL_MODEL_RECORD = f"ProcessingRun.matching.{MAGNITUDE_RECORD_KEY}.leaveOneEventOut.nullModelMae"
ONE_SCALE_NOTE = (
    "publicCum counts the public regional catalog's magnitudes of one type only; magnitudes are "
    "never compared across types (REQ-H2-13). GRCurve has no label field: the type is recorded "
    "here."
)
CENSORING_NOTE = (
    "Candidate magnitudes below H2's calibrated range are extrapolated, and near the detection "
    "limit they rest on the stations whose amplitude cleared the noise (biased upward, fewer "
    "stations), which flattens the low-magnitude end of the curve and the b-value (REQ-H2-13; "
    "ProcessingRun.matching.magnitude.magnitudes has the counts)."
)

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
    public_magnitudes: int  # public magnitudes of the curve's type (publicCum's sample)
    public_magnitudes_excluded: int  # public magnitudes of other types, left out (REQ-H2-13)
    recovered_magnitudes: int
    validation: Validation | None
    notes: ValidationNotes


@dataclass(frozen=True)
class MagnitudeSets:
    """The two magnitude samples of the G-R curve and what was left out of the public one."""

    public: np.ndarray  # CatalogEvent.mag of the events of ``mag_type``
    recovered: np.ndarray  # SeismicEvent.magnitude.value of every candidate event with one
    mag_type: str | None  # the public sample's magType; None when the catalog has no magnitude
    public_excluded_by_type: dict[str, int]  # other magTypes -> public magnitudes left out
    recovered_types: list[str]  # SeismicEvent.magnitude.type values seen (logged, never compared)


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


# --- the run record: bars, calibration type, gate ------------------------------------------------


def run_thresholds(run: ProcessingRun) -> Mapping[str, Any]:
    """The run's ``ProcessingRun.tiering`` with the bars H2's tier stage derived (REQ-H2-9), as
    ``assign_tiers(thresholds=...)`` takes it; a loud error naming H2's tier stage without."""
    return require_thresholds(run.tiering, f"run {run.id}: the validation reruns")


def magnitude_record(run: ProcessingRun) -> Mapping[str, Any] | None:
    """H2's magnitude-stage params (``ProcessingRun.matching["magnitude"]``, FYI-H2-7), or None
    when the stage has not recorded any."""
    record = run.matching.get(MAGNITUDE_RECORD_KEY)
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise ValidateError(
            f"ProcessingRun.matching[{MAGNITUDE_RECORD_KEY!r}] is a {type(record).__name__}, not "
            f"the dict H2's magnitude stage records (owner: {H2})"
        )
    return record


def _nested(record: Mapping[str, Any] | None, *keys: str) -> Any:
    """``record[k1][k2]...`` or None when any level is absent or not a mapping."""
    value: Any = record
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValidateError(f"{what} is {value!r}, not a number (owner: {H2})")
    return float(value)


def public_mag_type(
    record: Mapping[str, Any] | None, cfg: GRConfig, public_magnitudes: int
) -> tuple[str | None, str | None]:
    """The one ``CatalogEvent.magType`` the public curve is drawn from and where it came from:
    H2's recorded ``calibrationMagType``, else ``gr.publicMagType``. With public magnitudes and
    neither, a loud error: the curve never mixes scales (REQ-H2-13). Without public magnitudes
    there is nothing to select and both are None."""
    calibrated = _nested(record, CALIBRATION_TYPE_KEY)
    if calibrated is not None:
        if not isinstance(calibrated, str) or not calibrated:
            raise ValidateError(
                f"{CALIBRATION_TYPE_RECORD} is {calibrated!r}, not a magnitude type (owner: {H2})"
            )
        if cfg.publicMagType is not None and cfg.publicMagType != calibrated:
            log.warning(
                "validate: %s %r is not the run's calibration type %r (%s); the calibration "
                "type is used, as the candidates' magnitudes are on that scale",
                PUBLIC_MAG_TYPE_KNOB,
                cfg.publicMagType,
                calibrated,
                CALIBRATION_TYPE_RECORD,
            )
        return calibrated, CALIBRATION_TYPE_RECORD
    if cfg.publicMagType is not None:
        log.info(
            "validate: no %s in run.json (H2's magnitude stage has not recorded one); the public "
            "G-R curve uses %s %r",
            CALIBRATION_TYPE_RECORD,
            PUBLIC_MAG_TYPE_KNOB,
            cfg.publicMagType,
        )
        return cfg.publicMagType, PUBLIC_MAG_TYPE_KNOB
    if public_magnitudes > 0:
        raise ValidateError(
            f"{public_magnitudes} public magnitudes are present but neither "
            f"{CALIBRATION_TYPE_RECORD} (H2's magnitude stage, owner {H2}) nor "
            f"{PUBLIC_MAG_TYPE_KNOB} names the magnitude type the public G-R curve is drawn from; "
            "magnitudes are never compared across types (REQ-H2-13)"
        )
    return None, None


def effective_gr_config(record: Mapping[str, Any] | None, cfg: GRConfig) -> tuple[GRConfig, str]:
    """``cfg`` with ``maxLooMae`` replaced by the gate H2's magnitude stage applied
    (``gate.maxLooMae``) when recorded, so the two gates never disagree on a run (REQ-H2-13); a
    differing ``validate.yaml`` value is a warning. Returns the config and the gate's source."""
    gate = _nested(record, "gate", "maxLooMae")
    if gate is None:
        return cfg, GATE_KNOB
    value = _number(gate, GATE_RECORD)
    if value != cfg.maxLooMae:
        log.warning(
            "validate: %s (%g) differs from %s (%g); H2's recorded gate is applied. Keep the two "
            "equal (REQ-H2-13)",
            GATE_KNOB,
            cfg.maxLooMae,
            GATE_RECORD,
            value,
        )
    return cfg.model_copy(update={"maxLooMae": value}), GATE_RECORD


def null_model_mae(record: Mapping[str, Any] | None) -> float | None:
    """H2's ``leaveOneEventOut.nullModelMae`` (FYI-H2-7) when recorded."""
    value = _nested(record, "leaveOneEventOut", "nullModelMae")
    return None if value is None else _number(value, NULL_MODEL_RECORD)


# --- magnitudes ----------------------------------------------------------------------------------


def _magnitude_types(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(str(v) for v in frame[column].dropna().unique())


def public_magnitude_count(catalog: pd.DataFrame) -> int:
    """How many public regional catalog events carry a magnitude, of any type."""
    if CATALOG_MAG_COLUMN not in catalog.columns:
        return 0
    return len(as_magnitudes(catalog[CATALOG_MAG_COLUMN].to_numpy(dtype=np.float64)))


def magnitude_sets(
    catalog: pd.DataFrame, events: pd.DataFrame | None, mag_type: str | None
) -> MagnitudeSets:
    """The public regional catalog's magnitudes of ``mag_type`` (``CatalogEvent.mag`` where
    ``magType`` equals it and the value is not null; the others counted per type) and the
    candidate events' (``SeismicEvent.magnitude.value`` where not null). Types are logged,
    never compared; ``mag_type`` None (no public magnitude at all) selects nothing."""
    public = np.array([], dtype=np.float64)
    excluded: dict[str, int] = {}
    if CATALOG_MAG_COLUMN in catalog.columns and mag_type is not None:
        values = catalog[CATALOG_MAG_COLUMN].to_numpy(dtype=np.float64)
        has_mag = np.isfinite(values)
        if CATALOG_MAG_TYPE_COLUMN in catalog.columns:
            types = catalog[CATALOG_MAG_TYPE_COLUMN]
            keys = types.where(types.notna(), NO_MAG_TYPE).astype(str).to_numpy(dtype=object)
        else:
            keys = np.full(len(catalog), NO_MAG_TYPE, dtype=object)
        selected = has_mag & (keys == mag_type)
        public = values[selected]
        left_out = has_mag & ~selected
        for key in sorted({str(k) for k in keys[left_out]}):
            excluded[key] = int(np.sum(left_out & (keys == key)))
    if events is None or EVENT_MAG_COLUMN not in events.columns:
        recovered = np.array([], dtype=np.float64)
    else:
        recovered = as_magnitudes(events[EVENT_MAG_COLUMN].to_numpy(dtype=np.float64))
    recovered_types = [] if events is None else _magnitude_types(events, EVENT_MAG_TYPE_COLUMN)
    log.info(
        "validate: %d of %d public events carry a magnitude of type %r (publicCum's sample); "
        "%d public magnitudes of other types left out, per type %s; %d of %d candidate events "
        "carry a magnitude (types %s)",
        len(public),
        len(catalog),
        mag_type,
        sum(excluded.values()),
        excluded or "none",
        len(recovered),
        0 if events is None else len(events),
        recovered_types,
    )
    return MagnitudeSets(public, recovered, mag_type, excluded, recovered_types)


def gr_notes(
    sets: MagnitudeSets,
    mag_type_source: str | None,
    gr_cfg: GRConfig,
    gate_source: str,
    calibration: MagCalibration | None,
    null_mae: float | None,
) -> GRNotes:
    """The G-R provenance for ``validation_notes.json``: the public scale and its exclusions
    (REQ-H2-13), the gate applied, and ``looMae`` against H2's null model (FYI-H2-7), with a
    warning when the calibration is not better than predicting the mean."""
    loo = None if calibration is None else calibration.looMae
    skill = None if loo is None or null_mae is None else bool(loo < null_mae)
    notes = [ONE_SCALE_NOTE, CENSORING_NOTE]
    if null_mae is not None and loo is not None:
        notes.append(
            f"{NULL_MODEL_RECORD} is the MAE of predicting each calibration event as the mean "
            "catalog magnitude of the others; a looMae not below it shows no skill (FYI-H2-7)."
        )
        if not skill:
            log.warning(
                "validate: MagCalibration.looMae %.3f is not below H2's null-model MAE %.3f (%s): "
                "the calibration predicts no better than the mean catalog magnitude, so passing "
                "the gate (%g) shows no skill; do not present the magnitudes as calibrated",
                loo,
                null_mae,
                NULL_MODEL_RECORD,
                gr_cfg.maxLooMae,
            )
        else:
            log.info(
                "validate: MagCalibration.looMae %.3f is below H2's null-model MAE %.3f (%s)",
                loo,
                null_mae,
                NULL_MODEL_RECORD,
            )
    elif loo is not None:
        log.warning(
            "validate: no %s in run.json; looMae %.3f cannot be read against a null model "
            "(FYI-H2-7)",
            NULL_MODEL_RECORD,
            loo,
        )
    return GRNotes(
        magType=sets.mag_type,
        magTypeSource=mag_type_source,
        publicIncluded=len(sets.public),
        publicExcludedByType=dict(sets.public_excluded_by_type),
        recoveredMagTypes=list(sets.recovered_types),
        maxLooMae=gr_cfg.maxLooMae,
        maxLooMaeSource=gate_source,
        looMae=loo,
        nullModelMae=null_mae,
        skill=skill,
        notes=notes,
    )


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
    run_record = ctx.read_run()
    thresholds = run_thresholds(run_record)  # REQ-H2-9: the run's own bars, or a loud error
    thresholds_record = thresholds["thresholds"]
    if api is None:
        api = real_seismology_api(cache_dir=ctx.cache_dir, run_id=ctx.run_id)  # REQ-H2-8

    # VAL-02: chance associations.
    outcomes = null_shuffles(
        picks,
        stations,
        catalog,
        api,
        seismology_cfg,
        ctx.config.run,
        cfg.nullTest,
        cfg.pOnlyAssociator,
        thresholds=thresholds,
    )
    null_test = summarize(outcomes, cfg.nullTest)
    sidecars.NULL_TEST.write(ctx.run_dir, null_test)
    null_notes = rerun_notes(
        [o.tiering for o in outcomes],
        thresholds_record,
        profile_overrides((cfg.nullTest.profile,), cfg.pOnlyAssociator),
        "null test",
    )

    # VAL-01: the baseline table, when H1's STA/LTA picks exist.
    picks_stalta = read_optional_table(ctx.run_dir, PICKS_STALTA_TABLE, Pick)
    baseline: list[BaselineRow] = []
    baseline_notes: RerunNotes | None = None
    if picks_stalta is None:
        log.warning(
            "validate: the baseline comparison needs %s (the 'baseline' stage, owner %s); "
            "Validation.baseline stays empty and no neural-advantage gain is claimed",
            PICKS_STALTA_TABLE,
            H1,
        )
    else:
        reruns = baseline_reruns(
            picks,
            picks_stalta,
            stations,
            catalog,
            api,
            seismology_cfg,
            ctx.config.run,
            cfg.baseline,
            cfg.pOnlyAssociator,
            thresholds=thresholds,
        )
        baseline = [r.row for r in reruns]
        baseline_notes = rerun_notes(
            [r.tiering for r in reruns],
            thresholds_record,
            profile_overrides(cfg.baseline.profiles, cfg.pOnlyAssociator),
            "baseline",
        )
    sidecars.BASELINE.write(ctx.run_dir, baseline)
    gain = baseline_gain(baseline, cfg.baseline)

    # VAL-01: the Gutenberg-Richter curve, when magnitudes exist and H2's calibration passes.
    magnitude = read_magnitude(ctx.run_dir)
    mag_record = magnitude_record(run_record)
    events = read_optional_table(ctx.run_dir, EVENTS_TABLE, SeismicEvent)
    mag_type, mag_type_source = public_mag_type(mag_record, cfg.gr, public_magnitude_count(catalog))
    sets = magnitude_sets(catalog, events, mag_type)
    gr_cfg, gate_source = effective_gr_config(mag_record, cfg.gr)
    gr: GRCurve | None = None
    notes_gr: GRNotes | None = None
    n_mags = len(sets.public) + len(sets.recovered)
    if n_mags == 0 and magnitude is None and mag_record is None:
        log.warning(
            "validate: no magnitude in %s or %s (H2's magnitude stage fills them); Validation.gr "
            "is null",
            CATALOG_TABLE,
            EVENTS_TABLE,
        )
    else:
        notes_gr = gr_notes(
            sets, mag_type_source, gr_cfg, gate_source, magnitude, null_model_mae(mag_record)
        )
        if n_mags == 0:
            log.warning(
                "validate: no magnitude of type %r in %s and none in %s; Validation.gr is null",
                mag_type,
                CATALOG_TABLE,
                EVENTS_TABLE,
            )
        elif gr_allowed(magnitude, gr_cfg):
            gr = gr_curve(sets.public, sets.recovered, gr_cfg)
    gr_path = sidecars.GR.path(ctx.run_dir)
    if gr is not None:
        sidecars.GR.write(ctx.run_dir, gr)
    elif gr_path.is_file():
        gr_path.unlink()
        log.warning("validate: removed stale %s from an earlier rerun", gr_path)

    notes = ValidationNotes(nullTest=null_notes, baseline=baseline_notes, gr=notes_gr)
    sidecars.NOTES.write(ctx.run_dir, notes)
    log.info(
        "validate: wrote %s (null test: %d of %d reruns tiered; baseline: %s; gr type %s)",
        ctx.path(NOTES_JSON),
        null_notes.rerunsTiered,
        null_notes.reruns,
        "not run"
        if baseline_notes is None
        else f"{baseline_notes.rerunsTiered} of {baseline_notes.reruns} reruns tiered",
        "none" if notes_gr is None else notes_gr.magType,
    )

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
        public_magnitudes=len(sets.public),
        public_magnitudes_excluded=sum(sets.public_excluded_by_type.values()),
        recovered_magnitudes=len(sets.recovered),
        validation=validation,
        notes=notes,
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
        "nullRerunsTiered": out.notes.nullTest.rerunsTiered,
        "sweepPoints": len(validation.sweep) if validation is not None else 0,
        "baselineRows": len(out.baseline),
        "baselineGain": int(out.gain is not None),
        "publicMagnitudes": out.public_magnitudes,
        "publicMagnitudesExcluded": out.public_magnitudes_excluded,
        "recoveredMagnitudes": out.recovered_magnitudes,
        "grBins": len(out.gr.magBins) if out.gr is not None else 0,
        "hasMagnitude": int(out.magnitude is not None),
        "validationJson": int(validation is not None),
    }
    ctx.record(STAGE, runtime_s=time.perf_counter() - started, counts=counts)


__all__ = [
    "BASELINE_JSON",
    "GR_JSON",
    "NOTES_JSON",
    "NULL_TEST_JSON",
    "STAGE",
    "VALIDATION_JSON",
    "LaneSeismologyApi",
    "MagnitudeSets",
    "SeismologyApi",
    "ValidateError",
    "ValidateOutcome",
    "assemble_validation",
    "baseline_gain",
    "effective_gr_config",
    "gr_curve",
    "gr_notes",
    "magnitude_record",
    "magnitude_sets",
    "null_model_mae",
    "public_mag_type",
    "read_magnitude",
    "read_model_table",
    "read_sweep",
    "read_synthetic",
    "real_seismology_api",
    "run",
    "run_baseline",
    "run_null_test",
    "run_thresholds",
    "validate_run",
]

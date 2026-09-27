"""Run mode ``live``: station terms and tier bars borrowed from a showcase calibration run.

A live window (the API worker's last 2 hours) holds about 0-3 public regional catalog events:
too few for ``statics.mode`` referenceEvents terms (``minReferenceEvents`` per station-phase) or
for tier bars (``tiering.minMatched``). When ``run.json``'s mode is ``live``, stage locate applies
the calibration run's ``statics.parquet`` to every event (no terms are fit on the window, so a
second locate pass of the runner's reference-statics plan repeats the first) and stage tier
applies the calibration run's ``ProcessingRun.tiering`` thresholds. The calibration run is
``seismology.yaml``'s ``live.calibrationRun``, read from ``<data>/showcase/runs/<id>`` next to
the live run's own ``<data>/live/runs/<runId>``. Every other mode never reads it, so showcase
runs are unchanged.

A missing run directory, table or record fails loudly (``CalibrationError``); so does a
calibration run located with another ``locator.method``, whose terms are relative to other
travel times.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from hq_contracts.io import read_table

if TYPE_CHECKING:
    from hq.runs import RunContext

LIVE_MODE = "live"
CALIBRATION_MODE_DIR = "showcase"
RUNS_DIRNAME = "runs"
STATICS_TABLE = "statics.parquet"
RUN_JSON = "run.json"


class CalibrationError(ValueError):
    """The live run's calibration run is missing or unusable."""


@dataclass(frozen=True, eq=False)
class Calibration:
    """What a live run borrows: the statics table and the tiering record of one showcase run."""

    run_id: str
    run_dir: Path
    statics: pd.DataFrame  # stationId, phase, staticS, nEvents (StationStatic rows)
    statics_sha256: str
    tiering: dict[str, Any]  # the calibration run's ProcessingRun.tiering (with "thresholds")
    locator_method: str
    terms_record: list[dict[str, Any]]  # its ProcessingRun.locator statics.terms (rawS, madS)

    def terms(self) -> pd.DataFrame:
        """``hq.locate.statics.TERM_COLUMNS``: the statics table, with ``rawS`` / ``madS`` from
        the calibration run's record where it has them (else ``staticS`` / NaN)."""
        s = self.statics
        rec = {(str(r["stationId"]), str(r["phase"])): r for r in self.terms_record}
        keys = list(zip(s["stationId"].astype(str), s["phase"].astype(str), strict=True))

        def pick(key: tuple[str, str], name: str, default: float) -> float:
            value = rec.get(key, {}).get(name)
            return default if value is None else float(value)

        return pd.DataFrame({
            "stationId": s["stationId"].astype(str).to_numpy(dtype=object),
            "phase": s["phase"].astype(str).to_numpy(dtype=object),
            "staticS": s["staticS"].to_numpy(dtype=np.float64),
            "rawS": [pick(k, "rawS", float(v)) for k, v in zip(keys, s["staticS"], strict=True)],
            "nEvents": s["nEvents"].to_numpy(dtype=np.int64),
            "madS": [pick(k, "madS", math.nan) for k in keys],
        })

    def to_record(self) -> dict[str, Any]:
        s = self.statics
        return {
            "runId": self.run_id,
            # Relative to the data dir: no laptop path goes into a run record or bundle.
            "runDir": f"{CALIBRATION_MODE_DIR}/{RUNS_DIRNAME}/{self.run_id}",
            "staticsTable": STATICS_TABLE,
            "staticsSha256": self.statics_sha256,
            "stationPhases": len(s),
            "nonZero": int((s["staticS"] != 0.0).sum()),
            "thresholdsNMatched": self.tiering.get("thresholds", {}).get("nMatched"),
            "locatorMethod": self.locator_method,
        }


def run_mode(ctx: "RunContext") -> str | None:
    """``run.json``'s mode (``DataMode``)."""
    return getattr(ctx.read_run(), "mode", None)


def calibration_dir(ctx: "RunContext", run_id: str) -> Path:
    """``<data>/showcase/runs/<run_id>`` for a run dir ``<data>/<mode>/runs/<runId>``."""
    return ctx.run_dir.parent.parent.parent / CALIBRATION_MODE_DIR / RUNS_DIRNAME / run_id


def load_calibration(ctx: "RunContext") -> Calibration | None:
    """None unless the run's mode is live; then the calibration run, or ``CalibrationError``."""
    if run_mode(ctx) != LIVE_MODE:
        return None
    cfg = ctx.config.seismology
    run_id = cfg.live.calibrationRun
    run_dir = calibration_dir(ctx, run_id)
    statics_path, run_path = run_dir / STATICS_TABLE, run_dir / RUN_JSON
    missing = [p.name for p in (statics_path, run_path) if not p.is_file()]
    if missing:
        raise CalibrationError(
            f"calibration run {run_id} (seismology.yaml live.calibrationRun) lacks {missing} at "
            f"{run_dir}: copy the showcase run there (live runs read <data>/showcase/runs/<id>)")
    statics = read_table(statics_path)
    if statics.attrs.get("model") != "StationStatic":
        raise CalibrationError(f"{statics_path} holds {statics.attrs.get('model')!r} rows, not "
                               "StationStatic")
    values = statics["staticS"].to_numpy(dtype=np.float64)
    keys = list(zip(statics["stationId"].astype(str), statics["phase"].astype(str), strict=True))
    if not np.isfinite(values).all() or len(set(keys)) != len(keys):
        raise CalibrationError(f"{statics_path}: non-finite staticS or a repeated station-phase")
    record = json.loads(run_path.read_text(encoding="utf-8"))
    if record.get("mode") != CALIBRATION_MODE_DIR or record.get("id") != run_id:
        raise CalibrationError(f"{run_path} is run {record.get('id')!r} in mode "
                               f"{record.get('mode')!r}, not showcase run {run_id}")
    tiering = record.get("tiering") or {}
    if not isinstance(tiering.get("thresholds"), dict):
        raise CalibrationError(f"{run_path} has no tiering thresholds: run stage tier there first")
    locator = record.get("locator") or {}
    method = locator.get("method")
    if method != cfg.locator.method:
        raise CalibrationError(
            f"calibration run {run_id} was located with locator.method {method!r}, this config "
            f"uses {cfg.locator.method!r}: its terms are relative to other travel times")
    return Calibration(
        run_id=run_id, run_dir=run_dir, statics=statics.reset_index(drop=True),
        statics_sha256=hashlib.sha256(statics_path.read_bytes()).hexdigest(), tiering=tiering,
        locator_method=str(method),
        terms_record=list((locator.get("statics") or {}).get("terms") or []),
    )

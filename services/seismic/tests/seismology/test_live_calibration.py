"""Run mode live (``hq.locate.calibration``): loading the showcase calibration run.

The stages that use it are tested with their own: tier in ``test_tier.py``
(``test_live_stage_applies_the_calibration_runs_bars``), locate in ``test_statics.py``
(``test_live_stage_borrows_the_calibration_terms``).
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hq_contracts.io import write_table

from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig
from hq.locate.calibration import CalibrationError, load_calibration
from hq.locate.statics import TERM_COLUMNS

pytestmark = pytest.mark.smoke

THRESHOLDS = {"quantiles": {"A": 0.25, "B": 0.0}, "nMatched": 43}


def calibration_run(data: Path, cfg: SeismologyConfig, **record: Any) -> Path:
    run_id = cfg.live.calibrationRun
    run_dir = data / "showcase" / "runs" / run_id
    run_dir.mkdir(parents=True)
    doc = {"id": run_id, "mode": "showcase", "tiering": {"thresholds": THRESHOLDS},
           "locator": {"method": cfg.locator.method, "statics": {"terms": [
               {"stationId": "UU.A", "phase": "P", "staticS": 0.1, "rawS": 0.1, "nEvents": 12,
                "madS": 0.02}]}}, **record}
    (run_dir / "run.json").write_text(json.dumps(doc), encoding="utf-8")
    write_table(pd.DataFrame({"stationId": ["UU.A", "UU.A", "UU.B"], "phase": ["P", "S", "P"],
                              "staticS": [0.1, -0.4, 0.0], "nEvents": [12, 11, 2]}),
                run_dir / "statics.parquet", "StationStatic")
    return run_dir


def ctx_in(make_ctx: Any, run: RunSection, tmp_path: Path, mode: str) -> Any:
    run_dir = tmp_path / "data" / mode / "runs" / "window"
    run_dir.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(make_ctx(run), run_dir=run_dir, mode=mode)


def test_showcase_config_names_the_run_of_record(seismology_config: SeismologyConfig) -> None:
    assert seismology_config.live.calibrationRun == "20260926-0210-a04c611"
    raw = seismology_config.model_dump(mode="json")
    with pytest.raises(ValueError, match="calibrationRun"):
        SeismologyConfig.model_validate({**raw, "live": {"calibrationRun": "latest"}})


def test_showcase_mode_never_reads_a_calibration_run(
    make_ctx: Any, run_section: RunSection, tmp_path: Path
) -> None:
    # No calibration run exists anywhere: showcase (and every non-live mode) doesn't look.
    for mode in ("showcase", "mock", "snapshot"):
        assert load_calibration(ctx_in(make_ctx, run_section, tmp_path, mode)) is None


def test_live_mode_loads_terms_and_bars(
    make_ctx: Any, run_section: RunSection, seismology_config: SeismologyConfig, tmp_path: Path
) -> None:
    run_dir = calibration_run(tmp_path / "data", seismology_config)
    cal = load_calibration(ctx_in(make_ctx, run_section, tmp_path, "live"))
    assert cal is not None and cal.run_dir == run_dir
    assert cal.tiering["thresholds"] == THRESHOLDS
    terms = cal.terms()
    assert list(terms.columns) == TERM_COLUMNS
    assert terms["staticS"].tolist() == [0.1, -0.4, 0.0]
    assert terms["madS"].iloc[0] == 0.02 and np.isnan(terms["madS"].iloc[1])  # from run.json
    record = cal.to_record()
    assert record["runId"] == seismology_config.live.calibrationRun
    assert record["nonZero"] == 2 and len(record["staticsSha256"]) == 64
    assert record["runDir"] == f"showcase/runs/{record['runId']}"  # no laptop path


def test_live_mode_without_a_calibration_run_fails_loudly(
    make_ctx: Any, run_section: RunSection, seismology_config: SeismologyConfig, tmp_path: Path
) -> None:
    live = ctx_in(make_ctx, run_section, tmp_path, "live")
    with pytest.raises(CalibrationError, match="copy the showcase run there"):
        load_calibration(live)
    run_dir = calibration_run(tmp_path / "data", seismology_config)
    (run_dir / "statics.parquet").unlink()
    with pytest.raises(CalibrationError, match=r"lacks \['statics.parquet'\]"):
        load_calibration(live)


@pytest.mark.parametrize(("record", "match"), [
    ({"tiering": {}}, "no tiering thresholds"),
    ({"mode": "live"}, "not showcase run"),
    ({"locator": {"method": "grid3d"}}, "locator.method"),
])
def test_live_mode_refuses_an_unusable_calibration_run(
    make_ctx: Any, run_section: RunSection, seismology_config: SeismologyConfig, tmp_path: Path,
    record: dict[str, Any], match: str,
) -> None:
    calibration_run(tmp_path / "data", seismology_config, **record)
    with pytest.raises(CalibrationError, match=match):
        load_calibration(ctx_in(make_ctx, run_section, tmp_path, "live"))

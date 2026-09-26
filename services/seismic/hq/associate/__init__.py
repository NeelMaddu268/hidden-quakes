"""Association: picks + stations -> candidate events (docs/02 §5, LOC-03).

``associate`` runs PyOcto 0.2.0 on the same 1D layer model as the locator and returns the
``assoc_events`` / ``assoc_picks`` frames. How: ``hq.associate.core`` (steps), ``tables`` (PyOcto's
travel-time tables), ``frame`` (elevM/ENU <-> PyOcto's km frame), ``sweep``, and the stage in
``hq.associate.run``. The stage function is also this package's ``run``, because the stage
registry (docs/01, H4's ``hq.runs.STAGES``) resolves stage ``associate`` as ``hq.associate.run``
the attribute; ``importlib.import_module("hq.associate.run").run`` is the same function.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from hq.associate.core import associate_setup, prepared, record
from hq.associate.result import AssocResult
from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig
from hq.locate.velocity import LayerModel


def associate_detailed(
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    cfg: SeismologyConfig,
    run: RunSection,
    *,
    cache_dir: Path | None = None,
    model: LayerModel | None = None,
) -> tuple[AssocResult, dict[str, int], dict[str, Any]]:
    """``associate`` plus its counts and the ``ProcessingRun.associator`` record."""
    with prepared(stations, cfg, run, model=model, cache_dir=cache_dir) as setup:
        result, counts = associate_setup(picks, setup, cfg.associator)
        return result, counts, record(cfg.associator, setup)


def associate(
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    cfg: SeismologyConfig,
    run: RunSection,
    *,
    cache_dir: Path | None = None,
) -> AssocResult:
    """Associate ``picks`` (docs/02 ``Pick`` rows, any picker) at the stations used in the run.

    ``stations``: ``stations.parquet`` rows; the ``usedInRun`` ones take part, at ``sensorElevM``.
    Every pick must come from one of them. ``cache_dir`` (keyword only; the docs/02 call leaves
    it out) caches PyOcto's tables under ``<cache_dir>/ttgrids/pyocto/``; without it they are
    built in a temporary directory.
    """
    result, _, _ = associate_detailed(picks, stations, cfg, run, cache_dir=cache_dir)
    return result


# Last, so the package attribute ``run`` is the stage function, not the submodule.
from hq.associate.run import run

__all__ = ["AssocResult", "associate", "associate_detailed", "run"]

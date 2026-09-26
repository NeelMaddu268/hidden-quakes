"""Stage ``export`` (H4, ticket API-02): one run directory -> ``apps/web/public/data/<mode>/``.

``run(ctx)`` reads every table the pipeline wrote (``hq.export.tables``), derives what only the
exporter knows (``hq.export.summary``: reveal order, hero, catalog matches, ``AnalysisSummary``,
``SceneMeta``), cuts evidence snippets from the waveform cache through H1's reader and display
filter (``hq.export.evidence``, ``hq.export.waveforms``), writes one bundle per configured mode
atomically (``hq.export.bundle``) and re-validates it (``hq.export.check``). Every knob is in
``configs/showcase/export.yaml`` (``hq.config.export.ExportConfig``).

Lane dependencies are imported lazily, so this package loads before they are merged and a
missing one fails naming its owner: ``hq.ingest.cache.read_window`` and
``hq.preprocess.display_copy`` (H1), ``hq.export.features.load_features`` (FEAT-01).
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import hq
from hq.config.export import ExportConfig
from hq.export.bundle import ExportResult, export_bundle, load_features_lazily, replace_directory
from hq.export.check import BundleCheckError, check_bundle
from hq.export.errors import ExportError
from hq.export.tables import RunTables, load_run_tables
from hq.export.waveforms import LaneWaveformSource, WaveformSource, real_waveform_source

if TYPE_CHECKING:
    from hq.runs import RunContext

log = logging.getLogger(__name__)

STAGE = "export"


def checkout_root(start: Path | None = None) -> Path:
    """The git checkout holding the ``hq`` package: the first ancestor with a ``.git`` entry.

    In a linked worktree ``.git`` is a file; that worktree is still the right root, because the
    bundle is committed from the checkout whose code produced it (unlike ``data/``, which
    ``hq.cli`` shares across worktrees).
    """
    start = (start or Path(hq.__file__)).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise ExportError(
        f"no .git above {start}; set export.yaml outputDir to an absolute path to export from "
        "an installed copy"
    )


def output_root(cfg: ExportConfig) -> Path:
    """``<checkout root>/<outputDir>``, or ``outputDir`` itself when absolute."""
    out = Path(cfg.outputDir)
    return out if out.is_absolute() else checkout_root() / out


def run(ctx: "RunContext") -> None:
    """Stage entry: export every mode in ``export.yaml`` (each bundle is checked before it is
    swapped in) and record the first mode's counts plus ``modes``; every mode gets the same
    content, so the counts are not summed across modes."""
    started = time.perf_counter()
    cfg: ExportConfig = ctx.config.export
    tables = load_run_tables(ctx.run_dir)
    if tables.run.id != ctx.run_id:
        raise ExportError(
            f"{ctx.run_dir}/run.json has id {tables.run.id!r}, expected {ctx.run_id!r}"
        )
    source = real_waveform_source()
    root = output_root(cfg)
    counts: dict[str, int] = {}
    for mode in cfg.modes:
        result = export_bundle(
            tables, cfg, ctx.config.run, mode, source, cache_dir=ctx.cache_dir, out_dir=root / mode
        )
        if not counts:
            counts = dict(result.counts)
    counts["modes"] = len(cfg.modes)
    ctx.record(STAGE, runtime_s=time.perf_counter() - started, counts=counts)


__all__ = [
    "STAGE",
    "BundleCheckError",
    "ExportError",
    "ExportResult",
    "LaneWaveformSource",
    "RunTables",
    "WaveformSource",
    "check_bundle",
    "checkout_root",
    "export_bundle",
    "load_features_lazily",
    "load_run_tables",
    "output_root",
    "real_waveform_source",
    "replace_directory",
    "run",
]

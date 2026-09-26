"""``hq-api freeze-snapshot``: copy the last good live window into the snapshot bundle (API-05).

The web app's ``LiveProvider`` fails over to ``apps/web/public/data/snapshot/`` (``docs/02`` §6),
and the deployed static site only has the snapshot that is committed in git. The worker rewrites
``paths.snapshotDir`` after every good window on its own (``snapshot.enabled``); this command
does the same thing on demand from the state file, so the human can freeze the window they want
and commit it through ``main`` before the demo, with the worker stopped or running.

What it does: pick the newest good window in ``paths.stateFile`` that still has its live bundle
(a window with candidate events unless ``allow_empty``), re-check that bundle as ``live``, copy
it beside ``paths.snapshotDir`` with ``meta.mode`` set to ``snapshot`` (nothing else changes: the
run, the events and the evidence are the window's own), check the copy with
``check_bundle(mode="snapshot")`` and swap it in atomically. Any failure leaves the previous
snapshot exactly as it was.
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from hq_contracts.models import BundleMeta

from hq.config import ConfigError, load_config
from hq.config.export import ExportConfig
from hq.export import BundleCheckError, check_bundle, replace_directory
from hq.export.bundle import dump_json
from hq.export.files import META_JSON
from hq_api.config import LiveConfig, resolve_path
from hq_api.state import GOOD_OUTCOMES, LiveState, StateError, WindowRecord, read_state

log = logging.getLogger(__name__)

LIVE_MODE = "live"
SNAPSHOT_MODE = "snapshot"


class FreezeError(RuntimeError):
    """No window can be frozen, or the copy did not check out."""


@dataclass(frozen=True)
class FreezeResult:
    record: WindowRecord  # the window that was frozen
    source: Path  # its live bundle
    target: Path  # the snapshot bundle written
    counts: dict[str, int]  # what check_bundle counted in the copy


def freezable_window(state: LiveState, *, allow_empty: bool = False) -> WindowRecord | None:
    """The newest good window whose live bundle still exists; windows without candidate events
    are skipped unless ``allow_empty`` (the same rule as ``snapshot.writeEmptyWindows``)."""
    for record in state.history:  # newest first
        if record.outcome not in GOOD_OUTCOMES or record.bundleDir is None:
            continue
        if record.eventCount == 0 and not allow_empty:
            continue
        if Path(record.bundleDir).is_dir():
            return record
        log.warning(
            "freeze-snapshot: window %s has no bundle any more at %s (pruned by keepBundles?)",
            record.runId,
            record.bundleDir,
        )
    return None


def export_config_for(config: LiveConfig, root: Path) -> ExportConfig:
    """The exporter's caps and rounding from ``paths.configDir/export.yaml``, so the copy is
    checked exactly as the worker checked the original."""
    config_dir = resolve_path(config.paths.configDir, root)
    try:
        return load_config(config_dir).export
    except ConfigError as exc:
        raise FreezeError(f"cannot load export.yaml from {config_dir}: {exc}") from exc


def freeze_snapshot(config: LiveConfig, root: Path, *, allow_empty: bool = False) -> FreezeResult:
    """Copy the last good live bundle into ``paths.snapshotDir`` as a checked ``snapshot`` bundle."""
    state_file = resolve_path(config.paths.stateFile, root)
    try:
        state = read_state(state_file)
    except StateError as exc:
        raise FreezeError(str(exc)) from exc
    record = freezable_window(state, allow_empty=allow_empty)
    if record is None:
        raise FreezeError(
            f"no good live window with a bundle in {state_file}"
            + ("" if allow_empty else " (windows without candidate events are skipped; ")
            + ("" if allow_empty else "pass --allow-empty to freeze one anyway)")
        )
    assert record.bundleDir is not None  # freezable_window guarantees it
    source = Path(record.bundleDir)
    export_cfg = export_config_for(config, root)
    caps = {
        "rounding": export_cfg.rounding,
        "max_evidence_bytes": export_cfg.evidence.maxFileBytes,
        "max_bundle_bytes": export_cfg.maxBundleBytes,
    }
    try:
        check_bundle(source, mode=LIVE_MODE, **caps)
    except BundleCheckError as exc:
        raise FreezeError(
            f"live bundle {source} of window {record.runId} does not check: {exc}"
        ) from exc
    target = resolve_path(config.paths.snapshotDir, root)
    log.info(
        "freeze-snapshot: window %s (run %s, %d candidate events, updated %.0f) from %s -> %s",
        f"{record.windowStart:.0f}/{record.windowEnd:.0f}",
        record.runId,
        record.eventCount,
        record.updatedAt,
        source,
        target,
    )

    def build(tmp: Path) -> dict[str, int]:
        shutil.copytree(source, tmp, dirs_exist_ok=True)
        meta_path = tmp / META_JSON
        meta = BundleMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        frozen = meta.model_copy(update={"mode": SNAPSHOT_MODE})
        dump_json(meta_path, frozen.model_dump(mode="json"), pretty=True)
        try:
            return check_bundle(tmp, mode=SNAPSHOT_MODE, **caps)
        except BundleCheckError as exc:
            raise FreezeError(
                f"snapshot copy of window {record.runId} does not check: {exc}"
            ) from exc

    counts = replace_directory(build, target)
    log.info(
        "freeze-snapshot: wrote %s: %s; commit it through main so the deployed site has it",
        target,
        json.dumps(counts, sort_keys=True),
    )
    return FreezeResult(record=record, source=source, target=target, counts=counts)

"""Where evidence waveforms come from.

H1 provides ``hq.ingest.cache.read_window`` and ``hq.preprocess.display_copy`` (docs/02 §5). The
exporter never imports them at module load: ``real_waveform_source`` resolves them when the
stage runs, so the export package imports (and its tests run) before H1's modules are merged,
and a missing one fails with a message naming H1. Tests inject a ``WaveformSource`` built from
synthetic streams instead.
"""

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from obspy import Stream

from hq.export.errors import ExportError

log = logging.getLogger(__name__)

H1 = "H1 Signal"
READ_WINDOW_MODULE, READ_WINDOW_NAME = "hq.ingest.cache", "read_window"
DISPLAY_COPY_MODULE, DISPLAY_COPY_NAME = "hq.preprocess", "display_copy"

ReadWindow = Callable[..., Stream]  # (station_id, t0, t1, *, cache_dir) -> Stream
DisplayCopy = Callable[[Stream, tuple[float, float]], Stream]


class WaveformSource(Protocol):
    """The two H1 calls the evidence builder needs (docs/02 §5 signatures)."""

    def read_window(self, station_id: str, t0: float, t1: float, *, cache_dir: Path) -> Stream:
        """Raw counts for ``[t0, t1]``; gaps stay separate traces, nothing is zero-filled."""
        ...

    def display_copy(self, st: Stream, band_hz: tuple[float, float]) -> Stream:
        """Detrended, tapered, zero-phase bandpassed copy for display only."""
        ...


@dataclass(frozen=True)
class LaneWaveformSource:
    """A ``WaveformSource`` over two plain functions (H1's, or a test's)."""

    read_window: ReadWindow
    display_copy: DisplayCopy


def _resolve(module: str, name: str) -> Callable[..., Stream]:
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if module == missing or module.startswith(missing + "."):
            raise ExportError(
                f"evidence needs {module}.{name} (docs/02 §5), which is not merged yet "
                f"(owner: {H1})"
            ) from exc
        raise  # the module exists but one of its own imports is broken: a real error
    fn = getattr(mod, name, None)
    if not callable(fn):
        raise ExportError(
            f"evidence needs {module}.{name} (docs/02 §5), which {module} does not expose "
            f"(owner: {H1})"
        )
    return fn


def real_waveform_source() -> LaneWaveformSource:
    """H1's cache reader and display filter, imported now; a clear error names H1 if absent."""
    read_window = _resolve(READ_WINDOW_MODULE, READ_WINDOW_NAME)
    display_copy = _resolve(DISPLAY_COPY_MODULE, DISPLAY_COPY_NAME)
    log.info(
        "export: waveforms from %s.%s and %s.%s",
        READ_WINDOW_MODULE,
        READ_WINDOW_NAME,
        DISPLAY_COPY_MODULE,
        DISPLAY_COPY_NAME,
    )
    return LaneWaveformSource(read_window=read_window, display_copy=display_copy)

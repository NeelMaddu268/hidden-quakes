"""Hour-scale model chunks shared by the PhaseNet (SEIS-06) and STA/LTA (SEIS-07) pickers.

Both full-window pickers read the waveform cache through ``iter_model_chunks``, so they see
identical preprocessed traces and drop picks by identical rules (``pick_is_kept``).

Tiling. ``[t0, t1)`` is cut into keep intervals whose boundaries sit on multiples of
``preprocess.chunks.lengthS`` since the epoch, clipped to ``[t0, t1)``. The intervals cover
``[t0, t1)`` exactly once, so a run over a sub-window sees the same chunks as the full run.

Reads. Each keep interval is read over ``[keep0 - overlapS, keep1 + overlapS]`` so the taper,
the filter settling, the picker's own window and the STA/LTA warm-up all fall outside the keep
interval. ``read_window`` never fabricates samples: where the cache has no data the chunk simply
has less. Every read reaches ``edgeProbeS`` further on both sides, and the extra samples are
trimmed off before preprocessing. They only tell a real data edge from a chunk cut: data that runs
on past the read span means the span end is an artificial cut.

Data edges. ``ModelChunk.dataEdges`` lists, in real time, the first and last sample time of every
contiguous run of RAW samples (before preprocessing) on the selected channels, all components
together. Chunk cuts are not data edges; a place where the cached data genuinely stops (a gap, or
the end of what was ever downloaded) is one, even when it coincides with a read-span end.

Keep rule (``pick_is_kept``), applied to a pick's REAL time (``timemap.to_real`` of the model
time): outside the keep interval, the pick belongs to the neighbouring chunk and is discarded
without counting; within ``gap_edge_s`` of any data edge, it is dropped and counted (a picker
fires on the step at a gap edge); otherwise it is kept.
"""

from __future__ import annotations

import logging
import math
import time
from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, Protocol

from obspy import Stream, UTCDateTime

from hq.config.signal import SignalConfig
from hq.ingest.cache import read_window as cache_read_window
from hq.preprocess.profiles import TimeMap, for_picking

log = logging.getLogger(__name__)

KeepReason = Literal["ok", "outside_keep", "near_gap_edge"]
KEPT: KeepReason = "ok"
OUTSIDE_KEEP: KeepReason = "outside_keep"
NEAR_GAP_EDGE: KeepReason = "near_gap_edge"


class ReadWindow(Protocol):
    """Signature of ``hq.ingest.cache.read_window`` (docs/02 section 5)."""

    def __call__(self, station_id: str, t0: float, t1: float, *, cache_dir: Path) -> Stream: ...


@dataclass(frozen=True)
class ModelChunk:
    """One preprocessed chunk of one station, ready for a picker.

    ``stream`` is ``for_picking`` output: ``targetRateHz``, components Z/N/E, gap-separated
    segments as separate traces, times in MODEL time. ``keep`` and ``dataEdges`` are REAL time;
    convert picks with ``timemap.to_real`` before ``pick_is_kept``.
    """

    stationId: str
    profile: str
    stream: Stream
    timemap: TimeMap
    keep: tuple[float, float]  # real [start, end) this chunk owns
    dataEdges: tuple[float, ...]  # real times of raw data edges, sorted; chunk cuts excluded

    def __post_init__(self) -> None:
        if not self.keep[0] < self.keep[1]:
            raise ValueError(f"{self.stationId}: empty keep interval {self.keep}")
        if any(b < a for a, b in zip(self.dataEdges, self.dataEdges[1:], strict=False)):
            raise ValueError(f"{self.stationId}: dataEdges must be sorted")


@dataclass
class ChunkStats:
    """Counts one or more ``iter_model_chunks`` calls add to (pass one in to collect them)."""

    planned: int = 0  # keep intervals in [t0, t1)
    empty: int = 0  # no samples on the selected channels in the read span
    noSegments: int = 0  # samples, but for_picking dropped every segment as too short
    yielded: int = 0
    otherChannelTraces: int = 0  # traces read on channels not selected, dropped

    def add(self, other: ChunkStats) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def as_counts(self, prefix: str = "chunks") -> dict[str, int]:
        return {
            prefix + f.name[0].upper() + f.name[1:]: getattr(self, f.name) for f in fields(self)
        }


def plan_keeps(t0: float, t1: float, length_s: float) -> list[tuple[float, float]]:
    """Keep intervals tiling ``[t0, t1)`` exactly once, cut at multiples of ``length_s``.

    The first and last intervals are clipped to the window, so they may be shorter.
    """
    if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
        raise ValueError(f"chunk window must satisfy t0 < t1, got [{t0}, {t1})")
    if not length_s > 0.0:
        raise ValueError(f"chunk length must be positive, got {length_s}")
    out: list[tuple[float, float]] = []
    start = t0
    k = math.floor(t0 / length_s) + 1
    while start < t1:
        end = min(k * length_s, t1)
        k += 1
        if end <= start:  # t0 sits on a boundary up to rounding: no empty sliver
            continue
        out.append((start, end))
        start = end
    return out


def data_edges(raw: Stream, probe0: float, probe1: float) -> tuple[float, ...]:
    """Real data edges of ``raw``, read over ``[probe0, probe1]`` (both ends inclusive).

    A trace whose first sample lies within one sample interval of ``probe0`` may continue before
    it, so that start is a cut and is left out; the same holds for a last sample within one
    interval of ``probe1``. Every other trace start and end is a data edge.
    """
    edges: set[float] = set()
    for tr in raw:
        delta = float(tr.stats.delta)
        start = tr.stats.starttime.timestamp
        end = tr.stats.endtime.timestamp
        if start >= probe0 + delta:
            edges.add(start)
        if end <= probe1 - delta:
            edges.add(end)
    return tuple(sorted(edges))


def iter_model_chunks(
    station_id: str,
    channels: Sequence[str],
    profile: str,
    t0: float,
    t1: float,
    cfg: SignalConfig,
    *,
    cache_dir: Path,
    read_window: ReadWindow | None = None,
    stats: ChunkStats | None = None,
) -> Iterator[ModelChunk]:
    """Yield the preprocessed chunks of one station over ``[t0, t1)``, in time order.

    ``channels`` are the station's chosen channels (``Station.channels``); every other cached
    channel is dropped and counted. A chunk with no samples, or whose segments ``for_picking``
    drops as too short, yields nothing and is counted in ``stats`` and the log. Errors from
    ``read_window`` (e.g. ``CacheMissError``) and from ``for_picking`` propagate.
    """
    wanted = frozenset(channels)
    if not wanted:
        raise ValueError(f"{station_id}: no channels selected")
    reader: ReadWindow = cache_read_window if read_window is None else read_window
    local = ChunkStats()
    began = time.perf_counter()
    try:
        yield from _chunks(station_id, wanted, profile, t0, t1, cfg, cache_dir, reader, local)
    finally:  # also when the consumer stops early or an error propagates
        log.info(
            "%s chunks: planned=%d yielded=%d empty=%d no_segments=%d other_channel_traces=%d "
            "runtime_s=%.2f (includes the consumer's time between chunks)",
            station_id,
            local.planned,
            local.yielded,
            local.empty,
            local.noSegments,
            local.otherChannelTraces,
            time.perf_counter() - began,
        )
        if stats is not None:
            stats.add(local)


def _chunks(
    station_id: str,
    wanted: frozenset[str],
    profile: str,
    t0: float,
    t1: float,
    cfg: SignalConfig,
    cache_dir: Path,
    reader: ReadWindow,
    local: ChunkStats,
) -> Iterator[ModelChunk]:
    ccfg = cfg.preprocess.chunks
    for keep0, keep1 in plan_keeps(t0, t1, ccfg.lengthS):
        local.planned += 1
        read0, read1 = keep0 - ccfg.overlapS, keep1 + ccfg.overlapS
        probe0, probe1 = read0 - ccfg.edgeProbeS, read1 + ccfg.edgeProbeS
        raw = reader(station_id, probe0, probe1, cache_dir=cache_dir)
        selected = Stream(traces=[tr for tr in raw if tr.stats.channel in wanted])
        local.otherChannelTraces += len(raw) - len(selected)
        for tr in selected:
            if tr.stats.delta >= ccfg.edgeProbeS:
                raise ValueError(
                    f"{tr.id}: sample interval {tr.stats.delta} s is not below "
                    f"preprocess.chunks.edgeProbeS {ccfg.edgeProbeS} s"
                )
        edges = data_edges(selected, probe0, probe1)
        in_span = selected.trim(UTCDateTime(read0), UTCDateTime(read1), nearest_sample=False)
        if len(in_span) == 0:
            local.empty += 1
            log.info(
                "%s chunk [%s, %s): no samples on %s",
                station_id,
                UTCDateTime(keep0),
                UTCDateTime(keep1),
                ",".join(sorted(wanted)),
            )
            continue
        model, tmap = for_picking(in_span, profile, cfg)
        if len(model) == 0:
            local.noSegments += 1
            log.info(
                "%s chunk [%s, %s): every segment too short for %s",
                station_id,
                UTCDateTime(keep0),
                UTCDateTime(keep1),
                profile,
            )
            continue
        local.yielded += 1
        yield ModelChunk(
            stationId=station_id,
            profile=profile,
            stream=model,
            timemap=tmap,
            keep=(keep0, keep1),
            dataEdges=edges,
        )


def near_data_edge(t: float, edges: Sequence[float], gap_edge_s: float) -> bool:
    """True when real time ``t`` is within ``gap_edge_s`` of any of the sorted ``edges``."""
    i = bisect_left(edges, t)
    return any(0 <= j < len(edges) and abs(edges[j] - t) <= gap_edge_s for j in (i - 1, i))


def pick_is_kept(t: float, chunk: ModelChunk, gap_edge_s: float) -> tuple[bool, KeepReason]:
    """Whether a pick at REAL time ``t`` from ``chunk`` is kept, and why not.

    ``outside_keep``: the pick belongs to the neighbouring chunk; discard it without counting.
    ``near_gap_edge``: within ``gap_edge_s`` of a raw data edge; drop it and count it.
    """
    if gap_edge_s < 0.0:
        raise ValueError(f"gap_edge_s must be >= 0, got {gap_edge_s}")
    keep0, keep1 = chunk.keep
    if not keep0 <= t < keep1:
        return False, OUTSIDE_KEEP
    if near_data_edge(t, chunk.dataEdges, gap_edge_s):
        return False, NEAR_GAP_EDGE
    return True, KEPT

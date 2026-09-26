"""PhaseNet picking for one station's stream, without ever letting the model see a filled gap.

What seisbench 0.12.6 does with a stream (``seisbench/models/base.py``; read, not assumed):

* ``annotate_async`` calls ``stream.merge(-1)`` (base.py:1370). That is ObsPy's cleanup merge: it
  joins directly adjacent or identical overlapping traces and leaves real gaps as separate traces
  (obspy/core/stream.py:3099).
* Grouping then cuts the station into intervals (``GroupingHelper._get_intervals``, base.py:285).
  With the default ``strict=False`` an interval only needs *one* component (base.py:1183: "otherwise
  impute missing data with zeros"). ``stream_to_array`` allocates ``np.zeros`` for the whole
  interval (base.py:2418) and copies each trace in (base.py:2428); its docstring says "Every
  remaining gap is intended to be filled with zeros" (base.py:2382) and assumes "No overlapping
  traces of the same component exist" (base.py:2383). Intervals shorter than the model window are
  merged into neighbours by ``_merge_intervals`` (base.py:319) or deleted (base.py:432). So
  seisbench *will* zero-fill a missing component, and sub-sample misalignment between components
  can leave a zero sample at a block end after ``_align_fractional_samples`` (base.py:245, 2431).
* Windows are cut in ``_cut_fragments_array``; a group shorter than ``in_samples`` (3001 for
  PhaseNet, phasenet.py:75) yields no window at all (base.py:1742). It is not padded. The window
  step is ``in_samples - overlap`` (base.py:1740), so ``overlap`` must stay below ``in_samples``.
* ``annotate_stream_pre`` resamples every trace whose rate differs from the model's 100 Hz
  (base.py:2073, skipped when equal at base.py:2289) and then rejects mismatches (base.py:2133).
* ``classify`` starts from the weight file's ``default_args`` (base.py:2216), which carry
  per-weight thresholds (e.g. scedc 0.41). Every argument is therefore passed explicitly here.
* ``annotate_batch_post`` sets ``blinding`` samples at each window edge to NaN (phasenet.py:217),
  overlapping windows are stacked with a NaN-aware mean or max (base.py:1876), and
  ``_predictions_to_stream`` trims NaN edges (base.py:2019). With ``overlap >= sum(blinding)`` the
  only blinded samples are the first and last ``blinding`` samples of a block, so no pick can land
  there. That is an edge exclusion of ``blinding / 100`` model seconds, independent of
  ``gapEdgeS``; ``PickDiagnostics`` reports it (``edgeExclusionS``, ``secondsBlinded``).
* The model is put in ``eval()`` (base.py:1403) and run under ``torch.no_grad()`` (base.py:1965);
  PhaseNet has BatchNorm and no dropout. Picks come from ``trigger_onset(prob, thr, thr / 2)``
  with the peak inside the trigger window (base.py:2511).

Consequently this module does the gap handling itself: it splits the ``for_picking`` output into
contiguous blocks where all of Z, N and E exist, treats conflicting overlaps within one component
as gaps, cuts the three traces of a block to the same sample grid and length (so
``stream_to_array`` has nothing to fill), skips and counts blocks shorter than the model window,
and classifies each block separately. Picks are converted to real time with the profile's
``TimeMap`` and dropped (and counted) when they fall within ``gapEdgeS`` real seconds of a block
edge. The sampling rate is never a config value here: every block must already be at the model's
own rate, so seisbench never resamples.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
import obspy

from hq.config.signal import PickerConfig, SignalConfig

log = logging.getLogger(__name__)

# Component letters seisbench maps by the last character of the channel code (base.py:2460);
# for_picking renames borehole 1/2 to N/E, so these are the only letters a block may carry.
COMPONENTS: tuple[str, str, str] = ("Z", "N", "E")
PHASES: tuple[str, str] = ("P", "S")
# Numerical tolerance when mapping a float time onto a sample index, in samples. Absorbs float64
# rounding of epoch seconds (~1e-7 s); it is not a pipeline parameter.
_INDEX_TOL = 1e-3


class TimeMapLike(Protocol):
    """The part of ``hq.preprocess.TimeMap`` this module uses."""

    def to_real(self, t: float) -> float: ...


ForPicking = Callable[[obspy.Stream, str, SignalConfig], tuple[obspy.Stream, TimeMapLike]]


@dataclass(frozen=True)
class Block:
    """One contiguous span where Z, N and E all have data (model time), ready for the model."""

    stream: obspy.Stream  # three traces on one sample grid, identical start and npts
    npts: int
    samplingRateHz: float
    startModel: float
    endModel: float
    startReal: float
    endReal: float

    @property
    def realPerModelS(self) -> float:
        """Real seconds per model second (1 except for time-stretched profiles)."""
        span = self.endModel - self.startModel
        return (self.endReal - self.startReal) / span if span > 0 else 1.0


@dataclass(frozen=True)
class BlockSplit:
    """``split_blocks`` output: the blocks plus what had to be removed to build them."""

    blocks: list[obspy.Stream]
    nOverlaps: int  # conflicting overlaps between segments of one component
    overlapRegions: list[tuple[float, float]]  # model-time spans removed from every trace there


@dataclass(frozen=True)
class PreparedStation:
    """``for_picking`` output split into blocks. Independent of the weights, so reusable."""

    stationId: str
    profile: str
    timeMap: TimeMapLike
    blocks: tuple[Block, ...]
    nTraces: int
    missingComponents: tuple[str, ...]
    nOverlaps: int = 0
    secondsOverlapRemoved: float = 0.0  # real seconds


@dataclass
class PickDiagnostics:
    """Per-station counts for one (profile, weights) pass."""

    stationId: str
    profile: str
    weights: str
    nTraces: int = 0
    missingComponents: list[str] = field(default_factory=list)
    nOverlaps: int = 0
    secondsOverlapRemoved: float = 0.0  # real seconds treated as gaps (conflicting overlaps)
    nBlocks: int = 0
    nBlocksPicked: int = 0
    nBlocksTooShort: int = 0
    secondsPicked: float = 0.0  # real seconds of data the model saw
    secondsTooShort: float = 0.0  # real seconds in blocks shorter than the model window
    secondsBlinded: float = 0.0  # real seconds at block edges where blinding leaves no output
    edgeExclusionS: float = (
        0.0  # real s at each block edge where no pick survives (max over blocks)
    )
    picksP: int = 0
    picksS: int = 0
    droppedNearEdgeP: int = 0
    droppedNearEdgeS: int = 0
    droppedBelowThreshold: int = 0
    runtimeS: float = 0.0

    @property
    def droppedNearEdge(self) -> int:
        return self.droppedNearEdgeP + self.droppedNearEdgeS

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["droppedNearEdge"] = self.droppedNearEdge
        return out


# --- model --------------------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def load_model(weights: str, picker: PickerConfig) -> Any:
    """Load ``seisbench.models.PhaseNet`` weights once per process, in eval mode, seeded.

    The weight version is pinned by ``picker.weightsVersion`` so seisbench reads its local cache
    and never asks the remote for "latest".
    """
    if picker.model != "seisbench.PhaseNet":
        raise ValueError(f"unsupported picker model {picker.model!r}")
    if weights not in picker.candidateWeights:
        raise ValueError(f"weights {weights!r} not in picker.candidateWeights")

    import torch  # heavy; imported only when a real model is needed

    torch.set_num_threads(picker.torchThreads)
    torch.manual_seed(picker.seed)

    key = (weights, picker.weightsVersion)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    import seisbench.models as sbm

    t0 = time.perf_counter()
    model = sbm.PhaseNet.from_pretrained(weights, version_str=picker.weightsVersion)
    model.eval()
    if model.training:
        raise RuntimeError(f"PhaseNet {weights} still in training mode after eval()")
    check_model(model, picker)
    _MODEL_CACHE[key] = model
    log.info(
        "loaded PhaseNet weights=%s version=%s rate=%s Hz components=%s in %.2f s",
        weights,
        picker.weightsVersion,
        model.sampling_rate,
        model.component_order,
        time.perf_counter() - t0,
    )
    return model


def check_model(model: Any, picker: PickerConfig) -> None:
    """The model must have a fixed rate, map exactly Z/N/E, and fit the configured overlap."""
    if model.sampling_rate is None or float(model.sampling_rate) <= 0.0:
        raise ValueError(f"model has no fixed sampling rate: {model.sampling_rate!r}")
    if set(str(model.component_order)) != set(COMPONENTS):
        raise ValueError(f"model component order {model.component_order!r} is not a Z/N/E order")
    if picker.seisbench.overlap >= int(model.in_samples):
        raise ValueError(
            f"seisbench.overlap {picker.seisbench.overlap} must be below the model window "
            f"{model.in_samples} samples (the window step would be <= 0)"
        )


def classify_kwargs(picker: PickerConfig) -> dict[str, Any]:
    """Every argument handed to ``classify``; overrides the weight file's ``default_args``."""
    sb = picker.seisbench
    return {
        "P_threshold": picker.pThreshold,
        "S_threshold": picker.sThreshold,
        "batch_size": picker.batchSize,
        "overlap": sb.overlap,
        "stacking": sb.stacking,
        "blinding": tuple(sb.blinding),
        "strict": sb.strict,
        "flexible_horizontal_components": sb.flexibleHorizontalComponents,
    }


# --- picks --------------------------------------------------------------------------------------


def picker_name(weights: str) -> str:
    return f"phasenet:{weights}"


def make_pick(station_id: str, phase: str, t: float, prob: float, weights: str) -> dict[str, Any]:
    """A ``Pick`` row with the exact docs/02 field names."""
    if phase not in PHASES:
        raise ValueError(f"phase must be P or S, got {phase!r}")
    picker = picker_name(weights)
    return {
        "id": f"{picker}:{station_id}:{phase}:{t:.3f}",
        "stationId": station_id,
        "phase": phase,
        "t": float(t),
        "prob": float(prob),
        "picker": picker,
        "eventId": None,
        "residualS": None,
        "weight": None,
    }


# --- blocks -------------------------------------------------------------------------------------


def _intersect(
    a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Intersection of two sorted lists of disjoint closed intervals; empty overlaps dropped."""
    out: list[tuple[float, float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _span(tr: obspy.Trace) -> tuple[float, float]:
    return tr.stats.starttime.timestamp, tr.stats.endtime.timestamp


def _union(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lo, hi in sorted(regions):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _remove_overlaps(
    traces: list[obspy.Trace], sr: float
) -> tuple[list[obspy.Trace], int, list[tuple[float, float]]]:
    """Treat every span covered by two or more segments of one component as a gap.

    ``merge(-1)`` has already joined identical overlaps, so what is left disagrees and neither
    copy can be trusted. Returns disjoint traces sorted by start, the number of overlapping pairs
    and the removed regions (closed intervals, model time).
    """
    tol_s = _INDEX_TOL / sr
    spans = [_span(tr) for tr in traces]
    pairs: list[tuple[float, float]] = []
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            lo = max(spans[i][0], spans[j][0])
            hi = min(spans[i][1], spans[j][1])
            if lo <= hi + tol_s:
                pairs.append((lo, max(lo, hi)))
    if not pairs:
        return sorted(traces, key=lambda tr: tr.stats.starttime), 0, []
    regions = _union(pairs)
    out: list[obspy.Trace] = []
    for tr in traces:
        start = tr.stats.starttime.timestamp
        keep = np.ones(tr.stats.npts, dtype=bool)
        for lo, hi in regions:
            i0 = max(math.ceil((lo - start) * sr - _INDEX_TOL), 0)
            i1 = min(math.floor((hi - start) * sr + _INDEX_TOL), tr.stats.npts - 1)
            if i0 <= i1:
                keep[i0 : i1 + 1] = False
        edges = np.flatnonzero(np.diff(np.concatenate(([0], keep.astype(np.int8), [0]))))
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            piece = tr.copy()
            piece.data = np.array(tr.data[a:b], copy=True)
            piece.stats.starttime = tr.stats.starttime + a / sr
            out.append(piece)
    return sorted(out, key=lambda tr: tr.stats.starttime), len(pairs), regions


def _cut_block(
    by_comp: dict[str, list[obspy.Trace]], t0: float, t1: float, sr: float
) -> obspy.Stream | None:
    """Cut Z/N/E to [t0, t1] on one sample grid with one length, so nothing is left to fill."""
    delta = 1.0 / sr
    tol_s = _INDEX_TOL * delta
    pieces: dict[str, list[Any]] = {}
    for comp in COMPONENTS:
        matches = [
            tr
            for tr in by_comp[comp]
            if tr.stats.starttime.timestamp <= t0 + tol_s
            and tr.stats.endtime.timestamp >= t1 - tol_s
        ]
        if len(matches) != 1:  # spans are disjoint by construction; this is an internal check
            raise RuntimeError(f"block [{t0}, {t1}] is covered by {len(matches)} {comp} traces")
        tr = matches[0]
        start = tr.stats.starttime.timestamp
        i0 = max(math.ceil((t0 - start) * sr - _INDEX_TOL), 0)
        i1 = min(math.floor((t1 - start) * sr + _INDEX_TOL), tr.stats.npts - 1)
        pieces[comp] = [tr, i0, i1]

    # seisbench snaps every trace start to round(t * sr) / sr (base.py:245) and sizes its array from
    # the earliest start to the latest end. Put all three on the same grid index so it pads nothing.
    grid = {
        comp: round((tr.stats.starttime.timestamp + i0 / sr) * sr)
        for comp, (tr, i0, _) in pieces.items()
    }
    top = max(grid.values())
    for comp in COMPONENTS:
        pieces[comp][1] += top - grid[comp]
    n = min(i1 - i0 + 1 for _, i0, i1 in pieces.values())
    if n < 1:
        return None

    out = obspy.Stream()
    for comp in COMPONENTS:
        tr, i0, _ = pieces[comp]
        header = {
            key: tr.stats[key]
            for key in ("network", "station", "location", "channel", "sampling_rate")
        }
        header["starttime"] = tr.stats.starttime + i0 / sr
        # A copied Stats header would keep the parent's npts; build a fresh one from the data.
        out.append(obspy.Trace(data=np.array(tr.data[i0 : i0 + n], copy=True), header=header))
    return out


def split_blocks(st: obspy.Stream, sample_rate_hz: float) -> BlockSplit:
    """Contiguous three-component blocks: the intersection of the Z, N and E segment spans.

    ``st`` must be one instrument with one channel per component, named Z/N/E. Contiguous pieces
    of one component are joined first with ObsPy's cleanup merge (``merge(-1)``), which never
    fills gaps; spans where two segments of one component still overlap are removed from both and
    counted. Returns no blocks when a component is missing entirely.
    """
    st = st.copy()
    st.merge(method=-1)
    by_comp: dict[str, list[obspy.Trace]] = {}
    n_overlaps = 0
    regions: list[tuple[float, float]] = []
    for comp in COMPONENTS:
        traces = [tr for tr in st if tr.stats.channel[-1:] == comp and tr.stats.npts > 0]
        traces, n, removed = _remove_overlaps(traces, sample_rate_hz)
        by_comp[comp] = traces
        n_overlaps += n
        regions.extend(removed)
    if n_overlaps:
        log.warning(
            "%s: %d conflicting overlaps treated as gaps (%s)",
            st[0].id if len(st) else "?",
            n_overlaps,
            ", ".join(f"[{lo:.3f}, {hi:.3f}]" for lo, hi in _union(regions)),
        )
    if any(not traces for traces in by_comp.values()):
        return BlockSplit(blocks=[], nOverlaps=n_overlaps, overlapRegions=_union(regions))
    spans: list[tuple[float, float]] | None = None
    for comp in COMPONENTS:
        comp_spans = [_span(tr) for tr in by_comp[comp]]
        spans = comp_spans if spans is None else _intersect(spans, comp_spans)
    blocks: list[obspy.Stream] = []
    for t0, t1 in spans or []:
        block = _cut_block(by_comp, t0, t1, sample_rate_hz)
        if block is not None:
            blocks.append(block)
    return BlockSplit(blocks=blocks, nOverlaps=n_overlaps, overlapRegions=_union(regions))


def _validate_model_stream(st: obspy.Stream, station_id: str) -> float | None:
    """Checks on the ``for_picking`` output; returns its single sampling rate (None if empty)."""
    rates = {float(tr.stats.sampling_rate) for tr in st}
    if len(rates) > 1:
        raise ValueError(f"{station_id}: for_picking returned several rates {sorted(rates)} Hz")
    comps = {tr.stats.channel[-1:] for tr in st}
    if not comps <= set(COMPONENTS):
        raise ValueError(
            f"{station_id}: for_picking returned components {sorted(comps)}, not Z/N/E"
        )
    channels: dict[str, set[str]] = {}
    for tr in st:
        channels.setdefault(tr.stats.channel[-1:], set()).add(tr.stats.channel)
    doubled = {comp: sorted(codes) for comp, codes in channels.items() if len(codes) > 1}
    if doubled:
        raise ValueError(f"{station_id}: several channels per component {doubled}")
    instruments = {(tr.stats.network, tr.stats.station, tr.stats.location) for tr in st}
    if len(instruments) > 1:
        raise ValueError(f"{station_id}: for_picking returned several instruments {instruments}")
    masked = [tr.id for tr in st if np.ma.isMaskedArray(tr.data) and np.ma.is_masked(tr.data)]
    if masked:
        raise ValueError(f"{station_id}: masked (merged-over) gaps in {masked}")
    return next(iter(rates)) if rates else None


def _default_for_picking() -> ForPicking:
    from hq.preprocess import for_picking  # SEIS-03

    return for_picking


def prepare_station(
    raw: obspy.Stream,
    station_id: str,
    profile: str,
    cfg: SignalConfig,
    *,
    for_picking: ForPicking | None = None,
) -> PreparedStation:
    """Run ``for_picking`` and split its output into model-ready blocks (weights-independent)."""
    fp = for_picking if for_picking is not None else _default_for_picking()
    model_st, time_map = fp(raw, profile, cfg)
    sr = _validate_model_stream(model_st, station_id)
    present = {tr.stats.channel[-1:] for tr in model_st if tr.stats.npts > 0}
    missing = tuple(c for c in COMPONENTS if c not in present)
    if missing:
        log.warning(
            "%s (%s): components %s missing; station not picked", station_id, profile, missing
        )
    blocks: list[Block] = []
    n_overlaps = 0
    overlap_real_s = 0.0
    if sr is not None:
        split = split_blocks(model_st, sr)
        n_overlaps = split.nOverlaps
        overlap_real_s = sum(
            float(time_map.to_real(hi)) - float(time_map.to_real(lo))
            for lo, hi in split.overlapRegions
        )
        for block_st in split.blocks:
            start = max(tr.stats.starttime.timestamp for tr in block_st)
            end = min(tr.stats.endtime.timestamp for tr in block_st)
            blocks.append(
                Block(
                    stream=block_st,
                    npts=block_st[0].stats.npts,
                    samplingRateHz=sr,
                    startModel=start,
                    endModel=end,
                    startReal=float(time_map.to_real(start)),
                    endReal=float(time_map.to_real(end)),
                )
            )
    return PreparedStation(
        stationId=station_id,
        profile=profile,
        timeMap=time_map,
        blocks=tuple(blocks),
        nTraces=len(model_st),
        missingComponents=missing,
        nOverlaps=n_overlaps,
        secondsOverlapRemoved=overlap_real_s,
    )


def pick_prepared(
    prepared: PreparedStation, cfg: SignalConfig, model: Any, weights: str
) -> tuple[list[dict[str, Any]], PickDiagnostics]:
    """Classify each block separately; convert to real time; drop and count gap-edge picks."""
    t_start = time.perf_counter()
    picker = cfg.picker
    check_model(model, picker)
    model_sr = float(model.sampling_rate)
    min_samples = int(model.in_samples)
    kwargs = classify_kwargs(picker)
    thresholds = {"P": picker.pThreshold, "S": picker.sThreshold}
    blind_pre, blind_post = picker.seisbench.blinding
    diag = PickDiagnostics(
        stationId=prepared.stationId,
        profile=prepared.profile,
        weights=weights,
        nTraces=prepared.nTraces,
        missingComponents=list(prepared.missingComponents),
        nOverlaps=prepared.nOverlaps,
        secondsOverlapRemoved=prepared.secondsOverlapRemoved,
        nBlocks=len(prepared.blocks),
    )
    picks: list[dict[str, Any]] = []
    for block in prepared.blocks:
        if block.samplingRateHz != model_sr:
            raise ValueError(
                f"{prepared.stationId} ({prepared.profile}): for_picking delivered "
                f"{block.samplingRateHz} Hz, the model runs at {model_sr} Hz "
                "(the model must never resample)"
            )
        real_len = block.endReal - block.startReal
        if block.npts < min_samples:
            diag.nBlocksTooShort += 1
            diag.secondsTooShort += real_len
            continue
        diag.nBlocksPicked += 1
        diag.secondsPicked += real_len
        to_real_s = block.realPerModelS / model_sr  # real seconds per model sample
        diag.secondsBlinded += (blind_pre + blind_post) * to_real_s
        diag.edgeExclusionS = max(
            diag.edgeExclusionS, picker.gapEdgeS, max(blind_pre, blind_post) * to_real_s
        )
        output = model.classify(block.stream, **kwargs)
        for sb_pick in output.picks:
            phase = str(sb_pick.phase)
            if phase not in PHASES:
                raise ValueError(f"unexpected phase {phase!r} from PhaseNet {weights}")
            prob = float(sb_pick.peak_value)
            if prob < thresholds[phase]:
                diag.droppedBelowThreshold += 1
                continue
            t_real = float(prepared.timeMap.to_real(float(sb_pick.peak_time.timestamp)))
            if (
                t_real - block.startReal < picker.gapEdgeS
                or block.endReal - t_real < picker.gapEdgeS
            ):
                if phase == "P":
                    diag.droppedNearEdgeP += 1
                else:
                    diag.droppedNearEdgeS += 1
                continue
            picks.append(make_pick(prepared.stationId, phase, t_real, prob, weights))
    picks.sort(key=lambda p: (p["t"], p["phase"]))
    diag.picksP = sum(1 for p in picks if p["phase"] == "P")
    diag.picksS = sum(1 for p in picks if p["phase"] == "S")
    diag.runtimeS = time.perf_counter() - t_start
    log.debug(
        "%s %s %s: blocks %d (picked %d, too short %d) P %d S %d dropped-near-edge %d "
        "blinded %.1f s in %.2f s",
        prepared.stationId,
        prepared.profile,
        weights,
        diag.nBlocks,
        diag.nBlocksPicked,
        diag.nBlocksTooShort,
        diag.picksP,
        diag.picksS,
        diag.droppedNearEdge,
        diag.secondsBlinded,
        diag.runtimeS,
    )
    return picks, diag


def pick_stream(
    raw: obspy.Stream,
    station_id: str,
    profile: str,
    cfg: SignalConfig,
    model: Any,
    weights: str,
    *,
    for_picking: ForPicking | None = None,
) -> tuple[list[dict[str, Any]], PickDiagnostics]:
    """Pick one station's raw stream: ``for_picking`` -> blocks -> PhaseNet -> real-time picks.

    Returns ``Pick`` dicts (docs/02 field names, ``t`` in real epoch seconds) and the station's
    diagnostics (blocks, picks by phase, picks dropped near gap/data edges, blinded seconds).
    """
    t0 = time.perf_counter()
    prepared = prepare_station(raw, station_id, profile, cfg, for_picking=for_picking)
    picks, diag = pick_prepared(prepared, cfg, model, weights)
    diag.runtimeS = time.perf_counter() - t0
    log.info(
        "%s %s %s: %d blocks (%d too short), P %d, S %d, dropped near edges %d, "
        "edge exclusion %.2f s, overlaps %d, %.2f s",
        station_id,
        profile,
        weights,
        diag.nBlocks,
        diag.nBlocksTooShort,
        diag.picksP,
        diag.picksS,
        diag.droppedNearEdge,
        diag.edgeExclusionS,
        diag.nOverlaps,
        diag.runtimeS,
    )
    return picks, diag

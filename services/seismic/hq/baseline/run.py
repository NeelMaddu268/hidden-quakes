"""STA/LTA baseline picker (SEIS-07): classical picks plus a trigger-threshold sweep.

Stage ``baseline`` (docs/01): cache + ``stations.parquet`` -> ``picks_stalta.parquet`` and
``baseline_sweep.parquet`` in the run directory.

Input. Every ``usedInRun`` station is read through ``hq.preprocess.chunks.iter_model_chunks``,
the same hour chunks and the same ``for_picking`` output PhaseNet (SEIS-06) picks on, and every
pick goes through the same ``pick_is_kept`` rule with the same distance, ``picker.gapEdgeS``:
outside the chunk's keep interval it belongs to the neighbouring chunk; within ``gapEdgeS`` of a
raw data edge it is dropped and counted.

Characteristic function. Per gap-separated segment: a causal Butterworth bandpass chosen by the
station's preprocessing profile (``baseline.prefilter[profile]``, real Hz), then ObsPy's
``recursive_sta_lta``. P uses the vertical, S each horizontal, each with its own STA/LTA windows.
Windows are REAL seconds, so a time-stretched ``borehole-B`` segment gets the same physical
windows (``TimeMap.factor`` x more model samples). Each function is computed ONCE per segment and
window setting; every threshold of the sweep and the chosen thresholds then only re-run the cheap
``trigger_onset``.

Warm-up. The recursive LTA starts from zero at every segment start, so the ratio is inflated
there. A trigger that starts in the first ``warmupS`` of a segment is dropped; at the chosen
thresholds the ones inside the keep interval and not already within ``gapEdgeS`` of a data edge
are counted (``suppressedWarmupP``/``suppressedWarmupS``). After a gap the effective exclusion is
therefore ``warmupS`` (P 6 s, S 12 s in the showcase config), not ``gapEdgeS``. Segments shorter
than ``warmupS + minSegmentMarginS`` could never trigger and are skipped and counted.

Onsets. A pick is the first sample of a ``trigger_onset`` trigger, converted from model time to
real time with ``TimeMap.to_real`` (identity except ``borehole-B``). P: every vertical trigger.
S: for each P of the station (a P within ``gapEdgeS`` of a data edge does not count, and the S
picks lost that way are counted as ``sLostPNearGapEdge``), each horizontal is searched on its
own. On one horizontal, a trigger that is on at any time within ``pHorizontalTolS`` of the P
onset is that P's own energy on the component (an emergent P can cross the level there a few
tenths of a second after the vertical): it is never an S, and the search on that component
starts where it ends, so P-coda re-triggers do not become S picks. The component's S is its
earliest other trigger starting in ``[tP + minSMinusPS, tP + maxSMinusPS]``, and the pick is the
earlier of the two components'. Searching per component matters: a P trigger that stays on one
horizontal must not hide the S onset on the other. Earliest rather than strongest because the
first new horizontal energy is the onset and a later, stronger trigger is more often coda. Two Ps
choosing the same trigger give one S pick; horizontal triggers with no P ahead of them are not
picks. Known limitations, as for any classical S picker: when P energy keeps a horizontal trigger
on until the S arrives, that S has no onset of its own on that component; and an S less than
about ``pHorizontalTolS`` after the P is taken for the P's own trigger. Onsets are late-biased by
the STA build-up and by the causal prefilter's group delay, which is larger at low frequency, so
S is biased later than P (numbers in ``signal.yaml``).

Probability. STA/LTA has no calibrated confidence, so every pick gets ``prob = baseline.prob``
(1.0). The swept trigger thresholds are the baseline's only tuning dimension: with prob 1.0 no
association probability threshold ever drops a baseline pick. A baseline event's
``meanPickProb`` is therefore always 1.0 and says nothing about confidence; it is not comparable
with a PhaseNet event's.

Sweep. Grid ``sweep.pOn x sweep.sOn x sweep.offLevels`` (one off level for both phases). For
every point the stage records ``nP``, ``nS`` and ``nStations`` (stations with any pick). With
``sweep.scoreMode`` ``all`` (every point) or ``coordinate`` (a coordinate descent from
``baseline.chosen``), points are also scored through H2's pipeline exactly as H4's VAL-01
scores a ``BaselineRow`` (``hq.baseline.score``: the run's own tier bars, profile ``full``), and
the PhaseNet ``picks.parquet`` of the same stations and window is scored once the same way as
the reference. Scoring needs the run's ``ProcessingRun.tiering``, which stage ``tier`` writes
after this stage in pipeline order: it is for a rerun of this stage on a run that went through
``tier``, and without those bars the stage fails before it picks anything. While H2's modules
are not importable the stage logs a WARNING and writes every score as null. docs/02's
``SweepPoint`` has non-null ints, so the file is written with model name ``BaselineSweep``:
SweepPoint's columns (``params`` as JSON text, the io flattening rule for dicts), the rest of the
point's ``BaselineRow`` (``tierB``, ``tierC``, ``medianRmsS``, ``medianStations``; every score
null where a point was not scored) and ``nP``, ``nS``, ``nStations``. The reference row, the
best Tier A point, the search path and the rerun notes go to ``baseline_reference.json`` and
into ``ctx.record`` (``sweepScoring``); an unscored run removes a stale
``baseline_reference.json``. The stage logs the best Tier A point (ties: ``baseline.chosen``,
then grid order) and warns while it differs from ``baseline.chosen``. Any H2 error stops the
stage (``picks_stalta.parquet`` is already written).

Failures. A station with nothing cached (``CacheMissError`` on its first read) has no picks and
the reason is reported. Any other error for a station (conflicting overlapping pieces, a rate its
profile rejects, an ambiguous station id) stops the stage with the station named: those are data
or config faults PhaseNet would hit too, and the fix belongs upstream.

Outputs are sorted by ``(t, stationId, phase)``; identical config and cache give identical files.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from obspy import Trace, UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from scipy import signal as sps

from hq.baseline.score import (
    REFERENCE_FILE,
    Coords,
    Runner,
    ScoreMode,
    ScorePlan,
    SweepScorer,
    SweepScores,
    TieringProvider,
    make_runner,
    prepare_scoring,
    reference_picks,
    score_sweep,
    scoring_record,
)
from hq.config.run import RunSection
from hq.config.signal import (
    BaselineBandpass,
    BaselineConfig,
    BaselinePhase,
    PreprocessConfig,
    SignalConfig,
)
from hq.ingest.cache import CacheMissError
from hq.preprocess.chunks import (
    NEAR_GAP_EDGE,
    ChunkStats,
    ModelChunk,
    ReadWindow,
    iter_model_chunks,
    pick_is_kept,
)

if TYPE_CHECKING:
    from hq_contracts.models import BaselineRow

log = logging.getLogger(__name__)

STAGE = "baseline"  # registry name, ctx.record() key and the key params nest under
PICKER = "stalta"  # docs/02 Pick.picker for this baseline: a contract value, not a knob
PICKS_FILE = "picks_stalta.parquet"
SWEEP_FILE = "baseline_sweep.parquet"
SWEEP_MODEL = "BaselineSweep"  # SweepPoint's ints can't be null (see module docstring)
# SweepPoint's scores, then the rest of the point's BaselineRow; all null when not scored.
SWEEP_SCORE_DTYPES: dict[str, str] = {
    "candidates": "Int64",
    "recoveredPublic": "Int64",
    "tierA": "Int64",
    "tierB": "Int64",
    "tierC": "Int64",
    "medianRmsS": "float64",
    "medianStations": "float64",
}
SWEEP_SCORE_COLUMNS = tuple(SWEEP_SCORE_DTYPES)
SWEEP_COLUMNS = ("params", *SWEEP_SCORE_COLUMNS, "nP", "nS", "nStations")
SCORE_MODES: tuple[str, ...] = get_args(ScoreMode)
PHASES = ("P", "S")

_STATION_COLUMNS = ("id", "channels", "preprocessProfile", "usedInRun")

FloatArray = npt.NDArray[np.float64]
Pair = tuple[float, float]  # trigger (on, off)
Triggers = tuple[FloatArray, FloatArray]  # (onset, end) real times, sorted by onset


class StageContext(Protocol):
    """The parts of H4's ``RunContext`` (docs/02 section 4) this stage uses; scoring the sweep
    also reads ``read_run().tiering`` unless ``run_baseline`` gets ``tiering=``."""

    @property
    def run_id(self) -> str: ...

    @property
    def cache_dir(self) -> Path: ...

    @property
    def config(self) -> Any: ...  # RunConfig: .run, .signal, .seismology

    def path(self, name: str) -> Path: ...

    def record(
        self,
        stage: str,
        *,
        runtime_s: float,
        counts: dict[str, int],
        params: dict | None = None,
    ) -> None: ...


# --- thresholds -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GridPoint:
    """One set of trigger thresholds."""

    pOn: float
    pOff: float
    sOn: float
    sOff: float

    @property
    def p_pair(self) -> Pair:
        return (self.pOn, self.pOff)

    @property
    def s_pair(self) -> Pair:
        return (self.sOn, self.sOff)

    def params(self) -> dict[str, float]:
        return {"pOn": self.pOn, "pOff": self.pOff, "sOn": self.sOn, "sOff": self.sOff}

    def coords(self) -> Coords:
        """(pOn, sOn, off): the axes the scoring search moves along (off serves both phases)."""
        return (self.pOn, self.sOn, self.pOff)


def sweep_grid(cfg: BaselineConfig) -> list[GridPoint]:
    """``sweep.pOn x sweep.sOn x sweep.offLevels``, in that nesting order; off serves both phases."""
    sw = cfg.sweep
    grid = itertools.product(sw.pOn, sw.sOn, sw.offLevels)
    return [GridPoint(p, off, s, off) for p, s, off in grid]


def chosen_point(cfg: BaselineConfig) -> GridPoint:
    c = cfg.chosen
    return GridPoint(c.pOn, c.pOff, c.sOn, c.sOff)


def gap_edge_s(signal: SignalConfig) -> float:
    """The gap-edge distance both full-window pickers pass to ``pick_is_kept``."""
    return signal.picker.gapEdgeS


def real_nyquist_hz(pre: PreprocessConfig, profile: str) -> float:
    """Lowest real Nyquist a segment of ``profile`` can have after ``for_picking``.

    Every profile outputs ``targetRateHz`` of model time; ``stretch`` only relabels, so its real
    rate is the input rate (at least ``minRateHz``), the others resample to ``targetRateHz``.
    """
    prof = pre.profiles[profile]
    real_rate = prof.minRateHz if prof.method == "stretch" else pre.targetRateHz
    return real_rate / 2.0


def check_config(signal: SignalConfig) -> None:
    """Cross-section checks the per-section validators can't make (run at stage start)."""
    b = signal.baseline
    pre = signal.preprocess
    model = set(pre.modelComponents)
    for name, phase in (("p", b.p), ("s", b.s)):
        extra = set(phase.components) - model
        if extra:
            raise ValueError(
                f"baseline.{name}.components {sorted(extra)} are not model components "
                f"{pre.modelComponents!r}"
            )
    missing = sorted(set(pre.profiles) - set(b.prefilter))
    unknown = sorted(set(b.prefilter) - set(pre.profiles))
    if missing or unknown:
        raise ValueError(
            "baseline.prefilter needs one band per preprocess profile: "
            f"missing {missing}, not a profile {unknown}"
        )
    for profile, band in b.prefilter.items():
        nyquist = real_nyquist_hz(pre, profile)
        if band.highHz >= nyquist:
            raise ValueError(
                f"baseline.prefilter.{profile}.highHz {band.highHz} reaches the real Nyquist "
                f"{nyquist} Hz of profile {profile}"
            )
    overlap = pre.chunks.overlapS
    # A keep interval must start after every warm-up, and an S just inside it needs its P (up to
    # maxSMinusPS earlier, in the overlap) to be pickable too: past the P warm-up.
    need = max(b.s.warmupS, b.p.warmupS + b.maxSMinusPS)
    if overlap < need:
        raise ValueError(
            f"preprocess.chunks.overlapS {overlap} s is shorter than max(s.warmupS, p.warmupS + "
            f"maxSMinusPS) = {need} s: picks near a keep start would depend on the chunking"
        )
    # ... and the recursive LTA must have forgotten its zero start there.
    settle = max(b.settleLtaMultiple * b.s.ltaS, b.settleLtaMultiple * b.p.ltaS + b.maxSMinusPS)
    if overlap < settle:
        raise ValueError(
            f"preprocess.chunks.overlapS {overlap} s is shorter than settleLtaMultiple x ltaS "
            f"(+ maxSMinusPS for P) = {settle} s: the LTA at a keep start would depend on the "
            "chunking"
        )
    gap = gap_edge_s(signal)
    if gap >= overlap:
        raise ValueError(
            f"picker.gapEdgeS {gap} s must be below preprocess.chunks.overlapS {overlap} s: a data "
            "edge that close to a keep interval must lie inside the chunk's read span"
        )


# --- characteristic function and onsets -------------------------------------------------------------


@dataclass(frozen=True)
class SegmentCF:
    """Recursive STA/LTA of one segment; triggers starting before ``warmup`` samples are dropped."""

    cf: FloatArray
    warmup: int


def characteristic_function(
    tr: Trace,
    phase: BaselinePhase,
    prefilter: BaselineBandpass,
    factor: float,
    min_margin_s: float,
) -> SegmentCF | None:
    """STA/LTA of one segment, or ``None`` when it is shorter than ``warmupS + margin``.

    ``factor`` is the chunk's ``TimeMap.factor``: the segment's real sample rate is its model
    rate times ``factor``, and every window is converted to samples at that real rate.
    """
    fs_real = float(tr.stats.sampling_rate) * factor
    if tr.stats.npts / fs_real < phase.warmupS + min_margin_s:
        return None
    if prefilter.highHz >= fs_real / 2.0:
        raise ValueError(
            f"{tr.id}: prefilter highHz {prefilter.highHz} reaches Nyquist of {fs_real} Hz"
        )
    sos = sps.butter(
        prefilter.corners,
        [prefilter.lowHz, prefilter.highHz],
        btype="bandpass",
        fs=fs_real,
        output="sos",
    )
    data = sps.sosfilt(sos, np.asarray(tr.data, dtype=np.float64))
    nsta = round(phase.staS * fs_real)
    nlta = round(phase.ltaS * fs_real)
    if nsta < 1 or nlta <= nsta:
        raise ValueError(f"{tr.id}: STA/LTA windows {nsta}/{nlta} samples at {fs_real} Hz")
    cf: FloatArray = recursive_sta_lta(data, nsta, nlta)
    return SegmentCF(cf=cf, warmup=min(round(phase.warmupS * fs_real), cf.size))


def trigger_samples(cf: FloatArray, on: float, off: float) -> npt.NDArray[np.int64]:
    """``trigger_onset`` as an ``(n, 2)`` array of (first, last) sample indices."""
    return np.asarray(trigger_onset(cf, on, off), dtype=np.int64).reshape(-1, 2)


@dataclass(frozen=True)
class ChunkOnsets:
    """Triggers of one chunk in REAL time, per threshold pair.

    ``p``: vertical onsets, sorted. ``h``: per horizontal component, in ``s.components`` order,
    that component's triggers as (onset, end) arrays sorted by onset (empty when it has none).
    ``warmupP`` / ``warmupS``: onsets at the chosen thresholds that started inside a warm-up
    span and were dropped (vertical / horizontal).
    """

    p: dict[Pair, FloatArray]
    h: dict[Pair, tuple[Triggers, ...]]
    warmupP: FloatArray
    warmupS: FloatArray


def chunk_onsets(
    chunk: ModelChunk,
    cfg: BaselineConfig,
    prefilter: BaselineBandpass,
    p_pairs: Sequence[Pair],
    s_pairs: Sequence[Pair],
    chosen: GridPoint,
    counts: Counter[str],
) -> ChunkOnsets:
    """Compute each segment's characteristic function once and trigger it at every pair."""
    ons: dict[str, dict[Pair, list[FloatArray]]] = {
        "P": {pair: [] for pair in p_pairs},
        "S": {pair: [] for pair in s_pairs},
    }
    # horizontal triggers per (pair, component): onsets and ends
    h_on: dict[tuple[Pair, str], list[FloatArray]] = {}
    h_end: dict[tuple[Pair, str], list[FloatArray]] = {}
    warm: dict[str, list[FloatArray]] = {"P": [], "S": []}
    chosen_pair = {"P": chosen.p_pair, "S": chosen.s_pair}
    to_real = chunk.timemap.to_real
    for tr in chunk.stream:
        comp = tr.stats.channel[-1:]
        if comp in cfg.p.components:
            phase, phase_cfg = "P", cfg.p
        elif comp in cfg.s.components:
            phase, phase_cfg = "S", cfg.s
        else:
            counts["segmentsOtherComponent"] += 1
            continue
        seg = characteristic_function(
            tr, phase_cfg, prefilter, chunk.timemap.factor, cfg.minSegmentMarginS
        )
        if seg is None:
            counts[f"segmentsSkippedShort{phase}"] += 1
            continue
        counts[f"segments{phase}"] += 1
        start, delta = tr.stats.starttime.timestamp, float(tr.stats.delta)
        for pair in ons[phase]:
            trig = trigger_samples(seg.cf, *pair)
            late = trig[:, 0] >= seg.warmup
            if pair == chosen_pair[phase] and not late.all():
                warm[phase].append(to_real(start + trig[~late, 0] * delta))
            trig = trig[late]
            if trig.size == 0:
                continue
            if phase == "P":
                ons["P"][pair].append(to_real(start + trig[:, 0] * delta))
            else:
                h_on.setdefault((pair, comp), []).append(to_real(start + trig[:, 0] * delta))
                h_end.setdefault((pair, comp), []).append(to_real(start + trig[:, 1] * delta))
    h: dict[Pair, tuple[Triggers, ...]] = {}
    for pair in s_pairs:
        per_comp: list[Triggers] = []
        for comp in cfg.s.components:
            on_t, end_t = _cat(h_on.get((pair, comp), [])), _cat(h_end.get((pair, comp), []))
            order = np.argsort(on_t, kind="stable")
            per_comp.append((on_t[order], end_t[order]))
        h[pair] = tuple(per_comp)
    return ChunkOnsets(
        p={pair: np.sort(_cat(found)) for pair, found in ons["P"].items()},
        h=h,
        warmupP=np.sort(_cat(warm["P"])),
        warmupS=np.sort(_cat(warm["S"])),
    )


def _cat(parts: list[FloatArray]) -> FloatArray:
    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(parts).astype(np.float64, copy=False)


def _sorted(parts: list[FloatArray]) -> FloatArray:
    return np.sort(_cat(parts))


def edge_mask(t: FloatArray, edges: Sequence[float], gap_edge_s: float) -> npt.NDArray[np.bool_]:
    """Vectorised ``hq.preprocess.chunks.near_data_edge``."""
    if not edges or t.size == 0:
        return np.zeros(t.shape, dtype=bool)
    e = np.asarray(edges, dtype=np.float64)
    i = np.searchsorted(e, t, side="left")
    left = np.where(i > 0, e[np.maximum(i - 1, 0)], -np.inf)
    right = np.where(i < e.size, e[np.minimum(i, e.size - 1)], np.inf)
    near: npt.NDArray[np.bool_] = (t - left <= gap_edge_s) | (right - t <= gap_edge_s)
    return near


def keep_masks(
    t: FloatArray, chunk: ModelChunk, gap_edge_s: float
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Vectorised ``pick_is_kept``: (kept, dropped near a gap edge); the rest is outside keep."""
    in_keep = (t >= chunk.keep[0]) & (t < chunk.keep[1])
    near = edge_mask(t, chunk.dataEdges, gap_edge_s)
    return in_keep & ~near, in_keep & near


def s_on_component(
    p: FloatArray, h: Triggers, p_tol: float, min_s: float, max_s: float
) -> FloatArray:
    """Per P (aligned with ``p``), the S onset on ONE horizontal component, or NaN.

    ``h`` is that component's triggers as (onset, end), sorted by onset. A trigger that is on at
    any time within ``p_tol`` of the P onset is the P's own energy on this component: it is never
    the S, and the search starts where it ends. The S is the earliest other trigger starting in
    ``[p + min_s, p + max_s]``.
    """
    h_on, h_end = h
    out = np.full(p.shape, np.nan)
    if p.size == 0 or h_on.size == 0:
        return out
    # Triggers starting at or before p + p_tol, and the latest end among them. If that end
    # reaches p - p_tol, a trigger was on within p_tol of the P.
    n_upto = np.searchsorted(h_on, p + p_tol, side="right")
    last_end = np.maximum.accumulate(h_end)
    prev_end = np.where(n_upto > 0, last_end[np.maximum(n_upto - 1, 0)], -np.inf)
    busy = prev_end >= p - p_tol
    start = np.where(busy, np.maximum(p + min_s, prev_end), p + min_s)
    i = np.maximum(np.searchsorted(h_on, start, side="left"), n_upto)
    ok = i < h_on.size
    cand = np.full(p.shape, np.inf)
    cand[ok] = h_on[i[ok]]
    hit = cand <= p + max_s
    out[hit] = cand[hit]
    return out


def s_after_p(
    p: FloatArray,
    horizontals: Sequence[Triggers],
    p_tol: float,
    min_s: float,
    max_s: float,
) -> FloatArray:
    """For each P, the earliest S over the horizontal components; unique and sorted.

    Each component is searched on its own (``s_on_component``): a P trigger still on one
    horizontal does not hide an S onset on the other.
    """
    if p.size == 0 or not horizontals:
        return np.empty(0, dtype=np.float64)
    per = np.vstack([s_on_component(p, h, p_tol, min_s, max_s) for h in horizontals])
    best = np.where(np.isnan(per), np.inf, per).min(axis=0)
    return np.unique(best[np.isfinite(best)])


def s_picks(p: FloatArray, h: Sequence[Triggers], cfg: BaselineConfig) -> FloatArray:
    return s_after_p(p, h, cfg.pHorizontalTolS, cfg.minSMinusPS, cfg.maxSMinusPS)


def phase_picks(
    onsets: ChunkOnsets, point: GridPoint, chunk: ModelChunk, cfg: BaselineConfig, gap: float
) -> tuple[FloatArray, FloatArray]:
    """P and S candidates of one chunk at one threshold point, before the keep rule."""
    p = onsets.p[point.p_pair]
    p_valid = p[~edge_mask(p, chunk.dataEdges, gap)]
    return p, s_picks(p_valid, onsets.h[point.s_pair], cfg)


# --- one station -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StationRow:
    id: str
    channels: tuple[str, ...]
    profile: str


@dataclass
class StationResult:
    stationId: str
    profile: str
    picks: list[tuple[float, str]] = field(default_factory=list)  # chosen thresholds: (t, phase)
    sweepP: list[FloatArray] = field(default_factory=list)  # kept P times per grid point
    sweepS: list[FloatArray] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)
    chunks: ChunkStats = field(default_factory=ChunkStats)
    notCached: str | None = None
    runtimeS: float = 0.0

    def n(self, phase: str) -> int:
        return sum(1 for _, ph in self.picks if ph == phase)

    @property
    def lost(self) -> int:
        """Chosen-threshold candidates dropped by the edge, warm-up and P-near-edge rules."""
        c = self.counts
        return sum(
            c[k]
            for k in (
                "droppedNearGapEdgeP",
                "droppedNearGapEdgeS",
                "suppressedWarmupP",
                "suppressedWarmupS",
                "sLostPNearGapEdge",
            )
        )

    @property
    def reason(self) -> str:
        """Why a station has no picks at the chosen thresholds ("" when it has some)."""
        if self.picks:
            return ""
        if self.notCached is not None:
            return "not cached"
        if self.chunks.yielded == 0:
            c = self.chunks
            parts = [f"{c.empty} of {c.planned} chunks empty"] if c.empty else []
            if c.noSegments:
                parts.append(f"{c.noSegments} with every segment too short for for_picking")
            return "no usable data: " + ", ".join(parts)
        if self.counts["segmentsP"] + self.counts["segmentsS"] == 0:
            return "every segment shorter than warmupS + minSegmentMarginS"
        if self.lost:
            return "every trigger near a data edge or inside a warm-up span"
        return "no trigger above the chosen thresholds"


def pick_station(
    station: StationRow,
    t0: float,
    t1: float,
    cfg: SignalConfig,
    grid: Sequence[GridPoint],
    *,
    cache_dir: Path,
    read_window: ReadWindow | None = None,
) -> StationResult:
    """Chosen-threshold picks and per-grid-point kept pick times of one station over [t0, t1)."""
    began = perf_counter()
    b = cfg.baseline
    gap = gap_edge_s(cfg)
    prefilter = b.prefilter.get(station.profile)
    if prefilter is None:
        raise ValueError(f"{station.id}: no baseline.prefilter for profile {station.profile!r}")
    chosen = chosen_point(b)
    p_pairs = sorted({g.p_pair for g in grid} | {chosen.p_pair})
    s_pairs = sorted({g.s_pair for g in grid} | {chosen.s_pair})
    res = StationResult(stationId=station.id, profile=station.profile)
    sweep_p: list[list[FloatArray]] = [[] for _ in grid]
    sweep_s: list[list[FloatArray]] = [[] for _ in grid]
    try:
        for chunk in iter_model_chunks(
            station.id,
            station.channels,
            station.profile,
            t0,
            t1,
            cfg,
            cache_dir=cache_dir,
            read_window=read_window,
            stats=res.chunks,
        ):
            onsets = chunk_onsets(chunk, b, prefilter, p_pairs, s_pairs, chosen, res.counts)
            # Published picks: every candidate goes through the shared keep rule.
            p, s = phase_picks(onsets, chosen, chunk, b, gap)
            for phase, times in zip(PHASES, (p, s), strict=True):
                for t in times.tolist():
                    kept, why = pick_is_kept(t, chunk, gap)
                    if kept:
                        res.picks.append((t, phase))
                    elif why == NEAR_GAP_EDGE:
                        res.counts[f"droppedNearGapEdge{phase}"] += 1
            # What the other rules cost at the chosen thresholds, counted like drops: kept S
            # candidates whose only P was near a data edge, and onsets inside warm-up spans.
            s_all = s_picks(p, onsets.h[chosen.s_pair], b)
            lost = np.setdiff1d(s_all, s)
            res.counts["sLostPNearGapEdge"] += int(keep_masks(lost, chunk, gap)[0].sum())
            for phase, t_warm in zip(PHASES, (onsets.warmupP, onsets.warmupS), strict=True):
                n_warm = int(keep_masks(t_warm, chunk, gap)[0].sum())
                res.counts[f"suppressedWarmup{phase}"] += n_warm
            # Sweep counts: the same rule, vectorised (tests check it matches pick_is_kept).
            for k, point in enumerate(grid):
                p_k, s_k = phase_picks(onsets, point, chunk, b, gap)
                sweep_p[k].append(p_k[keep_masks(p_k, chunk, gap)[0]])
                sweep_s[k].append(s_k[keep_masks(s_k, chunk, gap)[0]])
    except CacheMissError as exc:  # nothing at all cached for the station: raised on the 1st read
        res.notCached = str(exc)
        log.warning("baseline %s: not cached, no picks (%s)", station.id, exc)
    except Exception as exc:
        # Deliberately not handled (see the module docstring); only name the station.
        exc.add_note(f"baseline stage, station {station.id} (profile {station.profile})")
        raise
    res.sweepP = [_sorted(parts) for parts in sweep_p]
    res.sweepS = [_sorted(parts) for parts in sweep_s]
    res.runtimeS = perf_counter() - began
    log.info(
        "baseline %s (%s): P=%d S=%d dropped_near_gap_edge P=%d S=%d s_lost_p_near_edge=%d "
        "suppressed_warmup P=%d S=%d segments P=%d S=%d skipped_short P=%d S=%d chunks %d/%d "
        "runtime_s=%.2f",
        station.id,
        station.profile,
        res.n("P"),
        res.n("S"),
        res.counts["droppedNearGapEdgeP"],
        res.counts["droppedNearGapEdgeS"],
        res.counts["sLostPNearGapEdge"],
        res.counts["suppressedWarmupP"],
        res.counts["suppressedWarmupS"],
        res.counts["segmentsP"],
        res.counts["segmentsS"],
        res.counts["segmentsSkippedShortP"],
        res.counts["segmentsSkippedShortS"],
        res.chunks.yielded,
        res.chunks.planned,
        res.runtimeS,
    )
    return res


# --- tables ------------------------------------------------------------------------------------------


def pick_id(station_id: str, phase: str, t: float) -> str:
    return f"{PICKER}:{station_id}:{phase}:{t:.3f}"


def picks_frame(rows: Sequence[tuple[float, str, str]], prob: float) -> pd.DataFrame:
    """``(t, stationId, phase)`` rows -> a docs/02 ``Pick`` table, sorted, ids unique."""
    from hq_contracts.io import to_frame
    from hq_contracts.models import Pick

    ordered = sorted(rows)
    models = [
        Pick(
            id=pick_id(sid, phase, t),
            stationId=sid,
            phase=phase,
            t=t,
            prob=prob,
            picker=PICKER,
            eventId=None,
            residualS=None,
            weight=None,
        )
        for t, sid, phase in ordered
    ]
    ids = [m.id for m in models]
    if len(set(ids)) != len(ids):
        dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
        raise ValueError(f"duplicate STA/LTA pick ids: {dupes[:5]}")
    # An empty frame would otherwise carry object columns; times stay float64 (docs/02 section 2).
    return to_frame(models, Pick).astype({"t": "float64", "prob": "float64"})


def chosen_rows(results: Sequence[StationResult]) -> list[tuple[float, str, str]]:
    return [(t, r.stationId, phase) for r in results for t, phase in r.picks]


def sweep_rows(results: Sequence[StationResult], k: int) -> list[tuple[float, str, str]]:
    rows: list[tuple[float, str, str]] = []
    for r in results:
        rows.extend((t, r.stationId, "P") for t in r.sweepP[k].tolist())
        rows.extend((t, r.stationId, "S") for t in r.sweepS[k].tolist())
    return rows


def score_cells(row: BaselineRow | None) -> dict[str, Any]:
    """A point's ``BaselineRow`` as sweep columns (every one null when not scored)."""
    if row is None:
        return dict.fromkeys(SWEEP_SCORE_COLUMNS)
    return {
        "candidates": row.candidates,
        "recoveredPublic": row.recoveredPublic,
        "tierA": row.tiers.A,
        "tierB": row.tiers.B,
        "tierC": row.tiers.C,
        "medianRmsS": row.medianRmsS,
        "medianStations": row.medianStations,
    }


def sweep_frame(
    results: Sequence[StationResult],
    grid: Sequence[GridPoint],
    rows: Sequence[BaselineRow | None],
) -> pd.DataFrame:
    """``baseline_sweep.parquet``: one row per grid point, in grid order."""
    out: list[dict[str, Any]] = []
    for k, point in enumerate(grid):
        out.append(
            {
                "params": json.dumps(point.params(), sort_keys=True),
                **score_cells(rows[k]),
                "nP": sum(r.sweepP[k].size for r in results),
                "nS": sum(r.sweepS[k].size for r in results),
                "nStations": sum(1 for r in results if r.sweepP[k].size + r.sweepS[k].size),
            }
        )
    df = pd.DataFrame(out, columns=list(SWEEP_COLUMNS))
    return df.astype({**SWEEP_SCORE_DTYPES, "nP": "int64", "nS": "int64", "nStations": "int64"})


def score_points(
    plan: ScorePlan,
    results: Sequence[StationResult],
    grid: Sequence[GridPoint],
    cfg: BaselineConfig,
    station_ids: Sequence[str],
    window: tuple[float, float],
    runner: Runner | None,
) -> SweepScores:
    """Score the sweep (``hq.baseline.score``); a point's picks are built only when it is sent."""
    scorer = SweepScorer(
        shared=plan.shared,
        runner=make_runner(cfg.sweep.scoreWorkers) if runner is None else runner,
        coords=[g.coords() for g in grid],
        params=[g.params() for g in grid],
        point_picks=lambda k: picks_frame(sweep_rows(results, k), cfg.prob),
        reference_picks=reference_picks(plan.phasenet_picks, station_ids, *window),
    )
    chosen = list(grid).index(chosen_point(cfg))
    return score_sweep(plan.mode, chosen, cfg.sweep.maxPasses, scorer)


def report_best(grid: Sequence[GridPoint], scores: SweepScores | None) -> None:
    if scores is None:
        return
    best, chosen = grid[scores.best], grid[scores.chosen]
    best_row, chosen_row = scores.best_row, scores.rows[scores.chosen]
    log.info(
        "baseline sweep (%s): most Tier A events (%d) at %s over %d scored point(s) in %.1f s; "
        "PhaseNet reference: %d Tier A of %d candidates",
        scores.mode,
        best_row.tiers.A,
        best.params(),
        scores.scored,
        scores.runtimeS,
        scores.reference.row.tiers.A,
        scores.reference.row.candidates,
    )
    if best != chosen:
        log.warning(
            "baseline sweep: baseline.chosen %s (Tier A %s) is not the best Tier A point %s; copy "
            "it into signal.yaml if the comparison should use the baseline's best thresholds",
            chosen.params(),
            "not scored" if chosen_row is None else chosen_row.tiers.A,
            best.params(),
        )


def write_reference(ctx: StageContext, record: Mapping[str, Any] | None) -> None:
    """``baseline_reference.json`` when scored; else remove a stale one (it would describe
    another sweep than the ``baseline_sweep.parquet`` written now)."""
    path = Path(ctx.path(REFERENCE_FILE))
    if record is not None:
        from hq.runs import write_text_atomic

        write_text_atomic(path, json.dumps(record, indent=2) + "\n")
        return
    if path.is_file():
        path.unlink()
        log.warning(
            "baseline: removed %s from an earlier scored sweep; this sweep is unscored", path
        )


def station_rows(stations_df: pd.DataFrame, station_ids: Sequence[str] | None) -> list[StationRow]:
    """``usedInRun`` stations (optionally a named subset), sorted by id."""
    missing = [c for c in _STATION_COLUMNS if c not in stations_df.columns]
    if missing:
        raise ValueError(f"stations table is missing columns {missing}")
    used_mask = stations_df["usedInRun"].map(lambda v: isinstance(v, bool | np.bool_) and bool(v))
    used = stations_df[used_mask.astype(bool)]
    log.info("baseline: %d of %d stations have usedInRun = true", len(used), len(stations_df))
    dupes = sorted(set(used.loc[used["id"].duplicated(), "id"]))
    if dupes:
        raise ValueError(f"stations table repeats station ids {dupes}")
    if station_ids is not None:
        unknown = sorted(set(station_ids) - set(used["id"]))
        if unknown:
            raise ValueError(f"--stations names ids that are not usedInRun stations: {unknown}")
        used = used[used["id"].isin(station_ids)]
    rows: list[StationRow] = []
    for row in used.sort_values("id", kind="mergesort").itertuples(index=False):
        channels = tuple(str(c) for c in row.channels)
        if not channels:
            raise ValueError(f"station {row.id} has no channels in the stations table")
        rows.append(
            StationRow(id=str(row.id), channels=channels, profile=str(row.preprocessProfile))
        )
    return rows


def format_station_table(results: Sequence[StationResult]) -> str:
    head = (
        f"{'station':<10} {'profile':<12} {'chunks':>7} {'P':>6} {'S':>6} {'dropP':>5} "
        f"{'dropS':>5} {'sNoP':>5} {'warmP':>5} {'warmS':>5} {'short':>5} {'sec':>6}  "
        "reason for no picks"
    )
    title = (
        "STA/LTA baseline at the chosen thresholds (drop = within picker.gapEdgeS of a data "
        "edge; sNoP = S lost because its P was near an edge; warm = onset inside a warm-up span; "
        "short = segments skipped as too short)"
    )
    lines = [title, head, "-" * len(head)]
    for r in results:
        c = r.counts
        short = c["segmentsSkippedShortP"] + c["segmentsSkippedShortS"]
        lines.append(
            f"{r.stationId:<10} {r.profile:<12} {r.chunks.yielded:>3}/{r.chunks.planned:<3} "
            f"{r.n('P'):>6} {r.n('S'):>6} {c['droppedNearGapEdgeP']:>5} "
            f"{c['droppedNearGapEdgeS']:>5} {c['sLostPNearGapEdge']:>5} "
            f"{c['suppressedWarmupP']:>5} {c['suppressedWarmupS']:>5} {short:>5} "
            f"{r.runtimeS:>6.1f}  {r.reason}"
        )
    return "\n".join(lines)


def format_sweep_table(sweep_df: pd.DataFrame) -> str:
    ints = ("candidates", "recoveredPublic", "tierA", "tierB", "tierC")
    head = (
        f"{'params':<50} {'nP':>7} {'nS':>7} {'nSta':>5} {'cand':>5} {'recov':>5} {'A':>5} "
        f"{'B':>5} {'C':>5} {'rmsS':>6}"
    )
    lines = [head]
    for row in sweep_df.to_dict("records"):
        cells = ["null" if pd.isna(row[c]) else str(int(row[c])) for c in ints]
        rms = "null" if pd.isna(row["medianRmsS"]) else f"{row['medianRmsS']:.3f}"
        lines.append(
            f"{row['params']:<50} {row['nP']:>7} {row['nS']:>7} {row['nStations']:>5} "
            + " ".join(f"{c:>5}" for c in cells)
            + f" {rms:>6}"
        )
    return "\n".join(lines)


# --- stage -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResult:
    stations: list[StationResult]
    picks: pd.DataFrame
    sweep: pd.DataFrame
    counts: dict[str, int]
    scores: SweepScores | None = None


def _map_stations(
    rows: Sequence[StationRow], work: Callable[[StationRow], StationResult], workers: int
) -> list[StationResult]:
    if workers == 1 or len(rows) <= 1:
        return [work(r) for r in rows]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="baseline") as pool:
        return list(pool.map(work, rows))  # input order: deterministic


_TOTALLED = (
    "droppedNearGapEdgeP",
    "droppedNearGapEdgeS",
    "sLostPNearGapEdge",
    "suppressedWarmupP",
    "suppressedWarmupS",
    "segmentsP",
    "segmentsS",
    "segmentsSkippedShortP",
    "segmentsSkippedShortS",
    "segmentsOtherComponent",
)


def record_params(
    signal: SignalConfig,
    grid: Sequence[GridPoint],
    scores: SweepScores | None,
    scoring: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """``ctx.record`` params, nested under the stage key (H4, REQ-H1-2: keys never collide).

    ``baseline.*`` from signal.yaml, plus what else decides the picks: the shared gap-edge
    distance and chunking; once the sweep is scored, the best Tier A point and ``sweepScoring``
    (the contents of ``baseline_reference.json``: mode, PhaseNet reference row, best and chosen
    rows, search path, rerun notes).
    """
    return {
        STAGE: {
            **signal.baseline.model_dump(mode="json"),
            "pickerGapEdgeS": gap_edge_s(signal),
            "preprocessChunks": signal.preprocess.chunks.model_dump(mode="json"),
            "sweepBestTierA": None if scores is None else grid[scores.best].params(),
            "sweepScoring": None if scoring is None else dict(scoring),
        }
    }


def run_baseline(
    ctx: StageContext,
    *,
    t0: float | None = None,
    t1: float | None = None,
    station_ids: Sequence[str] | None = None,
    read_window: ReadWindow | None = None,
    score_mode: ScoreMode | None = None,
    write_picks: bool = True,
    tiering: TieringProvider | None = None,
    runner: Runner | None = None,
) -> BaselineResult:
    """Pick every usedInRun station (or ``station_ids``) over the run window (or [t0, t1)).

    Writes ``picks_stalta.parquet`` (not with ``write_picks=False``: the CLI's
    ``--score-only``) and ``baseline_sweep.parquet``, scores the sweep in ``score_mode``
    (default ``baseline.sweep.scoreMode``) and writes ``baseline_reference.json`` when it does,
    prints the per-station and sweep tables and calls ``ctx.record("baseline", ...)``.
    ``tiering`` returns the run's ``ProcessingRun.tiering`` (default ``ctx.read_run().tiering``);
    ``runner`` scores the points (default: ``baseline.sweep.scoreWorkers`` processes).
    """
    from hq_contracts.io import read_table, write_table

    began = perf_counter()
    signal: SignalConfig = ctx.config.signal
    b = signal.baseline
    check_config(signal)
    mode: ScoreMode = b.sweep.scoreMode if score_mode is None else score_mode
    if mode not in SCORE_MODES:
        raise ValueError(f"score mode {mode!r} is not one of {SCORE_MODES}")
    if not write_picks and mode == "none":
        raise ValueError("score-only needs a score mode other than none: nothing would be written")
    run_section: RunSection = ctx.config.run
    t0 = run_section.window_start_s if t0 is None else t0
    t1 = run_section.window_end_s if t1 is None else t1
    stations_df = read_table(ctx.path("stations.parquet"))
    rows = station_rows(stations_df, station_ids)
    grid = sweep_grid(b)
    plan = prepare_scoring(ctx, mode, stations_df, tiering)  # a run that can't be scored fails now
    log.info(
        "baseline: %d stations over [%s, %s), %d sweep points (score mode %s, %d scoring "
        "worker(s)), %d picking worker(s), gapEdgeS %.2f s",
        len(rows),
        UTCDateTime(t0),
        UTCDateTime(t1),
        len(grid),
        mode if plan is not None else "none",
        b.sweep.scoreWorkers,
        b.maxWorkers,
        gap_edge_s(signal),
    )

    def work(row: StationRow) -> StationResult:
        return pick_station(
            row, t0, t1, signal, grid, cache_dir=ctx.cache_dir, read_window=read_window
        )

    results = _map_stations(rows, work, b.maxWorkers)
    picks = picks_frame(chosen_rows(results), b.prob)
    if write_picks:
        write_table(picks, ctx.path(PICKS_FILE), "Pick")
    else:
        log.info("baseline: score only; %s is left as it is", ctx.path(PICKS_FILE))
    scores = (
        None
        if plan is None
        else score_points(plan, results, grid, b, [r.id for r in rows], (t0, t1), runner)
    )
    sweep = sweep_frame(results, grid, [None] * len(grid) if scores is None else scores.rows)
    write_table(sweep, ctx.path(SWEEP_FILE), SWEEP_MODEL)
    scoring = (
        None
        if scores is None or plan is None
        else scoring_record(
            scores, [g.params() for g in grid], plan.shared.thresholds, b.sweep.scoreWorkers
        )
    )
    write_reference(ctx, scoring)
    report_best(grid, scores)

    chunk_totals = ChunkStats()
    totals: Counter[str] = Counter()
    for r in results:
        chunk_totals.add(r.chunks)
        totals.update(r.counts)
    counts = {
        "stations": len(results),
        "stationsWithPicks": sum(1 for r in results if r.picks),
        "stationsNotCached": sum(1 for r in results if r.notCached is not None),
        "nP": int((picks["phase"] == "P").sum()),
        "nS": int((picks["phase"] == "S").sum()),
        **{key: int(totals[key]) for key in _TOTALLED},
        **chunk_totals.as_counts(),
        "sweepPoints": len(grid),
        "sweepScored": 0 if scores is None else scores.scored,
    }
    if scores is not None:
        counts["sweepBestTierA"] = scores.best_row.tiers.A
        counts["phasenetReferenceTierA"] = scores.reference.row.tiers.A
    print(format_station_table(results))
    print(format_sweep_table(sweep))
    runtime_s = perf_counter() - began
    log.info(
        "baseline: %d P and %d S picks on %d of %d stations; %d near-gap-edge drops, %d S lost "
        "to a P near an edge, %d onsets inside warm-up spans; %d sweep points (%d scored); "
        "wrote %s%s%s in %.1f s",
        counts["nP"],
        counts["nS"],
        counts["stationsWithPicks"],
        counts["stations"],
        counts["droppedNearGapEdgeP"] + counts["droppedNearGapEdgeS"],
        counts["sLostPNearGapEdge"],
        counts["suppressedWarmupP"] + counts["suppressedWarmupS"],
        counts["sweepPoints"],
        counts["sweepScored"],
        f"{PICKS_FILE} and " if write_picks else "",
        SWEEP_FILE,
        f" and {REFERENCE_FILE}" if scoring is not None else "",
        runtime_s,
    )
    ctx.record(
        STAGE,
        runtime_s=runtime_s,
        counts=counts,
        params=record_params(signal, grid, scores, scoring),
    )
    return BaselineResult(stations=results, picks=picks, sweep=sweep, counts=counts, scores=scores)


def run(ctx: StageContext) -> None:
    """Stage ``baseline`` (docs/02 Stage API): every usedInRun station over the run window."""
    run_baseline(ctx)


# --- CLI ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _CliConfig:
    run: RunSection
    signal: SignalConfig


@dataclass(frozen=True)
class _CliContext:
    """Stand-in for ``RunContext`` on the command line; ``record`` only logs."""

    run_id: str
    run_dir: Path
    cache_dir: Path
    config: Any

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def read_run(self) -> Any:
        """The run's ``run.json`` (``ProcessingRun``): scoring reads its tier bars."""
        from hq.runs import read_run_json

        return read_run_json(self.run_dir)

    def record(
        self,
        stage: str,
        *,
        runtime_s: float,
        counts: dict[str, int],
        params: dict | None = None,
    ) -> None:
        log.info(
            "record %s (CLI, run.json not updated): runtime %.1f s, counts %s",
            stage,
            runtime_s,
            counts,
        )


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _cli_config(config_dir: Path) -> Any:
    """H4's full ``RunConfig`` when ``hq.config.load_config`` exists, else run + signal only."""
    load_config = getattr(importlib.import_module("hq.config"), "load_config", None)
    if load_config is not None:
        return load_config(config_dir)
    log.info("hq.config.load_config (RUN-01) not merged: loading run.yaml and signal.yaml only")
    return _CliConfig(
        run=RunSection.model_validate(_load_yaml(config_dir / "run.yaml")),
        signal=SignalConfig.model_validate(_load_yaml(config_dir / "signal.yaml")),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hq.baseline.run",
        description="STA/LTA baseline (SEIS-07) for a run directory, a station subset or a window.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="runs/<id> with stations.parquet; outputs go here",
    )
    parser.add_argument("--config-dir", type=Path, required=True, help="e.g. configs/showcase")
    parser.add_argument("--cache-dir", type=Path, required=True, help="cache root (holds mseed/)")
    parser.add_argument("--stations", help="comma-separated station ids (default: all usedInRun)")
    parser.add_argument("--start", help="ISO UTC start (default: windowStart from run.yaml)")
    parser.add_argument("--end", help="ISO UTC end, exclusive (default: windowEnd from run.yaml)")
    parser.add_argument(
        "--score-mode",
        choices=SCORE_MODES,
        help="override baseline.sweep.scoreMode (scoring needs the run's tier bars in run.json)",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help=f"recompute the triggers and score the sweep, but leave {PICKS_FILE} as it is",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
    )
    ctx = _CliContext(
        run_id=args.run_dir.name,
        run_dir=args.run_dir,
        cache_dir=args.cache_dir,
        config=_cli_config(args.config_dir),
    )
    mode = args.score_mode or ctx.config.signal.baseline.sweep.scoreMode
    if args.score_only and mode == "none":
        parser.error("--score-only needs a score mode other than none (--score-mode)")
    ids = [s.strip() for s in args.stations.split(",") if s.strip()] if args.stations else None
    run_baseline(
        ctx,
        t0=UTCDateTime(args.start).timestamp if args.start else None,
        t1=UTCDateTime(args.end).timestamp if args.end else None,
        station_ids=ids,
        score_mode=mode,
        write_picks=not args.score_only,
    )
    return 0


if __name__ == "__main__":
    # ``python -m`` runs a second copy of this module as __main__ (hq.baseline imported the real
    # one first, hence runpy's RuntimeWarning); delegate to the real module so there is one copy.
    from hq.baseline.run import main as _main

    sys.exit(_main())

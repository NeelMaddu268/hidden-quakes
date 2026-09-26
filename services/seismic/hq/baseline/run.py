"""STA/LTA baseline picker (SEIS-07): classical picks plus a trigger-threshold sweep.

Stage ``baseline`` (docs/01): cache + ``stations.parquet`` -> ``picks_stalta.parquet`` and
``baseline_sweep.parquet`` in the run directory.

Input. Every ``usedInRun`` station is read through ``hq.preprocess.chunks.iter_model_chunks``,
the same hour chunks and the same ``for_picking`` output PhaseNet (SEIS-06) picks on, and every
pick goes through the same ``pick_is_kept`` rule: outside the chunk's keep interval it belongs to
the neighbouring chunk; within ``baseline.gapEdgeS`` of a raw data edge it is dropped and counted.

Characteristic function. Per gap-separated segment: an optional causal Butterworth bandpass
(``baseline.prefilter``), then ObsPy's ``recursive_sta_lta``. P uses the vertical, S each
horizontal, each with its own STA/LTA windows. Windows are REAL seconds, so a time-stretched
``borehole-B`` segment gets the same physical windows (``TimeMap.factor`` x more model samples).
The recursive LTA starts from zero, so the first ``warmupS`` of every segment is zeroed (ObsPy
itself zeroes only the first LTA window). Segments shorter than ``ltaS + minSegmentMarginS`` are
skipped and counted. Each function is computed ONCE per segment and window setting; every
threshold of the sweep and the chosen thresholds then only re-run the cheap ``trigger_onset``.

Onsets. A pick is the first sample of a ``trigger_onset`` trigger, converted from model time to
real time with ``TimeMap.to_real`` (identity except ``borehole-B``). P: every vertical trigger.
S: for each P of the station (a P within ``gapEdgeS`` of a data edge does not count), the EARLIEST
trigger on either horizontal in ``[tP + minSMinusPS, tP + maxSMinusPS]``; earliest rather than
strongest because the first S energy is the onset and a later, stronger trigger is more often
coda. Two Ps choosing the same trigger give one S pick. Horizontal triggers with no P ahead of them
are not picks. Known limitation, as for any classical S picker: when P energy on a horizontal keeps
the trigger on until the S arrives, that S has no onset of its own and is missed.

Probability. STA/LTA has no calibrated confidence, so every pick gets ``prob = baseline.prob``
(1.0). How reliable the baseline is is controlled by the trigger thresholds alone, which the sweep
varies; prob 1.0 means an association probability threshold never drops a baseline pick, so the
baseline gets its best shot in the comparison.

Sweep. Grid ``sweep.pOn x sweep.sOn x sweep.offLevels`` (one off level for both phases). For
every point the stage records ``nP``, ``nS`` and ``nStations`` (stations with any pick). Tier A
and recovered-public counts need H2's ``associate``/``locate``/``match``/``assign_tiers``
(docs/02 section 5); until those are merged ``evaluate_with_h2`` raises
``H2PipelineMissingError``, the stage logs a WARNING and writes ``candidates``,
``recoveredPublic`` and ``tierA`` as null. docs/02's ``SweepPoint`` has non-null ints, so the
file is written with model name ``BaselineSweep``: SweepPoint's columns (``params`` as JSON text,
the io flattening rule for dicts) plus ``nP``, ``nS``, ``nStations``.

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
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from obspy import Trace, UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from scipy import signal as sps

from hq.config.run import RunSection
from hq.config.signal import BaselineBandpass, BaselineConfig, BaselinePhase, SignalConfig
from hq.ingest.cache import CacheMissError
from hq.preprocess.chunks import (
    NEAR_GAP_EDGE,
    ChunkStats,
    ModelChunk,
    ReadWindow,
    iter_model_chunks,
    pick_is_kept,
)

log = logging.getLogger(__name__)

STAGE = "baseline"  # registry name and ctx.record() key
PICKER = "stalta"  # docs/02 Pick.picker for this baseline: a contract value, not a knob
PICKS_FILE = "picks_stalta.parquet"
SWEEP_FILE = "baseline_sweep.parquet"
SWEEP_MODEL = "BaselineSweep"  # SweepPoint's ints can't be null (see module docstring)
SWEEP_SCORE_COLUMNS = ("candidates", "recoveredPublic", "tierA")
SWEEP_COLUMNS = ("params", *SWEEP_SCORE_COLUMNS, "nP", "nS", "nStations")
H2_MISSING_MESSAGE = "H2 pipeline not merged; sweep has pick counts only"
PHASES = ("P", "S")

# H2's library API (docs/02 section 5), imported lazily.
_H2_API = (
    ("hq.associate", "associate"),
    ("hq.locate", "locate"),
    ("hq.match", "match"),
    ("hq.tier", "assign_tiers"),
)
_STATION_COLUMNS = ("id", "channels", "preprocessProfile", "usedInRun")

FloatArray = npt.NDArray[np.float64]
Pair = tuple[float, float]  # trigger (on, off)


class StageContext(Protocol):
    """The parts of H4's ``RunContext`` (docs/02 section 4) this stage uses."""

    @property
    def cache_dir(self) -> Path: ...

    @property
    def config(self) -> Any: ...  # RunConfig: .run, .signal (and .seismology once H2 lands)

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


def sweep_grid(cfg: BaselineConfig) -> list[GridPoint]:
    """``sweep.pOn x sweep.sOn x sweep.offLevels``, in that nesting order; off serves both phases."""
    sw = cfg.sweep
    grid = itertools.product(sw.pOn, sw.sOn, sw.offLevels)
    return [GridPoint(p, off, s, off) for p, s, off in grid]


def chosen_point(cfg: BaselineConfig) -> GridPoint:
    c = cfg.chosen
    return GridPoint(c.pOn, c.pOff, c.sOn, c.sOff)


def check_config(signal: SignalConfig) -> None:
    """Cross-section checks the per-section validators can't make."""
    b = signal.baseline
    model = set(signal.preprocess.modelComponents)
    for name, phase in (("p", b.p), ("s", b.s)):
        extra = set(phase.components) - model
        if extra:
            raise ValueError(
                f"baseline.{name}.components {sorted(extra)} are not model components "
                f"{signal.preprocess.modelComponents!r}"
            )
    # A keep interval must start after every warm-up, and an S just inside it needs its P (up to
    # maxSMinusPS earlier, in the overlap) to be pickable too: past the P warm-up.
    overlap = signal.preprocess.chunks.overlapS
    need = max(b.s.warmupS, b.p.warmupS + b.maxSMinusPS)
    if overlap < need:
        raise ValueError(
            f"preprocess.chunks.overlapS {overlap} s is shorter than max(s.warmupS, p.warmupS + "
            f"maxSMinusPS) = {need} s: picks near a keep start would depend on the chunking"
        )
    if b.prefilter is not None and b.prefilter.highHz >= signal.preprocess.targetRateHz / 2.0:
        raise ValueError(
            f"baseline.prefilter.highHz {b.prefilter.highHz} reaches the model Nyquist "
            f"{signal.preprocess.targetRateHz / 2.0} Hz"
        )


# --- characteristic function and onsets -------------------------------------------------------------


def characteristic_function(
    tr: Trace,
    phase: BaselinePhase,
    prefilter: BaselineBandpass | None,
    factor: float,
    min_margin_s: float,
) -> FloatArray | None:
    """Recursive STA/LTA of one segment, or ``None`` when it is shorter than ``ltaS + margin``.

    ``factor`` is the chunk's ``TimeMap.factor``: the segment's real sample rate is its model
    rate times ``factor``, and every window is converted to samples at that real rate.
    """
    fs_real = float(tr.stats.sampling_rate) * factor
    if tr.stats.npts / fs_real < phase.ltaS + min_margin_s:
        return None
    data = np.asarray(tr.data, dtype=np.float64)
    if prefilter is not None:
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
        data = sps.sosfilt(sos, data)
    nsta = round(phase.staS * fs_real)
    nlta = round(phase.ltaS * fs_real)
    if nsta < 1 or nlta <= nsta:
        raise ValueError(f"{tr.id}: STA/LTA windows {nsta}/{nlta} samples at {fs_real} Hz")
    cf: FloatArray = recursive_sta_lta(data, nsta, nlta)
    cf[: min(round(phase.warmupS * fs_real), cf.size)] = 0.0
    return cf


def onset_samples(cf: FloatArray, on: float, off: float) -> npt.NDArray[np.int64]:
    """Sample index of the start of every ``trigger_onset`` trigger."""
    trig = np.asarray(trigger_onset(cf, on, off), dtype=np.int64).reshape(-1, 2)
    return trig[:, 0]


@dataclass(frozen=True)
class ChunkOnsets:
    """Trigger onsets of one chunk in REAL time, sorted, per threshold pair.

    ``p`` from the vertical segments; ``h`` from every horizontal segment, both components merged.
    """

    p: dict[Pair, FloatArray]
    h: dict[Pair, FloatArray]


def chunk_onsets(
    chunk: ModelChunk,
    cfg: BaselineConfig,
    p_pairs: Sequence[Pair],
    s_pairs: Sequence[Pair],
    counts: Counter[str],
) -> ChunkOnsets:
    """Compute each segment's characteristic function once and trigger it at every pair."""
    lists: dict[str, dict[Pair, list[FloatArray]]] = {
        "P": {pair: [] for pair in p_pairs},
        "S": {pair: [] for pair in s_pairs},
    }
    for tr in chunk.stream:
        comp = tr.stats.channel[-1:]
        if comp in cfg.p.components:
            phase, phase_cfg = "P", cfg.p
        elif comp in cfg.s.components:
            phase, phase_cfg = "S", cfg.s
        else:
            counts["segmentsOtherComponent"] += 1
            continue
        cf = characteristic_function(
            tr, phase_cfg, cfg.prefilter, chunk.timemap.factor, cfg.minSegmentMarginS
        )
        if cf is None:
            counts[f"segmentsSkippedShort{phase}"] += 1
            continue
        counts[f"segments{phase}"] += 1
        start, delta = tr.stats.starttime.timestamp, float(tr.stats.delta)
        for pair, found in lists[phase].items():
            idx = onset_samples(cf, *pair)
            if idx.size:
                found.append(chunk.timemap.to_real(start + idx * delta))
    return ChunkOnsets(
        p={pair: _sorted(found) for pair, found in lists["P"].items()},
        h={pair: _sorted(found) for pair, found in lists["S"].items()},
    )


def _sorted(parts: list[FloatArray]) -> FloatArray:
    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.sort(np.concatenate(parts))


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


def s_after_p(p: FloatArray, h: FloatArray, min_s: float, max_s: float) -> FloatArray:
    """For each P, the earliest horizontal onset in ``[p + min_s, p + max_s]``; unique, sorted."""
    if p.size == 0 or h.size == 0:
        return np.empty(0, dtype=np.float64)
    i = np.searchsorted(h, p + min_s, side="left")
    ok = i < h.size
    cand = h[i[ok]]
    return np.unique(cand[cand <= p[ok] + max_s])


def phase_picks(
    onsets: ChunkOnsets, point: GridPoint, chunk: ModelChunk, cfg: BaselineConfig
) -> tuple[FloatArray, FloatArray]:
    """P and S candidates of one chunk at one threshold point, before the keep rule."""
    p = onsets.p[point.p_pair]
    p_valid = p[~edge_mask(p, chunk.dataEdges, cfg.gapEdgeS)]
    return p, s_after_p(p_valid, onsets.h[point.s_pair], cfg.minSMinusPS, cfg.maxSMinusPS)


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
    def reason(self) -> str:
        """Why a station has no picks at the chosen thresholds ("" when it has some)."""
        if self.picks:
            return ""
        if self.notCached is not None:
            return "not cached"
        if self.chunks.yielded == 0:
            if self.chunks.noSegments:
                return "every segment too short for for_picking"
            return "no data in window"
        if self.counts["segmentsP"] + self.counts["segmentsS"] == 0:
            return "every segment shorter than ltaS + minSegmentMarginS"
        if self.counts["droppedNearGapEdgeP"] + self.counts["droppedNearGapEdgeS"]:
            return "every trigger within gapEdgeS of a data edge"
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
            onsets = chunk_onsets(chunk, b, p_pairs, s_pairs, res.counts)
            # Published picks: every candidate goes through the shared keep rule.
            for phase, times in zip(PHASES, phase_picks(onsets, chosen, chunk, b), strict=True):
                for t in times.tolist():
                    kept, why = pick_is_kept(t, chunk, b.gapEdgeS)
                    if kept:
                        res.picks.append((t, phase))
                    elif why == NEAR_GAP_EDGE:
                        res.counts[f"droppedNearGapEdge{phase}"] += 1
            # Sweep counts: the same rule, vectorised (tests check it matches pick_is_kept).
            for k, point in enumerate(grid):
                p, s = phase_picks(onsets, point, chunk, b)
                sweep_p[k].append(p[keep_masks(p, chunk, b.gapEdgeS)[0]])
                sweep_s[k].append(s[keep_masks(s, chunk, b.gapEdgeS)[0]])
    except CacheMissError as exc:  # nothing at all cached for the station: raised on the 1st read
        res.notCached = str(exc)
        log.warning("baseline %s: not cached, no picks (%s)", station.id, exc)
    res.sweepP = [_sorted(parts) for parts in sweep_p]
    res.sweepS = [_sorted(parts) for parts in sweep_s]
    res.runtimeS = perf_counter() - began
    log.info(
        "baseline %s (%s): P=%d S=%d dropped_near_gap_edge P=%d S=%d segments P=%d S=%d "
        "skipped_short P=%d S=%d chunks %d/%d runtime_s=%.2f",
        station.id,
        station.profile,
        res.n("P"),
        res.n("S"),
        res.counts["droppedNearGapEdgeP"],
        res.counts["droppedNearGapEdgeS"],
        res.counts["segmentsP"],
        res.counts["segmentsS"],
        res.counts["segmentsSkippedShortP"],
        res.counts["segmentsSkippedShortS"],
        res.chunks.yielded,
        res.chunks.planned,
        res.runtimeS,
    )
    return res


# --- H2 evaluation -----------------------------------------------------------------------------------


class H2PipelineMissingError(RuntimeError):
    """H2's associate/locate/match/assign_tiers (docs/02 section 5) are not importable yet."""


@dataclass(frozen=True)
class H2Pipeline:
    associate: Callable[..., Any]
    locate: Callable[..., Any]
    match: Callable[..., Any]
    assign_tiers: Callable[..., Any]


@dataclass(frozen=True)
class SweepScores:
    candidates: int
    recoveredPublic: int
    tierA: int


def load_h2_pipeline() -> H2Pipeline:
    """H2's four library functions; ``H2PipelineMissingError`` if any is not merged.

    A module that exists but fails to import for another reason (a missing dependency, a bug)
    raises its own error: only a missing H2 module or function counts as "not merged".
    """
    found: dict[str, Callable[..., Any]] = {}
    missing: list[str] = []
    for module_name, attr in _H2_API:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            missing.append(module_name)
            continue
        fn = getattr(module, attr, None)
        if callable(fn):
            found[attr] = fn
        else:
            missing.append(f"{module_name}.{attr}")
    if missing:
        raise H2PipelineMissingError(f"{H2_MISSING_MESSAGE} (missing: {', '.join(missing)})")
    return H2Pipeline(**found)


def evaluate_with_h2(
    picks_df: pd.DataFrame, stations_df: pd.DataFrame, ctx: StageContext
) -> SweepScores:
    """Associate, locate, match and tier one pick set with H2's pipeline (docs/02 section 5).

    ``candidates``: located events; ``recoveredPublic``: public catalog events matched to one;
    ``tierA``: Tier A events. Raises ``H2PipelineMissingError`` until H2's API is merged.
    """
    h2 = load_h2_pipeline()
    from hq_contracts.io import read_table

    seismology = ctx.config.seismology
    run_section = ctx.config.run
    catalog = read_table(ctx.path("catalog.parquet"))
    assoc = h2.associate(picks_df, stations_df, seismology, run_section)
    located = h2.locate(assoc, picks_df, stations_df, seismology, run_section)
    matched = h2.match(located.events, catalog, seismology)
    tiers = h2.assign_tiers(located.events, matched.matches, seismology)
    return SweepScores(
        candidates=len(located.events),
        recoveredPublic=int(matched.matches["eventId"].notna().sum()),
        tierA=int((tiers.events["tier"] == "A").sum()),
    )


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


def sweep_frame(
    results: Sequence[StationResult],
    grid: Sequence[GridPoint],
    scores: Sequence[SweepScores | None],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k, point in enumerate(grid):
        n_p = sum(r.sweepP[k].size for r in results)
        n_s = sum(r.sweepS[k].size for r in results)
        n_st = sum(1 for r in results if r.sweepP[k].size + r.sweepS[k].size)
        sc = scores[k]
        rows.append(
            {
                "params": json.dumps(point.params(), sort_keys=True),
                "candidates": None if sc is None else sc.candidates,
                "recoveredPublic": None if sc is None else sc.recoveredPublic,
                "tierA": None if sc is None else sc.tierA,
                "nP": n_p,
                "nS": n_s,
                "nStations": n_st,
            }
        )
    df = pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))
    return df.astype(
        {
            "candidates": "Int64",
            "recoveredPublic": "Int64",
            "tierA": "Int64",
            "nP": "int64",
            "nS": "int64",
            "nStations": "int64",
        }
    )


def score_sweep(
    results: Sequence[StationResult],
    grid: Sequence[GridPoint],
    stations_df: pd.DataFrame,
    prob: float,
    ctx: StageContext,
) -> list[SweepScores | None]:
    """H2 scores per grid point, or nulls (with a WARNING) while H2's pipeline is not merged."""
    try:
        return [
            evaluate_with_h2(picks_frame(sweep_rows(results, k), prob), stations_df, ctx)
            for k in range(len(grid))
        ]
    except H2PipelineMissingError as exc:
        log.warning(
            "baseline sweep: %s; candidates, recoveredPublic and tierA are null for all %d "
            "grid points",
            exc,
            len(grid),
        )
        return [None] * len(grid)


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
        f"{'dropS':>5} {'short':>5} {'sec':>6}  reason for no picks"
    )
    title = (
        "STA/LTA baseline at the chosen thresholds "
        "(drop = within gapEdgeS of a data edge, short = segments skipped as too short)"
    )
    lines = [
        title,
        head,
        "-" * len(head),
    ]
    for r in results:
        short = r.counts["segmentsSkippedShortP"] + r.counts["segmentsSkippedShortS"]
        lines.append(
            f"{r.stationId:<10} {r.profile:<12} {r.chunks.yielded:>3}/{r.chunks.planned:<3} "
            f"{r.n('P'):>6} {r.n('S'):>6} {r.counts['droppedNearGapEdgeP']:>5} "
            f"{r.counts['droppedNearGapEdgeS']:>5} {short:>5} {r.runtimeS:>6.1f}  {r.reason}"
        )
    return "\n".join(lines)


def format_sweep_table(sweep_df: pd.DataFrame) -> str:
    lines = [
        f"{'params':<50} {'nP':>7} {'nS':>7} {'nSta':>5} {'cand':>5} {'recov':>5} {'tierA':>5}"
    ]
    for row in sweep_df.itertuples(index=False):
        cells = [
            "null" if pd.isna(v) else str(int(v))
            for v in (row.candidates, row.recoveredPublic, row.tierA)
        ]
        lines.append(
            f"{row.params:<50} {row.nP:>7} {row.nS:>7} {row.nStations:>5} "
            f"{cells[0]:>5} {cells[1]:>5} {cells[2]:>5}"
        )
    return "\n".join(lines)


# --- stage -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResult:
    stations: list[StationResult]
    picks: pd.DataFrame
    sweep: pd.DataFrame
    counts: dict[str, int]


def _map_stations(
    rows: Sequence[StationRow], work: Callable[[StationRow], StationResult], workers: int
) -> list[StationResult]:
    if workers == 1 or len(rows) <= 1:
        return [work(r) for r in rows]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="baseline") as pool:
        return list(pool.map(work, rows))  # input order: deterministic


def run_baseline(
    ctx: StageContext,
    *,
    t0: float | None = None,
    t1: float | None = None,
    station_ids: Sequence[str] | None = None,
    read_window: ReadWindow | None = None,
) -> BaselineResult:
    """Pick every usedInRun station (or ``station_ids``) over the run window (or [t0, t1)).

    Writes ``picks_stalta.parquet`` and ``baseline_sweep.parquet``, prints the per-station and
    sweep tables and calls ``ctx.record("baseline", ...)``.
    """
    from hq_contracts.io import read_table, write_table

    began = perf_counter()
    signal: SignalConfig = ctx.config.signal
    b = signal.baseline
    check_config(signal)
    run_section: RunSection = ctx.config.run
    t0 = run_section.window_start_s if t0 is None else t0
    t1 = run_section.window_end_s if t1 is None else t1
    stations_df = read_table(ctx.path("stations.parquet"))
    rows = station_rows(stations_df, station_ids)
    grid = sweep_grid(b)
    log.info(
        "baseline: %d stations over [%s, %s), %d sweep points, %d worker(s)",
        len(rows),
        UTCDateTime(t0),
        UTCDateTime(t1),
        len(grid),
        b.maxWorkers,
    )

    def work(row: StationRow) -> StationResult:
        return pick_station(
            row, t0, t1, signal, grid, cache_dir=ctx.cache_dir, read_window=read_window
        )

    results = _map_stations(rows, work, b.maxWorkers)
    picks = picks_frame(chosen_rows(results), b.prob)
    write_table(picks, ctx.path(PICKS_FILE), "Pick")
    scores = score_sweep(results, grid, stations_df, b.prob, ctx)
    sweep = sweep_frame(results, grid, scores)
    write_table(sweep, ctx.path(SWEEP_FILE), SWEEP_MODEL)

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
        "droppedNearGapEdgeP": totals["droppedNearGapEdgeP"],
        "droppedNearGapEdgeS": totals["droppedNearGapEdgeS"],
        "segmentsP": totals["segmentsP"],
        "segmentsS": totals["segmentsS"],
        "segmentsSkippedShortP": totals["segmentsSkippedShortP"],
        "segmentsSkippedShortS": totals["segmentsSkippedShortS"],
        "segmentsOtherComponent": totals["segmentsOtherComponent"],
        **chunk_totals.as_counts(),
        "sweepPoints": len(grid),
        "sweepScored": sum(1 for s in scores if s is not None),
    }
    print(format_station_table(results))
    print(format_sweep_table(sweep))
    runtime_s = perf_counter() - began
    log.info(
        "baseline: %d P and %d S picks on %d of %d stations, %d near-gap-edge drops, "
        "%d sweep points (%d scored), wrote %s and %s in %.1f s",
        counts["nP"],
        counts["nS"],
        counts["stationsWithPicks"],
        counts["stations"],
        counts["droppedNearGapEdgeP"] + counts["droppedNearGapEdgeS"],
        counts["sweepPoints"],
        counts["sweepScored"],
        PICKS_FILE,
        SWEEP_FILE,
        runtime_s,
    )
    ctx.record(STAGE, runtime_s=runtime_s, counts=counts, params=b.model_dump(mode="json"))
    return BaselineResult(stations=results, picks=picks, sweep=sweep, counts=counts)


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
    ids = [s.strip() for s in args.stations.split(",") if s.strip()] if args.stations else None
    run_baseline(
        ctx,
        t0=UTCDateTime(args.start).timestamp if args.start else None,
        t1=UTCDateTime(args.end).timestamp if args.end else None,
        station_ids=ids,
    )
    return 0


if __name__ == "__main__":
    # ``python -m`` runs a second copy of this module as __main__ (hq.baseline imported the real
    # one first, hence runpy's RuntimeWarning); delegate to the real module so there is one copy.
    from hq.baseline.run import main as _main

    sys.exit(_main())

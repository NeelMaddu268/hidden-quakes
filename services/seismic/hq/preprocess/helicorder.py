"""Station-day helicorder: "A day at one station" (SEIS-10).

A drum plot of one borehole station's vertical channel over the run's UTC day, with candidate
events and public-catalog events marked at their origin times. Written as a PNG exactly
``layout.widthPx`` wide and a JSON manifest (the station-day asset contract the web panel reads).
Everything is configured in ``signal.yaml`` -> ``helicorder``.

Station. Among ``stations.parquet`` rows of ``stationKind`` (and ``usedInRun`` when configured),
the one with the most ``picks.parquet`` picks of ``pickPhases`` over the day (ties: station id
order). Its vertical channel (component ``component``) is drawn. The pick count and the next
``runnersUp`` stations go into the manifest.

Day. The run's window (``meta.json`` -> ``run.windowStart`` / ``windowEnd``, checked against
``run.json``), which must be exactly one UTC day.

Rendering, row by row (``rowMinutes`` per row):

1. ``read_window`` for the row plus ``padS`` at both ends (raw counts, the cache is read only).
2. Each gap-separated segment on its own: detrend, taper, zero-phase Butterworth bandpass
   (``bandHz``); nothing is merged with fill or interpolated across a gap. Samples outside the
   row are dropped after filtering, so filter and taper edges at the read edges fall in the pad.
   Segments too short for the bandpass's edge padding are dropped and counted.
3. Each segment becomes a per-pixel-column min / max envelope (one column per plot pixel), so a
   spike shorter than a pixel keeps its full height. A column no sample falls in is not drawn:
   gaps are left blank. Every segment is drawn as its own polygon.
4. One amplitude scale for the whole day: the ``refPercentile`` of the per-column peak ``|x|``
   over every drawn column maps to ``refHeightRows`` row spacings; every trace is clipped at
   ``clipRows`` row spacings, so a row never bleeds further into its neighbours.
5. Markers: candidate events (bundle ``events.json``) as vertical ticks in the candidate colour
   with the tier's opacity; public-catalog events (bundle ``catalog.json``) as hollow diamonds.
   Only origin times inside the day are drawn. The bundle is what the site shows, so its counts
   are the ones drawn; the run tables are cross-checked and any disagreement is logged.

CLI::

    uv run python -m hq.preprocess.helicorder --run-dir <run> --cache-dir <cache>
        --config-dir configs/showcase --bundle-dir <bundle> --out-dir <dir>

It writes nothing but ``<fileStem>.png`` and ``<fileStem>.json`` under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from obspy import Stream
from scipy import signal as sps

from hq.config.signal import HelicorderConfig, SignalConfig
from hq.ingest.cache import Segment, read_window, window_segments
from hq.preprocess.profiles import _detrend_taper, _min_filter_samples, _segments
from hq.preprocess.sonify import iso_utc, vertical_channel

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Labels fixed by the station-day asset contract, not knobs.
SOURCE = "EarthScope public waveforms"
GENERATOR = "hq.preprocess.helicorder"
MARKER_TIME = "origin"
DAY_S = 86400.0
TIERS = ("A", "B", "C")
LEGEND_KEYS = ("tierA", "tierB", "tierC", "public")
LEGEND_LABELS = {
    "tierA": "Candidate events, Tier A",
    "tierB": "Candidate events, Tier B",
    "tierC": "Candidate events, Tier C",
    "public": "Public-catalog events",
}
_MINUTE_S = 60.0
_PT_PER_IN = 72.0


# --- selection --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StationChoice:
    stationId: str
    channel: str
    seedId: str
    sensorDepthM: float | None
    pickCount: int
    runnersUp: tuple[tuple[str, int], ...]


def eligible_stations(stations: pd.DataFrame, cfg: HelicorderConfig) -> pd.DataFrame:
    """Stations of ``stationKind`` (and ``usedInRun`` when configured), sorted by id."""
    mask = stations["kind"] == cfg.stationKind
    if cfg.usedInRunOnly:
        mask &= stations["usedInRun"].astype(bool)
    out = stations[mask].sort_values("id", kind="stable")
    if out.empty:
        raise ValueError(f"no {cfg.stationKind} stations to choose from")
    return out


def ranked_by_picks(
    stations: pd.DataFrame, picks: pd.DataFrame, t0: float, t1: float, cfg: HelicorderConfig
) -> list[tuple[str, int]]:
    """Eligible stations with their pick counts (``pickPhases``, ``[t0, t1)``), most first; ties
    in station id order."""
    elig = eligible_stations(stations, cfg)
    inside = picks[
        (picks["t"] >= t0) & (picks["t"] < t1) & picks["phase"].isin(list(cfg.pickPhases))
    ]
    counts = inside.groupby("stationId").size()
    ranked = [(str(sid), int(counts.get(sid, 0))) for sid in elig["id"]]
    ranked.sort(key=lambda item: -item[1])  # stable: equal counts keep id order
    return ranked


def selection_rule(cfg: HelicorderConfig) -> str:
    """The station-choice rule as reader-facing text (the web panel shows it), from the config."""
    phases = list(cfg.pickPhases)
    named = phases[0] if len(phases) == 1 else f"{', '.join(phases[:-1])} and {phases[-1]}"
    used = " used in the run" if cfg.usedInRunOnly else ""
    return (
        f"Station chosen by code: the {cfg.stationKind} station{used} with the most {named} "
        "phase picks over the day (ties go to the first station ID); its vertical channel is shown."
    )


def choose_station(
    stations: pd.DataFrame, picks: pd.DataFrame, t0: float, t1: float, cfg: HelicorderConfig
) -> StationChoice:
    ranked = ranked_by_picks(stations, picks, t0, t1, cfg)
    station_id, n_picks = ranked[0]
    if n_picks == 0:
        raise ValueError("no picks at any eligible station over the day")
    row = stations.set_index("id").loc[station_id]
    channel = vertical_channel(station_id, list(row["channels"]), cfg.component)
    depth = row["sensorDepthM"]
    return StationChoice(
        stationId=station_id,
        channel=channel,
        seedId=f"{row['network']}.{row['station']}.{row['location'] or ''}.{channel}",
        sensorDepthM=None if pd.isna(depth) else float(depth),
        pickCount=n_picks,
        runnersUp=tuple(ranked[1 : 1 + cfg.runnersUp]),
    )


# --- day, rows and gaps (pure) ---------------------------------------------------------------------


def day_window(start: float, end: float) -> tuple[float, float, str]:
    """(t0, t1, "YYYY-MM-DD") of a window that must be exactly one UTC day."""
    if end - start != DAY_S or start % DAY_S != 0.0:
        raise ValueError(f"window [{start}, {end}) is not exactly one UTC day")
    return start, end, datetime.fromtimestamp(start, UTC).strftime("%Y-%m-%d")


def n_rows(t0: float, t1: float, row_s: float) -> int:
    rows = (t1 - t0) / row_s
    if rows != int(rows):
        raise ValueError(f"the window is not a whole number of {row_s} s rows")
    return int(rows)


def row_position(t: FloatArray, t0: float, row_s: float) -> tuple[IntArray, FloatArray]:
    """(row index, seconds into the row) of epoch times; rows start at ``t0``."""
    offset = np.asarray(t, dtype=np.float64) - t0
    rows = np.floor(offset / row_s).astype(np.int64)
    return rows, offset - rows * row_s


def covered_seconds(segments: Sequence[Segment], channel: str, t0: float, t1: float) -> float:
    """Seconds of ``[t0, t1)`` holding samples of ``channel`` (a sample covers ``delta``)."""
    spans = sorted(
        (max(s.start, t0), min(s.end + s.delta, t1)) for s in segments if s.channel == channel
    )
    covered, reach = 0.0, t0
    for lo, hi in spans:  # union, so overlapping records never count twice
        lo = max(lo, reach)
        if hi > lo:
            covered += hi - lo
            reach = hi
    return covered


# --- envelope (pure) -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    """One gap-free segment's per-column envelope within one row."""

    cols: IntArray  # ascending pixel columns that hold at least one sample
    lo: FloatArray  # min of the filtered samples in each column
    hi: FloatArray  # max


@dataclass
class RowResult:
    pieces: list[Piece] = field(default_factory=list)
    nSegments: int = 0
    nDropped: int = 0  # too short for the bandpass's edge padding


def column_envelope(
    start: float, rate_hz: float, data: FloatArray, row_t0: float, row_s: float, ncols: int
) -> Piece | None:
    """Per-column min / max of the samples of one segment that fall in ``[row_t0, row_t0 + row_s)``."""
    t = start + np.arange(data.size, dtype=np.float64) / rate_hz
    keep = (t >= row_t0) & (t < row_t0 + row_s)
    if not keep.any():
        return None
    x = data[keep]
    cols = np.floor((t[keep] - row_t0) / row_s * ncols).astype(np.int64)
    np.clip(cols, 0, ncols - 1, out=cols)
    starts = np.flatnonzero(np.r_[True, np.diff(cols) != 0])
    return Piece(
        cols=cols[starts],
        lo=np.minimum.reduceat(x, starts),
        hi=np.maximum.reduceat(x, starts),
    )


def row_envelope(
    st: Stream, row_t0: float, row_s: float, ncols: int, cfg: HelicorderConfig
) -> RowResult:
    """Filter every gap-separated segment of one channel on its own; envelope each into columns.

    ``st`` holds one channel's traces for the row plus pad. Segments are never merged with fill,
    and no sample is interpolated or placed inside a gap.
    """
    out = RowResult()
    if len(st) == 0:
        return out
    ids = {tr.id for tr in st}
    if len(ids) != 1:
        raise ValueError(f"one channel at a time, got {sorted(ids)}")
    rates = {float(tr.stats.sampling_rate) for tr in st}
    if len(rates) != 1:
        raise ValueError(f"one source rate expected, got {sorted(rates)} Hz")
    (rate_hz,) = rates
    low, high = cfg.bandHz
    if high >= rate_hz / 2.0:
        raise ValueError(f"band {cfg.bandHz} Hz reaches the source Nyquist of {rate_hz} Hz")
    sos = sps.butter(cfg.bandpassCorners, [low, high], btype="bandpass", fs=rate_hz, output="sos")
    segments, _ = _segments(st, cfg.joinMisalignmentSamples)
    segments.sort(key=lambda tr: tr.stats.starttime)
    out.nSegments = len(segments)
    for seg in segments:
        if seg.stats.npts <= _min_filter_samples(sos):
            out.nDropped += 1
            continue
        _detrend_taper(
            seg, cfg.detrend, cfg.taper.type, cfg.taper.maxPercentage, cfg.taper.maxLengthS
        )
        filtered = sps.sosfiltfilt(sos, seg.data)
        piece = column_envelope(
            seg.stats.starttime.timestamp, rate_hz, filtered, row_t0, row_s, ncols
        )
        if piece is not None:
            out.pieces.append(piece)
    return out


def day_reference(rows: Sequence[RowResult], percentile: float) -> float:
    """``percentile`` of the per-column peak ``|x|`` over every drawn column of the day."""
    peaks = [np.maximum(np.abs(p.lo), np.abs(p.hi)) for r in rows for p in r.pieces]
    if not peaks:
        raise ValueError("no samples in the day: nothing to draw")
    ref = float(np.percentile(np.concatenate(peaks), percentile))
    if not ref > 0.0:
        raise ValueError(f"reference amplitude is {ref}: the data is flat")
    return ref


def scaled_band(
    piece: Piece, row: int, gain: float, clip_rows: float, min_rows: float
) -> tuple[FloatArray, FloatArray]:
    """(top, bottom) y of one piece in row units (rows run downwards, ground motion up is up),
    clipped at ``clip_rows`` and at least ``min_rows`` tall."""
    up = np.clip(piece.hi * gain, -clip_rows, clip_rows)
    down = np.clip(piece.lo * gain, -clip_rows, clip_rows)
    top, bottom = row - up, row - down
    short = bottom - top < min_rows
    mid = (top + bottom) / 2.0
    top = np.where(short, mid - min_rows / 2.0, top)
    bottom = np.where(short, mid + min_rows / 2.0, bottom)
    return top, bottom


# --- markers (pure) --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Markers:
    """Row index and minute into the row of every marker, per legend key."""

    rows: dict[str, IntArray]
    minutes: dict[str, FloatArray]

    def count(self, key: str) -> int:
        return int(self.rows[key].size)


def select_markers(
    candidates: Sequence[tuple[float, str]],
    public: Sequence[float],
    t0: float,
    t1: float,
    row_s: float,
) -> Markers:
    """Candidate (origin time, tier) and public-catalog origin times inside ``[t0, t1)``."""
    rows: dict[str, IntArray] = {}
    minutes: dict[str, FloatArray] = {}
    groups: dict[str, list[float]] = {f"tier{tier}": [] for tier in TIERS}
    for t, tier in candidates:
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}")
        groups[f"tier{tier}"].append(t)
    groups["public"] = list(public)
    for key in LEGEND_KEYS:
        times = np.asarray(groups[key], dtype=np.float64)
        times = np.sort(times[(times >= t0) & (times < t1)])
        r, s = row_position(times, t0, row_s)
        rows[key], minutes[key] = r, s / _MINUTE_S
    return Markers(rows, minutes)


# --- manifest ---------------------------------------------------------------------------------------


def caption_text(
    *,
    station_id: str,
    seed_id: str,
    station_kind: str,
    sensor_depth_m: float | None,
    day_utc: str,
    band_hz: tuple[float, float],
    row_minutes: int,
) -> str:
    """One or two sentences built from the manifest's own fields."""
    depth = f", sensor {sensor_depth_m:g} m below the surface" if sensor_depth_m else ""
    return (
        f"The vertical channel {seed_id} of {station_kind} station {station_id}{depth}, over "
        f"the UTC day {day_utc}, bandpassed {band_hz[0]:g}-{band_hz[1]:g} Hz, {row_minutes} minutes per "
        "row; gaps in the recording are left blank. Amber ticks mark candidate events at their "
        "origin times (brighter for higher tiers), and hollow diamonds mark public-catalog events."
    )


def legend_entries(markers: Markers, cfg: HelicorderConfig) -> list[dict[str, Any]]:
    style = cfg.style
    out: list[dict[str, Any]] = []
    for key in LEGEND_KEYS:
        if key == "public":
            color, opacity, shape = style.public, 1.0, "diamond"
        else:
            color, shape = style.candidate, "tick"
            opacity = getattr(style.tierOpacity, key[-1])
        out.append(
            {
                "key": key,
                "label": LEGEND_LABELS[key],
                "color": color,
                "opacity": opacity,
                "shape": shape,
                "count": markers.count(key),
            }
        )
    return out


def build_manifest(
    *,
    cfg: HelicorderConfig,
    run_id: str,
    choice: StationChoice,
    t0: float,
    t1: float,
    day_utc: str,
    rows: int,
    gap_s: float,
    coverage: float,
    markers: Markers,
    width_px: int,
    height_px: int,
) -> dict[str, Any]:
    """The station-day asset contract, in its field order."""
    return {
        "image": f"{cfg.fileStem}.png",
        "widthPx": width_px,
        "heightPx": height_px,
        "title": cfg.title,
        "caption": caption_text(
            station_id=choice.stationId,
            seed_id=choice.seedId,
            station_kind=cfg.stationKind,
            sensor_depth_m=choice.sensorDepthM,
            day_utc=day_utc,
            band_hz=cfg.bandHz,
            row_minutes=cfg.rowMinutes,
        ),
        "runId": run_id,
        "stationId": choice.stationId,
        "seedId": choice.seedId,
        "channel": choice.channel,
        "stationKind": cfg.stationKind,
        "sensorDepthM": choice.sensorDepthM,
        "dayUtc": day_utc,
        "startUtc": iso_utc(t0),
        "endUtc": iso_utc(t1),
        "rowMinutes": cfg.rowMinutes,
        "rows": rows,
        "filterHz": [cfg.bandHz[0], cfg.bandHz[1]],
        "gapsFilled": False,
        "gapSeconds": gap_s,
        "coverageFraction": coverage,
        "markerTime": MARKER_TIME,
        "legend": legend_entries(markers, cfg),
        "selection": {
            "rule": selection_rule(cfg),
            "pickCount": choice.pickCount,
            "runnersUp": [{"stationId": s, "pickCount": n} for s, n in choice.runnersUp],
        },
        "source": SOURCE,
        "generator": GENERATOR,
    }


# --- drawing ----------------------------------------------------------------------------------------


def plot_columns(cfg: HelicorderConfig) -> int:
    """One envelope column per pixel of the plot area."""
    return cfg.layout.widthPx - cfg.layout.leftPx - cfg.layout.rightPx


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG's IHDR chunk."""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def _row_label(t0: float, row: int, row_s: float) -> str:
    return datetime.fromtimestamp(t0 + row * row_s, UTC).strftime("%H:%M")


def render_png(
    *,
    cfg: HelicorderConfig,
    rows: Sequence[RowResult],
    ref: float,
    markers: Markers,
    t0: float,
    choice: StationChoice,
    day_utc: str,
    run_id: str,
) -> bytes:
    """The helicorder as PNG bytes, exactly ``layout.widthPx`` x ``layout.heightPx``."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import patheffects
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    lay, style, mk = cfg.layout, cfg.style, cfg.markers
    row_s = cfg.rowMinutes * _MINUTE_S
    n = len(rows)
    ncols = plot_columns(cfg)
    plot_h = lay.heightPx - lay.topPx - lay.bottomPx
    span_rows = (n - 1) + lay.headroomRows + lay.footroomRows
    px_per_row = plot_h / span_rows
    px_to_pt = _PT_PER_IN / lay.dpi
    gain = cfg.scale.refHeightRows / ref
    min_rows = lay.minTraceHeightPx / px_per_row
    col_minutes = cfg.rowMinutes / ncols

    plt.rcParams["font.family"] = style.fontFamily
    fig = plt.figure(figsize=(lay.widthPx / lay.dpi, lay.heightPx / lay.dpi), dpi=lay.dpi)
    fig.patch.set_facecolor(style.background)
    ax = fig.add_axes(
        (
            lay.leftPx / lay.widthPx,
            lay.bottomPx / lay.heightPx,
            ncols / lay.widthPx,
            plot_h / lay.heightPx,
        )
    )
    ax.set_facecolor(style.background)
    ax.set_xlim(0.0, cfg.rowMinutes)
    ax.set_ylim(n - 1 + lay.footroomRows, -lay.headroomRows)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Faint minute grid behind everything.
    step = 5 if cfg.rowMinutes == 30 else 10
    minute_ticks = np.arange(0, cfg.rowMinutes + 1, step)
    for m in minute_ticks:
        ax.axvline(m, color=style.grid, linewidth=0.8, alpha=0.6, zorder=0)

    # Traces: every gap-free segment is its own polygon; columns without samples stay empty.
    for r, row in enumerate(rows):
        tone = style.traceTones[r % 2]
        for piece in row.pieces:
            top, bottom = scaled_band(piece, r, gain, cfg.scale.clipRows, min_rows)
            x = (piece.cols + 0.5) * col_minutes
            if piece.cols.size == 1:  # a single column: draw it one pixel wide
                x = np.array([x[0] - col_minutes / 2, x[0] + col_minutes / 2])
                top, bottom = np.repeat(top, 2), np.repeat(bottom, 2)
            breaks = np.flatnonzero(np.diff(piece.cols) > 1) + 1  # sub-segment column skips
            for xs, ts, bs in zip(
                np.split(x, breaks), np.split(top, breaks), np.split(bottom, breaks), strict=True
            ):
                ax.fill_between(xs, ts, bs, color=tone, linewidth=0.0, zorder=1)

    # Markers above each row's centre: ticks for candidate events, hollow diamonds for public.
    tick_lw = mk.tickWidthPx * px_to_pt
    halo = [
        patheffects.Stroke(
            linewidth=tick_lw + 2 * mk.haloPx * px_to_pt, foreground=style.background
        ),
        patheffects.Normal(),
    ]
    for key in ("tierC", "tierB", "tierA"):  # higher tiers on top
        if markers.count(key) == 0:
            continue
        rr = markers.rows[key].astype(np.float64)
        ax.vlines(
            markers.minutes[key],
            rr - mk.tickToRows,
            rr - mk.tickFromRows,
            colors=style.candidate,
            alpha=getattr(style.tierOpacity, key[-1]),
            linewidth=tick_lw,
            zorder=3,
            path_effects=halo,
        )
    if markers.count("public"):
        ax.scatter(
            markers.minutes["public"],
            markers.rows["public"] - mk.diamondRows,
            s=(mk.diamondSizePx * px_to_pt) ** 2,
            marker="D",
            facecolors="none",
            edgecolors=style.public,
            linewidths=mk.diamondEdgePx * px_to_pt,
            zorder=4,
            path_effects=[
                patheffects.Stroke(
                    linewidth=(mk.diamondEdgePx + 2 * mk.haloPx) * px_to_pt,
                    foreground=style.background,
                ),
                patheffects.Normal(),
            ],
        )

    # Axes: UTC row starts on the left (hours brighter), minutes into the row on top.
    ax.set_yticks(np.arange(n))
    labels = [_row_label(t0, r, row_s) for r in range(n)]
    ax.set_yticklabels(labels, fontsize=cfg.style.labelPt)
    for r, lab in enumerate(ax.get_yticklabels()):
        on_hour = (t0 + r * row_s) % 3600.0 == 0.0
        lab.set_color(style.text if on_hour else style.textDim)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(minute_ticks)
    ax.set_xticklabels([f"{m:d}" for m in minute_ticks], fontsize=style.labelPt)
    ax.tick_params(axis="x", colors=style.textDim, length=4, color=style.grid, pad=4)
    ax.set_xlabel("minutes into the row", color=style.textDim, fontsize=style.labelPt, labelpad=6)
    fig.text(
        (lay.leftPx - 12) / lay.widthPx,
        1 - (lay.topPx - 6) / lay.heightPx,
        "UTC",
        color=style.textDim,
        fontsize=style.labelPt,
        ha="right",
        va="bottom",
    )

    # Title block and legend.
    x_left = lay.leftPx / lay.widthPx * 0.25
    fig.text(
        x_left,
        1 - 28 / lay.heightPx,
        cfg.title,
        color=style.text,
        fontsize=style.titlePt,
        fontweight="bold",
        ha="left",
        va="top",
    )
    depth = (
        f"{cfg.stationKind} sensor {choice.sensorDepthM:g} m down"
        if choice.sensorDepthM
        else cfg.stationKind
    )
    subtitle = (
        f"{choice.stationId}  ·  {choice.channel}  ·  {depth}  ·  {day_utc} UTC  ·  "
        f"{cfg.bandHz[0]:g}–{cfg.bandHz[1]:g} Hz  ·  {cfg.rowMinutes} min per row  ·  "
        "gaps left blank"
    )
    fig.text(
        x_left,
        1 - 82 / lay.heightPx,
        subtitle,
        color=style.textDim,
        fontsize=style.subtitlePt,
        ha="left",
        va="top",
    )
    handles = []
    labels_legend = []
    for key in LEGEND_KEYS:
        if key == "public":
            handle = Line2D(
                [],
                [],
                linestyle="none",
                marker="D",
                markersize=mk.diamondSizePx * px_to_pt * 1.2,
                markerfacecolor="none",
                markeredgecolor=style.public,
                markeredgewidth=mk.diamondEdgePx * px_to_pt * 1.2,
            )
        else:
            handle = Line2D(
                [],
                [],
                linestyle="none",
                marker="|",
                markersize=16,
                markeredgewidth=tick_lw * 1.6,
                color=style.candidate,
                alpha=getattr(style.tierOpacity, key[-1]),
            )
        handles.append(handle)
        labels_legend.append(f"{LEGEND_LABELS[key]}  ({markers.count(key)})")
    fig.legend(
        handles,
        labels_legend,
        loc="upper right",
        bbox_to_anchor=(1 - lay.rightPx / lay.widthPx, 1 - 22 / lay.heightPx),
        ncol=2,
        frameon=False,
        fontsize=style.subtitlePt - 1,
        labelcolor=style.text,
        handlelength=1.2,
        handletextpad=0.6,
        columnspacing=1.8,
        borderaxespad=0.0,
    )

    # Footer.
    fig.text(
        x_left,
        14 / lay.heightPx,
        f"Source: {SOURCE}  ·  Hidden Quakes run {run_id}",
        color=style.textDim,
        fontsize=style.footerPt,
        ha="left",
        va="bottom",
    )
    fig.text(
        1 - lay.rightPx / lay.widthPx,
        14 / lay.heightPx,
        f"One amplitude scale for the whole day, clipped at {cfg.scale.clipRows:g} row "
        "spacings  ·  markers at origin times",
        color=style.textDim,
        fontsize=style.footerPt,
        ha="right",
        va="bottom",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=lay.dpi, facecolor=style.background)
    plt.close(fig)
    data = buf.getvalue()
    size = png_size(data)
    if size != (lay.widthPx, lay.heightPx):
        raise RuntimeError(f"PNG is {size}, configured {(lay.widthPx, lay.heightPx)}")
    if len(data) >= lay.maxBytes:
        raise RuntimeError(f"PNG is {len(data)} bytes, over maxBytes {lay.maxBytes}")
    return data


# --- inputs and CLI ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Inputs:
    runId: str
    t0: float
    t1: float
    dayUtc: str
    stations: pd.DataFrame
    picks: pd.DataFrame
    candidates: list[tuple[float, str]]  # bundle events.json (origin time, tier)
    public: list[float]  # bundle catalog.json origin times


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_inputs(run_dir: Path, bundle_dir: Path) -> Inputs:
    """Run tables through ``hq_contracts.io``, the bundle through the contract models, and the
    day from the bundle's run window (checked against ``run.json``)."""
    from hq_contracts.io import from_frame, read_table
    from hq_contracts.models import BundleMeta, CatalogEvent, Pick, SeismicEvent, Station

    stations = read_table(run_dir / "stations.parquet")
    picks = read_table(run_dir / "picks.parquet")
    from_frame(stations, Station)
    if not set(picks.columns) >= set(Pick.model_fields):
        raise ValueError(f"picks.parquet lacks Pick columns: {set(Pick.model_fields) - set(picks)}")
    meta = BundleMeta.model_validate(_read_json(bundle_dir / "meta.json"))
    run_id = run_dir.name
    if meta.scene.runId != run_id:
        raise ValueError(f"bundle is from run {meta.scene.runId}, run dir is {run_id}")
    run_json = _read_json(run_dir / "run.json")
    window = (meta.run.windowStart, meta.run.windowEnd)
    if (run_json["windowStart"], run_json["windowEnd"]) != window:
        raise ValueError(f"run.json window differs from the bundle's {window}")
    t0, t1, day = day_window(*window)
    events = [SeismicEvent.model_validate(e) for e in _read_json(bundle_dir / "events.json")]
    catalog = [CatalogEvent.model_validate(e) for e in _read_json(bundle_dir / "catalog.json")]
    foreign = {e.runId for e in events} - {run_id}
    if foreign:
        raise ValueError(f"events.json holds events of other runs: {sorted(foreign)}")
    _cross_check(run_dir, events, catalog)
    return Inputs(
        runId=run_id,
        t0=t0,
        t1=t1,
        dayUtc=day,
        stations=stations,
        picks=picks,
        candidates=[(e.t, str(e.tier)) for e in events],
        public=[c.t for c in catalog],
    )


def _cross_check(run_dir: Path, events: Sequence[Any], catalog: Sequence[Any]) -> None:
    """Log whether the run's events / catalog tables agree with the bundle (the bundle wins)."""
    bundle_tiers = pd.Series([str(e.tier) for e in events]).value_counts().to_dict()
    for name, count in (("events.parquet", None), ("catalog.parquet", len(catalog))):
        path = run_dir / name
        if not path.exists():
            log.warning("helicorder: %s missing, no cross-check", name)
            continue
        table = pd.read_parquet(path)
        if count is None:
            run_tiers = table["tier"].astype(str).value_counts().to_dict()
            same = run_tiers == bundle_tiers
            log.log(
                logging.INFO if same else logging.WARNING,
                "helicorder: candidate events per tier, bundle %s, run table %s%s",
                bundle_tiers,
                run_tiers,
                "" if same else ": they disagree, the bundle is drawn",
            )
        else:
            same = len(table) == count
            log.log(
                logging.INFO if same else logging.WARNING,
                "helicorder: public-catalog events, bundle %d, run table %d%s",
                count,
                len(table),
                "" if same else ": they disagree, the bundle is drawn",
            )


def read_rows(
    choice: StationChoice, inputs: Inputs, cfg: HelicorderConfig, cache_dir: Path
) -> list[RowResult]:
    row_s = cfg.rowMinutes * _MINUTE_S
    ncols = plot_columns(cfg)
    out: list[RowResult] = []
    for r in range(n_rows(inputs.t0, inputs.t1, row_s)):
        r0 = inputs.t0 + r * row_s
        st = read_window(
            choice.stationId, r0 - cfg.padS, r0 + row_s + cfg.padS, cache_dir=cache_dir
        )
        result = row_envelope(st.select(channel=choice.channel), r0, row_s, ncols, cfg)
        log.info(
            "helicorder: row %s, %d segments, %d dropped, %d pieces",
            _row_label(inputs.t0, r, row_s),
            result.nSegments,
            result.nDropped,
            len(result.pieces),
        )
        out.append(result)
    return out


def run(
    inputs: Inputs, cfg: HelicorderConfig, cache_dir: Path, out_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose, read, render and write; returns (manifest, report)."""
    started = perf_counter()
    choice = choose_station(inputs.stations, inputs.picks, inputs.t0, inputs.t1, cfg)
    log.info(
        "helicorder: %s %s with %d picks; runners-up %s",
        choice.stationId,
        choice.channel,
        choice.pickCount,
        choice.runnersUp,
    )
    segs = window_segments(choice.stationId, inputs.t0, inputs.t1, cache_dir=cache_dir)
    covered = covered_seconds(segs, choice.channel, inputs.t0, inputs.t1)
    day_s = inputs.t1 - inputs.t0
    rows = read_rows(choice, inputs, cfg, cache_dir)
    ref = day_reference(rows, cfg.scale.refPercentile)
    row_s = cfg.rowMinutes * _MINUTE_S
    markers = select_markers(inputs.candidates, inputs.public, inputs.t0, inputs.t1, row_s)
    png = render_png(
        cfg=cfg,
        rows=rows,
        ref=ref,
        markers=markers,
        t0=inputs.t0,
        choice=choice,
        day_utc=inputs.dayUtc,
        run_id=inputs.runId,
    )
    width, height = png_size(png)
    manifest = build_manifest(
        cfg=cfg,
        run_id=inputs.runId,
        choice=choice,
        t0=inputs.t0,
        t1=inputs.t1,
        day_utc=inputs.dayUtc,
        rows=len(rows),
        gap_s=round(day_s - covered, 6),  # epoch-float noise below a microsecond is not data
        coverage=round(covered / day_s, 9),
        markers=markers,
        width_px=width,
        height_px=height,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg.fileStem}.png").write_bytes(png)
    text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    (out_dir / f"{cfg.fileStem}.json").write_text(text, encoding="utf-8")
    report = {
        "pngBytes": len(png),
        "referenceCounts": ref,
        "segments": sum(r.nSegments for r in rows),
        "droppedTooShort": sum(r.nDropped for r in rows),
        "candidatesInBundle": len(inputs.candidates),
        "publicInBundle": len(inputs.public),
        "runtimeS": perf_counter() - started,
    }
    return manifest, report


def _load_signal(config_dir: Path) -> SignalConfig:
    with (config_dir / "signal.yaml").open(encoding="utf-8") as fh:
        return SignalConfig.model_validate(yaml.safe_load(fh))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hq.preprocess.helicorder",
        description="A day at one station: helicorder PNG + manifest (SEIS-10).",
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="runs/<id> (read only)")
    parser.add_argument("--cache-dir", type=Path, required=True, help="cache root (read only)")
    parser.add_argument("--config-dir", type=Path, required=True, help="folder with signal.yaml")
    parser.add_argument(
        "--bundle-dir", type=Path, required=True, help="web bundle: events, catalog, meta"
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="where the files are written")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = _load_signal(args.config_dir).helicorder
    inputs = load_inputs(args.run_dir, args.bundle_dir)
    manifest, report = run(inputs, cfg, args.cache_dir, args.out_dir)
    print(json.dumps(manifest, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

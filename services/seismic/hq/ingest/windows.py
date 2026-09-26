"""Known-event windows (SEIS-02): the largest public-catalog events and their station coverage.

Selects the ``known.nEvents`` largest public events inside the run window and bbox, cuts a
``[t - preS, t + postS]`` window around each, and reads it from the shared waveform cache.

Depends on SEIS-05's ``hq.ingest.cache``. The windows are read with ``read_window``, and the
download manifests next to the channel-day files say which spans were ever fetched. Nothing is
downloaded here: SEIS-05's downloader is the only writer of the channel-day cache, because a
second process rewriting the same channel-day file or manifest would lose data. A span the
downloader has not fetched is reported as "not downloaded", never as a data gap, so run this
once SEIS-05 has covered the event windows (a rerun after it has is cheap).

For every event and station it reports:

- ``components``: distinct component letters (Z/N/E/1/2) among the selected channels with data
- ``gapFraction``: share of the window without samples on the worst selected channel, whatever
  the cause; a selected channel with no data at all counts as 1.0
- ``epiDistM``: catalog epicenter to station latitude/longitude (``gps2dist_azimuth``)
- ``usable``: a Z plus N/E or 1/2 from one instrument, ``gapFraction <= maxGapFraction``, and
  every missing span fetched by the downloader; ``reason`` says why not

A window PASSes when at least ``minStations`` stations are usable. A hole in the samples is a gap
when it is longer than ``minGapSamples - 1`` sample intervals, so sub-sample jitter is not a gap
and a single missing sample is. SEIS-05's ``download.channel_gaps`` (``gaps.parquet``) shares
that interior rule but not the leading edge: here a leading hole must exceed ``minGapSamples``
intervals, there it needs at least ``minGapSamples - 0.5``. A station is usable here at
``gapFraction <= maxGapFraction``; Check A counts one as useful only below
``download.maxGapFraction``.

Outputs in ``runs/<id>/known/``:

- ``gaps_<eventId>.csv``: the gap report, one row per data gap (``stationId, channel, gapStart,
  gapEnd``), i.e. a span the downloader fetched and got no samples for. A human-readable report,
  not a docs/02 run table. Reports of events no longer selected are removed.
- ``windows.json``: the selected events with per-station coverage, plus the ``known:`` config
  block under ``params``. Written last. SEIS-04 reads it; :func:`load_windows` validates it.
- ``known_windows.record.json``: this sub-step's runtime, counts and ``known:`` params.

Not a registered pipeline stage (``hq.runs.STAGES`` rejects the name), so its record goes to
``known/known_windows.record.json``, never ``run.json``; run it with the CLI::

    uv run python -m hq.ingest.windows --run-dir <dir> --config-dir configs/showcase \\
        --cache-dir <dir>
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, Protocol

import obspy
import pandas as pd
import yaml
from obspy.geodetics import gps2dist_azimuth
from pydantic import BaseModel, ConfigDict

from hq.config.run import RunSection
from hq.config.signal import KnownEventsConfig, SignalConfig

log = logging.getLogger(__name__)

STAGE = "known_windows"  # sub-step name; its record file is known/<STAGE>.record.json
# The known-event sub-steps are not pipeline stages (H4's registry, hq.runs.STAGES, rejects
# their names in RunContext.record), so each writes its runtime, counts and params to its
# own record file next to its outputs, like H2's catalog.record.json.
RECORD_SUFFIX = ".record.json"
KNOWN_DIR = "known"
WINDOWS_FILE = "windows.json"
GAP_COLUMNS = ("stationId", "channel", "gapStart", "gapEnd")
CACHE_MODULE = "hq.ingest.cache"  # SEIS-05

# Station reasons that describe the cache rather than the data.
NOT_CACHED = "not cached"  # read_window found nothing at all cached for the station
NOT_DOWNLOADED = "not downloaded"  # part of the window was never fetched by the downloader
NO_DATA = "no data in window"  # fetched, and the data center returned no samples

# Three-component means a vertical plus a horizontal pair from one instrument (the channel code
# minus its last letter); borehole horizontals may be 1/2 instead of N/E. A definition, not a knob.
_VERTICAL = "Z"
_HORIZONTAL_PAIRS = (frozenset({"N", "E"}), frozenset({"1", "2"}))
_COMPONENT_LETTERS = frozenset({"Z", "N", "E", "1", "2"})

# miniSEED times resolve to 1 microsecond; shorter spans are floating-point noise.
_SPAN_EPS_S = 1e-6
# SEIS-05's manifest format (hq.ingest.download.MANIFEST_VERSION): a chunk is listed once the
# data center has answered it, with samples ("ok") or without ("nodata").
_MANIFEST_VERSION = 1
_ANSWERED = frozenset({"ok", "nodata"})

_CATALOG_COLUMNS = ("id", "t", "latitude", "longitude", "depthKm", "mag", "magType")
_STATION_COLUMNS = (
    "id",
    "network",
    "station",
    "location",
    "latitude",
    "longitude",
    "channels",
    "usedInRun",
)
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")

Span = tuple[float, float]


class ReadWindow(Protocol):
    """Signature of ``hq.ingest.cache.read_window`` (docs/02 section 5)."""

    def __call__(
        self, station_id: str, t0: float, t1: float, *, cache_dir: Path
    ) -> obspy.Stream: ...


class FetchedSpans(Protocol):
    """Spans of ``[t0, t1]`` the downloader requested and got an answer for, data or no data."""

    def __call__(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        t0: float,
        t1: float,
        *,
        cache_dir: Path,
    ) -> list[Span]: ...


class StageContext(Protocol):
    """The parts of H4's ``RunContext`` this module uses (docs/02 section 4)."""

    @property
    def cache_dir(self) -> Path: ...

    @property
    def config(self) -> Any: ...  # RunConfig: .run (RunSection), .signal (SignalConfig)

    def path(self, name: str) -> Path: ...

    def record(
        self,
        stage: str,
        *,
        runtime_s: float,
        counts: dict[str, int],
        params: dict | None = None,
    ) -> None: ...


# --- windows.json schema (SEIS-02 writes, SEIS-04 reads) --------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class KnownStationWindow(_Strict):
    stationId: str
    components: int  # distinct component letters among Z/N/E/1/2 on selected channels with data
    gapFraction: float  # worst selected channel; 1.0 when a selected channel has no data at all
    epiDistM: float  # catalog epicenter to station latitude/longitude
    usable: bool
    reason: str | None


class KnownEventWindow(_Strict):
    eventId: str
    t: float  # catalog origin time, epoch s UTC
    latitude: float
    longitude: float
    depthKm: float  # as published
    mag: float | None
    magType: str | None
    windowStart: float  # epoch s UTC
    windowEnd: float
    stations: list[KnownStationWindow]  # sorted by epiDistM


class KnownWindowsDoc(_Strict):
    events: list[KnownEventWindow]
    params: dict[str, Any]  # the known: config block


@dataclass(frozen=True)
class Gap:
    """One data gap on one channel, epoch s UTC, ``[gapStart, gapEnd)``."""

    stationId: str
    channel: str
    gapStart: float
    gapEnd: float


@dataclass(frozen=True)
class KnownWindows:
    """Everything :func:`assess_windows` found: the windows.json document plus the gap reports."""

    doc: dict[str, Any]
    gaps: dict[str, list[Gap]]  # eventId -> gap rows
    counts: dict[str, int]


@dataclass(frozen=True)
class _StationRow:
    id: str
    network: str
    station: str
    location: str
    latitude: float
    longitude: float
    channels: tuple[str, ...]

    @property
    def cache_key(self) -> str:
        """``NET.STA.LOC``: names exactly one location in the cache, even with an empty code."""
        return f"{self.network}.{self.station}.{self.location}"


@dataclass(frozen=True)
class _StationResult:
    entry: dict[str, Any]  # windows.json station entry, without epiDistM
    gaps: list[Gap]
    dropped: int  # traces on other channels or location codes
    notDownloadedS: float  # worst selected channel


# --- event selection --------------------------------------------------------------------------------


def select_known_events(
    catalog_df: pd.DataFrame, run: RunSection, cfg: KnownEventsConfig
) -> list[dict[str, Any]]:
    """The ``cfg.nEvents`` largest public events inside the run window and bbox.

    Window is ``[windowStart, windowEnd)``, bbox bounds are inclusive. Events without a magnitude
    are ignored and counted. Ranking is magnitude descending, then earlier origin time, then event
    id, so the result does not depend on row order. Each dict carries the catalog fields SEIS-04
    needs plus ``windowStart = t - preS`` and ``windowEnd = t + postS``. Other catalog events
    inside a selected window are logged, since their arrivals can be picked in its place.
    """
    missing = [c for c in _CATALOG_COLUMNS if c not in catalog_df.columns]
    if missing:
        raise ValueError(f"catalog is missing columns {missing}")
    df = catalog_df.loc[:, list(_CATALOG_COLUMNS)].copy()
    df["id"] = df["id"].astype(str)
    df["mag"] = pd.to_numeric(df["mag"])
    df["t"] = df["t"].astype("float64")

    run_t0, run_t1 = run.window_start_s, run.window_end_s
    min_lon, min_lat, max_lon, max_lat = run.bbox
    in_window = (df["t"] >= run_t0) & (df["t"] < run_t1)
    in_bbox = df["longitude"].between(min_lon, max_lon) & df["latitude"].between(min_lat, max_lat)
    has_mag = df["mag"].notna()
    candidates = df[in_window & in_bbox & has_mag]
    ranked = candidates.sort_values(
        ["mag", "t", "id"], ascending=[False, True, True], kind="mergesort"
    )
    chosen = ranked.head(cfg.nEvents)

    log.info(
        "known events: %d catalog rows, %d outside the run window, %d in window but outside "
        "the bbox, %d without a magnitude (ignored), %d candidates, %d selected (nEvents %d)",
        len(df),
        int((~in_window).sum()),
        int((in_window & ~in_bbox).sum()),
        int((in_window & in_bbox & ~has_mag).sum()),
        len(candidates),
        len(chosen),
        cfg.nEvents,
    )
    if len(chosen) < cfg.nEvents:
        log.warning(
            "only %d of nEvents %d public events with a magnitude in the window and bbox",
            len(chosen),
            cfg.nEvents,
        )
    _log_magnitude_types(candidates)

    events: list[dict[str, Any]] = []
    for row in chosen.itertuples(index=False):
        t = float(row.t)
        window_start, window_end = t - cfg.preS, t + cfg.postS
        event: dict[str, Any] = {
            "eventId": str(row.id),
            "t": t,
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "depthKm": float(row.depthKm),
            "mag": float(row.mag),
            "magType": None if pd.isna(row.magType) else str(row.magType),
            "windowStart": window_start,
            "windowEnd": window_end,
        }
        if window_start < run_t0 or window_end > run_t1:
            log.warning(
                "event %s: window [%s, %s] extends outside the run window; spans there that the "
                "downloader never fetched are reported as '%s'",
                event["eventId"],
                obspy.UTCDateTime(window_start),
                obspy.UTCDateTime(window_end),
                NOT_DOWNLOADED,
            )
        _warn_other_events(df, event)
        events.append(event)
    return events


def _log_magnitude_types(candidates: pd.DataFrame) -> None:
    """Count candidate magnitude types (case-folded); warn when the ranking compares across them."""
    per_type = dict(
        sorted(
            Counter(
                "none" if pd.isna(m) else str(m).casefold() for m in candidates["magType"]
            ).items()
        )
    )
    log.info("known events: candidate magnitude types %s", per_type)
    if len(per_type) > 1:
        log.warning(
            "candidates mix magnitude types %s; the ranking compares magnitudes across types",
            per_type,
        )


def _warn_other_events(df: pd.DataFrame, event: dict[str, Any]) -> None:
    """Warn about other catalog rows (any magnitude, bbox or not) inside an event's window."""
    inside = df[
        (df["t"] >= event["windowStart"])
        & (df["t"] <= event["windowEnd"])
        & (df["id"] != event["eventId"])
    ].sort_values(["t", "id"], kind="mergesort")
    if inside.empty:
        return
    listing = ", ".join(
        f"{r.id} (t{float(r.t) - event['t']:+.1f} s, mag "
        f"{'none' if pd.isna(r.mag) else f'{float(r.mag):.2f}'})"
        for r in inside.itertuples(index=False)
    )
    log.warning(
        "event %s: %d other catalog event(s) inside its window; their arrivals can be picked in "
        "place of this event's: %s",
        event["eventId"],
        len(inside),
        listing,
    )


# --- coverage ---------------------------------------------------------------------------------------


def coverage_gaps(
    traces: Sequence[obspy.Trace], t0: float, t1: float, min_gap_samples: float
) -> list[Span]:
    """Missing coverage of ``[t0, t1)`` for one channel's traces.

    A sample at time ``s`` covers ``[s, s + delta)``. A hole is a gap when it is longer than
    ``(min_gap_samples - 1) * delta``, with ``delta`` the channel's shortest sample interval:
    sub-sample timing jitter between segments is not a gap, one missing sample is. A read trimmed
    to the window starts at the first sample at or after ``t0``, up to one sample interval late,
    so the leading edge allows one extra interval. Overlapping traces are not double counted.
    """
    live = [tr for tr in traces if tr.stats.npts > 0]
    if not live:
        return [(t0, t1)]
    delta = min(float(tr.stats.delta) for tr in live)
    tol = (min_gap_samples - 1.0) * delta
    intervals = sorted(
        (
            max(tr.stats.starttime.timestamp, t0),
            min(tr.stats.endtime.timestamp + tr.stats.delta, t1),
        )
        for tr in live
    )
    gaps: list[Span] = []
    cursor = t0
    allowed = tol + delta  # leading edge
    for start, end in intervals:
        if end <= start:
            continue  # wholly outside the window
        if start - cursor > allowed:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
        allowed = tol
    if t1 - cursor > tol:
        gaps.append((cursor, t1))
    return gaps


def _merge_spans(spans: Sequence[Span]) -> list[Span]:
    merged: list[Span] = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if merged and a <= merged[-1][1] + _SPAN_EPS_S:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def split_missing(
    missing: Sequence[Span], fetched: Sequence[Span]
) -> tuple[list[Span], list[Span]]:
    """Split missing spans into (data gaps inside fetched spans, spans never fetched)."""
    covered = _merge_spans(fetched)
    inside: list[Span] = []
    outside: list[Span] = []
    for a, b in missing:
        cursor = a
        for c, d in covered:
            if d <= cursor:
                continue
            if c >= b:
                break
            if c > cursor:
                outside.append((cursor, c))
            hi = min(d, b)
            inside.append((max(c, cursor), hi))
            cursor = hi
            if cursor >= b:
                break
        if cursor < b:
            outside.append((cursor, b))
    return _drop_slivers(inside), _drop_slivers(outside)


def _drop_slivers(spans: Sequence[Span]) -> list[Span]:
    return [(a, b) for a, b in spans if b - a > _SPAN_EPS_S]


def _has_triplet(channels: Sequence[str]) -> bool:
    by_instrument: dict[str, set[str]] = {}
    for ch in channels:
        by_instrument.setdefault(ch[:-1], set()).add(ch[-1].upper())
    return any(
        _VERTICAL in letters and any(pair <= letters for pair in _HORIZONTAL_PAIRS)
        for letters in by_instrument.values()
    )


def _belongs(tr: obspy.Trace, sta: _StationRow) -> bool:
    s = tr.stats
    return (
        s.network == sta.network
        and s.station == sta.station
        and s.location == sta.location
        and s.channel in sta.channels
    )


def _assess_station(
    sta: _StationRow,
    raw: obspy.Stream,
    fetched_of: Callable[[str], list[Span]],
    t0: float,
    t1: float,
    cfg: KnownEventsConfig,
    *,
    not_cached: bool,
) -> _StationResult:
    """Coverage of one station over one window. ``fetched_of(channel)`` gives fetched spans."""
    if not_cached:
        entry = {
            "stationId": sta.id,
            "components": 0,
            "gapFraction": 1.0,
            "usable": False,
            "reason": NOT_CACHED,
        }
        return _StationResult(entry=entry, gaps=[], dropped=0, notDownloadedS=0.0)

    selected = [tr for tr in raw if _belongs(tr, sta)]
    by_channel: dict[str, list[obspy.Trace]] = {}
    for tr in selected:
        by_channel.setdefault(tr.stats.channel, []).append(tr)

    span = t1 - t0
    gaps: list[Gap] = []
    fractions: list[float] = []
    not_downloaded_s = 0.0
    for channel in sta.channels:
        missing = coverage_gaps(by_channel.get(channel, []), t0, t1, cfg.minGapSamples)
        fractions.append(sum(b - a for a, b in missing) / span)
        if not missing:
            continue
        data_gaps, never_fetched = split_missing(missing, fetched_of(channel))
        gaps.extend(Gap(sta.id, channel, a, b) for a, b in data_gaps)
        not_downloaded_s = max(not_downloaded_s, sum(b - a for a, b in never_fetched))

    letters = {ch[-1].upper() for ch in by_channel} & _COMPONENT_LETTERS
    gap_fraction = max(fractions)
    reasons: list[str] = []
    if not_downloaded_s > 0.0:
        reasons.append(f"{NOT_DOWNLOADED} ({not_downloaded_s:.1f} s of the window never fetched)")
    if not selected:
        if len(raw):
            got = ",".join(sorted({tr.id for tr in raw}))
            reasons.append(
                f"no selected channel cached (selected {sta.cache_key}"
                f".{{{','.join(sta.channels)}}}; got {got})"
            )
        elif gaps:
            reasons.append(NO_DATA)
    else:
        if not _has_triplet(list(by_channel)):
            if len(letters) < 3:
                reasons.append(f"{len(letters)} components ({''.join(sorted(letters))})")
            else:
                reasons.append(
                    "no Z + N/E or Z + 1/2 set from one instrument "
                    f"({','.join(sorted(by_channel))})"
                )
        if gap_fraction > cfg.maxGapFraction:
            reasons.append(
                f"gapFraction {gap_fraction:.3f} > maxGapFraction {cfg.maxGapFraction:.3f}"
            )
    entry = {
        "stationId": sta.id,
        "components": len(letters),
        "gapFraction": gap_fraction,
        "usable": not reasons,
        "reason": "; ".join(reasons) if reasons else None,
    }
    return _StationResult(
        entry=entry, gaps=gaps, dropped=len(raw) - len(selected), notDownloadedS=not_downloaded_s
    )


def _station_rows(stations_df: pd.DataFrame, cfg: KnownEventsConfig) -> list[_StationRow]:
    missing = [c for c in _STATION_COLUMNS if c not in stations_df.columns]
    if missing:
        raise ValueError(f"stations table is missing columns {missing}")
    df = stations_df
    if cfg.usedInRunOnly:
        keep = df["usedInRun"].astype(bool)
        log.info(
            "known windows: %d of %d stations have usedInRun = true (usedInRunOnly)",
            int(keep.sum()),
            len(df),
        )
        df = df[keep]
    duplicated = sorted(set(df.loc[df["id"].duplicated(), "id"]))
    if duplicated:
        raise ValueError(f"stations table repeats station ids {duplicated}")
    rows: list[_StationRow] = []
    for row in df.sort_values("id", kind="mergesort").itertuples(index=False):
        channels = tuple(str(c) for c in row.channels)
        if not channels:
            raise ValueError(f"station {row.id} has no channels in the stations table")
        rows.append(
            _StationRow(
                id=str(row.id),
                network=str(row.network),
                station=str(row.station),
                location="" if pd.isna(row.location) else str(row.location),  # docs/02 default ""
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                channels=channels,
            )
        )
    return rows


# --- the shared cache (SEIS-05) ---------------------------------------------------------------------


def manifest_fetched_spans(cache: ModuleType) -> FetchedSpans:
    """Fetched spans read from SEIS-05's per channel-day manifests, located by ``cache``."""

    def fetched_spans(
        network: str,
        station: str,
        location: str,
        channel: str,
        t0: float,
        t1: float,
        *,
        cache_dir: Path,
    ) -> list[Span]:
        spans: list[Span] = []
        for day in cache.days_covering(t0, t1):
            key = cache.ChannelDayKey(network, station, location, channel, day)
            path = Path(cache.manifest_path(cache_dir, key))
            if not path.is_file():
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("version") != _MANIFEST_VERSION:
                raise ValueError(
                    f"{path}: manifest version {doc.get('version')!r}, expected {_MANIFEST_VERSION}"
                )
            for chunk in doc["chunks"]:
                if chunk["status"] not in _ANSWERED:
                    raise ValueError(f"{path}: unknown chunk status {chunk['status']!r}")
                spans.append((float(chunk["start"]), float(chunk["end"])))
        return spans

    return fetched_spans


def _resolve_cache(
    read_window: ReadWindow | None,
    fetched_spans: FetchedSpans | None,
    not_cached_errors: tuple[type[Exception], ...] | None,
) -> tuple[ReadWindow, FetchedSpans, tuple[type[Exception], ...]]:
    """Fill in SEIS-05's ``read_window``, its manifests and ``CacheMissError``, imported lazily.

    ``CacheMissError`` (nothing cached at all for a station) marks that station "not cached"
    rather than failing the whole run.
    """
    if read_window is not None and fetched_spans is not None:
        return read_window, fetched_spans, not_cached_errors or ()
    try:
        cache = importlib.import_module(CACHE_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != CACHE_MODULE:
            raise
        raise RuntimeError(
            f"known windows read the shared waveform cache through {CACHE_MODULE} (SEIS-05), "
            "which is not importable here: merge SEIS-05, or pass read_window and fetched_spans"
        ) from exc
    reader: ReadWindow = read_window if read_window is not None else cache.read_window
    spans = fetched_spans if fetched_spans is not None else manifest_fetched_spans(cache)
    if not_cached_errors is None:
        miss = getattr(cache, "CacheMissError", None)
        if miss is None:
            log.warning(
                "%s defines no CacheMissError; a station with nothing cached reads as an empty "
                "stream and is reported by its manifests",
                CACHE_MODULE,
            )
        not_cached_errors = (miss,) if miss is not None else ()
    return reader, spans, not_cached_errors


def assess_windows(
    events: list[dict[str, Any]],
    stations_df: pd.DataFrame,
    cfg: KnownEventsConfig,
    cache_dir: Path,
    read_window: ReadWindow | None = None,
    *,
    fetched_spans: FetchedSpans | None = None,
    not_cached_errors: tuple[type[Exception], ...] | None = None,
) -> KnownWindows:
    """Read every (event, station) window from the cache and report coverage and gaps.

    ``read_window`` and ``fetched_spans`` default to SEIS-05's cache module and its download
    manifests. Stations are read by ``NET.STA.LOC`` so one location code is never ambiguous.
    One of ``not_cached_errors`` marks the station "not cached": counted and logged, not fatal.
    Traces on other channels or location codes are dropped and counted.
    """
    reader, spans_of, miss_errors = _resolve_cache(read_window, fetched_spans, not_cached_errors)
    stations = _station_rows(stations_df, cfg)
    counts = {
        "events": len(events),
        "stations": len(stations),
        "stationWindows": 0,
        "usable": 0,
        "notCached": 0,
        "notDownloaded": 0,
        "noData": 0,
        "droppedTraces": 0,
        "gapRows": 0,
        "eventsPass": 0,
    }
    out_events: list[dict[str, Any]] = []
    gaps_by_event: dict[str, list[Gap]] = {}
    for event in events:
        t0, t1 = float(event["windowStart"]), float(event["windowEnd"])
        entries: list[dict[str, Any]] = []
        event_gaps: list[Gap] = []
        n_not_cached = n_not_downloaded = 0
        for sta in stations:
            not_cached = False
            try:
                raw = reader(sta.cache_key, t0, t1, cache_dir=cache_dir)
            except miss_errors as exc:
                log.info("event %s station %s: not cached (%s)", event["eventId"], sta.id, exc)
                raw, not_cached = obspy.Stream(), True

            def fetched_of(
                channel: str, sta: _StationRow = sta, t0: float = t0, t1: float = t1
            ) -> list[Span]:
                return spans_of(
                    sta.network, sta.station, sta.location, channel, t0, t1, cache_dir=cache_dir
                )

            res = _assess_station(sta, raw, fetched_of, t0, t1, cfg, not_cached=not_cached)
            entry = res.entry
            entry["epiDistM"] = float(
                gps2dist_azimuth(
                    event["latitude"], event["longitude"], sta.latitude, sta.longitude
                )[0]
            )
            entries.append(entry)
            event_gaps.extend(res.gaps)
            n_not_cached += int(not_cached)
            n_not_downloaded += int(res.notDownloadedS > 0.0)
            counts["droppedTraces"] += res.dropped
            counts["noData"] += int(NO_DATA in (entry["reason"] or ""))
        entries.sort(key=lambda e: (e["epiDistM"], e["stationId"]))
        event_gaps.sort(key=lambda g: (g.stationId, g.channel, g.gapStart))
        n_usable = sum(1 for e in entries if e["usable"])
        counts["stationWindows"] += len(entries)
        counts["usable"] += n_usable
        counts["notCached"] += n_not_cached
        counts["notDownloaded"] += n_not_downloaded
        counts["gapRows"] += len(event_gaps)
        counts["eventsPass"] += int(n_usable >= cfg.minStations)
        log.info(
            "event %s: %d of %d stations usable, %d not cached, %d not fully downloaded, "
            "%d gap rows",
            event["eventId"],
            n_usable,
            len(entries),
            n_not_cached,
            n_not_downloaded,
            len(event_gaps),
        )
        out_events.append({**event, "stations": entries})
        gaps_by_event[str(event["eventId"])] = event_gaps
    if counts["droppedTraces"]:
        log.info(
            "known windows: dropped %d traces on unselected channels or location codes",
            counts["droppedTraces"],
        )
    if counts["notDownloaded"]:
        log.warning(
            "known windows: %d station windows include spans the downloader never fetched; "
            "rerun once SEIS-05 has covered them",
            counts["notDownloaded"],
        )
    doc = {"events": out_events, "params": cfg.model_dump(mode="json")}
    KnownWindowsDoc.model_validate(doc)
    return KnownWindows(doc=doc, gaps=gaps_by_event, counts=counts)


def build_windows(
    events: list[dict[str, Any]],
    stations_df: pd.DataFrame,
    cfg: KnownEventsConfig,
    cache_dir: Path,
    read_window: ReadWindow | None = None,
    *,
    fetched_spans: FetchedSpans | None = None,
    not_cached_errors: tuple[type[Exception], ...] | None = None,
) -> dict[str, Any]:
    """The windows.json document for ``events`` (see :func:`assess_windows`)."""
    return assess_windows(
        events,
        stations_df,
        cfg,
        cache_dir,
        read_window,
        fetched_spans=fetched_spans,
        not_cached_errors=not_cached_errors,
    ).doc


# --- verdicts and report ----------------------------------------------------------------------------


def usable_count(event: dict[str, Any]) -> int:
    return sum(1 for s in event["stations"] if s["usable"])


def event_passes(event: dict[str, Any], min_stations: int) -> bool:
    """Acceptance per window: at least ``min_stations`` usable three-component stations."""
    return usable_count(event) >= min_stations


def overall_pass(doc: dict[str, Any], cfg: KnownEventsConfig) -> bool:
    """All ``nEvents`` windows exist and every one passes."""
    events = doc["events"]
    return len(events) >= cfg.nEvents and all(event_passes(e, cfg.minStations) for e in events)


def format_report(doc: dict[str, Any], cfg: KnownEventsConfig) -> str:
    """Per-event station table, a PASS/FAIL line per event, and an overall line."""
    lines: list[str] = []
    events = doc["events"]
    for i, ev in enumerate(events, start=1):
        mag = "none" if ev["mag"] is None else f"{ev['mag']:.2f} {ev['magType'] or '?'}"
        lines.append(
            f"Known event {i}/{len(events)}: {ev['eventId']}  t={obspy.UTCDateTime(ev['t'])}  "
            f"mag={mag}  window=[{obspy.UTCDateTime(ev['windowStart'])}, "
            f"{obspy.UTCDateTime(ev['windowEnd'])}]"
        )
        lines.append(
            f"  {'stationId':<18}{'components':>11}{'gapFraction':>13}{'epiDistM':>11}"
            f"{'usable':>8}  reason"
        )
        for s in ev["stations"]:
            lines.append(
                f"  {s['stationId']:<18}{s['components']:>11d}{s['gapFraction']:>13.3f}"
                f"{s['epiDistM']:>11.0f}{'yes' if s['usable'] else 'no':>8}  {s['reason'] or '-'}"
            )
        verdict = "PASS" if event_passes(ev, cfg.minStations) else "FAIL"
        lines.append(
            f"{verdict} {ev['eventId']}: {usable_count(ev)} usable three-component stations "
            f"(minStations {cfg.minStations})"
        )
        lines.append("")
    n_pass = sum(1 for e in events if event_passes(e, cfg.minStations))
    lines.append(
        f"OVERALL {'PASS' if overall_pass(doc, cfg) else 'FAIL'}: {n_pass} of {len(events)} "
        f"windows pass; {len(events)} of nEvents {cfg.nEvents} events selected"
    )
    return "\n".join(lines)


# --- files ------------------------------------------------------------------------------------------


def gap_file_name(event_id: str) -> str:
    """``gaps_<eventId>.csv``, with characters outside ``[A-Za-z0-9._-]`` replaced by ``_``."""
    return f"gaps_{_UNSAFE_FILENAME.sub('_', event_id)}.csv"


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_outputs(result: KnownWindows, known_dir: Path) -> list[Path]:
    """Write one ``gaps_<eventId>.csv`` per event (header only if no gaps), then ``windows.json``.

    Every file name is checked before anything is written. Gap reports in ``known_dir`` of events
    that are no longer selected are removed. Returns ``[windows.json, gaps_*.csv...]``.
    """
    doc = KnownWindowsDoc.model_validate(result.doc).model_dump(mode="json")
    names: dict[str, str] = {}
    for ev in doc["events"]:
        event_id = ev["eventId"]
        name = gap_file_name(event_id)
        if name in names:
            raise ValueError(f"events {names[name]} and {event_id} map to the same file {name}")
        names[name] = event_id

    known_dir.mkdir(parents=True, exist_ok=True)
    gap_paths: list[Path] = []
    for name, event_id in names.items():
        rows = [
            (g.stationId, g.channel, g.gapStart, g.gapEnd) for g in result.gaps.get(event_id, [])
        ]
        text = pd.DataFrame(rows, columns=list(GAP_COLUMNS)).to_csv(
            index=False, float_format="%.6f", lineterminator="\n"
        )
        path = known_dir / name
        _write_atomic(path, text)
        gap_paths.append(path)

    stale = sorted(p for p in known_dir.glob("gaps_*.csv") if p.name not in names)
    for path in stale:
        path.unlink()
    if stale:
        log.info(
            "known/: removed %d gap reports of events no longer selected: %s",
            len(stale),
            [p.name for p in stale],
        )

    windows_path = known_dir / WINDOWS_FILE
    _write_atomic(windows_path, json.dumps(doc, indent=2, allow_nan=False) + "\n")
    return [windows_path, *gap_paths]


def load_windows(path: Path) -> dict[str, Any]:
    """Read and validate ``windows.json``; raises if the file does not match the schema."""
    text = Path(path).read_text(encoding="utf-8")
    return KnownWindowsDoc.model_validate_json(text).model_dump(mode="json")


# --- driver -----------------------------------------------------------------------------------------


def run_known_windows(
    ctx: StageContext,
    *,
    read_window: ReadWindow | None = None,
    fetched_spans: FetchedSpans | None = None,
    not_cached_errors: tuple[type[Exception], ...] | None = None,
) -> dict[str, Any]:
    """Select, assess and write the known-event windows of one run; returns windows.json.

    Reads ``catalog.parquet`` (H2) and ``stations.parquet`` (SEIS-01) from the run directory
    through ``hq_contracts.io``, prints the report and records counts and runtime.
    """
    from hq_contracts.io import from_frame, read_table
    from hq_contracts.models import CatalogEvent, Station

    started = perf_counter()
    cfg: KnownEventsConfig = ctx.config.signal.known
    run: RunSection = ctx.config.run
    catalog_path, stations_path = ctx.path("catalog.parquet"), ctx.path("stations.parquet")
    for path in (catalog_path, stations_path):
        if not path.is_file():
            raise FileNotFoundError(f"known windows need {path}")
    catalog_df = read_table(catalog_path)
    stations_df = read_table(stations_path)
    from_frame(catalog_df, CatalogEvent)  # schema check; raises on a mismatch
    from_frame(stations_df, Station)

    events = select_known_events(catalog_df, run, cfg)
    result = assess_windows(
        events,
        stations_df,
        cfg,
        ctx.cache_dir,
        read_window,
        fetched_spans=fetched_spans,
        not_cached_errors=not_cached_errors,
    )
    written = write_outputs(result, ctx.path(KNOWN_DIR))
    print(format_report(result.doc, cfg))

    runtime_s = perf_counter() - started
    log.info(
        "known windows: %d events, %d station windows, %d usable, %d not cached, "
        "%d not fully downloaded, %d gap rows, %d files written in %.2f s",
        result.counts["events"],
        result.counts["stationWindows"],
        result.counts["usable"],
        result.counts["notCached"],
        result.counts["notDownloaded"],
        result.counts["gapRows"],
        len(written),
        runtime_s,
    )
    record = write_step_record(
        ctx.path(KNOWN_DIR), STAGE, runtime_s, result.counts, cfg.model_dump(mode="json")
    )
    log.info("known windows: record written to %s", record)
    return result.doc


def write_step_record(
    out_dir: Path, step: str, runtime_s: float, counts: dict[str, int], params: dict | None
) -> Path:
    """Write ``<out_dir>/<step>.record.json`` (runtime, counts, params) atomically."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{step}{RECORD_SUFFIX}"
    tmp = path.with_name(path.name + ".part")
    payload = {"step": step, "runtimeS": runtime_s, "counts": counts, "params": params}
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


# --- CLI --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _CliConfig:
    run: RunSection
    signal: SignalConfig


@dataclass(frozen=True)
class _CliContext:
    """Stand-in for ``RunContext`` when run from the command line; ``record`` only logs."""

    run_id: str
    run_dir: Path
    cache_dir: Path
    config: _CliConfig

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
            "record %s (CLI, run.json not updated): runtime %.2f s, counts %s, params %s",
            stage,
            runtime_s,
            counts,
            params,
        )


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hq.ingest.windows",
        description="Known-event windows (SEIS-02). Exit 0 when every window passes, else 1.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="runs/<id> holding catalog.parquet and stations.parquet",
    )
    parser.add_argument(
        "--config-dir", type=Path, required=True, help="folder with run.yaml and signal.yaml"
    )
    parser.add_argument(
        "--cache-dir", type=Path, required=True, help="shared cache root (holds mseed/)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = _CliConfig(
        run=RunSection.model_validate(_load_yaml(args.config_dir / "run.yaml")),
        signal=SignalConfig.model_validate(_load_yaml(args.config_dir / "signal.yaml")),
    )
    ctx = _CliContext(
        run_id=args.run_dir.name, run_dir=args.run_dir, cache_dir=args.cache_dir, config=config
    )
    doc = run_known_windows(ctx)
    return 0 if overall_pass(doc, config.signal.known) else 1


if __name__ == "__main__":
    sys.exit(main())

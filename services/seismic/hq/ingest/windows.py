"""Known-event windows (SEIS-02): the largest public-catalog events and their station coverage.

Selects the ``known.nEvents`` largest public events inside the run window and bbox, cuts a
``[t - preS, t + postS]`` window around each, and reads it from the shared waveform cache through
``hq.ingest.cache.read_window``. Nothing is downloaded here: the full-window downloader (SEIS-05)
owns the channel-day cache, and a second writer on the same file would lose data.

For every event and station it reports the components present, the gap fraction, the epicentral
distance and whether the station is usable (three components and ``gapFraction <=
maxGapFraction``). A window PASSes when at least ``minStations`` stations are usable.

Outputs in ``runs/<id>/known/``:

- ``windows.json``: the selected events with per-station coverage, plus the ``known:`` config
  block under ``params``. SEIS-04 reads it with :func:`load_windows`.
- ``gaps_<eventId>.csv``: the gap report, one row per gap (``stationId, channel, gapStart,
  gapEnd``), including one full-window row for every selected channel with no data at all.

Not a registered pipeline stage: SEIS-04's known-event driver calls :func:`run_known_windows`,
and so does the CLI::

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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import obspy
import pandas as pd
import yaml
from obspy.geodetics import gps2dist_azimuth
from pydantic import BaseModel, ConfigDict

from hq.config.run import RunSection
from hq.config.signal import KnownEventsConfig, SignalConfig

log = logging.getLogger(__name__)

STAGE = "known_windows"  # key passed to ctx.record()
KNOWN_DIR = "known"
WINDOWS_FILE = "windows.json"
GAP_COLUMNS = ("stationId", "channel", "gapStart", "gapEnd")
NOT_CACHED = "not cached"

# Component letters that count toward "three-component": a vertical plus two horizontals, where
# borehole horizontals may be 1/2 instead of N/E. A definition, not a tunable knob.
_VERTICAL = "Z"
_HORIZONTALS = frozenset({"N", "E", "1", "2"})
_COMPONENT_LETTERS = _HORIZONTALS | {_VERTICAL}
_MIN_HORIZONTALS = 2

_CATALOG_COLUMNS = ("id", "t", "latitude", "longitude", "depthKm", "mag", "magType")
_STATION_COLUMNS = ("id", "latitude", "longitude", "channels", "usedInRun")
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


class ReadWindow(Protocol):
    """Signature of ``hq.ingest.cache.read_window`` (docs/02 section 5)."""

    def __call__(
        self, station_id: str, t0: float, t1: float, *, cache_dir: Path
    ) -> obspy.Stream: ...


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
    components: int  # distinct component letters among Z/N/E/1/2 on the selected channels
    gapFraction: float  # max over the components present; 1.0 when no data at all
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
    """One stretch of missing coverage on one channel, epoch s UTC, ``[gapStart, gapEnd)``."""

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
    latitude: float
    longitude: float
    channels: tuple[str, ...]


# --- event selection --------------------------------------------------------------------------------


def select_known_events(
    catalog_df: pd.DataFrame, run: RunSection, cfg: KnownEventsConfig
) -> list[dict[str, Any]]:
    """The ``cfg.nEvents`` largest public events inside the run window and bbox.

    Window is ``[windowStart, windowEnd)``, bbox bounds are inclusive. Events without a magnitude
    are ignored and counted. Ranking is magnitude descending, then earlier origin time, then event
    id, so the result does not depend on row order. Each dict carries the catalog fields SEIS-04
    needs plus ``windowStart = t - preS`` and ``windowEnd = t + postS``.
    """
    missing = [c for c in _CATALOG_COLUMNS if c not in catalog_df.columns]
    if missing:
        raise ValueError(f"catalog is missing columns {missing}")
    df = catalog_df.loc[:, list(_CATALOG_COLUMNS)].copy()
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
    mag_types = sorted({str(m) for m in chosen["magType"] if not pd.isna(m)})
    if len(mag_types) > 1:
        log.warning(
            "selected events mix magnitude types %s; the ranking compares across them", mag_types
        )

    events: list[dict[str, Any]] = []
    for row in chosen.itertuples(index=False):
        t = float(row.t)
        event = {
            "eventId": str(row.id),
            "t": t,
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "depthKm": float(row.depthKm),
            "mag": float(row.mag),
            "magType": None if pd.isna(row.magType) else str(row.magType),
            "windowStart": t - cfg.preS,
            "windowEnd": t + cfg.postS,
        }
        if event["windowStart"] < run_t0 or event["windowEnd"] > run_t1:
            log.warning(
                "event %s: window [%s, %s] extends outside the run window; coverage there "
                "depends on how far the downloader padded the cache",
                event["eventId"],
                obspy.UTCDateTime(event["windowStart"]),
                obspy.UTCDateTime(event["windowEnd"]),
            )
        events.append(event)
    return events


# --- coverage ---------------------------------------------------------------------------------------


def coverage_gaps(
    traces: Sequence[obspy.Trace], t0: float, t1: float, min_gap_s: float
) -> list[tuple[float, float]]:
    """Missing coverage of ``[t0, t1)`` for one channel's traces.

    A sample at time ``s`` covers ``[s, s + delta)``. Missing stretches shorter than
    ``min_gap_s`` (timing jitter between segments or at the window edges) are not gaps.
    """
    intervals = sorted(
        (
            max(tr.stats.starttime.timestamp, t0),
            min(tr.stats.endtime.timestamp + tr.stats.delta, t1),
        )
        for tr in traces
        if tr.stats.npts > 0
    )
    gaps: list[tuple[float, float]] = []
    cursor = t0
    for start, end in intervals:
        if end <= start:
            continue
        if start - cursor >= min_gap_s:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if t1 - cursor >= min_gap_s:
        gaps.append((cursor, t1))
    return gaps


def _is_three_component(letters: set[str]) -> bool:
    return _VERTICAL in letters and len(letters & _HORIZONTALS) >= _MIN_HORIZONTALS


def _assess_station(
    sta: _StationRow, raw: obspy.Stream, t0: float, t1: float, cfg: KnownEventsConfig
) -> tuple[dict[str, Any], list[Gap], int]:
    """Coverage of one station over one window: (station entry without epiDistM, gaps, dropped)."""
    selected = [tr for tr in raw if tr.stats.channel in sta.channels]
    dropped = len(raw) - len(selected)
    locations = sorted({tr.stats.location for tr in selected})
    if len(locations) > 1:
        raise ValueError(
            f"{sta.id}: read_window returned location codes {locations}; the id must name one"
        )
    by_channel: dict[str, list[obspy.Trace]] = {}
    for tr in selected:
        by_channel.setdefault(tr.stats.channel, []).append(tr)

    span = t1 - t0
    gaps: list[Gap] = []
    fractions: list[float] = []
    for channel in sta.channels:
        traces = by_channel.get(channel)
        if not traces:
            gaps.append(Gap(sta.id, channel, t0, t1))
            continue
        missing = coverage_gaps(traces, t0, t1, cfg.minGapS)
        gaps.extend(Gap(sta.id, channel, a, b) for a, b in missing)
        fractions.append(sum(b - a for a, b in missing) / span)

    letters = {ch[-1].upper() for ch in by_channel} & _COMPONENT_LETTERS
    gap_fraction = max(fractions) if fractions else 1.0  # no data covers none of the window
    reasons: list[str] = []
    if len(raw) == 0:
        reasons.append(NOT_CACHED)
    elif not selected:
        got = ",".join(sorted({tr.stats.channel for tr in raw}))
        reasons.append(f"no selected channel cached (selected {','.join(sta.channels)}; got {got})")
    else:
        if not _is_three_component(letters):
            reasons.append(f"{len(letters)} components ({''.join(sorted(letters))})")
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
    return entry, gaps, dropped


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
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                channels=channels,
            )
        )
    return rows


def _resolve_reader(
    read_window: ReadWindow | None, not_cached_errors: tuple[type[Exception], ...] | None
) -> tuple[ReadWindow, tuple[type[Exception], ...]]:
    """Default to SEIS-05's ``hq.ingest.cache.read_window``, imported lazily.

    When the cache module defines ``CacheMissError`` (nothing cached at all for a station), that
    error means "not cached" for this report rather than a failure of the whole run.
    """
    if read_window is not None:
        return read_window, not_cached_errors or ()
    cache = importlib.import_module("hq.ingest.cache")
    if not_cached_errors is None:
        miss = getattr(cache, "CacheMissError", None)
        not_cached_errors = (miss,) if miss is not None else ()
    return cache.read_window, not_cached_errors


def assess_windows(
    events: list[dict[str, Any]],
    stations_df: pd.DataFrame,
    cfg: KnownEventsConfig,
    cache_dir: Path,
    read_window: ReadWindow | None = None,
    *,
    not_cached_errors: tuple[type[Exception], ...] | None = None,
) -> KnownWindows:
    """Read every (event, station) window from the cache and report coverage and gaps.

    ``read_window`` defaults to ``hq.ingest.cache.read_window``. An empty stream, or one of
    ``not_cached_errors``, marks the station ``"not cached"``: counted and logged, not fatal.
    Traces on channels other than the station's selected ``channels`` are dropped and counted.
    """
    reader, miss_errors = _resolve_reader(read_window, not_cached_errors)
    stations = _station_rows(stations_df, cfg)
    counts = {
        "events": len(events),
        "stations": len(stations),
        "stationWindows": 0,
        "usable": 0,
        "notCached": 0,
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
        for sta in stations:
            try:
                raw = reader(sta.id, t0, t1, cache_dir=cache_dir)
            except miss_errors as exc:
                log.info("event %s station %s: not cached (%s)", event["eventId"], sta.id, exc)
                raw = obspy.Stream()
            entry, gaps, dropped = _assess_station(sta, raw, t0, t1, cfg)
            entry["epiDistM"] = float(
                gps2dist_azimuth(
                    event["latitude"], event["longitude"], sta.latitude, sta.longitude
                )[0]
            )
            entries.append(entry)
            event_gaps.extend(gaps)
            counts["droppedTraces"] += dropped
            counts["notCached"] += int(entry["reason"] == NOT_CACHED)
        entries.sort(key=lambda e: (e["epiDistM"], e["stationId"]))
        event_gaps.sort(key=lambda g: (g.stationId, g.channel, g.gapStart))
        n_usable = sum(1 for e in entries if e["usable"])
        counts["stationWindows"] += len(entries)
        counts["usable"] += n_usable
        counts["gapRows"] += len(event_gaps)
        counts["eventsPass"] += int(n_usable >= cfg.minStations)
        log.info(
            "event %s: %d of %d stations usable, %d not cached, %d gap rows",
            event["eventId"],
            n_usable,
            len(entries),
            sum(1 for e in entries if e["reason"] == NOT_CACHED),
            len(event_gaps),
        )
        out_events.append({**event, "stations": entries})
        gaps_by_event[str(event["eventId"])] = event_gaps
    if counts["droppedTraces"]:
        log.info("known windows: dropped %d traces on unselected channels", counts["droppedTraces"])
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
    not_cached_errors: tuple[type[Exception], ...] | None = None,
) -> dict[str, Any]:
    """The windows.json document for ``events`` (see :func:`assess_windows`)."""
    return assess_windows(
        events, stations_df, cfg, cache_dir, read_window, not_cached_errors=not_cached_errors
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


def write_outputs(result: KnownWindows, known_dir: Path) -> list[Path]:
    """Write ``windows.json`` and one ``gaps_<eventId>.csv`` per event (header-only if no gaps)."""
    doc = KnownWindowsDoc.model_validate(result.doc).model_dump(mode="json")
    known_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    windows_path = known_dir / WINDOWS_FILE
    windows_path.write_text(
        json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    written.append(windows_path)

    names: dict[str, str] = {}
    for ev in doc["events"]:
        event_id = ev["eventId"]
        name = gap_file_name(event_id)
        if name in names:
            raise ValueError(f"events {names[name]} and {event_id} map to the same file {name}")
        names[name] = event_id
        rows = [
            (g.stationId, g.channel, g.gapStart, g.gapEnd) for g in result.gaps.get(event_id, [])
        ]
        path = known_dir / name
        pd.DataFrame(rows, columns=list(GAP_COLUMNS)).to_csv(
            path, index=False, float_format="%.6f", lineterminator="\n"
        )
        written.append(path)

    stale = sorted(p.name for p in known_dir.glob("gaps_*.csv") if p.name not in names)
    if stale:
        log.warning("known/ holds gap reports from other events, left untouched: %s", stale)
    return written


def load_windows(path: Path) -> dict[str, Any]:
    """Read and validate ``windows.json``; raises if the file does not match the schema."""
    text = Path(path).read_text(encoding="utf-8")
    return KnownWindowsDoc.model_validate_json(text).model_dump(mode="json")


# --- driver -----------------------------------------------------------------------------------------


def run_known_windows(
    ctx: StageContext,
    *,
    read_window: ReadWindow | None = None,
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
        events, stations_df, cfg, ctx.cache_dir, read_window, not_cached_errors=not_cached_errors
    )
    written = write_outputs(result, ctx.path(KNOWN_DIR))
    print(format_report(result.doc, cfg))

    runtime_s = perf_counter() - started
    log.info(
        "known windows: %d events, %d station windows, %d usable, %d not cached, %d gap rows, "
        "%d files written in %.2f s",
        result.counts["events"],
        result.counts["stationWindows"],
        result.counts["usable"],
        result.counts["notCached"],
        result.counts["gapRows"],
        len(written),
        runtime_s,
    )
    ctx.record(STAGE, runtime_s=runtime_s, counts=result.counts, params=cfg.model_dump(mode="json"))
    return result.doc


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

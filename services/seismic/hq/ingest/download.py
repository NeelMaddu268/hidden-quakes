"""Full-window waveform download into the channel-day cache (SEIS-05).

The download unit is one (station, UTC day). A unit requests its hour chunks in order, one
``get_waveforms`` per chunk for the station's three channels, then writes each channel-day file
and its manifest once, atomically. Units run in parallel on a thread pool and never share a file
within one process; run one downloader per cache at a time. A chunk already in a manifest (and
not provisional) is never requested again, so a rerun over a cached window makes zero network
requests. Each manifest records the size of its data file, so a missing, truncated or replaced
data file is noticed and that channel-day is fetched again instead of being trusted.

Works for any window: the showcase day, a 10-minute event window, or the last two hours in Live
mode (chunks that ended less than ``provisionalLagS`` before they were fetched are refetched).
Live mode calls ``download_window`` directly with its own station list: the stage below keeps
only ``usedInRun`` stations, which SEIS-01 derives from daily availability.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Protocol

import numpy as np
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn.header import (
    FDSNBadRequestException,
    FDSNDoubleAuthenticationException,
    FDSNForbiddenException,
    FDSNInvalidRequestException,
    FDSNNoAuthenticationServiceException,
    FDSNNoDataException,
    FDSNNoServiceException,
    FDSNNotImplementedException,
    FDSNRequestTooLargeException,
    FDSNUnauthorizedException,
)

from hq.config.run import RunSection
from hq.config.signal import DownloadConfig, SignalConfig
from hq.ingest.cache import (
    CacheMissError,
    ChannelDayKey,
    Segment,
    day_of,
    manifest_path,
    mseed_dir,
    mseed_path,
    parse_name,
    window_segments,
)

log = logging.getLogger(__name__)

MANIFEST_VERSION = 1  # optional keys "fileBytes" and "stale" were added without a version bump
DAY_S = 86_400
# Half-open chunks [a, b): a sample stamped exactly at b belongs to the next chunk. miniSEED
# times resolve to 1 microsecond at best, so trimming at b minus this keeps everything before b.
_HALF_OPEN_EPS_S = 1e-6

# Fail the chunk at once: client-side request errors (retrying the same request cannot succeed;
# ObsPy maps HTTP 414 to NotImplementedError) and bugs, which backing off would only hide.
_FAIL_FAST: tuple[type[Exception], ...] = (
    FDSNBadRequestException,
    FDSNDoubleAuthenticationException,
    FDSNForbiddenException,
    FDSNInvalidRequestException,
    FDSNNoAuthenticationServiceException,
    FDSNNoServiceException,
    FDSNNotImplementedException,
    FDSNRequestTooLargeException,
    FDSNUnauthorizedException,
    NotImplementedError,
    TypeError,
    AttributeError,
    NameError,
    AssertionError,
)
# Everything else is retried with backoff: timeouts, HTTP 5xx / 429, connection resets, and
# ObsPy's decode errors on an empty or truncated response body.


class WaveformClient(Protocol):
    def get_waveforms(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
    ) -> obspy.Stream: ...


ClientFactory = Callable[[], WaveformClient]


@dataclass(frozen=True)
class StationRequest:
    """What to download for one station: its id and the channel triplet (docs/02 ``Station``)."""

    id: str
    network: str
    station: str
    location: str
    channels: tuple[str, ...]


@dataclass(frozen=True)
class Unit:
    station: StationRequest
    day: str  # YYYYMMDD
    chunks: tuple[tuple[float, float], ...]  # [start, end) epoch s, in order


@dataclass(frozen=True)
class ChunkFailure:
    stationId: str
    start: float
    end: float
    error: str

    def describe(self) -> str:
        return f"{self.stationId} {_iso(self.start)}..{_iso(self.end)}: {self.error}"


@dataclass
class UnitResult:
    stationId: str
    day: str
    cacheHit: bool
    requests: int = 0
    chunksOk: int = 0  # chunks where at least one channel returned samples
    chunksNodata: int = 0  # chunks with no samples on any channel
    samples: dict[str, int] = field(default_factory=dict)
    bytesWritten: int = 0
    runtimeS: float = 0.0
    staleChannels: list[str] = field(default_factory=list)  # cached file untrusted, refetched
    failures: list[ChunkFailure] = field(default_factory=list)


@dataclass
class DownloadResult:
    windowStart: float  # padded window actually requested
    windowEnd: float
    units: list[UnitResult]
    runtimeS: float

    @property
    def failures(self) -> list[ChunkFailure]:
        return [f for u in self.units for f in u.failures]

    def counts(self) -> dict[str, int]:
        return {
            "stations": len({u.stationId for u in self.units}),
            "units": len(self.units),
            "cacheHitUnits": sum(u.cacheHit for u in self.units),
            "requests": sum(u.requests for u in self.units),
            "chunksOk": sum(u.chunksOk for u in self.units),
            "chunksNodata": sum(u.chunksNodata for u in self.units),
            "chunksFailed": len(self.failures),
            "staleChannelDays": sum(len(u.staleChannels) for u in self.units),
            "samples": sum(sum(u.samples.values()) for u in self.units),
            "bytesWritten": sum(u.bytesWritten for u in self.units),
        }


class DownloadIncompleteError(RuntimeError):
    """Some chunks still failed after every retry. Rerunning resumes from the cache."""

    def __init__(self, result: DownloadResult) -> None:
        self.result = result
        lines = [f.describe() for f in result.failures]
        super().__init__(
            f"{len(lines)} chunk(s) failed after retries; rerun to resume from the cache:\n  "
            + "\n  ".join(lines)
        )


# --- window planning --------------------------------------------------------------------------


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")


def plan_chunks(t0: float, t1: float, chunk_s: int) -> list[tuple[float, float]]:
    """Split ``[t0, t1)`` on epoch multiples of ``chunk_s``; chunks never cross midnight UTC."""
    if DAY_S % chunk_s != 0:
        raise ValueError(f"chunkS={chunk_s} must divide a day ({DAY_S} s) evenly")
    if t1 <= t0:
        raise ValueError(f"window end {_iso(t1)} is not after start {_iso(t0)}")
    chunks: list[tuple[float, float]] = []
    a = t0
    while a < t1:
        b = min(float((math.floor(a / chunk_s) + 1) * chunk_s), t1)
        chunks.append((a, b))
        a = b
    return chunks


def plan_units(
    stations: Sequence[StationRequest], t0: float, t1: float, cfg: DownloadConfig
) -> list[Unit]:
    """(station, UTC day) units covering the padded window, ordered by day then station."""
    _check_unique_channels(stations)
    chunks = plan_chunks(t0 - cfg.padS, t1 + cfg.padS, cfg.chunkS)
    by_day: dict[str, list[tuple[float, float]]] = {}
    for chunk in chunks:
        by_day.setdefault(day_of(chunk[0]), []).append(chunk)
    return [
        Unit(station=s, day=day, chunks=tuple(day_chunks))
        for day, day_chunks in sorted(by_day.items())
        for s in stations
    ]


def _check_unique_channels(stations: Sequence[StationRequest]) -> None:
    seen: dict[tuple[str, str, str, str], str] = {}
    for s in stations:
        if not s.channels:
            raise ValueError(f"station {s.id} has no channels")
        for cha in s.channels:
            key = (s.network, s.station, s.location, cha)
            if key in seen:
                raise ValueError(f"channel {'.'.join(key)} requested by {seen[key]} and {s.id}")
            seen[key] = s.id


# --- manifests --------------------------------------------------------------------------------


def _read_manifest_doc(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        doc: dict[str, Any] = json.load(fh)
    if doc.get("version") != MANIFEST_VERSION:
        raise ValueError(f"{path}: manifest version {doc.get('version')} != {MANIFEST_VERSION}")
    return doc


def _data_problem(doc: dict[str, Any], data: Path) -> str | None:
    """Why a channel-day's data file can't be trusted against its manifest, or None."""
    if doc.get("stale"):
        return f"marked stale ({doc['stale']})"
    exists = data.is_file()
    expected = doc.get("fileBytes")
    if expected is not None:
        size = data.stat().st_size if exists else 0
        return None if size == expected else f"data file is {size} bytes, manifest says {expected}"
    # manifests written before fileBytes existed: at least the file must be there
    if not exists and any(c.get("samples") for c in doc["chunks"]):
        return "manifest lists samples but the data file is missing"
    return None


def _channel_day_state(
    cache_dir: Path, key: ChannelDayKey
) -> tuple[dict[str, Any] | None, str | None]:
    """(manifest document or None, reason the cached channel-day can't be trusted or None)."""
    data = mseed_path(cache_dir, key)
    try:
        doc = _read_manifest_doc(manifest_path(cache_dir, key))
    except json.JSONDecodeError as exc:
        return None, f"unreadable manifest ({exc})"
    if doc is None:
        return None, ("data file has no manifest" if data.is_file() else None)
    return doc, _data_problem(doc, data)


def _is_provisional(chunk: dict[str, Any], lag_s: float) -> bool:
    return float(chunk["fetchedAt"]) - float(chunk["end"]) < lag_s


def _covers(intervals: Iterable[tuple[float, float]], a: float, b: float) -> bool:
    cur = a
    for c, d in sorted(intervals):
        if c > cur + _HALF_OPEN_EPS_S:
            break
        cur = max(cur, d)
        if cur >= b - _HALF_OPEN_EPS_S:
            return True
    return cur >= b - _HALF_OPEN_EPS_S


def _subtract(chunks: list[dict[str, Any]], a: float, b: float) -> list[dict[str, Any]]:
    """Drop ``[a, b)`` from manifest entries, keeping the parts outside it.

    A split remainder's sample count is unknown, so it is set to None.
    """
    out: list[dict[str, Any]] = []
    for c in chunks:
        start, end = float(c["start"]), float(c["end"])
        if end <= a or start >= b:
            out.append(c)
            continue
        if start < a:
            out.append({**c, "end": a, "samples": None})
        if end > b:
            out.append({**c, "start": b, "samples": None})
    return out


def _cached_samples(entries: Iterable[dict[str, Any]], a: float, b: float) -> int:
    """Samples the manifest already holds for entries inside ``[a, b)`` (unknown counts as 0)."""
    return sum(
        int(c.get("samples") or 0)
        for c in entries
        if float(c["start"]) >= a and float(c["end"]) <= b
    )


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_manifest(
    path: Path,
    key: ChannelDayKey,
    chunks: list[dict[str, Any]],
    file_bytes: int | None,
    stale: str | None = None,
) -> None:
    doc: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "network": key.network,
        "station": key.station,
        "location": key.location,
        "channel": key.channel,
        "day": key.day,
        "fileBytes": file_bytes,  # size of the data file this manifest describes (0: no file)
        "chunks": sorted(chunks, key=lambda c: float(c["start"])),
    }
    if stale is not None:
        doc["stale"] = stale
    _atomic_write_text(path, json.dumps(doc, indent=1))


@dataclass
class _UnitPlan:
    todo: list[tuple[float, float]]
    entries: dict[str, list[dict[str, Any]]]  # trusted manifest entries per channel
    reset: set[str]  # channels whose cached channel-day is untrusted and rewritten from scratch


def _plan_unit(unit: Unit, cache_dir: Path, cfg: DownloadConfig) -> _UnitPlan:
    entries: dict[str, list[dict[str, Any]]] = {}
    reset: set[str] = set()
    covered: list[list[tuple[float, float]]] = []
    for cha in unit.station.channels:
        key = _key(unit, cha)
        doc, problem = _channel_day_state(cache_dir, key)
        chunks: list[dict[str, Any]] = [] if doc is None else list(doc["chunks"])
        if problem is not None:
            log.warning("%s: %s; fetching this channel-day again", key.stem, problem)
            reset.add(cha)
            chunks = []
        entries[cha] = chunks
        covered.append(
            [
                (float(c["start"]), float(c["end"]))
                for c in chunks
                if not _is_provisional(c, cfg.provisionalLagS)
            ]
        )
    todo = [ch for ch in unit.chunks if not all(_covers(cov, *ch) for cov in covered)]
    return _UnitPlan(todo=todo, entries=entries, reset=reset)


def needed_chunks(unit: Unit, cache_dir: Path, cfg: DownloadConfig) -> list[tuple[float, float]]:
    """Chunks of the unit not yet covered, on every channel, by trusted non-provisional entries."""
    return _plan_unit(unit, cache_dir, cfg).todo


def _key(unit: Unit, channel: str) -> ChannelDayKey:
    s = unit.station
    return ChannelDayKey(s.network, s.station, s.location, channel, unit.day)


# --- fetching ---------------------------------------------------------------------------------


class _ChunkFailedError(Exception):
    def __init__(self, message: str, requests: int) -> None:
        super().__init__(message)
        self.requests = requests


class _Fetcher:
    """Per-thread lazy client, so a pure cache hit never even runs service discovery."""

    def __init__(
        self, factory: ClientFactory, cfg: DownloadConfig, sleep: Callable[[float], None]
    ) -> None:
        self._factory = factory
        self._cfg = cfg
        self._sleep = sleep
        self._local = threading.local()

    def _client(self) -> WaveformClient:
        client: WaveformClient | None = getattr(self._local, "client", None)
        if client is None:
            client = self._factory()
            self._local.client = client
        return client

    def fetch(self, s: StationRequest, a: float, b: float) -> tuple[obspy.Stream | None, int]:
        """Returns (stream or None for no data, requests made). Raises _ChunkFailedError only."""
        requests = 0
        cfg = self._cfg
        for attempt in range(cfg.maxRetries + 1):
            try:
                client = self._client()
                requests += 1
                st = client.get_waveforms(
                    network=s.network,
                    station=s.station,
                    location=s.location or "--",  # FDSN spelling of an empty location code
                    channel=",".join(s.channels),
                    starttime=UTCDateTime(a),
                    endtime=UTCDateTime(b),
                )
                return st, requests
            except FDSNNoDataException:
                return None, requests
            except _FAIL_FAST as exc:
                raise _ChunkFailedError(f"{type(exc).__name__}: {exc}", requests) from exc
            except Exception as exc:  # transient (see _FAIL_FAST); retried, then listed
                if attempt == cfg.maxRetries:
                    raise _ChunkFailedError(
                        f"{type(exc).__name__} after {attempt + 1} attempts: {exc}", requests
                    ) from exc
                wait = min(cfg.backoffBaseS * 2**attempt, cfg.backoffMaxS)
                log.warning(
                    "%s %s: %s (attempt %d/%d), retrying in %.1f s",
                    s.id,
                    _iso(a),
                    type(exc).__name__,
                    attempt + 1,
                    cfg.maxRetries + 1,
                    wait,
                )
                self._sleep(wait)
        raise AssertionError("unreachable")


def _select(st: obspy.Stream, s: StationRequest, a: float, b: float) -> dict[str, obspy.Stream]:
    """Traces of the requested channels trimmed to [a, b); other ids are dropped and logged."""
    out: dict[str, obspy.Stream] = {cha: obspy.Stream() for cha in s.channels}
    foreign = 0
    for tr in st:
        ok = (
            tr.stats.network == s.network
            and tr.stats.station == s.station
            and tr.stats.location == s.location
            and tr.stats.channel in out
        )
        if not ok:
            foreign += 1
            continue
        out[tr.stats.channel].append(tr)
    if foreign:
        log.warning("%s %s: dropped %d trace(s) with other ids", s.id, _iso(a), foreign)
    for cha, cst in out.items():
        cst.trim(UTCDateTime(a), UTCDateTime(b) - _HALF_OPEN_EPS_S, nearest_sample=False)
        out[cha] = cst.split()
    return out


def _mseed_bytes(st: obspy.Stream) -> bytes:
    buf = io.BytesIO()
    st.write(buf, format="MSEED")
    return buf.getvalue()


def _cut_out(st: obspy.Stream, intervals: list[tuple[float, float]]) -> obspy.Stream:
    """Remove samples inside the half-open intervals, keeping everything else untouched."""
    out = obspy.Stream()
    ivs = sorted(intervals)
    for tr in st:
        keep_from = tr.stats.starttime.timestamp
        for a, b in ivs:
            if a > keep_from:
                piece = tr.slice(
                    UTCDateTime(keep_from), UTCDateTime(a) - _HALF_OPEN_EPS_S, nearest_sample=False
                )
                if piece.stats.npts:
                    out.append(piece.copy())
            keep_from = max(keep_from, b)
        piece = tr.slice(UTCDateTime(keep_from), None, nearest_sample=False)
        if piece.stats.npts:
            out.append(piece.copy())
    return out


def _fail(result: UnitResult, s: StationRequest, a: float, b: float, error: str) -> None:
    result.failures.append(ChunkFailure(s.id, a, b, error))
    log.error("%s %s: chunk failed: %s", s.id, _iso(a), error)


def _fetch_into_parts(
    unit: Unit,
    plan: _UnitPlan,
    handles: dict[str, IO[bytes]],
    cfg: DownloadConfig,
    fetcher: _Fetcher,
    now: Callable[[], float],
    result: UnitResult,
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, list[dict[str, Any]]]]:
    """Fetch the unit's chunks into the part files. Every per-chunk error becomes a listed
    failure; returns the chunks each channel will replace and their new manifest entries."""
    s = unit.station
    done: dict[str, list[tuple[float, float]]] = {cha: [] for cha in s.channels}
    new_entries: dict[str, list[dict[str, Any]]] = {cha: [] for cha in s.channels}
    for a, b in plan.todo:
        try:
            st, n_req = fetcher.fetch(s, a, b)
        except _ChunkFailedError as exc:
            result.requests += exc.requests
            _fail(result, s, a, b, str(exc))
            continue
        result.requests += n_req
        fetched_at = now()
        try:
            per_cha = _select(st, s, a, b) if st is not None else {}
            blobs = {cha: _mseed_bytes(cst) for cha, cst in per_cha.items() if len(cst)}
        except Exception as exc:  # a response ObsPy cannot trim or re-encode
            log.exception("%s %s: could not encode the response", s.id, _iso(a))
            error = f"could not encode the response: {type(exc).__name__}: {exc}"
            result.failures.append(ChunkFailure(s.id, a, b, error))
            continue

        take: list[tuple[str, int]] = []
        for cha in s.channels:
            n = sum(tr.stats.npts for tr in per_cha.get(cha, obspy.Stream()))
            cached = _cached_samples(plan.entries[cha], a, b)
            if n < cached:  # e.g. a transient 204 on a provisional refetch: never lose samples
                log.warning(
                    "%s %s %s: refetch returned %d samples but the cache holds %d; "
                    "keeping the cached samples (entry stays provisional)",
                    s.id,
                    cha,
                    _iso(a),
                    n,
                    cached,
                )
                continue
            take.append((cha, n))

        positions = {cha: handles[cha].tell() for cha in s.channels}
        try:
            for cha, n in take:
                if n:
                    handles[cha].write(blobs[cha])
        except OSError as exc:
            for cha, pos in positions.items():
                handles[cha].seek(pos)
                handles[cha].truncate()
            _fail(result, s, a, b, f"could not write the part file: {exc}")
            continue

        for cha, n in take:
            result.samples[cha] = result.samples.get(cha, 0) + n
            done[cha].append((a, b))
            new_entries[cha].append(
                {
                    "start": a,
                    "end": b,
                    "status": "ok" if n else "nodata",
                    "samples": n,
                    "fetchedAt": fetched_at,
                    "provisional": fetched_at - b < cfg.provisionalLagS,
                }
            )
        if blobs:
            result.chunksOk += 1
        else:
            result.chunksNodata += 1
    return done, new_entries


def run_unit(
    unit: Unit,
    cache_dir: Path,
    cfg: DownloadConfig,
    fetcher: _Fetcher,
    now: Callable[[], float],
) -> UnitResult:
    t_start = time.perf_counter()
    s = unit.station
    result = UnitResult(stationId=s.id, day=unit.day, cacheHit=False)
    plan = _plan_unit(unit, cache_dir, cfg)
    result.staleChannels = sorted(plan.reset)
    if not plan.todo:
        result.cacheHit = True
        result.runtimeS = time.perf_counter() - t_start
        log.info("%s %s: cache hit, 0 requests", s.id, unit.day)
        return result

    folder = mseed_dir(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex
    parts = {cha: folder / f"{_key(unit, cha).stem}.{tag}.part" for cha in s.channels}
    handles: dict[str, IO[bytes]] = {}
    try:
        try:
            for cha in s.channels:
                handles[cha] = parts[cha].open("wb")
            done, new_entries = _fetch_into_parts(unit, plan, handles, cfg, fetcher, now, result)
        finally:
            for fh in handles.values():
                fh.close()

        # commit whatever succeeded, even when other chunks failed: a rerun resumes from here
        for cha in s.channels:
            if not done[cha]:
                continue
            try:
                result.bytesWritten += _commit_channel(
                    cache_dir,
                    _key(unit, cha),
                    parts[cha],
                    done[cha],
                    plan.entries[cha],
                    new_entries[cha],
                    reset=cha in plan.reset,
                )
            except Exception as exc:  # e.g. PermissionError while a reader holds the file
                log.exception("%s %s: commit of %s failed", s.id, unit.day, cha)
                msg = f"commit of {cha} failed: {type(exc).__name__}: {exc}"
                result.failures.extend(ChunkFailure(s.id, a, b, msg) for a, b in done[cha])
    finally:
        for p in parts.values():
            p.unlink(missing_ok=True)

    result.runtimeS = time.perf_counter() - t_start
    log.info(
        "%s %s: %d request(s), chunks ok=%d nodata=%d failed=%d, samples=%s, %d bytes, %.1f s",
        s.id,
        unit.day,
        result.requests,
        result.chunksOk,
        result.chunksNodata,
        len(result.failures),
        result.samples,
        result.bytesWritten,
        result.runtimeS,
    )
    if result.chunksNodata and not result.chunksOk and not result.failures:
        log.warning("%s %s: no data on any channel for any requested chunk", s.id, unit.day)
    return result


def _commit_channel(
    cache_dir: Path,
    key: ChannelDayKey,
    part: Path,
    done: list[tuple[float, float]],
    old_entries: list[dict[str, Any]],
    new_entries: list[dict[str, Any]],
    *,
    reset: bool,
) -> int:
    """Write one channel-day file, then its manifest; returns the data file's size.

    Without a trusted existing file the part file simply becomes the channel-day file. Otherwise
    the old samples outside ``done`` are kept, the new ones added, and the file rewritten in time
    order.
    """
    final = mseed_path(cache_dir, key)
    has_new = part.stat().st_size > 0
    if reset or not final.is_file():
        if has_new:
            os.replace(part, final)
        else:
            final.unlink(missing_ok=True)
    else:
        st = _cut_out(obspy.read(str(final), format="MSEED"), done)
        if has_new:
            st += obspy.read(str(part), format="MSEED")
        st.sort(keys=["starttime", "endtime"])
        if len(st):
            tmp = final.with_name(f"{final.name}.{uuid.uuid4().hex}.tmp")
            try:
                with tmp.open("wb") as out:
                    st.write(out, format="MSEED")
                os.replace(tmp, final)
            finally:
                tmp.unlink(missing_ok=True)
        else:
            final.unlink(missing_ok=True)
    size = final.stat().st_size if final.is_file() else 0

    chunks = list(old_entries)
    for a, b in done:
        chunks = _subtract(chunks, a, b)
    _write_manifest(manifest_path(cache_dir, key), key, chunks + new_entries, size)
    return size


# --- public entry point -----------------------------------------------------------------------


def default_client_factory(cfg: DownloadConfig) -> ClientFactory:
    def make() -> WaveformClient:
        from obspy.clients.fdsn import Client

        client: WaveformClient = Client(cfg.client, timeout=cfg.timeoutS)
        return client

    return make


def download_window(
    stations: Sequence[StationRequest],
    t0: float,
    t1: float,
    cfg: DownloadConfig,
    *,
    cache_dir: Path,
    client_factory: ClientFactory | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
    max_workers: int | None = None,
) -> DownloadResult:
    """Fill the cache for ``[t0 - padS, t1 + padS)``; raise ``DownloadIncompleteError`` at the end
    if any chunk still failed after its retries (everything that succeeded is already cached)."""
    t_start = time.perf_counter()
    units = plan_units(stations, t0, t1, cfg)
    fetcher = _Fetcher(client_factory or default_client_factory(cfg), cfg, sleep)
    workers = max_workers if max_workers is not None else cfg.maxWorkers
    log.info(
        "download: %d station(s), %d unit(s), window %s..%s (padded by %.0f s), %d worker(s)",
        len(stations),
        len(units),
        _iso(t0 - cfg.padS),
        _iso(t1 + cfg.padS),
        cfg.padS,
        workers,
    )

    def work(u: Unit) -> UnitResult:
        try:
            return run_unit(u, cache_dir, cfg, fetcher, now)
        except Exception as exc:  # one broken unit must not discard every other unit's result
            log.exception("%s %s: unit aborted", u.station.id, u.day)
            error = f"unit aborted: {type(exc).__name__}: {exc}"
            return UnitResult(
                stationId=u.station.id,
                day=u.day,
                cacheHit=False,
                failures=[ChunkFailure(u.station.id, u.chunks[0][0], u.chunks[-1][1], error)],
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, units))
    result = DownloadResult(
        windowStart=t0 - cfg.padS,
        windowEnd=t1 + cfg.padS,
        units=results,
        runtimeS=time.perf_counter() - t_start,
    )
    counts = result.counts()
    if counts["cacheHitUnits"] == counts["units"]:
        log.info("download: pure cache hit, 0 network requests (%d units)", counts["units"])
    log.info("download: %s in %.1f s", counts, result.runtimeS)
    if result.failures:
        raise DownloadIncompleteError(result)
    return result


# --- cache verification -----------------------------------------------------------------------


@dataclass
class VerifyResult:
    channelDays: int = 0
    stamped: int = 0  # consistent channel-days whose manifest gained fileBytes
    stale: dict[str, str] = field(default_factory=dict)  # stem -> reason; refetched next run
    unordered: list[str] = field(default_factory=list)  # readable, but traces out of time order
    stray: dict[str, int] = field(default_factory=dict)  # leftover .part/.tmp files -> bytes


def _file_samples(data: Path) -> tuple[int, bool]:
    """(total samples from the record headers, traces in time order) of one data file."""
    if not data.is_file():
        return 0, True
    st = obspy.read(str(data), format="MSEED", headonly=True)
    starts = [tr.stats.starttime for tr in st]
    return sum(tr.stats.npts for tr in st), starts == sorted(starts)


def _manifest_samples(chunks: Iterable[dict[str, Any]]) -> int | None:
    total = 0
    for c in chunks:
        if c.get("samples") is None:
            return None
        total += int(c["samples"])
    return total


def verify_cache(cache_dir: Path) -> VerifyResult:
    """Check every channel-day against its manifest, e.g. before the cache is handed on.

    Compares the sample count in the file's record headers with the manifest's, which catches
    truncation. Consistent channel-days written before ``fileBytes`` existed get it stamped, so
    later runs can check them with a stat. Inconsistent ones are marked ``stale`` in their
    manifest and fetched again by the next download. Leftover temp files are listed, not deleted.
    """
    out = VerifyResult()
    folder = mseed_dir(cache_dir)
    if not folder.is_dir():
        return out
    keys: dict[str, ChannelDayKey] = {}
    for p in folder.iterdir():
        if p.name.endswith((".part", ".tmp")):
            out.stray[p.name] = p.stat().st_size
            continue
        parsed = parse_name(p.name)
        if parsed is not None:
            keys[parsed[0].stem] = parsed[0]
    for stem in sorted(keys):
        key = keys[stem]
        out.channelDays += 1
        data = mseed_path(cache_dir, key)
        doc, problem = _channel_day_state(cache_dir, key)
        if doc is not None and problem is None:
            npts, ordered = _file_samples(data)
            expected = _manifest_samples(doc["chunks"])
            if expected is not None and npts != expected:
                problem = f"data file holds {npts} samples, manifest lists {expected}"
            elif not ordered:
                out.unordered.append(stem)
        if problem is not None:
            out.stale[stem] = problem
            if doc is not None and not doc.get("stale"):
                _write_manifest(
                    manifest_path(cache_dir, key),
                    key,
                    doc["chunks"],
                    doc.get("fileBytes"),
                    stale=problem,
                )
        elif doc is not None and doc.get("fileBytes") is None:
            size = data.stat().st_size if data.is_file() else 0
            _write_manifest(manifest_path(cache_dir, key), key, doc["chunks"], size)
            out.stamped += 1
    return out


def verify_summary(res: VerifyResult) -> str:
    lines = [
        (
            f"verify: {res.channelDays} channel-day(s); fileBytes stamped on {res.stamped}; "
            f"stale {len(res.stale)}; out of time order {len(res.unordered)}; "
            f"leftover temp files {len(res.stray)}"
        )
    ]
    for stem, why in sorted(res.stale.items()):
        lines.append(f"  stale {stem}: {why} (the next download refetches it)")
    lines.extend(f"  unordered {stem}" for stem in res.unordered)
    for name, size in sorted(res.stray.items()):
        lines.append(f"  leftover {name}: {size} bytes (never read as data)")
    return "\n".join(lines)


# --- gaps and Check A -------------------------------------------------------------------------


GAP_COLUMNS = ["stationId", "channel", "gapStart", "gapEnd"]


@dataclass
class StationCoverage:
    stationId: str
    channels: list[str]
    componentsPresent: int
    gapFraction: dict[str, float]  # per expected channel
    maxGapFraction: float
    inBbox: bool | None  # reported only; None when the station list carries no coordinates
    useful: bool
    overlaps: int


def channel_gaps(
    segments: Sequence[Segment], t0: float, t1: float, min_gap_samples: float
) -> tuple[list[tuple[float, float]], int]:
    """Missing spans of one channel inside ``[t0, t1)`` and the number of overlaps.

    A sample at time t covers ``[t, t + delta)``. A hole shorter than
    ``(min_gap_samples - 1) * delta`` is timing jitter, not a gap. At the window start the
    first sample may legitimately sit anywhere in ``[t0, t0 + delta)``, so a leading gap needs
    ``(min_gap_samples - 0.5) * delta`` (one whole missing sample at the default 1.5).
    """
    gaps: list[tuple[float, float]] = []
    overlaps = 0
    if not segments:
        return [(t0, t1)], 0
    delta = min(seg.delta for seg in segments)
    tol = (min_gap_samples - 1.0) * delta
    lead_tol = (min_gap_samples - 0.5) * delta
    cursor = t0
    for i, seg in enumerate(sorted(segments, key=lambda x: x.start)):
        start = max(seg.start, t0)
        cover_end = min(seg.end + seg.delta, t1)
        if i == 0:
            if start - t0 >= lead_tol:
                gaps.append((t0, start))
        elif start - cursor > tol:
            gaps.append((cursor, start))
        elif cursor - start > tol:
            overlaps += 1
        cursor = max(cursor, cover_end)
    if t1 - cursor > tol:
        gaps.append((cursor, t1))
    return gaps, overlaps


def coverage(
    stations: Sequence[StationRequest],
    t0: float,
    t1: float,
    cfg: DownloadConfig,
    *,
    cache_dir: Path,
    bbox: tuple[float, float, float, float] | None = None,
    coords: dict[str, tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[StationCoverage]]:
    """Gap rows (``gaps.parquet`` schema) and per-station Check A coverage over ``[t0, t1)``.

    Check A's ``useful`` is exactly the lane doc's rule: three components present and every
    component's gap fraction below ``maxGapFraction``. ``inBbox`` is reported, not scored.
    """
    rows: list[dict[str, Any]] = []
    report: list[StationCoverage] = []
    span = t1 - t0
    for s in stations:
        exact_id = f"{s.network}.{s.station}.{s.location}"  # never ambiguous across locations
        try:
            segs = window_segments(exact_id, t0, t1, cache_dir=cache_dir)
        except CacheMissError:
            segs = []
        fractions: dict[str, float] = {}
        present = 0
        overlaps = 0
        for cha in s.channels:
            cha_segs = [g for g in segs if g.channel == cha]
            present += bool(cha_segs)
            gaps, n_over = channel_gaps(cha_segs, t0, t1, cfg.minGapSamples)
            overlaps += n_over
            fractions[cha] = sum(b - a for a, b in gaps) / span
            rows.extend(
                {"stationId": s.id, "channel": cha, "gapStart": a, "gapEnd": b} for a, b in gaps
            )
        if overlaps:
            log.warning("%s: %d overlap(s) in the window (not counted as gaps)", s.id, overlaps)
        worst = max(fractions.values())
        in_bbox: bool | None = None
        if bbox is not None and coords is not None and s.id in coords:
            lat, lon = coords[s.id]
            in_bbox = bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]
        useful = len(s.channels) == 3 and present == 3 and worst < cfg.maxGapFraction
        report.append(
            StationCoverage(
                stationId=s.id,
                channels=list(s.channels),
                componentsPresent=present,
                gapFraction=fractions,
                maxGapFraction=worst,
                inBbox=in_bbox,
                useful=useful,
                overlaps=overlaps,
            )
        )
    return rows, report


def check_a_table(
    report: Sequence[StationCoverage], cfg: DownloadConfig, *, title: str | None = None
) -> str:
    lines = [title] if title else []
    lines.append(
        f"{'station':<14} {'channels':<14} {'comp':>4} {'maxGap%':>8} {'inBbox':>6} {'useful':>6}"
    )
    for r in report:
        bbox = "?" if r.inBbox is None else ("yes" if r.inBbox else "no")
        lines.append(
            f"{r.stationId:<14} {','.join(r.channels):<14} {r.componentsPresent:>4} "
            f"{100 * r.maxGapFraction:>8.2f} {bbox:>6} {'yes' if r.useful else 'no':>6}"
        )
    n_useful = sum(r.useful for r in report)
    verdict = "PASS" if n_useful >= cfg.minUsefulStations else "FAIL"
    lines.append(f"useful stations: {n_useful} (need >= {cfg.minUsefulStations}) -> {verdict}")
    lines.append(
        f"useful = 3 components present and max gap fraction < {cfg.maxGapFraction:g}; "
        "inBbox is reported, not scored"
    )
    if any(r.inBbox is None for r in report):
        lines.append("inBbox '?': the station list carries no coordinates for that station")
    return "\n".join(lines)


def write_report(
    path: Path,
    result: DownloadResult | None,
    report: Sequence[StationCoverage],
    cfg: DownloadConfig,
    t0: float,
    t1: float,
) -> None:
    n_useful = sum(r.useful for r in report)
    doc = {
        "windowStart": t0,
        "windowEnd": t1,
        "usefulStations": n_useful,
        "minUsefulStations": cfg.minUsefulStations,
        "pass": n_useful >= cfg.minUsefulStations,
        "download": None if result is None else {**result.counts(), "runtimeS": result.runtimeS},
        "failures": [] if result is None else [asdict(f) for f in result.failures],
        "stations": [asdict(r) for r in report],
        "params": cfg.model_dump(mode="json"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, json.dumps(doc, indent=1))


# --- station lists ----------------------------------------------------------------------------


def station_requests(rows: Iterable[dict[str, Any]]) -> list[StationRequest]:
    return [
        StationRequest(
            id=str(r["id"]),
            network=str(r["network"]),
            station=str(r["station"]),
            location=str(r.get("location") or ""),
            channels=tuple(str(c) for c in r["channels"]),
        )
        for r in rows
    ]


def load_station_rows(path: Path) -> list[dict[str, Any]]:
    """Rows from ``stations.parquet`` (docs/02 ``Station``) or a JSON list with the same keys."""
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as fh:
            rows: list[dict[str, Any]] = json.load(fh)
        return rows
    if path.suffix == ".parquet":
        from hq_contracts.io import read_table

        df = read_table(path)
        return [{**r, "channels": list(r["channels"])} for r in df.to_dict(orient="records")]
    raise ValueError(f"station list must be .parquet or .json, got {path}")


def is_used(row: dict[str, Any]) -> bool:
    """``Station.usedInRun`` is a bool; a null (no availability answer) counts as unused."""
    value = row["usedInRun"]
    return isinstance(value, bool | np.bool_) and bool(value)


def _coords(rows: Iterable[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    return {
        str(r["id"]): (float(r["latitude"]), float(r["longitude"]))
        for r in rows
        if r.get("latitude") is not None and r.get("longitude") is not None
    }


# --- stage and CLI ----------------------------------------------------------------------------


class _StageConfig(Protocol):
    @property
    def run(self) -> RunSection: ...

    @property
    def signal(self) -> SignalConfig: ...


class StageContext(Protocol):
    """The part of H4's ``RunContext`` (docs/02 Stage API) this stage uses."""

    @property
    def cache_dir(self) -> Path: ...

    @property
    def config(self) -> _StageConfig: ...

    def path(self, name: str) -> Path: ...

    def record(
        self,
        stage: str,
        *,
        runtime_s: float,
        counts: dict[str, int],
        params: dict[str, Any] | None = None,
    ) -> None: ...


def _write_gaps(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd
    from hq_contracts.io import write_table

    df = pd.DataFrame(rows, columns=GAP_COLUMNS).astype(
        {"stationId": "string", "channel": "string", "gapStart": "float64", "gapEnd": "float64"}
    )
    write_table(df, path, "Gap")  # no Pydantic model for gaps in docs/02; H4 to confirm the name


def run(ctx: StageContext) -> None:
    """Stage ``download`` (docs/02 Stage API): stations.parquet -> cache + gaps.parquet."""
    t_start = time.perf_counter()
    cfg: DownloadConfig = ctx.config.signal.download
    window = ctx.config.run
    rows = load_station_rows(ctx.path("stations.parquet"))
    used = [r for r in rows if is_used(r)]  # SEIS-01: station has data in the window
    log.info("download stage: %d station row(s), %d with usedInRun", len(rows), len(used))
    stations = station_requests(used)
    t0, t1 = window.window_start_s, window.window_end_s
    result: DownloadResult | None
    error: DownloadIncompleteError | None = None
    try:
        result = download_window(stations, t0, t1, cfg, cache_dir=ctx.cache_dir)
    except DownloadIncompleteError as exc:
        result, error = exc.result, exc
    gap_rows, report = coverage(
        stations, t0, t1, cfg, cache_dir=ctx.cache_dir, bbox=window.bbox, coords=_coords(used)
    )
    _write_gaps(gap_rows, ctx.path("gaps.parquet"))
    write_report(ctx.path("download_report.json"), result, report, cfg, t0, t1)
    title = f"Check A (download stage): {len(stations)} usedInRun station(s) of {len(rows)}"
    log.info("%s", check_a_table(report, cfg, title=title))
    counts = {
        **result.counts(),
        "gapRows": len(gap_rows),
        "usefulStations": sum(r.useful for r in report),
    }
    ctx.record(
        "download",
        runtime_s=time.perf_counter() - t_start,
        counts=counts,
        params=cfg.model_dump(mode="json"),
    )
    if error is not None:
        raise error


def load_download_config(config_dir: Path) -> tuple[RunSection, DownloadConfig]:
    """(RunSection, DownloadConfig) from ``run.yaml`` and a fully validated ``signal.yaml``."""
    import yaml

    with (config_dir / "run.yaml").open(encoding="utf-8") as fh:
        run_section = RunSection.model_validate(yaml.safe_load(fh))
    with (config_dir / "signal.yaml").open(encoding="utf-8") as fh:
        signal = SignalConfig.model_validate(yaml.safe_load(fh))
    return run_section, signal.download


def _parse_time(text: str) -> float:
    return UTCDateTime(text).timestamp


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m hq.ingest.download", description=__doc__)
    p.add_argument("--stations", type=Path, help="stations.parquet or .json list")
    p.add_argument("--config-dir", type=Path, help="e.g. configs/showcase")
    p.add_argument("--cache-dir", type=Path, required=True, help="e.g. ../../data/cache")
    p.add_argument("--start", help="ISO UTC; default windowStart from run.yaml")
    p.add_argument("--end", help="ISO UTC; default windowEnd from run.yaml")
    p.add_argument("--run-dir", type=Path, help="write download_report.json (+ gaps.parquet)")
    p.add_argument("--report-only", action="store_true", help="skip the download, only report")
    p.add_argument(
        "--verify",
        action="store_true",
        help="only check every cached channel-day against its manifest (run before handing "
        "the cache on); exits 1 if any was marked stale",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    if args.verify:
        t_start = time.perf_counter()
        res = verify_cache(args.cache_dir)
        print(verify_summary(res))
        log.info("verify took %.1f s", time.perf_counter() - t_start)
        return 1 if res.stale else 0
    if args.stations is None or args.config_dir is None:
        p.error("--stations and --config-dir are required unless --verify is given")

    run_section, cfg = load_download_config(args.config_dir)
    t0 = _parse_time(args.start) if args.start else run_section.window_start_s
    t1 = _parse_time(args.end) if args.end else run_section.window_end_s
    rows = load_station_rows(args.stations)
    if rows and all("usedInRun" in r for r in rows):
        used = [r for r in rows if is_used(r)]
        selection = f"{len(used)} usedInRun station(s) of {len(rows)} in {args.stations.name}"
    else:
        used = rows
        selection = f"all {len(rows)} station(s) in {args.stations.name} (no usedInRun column)"
    stations = station_requests(used)
    result: DownloadResult | None = None
    error: DownloadIncompleteError | None = None
    if not args.report_only:
        try:
            result = download_window(stations, t0, t1, cfg, cache_dir=args.cache_dir)
        except DownloadIncompleteError as exc:
            result, error = exc.result, exc
    gap_rows, report = coverage(
        stations, t0, t1, cfg, cache_dir=args.cache_dir, bbox=run_section.bbox, coords=_coords(used)
    )
    title = f"Check A over {_iso(t0)}..{_iso(t1)} (unpadded): {selection}"
    print(check_a_table(report, cfg, title=title))
    code = 0
    if args.run_dir is not None:
        write_report(args.run_dir / "download_report.json", result, report, cfg, t0, t1)
        log.info("wrote %s (%d gap rows)", args.run_dir / "download_report.json", len(gap_rows))
        try:
            _write_gaps(gap_rows, args.run_dir / "gaps.parquet")
        except ImportError as exc:
            log.error("gaps.parquet NOT written, hq_contracts.io is unavailable: %s", exc)
            code = 3
    if error is not None:
        log.error("%s", error)
        code = 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())

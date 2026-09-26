"""Waveform cache: layout, reads and sample-coverage segments (SEIS-05, docs/02 section 5).

Layout (frozen; ``hq.ingest.download`` writes it, everyone else reads it through this module)::

    <cache_dir>/mseed/{net}.{sta}.{loc}.{cha}.{YYYYMMDD}.mseed   one file per channel and UTC day
    <cache_dir>/mseed/{net}.{sta}.{loc}.{cha}.{YYYYMMDD}.json    manifest: fetched chunks and
                                                                 the data file's size
    <cache_dir>/stationxml/{stationId}.xml                       response-level StationXML (SEIS-01)

An empty location code gives two adjacent dots, e.g. ``6K.CS01..GNZ.20260910.mseed``. A
channel-day file holds exactly the samples the data center returned: gaps are never filled.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import obspy
from obspy import UTCDateTime

log = logging.getLogger(__name__)

MSEED_DIR = "mseed"
STATIONXML_DIR = "stationxml"
MSEED_SUFFIX = ".mseed"
MANIFEST_SUFFIX = ".json"
DAY_FORMAT = "%Y%m%d"


class CacheMissError(LookupError):
    """Nothing at all is cached for the requested station."""


class AmbiguousStationError(LookupError):
    """A bare ``NET.STA`` id matches cached files from more than one location code."""


@dataclass(frozen=True)
class ChannelDayKey:
    network: str
    station: str
    location: str
    channel: str
    day: str  # YYYYMMDD, UTC

    @property
    def stem(self) -> str:
        return f"{self.network}.{self.station}.{self.location}.{self.channel}.{self.day}"


@dataclass(frozen=True)
class Segment:
    """One continuous run of samples: covers ``[start, end + delta)``."""

    channel: str
    start: float  # epoch s of the first sample
    end: float  # epoch s of the last sample
    delta: float  # s between samples


def mseed_dir(cache_dir: Path) -> Path:
    return Path(cache_dir) / MSEED_DIR


def mseed_path(cache_dir: Path, key: ChannelDayKey) -> Path:
    return mseed_dir(cache_dir) / f"{key.stem}{MSEED_SUFFIX}"


def manifest_path(cache_dir: Path, key: ChannelDayKey) -> Path:
    return mseed_dir(cache_dir) / f"{key.stem}{MANIFEST_SUFFIX}"


def stationxml_path(cache_dir: Path, station_id: str) -> Path:
    return Path(cache_dir) / STATIONXML_DIR / f"{station_id}.xml"


def day_of(t: float) -> str:
    """UTC day key (``YYYYMMDD``) of an epoch time."""
    return datetime.fromtimestamp(t, UTC).strftime(DAY_FORMAT)


def days_covering(t0: float, t1: float) -> list[str]:
    """Every UTC day key from the day of ``t0`` through the day of ``t1``, inclusive."""
    first = datetime.fromtimestamp(t0, UTC).date()
    last = datetime.fromtimestamp(t1, UTC).date()
    out: list[str] = []
    d = first
    while d <= last:
        out.append(d.strftime(DAY_FORMAT))
        d += timedelta(days=1)
    return out


def parse_name(name: str) -> tuple[ChannelDayKey, str] | None:
    """Split a cache file name into its key and suffix; ``None`` if it isn't a cache file."""
    for suffix in (MSEED_SUFFIX, MANIFEST_SUFFIX):
        if name.endswith(suffix):
            parts = name[: -len(suffix)].split(".")
            if len(parts) != 5 or not parts[4].isdigit() or len(parts[4]) != 8:
                return None
            return ChannelDayKey(*parts), suffix
    return None


def split_station_id(station_id: str) -> tuple[str, str, str | None]:
    """``NET.STA`` -> (net, sta, None); ``NET.STA.LOC`` -> (net, sta, loc)."""
    parts = station_id.split(".")
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"station id must be NET.STA or NET.STA.LOC, got {station_id!r}")


def station_keys(
    station_id: str, *, cache_dir: Path, days: Collection[str] | None = None
) -> list[tuple[ChannelDayKey, str]]:
    """Cached (key, suffix) pairs for one station, filtered by location when the id has one and
    by UTC day when ``days`` is given.

    Raises ``CacheMissError`` if nothing at all is cached for the station (on any day), and
    ``AmbiguousStationError`` if a bare ``NET.STA`` id spans several location codes within the
    selected days (another location cached on some other day does not make it ambiguous).
    """
    net, sta, loc = split_station_id(station_id)
    folder = mseed_dir(cache_dir)
    found: list[tuple[ChannelDayKey, str]] = []
    if folder.is_dir():
        for path in folder.glob(f"{net}.{sta}.*"):
            parsed = parse_name(path.name)
            if parsed is None:
                continue
            key, suffix = parsed
            if key.network != net or key.station != sta:
                continue
            if loc is not None and key.location != loc:
                continue
            found.append((key, suffix))
    if not found:
        raise CacheMissError(f"nothing cached for station {station_id} under {folder}")
    if days is not None:
        found = [(key, suffix) for key, suffix in found if key.day in days]
    locations = sorted({k.location for k, _ in found})
    if loc is None and len(locations) > 1:
        raise AmbiguousStationError(
            f"station id {station_id} matches location codes {locations} in the cache; "
            "use NET.STA.LOC"
        )
    return found


def _window_files(station_id: str, t0: float, t1: float, cache_dir: Path) -> list[Path]:
    if t1 < t0:
        raise ValueError(f"window end {t1} is before start {t0}")
    keys = station_keys(station_id, cache_dir=cache_dir, days=set(days_covering(t0, t1)))
    return sorted(mseed_path(cache_dir, key) for key, suffix in keys if suffix == MSEED_SUFFIX)


def read_window(station_id: str, t0: float, t1: float, *, cache_dir: Path) -> obspy.Stream:
    """Raw counts for ``[t0, t1]`` from the cache, both ends inclusive (ObsPy trim semantics).

    A sample stamped exactly at ``t1`` is returned, so consecutive windows ``[a, b]`` and
    ``[b, c]`` share that one sample; drop it on one side when concatenating. Gaps stay gaps:
    every continuous run of samples is its own trace and nothing is ever zero-filled. Contiguous
    pieces (e.g. across midnight) are joined by ObsPy's cleanup merge, which never fills. Missing
    data gives fewer or shorter traces, not fabricated samples.
    Returns every cached channel of the station; callers select ``Station.channels``.
    Read an hour or so at a time: a day of a 1,000 Hz triplet is about 1 GB in memory.
    """
    st = obspy.Stream()
    start, end = UTCDateTime(t0), UTCDateTime(t1)
    for path in _window_files(station_id, t0, t1, cache_dir):
        st += obspy.read(str(path), format="MSEED", starttime=start, endtime=end)
    st.trim(start, end, nearest_sample=False)
    st.merge(method=-1)
    st = st.split()  # a masked array would mean a gap; keep it as two traces instead
    st.sort(keys=["channel", "starttime"])
    return st


def window_segments(station_id: str, t0: float, t1: float, *, cache_dir: Path) -> list[Segment]:
    """Continuous sample runs overlapping ``[t0, t1)``, read from headers only (no samples).

    Used for gap accounting, so a day of 1,000 Hz data never has to be loaded into memory.
    Segments are not clipped: one that starts before ``t0`` or ends after ``t1`` is returned
    whole (``hq.ingest.download.channel_gaps`` clips).
    """
    raw: list[Segment] = []
    for path in _window_files(station_id, t0, t1, cache_dir):
        for tr in obspy.read(str(path), format="MSEED", headonly=True):
            raw.append(
                Segment(
                    channel=tr.stats.channel,
                    start=tr.stats.starttime.timestamp,
                    end=tr.stats.endtime.timestamp,
                    delta=tr.stats.delta,
                )
            )
    out: list[Segment] = []
    for seg in sorted(raw, key=lambda s: (s.channel, s.start)):
        if seg.end + seg.delta <= t0 or seg.start >= t1:
            continue
        out.append(seg)
    return out


def read_inventory(station_id: str, *, cache_dir: Path) -> obspy.Inventory:
    """Response-level StationXML for one station, as saved by SEIS-01."""
    path = stationxml_path(cache_dir, station_id)
    if not path.is_file():
        raise CacheMissError(f"no StationXML cached for {station_id}: expected {path}")
    return obspy.read_inventory(str(path), format="STATIONXML")

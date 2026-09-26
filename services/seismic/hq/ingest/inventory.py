"""SEIS-01 · Station inventory: public StationXML -> ``stations.parquet`` with real sensor depths.

Stage ``inventory`` (docs/02 -> Stage API). For the run window and bbox it

1. queries FDSN station metadata at channel level (raw XML cached under ``data/cache/stationxml/``),
2. forms three-component channel triplets that have a preprocessing profile and ranks them
   (velocity first, strong-motion only when a site has no usable velocity triplet),
3. resolves each chosen sensor's elevation. StationXML is inconsistent here: for some stations the
   station elevation is the wellhead surface, for others it is already the sensor. Each one is
   checked against a DEM at the sensor position. ``sensorDepthM`` is always the channel ``depth``,
   ``surfaceElevM`` is the ground surface at the sensor's site as resolved here (it can differ
   from the raw StationXML station elevation; every case is in ``inventory_report.json``), and
   ``sensorElevM = surfaceElevM - sensorDepthM`` always holds,
4. assigns ``kind``, ``preprocessProfile`` and ENU (UTM 12N minus the origin, docs/01),
5. estimates per-station data coverage of the window (MUSTANG daily availability, cached),
6. caches instrument responses per station for ``read_inventory`` (SEIS-05),

and writes ``stations.parquet`` plus ``inventory_report.json`` (every decision with its numbers).

``Station.id`` is ``NET.STA``; when two locations of one site are kept it is ``NET.STA.LOC``
with the raw location code, so an empty code gives ``NET.STA.`` (the same spelling SEIS-05 uses
for its cache lookups).

Acceptance driver::

    uv run python -m hq.ingest.inventory --config-dir configs/showcase \\
        --cache-dir ../../data/cache --run-dir ../../data/showcase/runs/<runId>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlencode

import requests
import yaml
from obspy import Inventory, UTCDateTime, read_inventory
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException
from pyproj import Geod, Transformer

from hq.config.run import Origin, RunSection
from hq.config.signal import (
    AvailabilityCheck,
    ElevationCheck,
    RateCheck,
    SignalConfig,
    StationQuery,
    StationSelection,
)

logger = logging.getLogger(__name__)

STAGE = "inventory"
REPORT_FILE = "inventory_report.json"
STATIONS_FILE = "stations.parquet"

# Cache layout, not a knob: CLAUDE.md fixes data/cache/stationxml/, and SEIS-05's
# hq.ingest.cache.read_inventory reads <cache_dir>/stationxml/<Station.id>.xml from it.
STATIONXML_DIR = "stationxml"

# docs/01 -> Conventions: horizontal ENU is UTM zone 12N metres minus the origin. A contract shared
# with every lane (SceneMeta.projection), not a tunable knob.
GEO_CRS = "EPSG:4326"
ENU_CRS = "EPSG:32612"

Kind = Literal["surface", "borehole", "strong_motion"]
Family = Literal["velocity", "accelerometer"]
# What StationXML's station elevation turned out to be: the surface, the sensor, or neither (the DEM
# at the sensor position is used as the surface).
Convention = Literal["surface", "sensor", "dem"]
Basis = Literal["dem", "both-match", "shallow", "mismatch", "assumed"]


# --- errors -------------------------------------------------------------------------------------


class InventoryError(RuntimeError):
    """Station metadata could not be turned into trustworthy Station rows."""


class AmbiguousElevationError(InventoryError):
    """The station elevation cannot be reconciled with the DEM and config says to stop."""


class DemError(InventoryError):
    """The DEM service failed or returned no data."""


class AvailabilityError(InventoryError):
    """The availability service failed, or a chosen channel has no measurement."""


# --- injectable I/O -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HttpResult:
    status: int
    text: str


HttpGet = Callable[[str, Mapping[str, str], float], HttpResult]
DemLookup = Callable[[float, float], float]  # (lat, lon) -> ground elevation, m ASL
# (network, station, location, channels, t0, t1) -> {channel: served sample rate}, or None when
# the service has no data in [t0, t1].
RateProbe = Callable[[str, str, str, Sequence[str], float, float], "dict[str, float] | None"]


class StationClient(Protocol):
    """The slice of ``obspy.clients.fdsn.Client`` this stage uses."""

    def get_stations(self, **kwargs: Any) -> Any: ...


def requests_get(url: str, params: Mapping[str, str], timeout_s: float) -> HttpResult:
    """Default HTTP GET. Tests pass their own ``HttpGet`` instead."""
    resp = requests.get(url, params=dict(params), timeout=timeout_s)
    return HttpResult(status=resp.status_code, text=resp.text)


def http_with_retries(
    http_get: HttpGet,
    url: str,
    params: Mapping[str, str],
    *,
    timeout_s: float,
    retries: int,
    backoff_s: float,
    ok: frozenset[int],
    what: str,
    error: type[InventoryError],
) -> HttpResult:
    """GET that retries transport errors, HTTP 429 and 5xx with linear backoff.

    Returns the first reply whose status is in ``ok``. Any other reply, or running out of
    attempts, raises ``error``.
    """
    last = ""
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(backoff_s * attempt)
        try:
            res = http_get(url, params, timeout_s)
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if res.status in ok:
                return res
            last = f"HTTP {res.status}: {res.text[:200]}"
            if res.status < 500 and res.status != 429:
                raise error(f"{what}: {last}")
        logger.warning("%s failed (attempt %d of %d): %s", what, attempt + 1, retries + 1, last)
    raise error(f"{what}: failed after {retries + 1} attempts: {last}")


def epqs_dem(cfg: StationSelection, http_get: HttpGet) -> DemLookup:
    """USGS 3DEP point elevation (EPQS). Retries transient failures, then raises ``DemError``."""
    ecfg = cfg.elevation

    def lookup(lat: float, lon: float) -> float:
        params = {
            "x": f"{lon}",
            "y": f"{lat}",
            "units": "Meters",
            "wkid": "4326",
            "includeDate": "false",
        }
        res = http_with_retries(
            http_get,
            ecfg.demUrl,
            params,
            timeout_s=ecfg.demTimeoutS,
            retries=ecfg.demRetries,
            backoff_s=ecfg.demBackoffS,
            ok=frozenset({200}),
            what=f"DEM lookup at lat={lat} lon={lon}",
            error=DemError,
        )
        try:
            value = float(json.loads(res.text)["value"])
        except (ValueError, KeyError, TypeError) as exc:
            raise DemError(f"unparseable DEM response at {lat},{lon}: {res.text[:200]}") from exc
        if value == ecfg.demNoDataValue:
            raise DemError(f"DEM has no data at lat={lat} lon={lon}")
        return value

    return lookup


class JsonCache:
    """A small key -> value JSON file. Reruns read it instead of the network."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.hits = 0
        self.misses = 0
        self.values: dict[str, Any] = {}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                self.values = dict(json.load(fh))

    def get(self, key: str) -> Any | None:
        if key in self.values:
            self.hits += 1
            return self.values[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        self.values[key] = value
        _write_json_atomic(self.path, dict(sorted(self.values.items())))


def _source_tagged(name: str, source: str) -> str:
    """``name`` with a short hash of ``source`` before the suffix, so sources never share a file."""
    p = Path(name)
    return f"{p.stem}_{hashlib.sha256(source.encode()).hexdigest()[:8]}{p.suffix}"


def dem_cache_path(xml_dir: Path, cfg: StationSelection) -> Path:
    return xml_dir / _source_tagged(cfg.elevation.demCacheFile, cfg.elevation.demUrl)


class DemCache:
    """DEM values cached as JSON keyed by rounded lat/lon, so reruns never touch the network."""

    def __init__(self, path: Path, lookup: DemLookup, decimals: int) -> None:
        self.store = JsonCache(path)
        self.lookup = lookup
        self.decimals = decimals

    def key(self, lat: float, lon: float) -> str:
        return f"{lat:.{self.decimals}f},{lon:.{self.decimals}f}"

    def elevation(self, lat: float, lon: float) -> float:
        k = self.key(lat, lon)
        cached = self.store.get(k)
        if cached is not None:
            return float(cached)
        value = float(self.lookup(lat, lon))
        self.store.put(k, value)
        return value


# --- channel triplets ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelRecord:
    """One channel epoch that overlaps the window, with its station epoch's position."""

    network: str
    station: str
    location: str
    code: str
    latitude: float
    longitude: float
    elevation_m: float
    depth_m: float
    sample_rate_hz: float
    overlap_s: float
    station_latitude: float
    station_longitude: float
    station_elevation_m: float


@dataclass(frozen=True)
class Triplet:
    """Three components of one band+instrument code at one location, ordered [Z, N|1, E|2]."""

    network: str
    station: str
    location: str
    code: str  # band + instrument, e.g. "GH"
    family: Family
    rank: int  # index in its family's priority list
    channels: tuple[str, str, str]
    sample_rate_hz: float
    depth_m: float
    latitude: float
    longitude: float
    channel_elev_m: float
    station_elev_m: float
    station_latitude: float
    station_longitude: float
    epoch_notes: tuple[str, ...] = ()  # channel metadata that changes inside the window

    @property
    def site(self) -> str:
        return f"{self.network}.{self.station}"

    @property
    def label(self) -> str:
        return f"{self.site}.{self.location}.{self.code}?"

    def sort_key(self) -> tuple[int, int, float, str]:
        return (
            0 if self.family == "velocity" else 1,
            self.rank,
            -self.sample_rate_hz,
            self.location,
        )


@dataclass
class SiteCandidates:
    triplets: list[Triplet] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    ignored_codes: set[str] = field(default_factory=set)


def _overlap_s(start: UTCDateTime | None, end: UTCDateTime | None, t0: float, t1: float) -> float:
    s = float(start.timestamp) if start is not None else -math.inf
    e = float(end.timestamp) if end is not None else math.inf
    return max(0.0, min(e, t1) - max(s, t0))


def _pick_epoch(epochs: list[ChannelRecord]) -> tuple[ChannelRecord, str | None]:
    """The epoch covering most of the window, plus a note when the overlapping epochs disagree."""
    ordered = sorted(epochs, key=lambda r: -r.overlap_s)
    best = ordered[0]
    if len(ordered) == 1:
        return best, None
    desc = (
        f"{best.network}.{best.station}.{best.location}.{best.code} has {len(ordered)} epochs in "
        "the window"
    )
    differ = sorted(
        {(r.depth_m, r.sample_rate_hz, r.latitude, r.longitude, r.elevation_m) for r in ordered}
    )
    if len(differ) == 1:
        logger.info(
            "%s with identical metadata; using the one covering %.0f s", desc, best.overlap_s
        )
        return best, None
    note = (
        f"{desc} with different (depth, rate, lat, lon, elev) {differ}; the one covering "
        f"{best.overlap_s:.0f} s of {sum(r.overlap_s for r in ordered):.0f} s is used for the "
        "whole window"
    )
    logger.warning("%s", note)
    return best, note


def collect_candidates(
    inv: Inventory, run: RunSection, cfg: StationSelection
) -> tuple[dict[str, SiteCandidates], dict[str, list[str]]]:
    """Group channel epochs into ranked triplets per site. Returns (sites, excluded networks)."""
    t0, t1 = run.window_start_s, run.window_end_s
    rules = cfg.channels
    vel = {c: i for i, c in enumerate(rules.velocityCodes)}
    acc = {c: i for i, c in enumerate(rules.accelerometerCodes)}
    excluded: dict[str, list[str]] = {}
    epochs: dict[str, dict[tuple[str, str], list[ChannelRecord]]] = {}
    sites: dict[str, SiteCandidates] = {}
    n_outside = n_ignored = 0
    for net in inv.networks:
        if net.code in cfg.query.excludeNetworks:
            excluded.setdefault(net.code, []).extend(s.code for s in net.stations)
            continue
        for sta in net.stations:
            site = f"{net.code}.{sta.code}"
            cand = sites.setdefault(site, SiteCandidates())
            per_chan = epochs.setdefault(site, {})
            for ch in sta.channels:
                overlap = _overlap_s(ch.start_date, ch.end_date, t0, t1)
                if overlap <= 0:
                    n_outside += 1
                    continue
                band_inst = ch.code[:2]
                if band_inst not in vel and band_inst not in acc:
                    # SOH, strain, pressure ...: never parsed further (SampleRate is optional)
                    cand.ignored_codes.add(band_inst)
                    n_ignored += 1
                    continue
                if cfg.query.skipRestricted and ch.restricted_status == "closed":
                    cand.problems.append(f"{ch.location_code}.{ch.code}: restricted")
                    continue
                values = {
                    "sample rate": ch.sample_rate,
                    "depth": ch.depth,
                    "latitude": ch.latitude,
                    "longitude": ch.longitude,
                    "elevation": ch.elevation,
                }
                missing = [k for k, v in values.items() if v is None]
                if missing:
                    cand.problems.append(
                        f"{ch.location_code}.{ch.code}: StationXML lacks {', '.join(missing)}"
                    )
                    continue
                rec = ChannelRecord(
                    network=net.code,
                    station=sta.code,
                    location=ch.location_code,
                    code=ch.code,
                    latitude=float(ch.latitude),
                    longitude=float(ch.longitude),
                    elevation_m=float(ch.elevation),
                    depth_m=float(ch.depth),
                    sample_rate_hz=float(ch.sample_rate),
                    overlap_s=overlap,
                    station_latitude=float(sta.latitude),
                    station_longitude=float(sta.longitude),
                    station_elevation_m=float(sta.elevation),
                )
                per_chan.setdefault((rec.location, rec.code), []).append(rec)
    logger.info(
        "%d channel epochs outside the window, %d with non-seismic codes: ignored",
        n_outside,
        n_ignored,
    )

    for site, per_chan in epochs.items():
        cand = sites[site]
        groups: dict[tuple[str, str], dict[str, tuple[ChannelRecord, str | None]]] = {}
        for (loc, code), recs in sorted(per_chan.items()):
            groups.setdefault((loc, code[:2]), {})[code[2:]] = _pick_epoch(recs)
        for (loc, band_inst), comps in sorted(groups.items()):
            where = f"{loc}.{band_inst}"
            vert = next((c for c in rules.verticalComponents if c in comps), None)
            pair = next((p for p in rules.horizontalPairs if p[0] in comps and p[1] in comps), None)
            if vert is None or pair is None:
                cand.problems.append(f"{where}: incomplete triplet (components {sorted(comps)})")
                continue
            picked = (comps[vert], comps[pair[0]], comps[pair[1]])
            recs3 = tuple(r for r, _ in picked)
            rates = sorted({r.sample_rate_hz for r in recs3})
            if len(rates) > 1:
                cand.problems.append(f"{where}: component sample rates differ {rates}")
                continue
            depths = [r.depth_m for r in recs3]
            if max(depths) - min(depths) > rules.componentDepthTolM:
                cand.problems.append(f"{where}: component depths differ {depths}")
                continue
            z = recs3[0]
            family: Family = "velocity" if band_inst in vel else "accelerometer"
            cand.triplets.append(
                Triplet(
                    network=z.network,
                    station=z.station,
                    location=loc,
                    code=band_inst,
                    family=family,
                    rank=vel[band_inst] if family == "velocity" else acc[band_inst],
                    channels=(recs3[0].code, recs3[1].code, recs3[2].code),
                    sample_rate_hz=rates[0],
                    depth_m=z.depth_m,
                    latitude=z.latitude,
                    longitude=z.longitude,
                    channel_elev_m=z.elevation_m,
                    station_elev_m=z.station_elevation_m,
                    station_latitude=z.station_latitude,
                    station_longitude=z.station_longitude,
                    epoch_notes=tuple(n for _, n in picked if n is not None),
                )
            )
    return sites, excluded


def classify_kind(depth_m: float, code: str, cfg: StationSelection) -> Kind:
    """Borehole by depth first; then strong-motion by instrument code; else surface."""
    if depth_m >= cfg.kind.boreholeMinDepthM:
        return "borehole"
    if code[1:2] in cfg.kind.strongMotionInstrumentCodes:
        return "strong_motion"
    return "surface"


def match_profile(rate_hz: float, cfg: StationSelection) -> str | None:
    """First profile rule whose inclusive rate range contains ``rate_hz``."""
    for rule in cfg.profiles:
        if rule.minRateHz <= rate_hz <= rule.maxRateHz:
            return rule.profile
    return None


@dataclass(frozen=True)
class Chosen:
    id: str
    triplet: Triplet
    kind: Kind
    profile: str


def choose_triplets(
    sites: Mapping[str, SiteCandidates], cfg: StationSelection
) -> tuple[list[Chosen], list[dict[str, str]], list[dict[str, str]]]:
    """Pick the triplets that become Station rows. Returns (chosen, skipped sites, dropped).

    Triplets whose sample rate has no preprocessing profile are dropped first, so an unusable
    velocity triplet never hides a usable accelerometer at the same site.
    """
    chosen: list[Chosen] = []
    skipped: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    tol = cfg.channels.locationDepthTolM
    for site in sorted(sites):
        cand = sites[site]
        for problem in cand.problems:
            logger.info("%s: %s", site, problem)
        viable: list[tuple[Triplet, str]] = []
        unprofiled: list[str] = []
        for t in cand.triplets:
            profile = match_profile(t.sample_rate_hz, cfg)
            if profile is None:
                reason = f"no preprocess profile for {t.sample_rate_hz:g} Hz"
                dropped.append({"triplet": t.label, "reason": reason})
                unprofiled.append(f"{t.location}.{t.code}: {reason}")
                continue
            viable.append((t, profile))
        velocity = [v for v in viable if v[0].family == "velocity"]
        pool = velocity or [v for v in viable if v[0].family == "accelerometer"]
        if not pool:
            parts = [*cand.problems, *unprofiled]
            if cand.ignored_codes:
                parts.append(
                    "no velocity or accelerometer codes (ignored: "
                    + ", ".join(sorted(cand.ignored_codes))
                    + ")"
                )
            reason = "; ".join(parts) or "no channel epoch overlaps the window"
            skipped.append({"site": site, "reason": reason})
            logger.info("skip %s: %s", site, reason)
            continue
        if velocity:
            for t, _ in viable:
                if t.family == "accelerometer":
                    dropped.append(
                        {"triplet": t.label, "reason": "accelerometer; site has a velocity triplet"}
                    )
        best_per_loc: dict[str, tuple[Triplet, str]] = {}
        for t, profile in sorted(pool, key=lambda v: v[0].sort_key()):
            if t.location in best_per_loc:
                better = best_per_loc[t.location][0]
                dropped.append(
                    {"triplet": t.label, "reason": f"lower priority than {better.label}"}
                )
                continue
            best_per_loc[t.location] = (t, profile)
        kept: list[tuple[Triplet, Kind, str]] = []
        for t, profile in sorted(best_per_loc.values(), key=lambda v: v[0].sort_key()):
            kind = classify_kind(t.depth_m, t.code, cfg)
            twin = next(
                (k for k, kk, _ in kept if kk == kind and abs(k.depth_m - t.depth_m) <= tol), None
            )
            if twin is not None:
                dropped.append(
                    {
                        "triplet": t.label,
                        "reason": f"same kind ({kind}) and depth as {twin.label}",
                    }
                )
                continue
            kept.append((t, kind, profile))
        for t, kind, profile in kept:
            # docs/02: "append .loc only if two locations coexist". The raw code is appended, so
            # an empty location gives "NET.STA." (SEIS-05 builds the same exact id).
            sid = site if len(kept) == 1 else f"{site}.{t.location}"
            chosen.append(Chosen(id=sid, triplet=t, kind=kind, profile=profile))
    for d in dropped:
        logger.info("not used %s: %s", d["triplet"], d["reason"])
    return chosen, skipped, dropped


# --- elevation and ENU --------------------------------------------------------------------------


@dataclass(frozen=True)
class ElevationDecision:
    convention: Convention
    basis: Basis
    surface_elev_m: float
    sensor_elev_m: float
    station_elev_m: float
    channel_elev_m: float
    depth_m: float
    dem_m: float
    flag: str | None = None  # set when a human should look at this station

    @property
    def label(self) -> str:
        return f"{self.convention}({self.basis})"


def resolve_elevation(
    station_elev_m: float,
    channel_elev_m: float,
    depth_m: float,
    dem_m: float,
    cfg: ElevationCheck,
    *,
    what: str,
) -> ElevationDecision | None:
    """Decide whether the station elevation is the site surface, the sensor, or neither.

    Two readings of StationXML's station elevation, each checked against the DEM at the sensor:
    ``surface`` (surfaceElevM = stationElev, sensorElevM = stationElev - depth) and ``sensor``
    (surfaceElevM = stationElev + depth, sensorElevM = stationElev). They differ by ``depth``.

    - One reading within ``toleranceM`` of the DEM: that one.
    - Both within tolerance (possible only for depth <= 2 tol): for depth <= tol the choice moves
      the sensor by at most tol, so ``surface`` is used; deeper, the closer one wins and the
      station is flagged, because the other reading would be off by the full depth.
    - Neither, and depth <= tol: the StationXML value is kept and flagged when it misses the DEM
      by at most ``maxShallowMismatchM``; beyond that ``onShallowMismatch`` decides.
    - Neither, and deeper: ambiguous, handled by ``onAmbiguous``.

    Returns None only when the config says to skip the station.
    """
    tol = cfg.toleranceM
    r_surface = station_elev_m - dem_m
    r_sensor = station_elev_m + depth_m - dem_m
    surface_ok = abs(r_surface) <= tol
    sensor_ok = abs(r_sensor) <= tol
    numbers = (
        f"stationElev {station_elev_m:.1f}, channelElev {channel_elev_m:.1f}, depth "
        f"{depth_m:.1f}, DEM {dem_m:.1f}"
    )
    convention: Convention
    basis: Basis
    flag: str | None = None
    if surface_ok and sensor_ok:
        basis = "both-match"
        if depth_m <= tol:
            convention = "surface"
        else:
            convention = "sensor" if abs(r_sensor) < abs(r_surface) else "surface"
            flag = (
                f"{what}: both readings of the station elevation are within {tol:g} m of the DEM "
                f"({numbers}); used the closer one ({convention}); the other would move the "
                f"sensor {depth_m:.1f} m"
            )
    elif surface_ok:
        convention, basis = "surface", "dem"
    elif sensor_ok:
        convention, basis = "sensor", "dem"
    elif depth_m <= tol:
        miss = (
            f"{what}: station elevation {station_elev_m:.1f} m differs from DEM {dem_m:.1f} m by "
            f"{r_surface:+.1f} m ({numbers})"
        )
        if abs(r_surface) <= cfg.maxShallowMismatchM:
            convention, basis = "surface", "shallow"
            flag = (
                f"{miss}; within maxShallowMismatchM {cfg.maxShallowMismatchM:g}, kept StationXML"
            )
        else:
            action = cfg.onShallowMismatch
            miss = f"{miss}; more than maxShallowMismatchM {cfg.maxShallowMismatchM:g}"
            if action == "error":
                raise AmbiguousElevationError(f"{miss} (onShallowMismatch=error)")
            if action == "skip":
                logger.warning("%s; skipping (onShallowMismatch=skip)", miss)
                return None
            if action == "keep":
                convention, basis = "surface", "shallow"
                flag = f"{miss}; kept StationXML (onShallowMismatch=keep)"
            else:
                convention, basis = "dem", "mismatch"
                flag = f"{miss}; used the DEM as the surface (onShallowMismatch=dem)"
    else:
        msg = (
            f"{what}: ambiguous elevation ({numbers}): neither stationElev (surface) nor "
            f"stationElev + depth (sensor) is within {tol:.1f} m of the DEM"
        )
        action_a = cfg.onAmbiguous
        if action_a == "error":
            raise AmbiguousElevationError(msg)
        if action_a == "skip":
            logger.warning("%s; skipping (onAmbiguous=skip)", msg)
            return None
        convention, basis = ("surface" if action_a == "surface" else "dem"), "assumed"
        flag = f"{msg}; used the {convention} reading (onAmbiguous={action_a})"
    surface = {
        "surface": station_elev_m,
        "sensor": station_elev_m + depth_m,
        "dem": dem_m,
    }[convention]
    decision = ElevationDecision(
        convention=convention,
        basis=basis,
        surface_elev_m=surface,
        sensor_elev_m=surface - depth_m,
        station_elev_m=station_elev_m,
        channel_elev_m=channel_elev_m,
        depth_m=depth_m,
        dem_m=dem_m,
        flag=flag,
    )
    logger.info(
        "%s: elevation %s: %s -> surfaceElevM %.1f, sensorElevM %.1f",
        what,
        decision.label,
        numbers,
        decision.surface_elev_m,
        decision.sensor_elev_m,
    )
    if flag is not None:
        logger.warning("FLAG %s", flag)
    return decision


class EnuProjector:
    """Geographic -> ENU: UTM 12N metres minus the origin; ``u = elevM - origin.elevM``."""

    def __init__(self, origin: Origin) -> None:
        self.origin = origin
        self._tf = Transformer.from_crs(GEO_CRS, ENU_CRS, always_xy=True)
        self._e0, self._n0 = self._tf.transform(origin.lon, origin.lat)

    def __call__(self, lat: float, lon: float, elev_m: float) -> dict[str, float]:
        e, n = self._tf.transform(lon, lat)
        return {
            "e": float(e - self._e0),
            "n": float(n - self._n0),
            "u": float(elev_m - self.origin.elevM),
        }


# --- availability -------------------------------------------------------------------------------


def _window_days(t0: float, t1: float) -> list[tuple[datetime, float]]:
    """UTC days touching [t0, t1) with the fraction of the window each day covers."""
    start = datetime.fromtimestamp(t0, UTC)
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    out: list[tuple[datetime, float]] = []
    span = t1 - t0
    while day.timestamp() < t1:
        nxt = day + timedelta(days=1)
        overlap = min(nxt.timestamp(), t1) - max(day.timestamp(), t0)
        if overlap > 0:
            out.append((day, overlap / span))
        day = nxt
    return out


def _availability_rows(res: HttpResult, metric: str, what: str) -> list[dict[str, Any]]:
    if res.status == 204:
        return []
    try:
        rows = json.loads(res.text)["measurements"].get(metric, [])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise AvailabilityError(f"{what}: unparseable reply: {res.text[:200]}") from exc
    if not isinstance(rows, list):
        raise AvailabilityError(f"{what}: unexpected reply: {res.text[:200]}")
    return rows


def fetch_coverage(
    chosen: Sequence[Chosen],
    run: RunSection,
    acfg: AvailabilityCheck,
    http_get: HttpGet,
    cache: JsonCache,
) -> dict[str, dict[str, float | None]]:
    """Per station id, per channel: fraction of the window with data (None = no measurement).

    MUSTANG publishes one availability value per channel-day, so a partial-day window gets the
    overlap-weighted daily values: an estimate that assumes data is spread evenly through each
    day. SEIS-05 measures real gaps after download. Replies (including 204 "no measurement") are
    cached under the query, so a rerun gives the same answer without the network.
    """
    days = _window_days(run.window_start_s, run.window_end_s)
    tw = f"{days[0][0]:%Y-%m-%dT%H:%M:%S},{days[-1][0] + timedelta(days=1):%Y-%m-%dT%H:%M:%S}"
    out: dict[str, dict[str, float | None]] = {}
    for c in chosen:
        t = c.triplet
        params = {
            "metric": acfg.metric,
            "net": t.network,
            "sta": t.station,
            "loc": t.location or "--",
            "cha": ",".join(t.channels),
            "timewindow": tw,
            "format": "json",
        }
        what = f"{c.id}: availability query"
        key = f"{acfg.url}?{urlencode(sorted(params.items()))}"
        cached = cache.get(key)
        if cached is not None:
            res = HttpResult(status=int(cached["status"]), text=str(cached["text"]))
            rows = _availability_rows(res, acfg.metric, what)
        else:
            res = http_with_retries(
                http_get,
                acfg.url,
                params,
                timeout_s=acfg.timeoutS,
                retries=acfg.retries,
                backoff_s=acfg.backoffS,
                ok=frozenset({200, 204}),
                what=what,
                error=AvailabilityError,
            )
            rows = _availability_rows(res, acfg.metric, what)  # parse before caching
            cache.put(key, {"status": res.status, "text": res.text})
        daily: dict[tuple[str, str], float] = {}
        for row in rows:
            if row.get("loc", "") != t.location or row.get("cha") not in t.channels:
                continue
            day_key = (str(row["cha"]), str(row["start"])[:10])
            # one row per quality code; the best one says whether data exists at all
            daily[day_key] = max(daily.get(day_key, 0.0), float(row["value"]) / 100.0)
        per_chan: dict[str, float | None] = {}
        for cha in t.channels:
            total = 0.0
            missing = False
            for day, weight in days:
                v = daily.get((cha, f"{day:%Y-%m-%d}"))
                if v is None:
                    missing = True
                    break
                total += v * weight
            if missing:
                msg = f"{c.id}: no {acfg.metric} measurement for {cha} in {tw}"
                if acfg.onMissing == "error":
                    raise AvailabilityError(msg)
                logger.warning("%s; coverage unknown (onMissing=%s)", msg, acfg.onMissing)
                per_chan[cha] = None
            else:
                per_chan[cha] = min(1.0, total)
        out[c.id] = per_chan
    logger.info("availability: %d cache hits, %d queried", cache.hits, cache.misses)
    return out


def station_coverage(per_chan: Mapping[str, float | None]) -> float | None:
    """Station coverage is the worst of its three components (three components or nothing)."""
    vals = list(per_chan.values())
    if any(v is None for v in vals):
        return None
    return min(v for v in vals if v is not None)


# --- StationXML fetch + cache -------------------------------------------------------------------


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


class _LazyClient:
    """Builds the FDSN client on first use, so a pure cache hit makes no network call."""

    def __init__(self, cfg: StationSelection, client: StationClient | None) -> None:
        self._cfg = cfg
        self._client = client

    def get(self) -> StationClient:
        if self._client is None:
            from obspy.clients.fdsn import Client

            self._client = Client(self._cfg.query.fdsnClient, timeout=self._cfg.query.timeoutS)
        return self._client


def _fetch_to_file(client: _LazyClient, qcfg: StationQuery, path: Path, **query: Any) -> None:
    """One station-service query saved to ``path``; retries transport errors and timeouts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    what = f"station query ({query.get('level')}) for {path.name}"
    last = ""
    for attempt in range(qcfg.retries + 1):
        if attempt:
            time.sleep(qcfg.backoffS * attempt)
        try:
            client.get().get_stations(filename=str(tmp), **query)
        except FDSNNoDataException as exc:
            raise InventoryError(f"{what}: the station service has no data ({exc})") from exc
        except (FDSNException, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "%s failed (attempt %d of %d): %s", what, attempt + 1, qcfg.retries + 1, last
            )
            continue
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise InventoryError(f"{what}: the station service returned nothing")
        os.replace(tmp, path)
        return
    raise InventoryError(f"{what}: failed after {qcfg.retries + 1} attempts: {last}")


def channel_query(run: RunSection, cfg: StationSelection) -> dict[str, Any]:
    q = cfg.query
    min_lon, min_lat, max_lon, max_lat = run.bbox
    return {
        "network": q.network,
        "station": q.station,
        "location": q.location,
        "channel": q.channel,
        "minlatitude": min_lat,
        "maxlatitude": max_lat,
        "minlongitude": min_lon,
        "maxlongitude": max_lon,
        "starttime": UTCDateTime(run.window_start_s),
        "endtime": UTCDateTime(run.window_end_s),
        "level": "channel",
    }


def channel_xml_path(run: RunSection, cfg: StationSelection, xml_dir: Path) -> Path:
    """Deterministic cache name derived from every query parameter."""
    params = {k: str(v) for k, v in channel_query(run, cfg).items()}
    params["client"] = cfg.query.fdsnClient
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    return xml_dir / f"channels_{digest}.xml"


def load_channel_inventory(
    run: RunSection, cfg: StationSelection, xml_dir: Path, client: _LazyClient
) -> tuple[Inventory, Path]:
    path = channel_xml_path(run, cfg, xml_dir)
    if path.exists():
        logger.info("station XML cache hit: %s", path)
    else:
        query = channel_query(run, cfg)
        logger.info(
            "querying %s station metadata (channel level) -> %s", cfg.query.fdsnClient, path
        )
        _fetch_to_file(client, cfg.query, path, **query)
        sidecar = {k: str(v) for k, v in query.items()} | {"client": cfg.query.fdsnClient}
        _write_json_atomic(path.with_suffix(".json"), sidecar)
    return read_inventory(str(path), format="STATIONXML"), path


def _uncovered_channels(path: Path, t: Triplet, t0: float, t1: float) -> list[str]:
    """Channels of ``t`` the cached XML lacks a full response for, over an epoch in [t0, t1)."""
    cached = read_inventory(str(path), format="STATIONXML")
    ok = {
        ch.code
        for net in cached.networks
        if net.code == t.network
        for sta in net.stations
        if sta.code == t.station
        for ch in sta.channels
        if ch.location_code == t.location
        and ch.code in t.channels
        and ch.response is not None
        and len(ch.response.response_stages) > 0
        and _overlap_s(ch.start_date, ch.end_date, t0, t1) > 0
    }
    return [c for c in t.channels if c not in ok]


def ensure_responses(
    chosen: Sequence[Chosen],
    run: RunSection,
    cfg: StationSelection,
    xml_dir: Path,
    client: _LazyClient,
) -> tuple[int, int]:
    """``<xml_dir>/<Station.id>.xml`` at response level for each chosen triplet. (hits, fetched)

    A cached file counts only if every chosen channel has response stages on an epoch that
    overlaps the window; otherwise it is fetched again for this window.
    """
    hits = fetched = 0
    t0, t1 = run.window_start_s, run.window_end_s
    for c in chosen:
        t = c.triplet
        path = xml_dir / f"{c.id}.xml"
        if path.exists():
            missing = _uncovered_channels(path, t, t0, t1)
            if not missing:
                hits += 1
                continue
            logger.info(
                "%s: cached response XML lacks a response covering the window for %s; refetching",
                c.id,
                missing,
            )
        _fetch_to_file(
            client,
            cfg.query,
            path,
            network=t.network,
            station=t.station,
            location=t.location or "--",
            channel=",".join(t.channels),
            starttime=UTCDateTime(t0),
            endtime=UTCDateTime(t1),
            level="response",
        )
        fetched += 1
    logger.info("response XML: %d cache hits, %d fetched", hits, fetched)
    return hits, fetched


# --- the stage ----------------------------------------------------------------------------------


@dataclass
class InventoryResult:
    rows: list[dict[str, Any]]  # docs/02 Station fields, sorted by id
    report: dict[str, Any]
    counts: dict[str, int]


def build_inventory(
    run: RunSection,
    cfg: StationSelection,
    cache_dir: Path,
    *,
    client: StationClient | None = None,
    dem: DemLookup | None = None,
    http_get: HttpGet | None = None,
    rate_probe: RateProbe | None = None,
) -> InventoryResult:
    """Everything except writing run outputs. Network access only through the injected callables
    (or their defaults), and only on cache misses (StationXML, DEM, availability, responses,
    dataselect sample-rate probes)."""
    get = http_get or requests_get
    lazy = _LazyClient(cfg, client)
    xml_dir = cache_dir / STATIONXML_DIR
    inv, xml_path = load_channel_inventory(run, cfg, xml_dir, lazy)

    sites, excluded = collect_candidates(inv, run, cfg)
    for net_code, stas in sorted(excluded.items()):
        logger.info("network %s excluded by config (%d stations)", net_code, len(stas))
    chosen, skipped, dropped = choose_triplets(sites, cfg)

    dem_cache = DemCache(
        dem_cache_path(xml_dir, cfg), dem or epqs_dem(cfg, get), cfg.elevation.demKeyDecimals
    )
    project = EnuProjector(run.origin)
    geod = Geod(ellps="WGS84")
    flags: list[dict[str, str]] = []
    resolved: list[tuple[Chosen, ElevationDecision]] = []
    for c in chosen:
        t = c.triplet
        if t.code in cfg.kind.boreholeLookingCodes and t.depth_m < cfg.kind.boreholeMinDepthM:
            msg = (
                f"{t.label} looks like a borehole geophone but sensorDepthM is {t.depth_m:g} "
                f"(< boreholeMinDepthM {cfg.kind.boreholeMinDepthM:g}); check its StationXML"
            )
            logger.warning("FLAG %s", msg)
            flags.append({"station": c.id, "flag": msg})
        _, _, dist_m = geod.inv(t.station_longitude, t.station_latitude, t.longitude, t.latitude)
        if dist_m > cfg.channels.coordTolM:
            msg = f"{t.label} channel position is {dist_m:.0f} m from the station position"
            logger.warning("FLAG %s", msg)
            flags.append({"station": c.id, "flag": msg})
        flags.extend({"station": c.id, "flag": note} for note in t.epoch_notes)
        dem_m = dem_cache.elevation(t.latitude, t.longitude)
        decision = resolve_elevation(
            t.station_elev_m, t.channel_elev_m, t.depth_m, dem_m, cfg.elevation, what=t.label
        )
        if decision is None:
            skipped.append({"site": c.id, "reason": f"{t.label}: elevation skipped by config"})
            continue
        if decision.flag is not None:
            flags.append({"station": c.id, "flag": decision.flag})
        resolved.append((c, decision))
    logger.info("DEM: %d cache hits, %d lookups", dem_cache.store.hits, dem_cache.store.misses)

    kept = [c for c, _ in resolved]
    avail_cache = JsonCache(xml_dir / cfg.availability.cacheFile)
    coverage = fetch_coverage(kept, run, cfg.availability, get, avail_cache)
    ensure_responses(kept, run, cfg, xml_dir, lazy)

    rcfg = cfg.rateCheck
    rate_cache = JsonCache(xml_dir / rcfg.cacheFile)
    probe = rate_probe or fdsn_rate_probe(cfg.query, rcfg)
    rates: dict[str, RateDecision] = {}
    rate_ok: list[tuple[Chosen, ElevationDecision]] = []
    mismatches = 0
    for c, d in resolved:
        t = c.triplet
        if station_coverage(coverage[c.id]) == 0:  # measured empty: usedInRun false, skip probe
            rd = RateDecision(t.sample_rate_hz, None, (), "unprobed")
        else:
            rd = check_rate(c, run, cfg.query, rcfg, probe, rate_cache)
        rates[c.id] = rd
        if rd.status == "nodata":
            if rcfg.onNoData == "error":
                raise InventoryError(f"{t.label}: no data at any rate probe {rcfg.probeOffsetsS}")
            flags.append(
                {
                    "station": c.id,
                    "flag": (
                        f"{t.label}: sample rate unverified (no data at any probe); kept "
                        f"StationXML {rd.metadata_hz:g} Hz"
                    ),
                }
            )
        elif rd.status == "mismatch":
            assert rd.data_hz is not None
            mismatches += 1
            msg = (
                f"{t.label}: StationXML says {rd.metadata_hz:g} Hz but the service serves "
                f"{rd.data_hz:g} Hz at all {len(rd.probe_times)} probe(s) with data"
            )
            if rcfg.onMismatch == "error":
                raise InventoryError(msg)
            if rcfg.onMismatch == "skip":
                logger.warning("FLAG %s; station skipped (onMismatch=skip)", msg)
                skipped.append({"site": c.id, "reason": f"{msg}; onMismatch=skip"})
                continue
            profile = match_profile(rd.data_hz, cfg)
            if profile is None:
                logger.warning("FLAG %s; no profile for the served rate, station skipped", msg)
                skipped.append({"site": c.id, "reason": f"{msg}; no profile for the served rate"})
                continue
            note = (
                f"{msg}; profile {c.profile} -> {profile}; the cached response XML still "
                f"describes the {rd.metadata_hz:g} Hz epoch"
            )
            logger.warning("FLAG %s", note)
            flags.append({"station": c.id, "flag": note})
            c = Chosen(id=c.id, triplet=c.triplet, kind=c.kind, profile=profile)
        rate_ok.append((c, d))
    resolved = rate_ok
    logger.info(
        "rate check: %d probe cache hits, %d probed; %d mismatch, %d unverified, %d unprobed",
        rate_cache.hits,
        rate_cache.misses,
        mismatches,
        sum(1 for r in rates.values() if r.status == "nodata"),
        sum(1 for r in rates.values() if r.status == "unprobed"),
    )

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for c, d in resolved:
        t = c.triplet
        rd = rates[c.id]
        rate_hz = rd.data_hz if rd.data_hz is not None else rd.metadata_hz
        cov = station_coverage(coverage[c.id])
        if cov is None:
            used = cfg.availability.onMissing == "used"
            note = ""
            if used and rd.status == "nodata" and cfg.availability.onMissingNoProbeData == "unused":
                # No availability measurement and nothing served at any rate probe: drop it, so
                # the station doesn't enter geometry-dependent work (onMissingNoProbeData).
                used = False
                note = (
                    "; no data at any rate probe either, so usedInRun False (onMissingNoProbeData)"
                )
            flags.append(
                {
                    "station": c.id,
                    "flag": (
                        f"no {cfg.availability.metric} measurement for the window; coverage "
                        f"unknown, usedInRun {used} (onMissing={cfg.availability.onMissing})"
                        f"{note}"
                    ),
                }
            )
        else:
            used = cov > 0
        rows.append(
            {
                "id": c.id,
                "network": t.network,
                "station": t.station,
                "location": t.location,
                "latitude": t.latitude,
                "longitude": t.longitude,
                "surfaceElevM": d.surface_elev_m,
                "sensorDepthM": t.depth_m,
                "sensorElevM": d.sensor_elev_m,
                "kind": c.kind,
                "channels": list(t.channels),
                "sampleRateHz": rate_hz,
                "enu": project(t.latitude, t.longitude, d.sensor_elev_m),
                "preprocessProfile": c.profile,
                "usedInRun": used,
                "staticsS": {},
            }
        )
        details.append(
            {
                "id": c.id,
                "triplet": t.label,
                "family": t.family,
                "kind": c.kind,
                "sampleRateHz": rate_hz,
                "rateCheck": {
                    "status": rd.status,
                    "metadataHz": rd.metadata_hz,
                    "dataHz": rd.data_hz,
                    "probeTimes": list(rd.probe_times),
                },
                "channels": list(t.channels),
                "profile": c.profile,
                "elevation": {
                    "convention": d.convention,
                    "basis": d.basis,
                    "stationElevM": d.station_elev_m,
                    "channelElevM": d.channel_elev_m,
                    "channelDepthM": d.depth_m,
                    "demM": d.dem_m,
                    "surfaceElevM": d.surface_elev_m,
                    "sensorElevM": d.sensor_elev_m,
                    "toleranceM": cfg.elevation.toleranceM,
                },
                "coverage": cov,
                "coverageByChannel": coverage[c.id],
                "usedInRun": used,
            }
        )
    rows.sort(key=lambda r: r["id"])
    details.sort(key=lambda r: r["id"])
    skipped.sort(key=lambda s: s["site"])
    flags.sort(key=lambda f: f["station"])

    counts = {
        "selected": len(rows),
        "usedInRun": sum(1 for r in rows if r["usedInRun"]),
        "withData": sum(1 for d in details if d["coverage"] is not None and d["coverage"] > 0),
        "noAvailabilityMeasurement": sum(1 for d in details if d["coverage"] is None),
        "borehole": sum(1 for r in rows if r["kind"] == "borehole"),
        "surface": sum(1 for r in rows if r["kind"] == "surface"),
        "strongMotion": sum(1 for r in rows if r["kind"] == "strong_motion"),
        "sensorLevelStationElev": sum(
            1 for d in details if d["elevation"]["convention"] == "sensor"
        ),
        "demSurface": sum(1 for d in details if d["elevation"]["convention"] == "dem"),
        "rateMismatch": mismatches,  # includes stations skipped for a mismatch
        "rateUnverified": sum(1 for d in details if d["rateCheck"]["status"] == "nodata"),
        "skippedSites": len(skipped),
        "droppedTriplets": len(dropped),
        "excludedStations": sum(len(v) for v in excluded.values()),
        "flags": len(flags),
    }
    bad = [r["id"] for r in rows if r["kind"] == "borehole" and r["sensorDepthM"] <= 0]
    if bad:  # impossible by construction (kind is depth-based); guards against a future edit
        raise InventoryError(f"borehole stations without sensor depth: {bad}")
    if not rows:
        raise InventoryError(f"no station selected in bbox {run.bbox} for the window")
    if counts["usedInRun"] == 0:
        raise InventoryError(
            f"none of the {len(rows)} selected stations has data in the window "
            f"({counts['withData']} measured with data, {counts['noAvailabilityMeasurement']} "
            f"unmeasured, onMissing={cfg.availability.onMissing}, "
            f"onMissingNoProbeData={cfg.availability.onMissingNoProbeData})"
        )
    report = {
        "stage": STAGE,
        "window": {"start": run.windowStart.isoformat(), "end": run.windowEnd.isoformat()},
        "bbox": list(run.bbox),
        "stationXml": xml_path.name,
        "availabilitySource": {"url": cfg.availability.url, "metric": cfg.availability.metric},
        "demSource": cfg.elevation.demUrl,
        "stations": details,
        "skippedSites": skipped,
        "droppedTriplets": dropped,
        "excludedNetworks": {k: sorted(set(v)) for k, v in sorted(excluded.items())},
        "flags": flags,
        "counts": counts,
    }
    logger.info(
        "inventory: %d stations selected (%d usedInRun, %d with measured data, %d borehole), "
        "%d sites skipped",
        counts["selected"],
        counts["usedInRun"],
        counts["withData"],
        counts["borehole"],
        counts["skippedSites"],
    )
    return InventoryResult(rows=rows, report=report, counts=counts)


class StageContext(Protocol):
    """The part of ``hq.runs.RunContext`` (docs/02 -> Stage API) this stage uses."""

    @property
    def run_dir(self) -> Path: ...
    @property
    def cache_dir(self) -> Path: ...
    @property
    def config(self) -> Any: ...
    def path(self, name: str) -> Path: ...
    def record(
        self,
        stage: str,
        *,
        runtime_s: float,
        counts: dict[str, int],
        params: dict | None = None,
    ) -> None: ...


def write_stations_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Validate every row as ``hq_contracts.models.Station`` and write via ``hq_contracts.io``."""
    try:
        from hq_contracts.io import to_frame, write_table
        from hq_contracts.models import Station
    except ModuleNotFoundError as exc:
        raise InventoryError(
            f"{path.name} not written: hq_contracts.models / hq_contracts.io (CONTRACT-01, H4) "
            f"are not available on this branch ({exc})"
        ) from exc

    models = [Station.model_validate(dict(r)) for r in rows]
    tmp = path.with_name(path.name + ".part")
    write_table(to_frame(models), tmp, "Station")
    os.replace(tmp, path)


def finish(ctx: StageContext, result: InventoryResult, runtime_s: float) -> None:
    """Write run outputs (the table first, so a failure leaves no report) and record the stage."""
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    write_stations_parquet(result.rows, ctx.path(STATIONS_FILE))
    _write_json_atomic(ctx.path(REPORT_FILE), result.report)
    cfg: SignalConfig = ctx.config.signal
    ctx.record(
        STAGE,
        runtime_s=runtime_s,
        counts=result.counts,
        params={STAGE: cfg.stations.model_dump(mode="json")},  # nested: RUN-01 maps it to picker
    )
    logger.info("inventory stage finished in %.1f s", runtime_s)


def run(ctx: StageContext) -> None:
    """Stage entry point (docs/02 -> Stage API)."""
    t_start = time.perf_counter()
    result = build_inventory(ctx.config.run, ctx.config.signal.stations, ctx.cache_dir)
    finish(ctx, result, time.perf_counter() - t_start)


# --- sample-rate check --------------------------------------------------------------------------


def fdsn_rate_probe(query: StationQuery, rcfg: RateCheck) -> RateProbe:
    """Default probe: a short dataselect request, retried on transport errors and timeouts."""
    holder: dict[str, Any] = {}

    def probe(
        net: str, sta: str, loc: str, channels: Sequence[str], t0: float, t1: float
    ) -> dict[str, float] | None:
        from obspy.clients.fdsn import Client

        for attempt in range(rcfg.retries + 1):
            try:
                if "client" not in holder:  # service discovery is retried like a request
                    holder["client"] = Client(query.fdsnClient, timeout=rcfg.timeoutS)
                st = holder["client"].get_waveforms(
                    net, sta, loc or "--", ",".join(channels), UTCDateTime(t0), UTCDateTime(t1)
                )
            except FDSNNoDataException:
                return None
            except (FDSNException, OSError) as exc:
                if attempt == rcfg.retries:
                    raise InventoryError(
                        f"rate probe {net}.{sta}.{loc} failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                time.sleep(rcfg.backoffS * (attempt + 1))
                continue
            rates: dict[str, set[float]] = {}
            for tr in st:
                rates.setdefault(tr.stats.channel, set()).add(float(tr.stats.sampling_rate))
            if not rates:
                return None
            mixed = {ch: sorted(r) for ch, r in rates.items() if len(r) > 1}
            if mixed:
                raise InventoryError(f"rate probe {net}.{sta}.{loc}: mixed rates {mixed}")
            return {ch: next(iter(r)) for ch, r in rates.items()}
        raise AssertionError("unreachable")

    return probe


@dataclass(frozen=True)
class RateDecision:
    metadata_hz: float
    data_hz: float | None  # None: no probe returned data (or the station was not probed)
    probe_times: tuple[float, ...]  # probe starts that returned data
    status: Literal["match", "mismatch", "nodata", "unprobed"]


def probe_starts(run: RunSection, rcfg: RateCheck) -> list[float]:
    """Probe start times inside the window; a config where none fits is an error."""
    starts = [
        run.window_start_s + off
        for off in rcfg.probeOffsetsS
        if run.window_start_s + off + rcfg.probeS <= run.window_end_s
    ]
    if not starts:
        raise InventoryError(
            f"rateCheck: no probe (offsets {rcfg.probeOffsetsS}, {rcfg.probeS:g} s) fits inside "
            f"the {run.window_end_s - run.window_start_s:g} s window"
        )
    return starts


def check_rate(
    c: Chosen,
    run: RunSection,
    query: StationQuery,
    rcfg: RateCheck,
    probe: RateProbe,
    cache: JsonCache,
) -> RateDecision:
    """Probe the chosen triplet at every in-window offset.

    Every probe that returns data must agree on one rate; a rate change inside the window fails
    loudly. Only replies with data are cached, so late-arriving data is probed again next run.
    """
    t = c.triplet
    meta = t.sample_rate_hz
    served_at: list[tuple[float, float]] = []
    for t0 in probe_starts(run, rcfg):
        t1 = t0 + rcfg.probeS
        key = (
            f"{query.fdsnClient}|{t.network}.{t.station}.{t.location}."
            f"{','.join(t.channels)}@{t0:.3f}+{rcfg.probeS:g}"
        )
        rates = cache.get(key)
        if rates is None:
            rates = probe(t.network, t.station, t.location, t.channels, t0, t1)
            if rates:
                cache.put(key, rates)
        if not rates:
            continue
        missing = sorted(set(t.channels) - set(rates))
        served = sorted(set(rates.values()))
        if len(served) > 1:
            raise InventoryError(f"{t.label}: components serve different rates {rates}")
        if missing:
            logger.warning(
                "%s: rate probe at %s got no data for %s", t.label, UTCDateTime(t0), missing
            )
        served_at.append((t0, served[0]))
    if not served_at:
        return RateDecision(metadata_hz=meta, data_hz=None, probe_times=(), status="nodata")
    distinct = sorted({hz for _, hz in served_at})
    if len(distinct) > 1:
        detail = ", ".join(f"{UTCDateTime(t0)}: {hz:g} Hz" for t0, hz in served_at)
        raise InventoryError(f"{t.label}: served sample rate changes inside the window ({detail})")
    data = distinct[0]
    status: Literal["match", "mismatch"] = (
        "match" if abs(data - meta) <= rcfg.relTol * meta else "mismatch"
    )
    return RateDecision(
        metadata_hz=meta,
        data_hz=data,
        probe_times=tuple(t0 for t0, _ in served_at),
        status=status,
    )


# --- acceptance driver --------------------------------------------------------------------------


@dataclass(frozen=True)
class _CliConfig:
    run: RunSection
    signal: SignalConfig


@dataclass(frozen=True)
class _CliContext:
    """Stand-in for H4's RunContext until ``hq.runs`` lands; ``record`` only logs."""

    run_id: str
    run_dir: Path
    cache_dir: Path
    config: _CliConfig
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

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
        self.records[stage] = {"runtime_s": runtime_s, "counts": counts, "params": params}
        logger.info("record %s: runtime %.1f s, counts %s", stage, runtime_s, counts)


def load_cli_config(config_dir: Path) -> _CliConfig:
    with (config_dir / "run.yaml").open(encoding="utf-8") as fh:
        run_section = RunSection.model_validate(yaml.safe_load(fh))
    with (config_dir / "signal.yaml").open(encoding="utf-8") as fh:
        signal = SignalConfig.model_validate(yaml.safe_load(fh))
    return _CliConfig(run=run_section, signal=signal)


def format_table(result: InventoryResult) -> str:
    details = {d["id"]: d for d in result.report["stations"]}
    head = (
        f"{'id':<14} {'kind':<13} {'rate':>6} {'channels':<12} {'depthM':>7} {'surfElevM':>9} "
        f"{'sensElevM':>9} {'convention':<19} {'coverage':>8} {'used':<5} profile"
    )
    lines = [head, "-" * len(head)]
    for r in result.rows:
        d = details[r["id"]]
        cov = d["coverage"]
        lines.append(
            f"{r['id']:<14} {r['kind']:<13} {r['sampleRateHz']:>6g} {','.join(r['channels']):<12} "
            f"{r['sensorDepthM']:>7.1f} {r['surfaceElevM']:>9.1f} {r['sensorElevM']:>9.1f} "
            f"{d['elevation']['convention'] + '(' + d['elevation']['basis'] + ')':<19} "
            f"{'n/a' if cov is None else f'{cov:.3f}':>8} {'yes' if r['usedInRun'] else 'no':<5} "
            f"{r['preprocessProfile']}"
        )
    c = result.counts
    lines += [
        "",
        f"selected stations:                 {c['selected']}",
        f"with any data in window (MUSTANG): {c['withData']}",
        f"no availability measurement:       {c['noAvailabilityMeasurement']}",
        f"usedInRun:                         {c['usedInRun']}",
        f"borehole (all sensorDepthM > 0):   {c['borehole']}",
        f"surface / strong_motion:           {c['surface']} / {c['strongMotion']}",
        f"station elev was sensor level:     {c['sensorLevelStationElev']}",
        f"surface taken from the DEM:        {c['demSurface']}",
        f"flagged:                           {c['flags']}",
    ]
    lines += [f"  ! {f['station']}: {f['flag']}" for f in result.report["flags"]]
    lines += [f"skipped sites:                     {c['skippedSites']}"]
    lines += [f"  - {s['site']}: {s['reason']}" for s in result.report["skippedSites"]]
    excluded = result.report["excludedNetworks"]
    lines += [f"  - network {k} excluded by config: {', '.join(v)}" for k, v in excluded.items()]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SEIS-01 station inventory (acceptance driver)")
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_cli_config(args.config_dir)
    ctx = _CliContext(
        run_id=args.run_dir.name,
        run_dir=args.run_dir,
        cache_dir=args.cache_dir,
        config=config,
    )
    t_start = time.perf_counter()
    try:
        result = build_inventory(config.run, config.signal.stations, ctx.cache_dir)
        print(format_table(result), flush=True)
        finish(ctx, result, time.perf_counter() - t_start)
    except InventoryError as exc:
        logger.error("inventory failed: %s", exc)
        return 2
    print(f"\nwrote {ctx.path(STATIONS_FILE)} and {ctx.path(REPORT_FILE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

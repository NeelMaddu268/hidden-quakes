"""SEIS-01 · Station inventory: public StationXML -> ``stations.parquet`` with real sensor depths.

Stage ``inventory`` (docs/02 -> Stage API). For the run window and bbox it

1. queries FDSN station metadata at channel level (raw XML cached under ``data/cache/stationxml/``),
2. forms three-component channel triplets and ranks them (velocity first, strong-motion only when a
   site has nothing else),
3. resolves each chosen sensor's elevation. StationXML is inconsistent here: for some stations the
   station elevation is the wellhead surface, for others it is already the sensor. Each one is
   checked against a DEM at the sensor position; ``sensorDepthM`` is always the channel ``depth``
   and ``sensorElevM = surfaceElevM - sensorDepthM`` always holds,
4. assigns ``kind``, ``preprocessProfile`` and ENU (UTM 12N minus the origin, docs/01),
5. measures per-station data coverage of the window (MUSTANG daily availability),
6. caches instrument responses per station for ``read_inventory`` (SEIS-05),

and writes ``stations.parquet`` plus ``inventory_report.json`` (every decision with its numbers).

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

import requests
import yaml
from obspy import Inventory, UTCDateTime, read_inventory
from pyproj import Geod, Transformer

from hq.config.run import Origin, RunSection
from hq.config.signal import SignalConfig, StationSelection

logger = logging.getLogger(__name__)

STAGE = "inventory"
REPORT_FILE = "inventory_report.json"
STATIONS_FILE = "stations.parquet"

# docs/01 -> Conventions: horizontal ENU is UTM zone 12N metres minus the origin. A contract shared
# with every lane (SceneMeta.projection), not a tunable knob.
GEO_CRS = "EPSG:4326"
ENU_CRS = "EPSG:32612"

Kind = Literal["surface", "borehole", "strong_motion"]
Family = Literal["velocity", "accelerometer"]
Convention = Literal["surface", "sensor"]
Basis = Literal["dem", "both-match", "shallow", "assumed"]


# --- errors -------------------------------------------------------------------------------------


class InventoryError(RuntimeError):
    """Station metadata could not be turned into trustworthy Station rows."""


class AmbiguousElevationError(InventoryError):
    """Neither elevation convention matches the DEM for a deep sensor."""


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


class StationClient(Protocol):
    """The slice of ``obspy.clients.fdsn.Client`` this stage uses."""

    def get_stations(self, **kwargs: Any) -> Any: ...


def requests_get(url: str, params: Mapping[str, str], timeout_s: float) -> HttpResult:
    """Default HTTP GET. Tests pass their own ``HttpGet`` instead."""
    resp = requests.get(url, params=dict(params), timeout=timeout_s)
    return HttpResult(status=resp.status_code, text=resp.text)


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
        last = ""
        for attempt in range(ecfg.demRetries + 1):
            if attempt:
                time.sleep(ecfg.demBackoffS * attempt)
            try:
                res = http_get(ecfg.demUrl, params, ecfg.demTimeoutS)
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}: {exc}"
                logger.warning("DEM query failed (attempt %d): %s", attempt + 1, last)
                continue
            if res.status == 200:
                try:
                    value = float(json.loads(res.text)["value"])
                except (ValueError, KeyError, TypeError) as exc:
                    raise DemError(
                        f"unparseable DEM response at {lat},{lon}: {res.text[:200]}"
                    ) from exc
                if value == ecfg.demNoDataValue:
                    raise DemError(f"DEM has no data at lat={lat} lon={lon}")
                return value
            last = f"HTTP {res.status}: {res.text[:200]}"
            logger.warning("DEM query failed (attempt %d): %s", attempt + 1, last)
            if res.status < 500:
                break
        raise DemError(f"DEM lookup failed at lat={lat} lon={lon}: {last}")

    return lookup


class DemCache:
    """DEM values cached as JSON keyed by rounded lat/lon, so reruns never touch the network."""

    def __init__(self, path: Path, lookup: DemLookup, decimals: int) -> None:
        self.path = path
        self.lookup = lookup
        self.decimals = decimals
        self.hits = 0
        self.misses = 0
        self.values: dict[str, float] = {}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                self.values = {str(k): float(v) for k, v in json.load(fh).items()}

    def key(self, lat: float, lon: float) -> str:
        return f"{lat:.{self.decimals}f},{lon:.{self.decimals}f}"

    def elevation(self, lat: float, lon: float) -> float:
        k = self.key(lat, lon)
        if k in self.values:
            self.hits += 1
            return self.values[k]
        self.misses += 1
        value = float(self.lookup(lat, lon))
        self.values[k] = value
        _write_json_atomic(self.path, dict(sorted(self.values.items())))
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


def _pick_epoch(epochs: list[ChannelRecord]) -> ChannelRecord:
    """The epoch covering most of the window; logs when overlapping epochs disagree."""
    ordered = sorted(epochs, key=lambda r: -r.overlap_s)
    best = ordered[0]
    if len(ordered) > 1:
        differ = {(r.depth_m, r.sample_rate_hz) for r in ordered}
        level = logging.WARNING if len(differ) > 1 else logging.INFO
        logger.log(
            level,
            "%s.%s.%s.%s has %d epochs in the window (depth/rate %s); using the one covering "
            "%.0f s",
            best.network,
            best.station,
            best.location,
            best.code,
            len(ordered),
            sorted(differ),
            best.overlap_s,
        )
    return best


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
    n_outside = 0
    for net in inv.networks:
        if net.code in cfg.query.excludeNetworks:
            excluded.setdefault(net.code, []).extend(s.code for s in net.stations)
            continue
        for sta in net.stations:
            site = f"{net.code}.{sta.code}"
            sites.setdefault(site, SiteCandidates())
            per_chan = epochs.setdefault(site, {})
            for ch in sta.channels:
                overlap = _overlap_s(ch.start_date, ch.end_date, t0, t1)
                if overlap <= 0:
                    n_outside += 1
                    continue
                if cfg.query.skipRestricted and ch.restricted_status == "closed":
                    sites[site].problems.append(f"{ch.location_code}.{ch.code}: restricted")
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
    if n_outside:
        logger.info("%d channel epochs do not overlap the window and were ignored", n_outside)

    for site, per_chan in epochs.items():
        cand = sites[site]
        groups: dict[tuple[str, str], dict[str, ChannelRecord]] = {}
        for (loc, code), recs in sorted(per_chan.items()):
            band_inst, comp = code[:2], code[2:]
            if band_inst not in vel and band_inst not in acc:
                cand.ignored_codes.add(band_inst)
                continue
            groups.setdefault((loc, band_inst), {})[comp] = _pick_epoch(recs)
        for (loc, band_inst), comps in sorted(groups.items()):
            where = f"{loc}.{band_inst}"
            vert = next((c for c in rules.verticalComponents if c in comps), None)
            pair = next((p for p in rules.horizontalPairs if p[0] in comps and p[1] in comps), None)
            if vert is None or pair is None:
                cand.problems.append(f"{where}: incomplete triplet (components {sorted(comps)})")
                continue
            recs3 = (comps[vert], comps[pair[0]], comps[pair[1]])
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


@dataclass(frozen=True)
class Chosen:
    id: str
    triplet: Triplet
    kind: Kind


def choose_triplets(
    sites: Mapping[str, SiteCandidates], cfg: StationSelection
) -> tuple[list[Chosen], list[dict[str, str]], list[dict[str, str]]]:
    """Pick the triplets that become Station rows. Returns (chosen, skipped sites, dropped)."""
    chosen: list[Chosen] = []
    skipped: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    tol = cfg.channels.locationDepthTolM
    for site in sorted(sites):
        cand = sites[site]
        for problem in cand.problems:
            logger.info("%s: %s", site, problem)
        velocity = [t for t in cand.triplets if t.family == "velocity"]
        pool = velocity or [t for t in cand.triplets if t.family == "accelerometer"]
        if not pool:
            parts = list(cand.problems)
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
            for t in cand.triplets:
                if t.family == "accelerometer":
                    dropped.append(
                        {"triplet": t.label, "reason": "accelerometer; site has a velocity triplet"}
                    )
        best_per_loc: dict[str, Triplet] = {}
        for t in sorted(pool, key=Triplet.sort_key):
            if t.location in best_per_loc:
                better = best_per_loc[t.location]
                dropped.append(
                    {"triplet": t.label, "reason": f"lower priority than {better.label}"}
                )
                continue
            best_per_loc[t.location] = t
        kept: list[tuple[Triplet, Kind]] = []
        for t in sorted(best_per_loc.values(), key=Triplet.sort_key):
            kind = classify_kind(t.depth_m, t.code, cfg)
            twin = next(
                (k for k, kk in kept if kk == kind and abs(k.depth_m - t.depth_m) <= tol), None
            )
            if twin is not None:
                dropped.append(
                    {
                        "triplet": t.label,
                        "reason": f"same kind ({kind}) and depth as {twin.label}",
                    }
                )
                continue
            kept.append((t, kind))
        for t, kind in kept:
            sid = site if len(kept) == 1 else f"{site}.{t.location or '--'}"
            chosen.append(Chosen(id=sid, triplet=t, kind=kind))
    for d in dropped:
        logger.info("not used %s: %s", d["triplet"], d["reason"])
    return chosen, skipped, dropped


# --- elevation, profile, ENU --------------------------------------------------------------------


@dataclass(frozen=True)
class ElevationDecision:
    convention: Convention  # what StationXML's station elevation turned out to be
    basis: Basis
    surface_elev_m: float
    sensor_elev_m: float
    station_elev_m: float
    channel_elev_m: float
    depth_m: float
    dem_m: float

    @property
    def label(self) -> str:
        return f"{self.convention}({self.basis})"


def resolve_elevation(
    station_elev_m: float,
    channel_elev_m: float,
    depth_m: float,
    dem_m: float,
    tol_m: float,
    on_ambiguous: Literal["error", "surface", "skip"],
    *,
    what: str,
) -> ElevationDecision | None:
    """Decide whether the station elevation is the site surface or already the sensor.

    ``surface``: surfaceElevM = stationElev, sensorElevM = stationElev - depth.
    ``sensor``:  surfaceElevM = stationElev + depth, sensorElevM = stationElev.
    Both conventions matching the DEM (only possible when depth <= 2 tol), or a shallow sensor
    (depth < 2 tol) matching neither, are indistinguishable and harmless: "surface" is used.
    A deep sensor matching neither is ambiguous and handled by ``on_ambiguous``.
    Returns None only when ``on_ambiguous == "skip"`` and the case is ambiguous.
    """
    surface_ok = abs(station_elev_m - dem_m) <= tol_m
    sensor_ok = abs(station_elev_m + depth_m - dem_m) <= tol_m
    convention: Convention
    basis: Basis
    if surface_ok and sensor_ok:
        convention, basis = "surface", "both-match"
    elif surface_ok:
        convention, basis = "surface", "dem"
    elif sensor_ok:
        convention, basis = "sensor", "dem"
    elif depth_m < 2 * tol_m:
        convention, basis = "surface", "shallow"
        logger.warning(
            "%s: station elevation %.1f m disagrees with DEM %.1f m by %.1f m (> %.1f); sensor is "
            "shallow (%.1f m), keeping StationXML station elevation as the surface",
            what,
            station_elev_m,
            dem_m,
            station_elev_m - dem_m,
            tol_m,
            depth_m,
        )
    else:
        msg = (
            f"{what}: ambiguous elevation: stationElev {station_elev_m:.1f}, channelElev "
            f"{channel_elev_m:.1f}, depth {depth_m:.1f}, DEM {dem_m:.1f}; neither stationElev "
            f"(surface) nor stationElev + depth (sensor) is within {tol_m:.1f} m of the DEM"
        )
        if on_ambiguous == "error":
            raise AmbiguousElevationError(msg)
        if on_ambiguous == "skip":
            logger.warning("%s; skipping", msg)
            return None
        logger.warning("%s; assuming surface convention (onAmbiguous=surface)", msg)
        convention, basis = "surface", "assumed"
    if convention == "surface":
        surface = station_elev_m
    else:
        surface = station_elev_m + depth_m
    decision = ElevationDecision(
        convention=convention,
        basis=basis,
        surface_elev_m=surface,
        sensor_elev_m=surface - depth_m,
        station_elev_m=station_elev_m,
        channel_elev_m=channel_elev_m,
        depth_m=depth_m,
        dem_m=dem_m,
    )
    logger.info(
        "%s: elevation %s: stationElev %.1f, channelElev %.1f, depth %.1f, DEM %.1f -> "
        "surfaceElevM %.1f, sensorElevM %.1f",
        what,
        decision.label,
        station_elev_m,
        channel_elev_m,
        depth_m,
        dem_m,
        decision.surface_elev_m,
        decision.sensor_elev_m,
    )
    return decision


def match_profile(rate_hz: float, cfg: StationSelection) -> str | None:
    """First profile rule whose inclusive rate range contains ``rate_hz``."""
    for rule in cfg.profiles:
        if rule.minRateHz <= rate_hz <= rule.maxRateHz:
            return rule.profile
    return None


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


def fetch_coverage(
    chosen: Sequence[Chosen], run: RunSection, cfg: StationSelection, http_get: HttpGet
) -> dict[str, dict[str, float | None]]:
    """Per station id, per channel: fraction of the window with data (None = no measurement).

    MUSTANG publishes one availability value per channel-day, so a partial-day window uses the
    overlap-weighted daily values. SEIS-05 measures real gaps after download.
    """
    acfg = cfg.availability
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
        res = http_get(acfg.url, params, acfg.timeoutS)
        if res.status == 204:
            rows: list[dict[str, Any]] = []
        elif res.status == 200:
            try:
                rows = json.loads(res.text)["measurements"].get(acfg.metric, [])
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise AvailabilityError(
                    f"{c.id}: unparseable availability response: {res.text[:200]}"
                ) from exc
        else:
            raise AvailabilityError(
                f"{c.id}: availability service HTTP {res.status}: {res.text[:200]}"
            )
        daily: dict[tuple[str, str], float] = {}
        for row in rows:
            if row.get("loc", "") != t.location or row.get("cha") not in t.channels:
                continue
            key = (str(row["cha"]), str(row["start"])[:10])
            # one row per quality code; the best one says whether data exists at all
            daily[key] = max(daily.get(key, 0.0), float(row["value"]) / 100.0)
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
                logger.warning("%s; marking unused (onMissing=unused)", msg)
                per_chan[cha] = None
            else:
                per_chan[cha] = min(1.0, total)
        out[c.id] = per_chan
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


def _fetch_to_file(client: StationClient, path: Path, **query: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    client.get_stations(filename=str(tmp), **query)
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise InventoryError(f"station service returned nothing for {query}")
    os.replace(tmp, path)


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
        _fetch_to_file(client.get(), path, **query)
        sidecar = {k: str(v) for k, v in query.items()} | {"client": cfg.query.fdsnClient}
        _write_json_atomic(path.with_suffix(".json"), sidecar)
    return read_inventory(str(path), format="STATIONXML"), path


def ensure_responses(
    chosen: Sequence[Chosen],
    run: RunSection,
    cfg: StationSelection,
    xml_dir: Path,
    client: _LazyClient,
) -> tuple[int, int]:
    """``<xml_dir>/<Station.id>.xml`` at response level for each chosen triplet. (hits, fetched)"""
    hits = fetched = 0
    t0, t1 = UTCDateTime(run.window_start_s), UTCDateTime(run.window_end_s)
    for c in chosen:
        t = c.triplet
        path = xml_dir / f"{c.id}.xml"
        if path.exists():
            cached = read_inventory(str(path), format="STATIONXML")
            have = {
                ch.code
                for net in cached.networks
                for sta in net.stations
                for ch in sta.channels
                if ch.location_code == t.location
            }
            if set(t.channels) <= have:
                hits += 1
                continue
            logger.info(
                "%s: cached response XML lacks %s; refetching", c.id, set(t.channels) - have
            )
        _fetch_to_file(
            client.get(),
            path,
            network=t.network,
            station=t.station,
            location=t.location or "--",
            channel=",".join(t.channels),
            starttime=t0,
            endtime=t1,
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
) -> InventoryResult:
    """Everything except writing run outputs. Network access only through the injected callables
    (or their defaults), and only on cache misses for StationXML and DEM."""
    get = http_get or requests_get
    lazy = _LazyClient(cfg, client)
    xml_dir = cache_dir / cfg.query.cacheSubdir
    inv, xml_path = load_channel_inventory(run, cfg, xml_dir, lazy)

    sites, excluded = collect_candidates(inv, run, cfg)
    for net_code, stas in sorted(excluded.items()):
        logger.info("network %s excluded by config (%d stations)", net_code, len(stas))
    chosen, skipped, dropped = choose_triplets(sites, cfg)

    dem_cache = DemCache(
        xml_dir / cfg.elevation.demCacheFile,
        dem or epqs_dem(cfg, get),
        cfg.elevation.demKeyDecimals,
    )
    project = EnuProjector(run.origin)
    geod = Geod(ellps="WGS84")
    flags: list[dict[str, str]] = []
    resolved: list[tuple[Chosen, str, ElevationDecision]] = []
    for c in chosen:
        t = c.triplet
        profile = match_profile(t.sample_rate_hz, cfg)
        if profile is None:
            reason = f"{t.label}: no preprocess profile for {t.sample_rate_hz:g} Hz"
            skipped.append({"site": c.id, "reason": reason})
            logger.warning("skip %s", reason)
            continue
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
        dem_m = dem_cache.elevation(t.latitude, t.longitude)
        decision = resolve_elevation(
            t.station_elev_m,
            t.channel_elev_m,
            t.depth_m,
            dem_m,
            cfg.elevation.toleranceM,
            cfg.elevation.onAmbiguous,
            what=t.label,
        )
        if decision is None:
            skipped.append({"site": c.id, "reason": f"{t.label}: ambiguous elevation"})
            continue
        if decision.basis in ("shallow", "assumed"):
            flags.append(
                {
                    "station": c.id,
                    "flag": (
                        f"{t.label} station elevation {decision.station_elev_m:.1f} m differs from "
                        f"DEM {decision.dem_m:.1f} m by "
                        f"{decision.station_elev_m - decision.dem_m:+.1f} m; kept StationXML value "
                        f"({decision.label})"
                    ),
                }
            )
        resolved.append((c, profile, decision))
    logger.info("DEM: %d cache hits, %d lookups", dem_cache.hits, dem_cache.misses)

    kept = [c for c, _, _ in resolved]
    coverage = fetch_coverage(kept, run, cfg, get)
    ensure_responses(kept, run, cfg, xml_dir, lazy)

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for c, profile, d in resolved:
        t = c.triplet
        cov = station_coverage(coverage[c.id])
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
                "sampleRateHz": t.sample_rate_hz,
                "enu": project(t.latitude, t.longitude, d.sensor_elev_m),
                "preprocessProfile": profile,
                "usedInRun": cov is not None and cov > 0,
                "staticsS": {},
            }
        )
        details.append(
            {
                "id": c.id,
                "triplet": t.label,
                "family": t.family,
                "kind": c.kind,
                "sampleRateHz": t.sample_rate_hz,
                "channels": list(t.channels),
                "profile": profile,
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
            }
        )
    rows.sort(key=lambda r: r["id"])
    details.sort(key=lambda r: r["id"])
    skipped.sort(key=lambda s: s["site"])

    counts = {
        "selected": len(rows),
        "withData": sum(1 for r in rows if r["usedInRun"]),
        "borehole": sum(1 for r in rows if r["kind"] == "borehole"),
        "surface": sum(1 for r in rows if r["kind"] == "surface"),
        "strongMotion": sum(1 for r in rows if r["kind"] == "strong_motion"),
        "sensorLevelStationElev": sum(
            1 for d in details if d["elevation"]["convention"] == "sensor"
        ),
        "skippedSites": len(skipped),
        "droppedTriplets": len(dropped),
        "excludedStations": sum(len(v) for v in excluded.values()),
        "noAvailabilityMeasurement": sum(1 for d in details if d["coverage"] is None),
        "flags": len(flags),
    }
    bad = [r["id"] for r in rows if r["kind"] == "borehole" and r["sensorDepthM"] <= 0]
    if bad:  # impossible by construction (kind is depth-based); guards against a future edit
        raise InventoryError(f"borehole stations without sensor depth: {bad}")
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
        "inventory: %d stations selected (%d with data in window, %d borehole), %d sites skipped",
        counts["selected"],
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
    write_table(to_frame(models), path, "Station")


def finish(ctx: StageContext, result: InventoryResult, runtime_s: float) -> None:
    """Write run outputs and record the stage."""
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(ctx.path(REPORT_FILE), result.report)
    write_stations_parquet(result.rows, ctx.path(STATIONS_FILE))
    cfg: SignalConfig = ctx.config.signal
    ctx.record(
        STAGE,
        runtime_s=runtime_s,
        counts=result.counts,
        params=cfg.stations.model_dump(mode="json"),
    )
    logger.info("inventory stage finished in %.1f s", runtime_s)


def run(ctx: StageContext) -> None:
    """Stage entry point (docs/02 -> Stage API)."""
    t_start = time.perf_counter()
    result = build_inventory(ctx.config.run, ctx.config.signal.stations, ctx.cache_dir)
    finish(ctx, result, time.perf_counter() - t_start)


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
        f"{'sensElevM':>9} {'convention':<19} {'coverage':>8} profile"
    )
    lines = [head, "-" * len(head)]
    for r in result.rows:
        d = details[r["id"]]
        cov = d["coverage"]
        lines.append(
            f"{r['id']:<14} {r['kind']:<13} {r['sampleRateHz']:>6g} {','.join(r['channels']):<12} "
            f"{r['sensorDepthM']:>7.1f} {r['surfaceElevM']:>9.1f} {r['sensorElevM']:>9.1f} "
            f"{d['elevation']['convention'] + '(' + d['elevation']['basis'] + ')':<19} "
            f"{'n/a' if cov is None else f'{cov:.3f}':>8} {r['preprocessProfile']}"
        )
    c = result.counts
    lines += [
        "",
        f"selected stations:                 {c['selected']}",
        f"with any data in window:           {c['withData']}",
        f"borehole (all sensorDepthM > 0):   {c['borehole']}",
        f"surface / strong_motion:           {c['surface']} / {c['strongMotion']}",
        f"station elev was sensor level:     {c['sensorLevelStationElev']}",
        f"no availability measurement:       {c['noAvailabilityMeasurement']}",
        f"flagged in log:                    {c['flags']}",
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

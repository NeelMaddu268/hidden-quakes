# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pillow>=10",
#   "pyproj>=3.6",
#   "pyyaml>=6",
# ]
# ///
"""Bake the web scene's terrain from public DEM tiles (WEB-02, H3).

Reads the origin from ``services/seismic/configs/showcase/run.yaml``, downloads AWS Terrain Tiles
(Terrarium encoding, public data) into ``data/cache/dem/``, resamples them onto a regular ENU grid
(UTM 12N / EPSG:32612 metres minus the origin, docs/01 conventions) and writes three files to
``apps/web/public/terrain/``:

* ``height.png``: elevation as a 16-bit value split across an 8-bit RGB PNG ("rg16": R = high byte,
  G = low byte, B = 0). Browsers decode 16-bit PNGs to 8 bits in canvas/WebGL, so a true 16-bit
  PNG would lose precision on the way into the scene. v = 0..65535 maps linearly onto
  ``elevMinM..elevMaxM``.
* ``hillshade.png``: 8-bit grayscale Horn/ESRI hillshade computed here from the same grid.
* ``meta.json``: grid geometry, encoding, origin, source + attribution, hillshade parameters and
  checksums the browser uses to prove its decode is bit-exact.

Row 0 of both PNGs is the northernmost row (n = nMax), column 0 the westernmost (e = eMin);
``enuBounds`` are pixel centres. Outputs are deterministic: no timestamps anywhere, so a rerun with
the same tiles and flags is byte-identical.

Usage::

    uv run scripts/bake-dem.py                  # bake with the defaults (16 x 16 km, 513 px)
    uv run scripts/bake-dem.py --half-width-km 10 --size 641
    uv run scripts/bake-dem.py --offline        # only use cached tiles; fail if one is missing
    uv run scripts/bake-dem.py --self-test      # offline checks of the pure functions
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml
from PIL import Image
from pyproj import Transformer

log = logging.getLogger("bake-dem")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_YAML = REPO_ROOT / "services/seismic/configs/showcase/run.yaml"
DEFAULT_OUT = REPO_ROOT / "apps/web/public/terrain"
DEFAULT_CACHE = REPO_ROOT / "data/cache/dem"

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_PX = 256
SOURCE_NAME = "AWS Terrain Tiles (Terrarium encoding)"
SOURCE_DOCS = "https://registry.opendata.aws/terrain-tiles/"
ATTRIBUTION_URL = "https://github.com/tilezen/joerd/blob/master/docs/attribution.md"
ATTRIBUTION = (
    "Terrain Tiles by Mapzen, hosted on the AWS Open Data Registry. In the contiguous US the "
    "elevation data are 3DEP (formerly NED), courtesy of the U.S. Geological Survey; other "
    "sources include SRTM, GMTED2010 and ETOPO1. Full attribution: " + ATTRIBUTION_URL
)
PROJECTION = "EPSG:32612 minus origin"
UTM_CRS = "EPSG:32612"
WGS84 = "EPSG:4326"

META_VERSION = 1
RG16_MAX = 65535
# Terrarium decodes anything; values outside this range mean a broken tile, not terrain.
PLAUSIBLE_ELEV_M = (-500.0, 9000.0)

FloatArray = npt.NDArray[np.float64]
U8Array = npt.NDArray[np.uint8]


# ------------------------------------------------------------------------------------------------
# Pure functions (covered by --self-test)
# ------------------------------------------------------------------------------------------------


def terrarium_decode(rgb: U8Array) -> FloatArray:
    """Terrarium tile pixels (..., 3) uint8 -> elevation in metres: R*256 + G + B/256 - 32768."""
    if rgb.shape[-1] < 3:
        raise ValueError(f"terrarium_decode needs RGB pixels, got shape {rgb.shape}")
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return r * 256.0 + g + b / 256.0 - 32768.0


def terrarium_encode(elev_m: FloatArray) -> U8Array:
    """Inverse of terrarium_decode (test helper; exact for multiples of 1/256 m)."""
    v = np.round((np.asarray(elev_m, dtype=np.float64) + 32768.0) * 256.0).astype(np.int64)
    out = np.empty((*v.shape, 3), dtype=np.uint8)
    out[..., 0] = (v >> 16) & 0xFF
    out[..., 1] = (v >> 8) & 0xFF
    out[..., 2] = v & 0xFF
    return out


def lonlat_to_global_px(
    lon: FloatArray, lat: FloatArray, zoom: int
) -> tuple[FloatArray, FloatArray]:
    """Web Mercator (XYZ tile scheme) global pixel coordinates at `zoom`.

    World pixel (i, j) covers [i, i + 1) x [j, j + 1); its centre is (i + 0.5, j + 0.5).
    """
    world = TILE_PX * (2**zoom)
    lat_r = np.radians(np.asarray(lat, dtype=np.float64))
    px = (np.asarray(lon, dtype=np.float64) + 180.0) / 360.0 * world
    py = (1.0 - np.arcsinh(np.tan(lat_r)) / math.pi) / 2.0 * world
    return px, py


def metres_per_px(lat: float, zoom: int) -> float:
    """Ground resolution of a Web Mercator pixel at latitude `lat` (WGS84 equatorial radius)."""
    return 2 * math.pi * 6378137.0 * math.cos(math.radians(lat)) / (TILE_PX * 2**zoom)


def bilinear_sample(grid: FloatArray, u: FloatArray, v: FloatArray) -> FloatArray:
    """Bilinear sample of `grid` at fractional (column u, row v); integer u, v are pixel centres.

    Raises instead of extrapolating: every sample must have all four neighbours inside the grid.
    """
    h, w = grid.shape
    if np.any(u < 0) or np.any(v < 0) or np.any(u > w - 1) or np.any(v > h - 1):
        raise ValueError(
            f"bilinear_sample out of range: u in [{u.min():.3f}, {u.max():.3f}] of 0..{w - 1}, "
            f"v in [{v.min():.3f}, {v.max():.3f}] of 0..{h - 1}"
        )
    u0 = np.minimum(np.floor(u).astype(np.int64), w - 2)
    v0 = np.minimum(np.floor(v).astype(np.int64), h - 2)
    fu = u - u0
    fv = v - v0
    top = grid[v0, u0] * (1 - fu) + grid[v0, u0 + 1] * fu
    bottom = grid[v0 + 1, u0] * (1 - fu) + grid[v0 + 1, u0 + 1] * fu
    return top * (1 - fv) + bottom * fv


def enu_axes(half_width_m: float, size: int) -> tuple[FloatArray, FloatArray]:
    """Pixel-centre coordinates: e increases with column (west -> east), n decreases with row
    (row 0 = north). Both span [-half_width_m, +half_width_m] inclusive."""
    if size < 2:
        raise ValueError(f"grid size must be >= 2, got {size}")
    e = np.linspace(-half_width_m, half_width_m, size)
    n = np.linspace(half_width_m, -half_width_m, size)
    return e, n


@dataclass(frozen=True)
class EnuFrame:
    """ENU <-> lon/lat for one origin: e, n = UTM 12N metres minus the origin's UTM coordinates."""

    origin_lon: float
    origin_lat: float
    origin_easting: float
    origin_northing: float
    to_utm: Transformer
    to_lonlat: Transformer

    @classmethod
    def for_origin(cls, lon: float, lat: float) -> EnuFrame:
        to_utm = Transformer.from_crs(WGS84, UTM_CRS, always_xy=True)
        to_lonlat = Transformer.from_crs(UTM_CRS, WGS84, always_xy=True)
        easting, northing = to_utm.transform(lon, lat)
        return cls(lon, lat, float(easting), float(northing), to_utm, to_lonlat)

    def enu_to_lonlat(self, e: FloatArray, n: FloatArray) -> tuple[FloatArray, FloatArray]:
        lon, lat = self.to_lonlat.transform(
            np.asarray(e, dtype=np.float64) + self.origin_easting,
            np.asarray(n, dtype=np.float64) + self.origin_northing,
        )
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)

    def lonlat_to_enu(self, lon: FloatArray, lat: FloatArray) -> tuple[FloatArray, FloatArray]:
        x, y = self.to_utm.transform(
            np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
        )
        return np.asarray(x) - self.origin_easting, np.asarray(y) - self.origin_northing


def rg16_range(elev_m: FloatArray) -> tuple[float, float]:
    """Encoding range: min/max rounded outward to the centimetre (never an empty range)."""
    lo = math.floor(float(np.min(elev_m)) * 100.0) / 100.0
    hi = math.ceil(float(np.max(elev_m)) * 100.0) / 100.0
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def rg16_quantize(elev_m: FloatArray, elev_min: float, elev_max: float) -> npt.NDArray[np.uint16]:
    """Elevation -> v in 0..65535, linear over [elev_min, elev_max]."""
    scaled = (np.asarray(elev_m, dtype=np.float64) - elev_min) / (elev_max - elev_min) * RG16_MAX
    return np.clip(np.round(scaled), 0, RG16_MAX).astype(np.uint16)


def rg16_encode(elev_m: FloatArray, elev_min: float, elev_max: float) -> U8Array:
    """Elevation grid (h, w) -> RGB uint8 (h, w, 3): R = high byte of v, G = low byte, B = 0."""
    v = rg16_quantize(elev_m, elev_min, elev_max)
    out = np.zeros((*v.shape, 3), dtype=np.uint8)
    out[..., 0] = (v >> 8).astype(np.uint8)
    out[..., 1] = (v & 0xFF).astype(np.uint8)
    return out


def rg16_values(rgb: U8Array) -> npt.NDArray[np.int64]:
    """RGB uint8 -> v = R*256 + G (the browser does exactly this)."""
    return rgb[..., 0].astype(np.int64) * 256 + rgb[..., 1].astype(np.int64)


def rg16_decode(rgb: U8Array, elev_min: float, elev_max: float) -> FloatArray:
    """RGB uint8 -> elevation in metres."""
    return elev_min + rg16_values(rgb).astype(np.float64) / RG16_MAX * (elev_max - elev_min)


def hillshade(
    elev_m: FloatArray,
    cell_e_m: float,
    cell_n_m: float,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
) -> FloatArray:
    """Horn (1981) slope/aspect with the ESRI hillshade formula, in [0, 1].

    `elev_m` is row 0 = north, column 0 = west. Borders use linear extrapolation (odd reflection),
    so a plane shades uniformly all the way to the edge.
    """
    z = np.pad(np.asarray(elev_m, dtype=np.float64), 1, mode="reflect", reflect_type="odd")
    a = z[:-2, :-2]
    b = z[:-2, 1:-1]
    c = z[:-2, 2:]
    d = z[1:-1, :-2]
    f = z[1:-1, 2:]
    g = z[2:, :-2]
    h = z[2:, 1:-1]
    i = z[2:, 2:]
    # dz/dx: east minus west. dz/dy: south minus north (ESRI's convention for top-down rows).
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cell_e_m)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * cell_n_m)
    slope = np.arctan(z_factor * np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    zenith = math.radians(90.0 - altitude_deg)
    azimuth_math = math.radians((360.0 - azimuth_deg + 90.0) % 360.0)
    shade = math.cos(zenith) * np.cos(slope) + math.sin(zenith) * np.sin(slope) * np.cos(
        azimuth_math - aspect
    )
    return np.clip(shade, 0.0, 1.0)


def hillshade_u8(shade: FloatArray) -> U8Array:
    return np.clip(np.round(shade * 255.0), 0, 255).astype(np.uint8)


def tile_range(px: FloatArray, py: FloatArray, margin_px: float = 1.5) -> tuple[int, int, int, int]:
    """Inclusive tile index range (x0, y0, x1, y1) covering every sample plus a bilinear margin."""
    x0 = math.floor((float(np.min(px)) - margin_px) / TILE_PX)
    x1 = math.floor((float(np.max(px)) + margin_px) / TILE_PX)
    y0 = math.floor((float(np.min(py)) - margin_px) / TILE_PX)
    y1 = math.floor((float(np.max(py)) + margin_px) / TILE_PX)
    return x0, y0, x1, y1


def sample_mosaic(
    mosaic_m: FloatArray, tile_x0: int, tile_y0: int, px: FloatArray, py: FloatArray
) -> FloatArray:
    """Sample a mosaic whose top-left tile is (tile_x0, tile_y0) at global pixel coords (px, py)."""
    u = px - tile_x0 * TILE_PX - 0.5
    v = py - tile_y0 * TILE_PX - 0.5
    return bilinear_sample(mosaic_m, u, v)


# ------------------------------------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Origin:
    lat: float
    lon: float
    elev_m: float
    ref_surface_elev_m: float


def read_origin(run_yaml: Path) -> Origin:
    with run_yaml.open() as fh:
        doc = yaml.safe_load(fh)
    try:
        o = doc["origin"]
        return Origin(
            lat=float(o["lat"]),
            lon=float(o["lon"]),
            elev_m=float(o["elevM"]),
            ref_surface_elev_m=float(doc["refSurfaceElevM"]),
        )
    except (KeyError, TypeError) as exc:
        raise SystemExit(
            f"{run_yaml}: missing origin.lat/lon/elevM or refSurfaceElevM ({exc})"
        ) from exc


def fetch_tile(z: int, x: int, y: int, cache_dir: Path, *, offline: bool, retries: int = 3) -> Path:
    """Cached tile path; downloads on a cache miss. Raises on any failure (never skips a tile)."""
    path = cache_dir / "terrarium" / str(z) / str(x) / f"{y}.png"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = TILE_URL.format(z=z, x=x, y=y)
    if offline:
        raise SystemExit(f"missing cached tile {path} (--offline); source {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hidden-quakes-bake-dem/1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
            if not body:
                raise ValueError("empty response")
            tmp = path.with_suffix(".png.part")
            tmp.write_bytes(body)
            tmp.replace(path)
            log.info("downloaded %s (%d bytes)", url, len(body))
            return path
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_err = exc
            log.warning("tile %s attempt %d/%d failed: %s", url, attempt, retries, exc)
            time.sleep(0.5 * attempt)
    raise SystemExit(f"could not download tile {url}: {last_err}")


def load_mosaic(
    zoom: int, rng: tuple[int, int, int, int], cache_dir: Path, *, offline: bool
) -> tuple[FloatArray, list[str]]:
    x0, y0, x1, y1 = rng
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    mosaic = np.empty((ny * TILE_PX, nx * TILE_PX), dtype=np.float64)
    names: list[str] = []
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            path = fetch_tile(zoom, tx, ty, cache_dir, offline=offline)
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            if rgb.shape != (TILE_PX, TILE_PX, 3):
                raise SystemExit(
                    f"tile {path} has shape {rgb.shape}, expected {TILE_PX}x{TILE_PX}x3"
                )
            r, c = (ty - y0) * TILE_PX, (tx - x0) * TILE_PX
            mosaic[r : r + TILE_PX, c : c + TILE_PX] = terrarium_decode(rgb)
            names.append(f"{zoom}/{tx}/{ty}")
    lo, hi = float(mosaic.min()), float(mosaic.max())
    if lo < PLAUSIBLE_ELEV_M[0] or hi > PLAUSIBLE_ELEV_M[1]:
        raise SystemExit(f"implausible DEM elevations {lo:.1f}..{hi:.1f} m; a tile is corrupt")
    return mosaic, names


def save_png(pixels: U8Array, path: Path, mode: str) -> int:
    """Write a PNG with no ancillary chunks (time, gamma, ICC): reruns are byte-identical."""
    Image.fromarray(pixels, mode=mode).save(path, format="PNG", optimize=True)
    return path.stat().st_size


# ------------------------------------------------------------------------------------------------
# Bake
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BakeArgs:
    run_yaml: Path
    out_dir: Path
    cache_dir: Path
    half_width_km: float
    size: int
    zoom: int
    azimuth_deg: float
    altitude_deg: float
    offline: bool


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def bake(args: BakeArgs) -> dict[str, object]:
    t_start = time.perf_counter()
    origin = read_origin(args.run_yaml)
    log.info(
        "origin lat %.6f lon %.6f elevM %.1f (refSurfaceElevM %.1f) from %s",
        origin.lat,
        origin.lon,
        origin.elev_m,
        origin.ref_surface_elev_m,
        repo_relative(args.run_yaml),
    )
    half_m = args.half_width_km * 1000.0
    spacing_m = 2 * half_m / (args.size - 1)
    res_m = metres_per_px(origin.lat, args.zoom)
    log.info(
        "grid %d x %d px, +/-%.0f m, spacing %.2f m; zoom %d tiles at %.2f m/px",
        args.size,
        args.size,
        half_m,
        spacing_m,
        args.zoom,
        res_m,
    )
    if res_m > spacing_m * 1.05:
        log.warning(
            "zoom %d (%.1f m/px) is coarser than the grid spacing (%.1f m)",
            args.zoom,
            res_m,
            spacing_m,
        )

    t = time.perf_counter()
    frame = EnuFrame.for_origin(origin.lon, origin.lat)
    e_axis, n_axis = enu_axes(half_m, args.size)
    ee, nn = np.meshgrid(e_axis, n_axis)
    lon, lat = frame.enu_to_lonlat(ee, nn)
    px, py = lonlat_to_global_px(lon, lat, args.zoom)
    rng = tile_range(px, py)
    log.info("projected %d grid points in %.2f s", ee.size, time.perf_counter() - t)

    t = time.perf_counter()
    mosaic, tiles = load_mosaic(args.zoom, rng, args.cache_dir, offline=args.offline)
    log.info(
        "mosaicked %d tiles (%d x %d px) in %.2f s",
        len(tiles),
        mosaic.shape[1],
        mosaic.shape[0],
        time.perf_counter() - t,
    )

    t = time.perf_counter()
    elev = sample_mosaic(mosaic, rng[0], rng[1], px, py)
    o_px, o_py = lonlat_to_global_px(np.array([origin.lon]), np.array([origin.lat]), args.zoom)
    dem_at_origin = float(sample_mosaic(mosaic, rng[0], rng[1], o_px, o_py)[0])
    log.info(
        "resampled in %.2f s: elev min %.1f max %.1f mean %.1f m; "
        "DEM at origin %.1f m vs run.yaml %.1f m",
        time.perf_counter() - t,
        elev.min(),
        elev.max(),
        elev.mean(),
        dem_at_origin,
        origin.elev_m,
    )

    t = time.perf_counter()
    elev_min, elev_max = rg16_range(elev)
    height_rgb = rg16_encode(elev, elev_min, elev_max)
    shade = hillshade(elev, spacing_m, spacing_m, args.azimuth_deg, args.altitude_deg)
    shade_u8 = hillshade_u8(shade)
    flat_u8 = int(hillshade_u8(np.array([math.sin(math.radians(args.altitude_deg))]))[0])
    log.info("encoded rg16 + hillshade in %.2f s", time.perf_counter() - t)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    height_path = args.out_dir / "height.png"
    shade_path = args.out_dir / "hillshade.png"
    height_bytes = save_png(height_rgb, height_path, "RGB")
    shade_bytes = save_png(shade_u8, shade_path, "L")

    # Verify what's on disk, exactly as the browser will read it.
    with Image.open(height_path) as im:
        back = np.asarray(im.convert("RGB"), dtype=np.uint8)
    with Image.open(shade_path) as im:
        shade_back = np.asarray(im, dtype=np.uint8)
    if not np.array_equal(back, height_rgb) or not np.array_equal(shade_back, shade_u8):
        raise SystemExit("PNG read-back differs from what was written")
    step_m = (elev_max - elev_min) / RG16_MAX
    max_err = float(np.max(np.abs(rg16_decode(back, elev_min, elev_max) - elev)))
    if max_err > step_m:
        raise SystemExit(f"rg16 round trip error {max_err:.4f} m exceeds one step ({step_m:.4f} m)")
    log.info("verified read-back: max rg16 error %.4f m (step %.4f m)", max_err, step_m)

    meta: dict[str, object] = {
        "version": META_VERSION,
        "encoding": "rg16",
        "encodingNote": (
            "height.png is 8-bit RGB: v = R*256 + G (B = 0), elevM = elevMinM + v / 65535 * "
            "(elevMaxM - elevMinM). A true 16-bit PNG would be truncated to 8 bits by browsers."
        ),
        "elevMinM": elev_min,
        "elevMaxM": elev_max,
        "sizePx": [args.size, args.size],
        "enuBounds": {
            "eMin": float(e_axis[0]),
            "eMax": float(e_axis[-1]),
            "nMin": float(n_axis[-1]),
            "nMax": float(n_axis[0]),
            "at": "pixelCenters",
        },
        "spacingM": [spacing_m, spacing_m],
        "rowOrder": "row 0 = nMax (north), column 0 = eMin (west)",
        "origin": {"lat": origin.lat, "lon": origin.lon, "elevM": origin.elev_m},
        "originUtm": {
            "crs": UTM_CRS,
            "easting": round(frame.origin_easting, 3),
            "northing": round(frame.origin_northing, 3),
        },
        "projection": PROJECTION,
        "refSurfaceElevM": origin.ref_surface_elev_m,
        "demElevAtOriginM": round(dem_at_origin, 2),
        "source": {
            "name": SOURCE_NAME,
            "url": TILE_URL,
            "docs": SOURCE_DOCS,
            "attribution": ATTRIBUTION,
            "zoom": args.zoom,
            "metresPerPx": round(res_m, 3),
            "tiles": tiles,
            "resampling": "bilinear at pixel centres, ENU grid -> lon/lat via pyproj",
        },
        "hillshade": {
            "azimuthDeg": args.azimuth_deg,
            "altitudeDeg": args.altitude_deg,
            "zFactor": 1.0,
            "method": "Horn (1981) gradient, ESRI hillshade formula",
            "flatValue": flat_u8,
        },
        "checksums": {
            "heightSumV": int(rg16_values(height_rgb).sum()),
            "hillshadeSum": int(shade_u8.astype(np.int64).sum()),
        },
        "files": {"height": "height.png", "hillshade": "hillshade.png"},
        "generator": "scripts/bake-dem.py",
        "runYaml": repo_relative(args.run_yaml),
    }
    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    log.info(
        "wrote %s (%d B), %s (%d B), %s in %.2f s total",
        repo_relative(height_path),
        height_bytes,
        repo_relative(shade_path),
        shade_bytes,
        repo_relative(meta_path),
        time.perf_counter() - t_start,
    )
    return meta


# ------------------------------------------------------------------------------------------------
# Self-test (offline; small synthetic arrays)
# ------------------------------------------------------------------------------------------------


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _test_terrarium_decode() -> None:
    elev = np.array([[-32768.0, 0.0, 1627.5], [4000.25, -10.5, 1234.00390625]])
    back = terrarium_decode(terrarium_encode(elev))
    _check(np.array_equal(back, elev), f"terrarium round trip {back} != {elev}")
    # Known pixel: R=134, G=91, B=128 -> 134*256 + 91 + 0.5 - 32768 = 1627.5
    px = np.array([[134, 91, 128]], dtype=np.uint8)
    _check(float(terrarium_decode(px)[0]) == 1627.5, "terrarium known pixel")


def _test_rg16_round_trip() -> None:
    rng = np.random.default_rng(12)
    elev = rng.uniform(1480.0, 2790.0, size=(33, 47))
    lo, hi = rg16_range(elev)
    _check(lo <= elev.min() and hi >= elev.max(), "rg16 range must contain the data")
    rgb = rg16_encode(elev, lo, hi)
    _check(rgb.dtype == np.uint8 and rgb.shape == (33, 47, 3), "rg16 shape/dtype")
    _check(int(rgb[..., 2].max()) == 0, "rg16 blue channel must be 0")
    v = rg16_quantize(elev, lo, hi).astype(np.int64)
    _check(np.array_equal(rg16_values(rgb), v), "rg16 bytes -> v")
    exact_v = (elev - lo) / (hi - lo) * RG16_MAX
    _check(float(np.max(np.abs(rg16_values(rgb) - exact_v))) <= 1.0, "rg16 within 1 unit")
    step = (hi - lo) / RG16_MAX
    _check(
        float(np.max(np.abs(rg16_decode(rgb, lo, hi) - elev))) <= step, "rg16 decode within 1 step"
    )
    ends = rg16_quantize(np.array([lo, hi]), lo, hi)
    _check(int(ends[0]) == 0 and int(ends[1]) == RG16_MAX, "rg16 endpoints map to 0 and 65535")
    flat_lo, flat_hi = rg16_range(np.full((3, 3), 1600.0))
    _check(flat_hi > flat_lo, "rg16 range never empty")


def _test_hillshade() -> None:
    cell = 30.0
    flat = hillshade(np.full((9, 11), 1700.0), cell, cell)
    expected_flat = math.sin(math.radians(45.0))
    _check(float(np.ptp(flat)) == 0.0, "flat plane must shade uniformly")
    _check(abs(float(flat[0, 0]) - expected_flat) < 1e-12, "flat plane shade = sin(altitude)")
    _check(int(hillshade_u8(flat)[0, 0]) == 180, "flat plane -> 180 at altitude 45")

    rows, cols = 12, 14
    n_idx, e_idx = np.mgrid[0:rows, 0:cols]
    e = e_idx * cell
    n = -n_idx * cell  # row 0 = north
    for az in (315.0, 0.0, 135.0, 250.0):
        for gx, gy in ((0.4, 0.0), (0.0, 0.4), (-0.3, 0.2), (0.25, -0.35), (1.2, 0.9)):
            z = 1500.0 + gx * e + gy * n  # dz/de = gx, dz/dn = gy
            got = hillshade(z, cell, cell, azimuth_deg=az, altitude_deg=45.0)
            # Independent derivation: Lambert term = dot(unit surface normal, unit sun vector).
            normal = np.array([-gx, -gy, 1.0]) / math.sqrt(gx * gx + gy * gy + 1.0)
            alt, azr = math.radians(45.0), math.radians(az)
            sun = np.array(
                [math.sin(azr) * math.cos(alt), math.cos(azr) * math.cos(alt), math.sin(alt)]
            )
            want = max(0.0, float(normal @ sun))
            _check(
                float(np.max(np.abs(got - want))) < 1e-9,
                f"slope ({gx},{gy}) az {az}: got {got[5, 5]:.6f}, want {want:.6f}",
            )
    # A slope facing the sun (NW for azimuth 315) is brighter than one facing away.
    toward = hillshade(1500.0 + 0.3 * e - 0.3 * n, cell, cell)  # rises to the SE, faces NW
    away = hillshade(1500.0 - 0.3 * e + 0.3 * n, cell, cell)
    _check(float(toward[4, 4]) > expected_flat > float(away[4, 4]), "sun-facing slope is brighter")


def _test_enu_round_trip() -> None:
    frame = EnuFrame.for_origin(-112.90, 38.51)
    e_axis, n_axis = enu_axes(8000.0, 17)
    ee, nn = np.meshgrid(e_axis, n_axis)
    lon, lat = frame.enu_to_lonlat(ee, nn)
    e2, n2 = frame.lonlat_to_enu(lon, lat)
    err = float(np.max(np.hypot(e2 - ee, n2 - nn)))
    _check(err < 0.01, f"ENU -> lon/lat -> ENU round trip {err} m")
    o_lon, o_lat = frame.enu_to_lonlat(np.array([0.0]), np.array([0.0]))
    _check(abs(o_lon[0] + 112.90) < 1e-9 and abs(o_lat[0] - 38.51) < 1e-9, "origin maps to itself")
    # Axes: +e is east (lon grows), +n is north (lat grows); row 0 is north.
    _check(lon[0, -1] > lon[0, 0] and lat[0, 0] > lat[-1, 0], "grid orientation")
    _check(e_axis[0] == -8000.0 and e_axis[-1] == 8000.0 and n_axis[0] == 8000.0, "pixel centres")


def _test_sampling_convention() -> None:
    # A mosaic whose value is its own global pixel x at the pixel centre: sampling anywhere must
    # return px - 0.5 exactly (bilinear is exact on linear fields): pins the centre convention.
    tx0, ty0 = 763, 1572
    rows, cols = 2 * TILE_PX, 2 * TILE_PX
    col_idx = np.arange(cols, dtype=np.float64)
    row_idx = np.arange(rows, dtype=np.float64)
    mosaic = np.add.outer(np.zeros(rows), tx0 * TILE_PX + col_idx) + 0.001 * np.add.outer(
        ty0 * TILE_PX + row_idx, np.zeros(cols)
    )
    px = np.array([tx0 * TILE_PX + 10.5, tx0 * TILE_PX + 300.25, tx0 * TILE_PX + 511.5])
    py = np.array([ty0 * TILE_PX + 0.5, ty0 * TILE_PX + 255.75, ty0 * TILE_PX + 400.0])
    got = sample_mosaic(mosaic, tx0, ty0, px, py)
    want = (px - 0.5) + 0.001 * (py - 0.5)
    _check(float(np.max(np.abs(got - want))) < 1e-9, f"sampling convention {got} vs {want}")
    try:
        sample_mosaic(
            mosaic, tx0, ty0, np.array([tx0 * TILE_PX + 0.2]), np.array([ty0 * TILE_PX + 5.0])
        )
    except ValueError:
        pass
    else:
        raise AssertionError("sampling outside the mosaic must raise, never extrapolate")
    x0, y0, x1, y1 = tile_range(
        np.array([256.0 * 5 + 1.0, 256.0 * 6 + 100.0]), np.array([256.0 * 2 + 128.0])
    )
    _check((x0, y0, x1, y1) == (4, 2, 6, 2), f"tile_range with bilinear margin {(x0, y0, x1, y1)}")
    _check(abs(metres_per_px(38.51, 12) - 29.91) < 0.01, "zoom 12 resolution at the site")


SELF_TESTS: list[Callable[[], None]] = [
    _test_terrarium_decode,
    _test_rg16_round_trip,
    _test_hillshade,
    _test_enu_round_trip,
    _test_sampling_convention,
]


def run_self_test() -> int:
    failed = 0
    t0 = time.perf_counter()
    for test in SELF_TESTS:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    elapsed = time.perf_counter() - t0
    print(f"{len(SELF_TESTS) - failed}/{len(SELF_TESTS)} self-tests passed in {elapsed:.2f} s")
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, BakeArgs]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-yaml", type=Path, default=DEFAULT_RUN_YAML, help="run section with the origin"
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="tile cache (gitignored)")
    p.add_argument("--half-width-km", type=float, default=8.0, help="half width of the square grid")
    p.add_argument(
        "--size", type=int, default=513, help="grid size in pixels (mesh = size-1 segments)"
    )
    p.add_argument("--zoom", type=int, default=12, help="tile zoom (12 is ~30 m/px at the site)")
    p.add_argument("--azimuth-deg", type=float, default=315.0, help="hillshade sun azimuth")
    p.add_argument("--altitude-deg", type=float, default=45.0, help="hillshade sun altitude")
    p.add_argument("--offline", action="store_true", help="use cached tiles only")
    p.add_argument("--self-test", action="store_true", help="run offline checks and exit")
    ns = p.parse_args(argv)
    if ns.size < 2 or ns.half_width_km <= 0 or not 0 < ns.altitude_deg <= 90:
        p.error("need --size >= 2, --half-width-km > 0 and 0 < --altitude-deg <= 90")
    return ns, BakeArgs(
        run_yaml=ns.run_yaml,
        out_dir=ns.out,
        cache_dir=ns.cache,
        half_width_km=ns.half_width_km,
        size=ns.size,
        zoom=ns.zoom,
        azimuth_deg=ns.azimuth_deg,
        altitude_deg=ns.altitude_deg,
        offline=ns.offline,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ns, args = parse_args(argv)
    if ns.self_test:
        return run_self_test()
    bake(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

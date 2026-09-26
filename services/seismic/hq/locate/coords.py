"""Geographic <-> local ENU coordinates (docs/01 -> Conventions).

``e``, ``n`` are UTM zone 12N (EPSG:32612) metres minus the origin's UTM coordinates, so they are
grid east/north, not true east/north (grid convergence is about -1.18 degrees at the showcase
origin: grid north lies west of true north).
``u = elevM - origin.elevM``. Both helpers are vectorized: they return float64 arrays shaped
like the inputs (0-d for scalars).
"""

from functools import lru_cache

import numpy as np
import numpy.typing as npt
from pyproj import Transformer

from hq.config.run import Origin

# Project-wide projection fixed by docs/01 (SceneMeta.projection "EPSG:32612 minus origin").
# A convention every lane shares, not a tunable knob.
PROJECTED_CRS = "EPSG:32612"
GEOGRAPHIC_CRS = "EPSG:4326"

ArrayLike = npt.ArrayLike
FloatArray = npt.NDArray[np.float64]


@lru_cache(maxsize=1)
def _to_utm() -> Transformer:
    return Transformer.from_crs(GEOGRAPHIC_CRS, PROJECTED_CRS, always_xy=True)


@lru_cache(maxsize=1)
def _from_utm() -> Transformer:
    return Transformer.from_crs(PROJECTED_CRS, GEOGRAPHIC_CRS, always_xy=True)


def _as_float(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _origin_utm(origin: Origin) -> tuple[float, float]:
    x0, y0 = _to_utm().transform(origin.lon, origin.lat, errcheck=True)
    return float(x0), float(y0)


def origin_utm(origin: Origin) -> tuple[float, float]:
    """The origin's EPSG:32612 easting and northing (m): ``e = easting - x0``, ``n = northing - y0``."""
    return _origin_utm(origin)


def to_enu(
    lat: ArrayLike, lon: ArrayLike, elev_m: ArrayLike, origin: Origin
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Latitude/longitude (degrees, WGS84) and elevation (m ASL) -> ``(e, n, u)`` in metres."""
    lat_a = _as_float(lat, "lat")
    lon_a = _as_float(lon, "lon")
    elev_a = _as_float(elev_m, "elev_m")
    x, y = _to_utm().transform(lon_a, lat_a, errcheck=True)
    x0, y0 = _origin_utm(origin)
    e = np.asarray(np.asarray(x, dtype=np.float64) - x0)
    n = np.asarray(np.asarray(y, dtype=np.float64) - y0)
    u = np.asarray(elev_a - origin.elevM)
    return e, n, u


def from_enu(
    e: ArrayLike, n: ArrayLike, u: ArrayLike, origin: Origin
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(e, n, u)`` in metres -> latitude, longitude (degrees, WGS84) and elevation (m ASL)."""
    e_a = _as_float(e, "e")
    n_a = _as_float(n, "n")
    u_a = _as_float(u, "u")
    x0, y0 = _origin_utm(origin)
    lon, lat = _from_utm().transform(e_a + x0, n_a + y0, errcheck=True)
    elev = np.asarray(u_a + origin.elevM)
    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64), elev

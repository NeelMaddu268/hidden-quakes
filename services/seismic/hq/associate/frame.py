"""PyOcto's coordinate frame and our search volume (LOC-03).

PyOcto 0.2.0 works in a local Cartesian frame in km with z pointing down.
``OctoAssociator.transform_stations`` (``associator.py``) puts ``z = -elevation / 1000`` and the
``StationSpecificVelocityModel1D`` docstring measures depth below sea level: km below sea level,
the same datum as our ``elevM``. The station-specific 1D model looks a point up at
table depth ``z - station.z + n_padding * delta`` (``VelocityModel.cpp``), so the event and the
station must share that datum; both use ``z = -elevM / 1000``, stations at ``sensorElevM``.

Horizontally, PyOcto's ``x``/``y`` are our ENU ``e``/``n`` in km (UTM 12N minus the run origin,
``hq.locate.coords``). PyOcto measures distance as the Euclidean norm in that frame. UTM's scale
factor across the showcase bbox is 0.99984-1.00006 (0.99994 at the origin; pyproj
``get_factors``), at most 16 m per 100 km, negligible next to ``minNodeSizeLocationKm``, so no
other projection is needed.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from hq.config.run import RunSection
from hq.config.seismology import AssociatorVolumeConfig
from hq.locate.coords import to_enu

M_PER_KM = 1000.0  # unit conversion

FloatArray = npt.NDArray[np.float64]


def pyocto_z_km(elev_m: npt.ArrayLike) -> FloatArray:
    """``elevM`` (m ASL, up positive) -> PyOcto ``z`` (km below sea level, down positive)."""
    return np.asarray(-np.asarray(elev_m, dtype=np.float64) / M_PER_KM)


def elev_m_from_pyocto_z(z_km: npt.ArrayLike) -> FloatArray:
    """PyOcto ``z`` (km below sea level) -> ``elevM`` (m ASL). Inverse of ``pyocto_z_km``."""
    return np.asarray(-np.asarray(z_km, dtype=np.float64) * M_PER_KM)


def pyocto_xy_km(e_m: npt.ArrayLike, n_m: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """ENU ``e``, ``n`` (m) -> PyOcto ``x``, ``y`` (km)."""
    return (
        np.asarray(np.asarray(e_m, dtype=np.float64) / M_PER_KM),
        np.asarray(np.asarray(n_m, dtype=np.float64) / M_PER_KM),
    )


def enu_from_pyocto_xy(x_km: npt.ArrayLike, y_km: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """PyOcto ``x``, ``y`` (km) -> ENU ``e``, ``n`` (m)."""
    return (
        np.asarray(np.asarray(x_km, dtype=np.float64) * M_PER_KM),
        np.asarray(np.asarray(y_km, dtype=np.float64) * M_PER_KM),
    )


@dataclass(frozen=True)
class SearchVolume:
    """The association search volume in ENU metres and elevM."""

    e_min_m: float
    e_max_m: float
    n_min_m: float
    n_max_m: float
    top_elev_m: float
    bottom_elev_m: float

    @property
    def xlim_km(self) -> tuple[float, float]:
        return (self.e_min_m / M_PER_KM, self.e_max_m / M_PER_KM)

    @property
    def ylim_km(self) -> tuple[float, float]:
        return (self.n_min_m / M_PER_KM, self.n_max_m / M_PER_KM)

    @property
    def zlim_km(self) -> tuple[float, float]:
        """PyOcto ``zlim``: (top, bottom) as km below sea level, so the smaller value first."""
        top, bottom = pyocto_z_km([self.top_elev_m, self.bottom_elev_m])
        return (float(top), float(bottom))

    def farthest_horizontal_m(self, e_m: FloatArray, n_m: FloatArray) -> FloatArray:
        """Per point, the largest horizontal distance (m) to any point of the volume (a corner)."""
        de = np.maximum(np.abs(e_m - self.e_min_m), np.abs(e_m - self.e_max_m))
        dn = np.maximum(np.abs(n_m - self.n_min_m), np.abs(n_m - self.n_max_m))
        return np.asarray(np.hypot(de, dn))

    def to_record(self) -> dict[str, Any]:
        return {
            "eMinM": self.e_min_m,
            "eMaxM": self.e_max_m,
            "nMinM": self.n_min_m,
            "nMaxM": self.n_max_m,
            "topElevM": self.top_elev_m,
            "bottomElevM": self.bottom_elev_m,
        }


def search_volume(cfg: AssociatorVolumeConfig, run: RunSection) -> SearchVolume:
    """The run bbox projected into ENU (bounding box of its corners) plus the margin, in elevM.

    The corners suffice for a bbox wholly west (or east) of the UTM zone's central meridian, as the
    showcase bbox is: along each edge easting and northing then change monotonically (tested).
    ``topElevM`` null means ``run.refSurfaceElevM``. Raises unless the top lies above the bottom.
    """
    min_lon, min_lat, max_lon, max_lat = run.bbox
    lats = np.array([min_lat, min_lat, max_lat, max_lat])
    lons = np.array([min_lon, max_lon, min_lon, max_lon])
    e, n, _ = to_enu(lats, lons, np.zeros(4), run.origin)
    margin = cfg.horizontalMarginM
    top = run.refSurfaceElevM if cfg.topElevM is None else cfg.topElevM
    if not top > cfg.bottomElevM:
        raise ValueError(
            f"association volume top {top} m ASL must lie above its bottom {cfg.bottomElevM} m ASL"
        )
    return SearchVolume(
        e_min_m=float(e.min()) - margin,
        e_max_m=float(e.max()) + margin,
        n_min_m=float(n.min()) - margin,
        n_max_m=float(n.max()) + margin,
        top_elev_m=float(top),
        bottom_elev_m=float(cfg.bottomElevM),
    )

"""Per-receiver 3D travel-time tables on the GDR 1800 3D velocity model (LOC-07), with scikit-fmm.

The model
    GDR 1800, "Cape EGS and Utah FORGE: Empirical 3D Seismic Velocity Model" (Nakata, Hopp, Wu,
    Robertson & Dadi, LBNL, 2025, CC BY 4.0); the paper is Nakata et al., SGP-TR-227 (2024),
    cached as ``data/cache/velocity/gdr1800/Nakata.pdf``. One NetCDF: ``Vp`` and ``Vs`` with dims
    (northing, easting, elevation), 601 x 601 x 201 nodes 50 m apart, float64 km/s, and no CRS
    attributes. What its coordinates and values mean was confirmed from the paper and the file
    before use (LOC-07 check, 2026-09-26, on the showcase run's origin and stations); every grid3d
    run repeats the per-station part in diagnostics.md:

    Horizontal CRS: easting/northing are EPSG:32612 (UTM zone 12N) metres. In a basin column the
    file follows the paper's p. 4 basin law Vp = 1.46 ln(6.7 x + 0.86) + 0.95 (x = depth below
    ground in km) with the depth stretched per column (the file name's "stretched"; p. 4 stretches
    the Vp/Vs profile to each column's basin depth), so the two topmost ground nodes give each
    column's stretch and ground elevation (a third node agrees within 0.2 m median). Placed in the
    run's ENU frame as EPSG:32612 minus the origin, that ground matches the 3DEP-based terrain
    baked for the web scene (apps/web/public/terrain: AWS Terrain Tiles, 3DEP in the US, resampled
    onto the same frame) with a median difference of +0.6 m and a MAD of 0.8 m over 73,851
    columns within 8 km of the origin. Shifting the model 50 m in any direction raises the MAD; on
    a 5 m search grid the best shift is (0, +5) m, zero within the terrain's 31 m pixels. A NAD27
    UTM 12N grid would sit about 65 m east and 203 m south of EPSG:32612 here, so the match rules
    it out; NAD83 UTM 12N differs by under a metre, which neither matters nor can be told apart.
    At the 19 showcase stations in basin columns the model's ground lies -4 to +7 m from their
    surfaceElevM (3DEP 1 m via EPQS). The paper's Fig. 2a marks the Cape site near (18, 17) km
    from the model's corner; its Fig. 1 puts it near 38.497 N, 112.905 W, which is (18.6, 16.5) km
    from the corner in EPSG:32612.

    Vertical: ``elevation`` is m above sea level, up positive (NAVD88, the 3DEP vertical): p. 3
    describes the model as "10 km deep (from a maximum altitude of 3000 m)" (nodes -7000..3000 m)
    with topography from the "1/3 arc second USGS 3D Elevation Program", and the ground match
    above ties the elevations to 3DEP heights. That is elevM, the pipeline's vertical everywhere.

    Air: p. 3 sets above-ground velocities to 0.01 km/s; the file does not. In a column holding any
    value other than its top value (a basin column), the nodes above the ground hold one constant
    (Vp 0.73271, Vs 0.40002 km/s) up to the model top. A column with one value at every elevation
    (Vp 5.8, Vs 3.392 km/s: basement from -7000 to 3000 m) carries no ground surface at all, so
    basement outcrop and air can't be told apart there. ``air_mask`` identifies air as the run of
    nodes from a column's top holding the file's air value (``velocity.model3d.airVp`` /
    ``airVs``).
    ``grid3d.airHandling`` then either gives those nodes the column's topmost ground velocity
    (``topSurfaceVelocity``) or keeps the file values (``asFile``). Constant columns keep their
    values either way: their air already holds their only velocity, so a ray there moves as
    through the basement it borders. Where such a column sits next to basin ground higher than
    its (unknown) ground, its air is faster than that basin ground; the model can't say where.
    Model features the 3D tables inherit: the west of the grid (easting below about 322350) is
    basement velocity at every elevation, although p. 3 says the basement surface was
    extrapolated to the model limits (clipped at the topography); p. 3 also notes a vertical
    discontinuity where the Mineral Mountains range front meets the basin fill.

    Receivers in constant columns (``grid3d.constantColumns``): such a column holds no ground
    surface and no basin data, so the file can't tell outcrop from missing data there (LOC-01:
    such a station may sit outside the real 3D model even inside the grid). ``fallback1d`` gives
    the receiver no 3D table (its 1D tables, named in the record); ``asFile`` builds one from the
    column's values, which puts basement rock from the model's bottom up to the sensor.

Frame and grid
    Tables live on a node lattice in the run's ENU frame (EPSG:32612 metres minus the run origin,
    ``hq.locate.coords``; elevM): nodes at multiples of ``spacingM`` in e and n, and at
    ``bottomElevM + k * spacingM`` in elevM. Horizontally the lattice covers the search volume and
    every receiver inside the model plus ``horizontalMarginM``, clipped to the model's node
    extent; its top covers the highest in-model receiver or the volume top plus ``topMarginM``
    (``make_grid3d``). A receiver outside the model (outside the model's node extent, off the
    lattice, or in a column with no finite value), or in a constant column under ``fallback1d``,
    gets no 3D table: the locator uses its 1D tables (``hq.locate.tt_grid``) and the record names
    it with the reason. The lattice covers constant-column receivers either way, so the knob
    never changes another receiver's table.

Resampling
    Each lattice node carries the mean slowness of its cell ``[x - h/2, x + h/2]`` on all three
    axes over the model's 50 m cells, weighted by overlap (``resample``; a separable weighted
    mean, cells without a finite value left out). This is tt_grid's cell-mean slowness in 3D.

Solve
    Per receiver and phase, with the source at the sensor (reciprocity) at its true (e, n,
    sensorElevM), and the receiver's own model column (50 m cells, air handled as configured,
    equal neighbouring cells merged) as a 1D layer model (``column_model``)::

        T = C(r, elevM) + F3d - Fcol

    ``C`` is the column's own 2D (r, elevM) eikonal table (``tt_grid.solve_table`` with the 1D
    ``grids`` knobs: 25 m, exact layered seed); ``F3d`` and ``Fcol`` are ``skfmm.travel_time``
    on the lattice with the resampled 3D speeds and with the column's speeds at every node
    (resampled the same way), both seeded alike. Seeding follows tt_grid: every node within
    ``seedRadiusM`` of the receiver gets the column's exact layered first arrival
    (``tt_grid.layered_first_arrival``) and keeps it, and the solve starts from an isochron inside
    that region (the earliest seeded time in its outer two cells; tt_grid's ``(R - 2 h) / vMax``
    bound leaves no node inside it on 100-200 m cells in slow near-surface rock). The lattice's
    discretisation error, common to ``F3d`` and ``Fcol``, cancels; a laterally uniform model
    gives exactly the column's 2D table on the nodes. What stays: the seed assumes the model is
    laterally uniform within the seed radius (it carries the station's near-surface column,
    which 100-200 m cells can't resolve), and trilinear interpolation between nodes.
    Measured (tests/seismology/test_tt_grid3d.py, and LOC-07's probes): at source depths
    (-4500..-500 m) a layered model's 200 m table is within 8 ms (S) / 4 ms (P) of the exact
    layered times, 100 m within 6.5 / 3.7 ms; within a few cells of a receiver interpolation
    reaches ~30 ms at 200 m. A laterally varying model with a closed form (velocity gradient
    0.35 /s down, 0.07 /s sideways): 200 m median +1 to +3 ms, max 9-16 ms (seed radius 1000 m;
    800 m doubled it), 100 m within 3.5 ms; without the column correction 200 m was biased
    +9 ms (median) with p95 18 ms.

Cache
    ``<cache_dir>/ttgrids/3d/<key>.npy`` plus a ``<key>.json`` sidecar. The key is a SHA-256 of the
    model file's SHA-256 and units, the air handling, the lattice (with the origin's EPSG:32612
    coordinates), the receiver position, the phase, the seed radius, the stencil order, the
    column table's 1D grid knobs, the scikit-fmm version, ``ALGORITHM`` and the SHA-256 of this
    module's and tt_grid's source. When
    every table is cached the model file is never read beyond its coordinates and the station
    columns. Lookups are trilinear (``Table3d.lookup``; ``lookup_box`` does the same separably on
    a tensor-product box).
"""

import hashlib
import io
import json
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import skfmm
from numpy.typing import ArrayLike, NDArray

from hq.config.seismology import Grid3dConfig, GridsConfig, Velocity3dConfig, VelocityConfig
from hq.locate import tt_grid
from hq.locate.tt_grid import PHASES, GridSpec, Phase, TravelTimeTable, layered_first_arrival
from hq.locate.velocity import LayerModel, SourceRef

if TYPE_CHECKING:
    from hq.locate.locator import TravelTimeProvider

log = logging.getLogger(__name__)

METHOD = "grid3d"  # LocationQuality.method
ALGORITHM = ("tt_grid3d/1: column 2D table + skfmm 3D minus column-speed 3D, overlap-weighted "
             "cell-mean slowness, exact receiver-column seed")
SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes() + Path(tt_grid.__file__).read_bytes()
).hexdigest()
CACHE_SUBDIR = Path(tt_grid.CACHE_SUBDIR) / "3d"
VELOCITY_CACHE_SUBDIR = "velocity"  # docs/01: velocity models live in data/cache/velocity/
DIMS = ("northing", "easting", "elevation")  # the file's dimension names, in array order
VARIABLES: dict[Phase, str] = {"P": "Vp", "S": "Vs"}
UNITS_TO_M_PER_S = {"km/s": 1000.0, "m/s": 1.0}
TOP_SURFACE_VELOCITY = "topSurfaceVelocity"
AS_FILE = "asFile"
FALLBACK_1D = "fallback1d"  # grid3d.constantColumns
CONSTANT_COLUMN_REASON = ("its model column holds one value at every elevation (no ground surface "
                          "or basin data in the file; grid3d.constantColumns fallback1d)")
SHA_CHUNK = 1 << 24

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

CRS_EVIDENCE = (
    "easting/northing are EPSG:32612 (UTM 12N) metres: the ground surface recovered from the "
    "model's basin columns (paper p. 4 basin Vp law, depth stretched per column) matches the "
    "3DEP-based terrain in the run's ENU frame at zero shift (LOC-07 check: median +0.6 m, MAD "
    "0.8 m over 73,851 columns; a 50 m shift in any direction raises the MAD; NAD27 would sit "
    "~65 m E / 203 m S and is ruled out). The file has no CRS attributes."
)
VERTICAL_EVIDENCE = (
    "elevation is m above sea level (NAVD88, the 3DEP vertical), up positive, nodes -7000..3000 m: "
    "Nakata.pdf p. 3 (10 km deep from a maximum altitude of 3000 m; topography from 3DEP 1/3 "
    "arc-second) and the ground match above."
)
AIR_EVIDENCE = (
    "Nakata.pdf p. 3 says 0.01 km/s above ground; the file differs: above-ground nodes of a basin "
    "column hold one constant (Vp 0.73271, Vs 0.40002 km/s) up to the model top, and a column "
    "with one value at every elevation (basement 5.8 / 3.392 km/s) carries no ground surface. Air "
    "= the run of nodes from a column's top holding that constant (velocity.model3d.airVp/airVs)."
)
AIR_RTOL = 1e-9  # the air value compares equal to this relative tolerance (unit conversion)


def _frozen(values: ArrayLike) -> FloatArray:
    arr = np.array(values, dtype=np.float64)
    arr.setflags(write=False)
    return arr


def _uniform_step(axis: FloatArray, name: str) -> float:
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"model axis {name} needs at least 2 nodes")
    steps = np.diff(axis)
    if not (np.all(steps > 0) and np.allclose(steps, steps[0], rtol=0, atol=1e-6)):
        raise ValueError(f"model axis {name} must be ascending and uniformly spaced")
    return float(steps[0])


# --- the model ------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Model3dSource:
    """A 3D P/S model: its identity, node axes and provenance; the values load on demand.

    Axes are node coordinates: ``easting_m`` / ``northing_m`` in EPSG:32612 metres, ``elev_m`` in
    m ASL, each ascending and uniformly spaced. Values are indexed (northing, easting, elevation).
    ``path`` is the NetCDF file; an in-memory model (tests) carries ``arrays`` (Vp, Vs in m/s)
    instead. ``sha256`` is the file's (or the arrays') SHA-256, part of every table key.
    """

    name: str
    sha256: str
    easting_m: FloatArray
    northing_m: FloatArray
    elev_m: FloatArray
    units: str  # of the stored values: "km/s" or "m/s"
    record: Mapping[str, Any]  # provenance for ProcessingRun.velocityModel
    path: Path | None = None
    arrays: tuple[FloatArray, FloatArray] | None = None
    air_m_per_s: tuple[float, float] | None = None  # (Vp, Vs) the model holds above ground

    def __post_init__(self) -> None:
        for attr in ("easting_m", "northing_m", "elev_m"):
            object.__setattr__(self, attr, _frozen(getattr(self, attr)))
            _uniform_step(getattr(self, attr), attr)
        if self.units not in UNITS_TO_M_PER_S:
            raise ValueError(f"units must be one of {sorted(UNITS_TO_M_PER_S)}, got {self.units!r}")
        if (self.path is None) == (self.arrays is None):
            raise ValueError("a 3D model needs exactly one of path and arrays")
        if self.arrays is not None:
            shape = (self.northing_m.size, self.easting_m.size, self.elev_m.size)
            for arr in self.arrays:
                if arr.shape != shape:
                    raise ValueError(f"model arrays have shape {arr.shape}, axes give {shape}")

    @property
    def spacing_m(self) -> tuple[float, float, float]:
        """Node spacing along (northing, easting, elevation)."""
        return (_uniform_step(self.northing_m, "northing"), _uniform_step(self.easting_m, "easting"),
                _uniform_step(self.elev_m, "elevation"))

    @property
    def factor(self) -> float:
        return UNITS_TO_M_PER_S[self.units]

    def identity(self) -> dict[str, Any]:
        """What a table key needs from the model."""
        return {"name": self.name, "sha256": self.sha256, "units": self.units,
                "toMPerS": self.factor,
                "airMPerS": None if self.air_m_per_s is None else list(self.air_m_per_s)}

    def load(self) -> tuple[FloatArray, FloatArray]:
        """(Vp, Vs) in m/s, float64, (northing, easting, elevation); NaN where the file has none."""
        started = time.perf_counter()
        if self.arrays is not None:
            vp, vs = (np.array(a, dtype=np.float64) for a in self.arrays)
        else:
            vp, vs = _read_variables(Path(str(self.path)))
        vp *= self.factor
        vs *= self.factor
        log.info("3D model %s: loaded Vp and Vs %s in %.1f s", self.name, vp.shape,
                 time.perf_counter() - started)
        return vp, vs

    def column(self, i_n: int, j_e: int) -> tuple[FloatArray, FloatArray]:
        """(Vp, Vs) of one column in m/s, bottom to top."""
        vp, vs = self.columns([i_n], [j_e])
        return vp[0], vs[0]

    def columns(self, i_n: ArrayLike, j_e: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """(Vp, Vs) in m/s of the columns at (northing, easting) indices, shape (len, elevation),
        bottom to top."""
        i = np.atleast_1d(np.asarray(i_n, dtype=np.intp))
        j = np.atleast_1d(np.asarray(j_e, dtype=np.intp))
        if self.arrays is not None:
            vp, vs = (np.array(a[i, j, :], dtype=np.float64) for a in self.arrays)
        else:
            import xarray as xr

            at = {"northing": xr.DataArray(i, dims="point"),
                  "easting": xr.DataArray(j, dims="point")}
            with xr.open_dataset(Path(str(self.path))) as ds:
                vp, vs = (ds[VARIABLES[ph]].isel(at).transpose("point", "elevation").to_numpy()
                          .astype(np.float64) for ph in PHASES)
        return vp * self.factor, vs * self.factor

    def to_record(self) -> dict[str, Any]:
        n, e, z = self.spacing_m
        return {
            **dict(self.record),
            "sha256": self.sha256,
            "units": self.units,
            "airMPerS": None if self.air_m_per_s is None else list(self.air_m_per_s),
            "nodes": {"northing": int(self.northing_m.size), "easting": int(self.easting_m.size),
                      "elevation": int(self.elev_m.size)},
            "spacingM": {"northing": n, "easting": e, "elevation": z},
            "eastingRangeM": [float(self.easting_m[0]), float(self.easting_m[-1])],
            "northingRangeM": [float(self.northing_m[0]), float(self.northing_m[-1])],
            "elevRangeM": [float(self.elev_m[0]), float(self.elev_m[-1])],
        }

    @classmethod
    def from_arrays(
        cls,
        name: str,
        easting_m: ArrayLike,
        northing_m: ArrayLike,
        elev_m: ArrayLike,
        vp_m_per_s: ArrayLike,
        vs_m_per_s: ArrayLike,
        record: Mapping[str, Any] | None = None,
        air_m_per_s: tuple[float, float] | None = None,
    ) -> "Model3dSource":
        """An in-memory model (m/s); its sha256 hashes the axes and values."""
        arrays = (_frozen(vp_m_per_s), _frozen(vs_m_per_s))
        digest = hashlib.sha256()
        for arr in (np.asarray(easting_m, np.float64), np.asarray(northing_m, np.float64),
                    np.asarray(elev_m, np.float64), *arrays):
            digest.update(np.ascontiguousarray(arr).tobytes())
        return cls(name=name, sha256=digest.hexdigest(), easting_m=_frozen(easting_m),
                   northing_m=_frozen(northing_m), elev_m=_frozen(elev_m), units="m/s",
                   record=dict(record or {"name": name}), arrays=arrays,
                   air_m_per_s=air_m_per_s)


def _read_variables(path: Path) -> tuple[FloatArray, FloatArray]:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        out = [ds[VARIABLES[ph]].transpose(*DIMS).to_numpy().astype(np.float64) for ph in PHASES]
    return out[0], out[1]


@lru_cache(maxsize=4)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    started = time.perf_counter()
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(SHA_CHUNK):
            digest.update(chunk)
    log.info("3D model: SHA-256 of %s (%d bytes) in %.1f s", path, size,
             time.perf_counter() - started)
    return digest.hexdigest()


def model3d_path(velocity: VelocityConfig, cache_dir: Path) -> Path:
    """``<cache_dir>/velocity/<model3d.cacheFile>``."""
    return Path(cache_dir) / VELOCITY_CACHE_SUBDIR / velocity.model3d.cacheFile


def open_model3d(path: Path, cfg: Velocity3dConfig) -> Model3dSource:
    """The configured 3D model file: axes and dims checked, SHA-256 taken, values not read."""
    import xarray as xr

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"3D velocity model {path} is missing: download {cfg.url} ({cfg.expectedBytes} bytes) "
            f"into {path.parent}"
        )
    stat = path.stat()
    if stat.st_size != cfg.expectedBytes:
        raise ValueError(f"{path} has {stat.st_size} bytes; velocity.model3d.expectedBytes is "
                         f"{cfg.expectedBytes} (partial download?)")
    with xr.open_dataset(path) as ds:
        for ph, var in VARIABLES.items():
            if var not in ds or set(ds[var].dims) != set(DIMS):
                raise ValueError(f"{path}: expected variable {var} ({ph}) with dims {DIMS}")
        axes = {dim: ds[dim].to_numpy().astype(np.float64) for dim in DIMS}
    record = {
        "name": path.stem,
        "source": SourceRef(citation=cfg.citation, url=cfg.submissionUrl, verified=True).to_record(),
        "verifiedBasis": "the values come from the GDR 1800 file itself; the CRS, vertical datum "
        "and air readings are ours, with the evidence below",
        "sourceFile": cfg.url,
        "license": cfg.license,
        "crs": cfg.crs,
        "datum": "elevation in m above sea level (NAVD88, the 3DEP vertical), up positive = elevM",
        "evidence": {"crs": CRS_EVIDENCE, "vertical": VERTICAL_EVIDENCE, "air": AIR_EVIDENCE,
                     "where": "hq.locate.tt_grid3d module docstring; diagnostics.md 3D section"},
    }
    return Model3dSource(
        name=path.stem,
        sha256=_file_sha256(str(path), stat.st_size, stat.st_mtime_ns),
        easting_m=axes["easting"], northing_m=axes["northing"], elev_m=axes["elevation"],
        units=cfg.units, record=record, path=path,
        air_m_per_s=(cfg.airVp * UNITS_TO_M_PER_S[cfg.units],
                     cfg.airVs * UNITS_TO_M_PER_S[cfg.units]),
    )


# --- air --------------------------------------------------------------------------------------


def air_mask(v: FloatArray, air_value: float) -> NDArray[np.bool_]:
    """Above-ground nodes of a (northing, easting, elevation) model (elevation ascending): per
    column, the run of nodes from the top holding ``air_value``. A column holding it at every
    node has no ground and gets none."""
    same = np.isclose(v, air_value, rtol=AIR_RTOL, atol=0.0)
    run = np.cumprod(same[:, :, ::-1], axis=2, dtype=np.int32).sum(axis=2)
    run = np.where(run == v.shape[2], 0, run)
    depth_from_top = np.arange(v.shape[2])[::-1]  # 0 at the top node
    return depth_from_top[None, None, :] < run[:, :, None]


def fill_air(v: FloatArray, air: NDArray[np.bool_]) -> FloatArray:
    """``v`` with every air node given its column's topmost ground value (a copy)."""
    n_air = air.sum(axis=2)
    ground_top = v.shape[2] - 1 - n_air  # index of the topmost ground node
    top_value = np.take_along_axis(v, ground_top[:, :, None], axis=2)[:, :, 0]
    out = v.copy()
    for k in range(v.shape[2]):
        sel = air[:, :, k]
        out[:, :, k][sel] = top_value[sel]
    return out


def handle_air(
    vp: FloatArray, vs: FloatArray, mode: str, air: tuple[float, float] | None
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Air per ``grid3d.airHandling`` (``air``: the model's above-ground Vp, Vs; None: the model
    holds no air value). Also returns counts for the record. Air is identified on Vp; Vs must
    hold its air value on the same nodes, or this raises."""
    if mode not in (AS_FILE, TOP_SURFACE_VELOCITY):
        raise ValueError(f"unknown airHandling {mode!r}")
    finite_cols = np.isfinite(vp).any(axis=2)
    constant = np.all(vp == vp[:, :, :1], axis=2)
    if air is None:
        mask = np.zeros(vp.shape, dtype=bool)
    else:
        mask = air_mask(vp, air[0])
        if np.any(mask & ~np.isclose(vs, air[1], rtol=AIR_RTOL, atol=0.0)):
            raise ValueError("3D model: Vs does not hold its air value on nodes that are air on Vp")
    columns = mask.any(axis=2)
    counts = {
        "airHandling": mode,
        "airValueMPerS": None if air is None else list(air),
        "airNodes": int(mask.sum()),
        "columnsWithAir": int(columns.sum()),
        "constantColumns": int((finite_cols & constant).sum()),
        "columnsWithoutFiniteValue": int((~finite_cols).sum()),
        "columns": int(columns.size),
    }
    if mode == AS_FILE or air is None:
        return vp, vs, counts
    return fill_air(vp, mask), fill_air(vs, mask), counts


def ground_elev_m(vp_columns: FloatArray, source: "Model3dSource") -> FloatArray:
    """Per column (rows of ``vp_columns``, m/s, bottom to top): the model's ground, midway between
    its topmost ground node and its lowest air node; NaN where the column holds no air (no ground
    surface in the file)."""
    cols = np.atleast_2d(np.asarray(vp_columns, dtype=np.float64))
    if source.air_m_per_s is None:
        return np.full(cols.shape[0], np.nan)
    n_air = air_mask(cols[None, :, :], source.air_m_per_s[0])[0].sum(axis=1)
    elev = np.asarray(source.elev_m)
    lowest_air = elev[np.clip(elev.size - n_air, 0, elev.size - 1)]
    return np.where(n_air > 0, lowest_air - source.spacing_m[2] / 2.0, np.nan)


# --- lattice --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid3dSpec:
    """Node lattice of the 3D tables: ENU ``e = e0 + i h``, ``n = n0 + j h``, ``elevM = z0 + k h``.

    ``origin_utm_*`` is the run origin's EPSG:32612 position, so the lattice is fixed on the model.
    """

    spacing_m: float
    e0_m: float
    n0_m: float
    z0_m: float
    n_e: int
    n_n: int
    n_z: int
    origin_utm_e_m: float
    origin_utm_n_m: float

    def __post_init__(self) -> None:
        if min(self.n_e, self.n_n, self.n_z) < 2:
            raise ValueError("a 3D table lattice needs at least 2 nodes along each axis")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Array shape (elevM, n, e)."""
        return (self.n_z, self.n_n, self.n_e)

    @property
    def e_max_m(self) -> float:
        return self.e0_m + (self.n_e - 1) * self.spacing_m

    @property
    def n_max_m(self) -> float:
        return self.n0_m + (self.n_n - 1) * self.spacing_m

    @property
    def top_elev_m(self) -> float:
        return self.z0_m + (self.n_z - 1) * self.spacing_m

    def e_nodes(self) -> FloatArray:
        return self.e0_m + np.arange(self.n_e, dtype=np.float64) * self.spacing_m

    def n_nodes(self) -> FloatArray:
        return self.n0_m + np.arange(self.n_n, dtype=np.float64) * self.spacing_m

    def z_nodes(self) -> FloatArray:
        return self.z0_m + np.arange(self.n_z, dtype=np.float64) * self.spacing_m

    def covers(self, e_m: ArrayLike, n_m: ArrayLike, elev_m: ArrayLike) -> NDArray[np.bool_]:
        e, n, z = (np.asarray(x, dtype=np.float64) for x in (e_m, n_m, elev_m))
        return ((e >= self.e0_m) & (e <= self.e_max_m) & (n >= self.n0_m) & (n <= self.n_max_m)
                & (z >= self.z0_m) & (z <= self.top_elev_m))

    def to_record(self) -> dict[str, Any]:
        return {
            "spacingM": self.spacing_m,
            "eRangeM": [self.e0_m, self.e_max_m],
            "nRangeM": [self.n0_m, self.n_max_m],
            "elevRangeM": [self.z0_m, self.top_elev_m],
            "nodes": {"e": self.n_e, "n": self.n_n, "elevM": self.n_z},
            "frame": "ENU = EPSG:32612 minus the run origin; elevM",
            "originUtmM": [self.origin_utm_e_m, self.origin_utm_n_m],
        }


@dataclass(frozen=True)
class VolumeBox:
    """The search volume's bounds (ENU m, elevM)."""

    e_min_m: float
    e_max_m: float
    n_min_m: float
    n_max_m: float
    bottom_elev_m: float
    top_elev_m: float


def model_extent_enu(source: Model3dSource, origin_utm: tuple[float, float]) -> dict[str, float]:
    """The model's node extent in the run's ENU frame."""
    x0, y0 = origin_utm
    return {"eMin": float(source.easting_m[0] - x0), "eMax": float(source.easting_m[-1] - x0),
            "nMin": float(source.northing_m[0] - y0), "nMax": float(source.northing_m[-1] - y0),
            "zMin": float(source.elev_m[0]), "zMax": float(source.elev_m[-1])}


def nearest_columns(source: Model3dSource, e_m: ArrayLike, n_m: ArrayLike,
                    origin_utm: tuple[float, float]) -> tuple[IntArray, IntArray]:
    """(northing, easting) indices of the model column nearest each ENU point; raises for a
    point outside the model's horizontal node extent."""
    x0, y0 = origin_utm
    dn, de, _ = source.spacing_m
    i = np.rint((np.atleast_1d(np.asarray(n_m, dtype=np.float64)) + y0 - source.northing_m[0])
                / dn).astype(np.intp)
    j = np.rint((np.atleast_1d(np.asarray(e_m, dtype=np.float64)) + x0 - source.easting_m[0])
                / de).astype(np.intp)
    bad = (i < 0) | (i >= source.northing_m.size) | (j < 0) | (j >= source.easting_m.size)
    if np.any(bad):
        raise ValueError(f"{int(bad.sum())} point(s) lie outside the 3D model's horizontal extent")
    return i, j


def make_grid3d(
    source: Model3dSource,
    cfg: Grid3dConfig,
    origin_utm: tuple[float, float],
    volume: VolumeBox,
    receivers: pd.DataFrame,
) -> Grid3dSpec:
    """The lattice covering the volume and ``receivers`` (e, n, sensorElevM columns) inside the
    model, plus the margins, clipped to the model's node extent. Raises if the search volume
    reaches outside it."""
    h = cfg.spacingM
    ext = model_extent_enu(source, origin_utm)
    e_all = [volume.e_min_m, volume.e_max_m, *receivers["enu_e"].astype(float)]
    n_all = [volume.n_min_m, volume.n_max_m, *receivers["enu_n"].astype(float)]
    e_lo = max(math.floor((min(e_all) - cfg.horizontalMarginM) / h), math.ceil(ext["eMin"] / h))
    e_hi = min(math.ceil((max(e_all) + cfg.horizontalMarginM) / h), math.floor(ext["eMax"] / h))
    n_lo = max(math.floor((min(n_all) - cfg.horizontalMarginM) / h), math.ceil(ext["nMin"] / h))
    n_hi = min(math.ceil((max(n_all) + cfg.horizontalMarginM) / h), math.floor(ext["nMax"] / h))
    if cfg.bottomElevM < ext["zMin"]:
        raise ValueError(f"grid3d.bottomElevM {cfg.bottomElevM} lies below the 3D model's bottom "
                         f"node {ext['zMin']}")
    highest = max([volume.top_elev_m, *receivers["sensorElevM"].astype(float)])
    k_top = math.ceil((highest + cfg.topMarginM - cfg.bottomElevM) / h)
    k_top = min(k_top, math.floor((ext["zMax"] - cfg.bottomElevM) / h))
    grid = Grid3dSpec(
        spacing_m=h, e0_m=e_lo * h, n0_m=n_lo * h, z0_m=cfg.bottomElevM, n_e=e_hi - e_lo + 1,
        n_n=n_hi - n_lo + 1, n_z=k_top + 1, origin_utm_e_m=float(origin_utm[0]),
        origin_utm_n_m=float(origin_utm[1]),
    )
    corners = grid.covers([volume.e_min_m, volume.e_max_m], [volume.n_min_m, volume.n_max_m],
                          [volume.bottom_elev_m, volume.top_elev_m])
    if not corners.all():
        raise ValueError(
            f"the search volume (e {volume.e_min_m}..{volume.e_max_m}, n {volume.n_min_m}.."
            f"{volume.n_max_m}, elevM {volume.bottom_elev_m}..{volume.top_elev_m}) reaches outside "
            f"the 3D lattice {grid.to_record()} (model extent in ENU {ext})"
        )
    return grid


# --- resampling -----------------------------------------------------------------------------------


def overlap_weights(nodes: FloatArray, h: float, cells: FloatArray) -> FloatArray:
    """(len(nodes), len(cells)): overlap length of each node's cell [x - h/2, x + h/2] with each
    model cell [c - d/2, c + d/2] (d the model spacing)."""
    d = _uniform_step(cells, "model axis")
    lo = np.maximum(nodes[:, None] - h / 2, cells[None, :] - d / 2)
    hi = np.minimum(nodes[:, None] + h / 2, cells[None, :] + d / 2)
    return np.clip(hi - lo, 0.0, None)


def resample(
    v: FloatArray, source: Model3dSource, grid: Grid3dSpec
) -> FloatArray:
    """Speed (m/s) on the lattice, shape (elevM, n, e): 1 / overlap-weighted mean slowness.

    ``v``: model velocities (m/s), (northing, easting, elevation). Model cells without a finite
    value are left out of each mean; a node whose cell holds none raises.
    """
    x0, y0 = grid.origin_utm_e_m, grid.origin_utm_n_m
    w_n = overlap_weights(grid.n_nodes() + y0, grid.spacing_m, np.asarray(source.northing_m))
    w_e = overlap_weights(grid.e_nodes() + x0, grid.spacing_m, np.asarray(source.easting_m))
    w_z = overlap_weights(grid.z_nodes(), grid.spacing_m, np.asarray(source.elev_m))
    spans = [np.flatnonzero(w.any(axis=0)) for w in (w_n, w_e, w_z)]
    sub = tuple(slice(s[0], s[-1] + 1) for s in spans)
    block = v[sub]
    finite = np.isfinite(block)
    with np.errstate(divide="ignore", invalid="ignore"):
        slowness = 1.0 / block
    slowness[~finite] = 0.0
    w_n, w_e, w_z = w_n[:, sub[0]], w_e[:, sub[1]], w_z[:, sub[2]]

    def contract(a: FloatArray) -> FloatArray:
        a = np.tensordot(a, w_z, axes=([2], [1]))  # (N, E, z)
        a = np.tensordot(a, w_e, axes=([1], [1]))  # (N, z, e)
        a = np.tensordot(a, w_n, axes=([0], [1]))  # (z, e, n)
        return np.ascontiguousarray(np.transpose(a, (0, 2, 1)))  # (z, n, e)

    num = contract(slowness)
    den = contract(finite.astype(np.float64))
    empty = den <= 0
    if np.any(empty):
        k = np.argwhere(empty)[0]
        raise ValueError(
            f"{int(empty.sum())} lattice node(s) have no finite model value in their cell (e.g. "
            f"e {grid.e_nodes()[k[2]]:.0f}, n {grid.n_nodes()[k[1]]:.0f}, elevM "
            f"{grid.z_nodes()[k[0]]:.0f}); shrink grid3d.horizontalMarginM"
        )
    return den / num


def resample_column(v_col: FloatArray, source: Model3dSource, grid: Grid3dSpec) -> FloatArray:
    """One model column's speed (m/s) at the lattice elevations, resampled as ``resample`` does
    (overlap-weighted mean slowness), so a laterally uniform model gives equal speeds."""
    w_z = overlap_weights(grid.z_nodes(), grid.spacing_m, np.asarray(source.elev_m))
    if not np.all(np.isfinite(v_col)):
        raise ValueError("a column with non-finite values has no lattice speeds")
    return np.asarray(w_z.sum(axis=1) / (w_z @ (1.0 / v_col)), dtype=np.float64)


def column_model(source: Model3dSource, vp_col: FloatArray, vs_col: FloatArray,
                 label: str) -> LayerModel:
    """A 1D layer model of one model column (m/s, bottom to top): each node is a layer from half a
    spacing below it to half above, equal neighbours merged; the bottom node is the half-space."""
    if not (np.all(np.isfinite(vp_col)) and np.all(np.isfinite(vs_col))):
        raise ValueError(f"column {label} has non-finite values")
    dz = source.spacing_m[2]
    tops = np.asarray(source.elev_m) + dz / 2
    vp_td, vs_td, tops_td = vp_col[::-1], vs_col[::-1], tops[::-1]  # top-down
    keep = np.ones(tops_td.size, dtype=bool)
    keep[1:] = (vp_td[1:] != vp_td[:-1]) | (vs_td[1:] != vs_td[:-1])
    ref = source.record.get("source")
    citation = str(ref["citation"]) if isinstance(ref, Mapping) and "citation" in ref else source.name
    return LayerModel(
        name=f"{source.name} column {label}",
        datum="topElevM is m above sea level (the 3D model's elevation), up positive.",
        source=SourceRef(citation=citation, url=str(source.record.get("sourceFile", "in-memory")),
                         verified=False),
        top_elev_m=tops_td[keep],
        vp_m_per_s=vp_td[keep],
        vs_m_per_s=vs_td[keep],
        source_file=str(source.record.get("sourceFile", "in-memory")),
        license=str(source.record.get("license", "n/a")),
    )


# --- solve ----------------------------------------------------------------------------------------


def _seeded_fmm(
    speed: FloatArray,
    grid: Grid3dSpec,
    receiver: tuple[float, float, float],
    column: LayerModel,
    phase: Phase,
    *,
    seed_radius_m: float,
    fmm_order: int,
) -> FloatArray:
    """Eikonal times (s), shape (elevM, n, e), for ``speed`` from a source at ``receiver``; every
    node within ``seed_radius_m`` keeps the exact layered time of ``column``."""
    h = grid.spacing_m
    er, nr, zr = (float(x) for x in receiver)
    z, n, e = grid.z_nodes(), grid.n_nodes(), grid.e_nodes()
    iz = np.flatnonzero(np.abs(z - zr) <= seed_radius_m)
    i_n = np.flatnonzero(np.abs(n - nr) <= seed_radius_m)
    ie = np.flatnonzero(np.abs(e - er) <= seed_radius_m)
    zz, nn, ee = np.meshgrid(z[iz], n[i_n], e[ie], indexing="ij")
    rh = np.hypot(ee - er, nn - nr)
    dist = np.sqrt(rh**2 + (zz - zr) ** 2)
    inside = dist <= seed_radius_m
    seed = np.full(zz.shape, np.nan)
    seed[inside] = layered_first_arrival(column, phase, zr, rh[inside], zz[inside])
    block = (slice(iz[0], iz[-1] + 1), slice(i_n[0], i_n[-1] + 1), slice(ie[0], ie[-1] + 1))
    # Start isochron: the earliest seeded time in the outer shell (the last two cells of the seed
    # radius), so every node earlier than it, and its neighbours, carry exact seeded times. (tt_grid
    # bounds it with the fastest velocity instead; on 3D cells of 100-200 m in slow near-surface
    # rock that bound leaves no node inside the isochron.)
    shell = inside & (dist > seed_radius_m - 2.0 * h)
    t1 = float(np.min(seed[shell]))
    phi = np.full(grid.shape, seed_radius_m)
    phi_block = phi[block]
    phi_block[inside] = (seed[inside] - t1) * speed[block][inside]
    if not np.any(phi_block < 0):
        raise ValueError("seed isochron is empty; seed_radius_m is too small for this lattice")
    times = skfmm.travel_time(phi, speed, dx=[h, h, h], order=fmm_order)
    if np.ma.is_masked(times):
        raise RuntimeError("scikit-fmm left unreached nodes in the 3D travel-time table")
    out = np.asarray(times, dtype=np.float64) + t1
    out_block = out[block]
    out_block[inside] = seed[inside]
    return out


def column_table(
    column: LayerModel, phase: Phase, receiver: tuple[float, float, float], grid: Grid3dSpec,
    cfg: GridsConfig,
) -> TravelTimeTable:
    """The receiver column's own 2D (r, elevM) table (``tt_grid.solve_table`` with the 1D grid
    knobs ``grids.drM``, ``dzM``, ``seedRadiusM``, ``fmmOrder``), reaching every lattice node."""
    er, nr, zr = (float(x) for x in receiver)
    reach = max(math.hypot(ce - er, cn - nr) for ce in (grid.e0_m, grid.e_max_m)
                for cn in (grid.n0_m, grid.n_max_m))
    bottom = grid.z0_m - cfg.dzM
    spec = GridSpec(
        dr_m=cfg.drM, dz_m=cfg.dzM, n_r=math.ceil(reach / cfg.drM) + 1,
        n_z=math.ceil((grid.top_elev_m - bottom) / cfg.dzM) + 1, bottom_elev_m=bottom,
    )
    times = tt_grid.solve_table(column, phase, zr, spec, seed_radius_m=cfg.seedRadiusM,
                                fmm_order=cfg.fmmOrder)
    times.setflags(write=False)
    return TravelTimeTable(phase, zr, spec, times, key="column")


def solve_table3d(
    speed: FloatArray,
    column_speed: FloatArray,
    grid: Grid3dSpec,
    receiver: tuple[float, float, float],
    column: LayerModel,
    phase: Phase,
    *,
    seed_radius_m: float,
    fmm_order: int,
    column_grid: GridsConfig,
) -> FloatArray:
    """First-arrival times (s), shape (elevM, n, e), from a source at ``receiver`` (e, n, elevM).

    ``speed``: this phase's lattice speeds (m/s); ``column``: the receiver's own model column and
    ``column_speed`` its speeds at the lattice elevations (resampled the same way). The table is
    the column's own 2D table (``column_table``) plus the 3D solve minus the same solve with the
    column's speeds everywhere, both seeded with the column's exact layered times within
    ``seed_radius_m``: the lattice's discretisation error, common to both solves, cancels, and a
    laterally uniform model gives the column's 2D table exactly.
    """
    h = grid.spacing_m
    if speed.shape != grid.shape or column_speed.shape != (grid.n_z,):
        raise ValueError(f"speeds {speed.shape} / {column_speed.shape} don't fit the lattice "
                         f"{grid.shape}")
    if seed_radius_m < 4.0 * h:
        raise ValueError("seed_radius_m must be at least 4 grid cells")
    er, nr, zr = (float(x) for x in receiver)
    if not bool(grid.covers(er, nr, zr)):
        raise ValueError(f"receiver {receiver} lies outside the 3D lattice")
    kw = {"seed_radius_m": seed_radius_m, "fmm_order": fmm_order}
    f3d = _seeded_fmm(speed, grid, receiver, column, phase, **kw)
    uniform = np.ascontiguousarray(np.broadcast_to(column_speed[:, None, None], grid.shape))
    fcol = _seeded_fmm(uniform, grid, receiver, column, phase, **kw)
    ref = column_table(column, phase, receiver, grid, column_grid)
    rh = np.hypot(grid.e_nodes()[None, :] - er, grid.n_nodes()[:, None] - nr)
    out = ref.lookup(rh[None, :, :], grid.z_nodes()[:, None, None]) + (f3d - fcol)
    if not np.all(np.isfinite(out)) or np.any(out < 0):
        raise RuntimeError("3D travel-time table has non-finite or negative values")
    return np.ascontiguousarray(out)


# --- tables and lookups ---------------------------------------------------------------------------


def _axis_index(x: FloatArray, x0: float, h: float, n: int, name: str,
                what: str) -> tuple[IntArray, FloatArray]:
    f = (x - x0) / h
    bad = ~np.isfinite(f) | (f < 0) | (f > n - 1)
    if np.any(bad):
        raise ValueError(
            f"{int(np.count_nonzero(bad))} lookup point(s) have {name} outside the 3D table "
            f"{what} ({x0} to {x0 + (n - 1) * h} m)"
        )
    i = np.minimum(f.astype(np.intp), n - 2)
    return i, f - i


@dataclass(frozen=True, eq=False)
class Table3d:
    """``times_s[k, j, i]``: first-arrival time from the receiver to node (e_i, n_j, elevM_k)."""

    station_id: str
    phase: Phase
    receiver: tuple[float, float, float]  # e, n, elevM
    grid: Grid3dSpec
    times_s: FloatArray  # read-only, shape grid.shape
    key: str

    def __post_init__(self) -> None:
        if self.times_s.shape != self.grid.shape:
            raise ValueError(f"table shape {self.times_s.shape} does not match its lattice")
        if self.times_s.flags.writeable:
            raise ValueError("table arrays must be read-only")

    @property
    def _what(self) -> str:
        return f"{self.phase} for {self.station_id}"

    def lookup(self, e_m: ArrayLike, n_m: ArrayLike, elev_m: ArrayLike) -> FloatArray:
        """Trilinear T(e, n, elevM) for broadcastable arrays; raises for a point off the lattice."""
        g = self.grid
        e, n, z = np.broadcast_arrays(*(np.asarray(x, dtype=np.float64) for x in (e_m, n_m, elev_m)))
        ie, ae = _axis_index(e, g.e0_m, g.spacing_m, g.n_e, "e", self._what)
        jn, an = _axis_index(n, g.n0_m, g.spacing_m, g.n_n, "n", self._what)
        kz, az = _axis_index(z, g.z0_m, g.spacing_m, g.n_z, "elevM", self._what)
        t = self.times_s
        c00 = (1 - ae) * t[kz, jn, ie] + ae * t[kz, jn, ie + 1]
        c01 = (1 - ae) * t[kz, jn + 1, ie] + ae * t[kz, jn + 1, ie + 1]
        c10 = (1 - ae) * t[kz + 1, jn, ie] + ae * t[kz + 1, jn, ie + 1]
        c11 = (1 - ae) * t[kz + 1, jn + 1, ie] + ae * t[kz + 1, jn + 1, ie + 1]
        low = (1 - an) * c00 + an * c01
        high = (1 - an) * c10 + an * c11
        return np.asarray((1 - az) * low + az * high, dtype=np.float64)

    def lookup_box(self, e_m: ArrayLike, n_m: ArrayLike, elev_m: ArrayLike) -> FloatArray:
        """Trilinear times on the tensor-product box of 1D ``e``, ``n``, ``elevM`` axes, shape
        (len(elevM), len(n), len(e)), interpolated one axis at a time (the same values as
        ``lookup`` on the meshgrid, to float rounding)."""
        g = self.grid
        e, n, z = (np.asarray(x, dtype=np.float64).ravel() for x in (e_m, n_m, elev_m))
        ie, ae = _axis_index(e, g.e0_m, g.spacing_m, g.n_e, "e", self._what)
        jn, an = _axis_index(n, g.n0_m, g.spacing_m, g.n_n, "n", self._what)
        kz, az = _axis_index(z, g.z0_m, g.spacing_m, g.n_z, "elevM", self._what)
        z0, z1 = int(kz.min()), int(kz.max()) + 2
        n0, n1 = int(jn.min()), int(jn.max()) + 2
        sub = self.times_s[z0:z1, n0:n1, :]
        te = sub[:, :, ie] * (1 - ae) + sub[:, :, ie + 1] * ae
        jr = jn - n0
        tn = te[:, jr, :] * (1 - an)[None, :, None] + te[:, jr + 1, :] * an[None, :, None]
        kr = kz - z0
        out = tn[kr] * (1 - az)[:, None, None] + tn[kr + 1] * az[:, None, None]
        return np.asarray(out, dtype=np.float64)


def table_key(
    source: Model3dSource,
    air_handling: str,
    grid: Grid3dSpec,
    phase: Phase,
    receiver: tuple[float, float, float],
    *,
    seed_radius_m: float,
    fmm_order: int,
    column_grid: GridsConfig,
) -> tuple[str, dict[str, Any]]:
    """The cache key (SHA-256 hex) of a 3D table and the JSON payload it hashes."""
    payload = {
        "algorithm": ALGORITHM,
        "solverSourceSha256": SOURCE_SHA256,
        "scikitFmm": skfmm.__version__,
        "model": source.identity(),
        "airHandling": air_handling,
        "grid": grid.to_record(),
        "phase": phase,
        "receiver": {"eM": float(receiver[0]), "nM": float(receiver[1]),
                     "elevM": float(receiver[2])},
        "seedRadiusM": float(seed_radius_m),
        "fmmOrder": int(fmm_order),
        "columnTable": {"drM": column_grid.drM, "dzM": column_grid.dzM,
                        "seedRadiusM": column_grid.seedRadiusM, "fmmOrder": column_grid.fmmOrder},
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), payload


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _npy_bytes(array: FloatArray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


# --- station tables -------------------------------------------------------------------------------


@dataclass(frozen=True)
class StationColumn:
    """Where a receiver sits in the 3D model and what its column holds."""

    station_id: str
    in_model: bool
    reason: str | None  # why it has no 3D table (None: it has one)
    # "basin": air nodes above a ground surface; "constant": one value at every elevation (no
    # ground surface in the file); "other": varies with elevation, no air value on top.
    kind: str | None
    ground_low_elev_m: float | None  # basin: the ground lies above this node ...
    ground_high_elev_m: float | None  # ... and at or below this one (the lowest air node)
    column_index: tuple[int, int] | None  # (northing, easting) index of the nearest model column

    def to_record(self) -> dict[str, Any]:
        return {"inModel": self.in_model, "reason": self.reason, "columnKind": self.kind,
                "groundBetweenElevM": (None if self.ground_low_elev_m is None else
                                       [self.ground_low_elev_m, self.ground_high_elev_m])}


def station_columns(
    source: Model3dSource, stations: pd.DataFrame, origin_utm: tuple[float, float]
) -> dict[str, StationColumn]:
    """Per station: inside the model's node extent (horizontally and in elevation) with a finite
    nearest column, that column's kind, and for a basin column the nodes the ground lies between."""
    x0, y0 = origin_utm
    ext = model_extent_enu(source, origin_utm)
    dn, de, _ = source.spacing_m
    out: dict[str, StationColumn] = {}
    for sid, e, n, z in stations[["id", "enu_e", "enu_n", "sensorElevM"]].itertuples(index=False):
        sid, e, n, z = str(sid), float(e), float(n), float(z)
        if not (ext["eMin"] <= e <= ext["eMax"] and ext["nMin"] <= n <= ext["nMax"]):
            out[sid] = StationColumn(sid, False, "outside the model's horizontal extent", None,
                                     None, None, None)
            continue
        if not ext["zMin"] <= z <= ext["zMax"]:
            out[sid] = StationColumn(sid, False, f"sensorElevM {z:.1f} outside the model's "
                                     f"elevations", None, None, None, None)
            continue
        i_n = round(float(n + y0 - source.northing_m[0]) / dn)
        j_e = round(float(e + x0 - source.easting_m[0]) / de)
        vp, vs = source.column(i_n, j_e)
        if not (np.all(np.isfinite(vp)) and np.all(np.isfinite(vs))):
            out[sid] = StationColumn(sid, False, "its model column has no (or partial) values",
                                     None, None, None, (i_n, j_e))
            continue
        air = (np.zeros(vp.size, dtype=bool) if source.air_m_per_s is None
               else air_mask(vp[None, None, :], source.air_m_per_s[0])[0, 0])
        if not air.any():
            kind = "constant" if np.all(vp == vp[0]) else "other"
            out[sid] = StationColumn(sid, True, None, kind, None, None, (i_n, j_e))
            continue
        k_air = int(np.flatnonzero(air)[0])
        out[sid] = StationColumn(sid, True, None, "basin", float(source.elev_m[k_air - 1]),
                                 float(source.elev_m[k_air]), (i_n, j_e))
    return out


@dataclass(frozen=True, eq=False)
class StationTables3d:
    """3D tables of the receivers inside the model, one per (station, phase)."""

    source: Model3dSource
    grid: Grid3dSpec
    tables: Mapping[tuple[str, Phase], Table3d]
    columns: Mapping[str, StationColumn]  # every station passed in
    air_handling: str
    constant_columns: str  # grid3d.constantColumns
    seed_radius_m: float
    fmm_order: int
    resampling: Mapping[str, Any]  # air counts etc. from the build that wrote the tables
    n_built: int
    n_loaded: int
    build_s: float

    @property
    def stations_3d(self) -> tuple[str, ...]:
        return tuple(sorted({sid for sid, _ in self.tables}))

    @property
    def fallback(self) -> dict[str, str]:
        """Stations without a 3D table and why."""
        return {sid: str(c.reason) for sid, c in sorted(self.columns.items()) if not c.in_model}

    def has(self, station_id: str) -> bool:
        return (station_id, "P") in self.tables

    def table(self, station_id: str, phase: Phase) -> Table3d:
        try:
            return self.tables[(station_id, phase)]
        except KeyError as err:
            raise KeyError(f"no 3D travel-time table for station {station_id!r} {phase}") from err

    def column_model_of(self, station_id: str) -> LayerModel:
        """The model column a 3D table's receiver sits in, air handled as for its table (the
        layers its near-receiver seed used)."""
        if not self.has(station_id):
            raise KeyError(f"station {station_id!r} has no 3D table")
        index = self.columns[station_id].column_index
        if index is None:
            raise ValueError(f"station {station_id!r} has no model column")
        vp, vs = self.source.columns([index[0]], [index[1]])
        vp, vs, _ = handle_air(vp[None], vs[None], self.air_handling, self.source.air_m_per_s)
        return column_model(self.source, vp[0, 0], vs[0, 0], f"{station_id} {index}")

    def to_record(self) -> dict[str, Any]:
        return {
            "algorithm": ALGORITHM,
            "solverSourceSha256": SOURCE_SHA256,
            "scikitFmm": skfmm.__version__,
            "grid": self.grid.to_record(),
            "airHandling": self.air_handling,
            "constantColumns": self.constant_columns,
            "seedRadiusM": self.seed_radius_m,
            "fmmOrder": self.fmm_order,
            "resampling": dict(self.resampling),
            "model": self.source.to_record(),
            "stations3d": list(self.stations_3d),
            "fallback1d": self.fallback,
            "stationColumns": {sid: c.to_record() for sid, c in sorted(self.columns.items())},
            "tables": [{"stationId": sid, "phase": ph, "key": t.key}
                       for (sid, ph), t in sorted(self.tables.items())],
            "nBuilt": self.n_built,
            "nLoaded": self.n_loaded,
            "buildS": self.build_s,
        }


def _load_cached(path: Path, sidecar_path: Path, payload: dict[str, Any],
                 grid: Grid3dSpec) -> tuple[FloatArray, dict[str, Any]]:
    times = np.load(path, mmap_mode="r", allow_pickle=False).view(np.ndarray)
    if times.shape != grid.shape or times.dtype != np.float64:
        raise ValueError(f"cached 3D table {path} has shape {times.shape} {times.dtype}; delete it")
    if not sidecar_path.exists():
        raise ValueError(f"cached 3D table {path} has no sidecar {sidecar_path.name}; delete it")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    build = sidecar.pop("build", None)
    if sidecar != payload or build is None:
        raise ValueError(f"sidecar {sidecar_path} does not match the table key; delete both")
    return times, build


def build_station_tables3d(
    stations: pd.DataFrame,
    source: Model3dSource,
    cfg: Grid3dConfig,
    *,
    origin_utm: tuple[float, float],
    volume: VolumeBox,
    cache_dir: Path,
    vp_range_m_per_s: tuple[float, float],
    vs_range_m_per_s: tuple[float, float],
    column_grid: GridsConfig,
) -> StationTables3d:
    """P and S 3D tables for every station inside the model (columns id, enu_e, enu_n,
    sensorElevM), built or loaded from ``<cache_dir>/ttgrids/3d/``.

    Stations outside the model, and under ``constantColumns: fallback1d`` those in a constant
    column, get none (``StationTables3d.fallback`` says why). The lattice covers every station
    inside the model's extent, whatever the knob says. The model's values are read only when a
    table has to be built; they must lie inside the configured unit guards
    (``velocity.plausibleVpMPerS`` / ``plausibleVsMPerS``, in m/s after conversion).
    ``column_grid`` (the 1D ``grids`` knobs) sets each receiver column's own 2D table
    (``solve_table3d``).
    """
    started = time.perf_counter()
    for column in ("id", "enu_e", "enu_n", "sensorElevM"):
        if column not in stations.columns:
            raise ValueError(f"stations need column {column!r}")
    columns = station_columns(source, stations, origin_utm)
    in_model = stations[np.array([columns[str(s)].in_model for s in stations["id"]], dtype=bool)]
    grid = make_grid3d(source, cfg, origin_utm, volume, in_model)
    for sid, e, n, z in in_model[["id", "enu_e", "enu_n", "sensorElevM"]].itertuples(index=False):
        c = columns[str(sid)]
        if not bool(grid.covers(e, n, z)):
            reason = "outside the 3D lattice (model edge)"
        elif c.kind == "constant" and cfg.constantColumns == FALLBACK_1D:
            reason = CONSTANT_COLUMN_REASON
        else:
            continue
        columns[str(sid)] = StationColumn(str(sid), False, reason, c.kind, c.ground_low_elev_m,
                                          c.ground_high_elev_m, c.column_index)
    receivers = {
        str(sid): (float(e), float(n), float(z))
        for sid, e, n, z in stations[["id", "enu_e", "enu_n", "sensorElevM"]].itertuples(index=False)
        if columns[str(sid)].in_model
    }
    folder = Path(cache_dir) / CACHE_SUBDIR
    plan: dict[tuple[str, Phase], tuple[str, dict[str, Any]]] = {
        (sid, ph): table_key(source, cfg.airHandling, grid, ph, rec,
                             seed_radius_m=cfg.seedRadiusM, fmm_order=cfg.fmmOrder,
                             column_grid=column_grid)
        for sid, rec in receivers.items() for ph in PHASES
    }
    missing = [k for k, (key, _) in plan.items() if not (folder / f"{key}.npy").exists()]
    built: dict[tuple[str, Phase], FloatArray] = {}
    resampling: dict[str, Any] = {}
    if missing:
        built, resampling = _build(source, cfg, grid, columns, receivers, missing, plan, folder,
                                   vp_range_m_per_s, vs_range_m_per_s, column_grid)
    tables: dict[tuple[str, Phase], Table3d] = {}
    for (sid, ph), (key, payload) in plan.items():
        if (sid, ph) in built:
            times = built[(sid, ph)]
        else:
            times, build = _load_cached(folder / f"{key}.npy", folder / f"{key}.json", payload, grid)
            resampling = resampling or dict(build.get("resampling", {}))
        tables[(sid, ph)] = Table3d(sid, ph, receivers[sid], grid, times, key)
    elapsed = time.perf_counter() - started
    fallback = {sid: c.reason for sid, c in columns.items() if not c.in_model}
    log.info(
        "3D travel-time tables: %d of %d stations on the 3D model (%d tables: %d built, %d from "
        "cache) on a %d x %d x %d lattice at %.0f m in %.1f s; 1D fallback: %s",
        len(receivers), len(columns), len(tables), len(built), len(tables) - len(built),
        grid.n_e, grid.n_n, grid.n_z, grid.spacing_m, elapsed, fallback or "none",
    )
    return StationTables3d(
        source=source, grid=grid, tables=tables, columns=columns, air_handling=cfg.airHandling,
        constant_columns=cfg.constantColumns, seed_radius_m=cfg.seedRadiusM,
        fmm_order=cfg.fmmOrder, resampling=resampling,
        n_built=len(built), n_loaded=len(tables) - len(built), build_s=elapsed,
    )


def _check_range(v: FloatArray, bounds: tuple[float, float], name: str) -> None:
    finite = v[np.isfinite(v)]
    if finite.size and (finite.min() < bounds[0] or finite.max() > bounds[1]):
        raise ValueError(
            f"3D model {name} spans {finite.min():.1f}..{finite.max():.1f} m/s, outside the unit "
            f"guard {list(bounds)} m/s (velocity.model3d.units wrong?)"
        )


def _build(
    source: Model3dSource,
    cfg: Grid3dConfig,
    grid: Grid3dSpec,
    columns: Mapping[str, StationColumn],
    receivers: Mapping[str, tuple[float, float, float]],
    missing: Sequence[tuple[str, Phase]],
    plan: Mapping[tuple[str, Phase], tuple[str, dict[str, Any]]],
    folder: Path,
    vp_range: tuple[float, float],
    vs_range: tuple[float, float],
    column_grid: GridsConfig,
) -> tuple[dict[tuple[str, Phase], FloatArray], dict[str, Any]]:
    t_start = time.perf_counter()
    vp, vs = source.load()
    _check_range(vp, vp_range, "Vp")
    _check_range(vs, vs_range, "Vs")
    vp, vs, counts = handle_air(vp, vs, cfg.airHandling, source.air_m_per_s)
    t_load = time.perf_counter() - t_start
    speeds = {"P": resample(vp, source, grid), "S": resample(vs, source, grid)}
    t_resample = time.perf_counter() - t_start - t_load
    resampling = {**counts, "loadS": t_load, "resampleS": t_resample,
                  "method": "overlap-weighted mean slowness over the model's cells"}
    log.info("3D model resampled to %.0f m (%d x %d x %d nodes) in %.1f s (load and air %.1f s); "
             "air nodes %d in %d columns, %d constant columns", grid.spacing_m, grid.n_e,
             grid.n_n, grid.n_z, t_resample, t_load, counts["airNodes"],
             counts["columnsWithAir"], counts["constantColumns"])
    folder.mkdir(parents=True, exist_ok=True)
    built: dict[tuple[str, Phase], FloatArray] = {}
    for sid, ph in missing:
        t0 = time.perf_counter()
        c = columns[sid]
        if c.column_index is None:
            raise ValueError(f"station {sid} has no model column")
        i_n, j_e = c.column_index
        col = column_model(source, vp[i_n, j_e, :], vs[i_n, j_e, :], f"{sid} ({i_n}, {j_e})")
        v_col = (vp if ph == "P" else vs)[i_n, j_e, :]
        times = solve_table3d(speeds[ph], resample_column(v_col, source, grid), grid,
                              receivers[sid], col, ph, seed_radius_m=cfg.seedRadiusM,
                              fmm_order=cfg.fmmOrder, column_grid=column_grid)
        key, payload = plan[(sid, ph)]
        solve_s = time.perf_counter() - t0
        build = {"stationId": sid, "solveS": solve_s, "columnLayers": col.n_layers,
                 "columnKind": c.kind, "resampling": resampling}
        text = json.dumps({**payload, "build": build}, indent=2, sort_keys=True, allow_nan=False)
        _write_atomic(folder / f"{key}.json", text.encode("utf-8"))
        _write_atomic(folder / f"{key}.npy", _npy_bytes(times))
        times.setflags(write=False)
        built[(sid, ph)] = times
        log.info("built 3D %s table for %s (%s column, %d layers) in %.2f s -> %s", ph, sid, c.kind,
                 col.n_layers, solve_s, key[:16])
    return built, resampling


# --- the grid3d travel-time provider ---------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Grid3dProvider:
    """Travel times from the 3D tables, and from ``fallback`` (the 1D provider) for stations
    without one. The Locator's ``TravelTimeProvider`` for ``locator.method: grid3d``."""

    tables3d: StationTables3d
    fallback: "TravelTimeProvider"
    method: str = field(default=METHOD)

    def times(self, station_id: str, phase: Phase, e_m: ArrayLike, n_m: ArrayLike,
              elev_m: ArrayLike) -> FloatArray:
        if self.tables3d.has(station_id):
            return self.tables3d.table(station_id, phase).lookup(e_m, n_m, elev_m)
        return self.fallback.times(station_id, phase, e_m, n_m, elev_m)

    def times_box(self, station_id: str, phase: Phase, e_m: FloatArray, n_m: FloatArray,
                  elev_m: FloatArray) -> FloatArray:
        if self.tables3d.has(station_id):
            return self.tables3d.table(station_id, phase).lookup_box(e_m, n_m, elev_m)
        return self.fallback.times_box(station_id, phase, e_m, n_m, elev_m)

    def covers(self, e_m: float, n_m: float, elev_m: float) -> bool:
        return bool(self.tables3d.grid.covers(e_m, n_m, elev_m)) and self.fallback.covers(
            e_m, n_m, elev_m)

    def to_record(self) -> dict[str, Any]:
        return {"grid3d": self.tables3d.to_record(), "grid1dFallback": self.fallback.to_record()}

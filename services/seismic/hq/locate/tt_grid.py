"""Per-receiver 2D (r, elevM) travel-time tables for a 1D layered model, solved with scikit-fmm.

Reciprocity: the time from a hypocentre to a sensor equals the time from the sensor to the
hypocentre, so each table puts the source at the sensor's true ``sensorElevM`` (on the axis r = 0)
and holds the first-arrival time ``T(r, elevM)`` to every node. For a laterally uniform model the
first arrival depends only on the epicentral distance r and the elevation, so the axisymmetric 2D
eikonal ``|grad T| = 1 / v(elevM)`` in the (r, elevM) plane is exact up to grid error.

Grid: nodes ``r = i * drM`` for ``i = 0 .. rMaxM / drM`` and ``elevM = bottomElevM + k * dzM`` up to
the grid top, which is the highest receiver (or the search-volume top, if higher) plus
``topMarginM``, snapped up onto that lattice. The model's top layer is extended up to the grid top
explicitly (``LayerModel.with_top_extended_to``), and the extension is part of the table key and
of the recorded velocity model.

Node speeds: each node carries the mean slowness of its cell ``[elevM - dzM/2, elevM + dzM/2]``
(clipped to the grid), so an interface between nodes is placed to first order in its true
position instead of snapping to the nearest node.

Near-source initialisation: a plain point source makes the fast marching method first-order
accurate. Instead, every node within ``seedRadiusM`` of the source gets the exact 1D layered
first-arrival time (``layered_first_arrival``: direct/transmitted rays and head waves, including
interfaces close to the source). The eikonal solve starts from the isochron ``T = t1`` inside that
region, with ``t1 = (seedRadiusM - 2 * max(drM, dzM)) / vMax``, so the whole isochron and its grid
neighbours lie inside the seeded region, and the seeded nodes keep their exact times.

Accuracy record: cell-mean slowness is first order at interfaces, so a table is slow by up to about
14 ms (S) at nodes just below a shallow interface, and by about 3 ms or less at elevM <= 0 for the
FORGE model. After each solve the table is compared with ``layered_first_arrival`` at every
elevation node and every ``accuracyCheckStrideR``-th distance node; the largest error in each model
layer is kept in the sidecar and in ``StationTables.to_record()``. The check never changes a table.

Tables are cached as ``.npy`` under ``<cache_dir>/ttgrids/<key>.npy`` with a ``<key>.json`` sidecar
holding the key's inputs and the accuracy record. The key is a SHA-256 of the velocity record (with
the top extension), the phase, the receiver elevation, the grid, the seed radius, the stencil
order, the scikit-fmm version, ``ALGORITHM`` and the SHA-256 of this module's source, so any edit to
the solver code gives new keys instead of stale tables. Receivers at the same ``sensorElevM`` share
a table.
"""

import hashlib
import io
import json
import logging
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import skfmm
from numpy.typing import ArrayLike, NDArray

from hq.config.seismology import GridsConfig
from hq.locate.velocity import LayerModel

log = logging.getLogger(__name__)

Phase = Literal["P", "S"]
PHASES: tuple[Phase, ...] = ("P", "S")
# Human-readable solver label, part of the cache key. The key also hashes this module's source
# (SOURCE_SHA256), so a solver edit can't serve stale tables even if nobody bumps this label.
ALGORITHM = "tt_grid/2: skfmm, cell-mean slowness, exact 1D layered seed"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CACHE_SUBDIR = "ttgrids"  # docs/01: travel-time grids live in data/cache/ttgrids/
# Bisection halvings for the ray parameter: 64 take the bracket below float64 resolution.
_BISECTION_STEPS = 64

FloatArray = NDArray[np.float64]


def _velocities(model: LayerModel, phase: Phase) -> FloatArray:
    if phase == "P":
        return model.vp_m_per_s
    if phase == "S":
        return model.vs_m_per_s
    raise ValueError(f"phase must be 'P' or 'S', got {phase!r}")


def _layer_bottoms(model: LayerModel) -> FloatArray:
    return np.append(model.top_elev_m[1:], -np.inf)


def _thickness(low: FloatArray, high: FloatArray, model: LayerModel) -> FloatArray:
    """Thickness (m) of each layer between elevations ``low <= high``; shape (N, nLayers)."""
    tops = model.top_elev_m[None, :]
    bottoms = _layer_bottoms(model)[None, :]
    overlap = np.minimum(high[:, None], tops) - np.maximum(low[:, None], bottoms)
    return np.clip(overlap, 0.0, None)


def _head_wave(
    path: FloatArray, v: FloatArray, v_refractor: float, r: FloatArray, allowed: NDArray[np.bool_]
) -> FloatArray:
    """Head-wave time along a refractor of speed ``v_refractor`` (inf where it does not exist).

    ``path`` holds the layer thicknesses crossed from both endpoints to the refractor face. The
    head wave exists when every crossed layer is slower than the refractor and r reaches the
    critical distance.
    """
    crossed = path > 0
    slower = np.all(~crossed | (v[None, :] < v_refractor), axis=1)
    ok = allowed & slower
    ratio = np.where(crossed & ok[:, None], v[None, :] / v_refractor, 0.0)
    cos = np.sqrt(1.0 - ratio**2)  # ratio < 1 wherever it is used
    x_crit = np.sum(path * ratio / cos, axis=1)
    delay = np.sum(path * cos / v[None, :], axis=1)
    time_s = r / v_refractor + delay
    return np.where(ok & (r >= x_crit), time_s, np.inf)


def layered_first_arrival(
    model: LayerModel, phase: Phase, source_elev_m: float, r_m: ArrayLike, elev_m: ArrayLike
) -> FloatArray:
    """Exact first-arrival time (s) in a 1D model of constant-velocity layers.

    The source sits at ``source_elev_m`` on the axis; each point is at epicentral distance
    ``r_m`` and elevation ``elev_m`` (m ASL). The result is the minimum over the transmitted ray
    (the direct ray within one layer), solved by bisection on the ray parameter p so that the
    horizontal range X(p) equals r, and every head wave along the top face of a layer below both
    endpoints or the bottom face of a layer above both. Travel time is written as
    ``tau(p) + p * r``, which is stationary in p at the ray, so bisection error is second order.
    Raises for points above the model top or non-finite input.
    """
    v = _velocities(model, phase)
    r_b, z_b = np.broadcast_arrays(np.asarray(r_m, np.float64), np.asarray(elev_m, np.float64))
    shape = r_b.shape
    r = r_b.ravel()
    z = z_b.ravel()
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(z)) and math.isfinite(source_elev_m)):
        raise ValueError("layered_first_arrival needs finite distances and elevations")
    if np.any(r < 0):
        raise ValueError("epicentral distances must be >= 0")
    model.layer_index(np.append(z, source_elev_m))  # raises above the model top
    zs = np.full_like(z, float(source_elev_m))
    low = np.minimum(zs, z)
    high = np.maximum(zs, z)

    # Transmitted (or direct) ray.
    path = _thickness(low, high, model)
    crossed = path > 0
    v_fast = np.max(np.where(crossed, v[None, :], 0.0), axis=1)
    level = v_fast == 0.0  # both endpoints at one elevation: a horizontal ray in that layer
    v_level = v[model.layer_index(z)]
    p_hi = np.where(level, 0.0, 1.0 / np.where(level, 1.0, v_fast))
    p_lo = np.zeros_like(r)
    slowness2 = 1.0 / v**2
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(_BISECTION_STEPS):
            p = 0.5 * (p_lo + p_hi)
            q = np.sqrt(np.clip(slowness2[None, :] - p[:, None] ** 2, 0.0, None))
            x = np.sum(np.where(crossed, path * p[:, None] / q, 0.0), axis=1)
            short = x < r
            p_lo = np.where(short, p, p_lo)
            p_hi = np.where(short, p_hi, p)
    p = p_lo
    tau = np.sum(path * np.sqrt(np.clip(slowness2[None, :] - p[:, None] ** 2, 0.0, None)), axis=1)
    best = np.where(level, r / v_level, tau + p * r)

    # Head waves.
    tops = model.top_elev_m
    bottoms = _layer_bottoms(model)
    for layer in range(model.n_layers):
        face = np.full_like(z, tops[layer])
        below_both = low >= tops[layer]
        down = _thickness(face, zs, model) + _thickness(face, z, model)
        best = np.minimum(best, _head_wave(down, v, float(v[layer]), r, below_both))
        if layer + 1 < model.n_layers:
            face = np.full_like(z, bottoms[layer])
            above_both = high <= bottoms[layer]
            up = _thickness(zs, face, model) + _thickness(z, face, model)
            best = np.minimum(best, _head_wave(up, v, float(v[layer]), r, above_both))
    return best.reshape(shape)


@dataclass(frozen=True)
class GridSpec:
    """Node lattice of a table: ``r = i * dr_m`` and ``elevM = bottom_elev_m + k * dz_m``."""

    dr_m: float
    dz_m: float
    n_r: int
    n_z: int
    bottom_elev_m: float

    def __post_init__(self) -> None:
        if self.n_r < 2 or self.n_z < 2:
            raise ValueError("a table grid needs at least 2 nodes along each axis")

    @property
    def top_elev_m(self) -> float:
        return self.bottom_elev_m + (self.n_z - 1) * self.dz_m

    @property
    def r_max_m(self) -> float:
        return (self.n_r - 1) * self.dr_m

    def r_nodes(self) -> FloatArray:
        return np.arange(self.n_r, dtype=np.float64) * self.dr_m

    def z_nodes(self) -> FloatArray:
        return self.bottom_elev_m + np.arange(self.n_z, dtype=np.float64) * self.dz_m

    def to_record(self) -> dict[str, Any]:
        return {
            "drM": self.dr_m,
            "dzM": self.dz_m,
            "nR": self.n_r,
            "nZ": self.n_z,
            "rMaxM": self.r_max_m,
            "bottomElevM": self.bottom_elev_m,
            "topElevM": self.top_elev_m,
        }


def make_grid(cfg: GridsConfig, cover_elev_m: float) -> GridSpec:
    """The table lattice whose top reaches ``cover_elev_m + cfg.topMarginM`` (snapped up)."""
    wanted_top = float(cover_elev_m) + cfg.topMarginM
    if not wanted_top > cfg.bottomElevM:
        raise ValueError(f"grid top {wanted_top} must lie above bottomElevM {cfg.bottomElevM}")
    n_cells = math.ceil((wanted_top - cfg.bottomElevM) / cfg.dzM)
    return GridSpec(
        dr_m=cfg.drM,
        dz_m=cfg.dzM,
        n_r=round(cfg.rMaxM / cfg.drM) + 1,
        n_z=n_cells + 1,
        bottom_elev_m=cfg.bottomElevM,
    )


def cell_slowness(model: LayerModel, phase: Phase, grid: GridSpec) -> FloatArray:
    """Mean slowness (s/m) of each node's cell ``[z - dz/2, z + dz/2]``, clipped to the grid."""
    z = grid.z_nodes()
    low = np.maximum(z - 0.5 * grid.dz_m, grid.bottom_elev_m)
    high = np.minimum(z + 0.5 * grid.dz_m, grid.top_elev_m)
    parts = _thickness(low, high, model)
    return np.sum(parts / _velocities(model, phase)[None, :], axis=1) / (high - low)


def solve_table(
    model: LayerModel,
    phase: Phase,
    receiver_elev_m: float,
    grid: GridSpec,
    *,
    seed_radius_m: float,
    fmm_order: int,
) -> FloatArray:
    """First-arrival times (s), shape ``(n_z, n_r)``, from a source at ``(0, receiver_elev_m)``.

    ``model`` must already reach the grid top (see ``build_station_tables``).
    """
    if model.top_of_model_elev_m < grid.top_elev_m:
        raise ValueError(
            f"velocity model top {model.top_of_model_elev_m} m ASL is below the grid top "
            f"{grid.top_elev_m} m ASL; extend it with with_top_extended_to first"
        )
    zs = float(receiver_elev_m)
    if not grid.bottom_elev_m < zs <= grid.top_elev_m:
        raise ValueError(f"receiver elevation {zs} lies outside the grid elevations")
    cell = max(grid.dr_m, grid.dz_m)
    if seed_radius_m < 4.0 * cell:
        raise ValueError("seed_radius_m must be at least 4 grid cells")
    r = grid.r_nodes()
    z = grid.z_nodes()
    speed_z = 1.0 / cell_slowness(model, phase, grid)
    speed = np.repeat(speed_z[:, None], grid.n_r, axis=1)

    # Seeded region: nodes within seed_radius_m of the source get exact layered times.
    iz = np.flatnonzero(np.abs(z - zs) <= seed_radius_m)
    ir = np.flatnonzero(r <= seed_radius_m)
    rr, zz = np.meshgrid(r[ir], z[iz])
    inside = np.hypot(rr, zz - zs) <= seed_radius_m
    seed = np.full(rr.shape, np.nan)
    seed[inside] = layered_first_arrival(model, phase, zs, rr[inside], zz[inside])

    # Start the march from the isochron t1: every point with T <= t1 is within
    # seed_radius_m - 2 cells of the source, since no layer is faster than v_max.
    v_max = float(np.max(_velocities(model, phase)))
    t1 = (seed_radius_m - 2.0 * cell) / v_max
    phi = np.full((grid.n_z, grid.n_r), seed_radius_m)
    block = (slice(iz[0], iz[-1] + 1), slice(ir[0], ir[-1] + 1))
    phi_block = phi[block]
    phi_block[inside] = (seed[inside] - t1) * speed[block][inside]
    if not np.any(phi_block < 0):
        raise ValueError("seed isochron is empty; seed_radius_m is too small for this grid")
    times = skfmm.travel_time(phi, speed, dx=[grid.dz_m, grid.dr_m], order=fmm_order)
    if np.ma.is_masked(times):
        raise RuntimeError("scikit-fmm left unreached nodes in the travel-time table")
    out = np.asarray(times, dtype=np.float64) + t1
    out_block = out[block]
    out_block[inside] = seed[inside]
    if not np.all(np.isfinite(out)) or np.any(out < 0):
        raise RuntimeError("travel-time table has non-finite or negative values")
    return np.ascontiguousarray(out)


@dataclass(frozen=True, eq=False)
class TravelTimeTable:
    """``times_s[k, i]`` is the first-arrival time from the receiver to ``(r_i, elevM_k)``."""

    phase: Phase
    receiver_elev_m: float
    grid: GridSpec
    times_s: FloatArray  # read-only, shape (n_z, n_r)
    key: str
    accuracy: Mapping[str, Any] | None = None  # table vs exact layered times (see table_accuracy)

    def __post_init__(self) -> None:
        if self.times_s.shape != (self.grid.n_z, self.grid.n_r):
            raise ValueError(f"table shape {self.times_s.shape} does not match its grid")
        if self.times_s.flags.writeable:
            raise ValueError("table arrays must be read-only")

    def lookup(self, r_m: ArrayLike, elev_m: ArrayLike) -> FloatArray:
        """Bilinear T(r, elevM) for broadcastable arrays; raises for any point off the grid."""
        g = self.grid
        fr = np.asarray(r_m, dtype=np.float64) / g.dr_m
        fz = (np.asarray(elev_m, dtype=np.float64) - g.bottom_elev_m) / g.dz_m
        for name, f, n in (("r", fr, g.n_r), ("elevM", fz, g.n_z)):
            bad = ~np.isfinite(f) | (f < 0) | (f > n - 1)
            if np.any(bad):
                raise ValueError(
                    f"{int(np.count_nonzero(bad))} lookup point(s) have {name} outside the "
                    f"{self.phase} table for receiver {self.receiver_elev_m} m ASL (r 0 to "
                    f"{g.r_max_m} m, elevM {g.bottom_elev_m} to {g.top_elev_m} m)"
                )
        ir = np.minimum(fr.astype(np.intp), g.n_r - 2)
        iz = np.minimum(fz.astype(np.intp), g.n_z - 2)
        ar = fr - ir
        az = fz - iz
        t = self.times_s
        low = (1.0 - ar) * t[iz, ir] + ar * t[iz, ir + 1]
        high = (1.0 - ar) * t[iz + 1, ir] + ar * t[iz + 1, ir + 1]
        return np.asarray((1.0 - az) * low + az * high, dtype=np.float64)


def table_key(
    model: LayerModel,
    phase: Phase,
    receiver_elev_m: float,
    grid: GridSpec,
    *,
    seed_radius_m: float,
    fmm_order: int,
) -> tuple[str, dict[str, Any]]:
    """The cache key (SHA-256 hex) of a table and the JSON payload it hashes."""
    payload = {
        "algorithm": ALGORITHM,
        "solverSourceSha256": SOURCE_SHA256,
        "scikitFmm": skfmm.__version__,
        "velocityModel": model.to_record(),
        "phase": phase,
        "receiverElevM": float(receiver_elev_m),
        "grid": grid.to_record(),
        "seedRadiusM": float(seed_radius_m),
        "fmmOrder": int(fmm_order),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), payload


def table_accuracy(
    model: LayerModel,
    phase: Phase,
    receiver_elev_m: float,
    grid: GridSpec,
    times_s: FloatArray,
    *,
    stride_r: int,
) -> dict[str, Any]:
    """Largest ``|table - exact layered time|`` (s), overall and per model layer.

    Checked at every elevation node and every ``stride_r``-th distance node. A node on an
    interface belongs to the layer below it (``LayerModel.layer_index``).
    """
    r = grid.r_nodes()[::stride_r]
    z = grid.z_nodes()
    rr, zz = np.meshgrid(r, z)
    err = np.abs(times_s[:, ::stride_r] - layered_first_arrival(model, phase, receiver_elev_m, rr, zz))
    k = np.unravel_index(int(np.argmax(err)), err.shape)
    row_max = err.max(axis=1)
    layer = model.layer_index(z)
    by_layer = [
        {
            "topElevM": float(model.top_elev_m[i]),
            "maxErrS": float(row_max[layer == i].max()),
        }
        for i in range(model.n_layers)
        if np.any(layer == i)
    ]
    return {
        "strideR": int(stride_r),
        "maxErrS": float(err[k]),
        "atElevM": float(zz[k]),
        "atRM": float(rr[k]),
        "byLayer": by_layer,
    }


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file, so readers never see a partial file."""
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


def load_or_build_table(
    model: LayerModel,
    phase: Phase,
    receiver_elev_m: float,
    grid: GridSpec,
    cfg: GridsConfig,
    *,
    cache_dir: Path,
) -> tuple[TravelTimeTable, bool]:
    """The table from ``<cache_dir>/ttgrids/`` if present, else solved and cached.

    Returns the table and whether it was built (False: loaded). A cached file with the wrong
    shape or non-finite values, or a sidecar that is missing or does not match the key's inputs,
    raises: delete both files and rerun.
    """
    key, payload = table_key(
        model, phase, receiver_elev_m, grid, seed_radius_m=cfg.seedRadiusM, fmm_order=cfg.fmmOrder
    )
    folder = Path(cache_dir) / CACHE_SUBDIR
    path = folder / f"{key}.npy"
    sidecar_path = folder / f"{key}.json"
    if path.exists():
        # A plain read-only ndarray view of the memory map (fancy indexing on np.memmap is slow);
        # spawned workers share the pages.
        times = np.load(path, mmap_mode="r", allow_pickle=False).view(np.ndarray)
        if times.shape != (grid.n_z, grid.n_r) or times.dtype != np.float64:
            raise ValueError(f"cached table {path} has shape {times.shape} {times.dtype}; delete it")
        if not np.all(np.isfinite(times)):
            raise ValueError(f"cached table {path} has non-finite values; delete it")
        if not sidecar_path.exists():
            raise ValueError(f"cached table {path} has no sidecar {sidecar_path.name}; delete it")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        accuracy = sidecar.pop("accuracyVsExact", None)
        if sidecar != payload or accuracy is None:
            raise ValueError(f"sidecar {sidecar_path} does not match the table key; delete both")
        return TravelTimeTable(phase, float(receiver_elev_m), grid, times, key, accuracy), False
    started = time.perf_counter()
    times = solve_table(
        model,
        phase,
        receiver_elev_m,
        grid,
        seed_radius_m=cfg.seedRadiusM,
        fmm_order=cfg.fmmOrder,
    )
    accuracy = table_accuracy(
        model, phase, receiver_elev_m, grid, times, stride_r=cfg.accuracyCheckStrideR
    )
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        {**payload, "accuracyVsExact": accuracy}, indent=2, sort_keys=True, allow_nan=False
    )
    _write_atomic(sidecar_path, text.encode("utf-8"))
    _write_atomic(path, _npy_bytes(times))
    times.setflags(write=False)
    log.info(
        "built %s table for receiver %.2f m ASL: %d x %d nodes in %.2f s (max %.1f ms from the "
        "exact layered times, at elevM %.0f m) -> %s",
        phase, receiver_elev_m, grid.n_z, grid.n_r, time.perf_counter() - started,
        accuracy["maxErrS"] * 1e3, accuracy["atElevM"], path.name,
    )
    return TravelTimeTable(phase, float(receiver_elev_m), grid, times, key, accuracy), True


@dataclass(frozen=True, eq=False)
class StationTables:
    """Tables for a station set: one per (phase, receiver elevation), shared across stations."""

    model: LayerModel  # the velocity model with its top extended to the grid top
    grid: GridSpec
    receiver_elev_m: Mapping[str, float]  # station id -> sensorElevM
    tables: Mapping[tuple[Phase, float], TravelTimeTable]
    seed_radius_m: float
    fmm_order: int
    n_built: int
    n_loaded: int
    build_s: float

    def table(self, station_id: str, phase: Phase) -> TravelTimeTable:
        try:
            elev = self.receiver_elev_m[station_id]
        except KeyError as err:
            raise KeyError(f"no travel-time table for station {station_id!r}") from err
        return self.tables[(phase, elev)]

    def accuracy_by_layer(self) -> dict[str, list[dict[str, float]]]:
        """Per phase and model layer, the largest table-vs-exact error (s) over all tables."""
        out: dict[str, list[dict[str, float]]] = {}
        for phase in PHASES:
            worst: dict[float, float] = {}
            for (ph, _), table in self.tables.items():
                if ph != phase:
                    continue
                if table.accuracy is None:
                    raise ValueError(f"table {table.key} carries no accuracy record")
                for row in table.accuracy["byLayer"]:
                    top = float(row["topElevM"])
                    worst[top] = max(worst.get(top, 0.0), float(row["maxErrS"]))
            out[phase] = [
                {"topElevM": top, "maxErrS": err} for top, err in sorted(worst.items(), reverse=True)
            ]
        return out

    def to_record(self) -> dict[str, Any]:
        """Grid, seed, velocity model (with extension), table keys and accuracy, for the record."""
        by_table: dict[tuple[Phase, float], list[str]] = {}
        for sid, elev in sorted(self.receiver_elev_m.items()):
            for phase in PHASES:
                by_table.setdefault((phase, elev), []).append(sid)
        return {
            "algorithm": ALGORITHM,
            "solverSourceSha256": SOURCE_SHA256,
            "scikitFmm": skfmm.__version__,
            "grid": self.grid.to_record(),
            "seedRadiusM": self.seed_radius_m,
            "fmmOrder": self.fmm_order,
            "velocityModel": self.model.to_record(),
            "accuracyVsExactByLayer": self.accuracy_by_layer(),
            "tables": [
                {
                    "phase": phase,
                    "receiverElevM": elev,
                    "key": self.tables[(phase, elev)].key,
                    "stationIds": ids,
                    "maxErrVsExactS": _max_err(self.tables[(phase, elev)]),
                }
                for (phase, elev), ids in sorted(by_table.items())
            ],
            "nBuilt": self.n_built,
            "nLoaded": self.n_loaded,
            "buildS": self.build_s,
        }


def _max_err(table: TravelTimeTable) -> float:
    if table.accuracy is None:
        raise ValueError(f"table {table.key} carries no accuracy record")
    return float(table.accuracy["maxErrS"])


def build_station_tables(
    stations: pd.DataFrame,
    model: LayerModel,
    cfg: GridsConfig,
    *,
    cache_dir: Path,
    max_extension_m: float,
    cover_top_elev_m: float | None = None,
) -> StationTables:
    """P and S tables for every distinct ``sensorElevM`` in ``stations`` (columns id, sensorElevM).

    The grid top covers the highest receiver and ``cover_top_elev_m`` (the search-volume top)
    plus ``cfg.topMarginM``; the model's top layer is extended to it, capped by
    ``max_extension_m`` (config ``velocity.maxTopExtensionM``).
    """
    started = time.perf_counter()
    for column in ("id", "sensorElevM"):
        if column not in stations.columns:
            raise ValueError(f"stations need column {column!r}")
    if stations.empty:
        raise ValueError("no stations")
    ids = stations["id"].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"duplicate station ids: {sorted(ids[ids.duplicated()].unique())}")
    elevs = stations["sensorElevM"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(elevs)):
        raise ValueError("stations have non-finite sensorElevM")
    highest = float(np.max(elevs))
    cover = highest if cover_top_elev_m is None else max(highest, float(cover_top_elev_m))
    grid = make_grid(cfg, cover)
    if float(np.min(elevs)) <= grid.bottom_elev_m:
        raise ValueError("a receiver lies at or below grids.bottomElevM")
    extended = model.with_top_extended_to(grid.top_elev_m, max_extension_m=max_extension_m)
    receivers = dict(zip(ids, (float(e) for e in elevs), strict=True))
    unique = sorted(set(receivers.values()))
    tables: dict[tuple[Phase, float], TravelTimeTable] = {}
    n_built = 0
    for elev in unique:
        for phase in PHASES:
            table, built = load_or_build_table(
                extended, phase, elev, grid, cfg, cache_dir=Path(cache_dir)
            )
            tables[(phase, elev)] = table
            n_built += int(built)
    elapsed = time.perf_counter() - started
    log.info(
        "travel-time tables: %d stations, %d receiver elevations, %d tables (%d built, %d from "
        "cache) on a %d x %d grid, top %.1f m ASL, in %.2f s",
        len(receivers), len(unique), len(tables), n_built, len(tables) - n_built,
        grid.n_z, grid.n_r, grid.top_elev_m, elapsed,
    )
    return StationTables(
        model=extended,
        grid=grid,
        receiver_elev_m=receivers,
        tables=tables,
        seed_radius_m=cfg.seedRadiusM,
        fmm_order=cfg.fmmOrder,
        n_built=n_built,
        n_loaded=len(tables) - n_built,
        build_s=elapsed,
    )

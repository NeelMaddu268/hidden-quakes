"""PyOcto station-specific travel-time tables from the 1D layer model (LOC-03).

PyOcto 0.2.0's ``StationSpecificVelocityModel1D`` reads a directory holding, per station,
``<stationId>.pyocto``: int32 ``nx``, int32 ``nz``, float64 ``delta`` (km), then ``nx * nz`` float64
P times and ``nx * nz`` float64 S times (s), C order ``[ix, iz]``; plus ``n_padding`` (int32).
Node ``(ix, iz)`` lies ``ix * delta`` km from the station horizontally and ``iz * delta`` km below
the table top, which is ``n_padding * delta`` km above the sensor (the source node is
``(0, n_padding)``). PyOcto evaluates an event at ``z`` (km below sea level) at table depth
``z - station.z + n_padding * delta``, interpolates bilinearly, returns NaN from the last node row
or column on, and reads out of bounds for a negative table depth (``VelocityModel.cpp``). The
tables below are sized so neither happens anywhere in the search volume.

PyOcto's own builder (``create_model``) needs pyrocko, which is not a dependency, and it
interpolates velocities linearly between rows instead of honouring layer steps. Here the same
format is written with scikit-fmm on the layer model itself: by reciprocity the source is the
sensor at its true ``sensorElevM``, and each node row takes the model's slowness averaged over the
row's cell ``[elev - spacing / 2, elev + spacing / 2]`` (the exact integral of the layer steps), so
an interface between two node rows sits at its true elevation to first order instead of snapping
to a row. Against a 10 m reference this keeps 50 m tables of the FORGE model within about 20 ms
(100 m tables without averaging: up to about 90 ms in S). Inside ``seedRadiusM`` of the sensor
the times are straight rays at the velocity of the sensor's layer (``LayerModel.vp_at``/``vs_at``,
a sensor on a boundary taking the layer below), and the eikonal solve starts from that circle. Where a layer boundary crosses the circle this is
approximate; the bound ``(seedRadiusM - |boundary - sensor|) * |1/v_above - 1/v_below|`` is computed
per station, recorded, and must stay below ``maxSeedErrorS``.

Model top: the layer model is extended (``with_top_extended_to``, recorded) to the highest sensor
or the volume top, whichever is higher. PyOcto's ``n_padding`` is shared by every station, so the
tables of high stations have rows above that; they repeat the top layer. No queried time depends
on them: a path leaving the extended model through its top returns into the same top layer, and
the straight segment between those two points, inside the top layer, is shorter at equal velocity.

Tables are cached under ``<cache_dir>/ttgrids/pyocto/<key>/`` with a ``manifest.json``; the key
hashes everything that determines their bytes.
"""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import skfmm

from hq.associate.frame import M_PER_KM, SearchVolume
from hq.config.seismology import AssociatorTablesConfig
from hq.locate.velocity import LayerModel

log = logging.getLogger(__name__)

FORMAT = "pyocto-0.2.0-station-specific-1d"  # PyOcto's reader this layout is written for
BUILDER_VERSION = 1  # bump when the builder changes what it writes
CACHE_SUBDIR = Path("ttgrids") / "pyocto"
HEADER = struct.Struct("=iid")  # nx, nz, delta_km, native order as PyOcto's fread expects
PADDING = struct.Struct("=i")
PADDING_FILE = "n_padding"
MANIFEST_FILE = "manifest.json"
TABLE_SUFFIX = ".pyocto"
SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")
KEY_CHARS = 16  # hex characters of the manifest hash in the directory name

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TableSpec:
    """Shape of every station's table; all stations share one ``nx``, ``nz`` and ``n_padding``."""

    spacing_m: float
    nx: int
    nz: int
    n_padding: int

    @property
    def delta_km(self) -> float:
        return self.spacing_m / M_PER_KM

    @property
    def r_max_m(self) -> float:
        return (self.nx - 1) * self.spacing_m

    def to_record(self) -> dict[str, Any]:
        return {
            "spacingM": self.spacing_m,
            "deltaKm": self.delta_km,
            "nx": self.nx,
            "nz": self.nz,
            "nPadding": self.n_padding,
            "rMaxM": self.r_max_m,
        }


@dataclass(frozen=True)
class TableSet:
    """Tables on disk for one set of stations, ready for ``StationSpecificVelocityModel1D``."""

    directory: Path
    key: str
    spec: TableSpec
    built: bool  # False: reused from the cache
    model_top_elev_m: float  # the layer model was extended up to here
    seed_error_bound_s: dict[str, float]  # per station, the larger of P and S
    max_s_time_in_volume_s: float  # largest S time from any station to any point of the volume
    build_runtime_s: float

    def to_record(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "builderVersion": BUILDER_VERSION,
            "directory": str(self.directory),
            "key": self.key,
            "reusedFromCache": not self.built,
            **self.spec.to_record(),
            "modelTopElevM": self.model_top_elev_m,
            "rowsAboveModelTop": "repeat the top layer's velocities (PyOcto's n_padding is shared "
            "by all stations); no queried travel time passes through them",
            "seedErrorBoundS": self.seed_error_bound_s,
            "maxSTimeInVolumeS": self.max_s_time_in_volume_s,
            "skfmmVersion": skfmm.__version__,
        }


def plan_tables(
    sensor_elev_m: FloatArray,
    farthest_m: FloatArray,
    volume: SearchVolume,
    cfg: AssociatorTablesConfig,
) -> TableSpec:
    """Size the tables so every station reaches every point of the volume inside the table.

    ``farthest_m``: per station, the largest horizontal distance to the volume. PyOcto returns NaN
    from index ``nx - 1`` (``nz - 1``) on, so the last needed node must lie before that.
    """
    step = cfg.spacingM
    if sensor_elev_m.size == 0:
        raise ValueError("tables need at least one station")
    # Rows above the highest-needed point of the shallowest sensor: the volume top.
    n_padding = max(0, math.ceil((volume.top_elev_m - float(sensor_elev_m.min())) / step))
    below = math.ceil((float(sensor_elev_m.max()) - volume.bottom_elev_m) / step)
    nz = n_padding + below + 2
    nx = math.ceil(float(farthest_m.max()) / step) + 2
    return TableSpec(spacing_m=step, nx=nx, nz=nz, n_padding=n_padding)


def seed_error_bound_s(model: LayerModel, sensor_elev_m: float, seed_radius_m: float) -> float:
    """Upper bound (s) of the straight-ray seed's error for a sensor, over both phases."""
    bound = 0.0
    tops = model.top_elev_m
    for i in range(1, model.n_layers):
        gap = abs(float(tops[i]) - sensor_elev_m)
        if gap >= seed_radius_m:
            continue
        for vel in (model.vp_m_per_s, model.vs_m_per_s):
            slowness_step = abs(1.0 / float(vel[i - 1]) - 1.0 / float(vel[i]))
            bound = max(bound, (seed_radius_m - gap) * slowness_step)
    return bound


def cell_slowness(
    top_elev_m: FloatArray, vel_m_per_s: FloatArray, row_elev_m: FloatArray, spacing_m: float
) -> FloatArray:
    """Slowness (s/m) of layered velocities averaged over each row's cell ``elev +- spacing/2``.

    The top layer continues upward without limit (rows above the model top repeat it) and the
    half-space downward. The average is the exact integral of the piecewise-constant slowness.
    """
    upper = np.r_[np.inf, top_elev_m[1:]]  # layer i spans (lower[i], upper[i]]
    lower = np.r_[top_elev_m[1:], -np.inf]
    lo = row_elev_m[:, None] - spacing_m / 2.0
    hi = row_elev_m[:, None] + spacing_m / 2.0
    overlap = np.clip(np.minimum(hi, upper[None, :]) - np.maximum(lo, lower[None, :]), 0.0, None)
    return np.asarray((overlap / vel_m_per_s[None, :]).sum(axis=1) / spacing_m)


def station_times(
    model: LayerModel,
    sensor_elev_m: float,
    spec: TableSpec,
    cfg: AssociatorTablesConfig,
) -> tuple[FloatArray, FloatArray]:
    """P and S first-arrival times (s), shape ``(nx, nz)``, for a sensor at ``sensor_elev_m``.

    ``model`` must already reach up to every row that matters (see the module docstring); rows
    above its top repeat the top layer. Rows carry the cell-averaged slowness (``cell_slowness``).
    """
    step_km = spec.delta_km
    rows = np.arange(spec.nz, dtype=np.float64)
    row_elev_m = sensor_elev_m + (spec.n_padding - rows) * spec.spacing_m
    r_km = np.arange(spec.nx, dtype=np.float64)[:, None] * step_km
    dz_km = (rows[None, :] - spec.n_padding) * step_km
    dist_km = np.hypot(r_km, dz_km)
    seed_km = cfg.seedRadiusM / M_PER_KM
    inside = dist_km <= seed_km
    phi = dist_km - seed_km
    out: list[FloatArray] = []
    for layer_v, v_src_m_per_s in (
        (model.vp_m_per_s, model.vp_at(sensor_elev_m)),
        (model.vs_m_per_s, model.vs_at(sensor_elev_m)),
    ):
        slowness = cell_slowness(model.top_elev_m, layer_v, row_elev_m, spec.spacing_m)
        speed = np.ascontiguousarray(
            np.broadcast_to(1.0 / (slowness * M_PER_KM), (spec.nx, spec.nz))
        )
        v_src = float(v_src_m_per_s) / M_PER_KM
        fmm = np.asarray(skfmm.travel_time(phi, speed, dx=step_km, order=cfg.fmmOrder))
        times = np.where(inside, dist_km / v_src, fmm + seed_km / v_src)
        if not np.all(np.isfinite(times)):
            raise RuntimeError(f"non-finite travel times for a sensor at {sensor_elev_m} m ASL")
        out.append(np.ascontiguousarray(times, dtype=np.float64))
    return out[0], out[1]


def _write_table(path: Path, spec: TableSpec, p: FloatArray, s: FloatArray) -> None:
    with path.open("wb") as fh:
        fh.write(HEADER.pack(spec.nx, spec.nz, spec.delta_km))
        fh.write(p.tobytes(order="C"))
        fh.write(s.tobytes(order="C"))


def read_table(path: Path) -> tuple[TableSpec, FloatArray, FloatArray]:
    """Read one ``.pyocto`` table back (spacing from the header; ``n_padding`` is not in it)."""
    raw = path.read_bytes()
    nx, nz, delta_km = HEADER.unpack_from(raw)
    expected = HEADER.size + 2 * nx * nz * 8
    if len(raw) != expected:
        raise ValueError(f"{path}: {len(raw)} bytes, expected {expected} for nx={nx}, nz={nz}")
    times = np.frombuffer(raw, dtype=np.float64, offset=HEADER.size).reshape(2, nx, nz)
    spec = TableSpec(spacing_m=delta_km * M_PER_KM, nx=nx, nz=nz, n_padding=-1)
    return spec, times[0], times[1]


def _manifest(
    model: LayerModel, station_ids: list[str], sensor_elev_m: FloatArray, spec: TableSpec,
    cfg: AssociatorTablesConfig,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "builderVersion": BUILDER_VERSION,
        "skfmmVersion": skfmm.__version__,
        "spec": spec.to_record(),
        "seedRadiusM": cfg.seedRadiusM,
        "fmmOrder": cfg.fmmOrder,
        "velocityModel": model.to_record(),
        # repr keeps every bit of the elevation, so a changed sensor elevation changes the key
        "stations": [
            {"id": sid, "sensorElevM": repr(float(elev))}
            for sid, elev in zip(station_ids, sensor_elev_m, strict=True)
        ],
    }


def _cache_is_complete(directory: Path, manifest: dict[str, Any], station_ids: list[str],
                       spec: TableSpec) -> bool:
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        return False
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"{manifest_path} does not match its key; remove the directory")
    size = HEADER.size + 2 * spec.nx * spec.nz * 8
    for sid in station_ids:
        table = directory / f"{sid}{TABLE_SUFFIX}"
        if not table.is_file() or table.stat().st_size != size:
            raise RuntimeError(f"cached table {table} is missing or truncated; remove {directory}")
    padding = directory / PADDING_FILE
    if PADDING.unpack(padding.read_bytes()) != (spec.n_padding,):
        raise RuntimeError(f"{padding} does not hold n_padding {spec.n_padding}")
    return True


def _max_s_time_in_volume(
    directory: Path, station_ids: list[str], sensor_elev_m: FloatArray, farthest_m: FloatArray,
    spec: TableSpec, volume: SearchVolume,
) -> float:
    """Largest S time from any station to any volume point, read from the tables on disk."""
    largest = 0.0
    for sid, elev, far in zip(station_ids, sensor_elev_m, farthest_m, strict=True):
        _, _, s_times = read_table(directory / f"{sid}{TABLE_SUFFIX}")
        ix = min(spec.nx, math.ceil(float(far) / spec.spacing_m) + 1)
        iz0 = max(0, math.floor((float(elev) - volume.top_elev_m) / spec.spacing_m) + spec.n_padding)
        iz1 = min(
            spec.nz,
            math.ceil((float(elev) - volume.bottom_elev_m) / spec.spacing_m) + spec.n_padding + 1,
        )
        largest = max(largest, float(s_times[:ix, iz0:iz1].max()))
    return largest


def build_tables(
    model: LayerModel,
    station_ids: list[str],
    sensor_elev_m: FloatArray,
    farthest_m: FloatArray,
    volume: SearchVolume,
    cfg: AssociatorTablesConfig,
    root: Path,
) -> TableSet:
    """Build (or reuse from ``root``) the tables for these stations; returns where they are.

    ``model`` must reach up to ``max(sensor_elev_m.max(), volume.top_elev_m)``; the caller extends
    it (recorded) with ``with_top_extended_to``. Raises on any station id that is not a safe file
    name, on a seed error bound above ``cfg.maxSeedErrorS``, and on a corrupt cache entry.
    """
    started = time.perf_counter()
    need_top = max(float(sensor_elev_m.max()), volume.top_elev_m)
    if model.top_of_model_elev_m < need_top:
        raise ValueError(
            f"velocity model top {model.top_of_model_elev_m} m ASL is below {need_top} m ASL; "
            "extend it with with_top_extended_to first"
        )
    bad = [sid for sid in station_ids if not SAFE_ID.fullmatch(sid) or sid == PADDING_FILE]
    if bad:
        raise ValueError(f"station ids {bad} cannot be PyOcto table file names")
    if len(set(station_ids)) != len(station_ids):
        raise ValueError("station ids must be unique")
    bounds = {
        sid: seed_error_bound_s(model, float(elev), cfg.seedRadiusM)
        for sid, elev in zip(station_ids, sensor_elev_m, strict=True)
    }
    worst = max(bounds, key=lambda sid: bounds[sid])
    if bounds[worst] > cfg.maxSeedErrorS:
        raise ValueError(
            f"station {worst}: a layer boundary inside the {cfg.seedRadiusM} m seed disc bounds "
            f"the seed error at {bounds[worst]:.3f} s > maxSeedErrorS {cfg.maxSeedErrorS}; "
            "reduce tables.seedRadiusM"
        )

    spec = plan_tables(sensor_elev_m, farthest_m, volume, cfg)
    manifest = _manifest(model, station_ids, sensor_elev_m, spec, cfg)
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    directory = root / CACHE_SUBDIR / key[:KEY_CHARS]
    built = False
    if not _cache_is_complete(directory, manifest, station_ids, spec):
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{key[:KEY_CHARS]}-", dir=directory.parent))
        try:
            for sid, elev in zip(station_ids, sensor_elev_m, strict=True):
                p, s = station_times(model, float(elev), spec, cfg)
                _write_table(staging / f"{sid}{TABLE_SUFFIX}", spec, p, s)
            (staging / PADDING_FILE).write_bytes(PADDING.pack(spec.n_padding))
            (staging / MANIFEST_FILE).write_text(
                json.dumps(manifest, sort_keys=True, indent=1), encoding="utf-8"
            )
            if directory.exists():  # a partial directory from an interrupted build
                shutil.rmtree(directory)
            os.replace(staging, directory)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        built = True
    max_s = _max_s_time_in_volume(directory, station_ids, sensor_elev_m, farthest_m, spec, volume)
    runtime = time.perf_counter() - started
    log.info(
        "associate: %s %d PyOcto tables (nx %d, nz %d, n_padding %d, %.0f m) in %s (%.2f s); "
        "largest S time in the volume %.2f s; worst seed error bound %.4f s (%s)",
        "built" if built else "reused", len(station_ids), spec.nx, spec.nz, spec.n_padding,
        spec.spacing_m, directory, runtime, max_s, bounds[worst], worst,
    )
    return TableSet(
        directory=directory,
        key=key,
        spec=spec,
        built=built,
        model_top_elev_m=model.top_of_model_elev_m,
        seed_error_bound_s=bounds,
        max_s_time_in_volume_s=max_s,
        build_runtime_s=runtime,
    )

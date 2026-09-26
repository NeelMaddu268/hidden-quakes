"""Grid-search hypocentre locator on per-station 1D travel-time tables (L1, origin time removed).

Search volume (ENU metres around the run origin, elevM): ``e`` and ``n`` in
``[-halfWidthM, +halfWidthM]`` and elevM from ``volume.bottomElevM`` up to ``volume.topElevM``
(null: ``run.refSurfaceElevM``, the ground at the origin, because a 1D search has no DEM). Every
grid is anchored at the volume's lower corner; the top is snapped down onto the fine lattice, so
no hypocentre lies above the configured top. Where the ground in the volume lies below that top,
hypocentres can still land above the local ground; LOC-04 checks them against a DEM.

Misfit at a node, over the picks in use:
    d_i = t_obs_i - T_i(node) - static_i
    t0 = weighted median of d_i with weights w_i = prob_i / sigma_i  (the lower weighted median:
         the first sorted d_i whose cumulative weight reaches half the total; it minimises the sum)
    misfit = sum_i w_i * |d_i - t0|
``static_i`` is the per station-phase static (default 0; LOC-05 supplies them). ``sigma_i`` is
``pickSigmaS[phase]`` unless the station's ``preprocessProfile`` has an entry in
``profilePickSigmaS``.

Search, per event (``cutoff`` = ``pdfCutoff / pdfMisfitScale`` in misfit units, so nodes beyond it
hold less than exp(-pdfCutoff) of the peak node's PDF mass):
1. Coarse: every ``coarseSpacingM`` node of the volume (travel times cached across events).
2. Stage pass: the misfit every ``fineStageSpacingM`` over a box that starts at
   ``+/- fineHalfWidthM`` around the best coarse node (clipped to the volume). While the stage
   nodes within ``cutoff`` of the stage minimum reach a face of the box that is not a volume face,
   that face moves out by the box's width on that axis (at least ``fineHalfWidthM``).
3. Fine region: the fine (``fineSpacingM``) nodes over the bounding box of those stage nodes plus
   one stage step, clipped to the volume. It grows by a stage step on any face that is not a volume
   face and still holds a fine node within ``cutoff`` of the minimum. So the reported PDF lives on
   fine nodes and covers the whole region within ``cutoff`` of the minimum, however far that
   reaches, up to the search volume. (A separate basin missed by the stage pass and cut off by a
   ridge higher than ``cutoff`` is not covered.)
4. Node budget: if the stage region's bounding box, or a grown fine region, would exceed
   ``maxPdfNodes`` fine nodes, growth stops. After a stage-pass stop the fine region is
   ``+/- fineHalfWidthM`` around the best stage node. The PDF is then truncated wherever a face of
   the evaluated region that is not the volume's top or bottom still holds a node within
   ``cutoff``; lateral volume faces count too. A truncated PDF gives hErrM = vErrM = None and
   ``pdfTruncated`` in the search record (docs/02 allows None). The volume's top and bottom are
   the prior's bounds, not truncation: ``depthOnEdge`` reports them.
5. The hypocentre is the fine node with the least misfit (MAP); the PDF gives the formal errors
   and ``depthOnEdge`` (``hq.locate.uncertainty``).
6. Outlier pass: residuals at the MAP node; picks with ``|residual| > max(madK * MAD, floorS)``
   (MAD = median of |r - median(r)|, unscaled, over every phase) are dropped and the event is
   relocated once from step 1. If dropping would leave fewer than ``minPicks`` picks, nothing is
   dropped and the result says so.

Callers pass only the stations in use (``usedInRun``); every station passed must lie within
``grids.rMaxM`` of every search-volume corner, or the Locator raises.
"""

import logging
import math
import multiprocessing
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from hq.config.run import RunSection
from hq.config.seismology import LocatorConfig, SeismologyConfig
from hq.locate.tt_grid import PHASES, Phase, StationTables, build_station_tables
from hq.locate.uncertainty import PdfSummary, conventions, summarize_pdf
from hq.locate.velocity import LayerModel

log = logging.getLogger(__name__)

METHOD = "grid1d"  # LocationQuality.method
PICK_COLUMNS = ("id", "stationId", "phase", "t", "prob")
STATION_COLUMNS = ("id", "enu_e", "enu_n", "enu_u", "sensorElevM")

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Statics = Mapping[tuple[str, str], float]
# Face names of a (z, n, e) box per axis: (low index side, high index side).
AXIS_FACES = (("bottom", "top"), ("south", "north"), ("west", "east"))


def weighted_median(values: ArrayLike, weights: ArrayLike) -> float:
    """Lower weighted median: the first sorted value whose cumulative weight reaches half."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.ndim != 1 or v.shape != w.shape or v.size == 0:
        raise ValueError("values and weights must be equal-length, non-empty 1D arrays")
    if np.any(w <= 0) or not np.all(np.isfinite(w)) or not np.all(np.isfinite(v)):
        raise ValueError("weights must be positive and everything finite")
    _, t0 = l1_misfit(v[None, :], w)
    return float(t0[0])


def l1_misfit(d: FloatArray, w: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Per row of ``d`` (nodes x picks): the weighted-L1 misfit and its origin time.

    ``t0`` is the lower weighted median of the row and ``misfit = sum_i w_i |d_i - t0|``, the
    minimum over all origin times.
    """
    order = np.argsort(d, axis=1)
    ds = np.take_along_axis(d, order, axis=1)
    cw = np.cumsum(w[order], axis=1)
    k = np.argmax(cw >= 0.5 * float(w.sum()), axis=1)
    t0 = ds[np.arange(d.shape[0]), k]
    misfit = (np.abs(d - t0[:, None]) * w[None, :]).sum(axis=1)
    return misfit, t0


def azimuthal_gap_deg(de: ArrayLike, dn: ArrayLike) -> float:
    """Largest azimuthal gap (deg) of stations at offsets (de, dn) from the epicentre.

    A station exactly above the epicentre has no azimuth and is left out; fewer than two
    stations with an azimuth give 360.
    """
    de_a = np.asarray(de, dtype=np.float64)
    dn_a = np.asarray(dn, dtype=np.float64)
    keep = np.hypot(de_a, dn_a) > 0
    az = np.sort(np.degrees(np.arctan2(de_a[keep], dn_a[keep])) % 360.0)
    if az.size < 2:
        return 360.0
    gaps = np.append(np.diff(az), az[0] + 360.0 - az[-1])
    return float(gaps.max())


@dataclass(frozen=True)
class SearchVolume:
    """The fine lattice of the search volume; coarse nodes are every ``coarse_stride``-th node."""

    e_min_m: float
    n_min_m: float
    bottom_elev_m: float
    fine_m: float
    n_e: int
    n_n: int
    n_z: int
    coarse_stride: int
    configured_top_elev_m: float  # before snapping onto the fine lattice

    @property
    def top_elev_m(self) -> float:
        return self.bottom_elev_m + (self.n_z - 1) * self.fine_m

    @property
    def e_max_m(self) -> float:
        return self.e_min_m + (self.n_e - 1) * self.fine_m

    @property
    def n_max_m(self) -> float:
        return self.n_min_m + (self.n_n - 1) * self.fine_m

    def e_at(self, idx: int | IntArray) -> FloatArray:
        return self.e_min_m + np.asarray(idx, dtype=np.float64) * self.fine_m

    def n_at(self, idx: int | IntArray) -> FloatArray:
        return self.n_min_m + np.asarray(idx, dtype=np.float64) * self.fine_m

    def z_at(self, idx: int | IntArray) -> FloatArray:
        return self.bottom_elev_m + np.asarray(idx, dtype=np.float64) * self.fine_m

    @property
    def upper(self) -> IntArray:
        """Largest fine index along (z, n, e)."""
        return np.array([self.n_z - 1, self.n_n - 1, self.n_e - 1], dtype=np.intp)

    def coarse_axes(self) -> tuple[IntArray, IntArray, IntArray]:
        """Fine indices of the coarse nodes along e, n and z."""
        s = self.coarse_stride
        return np.arange(0, self.n_e, s), np.arange(0, self.n_n, s), np.arange(0, self.n_z, s)

    def to_record(self) -> dict[str, Any]:
        return {
            "eMinM": self.e_min_m,
            "eMaxM": self.e_max_m,
            "nMinM": self.n_min_m,
            "nMaxM": self.n_max_m,
            "bottomElevM": self.bottom_elev_m,
            "topElevM": self.top_elev_m,
            "configuredTopElevM": self.configured_top_elev_m,
            "fineSpacingM": self.fine_m,
            "coarseSpacingM": self.fine_m * self.coarse_stride,
        }


def make_volume(cfg: LocatorConfig, ref_surface_elev_m: float) -> SearchVolume:
    """The search volume from config; ``volume.topElevM`` null means ``ref_surface_elev_m``."""
    top = cfg.volume.topElevM if cfg.volume.topElevM is not None else float(ref_surface_elev_m)
    bottom = cfg.volume.bottomElevM
    if top - bottom < cfg.coarseSpacingM:
        raise ValueError(f"search volume {bottom}..{top} m ASL is thinner than one coarse cell")
    fine = cfg.fineSpacingM
    width = 2.0 * cfg.volume.halfWidthM
    return SearchVolume(
        e_min_m=-cfg.volume.halfWidthM,
        n_min_m=-cfg.volume.halfWidthM,
        bottom_elev_m=bottom,
        fine_m=fine,
        n_e=round(width / fine) + 1,
        n_n=round(width / fine) + 1,
        n_z=math.floor((top - bottom) / fine) + 1,
        coarse_stride=round(cfg.coarseSpacingM / fine),
        configured_top_elev_m=top,
    )


@dataclass(frozen=True)
class _Picks:
    ids: list[str]
    station_ids: list[str]
    phases: list[Phase]
    t_obs: FloatArray  # epoch s
    t_rel: FloatArray  # s after the earliest pick
    t_ref: float
    prob: FloatArray
    sigma: FloatArray
    w: FloatArray
    static: FloatArray
    se: FloatArray  # station east (m)
    sn: FloatArray  # station north (m)


@dataclass(frozen=True)
class _Search:
    iz: int  # MAP node, fine indices
    i_n: int
    ie: int
    t0_rel: float
    misfit: float
    pdf: PdfSummary
    pdf_truncated: bool
    box: dict[str, Any]


def _stage_axis(center: int, lo: int, hi: int, step: int) -> IntArray:
    """Stage-lattice fine indices in ``[lo, hi]``: every ``step``-th from ``center``, both ends."""
    first = center - ((center - lo) // step) * step
    return np.unique(np.concatenate(([lo], np.arange(first, hi + 1, step), [hi]))).astype(np.intp)


def _n_nodes(lo: IntArray, hi: IntArray) -> int:
    return int(np.prod(hi - lo + 1))


@dataclass(frozen=True, eq=False)
class EventLocation:
    """One located event. ``quality()`` gives the docs/02 ``LocationQuality`` fields."""

    e_m: float
    n_m: float
    u_m: float  # elevM - run origin elevM
    elev_m: float
    t0: float  # origin time, epoch s UTC
    arrivals: pd.DataFrame  # one row per pick (see ARRIVAL_COLUMNS)
    n_stations: int
    n_p: int
    n_s: int
    rms_s: float
    gap_deg: float
    min_epi_dist_m: float
    h_err_m: float | None  # None when the PDF is truncated (pdf_truncated)
    v_err_m: float | None
    depth_on_edge: bool
    statics_applied: bool  # at least one pick used in the location carries a non-zero static
    pdf: PdfSummary  # computed even when truncated, for the record
    pdf_truncated: bool
    misfit: float
    dropped_pick_ids: tuple[str, ...]
    outlier_mad_s: float  # MAD of the first-pass residuals (unscaled)
    outlier_threshold_s: float
    relocated: bool
    outlier_note: str | None  # set when the outlier pass was skipped
    search: dict[str, Any]  # evaluated region, faces, truncation, node counts

    def quality(self) -> dict[str, Any]:
        return {
            "method": METHOD,
            "statics": self.statics_applied,
            "nStations": self.n_stations,
            "nP": self.n_p,
            "nS": self.n_s,
            "rmsS": self.rms_s,
            "gapDeg": self.gap_deg,
            "minEpiDistM": self.min_epi_dist_m,
            "hErrM": self.h_err_m,
            "vErrM": self.v_err_m,
            "depthOnEdge": self.depth_on_edge,
        }


ARRIVAL_COLUMNS = (
    "pickId",
    "stationId",
    "phase",
    "tObs",
    "travelTimeS",
    "staticS",
    "tPred",
    "residualS",
    "sigmaS",
    "weight",
    "usedInLocation",  # docs/02 arrivals.parquet column name
)


class Locator:
    """Locates events for one station set. Coarse-grid travel times are cached across events.

    ``stations`` must hold only the stations in use (``usedInRun``), with columns id, enu_e,
    enu_n, enu_u, sensorElevM (and preprocessProfile when ``profilePickSigmaS`` is set).
    """

    def __init__(
        self,
        stations: pd.DataFrame,
        tables: StationTables,
        cfg: LocatorConfig,
        *,
        origin_elev_m: float,
        ref_surface_elev_m: float,
    ) -> None:
        missing = [c for c in STATION_COLUMNS if c not in stations.columns]
        if missing:
            raise ValueError(f"stations lack columns {missing}")
        if cfg.profilePickSigmaS and "preprocessProfile" not in stations.columns:
            raise ValueError("profilePickSigmaS is set but stations lack preprocessProfile")
        self.cfg = cfg
        self.tables = tables
        self.origin_elev_m = float(origin_elev_m)
        self.volume = make_volume(cfg, ref_surface_elev_m)
        ids = stations["id"].astype(str).tolist()
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate station ids")
        e = stations["enu_e"].to_numpy(dtype=np.float64)
        n = stations["enu_n"].to_numpy(dtype=np.float64)
        u = stations["enu_u"].to_numpy(dtype=np.float64)
        elev = stations["sensorElevM"].to_numpy(dtype=np.float64)
        if not all(np.all(np.isfinite(a)) for a in (e, n, u, elev)):
            raise ValueError("station coordinates must be finite")
        off = np.abs(u + self.origin_elev_m - elev)
        if np.any(off > cfg.enuConsistencyTolM):
            bad = [ids[i] for i in np.flatnonzero(off > cfg.enuConsistencyTolM)]
            raise ValueError(f"stations {bad}: enu_u + origin elevM differs from sensorElevM")
        for sid, z in zip(ids, elev, strict=True):
            if tables.receiver_elev_m.get(sid) != float(z):
                raise ValueError(f"station {sid}: no table at its sensorElevM {z}")
        self._index = {sid: i for i, sid in enumerate(ids)}
        self._e = e
        self._n = n
        self._profile = (
            stations["preprocessProfile"].astype(str).tolist()
            if "preprocessProfile" in stations.columns
            else [""] * len(ids)
        )
        unused = sorted(set(cfg.profilePickSigmaS) - set(self._profile))
        if unused:
            log.warning(
                "profilePickSigmaS keys %s match no station's preprocessProfile (typo?); those "
                "overrides apply to no station", unused,
            )
        self._check_reach(ids)
        vol = self.volume
        self._coarse_cache: dict[tuple[str, Phase], FloatArray] = {}
        ce, cn, cz = vol.coarse_axes()
        self._coarse_axes = (ce, cn, cz)
        self._n_coarse = ce.size * cn.size * cz.size
        log.info(
            "locator: %d stations, volume e/n %.0f..%.0f m, elevM %.1f..%.1f m ASL, %d coarse "
            "nodes at %.0f m, fine %.0f m",
            len(ids), vol.e_min_m, vol.e_max_m, vol.bottom_elev_m, vol.top_elev_m,
            self._n_coarse, vol.fine_m * vol.coarse_stride, vol.fine_m,
        )

    def _check_reach(self, ids: list[str]) -> None:
        vol = self.volume
        grid = self.tables.grid
        if vol.bottom_elev_m < grid.bottom_elev_m or vol.top_elev_m > grid.top_elev_m:
            raise ValueError(
                f"search volume elevM {vol.bottom_elev_m}..{vol.top_elev_m} is outside the "
                f"travel-time grid {grid.bottom_elev_m}..{grid.top_elev_m}"
            )
        far = []
        for sid, se, sn in zip(ids, self._e, self._n, strict=True):
            reach = max(
                math.hypot(ce - se, cn - sn)
                for ce in (vol.e_min_m, vol.e_max_m)
                for cn in (vol.n_min_m, vol.n_max_m)
            )
            if reach > grid.r_max_m:
                far.append(f"{sid} ({reach:.0f} m)")
        if far:
            raise ValueError(
                f"stations farther from a search-volume corner than grids.rMaxM "
                f"{grid.r_max_m:.0f} m: {far}; drop them or raise rMaxM"
            )

    # --- picks -----------------------------------------------------------------------------

    def _sigma(self, station_index: int, phase: Phase) -> float:
        override = self.cfg.profilePickSigmaS.get(self._profile[station_index])
        sig = override if override is not None else self.cfg.pickSigmaS
        return float(sig.P if phase == "P" else sig.S)

    def pick_sigma(self, station_id: str, phase: Phase) -> float:
        """Pick sigma (s) the locator uses for ``station_id`` and ``phase``."""
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        return self._sigma(self._index[station_id], phase)

    def _check_statics(self, statics: Statics) -> None:
        bad = sorted(
            f"{key!r}"
            for key in statics
            if not (
                isinstance(key, tuple)
                and len(key) == 2
                and key[0] in self._index
                and key[1] in PHASES
            )
        )
        if bad:
            raise ValueError(
                f"statics keys must be (stationId, phase) with a known station and phase P or S; "
                f"got {bad}"
            )

    def _prepare(self, picks: pd.DataFrame, statics: Statics | None) -> _Picks:
        missing = [c for c in PICK_COLUMNS if c not in picks.columns]
        if missing:
            raise ValueError(f"picks lack columns {missing}")
        if len(picks) < self.cfg.minPicks:
            raise ValueError(f"{len(picks)} picks; the locator needs at least {self.cfg.minPicks}")
        station_ids = picks["stationId"].astype(str).tolist()
        raw_phases = picks["phase"].astype(str).tolist()
        unknown = sorted({s for s in station_ids if s not in self._index})
        if unknown:
            raise ValueError(f"picks on stations without coordinates or tables: {unknown}")
        bad_phase = sorted({p for p in raw_phases if p not in PHASES})
        if bad_phase:
            raise ValueError(f"unknown phases {bad_phase}")
        phases = [cast(Phase, p) for p in raw_phases]
        pairs = list(zip(station_ids, phases, strict=True))
        if len(set(pairs)) != len(pairs):
            raise ValueError("more than one pick for a station and phase in one event")
        t = picks["t"].to_numpy(dtype=np.float64)
        prob = picks["prob"].to_numpy(dtype=np.float64)
        if not (np.all(np.isfinite(t)) and np.all((prob > 0) & (prob <= 1))):
            raise ValueError("pick times must be finite and probabilities in (0, 1]")
        if statics is not None:
            self._check_statics(statics)
        idx = [self._index[s] for s in station_ids]
        sigma = np.array([self._sigma(i, p) for i, p in zip(idx, phases, strict=True)])
        static = np.array([float((statics or {}).get(pair, 0.0)) for pair in pairs])
        if not np.all(np.isfinite(static)):
            raise ValueError("statics must be finite")
        t_ref = float(t.min())
        return _Picks(
            ids=picks["id"].astype(str).tolist(),
            station_ids=station_ids,
            phases=phases,
            t_obs=t,
            t_rel=t - t_ref,
            t_ref=t_ref,
            prob=prob,
            sigma=sigma,
            w=prob / sigma,
            static=static,
            se=self._e[idx],
            sn=self._n[idx],
        )

    # --- misfit evaluation -----------------------------------------------------------------

    def _coarse_times(self, station_id: str, phase: Phase) -> FloatArray:
        key = (station_id, phase)
        cached = self._coarse_cache.get(key)
        if cached is None:
            ce, cn, cz = self._coarse_axes
            vol = self.volume
            i = self._index[station_id]
            r = np.hypot(vol.e_at(ce)[None, :] - self._e[i], vol.n_at(cn)[:, None] - self._n[i])
            times = self.tables.table(station_id, phase).lookup(
                r[None, :, :], vol.z_at(cz)[:, None, None]
            )
            cached = np.ascontiguousarray(times.ravel())
            cached.setflags(write=False)
            self._coarse_cache[key] = cached
        return cached

    def _coarse_best(self, p: _Picks, used: NDArray[np.bool_]) -> tuple[int, int, int]:
        cols = np.flatnonzero(used)
        columns = [self._coarse_times(p.station_ids[j], p.phases[j]) for j in cols]
        offset = p.t_rel[cols] - p.static[cols]
        w = p.w[cols]
        chunk = self.cfg.evalChunkNodes
        best_val = np.inf
        best_idx = -1
        for start in range(0, self._n_coarse, chunk):
            stop = min(start + chunk, self._n_coarse)
            d = offset[None, :] - np.stack([c[start:stop] for c in columns], axis=1)
            misfit, _ = l1_misfit(d, w)
            k = int(np.argmin(misfit))
            if misfit[k] < best_val:
                best_val = float(misfit[k])
                best_idx = start + k
        ce, cn, cz = self._coarse_axes
        kz, kn, ke = np.unravel_index(best_idx, (cz.size, cn.size, ce.size))
        return int(cz[kz]), int(cn[kn]), int(ce[ke])

    def _box_misfit(
        self, p: _Picks, used: NDArray[np.bool_], ie: IntArray, i_n: IntArray, iz: IntArray
    ) -> tuple[FloatArray, FloatArray]:
        """Misfit and origin time (relative) on the box ``iz x i_n x ie``; shape (nz, nn, ne)."""
        vol = self.volume
        cols = np.flatnonzero(used)
        e = vol.e_at(ie)
        n = vol.n_at(i_n)
        z = vol.z_at(iz)
        r_by_station: dict[str, FloatArray] = {}
        for j in cols:
            sid = p.station_ids[j]
            if sid not in r_by_station:
                r_by_station[sid] = np.hypot(e[None, :] - p.se[j], n[:, None] - p.sn[j])
        offset = p.t_rel[cols] - p.static[cols]
        w = p.w[cols]
        plane = n.size * e.size
        levels = max(1, self.cfg.evalChunkNodes // plane)
        misfit = np.empty((z.size, n.size, e.size))
        t0 = np.empty_like(misfit)
        for start in range(0, z.size, levels):
            zc = z[start : start + levels]
            times = np.stack(
                [
                    self.tables.table(p.station_ids[j], p.phases[j])
                    .lookup(r_by_station[p.station_ids[j]][None, :, :], zc[:, None, None])
                    .ravel()
                    for j in cols
                ],
                axis=1,
            )
            m, t = l1_misfit(offset[None, :] - times, w)
            misfit[start : start + zc.size] = m.reshape(zc.size, n.size, e.size)
            t0[start : start + zc.size] = t.reshape(zc.size, n.size, e.size)
        return misfit, t0

    def _search(self, p: _Picks, used: NDArray[np.bool_]) -> _Search:
        cfg = self.cfg
        vol = self.volume
        cutoff = cfg.pdfCutoff / cfg.pdfMisfitScale  # in misfit units
        budget = cfg.maxPdfNodes
        upper = vol.upper
        center = np.array(self._coarse_best(p, used), dtype=np.intp)
        half = round(cfg.fineHalfWidthM / vol.fine_m)
        step = round(cfg.fineStageSpacingM / vol.fine_m)
        lo = np.maximum(center - half, 0)
        hi = np.minimum(center + half, upper)
        first_box = (lo.copy(), hi.copy())

        # Stage pass: grow the box until the stage nodes within cutoff clear every face that is
        # not a volume face, or their bounding box would exceed the node budget.
        n_stage = 0
        n_stage_grow = 0
        over_budget = False
        while True:
            axes = [_stage_axis(int(center[a]), int(lo[a]), int(hi[a]), step) for a in range(3)]
            m_stage, _ = self._box_misfit(p, used, axes[2], axes[1], axes[0])
            n_stage += int(m_stage.size)
            region = m_stage <= m_stage.min() + cutoff
            sub_lo = np.empty(3, dtype=np.intp)
            sub_hi = np.empty(3, dtype=np.intp)
            new_lo, new_hi = lo.copy(), hi.copy()
            for a in range(3):
                hit = np.flatnonzero(region.any(axis=tuple(b for b in range(3) if b != a)))
                sub_lo[a] = max(axes[a][hit[0]] - step, 0)
                sub_hi[a] = min(axes[a][hit[-1]] + step, upper[a])
                grow = max(int(hi[a] - lo[a]), half)
                if hit[0] == 0 and lo[a] > 0:
                    new_lo[a] = max(lo[a] - grow, 0)
                if hit[-1] == axes[a].size - 1 and hi[a] < upper[a]:
                    new_hi[a] = min(hi[a] + grow, upper[a])
            if _n_nodes(sub_lo, sub_hi) > budget:
                over_budget = True
                kz, kn, ke = np.unravel_index(int(np.argmin(m_stage)), m_stage.shape)
                best = np.array([axes[0][kz], axes[1][kn], axes[2][ke]], dtype=np.intp)
                sub_lo = np.maximum(best - half, 0)
                sub_hi = np.minimum(best + half, upper)
                break
            if np.array_equal(new_lo, lo) and np.array_equal(new_hi, hi):
                break
            lo, hi = new_lo, new_hi
            n_stage_grow += 1

        # Fine region: grow by a stage step on every non-volume face that still holds a node
        # within cutoff, unless that would exceed the node budget.
        n_fine_grow = 0
        while True:
            iz = np.arange(sub_lo[0], sub_hi[0] + 1)
            i_n = np.arange(sub_lo[1], sub_hi[1] + 1)
            ie = np.arange(sub_lo[2], sub_hi[2] + 1)
            misfit, t0 = self._box_misfit(p, used, ie, i_n, iz)
            level = misfit.min() + cutoff
            touch = [
                (
                    bool(misfit.take(0, axis=a).min() <= level),
                    bool(misfit.take(-1, axis=a).min() <= level),
                )
                for a in range(3)
            ]
            if over_budget:
                break
            new_lo, new_hi = sub_lo.copy(), sub_hi.copy()
            for a in range(3):
                if touch[a][0] and sub_lo[a] > 0:
                    new_lo[a] = max(sub_lo[a] - step, 0)
                if touch[a][1] and sub_hi[a] < upper[a]:
                    new_hi[a] = min(sub_hi[a] + step, upper[a])
            if np.array_equal(new_lo, sub_lo) and np.array_equal(new_hi, sub_hi):
                break
            if _n_nodes(new_lo, new_hi) > budget:
                over_budget = True
                break
            sub_lo, sub_hi = new_lo, new_hi
            n_fine_grow += 1

        at_volume = [(bool(sub_lo[a] == 0), bool(sub_hi[a] == upper[a])) for a in range(3)]
        truncated = [
            AXIS_FACES[a][side]
            for a in range(3)
            for side in (0, 1)
            if touch[a][side] and not (a == 0 and at_volume[a][side])
        ]
        pdf = summarize_pdf(
            misfit,
            vol.e_at(ie),
            vol.n_at(i_n),
            vol.z_at(iz),
            spacing_h_m=vol.fine_m,
            spacing_z_m=vol.fine_m,
            misfit_scale=cfg.pdfMisfitScale,
            top_is_volume_top=at_volume[0][1],
            bottom_is_volume_bottom=at_volume[0][0],
            confidence=cfg.errConfidence,
            edge_fraction=cfg.depthOnEdgeMassFraction,
        )
        kz, kn, ke = np.unravel_index(int(np.argmin(misfit)), misfit.shape)
        f_lo, f_hi = first_box
        box = {
            "firstFineBoxElevM": [float(vol.z_at(f_lo[0])), float(vol.z_at(f_hi[0]))],
            "firstFineBoxNM": [float(vol.n_at(f_lo[1])), float(vol.n_at(f_hi[1]))],
            "firstFineBoxEM": [float(vol.e_at(f_lo[2])), float(vol.e_at(f_hi[2]))],
            "evaluatedElevM": [float(vol.z_at(iz[0])), float(vol.z_at(iz[-1]))],
            "evaluatedNM": [float(vol.n_at(i_n[0])), float(vol.n_at(i_n[-1]))],
            "evaluatedEM": [float(vol.e_at(ie[0])), float(vol.e_at(ie[-1]))],
            "atVolumeTop": at_volume[0][1],
            "atVolumeBottom": at_volume[0][0],
            "volumeFaces": [
                AXIS_FACES[a][side] for a in range(3) for side in (0, 1) if at_volume[a][side]
            ],
            "faceMass": dict(pdf.face_mass),
            "mapOnVolumeTop": pdf.map_on_volume_top,
            "mapOnVolumeBottom": pdf.map_on_volume_bottom,
            "pdfTruncated": bool(truncated),
            "truncatedFaces": truncated,
            "nodeBudgetHit": over_budget,
            "nStageGrow": n_stage_grow,
            "nFineGrow": n_fine_grow,
            "nCoarse": self._n_coarse,
            "nStage": n_stage,
            "nFine": int(misfit.size),
        }
        return _Search(
            iz=int(iz[kz]),
            i_n=int(i_n[kn]),
            ie=int(ie[ke]),
            t0_rel=float(t0[kz, kn, ke]),
            misfit=float(misfit[kz, kn, ke]),
            pdf=pdf,
            pdf_truncated=bool(truncated),
            box=box,
        )

    def _travel_times(self, p: _Picks, e: float, n: float, z: float) -> FloatArray:
        return np.array(
            [
                float(
                    self.tables.table(sid, ph).lookup(math.hypot(e - se, n - sn), z)
                )
                for sid, ph, se, sn in zip(p.station_ids, p.phases, p.se, p.sn, strict=True)
            ]
        )

    # --- public ----------------------------------------------------------------------------

    def locate(self, picks: pd.DataFrame, *, statics: Statics | None = None) -> EventLocation:
        """Locate one event from its picks (columns id, stationId, phase, t, prob).

        ``statics`` maps (stationId, phase) to an additive static (s); missing pairs use 0.
        """
        p = self._prepare(picks, statics)
        all_used = np.ones(len(p.ids), dtype=bool)
        first = self._search(p, all_used)
        res = self._residuals(p, first)
        mad = float(np.median(np.abs(res - np.median(res))))
        threshold = max(self.cfg.outlier.madK * mad, self.cfg.outlier.floorS)
        drop = np.abs(res) > threshold
        final, used, note = first, all_used, None
        if drop.any():
            if int((~drop).sum()) < self.cfg.minPicks:
                note = (
                    f"outlier pass skipped: dropping {int(drop.sum())} picks would leave "
                    f"{int((~drop).sum())} < minPicks {self.cfg.minPicks}"
                )
            else:
                used = ~drop
                final = self._search(p, used)
        return self._result(p, final, used, mad, threshold, note)

    def _residuals(self, p: _Picks, s: _Search) -> FloatArray:
        vol = self.volume
        e, n, z = float(vol.e_at(s.ie)), float(vol.n_at(s.i_n)), float(vol.z_at(s.iz))
        return p.t_rel - self._travel_times(p, e, n, z) - p.static - s.t0_rel

    def _result(
        self,
        p: _Picks,
        s: _Search,
        used: NDArray[np.bool_],
        mad: float,
        threshold: float,
        note: str | None,
    ) -> EventLocation:
        vol = self.volume
        e, n, z = float(vol.e_at(s.ie)), float(vol.n_at(s.i_n)), float(vol.z_at(s.iz))
        tt = self._travel_times(p, e, n, z)
        t0 = p.t_ref + s.t0_rel
        residual = p.t_obs - (t0 + tt + p.static)
        arrivals = pd.DataFrame(
            {
                "pickId": p.ids,
                "stationId": p.station_ids,
                "phase": p.phases,
                "tObs": p.t_obs,
                "travelTimeS": tt,
                "staticS": p.static,
                "tPred": t0 + tt + p.static,
                "residualS": residual,
                "sigmaS": p.sigma,
                "weight": p.w,
                "usedInLocation": used,
            },
            columns=list(ARRIVAL_COLUMNS),
        )
        used_stations = sorted({p.station_ids[j] for j in np.flatnonzero(used)})
        si = [self._index[sid] for sid in used_stations]
        de = self._e[si] - e
        dn = self._n[si] - n
        used_phases = [p.phases[j] for j in np.flatnonzero(used)]
        return EventLocation(
            e_m=e,
            n_m=n,
            u_m=z - self.origin_elev_m,
            elev_m=z,
            t0=t0,
            arrivals=arrivals,
            n_stations=len(used_stations),
            n_p=used_phases.count("P"),
            n_s=used_phases.count("S"),
            rms_s=float(np.sqrt(np.mean(residual[used] ** 2))),
            gap_deg=azimuthal_gap_deg(de, dn),
            min_epi_dist_m=float(np.min(np.hypot(de, dn))),
            h_err_m=None if s.pdf_truncated else s.pdf.h_err_m,
            v_err_m=None if s.pdf_truncated else s.pdf.v_err_m,
            depth_on_edge=s.pdf.depth_on_edge,
            statics_applied=bool(np.any(p.static[used] != 0.0)),
            pdf=s.pdf,
            pdf_truncated=s.pdf_truncated,
            misfit=s.misfit,
            dropped_pick_ids=tuple(p.ids[j] for j in np.flatnonzero(~used)),
            outlier_mad_s=mad,
            outlier_threshold_s=threshold,
            relocated=bool((~used).any()),
            outlier_note=note,
            search=dict(s.box),
        )

    def to_record(self) -> dict[str, Any]:
        """Locator parameters and conventions for ``ProcessingRun.locator``.

        ``tables.velocityModel`` is the top-extended model the tables were solved on (with its
        ``topExtension``); ``ProcessingRun.velocityModel`` must be that record
        (``velocity_model_record()``), not the source model's.
        """
        return {
            "method": METHOD,
            "config": self.cfg.model_dump(mode="json"),
            "volume": self.volume.to_record(),
            "stationIds": list(self._index),
            "misfit": "sum_i w_i |t_obs_i - T_i - static_i - t0|, w_i = prob_i / sigma_i; t0 = "
            "lower weighted median of (t_obs_i - T_i - static_i) (eliminated analytically)",
            "search": "coarse grid over the volume; a fineStageSpacingM pass from the first fine "
            "box (fineHalfWidthM around the best coarse node) grown until the region within "
            "pdfCutoff clears every non-volume face; fine nodes over that region, grown the same "
            "way; both stop at maxPdfNodes (then hErrM/vErrM are None, pdfTruncated); hypocentre "
            "= fine MAP node",
            "outliers": "drop |residual| > max(madK * MAD, floorS), MAD unscaled over both "
            "phases; relocate once",
            "statics": "additive per (stationId, phase), default 0; LocationQuality.statics is "
            "true when a pick used in the location carries a non-zero static",
            "uncertainty": conventions(
                self.cfg.errConfidence, self.cfg.depthOnEdgeMassFraction, self.cfg.pdfMisfitScale
            ),
            "tables": self.tables.to_record(),
        }

    def velocity_model_record(self) -> dict[str, Any]:
        """``ProcessingRun.velocityModel``: the top-extended model the tables were solved on."""
        return self.tables.model.to_record()


@dataclass(frozen=True)
class LocatorSetup:
    """Everything a (possibly spawned) process needs to rebuild the same Locator."""

    stations: pd.DataFrame
    model: LayerModel  # the source model; the top extension happens in build_station_tables
    config: SeismologyConfig
    run: RunSection
    cache_dir: Path


def build_locator(setup: LocatorSetup) -> Locator:
    """Tables (built or loaded from ``<cache_dir>/ttgrids``) and the Locator for ``setup``."""
    cfg = setup.config
    volume = make_volume(cfg.locator, setup.run.refSurfaceElevM)
    tables = build_station_tables(
        setup.stations,
        setup.model,
        cfg.grids,
        cache_dir=setup.cache_dir,
        max_extension_m=cfg.velocity.maxTopExtensionM,
        cover_top_elev_m=volume.top_elev_m,
    )
    return Locator(
        setup.stations,
        tables,
        cfg.locator,
        origin_elev_m=setup.run.origin.elevM,
        ref_surface_elev_m=setup.run.refSurfaceElevM,
    )


_WORKER: Locator | None = None


def _init_worker(setup: LocatorSetup) -> None:
    global _WORKER  # one Locator per spawned worker process
    _WORKER = build_locator(setup)


def _locate_in_worker(args: tuple[pd.DataFrame, Statics | None]) -> EventLocation:
    if _WORKER is None:
        raise RuntimeError("worker process was not initialised")
    picks, statics = args
    return _WORKER.locate(picks, statics=statics)


def locate_many(
    setup: LocatorSetup,
    events: Sequence[pd.DataFrame],
    *,
    statics: Statics | None = None,
    locator: Locator | None = None,
) -> list[EventLocation]:
    """Locate every event (one picks frame each), in order.

    With ``locator.nWorkers > 1`` the events are split over spawned processes that load the
    same cached tables; each event is located independently, so results do not depend on the
    worker count. ``locator`` (built from ``setup``) is reused for the serial path; one built
    from another config or station set raises, since the workers rebuild from ``setup``.
    """
    started = time.perf_counter()
    if locator is not None and (
        locator.cfg != setup.config.locator
        or list(locator._index) != setup.stations["id"].astype(str).tolist()
    ):
        raise ValueError("locator was not built from setup (config or stations differ)")
    n_workers = min(setup.config.locator.nWorkers, len(events))
    if n_workers <= 1:
        loc = locator if locator is not None else build_locator(setup)
        out = [loc.locate(ev, statics=statics) for ev in events]
    else:
        if locator is None:
            build_locator(setup)  # build and cache the tables once, before the workers load them
        ctx = multiprocessing.get_context("spawn")
        chunk = max(1, math.ceil(len(events) / (4 * n_workers)))
        with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=(setup,)) as pool:
            out = list(pool.map(_locate_in_worker, [(ev, statics) for ev in events],
                                chunksize=chunk))
    elapsed = time.perf_counter() - started
    log.info(
        "located %d events with %d worker(s) in %.1f s (%d relocated after the outlier pass)",
        len(out), max(n_workers, 1), elapsed, sum(loc_.relocated for loc_ in out),
    )
    return out

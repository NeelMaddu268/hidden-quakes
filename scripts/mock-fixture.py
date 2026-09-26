#!/usr/bin/env python3
"""Generate the synthetic mock bundle for ``apps/web/public/data/mock/`` (ticket FIX-01).

This is the only place in the repo that may invent seismic data (CLAUDE.md rule 5). Every
record it writes is made up from the knobs below: no station code, event, pick, waveform or
validation number comes from a real catalog, a real run or pre-event outputs. The bundle carries
``isSynthetic: true`` in ``meta.scene`` and ``meta.run`` so the web app shows its SYNTHETIC banner.

What it produces (docs/01 -> Data bundle), every file validated by ``hq_contracts.models``::

    meta.json                BundleMeta   mode "mock", scene, run, summary recomputed from the data
    stations.json            Station[]    12 surface stations on a ring + 3 borehole sensors
    catalog.json             CatalogEvent[]  43 public points, 38 matched to a candidate
    events.json              SeismicEvent[]  ~500 candidates: Tier A cluster, B around it, C scattered
    features.json            GeoFeature[]  one synthetic well, one facility, one boundary; all unverified
    validation.json          Validation   baseline table, sweep, null test, G-R, calibration, synthetic
    evidence/{eventId}.json  EventEvidence  20 events with synthetic wavelets at predicted times

Usage (from ``services/seismic`` so ``hq`` and ``hq_contracts`` resolve)::

    uv run python ../../scripts/mock-fixture.py [--out DIR] [--seed N]

The same seed gives byte-identical output: one seeded numpy generator drives every random draw,
every float is rounded to a fixed number of decimals, and JSON keys are sorted.

Coordinates follow docs/01 -> Conventions: one vertical (``elevM``, m ASL); ``enu`` is offset
from the run.yaml origin (a local equirectangular approximation stands in for UTM 12N here);
``depthKm = (refSurfaceElevM - elevM) / 1000``. The window, bbox, origin and reference surface
come from ``configs/showcase/run.yaml`` so the mock sits in the real region.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pydantic
import yaml
from hq_contracts import models as m

from hq.config.run import RunSection

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_YAML = REPO_ROOT / "services" / "seismic" / "configs" / "showcase" / "run.yaml"
DEFAULT_OUT = REPO_ROOT / "apps" / "web" / "public" / "data" / "mock"
DEFAULT_SEED = 13
DOCS_URL = "https://github.com/NeelMaddu268/hidden-quakes/blob/main/docs/lanes/H4-platform.md"

TIERS: tuple[str, ...] = ("A", "B", "C")
BUNDLE_FILES: tuple[str, ...] = (
    "meta.json",
    "stations.json",
    "catalog.json",
    "events.json",
    "features.json",
    "validation.json",
)

# Local equirectangular scale: the mock derives latitude/longitude from ENU meters with this
# (the pipeline uses UTM 12N; docs/01). Longitude uses M_PER_DEG_LAT * cos(originLat).
M_PER_DEG_LAT = 111_320.0


# ------------------------------------------------------------------------------------------
# Knobs: every number the generator uses lives here (CLAUDE.md rule 8). The code below only
# adds unit conversions (m <-> km, s <-> h) and index arithmetic.
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Knobs:
    """Generator parameters. Fields keyed "A"/"B"/"C" are per tier."""

    # --- counts -------------------------------------------------------------------------
    n_public: int = 43  # public regional catalog rows (the ticket's "43 public points")
    n_matched: dict[str, int] = field(default_factory=lambda: {"A": 26, "B": 10, "C": 2})  # =38
    n_events: dict[str, int] = field(default_factory=lambda: {"A": 150, "B": 200, "C": 150})
    n_evidence: dict[str, int] = field(default_factory=lambda: {"A": 12, "B": 5, "C": 3})  # =20
    n_surface_stations: int = 12
    n_unused_surface_stations: int = 1  # usedInRun false (e.g. too many gaps), exercises the UI
    max_sample_attempts: int = 50  # redraws allowed before an event's tier is declared unreachable

    # --- fixed identifiers ----------------------------------------------------------------
    network: str = "XX"  # clearly synthetic network code; never a real one
    station_prefix: str = "M"  # "XX.M01" ...
    run_id_prefix: str = "mock"  # runId = "mock-<seed>"
    git_sha: str = "mock0000"
    created_at: str = "2026-09-11T01:00:00Z"  # fixed so output is byte-identical per seed
    picker_model: str = "seisbench.PhaseNet"
    picker_weights: str = "synthetic"  # picks are invented; picker id "phasenet:synthetic"
    catalog_source: str = "synthetic public catalog (mock fixture)"
    catalog_id_prefix: str = "mockpub"
    catalog_depth_datum: str = "sea level"
    well_name: str = "Synthetic well A"

    # --- station geometry (meters from the run.yaml origin) --------------------------------
    ring_radius_m: float = 10_000.0  # 12 surface stations on a ~20 km diameter ring
    ring_radius_jitter_m: float = 1_200.0
    ring_angle_jitter_deg: float = 8.0
    surface_elev_min_m: float = 1_500.0  # plausible ASL range for the ring stations
    surface_elev_max_m: float = 2_200.0
    terrain_amplitude_m: tuple[float, float, float] = (180.0, 140.0, 90.0)  # synthetic relief
    terrain_wavelength_m: tuple[float, float, float] = (7_000.0, 5_500.0, 4_200.0)
    terrain_phase: tuple[float, float] = (0.4, -0.9)
    borehole_en_m: tuple[tuple[float, float], ...] = (
        (1_150.0, -550.0),
        (2_650.0, -1_750.0),
        (900.0, -2_150.0),
    )
    borehole_depth_m: tuple[float, ...] = (380.0, 720.0, 1_050.0)
    surface_sample_rate_hz: float = 100.0
    borehole_sample_rate_hz: float = 1_000.0
    surface_channels: tuple[str, ...] = ("HHZ", "HHN", "HHE")
    borehole_channels: tuple[str, ...] = ("DPZ", "DP1", "DP2")
    surface_profile: str = "surface"
    borehole_profile: str = "borehole"
    statics_fraction: float = 0.6  # share of used stations with staticsS filled
    statics_max_s: float = 0.06

    # --- event clusters (meters; depth is below refSurfaceElevM) ------------------------
    cluster_center_en_m: tuple[float, float] = (1_800.0, -1_200.0)
    cluster_center_depth_m: float = 2_000.0
    # Tier A: compact ellipsoid ~2 km x 0.5 km x 1.2 km (2 sigma), striking 40 deg, dipping 70 deg
    a_sigma_m: tuple[float, float, float] = (550.0, 110.0, 300.0)  # (strike, normal, dip)
    a_strike_deg: float = 40.0
    a_dip_deg: float = 70.0
    a_depth_range_m: tuple[float, float] = (900.0, 3_200.0)
    # Tier B: diffuse halo around the same center
    b_sigma_m: tuple[float, float, float] = (1_200.0, 700.0, 650.0)
    b_depth_range_m: tuple[float, float] = (200.0, 5_500.0)
    # Tier C: scattered through the search volume
    c_half_extent_m: float = 12_000.0
    grid_depth_range_m: tuple[float, float] = (0.0, 8_000.0)  # locator grid; edge events pin here
    edge_band_m: float = 60.0  # how close to the grid face a depthOnEdge event sits
    edge_top_frac: float = 0.5  # share of Tier C edge events pinned at the top (rest: bottom)

    # --- origin times (hours after windowStart) -------------------------------------------
    swarm_center_h: tuple[float, ...] = (3.3, 11.7, 19.2)
    swarm_sigma_h: tuple[float, ...] = (0.6, 1.1, 0.8)
    swarm_weight: tuple[float, ...] = (0.35, 0.40, 0.25)
    swarm_fraction: dict[str, float] = field(
        default_factory=lambda: {"A": 0.85, "B": 0.65, "C": 0.2}
    )

    # --- fake velocity used for predicted arrivals and the synthetic wavelets --------------
    vp_km_s: float = 5.5
    vs_km_s: float = 3.2
    pick_sigma_s: dict[str, float] = field(default_factory=lambda: {"P": 0.03, "S": 0.06})

    # --- per-tier quality ranges (sampled so each tier's thresholds hold by construction) --
    n_stations_range: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"A": (8, 14), "B": (5, 11), "C": (4, 7)}
    )
    s_dropout_max: dict[str, int] = field(default_factory=lambda: {"A": 3, "B": 4, "C": 3})
    rms_range_s: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"A": (0.015, 0.055), "B": (0.03, 0.12), "C": (0.06, 0.25)}
    )
    h_err_range_m: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"A": (60.0, 350.0), "B": (200.0, 900.0), "C": (600.0, 3_000.0)}
    )
    v_err_range_m: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "A": (100.0, 500.0),
            "B": (300.0, 1_500.0),
            "C": (900.0, 4_000.0),
        }
    )
    v_err_missing_frac: dict[str, float] = field(
        default_factory=lambda: {"A": 0.0, "B": 0.1, "C": 0.4}
    )
    depth_on_edge_frac: dict[str, float] = field(
        default_factory=lambda: {"A": 0.0, "B": 0.15, "C": 0.5}
    )
    pick_prob_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"A": (0.7, 0.99), "B": (0.5, 0.9), "C": (0.3, 0.75)}
    )
    statics_applied: dict[str, bool] = field(
        default_factory=lambda: {"A": True, "B": True, "C": False}
    )
    location_method: str = "grid1d"

    # --- tier thresholds, recorded verbatim in meta.run.tiering ----------------------------
    tier_a: dict[str, float] = field(
        default_factory=lambda: {
            "minStations": 8,
            "maxRmsS": 0.055,
            "maxHErrM": 350.0,
            "maxVErrM": 500.0,
            "maxGapDeg": 180.0,
        }
    )
    tier_b: dict[str, float] = field(
        default_factory=lambda: {"minStations": 5, "maxRmsS": 0.12, "maxHErrM": 900.0}
    )
    tier_a_quantile_label: str = "p75 of matched"  # what the A thresholds were derived from
    tier_a_station_quantile_label: str = "p25 of matched"
    tier_b_quantile_label: str = "p95 of matched"

    # --- magnitudes -----------------------------------------------------------------------
    mag_type: str = "ML_cal"
    mag_min: float = -0.6  # Gutenberg-Richter sampling floor
    mag_max: float = 2.4
    mag_b_value: float = 1.0
    mag_sigma_range: tuple[float, float] = (0.12, 0.30)
    mag_missing_frac: float = 0.06
    catalog_mag_type: str = "ML"
    catalog_mag_alt_type: str = "Md"
    catalog_mag_alt_frac: float = 0.15
    catalog_mag_bias: float = 0.15  # public ML runs a little higher than our ML_cal
    catalog_mag_sigma: float = 0.2
    catalog_mag_missing: int = 6  # public rows with no magnitude
    unmatched_mag_range: tuple[float, float] = (0.8, 1.9)

    # --- catalog matching -----------------------------------------------------------------
    match_dt_sigma_s: float = 0.4
    match_dt_max_s: float = 3.0
    match_dist_sigma_m: float = 500.0
    match_dist_max_m: float = 5_000.0
    catalog_depth_sigma_km: float = 0.8  # published depths are rougher than ours
    catalog_depth_decimals: int = 2
    unmatched_near_radius_m: tuple[float, float] = (6_000.0, 9_000.0)  # in the ring, off-cluster
    unmatched_far_radius_m: tuple[float, float] = (18_000.0, 22_000.0)  # outside the ring
    n_unmatched_near: int = 2
    unmatched_depth_range_m: tuple[float, float] = (1_000.0, 6_000.0)

    # --- evidence snippets ----------------------------------------------------------------
    evidence_dt_s: float = 0.01
    evidence_min_s: float = 4.0
    evidence_max_s: float = 8.0
    evidence_sample_budget: int = 7_600  # samples per file; keeps each file under 60 KB
    evidence_max_traces: int = 16
    evidence_pre_p_s: float = 1.0  # window starts this long before the predicted P
    evidence_post_s_s: float = 0.8  # and must extend this long past the predicted S
    filter_hz: tuple[float, float] = (2.0, 20.0)
    p_wavelet_hz: float = 10.0
    s_wavelet_hz: float = 6.0
    p_amp: float = 0.45
    s_amp: float = 1.0
    amp_ref_dist_m: float = 2_000.0  # amplitude decays as (ref / r) ** amp_decay
    amp_decay: float = 0.7
    noise_amp_range: tuple[float, float] = (0.06, 0.15)
    noise_smooth_samples: int = 5
    coda_amp: float = 0.6
    coda_decay_s: float = 1.2
    surface_vertical_channel: str = "HHZ"
    borehole_vertical_channel: str = "DPZ"

    # --- features -------------------------------------------------------------------------
    well_head_en_m: tuple[float, float] = (1_400.0, -900.0)
    well_bottom_elev_m: float = -1_500.0
    well_points: int = 12
    well_deviation_m: tuple[float, float] = (350.0, -260.0)  # horizontal drift head -> toe
    facility_en_m: tuple[float, float] = (1_050.0, -450.0)
    boundary_center_en_m: tuple[float, float] = (1_700.0, -1_100.0)
    boundary_radius_m: float = 3_200.0
    boundary_points: int = 6

    # --- validation ------------------------------------------------------------------------
    # Baseline rows as fractions of the phasenet/full row (candidates, recovered, A, B, C, rms, sta)
    baseline_factors: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "phasenet/p_only": (0.74, 0.92, 0.68, 0.80, 0.85, 1.3, 0.9),
            "stalta/full": (0.41, 0.76, 0.31, 0.45, 0.55, 1.6, 0.8),
            "stalta/p_only": (0.30, 0.63, 0.22, 0.35, 0.45, 1.9, 0.75),
        }
    )
    sweep_n_p_and_s_min: tuple[int, ...] = (4, 5, 6, 7, 8)
    sweep_chosen_n_p_and_s_min: int = 6
    sweep_tolerances_s: tuple[float, ...] = (1.0, 1.5, 2.0)
    sweep_chosen_tolerance_s: float = 1.5
    # (candidates, recovered, tierA) multipliers per nPAndSMin, relative to the chosen point
    sweep_factors: dict[int, tuple[float, float, float]] = field(
        default_factory=lambda: {
            4: (1.9, 1.0, 0.93),
            5: (1.35, 1.0, 0.98),
            6: (1.0, 1.0, 1.0),
            7: (0.72, 0.92, 0.85),
            8: (0.5, 0.8, 0.65),
        }
    )
    sweep_tolerance_factors: dict[float, tuple[float, float, float]] = field(
        default_factory=lambda: {
            1.0: (0.86, 0.95, 0.9),
            1.5: (1.0, 1.0, 1.0),
            2.0: (1.22, 1.0, 0.96),
        }
    )
    null_shuffles: int = 20
    null_shift_range_s: float = 30.0
    null_mean_events: float = 2.4
    null_mean_strict: float = 0.1
    null_std_events: float = 1.3
    gr_min_mag: float = -0.5
    gr_max_mag: float = 2.5
    gr_bin: float = 0.25
    gr_mc_offset: float = 0.2  # Mc = maximum curvature + 0.2
    shi_bolt_factor: float = 2.3  # ln(10), as in the Shi & Bolt sigma_b formula
    calib_coefficients: dict[str, float] = field(
        default_factory=lambda: {"a": 1.0, "b": 1.11, "c": 0.00189, "d": -2.09}
    )
    synthetic_n_events: int = 200
    synthetic_median_h_err_m: float = 180.0
    synthetic_median_v_err_m: float = 260.0
    synthetic_p90_v_err_m: float = 620.0
    synthetic_median_depth_bias_m: float = -40.0

    # --- run metadata (plausible synthetic params; recorded verbatim) ----------------------
    runtime_s: dict[str, float] = field(
        default_factory=lambda: {
            "inventory": 4.2,
            "catalog": 6.1,
            "download": 812.0,
            "pick": 1460.5,
            "baseline": 220.3,
            "associate": 95.4,
            "locate": 310.8,
            "match": 1.7,
            "tier": 0.9,
            "magnitude": 48.2,
            "validate": 640.0,
            "export": 12.5,
        }
    )
    picker_params: dict[str, object] = field(
        default_factory=lambda: {
            "pThreshold": 0.3,
            "sThreshold": 0.3,
            "storeThreshold": 0.1,
            "windowS": 60.0,
            "overlapS": 10.0,
            "profiles": {"surface": "synthetic", "borehole": "synthetic"},
        }
    )
    associator_params: dict[str, object] = field(
        default_factory=lambda: {
            "engine": "pyocto (synthetic parameters)",
            "velocityModel": "mock-1d-halfspace",
            "timeBeforeS": 300.0,
            "nPMin": 4,
            "nSMin": 2,
            "nPAndSMin": 6,
            "pickMatchToleranceS": 1.5,
            "minPickFraction": 0.25,
            "zlimM": [0.0, 8000.0],
        }
    )
    locator_params: dict[str, object] = field(
        default_factory=lambda: {
            "method": "grid1d",
            "gridSpacingM": 100.0,
            "depthRangeM": [0.0, 8000.0],
            "statics": True,
            "staticsIterations": 3,
            "errorLevel": 0.68,
            "edgeMassThreshold": 0.05,
        }
    )
    matching_params: dict[str, object] = field(
        default_factory=lambda: {"dtMaxS": 3.0, "distMaxM": 5000.0, "oneToOne": True}
    )

    # --- rounding (decimals) so identical seeds give identical bytes -----------------------
    dec_m: int = 1
    dec_s: int = 3
    dec_deg: int = 6
    dec_prob: int = 3
    dec_mag: int = 2
    dec_sample: int = 3
    dec_depth_km: int = 4
    dec_recall: int = 4


# ------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------


def rnd(x: float, decimals: int) -> float:
    """Round to a fixed number of decimals and normalise -0.0 to 0.0."""
    return round(float(x), decimals) + 0.0


@dataclass(frozen=True)
class Frame:
    """Coordinate frame from run.yaml: origin for ENU and reference surface for display depth."""

    origin_lat: float
    origin_lon: float
    origin_elev_m: float
    ref_surface_elev_m: float
    k: Knobs

    def latitude(self, n: float) -> float:
        return rnd(self.origin_lat + n / M_PER_DEG_LAT, self.k.dec_deg)

    def longitude(self, e: float) -> float:
        scale = M_PER_DEG_LAT * math.cos(math.radians(self.origin_lat))
        return rnd(self.origin_lon + e / scale, self.k.dec_deg)

    def elev(self, elev_m: float) -> float:
        """Canonical rounded ``elevM``; every other vertical field derives from this value."""
        return rnd(elev_m, self.k.dec_m)

    def enu(self, e: float, n: float, elev_m: float) -> m.Enu:
        return m.Enu(
            e=rnd(e, self.k.dec_m),
            n=rnd(n, self.k.dec_m),
            u=rnd(self.elev(elev_m) - self.origin_elev_m, self.k.dec_m),
        )

    def depth_km(self, elev_m: float) -> float:
        return rnd((self.ref_surface_elev_m - self.elev(elev_m)) / 1000.0, self.k.dec_depth_km)

    def surface_elev_m(self, e: float, n: float) -> float:
        """Smooth synthetic relief, equal to the reference surface at the origin."""
        amp = self.k.terrain_amplitude_m
        wl = self.k.terrain_wavelength_m
        ph = self.k.terrain_phase

        def relief(x: float, y: float) -> float:
            return (
                amp[0] * math.sin(x / wl[0] + ph[0])
                + amp[1] * math.cos(y / wl[1] + ph[1])
                + amp[2] * math.sin((x + y) / wl[2])
            )

        return self.ref_surface_elev_m + relief(e, n) - relief(0.0, 0.0)


def window_label(start: datetime, end: datetime) -> str:
    """'2026-09-10 00:00-24:00 UTC' for one whole day; otherwise explicit bounds."""
    whole_day = end - start == timedelta(days=1) and start.hour == 0 and start.minute == 0
    if whole_day:
        return f"{start:%Y-%m-%d} 00:00-24:00 UTC"
    if end.date() == start.date():
        return f"{start:%Y-%m-%d %H:%M}-{end:%H:%M} UTC"
    return f"{start:%Y-%m-%d %H:%M}-{end:%Y-%m-%d %H:%M} UTC"


def azimuthal_gap_deg(azimuths_deg: np.ndarray) -> float:
    a = np.sort(np.mod(azimuths_deg, 360.0))
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return float(gaps.max())


def ricker(t: np.ndarray, freq_hz: float) -> np.ndarray:
    a = (math.pi * freq_hz * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


# ------------------------------------------------------------------------------------------
# Drafts: mutable working records before they become contract models
# ------------------------------------------------------------------------------------------


@dataclass
class StationDraft:
    id: str
    e: float
    n: float
    surface_elev_m: float  # rounded, canonical
    sensor_depth_m: float  # rounded, canonical
    kind: str
    used: bool

    @property
    def sensor_elev_m(self) -> float:
        return self.surface_elev_m - self.sensor_depth_m


@dataclass
class PickDraft:
    id: str
    station: StationDraft
    phase: str
    t: float
    prob: float


@dataclass
class EventDraft:
    tier: str
    e: float
    n: float
    elev_m: float
    t: float
    n_stations: int
    n_p: int
    n_s: int
    rms_s: float
    h_err_m: float | None
    v_err_m: float | None
    depth_on_edge: bool
    magnitude: m.Magnitude | None
    picks: list[PickDraft] = field(default_factory=list)
    gap_deg: float = 0.0
    min_epi_dist_m: float = 0.0
    mean_pick_prob: float = 0.0
    id: str = ""
    reveal_order: int = -1
    catalog_match: m.CatalogMatch | None = None
    tier_reasons: list[str] = field(default_factory=list)

    def epi_dist_m(self, st: StationDraft) -> float:
        return math.hypot(st.e - self.e, st.n - self.n)

    def hypo_dist_m(self, st: StationDraft) -> float:
        return math.sqrt(self.epi_dist_m(st) ** 2 + (st.sensor_elev_m - self.elev_m) ** 2)


# ------------------------------------------------------------------------------------------
# Tiering: thresholds live in Knobs and are written verbatim to meta.run.tiering; the reasons
# on each event quote them, so every number on screen traces to that config block.
# ------------------------------------------------------------------------------------------


def _fmt_check(name: str, value: float | None, op: str, limit: float, label: str) -> str:
    if value is None:
        return f"{name} missing"
    shown = f"{value:.3f}" if name == "rmsS" else f"{value:.0f}"
    lim = f"{limit}" if name == "rmsS" else f"{limit:.0f}"
    return f"{name} {shown} {op} {lim} ({label})"


def _le_check(name: str, value: float | None, limit: float, label: str) -> tuple[bool, str, str]:
    ok = value is not None and value <= limit
    return (
        ok,
        _fmt_check(name, value, "<=", limit, label),
        _fmt_check(name, value, ">", limit, label),
    )


def _ge_check(name: str, value: int, limit: float, label: str) -> tuple[bool, str, str]:
    ok = value >= limit
    return ok, f"{name} {value} >= {limit} ({label})", f"{name} {value} < {limit} ({label})"


def _split(checks: list[tuple[bool, str, str]]) -> tuple[list[str], list[str]]:
    passed = [ok_msg for ok, ok_msg, _ in checks if ok]
    failed = [bad_msg for ok, _, bad_msg in checks if not ok]
    return passed, failed


def tier_a_checks(q: m.LocationQuality, thresholds: dict, k: Knobs) -> tuple[list[str], list[str]]:
    a = thresholds["A"]
    q_sta, q_val = k.tier_a_station_quantile_label, k.tier_a_quantile_label
    return _split(
        [
            _ge_check("nStations", q.nStations, a["minStations"], q_sta),
            _le_check("rmsS", q.rmsS, a["maxRmsS"], q_val),
            _le_check("hErrM", q.hErrM, a["maxHErrM"], q_val),
            _le_check("vErrM", q.vErrM, a["maxVErrM"], q_val),
            _le_check("gapDeg", q.gapDeg, a["maxGapDeg"], q_val),
            (not q.depthOnEdge, "depthOnEdge false", "depthOnEdge true"),
        ]
    )


def tier_b_checks(q: m.LocationQuality, thresholds: dict, k: Knobs) -> tuple[list[str], list[str]]:
    b = thresholds["B"]
    return _split(
        [
            _ge_check("nStations", q.nStations, b["minStations"], "Tier B floor"),
            _le_check("rmsS", q.rmsS, b["maxRmsS"], k.tier_b_quantile_label),
            _le_check("hErrM", q.hErrM, b["maxHErrM"], k.tier_b_quantile_label),
        ]
    )


def tier_of(q: m.LocationQuality, thresholds: dict, k: Knobs) -> tuple[str, list[str]]:
    """Derive (tier, tierReasons) from a quality block and the recorded thresholds."""
    a_pass, a_fail = tier_a_checks(q, thresholds, k)
    if not a_fail:
        return "A", a_pass
    b_pass, b_fail = tier_b_checks(q, thresholds, k)
    if not b_fail:
        return "B", [f"not A: {r}" for r in a_fail] + b_pass
    return "C", [f"not A: {r}" for r in a_fail] + [f"not B: {r}" for r in b_fail]


def tiering_config(k: Knobs) -> dict:
    a_lbl, b_lbl, s_lbl = (
        k.tier_a_quantile_label,
        k.tier_b_quantile_label,
        k.tier_a_station_quantile_label,
    )
    return {
        "method": "thresholds from quantiles of matched public events (synthetic values)",
        "thresholds": {"A": dict(k.tier_a), "B": dict(k.tier_b)},
        "quantiles": {
            "nStations": {s_lbl: k.tier_a["minStations"]},
            "rmsS": {a_lbl: k.tier_a["maxRmsS"], b_lbl: k.tier_b["maxRmsS"]},
            "hErrM": {a_lbl: k.tier_a["maxHErrM"], b_lbl: k.tier_b["maxHErrM"]},
            "vErrM": {a_lbl: k.tier_a["maxVErrM"]},
            "gapDeg": {a_lbl: k.tier_a["maxGapDeg"]},
        },
        "tierA": "every A threshold holds, vErrM present, depthOnEdge false",
        "tierB": "every B threshold holds",
        "tierC": "everything else",
    }


# ------------------------------------------------------------------------------------------
# Stations
# ------------------------------------------------------------------------------------------


def make_stations(rng: np.random.Generator, frame: Frame, k: Knobs) -> list[StationDraft]:
    drafts: list[StationDraft] = []
    n_ring = k.n_surface_stations
    unused = set(rng.choice(n_ring, size=k.n_unused_surface_stations, replace=False).tolist())
    for i in range(n_ring):
        jitter = rng.uniform(-k.ring_angle_jitter_deg, k.ring_angle_jitter_deg)
        angle = math.radians(360.0 * i / n_ring + jitter)
        radius = k.ring_radius_m + rng.uniform(-k.ring_radius_jitter_m, k.ring_radius_jitter_m)
        e, n = radius * math.sin(angle), radius * math.cos(angle)
        elev = float(
            np.clip(frame.surface_elev_m(e, n), k.surface_elev_min_m, k.surface_elev_max_m)
        )
        drafts.append(
            StationDraft(
                id=f"{k.network}.{k.station_prefix}{i + 1:02d}",
                e=e,
                n=n,
                surface_elev_m=frame.elev(elev),
                sensor_depth_m=0.0,
                kind="surface",
                used=i not in unused,
            )
        )
    for j, ((e, n), depth) in enumerate(zip(k.borehole_en_m, k.borehole_depth_m, strict=True)):
        drafts.append(
            StationDraft(
                id=f"{k.network}.{k.station_prefix}{n_ring + j + 1:02d}",
                e=e,
                n=n,
                surface_elev_m=frame.elev(frame.surface_elev_m(e, n)),
                sensor_depth_m=rnd(depth, k.dec_m),
                kind="borehole",
                used=True,
            )
        )
    return drafts


def station_models(
    rng: np.random.Generator, drafts: list[StationDraft], frame: Frame, k: Knobs
) -> list[m.Station]:
    out: list[m.Station] = []
    for st in drafts:
        borehole = st.kind == "borehole"
        statics: dict[str, float] = {}
        if st.used and rng.random() < k.statics_fraction:
            statics["P"] = rnd(rng.uniform(-k.statics_max_s, k.statics_max_s), k.dec_s)
            if not borehole:
                statics["S"] = rnd(rng.uniform(-k.statics_max_s, k.statics_max_s), k.dec_s)
        net, sta = st.id.split(".")
        out.append(
            m.Station(
                id=st.id,
                network=net,
                station=sta,
                latitude=frame.latitude(st.n),
                longitude=frame.longitude(st.e),
                surfaceElevM=st.surface_elev_m,
                sensorDepthM=st.sensor_depth_m,
                sensorElevM=rnd(st.sensor_elev_m, k.dec_m),
                kind="borehole" if borehole else "surface",
                channels=list(k.borehole_channels if borehole else k.surface_channels),
                sampleRateHz=k.borehole_sample_rate_hz if borehole else k.surface_sample_rate_hz,
                enu=frame.enu(st.e, st.n, st.sensor_elev_m),
                preprocessProfile=k.borehole_profile if borehole else k.surface_profile,
                usedInRun=st.used,
                staticsS=statics,
            )
        )
    return out


# ------------------------------------------------------------------------------------------
# Events
# ------------------------------------------------------------------------------------------


def rotated_offset(
    rng: np.random.Generator, sigma: tuple[float, float, float], strike_deg: float, dip_deg: float
) -> tuple[float, float, float]:
    """Gaussian offset in a (strike, normal, dip) frame, returned as (dE, dN, dUp) meters."""
    a, b, c = rng.standard_normal(3) * np.asarray(sigma)
    dip = math.radians(dip_deg)
    # Before the strike rotation the strike axis points east and the plane dips toward north.
    x = a
    y = c * math.cos(dip) + b * math.sin(dip)
    up = -c * math.sin(dip) + b * math.cos(dip)
    phi = math.radians(90.0 - strike_deg)  # rotate +E onto the strike azimuth (CW from N)
    de = x * math.cos(phi) - y * math.sin(phi)
    dn = x * math.sin(phi) + y * math.cos(phi)
    return de, dn, up


def sample_position(
    rng: np.random.Generator, tier: str, depth_on_edge: bool, k: Knobs
) -> tuple[float, float, float]:
    """(e, n, depth below refSurface) for one event of the given tier."""
    ce, cn = k.cluster_center_en_m
    top, bottom = k.grid_depth_range_m
    if tier == "A":
        de, dn, up = rotated_offset(rng, k.a_sigma_m, k.a_strike_deg, k.a_dip_deg)
        depth = float(np.clip(k.cluster_center_depth_m - up, *k.a_depth_range_m))
        return ce + de, cn + dn, depth
    if tier == "B":
        de, dn, up = rotated_offset(rng, k.b_sigma_m, k.a_strike_deg, k.a_dip_deg)
        depth = float(np.clip(k.cluster_center_depth_m - up, *k.b_depth_range_m))
        if depth_on_edge:
            depth = top + rng.uniform(0.0, k.edge_band_m)
        return ce + de, cn + dn, depth
    e = rng.uniform(-k.c_half_extent_m, k.c_half_extent_m)
    n = rng.uniform(-k.c_half_extent_m, k.c_half_extent_m)
    if not depth_on_edge:
        depth = rng.uniform(top + k.edge_band_m, bottom - k.edge_band_m)
    elif rng.random() < k.edge_top_frac:
        depth = top + rng.uniform(0.0, k.edge_band_m)
    else:
        depth = bottom - rng.uniform(0.0, k.edge_band_m)
    return e, n, depth


def sample_times(
    rng: np.random.Generator, n: int, tier: str, t_start: float, t_end: float, k: Knobs
) -> np.ndarray:
    """Origin times: a mixture of swarms and uniform background, all inside [t_start, t_end)."""
    s_per_h = 3600.0
    centers = np.asarray(k.swarm_center_h) * s_per_h
    sigmas = np.asarray(k.swarm_sigma_h) * s_per_h
    weights = np.asarray(k.swarm_weight) / np.sum(k.swarm_weight)
    span = t_end - t_start
    out: list[float] = []
    while len(out) < n:
        draw = 2 * (n - len(out))
        in_swarm = rng.random(draw) < k.swarm_fraction[tier]
        which = rng.choice(len(centers), size=draw, p=weights)
        swarm_t = centers[which] + sigmas[which] * rng.standard_normal(draw)
        background_t = rng.uniform(0.0, span, size=draw)
        rel = np.where(in_swarm, swarm_t, background_t)
        out.extend(float(x) for x in rel[(rel >= 0.0) & (rel < span)])
    return t_start + np.asarray(out[:n])


def sample_magnitude(rng: np.random.Generator, k: Knobs) -> m.Magnitude | None:
    if rng.random() < k.mag_missing_frac:
        return None
    value = k.mag_min - math.log10(rng.uniform(np.finfo(float).tiny, 1.0)) / k.mag_b_value
    return m.Magnitude(
        value=rnd(min(value, k.mag_max), k.dec_mag),
        type=k.mag_type,
        sigma=rnd(rng.uniform(*k.mag_sigma_range), k.dec_mag),
    )


def make_picks(
    rng: np.random.Generator, ev: EventDraft, stations: list[StationDraft], k: Knobs
) -> list[PickDraft]:
    """P picks on the n_stations nearest used stations; S picks on the n_s nearest of those."""
    used = [s for s in stations if s.used]
    order = sorted(used, key=lambda s: (ev.hypo_dist_m(s), s.id))[: ev.n_stations]
    lo, hi = k.pick_prob_range[ev.tier]
    picks: list[PickDraft] = []
    for i, st in enumerate(order):
        r_km = ev.hypo_dist_m(st) / 1000.0
        for phase, v in (("P", k.vp_km_s), ("S", k.vs_km_s)):
            if phase == "S" and i >= ev.n_s:
                continue
            t_pick = rnd(ev.t + r_km / v + rng.normal(0.0, k.pick_sigma_s[phase]), k.dec_s)
            pick_id = f"phasenet:{k.picker_weights}:{st.id}:{phase}:{t_pick:.3f}"
            picks.append(
                PickDraft(pick_id, st, phase, t_pick, rnd(rng.uniform(lo, hi), k.dec_prob))
            )
    return picks


def quality_of(ev: EventDraft, k: Knobs) -> m.LocationQuality:
    return m.LocationQuality(
        method=k.location_method,
        statics=k.statics_applied[ev.tier],
        nStations=ev.n_stations,
        nP=ev.n_p,
        nS=ev.n_s,
        rmsS=ev.rms_s,
        gapDeg=ev.gap_deg,
        minEpiDistM=ev.min_epi_dist_m,
        hErrM=ev.h_err_m,
        vErrM=ev.v_err_m,
        depthOnEdge=ev.depth_on_edge,
    )


def fix_better_tier(rng: np.random.Generator, ev: EventDraft, thresholds: dict, k: Knobs) -> None:
    """Sampling keeps each tier's own thresholds; make sure the *better* tier fails as well.

    A B-event that happens to pass every A check gets an rmsS above the A threshold; a C-event
    that passes every B check gets one above the B threshold.
    """
    derived, _ = tier_of(quality_of(ev, k), thresholds, k)
    if ev.tier == "B" and derived == "A":
        floor, ceiling = k.tier_a["maxRmsS"], k.rms_range_s["B"][1]
    elif ev.tier == "C" and derived != "C":
        floor, ceiling = k.tier_b["maxRmsS"], k.rms_range_s["C"][1]
    else:
        return
    step = 10.0**-k.dec_s
    ev.rms_s = rnd(rng.uniform(floor + step, ceiling), k.dec_s)
    if ev.rms_s <= floor:
        ev.rms_s = rnd(floor + step, k.dec_s)


def sample_event(
    rng: np.random.Generator,
    tier: str,
    t: float,
    stations: list[StationDraft],
    frame: Frame,
    thresholds: dict,
    k: Knobs,
) -> EventDraft | None:
    """One event draft of the given tier, or None when the draw landed in another tier."""
    depth_on_edge = bool(rng.random() < k.depth_on_edge_frac[tier])
    e, n, depth = sample_position(rng, tier, depth_on_edge, k)
    lo_sta, hi_sta = k.n_stations_range[tier]
    n_sta = int(rng.integers(lo_sta, hi_sta + 1))
    n_s = max(0, n_sta - int(rng.integers(0, k.s_dropout_max[tier] + 1)))
    missing_v = rng.random() < k.v_err_missing_frac[tier]
    ev = EventDraft(
        tier=tier,
        e=e,
        n=n,
        elev_m=frame.elev(frame.ref_surface_elev_m - depth),
        t=rnd(t, k.dec_s),
        n_stations=n_sta,
        n_p=n_sta,
        n_s=n_s,
        rms_s=rnd(rng.uniform(*k.rms_range_s[tier]), k.dec_s),
        h_err_m=rnd(rng.uniform(*k.h_err_range_m[tier]), k.dec_m),
        v_err_m=None if missing_v else rnd(rng.uniform(*k.v_err_range_m[tier]), k.dec_m),
        depth_on_edge=depth_on_edge,
        magnitude=sample_magnitude(rng, k),
    )
    ev.picks = make_picks(rng, ev, stations, k)
    used = list({p.station.id: p.station for p in ev.picks}.values())
    az = np.asarray([math.degrees(math.atan2(s.e - ev.e, s.n - ev.n)) for s in used])
    ev.gap_deg = rnd(azimuthal_gap_deg(az), k.dec_m)
    ev.min_epi_dist_m = rnd(min(ev.epi_dist_m(s) for s in used), k.dec_m)
    ev.mean_pick_prob = rnd(float(np.mean([p.prob for p in ev.picks])), k.dec_prob)
    fix_better_tier(rng, ev, thresholds, k)
    derived, reasons = tier_of(quality_of(ev, k), thresholds, k)
    if derived != tier:
        return None
    ev.tier_reasons = reasons
    return ev


def make_events(
    rng: np.random.Generator, stations: list[StationDraft], frame: Frame, run: RunSection, k: Knobs
) -> list[EventDraft]:
    thresholds = tiering_config(k)["thresholds"]
    drafts: list[EventDraft] = []
    for tier in TIERS:
        times = sample_times(rng, k.n_events[tier], tier, run.window_start_s, run.window_end_s, k)
        for t in times:
            for _ in range(k.max_sample_attempts):
                ev = sample_event(rng, tier, float(t), stations, frame, thresholds, k)
                if ev is not None:
                    drafts.append(ev)
                    break
            else:
                raise RuntimeError(
                    f"could not draw a Tier {tier} event in {k.max_sample_attempts} attempts; "
                    "the quality ranges in Knobs no longer fit the tier thresholds"
                )
    drafts.sort(key=lambda d: (d.t, d.tier, d.e))
    return drafts


def assign_ids_and_reveal(drafts: list[EventDraft], run_id: str) -> None:
    """Ids in time order (like the pipeline); revealOrder Tier A -> B -> C, time-ordered within."""
    for i, ev in enumerate(drafts):
        ev.id = f"hq-{run_id}-{i + 1:06d}"
    rank = {tier: i for i, tier in enumerate(TIERS)}
    for order, ev in enumerate(sorted(drafts, key=lambda d: (rank[d.tier], d.t, d.id))):
        ev.reveal_order = order


def choose_hero(drafts: list[EventDraft]) -> EventDraft:
    """The Tier A event with the most stations (ties: lowest rmsS, then earliest id)."""
    tier_a = [d for d in drafts if d.tier == "A"]
    return min(tier_a, key=lambda d: (-d.n_stations, d.rms_s, d.id))


def event_models(
    drafts: list[EventDraft], frame: Frame, run_id: str, k: Knobs
) -> list[m.SeismicEvent]:
    return [
        m.SeismicEvent(
            id=ev.id,
            runId=run_id,
            source="hq-pipeline",
            t=ev.t,
            latitude=frame.latitude(ev.n),
            longitude=frame.longitude(ev.e),
            elevM=ev.elev_m,
            depthKm=frame.depth_km(ev.elev_m),
            enu=frame.enu(ev.e, ev.n, ev.elev_m),
            quality=quality_of(ev, k),
            tier=ev.tier,
            tierReasons=ev.tier_reasons,
            meanPickProb=ev.mean_pick_prob,
            magnitude=ev.magnitude,
            catalogMatch=ev.catalog_match,
            revealOrder=ev.reveal_order,
            pickIds=[p.id for p in ev.picks],
        )
        for ev in drafts
    ]


# ------------------------------------------------------------------------------------------
# Public catalog and matching
# ------------------------------------------------------------------------------------------


@dataclass
class CatalogDraft:
    t: float
    e: float
    n: float
    depth_km: float  # as published, relative to the catalog datum (sea level)
    mag: float
    event: EventDraft | None  # the matched candidate, if any


def _magnitude_key(d: EventDraft) -> tuple[float, str]:
    return (d.magnitude.value if d.magnitude else -math.inf, d.id)


def make_catalog(
    rng: np.random.Generator, drafts: list[EventDraft], frame: Frame, run: RunSection, k: Knobs
) -> list[m.CatalogEvent]:
    """Public rows: the largest A/B (and a couple of C) candidates get a match; the rest don't."""
    rows: list[CatalogDraft] = []
    for tier in TIERS:
        pool = sorted((d for d in drafts if d.tier == tier), key=_magnitude_key, reverse=True)
        for ev in pool[: k.n_matched[tier]]:
            dt = float(
                np.clip(rng.normal(0.0, k.match_dt_sigma_s), -k.match_dt_max_s, k.match_dt_max_s)
            )
            angle = rng.uniform(0.0, 2.0 * math.pi)
            dist = min(abs(rng.normal(0.0, k.match_dist_sigma_m)), k.match_dist_max_m)
            depth_noise = rng.normal(0.0, k.catalog_depth_sigma_km)
            if ev.magnitude is not None:
                mag = ev.magnitude.value + rng.normal(k.catalog_mag_bias, k.catalog_mag_sigma)
            else:
                mag = rng.uniform(*k.unmatched_mag_range)
            rows.append(
                CatalogDraft(
                    t=rnd(ev.t + dt, k.dec_s),
                    e=ev.e + dist * math.sin(angle),
                    n=ev.n + dist * math.cos(angle),
                    depth_km=rnd(-ev.elev_m / 1000.0 + depth_noise, k.catalog_depth_decimals),
                    mag=mag,
                    event=ev,
                )
            )
    n_unmatched = k.n_public - sum(k.n_matched.values())
    for i in range(n_unmatched):
        near = i < k.n_unmatched_near
        radius = rng.uniform(*(k.unmatched_near_radius_m if near else k.unmatched_far_radius_m))
        angle = rng.uniform(0.0, 2.0 * math.pi)
        depth_below = rng.uniform(*k.unmatched_depth_range_m)
        rows.append(
            CatalogDraft(
                t=rnd(rng.uniform(run.window_start_s, run.window_end_s), k.dec_s),
                e=radius * math.sin(angle),
                n=radius * math.cos(angle),
                depth_km=rnd(
                    (depth_below - frame.ref_surface_elev_m) / 1000.0, k.catalog_depth_decimals
                ),
                mag=rng.uniform(*k.unmatched_mag_range),
                event=None,
            )
        )
    rows.sort(key=lambda r: (r.t, r.e))
    missing = set(rng.choice(len(rows), size=k.catalog_mag_missing, replace=False).tolist())
    out: list[m.CatalogEvent] = []
    for i, row in enumerate(rows):
        cat_id = f"{k.catalog_id_prefix}{i + 1:03d}"
        elev_m = frame.elev(-row.depth_km * 1000.0)
        if row.event is not None:
            row.event.catalog_match = m.CatalogMatch(
                catalogId=cat_id,
                dtS=rnd(row.t - row.event.t, k.dec_s),
                distM=rnd(math.hypot(row.e - row.event.e, row.n - row.event.n), k.dec_m),
            )
        has_mag = i not in missing
        alt_type = rng.random() < k.catalog_mag_alt_frac
        mag_type = k.catalog_mag_alt_type if alt_type else k.catalog_mag_type
        out.append(
            m.CatalogEvent(
                id=cat_id,
                source=k.catalog_source,
                t=row.t,
                latitude=frame.latitude(row.n),
                longitude=frame.longitude(row.e),
                depthKm=row.depth_km,
                depthDatum=k.catalog_depth_datum,
                elevM=elev_m,
                mag=rnd(row.mag, k.dec_mag) if has_mag else None,
                magType=mag_type if has_mag else None,
                enu=frame.enu(row.e, row.n, elev_m),
                matchedEventId=row.event.id if row.event is not None else None,
            )
        )
    return out


# ------------------------------------------------------------------------------------------
# Features
# ------------------------------------------------------------------------------------------


def make_features(frame: Frame, k: Knobs) -> list[m.GeoFeature]:
    def unverified(what: str) -> m.SourceRef:
        return m.SourceRef(
            citation=f"synthetic {what} (mock fixture)", url=DOCS_URL, verified=False
        )

    he, hn = k.well_head_en_m
    head_elev = frame.surface_elev_m(he, hn)
    de, dn = k.well_deviation_m
    well: list[m.Enu] = []
    for i in range(k.well_points):
        f = i / (k.well_points - 1)
        # Vertical near the head, drifting toward the toe: quadratic horizontal deviation.
        elev = head_elev + (k.well_bottom_elev_m - head_elev) * f
        well.append(frame.enu(he + de * f * f, hn + dn * f * f, elev))

    fe, fn = k.facility_en_m
    be, bn = k.boundary_center_en_m
    boundary: list[m.Enu] = []
    for i in range(k.boundary_points + 1):  # +1 closes the ring
        angle = 2.0 * math.pi * (i % k.boundary_points) / k.boundary_points
        e, n = (
            be + k.boundary_radius_m * math.sin(angle),
            bn + k.boundary_radius_m * math.cos(angle),
        )
        boundary.append(frame.enu(e, n, frame.surface_elev_m(e, n)))

    return [
        m.GeoFeature(
            id="well-synthetic-a",
            kind="well",
            name=k.well_name,
            path=well,
            source=unverified("well path"),
        ),
        m.GeoFeature(
            id="facility-synthetic-pad",
            kind="facility",
            name="Synthetic pad",
            path=[frame.enu(fe, fn, frame.surface_elev_m(fe, fn))],
            source=unverified("facility point"),
        ),
        m.GeoFeature(
            id="boundary-synthetic-lease",
            kind="boundary",
            name="Synthetic lease outline",
            path=boundary,
            source=unverified("outline"),
        ),
    ]


# ------------------------------------------------------------------------------------------
# Evidence
# ------------------------------------------------------------------------------------------


def choose_evidence(
    rng: np.random.Generator, drafts: list[EventDraft], hero: EventDraft, k: Knobs
) -> list[EventDraft]:
    chosen: list[EventDraft] = [hero]
    for tier in TIERS:
        pool = [d for d in drafts if d.tier == tier and d is not hero]
        want = k.n_evidence[tier] - (1 if tier == hero.tier else 0)
        idx = rng.choice(len(pool), size=want, replace=False)
        chosen.extend(pool[int(i)] for i in sorted(idx.tolist()))
    return chosen


def synth_trace(
    rng: np.random.Generator, t: np.ndarray, t_p: float, t_s: float, dist_m: float, k: Knobs
) -> list[float]:
    """Smoothed noise + a P Ricker + a bigger S Ricker with a decaying coda, scaled to [-1, 1]."""
    n = len(t)
    kernel = np.ones(k.noise_smooth_samples) / k.noise_smooth_samples
    x = rng.uniform(*k.noise_amp_range) * np.convolve(rng.standard_normal(n), kernel, mode="same")
    decay = min(1.0, (k.amp_ref_dist_m / max(dist_m, 1.0)) ** k.amp_decay)
    x += k.p_amp * decay * ricker(t - t_p, k.p_wavelet_hz)
    x += k.s_amp * decay * ricker(t - t_s, k.s_wavelet_hz)
    coda = np.where(t > t_s, np.exp(-(t - t_s) / k.coda_decay_s), 0.0)
    x += (
        k.coda_amp
        * k.s_amp
        * decay
        * coda
        * np.convolve(rng.standard_normal(n), kernel, mode="same")
    )
    x /= np.max(np.abs(x))
    return [rnd(v, k.dec_sample) for v in x.tolist()]


def make_evidence(rng: np.random.Generator, ev: EventDraft, k: Knobs) -> m.EventEvidence:
    """Nearest traces sorted by epiDistM; each trace's window opens just before its predicted P."""
    by_station: dict[str, dict[str, PickDraft]] = {}
    for p in ev.picks:
        by_station.setdefault(p.station.id, {})[p.phase] = p
    stations = sorted(
        (picks["P"].station for picks in by_station.values()),
        key=lambda s: (ev.epi_dist_m(s), s.id),
    )[: k.evidence_max_traces]
    rate = 1.0 / k.evidence_dt_s
    min_n, max_n = int(k.evidence_min_s * rate), int(k.evidence_max_s * rate)
    s_minus_p_per_km = 1.0 / k.vs_km_s - 1.0 / k.vp_km_s
    # Drop the farthest trace until every S (plus a margin) fits inside the per-file sample budget.
    while True:
        n_samples = min(max_n, max(min_n, k.evidence_sample_budget // len(stations)))
        needed_s = max(ev.hypo_dist_m(s) / 1000.0 * s_minus_p_per_km for s in stations)
        needed_s += k.evidence_pre_p_s + k.evidence_post_s_s
        if needed_s * rate <= n_samples or len(stations) == 1:
            break
        stations = stations[:-1]
    traces: list[m.WaveformSnippet] = []
    for st in stations:
        picks = by_station[st.id]
        r_km = ev.hypo_dist_m(st) / 1000.0
        pred_p, pred_s = ev.t + r_km / k.vp_km_s, ev.t + r_km / k.vs_km_s
        t0 = rnd(pred_p - k.evidence_pre_p_s, k.dec_s)
        t = t0 + k.evidence_dt_s * np.arange(n_samples)
        p_pick, s_pick = picks["P"], picks.get("S")
        t_s = s_pick.t if s_pick is not None else pred_s
        borehole = st.kind == "borehole"
        traces.append(
            m.WaveformSnippet(
                stationId=st.id,
                channel=k.borehole_vertical_channel if borehole else k.surface_vertical_channel,
                epiDistM=rnd(ev.epi_dist_m(st), k.dec_m),
                t0=t0,
                dt=k.evidence_dt_s,
                samples=synth_trace(rng, t, p_pick.t, t_s, ev.hypo_dist_m(st), k),
                pickP=p_pick.t,
                pickS=s_pick.t if s_pick is not None else None,
                probP=p_pick.prob,
                probS=s_pick.prob if s_pick is not None else None,
                predP=rnd(pred_p, k.dec_s),
                predS=rnd(pred_s, k.dec_s),
            )
        )
    return m.EventEvidence(eventId=ev.id, filterHz=k.filter_hz, traces=traces)


# ------------------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------------------


def tier_counts(events: list[m.SeismicEvent]) -> m.TierCounts:
    return m.TierCounts(**{tier: sum(1 for e in events if e.tier == tier) for tier in TIERS})


def median_rms_s(events: list[m.SeismicEvent], k: Knobs) -> float:
    return rnd(float(np.median([e.quality.rmsS for e in events])), k.dec_s)


def median_stations(events: list[m.SeismicEvent]) -> float:
    return float(np.median([e.quality.nStations for e in events]))


def baseline_rows(events: list[m.SeismicEvent], recovered: int, k: Knobs) -> list[m.BaselineRow]:
    tiers = tier_counts(events)
    rms, sta = median_rms_s(events, k), median_stations(events)
    rows = [
        m.BaselineRow(
            method="phasenet",
            associationProfile="full",
            candidates=len(events),
            recoveredPublic=recovered,
            tiers=tiers,
            medianRmsS=rms,
            medianStations=sta,
        )
    ]
    for key, (f_cand, f_rec, f_a, f_b, f_c, f_rms, f_sta) in k.baseline_factors.items():
        method, profile = key.split("/")
        rows.append(
            m.BaselineRow(
                method=method,
                associationProfile=profile,
                candidates=round(len(events) * f_cand),
                recoveredPublic=round(recovered * f_rec),
                tiers=m.TierCounts(
                    A=round(tiers.A * f_a), B=round(tiers.B * f_b), C=round(tiers.C * f_c)
                ),
                medianRmsS=rnd(rms * f_rms, k.dec_s),
                medianStations=rnd(sta * f_sta, 1),
            )
        )
    return rows


def sweep_points(events: list[m.SeismicEvent], recovered: int, k: Knobs) -> list[m.SweepPoint]:
    n_a = sum(1 for e in events if e.tier == "A")
    combos = [(n, k.sweep_chosen_tolerance_s) for n in k.sweep_n_p_and_s_min]
    combos += [
        (k.sweep_chosen_n_p_and_s_min, tol)
        for tol in k.sweep_tolerances_s
        if tol != k.sweep_chosen_tolerance_s
    ]
    points: list[m.SweepPoint] = []
    for n_min, tol in combos:
        f_cand, f_rec, f_a = (
            x * y
            for x, y in zip(k.sweep_factors[n_min], k.sweep_tolerance_factors[tol], strict=True)
        )
        points.append(
            m.SweepPoint(
                params={"nPAndSMin": n_min, "pickMatchToleranceS": tol},
                candidates=round(len(events) * f_cand),
                recoveredPublic=min(recovered, round(recovered * f_rec)),
                tierA=round(n_a * f_a),
            )
        )
    return points


def gr_curve(public_mags: np.ndarray, recovered_mags: np.ndarray, k: Knobs) -> m.GRCurve:
    """Cumulative counts per bin from the actual magnitudes; Mc by maximum curvature + offset;
    Aki-Utsu b (with the half-bin correction) and Shi-Bolt sigma on the recovered set."""
    bins = np.round(np.arange(k.gr_min_mag, k.gr_max_mag + k.gr_bin / 2.0, k.gr_bin), k.dec_mag)
    eps = 10.0 ** -(k.dec_mag + 1)

    def cumulative(mags: np.ndarray) -> list[int]:
        return [int(np.sum(mags >= b - eps)) for b in bins]

    def mc(mags: np.ndarray) -> float:
        counts = [np.sum((mags >= b - eps) & (mags < b + k.gr_bin - eps)) for b in bins]
        return rnd(bins[int(np.argmax(counts))] + k.gr_mc_offset, k.dec_mag)

    mc_rec = mc(recovered_mags)
    above = recovered_mags[recovered_mags >= mc_rec - eps]
    n = len(above)
    b = math.log10(math.e) / (float(np.mean(above)) - (mc_rec - k.gr_bin / 2.0))
    spread = float(np.sum((above - np.mean(above)) ** 2)) / (n * (n - 1))
    b_sigma = k.shi_bolt_factor * b * b * math.sqrt(spread)
    return m.GRCurve(
        magBins=[float(b_) for b_ in bins],
        publicCum=cumulative(public_mags),
        recoveredCum=cumulative(recovered_mags),
        mcPublic=mc(public_mags),
        mcRecovered=mc_rec,
        bValue=rnd(b, k.dec_s),
        bSigma=rnd(b_sigma, k.dec_s),
    )


def make_validation(
    events: list[m.SeismicEvent], catalog: list[m.CatalogEvent], k: Knobs
) -> m.Validation:
    recovered = sum(1 for c in catalog if c.matchedEventId is not None)
    public_mags = np.asarray([c.mag for c in catalog if c.mag is not None])
    recovered_mags = np.asarray([e.magnitude.value for e in events if e.magnitude is not None])
    by_id = {e.id: e for e in events}
    pairs: list[tuple[float, float]] = []
    for c in catalog:
        ours = by_id[c.matchedEventId].magnitude if c.matchedEventId is not None else None
        if ours is not None and c.mag is not None:
            pairs.append((ours.value, c.mag))
    loo_mae = float(np.mean([abs(ours - theirs) for ours, theirs in pairs]))
    return m.Validation(
        baseline=baseline_rows(events, recovered, k),
        sweep=sweep_points(events, recovered, k),
        nullTest=m.NullTest(
            nShuffles=k.null_shuffles,
            shiftRangeS=k.null_shift_range_s,
            meanChanceEvents=k.null_mean_events,
            meanChanceStrict=k.null_mean_strict,
            stdChanceEvents=k.null_std_events,
        ),
        gr=gr_curve(public_mags, recovered_mags, k),
        magnitude=m.MagCalibration(
            n=len(pairs), looMae=rnd(loo_mae, k.dec_s), coefficients=dict(k.calib_coefficients)
        ),
        synthetic=m.SyntheticTest(
            nEvents=k.synthetic_n_events,
            pickSigmaS=dict(k.pick_sigma_s),
            medianHErrM=k.synthetic_median_h_err_m,
            medianVErrM=k.synthetic_median_v_err_m,
            p90VErrM=k.synthetic_p90_v_err_m,
            medianDepthBiasM=k.synthetic_median_depth_bias_m,
        ),
    )


# ------------------------------------------------------------------------------------------
# Meta: run, scene, summary
# ------------------------------------------------------------------------------------------


def make_summary(
    run_id: str,
    events: list[m.SeismicEvent],
    catalog: list[m.CatalogEvent],
    validation: m.Validation,
    k: Knobs,
) -> m.AnalysisSummary:
    """Counts recomputed from the event and catalog lists (the test recomputes them again)."""
    public = len(catalog)
    recovered = sum(1 for c in catalog if c.matchedEventId is not None)
    additional = [e for e in events if e.catalogMatch is None]
    rows = {(r.method, r.associationProfile): r for r in validation.baseline}
    gains = {
        profile: rows[("phasenet", profile)].tiers.A / rows[("stalta", profile)].tiers.A
        for profile in ("full", "p_only")
    }
    baseline: m.BaselineGain | None = None
    if all(g > 1.0 for g in gains.values()):  # only claim a gain when it holds in both profiles
        baseline = m.BaselineGain(
            associationProfile="full",
            strictPhasenet=rows[("phasenet", "full")].tiers.A,
            strictStalta=rows[("stalta", "full")].tiers.A,
            gain=rnd(gains["full"], k.dec_mag),
        )
    return m.AnalysisSummary(
        runId=run_id,
        publicCatalogCount=public,
        recoveredCatalogCount=recovered,
        recall=rnd(recovered / public, k.dec_recall),
        unmatchedPublicIds=[c.id for c in catalog if c.matchedEventId is None],
        candidateCount=len(events),
        additionalCount=len(additional),
        additional=tier_counts(additional),
        strictQualityCount=sum(1 for e in events if e.tier == "A"),
        strictAdditionalCount=sum(1 for e in additional if e.tier == "A"),
        medianStations=median_stations(events),
        medianRmsS=median_rms_s(events, k),
        baseline=baseline,
    )


def make_run(run_id: str, run: RunSection, stations: list[m.Station], k: Knobs) -> m.ProcessingRun:
    return m.ProcessingRun(
        id=run_id,
        mode="mock",
        createdAt=k.created_at,
        gitSha=k.git_sha,
        windowStart=run.window_start_s,
        windowEnd=run.window_end_s,
        windowLabel=window_label(run.windowStart, run.windowEnd),
        bbox=run.bbox,
        stationIds=[s.id for s in stations if s.usedInRun],
        pickerModel=k.picker_model,
        pickerWeights=k.picker_weights,
        softwareVersions={
            "mock-fixture": "synthetic",
            "hq_contracts": m.SCHEMA_VERSION,
            "numpy": np.__version__,
            "pydantic": pydantic.VERSION,
        },
        runtimeS=dict(k.runtime_s),
        picker={"model": k.picker_model, "weights": k.picker_weights, **k.picker_params},
        associator=dict(k.associator_params),
        velocityModel={
            "name": "mock-1d-halfspace",
            "vpKmS": k.vp_km_s,
            "vsKmS": k.vs_km_s,
            "datum": "site surface",
            "source": {
                "citation": "synthetic velocity model (mock fixture)",
                "url": DOCS_URL,
                "verified": False,
            },
        },
        locator=dict(k.locator_params),
        tiering=tiering_config(k),
        matching=dict(k.matching_params),
        isSynthetic=True,
    )


def make_scene(run_id: str, run: RunSection, hero_id: str) -> m.SceneMeta:
    return m.SceneMeta(
        runId=run_id,
        originLat=run.origin.lat,
        originLon=run.origin.lon,
        originElevM=run.origin.elevM,
        refSurfaceElevM=run.refSurfaceElevM,
        projection="EPSG:32612 minus origin",
        verticalExaggeration=1.0,
        depthLabel=f"Depth below site surface (ref {run.refSurfaceElevM:g} m ASL)",
        heroEventId=hero_id,
        isSynthetic=True,
    )


# ------------------------------------------------------------------------------------------
# Bundle assembly and output
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MockBundle:
    meta: m.BundleMeta
    stations: list[m.Station]
    catalog: list[m.CatalogEvent]
    events: list[m.SeismicEvent]
    features: list[m.GeoFeature]
    validation: m.Validation
    evidence: list[m.EventEvidence]


def load_run_section(path: Path = RUN_YAML) -> RunSection:
    return RunSection.model_validate(yaml.safe_load(path.read_text()))


def build_bundle(seed: int, run: RunSection, k: Knobs | None = None) -> MockBundle:
    k = k or Knobs()
    rng = np.random.default_rng(seed)
    run_id = f"{k.run_id_prefix}-{seed}"
    frame = Frame(run.origin.lat, run.origin.lon, run.origin.elevM, run.refSurfaceElevM, k)

    station_drafts = make_stations(rng, frame, k)
    stations = station_models(rng, station_drafts, frame, k)
    drafts = make_events(rng, station_drafts, frame, run, k)
    assign_ids_and_reveal(drafts, run_id)
    catalog = make_catalog(rng, drafts, frame, run, k)  # also fills each draft's catalog_match
    events = event_models(drafts, frame, run_id, k)
    hero = choose_hero(drafts)
    evidence = [make_evidence(rng, ev, k) for ev in choose_evidence(rng, drafts, hero, k)]
    validation = make_validation(events, catalog, k)
    meta = m.BundleMeta(
        mode="mock",
        scene=make_scene(run_id, run, hero.id),
        run=make_run(run_id, run, stations, k),
        summary=make_summary(run_id, events, catalog, validation, k),
    )
    return MockBundle(
        meta, stations, catalog, events, make_features(frame, k), validation, evidence
    )


def dump_json(path: Path, data: object, *, indent: int | None) -> int:
    """Write sorted-key JSON (pretty when ``indent``, compact otherwise); returns bytes written."""
    separators = None if indent else (",", ":")
    text = json.dumps(data, sort_keys=True, indent=indent, separators=separators) + "\n"
    path.write_text(text)
    return len(text.encode())


def write_bundle(bundle: MockBundle, out_dir: Path) -> dict[str, int]:
    """Write every bundle file; returns {relative path: bytes}. Stale evidence files are removed."""
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for stale in evidence_dir.glob("*.json"):
        stale.unlink()
    pretty = {
        "meta.json": bundle.meta.model_dump(mode="json"),
        "stations.json": [s.model_dump(mode="json") for s in bundle.stations],
        "catalog.json": [c.model_dump(mode="json") for c in bundle.catalog],
        "features.json": [f.model_dump(mode="json") for f in bundle.features],
        "validation.json": bundle.validation.model_dump(mode="json"),
    }
    sizes = {name: dump_json(out_dir / name, data, indent=2) for name, data in pretty.items()}
    events = [e.model_dump(mode="json") for e in bundle.events]
    sizes["events.json"] = dump_json(out_dir / "events.json", events, indent=None)
    for ev in bundle.evidence:
        rel = f"evidence/{ev.eventId}.json"
        sizes[rel] = dump_json(out_dir / rel, ev.model_dump(mode="json"), indent=None)
    return sizes


def generate(out_dir: Path, seed: int = DEFAULT_SEED, run_yaml: Path = RUN_YAML) -> MockBundle:
    """Build and write the mock bundle. Returns the in-memory bundle for callers and tests."""
    bundle = build_bundle(seed, load_run_section(run_yaml))
    sizes = write_bundle(bundle, out_dir)
    s = bundle.meta.summary
    tiers = tier_counts(bundle.events)
    evidence_bytes = [n for p, n in sizes.items() if p.startswith("evidence/")]
    print(f"mock bundle -> {out_dir} (seed {seed}, run {bundle.meta.run.id}, synthetic)")
    print(
        f"  stations {len(bundle.stations)}  catalog {s.publicCatalogCount}  "
        f"events {s.candidateCount} (A {tiers.A}, B {tiers.B}, C {tiers.C})  "
        f"recovered {s.recoveredCatalogCount}  hero {bundle.meta.scene.heroEventId}"
    )
    print(
        f"  evidence {len(evidence_bytes)} files, max {max(evidence_bytes) / 1024:.1f} KB; "
        f"events.json {sizes['events.json'] / 1024:.1f} KB; "
        f"total {sum(sizes.values()) / 1024:.1f} KB"
    )
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic mock bundle (FIX-01). Everything it writes is invented."
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help=f"output dir (default {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"RNG seed (default {DEFAULT_SEED})"
    )
    args = parser.parse_args(argv)
    generate(args.out.resolve(), args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

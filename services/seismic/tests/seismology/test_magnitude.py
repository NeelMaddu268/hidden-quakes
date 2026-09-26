"""MAG-01: magnitude amplitudes, calibration and the stage, on tiny synthetic data built here.

The stage tests build a made-up world in ``tmp_path``: five 100 Hz stations with StationXML and
MiniSEED in the H1 cache layout (one whose response describes 200 Hz, one with no waveform data),
twelve events with known magnitudes whose S-window Wood-Anderson amplitudes follow
``M = logA + b logR + c + s`` exactly, and the run tables stages ``tier`` and ``match`` leave.
Everything is seeded and offline.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from hq_contracts.io import read_table, to_frame, write_table
from hq_contracts.models import (
    CatalogEvent,
    CatalogMatch,
    Enu,
    LocationQuality,
    MagCalibration,
    SeismicEvent,
    Station,
)
from obspy import Stream, Trace, UTCDateTime
from obspy.core.inventory import Channel, Inventory, Network, Response
from obspy.core.inventory import Station as InvStation
from obspy.core.inventory.util import FloatWithUncertaintiesAndUnit, Frequency

import hq.magnitude as magnitude_pkg
from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig, WoodAndersonConfig
from hq.magnitude import (
    MagnitudeError,
    StationScreen,
    event_magnitudes,
    fit_calibration,
    leave_one_event_out,
    measure_amplitudes,
    plan_windows,
    screen_station,
)
from hq.magnitude.amplitude import (
    WINDOW_COLUMNS,
    nfft_for,
    read_groups,
    response_filter,
    to_wood_anderson,
    wood_anderson_response,
)
from hq.runs import resolve_stage, stage_spec

SR = 100.0
DAY = UTCDateTime("2026-09-10T00:00:00")
T_DATA = DAY.timestamp + 3600.0  # first sample of every synthetic trace
DATA_S = 900.0
SENSITIVITY = 1.0e9  # counts per m of ground displacement (a flat displacement response)
GAIN = {"T.A02": 4.0e9}  # one station with another gain: responses must not be shared
WAVELET_HZ = 5.0
TRUE_B, TRUE_C = 1.3, 0.4
TRUE_TERMS = {"T.A01": 0.15, "T.A02": -0.1, "T.A03": 0.05, "T.A04": -0.1}
STATION_ENU = {  # e, n (m); surface sensors at the run origin's elevation
    "T.A01": (3000.0, 1000.0),
    "T.A02": (-2500.0, 3500.0),
    "T.A03": (-4000.0, -3000.0),
    "T.A04": (5000.0, -6000.0),
    "T.BAD": (1000.0, 6000.0),  # StationXML says 200 Hz, the data is 100 Hz
    "T.NOD": (-6000.0, 500.0),  # StationXML only, no MiniSEED
}
N_EVENTS = 12
ML_EVENTS = tuple(range(9))  # matched, magType ml
MD_EVENTS = (9, 10)  # matched, magType md; event 11 stays unmatched


def _cfg(base: SeismologyConfig, **magnitude: Any) -> SeismologyConfig:
    raw = base.model_dump()
    fit = {**raw["magnitude"]["fit"], **magnitude.pop("fit", {})}
    raw["magnitude"] = {**raw["magnitude"], **magnitude, "fit": fit}
    return SeismologyConfig.model_validate(raw)


def _gain(sid: str) -> float:
    return GAIN.get(sid, SENSITIVITY)


def _flat_response(rate_hz: float | None, gain: float = SENSITIVITY) -> Response:
    resp = Response.from_paz(
        zeros=[], poles=[], stage_gain=gain, input_units="M", output_units="COUNTS"
    )
    if rate_hz is not None:
        stage = resp.response_stages[0]
        stage.decimation_input_sample_rate = Frequency(rate_hz)
        stage.decimation_factor = 1
        stage.decimation_offset = 0
        stage.decimation_delay = FloatWithUncertaintiesAndUnit(0.0)
        stage.decimation_correction = FloatWithUncertaintiesAndUnit(0.0)
    return resp


def _inventory(sid: str, response_rate: float | None) -> Inventory:
    net, sta = sid.split(".")
    channels = [
        Channel(
            code=cha, location_code="", latitude=38.5, longitude=-112.9, elevation=1650.0,
            depth=0.0, azimuth=az, dip=dip, sample_rate=SR, start_date=UTCDateTime(2020, 1, 1),
            response=_flat_response(response_rate, _gain(sid)),
        )
        for cha, az, dip in (("HHZ", 0.0, -90.0), ("HHN", 0.0, 0.0), ("HHE", 90.0, 0.0))
    ]
    station = InvStation(
        code=sta, latitude=38.5, longitude=-112.9, elevation=1650.0, channels=channels
    )
    return Inventory(networks=[Network(code=net, stations=[station])], source="test")


def _station_row(sid: str) -> Station:
    net, sta = sid.split(".")
    e, n = STATION_ENU[sid]
    return Station(
        id=sid, network=net, station=sta, location="", latitude=38.5, longitude=-112.9,
        surfaceElevM=1650.0, sensorDepthM=0.0, sensorElevM=1650.0, kind="surface",
        channels=["HHZ", "HHN", "HHE"], sampleRateHz=SR, enu=Enu(e=e, n=n, u=0.0),
        preprocessProfile="surface-100", usedInRun=True,
    )  # fmt: skip


def _wavelet(t: np.ndarray) -> np.ndarray:
    return np.exp(-((t / 0.25) ** 2)) * np.sin(2.0 * np.pi * WAVELET_HZ * t)


def _write_mseed(cache_dir: Path, sid: str, cha: str, data: np.ndarray) -> None:
    net, sta = sid.split(".")
    header = {"network": net, "station": sta, "location": "", "channel": cha,
              "sampling_rate": SR, "starttime": UTCDateTime(T_DATA)}  # fmt: skip
    Stream([Trace(data=np.round(data).astype(np.int32), header=header)]).write(
        str(cache_dir / "mseed" / f"{net}.{sta}..{cha}.20260910.mseed"), format="MSEED"
    )


class World:
    """The synthetic run: tables in ``run_dir``, waveforms and StationXML in ``cache_dir``."""

    def __init__(self, root: Path, run: RunSection, wa: WoodAndersonConfig) -> None:
        rng = np.random.default_rng(20260926)
        self.run_dir = root / "runs" / "test-run"
        self.cache_dir = root / "cache"
        self.run_dir.mkdir(parents=True)
        (self.cache_dir / "mseed").mkdir(parents=True)
        (self.cache_dir / "stationxml").mkdir(parents=True)
        self.mags = np.round(rng.permutation(np.linspace(0.6, 2.0, N_EVENTS)), 3)
        self.ids = [f"hq-test-run-{i:06d}" for i in range(N_EVENTS)]
        self.true_log_a: dict[tuple[str, str], float] = {}
        self.t = T_DATA + 60.0 + 60.0 * np.arange(N_EVENTS)
        self.enu = [(rng.uniform(-800, 800), rng.uniform(-800, 800), -4000.0 - rng.uniform(0, 800))
                    for _ in range(N_EVENTS)]  # fmt: skip
        wa_gain = float(np.abs(wood_anderson_response(np.array([WAVELET_HZ]), wa))[0])
        n_samples = int(DATA_S * SR)
        t_axis = T_DATA + np.arange(n_samples) / SR
        arrivals: list[dict[str, Any]] = []
        for sid, (e_s, n_s) in STATION_ENU.items():
            north = rng.normal(0.0, 1.0, n_samples)
            for i, eid in enumerate(self.ids):
                e, n, u = self.enu[i]
                r_m = math.sqrt((e - e_s) ** 2 + (n - n_s) ** 2 + u**2)
                t_p, t_s = self.t[i] + r_m / 6000.0, self.t[i] + r_m / 3500.0
                observed = sid in ("T.A01", "T.A03")
                for phase, t_pred in (("P", t_p), ("S", t_s)):
                    arrivals.append({
                        "eventId": eid, "stationId": sid, "phase": phase, "tPred": t_pred,
                        "tObs": t_pred + 0.01 if observed else math.nan,
                        "residualS": 0.01 if observed else math.nan,
                        "pickId": f"p:{sid}:{phase}:{t_pred:.3f}" if observed else None,
                        "usedInLocation": observed,
                    })  # fmt: skip
                log_a = (self.mags[i] - TRUE_B * math.log10(r_m / 1000.0) - TRUE_C
                         - TRUE_TERMS.get(sid, 0.0))  # fmt: skip
                self.true_log_a[(eid, sid)] = log_a
                disp_m = 10.0**log_a / (wa_gain * 1000.0)  # WA mm at 5 Hz -> ground m
                north += _gain(sid) * disp_m * _wavelet(t_axis - (t_s + 0.5))
            response_rate = 2.0 * SR if sid == "T.BAD" else SR
            _inventory(sid, response_rate).write(
                str(self.cache_dir / "stationxml" / f"{sid}.xml"), format="STATIONXML"
            )
            if sid != "T.NOD":
                _write_mseed(self.cache_dir, sid, "HHN", north)
                _write_mseed(self.cache_dir, sid, "HHE", 0.3 * north + rng.normal(0, 1, n_samples))
                _write_mseed(self.cache_dir, sid, "HHZ", rng.normal(0.0, 1.0, n_samples))

        matched = {i: f"uu{i:04d}" for i in (*ML_EVENTS, *MD_EVENTS)}
        events = [
            SeismicEvent(
                id=eid, runId="test-run", t=float(self.t[i]), latitude=38.5, longitude=-112.9,
                elevM=run.origin.elevM + self.enu[i][2],
                depthKm=(run.refSurfaceElevM - (run.origin.elevM + self.enu[i][2])) / 1000.0,
                enu=Enu(e=self.enu[i][0], n=self.enu[i][1], u=self.enu[i][2]),
                quality=LocationQuality(
                    method="grid1d", statics=False, nStations=5, nP=5, nS=4, rmsS=0.05,
                    gapDeg=90.0, minEpiDistM=2000.0, hErrM=200.0, vErrM=None,
                    depthOnEdge=False,
                ),
                tier="B", tierReasons=["test", "reason"], meanPickProb=0.8,
                catalogMatch=(CatalogMatch(catalogId=matched[i], dtS=0.1, distM=500.0)
                              if i in matched else None),
                revealOrder=-1, pickIds=[f"p:{i}:a", f"p:{i}:b"],
            )
            for i, eid in enumerate(self.ids)
        ]
        catalog = [
            CatalogEvent(
                id=cid, source="UU via test", t=float(self.t[i]), latitude=38.5,
                longitude=-112.9, depthKm=4.0, depthDatum="test", elevM=-4000.0,
                mag=float(self.mags[i]), magType="ml" if i in ML_EVENTS else "md",
                enu=Enu(e=0.0, n=0.0, u=0.0), matchedEventId=self.ids[i],
            )
            for i, cid in matched.items()
        ]
        catalog.append(
            CatalogEvent(
                id="uu9999", source="UU via test", t=T_DATA + 5.0, latitude=38.5,
                longitude=-112.9, depthKm=4.0, depthDatum="test", elevM=-4000.0, mag=0.5,
                magType="ml", enu=Enu(e=0.0, n=0.0, u=0.0),
            )
        )
        matches = pd.DataFrame(
            {
                "catalogId": pd.array([*matched.values(), "uu9999"], dtype="string"),
                "eventId": pd.array([self.ids[i] for i in matched] + [None], dtype="string"),
                "dtS": [0.1] * len(matched) + [math.nan],
                "distM": [500.0] * len(matched) + [math.nan],
                "reason": pd.array([None] * len(matched) + ["no data"], dtype="string"),
            }
        )
        arr = pd.DataFrame(arrivals)
        for col in ("eventId", "stationId", "phase", "pickId"):
            arr[col] = arr[col].astype("string")
        stations = to_frame([_station_row(s) for s in STATION_ENU], Station)
        write_table(to_frame(events, SeismicEvent), self.path("events.parquet"), "SeismicEvent")
        write_table(arr, self.path("arrivals.parquet"), "Arrival")
        write_table(stations, self.path("stations.parquet"), "Station")
        write_table(matches, self.path("matches.parquet"), "Match")
        write_table(to_frame(catalog, CatalogEvent), self.path("catalog.parquet"), "CatalogEvent")

    def path(self, name: str) -> Path:
        return self.run_dir / name


@pytest.fixture
def world(tmp_path: Path, run_section: RunSection, seismology_config: SeismologyConfig) -> World:
    return World(tmp_path, run_section, seismology_config.magnitude.woodAnderson)


def _ctx(make_ctx: Any, world: World, run: RunSection, cfg: SeismologyConfig) -> Any:
    ctx = make_ctx(run, cfg)
    assert ctx.run_dir == world.run_dir
    return type(ctx)(ctx.run_id, ctx.run_dir, world.cache_dir, ctx.config)


def _stage_cfg(seismology_config: SeismologyConfig, **overrides: Any) -> SeismologyConfig:
    # Four usable stations and nine ml events: small minimums, every other knob the showcase's.
    fit = {"minStationObs": 3, **overrides.pop("fit", {})}
    return _cfg(seismology_config, **{"minStations": 3, "minCalibrationEvents": 5, **overrides},
                fit=fit)  # fmt: skip


# --- Wood-Anderson and response removal ---------------------------------------------------------


@pytest.mark.smoke
def test_wood_anderson_response_limits(seismology_config: SeismologyConfig) -> None:
    wa = seismology_config.magnitude.woodAnderson
    high = np.abs(wood_anderson_response(np.array([1000.0]), wa))[0]
    at_f0 = np.abs(wood_anderson_response(np.array([1.0 / wa.periodS]), wa))[0]
    assert high == pytest.approx(wa.gain, rel=1e-3)  # flat to displacement above f0
    assert at_f0 == pytest.approx(wa.gain / (2.0 * wa.damping), rel=1e-9)
    assert np.abs(wood_anderson_response(np.array([0.0]), wa))[0] == 0.0


@pytest.mark.smoke
def test_response_removal_recovers_a_known_ground_displacement(
    seismology_config: SeismologyConfig,
) -> None:
    """A 5 Hz ground displacement through a velocity geophone response comes back as Wood-Anderson
    mm = D |H_WA(5 Hz)| x 1000, independent of the instrument."""
    cfg = seismology_config.magnitude
    zeros = [0j, 0j]
    poles = [-4.44 + 4.44j, -4.44 - 4.44j]
    s_n = 2j * np.pi * 5.0
    a0 = 1.0 / abs(np.prod([s_n - z for z in zeros]) / np.prod([s_n - p for p in poles]))
    resp = Response.from_paz(
        zeros=zeros, poles=poles, stage_gain=3.0e8, stage_gain_frequency=5.0, input_units="M/S",
        output_units="COUNTS", normalization_frequency=5.0, normalization_factor=a0,
    )  # fmt: skip
    f0, disp = 5.0, 2.0e-6
    r_disp = resp.get_evalresp_response_for_frequencies([f0], output="DISP")[0]
    t = np.arange(int(20 * SR)) / SR
    counts = np.real(r_disp * disp * np.exp(2j * np.pi * f0 * t))
    filt, clip = response_filter(resp, 1.0 / SR, nfft_for(len(t)), cfg)
    wa_mm = to_wood_anderson(counts, filt, round(cfg.window.padS * SR))
    middle = wa_mm[int(6 * SR) : int(14 * SR)]
    expected = disp * np.abs(wood_anderson_response(np.array([f0]), cfg.woodAnderson))[0] * 1000.0
    assert np.abs(middle).max() == pytest.approx(expected, rel=0.01)
    assert clip is None


@pytest.mark.parametrize("above_f2", [1.25, 2.5])
@pytest.mark.smoke
def test_accelerometer_at_1000_hz_is_not_high_passed_in_band(
    seismology_config: SeismologyConfig, above_f2: float
) -> None:
    """A flat 1000 Hz accelerometer (M/S**2): as displacement its response peaks at Nyquist, 80 dB
    above its 5 Hz value, so a 60 dB water level on the displacement response would pass a tenth
    of a 5 Hz signal and 40% of a 10 Hz one. In native units nothing in the band is clipped: a
    ground displacement low in the flat band [f2, f3] comes back at full amplitude."""
    cfg = seismology_config.magnitude
    f0 = above_f2 * cfg.response.preFiltHz[1]
    rate = 1000.0
    resp = Response.from_paz(
        zeros=[], poles=[], stage_gain=4.0e5, input_units="M/S**2", output_units="COUNTS"
    )
    disp = 1.0e-7
    r_disp = resp.get_evalresp_response_for_frequencies([f0], output="DISP")[0]
    t = np.arange(int(12 * rate)) / rate
    counts = np.real(r_disp * disp * np.exp(2j * np.pi * f0 * t))
    filt, clip = response_filter(resp, 1.0 / rate, nfft_for(len(t)), cfg)
    wa_mm = to_wood_anderson(counts, filt, round(cfg.window.padS * rate))
    expected = disp * np.abs(wood_anderson_response(np.array([f0]), cfg.woodAnderson))[0] * 1000.0
    assert np.abs(wa_mm[int(4 * rate) : int(8 * rate)]).max() == pytest.approx(expected, rel=0.01)
    assert clip is None


@pytest.mark.smoke
def test_water_level_clip_inside_the_band_is_reported(seismology_config: SeismologyConfig) -> None:
    """A displacement sensor with four poles at 5 Hz falls 60 dB below its maximum at about
    27.5 Hz: the water level clips from there to preFiltHz f3."""
    cfg = seismology_config.magnitude
    pole = -2.0 * np.pi * 5.0
    resp = Response.from_paz(
        zeros=[], poles=[pole] * 4, stage_gain=1.0e9, input_units="M", output_units="COUNTS",
        normalization_frequency=0.0, normalization_factor=pole**4,
    )  # fmt: skip
    _, clip = response_filter(resp, 1.0 / SR, nfft_for(int(20 * SR)), cfg)
    assert clip is not None
    f_60db = 5.0 * math.sqrt(10.0**1.5 - 1.0)  # (1 + (f/5)^2)^2 = 10^3
    assert clip[0] == pytest.approx(f_60db, abs=0.1)
    assert clip[1] == pytest.approx(cfg.response.preFiltHz[2], abs=0.1)


def test_measured_amplitudes_follow_the_truth_at_every_station(
    world: World, run_section: RunSection, seismology_config: SeismologyConfig
) -> None:
    """Measured log10 A minus the planted one is one constant (the wavelet's peak factor) for
    every event and station, whatever the station's gain; excluded stations are marked."""
    cfg = _cfg(seismology_config, readChunkS=200.0).magnitude  # several reads per station
    stations = read_table(world.path("stations.parquet"))
    plan = plan_windows(
        read_table(world.path("events.parquet")), read_table(world.path("arrivals.parquet")),
        stations, cfg,
    )  # fmt: skip
    window = (run_section.window_start_s, run_section.window_end_s)
    screens = {str(r["id"]): screen_station(r, cfg, window, cache_dir=world.cache_dir)
               for _, r in stations.iterrows()}  # fmt: skip
    result = measure_amplitudes(plan, screens, cfg, cache_dir=world.cache_dir)
    table = result.table
    ok = table[table["status"] == "ok"]
    assert set(ok["stationId"]) == {"T.A01", "T.A02", "T.A03", "T.A04"}
    assert len(ok) == 4 * N_EVENTS
    assert set(table.loc[table["stationId"].isin(["T.BAD", "T.NOD"]), "status"]) == {
        "stationExcluded"
    }
    offset = ok["logA"].to_numpy() - np.array(
        [world.true_log_a[(e, s)] for e, s in zip(ok["eventId"], ok["stationId"], strict=True)]
    )
    assert np.ptp(offset) < 0.05  # a shared response would put T.A02 off by log10(4)
    assert (ok["snr"] > cfg.minSnr).all()
    assert result.record["responseEvaluations"] == 4 * 2  # one per station and horizontal
    assert result.record["reads"] > 4


@pytest.mark.smoke
def test_read_groups_span_at_most_the_chunk() -> None:
    rng = np.random.default_rng(1)
    starts = np.sort(rng.uniform(0, 10000, 200))
    ends = starts + rng.uniform(5, 20, 200)
    groups = read_groups(starts, ends, 600.0)
    assert sorted(np.concatenate(groups).tolist()) == list(range(200))
    for g in groups:
        assert ends[g].max() - starts[g].min() <= 600.0 or len(g) == 1
    assert len(read_groups(np.array([0.0]), np.array([1000.0]), 600.0)) == 1


@pytest.mark.smoke
def test_window_statuses_gap_rate_flat_clipped(seismology_config: SeismologyConfig) -> None:
    """One window per made-up station, each read returning a trace with one defect."""
    cfg = seismology_config.magnitude
    w, sat = cfg.window, cfg.saturation
    t_p, t_s = T_DATA + 20.0, T_DATA + 23.0
    row: dict[str, Any] = {
        "eventId": "e0", "hypoDistM": 5000.0, "tP": t_p, "tS": t_s, "pAnchor": "observed",
        "sAnchor": "observed", "noiseStart": t_p - w.noiseGapS - w.noiseLenS,
        "noiseEnd": t_p - w.noiseGapS, "signalStart": t_s - w.sPreS, "signalEnd": t_s + w.sPostS,
    }  # fmt: skip
    row["readStart"], row["readEnd"] = row["noiseStart"] - w.padS, row["signalEnd"] + w.padS
    kinds = {"T.OK": "ok", "T.GAP": "gap", "T.RATE": "rateMismatch", "T.FLAT": "flat",
             "T.CLIP": "clipped"}  # fmt: skip
    plan = pd.DataFrame([{**row, "stationId": sid} for sid in kinds])[list(WINDOW_COLUMNS)]
    resp = _flat_response(SR)
    screens = {
        sid: StationScreen(stationId=sid, usable=True, reason=None, location="",
                           channels=("HHN", "HHE"), dataRateHz=SR,
                           responses={"HHN": resp, "HHE": resp})
        for sid in kinds
    }  # fmt: skip
    n = int(60 * SR)
    clean = np.random.default_rng(3).normal(0.0, 10.0, n)
    clean += 1.0e4 * _wavelet(T_DATA + np.arange(n) / SR - (t_s + 0.5))

    def reader(sid: str, t0: float, t1: float, *, cache_dir: Path) -> Stream:
        data, rate = clean.copy(), SR
        if sid == "T.FLAT":
            data[:] = 0.0  # a dead channel
        elif sid == "T.CLIP":
            data[int((t_s + 0.5 - T_DATA) * SR)] = 0.96 * sat.fullScaleCounts
        elif sid == "T.RATE":
            rate = SR / 2.0  # served at another rate than Station.sampleRateHz
        st = Stream()
        for cha in ("HHN", "HHE"):
            header = {"network": "T", "station": sid[2:], "location": "", "channel": cha,
                      "sampling_rate": rate, "starttime": UTCDateTime(T_DATA)}  # fmt: skip
            tr = Trace(data=np.round(data).astype(np.int32), header=header)
            if sid == "T.GAP" and cha == "HHN":  # 1 s missing inside the S window
                cut = int((t_s - T_DATA) * SR)
                st += tr.copy().slice(endtime=UTCDateTime(T_DATA + (cut - 1) / SR))
                st += tr.copy().slice(starttime=UTCDateTime(T_DATA + (cut + int(SR)) / SR))
                continue
            st += tr
        return st

    result = measure_amplitudes(plan, screens, cfg, cache_dir=Path("unused"), reader=reader)
    status = dict(zip(result.table["stationId"], result.table["status"], strict=True))
    assert status == kinds
    per = result.record["perStation"]
    assert per["T.CLIP"]["maxFullScaleFraction"] == pytest.approx(0.96, abs=1e-4)
    assert per["T.OK"]["maxFullScaleFraction"] < sat.maxFraction
    assert result.table.loc[result.table["stationId"] == "T.OK", "logA"].notna().all()
    assert result.table.loc[result.table["stationId"] != "T.OK", "logA"].isna().all()
    assert result.record["status"] == {v: 1 for v in kinds.values()}


# --- windows and station screening --------------------------------------------------------------


@pytest.mark.smoke
def test_plan_windows_anchors_and_distance(
    world: World, run_section: RunSection, seismology_config: SeismologyConfig
) -> None:
    cfg = seismology_config.magnitude
    events = read_table(world.path("events.parquet"))
    arrivals = read_table(world.path("arrivals.parquet"))
    stations = read_table(world.path("stations.parquet"))
    plan = plan_windows(events, arrivals, stations, cfg)
    assert len(plan) == N_EVENTS * len(STATION_ENU)
    row = plan[(plan["eventId"] == world.ids[0]) & (plan["stationId"] == "T.A01")].iloc[0]
    e, n, u = world.enu[0]
    assert row["hypoDistM"] == pytest.approx(math.sqrt((e - 3000) ** 2 + (n - 1000) ** 2 + u**2))
    assert row["sAnchor"] == "observed" and row["pAnchor"] == "observed"
    pred = plan[(plan["eventId"] == world.ids[0]) & (plan["stationId"] == "T.A02")].iloc[0]
    assert pred["sAnchor"] == "predicted"
    w = cfg.window
    assert row["signalStart"] == pytest.approx(row["tS"] - w.sPreS)
    assert row["noiseEnd"] == pytest.approx(row["tP"] - w.noiseGapS)
    assert row["readStart"] == pytest.approx(row["noiseStart"] - w.padS)
    assert row["readEnd"] == pytest.approx(row["signalEnd"] + w.padS)
    with pytest.raises(MagnitudeError, match="no arrivals"):
        plan_windows(events, arrivals[arrivals["eventId"] != world.ids[3]], stations, cfg)
    # a borehole sensor 1200 m down: R runs to the sensor's ENU (enu_u = sensorElevM - origin)
    bore = stations.copy()
    k = bore.index[bore["id"] == "T.A01"][0]
    sensor_elev = bore.loc[k, "surfaceElevM"] - 1200.0
    bore.loc[k, ["sensorDepthM", "sensorElevM", "enu_u"]] = [
        1200.0, sensor_elev, sensor_elev - run_section.origin.elevM
    ]  # fmt: skip
    deep = plan_windows(events, arrivals, bore, cfg)
    drow = deep[(deep["eventId"] == world.ids[0]) & (deep["stationId"] == "T.A01")].iloc[0]
    expected = math.sqrt(
        (e - 3000) ** 2 + (n - 1000) ** 2 + (u - (sensor_elev - run_section.origin.elevM)) ** 2
    )
    assert drow["hypoDistM"] == pytest.approx(expected)
    assert drow["hypoDistM"] < row["hypoDistM"]  # the events sit below the sensor


def test_screen_station_reasons(
    world: World, run_section: RunSection, seismology_config: SeismologyConfig
) -> None:
    cfg = seismology_config.magnitude
    stations = read_table(world.path("stations.parquet")).set_index("id", drop=False)
    window = (run_section.window_start_s, run_section.window_end_s)
    good = screen_station(stations.loc["T.A01"], cfg, window, cache_dir=world.cache_dir)
    assert good.usable and good.channels == ("HHN", "HHE") and good.flags == ()
    bad = screen_station(stations.loc["T.BAD"], cfg, window, cache_dir=world.cache_dir)
    assert not bad.usable and "200.0 Hz" in (bad.reason or "")
    nod = screen_station(stations.loc["T.NOD"], cfg, window, cache_dir=world.cache_dir)
    assert not nod.usable and "no waveform data" in (nod.reason or "")
    # a response without decimation stages is checked against Channel.sample_rate and flagged
    _inventory("T.A02", None).write(
        str(world.cache_dir / "stationxml" / "T.A02.xml"), format="STATIONXML"
    )
    flagged = screen_station(stations.loc["T.A02"], cfg, window, cache_dir=world.cache_dir)
    assert flagged.usable and any("responseRateFromChannel" in f for f in flagged.flags)
    high = _cfg(seismology_config, response={**cfg.response.model_dump(),
                                             "preFiltHz": (0.5, 1.0, 45.0, 55.0)})  # fmt: skip
    nyq = screen_station(stations.loc["T.A01"], high.magnitude, window, cache_dir=world.cache_dir)
    assert not nyq.usable and "Nyquist" in (nyq.reason or "")


# --- calibration ---------------------------------------------------------------------------------


def _synthetic_obs(rng: np.random.Generator, n_events: int, noise: float) -> pd.DataFrame:
    stations = {f"S{j}": s for j, s in enumerate([0.2, -0.1, 0.05, -0.15, 0.0])}
    rows = []
    for i in range(n_events):
        mag = rng.uniform(0.5, 2.5)
        for sid, term in stations.items():
            log_r = math.log10(rng.uniform(2.0, 30.0))
            log_a = (mag - 1.2 * log_r - 0.5 - term) / 0.9 + rng.normal(0.0, noise)
            rows.append({"eventId": f"e{i}", "stationId": sid, "logA": log_a, "logR": log_r,
                         "mag": mag})  # fmt: skip
    return pd.DataFrame(rows)


@pytest.mark.smoke
def test_fit_recovers_known_coefficients(seismology_config: SeismologyConfig) -> None:
    obs = _synthetic_obs(np.random.default_rng(7), 30, 0.0)
    fit = _cfg(seismology_config, fit={"amplitudeSlope": None, "stationTermRidge": 1e-9,
                                       "loss": "linear"}).magnitude.fit  # fmt: skip
    cal = fit_calibration(obs, fit)
    assert (cal.a, cal.b, cal.c) == pytest.approx((0.9, 1.2, 0.5), abs=1e-6)
    assert cal.stationTerms["S0"] == pytest.approx(0.2, abs=1e-6)
    assert not cal.aFixed and cal.nEvents == 30 and cal.nObs == 150
    robust = fit_calibration(obs, fit.model_copy(update={"loss": "soft_l1"}))
    assert (robust.a, robust.b) == pytest.approx((0.9, 1.2), abs=1e-5)


@pytest.mark.smoke
def test_ridge_terms_sum_to_zero_and_robust_loss_resists_an_outlier(
    seismology_config: SeismologyConfig,
) -> None:
    obs = _synthetic_obs(np.random.default_rng(8), 25, 0.05)
    obs.loc[3, "logA"] += 3.0  # one wild amplitude
    fit = seismology_config.magnitude.fit  # showcase: a fixed at 1, ridge, soft_l1
    cal = fit_calibration(obs, fit)
    assert cal.aFixed and cal.a == fit.amplitudeSlope
    assert sum(cal.stationTerms.values()) == pytest.approx(0.0, abs=1e-9)
    clean = obs.drop(index=3)
    lin = fit.model_copy(update={"loss": "linear"})
    shift_robust = abs(fit_calibration(obs, fit).c - fit_calibration(clean, fit).c)
    shift_linear = abs(fit_calibration(obs, lin).c - fit_calibration(clean, lin).c)
    assert shift_robust < shift_linear


@pytest.mark.smoke
def test_min_station_obs_and_event_magnitudes(seismology_config: SeismologyConfig) -> None:
    obs = _synthetic_obs(np.random.default_rng(9), 10, 0.0)
    sparse = pd.DataFrame([{"eventId": "e0", "stationId": "SX", "logA": 0.0, "logR": 1.0,
                            "mag": obs["mag"].iloc[0]}])  # fmt: skip
    obs = pd.concat([obs, sparse], ignore_index=True)
    fit = _cfg(seismology_config, fit={"minStationObs": 3}).magnitude.fit
    cal = fit_calibration(obs, fit)
    assert cal.stationsWithoutTerm == {"SX": 1} and "SX" not in cal.stationTerms
    mags = event_magnitudes(cal, obs, min_stations=3)
    first = mags.iloc[0]
    assert first["eventId"] == "e0" and first["nStations"] == 5  # SX has no term
    station_mags = cal.station_magnitudes(obs[(obs["eventId"] == "e0") & (obs["stationId"] != "SX")])
    assert first["value"] == pytest.approx(np.median(station_mags))
    nmad = 1.4826 * np.median(np.abs(station_mags - np.median(station_mags)))
    assert first["sigma"] == pytest.approx(nmad)
    assert event_magnitudes(cal, obs, min_stations=6)["value"].isna().all()


@pytest.mark.smoke
def test_leave_one_event_out_refits_without_the_event(seismology_config: SeismologyConfig) -> None:
    obs = _synthetic_obs(np.random.default_rng(10), 12, 0.1)
    cfg = _cfg(seismology_config, minStations=3).magnitude
    loo = leave_one_event_out(obs, cfg)
    assert list(loo["eventId"]) == [f"e{i}" for i in range(12)]
    held = obs[obs["eventId"] == "e4"]
    fold = fit_calibration(obs[obs["eventId"] != "e4"], cfg.fit)
    expected = event_magnitudes(fold, held, cfg.minStations)["value"].iloc[0]
    row = loo.set_index("eventId").loc["e4"]
    assert row["looMag"] == pytest.approx(expected)
    assert row["looError"] == pytest.approx(expected - held["mag"].iloc[0])
    others = obs[obs["eventId"] != "e4"].groupby("eventId")["mag"].first().mean()
    assert row["nullError"] == pytest.approx(others - held["mag"].iloc[0])


# --- the stage -----------------------------------------------------------------------------------


def test_stage_writes_calibrated_magnitudes(
    world: World, make_ctx: Any, run_section: RunSection, seismology_config: SeismologyConfig
) -> None:
    cfg = _stage_cfg(seismology_config)
    ctx = _ctx(make_ctx, world, run_section, cfg)
    before = pq.read_table(world.path("events.parquet"))
    magnitude_pkg.run(ctx)

    doc = json.loads(world.path("magnitude.json").read_text())
    assert set(doc) == {"n", "looMae", "coefficients"}
    calibration = MagCalibration.model_validate(doc)
    assert calibration.n == len(ML_EVENTS)  # ml only: the md events are never calibrated on
    assert calibration.looMae < 0.1
    assert calibration.coefficients["a"] == cfg.magnitude.fit.amplitudeSlope
    assert set(calibration.coefficients) == {"a", "b", "c"}

    after = pq.read_table(world.path("events.parquet"))
    assert after.schema.equals(before.schema, check_metadata=False)
    for key in (b"schemaVersion", b"model"):
        assert after.schema.metadata[key] == before.schema.metadata[key]
    for name in before.schema.names:
        if not name.startswith("magnitude_"):
            assert after.column(name).equals(before.column(name)), name
    events = read_table(world.path("events.parquet")).set_index("id")
    assert (events["magnitude_type"] == "ML_cal").all()
    np.testing.assert_allclose(events.loc[world.ids, "magnitude_value"], world.mags, atol=0.1)
    assert (events["magnitude_sigma"] >= 0).all()

    [rec] = ctx.records
    assert rec["stage"] == "magnitude" and rec["field"] == "matching"
    params = rec["params"]["magnitude"]
    assert params["gate"]["passed"] is True
    assert "200.0 Hz" in params["stationsExcluded"]["T.BAD"]
    assert "no waveform data" in params["stationsExcluded"]["T.NOD"]
    assert params["calibrationSet"]["matchedByCatalogMagType"] == {"md": 2, "ml": 9}
    assert params["calibrationSet"]["isMostMatchedType"] is True
    assert params["preprocessing"]["responseRemoval"]["preFiltHz"] == list(
        cfg.magnitude.response.preFiltHz
    )
    assert params["config"] == cfg.magnitude.model_dump(mode="json")
    assert len(params["leaveOneEventOut"]["events"]) == len(ML_EVENTS)
    assert params["leaveOneEventOut"]["nullModelMae"] > calibration.looMae
    assert rec["counts"]["magnitudes"] == N_EVENTS
    ml = world.mags[list(ML_EVENTS)]
    assert params["magnitudes"]["calibratedRange"] == pytest.approx([ml.min(), ml.max()])
    below = int((events["magnitude_value"] < ml.min()).sum())
    assert params["magnitudes"]["belowCalibratedRange"] == below
    assert rec["counts"]["stationsExcluded"] == 2
    assert not list(world.run_dir.glob("*.part"))
    censoring = params["magnitudes"]["censoring"]["byDecile"]
    assert sum(r["events"] for r in censoring) == N_EVENTS
    assert all(r["nStationsMin"] >= cfg.magnitude.minStations for r in censoring)
    spread = params["fit"]["bIdentification"]
    assert spread["withinStationLogRSd"] >= 0 and spread["betweenStationLogRSd"] > 0
    assert params["preprocessing"]["saturation"]["maxFraction"] == cfg.magnitude.saturation.maxFraction
    per = params["amplitudes"]["perStation"]
    assert all(per[s]["waterLevelClipHz"] == {} for s in per)  # flat responses clip nowhere
    assert all(per[s]["maxFullScaleFraction"] < 1e-3 for s in per)


def test_stage_gate_leaves_magnitudes_null(
    world: World,
    make_ctx: Any,
    run_section: RunSection,
    seismology_config: SeismologyConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    events = read_table(world.path("events.parquet"))
    events["magnitude_value"] = 9.9  # a stale magnitude from an earlier run must not survive
    events["magnitude_type"] = pd.array(["ML_cal"] * len(events), dtype="string")
    write_table(events, world.path("events.parquet"), "SeismicEvent")
    cfg = _stage_cfg(seismology_config, maxLooMae=1e-6)
    ctx = _ctx(make_ctx, world, run_section, cfg)
    with caplog.at_level(logging.ERROR, logger="hq.magnitude.run"):
        magnitude_pkg.run(ctx)
    assert "kill switch" in caplog.text
    out = read_table(world.path("events.parquet"))
    assert out["magnitude_value"].isna().all() and out["magnitude_type"].isna().all()
    assert MagCalibration.model_validate_json(world.path("magnitude.json").read_text()).looMae > 0
    [rec] = ctx.records
    assert rec["counts"]["magnitudes"] == 0 and rec["counts"]["gatePassed"] == 0


def test_stage_fails_loudly_without_enough_calibration_events(
    world: World, make_ctx: Any, run_section: RunSection, seismology_config: SeismologyConfig
) -> None:
    """A failure clears what an earlier run left: magnitude.json, magnitudes, the record."""
    stale = MagCalibration(n=9, looMae=0.05, coefficients={"a": 1.0, "b": 1.0, "c": 0.0})
    world.path("magnitude.json").write_text(stale.model_dump_json())
    events = read_table(world.path("events.parquet"))
    events["magnitude_value"] = 1.5
    events["magnitude_type"] = pd.array(["ML_cal"] * len(events), dtype="string")
    write_table(events, world.path("events.parquet"), "SeismicEvent")
    before = pq.read_table(world.path("events.parquet"))
    cfg = _stage_cfg(seismology_config, minStations=5)  # only four stations are usable
    ctx = _ctx(make_ctx, world, run_section, cfg)
    with pytest.raises(MagnitudeError, match="calibration events"):
        magnitude_pkg.run(ctx)
    assert not world.path("magnitude.json").exists()
    after = pq.read_table(world.path("events.parquet"))
    for name in before.schema.names:
        if not name.startswith("magnitude_"):
            assert after.column(name).equals(before.column(name)), name
    out = read_table(world.path("events.parquet"))
    assert out["magnitude_value"].isna().all() and out["magnitude_type"].isna().all()
    [rec] = ctx.records
    assert rec["counts"] == {"magnitudes": 0, "failed": 1}
    assert "calibration events" in rec["params"]["magnitude"]["failed"]
    assert not list(world.run_dir.glob("*.part"))


@pytest.mark.smoke
def test_stage_rejects_stale_matches(
    world: World, make_ctx: Any, run_section: RunSection, seismology_config: SeismologyConfig
) -> None:
    matches = read_table(world.path("matches.parquet"))
    matches.loc[0, "eventId"] = world.ids[11]
    write_table(matches, world.path("matches.parquet"), "Match")
    with pytest.raises(MagnitudeError, match="disagrees"):
        magnitude_pkg.run(_ctx(make_ctx, world, run_section, _stage_cfg(seismology_config)))


@pytest.mark.smoke
def test_resolve_stage_magnitude() -> None:
    fn = resolve_stage(stage_spec("magnitude"))
    assert fn is magnitude_pkg.run
    assert importlib.import_module("hq.magnitude.run").run is fn

"""Station amplitudes for the local magnitude (MAG-01): Wood-Anderson peaks in S windows.

For every located event and every used station, ``plan_windows`` sets three windows from the
event's rows in ``arrivals.parquet``: the S window ``[tS - sPreS, tS + sPostS]``, the noise window
``[tP - noiseGapS - noiseLenS, tP - noiseGapS]`` and the processing window, which spans both plus
``padS`` on each side. The anchors ``tP``/``tS`` are the observed picks used in the final location,
else the predicted times (``tPred``). ``R`` is the hypocentral distance from the event's ENU to the
sensor's ENU (``Station.enu`` is the sensor, ``sensorElevM`` included).

``screen_station`` decides once per station whether its horizontals can be deconvolved: exactly
two horizontal channels in ``Station.channels``, one StationXML epoch per channel covering the run
window, response input units of displacement, velocity or acceleration, a response sample rate
equal to the data rate (``rateRelTol``; UU.FORU's cached StationXML describes the 200 Hz epoch
while 100 Hz is served), and ``preFiltHz`` f4 below the Nyquist. The response rate comes from the
last decimation stage; a response without decimation stages (an analog sensor's single PAZ stage)
is checked against ``Channel.sample_rate`` and flagged ``responseRateFromChannel``.

``measure_amplitudes`` reads each station's windows in groups spanning at most ``readChunkS``
(one ``hq.ingest.cache.read_window`` call per group, so a station-day is read about once), and for
each window and horizontal channel: linear detrend, cosine taper over ``padS`` at each end, FFT,
times ``preFiltHz``'s cosine taper, times the inverted displacement response (water level
``waterLevelDb``, as ObsPy's ``remove_response``), times the Wood-Anderson response
``gain * s^2 / (s^2 + 2 h w0 s + w0^2)``, inverse FFT, in mm. The response spectrum is evaluated
once per station, channel, sample interval and FFT length (evalresp costs about a second at
1000 Hz).
The amplitude is the peak of ``sqrt(h1^2 + h2^2)`` in the S window; the noise level is the same
peak in the noise window, and a window is usable when their ratio is at least ``minSnr``.
A window whose processing span is not covered by one continuous trace on both horizontals is a
``gap`` (gaps are never filled).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import obspy
import pandas as pd
from obspy import UTCDateTime
from obspy.core.inventory import Inventory, Response
from obspy.signal.invsim import cosine_sac_taper, invert_spectrum
from scipy.signal import detrend
from scipy.signal.windows import tukey

from hq.config.seismology import MagnitudeConfig, WoodAndersonConfig
from hq.ingest.cache import CacheMissError, read_inventory, read_window, station_keys

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
ReadWindow = Callable[..., obspy.Stream]

AMPLITUDE_UNITS = "mm"  # Wood-Anderson trace amplitude
DISTANCE_UNITS = "km"  # log10(R) with R in km
M_TO_MM = 1000.0  # unit conversion, not a knob
M_PER_KM = 1000.0
RESPONSE_OUTPUT = "DISP"  # evalresp output: ground displacement (m)
SUPPORTED_INPUT_UNITS = ("M", "M/S", "M/S**2")  # displacement, velocity, acceleration
PHASES = ("P", "S")

# Window status values (one per event x station).
OK = "ok"
LOW_SNR = "lowSnr"
GAP = "gap"
RATE_MISMATCH = "rateMismatch"
FLAT = "flat"  # zero amplitude in the S window: a dead channel, never a magnitude
EXCLUDED = "stationExcluded"

WINDOW_COLUMNS: tuple[str, ...] = (
    "eventId",
    "stationId",
    "hypoDistM",
    "tP",
    "tS",
    "pAnchor",
    "sAnchor",
    "noiseStart",
    "noiseEnd",
    "signalStart",
    "signalEnd",
    "readStart",
    "readEnd",
)


class MagnitudeError(RuntimeError):
    """A magnitude input or calibration problem that must stop the stage."""


# --- windows --------------------------------------------------------------------------------------


def _anchors(arrivals: pd.DataFrame) -> pd.DataFrame:
    """Per (eventId, stationId): tP, tS and whether each is the observed pick or the prediction."""
    missing = [
        c
        for c in ("eventId", "stationId", "phase", "tPred", "tObs", "usedInLocation")
        if c not in arrivals.columns
    ]
    if missing:
        raise MagnitudeError(f"arrivals lack columns {missing}")
    frame = arrivals[["eventId", "stationId", "phase", "tPred", "tObs", "usedInLocation"]].copy()
    frame["eventId"] = frame["eventId"].astype(str)
    frame["stationId"] = frame["stationId"].astype(str)
    frame["phase"] = frame["phase"].astype(str)
    if frame.duplicated(["eventId", "stationId", "phase"]).any():
        raise MagnitudeError("arrivals: an (eventId, stationId, phase) appears twice")
    t_pred = frame["tPred"].to_numpy(dtype=np.float64)
    if not np.isfinite(t_pred).all():
        raise MagnitudeError("arrivals: tPred must be set on every row")
    t_obs = frame["tObs"].to_numpy(dtype=np.float64)
    used = frame["usedInLocation"].to_numpy(dtype=bool) & np.isfinite(t_obs)
    frame["t"] = np.where(used, t_obs, t_pred)
    frame["anchor"] = np.where(used, "observed", "predicted")
    wide = frame.pivot(index=["eventId", "stationId"], columns="phase", values=["t", "anchor"])
    for phase in PHASES:
        if ("t", phase) not in wide.columns:
            raise MagnitudeError(f"arrivals hold no {phase} rows")
    if wide[[("t", p) for p in PHASES]].isna().any().any():
        raise MagnitudeError("arrivals: every event-station needs both a P and an S row")
    out = pd.DataFrame(
        {
            "eventId": wide.index.get_level_values(0),
            "stationId": wide.index.get_level_values(1),
            "tP": wide[("t", "P")].to_numpy(dtype=np.float64),
            "tS": wide[("t", "S")].to_numpy(dtype=np.float64),
            "pAnchor": wide[("anchor", "P")].to_numpy(dtype=object),
            "sAnchor": wide[("anchor", "S")].to_numpy(dtype=object),
        }
    )
    return out


def plan_windows(
    events: pd.DataFrame, arrivals: pd.DataFrame, stations: pd.DataFrame, cfg: MagnitudeConfig
) -> pd.DataFrame:
    """One row per (event, station) in ``arrivals`` (``WINDOW_COLUMNS``), in event then station
    order. Every event must have arrivals and every arrival station a ``stations`` row."""
    ids = events["id"].astype(str)
    if ids.duplicated().any():
        raise MagnitudeError("events: duplicate ids")
    anchors = _anchors(arrivals)
    unknown_events = sorted(set(anchors["eventId"]) - set(ids))
    if unknown_events:
        raise MagnitudeError(
            f"arrivals name {len(unknown_events)} events not in events.parquet, e.g. "
            f"{unknown_events[0]}; the tables are from different runs"
        )
    no_arrivals = sorted(set(ids) - set(anchors["eventId"]))
    if no_arrivals:
        raise MagnitudeError(f"{len(no_arrivals)} events have no arrivals, e.g. {no_arrivals[0]}")
    station_ids = stations["id"].astype(str)
    unknown_stations = sorted(set(anchors["stationId"]) - set(station_ids))
    if unknown_stations:
        raise MagnitudeError(f"arrivals name stations not in stations.parquet: {unknown_stations}")

    ev = pd.DataFrame(
        {
            "eventId": ids.to_numpy(),
            "ev_e": events["enu_e"].to_numpy(dtype=np.float64),
            "ev_n": events["enu_n"].to_numpy(dtype=np.float64),
            "ev_u": events["enu_u"].to_numpy(dtype=np.float64),
            "order": np.arange(len(events)),
        }
    )
    st = pd.DataFrame(
        {
            "stationId": station_ids.to_numpy(),
            "st_e": stations["enu_e"].to_numpy(dtype=np.float64),
            "st_n": stations["enu_n"].to_numpy(dtype=np.float64),
            "st_u": stations["enu_u"].to_numpy(dtype=np.float64),
        }
    )
    plan = anchors.merge(ev, on="eventId", how="left").merge(st, on="stationId", how="left")
    plan = plan.sort_values(["order", "stationId"], kind="stable").reset_index(drop=True)
    plan["hypoDistM"] = np.sqrt(
        (plan["ev_e"] - plan["st_e"]) ** 2
        + (plan["ev_n"] - plan["st_n"]) ** 2
        + (plan["ev_u"] - plan["st_u"]) ** 2
    )
    if not np.isfinite(plan["hypoDistM"].to_numpy(dtype=np.float64)).all():
        raise MagnitudeError("non-finite ENU in events or stations")
    if (plan["hypoDistM"] <= 0).any():
        raise MagnitudeError("an event sits exactly on a sensor (hypocentral distance 0)")
    w = cfg.window
    plan["noiseEnd"] = plan["tP"] - w.noiseGapS
    plan["noiseStart"] = plan["noiseEnd"] - w.noiseLenS
    plan["signalStart"] = plan["tS"] - w.sPreS
    plan["signalEnd"] = plan["tS"] + w.sPostS
    plan["readStart"] = np.minimum(plan["noiseStart"], plan["signalStart"]) - w.padS
    plan["readEnd"] = np.maximum(plan["noiseEnd"], plan["signalEnd"]) + w.padS
    return plan[list(WINDOW_COLUMNS)]


# --- station screening ----------------------------------------------------------------------------


@dataclass(frozen=True)
class StationScreen:
    """Whether a station's horizontals can be deconvolved, and with which responses."""

    stationId: str
    usable: bool
    reason: str | None
    location: str
    channels: tuple[str, ...]  # the two horizontal channel codes, in Station.channels order
    dataRateHz: float
    responses: dict[str, Response] = field(default_factory=dict, repr=False)
    responseRateHz: dict[str, float | None] = field(default_factory=dict)
    inputUnits: dict[str, str] = field(default_factory=dict)
    flags: tuple[str, ...] = ()

    def record(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "reason": self.reason,
            "location": self.location,
            "channels": list(self.channels),
            "dataRateHz": self.dataRateHz,
            "responseRateHz": dict(self.responseRateHz),
            "inputUnits": dict(self.inputUnits),
            "flags": list(self.flags),
        }


def response_rate_hz(response: Response) -> float | None:
    """Output sample rate of the last decimation stage, or None when no stage carries one."""
    for stage in reversed(response.response_stages):
        rate = stage.decimation_input_sample_rate
        factor = stage.decimation_factor
        if rate is not None and factor:
            return float(rate) / float(factor)
    return None


def _excluded(station: pd.Series, horizontals: tuple[str, ...], reason: str) -> StationScreen:
    return StationScreen(
        stationId=str(station["id"]),
        usable=False,
        reason=reason,
        location=str(station["location"]),
        channels=horizontals,
        dataRateHz=float(station["sampleRateHz"]),
    )


def screen_station(
    station: pd.Series,
    cfg: MagnitudeConfig,
    window: tuple[float, float],
    *,
    cache_dir: Path,
) -> StationScreen:
    """Check one ``stations.parquet`` row against its cached StationXML and waveform files."""
    sid = str(station["id"])
    channels = tuple(str(c) for c in station["channels"])
    horizontals = tuple(c for c in channels if not c.endswith("Z"))
    data_rate = float(station["sampleRateHz"])
    if len(horizontals) != 2:
        return _excluded(station, horizontals, f"needs 2 horizontal channels, has {channels}")
    try:
        station_keys(sid, cache_dir=cache_dir)
    except CacheMissError:
        return _excluded(station, horizontals, "no waveform data in the cache")
    try:
        inventory: Inventory = read_inventory(sid, cache_dir=cache_dir)
    except CacheMissError:
        return _excluded(station, horizontals, "no StationXML in the cache")
    f4 = cfg.response.preFiltHz[3]
    if f4 >= data_rate / 2.0:
        return _excluded(
            station, horizontals, f"preFiltHz f4 {f4} Hz is not below the {data_rate / 2} Hz Nyquist"
        )
    start, end = UTCDateTime(window[0]), UTCDateTime(window[1])
    responses: dict[str, Response] = {}
    rates: dict[str, float | None] = {}
    units: dict[str, str] = {}
    flags: list[str] = []
    net, sta = sid.split(".")[:2]
    for cha in horizontals:
        selected = inventory.select(
            network=net, station=sta, location=str(station["location"]), channel=cha
        )
        epochs = [ch for n in selected for s in n for ch in s]
        matches = [
            ch
            for ch in epochs
            if ch.start_date <= start and (ch.end_date is None or ch.end_date >= end)
        ]
        if len(matches) != 1:
            return _excluded(
                station,
                horizontals,
                f"{cha}: {len(matches)} StationXML epochs cover the run window "
                f"({len(epochs)} epochs in the file); need exactly 1",
            )
        channel = matches[0]
        response = channel.response
        if response is None or response.instrument_sensitivity is None:
            return _excluded(station, horizontals, f"{cha}: no response in the StationXML")
        unit = str(response.instrument_sensitivity.input_units or "").upper()
        units[cha] = unit
        if unit not in SUPPORTED_INPUT_UNITS:
            return _excluded(
                station, horizontals, f"{cha}: response input units {unit!r} are not ground motion"
            )
        rate = response_rate_hz(response)
        rates[cha] = rate
        if rate is None:
            rate = float(channel.sample_rate)
            flags.append(f"{cha}: responseRateFromChannel ({rate} Hz, no decimation stage)")
        if abs(rate / data_rate - 1.0) > cfg.response.rateRelTol:
            return StationScreen(
                stationId=sid,
                usable=False,
                reason=f"{cha}: the StationXML response describes {rate} Hz but the data is "
                f"{data_rate} Hz (a response epoch that does not match the served data)",
                location=str(station["location"]),
                channels=horizontals,
                dataRateHz=data_rate,
                responseRateHz=rates,
                inputUnits=units,
            )
        responses[cha] = response
    return StationScreen(
        stationId=sid,
        usable=True,
        reason=None,
        location=str(station["location"]),
        channels=horizontals,
        dataRateHz=data_rate,
        responses=responses,
        responseRateHz=rates,
        inputUnits=units,
        flags=tuple(flags),
    )


# --- response removal + Wood-Anderson -------------------------------------------------------------


def wood_anderson_response(freqs: FloatArray, wa: WoodAndersonConfig) -> ComplexArray:
    """Wood-Anderson trace displacement per unit ground displacement at ``freqs`` (Hz):
    ``gain * s^2 / (s^2 + 2 h w0 s + w0^2)``, ``s = 2 pi i f``, ``w0 = 2 pi / periodS``."""
    s = 2j * np.pi * np.asarray(freqs, dtype=np.float64)
    w0 = 2.0 * np.pi / wa.periodS
    return np.asarray(wa.gain * s**2 / (s**2 + 2.0 * wa.damping * w0 * s + w0**2))


def nfft_for(npts: int) -> int:
    """FFT length: the power of two at or above ``2 * npts`` (no wrap-around)."""
    return 1 << max(1, math.ceil(math.log2(2 * npts)))


def response_filter(
    response: Response, delta: float, nfft: int, cfg: MagnitudeConfig
) -> ComplexArray:
    """The rfft-domain filter from raw counts to Wood-Anderson mm: preFiltHz taper x inverted
    displacement response (water level) x Wood-Anderson response x 1000."""
    resp, freqs = response.get_evalresp_response(t_samp=delta, nfft=nfft, output=RESPONSE_OUTPUT)
    resp = np.asarray(resp, dtype=np.complex128)
    invert_spectrum(resp, cfg.response.waterLevelDb)
    taper = cosine_sac_taper(np.asarray(freqs), flimit=cfg.response.preFiltHz)
    return np.asarray(
        resp * taper * wood_anderson_response(np.asarray(freqs), cfg.woodAnderson) * M_TO_MM
    )


def to_wood_anderson(data: FloatArray, filt: ComplexArray, pad_samples: int) -> FloatArray:
    """Raw counts -> Wood-Anderson mm with a precomputed ``response_filter`` (its length fixes
    the FFT length): linear detrend, cosine taper over ``pad_samples`` at each end, filter."""
    npts = len(data)
    nfft = 2 * (len(filt) - 1)
    if nfft < npts:
        raise ValueError(f"filter for nfft {nfft} is shorter than the {npts}-sample window")
    x = detrend(np.asarray(data, dtype=np.float64), type="linear")
    alpha = min(1.0, 2.0 * pad_samples / npts) if npts > 1 else 0.0
    x = x * tukey(npts, alpha=alpha)
    spec = np.fft.rfft(x, n=nfft) * filt
    spec[-1] = abs(spec[-1]) + 0.0j  # as ObsPy's remove_response: real Nyquist bin
    return np.asarray(np.fft.irfft(spec, n=nfft)[:npts])


# --- measurement ----------------------------------------------------------------------------------


def read_groups(starts: FloatArray, ends: FloatArray, chunk_s: float) -> list[npt.NDArray[np.intp]]:
    """Indices grouped so each group's span ``[min start, max end]`` is at most ``chunk_s``
    (a single window longer than that is its own group), in start order."""
    order = np.argsort(starts, kind="stable")
    groups: list[list[int]] = []
    g_start = g_end = math.nan
    for i in order:
        s, e = float(starts[i]), float(ends[i])
        if groups and max(g_end, e) - g_start <= chunk_s:
            groups[-1].append(int(i))
            g_end = max(g_end, e)
        else:
            groups.append([int(i)])
            g_start, g_end = s, e
    return [np.asarray(g, dtype=np.intp) for g in groups]


@dataclass
class _Filters:
    """Response filters per (station, channel, delta, nfft), evaluated once each."""

    cfg: MagnitudeConfig
    cache: dict[tuple[str, str, float, int], ComplexArray] = field(default_factory=dict)
    evaluations: int = 0

    def get(
        self, station_id: str, cha: str, response: Response, delta: float, nfft: int
    ) -> ComplexArray:
        key = (station_id, cha, delta, nfft)
        if key not in self.cache:
            self.cache[key] = response_filter(response, delta, nfft, self.cfg)
            self.evaluations += 1
        return self.cache[key]


def _covering(st: obspy.Stream, location: str, cha: str, t0: float, t1: float) -> obspy.Trace | None:
    """The continuous trace holding every sample of ``[t0, t1]``: its first sample at most one
    sample interval after ``t0`` and its last at most one before ``t1`` (samples sit on the
    trace's own grid, and a read trims to the samples inside the requested span)."""
    for tr in st:
        if tr.stats.channel != cha or tr.stats.location != location:
            continue
        delta = float(tr.stats.delta)
        if tr.stats.starttime.timestamp - t0 < delta and t1 - tr.stats.endtime.timestamp < delta:
            return tr
    return None


def _measure_one(
    row: pd.Series,
    st: obspy.Stream,
    screen: StationScreen,
    filters: _Filters,
    cfg: MagnitudeConfig,
) -> dict[str, Any]:
    t0, t1 = float(row["readStart"]), float(row["readEnd"])
    wa: list[FloatArray] = []
    start: float | None = None
    delta = 0.0
    for cha in screen.channels:
        tr = _covering(st, screen.location, cha, t0, t1)
        if tr is None:
            return {"status": GAP}
        rate = float(tr.stats.sampling_rate)
        if abs(rate / screen.dataRateHz - 1.0) > cfg.response.rateRelTol:
            return {"status": RATE_MISMATCH}
        piece = tr.slice(UTCDateTime(t0), UTCDateTime(t1))
        delta = float(piece.stats.delta)
        npts = piece.stats.npts
        filt = filters.get(screen.stationId, cha, screen.responses[cha], delta, nfft_for(npts))
        pad = round(cfg.window.padS / delta)
        wa.append(to_wood_anderson(piece.data, filt, pad))
        if start is None:
            start = piece.stats.starttime.timestamp
    assert start is not None
    n = min(len(x) for x in wa)
    vec = np.hypot(wa[0][:n], wa[1][:n])
    t = start + np.arange(n) * delta
    sig = vec[(t >= row["signalStart"]) & (t <= row["signalEnd"])]
    noise = vec[(t >= row["noiseStart"]) & (t <= row["noiseEnd"])]
    if sig.size == 0 or noise.size == 0:
        return {"status": GAP}
    peak = float(sig.max())
    noise_peak = float(noise.max())
    if peak <= 0.0:
        return {"status": FLAT, "ampMm": peak, "noiseMm": noise_peak}
    snr = peak / noise_peak if noise_peak > 0.0 else math.inf
    status = OK if snr >= cfg.minSnr else LOW_SNR
    return {"status": status, "ampMm": peak, "noiseMm": noise_peak, "snr": snr}


@dataclass(frozen=True)
class AmplitudeResult:
    """``table``: one row per planned window (``WINDOW_COLUMNS`` plus ``status``, ``ampMm``,
    ``noiseMm``, ``snr``, ``logA``, ``logR``); ``record``: counts and timings for the run."""

    table: pd.DataFrame
    record: dict[str, Any]


def measure_amplitudes(
    plan: pd.DataFrame,
    screens: dict[str, StationScreen],
    cfg: MagnitudeConfig,
    *,
    cache_dir: Path,
    reader: ReadWindow = read_window,
) -> AmplitudeResult:
    """Amplitude, noise level and status for every planned window (see the module docstring)."""
    started = time.perf_counter()
    out = plan.copy()
    out["status"] = EXCLUDED
    out["ampMm"] = np.nan
    out["noiseMm"] = np.nan
    out["snr"] = np.nan
    filters = _Filters(cfg)
    reads = 0
    per_station: dict[str, dict[str, Any]] = {}
    for sid, idx in out.groupby("stationId", sort=True).groups.items():
        screen = screens[str(sid)]
        if not screen.usable:
            continue
        s_started = time.perf_counter()
        rows = out.loc[idx]
        groups = read_groups(
            rows["readStart"].to_numpy(dtype=np.float64),
            rows["readEnd"].to_numpy(dtype=np.float64),
            cfg.readChunkS,
        )
        for g in groups:
            sub = rows.iloc[g]
            st = reader(
                str(sid),
                float(sub["readStart"].min()),
                float(sub["readEnd"].max()),
                cache_dir=cache_dir,
            )
            reads += 1
            for label, row in sub.iterrows():
                for key, value in _measure_one(row, st, screen, filters, cfg).items():
                    out.at[label, key] = value
            del st
        statuses = out.loc[idx, "status"].value_counts().to_dict()
        per_station[str(sid)] = {
            "reads": len(groups),
            "windows": len(idx),
            "status": {str(k): int(v) for k, v in statuses.items()},
            "runtimeS": round(time.perf_counter() - s_started, 3),
        }
        log.info(
            "magnitude: %s: %d windows in %d reads, %s (%.1f s)",
            sid, len(idx), len(groups), per_station[str(sid)]["status"],
            per_station[str(sid)]["runtimeS"],
        )  # fmt: skip
    amp = out["ampMm"].to_numpy(dtype=np.float64)
    ok = (out["status"] == OK).to_numpy()
    out["logA"] = np.where(ok, np.log10(np.where(ok, amp, 1.0)), np.nan)
    out["logR"] = np.log10(out["hypoDistM"].to_numpy(dtype=np.float64) / M_PER_KM)
    status_counts = {str(k): int(v) for k, v in out["status"].value_counts().to_dict().items()}
    record = {
        "windows": len(out),
        "status": status_counts,
        "reads": reads,
        "responseEvaluations": filters.evaluations,
        "perStation": per_station,
        "runtimeS": round(time.perf_counter() - started, 3),
    }
    log.info(
        "magnitude: %d windows (%s), %d cache reads, %d response evaluations in %.1f s",
        len(out), status_counts, reads, filters.evaluations, record["runtimeS"],
    )  # fmt: skip
    return AmplitudeResult(table=out, record=record)

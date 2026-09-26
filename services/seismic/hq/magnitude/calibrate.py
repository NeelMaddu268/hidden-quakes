"""Magnitude calibration (MAG-01): robust least squares with ridge station terms, leave-one-out.

Observations are station amplitudes of calibration events: one row per (event, station) with
``logA`` (log10 Wood-Anderson mm), ``logR`` (log10 hypocentral km) and ``mag`` (the event's
public regional catalog magnitude, one magnitude type only). The model is

    mag = a logA + b logR + c + s_station

fitted over all rows with scipy's ``least_squares`` from the ordinary (ridge) least-squares start.
``fit.amplitudeSlope`` null fits ``a``; a number fixes it (the rows then fit ``mag - a logA``).

Station terms carry a ridge constraint: the objective adds ``stationTermRidge * sum(s_j^2)``,
kept quadratic (only the data rows go through the robust ``fit.loss`` with ``fit.fScaleMag``);
with the free intercept ``c`` the terms also sum to zero at the optimum. With the Gaussian
reading of the penalty, ``stationTermRidge = (residual sd / station-term sd)^2``.

``b`` is not identified by a clustered calibration set. When the calibration events sit within a
few km of each other, each station's hypocentral distance barely changes between them
(``Calibration.distanceSpread["withinStationLogRSd"]``), so free station terms would absorb the
distance dependence and leave ``b`` to that small spread. The ridge decides how much of the
between-station distance spread (``betweenStationLogRSd``) is credited to ``b`` rather than to the
terms, so the fitted ``b`` follows ``stationTermRidge`` and ``fit.loss`` as much as the data. The
leave-one-event-out MAE cannot test ``b``: every held-out event shares the others' source region.
Magnitudes of events far from the calibration events depend on it.

A station with fewer than ``fit.minStationObs`` rows gets no term: its rows are left out of the
fit and it gives no station magnitude from that fit.

A station magnitude is the model value for one row; an event magnitude is the median of its
station magnitudes (stations with a term only), with ``sigma`` the normalized median absolute
deviation (1.4826 x MAD) of those station magnitudes, and no magnitude below ``minStations``.

Leave-one-event-out: for each calibration event, refit without all of its rows (the station set
and every coefficient come from the remaining events) and predict its magnitude from its own
amplitudes. ``looMae`` is the mean absolute difference from the catalog magnitude. The record
also gets the MAE of the null model that predicts each event as the mean catalog magnitude of
the others, so the MAE can be read against the spread of the calibration set.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import least_squares

from hq.config.seismology import MagnitudeConfig, MagnitudeFitConfig
from hq.magnitude.amplitude import MagnitudeError

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
NMAD_FACTOR = 1.4826  # MAD -> standard deviation for normal data (a constant, not a knob)
OBS_COLUMNS: tuple[str, ...] = ("eventId", "stationId", "logA", "logR")


def _check_obs(obs: pd.DataFrame, need_mag: bool) -> None:
    cols = [*OBS_COLUMNS, "mag"] if need_mag else list(OBS_COLUMNS)
    missing = [c for c in cols if c not in obs.columns]
    if missing:
        raise MagnitudeError(f"observations lack columns {missing}")
    for c in cols[2:]:
        if not np.isfinite(obs[c].to_numpy(dtype=np.float64)).all():
            raise MagnitudeError(f"observations: non-finite {c}")


def _rho(name: str, z: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """scipy ``least_squares`` loss ``rho(z)`` and its first two derivatives (``z`` = squared
    residual over ``f_scale^2``), for the named losses as scipy defines them."""
    if name == "linear":
        return z, np.ones_like(z), np.zeros_like(z)
    if name == "soft_l1":
        t = 1.0 + z
        return 2.0 * (np.sqrt(t) - 1.0), t**-0.5, -0.5 * t**-1.5
    if name == "huber":
        small = z <= 1.0
        root = np.sqrt(np.where(small, 1.0, z))
        return (
            np.where(small, z, 2.0 * root - 1.0),
            np.where(small, 1.0, 1.0 / root),
            np.where(small, 0.0, -0.5 / root**3),
        )
    if name == "cauchy":
        return np.log1p(z), 1.0 / (1.0 + z), -1.0 / (1.0 + z) ** 2
    if name == "arctan":
        return np.arctan(z), 1.0 / (1.0 + z**2), -2.0 * z / (1.0 + z**2) ** 2
    raise MagnitudeError(f"unknown loss {name!r}")


def _mixed_loss(name: str, n_data: int) -> Callable[[FloatArray], FloatArray]:
    """``name`` on the first ``n_data`` residuals (the observations), linear on the rest (the
    ridge rows), so the station-term penalty stays exactly quadratic."""

    def loss(z: FloatArray) -> FloatArray:
        out = np.empty((3, z.size))
        out[:, :n_data] = np.vstack(_rho(name, z[:n_data]))
        out[:, n_data:] = np.vstack(_rho("linear", z[n_data:]))
        return out

    return loss


@dataclass(frozen=True)
class Calibration:
    """A fitted model. ``stationsWithoutTerm`` maps each station left out of the fit (fewer
    than ``minStationObs`` rows) to its row count."""

    a: float
    b: float
    c: float
    aFixed: bool
    stationTerms: dict[str, float]
    nEvents: int
    nObs: int
    stationsWithoutTerm: dict[str, int] = field(default_factory=dict)
    robust: dict[str, Any] = field(default_factory=dict)
    # log10(R km) spread of the fitted rows: within stations (rms about each station's mean, the
    # only distance information once terms are free) and between the stations' means.
    distanceSpread: dict[str, float] = field(default_factory=dict)

    def coefficients(self) -> dict[str, float]:
        """``MagCalibration.coefficients``: the global terms (station terms go to the record)."""
        return {"a": self.a, "b": self.b, "c": self.c}

    def station_magnitudes(self, obs: pd.DataFrame) -> FloatArray:
        """One magnitude per row; NaN where the station has no term."""
        _check_obs(obs, need_mag=False)
        terms = obs["stationId"].astype(str).map(self.stationTerms).to_numpy(dtype=np.float64)
        return np.asarray(
            self.a * obs["logA"].to_numpy(dtype=np.float64)
            + self.b * obs["logR"].to_numpy(dtype=np.float64)
            + self.c
            + terms
        )


def fit_calibration(obs: pd.DataFrame, fit: MagnitudeFitConfig) -> Calibration:
    """Fit ``mag = a logA + b logR + c + s_station`` (see the module docstring)."""
    _check_obs(obs, need_mag=True)
    counts = obs["stationId"].astype(str).value_counts()
    with_term = sorted(str(s) for s in counts.index[counts >= fit.minStationObs])
    without = {str(s): int(n) for s, n in counts.items() if n < fit.minStationObs}
    if not with_term:
        raise MagnitudeError(f"no station has {fit.minStationObs} calibration observations")
    use = obs[obs["stationId"].astype(str).isin(with_term)]
    x_a = use["logA"].to_numpy(dtype=np.float64)
    x_r = use["logR"].to_numpy(dtype=np.float64)
    y = use["mag"].to_numpy(dtype=np.float64)
    columns: list[FloatArray] = []
    names: list[str] = []
    if fit.amplitudeSlope is None:
        columns.append(x_a)
        names.append("a")
    else:
        y = y - fit.amplitudeSlope * x_a
    columns += [x_r, np.ones_like(x_r)]
    names += ["b", "c"]
    global_part = np.column_stack(columns)
    n_rows = len(y)
    if n_rows <= len(names) or int(np.linalg.matrix_rank(global_part)) < len(names):
        raise MagnitudeError(
            f"{n_rows} calibration observations cannot determine {names} (too few rows, or "
            "every observation at one distance or one amplitude)"
        )
    index = {s: k for k, s in enumerate(with_term)}
    k = use["stationId"].astype(str).map(index).to_numpy(dtype=np.int64)
    n_terms = len(with_term)
    indicators = np.zeros((n_rows, n_terms))
    indicators[np.arange(n_rows), k] = 1.0
    ridge = np.hstack(
        [np.zeros((n_terms, len(names))), math.sqrt(fit.stationTermRidge) * np.eye(n_terms)]
    )
    design = np.vstack([np.hstack([global_part, indicators]), ridge])
    target = np.concatenate([y, np.zeros(n_terms)])
    p0, *_ = np.linalg.lstsq(design, target, rcond=None)
    robust: dict[str, Any] = {"loss": fit.loss, "fScaleMag": fit.fScaleMag}
    if fit.loss == "linear":
        p = p0
    else:
        result = least_squares(
            lambda q: design @ q - target,
            p0,
            jac=lambda q: design,
            loss=_mixed_loss(fit.loss, n_rows),
            f_scale=fit.fScaleMag,
            method="trf",
        )
        if not result.success:
            raise MagnitudeError(f"robust calibration fit did not converge: {result.message}")
        p = result.x
        robust["nfev"] = int(result.nfev)
        resid = (design @ p - target)[:n_rows]
        robust["beyondFScale"] = int((np.abs(resid) > fit.fScaleMag).sum())
    values = dict(zip(names, (float(v) for v in p[: len(names)]), strict=True))
    terms = [float(v) for v in p[len(names) :]]
    robust["stationTermSum"] = float(sum(terms))
    station_mean_log_r = np.bincount(k, weights=x_r, minlength=n_terms) / np.bincount(
        k, minlength=n_terms
    )
    spread = {
        "withinStationLogRSd": float(np.sqrt(np.mean((x_r - station_mean_log_r[k]) ** 2))),
        "betweenStationLogRSd": float(np.std(station_mean_log_r)),
    }
    return Calibration(
        a=values["a"] if fit.amplitudeSlope is None else float(fit.amplitudeSlope),
        b=values["b"],
        c=values["c"],
        aFixed=fit.amplitudeSlope is not None,
        stationTerms=dict(zip(with_term, terms, strict=True)),
        nEvents=int(use["eventId"].nunique()),
        nObs=int(n_rows),
        stationsWithoutTerm=without,
        robust=robust,
        distanceSpread=spread,
    )


def event_magnitudes(cal: Calibration, obs: pd.DataFrame, min_stations: int) -> pd.DataFrame:
    """Per event in ``obs`` (first-appearance order): ``nStations`` (station magnitudes with a
    term), ``value`` (their median) and ``sigma`` (1.4826 x MAD); value and sigma are NaN below
    ``min_stations``."""
    station_mag = cal.station_magnitudes(obs)
    frame = pd.DataFrame({"eventId": obs["eventId"].astype(str).to_numpy(), "m": station_mag})
    frame = frame[np.isfinite(frame["m"].to_numpy(dtype=np.float64))]
    grouped = {str(k): g["m"].to_numpy(dtype=np.float64) for k, g in frame.groupby("eventId")}
    rows: list[dict[str, Any]] = []
    for eid in pd.unique(obs["eventId"].astype(str)):
        m = grouped.get(eid, np.empty(0))
        n = int(m.size)
        if n >= min_stations:
            med = float(np.median(m))
            sigma = float(NMAD_FACTOR * np.median(np.abs(m - med)))
        else:
            med = sigma = math.nan
        rows.append({"eventId": eid, "nStations": n, "value": med, "sigma": sigma})
    return pd.DataFrame(rows, columns=["eventId", "nStations", "value", "sigma"])


def leave_one_event_out(obs: pd.DataFrame, cfg: MagnitudeConfig) -> pd.DataFrame:
    """Per calibration event: ``catalogMag``, ``looMag`` (NaN when the refit leaves it fewer than
    ``minStations`` stations with a term), ``nStations``, ``looError`` = looMag - catalogMag,
    and ``nullError``: the mean catalog magnitude of the other events minus catalogMag."""
    _check_obs(obs, need_mag=True)
    rows: list[dict[str, Any]] = []
    event_ids = obs["eventId"].astype(str)
    catalog = obs.groupby(event_ids, sort=False)["mag"].first()
    for eid in pd.unique(event_ids):
        held = obs[event_ids == eid]
        fold = fit_calibration(obs[event_ids != eid], cfg.fit)
        pred = event_magnitudes(fold, held, cfg.minStations).iloc[0]
        cat = float(catalog[eid])
        rows.append(
            {
                "eventId": eid,
                "catalogMag": cat,
                "looMag": float(pred["value"]),
                "nStations": int(pred["nStations"]),
                "looError": float(pred["value"]) - cat,
                "nullError": float(catalog.drop(eid).mean()) - cat,
            }
        )
    return pd.DataFrame(
        rows, columns=["eventId", "catalogMag", "looMag", "nStations", "looError", "nullError"]
    )

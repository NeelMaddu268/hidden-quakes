"""Stage ``magnitude`` (MAG-01): calibrated local magnitude -> ``magnitude.json``, ``events.parquet``.

Reads ``events.parquet`` (stage ``tier``), ``arrivals.parquet``, ``stations.parquet``,
``matches.parquet``, ``catalog.parquet`` and the waveform cache (``ctx.cache_dir``), then:

1. screens every station in ``arrivals.parquet`` once (``hq.magnitude.amplitude.screen_station``:
   cached data, one StationXML epoch per horizontal, ground-motion units, response rate equal to
   the data rate, ``preFiltHz`` below Nyquist); excluded stations are logged and recorded;
2. measures the Wood-Anderson S-window amplitude of every event at every usable station
   (``measure_amplitudes``, reading each station's windows in ``readChunkS`` groups);
3. calibrates on the matched public regional catalog events of ``calibrationMagType`` only (every
   other type is counted and left out) with at least ``minStations`` usable amplitudes, and
   refits without each of them for the leave-one-event-out MAE (``hq.magnitude.calibrate``);
4. writes ``magnitude.json`` (``MagCalibration``: exactly ``n``, ``looMae``, ``coefficients``) and
   ``events.parquet`` with ``magnitude`` = (value, ``ML_cal``, sigma) for events with at least
   ``minStations`` usable station magnitudes, only when ``looMae <= maxLooMae``. Otherwise (the
   docs/03 kill switch) every magnitude is null and the stage logs it as an error. Every other
   ``events.parquet`` column is kept identical (checked on the written file with pyarrow before it
   replaces the old one). Both files are written and checked under ``.part`` names; then the old
   ``magnitude.json`` is removed, ``events.parquet`` moved into place, then ``magnitude.json``
   (two moves, not one atomic step: a crash between them leaves no ``magnitude.json``).

On any failure (too few calibration events, stale matches, a missing column, ...) the stage
removes ``magnitude.json``, sets every ``events.parquet`` magnitude to null, records the failure
under ``ProcessingRun.matching["magnitude"]["failed"]`` and raises again, so no calibration or
magnitude from an earlier run stays next to tables it was not computed from.

``n`` is the number of calibration events with a leave-one-event-out prediction (the events the
MAE averages over). ``ctx.record`` gets the counts and, under ``ProcessingRun.matching["magnitude"]``
(the run model has no magnitude field; the calibration uses the matches), every knob, the
preprocessing, per-station screening and amplitude counts, station terms, the per-event LOO
table, the gate and the runtimes.

The package attribute ``hq.magnitude.run`` is this module's ``run`` function (the stage registry
resolves it there); reach the module with ``importlib.import_module("hq.magnitude.run")``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hq_contracts.io import from_frame, read_table, write_table
from hq_contracts.models import MagCalibration, SeismicEvent

from hq.locate.provenance import stage_provenance
from hq.magnitude.amplitude import (
    AMPLITUDE_UNITS,
    DISTANCE_UNITS,
    LOW_SNR,
    NATIVE_OUTPUT,
    OK,
    MagnitudeError,
    measure_amplitudes,
    plan_windows,
    screen_station,
)
from hq.magnitude.calibrate import (
    NMAD_FACTOR,
    event_magnitudes,
    fit_calibration,
    leave_one_event_out,
)

if TYPE_CHECKING:
    from hq.runs import RunContext

log = logging.getLogger(__name__)

STAGE = "magnitude"
MAG_TYPE = "ML_cal"  # docs/02 Magnitude.type for this lane's calibrated magnitude
RECORD_FIELD = "matching"  # ProcessingRun has no magnitude field; params nest under RECORD_KEY
RECORD_KEY = "magnitude"
EVENTS_TABLE = "events.parquet"
ARRIVALS_TABLE = "arrivals.parquet"
STATIONS_TABLE = "stations.parquet"
MATCHES_TABLE = "matches.parquet"
CATALOG_TABLE = "catalog.parquet"
MAGNITUDE_JSON = "magnitude.json"
PART_SUFFIX = ".part"
MAG_COLUMNS: tuple[str, ...] = ("magnitude_value", "magnitude_type", "magnitude_sigma")
CENSORING_BINS = 10  # ML_cal deciles in the record's censoring table (a summary, not a knob)
INPUT_MODELS = {
    "events": "SeismicEvent",
    "arrivals": "Arrival",
    "stations": "Station",
    "matches": "Match",
    "catalog": "CatalogEvent",
}


def _part(path: Path) -> Path:
    return path.with_name(path.name + PART_SUFFIX)


def _read(path: Path, what: str) -> pd.DataFrame:
    frame = read_table(path)
    model = frame.attrs.get("model")
    if model != INPUT_MODELS[what]:
        raise MagnitudeError(
            f"{path} holds {model!r} rows; the {what} input must hold {INPUT_MODELS[what]!r} rows"
        )
    return frame


def calibration_events(
    events: pd.DataFrame, matches: pd.DataFrame, catalog: pd.DataFrame, mag_type: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Matched events whose public regional catalog magnitude has type ``mag_type``: columns
    ``eventId``, ``catalogId``, ``mag``. Checks ``matches`` against ``events.parquet``'s
    ``catalogMatch`` (stale tables fail) and counts the matched events by catalog type."""
    matched = matches[matches["eventId"].notna()]
    pairs = pd.DataFrame(
        {
            "eventId": matched["eventId"].astype(str).to_numpy(),
            "catalogId": matched["catalogId"].astype(str).to_numpy(),
        }
    )
    in_events = events[events["catalogMatch_catalogId"].notna()]
    stored = dict(
        zip(
            in_events["id"].astype(str),
            in_events["catalogMatch_catalogId"].astype(str),
            strict=True,
        )
    )
    expected = dict(zip(pairs["eventId"], pairs["catalogId"], strict=True))
    if stored != expected:
        raise MagnitudeError(
            f"{MATCHES_TABLE} ({len(expected)} pairs) disagrees with events.parquet catalogMatch "
            f"({len(stored)} pairs); rerun stage tier after match"
        )
    cat = pd.DataFrame(
        {
            "catalogId": catalog["id"].astype(str).to_numpy(),
            "mag": catalog["mag"].to_numpy(dtype=np.float64),
            "magType": catalog["magType"].astype(object).to_numpy(),
        }
    )
    missing = sorted(set(pairs["catalogId"]) - set(cat["catalogId"]))
    if missing:
        raise MagnitudeError(f"matched catalog ids not in {CATALOG_TABLE}: {missing}")
    joined = pairs.merge(cat, on="catalogId", how="left")
    types = joined["magType"].where(joined["magType"].notna(), "null").astype(str)
    by_type = {str(k): int(v) for k, v in types.value_counts().sort_index().items()}
    use = joined[(joined["magType"] == mag_type) & np.isfinite(joined["mag"])]
    info = {
        "matchedEvents": len(joined),
        "matchedByCatalogMagType": by_type,
        "withCalibrationMagType": len(use),
        "otherTypesLeftOut": len(joined) - len(use),
        # the configured type has at least as many matched events as any other type
        "isMostMatchedType": bool(by_type) and by_type.get(mag_type, 0) == max(by_type.values()),
    }
    if not info["isMostMatchedType"]:
        log.warning(
            "magnitude: calibrationMagType %r is not the most matched catalog type %s",
            mag_type, by_type,
        )  # fmt: skip
    return use[["eventId", "catalogId", "mag"]].reset_index(drop=True), info


def censoring_summary(mags: pd.DataFrame, table: pd.DataFrame) -> dict[str, Any]:
    """Station support of the event magnitudes by ML_cal decile (``CENSORING_BINS`` equal-count
    bins): usable station magnitudes per event (median, min) and the median share of the event's
    measured windows (``ok`` + ``lowSnr``) that fell below ``minSnr``."""
    have = mags[np.isfinite(mags["value"].to_numpy(dtype=np.float64))]
    measured = table[table["status"].isin([OK, LOW_SNR])]
    low_share = (measured["status"] == LOW_SNR).groupby(measured["eventId"]).mean()
    rows: list[dict[str, Any]] = []
    if len(have):
        have = have.assign(lowSnrShare=have["eventId"].map(low_share).to_numpy(dtype=np.float64))
        bins = pd.qcut(have["value"], CENSORING_BINS, labels=False, duplicates="drop")
        for _, g in have.groupby(bins.to_numpy(), sort=True):
            rows.append(
                {
                    "mlCal": [round(float(g["value"].min()), 3), round(float(g["value"].max()), 3)],
                    "events": len(g),
                    "nStationsMedian": float(g["nStations"].median()),
                    "nStationsMin": int(g["nStations"].min()),
                    "lowSnrShareMedian": round(float(g["lowSnrShare"].median()), 3),
                }
            )
    return {
        "note": "only windows with SNR >= minSnr give station magnitudes, so near the detection "
        "limit an event's magnitude rests on the stations whose amplitude happened to clear the "
        "noise: fewer stations, biased upward. Read belowCalibratedRange and the low-magnitude "
        "end of any G-R curve or b-value with this table",
        "byDecile": rows,
    }


def _with_magnitudes(events: pd.DataFrame, mags: pd.DataFrame | None) -> pd.DataFrame:
    """``events`` with every magnitude reset to null, then set from ``mags`` (eventId, value,
    sigma; NaN value = no magnitude) when given. Column dtypes are kept."""
    out = events.copy()
    out["magnitude_value"] = np.nan
    out["magnitude_sigma"] = np.nan
    out["magnitude_type"] = pd.array([pd.NA] * len(out), dtype=events["magnitude_type"].dtype)
    if mags is not None:
        have = mags[np.isfinite(mags["value"].to_numpy(dtype=np.float64))]
        pos = pd.Index(out["id"].astype(str)).get_indexer(have["eventId"])
        if (pos < 0).any():
            raise MagnitudeError("a magnitude names an event not in events.parquet")
        value_col = out.columns.get_loc("magnitude_value")
        sigma_col = out.columns.get_loc("magnitude_sigma")
        type_col = out.columns.get_loc("magnitude_type")
        out.iloc[pos, value_col] = have["value"].to_numpy(dtype=np.float64)
        out.iloc[pos, sigma_col] = have["sigma"].to_numpy(dtype=np.float64)
        out.iloc[pos, type_col] = MAG_TYPE
    for col in MAG_COLUMNS:
        out[col] = out[col].astype(events[col].dtype)
    return out


def check_round_trip(original: Path, written: Path, expected: pd.DataFrame) -> None:
    """Every column of ``written`` except the magnitude ones equals ``original`` exactly (Arrow
    types and values, file metadata included); the magnitude columns hold ``expected``'s."""
    old = pq.read_table(original)
    new = pq.read_table(written)
    if old.schema.names != new.schema.names:
        raise MagnitudeError(f"{written}: columns {new.schema.names} != {old.schema.names}")
    if not old.schema.equals(new.schema, check_metadata=False):
        raise MagnitudeError(f"{written}: column types changed: {new.schema} vs {old.schema}")
    for key in (b"schemaVersion", b"model"):
        if (old.schema.metadata or {}).get(key) != (new.schema.metadata or {}).get(key):
            raise MagnitudeError(f"{written}: file metadata {key!r} changed")
    # pandas' per-column metadata rebuilds the dtypes on read; only the magnitude columns may
    # change there (an all-null column is described differently from a filled one).
    old_pd, new_pd = (
        {
            c["name"]: c
            for c in json.loads((t.schema.metadata or {}).get(b"pandas", b"{}"))["columns"]
            if c["name"] not in MAG_COLUMNS
        }
        for t in (old, new)
    )
    if old_pd != new_pd:
        raise MagnitudeError(f"{written}: pandas column metadata changed outside magnitude")
    changed = [
        name
        for name in old.schema.names
        if name not in MAG_COLUMNS and not old.column(name).equals(new.column(name))
    ]
    if changed:
        raise MagnitudeError(f"{written}: columns other than magnitude changed: {changed}")
    back = read_table(written)
    for col in ("magnitude_value", "magnitude_sigma"):
        a = back[col].to_numpy(dtype=np.float64)
        b = expected[col].to_numpy(dtype=np.float64)
        if not np.array_equal(a, b, equal_nan=True):
            raise MagnitudeError(f"{written}: {col} did not round-trip")
    if not back["magnitude_type"].astype(object).fillna("").equals(
        expected["magnitude_type"].astype(object).fillna("")
    ):
        raise MagnitudeError(f"{written}: magnitude_type did not round-trip")


def _write(
    ctx: RunContext, events_path: Path, events_out: pd.DataFrame, calibration: MagCalibration
) -> None:
    json_path = ctx.path(MAGNITUDE_JSON)
    targets = [events_path, json_path]
    try:
        write_table(events_out, _part(events_path), SeismicEvent.__name__)
        check_round_trip(events_path, _part(events_path), events_out)
        text = json.dumps(calibration.model_dump(mode="json"), indent=2, allow_nan=False)
        _part(json_path).write_text(text + "\n", encoding="utf-8")
        MagCalibration.model_validate_json(_part(json_path).read_text(encoding="utf-8"))
        # Not atomic as a pair: the old magnitude.json goes first, so a crash between the two
        # moves leaves the new events.parquet with no magnitude.json, never with the old one.
        json_path.unlink(missing_ok=True)
        os.replace(_part(events_path), events_path)
        os.replace(_part(json_path), json_path)
    finally:
        for path in targets:
            _part(path).unlink(missing_ok=True)


def _clear_outputs(ctx: RunContext, error: BaseException, runtime_s: float) -> None:
    """After a failure: no magnitude.json, no magnitude in events.parquet, and a run record that
    says the stage failed, so nothing from an earlier run outlives the tables it came from."""
    json_path = ctx.path(MAGNITUDE_JSON)
    removed = json_path.is_file()
    json_path.unlink(missing_ok=True)
    events_path = ctx.path(EVENTS_TABLE)
    nulled = 0
    if events_path.is_file():
        events = read_table(events_path)
        nulled = int(events["magnitude_value"].notna().sum())
        if nulled:
            events_out = _with_magnitudes(events, None)
            try:
                write_table(events_out, _part(events_path), SeismicEvent.__name__)
                check_round_trip(events_path, _part(events_path), events_out)
                os.replace(_part(events_path), events_path)
            finally:
                _part(events_path).unlink(missing_ok=True)
    log.error(
        "magnitude: stage failed (%s: %s); %s, %d events.parquet magnitudes set to null",
        type(error).__name__, error,
        f"removed {MAGNITUDE_JSON}" if removed else f"no {MAGNITUDE_JSON} to remove", nulled,
    )  # fmt: skip
    ctx.record(
        STAGE,
        runtime_s=runtime_s,
        counts={"magnitudes": 0, "failed": 1},
        params={
            RECORD_KEY: {
                "failed": f"{type(error).__name__}: {error}",
                "outputs": f"{MAGNITUDE_JSON} removed and every {EVENTS_TABLE} magnitude null",
            }
        },
        field=RECORD_FIELD,
    )


def _preprocessing_record(ctx: RunContext) -> dict[str, Any]:
    cfg = ctx.config.seismology.magnitude
    wa = cfg.woodAnderson
    return {
        "amplitude": "peak of the horizontal vector amplitude sqrt(h1^2 + h2^2) of the simulated "
        "Wood-Anderson trace in the S window [tS - sPreS, tS + sPostS]; rotation-invariant, so "
        "borehole 1/2 components of unknown azimuth need no rotation",
        "amplitudeUnits": f"{AMPLITUDE_UNITS} (Wood-Anderson trace amplitude)",
        "distance": "hypocentral, from the located hypocentre's ENU to the sensor's ENU "
        "(Station.enu, sensorElevM included)",
        "distanceUnits": DISTANCE_UNITS,
        "anchors": "tP / tS: the observed pick when usedInLocation, else arrivals.tPred",
        "noiseWindow": "[tP - noiseGapS - noiseLenS, tP - noiseGapS]",
        "snr": "S-window peak / noise-window peak of the same processed vector trace",
        "responseRemoval": {
            "evalrespOutputByInputUnits": {u: out for u, (out, _) in NATIVE_OUTPUT.items()},
            "integrationsToDisplacement": {u: k for u, (_, k) in NATIVE_OUTPUT.items()},
            "outputUnits": "m (ground displacement)",
            "method": "linear detrend, cosine taper over padS at each end, rfft (nfft = next "
            "power of two >= 2 npts), x ObsPy cosine_sac_taper(preFiltHz), x inverted evalresp "
            "response in the sensor's own input units (ObsPy invert_spectrum(waterLevelDb): the "
            "water level is waterLevelDb below that native-unit response's maximum), "
            "/ (2 pi i f)^k to ground displacement",
            "band": "flat between preFiltHz f2 and f3, cosine tapers to 0 at f1 and f4, then the "
            "Wood-Anderson response (flat to displacement above 1/periodS, falling as f^2 below); "
            "the same at every station except inside amplitudes.perStation.*.waterLevelClipHz, "
            "where the water level caps the inverse response",
            "preFiltHz": list(cfg.response.preFiltHz),
            "waterLevelDb": cfg.response.waterLevelDb,
        },
        "saturation": {
            "rule": "status clipped when the raw |counts| of either horizontal reach maxFraction "
            "x fullScaleCounts anywhere in the processed span",
            "fullScaleCounts": cfg.saturation.fullScaleCounts,
            "maxFraction": cfg.saturation.maxFraction,
        },
        "woodAnderson": {
            "response": "gain * s^2 / (s^2 + 2 h w0 s + w0^2), w0 = 2 pi / periodS, applied to "
            "ground displacement",
            "periodS": wa.periodS,
            "damping": wa.damping,
            "gain": wa.gain,
            "outputUnits": AMPLITUDE_UNITS,
        },
    }


def run(ctx: RunContext) -> None:
    """Stage ``magnitude`` (docs/02 §4); see the module docstring. On any failure the outputs of
    an earlier run are cleared (``_clear_outputs``) and the error is raised again."""
    started = time.perf_counter()
    try:
        _run(ctx, started)
    except Exception as error:
        _clear_outputs(ctx, error, time.perf_counter() - started)
        raise


def _run(ctx: RunContext, started: float) -> None:
    cfg = ctx.config.seismology.magnitude
    run_section = ctx.config.run
    events_path = ctx.path(EVENTS_TABLE)
    events = _read(events_path, "events")
    arrivals = _read(ctx.path(ARRIVALS_TABLE), "arrivals")
    stations = _read(ctx.path(STATIONS_TABLE), "stations")
    matches = _read(ctx.path(MATCHES_TABLE), "matches")
    catalog = _read(ctx.path(CATALOG_TABLE), "catalog")
    log.info(
        "magnitude: %d events, %d arrivals, %d stations, %d matches rows, %d catalog events",
        len(events), len(arrivals), len(stations), len(matches), len(catalog),
    )  # fmt: skip

    cal_events, cal_info = calibration_events(events, matches, catalog, cfg.calibrationMagType)
    log.info(
        "magnitude: matched events by public regional catalog magType %s; calibrating on %r "
        "only (%d events, %d of other types left out)",
        cal_info["matchedByCatalogMagType"], cfg.calibrationMagType,
        cal_info["withCalibrationMagType"], cal_info["otherTypesLeftOut"],
    )  # fmt: skip

    arrival_stations = set(arrivals["stationId"].astype(str))
    used = stations[stations["id"].astype(str).isin(arrival_stations)].reset_index(drop=True)
    plan = plan_windows(events, arrivals, used, cfg)
    window = (run_section.window_start_s, run_section.window_end_s)
    screens = {
        str(row["id"]): screen_station(row, cfg, window, cache_dir=ctx.cache_dir)
        for _, row in used.iterrows()
    }
    for screen in screens.values():
        if not screen.usable:
            log.warning("magnitude: station %s excluded: %s", screen.stationId, screen.reason)
        for flag in screen.flags:
            log.warning("magnitude: station %s flagged: %s", screen.stationId, flag)
    amps = measure_amplitudes(plan, screens, cfg, cache_dir=ctx.cache_dir)
    table = amps.table
    usable = table[table["status"] == OK]

    obs = usable.merge(cal_events, on="eventId", how="inner")
    per_event = obs.groupby("eventId").size()
    enough = set(per_event.index[per_event >= cfg.minStations])
    too_few = len(cal_events) - len(enough)
    obs = obs[obs["eventId"].isin(enough)].reset_index(drop=True)
    if too_few:
        log.warning(
            "magnitude: %d %r calibration events have fewer than %d usable station amplitudes "
            "and are left out", too_few, cfg.calibrationMagType, cfg.minStations,
        )  # fmt: skip
    if len(enough) < cfg.minCalibrationEvents:
        raise MagnitudeError(
            f"{len(enough)} calibration events with at least {cfg.minStations} usable amplitudes, "
            f"fewer than minCalibrationEvents {cfg.minCalibrationEvents}"
        )
    fit_started = time.perf_counter()
    final = fit_calibration(obs, cfg.fit)
    loo = leave_one_event_out(obs, cfg)
    fit_runtime = time.perf_counter() - fit_started
    predicted = loo[np.isfinite(loo["looMag"].to_numpy(dtype=np.float64))]
    if len(predicted) < len(loo):
        log.warning(
            "magnitude: %d calibration events got no leave-one-out prediction (fewer than %d "
            "stations with a term once refitted without them); looMae averages the other %d",
            len(loo) - len(predicted), cfg.minStations, len(predicted),
        )  # fmt: skip
    if len(predicted) < cfg.minCalibrationEvents:
        raise MagnitudeError(
            f"only {len(predicted)} calibration events have a leave-one-out prediction, fewer "
            f"than minCalibrationEvents {cfg.minCalibrationEvents}"
        )
    loo_mae = float(predicted["looError"].abs().mean())
    calibration = MagCalibration(
        n=len(predicted), looMae=loo_mae, coefficients=final.coefficients()
    )
    passed = loo_mae <= cfg.maxLooMae

    mags = event_magnitudes(final, usable, cfg.minStations)
    mags = pd.DataFrame({"eventId": events["id"].astype(str)}).merge(mags, on="eventId", how="left")
    mags["nStations"] = mags["nStations"].fillna(0).astype(int)
    with_mag = int(np.isfinite(mags["value"].to_numpy(dtype=np.float64)).sum())
    without_mag = len(events) - with_mag
    if passed:
        events_out = _with_magnitudes(events, mags)
        log.info(
            "magnitude: LOO MAE %.3f <= maxLooMae %.3f (n=%d): %s written for %d of %d events; "
            "%d events have fewer than %d usable station magnitudes and stay null",
            loo_mae, cfg.maxLooMae, calibration.n, MAG_TYPE, with_mag, len(events),
            without_mag, cfg.minStations,
        )  # fmt: skip
    else:
        events_out = _with_magnitudes(events, None)
        log.error(
            "magnitude: LOO MAE %.3f > maxLooMae %.3f (n=%d): docs/03 magnitude kill switch; "
            "every events.parquet magnitude left null (%d events would have had one); "
            "magnitude.json still written",
            loo_mae, cfg.maxLooMae, calibration.n, with_mag,
        )  # fmt: skip
    from_frame(events_out, SeismicEvent)  # every row still validates against the contract
    _write(ctx, events_path, events_out, calibration)

    written = int(events_out["magnitude_value"].notna().sum())
    cal_range = (float(obs["mag"].min()), float(obs["mag"].max()))
    values = mags["value"].to_numpy(dtype=np.float64)
    below_range = int((values < cal_range[0]).sum())  # NaN compares False
    above_range = int((values > cal_range[1]).sum())
    log.info(
        "magnitude: %d of %d event magnitudes lie below the calibrated %r range [%.2f, %.2f] and "
        "%d above it: extrapolated (a %s)",
        below_range, with_mag, cfg.calibrationMagType, *cal_range, above_range,
        f"fixed at {cfg.fit.amplitudeSlope}" if final.aFixed else "fitted",
    )  # fmt: skip
    insample = event_magnitudes(final, obs, cfg.minStations)
    loo_table = loo.merge(cal_events[["eventId", "catalogId"]], on="eventId", how="left").merge(
        insample[["eventId", "value", "sigma"]].rename(
            columns={"value": "fitMag", "sigma": "fitSigma"}
        ),
        on="eventId",
        how="left",
    )
    errors = predicted["looError"].to_numpy(dtype=np.float64)
    screens_record = {sid: s.record() for sid, s in sorted(screens.items())}
    excluded = {sid: s.reason for sid, s in sorted(screens.items()) if not s.usable}
    runtime_s = time.perf_counter() - started
    counts = {
        "events": len(events),
        "magnitudes": written,
        "eventsBelowMinStations": without_mag,
        "calibrationEvents": len(enough),
        "looPredicted": calibration.n,
        "stationsUsed": sum(1 for s in screens.values() if s.usable),
        "stationsExcluded": len(excluded),
        "windows": len(table),
        "windowsUsable": len(usable),
        "gatePassed": int(passed),
    }
    params: dict[str, Any] = {
        "config": cfg.model_dump(mode="json"),
        "model": "M_cat = a log10(A) + b log10(R) + c + s_station; event magnitude = median of "
        "station magnitudes",
        "type": MAG_TYPE,
        "calibrationMagType": cfg.calibrationMagType,
        "calibrationMagTypeReason": "the public regional catalog carries several magnitude "
        "types; only one is calibrated on (never compared across types). The configured type is "
        "chosen for measuring what this model measures (ml: a peak amplitude; md: a duration); "
        "calibrationSet.isMostMatchedType says whether it also has the most matched events",
        "calibrationSet": {**cal_info, "withEnoughStations": len(enough), "tooFewStations": too_few},
        "fit": {
            "method": "scipy.optimize.least_squares from the ordinary (ridge) least-squares "
            "start; the robust loss acts on the observations only",
            "loss": cfg.fit.loss,
            "fScaleMag": cfg.fit.fScaleMag,
            "aFixed": final.aFixed,
            "stationTermConstraint": f"ridge: stationTermRidge ({cfg.fit.stationTermRidge}) x "
            "sum of squared station terms, quadratic; with the free intercept c the terms also "
            "sum to zero at the optimum",
            "bIdentification": {
                **final.distanceSpread,
                "note": "b is not identified by a clustered calibration set: with free station "
                "terms only withinStationLogRSd informs it, and stationTermRidge decides how much "
                "of betweenStationLogRSd is credited to b instead of to the terms, so b follows "
                "that knob and the loss as much as the data. The leave-one-event-out MAE does not "
                "test b (the held-out events share the others' source region); magnitudes of "
                "events far from the calibration events depend on it",
            },
            "coefficients": final.coefficients(),
            "stationTerms": final.stationTerms,
            "stationsWithoutTerm": final.stationsWithoutTerm,
            "nObs": final.nObs,
            "nEvents": final.nEvents,
            "robust": final.robust,
        },
        "sigma": f"{NMAD_FACTOR} x median absolute deviation of the event's station magnitudes "
        "(normalized MAD, a robust standard deviation)",
        "leaveOneEventOut": {
            "n": calibration.n,
            "looMae": loo_mae,
            "meanError": float(errors.mean()),
            "rmse": float(math.sqrt(float((errors**2).mean()))),
            "maxAbsError": float(np.abs(errors).max()),
            # Predicting every event as the mean catalog magnitude of the others: the MAE a
            # model with no skill gets on this calibration set (read looMae against it).
            "nullModelMae": float(predicted["nullError"].abs().mean()),
            "events": [
                {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                 for k, v in row.items()}
                for row in loo_table.to_dict("records")
            ],
        },
        "gate": {
            "maxLooMae": cfg.maxLooMae,
            "looMae": loo_mae,
            "passed": passed,
            "action": "magnitudes written" if passed else "every magnitude left null (docs/03)",
        },
        "magnitudes": {
            "written": written,
            "withEnoughStations": with_mag,
            "belowMinStations": without_mag,
            "minStations": cfg.minStations,
            # catalog magnitudes of the calibration events; values outside are extrapolated
            "calibratedRange": list(cal_range),
            "belowCalibratedRange": below_range,
            "aboveCalibratedRange": above_range,
            "censoring": censoring_summary(mags, table),
        },
        "preprocessing": _preprocessing_record(ctx),
        "stations": screens_record,
        "stationsExcluded": excluded,
        "amplitudes": amps.record,
        "outputs": [EVENTS_TABLE, MAGNITUDE_JSON],
        "provenance": stage_provenance(),
        "runtimeS": {
            "amplitudes": amps.record["runtimeS"],
            "fitAndLoo": round(fit_runtime, 3),
            "total": round(runtime_s, 3),
        },
    }
    log.info(
        "magnitude: calibration %r n=%d a=%.3f b=%.3f c=%.3f LOO MAE %.3f (null model %.3f, gate "
        "%s); %d magnitudes written; %d stations excluded %s; %.1f s",
        cfg.calibrationMagType, calibration.n, final.a, final.b, final.c, loo_mae,
        params["leaveOneEventOut"]["nullModelMae"], "passed" if passed else "FAILED", written,
        len(excluded), sorted(excluded), runtime_s,
    )  # fmt: skip
    ctx.record(
        STAGE, runtime_s=runtime_s, counts=counts, params={RECORD_KEY: params}, field=RECORD_FIELD
    )

"""Synthetic recovery test: can this station geometry and locator recover known hypocentres?

``nEvents`` hypocentres are drawn uniformly in the configured zone (a vertical cylinder in ENU
metres and elevM around the run origin; config values only, never catalog values) with origin
times uniform in the run window. Every station gets a P pick; its S pick is kept with probability
``sKeepProb``. Pick times are the exact 1D layered first-arrival times
(``tt_grid.layered_first_arrival``, an independent forward model, so table discretisation error
is part of what the test measures) plus Gaussian noise with the standard deviation the locator
uses for that station and phase (``Locator.pick_sigma``: ``pickSigmaS``, or the station's
``profilePickSigmaS`` override). Every pick gets probability ``pickProb``. In seismology.yaml both
may be null: stage ``locate`` then measures them from the run's located events
(``measured_pick_stats``: median nS / nStations and median used-pick probability) and runs the
test with those values (``with_pick_stats``); ``run_synthetic`` itself needs numbers. Every
station still gets a P pick, so the synthetic picks stay optimistic for weak events, and the
params say so. The same events are located twice, with and without noise (same S picks kept).
One seeded generator draws everything in a fixed order, so identical config gives identical
results.

Errors against the truth: horizontal ``hypot(de, dn)``, vertical ``|dElevM|``, and depth bias
``elevM_true - elevM_located`` (positive = located too deep; docs/02 ``SyntheticTest`` carries no
sign, so the params record it). Formal-error calibration is the fraction of events whose true
horizontal (vertical) error is within ``hErrM`` (``vErrM``), over events whose PDF is not
truncated. Compare Tier A formal ``vErrM`` with ``params["noisy"]["medianFormalVErrM"]`` (a formal
error), not with ``medianVErrM`` (the median true error).

``synthetic.json`` holds exactly the docs/02 ``SyntheticTest`` fields (``hq_contracts``); everything
else goes into the params dict (for ``ctx.record``) and the log, including the station geometry
(ids, sensor positions, a hash of them and a caller-supplied label) and the locator record, so a
report can be traced to the geometry and tables it used. docs/00 licenses the "resolves depth to
about +/- N m" claim only for a run on the real station geometry.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hq_contracts.models import SyntheticTest

from hq.config.seismology import SeismologyConfig
from hq.locate.locator import EventLocation, Locator, LocatorSetup, build_locator, locate_many
from hq.locate.tt_grid import PHASES, layered_first_arrival

log = logging.getLogger(__name__)

DEPTH_BIAS_SIGN = "elevM_true - elevM_located; positive = located too deep"
GEOMETRY_COLUMNS = ("id", "enu_e", "enu_n", "sensorElevM")


def pick_probabilities(cfg: SeismologyConfig) -> tuple[float, float]:
    """(sKeepProb, pickProb) from ``cfg.synthetic``; fails when either is still null."""
    syn = cfg.synthetic
    if syn.sKeepProb is None or syn.pickProb is None:
        raise ValueError(
            "synthetic.sKeepProb and synthetic.pickProb must be numbers here; null means stage "
            "locate measures them from the run (measured_pick_stats, with_pick_stats)"
        )
    return float(syn.sKeepProb), float(syn.pickProb)


def measured_pick_stats(events: pd.DataFrame, picks: pd.DataFrame) -> dict[str, Any]:
    """The run's own S-pick fraction and used-pick probability, for the synthetic test.

    ``events``: ``events_located`` rows (``quality_nS``, ``quality_nStations``, ``pickIds``);
    ``picks``: the picks table they were located from (``id``, ``prob``). ``sKeepProb`` is the
    median over events of nS / nStations (S picks per station with a used pick) and ``pickProb``
    the median ``prob`` of the picks used in the final locations. Fails with no located event.
    """
    if events.empty:
        raise ValueError(
            "no located events to measure synthetic.sKeepProb / pickProb from; set numbers in "
            "seismology.yaml (synthetic) to run the synthetic test on this run"
        )
    n_s = events["quality_nS"].to_numpy(dtype=np.float64)
    n_st = events["quality_nStations"].to_numpy(dtype=np.float64)
    used = [str(p) for ids in events["pickIds"] for p in ids]
    prob = picks.assign(id=picks["id"].astype(str)).set_index("id")["prob"]
    missing = sorted(set(used) - set(prob.index))
    if missing:
        raise ValueError(f"used pick ids missing from the picks table: {missing[:5]}")
    probs = prob.loc[used].to_numpy(dtype=np.float64)
    return {
        "sKeepProb": float(np.median(n_s / n_st)),
        "pickProb": float(np.median(probs)),
        "from": "located events of this run: sKeepProb = median over events of nS / nStations, "
        "pickProb = median prob of the picks used in the final locations",
        "nEvents": len(events),
        "nUsedPicks": len(used),
        "medianNS": float(np.median(n_s)),
        "medianNStations": float(np.median(n_st)),
        "medianNPPerStation": float(
            np.median(events["quality_nP"].to_numpy(dtype=np.float64) / n_st)
        ),
    }


def with_pick_stats(cfg: SeismologyConfig, stats: dict[str, Any]) -> SeismologyConfig:
    """``cfg`` with each null ``synthetic.sKeepProb`` / ``pickProb`` set from ``stats``."""
    syn = cfg.synthetic
    update = {
        key: stats[key] for key in ("sKeepProb", "pickProb") if getattr(syn, key) is None
    }
    if not update:
        return cfg
    raw = cfg.model_dump(mode="json")
    raw["synthetic"].update(update)
    return SeismologyConfig.model_validate(raw)


def geometry_record(stations: pd.DataFrame, label: str | None) -> dict[str, Any]:
    """Station ids, sensor positions and a SHA-256 of them, plus the caller's geometry label."""
    rows = [
        [str(sid), float(e), float(n), float(z)]
        for sid, e, n, z in stations[list(GEOMETRY_COLUMNS)].itertuples(index=False)
    ]
    text = json.dumps(rows, separators=(",", ":"), allow_nan=False)
    return {
        "label": label,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "hashOf": "JSON list of [id, enu_e, enu_n, sensorElevM] per station, in frame order",
        "stationIds": [r[0] for r in rows],
        "enuEM": [r[1] for r in rows],
        "enuNM": [r[2] for r in rows],
        "sensorElevM": [r[3] for r in rows],
        "nBorehole": int((stations["kind"] == "borehole").sum())
        if "kind" in stations.columns
        else None,
    }


@dataclass(frozen=True, eq=False)
class SyntheticResult:
    report: SyntheticTest
    params: dict[str, Any]  # everything else, for ctx.record and the log
    events: pd.DataFrame  # one row per event: truth, noisy and noise-free locations


def draw_hypocentres(
    setup: LocatorSetup, n_events: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Truth hypocentres (e, n, elevM) and origin times, drawn in the configured zone."""
    zone = setup.config.synthetic.zone
    radius = zone.radiusM * np.sqrt(rng.random(n_events))
    angle = rng.random(n_events) * 2.0 * np.pi
    elev = zone.bottomElevM + rng.random(n_events) * (zone.topElevM - zone.bottomElevM)
    run = setup.run
    t0 = run.window_start_s + rng.random(n_events) * (run.window_end_s - run.window_start_s)
    return pd.DataFrame(
        {
            "e": zone.centerEM + radius * np.sin(angle),
            "n": zone.centerNM + radius * np.cos(angle),
            "elevM": elev,
            "t0": t0,
        }
    )


def synthetic_picks(
    setup: LocatorSetup, locator: Locator, truth: pd.DataFrame, rng: np.random.Generator
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Noisy and noise-free pick frames per event (same kept S picks in both).

    The forward model is the tables' top-extended model; pick noise uses the locator's sigma for
    each station and phase.
    """
    cfg = setup.config
    stations = setup.stations
    model = locator.tables.model
    ids = stations["id"].astype(str).tolist()
    s_keep_prob, pick_prob = pick_probabilities(cfg)
    n_ev, n_st = len(truth), len(ids)
    keep_s = rng.random((n_ev, n_st)) < s_keep_prob
    noise = {
        ph: rng.normal(0.0, 1.0, (n_ev, n_st))
        * np.array([locator.pick_sigma(sid, ph) for sid in ids])[None, :]
        for ph in PHASES
    }
    e = truth["e"].to_numpy()
    n = truth["n"].to_numpy()
    elev = truth["elevM"].to_numpy()
    tt = {}
    for k, row in enumerate(stations.itertuples(index=False)):
        r = np.hypot(e - row.enu_e, n - row.enu_n)
        for ph in PHASES:
            # Reciprocity: source at the sensor, receiver at each hypocentre.
            tt[(k, ph)] = layered_first_arrival(model, ph, float(row.sensorElevM), r, elev)
    noisy: list[pd.DataFrame] = []
    clean: list[pd.DataFrame] = []
    t0 = truth["t0"].to_numpy()
    for i in range(n_ev):
        rows = []
        for k, sid in enumerate(ids):
            for ph in PHASES:
                if ph == "S" and not keep_s[i, k]:
                    continue
                exact = t0[i] + tt[(k, ph)][i]
                rows.append((sid, ph, exact, exact + noise[ph][i, k]))
        frame = pd.DataFrame(rows, columns=["stationId", "phase", "tExact", "tNoisy"])
        base = {
            "id": [f"synthetic:{i}:{s}:{p}" for s, p in zip(frame["stationId"], frame["phase"],
                                                             strict=True)],
            "stationId": frame["stationId"],
            "phase": frame["phase"],
            "prob": pick_prob,
        }
        noisy.append(pd.DataFrame({**base, "t": frame["tNoisy"]}))
        clean.append(pd.DataFrame({**base, "t": frame["tExact"]}))
    return noisy, clean


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _errors(truth: pd.DataFrame, located: list[EventLocation], prefix: str) -> pd.DataFrame:
    e = np.array([loc.e_m for loc in located])
    n = np.array([loc.n_m for loc in located])
    z = np.array([loc.elev_m for loc in located])
    return pd.DataFrame(
        {
            f"{prefix}E": e,
            f"{prefix}N": n,
            f"{prefix}ElevM": z,
            f"{prefix}HErrTrueM": np.hypot(e - truth["e"], n - truth["n"]),
            f"{prefix}VErrTrueM": np.abs(z - truth["elevM"]),
            f"{prefix}DepthBiasM": truth["elevM"].to_numpy() - z,
            f"{prefix}HErrM": [_nan_if_none(loc.h_err_m) for loc in located],  # NaN: truncated
            f"{prefix}VErrM": [_nan_if_none(loc.v_err_m) for loc in located],
            f"{prefix}PdfTruncated": [loc.pdf_truncated for loc in located],
            f"{prefix}DepthOnEdge": [loc.depth_on_edge for loc in located],
            f"{prefix}MapOnVolumeTop": [loc.map_on_volume_top for loc in located],
            f"{prefix}MapOnVolumeBottom": [loc.map_on_volume_bottom for loc in located],
            f"{prefix}NDropped": [len(loc.dropped_pick_ids) for loc in located],
            f"{prefix}NS": [loc.n_s for loc in located],
            f"{prefix}RmsS": [loc.rms_s for loc in located],
        }
    )


def _summary(df: pd.DataFrame, prefix: str) -> dict[str, float | int | None]:
    h = df[f"{prefix}HErrTrueM"]
    v = df[f"{prefix}VErrTrueM"]
    formal = ~df[f"{prefix}PdfTruncated"]  # events with formal errors
    n_formal = int(formal.sum())

    def frac(true_err: pd.Series, formal_err: pd.Series) -> float | None:
        if n_formal == 0:
            return None
        return float((true_err[formal] <= formal_err[formal]).mean())

    def median(values: pd.Series) -> float | None:
        return float(values[formal].median()) if n_formal else None

    return {
        "medianHErrM": float(h.median()),
        "p90HErrM": float(h.quantile(0.9)),
        "medianVErrM": float(v.median()),
        "p90VErrM": float(v.quantile(0.9)),
        "medianDepthBiasM": float(df[f"{prefix}DepthBiasM"].median()),
        "meanDepthBiasM": float(df[f"{prefix}DepthBiasM"].mean()),
        "fracHWithinHErrM": frac(h, df[f"{prefix}HErrM"]),
        "fracVWithinVErrM": frac(v, df[f"{prefix}VErrM"]),
        "medianFormalHErrM": median(df[f"{prefix}HErrM"]),
        "medianFormalVErrM": median(df[f"{prefix}VErrM"]),
        "nPdfTruncated": int(df[f"{prefix}PdfTruncated"].sum()),
        "nDepthOnEdge": int(df[f"{prefix}DepthOnEdge"].sum()),
        "nMapOnVolumeTop": int(df[f"{prefix}MapOnVolumeTop"].sum()),
        "nMapOnVolumeBottom": int(df[f"{prefix}MapOnVolumeBottom"].sum()),
        "nEventsWithDroppedPicks": int((df[f"{prefix}NDropped"] > 0).sum()),
    }


def run_synthetic(
    setup: LocatorSetup,
    *,
    n_events: int | None = None,
    geometry_label: str | None = None,
    pick_stats: dict[str, Any] | None = None,
) -> SyntheticResult:
    """Run the synthetic recovery test on ``setup``'s station geometry and configuration.

    ``geometry_label`` names where the station geometry came from (e.g. an H1 runId, or
    "PROVISIONAL"); it goes into the params with the geometry itself. ``pick_stats`` (from
    ``measured_pick_stats``) is recorded when the config's sKeepProb / pickProb came from it.
    """
    started = time.perf_counter()
    cfg = setup.config
    s_keep_prob, pick_prob = pick_probabilities(cfg)
    count = cfg.synthetic.nEvents if n_events is None else int(n_events)
    if count < 1:
        raise ValueError("n_events must be >= 1")
    locator = build_locator(setup)
    if cfg.synthetic.zone.topElevM > locator.volume.top_elev_m:
        raise ValueError("the synthetic zone reaches above the search-volume top")
    rng = np.random.default_rng(cfg.synthetic.seed)
    truth = draw_hypocentres(setup, count, rng)
    noisy, clean = synthetic_picks(setup, locator, truth, rng)
    located = locate_many(setup, noisy, locator=locator)
    located_clean = locate_many(setup, clean, locator=locator)
    events = pd.concat(
        [truth, _errors(truth, located, "noisy"), _errors(truth, located_clean, "clean")], axis=1
    )
    noisy_s = _summary(events, "noisy")
    clean_s = _summary(events, "clean")
    sigma = cfg.locator.pickSigmaS
    report = SyntheticTest(
        nEvents=count,
        pickSigmaS={"P": sigma.P, "S": sigma.S},
        medianHErrM=noisy_s["medianHErrM"],
        medianVErrM=noisy_s["medianVErrM"],
        p90VErrM=noisy_s["p90VErrM"],
        medianDepthBiasM=noisy_s["medianDepthBiasM"],
    )
    stations = setup.stations
    ids = stations["id"].astype(str).tolist()
    runtime = time.perf_counter() - started
    params: dict[str, Any] = {
        "synthetic": cfg.synthetic.model_dump(mode="json"),
        "pickStats": {
            "sKeepProb": s_keep_prob,
            "pickProb": pick_prob,
            "source": "measured from the run" if pick_stats is not None else "seismology.yaml",
            "measured": pick_stats,
            "caveat": "every station gets a P pick, so the synthetic picks are optimistic for "
            "weak events",
        },
        "forwardModel": "exact 1D layered first arrivals (tt_grid.layered_first_arrival) on the "
        "tables' top-extended velocity model, plus Gaussian noise at the locator's sigma per "
        "station and phase (pickSigmaS or the profilePickSigmaS override)",
        "pickSigmaSByStation": {
            sid: {ph: locator.pick_sigma(sid, ph) for ph in PHASES} for sid in ids
        },
        "depthBiasSign": DEPTH_BIAS_SIGN,
        "reportFields": "synthetic.json medianHErrM/medianVErrM/p90VErrM are TRUE errors of the "
        "noisy run; compare Tier A formal vErrM with noisy.medianFormalVErrM",
        "stationGeometry": geometry_record(stations, geometry_label),
        "nStations": len(stations),
        "nPicksMean": float(np.mean([len(p) for p in noisy])),
        "nSMean": float(np.mean([int((p["phase"] == "S").sum()) for p in noisy])),
        "noisy": noisy_s,
        "noiseFree": clean_s,
        "locator": locator.to_record(),
        "velocityModel": locator.velocity_model_record(),
        "runtimeS": runtime,
        "nWorkers": min(cfg.locator.nWorkers, count),
    }
    log.info(
        "synthetic test: %d events on %d stations (geometry %s) in %.1f s; noisy median h %.1f m, "
        "v %.1f m, p90 v %.1f m, depth bias %+.1f m; noise-free depth bias %+.1f m; within hErrM "
        "%s, within vErrM %s; %d noisy PDFs truncated",
        count, len(stations), geometry_label, runtime, noisy_s["medianHErrM"],
        noisy_s["medianVErrM"], noisy_s["p90VErrM"], noisy_s["medianDepthBiasM"],
        clean_s["medianDepthBiasM"], noisy_s["fracHWithinHErrM"], noisy_s["fracVWithinVErrM"],
        noisy_s["nPdfTruncated"],
    )
    return SyntheticResult(report=report, params=params, events=events)


def write_synthetic_json(path: Path, report: SyntheticTest) -> Path:
    """Write ``report`` as ``synthetic.json`` (exactly the docs/02 fields), atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report.model_dump(mode="json"), indent=2, allow_nan=False) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    log.info("wrote %s", path)
    return path

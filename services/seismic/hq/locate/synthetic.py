"""Synthetic recovery test: can this station geometry and locator recover known hypocentres?

``nEvents`` hypocentres are drawn uniformly in the configured zone (a vertical cylinder in ENU
metres and elevM around the run origin; config values only, never catalog values) with origin
times uniform in the run window. Every station gets a P pick; its S pick is kept with probability
``sKeepProb``. Pick times are the exact 1D layered first-arrival times
(``tt_grid.layered_first_arrival``, an independent forward model, so table discretisation error
is part of what the test measures) plus Gaussian noise with standard deviation
``locator.pickSigmaS`` per phase. Every pick gets probability ``pickProb``. The same events are
located twice, with and without noise (same S picks kept). One seeded generator draws everything
in a fixed order, so identical config gives identical results.

Errors against the truth: horizontal ``hypot(de, dn)``, vertical ``|dElevM|``, and depth bias
``elevM_true - elevM_located`` (positive = located too deep). Formal-error calibration is the
fraction of events whose true horizontal (vertical) error is within ``hErrM`` (``vErrM``).

``synthetic.json`` holds exactly the docs/02 ``SyntheticTest`` fields; everything else goes into
the params dict (for ``ctx.record``) and the log.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hq.locate.locator import EventLocation, LocatorSetup, build_locator, locate_many
from hq.locate.tt_grid import PHASES, layered_first_arrival
from hq.locate.velocity import LayerModel

log = logging.getLogger(__name__)


# TODO(CONTRACT-01): replace with hq_contracts.models.SyntheticTest once it lands; keep the fields
# equal to docs/02 until then (a smoke test checks them).
@dataclass(frozen=True)
class SyntheticTest:
    """Same fields and meaning as ``SyntheticTest`` in docs/02."""

    nEvents: int
    pickSigmaS: dict[str, float]  # keys "P" and "S"
    medianHErrM: float
    medianVErrM: float
    p90VErrM: float
    medianDepthBiasM: float  # elevM_true - elevM_located: positive = located too deep


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
    setup: LocatorSetup, model: LayerModel, truth: pd.DataFrame, rng: np.random.Generator
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Noisy and noise-free pick frames per event (same kept S picks in both).

    ``model`` is the forward model (the tables' top-extended model).
    """
    cfg = setup.config
    stations = setup.stations
    ids = stations["id"].astype(str).tolist()
    n_ev, n_st = len(truth), len(ids)
    keep_s = rng.random((n_ev, n_st)) < cfg.synthetic.sKeepProb
    sigma = cfg.locator.pickSigmaS
    noise = {ph: rng.normal(0.0, getattr(sigma, ph), (n_ev, n_st)) for ph in PHASES}
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
            "prob": cfg.synthetic.pickProb,
        }
        noisy.append(pd.DataFrame({**base, "t": frame["tNoisy"]}))
        clean.append(pd.DataFrame({**base, "t": frame["tExact"]}))
    return noisy, clean


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
            f"{prefix}HErrM": [loc.h_err_m for loc in located],
            f"{prefix}VErrM": [loc.v_err_m for loc in located],
            f"{prefix}DepthOnEdge": [loc.depth_on_edge for loc in located],
            f"{prefix}NDropped": [len(loc.dropped_pick_ids) for loc in located],
            f"{prefix}NS": [loc.n_s for loc in located],
            f"{prefix}RmsS": [loc.rms_s for loc in located],
        }
    )


def _summary(df: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    h = df[f"{prefix}HErrTrueM"]
    v = df[f"{prefix}VErrTrueM"]
    return {
        "medianHErrM": float(h.median()),
        "p90HErrM": float(h.quantile(0.9)),
        "medianVErrM": float(v.median()),
        "p90VErrM": float(v.quantile(0.9)),
        "medianDepthBiasM": float(df[f"{prefix}DepthBiasM"].median()),
        "meanDepthBiasM": float(df[f"{prefix}DepthBiasM"].mean()),
        "fracHWithinHErrM": float((h <= df[f"{prefix}HErrM"]).mean()),
        "fracVWithinVErrM": float((v <= df[f"{prefix}VErrM"]).mean()),
        "medianFormalHErrM": float(df[f"{prefix}HErrM"].median()),
        "medianFormalVErrM": float(df[f"{prefix}VErrM"].median()),
        "nDepthOnEdge": int(df[f"{prefix}DepthOnEdge"].sum()),
        "nEventsWithDroppedPicks": int((df[f"{prefix}NDropped"] > 0).sum()),
    }


def run_synthetic(setup: LocatorSetup, *, n_events: int | None = None) -> SyntheticResult:
    """Run the synthetic recovery test on ``setup``'s station geometry and configuration."""
    started = time.perf_counter()
    cfg = setup.config
    count = cfg.synthetic.nEvents if n_events is None else int(n_events)
    if count < 1:
        raise ValueError("n_events must be >= 1")
    locator = build_locator(setup)
    if cfg.synthetic.zone.topElevM > locator.volume.top_elev_m:
        raise ValueError("the synthetic zone reaches above the search-volume top")
    rng = np.random.default_rng(cfg.synthetic.seed)
    truth = draw_hypocentres(setup, count, rng)
    noisy, clean = synthetic_picks(setup, locator.tables.model, truth, rng)
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
    runtime = time.perf_counter() - started
    params: dict[str, Any] = {
        "synthetic": cfg.synthetic.model_dump(mode="json"),
        "forwardModel": "exact 1D layered first arrivals (tt_grid.layered_first_arrival) on the "
        "tables' top-extended velocity model, plus Gaussian noise at locator.pickSigmaS",
        "depthBiasSign": "elevM_true - elevM_located; positive = located too deep",
        "nStations": len(stations),
        "nPicksMean": float(np.mean([len(p) for p in noisy])),
        "nSMean": float(np.mean([int((p["phase"] == "S").sum()) for p in noisy])),
        "noisy": noisy_s,
        "noiseFree": clean_s,
        "runtimeS": runtime,
        "nWorkers": min(cfg.locator.nWorkers, count),
    }
    log.info(
        "synthetic test: %d events on %d stations in %.1f s; noisy median h %.1f m, v %.1f m, "
        "p90 v %.1f m, depth bias %+.1f m; noise-free depth bias %+.1f m; within hErrM %.2f, "
        "within vErrM %.2f",
        count, len(stations), runtime, noisy_s["medianHErrM"], noisy_s["medianVErrM"],
        noisy_s["p90VErrM"], noisy_s["medianDepthBiasM"], clean_s["medianDepthBiasM"],
        noisy_s["fracHWithinHErrM"], noisy_s["fracVWithinVErrM"],
    )
    return SyntheticResult(report=report, params=params, events=events)


def write_synthetic_json(path: Path, report: SyntheticTest) -> Path:
    """Write ``report`` as ``synthetic.json`` (exactly the docs/02 fields), atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(report), indent=2, allow_nan=False) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    log.info("wrote %s", path)
    return path

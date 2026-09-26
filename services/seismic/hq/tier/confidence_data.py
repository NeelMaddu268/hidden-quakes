"""Training data for the chance-association classifier (ML-01, issue #100).

Positives: the run's real picks rerun through ``associate -> locate -> match -> assign_tiers``.
Negatives (decoys): the same reruns on scrambled-clock picks, every station's picks moved
together by one ``uniform(-shiftS, +shiftS)`` draw from ``numpy.random.default_rng([seed, i])``,
exactly the H4 null test (``hq.validate.null_test``). Shuffles ``0 .. nShuffles-1`` are the
published null test's; indices from ``nShuffles`` on continue the same seed stream for more
decoys.

Both sets go through one function path: ``hq.validate.null_test.rerun_pipeline`` with
``hq.associate.associate``, ``hq.match.match`` and ``hq.tier.assign_tiers`` as the validate stage
binds them, the run's own tier bars (``run.json`` ``tiering``) and the run's ``statics.parquet``.
Locate is ``hq.locate.locate_detailed(..., statics=statics_map(statics))``, which is what
``hq.locate.locate(..., statics=statics)`` runs and returns ``.result`` of; calling it directly
keeps the per-event locate flags (PDF truncation, MAP on a volume face) for the features.
``associator.nThreads`` is set to 1 and ``locator.nWorkers`` to ``--workers`` for the real
rerun (run first, alone) and to 1 inside each shuffle worker process (the config says results
do not depend on either), so at most ``--workers`` processes locate at once.

Per event, only quality and pick/residual statistics of the rerun are kept as features
(``FEATURES``); origin time, position and ids are metadata columns, never features.

    python -m hq.tier.confidence_data --run-dir <run> --config-dir configs/showcase \
        --cache-dir <data>/cache --out-dir <scratch>/data --workers 6 --budget-min 40
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hq_contracts.io import read_table

log = logging.getLogger(__name__)

REAL = -1  # shuffle index of the real-picks rerun

# Features: rerun outputs only; no origin time, position or ids.
QUALITY_FEATURES = (
    "quality_nStations",
    "quality_nP",
    "quality_nS",
    "quality_rmsS",
    "quality_gapDeg",
    "quality_minEpiDistM",
    "quality_hErrM",
    "quality_vErrM",
    "quality_depthOnEdge",
    "meanPickProb",
)
DERIVED_FEATURES = (
    "hErrNull",
    "vErrNull",
    "medianPickProb",
    "minPickProb",
    "medianPickProbP",
    "medianPickProbS",
    "nPicksAssociated",
    "nPicksUsed",
    "nPicksDropped",
    "fracPicksDropped",
    "medAbsResidualS",
    "meanAbsResidualS",
    "maxAbsResidualS",
    "medAbsResidualP_S",
    "medAbsResidualS_S",
    "nStationsPandS",
    "fracStationsUsed",
)
FLAG_FEATURES = (
    "pdfTruncated",
    "mapOnVolumeTop",
    "mapOnVolumeBottom",
    "depthOnEdgeTop",
    "depthOnEdgeBottom",
    "nodeBudgetHit",
    "hErrGridLimited",
    "vErrGridLimited",
    "outlierPassSkipped",
    "aboveNearestStationSurface",
)
FEATURES = QUALITY_FEATURES + DERIVED_FEATURES + FLAG_FEATURES
META = ("label", "shuffle", "seed", "eventId", "rerunEventId", "assocId", "tier",
        "catalogMatched")  # fmt: skip


@dataclass(frozen=True)
class Inputs:
    picks: pd.DataFrame
    stations: pd.DataFrame
    catalog: pd.DataFrame
    statics: pd.DataFrame
    tiering: dict[str, Any]
    seismology: Any  # SeismologyConfig, nWorkers/nThreads set to 1
    run: Any  # RunSection
    null_cfg: Any  # NullTestConfig
    run_id: str
    cache_dir: Path | None


def load_inputs(
    run_dir: Path, config_dir: Path, cache_dir: Path | None, locate_workers: int = 1
) -> Inputs:
    from hq.config import load_config
    from hq.validate.null_test import require_thresholds

    config = load_config(config_dir)
    seis = config.section("seismology")
    seis = seis.model_copy(
        update={
            "locator": seis.locator.model_copy(update={"nWorkers": locate_workers}),
            "associator": seis.associator.model_copy(update={"nThreads": 1}),
        }
    )
    record = json.loads((run_dir / "run.json").read_text())
    tiering = dict(require_thresholds(record.get("tiering"), "ML-01 confidence data"))
    return Inputs(
        picks=read_table(run_dir / "picks.parquet"),
        stations=read_table(run_dir / "stations.parquet"),
        catalog=read_table(run_dir / "catalog.parquet"),
        statics=read_table(run_dir / "statics.parquet"),
        tiering=tiering,
        seismology=seis,
        run=config.run,
        null_cfg=config.validate.nullTest,
        run_id=str(record["id"]),
        cache_dir=cache_dir,
    )


class _CapturingLocate:
    """``hq.locate.locate(..., statics=)`` as the validate stage binds it, keeping the details."""

    def __init__(self, inputs: Inputs) -> None:
        from hq.locate.statics import statics_map

        self.inputs = inputs
        self.statics = statics_map(inputs.statics)
        self.details: Any = None

    def __call__(self, assoc, picks, stations, cfg, run):
        from hq.locate import locate_detailed

        self.details = locate_detailed(
            assoc, picks, stations, cfg, run, run_id=self.inputs.run_id,
            cache_dir=self.inputs.cache_dir, statics=self.statics,
        )  # fmt: skip
        return self.details.result


def shuffled_picks(inputs: Inputs, index: int) -> pd.DataFrame:
    """The picks rerun ``index`` feeds in: the null test's shifted picks, or the real ones."""
    from hq.validate.null_test import select_profile, shift_picks, station_shifts

    selected = select_profile(inputs.picks, inputs.null_cfg.profile)
    if index == REAL:
        return selected
    station_ids = sorted(set(selected["stationId"].astype(str)))
    rng = np.random.default_rng([inputs.null_cfg.seed, index])
    return shift_picks(selected, station_shifts(station_ids, rng, inputs.null_cfg.shiftS))


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def event_features(
    events: pd.DataFrame,
    arrivals: pd.DataFrame,
    flags: pd.DataFrame,
    matches: pd.DataFrame,
    picks: pd.DataFrame,
    n_used_stations: int,
) -> pd.DataFrame:
    """One row per final event: ``FEATURES`` plus the rerun's ids, tier and match flag."""
    prob = dict(zip(picks["id"].astype(str), picks["prob"].astype(float), strict=True))
    flag_by_id = flags.set_index("eventId")
    matched = set(matches.loc[matches["eventId"].notna(), "eventId"].astype(str))
    by_event = {k: g for k, g in arrivals.groupby("eventId", sort=False)}
    rows = []
    for ev in events.itertuples(index=False):
        eid = str(ev.id)
        arr = by_event[eid]
        picked = arr[arr["pickId"].notna()]
        used = picked[picked["usedInLocation"].to_numpy(dtype=bool)]
        res = np.abs(used["residualS"].to_numpy(dtype=np.float64))
        is_p = used["phase"].astype(str).to_numpy() == "P"
        probs = np.array([prob[str(p)] for p in used["pickId"]], dtype=np.float64)
        per_station = used.groupby("stationId")["phase"].nunique()
        fl = flag_by_id.loc[eid]
        row: dict[str, Any] = {c: getattr(ev, c) for c in QUALITY_FEATURES}
        row.update(
            hErrNull=bool(pd.isna(ev.quality_hErrM)),
            vErrNull=bool(pd.isna(ev.quality_vErrM)),
            medianPickProb=_median(probs),
            minPickProb=float(probs.min()) if len(probs) else float("nan"),
            medianPickProbP=_median(probs[is_p]),
            medianPickProbS=_median(probs[~is_p]),
            nPicksAssociated=len(picked),
            nPicksUsed=len(used),
            nPicksDropped=len(picked) - len(used),
            fracPicksDropped=(len(picked) - len(used)) / len(picked) if len(picked) else 0.0,
            medAbsResidualS=_median(res),
            meanAbsResidualS=float(res.mean()) if len(res) else float("nan"),
            maxAbsResidualS=float(res.max()) if len(res) else float("nan"),
            medAbsResidualP_S=_median(res[is_p]),
            medAbsResidualS_S=_median(res[~is_p]),
            nStationsPandS=int((per_station >= 2).sum()),
            fracStationsUsed=float(ev.quality_nStations) / n_used_stations,
        )
        row.update({c: bool(fl[c]) for c in FLAG_FEATURES})
        row.update(
            rerunEventId=eid,
            assocId=str(fl["assocId"]),
            tier=str(ev.tier),
            catalogMatched=eid in matched,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def rerun_one(inputs: Inputs, index: int) -> tuple[int, pd.DataFrame, dict[str, Any]]:
    """One rerun (``REAL`` or a shuffle index) through the null test's pipeline -> features."""
    from hq.associate import associate
    from hq.config.validate import POnlyAssociatorConfig
    from hq.match import match
    from hq.tier import assign_tiers
    from hq.validate.lanes import LaneSeismologyApi
    from hq.validate.null_test import profile_config, rerun_pipeline

    started = time.perf_counter()
    loc = _CapturingLocate(inputs)
    api = LaneSeismologyApi(associate=associate, locate=loc, match=match, assign_tiers=assign_tiers)
    seis = profile_config(inputs.seismology, inputs.null_cfg.profile, POnlyAssociatorConfig())
    picks = shuffled_picks(inputs, index)
    rerun = rerun_pipeline(
        picks, inputs.stations, inputs.catalog, api, seis, inputs.run, thresholds=inputs.tiering
    )
    if rerun.n_events:
        details = loc.details
        feats = event_features(
            rerun.events, details.result.arrivals, details.flags, rerun.matches, picks,
            len(details.stations),
        )  # fmt: skip
    else:
        feats = pd.DataFrame(columns=[*FEATURES, "rerunEventId", "assocId", "tier",
                                      "catalogMatched"])  # fmt: skip
    feats.insert(0, "shuffle", index)
    feats.insert(0, "label", int(index == REAL))
    feats["seed"] = inputs.null_cfg.seed
    summary = {
        "shuffle": index,
        "nEvents": rerun.n_events,
        "nStrict": rerun.n_strict,
        "tiers": {k: int(v) for k, v in feats["tier"].value_counts().items()},
        "runtimeS": round(time.perf_counter() - started, 1),
    }
    return index, feats, summary


_INPUTS: Inputs | None = None


def _init_worker(run_dir: Path, config_dir: Path, cache_dir: Path | None) -> None:
    global _INPUTS
    logging.basicConfig(level=logging.WARNING)
    _INPUTS = load_inputs(run_dir, config_dir, cache_dir)


def _work(index: int) -> tuple[int, pd.DataFrame, dict[str, Any]]:
    assert _INPUTS is not None
    return rerun_one(_INPUTS, index)


def map_positives(real: pd.DataFrame, run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Canonical ``events.parquet`` id for each real-rerun event, through the association id
    (``locate_flags.parquet`` holds it for the run's events)."""
    flags = read_table(run_dir / "locate_flags.parquet")[["eventId", "assocId"]]
    canon = read_table(run_dir / "events.parquet")[["id", "tier"]]
    by_assoc = dict(zip(flags["assocId"].astype(str), flags["eventId"].astype(str), strict=True))
    tier_by_id = dict(zip(canon["id"].astype(str), canon["tier"].astype(str), strict=True))
    out = real.copy()
    out["eventId"] = out["assocId"].map(by_assoc)
    unmapped = out.loc[out["eventId"].isna(), "assocId"].tolist()
    mapped = out["eventId"].dropna()
    report = {
        "rerunEvents": len(out),
        "canonicalEvents": len(canon),
        "mapped": int(mapped.notna().sum()),
        "unmappedAssocIds": unmapped,
        "canonicalNotInRerun": sorted(set(canon["id"].astype(str)) - set(mapped)),
        "duplicateCanonical": int(mapped.duplicated().sum()),
        "rerunIdEqualsCanonicalId": int((out["eventId"] == out["rerunEventId"]).sum()),
        "tierEqualsCanonical": int((out["eventId"].map(tier_by_id) == out["tier"]).sum()),
        "rerunTiers": {k: int(v) for k, v in out["tier"].value_counts().items()},
        "canonicalTiers": {k: int(v) for k, v in canon["tier"].value_counts().items()},
    }
    return out, report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--budget-min", type=float, default=40.0, help="stop adding extra shuffles")
    p.add_argument("--max-extra", type=int, default=200)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.monotonic()
    inputs = load_inputs(args.run_dir, args.config_dir, args.cache_dir, args.workers)
    n_pub = inputs.null_cfg.nShuffles
    parts = args.out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    results: dict[int, pd.DataFrame] = {}
    summaries: dict[int, dict[str, Any]] = {}
    real_part = parts / f"rerun_{REAL:+05d}.parquet"
    if real_part.is_file() and (parts / "real_summary.json").is_file():  # resume
        results[REAL] = pd.read_parquet(real_part)
        summaries[REAL] = json.loads((parts / "real_summary.json").read_text())
    else:  # the real rerun alone, its events located by --workers locate processes
        _, results[REAL], summaries[REAL] = rerun_one(inputs, REAL)
        results[REAL].to_parquet(real_part, index=False)
        (parts / "real_summary.json").write_text(json.dumps(summaries[REAL]))
    log.info("real rerun: %s (%.0f s elapsed)", summaries[REAL], time.monotonic() - started)
    queue = list(range(n_pub))
    extra = iter(range(n_pub, n_pub + args.max_extra))
    deadline = started + args.budget_min * 60
    ctx = get_context("spawn")
    init = partial(_init_worker, args.run_dir, args.config_dir, args.cache_dir)
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init) as pool:
        running: dict[Future, int] = {}

        def top_up() -> None:
            while len(running) < args.workers:
                if queue:
                    index = queue.pop(0)
                else:
                    runtimes = [s["runtimeS"] for s in summaries.values() if s["shuffle"] >= 0]
                    est = float(np.median(runtimes)) if runtimes else 120.0
                    if time.monotonic() + est > deadline:
                        return
                    index = next(extra, None)
                    if index is None:
                        return
                running[pool.submit(_work, index)] = index

        top_up()
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                running.pop(fut)
                index, feats, summary = fut.result()
                feats.to_parquet(parts / f"rerun_{index:+05d}.parquet", index=False)
                results[index], summaries[index] = feats, summary
                log.info("rerun %d done: %s (%.0f s elapsed)", index, summary,
                         time.monotonic() - started)  # fmt: skip
            top_up()

    positives, mapping = map_positives(results[REAL], args.run_dir)
    negatives = pd.concat([results[i] for i in sorted(results) if i != REAL], ignore_index=True)
    negatives["eventId"] = negatives["rerunEventId"]  # decoys have no canonical id
    cols = [*META, *FEATURES]
    positives[cols].to_parquet(args.out_dir / "positives.parquet", index=False)
    negatives[cols].to_parquet(args.out_dir / "negatives.parquet", index=False)
    published = [summaries[i] for i in range(n_pub)]
    counts = np.array([s["nEvents"] for s in published], dtype=np.float64)
    manifest = {
        "runDir": str(args.run_dir),
        "runId": inputs.run_id,
        "nullTest": {
            "nShuffles": n_pub,
            "shiftS": inputs.null_cfg.shiftS,
            "seed": inputs.null_cfg.seed,
            "profile": inputs.null_cfg.profile,
            "publishedIndices": [0, n_pub - 1],
            "meanChanceEvents": float(counts.mean()),
            "stdChanceEvents": float(counts.std(ddof=1)),
            "meanChanceStrict": float(np.mean([s["nStrict"] for s in published])),
            "totalChanceEvents": int(counts.sum()),
        },
        "extraShuffles": sorted(i for i in summaries if i >= n_pub),
        "positives": mapping,
        "nPositives": len(positives),
        "nNegatives": len(negatives),
        "features": list(FEATURES),
        "meta": list(META),
        "reruns": [summaries[i] for i in sorted(summaries)],
        "wallS": round(time.monotonic() - started, 1),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote %d positives, %d negatives to %s in %.0f s", len(positives), len(negatives),
             args.out_dir, time.monotonic() - started)  # fmt: skip


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()

"""Chance-association classifier (ML-01, issue #100), trained on this run's own data.

Each candidate event gets a score in [0, 1] for how much its station count, arrival-time fit and
pick confidence look like an association made on real pick timing rather than one made on
scrambled-clock picks. Label 1 is an event from the run's real picks, label 0 a decoy event from
the null test's shuffles (every station's picks moved together by one uniform(-shiftS, +shiftS)
draw), both built by ``hq.tier.confidence_data`` through the same associate -> locate -> match ->
tier path.

The score is NOT a calibrated probability that an event is an earthquake: the real-run set
itself holds some chance associations, and the decoys are much smaller than typical real
candidates (almost none has more than 8 stations), so held-out results are also reported among
events of equal station count, and scores above the decoys' station range are extrapolated.

Candidates, simplest first: a 3-feature logistic regression (station count, arrival-time rms,
mean pick probability), the 34-feature logistic regression, and a one-hidden-layer torch MLP on
the 34 features. A larger one replaces the simpler choice only if it beats it by more than
``MARGIN`` on both the held-out ROC AUC and the equal-station-count AUC. Folds: decoys split by
shuffle (grouped), real events at random; every event's score comes from a model that did not
see it.

    python -m hq.tier.confidence --run <runId> --data-dir <scratch>/data --out <scratch>/confidence.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata, spearmanr

from hq.tier.confidence_data import FEATURES

log = logging.getLogger(__name__)

# Timing residuals and rms span orders of magnitude (1 ms to seconds); distances and errors too.
SECONDS_FEATURES = (
    "quality_rmsS",
    "medAbsResidualS",
    "meanAbsResidualS",
    "maxAbsResidualS",
    "medAbsResidualP_S",
    "medAbsResidualS_S",
)
METERS_FEATURES = ("quality_minEpiDistM", "quality_hErrM", "quality_vErrM")
SECONDS_FLOOR = 1e-3
METERS_FLOOR = 1.0
# Event-size features, left out in the "no size" ablation.
SIZE_FEATURES = (
    "quality_nStations",
    "quality_nP",
    "quality_nS",
    "nPicksAssociated",
    "nPicksUsed",
    "nPicksDropped",
    "nStationsPandS",
    "fracStationsUsed",
)

# Exact duplicates in this data, by construction: fracStationsUsed is nStations over the run's
# fixed station count, and pdfTruncated == hErrNull == vErrNull. The model keeps one of each so
# the logistic weights read cleanly (L2 would otherwise split one weight across copies).
REDUNDANT_FEATURES = ("fracStationsUsed", "vErrNull", "pdfTruncated")
MODEL_FEATURES = tuple(f for f in FEATURES if f not in REDUNDANT_FEATURES)
# The three features that carry nearly all of the 34-feature model's held-out performance:
# event size, timing coherence (what the scramble destroys) and pick confidence. Decoys are
# re-assembled strong picks from real events, so pick confidence gets a negative weight: an
# artifact of how decoys are made, disclosed rather than hidden.
CORE_FEATURES = ("quality_nStations", "quality_rmsS", "meanPickProb")
# Pick-confidence features, left out in the "timing and geometry only" ablations.
PICK_PROB_FEATURES = (
    "meanPickProb",
    "medianPickProb",
    "minPickProb",
    "medianPickProbP",
    "medianPickProbS",
)

N_FOLDS = 5
SEED = 0
L2 = 1e-2
MLP_HIDDEN = 16
MLP_EPOCHS = 400
MLP_LR = 1e-2
MLP_WEIGHT_DECAY = 1e-3
HARD_NSTATIONS = (5, 10)
# Model candidates, simplest first. A later one replaces the current choice only if its mean
# held-out ROC AUC and its station-matched AUC are both higher by more than MARGIN.
CANDIDATES = ("logisticCore", "logistic", "mlp")
MARGIN = 0.01
# Decoy support: the largest station count with at least this many decoys. Above it the score
# is an extrapolation of the station-count weight, not a comparison with decoys.
SUPPORT_MIN_DECOYS = 5


# --- features -------------------------------------------------------------------------------


def transform(df: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    """Raw feature matrix: bools to 0/1, log10 for seconds/meters features; NaN stays NaN."""
    cols = []
    for name in names:
        x = df[name].to_numpy(dtype=float)
        if name in SECONDS_FEATURES:
            x = np.log10(x + SECONDS_FLOOR)
        elif name in METERS_FEATURES:
            x = np.log10(x + METERS_FLOOR)
        cols.append(x)
    return np.column_stack(cols) if cols else np.empty((len(df), 0))


@dataclass(frozen=True)
class Prep:
    """Imputation and standardization fitted on training rows only."""

    names: tuple[str, ...]
    fill: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def fit_prep(df: pd.DataFrame, names: Sequence[str]) -> Prep:
    """Nulls: hErrM/vErrM (PDF truncated; hErrNull/vErrNull flag them) and the S-pick stats
    (no used S pick; nS carries it) are filled with the training median of the non-null values.
    A column that is null in every training row is filled with 0; a constant column keeps std 1.
    """
    x = transform(df, names)
    with np.errstate(all="ignore"):
        fill = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in x.T])
    xf = np.where(np.isnan(x), fill, x)
    mean = xf.mean(axis=0)
    std = xf.std(axis=0)
    std[std == 0] = 1.0
    return Prep(tuple(names), fill, mean, std)


def apply_prep(prep: Prep, df: pd.DataFrame) -> np.ndarray:
    x = transform(df, prep.names)
    x = np.where(np.isnan(x), prep.fill, x)
    return (x - prep.mean) / prep.std


def class_weights(y: np.ndarray) -> np.ndarray:
    """Balanced weights: each class carries half the total weight."""
    y = np.asarray(y)
    n = len(y)
    n1 = int((y == 1).sum())
    n0 = n - n1
    return np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))


# --- models ---------------------------------------------------------------------------------

Predictor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Logistic:
    intercept: float
    coef: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return expit(self.intercept + x @ self.coef)


def fit_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = L2) -> Logistic:
    """Weighted log-loss (weights normalized to sum 1) + l2/2 * |coef|^2; intercept unpenalized."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float) / np.sum(w)
    sign = 2.0 * y - 1.0

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        b, coef = theta[0], theta[1:]
        z = b + x @ coef
        value = float(np.sum(w * np.logaddexp(0.0, -sign * z)) + 0.5 * l2 * coef @ coef)
        gz = w * (expit(z) - y)
        grad = np.concatenate([[gz.sum()], x.T @ gz + l2 * coef])
        return value, grad

    res = minimize(
        loss,
        np.zeros(x.shape[1] + 1),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "gtol": 1e-9},
    )
    return Logistic(float(res.x[0]), res.x[1:].copy())


def fit_mlp(x: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int = SEED) -> Predictor:
    """One hidden layer (ReLU), full-batch Adam, class-weighted BCE, weight decay."""
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], MLP_HIDDEN), torch.nn.ReLU(), torch.nn.Linear(MLP_HIDDEN, 1)
    ).double()
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=float))
    yt = torch.from_numpy(np.asarray(y, dtype=float))
    wt = torch.from_numpy(np.asarray(w, dtype=float) / np.sum(w))
    for _ in range(MLP_EPOCHS):
        opt.zero_grad()
        logits = model(xt).squeeze(1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, reduction="none")
        (wt * bce).sum().backward()
        opt.step()
    model.eval()

    def predict(xn: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(torch.from_numpy(np.ascontiguousarray(xn, dtype=float))).squeeze(1)
            return torch.sigmoid(out).numpy()

    return predict


# --- metrics --------------------------------------------------------------------------------


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney ROC AUC, ties count one half. NaN if a class is missing."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Sum over distinct score thresholds of (recall step) x precision, as sklearn defines it."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    n1 = int((y == 1).sum())
    if n1 == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]  # end of each tie block
    tp = np.cumsum(ys == 1)[last]
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n1
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def matched_auc(y: np.ndarray, s: np.ndarray, strata: np.ndarray) -> tuple[float, int]:
    """ROC AUC over (real, decoy) pairs in the same stratum only (e.g. equal station count),
    so a score that only reads the stratum gets 0.5. Returns (auc, number of pairs)."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    strata = np.asarray(strata)
    num = 0.0
    pairs = 0
    for k in np.unique(strata):
        m = strata == k
        n1 = int((y[m] == 1).sum())
        n0 = int(m.sum()) - n1
        if n1 and n0:
            num += roc_auc(y[m], s[m]) * n1 * n0
            pairs += n1 * n0
    return (num / pairs if pairs else float("nan")), pairs


def threshold_at_fpr(decoy_scores: np.ndarray, fpr: float) -> float:
    """Smallest threshold t such that at most `fpr` of decoys score strictly above t."""
    return float(np.quantile(np.asarray(decoy_scores, dtype=float), 1.0 - fpr, method="higher"))


def decoy_exceedance(score: np.ndarray, labels: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Per event, (k + 1) / (n + 1) where k of the n decoys held out in the same fold (scored by
    the same model) score at least as high: how often a scrambled-clock decoy looks this real."""
    score = np.asarray(score, dtype=float)
    labels = np.asarray(labels)
    out = np.full(len(score), np.nan)
    for k in np.unique(fold):
        m = fold == k
        dec = np.sort(score[m & (labels == 0)])
        n_at_least = len(dec) - np.searchsorted(dec, score[m], side="left")
        out[m] = (n_at_least + 1) / (len(dec) + 1)
    return out


# --- folds and cross-validation -------------------------------------------------------------


def make_folds(
    labels: np.ndarray, shuffles: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED
) -> np.ndarray:
    """Fold id per row: real events (label 1) at random; decoys by shuffle, whole shuffles
    per fold, so no fold's decoys share a scramble with its training decoys."""
    labels = np.asarray(labels)
    shuffles = np.asarray(shuffles)
    rng = np.random.default_rng(seed)
    fold = np.full(len(labels), -1, dtype=int)
    pos = rng.permutation(np.flatnonzero(labels == 1))
    fold[pos] = np.arange(len(pos)) % n_folds
    groups = rng.permutation(np.unique(shuffles[labels == 0]))
    group_fold = {g: i % n_folds for i, g in enumerate(groups)}
    neg = labels == 0
    fold[neg] = [group_fold[g] for g in shuffles[neg]]
    return fold


@dataclass
class CvResult:
    oof: np.ndarray
    folds: list[dict[str, float]]
    coefs: list[np.ndarray]  # logistic only


def cross_validate(df: pd.DataFrame, fold: np.ndarray, kind: str, names: Sequence[str]) -> CvResult:
    y = df["label"].to_numpy()
    oof = np.full(len(df), np.nan)
    per_fold: list[dict[str, float]] = []
    coefs: list[np.ndarray] = []
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        prep = fit_prep(df[tr], names)
        xtr, xte = apply_prep(prep, df[tr]), apply_prep(prep, df[te])
        w = class_weights(y[tr])
        if kind == "logistic":
            model = fit_logistic(xtr, y[tr], w)
            coefs.append(model.coef)
            oof[te] = model(xte)
        elif kind == "mlp":
            oof[te] = fit_mlp(xtr, y[tr], w, seed=SEED + int(k))(xte)
        else:
            raise ValueError(kind)
        per_fold.append(fold_metrics(df[te], oof[te]))
    return CvResult(oof, per_fold, coefs)


def fold_metrics(df: pd.DataFrame, s: np.ndarray) -> dict[str, float]:
    y = df["label"].to_numpy()
    nsta = df["quality_nStations"].to_numpy()
    lo, hi = HARD_NSTATIONS
    hard = (nsta >= lo) & (nsta <= hi)
    tier_c = (y == 0) | (df["tier"].to_numpy() == "C")
    return {
        "nReal": int((y == 1).sum()),
        "nDecoy": int((y == 0).sum()),
        "rocAuc": roc_auc(y, s),
        "averagePrecision": average_precision(y, s),
        "rocAucNSta5to10": roc_auc(y[hard], s[hard]),
        "rocAucTierC": roc_auc(y[tier_c], s[tier_c]),
        "matchedAucNSta": matched_auc(y, s, nsta)[0],
        "matchedAucNPicks": matched_auc(y, s, df["nPicksUsed"].to_numpy())[0],
    }


def summarize(folds: list[dict[str, float]]) -> dict[str, list[float]]:
    """mean, std (ddof=1), min, max over folds for each metric."""
    out: dict[str, list[float]] = {}
    for key in folds[0]:
        v = np.array([f[key] for f in folds], dtype=float)
        out[key] = [
            round(float(v.mean()), 4),
            round(float(v.std(ddof=1)), 4),
            round(float(v.min()), 4),
            round(float(v.max()), 4),
        ]
    return out


def pooled(df: pd.DataFrame, s: np.ndarray) -> dict[str, Any]:
    """Metrics on all out-of-fold scores pooled, with subset sizes."""
    y = df["label"].to_numpy()
    nsta = df["quality_nStations"].to_numpy()
    lo, hi = HARD_NSTATIONS
    hard = (nsta >= lo) & (nsta <= hi)
    le10 = nsta <= hi
    tier_c = (y == 0) | (df["tier"].to_numpy() == "C")
    mauc, pairs = matched_auc(y, s, nsta)
    mauc_p, pairs_p = matched_auc(y, s, df["nPicksUsed"].to_numpy())
    return {
        "rocAuc": round(roc_auc(y, s), 4),
        "averagePrecision": round(average_precision(y, s), 4),
        "prevalence": round(float(y.mean()), 4),
        "nSta5to10": {
            "rocAuc": round(roc_auc(y[hard], s[hard]), 4),
            "averagePrecision": round(average_precision(y[hard], s[hard]), 4),
            "nReal": int((y[hard] == 1).sum()),
            "nDecoy": int((y[hard] == 0).sum()),
        },
        "nStaLe10": {
            "rocAuc": round(roc_auc(y[le10], s[le10]), 4),
            "nReal": int((y[le10] == 1).sum()),
            "nDecoy": int((y[le10] == 0).sum()),
        },
        "tierC": {
            "rocAuc": round(roc_auc(y[tier_c], s[tier_c]), 4),
            "averagePrecision": round(average_precision(y[tier_c], s[tier_c]), 4),
            "nReal": int((y[tier_c] == 1).sum()),
            "nDecoy": int((y[tier_c] == 0).sum()),
        },
        "matchedNSta": {"rocAuc": round(mauc, 4), "pairs": pairs},
        "matchedNPicksUsed": {"rocAuc": round(mauc_p, 4), "pairs": pairs_p},
    }


def by_station_count(df: pd.DataFrame, s: np.ndarray) -> list[dict[str, Any]]:
    """Pooled out-of-fold ROC AUC within each station count that has both classes."""
    y = df["label"].to_numpy()
    nsta = df["quality_nStations"].to_numpy()
    rows = []
    for k in np.unique(nsta):
        m = nsta == k
        n1, n0 = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        auc = roc_auc(y[m], s[m]) if n1 and n0 else float("nan")
        rows.append({"nStations": int(k), "nReal": n1, "nDecoy": n0, "rocAuc": round(auc, 4)})
    return rows


def decoy_support(nsta_decoys: np.ndarray, min_decoys: int = SUPPORT_MIN_DECOYS) -> int:
    """Largest station count held by at least ``min_decoys`` decoys (0 if none)."""
    counts = pd.Series(np.asarray(nsta_decoys)).value_counts()
    ok = counts[counts >= min_decoys]
    return int(ok.index.max()) if len(ok) else 0


def permutation_importance(
    df: pd.DataFrame, fold: np.ndarray, names: Sequence[str], seed: int = SEED
) -> pd.DataFrame:
    """Held-out drop in ROC AUC and station-matched AUC when one feature is shuffled within
    each test fold (logistic model), averaged over folds."""
    y = df["label"].to_numpy()
    nsta = df["quality_nStations"].to_numpy()
    rng = np.random.default_rng(seed)
    drops: dict[str, list[tuple[float, float]]] = {n: [] for n in names}
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        prep = fit_prep(df[tr], names)
        model = fit_logistic(apply_prep(prep, df[tr]), y[tr], class_weights(y[tr]))
        xte = apply_prep(prep, df[te])
        base = model(xte)
        b_auc, b_m = roc_auc(y[te], base), matched_auc(y[te], base, nsta[te])[0]
        for j, name in enumerate(names):
            xp = xte.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            s = model(xp)
            drops[name].append(
                (b_auc - roc_auc(y[te], s), b_m - matched_auc(y[te], s, nsta[te])[0])
            )
    rows = [
        (n, float(np.mean([d[0] for d in v])), float(np.mean([d[1] for d in v])))
        for n, v in drops.items()
    ]
    return pd.DataFrame(rows, columns=["feature", "aucDrop", "matchedAucDrop"]).sort_values(
        "aucDrop", ascending=False, ignore_index=True
    )


def choose(
    summary: dict[str, dict[str, list[float]]], candidates: Sequence[str] = CANDIDATES
) -> str:
    """The simplest candidate, replaced by a later (larger) one only if that one's mean held-out
    ROC AUC and station-matched AUC both beat the current choice by more than MARGIN."""
    chosen = candidates[0]
    for c in candidates[1:]:
        cur, new = summary[chosen], summary[c]
        if all(new[k][0] - cur[k][0] > MARGIN for k in ("rocAuc", "matchedAucNSta")):
            chosen = c
    return chosen


# --- run ------------------------------------------------------------------------------------


def _describe(s: pd.Series) -> dict[str, float]:
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "n": int(s.size),
        "mean": round(float(s.mean()), 3),
        **{f"q{int(p * 100)}": round(float(v), 3) for p, v in q.items()},
    }


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _weights(df: pd.DataFrame, names: Sequence[str], fold_coefs: list[np.ndarray]) -> list[dict]:
    """Logistic coefficients per standard deviation from a full-data fit, with their across-fold
    sd, largest first."""
    y = df["label"].to_numpy()
    prep = fit_prep(df, names)
    full = fit_logistic(apply_prep(prep, df), y, class_weights(y))
    out = pd.DataFrame(
        {
            "feature": list(names),
            "coef": full.coef,
            "coefFoldStd": np.array(fold_coefs).std(axis=0, ddof=1),
        }
    )
    out = out.sort_values("coef", key=np.abs, ascending=False, ignore_index=True)
    return out.round(4).to_dict("records")


def run(data_dir: Path, out_path: Path, run_id: str | None = None) -> dict[str, Any]:
    """Train and evaluate every candidate on the same folds; write ``out_path``
    (``confidence.json``) and, beside it, ``scores.parquet`` and ``report.json``."""
    pos = pd.read_parquet(data_dir / "positives.parquet")
    neg = pd.read_parquet(data_dir / "negatives.parquet")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if run_id is not None and manifest["runId"] != run_id:
        raise ValueError(
            f"{data_dir} holds training data for run {manifest['runId']!r}, not {run_id!r}"
        )
    df = pd.concat([pos, neg], ignore_index=True)
    y = df["label"].to_numpy()
    fold = make_folds(y, df["shuffle"].to_numpy())
    names = list(MODEL_FEATURES)
    feature_sets = {
        "logisticCore": list(CORE_FEATURES),
        "logistic": names,
        "mlp": names,
        "logisticCoreNoPickProb": [n for n in CORE_FEATURES if n not in PICK_PROB_FEATURES],
        "logisticNoSize": [n for n in names if n not in SIZE_FEATURES],
        "logisticNoPickProb": [n for n in names if n not in PICK_PROB_FEATURES],
    }
    cv = {
        k: cross_validate(df, fold, "mlp" if k == "mlp" else "logistic", v)
        for k, v in feature_sets.items()
    }
    # One-feature baselines, no training: bigger events and tighter arrival-time fits look real.
    baselines = {
        "nStationsOnly": df["quality_nStations"].to_numpy(dtype=float),
        "rmsOnly": -df["quality_rmsS"].to_numpy(dtype=float),
    }
    fold_summary = {k: summarize(v.folds) for k, v in cv.items()}
    pooled_all = {k: pooled(df, v.oof) for k, v in cv.items()}
    for k, b in baselines.items():
        fold_summary[k] = summarize(
            [fold_metrics(df[fold == f], b[fold == f]) for f in np.unique(fold)]
        )
        pooled_all[k] = pooled(df, b)
    chosen = choose(fold_summary)
    chosen_names = feature_sets[chosen]
    score = cv[chosen].oof

    real = y == 1
    sc = pd.DataFrame(
        {
            "eventId": df["eventId"],
            "label": y,
            "shuffle": df["shuffle"],
            "fold": fold,
            "tier": df["tier"],
            "catalogMatched": df["catalogMatched"],
            "nStations": df["quality_nStations"],
            "score": score,
            "scoreLogisticCore": cv["logisticCore"].oof,
            "scoreLogistic": cv["logistic"].oof,
            "scoreMlp": cv["mlp"].oof,
            "scoreLogisticNoSize": cv["logisticNoSize"].oof,
            "decoyExceedance": decoy_exceedance(score, y, fold),
        }
    )
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    sc.to_parquet(out_dir / "scores.parquet", index=False)

    dec = sc[~real]
    rs = sc[real]
    support = decoy_support(dec["nStations"].to_numpy())
    thresholds = {}
    for fpr in (0.10, 0.05, 0.01):
        t = threshold_at_fpr(dec["score"].to_numpy(), fpr)
        thresholds[f"decoyFpr{int(fpr * 100)}pct"] = {
            "threshold": round(t, 4),
            "decoysAbove": int((dec["score"] > t).sum()),
            "realAbove": int((rs["score"] > t).sum()),
            "realTierCAbove": f"{int((rs[rs.tier == 'C'].score > t).sum())}/{int((rs.tier == 'C').sum())}",
            "realWithinSupportAbove": f"{int((rs[rs.nStations <= support].score > t).sum())}/{int((rs.nStations <= support).sum())}",
        }
    for cut in (0.5, 0.9):
        thresholds[f"score{cut}"] = {
            "decoysAbove": f"{int((dec.score > cut).sum())}/{len(dec)}",
            "realAbove": f"{int((rs.score > cut).sum())}/{len(rs)}",
        }

    top_decoys = dec.sort_values("nStations", ascending=False).head(10)
    sanity = {
        "decoySupport": {
            "maxStations": support,
            "minDecoysPerStationCount": SUPPORT_MIN_DECOYS,
            "decoysAbove": int((dec.nStations > support).sum()),
            "realWithin": int((rs.nStations <= support).sum()),
            "realAbove": int((rs.nStations > support).sum()),
        },
        "byStationCount": by_station_count(df, score),
        "realWithinSupport": _describe(rs[rs.nStations <= support].score),
        # All catalog-matched events are far above the decoy support, so this is trivially met.
        "catalogMatched": _describe(rs[rs.catalogMatched].score),
        "catalogMatchedMinNStations": (
            int(rs[rs.catalogMatched].nStations.min()) if rs.catalogMatched.any() else None
        ),
        "byTier": {t: _describe(g.score) for t, g in rs.groupby("tier")},
        "decoys": _describe(dec.score),
        "decoysReachedTierB": int((dec.tier != "C").sum()),
        "largestDecoys": top_decoys[["shuffle", "eventId", "nStations", "score"]]
        .round(4)
        .to_dict("records"),
        "realDecoyExceedance": {
            f"<={c}": int((rs.decoyExceedance <= c).sum()) for c in (0.01, 0.05)
        },
        "realAtLeast0.99": int((rs.score >= 0.99).sum()),
        "spearmanScoreVsNStationsReal": round(float(spearmanr(rs.score, rs.nStations)[0]), 3),
    }

    weights = {
        k: _weights(df, feature_sets[k], cv[k].coefs)
        for k in ("logisticCore", "logistic", "logisticCoreNoPickProb")
    }
    perm_names = chosen_names if chosen != "mlp" else names
    perm = permutation_importance(df, fold, perm_names)

    report = {
        "schema": "hq.confidence-report/1",
        "runId": manifest["runId"],
        "gitSha": _git_sha(),
        "createdAt": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": {
            "real": int(real.sum()),
            "decoys": int((~real).sum()),
            "shuffles": int(df.loc[~real, "shuffle"].nunique()),
            "shiftS": manifest["nullTest"]["shiftS"],
            "features": chosen_names,
            "allFeatures": names,
            "droppedDuplicates": list(REDUNDANT_FEATURES),
        },
        "folds": {
            "n": N_FOLDS,
            "seed": SEED,
            "scheme": "real events random; decoys by shuffle",
            "sizes": {
                int(k): {
                    "real": int(((fold == k) & real).sum()),
                    "decoys": int(((fold == k) & ~real).sum()),
                }
                for k in np.unique(fold)
            },
        },
        "hyper": {
            "l2": L2,
            "mlpHidden": MLP_HIDDEN,
            "mlpEpochs": MLP_EPOCHS,
            "mlpLr": MLP_LR,
            "mlpWeightDecay": MLP_WEIGHT_DECAY,
            "candidates": list(CANDIDATES),
            "margin": MARGIN,
            "supportMinDecoys": SUPPORT_MIN_DECOYS,
        },
        "featureSets": feature_sets,
        "foldSummary(mean,std,min,max)": fold_summary,
        "pooledOutOfFold": pooled_all,
        "chosen": chosen,
        "thresholds": thresholds,
        "sanity": sanity,
        "logisticWeights": weights,
        "permutationImportance": perm.round(4).to_dict("records"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
    doc = confidence_doc(report, sc[real])
    out_path.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
    return report


LABEL = "Scramble test"
DESCRIPTION = (
    "How much this event's station count, arrival-time fit and pick confidence look like an"
    " association on real pick timing rather than a scrambled-clock decoy, from a logistic"
    " regression trained on this run (not a probability that it is an earthquake)"
)
MODEL_TYPES = {
    "logisticCore": "logistic regression, 3 features, class-weighted; out-of-fold scores",
    "logistic": "logistic regression, 34 features, class-weighted; out-of-fold scores",
    "mlp": "MLP (one hidden layer), 34 features, class-weighted; out-of-fold scores",
}


def confidence_doc(report: dict[str, Any], real: pd.DataFrame) -> dict[str, Any]:
    """The ``hq.confidence/1`` document the exporter reads from a run directory: out-of-fold
    scores of the real candidate events, rounded to 3 decimals, with the held-out numbers and
    the baselines they must be read against."""
    chosen = report["chosen"]
    fs = report["foldSummary(mean,std,min,max)"]
    data = report["data"]
    support = report["sanity"]["decoySupport"]

    def held(key: str) -> dict[str, Any]:
        return {
            "rocAuc": fs[key]["rocAuc"][0],
            "rocAucEqualStationCount": fs[key]["matchedAucNSta"][0],
        }

    weights = report["logisticWeights"].get(chosen) if chosen != "mlp" else None
    return {
        "schema": "hq.confidence/1",
        "runId": report["runId"],
        "model": {
            "type": MODEL_TYPES[chosen],
            "features": data["features"],
            "coefPerSd": {w["feature"]: w["coef"] for w in weights} if weights else None,
            "trainedOn": {
                "positives": data["real"],
                "decoys": data["decoys"],
                "shuffles": data["shuffles"],
                "shiftS": data["shiftS"],
            },
            "heldOut": {
                "rocAuc": fs[chosen]["rocAuc"][0],
                "rocAucStd": fs[chosen]["rocAuc"][1],
                "rocAucRange": fs[chosen]["rocAuc"][2:],
                "averagePrecision": fs[chosen]["averagePrecision"][0],
                "rocAucEqualStationCount": fs[chosen]["matchedAucNSta"][0],
                "rocAucEqualStationCountRange": fs[chosen]["matchedAucNSta"][2:],
                "folds": report["folds"]["n"],
                "baselines": {
                    "stationCountOnly": held("nStationsOnly"),
                    "rmsOnly": held("rmsOnly"),
                },
                "comparedWith": {k: held(k) for k in CANDIDATES if k != chosen},
            },
            "decoySupport": {
                "maxStations": support["maxStations"],
                "note": (
                    "Decoys with more stations than this are too few to compare against; scores"
                    " of larger events extrapolate the station-count weight."
                ),
            },
            "createdAt": report["createdAt"],
            "gitSha": report["gitSha"],
        },
        "label": LABEL,
        "description": DESCRIPTION,
        "events": {str(e): round(float(v), 3) for e, v in zip(real["eventId"], real["score"])},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="runId the training data must come from")
    ap.add_argument(
        "--data-dir", type=Path, required=True, help="output dir of hq.tier.confidence_data"
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="confidence.json to write; scores.parquet and report.json are written beside it",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = run(args.data_dir, args.out, run_id=args.run)
    log.info(
        "chosen=%s heldOut=%s wrote %s",
        report["chosen"],
        json.dumps(report["foldSummary(mean,std,min,max)"][report["chosen"]]),
        args.out,
    )


if __name__ == "__main__":
    main()

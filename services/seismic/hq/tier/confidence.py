"""Chance-association classifier (ML-01, issue #100), trained on this run's own data.

Each candidate event gets a score in [0, 1] for how much it looks like an association made on
real pick timing rather than one made on scrambled-clock picks. Label 1 is an event from the
run's real picks, label 0 a decoy event from the null test's shuffles (every station's picks
moved together by one uniform(-shiftS, +shiftS) draw), both built by ``hq.tier.confidence_data``
through the same associate -> locate -> match -> tier path.

The score is NOT a calibrated probability that an event is an earthquake: the real-run set
itself holds some chance associations (the null test's mean is 94 per shuffle), and the
decoys are much smaller than typical real candidates, so held-out results are also reported
among events of equal station count.

Two models, both class-weighted and deterministic: a standardized L2 logistic regression
(numpy/scipy) and a one-hidden-layer MLP (torch). Folds: decoys split by shuffle (grouped),
real events at random; every event's score comes from a model that did not see it.

    python -m hq.tier.confidence --data-dir <scratch>/data --out-dir <scratch>
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
# Pick-confidence features, left out in the "timing and geometry only" ablation.
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
# The MLP replaces the logistic regression only if its mean held-out ROC AUC and its
# station-matched AUC are both higher by more than this.
MLP_MARGIN = 0.01


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
    }


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


def choose(summary: dict[str, dict[str, list[float]]]) -> str:
    """Logistic unless the MLP's mean held-out ROC AUC and station-matched AUC both beat it by
    more than MLP_MARGIN."""
    lg, ml = summary["logistic"], summary["mlp"]
    better = all(ml[k][0] - lg[k][0] > MLP_MARGIN for k in ("rocAuc", "matchedAucNSta"))
    return "mlp" if better else "logistic"


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


def run(data_dir: Path, out_dir: Path) -> dict[str, Any]:
    pos = pd.read_parquet(data_dir / "positives.parquet")
    neg = pd.read_parquet(data_dir / "negatives.parquet")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    df = pd.concat([pos, neg], ignore_index=True)
    y = df["label"].to_numpy()
    fold = make_folds(y, df["shuffle"].to_numpy())
    names = list(MODEL_FEATURES)
    no_size = [n for n in names if n not in SIZE_FEATURES]
    no_prob = [n for n in names if n not in PICK_PROB_FEATURES]

    cv = {
        "logistic": cross_validate(df, fold, "logistic", names),
        "mlp": cross_validate(df, fold, "mlp", names),
        "logisticNoSize": cross_validate(df, fold, "logistic", no_size),
        "logisticNoPickProb": cross_validate(df, fold, "logistic", no_prob),
    }
    baseline = df["quality_nStations"].to_numpy(dtype=float)
    fold_summary = {k: summarize(v.folds) for k, v in cv.items()}
    fold_summary["nStationsOnly"] = summarize(
        [fold_metrics(df[fold == k], baseline[fold == k]) for k in np.unique(fold)]
    )
    pooled_all = {k: pooled(df, v.oof) for k, v in cv.items()}
    pooled_all["nStationsOnly"] = pooled(df, baseline)
    chosen = choose(fold_summary)
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
            "scoreLogistic": cv["logistic"].oof,
            "scoreMlp": cv["mlp"].oof,
            "scoreLogisticNoSize": cv["logisticNoSize"].oof,
            "decoyExceedance": decoy_exceedance(score, y, fold),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sc.to_parquet(out_dir / "scores.parquet", index=False)

    dec = sc[~real]
    rs = sc[real]
    thresholds = {}
    for fpr in (0.10, 0.05, 0.01):
        t = threshold_at_fpr(dec["score"].to_numpy(), fpr)
        thresholds[f"decoyFpr{int(fpr * 100)}pct"] = {
            "threshold": round(t, 4),
            "decoysAbove": int((dec["score"] > t).sum()),
            "realAbove": int((rs["score"] > t).sum()),
            "realTierCAbove": f"{int((rs[rs.tier == 'C'].score > t).sum())}/{int((rs.tier == 'C').sum())}",
            "realNStaLe10Above": f"{int((rs[rs.nStations <= 10].score > t).sum())}/{int((rs.nStations <= 10).sum())}",
        }
    for cut in (0.5, 0.9):
        thresholds[f"score{cut}"] = {
            "decoysAbove": f"{int((dec.score > cut).sum())}/{len(dec)}",
            "realAbove": f"{int((rs.score > cut).sum())}/{len(rs)}",
        }

    top_decoys = dec.sort_values("nStations", ascending=False).head(10)
    sanity = {
        "catalogMatched": _describe(rs[rs.catalogMatched].score),
        "catalogMatchedMin": round(float(rs[rs.catalogMatched].score.min()), 4),
        "byTier": {t: _describe(g.score) for t, g in rs.groupby("tier")},
        "realNSta5to10": _describe(rs[(rs.nStations >= 5) & (rs.nStations <= 10)].score),
        "decoys": _describe(dec.score),
        "decoysNSta5to10": _describe(dec[(dec.nStations >= 5) & (dec.nStations <= 10)].score),
        "decoysReachedTierB": int((dec.tier != "C").sum()),
        "largestDecoys": top_decoys[["shuffle", "eventId", "nStations", "score"]]
        .round(4)
        .to_dict("records"),
        "realDecoyExceedance": {
            f"<={c}": int((rs.decoyExceedance <= c).sum()) for c in (0.01, 0.05)
        },
        "spearmanScoreVsNStationsReal": round(float(spearmanr(rs.score, rs.nStations)[0]), 3),
    }

    prep_all = fit_prep(df, names)
    full = fit_logistic(apply_prep(prep_all, df), y, class_weights(y))
    coef_folds = np.array(cv["logistic"].coefs)
    weights = pd.DataFrame(
        {
            "feature": names,
            "coef": full.coef,
            "coefFoldStd": coef_folds.std(axis=0, ddof=1),
        }
    ).sort_values("coef", key=np.abs, ascending=False, ignore_index=True)
    perm = permutation_importance(df, fold, names)

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
            "features": names,
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
            "mlpMargin": MLP_MARGIN,
        },
        "foldSummary(mean,std,min,max)": fold_summary,
        "pooledOutOfFold": pooled_all,
        "chosen": chosen,
        "thresholds": thresholds,
        "sanity": sanity,
        "logisticWeights": weights.round(4).to_dict("records"),
        "permutationImportance": perm.round(4).to_dict("records"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
    doc = confidence_doc(report, sc[real])
    (out_dir / "confidence.json").write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
    return report


LABEL = "Decoy test"
DESCRIPTION = (
    "How much this event's pick timing looks like a real association rather than a"
    " scrambled-clock decoy, from a model trained on this run (not a probability that it is"
    " an earthquake)"
)


def confidence_doc(report: dict[str, Any], real: pd.DataFrame) -> dict[str, Any]:
    """The ``hq.confidence/1`` document the exporter reads from a run directory: out-of-fold
    scores of the real candidate events, rounded to 3 decimals."""
    chosen = report["chosen"]
    folds = report["foldSummary(mean,std,min,max)"][chosen]
    data = report["data"]
    return {
        "schema": "hq.confidence/1",
        "runId": report["runId"],
        "model": {
            "type": f"{chosen} (class-weighted), out-of-fold scores",
            "features": data["features"],
            "trainedOn": {
                "positives": data["real"],
                "decoys": data["decoys"],
                "shuffles": data["shuffles"],
                "shiftS": data["shiftS"],
            },
            "heldOut": {
                "rocAuc": folds["rocAuc"][0],
                "rocAucStd": folds["rocAuc"][1],
                "averagePrecision": folds["averagePrecision"][0],
                "rocAucEqualStationCount": folds["matchedAucNSta"][0],
                "folds": report["folds"]["n"],
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
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = run(args.data_dir, args.out_dir)
    log.info("chosen=%s pooled=%s", report["chosen"], json.dumps(report["pooledOutOfFold"]))


if __name__ == "__main__":
    main()

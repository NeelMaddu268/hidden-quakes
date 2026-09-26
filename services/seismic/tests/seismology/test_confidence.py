"""ML-01 chance-association classifier: pure functions on small synthetic tables."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hq.tier import confidence as cf
from hq.tier.confidence_data import FEATURES, META

pytestmark = pytest.mark.smoke


def test_model_features_carry_no_labels_or_metadata() -> None:
    assert set(cf.MODEL_FEATURES) <= set(FEATURES)
    assert set(cf.MODEL_FEATURES).isdisjoint(META)
    assert set(cf.REDUNDANT_FEATURES).isdisjoint(cf.MODEL_FEATURES)


def test_transform_logs_seconds_and_meters_and_keeps_nan() -> None:
    df = pd.DataFrame(
        {
            "quality_rmsS": [0.009, 0.099],
            "quality_hErrM": [99.0, np.nan],
            "hErrNull": [False, True],
            "quality_nStations": [5, 12],
        }
    )
    x = cf.transform(df, ["quality_rmsS", "quality_hErrM", "hErrNull", "quality_nStations"])
    np.testing.assert_allclose(x[:, 0], [-2.0, -1.0])
    assert x[0, 1] == pytest.approx(2.0) and math.isnan(x[1, 1])
    np.testing.assert_array_equal(x[:, 2], [0.0, 1.0])
    np.testing.assert_array_equal(x[:, 3], [5.0, 12.0])


def test_prep_imputes_training_median_and_handles_constant_and_all_null() -> None:
    train = pd.DataFrame(
        {
            "quality_nS": [1.0, 2.0, 3.0, np.nan],
            "minPickProb": [0.5] * 4,
            "medianPickProbS": [np.nan] * 4,
        }
    )
    names = ["quality_nS", "minPickProb", "medianPickProbS"]
    prep = cf.fit_prep(train, names)
    np.testing.assert_allclose(prep.fill, [2.0, 0.5, 0.0])
    assert prep.std[1] == 1.0 and prep.std[2] == 1.0
    test = pd.DataFrame(
        {"quality_nS": [np.nan, 100.0], "minPickProb": [0.5, 0.5], "medianPickProbS": [np.nan, 0.7]}
    )
    x = cf.apply_prep(prep, test)
    assert not np.isnan(x).any()
    assert x[0, 0] == pytest.approx(0.0)  # filled with the training median = training mean
    np.testing.assert_allclose(x[:, 1], 0.0)
    # fitting on the test rows would change the fill; applying must not
    np.testing.assert_allclose(cf.fit_prep(train, names).fill, prep.fill)


def test_class_weights_balance_the_classes() -> None:
    y = np.array([1, 0, 0, 0, 0, 0])
    w = cf.class_weights(y)
    assert w[y == 1].sum() == pytest.approx(3.0)
    assert w[y == 0].sum() == pytest.approx(3.0)


def test_roc_auc_known_values() -> None:
    y = np.array([0, 0, 1, 1])
    assert cf.roc_auc(y, np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert cf.roc_auc(y, np.array([0.4, 0.3, 0.2, 0.1])) == 0.0
    assert cf.roc_auc(y, np.ones(4)) == 0.5
    assert cf.roc_auc(y, np.array([0.1, 0.35, 0.3, 0.4])) == 0.75
    assert math.isnan(cf.roc_auc(np.ones(3), np.arange(3)))


def test_average_precision_known_values() -> None:
    y = np.array([1, 0, 1, 0])
    assert cf.average_precision(y, np.array([0.9, 0.8, 0.7, 0.6])) == pytest.approx(
        0.5 * 1.0 + 0.5 * 2 / 3
    )
    assert cf.average_precision(y, np.array([0.9, 0.1, 0.8, 0.2])) == 1.0
    # all tied: one threshold, precision = prevalence
    assert cf.average_precision(y, np.ones(4)) == pytest.approx(0.5)


def test_matched_auc_ignores_the_stratum_itself() -> None:
    strata = np.array([5, 5, 5, 10, 10, 10])
    y = np.array([1, 0, 0, 1, 1, 0])
    auc, pairs = cf.matched_auc(y, strata.astype(float), strata)  # score = stratum only
    assert auc == 0.5 and pairs == 1 * 2 + 2 * 1
    auc, _ = cf.matched_auc(y, np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.05]), strata)
    assert auc == 1.0
    assert math.isnan(cf.matched_auc(y, y.astype(float), np.arange(6))[0])


def test_threshold_at_fpr_bounds_decoys_above() -> None:
    dec = np.linspace(0.0, 1.0, 101)
    for fpr in (0.01, 0.05, 0.10):
        t = cf.threshold_at_fpr(dec, fpr)
        assert (dec > t).mean() <= fpr + 1e-12


def test_decoy_exceedance_uses_same_fold_decoys() -> None:
    score = np.array([0.99, 0.5, 0.0, 0.1, 0.6, 0.2, 0.9])
    labels = np.array([1, 1, 1, 0, 0, 0, 0])
    fold = np.array([0, 0, 0, 0, 0, 0, 1])  # the 0.9 decoy is in another fold
    out = cf.decoy_exceedance(score, labels, fold)
    assert out[0] == pytest.approx(1 / 4)  # no fold-0 decoy >= 0.99
    assert out[1] == pytest.approx(2 / 4)  # 0.6 >= 0.5
    assert out[2] == pytest.approx(4 / 4)


def test_make_folds_groups_decoys_by_shuffle_and_is_deterministic() -> None:
    labels = np.r_[np.ones(50, int), np.zeros(200, int)]
    shuffles = np.r_[np.full(50, -1), np.repeat(np.arange(20), 10)]
    fold = cf.make_folds(labels, shuffles, n_folds=5, seed=0)
    assert (fold >= 0).all()
    for s in range(20):
        assert len(set(fold[shuffles == s])) == 1
    assert np.bincount(fold[labels == 1], minlength=5).tolist() == [10] * 5
    assert np.bincount(fold[labels == 0], minlength=5).tolist() == [40] * 5
    np.testing.assert_array_equal(fold, cf.make_folds(labels, shuffles, n_folds=5, seed=0))


def _toy(n: int = 300, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    x = rng.normal(size=(n, 3))
    x[:, 0] += 2.0 * y
    x[:, 1] -= 1.0 * y
    return x, y


def test_fit_logistic_recovers_signs_and_shrinks_with_l2() -> None:
    x, y = _toy()
    w = cf.class_weights(y)
    model = cf.fit_logistic(x, y, w)
    assert model.coef[0] > 0 > model.coef[1]
    assert abs(model.coef[2]) < abs(model.coef[1])
    assert cf.roc_auc(y, model(x)) > 0.9
    strong = cf.fit_logistic(x, y, w, l2=10.0)
    assert np.linalg.norm(strong.coef) < np.linalg.norm(model.coef)


def test_fit_mlp_is_deterministic_and_learns() -> None:
    x, y = _toy()
    w = cf.class_weights(y)
    p1 = cf.fit_mlp(x, y, w, seed=3)(x)
    p2 = cf.fit_mlp(x, y, w, seed=3)(x)
    np.testing.assert_array_equal(p1, p2)
    assert ((p1 >= 0) & (p1 <= 1)).all()
    assert cf.roc_auc(y, p1) > 0.9


def test_cross_validate_scores_every_row_out_of_fold() -> None:
    x, y = _toy(n=200)
    df = pd.DataFrame(
        {
            "quality_nP": x[:, 0],
            "quality_nS": x[:, 1],
            "minPickProb": x[:, 2],
            "quality_nStations": np.where(y == 1, 8, 6),
            "label": y,
            "shuffle": np.where(y == 1, -1, np.arange(200) % 10),
            "tier": "C",
        }
    )
    fold = cf.make_folds(df["label"].to_numpy(), df["shuffle"].to_numpy())
    res = cf.cross_validate(df, fold, "logistic", ["quality_nP", "quality_nS", "minPickProb"])
    assert not np.isnan(res.oof).any()
    assert len(res.folds) == cf.N_FOLDS and len(res.coefs) == cf.N_FOLDS
    assert res.folds[0]["rocAuc"] > 0.8


def test_choose_prefers_logistic_unless_mlp_clearly_better() -> None:
    def s(auc: float, matched: float) -> dict[str, list[float]]:
        return {"rocAuc": [auc], "matchedAucNSta": [matched]}

    assert cf.choose({"logistic": s(0.95, 0.90), "mlp": s(0.955, 0.95)}) == "logistic"
    assert cf.choose({"logistic": s(0.95, 0.90), "mlp": s(0.97, 0.92)}) == "mlp"


def test_confidence_doc_format_and_wording() -> None:
    report = {
        "chosen": "logistic",
        "runId": "r1",
        "createdAt": "2026-09-26T00:00:00Z",
        "gitSha": "abc",
        "data": {"features": ["quality_nP"], "real": 2, "decoys": 3, "shuffles": 1, "shiftS": 30.0},
        "folds": {"n": 5},
        "foldSummary(mean,std,min,max)": {
            "logistic": {
                "rocAuc": [0.99, 0.01, 0.98, 1.0],
                "averagePrecision": [0.98, 0.01, 0.97, 0.99],
                "matchedAucNSta": [0.95, 0.03, 0.9, 0.98],
            }
        },
    }
    real = pd.DataFrame({"eventId": ["e1", "e2"], "score": [0.12345, 1.0]})
    doc = cf.confidence_doc(report, real)
    assert doc["schema"] == "hq.confidence/1" and doc["runId"] == "r1"
    assert doc["events"] == {"e1": 0.123, "e2": 1.0}
    assert doc["model"]["heldOut"]["rocAuc"] == 0.99
    # docs/00 language rules (the exporter rejects these phrases in label/description)
    for text in (doc["label"], doc["description"]):
        for phrase in ("confirmed earthquake", "caused by", "predict", "official"):
            assert phrase not in text.lower()

"""ML-01 training data: decoys are the null test's shuffles, features carry no time/position/id.

Small synthetic tables built here; nothing locates.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hq.tier import confidence_data as cd
from hq.validate.null_test import shift_picks, station_shifts


def _picks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["p1", "p2", "p3", "p4", "p5"],
            "stationId": ["A", "A", "B", "B", "C"],
            "phase": ["P", "S", "P", "S", "P"],
            "t": [10.0, 11.0, 10.5, 11.8, 10.9],
            "prob": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )


def _inputs(picks: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        picks=picks, null_cfg=SimpleNamespace(profile="full", seed=0, shiftS=30.0)
    )


def test_real_rerun_takes_the_picks_unchanged() -> None:
    picks = _picks()
    pd.testing.assert_frame_equal(cd.shuffled_picks(_inputs(picks), cd.REAL), picks)


def test_shuffle_i_is_the_null_tests_shuffle_i() -> None:
    picks = _picks()
    for i in (0, 7, 25):
        rng = np.random.default_rng([0, i])
        want = shift_picks(picks, station_shifts(["A", "B", "C"], rng, 30.0))
        pd.testing.assert_frame_equal(cd.shuffled_picks(_inputs(picks), i), want)


def test_features_exclude_time_position_and_ids() -> None:
    leaks = {"t", "latitude", "longitude", "elevM", "depthKm", "enu_e", "enu_n", "enu_u", "id",
             "eventId", "rerunEventId", "assocId", "tier", "catalogMatched", "label", "shuffle"}  # fmt: skip
    assert not leaks & set(cd.FEATURES)
    assert len(set(cd.FEATURES)) == len(cd.FEATURES)


def test_event_features_from_arrivals_and_flags() -> None:
    events = pd.DataFrame(
        {
            "id": ["e0"],
            "quality_nStations": [2],
            "quality_nP": [2],
            "quality_nS": [1],
            "quality_rmsS": [0.05],
            "quality_gapDeg": [180.0],
            "quality_minEpiDistM": [1000.0],
            "quality_hErrM": [None],
            "quality_vErrM": [120.0],
            "quality_depthOnEdge": [False],
            "meanPickProb": [0.8],
            "tier": ["C"],
        }
    )
    arrivals = pd.DataFrame(
        {
            "eventId": ["e0"] * 6,
            "stationId": ["A", "A", "B", "B", "C", "C"],
            "phase": ["P", "S", "P", "S", "P", "S"],
            "residualS": [0.1, -0.2, 0.3, math.nan, -0.4, math.nan],
            "pickId": ["p1", "p2", "p3", None, "p5", None],
            "usedInLocation": [True, True, True, False, False, False],
        }
    )
    flags = pd.DataFrame(
        {"eventId": ["e0"], "assocId": ["assoc-000000"], **{c: [False] for c in cd.FLAG_FEATURES}}
    )
    matches = pd.DataFrame({"catalogId": ["x"], "eventId": [None]})
    row = cd.event_features(events, arrivals, flags, matches, _picks(), 25).iloc[0]
    assert row["nPicksAssociated"] == 4 and row["nPicksUsed"] == 3 and row["nPicksDropped"] == 1
    assert row["fracPicksDropped"] == 0.25
    assert math.isclose(row["medAbsResidualS"], 0.2) and math.isclose(row["maxAbsResidualS"], 0.3)
    assert math.isclose(row["medAbsResidualP_S"], 0.2)
    assert math.isclose(row["medianPickProb"], 0.8) and math.isclose(row["minPickProb"], 0.7)
    assert row["nStationsPandS"] == 1 and math.isclose(row["fracStationsUsed"], 2 / 25)
    assert row["hErrNull"] and not row["vErrNull"]
    assert row["assocId"] == "assoc-000000" and not row["catalogMatched"]

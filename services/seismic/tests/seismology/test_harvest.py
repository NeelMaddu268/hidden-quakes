"""LOC-10: pick harvest at predicted arrivals (hq.locate.harvest) and its config section.

Offline. The smoke tests work on small tables built here.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hq.config.seismology import PhaseSigma, SeismologyConfig
from hq.locate.harvest import (
    FREE_COLUMNS,
    SLOT_COLUMNS,
    UNTRUSTED,
    analytic_chance,
    free_picks,
    select,
    slot_table,
    untrusted_flags,
)

# --- config (smoke) ----------------------------------------------------------------------------


def _with_harvest(raw: dict[str, Any], **harvest: Any) -> dict[str, Any]:
    return {**raw, "harvest": {**raw["harvest"], **harvest}}


@pytest.mark.smoke
def test_showcase_config_ships_harvest_off(seismology_config: SeismologyConfig) -> None:
    h = seismology_config.harvest
    assert h.enabled is False
    assert h.windowS.P <= seismology_config.locator.outlier.floorS
    assert h.windowS.S <= seismology_config.locator.outlier.floorS
    assert h.minProb >= seismology_config.associator.minPickProb


@pytest.mark.smoke
def test_harvest_config_rejects_bad_values(seismology_config: SeismologyConfig) -> None:
    raw = seismology_config.model_dump(mode="json")
    floor = raw["locator"]["outlier"]["floorS"]
    with pytest.raises(ValueError, match="must not exceed locator.outlier.floorS"):
        SeismologyConfig.model_validate(_with_harvest(raw, windowS={"P": 0.1, "S": floor + 0.01}))
    with pytest.raises(ValueError, match="must not exceed locator.outlier.floorS"):
        SeismologyConfig.model_validate(_with_harvest(raw, windowS={"P": floor + 0.01, "S": 0.1}))
    below = raw["associator"]["minPickProb"] - 0.05
    with pytest.raises(ValueError, match="must not be below associator.minPickProb"):
        SeismologyConfig.model_validate(_with_harvest(raw, minProb=below))
    for bad in ({"iterations": 1}, {"windowS": {"P": 0.0, "S": 0.1}}, {"minProb": 1.5}):
        with pytest.raises(ValueError):
            SeismologyConfig.model_validate(_with_harvest(raw, **bad))
    missing = {k: v for k, v in raw.items() if k != "harvest"}
    with pytest.raises(ValueError, match="harvest"):
        SeismologyConfig.model_validate(missing)
    # The boundary itself is allowed.
    ok = SeismologyConfig.model_validate(_with_harvest(raw, windowS={"P": floor, "S": floor}))
    assert ok.harvest.windowS.S == floor


# --- selection rules on small tables (smoke) --------------------------------------------------

WINDOW = PhaseSigma(P=0.10, S=0.15)


def _slots(*rows: tuple[int, str, str, float, bool, bool]) -> pd.DataFrame:
    """(event, stationId, phase, tPred, filled, eligible) rows."""
    return pd.DataFrame(list(rows), columns=list(SLOT_COLUMNS))


def _free(*rows: tuple[str, str, str, float, float]) -> pd.DataFrame:
    """(id, stationId, phase, t, prob) rows."""
    return pd.DataFrame(list(rows), columns=list(FREE_COLUMNS))


def _picked(chosen: pd.DataFrame) -> dict[tuple[int, str, str], str]:
    return {(int(r.event), r.stationId, r.phase): r.pickId for r in chosen.itertuples()}


@pytest.mark.smoke
def test_window_is_per_phase_and_inclusive() -> None:
    slots = _slots((0, "A", "P", 100.0, False, True), (0, "B", "P", 100.0, False, True),
                   (0, "C", "P", 100.0, False, True), (0, "A", "S", 102.0, False, True),
                   (0, "B", "S", 102.0, False, True), (0, "C", "S", 102.0, False, True))
    free = _free(("pA", "A", "P", 100.0 + 0.10, 0.9),  # on the P boundary: in
                 ("pB", "B", "P", 100.0 - 0.10 - 1e-6, 0.9),  # just outside
                 ("pC", "C", "P", 100.0 + 0.12, 0.9),  # inside the S width, outside P's
                 ("sA", "A", "S", 102.0 - 0.15, 0.9),  # on the S boundary: in
                 ("sB", "B", "S", 102.0 + 0.15 + 1e-6, 0.9),  # just outside
                 ("sC", "C", "S", 102.0 + 0.12, 0.9))  # S width: in
    chosen, counts = select(slots, free, WINDOW)
    assert _picked(chosen) == {(0, "A", "P"): "pA", (0, "A", "S"): "sA", (0, "C", "S"): "sC"}
    got = chosen.set_index("pickId")
    assert got.loc["sC", "offsetS"] == pytest.approx(0.12)
    assert got.loc["sA", "tPred"] == 102.0
    assert counts == {"ambiguousPicks": 0, "skippedUntrustedSlots": 0,
                      "skippedMultiCandidateSlots": 0}


@pytest.mark.smoke
def test_free_picks_keep_unassociated_picks_above_min_prob_on_locator_stations() -> None:
    picks = pd.DataFrame({
        "id": ["a1", "f1", "f2", "f3", "f4", "f5"],
        "stationId": ["A", "A", "A", "Z", "A", "A"],
        "phase": ["S", "S", "P", "S", "S", "X"],
        "t": [10.0, 11.0, 9.0, 11.0, 12.0, 13.0],
        "prob": [0.9, 0.9, 0.29, 0.9, 0.3, 0.9],
        "picker": "phasenet:test",
    })
    free = free_picks(picks, {"a1"}, ["A", "B"], 0.3)
    # a1: associated; f2: below minProb; f3: not a locator station; f5: unknown phase.
    assert free["id"].tolist() == ["f1", "f4"]
    assert list(free.columns) == list(FREE_COLUMNS)


@pytest.mark.smoke
def test_never_steals_an_associated_pick_or_refills_a_slot() -> None:
    # The associated pick a1 sits right at event 1's empty slot; it is not free, so it is never
    # a candidate. Event 0's A S slot holds its own pick (filled; an outlier-dropped associated
    # pick counts the same), so the free pick next to it is not added.
    picks = pd.DataFrame({"id": ["a1", "f1"], "stationId": ["A", "A"], "phase": ["S", "S"],
                          "t": [50.0, 20.02], "prob": [0.9, 0.9]})
    free = free_picks(picks, {"a1"}, ["A"], 0.3)
    slots = _slots((0, "A", "S", 20.0, True, True), (1, "A", "S", 50.0, False, True))
    chosen, _ = select(slots, free, WINDOW)
    assert chosen.empty


@pytest.mark.smoke
def test_a_slot_with_two_candidates_is_skipped_and_counted() -> None:
    slots = _slots((0, "A", "S", 20.0, False, True), (0, "B", "S", 21.0, False, True))
    free = _free(("f1", "A", "S", 19.95, 0.9), ("f2", "A", "S", 20.05, 0.8),
                 ("f3", "B", "S", 21.01, 0.9))
    chosen, counts = select(slots, free, WINDOW)
    assert _picked(chosen) == {(0, "B", "S"): "f3"}
    assert counts["skippedMultiCandidateSlots"] == 1


@pytest.mark.smoke
def test_a_pick_near_two_events_is_ambiguous_even_when_one_slot_is_filled() -> None:
    # f1 lies within the S window of events 0 (empty slot) and 1 (filled slot): likely event 1's
    # arrival picked twice, so neither gets it. f2 lies within two empty slots: nobody gets it.
    slots = _slots((0, "A", "S", 20.00, False, True), (1, "A", "S", 20.20, True, True),
                   (2, "A", "S", 40.00, False, True), (3, "A", "S", 40.25, False, True))
    free = _free(("f1", "A", "S", 20.10, 0.9), ("f2", "A", "S", 40.12, 0.9))
    chosen, counts = select(slots, free, WINDOW)
    assert chosen.empty
    assert counts["ambiguousPicks"] == 2
    # Without event 1's row, f1 is event 0's alone.
    alone, _ = select(slots[slots["event"] != 1], free, WINDOW)
    assert _picked(alone) == {(0, "A", "S"): "f1"}


class _FakeLocator:
    station_ids = ("A", "B")

    @staticmethod
    def travel_times(e_m: float, n_m: float, elev_m: float) -> pd.DataFrame:
        return pd.DataFrame({"stationId": ["A", "A", "B", "B"], "phase": ["P", "S", "P", "S"],
                             "travelTimeS": [1.0, 2.0, 1.5, 3.0]})


def _loc(t0: float, **flags: bool) -> SimpleNamespace:
    return SimpleNamespace(e_m=0.0, n_m=0.0, elev_m=0.0, t0=t0,
                           map_on_volume_top=flags.get("mapOnVolumeTop", False),
                           map_on_volume_bottom=flags.get("mapOnVolumeBottom", False),
                           pdf_truncated=flags.get("pdfTruncated", False),
                           depth_on_edge=flags.get("depthOnEdge", False))


@pytest.mark.smoke
def test_slot_table_predicts_with_each_events_statics_and_marks_untrusted_events() -> None:
    frames = [pd.DataFrame({"id": ["a"], "stationId": ["A"], "phase": ["P"], "t": [101.0],
                            "prob": [0.9]}),
              pd.DataFrame({"id": ["b"], "stationId": ["B"], "phase": ["S"], "t": [203.0],
                            "prob": [0.9]})]
    slots = slot_table(_FakeLocator(), [_loc(100.0), _loc(200.0, pdfTruncated=True)], frames,
                       [{("A", "S"): 0.5}, {}])
    s = slots.set_index(["event", "stationId", "phase"])
    assert s.loc[(0, "A", "S"), "tPred"] == 100.0 + 2.0 + 0.5
    assert s.loc[(1, "A", "S"), "tPred"] == 200.0 + 2.0
    assert bool(s.loc[(0, "A", "P"), "filled"]) and not bool(s.loc[(0, "A", "S"), "filled"])
    assert s.loc[0, "eligible"].all() and not s.loc[1, "eligible"].any()
    for flag in UNTRUSTED:
        assert untrusted_flags(_loc(0.0, **{flag: True})) == [flag]  # type: ignore[arg-type]
    assert untrusted_flags(_loc(0.0)) == []  # type: ignore[arg-type]


@pytest.mark.smoke
def test_untrusted_events_are_skipped_but_their_windows_still_count() -> None:
    slots = _slots((0, "A", "S", 20.00, False, False),  # untrusted, empty: skipped and counted
                   (1, "A", "S", 60.00, False, True), (2, "A", "S", 60.10, False, False))
    free = _free(("f1", "A", "S", 20.01, 0.9), ("f2", "A", "S", 60.05, 0.9))
    chosen, counts = select(slots, free, WINDOW)
    assert chosen.empty  # f2 is ambiguous: untrusted event 2's window counts
    assert counts["skippedUntrustedSlots"] == 1 and counts["ambiguousPicks"] == 1


@pytest.mark.smoke
def test_decoys_are_rejected_and_the_true_pick_found_in_any_input_order() -> None:
    rng = np.random.default_rng(10)
    slots = _slots(
        (0, "A", "S", 100.0, False, True),  # the true pick f_true
        (0, "A", "P", 99.0, False, True),  # decoy: a P-labelled pick sits at S tPred
        (1, "B", "S", 300.0, False, True), (2, "B", "S", 300.1, False, True),  # share f_amb
        (3, "C", "S", 500.0, False, False),  # a face event: its pick stays
        (4, "D", "S", 700.0, False, True),  # chance picks far from tPred only
    )
    free = _free(("f_true", "A", "S", 100.03, 0.8), ("f_lbl", "A", "P", 100.0, 0.9),
                 ("f_amb", "B", "S", 300.05, 0.9), ("f_face", "C", "S", 500.01, 0.9),
                 *[(f"n{k}", "D", "S", float(t), 0.5)
                   for k, t in enumerate(rng.uniform(701.0, 900.0, 20))])
    chosen, counts = select(slots, free, WINDOW)
    assert _picked(chosen) == {(0, "A", "S"): "f_true"}
    assert counts == {"ambiguousPicks": 1, "skippedUntrustedSlots": 1,
                      "skippedMultiCandidateSlots": 0}
    for seed in range(3):
        shuffled, again = select(slots.sample(frac=1.0, random_state=seed),
                                 free.sample(frac=1.0, random_state=seed + 7), WINDOW)
        pd.testing.assert_frame_equal(shuffled, chosen)
        assert again == counts


@pytest.mark.smoke
def test_analytic_chance_counts_open_slots_of_trusted_events_only() -> None:
    slots = _slots((0, "A", "S", 10.0, False, True), (1, "A", "S", 50.0, True, True),
                   (2, "A", "S", 90.0, False, False))
    free = _free(*[(f"f{k}", "A", "S", float(k), 0.9) for k in range(10)])
    # rate 10 picks / 100 s, one open trusted slot, window 2 x 0.15 s.
    assert analytic_chance(slots, free, WINDOW, 100.0) == pytest.approx(0.1 * 0.3)

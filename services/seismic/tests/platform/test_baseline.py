"""VAL-01 acceptance: the baseline table (PhaseNet vs STA/LTA x association profiles) through
H2's associate -> locate -> match -> assign_tiers reproduces byte for byte from the stored
tables, the gain is claimed only when it holds in both profiles, an STA/LTA set that matches
PhaseNet's strict count gives no gain (and logs the docs/03 kill switch), and the ``validate``
stage writes ``baseline.json``, ``gr.json``, ``validation_notes.json`` and ``validation.json``
with counts. The G-R public curve is drawn on one magnitude scale (REQ-H2-13) and H2's gate and
null model are read from the run record (FYI-H2-7). Offline: the toy seismology API and planted
picks come from ``test_null_test``."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from hq_contracts import models as m
from hq_contracts.io import read_table, to_frame, write_models

from hq import runs
from hq.config import RunConfig
from hq.config.run import RunSection
from hq.config.validate import BaselineConfig, GRConfig
from hq.validate import (
    BASELINE_JSON,
    CALIBRATION_TYPE_RECORD,
    GATE_RECORD,
    GR_JSON,
    NOTES_JSON,
    NULL_TEST_JSON,
    PUBLIC_MAG_TYPE_KNOB,
    VALIDATION_JSON,
    ValidateError,
    effective_gr_config,
    magnitude_sets,
    public_mag_type,
    validate_run,
)
from hq.validate import run as validate_stage
from hq.validate.baseline import baseline_gain, baseline_reruns, index_rows, run_baseline
from hq.validate.sidecars import BASELINE, GR, MAGNITUDE
from tests.platform.test_null_test import (
    CATALOG_EVENT_INDICES,
    CATALOG_MAG_TYPE,
    N_MATCHED,
    N_STATIONS,
    NOW,
    RUN_TIERING,
    SHOWCASE_DIR,
    TIER_A_MIN_STATIONS,
    FakeSeismologyApi,
    SyntheticPicks,
    build_picks,
    read_notes,
    write_h2_validation_inputs,
    write_run_tables,
)

pytestmark = pytest.mark.smoke

N_KEEP = 6  # planted events kept (fewer picks than the null test uses: 4 reruns must be quick)
FULL_STATIONS = tuple(range(N_STATIONS))  # station indices STA/LTA sees on even events
SPARSE_STATIONS = tuple(range(TIER_A_MIN_STATIONS - 1))  # on odd events: one short of Tier A
NULL_SHUFFLES = 2  # the stage's null test, trimmed for speed (VAL-02 tests it at full length)
GR_MIN_EVENTS = 5  # trimmed too, so a handful of synthetic magnitudes makes a curve
N_MAG_EVENTS = 40
MAG_B = 1.0
MAG_MIN = 0.5
MAG_BIN = 0.1
STALTA_PICKER = "stalta"
ROW_ORDER = [("phasenet", "full"), ("phasenet", "p_only"), ("stalta", "full"), ("stalta", "p_only")]
OTHER_MAG_TYPE = "md"  # a second public scale (duration based) the curve must leave out
RECOVERED_MAG_TYPE = "ML_cal"


# --- fixtures -----------------------------------------------------------------------------------


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """configs/showcase with a faster validate.yaml (fewer null-test reruns, a low G-R floor)."""
    target = tmp_path / "config"
    target.mkdir()
    for path in sorted(SHOWCASE_DIR.glob("*.yaml")):
        shutil.copy(path, target / path.name)
    validate = yaml.safe_load((target / "validate.yaml").read_text())
    validate["nullTest"]["nShuffles"] = NULL_SHUFFLES
    validate["gr"]["minEvents"] = GR_MIN_EVENTS
    validate["gr"]["publicMagType"] = CATALOG_MAG_TYPE  # the fallback scale (REQ-H2-13)
    (target / "validate.yaml").write_text(yaml.safe_dump(validate))
    return target


@pytest.fixture
def ctx(config_dir: Path, tmp_path: Path) -> runs.RunContext:
    return runs.create_run(config_dir, tmp_path / "data", now=NOW)


@pytest.fixture
def section(ctx: runs.RunContext) -> RunSection:
    return ctx.config.run


@pytest.fixture
def synthetic(section: RunSection) -> SyntheticPicks:
    """The first ``N_KEEP`` planted events of ``test_null_test.build_picks``."""
    full = build_picks(section)
    cutoff = full.event_times[N_KEEP]
    kept = [i for i in CATALOG_EVENT_INDICES if i < N_KEEP]
    return SyntheticPicks(
        full.stations,
        [p for p in full.picks if p.t < cutoff],
        [c for c, i in zip(full.catalog, CATALOG_EVENT_INDICES, strict=True) if i in kept],
        full.event_times[:N_KEEP],
    )


def stalta_picks(synthetic: SyntheticPicks, *, sparse: bool = True) -> list[m.Pick]:
    """STA/LTA picks derived from the planted P picks: every station on even events, only
    ``SPARSE_STATIONS`` on odd ones (so half the events fall short of Tier A). With
    ``sparse=False`` it is PhaseNet's P set relabelled: same events, same strict count."""
    station_index = {s.id: i for i, s in enumerate(synthetic.stations)}
    times = np.array(synthetic.event_times)
    out: list[m.Pick] = []
    for p in synthetic.picks:
        if p.phase != "P":
            continue
        k = int(np.argmin(np.abs(times - p.t)))  # the planted event this pick belongs to
        allowed = FULL_STATIONS if (k % 2 == 0 or not sparse) else SPARSE_STATIONS
        if station_index[p.stationId] not in allowed:
            continue
        out.append(
            m.Pick(
                id=f"{STALTA_PICKER}:{p.stationId}:P:{p.t:.3f}",
                stationId=p.stationId,
                phase="P",
                t=p.t,
                prob=1.0,
                picker=STALTA_PICKER,
            )
        )
    return out


def magnitude_events(section: RunSection, ctx: runs.RunContext) -> list[m.SeismicEvent]:
    """Final ``SeismicEvent`` rows with magnitudes on a Gutenberg-Richter sample (b = 1),
    seeded, rounded to the bin width; only the magnitudes matter to the validate stage."""
    rng = np.random.default_rng(7)
    mags = MAG_MIN + rng.exponential(np.log10(np.e) / MAG_B, size=N_MAG_EVENTS)
    elev = section.refSurfaceElevM - 2_000.0
    events = []
    for k, mag in enumerate(np.round(mags, 1)):
        events.append(
            m.SeismicEvent(
                id=f"hq-{ctx.run_id}-{k + 1:06d}",
                runId=ctx.run_id,
                t=section.window_start_s + 10.0 * k,
                latitude=section.origin.lat,
                longitude=section.origin.lon,
                elevM=elev,
                depthKm=(section.refSurfaceElevM - elev) / 1000.0,
                enu=m.Enu(e=0.0, n=0.0, u=elev - section.origin.elevM),
                quality=m.LocationQuality(
                    method="grid1d",
                    statics=False,
                    nStations=8,
                    nP=8,
                    nS=6,
                    rmsS=0.04,
                    gapDeg=90.0,
                    minEpiDistM=3000.0,
                    hErrM=150.0,
                    vErrM=300.0,
                    depthOnEdge=False,
                ),
                tier="A",
                tierReasons=["test"],
                meanPickProb=0.9,
                magnitude=m.Magnitude(value=float(mag), type=RECOVERED_MAG_TYPE, sigma=0.2),
                revealOrder=-1,
                pickIds=[],
            )
        )
    return events


def calibration(loo_mae: float) -> m.MagCalibration:
    return m.MagCalibration(n=12, looMae=loo_mae, coefficients={"a": 1.0, "b": -1.5})


def record_magnitude_stage(
    ctx: runs.RunContext,
    *,
    mag_type: str = CATALOG_MAG_TYPE,
    gate: float = 0.4,
    null_model_mae: float | None = 0.5,
) -> None:
    """As H2's magnitude stage records itself (FYI-H2-7): the calibration type, the gate and
    the leave-one-event-out null model under ``ProcessingRun.matching["magnitude"]``."""
    params: dict[str, Any] = {
        "calibrationMagType": mag_type,
        "gate": {"maxLooMae": gate, "passed": True},
    }
    if null_model_mae is not None:
        params["leaveOneEventOut"] = {"n": 12, "nullModelMae": null_model_mae}
    ctx.record("magnitude", runtime_s=0.0, counts={}, params={"magnitude": params},
               field="matching")  # fmt: skip


def mixed_type_catalog(synthetic: SyntheticPicks, n_other: int) -> list[m.CatalogEvent]:
    """The planted catalog plus ``n_other`` public events on another magnitude scale, placed
    where no candidate matches them."""
    extra = [
        c.model_copy(
            update={
                "id": f"testpub-other{k + 1:03d}",
                "t": c.t + 7.0 * (k + 1),
                "mag": round(1.0 + 0.1 * k, 2),
                "magType": OTHER_MAG_TYPE,
            }
        )
        for k, c in enumerate(synthetic.catalog * (n_other // len(synthetic.catalog) + 1))
    ][:n_other]
    return [*synthetic.catalog, *extra]


def run_rows(
    synthetic: SyntheticPicks,
    stalta: list[m.Pick],
    ctx: runs.RunContext,
    api: FakeSeismologyApi | None = None,
    cfg: BaselineConfig | None = None,
) -> list[m.BaselineRow]:
    return run_baseline(
        synthetic.picks_frame,
        to_frame(stalta, m.Pick),
        synthetic.stations_frame,
        synthetic.catalog_frame,
        api or FakeSeismologyApi(),
        ctx.config.seismology,
        ctx.config.run,
        cfg or ctx.config.validate.baseline,
        thresholds=RUN_TIERING,
    )


# --- the table ---------------------------------------------------------------------------------


def test_rows_reproduce_byte_identically_from_stored_tables(
    synthetic: SyntheticPicks, ctx: runs.RunContext
) -> None:
    stalta = stalta_picks(synthetic)
    api = FakeSeismologyApi()
    rows = run_rows(synthetic, stalta, ctx, api)
    assert api.calls == {"associate": 4, "locate": 4, "match": 4, "assign_tiers": 4}
    assert [(r.method, r.associationProfile) for r in rows] == ROW_ORDER
    # REQ-H2-9: every rerun tiered with the run's bars, its arrivals and the stations table.
    assert all(kw["thresholds"] is RUN_TIERING for kw in api.tier_kwargs)
    assert all(kw["arrivals"] is not None and kw["stations"] is not None for kw in api.tier_kwargs)
    reruns = baseline_reruns(
        synthetic.picks_frame, to_frame(stalta, m.Pick), synthetic.stations_frame,
        synthetic.catalog_frame, FakeSeismologyApi(), ctx.config.seismology, ctx.config.run,
        ctx.config.validate.baseline, thresholds=RUN_TIERING,
    )  # fmt: skip
    assert [r.row for r in reruns] == rows
    assert all(r.tiering is not None and r.tiering["thresholdSource"] == "supplied" for r in reruns)
    with pytest.raises(ValidateError, match="baseline comparison.*'tier'.*H2 Seismology"):
        run_baseline(
            synthetic.picks_frame, to_frame(stalta, m.Pick), synthetic.stations_frame,
            synthetic.catalog_frame, FakeSeismologyApi(), ctx.config.seismology, ctx.config.run,
            ctx.config.validate.baseline, thresholds={},
        )  # fmt: skip
    n_public = len(synthetic.catalog)
    for row in rows[:2]:  # PhaseNet: every station picks every event -> all Tier A
        assert (row.candidates, row.recoveredPublic) == (N_KEEP, n_public)
        assert row.tiers == m.TierCounts(A=N_KEEP, B=0, C=0)
        assert (row.medianRmsS, row.medianStations) == (0.05, float(N_STATIONS))
    for row in rows[2:]:  # STA/LTA: odd events one station short of Tier A
        assert (row.candidates, row.recoveredPublic) == (N_KEEP, n_public)
        assert row.tiers == m.TierCounts(A=N_KEEP // 2, B=N_KEEP // 2, C=0)
        assert row.medianStations == float(np.median([N_STATIONS, len(SPARSE_STATIONS)]))
    # The same tables from parquet (what the stage reads) give the same bytes.
    write_models(synthetic.picks, ctx.path("picks.parquet"))
    write_models(stalta, ctx.path("picks_stalta.parquet"))
    again = run_baseline(
        read_table(ctx.path("picks.parquet")),
        read_table(ctx.path("picks_stalta.parquet")),
        synthetic.stations_frame,
        synthetic.catalog_frame,
        FakeSeismologyApi(),
        ctx.config.seismology,
        ctx.config.run,
        ctx.config.validate.baseline,
        thresholds=RUN_TIERING,
    )
    assert BASELINE.adapter.dump_json(again) == BASELINE.adapter.dump_json(rows)
    with pytest.raises(ValidateError, match="stalta picks name stations missing"):
        stray = stalta[0].model_copy(update={"stationId": "XT.S99"})
        run_rows(synthetic, [stray, *stalta[1:]], ctx)


def test_gain_only_when_it_holds_in_both_profiles(
    synthetic: SyntheticPicks, ctx: runs.RunContext
) -> None:
    cfg = ctx.config.validate.baseline
    api = FakeSeismologyApi()
    rows = run_rows(synthetic, stalta_picks(synthetic), ctx, api=api)
    gain = baseline_gain(rows, cfg)
    assert gain == m.BaselineGain(
        associationProfile="full", strictPhasenet=N_KEEP, strictStalta=N_KEEP // 2, gain=2.0
    )
    by_key = index_rows(rows)

    def with_strict(method: str, profile: str, a: int) -> list[m.BaselineRow]:
        row = by_key[(method, profile)]
        edited = row.model_copy(update={"tiers": row.tiers.model_copy(update={"A": a})})
        return [edited if (r.method, r.associationProfile) == (method, profile) else r
                for r in rows]  # fmt: skip

    assert baseline_gain([], cfg) is None
    assert baseline_gain(with_strict("stalta", "p_only", N_KEEP), cfg) is None  # p_only fails
    assert baseline_gain(with_strict("stalta", "full", N_KEEP + 1), cfg) is None  # full fails
    assert baseline_gain(with_strict("stalta", "p_only", 0), cfg) is None  # no STA/LTA strict
    assert baseline_gain(rows[:3], cfg) is None  # a profile missing
    assert baseline_gain([r for r in rows if r.associationProfile == "full"], cfg) is None
    with pytest.raises(ValidateError, match="several rows"):
        baseline_gain(rows + rows[:1], cfg)
    # The gain must exceed minGain; the kill-switch fraction never decides.
    assert baseline_gain(rows, cfg.model_copy(update={"minGain": 2.0})) is None
    assert baseline_gain(rows, cfg.model_copy(update={"minGain": 1.5})) == gain
    assert baseline_gain(rows, cfg.model_copy(update={"comparableFraction": 1.0})) == gain
    # p_only for PhaseNet feeds P picks only; the toy associator sees the same events.
    assert by_key[("phasenet", "p_only")].candidates == by_key[("phasenet", "full")].candidates
    # REQ-H2-7: the two p_only reruns used the associator overrides, the two full ones did not.
    overridden = [c for c in api.cfgs_seen if c is not ctx.config.seismology]
    assert len(api.cfgs_seen) == 4 and len(overridden) == 2
    assert {(c.associator.nSPicks, c.associator.nPAndSPicks) for c in overridden} == {(0, 0)}


def test_matching_strict_counts_give_no_gain_and_log_the_kill_switch(
    synthetic: SyntheticPicks, ctx: runs.RunContext, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="hq.validate")
    cfg = ctx.config.validate.baseline
    rows = run_rows(synthetic, stalta_picks(synthetic, sparse=False), ctx)
    assert all(r.tiers.A == N_KEEP for r in rows)
    assert baseline_gain(rows, cfg) is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert sum("kill switch" in w and "docs/03" in w for w in warnings) == 2  # both profiles
    assert any("not exceed" in r.getMessage() for r in caplog.records)
    # Within the fraction but still above minGain: the gain is claimed, the switch only logs.
    caplog.clear()
    close = [
        r.model_copy(
            update={"tiers": m.TierCounts(A=10 if r.method == "phasenet" else 9, B=0, C=0)}
        )
        for r in rows
    ]
    gain = baseline_gain(close, cfg)
    assert gain is not None and (gain.strictPhasenet, gain.strictStalta) == (10, 9)
    assert any("kill switch" in r.getMessage() for r in caplog.records)


# --- the stage -----------------------------------------------------------------------------------


def stage_counts(ctx: runs.RunContext) -> dict[str, Any]:
    return json.loads(ctx.path("stages.json").read_text())["validate"]["counts"]


def test_stage_writes_baseline_gr_and_validation_and_reproduces(
    ctx: runs.RunContext,
    synthetic: SyntheticPicks,
    section: RunSection,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="hq.validate")
    write_run_tables(ctx, synthetic)
    write_models(stalta_picks(synthetic), ctx.path("picks_stalta.parquet"))
    synthetic_test, sweep = write_h2_validation_inputs(ctx)
    events = magnitude_events(section, ctx)
    write_models(events, ctx.path("events.parquet"))
    MAGNITUDE.write(ctx.run_dir, calibration(0.2))
    api = FakeSeismologyApi()
    api.install(monkeypatch)
    cfg = ctx.config.validate

    runs.run_stage(ctx, "validate")

    assert api.calls["associate"] == NULL_SHUFFLES + 2 * len(cfg.baseline.profiles)
    validation = m.Validation.model_validate_json(ctx.path(VALIDATION_JSON).read_text())
    assert validation.synthetic == synthetic_test and validation.sweep == sweep
    assert validation.nullTest == m.NullTest.model_validate_json(
        ctx.path(NULL_TEST_JSON).read_text()
    )
    rows = BASELINE.read(ctx.run_dir, ValidateError)
    assert validation.baseline == rows and len(rows) == 4
    assert [(r.method, r.associationProfile) for r in rows] == ROW_ORDER
    gr = GR.read(ctx.run_dir, ValidateError)
    assert validation.gr == gr and gr is not None
    public_mags = sorted(c.mag for c in synthetic.catalog if c.mag is not None)
    recovered_mags = [e.magnitude.value for e in events if e.magnitude is not None]
    assert gr.magBins[0] <= min(public_mags + recovered_mags) <= gr.magBins[1]
    assert gr.magBins[-1] >= max(public_mags + recovered_mags)
    assert gr.publicCum[0] == len(public_mags) and gr.recoveredCum[0] == len(recovered_mags)
    assert gr.mcPublic is None  # fewer public magnitudes than minEvents
    assert gr.mcRecovered is not None and gr.bValue is not None and gr.bSigma is not None
    assert 0.6 < gr.bValue < 1.6 and 0.0 < gr.bSigma < 1.0  # a 40-event sample, loosely
    assert validation.magnitude == calibration(0.2)
    notes = read_notes(ctx)
    assert notes.baseline is not None and notes.baseline.reruns == 4
    assert notes.baseline.rerunsTiered == 4 and notes.baseline.staticsApplied is False
    assert notes.baseline.thresholds.nMatched == N_MATCHED
    assert notes.baseline.associatorOverrides == {"p_only": {"nSPicks": 0, "nPAndSPicks": 0}}
    assert notes.baseline.tieringRules is not None
    assert notes.baseline.tieringRules.mapOnVolumeTopApplied is False
    assert notes.gr is not None
    assert (notes.gr.magType, notes.gr.magTypeSource) == (CATALOG_MAG_TYPE, PUBLIC_MAG_TYPE_KNOB)
    assert notes.gr.publicIncluded == len(public_mags) and notes.gr.publicExcludedByType == {}
    assert notes.gr.recoveredMagTypes == [RECOVERED_MAG_TYPE]
    assert (notes.gr.maxLooMae, notes.gr.looMae) == (cfg.gr.maxLooMae, 0.2)
    assert (notes.gr.nullModelMae, notes.gr.skill) == (None, None)  # no H2 record on this run
    counts = stage_counts(ctx)
    assert counts["baselineRows"] == 4 and counts["baselineGain"] == 1
    assert counts["publicMagnitudes"] == len(public_mags)
    assert counts["publicMagnitudesExcluded"] == 0
    assert counts["recoveredMagnitudes"] == len(recovered_mags)
    assert counts["grBins"] == len(gr.magBins) and counts["hasMagnitude"] == 1
    assert counts["validationJson"] == 1 and counts["nullShuffles"] == NULL_SHUFFLES
    messages = [r.getMessage() for r in caplog.records]
    assert sum(msg.startswith(("baseline phasenet/", "baseline stalta/")) for msg in messages) == 4
    assert any(msg.startswith("baseline: PhaseNet gain") for msg in messages)
    assert any(msg.startswith("G-R:") and "bins" in msg for msg in messages)
    assert not list(ctx.run_dir.glob("*.tmp"))

    # Same tables, same config: every file reproduces byte for byte.
    before = {
        name: ctx.path(name).read_bytes()
        for name in (NULL_TEST_JSON, BASELINE_JSON, GR_JSON, NOTES_JSON, VALIDATION_JSON)
    }
    validate_stage(ctx)
    assert {name: ctx.path(name).read_bytes() for name in before} == before

    # docs/03 magnitude kill switch: G-R skipped, the calibration still embedded, no stale gr.json.
    caplog.clear()
    MAGNITUDE.write(ctx.run_dir, calibration(cfg.gr.maxLooMae + 0.1))
    out = validate_run(ctx)
    assert out.gr is None and out.magnitude == calibration(cfg.gr.maxLooMae + 0.1)
    assert not ctx.path(GR_JSON).exists()
    validation = m.Validation.model_validate_json(ctx.path(VALIDATION_JSON).read_text())
    assert validation.gr is None and validation.magnitude == out.magnitude
    assert validation.baseline == rows
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("magnitude kill switch" in w and "looMae" in w for w in warnings)
    assert any("stale" in w and GR_JSON in w for w in warnings)


def test_stage_without_stalta_picks_or_magnitudes_names_the_owners(
    ctx: runs.RunContext,
    synthetic: SyntheticPicks,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="hq.validate")
    write_run_tables(ctx, synthetic)
    write_h2_validation_inputs(ctx)
    FakeSeismologyApi().install(monkeypatch)
    runs.run_stage(ctx, "validate")
    validation = m.Validation.model_validate_json(ctx.path(VALIDATION_JSON).read_text())
    assert validation.baseline == [] and validation.gr is None and validation.magnitude is None
    assert BASELINE.read(ctx.run_dir, ValidateError) == []
    assert not ctx.path(GR_JSON).exists()
    notes = read_notes(ctx)
    assert notes.baseline is None and notes.gr is not None and notes.gr.looMae is None
    counts = stage_counts(ctx)
    assert (counts["baselineRows"], counts["baselineGain"], counts["grBins"]) == (0, 0, 0)
    assert counts["hasMagnitude"] == 0 and counts["recoveredMagnitudes"] == 0
    assert counts["publicMagnitudes"] == sum(c.mag is not None for c in synthetic.catalog)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("picks_stalta.parquet" in w and "H1 Signal" in w for w in warnings)
    assert any("events.parquet" in w and "'tier'" in w and "H2 Seismology" in w for w in warnings)
    assert any("fewer than minEvents" in w for w in warnings)  # the catalog's few magnitudes
    # A corrupt magnitude.json fails loudly, naming the model.
    ctx.path("magnitude.json").write_text('{"n": 1}')
    with pytest.raises(ValidateError, match="not a valid MagCalibration"):
        validate_stage(ctx)


# --- G-R on one magnitude scale (REQ-H2-13) and H2's gate and null model (FYI-H2-7) -------------


def test_gr_public_curve_uses_the_calibration_type_only(
    ctx: runs.RunContext,
    synthetic: SyntheticPicks,
    section: RunSection,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A public catalog on two scales: publicCum counts the calibration type's magnitudes only,
    the others are counted per type in the notes and the run record's type wins over the knob."""
    caplog.set_level(logging.INFO, logger="hq.validate")
    write_run_tables(ctx, synthetic)
    n_other = 7
    catalog = mixed_type_catalog(synthetic, n_other)
    write_models(catalog, ctx.path("catalog.parquet"))
    write_h2_validation_inputs(ctx)
    events = magnitude_events(section, ctx)
    write_models(events, ctx.path("events.parquet"))
    MAGNITUDE.write(ctx.run_dir, calibration(0.2))
    record_magnitude_stage(ctx, mag_type=CATALOG_MAG_TYPE, null_model_mae=0.5)
    FakeSeismologyApi().install(monkeypatch)

    runs.run_stage(ctx, "validate")

    gr = GR.read(ctx.run_dir, ValidateError)
    assert gr is not None
    same_scale = [c.mag for c in catalog if c.magType == CATALOG_MAG_TYPE and c.mag is not None]
    assert gr.publicCum[0] == len(same_scale) == len(synthetic.catalog)
    assert gr.recoveredCum[0] == len(events)
    notes = read_notes(ctx)
    assert notes.gr is not None
    assert (notes.gr.magType, notes.gr.magTypeSource) == (CATALOG_MAG_TYPE, CALIBRATION_TYPE_RECORD)
    assert notes.gr.publicIncluded == len(same_scale)
    assert notes.gr.publicExcludedByType == {OTHER_MAG_TYPE: n_other}
    assert (notes.gr.maxLooMae, notes.gr.maxLooMaeSource) == (0.4, GATE_RECORD)
    assert (notes.gr.looMae, notes.gr.nullModelMae, notes.gr.skill) == (0.2, 0.5, True)
    counts = stage_counts(ctx)
    assert counts["publicMagnitudes"] == len(same_scale)
    assert counts["publicMagnitudesExcluded"] == n_other
    messages = [r.getMessage() for r in caplog.records]
    assert any(f"{n_other} public magnitudes of other types left out" in msg for msg in messages)
    assert any("looMae 0.200 is below" in msg for msg in messages)
    assert not any("no skill" in msg for msg in messages)

    # The other type as calibration type: the curve flips to it; the knob's type is a warning.
    caplog.clear()
    record_magnitude_stage(ctx, mag_type=OTHER_MAG_TYPE, null_model_mae=0.5)
    out = validate_run(ctx)
    assert out.gr is not None and out.gr.publicCum[0] == n_other
    assert out.notes.gr is not None
    assert out.notes.gr.publicExcludedByType == {CATALOG_MAG_TYPE: len(synthetic.catalog)}
    assert (out.public_magnitudes, out.public_magnitudes_excluded) == (n_other, len(same_scale))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(PUBLIC_MAG_TYPE_KNOB in w and "calibration type" in w for w in warnings)

    # FYI-H2-7: a looMae not below H2's null model is a warning and skill false.
    caplog.clear()
    record_magnitude_stage(ctx, mag_type=CATALOG_MAG_TYPE, null_model_mae=0.15)
    out = validate_run(ctx)
    assert out.notes.gr is not None and out.notes.gr.skill is False
    assert out.notes.gr.nullModelMae == 0.15 and out.gr is not None  # the gate still passes
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("not below H2's null-model MAE" in w and "no skill" in w for w in warnings)

    # REQ-H2-13: H2's recorded gate is applied; a differing validate.yaml gate is a warning.
    caplog.clear()
    record_magnitude_stage(ctx, gate=0.1, null_model_mae=0.5)
    out = validate_run(ctx)
    assert out.gr is None and not ctx.path(GR_JSON).exists()
    assert out.notes.gr is not None and out.notes.gr.maxLooMae == 0.1
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("differs from" in w and GATE_RECORD in w for w in warnings)
    assert any("magnitude kill switch" in w and "0.100" in w for w in warnings)
    assert not list(ctx.run_dir.glob("*.tmp"))


def test_gr_without_a_public_magnitude_type_fails_naming_h2_and_the_knob(
    ctx: runs.RunContext, synthetic: SyntheticPicks, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_run_tables(ctx, synthetic)  # the planted catalog carries magnitudes
    FakeSeismologyApi().install(monkeypatch)
    cfg = ctx.config
    no_knob = cfg.validate.model_copy(
        update={"gr": cfg.validate.gr.model_copy(update={"publicMagType": None})}
    )
    bare = runs.RunContext(
        ctx.run_id, ctx.run_dir, ctx.cache_dir,
        RunConfig(run=cfg.run, signal=cfg.signal, seismology=cfg.seismology, export=cfg.export,
                  validate=no_knob),
    )  # fmt: skip
    with pytest.raises(ValidateError, match=r"public magnitudes.*H2 Seismology.*gr\.publicMagType"):
        validate_stage(bare)
    # With H2's record the knob is not needed; without public magnitudes neither is.
    assert public_mag_type({"calibrationMagType": "ml"}, GRConfig(), 3) == (
        "ml", CALIBRATION_TYPE_RECORD,
    )  # fmt: skip
    assert public_mag_type(None, GRConfig(), 0) == (None, None)
    assert public_mag_type({"failed": "boom"}, GRConfig(publicMagType="ml"), 3) == (
        "ml", PUBLIC_MAG_TYPE_KNOB,
    )  # fmt: skip
    with pytest.raises(ValidateError, match="not a magnitude type"):
        public_mag_type({"calibrationMagType": ""}, GRConfig(), 3)
    with pytest.raises(ValidateError, match="not a number"):
        effective_gr_config({"gate": {"maxLooMae": "0.4"}}, GRConfig())
    assert effective_gr_config(None, GRConfig()) == (GRConfig(), "validate.yaml gr.maxLooMae")
    # magnitude_sets: null magTypes are counted under "null"; no type selects nothing.
    catalog = synthetic.catalog_frame
    catalog.loc[0, "magType"] = None
    sets = magnitude_sets(catalog, None, CATALOG_MAG_TYPE)
    assert len(sets.public) == len(synthetic.catalog) - 1
    assert sets.public_excluded_by_type == {"null": 1}
    assert len(magnitude_sets(catalog, None, None).public) == 0

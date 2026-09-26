"""VAL-01 acceptance: the baseline table (PhaseNet vs STA/LTA x association profiles) through
H2's associate -> locate -> match -> assign_tiers reproduces byte for byte from the stored
tables, the gain is claimed only when it holds in both profiles, an STA/LTA set that matches
PhaseNet's strict count gives no gain (and logs the docs/03 kill switch), and the ``validate``
stage writes ``baseline.json``, ``gr.json`` and ``validation.json`` with counts. Offline: the
toy seismology API and planted picks come from ``test_null_test``."""

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
from hq.config.run import RunSection
from hq.config.validate import BaselineConfig
from hq.validate import (
    BASELINE_JSON,
    GR_JSON,
    NULL_TEST_JSON,
    VALIDATION_JSON,
    ValidateError,
    validate_run,
)
from hq.validate import run as validate_stage
from hq.validate.baseline import baseline_gain, index_rows, run_baseline
from hq.validate.sidecars import BASELINE, GR, MAGNITUDE
from tests.platform.test_null_test import (
    CATALOG_EVENT_INDICES,
    N_STATIONS,
    NOW,
    SHOWCASE_DIR,
    TIER_A_MIN_STATIONS,
    FakeSeismologyApi,
    SyntheticPicks,
    build_picks,
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
                magnitude=m.Magnitude(value=float(mag), type="ML_cal", sigma=0.2),
                revealOrder=-1,
                pickIds=[],
            )
        )
    return events


def calibration(loo_mae: float) -> m.MagCalibration:
    return m.MagCalibration(n=12, looMae=loo_mae, coefficients={"a": 1.0, "b": -1.5})


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
    counts = stage_counts(ctx)
    assert counts["baselineRows"] == 4 and counts["baselineGain"] == 1
    assert counts["publicMagnitudes"] == len(public_mags)
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
        for name in (NULL_TEST_JSON, BASELINE_JSON, GR_JSON, VALIDATION_JSON)
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

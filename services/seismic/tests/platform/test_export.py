"""API-02 acceptance: the exporter turns a run directory into a bundle whose every file
validates, whose summary recomputes from ``events.json``, whose evidence files stay under the
byte cap, and which is byte-identical run to run and never half-written. Offline: the run
tables and the waveforms are synthetic, built here with a seeded generator, and H1's cache
reader / display filter and FEAT-01's feature loader are stand-ins injected per test."""

import json
import logging
import math
import shutil
import sys
import types
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hq_contracts import models as m
from hq_contracts.io import write_models, write_table
from obspy import Stream, Trace, UTCDateTime

from hq import runs
from hq.config import load_config
from hq.config.export import EvidenceConfig, ExportConfig
from hq.config.run import RunSection
from hq.export import (
    BundleCheckError,
    ExportError,
    LaneWaveformSource,
    check_bundle,
    export_bundle,
    load_run_tables,
    output_root,
    real_waveform_source,
)
from hq.export.bundle import load_features_lazily
from hq.export.summary import baseline_gain, reveal_key
from hq.export.tables import is_null, str_or_none
from hq.locate.coords import from_enu
from hq.locate.result import ARRIVAL_DTYPES, ARRIVALS_MODEL, STATIC_DTYPES, STATICS_MODEL

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[4]
SHOWCASE_DIR = REPO_ROOT / "services" / "seismic" / "configs" / "showcase"
MOCK_BUNDLE = REPO_ROOT / "apps" / "web" / "public" / "data" / "mock"
NOW = datetime(2026, 9, 10, 4, 5, tzinfo=UTC)
SEED = 7
BUNDLE_FILES = ("meta.json", "stations.json", "catalog.json", "events.json", "features.json")
MAX_EVIDENCE_BYTES = 60 * 1024

# --- synthetic run knobs (test-local; CLAUDE.md rule 5 allows small synthetic data in tests) ---
VP_KM_S, VS_KM_S = 5.5, 3.2
RING_RADIUS_M = 6_000.0
N_RING = 6
UNUSED_RADIUS_M = 9_000.0
BOREHOLE_EN_M, BOREHOLE_DEPTH_M = (500.0, -300.0), 600.0
SURFACE_RATE_HZ, BOREHOLE_RATE_HZ = 100.0, 1000.0
PICKER = "phasenet:test"
TIER_PLAN = "ABCABCABCABC"  # interleaved in time, so reveal order != time order
N_STATIONS = {"A": 7, "B": 5, "C": 3}
RMS_S = {"A": (0.02, 0.05), "B": (0.06, 0.11), "C": (0.12, 0.25)}
H_ERR_M = {"A": (80.0, 300.0), "B": (300.0, 800.0), "C": (800.0, 2500.0)}
PROB = {"A": (0.75, 0.98), "B": (0.55, 0.85), "C": (0.35, 0.7)}
PICK_SIGMA_S = {"P": 0.03, "S": 0.06}
MATCHED_EVENT_INDICES = (0, 2, 5, 7, 9)  # 5 of 6 public rows match; the sixth is unmatched
UNFILLED_MATCH_INDEX = 5  # this event's catalogMatch is left null in events.parquet
NO_PICK_STATION_EVENT_INDEX = 1  # a Tier B event with an extra predicted-only station
BLANK_PICK_ID_EVENT_INDEX = 4  # one S arrival with a null pickId although the event has the pick
GAP_STATION_ID = "XT.R03"  # returns two short pieces: dropped as a gap
EMPTY_STATION_ID = "XT.R05"  # returns no data at all: dropped
NOISE_AMP, P_AMP, S_AMP = 0.08, 0.6, 1.0
P_HZ, S_HZ = 8.0, 5.0
COUNTS_SCALE = 1000.0


def rnd(x: float, decimals: int) -> float:
    return round(float(x), decimals) + 0.0


def ricker(t: np.ndarray, freq_hz: float) -> np.ndarray:
    a = (math.pi * freq_hz * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


# --- fixtures: config, run context, synthetic run tables --------------------------------------


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """configs/showcase with ``outputDir`` pointed into tmp_path."""
    target = tmp_path / "config"
    target.mkdir()
    for path in sorted(SHOWCASE_DIR.glob("*.yaml")):
        shutil.copy(path, target / path.name)
    export = target / "export.yaml"
    text = export.read_text()
    assert "outputDir: apps/web/public/data" in text
    export.write_text(
        text.replace("outputDir: apps/web/public/data", f"outputDir: {tmp_path / 'bundles'}")
    )
    return target


@pytest.fixture
def ctx(config_dir: Path, tmp_path: Path) -> runs.RunContext:
    return runs.create_run(config_dir, tmp_path / "data", now=NOW)


@dataclass
class SyntheticRun:
    """The tables written into the run directory, kept for assertions."""

    section: RunSection
    stations: list[m.Station]
    events: list[m.SeismicEvent]
    catalog: list[m.CatalogEvent]
    picks: list[m.Pick]
    arrivals: pd.DataFrame
    matches: pd.DataFrame

    @property
    def used(self) -> list[m.Station]:
        return [s for s in self.stations if s.usedInRun]


def make_station(
    section: RunSection, sid: str, e: float, n: float, surface_m: float, depth_m: float, used: bool
) -> m.Station:
    sensor_elev = surface_m - depth_m
    lat, lon, _ = from_enu(e, n, sensor_elev - section.origin.elevM, section.origin)
    borehole = depth_m > 0.0
    net, sta = sid.split(".")
    return m.Station(
        id=sid,
        network=net,
        station=sta,
        latitude=float(lat),
        longitude=float(lon),
        surfaceElevM=surface_m,
        sensorDepthM=depth_m,
        sensorElevM=sensor_elev,
        kind="borehole" if borehole else "surface",
        channels=["DPZ", "DP1", "DP2"] if borehole else ["HHZ", "HHN", "HHE"],
        sampleRateHz=BOREHOLE_RATE_HZ if borehole else SURFACE_RATE_HZ,
        enu=m.Enu(e=e, n=n, u=sensor_elev - section.origin.elevM),
        preprocessProfile="borehole" if borehole else "surface",
        usedInRun=used,
        staticsS={"P": 0.01} if used and borehole else {},
    )


def make_stations(section: RunSection) -> list[m.Station]:
    out: list[m.Station] = []
    for i in range(N_RING):
        angle = 2.0 * math.pi * i / N_RING
        e, n = RING_RADIUS_M * math.sin(angle), RING_RADIUS_M * math.cos(angle)
        surface = section.refSurfaceElevM + 40.0 * i
        out.append(make_station(section, f"XT.R{i + 1:02d}", e, n, surface, 0.0, True))
    be, bn = BOREHOLE_EN_M
    out.append(
        make_station(
            section, "XT.B01", be, bn, section.refSurfaceElevM + 12.0, BOREHOLE_DEPTH_M, True
        )
    )
    out.append(
        make_station(section, "XT.U01", UNUSED_RADIUS_M, 0.0, section.refSurfaceElevM, 0.0, False)
    )
    return out


def hypo_dist_m(ev_e: float, ev_n: float, ev_elev: float, st: m.Station) -> float:
    return math.sqrt(
        (st.enu.e - ev_e) ** 2 + (st.enu.n - ev_n) ** 2 + (st.sensorElevM - ev_elev) ** 2
    )


def build_run(ctx: runs.RunContext, seed: int = SEED) -> SyntheticRun:
    """A dozen events across tiers, stations incl. a borehole, picks, arrivals, matches, catalog."""
    section: RunSection = ctx.config.run
    rng = np.random.default_rng(seed)
    stations = make_stations(section)
    used = [s for s in stations if s.usedInRun]
    run_id = ctx.run_id
    events: list[m.SeismicEvent] = []
    picks: list[m.Pick] = []
    arrival_rows: list[dict[str, object]] = []
    n_events = len(TIER_PLAN)
    for i, tier in enumerate(TIER_PLAN):
        t = section.window_start_s + 3600.0 * (1.0 + 1.7 * i) + float(rng.uniform(0.0, 600.0))
        t = rnd(t, 3)
        e, n = float(rng.uniform(-3000.0, 3000.0)), float(rng.uniform(-3000.0, 3000.0))
        elev = section.refSurfaceElevM - float(rng.uniform(1000.0, 4000.0))
        n_sta = N_STATIONS[tier]
        nearest = sorted(used, key=lambda s: (hypo_dist_m(e, n, elev, s), s.id))
        chosen, rest = nearest[:n_sta], nearest[n_sta:]
        n_s = n_sta - 1
        event_id = f"hq-{run_id}-{i + 1:06d}"
        pick_ids: list[str] = []
        probs: list[float] = []
        for j, st in enumerate(chosen):
            r_km = hypo_dist_m(e, n, elev, st) / 1000.0
            for phase, v in (("P", VP_KM_S), ("S", VS_KM_S)):
                if phase == "S" and j >= n_s:
                    continue
                t_pred = rnd(t + r_km / v, 4)
                t_pick = rnd(t_pred + rng.normal(0.0, PICK_SIGMA_S[phase]), 3)
                pick = m.Pick(
                    id=f"{PICKER}:{st.id}:{phase}:{t_pick:.3f}",
                    stationId=st.id,
                    phase=phase,
                    t=t_pick,
                    prob=rnd(rng.uniform(*PROB[tier]), 3),
                    picker=PICKER,
                )
                picks.append(pick)
                pick_ids.append(pick.id)
                probs.append(pick.prob)
                arrival_rows.append(
                    {
                        "eventId": event_id,
                        "stationId": st.id,
                        "phase": phase,
                        "tPred": t_pred,
                        "tObs": t_pick,
                        "residualS": rnd(t_pick - t_pred, 4),
                        "pickId": pick.id,
                        "usedInLocation": True,
                    }
                )
        if i == BLANK_PICK_ID_EVENT_INDEX:  # the locator kept the pick out of this arrival row
            row = next(
                r
                for r in reversed(arrival_rows)
                if r["eventId"] == event_id
                and r["phase"] == "S"
                and r["stationId"] not in (GAP_STATION_ID, EMPTY_STATION_ID)
            )
            row["pickId"] = None
            row["usedInLocation"] = False
        if i == NO_PICK_STATION_EVENT_INDEX:  # predicted-only rows for one more station
            st = next(s for s in rest if s.id not in (GAP_STATION_ID, EMPTY_STATION_ID))
            r_km = hypo_dist_m(e, n, elev, st) / 1000.0
            for phase, v in (("P", VP_KM_S), ("S", VS_KM_S)):
                arrival_rows.append(
                    {
                        "eventId": event_id,
                        "stationId": st.id,
                        "phase": phase,
                        "tPred": rnd(t + r_km / v, 4),
                        "tObs": None,
                        "residualS": None,
                        "pickId": None,
                        "usedInLocation": False,
                    }
                )
        azimuths = np.sort(
            np.mod([math.degrees(math.atan2(s.enu.e - e, s.enu.n - n)) for s in chosen], 360.0)
        )
        gap = float(np.max(np.diff(np.concatenate([azimuths, [azimuths[0] + 360.0]]))))
        rms_lo, rms_hi = RMS_S[tier]
        rms = rnd(rng.uniform(rms_lo, rms_hi), 3)
        if tier == "A" and i == 0:
            rms = rms_lo  # the hero: every Tier A event has 7 stations, so the tie breaks on rmsS
        lat, lon, _ = from_enu(e, n, elev - section.origin.elevM, section.origin)
        events.append(
            m.SeismicEvent(
                id=event_id,
                runId=run_id,
                t=t,
                latitude=float(lat),
                longitude=float(lon),
                elevM=elev,
                depthKm=(section.refSurfaceElevM - elev) / 1000.0,
                enu=m.Enu(e=e, n=n, u=elev - section.origin.elevM),
                quality=m.LocationQuality(
                    method="grid1d",
                    statics=tier != "C",
                    nStations=n_sta,
                    nP=n_sta,
                    nS=n_s,
                    rmsS=rms,
                    gapDeg=rnd(gap, 1),
                    minEpiDistM=rnd(min(math.hypot(s.enu.e - e, s.enu.n - n) for s in chosen), 1),
                    hErrM=rnd(rng.uniform(*H_ERR_M[tier]), 1),
                    vErrM=None if tier == "C" else rnd(rng.uniform(*H_ERR_M[tier]) * 1.5, 1),
                    depthOnEdge=False,
                ),
                tier=tier,
                tierReasons=[f"synthetic tier {tier} (test)"],
                meanPickProb=rnd(float(np.mean(probs)), 3),
                magnitude=m.Magnitude(value=rnd(rng.uniform(-0.5, 2.0), 2), type="ML_cal"),
                catalogMatch=None,
                revealOrder=-1,
                pickIds=pick_ids,
            )
        )
    assert len(events) == n_events

    catalog: list[m.CatalogEvent] = []
    match_rows: list[dict[str, object]] = []
    for k, idx in enumerate(MATCHED_EVENT_INDICES):
        ev = events[idx]
        cat_id = f"testpub{k + 1:03d}"
        dt_s = rnd(rng.uniform(-0.3, 0.3), 3)
        de, dn = float(rng.uniform(-200.0, 200.0)), float(rng.uniform(-200.0, 200.0))
        depth_km = rnd(-ev.elevM / 1000.0 + rng.normal(0.0, 0.5), 2)  # published below sea level
        elev = -depth_km * 1000.0
        lat, lon, _ = from_enu(
            ev.enu.e + de, ev.enu.n + dn, elev - section.origin.elevM, section.origin
        )
        catalog.append(
            m.CatalogEvent(
                id=cat_id,
                source="synthetic public catalog (test)",
                t=rnd(ev.t + dt_s, 3),
                latitude=float(lat),
                longitude=float(lon),
                depthKm=depth_km,
                depthDatum="sea level (test)",
                elevM=elev,
                mag=rnd(rng.uniform(0.5, 2.0), 2),
                magType="ML",
                enu=m.Enu(e=ev.enu.e + de, n=ev.enu.n + dn, u=elev - section.origin.elevM),
            )
        )
        dist_m = rnd(math.hypot(de, dn), 1)
        match_rows.append(
            {"catalogId": cat_id, "eventId": ev.id, "dtS": dt_s, "distM": dist_m, "reason": None}
        )
        if idx != UNFILLED_MATCH_INDEX:  # H2's tier stage sets this; one row left null on purpose
            events[idx] = ev.model_copy(
                update={"catalogMatch": m.CatalogMatch(catalogId=cat_id, dtS=dt_s, distM=dist_m)}
            )
    unmatched_id = f"testpub{len(MATCHED_EVENT_INDICES) + 1:03d}"
    lat, lon, _ = from_enu(8000.0, 8000.0, -2000.0 - section.origin.elevM, section.origin)
    catalog.append(
        m.CatalogEvent(
            id=unmatched_id,
            source="synthetic public catalog (test)",
            t=rnd(section.window_start_s + 20.0 * 3600.0, 3),
            latitude=float(lat),
            longitude=float(lon),
            depthKm=2.0,
            depthDatum="sea level (test)",
            elevM=-2000.0,
            mag=None,
            magType=None,
            enu=m.Enu(e=8000.0, n=8000.0, u=-2000.0 - section.origin.elevM),
        )
    )
    match_rows.append(
        {
            "catalogId": unmatched_id,
            "eventId": None,
            "dtS": None,
            "distM": None,
            "reason": "no candidate within tolerance",
        }
    )
    catalog.sort(key=lambda c: (c.t, c.id))

    arrivals = pd.DataFrame(arrival_rows)
    matches = pd.DataFrame(match_rows)
    write_models(stations, ctx.path("stations.parquet"))
    write_models(catalog, ctx.path("catalog.parquet"))
    write_models(events, ctx.path("events.parquet"))
    write_models(picks, ctx.path("picks.parquet"))
    write_table(arrivals, ctx.path("arrivals.parquet"), "Arrival")
    write_table(matches, ctx.path("matches.parquet"), "Match")
    ctx.update_run(
        stationIds=[s.id for s in used], pickerModel="seisbench.PhaseNet", pickerWeights="test"
    )
    ctx.record("locate", runtime_s=1.0, counts={"events": n_events}, params={"method": "grid1d"})
    return SyntheticRun(section, stations, events, catalog, picks, arrivals, matches)


@pytest.fixture
def synthetic_run(ctx: runs.RunContext) -> SyntheticRun:
    return build_run(ctx)


# --- the synthetic waveform source ------------------------------------------------------------


class SyntheticSource:
    """Streams built inside the test: smoothed noise plus a Ricker wavelet at every predicted P
    and S of the station (from the arrivals table), as raw integer counts. One station returns
    two short pieces around a gap and one returns nothing, to exercise the drop paths."""

    def __init__(
        self,
        stations: list[m.Station],
        arrivals: pd.DataFrame,
        *,
        seed: int = SEED,
        gap_station: str | None = GAP_STATION_ID,
        empty_station: str | None = EMPTY_STATION_ID,
        fail_after: int | None = None,
    ) -> None:
        self.stations = {s.id: s for s in stations}
        self.arrivals = arrivals
        self.seed = seed
        self.gap_station = gap_station
        self.empty_station = empty_station
        self.fail_after = fail_after
        self.reads: list[tuple[str, float, float]] = []

    def _data(self, station_id: str, t0: float, t1: float, rate: float) -> np.ndarray:
        n = round((t1 - t0) * rate) + 1
        times = t0 + np.arange(n) / rate
        rng = np.random.default_rng(self.seed + zlib.crc32(station_id.encode()) + int(t0))
        data = NOISE_AMP * np.convolve(rng.standard_normal(n), np.ones(5) / 5.0, mode="same")
        rows = self.arrivals[self.arrivals["stationId"] == station_id]
        for _, row in rows.iterrows():
            amp, freq = (P_AMP, P_HZ) if row["phase"] == "P" else (S_AMP, S_HZ)
            data += amp * ricker(times - float(row["tPred"]), freq)
        return np.round(data * COUNTS_SCALE).astype(np.int32)

    def read_window(self, station_id: str, t0: float, t1: float, *, cache_dir: Path) -> Stream:
        assert cache_dir.is_dir(), "the stage passes ctx.cache_dir"
        self.reads.append((station_id, t0, t1))
        if self.fail_after is not None and len(self.reads) > self.fail_after:
            raise OSError(f"synthetic cache failure after {self.fail_after} reads")
        if station_id == self.empty_station:
            return Stream()
        station = self.stations[station_id]
        rate = station.sampleRateHz
        data = self._data(station_id, t0, t1, rate)
        net, sta = station_id.split(".")
        traces: list[Trace] = []
        for channel in station.channels:
            header = {
                "network": net,
                "station": sta,
                "channel": channel,
                "sampling_rate": rate,
                "starttime": UTCDateTime(t0),
            }
            if station_id == self.gap_station:  # two 1.5 s pieces, a hole in between
                piece = int(1.5 * rate)
                traces.append(Trace(data=data[:piece].copy(), header=dict(header)))
                tail = dict(header, starttime=UTCDateTime(t0 + (len(data) - piece) / rate))
                traces.append(Trace(data=data[-piece:].copy(), header=tail))
            else:
                traces.append(Trace(data=data.copy(), header=header))
        return Stream(traces)

    def display_copy(self, st: Stream, band_hz: tuple[float, float]) -> Stream:
        out = st.copy()
        out.detrend("linear")
        out.taper(max_percentage=0.05, type="cosine")
        out.filter("bandpass", freqmin=band_hz[0], freqmax=band_hz[1], corners=4, zerophase=True)
        return out


def fake_features(cfg: ExportConfig, section: RunSection) -> list[m.GeoFeature]:
    assert isinstance(cfg, ExportConfig) and isinstance(section, RunSection)
    return [
        m.GeoFeature(
            id="test-outline",
            kind="boundary",
            name="Synthetic outline (test)",
            path=[m.Enu(e=0.0, n=0.0, u=0.0), m.Enu(e=100.0, n=0.0, u=0.0)],
            source=m.SourceRef(
                citation="synthetic (test)", url="https://example.invalid", verified=False
            ),
        )
    ]


def do_export(
    ctx: runs.RunContext,
    out_dir: Path,
    source: SyntheticSource | None = None,
    *,
    run: SyntheticRun | None = None,
    cfg: ExportConfig | None = None,
) -> tuple[Path, object]:
    tables = load_run_tables(ctx.run_dir)
    if source is None:
        assert run is not None
        source = SyntheticSource(run.stations, run.arrivals)
    cfg = cfg or ctx.config.export
    result = export_bundle(
        tables,
        cfg,
        ctx.config.run,
        "showcase",
        source,
        cache_dir=ctx.cache_dir,
        out_dir=out_dir,
        baseline_cfg=ctx.config.validate.baseline,
        features_loader=fake_features,
    )
    return result.out_dir, result


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


class Bundle:
    def __init__(self, out_dir: Path) -> None:
        self.dir = out_dir
        self.meta = m.BundleMeta.model_validate(read_json(out_dir / "meta.json"))
        self.stations = [m.Station.model_validate(x) for x in read_json(out_dir / "stations.json")]
        self.catalog = [
            m.CatalogEvent.model_validate(x) for x in read_json(out_dir / "catalog.json")
        ]
        self.events = [m.SeismicEvent.model_validate(x) for x in read_json(out_dir / "events.json")]
        self.features = [
            m.GeoFeature.model_validate(x) for x in read_json(out_dir / "features.json")
        ]
        path = out_dir / "validation.json"
        self.validation = m.Validation.model_validate(read_json(path)) if path.is_file() else None
        self.evidence = {
            p.stem: m.EventEvidence.model_validate(read_json(p))
            for p in sorted((out_dir / "evidence").glob("*.json"))
        }
        self.events_by_id = {e.id: e for e in self.events}


@pytest.fixture
def exported(ctx: runs.RunContext, synthetic_run: SyntheticRun, tmp_path: Path) -> Bundle:
    out_dir, _ = do_export(ctx, tmp_path / "bundles" / "showcase", run=synthetic_run)
    return Bundle(out_dir)


# --- acceptance: every file validates, counts recompute, evidence under the cap ---------------


def test_bundle_files_validate_and_check_passes(exported: Bundle, ctx: runs.RunContext) -> None:
    top = sorted(p.name for p in exported.dir.iterdir())
    assert top == sorted([*BUNDLE_FILES, "validation.json", "evidence"]) or top == sorted(
        [*BUNDLE_FILES, "evidence"]
    )
    counts = check_bundle(exported.dir)
    assert counts["events"] == len(TIER_PLAN) and counts["evidenceFiles"] == len(TIER_PLAN)
    meta = exported.meta
    assert meta.schemaVersion == m.SCHEMA_VERSION and meta.mode == "showcase"
    assert meta.run.id == ctx.run_id == meta.scene.runId == meta.summary.runId
    assert meta.run.isSynthetic is False and meta.scene.isSynthetic is False
    assert meta.run.stationIds == [s.id for s in exported.stations if s.usedInRun]
    assert meta.run.locator == {"method": "grid1d"}  # ProcessingRun copied verbatim
    assert all(e.runId == ctx.run_id for e in exported.events)
    assert [f.id for f in exported.features] == ["test-outline"]


def test_summary_recomputes_from_events_and_catalog(exported: Bundle) -> None:
    s = exported.meta.summary
    events, catalog = exported.events, exported.catalog
    recovered = [c for c in catalog if c.matchedEventId is not None]
    additional = [e for e in events if e.catalogMatch is None]
    assert s.publicCatalogCount == len(catalog) == len(MATCHED_EVENT_INDICES) + 1
    assert s.recoveredCatalogCount == len(recovered) == len(MATCHED_EVENT_INDICES)
    assert s.recall == round(len(recovered) / len(catalog), 4)
    assert s.unmatchedPublicIds == [c.id for c in catalog if c.matchedEventId is None]
    assert len(s.unmatchedPublicIds) == 1
    assert s.candidateCount == len(events) == len(TIER_PLAN)
    assert s.additionalCount == len(additional) == len(TIER_PLAN) - len(MATCHED_EVENT_INDICES)
    for tier in "ABC":
        assert getattr(s.additional, tier) == sum(1 for e in additional if e.tier == tier)
    assert s.strictQualityCount == sum(1 for e in events if e.tier == "A") == TIER_PLAN.count("A")
    assert s.strictAdditionalCount == sum(1 for e in additional if e.tier == "A")
    assert s.medianStations == float(np.median([e.quality.nStations for e in events]))
    assert s.medianRmsS == round(float(np.median([e.quality.rmsS for e in events])), 3)
    assert s.baseline is None  # no baseline table in this run


def test_scene_meta_comes_from_run_yaml_and_export_yaml(
    exported: Bundle, ctx: runs.RunContext
) -> None:
    scene, section, cfg = exported.meta.scene, ctx.config.run, ctx.config.export
    assert (scene.originLat, scene.originLon, scene.originElevM) == (
        section.origin.lat,
        section.origin.lon,
        section.origin.elevM,
    )
    assert scene.refSurfaceElevM == section.refSurfaceElevM
    assert scene.projection == "EPSG:32612 minus origin"
    assert scene.verticalExaggeration == cfg.scene.verticalExaggeration
    assert scene.depthLabel == cfg.scene.depthLabel.format(refSurfaceElevM=section.refSurfaceElevM)
    assert f"{section.refSurfaceElevM:g}" in scene.depthLabel


def test_reveal_order_is_tier_then_time(exported: Bundle) -> None:
    orders = sorted(e.revealOrder for e in exported.events)
    assert orders == list(range(len(exported.events)))
    by_order = sorted(exported.events, key=lambda e: e.revealOrder)
    keys = [reveal_key(e) for e in by_order]
    assert keys == sorted(keys)
    assert [e.tier for e in by_order] == sorted(TIER_PLAN)
    times = [e.t for e in exported.events]  # events.json itself is time-ordered
    assert times == sorted(times)
    assert [e.tier for e in exported.events] == list(TIER_PLAN)


def test_hero_is_tier_a_with_most_stations_and_has_full_evidence(exported: Bundle) -> None:
    hero_id = exported.meta.scene.heroEventId
    assert hero_id is not None
    hero = exported.events_by_id[hero_id]
    assert hero.tier == "A"
    tier_a = [e for e in exported.events if e.tier == "A"]
    assert hero.quality.nStations == max(e.quality.nStations for e in tier_a)
    assert hero.quality.rmsS == min(e.quality.rmsS for e in tier_a)  # the tie-break
    assert hero_id == exported.events[0].id  # by construction the first event
    evidence = exported.evidence[hero_id]
    used = {s.id for s in exported.stations if s.usedInRun}
    # every used station except the gap and the empty one contributes a trace
    assert {t.stationId for t in evidence.traces} == used - {GAP_STATION_ID, EMPTY_STATION_ID}


def test_catalog_matches_filled_from_matches_table(
    exported: Bundle, synthetic_run: SyntheticRun, caplog: pytest.LogCaptureFixture
) -> None:
    matched = {c.id: c.matchedEventId for c in exported.catalog if c.matchedEventId is not None}
    want = {
        str(row["catalogId"]): str(row["eventId"])
        for row in synthetic_run.matches.to_dict("records")
        if not pd.isna(row["eventId"])
    }
    assert matched == want
    for cat_id, event_id in matched.items():
        ev = exported.events_by_id[event_id]
        assert ev.catalogMatch is not None and ev.catalogMatch.catalogId == cat_id
    # the event whose catalogMatch was null in events.parquet got it from matches.parquet
    filled = synthetic_run.events[UNFILLED_MATCH_INDEX]
    assert filled.catalogMatch is None
    got = exported.events_by_id[filled.id].catalogMatch
    assert got is not None and got.catalogId in matched
    assert [c.id for c in exported.catalog] == [c.id for c in synthetic_run.catalog]


def test_evidence_files_are_small_sorted_and_filled(
    exported: Bundle, synthetic_run: SyntheticRun, ctx: runs.RunContext
) -> None:
    cfg: EvidenceConfig = ctx.config.export.evidence
    picks = {p.id: p for p in synthetic_run.picks}
    assert set(exported.evidence) == set(exported.events_by_id)
    for event_id, ev in exported.evidence.items():
        assert ev.eventId == event_id
        size = (exported.dir / "evidence" / f"{event_id}.json").stat().st_size
        assert size < MAX_EVIDENCE_BYTES
        assert size <= cfg.maxFileBytes
        assert ev.filterHz == cfg.bandHz
        assert 1 <= len(ev.traces) <= cfg.maxTraces
        dists = [t.epiDistM for t in ev.traces]
        assert dists == sorted(dists)
        event = exported.events_by_id[event_id]
        stations = {s.id: s for s in exported.stations}
        arrivals = synthetic_run.arrivals[synthetic_run.arrivals["eventId"] == event_id]
        for t in ev.traces:
            st = stations[t.stationId]
            assert st.usedInRun and t.stationId not in (GAP_STATION_ID, EMPTY_STATION_ID)
            assert t.channel == ("DPZ" if st.kind == "borehole" else "HHZ")
            assert t.epiDistM == pytest.approx(
                math.hypot(st.enu.e - event.enu.e, st.enu.n - event.enu.n), abs=0.1
            )
            assert t.dt == 1.0 / cfg.displayRateHz
            seconds = len(t.samples) * t.dt
            assert cfg.minLengthS <= seconds <= cfg.maxLengthS
            assert abs(seconds - (cfg.beforeS + cfg.afterS)) <= 2 * t.dt
            assert min(t.samples) >= -1.0 and max(t.samples) <= 1.0
            assert max(abs(x) for x in t.samples) == 1.0
            assert all(x == round(x, 3) for x in t.samples)
            rows = arrivals[arrivals["stationId"] == t.stationId].set_index("phase")
            assert t.predP == float(rows.loc["P", "tPred"])
            assert t.predS == (float(rows.loc["S", "tPred"]) if "S" in rows.index else None)
            assert t.t0 == pytest.approx(t.predP - cfg.beforeS, abs=t.dt)
            pick_id = rows.loc["P", "pickId"]
            if pick_id is None or (isinstance(pick_id, float) and np.isnan(pick_id)):
                assert t.pickP is None and t.probP is None
            else:
                assert (t.pickP, t.probP) == (picks[pick_id].t, picks[pick_id].prob)
            if "S" in rows.index and rows.loc["S", "pickId"] is not None:
                s_id = rows.loc["S", "pickId"]
                if not (isinstance(s_id, float) and np.isnan(s_id)):
                    assert (t.pickS, t.probS) == (picks[s_id].t, picks[s_id].prob)
            # the largest swing sits on the synthetic S wavelet (on P where the station has no
            # S prediction): the trace is placed by its own t0
            samples = np.asarray(t.samples)
            t_peak = t.t0 + int(np.argmax(np.abs(samples))) * t.dt
            assert abs(t_peak - (t.predS if t.predS is not None else t.predP)) < 0.3
    # a null arrivals.pickId shows no pick even though the event's pickIds hold one (no substitution)
    blank_event = synthetic_run.events[BLANK_PICK_ID_EVENT_INDEX]
    blank_rows = synthetic_run.arrivals[
        (synthetic_run.arrivals["eventId"] == blank_event.id)
        & (synthetic_run.arrivals["phase"] == "S")
        & synthetic_run.arrivals["pickId"].isna()
    ]
    assert len(blank_rows) == 1
    blank_station = str(blank_rows.iloc[0]["stationId"])
    assert any(p.stationId == blank_station and p.phase == "S" for p in synthetic_run.picks)
    blank_trace = next(
        t for t in exported.evidence[blank_event.id].traces if t.stationId == blank_station
    )
    assert blank_trace.pickS is None and blank_trace.probS is None
    assert blank_trace.predS == float(blank_rows.iloc[0]["tPred"])
    assert blank_trace.pickP is not None
    # the predicted-only station shows predictions and no picks
    ev = exported.evidence[synthetic_run.events[NO_PICK_STATION_EVENT_INDEX].id]
    no_pick = [t for t in ev.traces if t.pickP is None]
    assert len(no_pick) == 1 and no_pick[0].predP is not None and no_pick[0].predS is not None
    assert no_pick[0].pickS is None and no_pick[0].probS is None
    arrival_stations = set(
        synthetic_run.arrivals.loc[synthetic_run.arrivals["eventId"] == ev.eventId, "stationId"]
    )
    assert len(arrival_stations) == N_STATIONS["B"] + 1
    assert {t.stationId for t in ev.traces} == arrival_stations - {GAP_STATION_ID, EMPTY_STATION_ID}


def test_dropped_traces_are_logged_never_filled(
    ctx: runs.RunContext,
    synthetic_run: SyntheticRun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="hq.export")
    out_dir, result = do_export(ctx, tmp_path / "b" / "showcase", run=synthetic_run)
    messages = [r.getMessage() for r in caplog.records]
    assert any(GAP_STATION_ID in msg and "minLengthS" in msg for msg in messages)
    assert any(EMPTY_STATION_ID in msg and "no data" in msg for msg in messages)
    assert result.counts["evidenceTracesDropped"] > 0
    assert result.counts["evidenceFiles"] == len(TIER_PLAN)
    assert result.counts["bytes"] == sum(p.stat().st_size for p in out_dir.rglob("*.json"))
    for ev in Bundle(out_dir).evidence.values():
        assert all(t.stationId not in (GAP_STATION_ID, EMPTY_STATION_ID) for t in ev.traces)


# --- validation fallbacks and the baseline gain -------------------------------------------------


def test_missing_validation_names_val01(
    ctx: runs.RunContext,
    synthetic_run: SyntheticRun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="hq.export")
    out_dir, _ = do_export(ctx, tmp_path / "b1" / "showcase", run=synthetic_run)
    assert not (out_dir / "validation.json").exists()
    assert any("VAL-01" in r.getMessage() for r in caplog.records)
    check_bundle(out_dir)

    synthetic = m.SyntheticTest(
        nEvents=10, pickSigmaS={"P": 0.03, "S": 0.06}, medianHErrM=100.0, medianVErrM=200.0,
        p90VErrM=400.0, medianDepthBiasM=-10.0,
    )  # fmt: skip
    ctx.path("synthetic.json").write_text(synthetic.model_dump_json())
    write_models([m.SweepPoint(params={"nPAndSMin": 6}, candidates=12, recoveredPublic=5, tierA=4)],
                 ctx.path("sweep.parquet"))  # fmt: skip
    caplog.clear()
    out_dir, _ = do_export(ctx, tmp_path / "b2" / "showcase", run=synthetic_run)
    bundle = Bundle(out_dir)
    assert bundle.validation is not None
    assert bundle.validation.synthetic == synthetic
    assert len(bundle.validation.sweep) == 1 and bundle.validation.baseline == []
    assert bundle.validation.nullTest is None and bundle.validation.gr is None
    assert any("VAL-01" in r.getMessage() for r in caplog.records)
    check_bundle(out_dir)


def full_validation(gain_p_only: bool) -> m.Validation:
    def row(method: str, profile: str, a: int) -> m.BaselineRow:
        return m.BaselineRow(
            method=method, associationProfile=profile, candidates=a + 6, recoveredPublic=5,
            tiers=m.TierCounts(A=a, B=4, C=2), medianRmsS=0.05, medianStations=6.0,
        )  # fmt: skip

    return m.Validation(
        baseline=[
            row("phasenet", "full", 8),
            row("stalta", "full", 4),
            row("phasenet", "p_only", 6),
            row("stalta", "p_only", 3 if gain_p_only else 6),
        ],
        sweep=[],
        synthetic=m.SyntheticTest(
            nEvents=10,
            pickSigmaS={"P": 0.03, "S": 0.06},
            medianHErrM=100.0,
            medianVErrM=200.0,
            p90VErrM=400.0,
            medianDepthBiasM=-10.0,
        ),
    )


def test_baseline_gain_only_when_it_holds_in_both_profiles(
    ctx: runs.RunContext, synthetic_run: SyntheticRun, tmp_path: Path
) -> None:
    rounding, baseline_cfg = ctx.config.export.rounding, ctx.config.validate.baseline
    assert baseline_gain(None, rounding, baseline_cfg) is None
    assert baseline_gain(full_validation(gain_p_only=False), rounding, baseline_cfg) is None
    assert baseline_gain(full_validation(gain_p_only=True), rounding, None) is None  # no config
    gain = baseline_gain(full_validation(gain_p_only=True), rounding, baseline_cfg)
    assert gain is not None and (gain.strictPhasenet, gain.strictStalta, gain.gain) == (8, 4, 2.0)
    assert gain.associationProfile == "full"

    ctx.path("validation.json").write_text(full_validation(gain_p_only=True).model_dump_json())
    out_dir, _ = do_export(ctx, tmp_path / "b" / "showcase", run=synthetic_run)
    bundle = Bundle(out_dir)
    assert bundle.meta.summary.baseline == gain
    assert bundle.validation is not None and len(bundle.validation.baseline) == 4
    check_bundle(out_dir)


def test_validation_assembled_from_sidecars_when_validation_json_is_absent(
    ctx: runs.RunContext,
    synthetic_run: SyntheticRun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The validate stage may have run before H2's synthetic.json existed: its sidecars still
    carry the results, and the exporter assembles the Validation from them."""
    caplog.set_level(logging.WARNING, logger="hq.export")
    full = full_validation(gain_p_only=True)
    ctx.path("synthetic.json").write_text(full.synthetic.model_dump_json())
    null_test = m.NullTest(
        nShuffles=4, shiftRangeS=30.0, meanChanceEvents=1.5, meanChanceStrict=0.25,
        stdChanceEvents=0.5,
    )  # fmt: skip
    gr = m.GRCurve(magBins=[0.5, 0.6, 0.7], publicCum=[3, 2, 1], recoveredCum=[9, 5, 2],
                   mcPublic=None, mcRecovered=0.7, bValue=1.05, bSigma=0.2)  # fmt: skip
    calibration = m.MagCalibration(n=6, looMae=0.3, coefficients={"a": 1.0, "b": -1.5})
    ctx.path("null_test.json").write_text(null_test.model_dump_json())
    ctx.path("baseline.json").write_text(
        json.dumps([row.model_dump(mode="json") for row in full.baseline])
    )
    ctx.path("gr.json").write_text(gr.model_dump_json())
    ctx.path("magnitude.json").write_text(calibration.model_dump_json())

    out_dir, _ = do_export(ctx, tmp_path / "b" / "showcase", run=synthetic_run)
    bundle = Bundle(out_dir)
    assert bundle.validation is not None
    assert bundle.validation.synthetic == full.synthetic
    assert bundle.validation.nullTest == null_test
    assert bundle.validation.baseline == full.baseline
    assert bundle.validation.gr == gr and bundle.validation.magnitude == calibration
    assert bundle.validation.sweep == []
    expected_gain = baseline_gain(full, ctx.config.export.rounding, ctx.config.validate.baseline)
    assert bundle.meta.summary.baseline == expected_gain and expected_gain is not None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assembled = [w for w in warnings if "validation.json not found" in w]
    assert len(assembled) == 1 and "VAL-01" in assembled[0]
    for name in ("null_test.json", "baseline.json", "gr.json", "magnitude.json"):
        assert name in assembled[0]
    assert "with sweep empty" in assembled[0]
    check_bundle(out_dir)

    # With the run's G-R config the magnitude kill switch is applied to the sidecar gr.json
    # again: a calibration that fails it drops the curve, as the validate stage would.
    gr_cfg = ctx.config.validate.gr
    failing = calibration.model_copy(update={"looMae": gr_cfg.maxLooMae + 0.1})
    ctx.path("magnitude.json").write_text(failing.model_dump_json())
    caplog.clear()
    tables = load_run_tables(ctx.run_dir, gr_cfg=gr_cfg)
    assert tables.validation is not None
    assert tables.validation.gr is None and tables.validation.magnitude == failing
    assert any("gr.json dropped" in r.getMessage() and "kill switch" in r.getMessage()
               for r in caplog.records)  # fmt: skip
    assert load_run_tables(ctx.run_dir).validation.gr == gr  # no config: sidecars as they are
    assert load_run_tables(ctx.run_dir, gr_cfg=gr_cfg).validation_source.count("gr.json") == 0
    ctx.path("magnitude.json").write_text(calibration.model_dump_json())
    assert load_run_tables(ctx.run_dir, gr_cfg=gr_cfg).validation.gr == gr  # passes: kept
    # A corrupt sidecar fails loudly, naming the model it should hold.
    ctx.path("baseline.json").write_text("[{}]")
    with pytest.raises(ExportError, match="not a valid list of BaselineRow"):
        load_run_tables(ctx.run_dir)


# --- determinism, atomicity, caps ----------------------------------------------------------------


def test_same_run_gives_identical_bytes(
    ctx: runs.RunContext, synthetic_run: SyntheticRun, tmp_path: Path
) -> None:
    first, _ = do_export(ctx, tmp_path / "one" / "showcase", run=synthetic_run)
    second, _ = do_export(ctx, tmp_path / "two" / "showcase", run=synthetic_run)
    a = sorted(p.relative_to(first) for p in first.rglob("*.json"))
    b = sorted(p.relative_to(second) for p in second.rglob("*.json"))
    assert a == b and len(a) == len(BUNDLE_FILES) + len(TIER_PLAN)
    for rel in a:
        assert (first / rel).read_bytes() == (second / rel).read_bytes(), rel


def test_failed_export_leaves_no_partial_bundle(
    ctx: runs.RunContext, synthetic_run: SyntheticRun, tmp_path: Path
) -> None:
    out_dir = tmp_path / "bundles" / "showcase"
    failing = SyntheticSource(synthetic_run.stations, synthetic_run.arrivals, fail_after=5)
    with pytest.raises(OSError, match="synthetic cache failure"):
        do_export(ctx, out_dir, failing)
    assert not out_dir.exists()
    assert list(out_dir.parent.iterdir()) == [], "no temp directory left behind"

    good, _ = do_export(ctx, out_dir, run=synthetic_run)
    before = {p.relative_to(good): p.read_bytes() for p in good.rglob("*.json")}
    failing = SyntheticSource(synthetic_run.stations, synthetic_run.arrivals, fail_after=5)
    with pytest.raises(OSError):
        do_export(ctx, out_dir, failing)
    after = {p.relative_to(good): p.read_bytes() for p in good.rglob("*.json")}
    assert after == before, "the previous bundle is untouched"
    assert [p.name for p in out_dir.parent.iterdir()] == ["showcase"]


def test_max_events_and_byte_budget(
    ctx: runs.RunContext, synthetic_run: SyntheticRun, tmp_path: Path
) -> None:
    base = ctx.config.export
    cfg = base.model_copy(update={"evidence": base.evidence.model_copy(update={"maxEvents": 3})})
    out_dir, _ = do_export(ctx, tmp_path / "capped" / "showcase", run=synthetic_run, cfg=cfg)
    bundle = Bundle(out_dir)
    by_order = sorted(bundle.events, key=lambda e: e.revealOrder)
    hero = bundle.meta.scene.heroEventId
    expected = [hero] + [e.id for e in by_order if e.id != hero][:2]
    assert sorted(bundle.evidence) == sorted(expected)
    check_bundle(out_dir)

    small = base.model_copy(
        update={"evidence": base.evidence.model_copy(update={"maxFileBytes": 9_000})}
    )
    out_dir, result = do_export(ctx, tmp_path / "small" / "showcase", run=synthetic_run, cfg=small)
    assert result.counts["evidenceTracesOverBudget"] > 0
    for path in (out_dir / "evidence").glob("*.json"):
        assert path.stat().st_size < 9_000
    check_bundle(out_dir, max_evidence_bytes=9_000)

    over = base.model_copy(update={"maxBundleBytes": 20_000})
    with pytest.raises(ExportError, match="maxBundleBytes"):
        do_export(ctx, tmp_path / "over" / "showcase", run=synthetic_run, cfg=over)
    assert not (tmp_path / "over" / "showcase").exists()
    assert list((tmp_path / "over").iterdir()) == [], "the failed build was cleaned up"
    with pytest.raises(BundleCheckError, match="maxBundleBytes"):
        check_bundle(out_dir, max_evidence_bytes=9_000, max_bundle_bytes=20_000)

    tiny = base.model_copy(
        update={"evidence": base.evidence.model_copy(update={"maxFileBytes": 500})}
    )
    with pytest.raises(ExportError, match="one trace alone"):
        do_export(ctx, tmp_path / "tiny" / "showcase", run=synthetic_run, cfg=tiny)


def test_bad_tables_fail_loudly(ctx: runs.RunContext, synthetic_run: SyntheticRun) -> None:
    events = synthetic_run.events
    broken = [*events[:-1], events[-1].model_copy(update={"pickIds": ["phasenet:test:nope"]})]
    write_models(broken, ctx.path("events.parquet"))
    with pytest.raises(ExportError, match="not in .*picks.parquet"):
        load_run_tables(ctx.run_dir)
    write_models(events, ctx.path("events.parquet"))

    matches = synthetic_run.matches.copy()
    matches.loc[matches["eventId"].notna(), "eventId"] = events[3].id  # many-to-one
    write_table(matches, ctx.path("matches.parquet"), "Match")
    with pytest.raises(ExportError, match="not one-to-one|carries catalogMatch"):
        do_export(ctx, ctx.run_dir / "out", run=synthetic_run)
    write_table(synthetic_run.matches, ctx.path("matches.parquet"), "Match")

    ctx.path("arrivals.parquet").unlink()
    with pytest.raises(ExportError, match="arrivals.parquet not found"):
        load_run_tables(ctx.run_dir)


# --- the stage through the runner, lazy lane imports ----------------------------------------------


def test_run_stage_export_end_to_end(
    ctx: runs.RunContext,
    synthetic_run: SyntheticRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SyntheticSource(synthetic_run.stations, synthetic_run.arrivals)
    install_module(monkeypatch, "hq.ingest.cache", read_window=source.read_window)
    install_module(monkeypatch, "hq.preprocess", display_copy=source.display_copy)
    install_module(monkeypatch, "hq.export.features", load_features=fake_features)

    runs.run_stage(ctx, "export")

    root = output_root(ctx.config.export)
    assert root == tmp_path / "bundles"
    out_dir = root / "showcase"
    counts = check_bundle(out_dir)
    assert counts["events"] == len(TIER_PLAN)
    run = ctx.read_run()
    assert run.runtimeS["export"] > 0.0
    stages = json.loads(ctx.path("stages.json").read_text())
    recorded = stages["export"]["counts"]
    assert recorded["events"] == len(TIER_PLAN)
    assert recorded["evidenceFiles"] == len(TIER_PLAN)
    assert recorded["modes"] == 1 and recorded["bytes"] > 0
    assert recorded["bytes"] == sum(p.stat().st_size for p in out_dir.rglob("*.json"))
    assert Bundle(out_dir).meta.run.id == ctx.run_id


def test_missing_lane_modules_name_their_owner(
    ctx: runs.RunContext, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "hq.ingest.cache", None)
    with pytest.raises(ExportError, match=r"hq\.ingest\.cache\.read_window.*H1 Signal"):
        real_waveform_source()
    install_module(monkeypatch, "hq.ingest.cache", read_window=lambda *a, **k: Stream())
    monkeypatch.setitem(sys.modules, "hq.preprocess", None)
    with pytest.raises(ExportError, match=r"hq\.preprocess\.display_copy.*H1 Signal"):
        real_waveform_source()
    install_module(monkeypatch, "hq.preprocess")  # exists but exposes nothing
    with pytest.raises(ExportError, match="does not expose.*H1 Signal"):
        real_waveform_source()
    install_module(monkeypatch, "hq.preprocess", display_copy=lambda st, band: st)
    source = real_waveform_source()
    assert isinstance(source, LaneWaveformSource)

    monkeypatch.setitem(sys.modules, "hq.export.features", None)
    caplog.set_level(logging.WARNING, logger="hq.export")
    # The showcase config lists FEAT-01's features; the fallback is for an empty list only.
    empty = ctx.config.export.model_copy(update={"features": []})
    assert load_features_lazily(empty, ctx.config.run) == []
    assert any("FEAT-01" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    listed = ctx.config.export.model_copy(update={"features": [{"id": "well-1"}]})
    with pytest.raises(ExportError, match="lists features.*FEAT-01"):
        load_features_lazily(listed, ctx.config.run)
    install_module(monkeypatch, "hq.export.features", load_features=fake_features)
    assert [f.id for f in load_features_lazily(ctx.config.export, ctx.config.run)] == [
        "test-outline"
    ]


def test_export_refuses_a_synthetic_run(
    ctx: runs.RunContext, synthetic_run: SyntheticRun, tmp_path: Path
) -> None:
    tables = load_run_tables(ctx.run_dir)
    fake = tables.run.model_copy(update={"isSynthetic": True})
    synthetic_tables = type(tables)(**{**tables.__dict__, "run": fake})
    with pytest.raises(ExportError, match="mock-fixture"):
        export_bundle(
            synthetic_tables,
            ctx.config.export,
            ctx.config.run,
            "showcase",
            SyntheticSource(synthetic_run.stations, synthetic_run.arrivals),
            cache_dir=ctx.cache_dir,
            out_dir=tmp_path / "x",
            baseline_cfg=ctx.config.validate.baseline,
            features_loader=fake_features,
        )
    assert not (tmp_path / "x").exists()


# --- the checker itself ----------------------------------------------------------------------------


def test_mock_bundle_passes_check_bundle() -> None:
    counts = check_bundle(MOCK_BUNDLE)
    assert counts["events"] > 0 and counts["evidenceFiles"] > 0 and counts["hasValidation"] == 1


def test_check_bundle_reports_every_problem(exported: Bundle, tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(exported.dir, broken)
    meta = read_json(broken / "meta.json")
    meta["summary"]["candidateCount"] += 1
    meta["scene"]["heroEventId"] = "hq-nope"
    meta["run"]["isSynthetic"] = True  # only a mock bundle may be synthetic
    (broken / "meta.json").write_text(json.dumps(meta))
    events = read_json(broken / "events.json")
    events[0]["revealOrder"] = events[1]["revealOrder"]
    (broken / "events.json").write_text(json.dumps(events))
    stray = broken / "evidence" / "hq-stray.json"
    stray.write_text((broken / "evidence" / f"{exported.events[0].id}.json").read_text())
    (broken / "notes.txt").write_text("x")
    with pytest.raises(BundleCheckError) as info:
        check_bundle(broken)
    problems = "\n".join(info.value.problems)
    for needle in (
        "candidateCount",
        "heroEventId",
        "permutation",
        "hq-stray",
        "unexpected entries",
        "isSynthetic",
        "'broken' (the bundle directory)",
    ):
        assert needle in problems, needle
    # the same bundle under its own mode name passes the mode check but nothing else
    with pytest.raises(BundleCheckError) as info:
        check_bundle(broken, mode="showcase")
    assert "bundle directory" not in "\n".join(info.value.problems)
    with pytest.raises(BundleCheckError, match="not a directory"):
        check_bundle(tmp_path / "nothing-here")
    (tmp_path / "empty").mkdir()
    with pytest.raises(BundleCheckError, match="missing files"):
        check_bundle(tmp_path / "empty")


def test_check_bundle_verifies_a_claimed_baseline_gain_against_its_rows(
    ctx: runs.RunContext, synthetic_run: SyntheticRun, tmp_path: Path
) -> None:
    """A claimed summary.baseline must be what the bundle's own validation.baseline rows give
    under the shared rule; every way the claim can drift from the rows is a problem."""
    ctx.path("validation.json").write_text(full_validation(gain_p_only=True).model_dump_json())
    out_dir, _ = do_export(ctx, tmp_path / "ok" / "showcase", run=synthetic_run)
    check_bundle(out_dir)
    claim = read_json(out_dir / "meta.json")["summary"]["baseline"]
    assert claim is not None

    def broken_with(
        name: str,
        *,
        rows: object = None,
        summary_baseline: object = None,
    ) -> list[str]:
        broken = tmp_path / name
        shutil.copytree(out_dir, broken)
        if rows is not None:
            validation = read_json(broken / "validation.json")
            validation["baseline"] = rows
            (broken / "validation.json").write_text(json.dumps(validation))
        if summary_baseline is not None:
            meta = read_json(broken / "meta.json")
            meta["summary"]["baseline"] = summary_baseline
            (broken / "meta.json").write_text(json.dumps(meta))
        with pytest.raises(BundleCheckError) as info:
            check_bundle(broken, mode="showcase")  # the copy's directory is not a mode name
        return info.value.problems

    good_rows = read_json(out_dir / "validation.json")["baseline"]

    def rows_with(method: str, profile: str, a: int) -> list[dict]:
        return [
            {**r, "tiers": {**r["tiers"], "A": a}}
            if (r["method"], r["associationProfile"]) == (method, profile)
            else r
            for r in good_rows
        ]

    partial = [r for r in good_rows if r["method"] == "phasenet"]
    cases: dict[str, tuple[dict[str, object], str]] = {
        "no-rows": ({"rows": []}, "no baseline rows"),
        "partial-rows": ({"rows": partial}, "no (stalta, full) row"),
        "duplicate-rows": ({"rows": good_rows + good_rows[:1]}, "several rows"),
        "count-mismatch": (
            {"summary_baseline": {**claim, "strictPhasenet": claim["strictPhasenet"] + 1}},
            "summary.baseline.strictPhasenet",
        ),
        "p-only-no-gain": ({"rows": rows_with("stalta", "p_only", 6)}, "does not support"),
        "stalta-no-strict": (
            {
                "rows": rows_with("stalta", "full", 0),
                "summary_baseline": {**claim, "strictStalta": 0},
            },
            "does not support",
        ),
        "ratio": (
            {"summary_baseline": {**claim, "gain": claim["gain"] + 1.0}},
            "!= strictPhasenet",
        ),
        "profile": (
            {"summary_baseline": {**claim, "associationProfile": "p_only"}},
            "associationProfile",
        ),
    }
    for name, (edits, needle) in cases.items():
        problems = broken_with(name, **edits)
        assert any(needle in p for p in problems), (name, problems)
    # A bundle that claims nothing is not checked against the rows at all.
    no_claim = tmp_path / "no-claim"
    shutil.copytree(out_dir, no_claim)
    meta = read_json(no_claim / "meta.json")
    meta["summary"]["baseline"] = None
    (no_claim / "meta.json").write_text(json.dumps(meta))
    validation = read_json(no_claim / "validation.json")
    validation["baseline"] = []
    (no_claim / "validation.json").write_text(json.dumps(validation))
    check_bundle(no_claim, mode="showcase")


def test_load_config_accepts_the_new_knobs(config_dir: Path) -> None:
    cfg = load_config(config_dir).export
    assert cfg.evidence.maxEvents is not None and cfg.evidence.channelPriority == ["Z"]
    assert cfg.evidence.maxEvents * cfg.evidence.maxFileBytes < cfg.maxBundleBytes
    assert cfg.evidence.maxFileBytes <= MAX_EVIDENCE_BYTES
    assert cfg.evidence.beforeS + cfg.evidence.afterS <= cfg.evidence.maxLengthS
    assert cfg.rounding.sample >= 1 and cfg.scene.verticalExaggeration == 1.0
    with pytest.raises(ValueError, match="channelPriority"):
        EvidenceConfig(channelPriority=["ZZ"])
    with pytest.raises(ValueError, match="channelPriority"):
        EvidenceConfig(channelPriority=["Z", "Z"])


# --- tables as stage locate writes them: `string` columns hold pd.NA (docs/02 §2) -----------------

MATCH_DTYPES = {
    "catalogId": "string",
    "eventId": "string",
    "dtS": "float64",
    "distM": "float64",
    "reason": "string",
}


def test_str_or_none_treats_every_table_null_as_none() -> None:
    """REQ-H2-10: a null in a ``string`` column reads back as ``pd.NA``; it must never become
    the id ``'<NA>'`` (nor ``'nan'`` / ``'None'``)."""
    for null in (None, float("nan"), np.nan, pd.NA, pd.NaT, np.float64("nan")):
        assert is_null(null)
        assert str_or_none(null) is None
    for value in ("phasenet:test:XT.R01:P:1.000", "", 0, 1.5, np.str_("x"), [], ["a"]):
        assert not is_null(value)
    assert str_or_none("XT.R01") == "XT.R01"
    assert str_or_none(np.str_("P")) == "P"
    assert str_or_none([]) == "[]"  # a list cell is a value, never null


def test_string_dtype_nulls_in_arrivals_matches_and_statics(
    ctx: runs.RunContext, synthetic_run: SyntheticRun, tmp_path: Path
) -> None:
    """REQ-H2-10 + FYI-H2-8: the run tables typed as stage locate / match write them (H2's
    ``ARRIVAL_DTYPES`` / ``STATIC_DTYPES``; ``matches`` with ``string`` ids): ``pickId`` is
    ``pd.NA`` on predicted-only arrival rows and ``eventId`` on the unmatched public event. The
    export succeeds, those traces carry no pick, the public event stays unmatched, and
    ``Station.staticsS`` comes from ``statics.parquet``."""
    arrivals = synthetic_run.arrivals[list(ARRIVAL_DTYPES)].astype(ARRIVAL_DTYPES)
    assert str(arrivals["pickId"].dtype) == "string"
    predicted_only = arrivals["pickId"].isna()
    assert predicted_only.sum() == 3  # two rows of the extra station, one blanked S row
    assert all(v is pd.NA for v in arrivals.loc[predicted_only, "pickId"])
    write_table(arrivals, ctx.path("arrivals.parquet"), ARRIVALS_MODEL)
    matches = synthetic_run.matches[list(MATCH_DTYPES)].astype(MATCH_DTYPES)
    assert matches["eventId"].isna().sum() == 1
    write_table(matches, ctx.path("matches.parquet"), "Match")
    statics = pd.DataFrame.from_records(
        [
            {"stationId": "XT.R01", "phase": "P", "staticS": -0.012, "nEvents": 4},
            {"stationId": "XT.R01", "phase": "S", "staticS": 0.03, "nEvents": 3},
            {"stationId": "XT.B01", "phase": "P", "staticS": 0.0, "nEvents": 0},
        ],
        columns=list(STATIC_DTYPES),
    ).astype(STATIC_DTYPES)
    write_table(statics, ctx.path("statics.parquet"), STATICS_MODEL)

    tables = load_run_tables(ctx.run_dir)
    assert set(tables.picks) == {p.id for p in synthetic_run.picks}
    assert not {"<NA>", "nan", "None"} & set(tables.picks)
    assert [s.id for s in tables.stations] == [s.id for s in synthetic_run.stations]

    out_dir, _ = do_export(ctx, tmp_path / "bundles" / "showcase", run=synthetic_run)
    bundle = Bundle(out_dir)
    ev = bundle.evidence[synthetic_run.events[NO_PICK_STATION_EVENT_INDEX].id]
    no_pick = [t for t in ev.traces if t.pickP is None]
    assert len(no_pick) == 1
    assert no_pick[0].predP is not None and no_pick[0].predS is not None
    assert no_pick[0].pickS is None and no_pick[0].probP is None and no_pick[0].probS is None
    blank_event = synthetic_run.events[BLANK_PICK_ID_EVENT_INDEX]
    blank_station = str(
        arrivals.loc[(arrivals["eventId"] == blank_event.id) & predicted_only, "stationId"].iloc[0]
    )
    blank = next(t for t in bundle.evidence[blank_event.id].traces if t.stationId == blank_station)
    assert blank.pickP is not None and blank.pickS is None and blank.predS is not None
    matched = {c.id: c.matchedEventId for c in bundle.catalog if c.matchedEventId is not None}
    assert len(matched) == len(MATCHED_EVENT_INDICES)
    unmatched_id = str(matches.loc[matches["eventId"].isna(), "catalogId"].iloc[0])
    assert unmatched_id not in matched
    assert next(c for c in bundle.catalog if c.id == unmatched_id).matchedEventId is None
    statics_by_station = {s.id: s.staticsS for s in bundle.stations}
    assert statics_by_station["XT.R01"] == {"P": -0.012, "S": 0.03}
    assert statics_by_station["XT.B01"] == {"P": 0.0}  # the table's term replaces H1's value
    assert statics_by_station["XT.R02"] == {}  # not in the table: stations.parquet's value


def test_statics_table_fills_station_statics(
    ctx: runs.RunContext, synthetic_run: SyntheticRun, caplog: pytest.LogCaptureFixture
) -> None:
    """FYI-H2-8: ``Station.staticsS`` is filled from ``statics.parquet`` when it exists (keys
    P/S), stations the table does not name keep H1's value, counts are logged, and a row naming
    an unknown station or phase, a duplicate station-phase or a non-finite term fails loudly."""
    h1 = {s.id: s.staticsS for s in synthetic_run.stations}
    assert h1["XT.B01"] == {"P": 0.01} and h1["XT.R01"] == {}
    with caplog.at_level(logging.INFO, logger="hq.export.tables"):
        assert {s.id: s.staticsS for s in load_run_tables(ctx.run_dir).stations} == h1
    assert "no statics.parquet" in caplog.text

    rows = [
        {"stationId": "XT.R01", "phase": "P", "staticS": -0.012, "nEvents": 4},
        {"stationId": "XT.R01", "phase": "S", "staticS": 0.03, "nEvents": 3},
        {"stationId": "XT.B01", "phase": "P", "staticS": 0.0, "nEvents": 0},
    ]
    path = ctx.path("statics.parquet")

    def write_statics(extra: dict[str, object] | None = None) -> None:
        frame = pd.DataFrame.from_records(
            [*rows, *([extra] if extra else [])], columns=list(STATIC_DTYPES)
        ).astype(STATIC_DTYPES)
        write_table(frame, path, STATICS_MODEL)

    write_statics()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="hq.export.tables"):
        stations = {s.id: s.staticsS for s in load_run_tables(ctx.run_dir).stations}
    assert stations == {**h1, "XT.R01": {"P": -0.012, "S": 0.03}, "XT.B01": {"P": 0.0}}
    assert "3 static term(s) for 2 station(s) (2 non-zero, 7 event use(s))" in caplog.text
    assert f"{len(h1) - 2} station(s) not in the table keep stations.parquet" in caplog.text

    for extra, match in (
        (
            {"stationId": "XT.NOPE", "phase": "P", "staticS": 0.0, "nEvents": 1},
            "missing from stations.parquet",
        ),
        ({"stationId": "XT.R02", "phase": "X", "staticS": 0.0, "nEvents": 1}, "phase must be"),
        ({"stationId": "XT.R01", "phase": "P", "staticS": 0.0, "nEvents": 1}, "several P rows"),
        (
            {"stationId": "XT.R02", "phase": "P", "staticS": float("nan"), "nEvents": 1},
            "no finite staticS",
        ),
    ):
        write_statics(extra)
        with pytest.raises(ExportError, match=match):
            load_run_tables(ctx.run_dir)
    write_table(pd.DataFrame({"stationId": ["XT.R01"]}), path, STATICS_MODEL)
    with pytest.raises(ExportError, match="lacks columns"):
        load_run_tables(ctx.run_dir)

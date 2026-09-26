"""Fixtures for the live worker tests: a fake ``PipelineRunner`` that creates a real ``live``
run through ``hq.runs`` (per-window config, ``create_run``) and writes minimal synthetic run
tables into it, a synthetic waveform source for the exporter, a controllable clock, and a live
config whose paths all point into ``tmp_path``. Offline; nothing touches the network."""

import math
import shutil
import socket
import threading
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hq_contracts import models as m
from hq_contracts.io import write_models, write_table
from obspy import Stream, Trace, UTCDateTime

from hq import runs
from hq.config.export import ExportConfig
from hq.config.run import RunSection
from hq.export import LaneWaveformSource
from hq.export.tables import ARRIVAL_COLUMNS, MATCH_COLUMNS
from hq.locate.coords import from_enu
from hq_api.config import DEFAULT_CONFIG_FILE, LiveConfig, PathsConfig, load_live_config
from hq_api.runner import LiveWindow, PipelineRun, write_window_config

API_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = API_DIR.parents[1]
SHOWCASE_DIR = REPO_ROOT / "services" / "seismic" / "configs" / "showcase"
CONFIG_YAML = API_DIR / DEFAULT_CONFIG_FILE

# Test-local synthetic knobs (CLAUDE.md rule 5 allows small synthetic data inside tests).
T_START = datetime(2026, 9, 26, 12, 0, tzinfo=UTC).timestamp()
SEED = 11
STATION_RADIUS_M = 5_000.0
N_STATIONS = 4
RATE_HZ = 100.0
VP_KM_S, VS_KM_S = 5.5, 3.2
PICKER = "phasenet:test"
NOISE_AMP, P_AMP, S_AMP = 0.05, 0.6, 1.0
P_HZ, S_HZ = 8.0, 5.0
COUNTS_SCALE = 1000.0
MISSING_STAGE_MODULE = "hq_api_test_missing_pick"
MISSING_STAGE_OWNER = "H1 Signal"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that opens a network connection, even indirectly."""

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network access in an offline test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


class FakeClock:
    """Wall clock the tests advance by hand, so run ids and window ends are distinct."""

    def __init__(self, t: float = T_START) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def showcase_config(tmp_path: Path) -> Path:
    """A copy of configs/showcase/*.yaml (what paths.configDir points at)."""
    target = tmp_path / "showcase"
    target.mkdir()
    for path in sorted(SHOWCASE_DIR.glob("*.yaml")):
        shutil.copy(path, target / path.name)
    return target


@pytest.fixture
def live_config(tmp_path: Path, showcase_config: Path) -> LiveConfig:
    """The real config.yaml with every path moved into tmp_path."""
    base = load_live_config(CONFIG_YAML)
    paths = PathsConfig(
        configDir=str(showcase_config),
        dataDir=str(tmp_path / "data"),
        windowConfigDir=str(tmp_path / "data" / "live" / "configs"),
        stateFile=str(tmp_path / "data" / "live" / "latest.json"),
        bundlesDir=str(tmp_path / "data" / "live" / "bundles"),
        snapshotDir=str(tmp_path / "web" / "snapshot"),
        keepBundles=base.paths.keepBundles,
    )
    return base.model_copy(update={"paths": paths})


def rnd(x: float, decimals: int) -> float:
    return round(float(x), decimals) + 0.0


def ricker(t: np.ndarray, freq_hz: float) -> np.ndarray:
    a = (math.pi * freq_hz * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def make_stations(section: RunSection) -> list[m.Station]:
    out: list[m.Station] = []
    for i in range(N_STATIONS):
        angle = 2.0 * math.pi * i / N_STATIONS
        e, n = STATION_RADIUS_M * math.sin(angle), STATION_RADIUS_M * math.cos(angle)
        surface = section.refSurfaceElevM + 20.0 * i
        lat, lon, _ = from_enu(e, n, surface - section.origin.elevM, section.origin)
        out.append(
            m.Station(
                id=f"XT.L{i + 1:02d}",
                network="XT",
                station=f"L{i + 1:02d}",
                latitude=float(lat),
                longitude=float(lon),
                surfaceElevM=surface,
                sensorDepthM=0.0,
                sensorElevM=surface,
                kind="surface",
                channels=["HHZ", "HHN", "HHE"],
                sampleRateHz=RATE_HZ,
                enu=m.Enu(e=e, n=n, u=surface - section.origin.elevM),
                preprocessProfile="surface",
                usedInRun=True,
            )
        )
    return out


@dataclass
class SyntheticTables:
    stations: list[m.Station]
    events: list[m.SeismicEvent]
    picks: list[m.Pick]
    arrivals: pd.DataFrame


def build_tables(ctx: runs.RunContext, window: LiveWindow, n_events: int) -> SyntheticTables:
    """``n_events`` Tier A events inside the window, picked at every station."""
    section: RunSection = ctx.config.run
    rng = np.random.default_rng(SEED + n_events)
    stations = make_stations(section)
    events: list[m.SeismicEvent] = []
    picks: list[m.Pick] = []
    rows: list[dict[str, object]] = []
    for i in range(n_events):
        t = rnd(window.start + (i + 1) * window.length_s / (n_events + 1), 3)
        e, n = float(rng.uniform(-1500.0, 1500.0)), float(rng.uniform(-1500.0, 1500.0))
        elev = section.refSurfaceElevM - float(rng.uniform(1500.0, 3500.0))
        event_id = f"hq-{ctx.run_id}-{i + 1:06d}"
        pick_ids: list[str] = []
        probs: list[float] = []
        for st in stations:
            r_km = math.dist((st.enu.e, st.enu.n, st.sensorElevM), (e, n, elev)) / 1000.0
            for phase, v in (("P", VP_KM_S), ("S", VS_KM_S)):
                t_pred = rnd(t + r_km / v, 4)
                t_pick = rnd(t_pred + rng.normal(0.0, 0.03), 3)
                pick = m.Pick(
                    id=f"{PICKER}:{st.id}:{phase}:{t_pick:.3f}",
                    stationId=st.id,
                    phase=phase,
                    t=t_pick,
                    prob=rnd(rng.uniform(0.7, 0.98), 3),
                    picker=PICKER,
                )
                picks.append(pick)
                pick_ids.append(pick.id)
                probs.append(pick.prob)
                rows.append(
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
        lat, lon, _ = from_enu(e, n, elev - section.origin.elevM, section.origin)
        events.append(
            m.SeismicEvent(
                id=event_id,
                runId=ctx.run_id,
                t=t,
                latitude=float(lat),
                longitude=float(lon),
                elevM=elev,
                depthKm=(section.refSurfaceElevM - elev) / 1000.0,
                enu=m.Enu(e=e, n=n, u=elev - section.origin.elevM),
                quality=m.LocationQuality(
                    method="grid1d",
                    statics=False,
                    nStations=N_STATIONS,
                    nP=N_STATIONS,
                    nS=N_STATIONS,
                    rmsS=rnd(rng.uniform(0.02, 0.05), 3),
                    gapDeg=90.0,
                    minEpiDistM=rnd(min(math.hypot(s.enu.e - e, s.enu.n - n) for s in stations), 1),
                    hErrM=rnd(rng.uniform(80.0, 300.0), 1),
                    vErrM=rnd(rng.uniform(100.0, 400.0), 1),
                    depthOnEdge=False,
                ),
                tier="A",
                tierReasons=["synthetic tier A (test)"],
                meanPickProb=rnd(float(np.mean(probs)), 3),
                revealOrder=-1,
                pickIds=pick_ids,
            )
        )
    arrivals = pd.DataFrame(rows, columns=list(ARRIVAL_COLUMNS))
    matches = pd.DataFrame(columns=list(MATCH_COLUMNS))
    write_models(stations, ctx.path("stations.parquet"))
    write_models([], ctx.path("catalog.parquet"), m.CatalogEvent)
    write_models(events, ctx.path("events.parquet"), m.SeismicEvent)
    write_models(picks, ctx.path("picks.parquet"), m.Pick)
    write_table(arrivals, ctx.path("arrivals.parquet"), "Arrival")
    write_table(matches, ctx.path("matches.parquet"), "Match")
    ctx.update_run(
        stationIds=[s.id for s in stations], pickerModel="seisbench.PhaseNet", pickerWeights="test"
    )
    ctx.record("tier", runtime_s=0.1, counts={"events": n_events}, params={"rule": "test"})
    return SyntheticTables(stations, events, picks, arrivals)


class SyntheticSource:
    """Smoothed noise plus a Ricker wavelet at every predicted arrival of the station."""

    def __init__(self, stations: list[m.Station], arrivals: pd.DataFrame) -> None:
        self.stations = {s.id: s for s in stations}
        self.arrivals = arrivals

    def read_window(self, station_id: str, t0: float, t1: float, *, cache_dir: Path) -> Stream:
        station = self.stations[station_id]
        n = round((t1 - t0) * RATE_HZ) + 1
        times = t0 + np.arange(n) / RATE_HZ
        rng = np.random.default_rng(SEED + zlib.crc32(station_id.encode()) + int(t0))
        data = NOISE_AMP * np.convolve(rng.standard_normal(n), np.ones(5) / 5.0, mode="same")
        for _, row in self.arrivals[self.arrivals["stationId"] == station_id].iterrows():
            amp, freq = (P_AMP, P_HZ) if row["phase"] == "P" else (S_AMP, S_HZ)
            data += amp * ricker(times - float(row["tPred"]), freq)
        counts = np.round(data * COUNTS_SCALE).astype(np.int32)
        net, sta = station_id.split(".")
        return Stream(
            [
                Trace(
                    data=counts.copy(),
                    header={
                        "network": net,
                        "station": sta,
                        "channel": channel,
                        "sampling_rate": RATE_HZ,
                        "starttime": UTCDateTime(t0),
                    },
                )
                for channel in station.channels
            ]
        )

    def display_copy(self, st: Stream, band_hz: tuple[float, float]) -> Stream:
        out = st.copy()
        out.detrend("linear")
        out.taper(max_percentage=0.05, type="cosine")
        out.filter("bandpass", freqmin=band_hz[0], freqmax=band_hz[1], corners=4, zerophase=True)
        return out


def no_features(cfg: ExportConfig, section: RunSection) -> list[m.GeoFeature]:
    return []


@dataclass
class FakeRunner:
    """A ``PipelineRunner`` that makes a real ``live`` run and fills it with synthetic tables.

    ``n_events`` sets how many candidate events the next window holds; ``fail`` makes the run
    fail the way a missing lane stage does (``hq.runs.StageMissingError`` naming the owner);
    ``block`` holds the run until the event is set (overlap tests).
    """

    config: LiveConfig
    n_events: int = 3
    fail: bool = False
    block: threading.Event | None = None
    started: threading.Event = field(default_factory=threading.Event)
    windows: list[LiveWindow] = field(default_factory=list)

    def run_window(self, window: LiveWindow) -> PipelineRun:
        self.windows.append(window)
        self.started.set()
        if self.block is not None:
            self.block.wait()
        paths = self.config.paths
        assert paths.dataDir is not None
        config_dir = write_window_config(
            Path(paths.configDir), Path(paths.windowConfigDir) / f"w{len(self.windows)}", window
        )
        ctx = runs.create_run(
            config_dir,
            Path(paths.dataDir),
            "live",
            now=datetime.fromtimestamp(window.end, tz=UTC),
        )
        if self.fail:
            registry = (runs.StageSpec("pick", MISSING_STAGE_MODULE, MISSING_STAGE_OWNER),)
            runs.run_stages(ctx, ["pick"], registry=registry)  # raises StageMissingError
        tables = build_tables(ctx, window, self.n_events)
        return PipelineRun(
            run_id=ctx.run_id,
            run_dir=ctx.run_dir,
            cache_dir=ctx.cache_dir,
            config=ctx.config,
            waveforms=LaneWaveformSource(
                read_window=SyntheticSource(tables.stations, tables.arrivals).read_window,
                display_copy=SyntheticSource(tables.stations, tables.arrivals).display_copy,
            ),
            stages_ran=("tier",),
        )

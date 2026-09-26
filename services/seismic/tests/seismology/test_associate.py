"""LOC-03: association with PyOcto on the layer model (hq.associate).

Synthetic data only, built here: a geometry of 12 stations around the run origin (3 borehole
sensors a few hundred metres below their wellheads), hypocentres at known elevM, P and S times
forward-modelled analytically (homogeneous model, straight rays) with Gaussian noise, random
false picks. Offline and seeded.
"""

import inspect
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyocto
import pytest
from hq_contracts.io import read_models, read_table, write_models, write_table
from hq_contracts.models import Pick, SweepPoint

from hq.associate import AssocResult, associate, associate_detailed
from hq.associate.core import (
    drop_outside_window,
    finish,
    merge_duplicates,
    pick_counts,
    prepared,
    require_pyocto,
    resolve_shared_picks,
    resolve_with_minimums,
    run_pyocto,
)
from hq.associate.frame import (
    elev_m_from_pyocto_z,
    enu_from_pyocto_xy,
    pyocto_xy_km,
    pyocto_z_km,
    search_volume,
)
from hq.associate.result import EVENT_DTYPES, PICK_DTYPES
from hq.associate.run import run as stage_run
from hq.associate.sweep import SweepScore, grid, run_sweep, sweep_points
from hq.associate.tables import (
    HEADER,
    PADDING,
    TableSpec,
    build_tables,
    seed_error_bound_s,
    station_times,
)
from hq.associate.tables import (
    read_table as read_tt_table,
)
from hq.config.run import RunSection
from hq.config.seismology import AssociatorConfig, SeismologyConfig
from hq.locate.coords import from_enu, to_enu
from hq.locate.velocity import LayerModel, SourceRef, load_configured_model

VP, VS = 5000.0, 2900.0  # homogeneous toy model (m/s)
SIGMA = {"P": 0.02, "S": 0.04}  # pick noise (s)
SEED = 20260926
N_EVENTS = 10
EVENT_SPACING_S = 25.0
PICKER = "phasenet:synthetic"


# --- synthetic world ---------------------------------------------------------------------------


def toy_model(tops: list[float], vp: list[float], vs: list[float]) -> LayerModel:
    return LayerModel(
        name="toy",
        datum="topElevM is m above mean sea level.",
        source=SourceRef(citation="test", url="https://example.invalid", verified=False),
        top_elev_m=np.array(tops),
        vp_m_per_s=np.array(vp),
        vs_m_per_s=np.array(vs),
        source_file="https://example.invalid/toy.csv",
        license="CC0",
    )


HOMOGENEOUS = toy_model([3000.0], [VP], [VS])


def make_stations(run: RunSection) -> pd.DataFrame:
    """12 stations within ~12 km of the origin; 3 borehole sensors 150-400 m below the wellhead."""
    rng = np.random.default_rng(SEED)
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False) + rng.uniform(-0.2, 0.2, 12)
    radii = np.r_[1500.0, 2500.0, 3000.0, rng.uniform(4000.0, 12000.0, 9)]
    e, n = radii * np.cos(angles), radii * np.sin(angles)
    surface = rng.uniform(1500.0, 1900.0, 12)
    depth = np.zeros(12)
    depth[:3] = [400.0, 290.0, 150.0]
    sensor = surface - depth
    lat, lon, _ = from_enu(e, n, sensor - run.origin.elevM, run.origin)
    ee, nn, uu = to_enu(lat, lon, sensor, run.origin)
    return pd.DataFrame(
        {
            "id": [f"XX.S{i:02d}" for i in range(12)],
            "network": "XX",
            "station": [f"S{i:02d}" for i in range(12)],
            "location": "",
            "latitude": lat,
            "longitude": lon,
            "surfaceElevM": surface,
            "sensorDepthM": depth,
            "sensorElevM": sensor,
            "kind": ["borehole"] * 3 + ["surface"] * 9,
            "channels": [["HHZ", "HHN", "HHE"]] * 12,
            "sampleRateHz": 100.0,
            "enu_e": ee,
            "enu_n": nn,
            "enu_u": uu,
            "preprocessProfile": "surface-A",
            "usedInRun": True,
            "staticsS": "{}",
        }
    )


def make_events(run: RunSection) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    return pd.DataFrame(
        {
            "t": run.window_start_s + 600.0 + EVENT_SPACING_S * np.arange(N_EVENTS),
            "e": rng.uniform(-4000.0, 4000.0, N_EVENTS),
            "n": rng.uniform(-4000.0, 4000.0, N_EVENTS),
            "elevM": rng.uniform(-4500.0, -500.0, N_EVENTS),
        }
    )


def pick_row(station: str, phase: str, t: float, prob: float, picker: str = PICKER) -> dict:
    prefix = "stalta" if picker == "stalta" else picker
    return {"id": f"{prefix}:{station}:{phase}:{t:.3f}", "stationId": station, "phase": phase,
            "t": t, "prob": prob, "picker": picker}


def make_picks(
    stations: pd.DataFrame,
    events: pd.DataFrame,
    *,
    n_false: int = 60,
    travel_time: Any = None,
    seed: int = SEED + 2,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Picks (docs/02 Pick columns) and the true event index of every pick (-1: false pick)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    truth: dict[str, int] = {}
    se, sn, sz = (stations[c].to_numpy() for c in ("enu_e", "enu_n", "sensorElevM"))
    for k, ev in events.iterrows():
        for j, sid in enumerate(stations["id"]):
            for phase, vel in (("P", VP), ("S", VS)):
                if travel_time is None:
                    tt = math.dist((se[j], sn[j], sz[j]), (ev.e, ev.n, ev.elevM)) / vel
                else:
                    tt = travel_time(sid, float(sz[j]), math.hypot(se[j] - ev.e, sn[j] - ev.n),
                                     float(ev.elevM), phase)
                t = float(ev.t + tt + rng.normal(0.0, SIGMA[phase]))
                row = pick_row(sid, phase, t, float(rng.uniform(0.5, 1.0)))
                rows.append(row)
                truth[row["id"]] = int(k)
    t0, t1 = float(events.t.min()) - 30.0, float(events.t.max()) + 60.0
    for _ in range(n_false):
        sid = str(stations["id"].iloc[rng.integers(len(stations))])
        row = pick_row(sid, "PS"[rng.integers(2)], float(rng.uniform(t0, t1)),
                       float(rng.uniform(0.3, 1.0)))
        rows.append(row)
        truth[row["id"]] = -1
    picks = pd.DataFrame(rows)
    assert not picks["id"].duplicated().any()
    return picks, truth


def _cfg(base: SeismologyConfig, **assoc: Any) -> SeismologyConfig:
    data = base.model_dump()
    data["associator"].update(assoc)
    return SeismologyConfig.model_validate(data)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One synthetic scenario shared by the module (tables cached in a module cache dir)."""
    import yaml

    from hq.config.seismology import SEISMIC_ROOT

    showcase = SEISMIC_ROOT / "configs" / "showcase"
    run = RunSection.model_validate(yaml.safe_load((showcase / "run.yaml").read_text()))
    base = SeismologyConfig.model_validate(
        yaml.safe_load((showcase / "seismology.yaml").read_text())
    )
    stations = make_stations(run)
    events = make_events(run)
    picks, truth = make_picks(stations, events)
    return {
        "run": run,
        "base": base,
        "cfg": _cfg(base, minPickProb=0.3, minStations=5),
        "stations": stations,
        "events": events,
        "picks": picks,
        "truth": truth,
        "cache": tmp_path_factory.mktemp("cache"),
    }


def _associate_toy(world: dict[str, Any], picks: pd.DataFrame | None = None,
                   cfg: SeismologyConfig | None = None) -> tuple[AssocResult, dict, dict]:
    return associate_detailed(
        world["picks"] if picks is None else picks,
        world["stations"],
        world["cfg"] if cfg is None else cfg,
        world["run"],
        cache_dir=world["cache"],
        model=HOMOGENEOUS,
    )


def _score(result: AssocResult, truth: dict[str, int]) -> dict[str, Any]:
    """Majority true event per associated event, and pick precision/recall."""
    labelled = result.picks.assign(truth=result.picks["pickId"].map(truth))
    majority = labelled.groupby("assocId")["truth"].agg(lambda s: s.value_counts().index[0])
    true_ids = [pid for pid, k in truth.items() if k >= 0]
    correct = int(
        (labelled["truth"] == labelled["assocId"].map(majority)).where(labelled["truth"] >= 0,
                                                                      False).sum()
    )
    return {
        "majority": majority,
        "precision": correct / max(len(labelled), 1),
        "recall": correct / len(true_ids),
    }


# --- frame and tables ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_other_pyocto_versions_fail_loudly() -> None:
    require_pyocto(pyocto.__version__)  # the installed one passes (the module imported)
    for version in ("0.1.4", "0.2.1"):
        with pytest.raises(ImportError, match=r"pyocto==0\.2\.0"):
            require_pyocto(version)


@pytest.mark.smoke
def test_elev_to_pyocto_depth_round_trip() -> None:
    rng = np.random.default_rng(1)
    elev = np.r_[0.0, 1627.7, -10000.0, 2421.0, rng.uniform(-12000.0, 4000.0, 10000)]
    z = pyocto_z_km(elev)
    assert z[0] == 0.0 and z[2] == 10.0 and z[3] == -2.421  # km below sea level, down positive
    assert np.all(np.sign(z[1:]) == -np.sign(elev[1:]))
    back = elev_m_from_pyocto_z(z)
    # ÷1000 then ×1000 is exact up to one rounding of the double (never a datum or unit shift)
    np.testing.assert_allclose(back, elev, rtol=0.0, atol=1e-9)
    e, n = rng.uniform(-30000.0, 30000.0, (2, 1000))
    ee, nn = enu_from_pyocto_xy(*pyocto_xy_km(e, n))
    np.testing.assert_allclose(ee, e, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(nn, n, rtol=0.0, atol=1e-9)


@pytest.mark.smoke
def test_search_volume_covers_bbox_and_uses_ref_surface(world: dict[str, Any]) -> None:
    run, acfg = world["run"], world["base"].associator
    vol = search_volume(acfg.volume, run)
    assert vol.top_elev_m == run.refSurfaceElevM  # topElevM null
    assert vol.bottom_elev_m == acfg.volume.bottomElevM
    min_lon, min_lat, max_lon, max_lat = run.bbox
    lat, lon = np.meshgrid(np.linspace(min_lat, max_lat, 41), np.linspace(min_lon, max_lon, 41))
    e, n, _ = to_enu(lat.ravel(), lon.ravel(), np.zeros(lat.size), run.origin)
    assert vol.e_min_m <= e.min() and e.max() <= vol.e_max_m
    assert vol.n_min_m <= n.min() and n.max() <= vol.n_max_m
    assert vol.zlim_km == (-run.refSurfaceElevM / 1000.0, -acfg.volume.bottomElevM / 1000.0)


def _spec(spacing_m: float, r_max_m: float, below_m: float, above_m: float) -> TableSpec:
    """A table shape at the configured spacing reaching the given extents around the sensor."""
    pad = round(above_m / spacing_m)
    return TableSpec(spacing_m=spacing_m, nx=round(r_max_m / spacing_m) + 1,
                     nz=pad + round(below_m / spacing_m) + 1, n_padding=pad)


@pytest.mark.smoke
def test_tables_match_straight_rays_in_a_homogeneous_model(world: dict[str, Any]) -> None:
    tcfg = world["base"].associator.tables
    spec = _spec(tcfg.spacingM, 25000.0, 11000.0, 500.0)
    for sensor in (1650.0, 1350.0):  # surface and borehole sensor elevations
        p, s = station_times(HOMOGENEOUS, sensor, spec, tcfg)
        assert p.shape == s.shape == (spec.nx, spec.nz)
        assert p[0, spec.n_padding] == 0.0 and s[0, spec.n_padding] == 0.0  # source = sensor
        r = np.arange(spec.nx)[:, None] * spec.spacing_m
        dz = (np.arange(spec.nz)[None, :] - spec.n_padding) * spec.spacing_m
        dist = np.hypot(r, dz)
        assert np.max(np.abs(p - dist / VP)) < 0.005
        assert np.max(np.abs(s - dist / VS)) < 0.008


@pytest.mark.smoke
def test_tables_honour_layers_head_wave(world: dict[str, Any]) -> None:
    """Two layers: past the crossover a surface-to-surface first arrival is the head wave."""
    tcfg = world["base"].associator.tables
    v1, v2, interface, sensor = 3000.0, 6000.0, 0.0, 1000.0
    model = toy_model([3000.0, interface], [v1, v2], [v1 / 1.8, v2 / 1.8])
    spec = _spec(tcfg.spacingM, 40000.0, 11000.0, 500.0)
    p, _ = station_times(model, sensor, spec, tcfg)
    h = sensor - interface
    for r in (1000.0, 2000.0, 10000.0, 20000.0, 30000.0, 39000.0):
        direct = r / v1
        head = r / v2 + 2.0 * h * math.sqrt(1.0 / v1**2 - 1.0 / v2**2)
        expected = min(direct, head)
        got = p[round(r / spec.spacing_m), spec.n_padding]
        assert abs(got - expected) < 0.02, (r, got, expected)
    assert p[int(20000 / spec.spacing_m), spec.n_padding] < 20000.0 / v1 - 1.0  # head wave wins


@pytest.mark.smoke
def test_seed_error_bound() -> None:
    model = toy_model([3000.0, 1000.0], [3000.0, 4000.0], [1500.0, 2000.0])
    assert seed_error_bound_s(model, 1500.0, 150.0) == 0.0  # interface 500 m away
    bound = seed_error_bound_s(model, 1050.0, 150.0)  # interface 50 m below, S step worst
    assert bound == pytest.approx(100.0 * (1.0 / 1500.0 - 1.0 / 2000.0))


def test_table_files_and_cache(world: dict[str, Any], tmp_path: Path) -> None:
    run, acfg = world["run"], world["base"].associator
    stations = world["stations"]
    vol = search_volume(acfg.volume, run)
    ids = list(stations["id"])
    elev = stations["sensorElevM"].to_numpy()
    far = vol.farthest_horizontal_m(stations["enu_e"].to_numpy(), stations["enu_n"].to_numpy())
    first = build_tables(HOMOGENEOUS, ids, elev, far, vol, acfg.tables, tmp_path)
    assert first.built and first.directory.parent == tmp_path / "ttgrids" / "pyocto"
    spec = first.spec
    # every volume point lies inside every station's table (PyOcto returns NaN at nx-1 / nz-1)
    assert (spec.nx - 2) * spec.spacing_m >= far.max()
    assert spec.n_padding * spec.spacing_m >= vol.top_elev_m - elev.min()
    assert (spec.nz - 2 - spec.n_padding) * spec.spacing_m >= elev.max() - vol.bottom_elev_m
    raw = (first.directory / f"{ids[0]}.pyocto").read_bytes()
    assert HEADER.unpack_from(raw) == (spec.nx, spec.nz, spec.spacing_m / 1000.0)
    assert PADDING.unpack((first.directory / "n_padding").read_bytes()) == (spec.n_padding,)
    got, p, s = read_tt_table(first.directory / f"{ids[0]}.pyocto")
    assert (got.nx, got.nz) == (spec.nx, spec.nz) and np.all(s >= p)
    again = build_tables(HOMOGENEOUS, ids, elev, far, vol, acfg.tables, tmp_path)
    assert not again.built and again.directory == first.directory
    moved = elev.copy()
    moved[0] += 1.0  # a changed sensor elevation is a different table set
    other = build_tables(HOMOGENEOUS, ids, moved, far, vol, acfg.tables, tmp_path)
    assert other.built and other.directory != first.directory


def test_model_top_extended_to_highest_station_and_recorded(world: dict[str, Any]) -> None:
    cfg, run = world["base"], world["run"]
    stations = world["stations"].copy()
    stations.loc[3, ["surfaceElevM", "sensorElevM"]] = 2400.0  # above the FORGE model top (1800)
    stations.loc[3, "enu_u"] = 2400.0 - run.origin.elevM
    source_top = load_configured_model(cfg.velocity).top_of_model_elev_m
    with prepared(stations, cfg, run, cache_dir=world["cache"]) as setup:
        ext = setup.velocity_model["topExtension"]
        assert ext == {**ext, "fromElevM": source_top, "toElevM": 2400.0}
        assert setup.tables.model_top_elev_m == 2400.0
        above = setup.station_note["aboveModelSourceTopM"]
        expected = {
            str(sid): float(z) - source_top
            for sid, z in zip(stations["id"], stations["sensorElevM"], strict=True)
            if z > source_top
        }
        assert above == expected and "XX.S03" in above


def test_pyocto_stations_sit_at_the_sensor_not_the_wellhead(world: dict[str, Any]) -> None:
    """PyOcto station z is -sensorElevM / 1000 (km below sea level), boreholes included."""
    stations = world["stations"]
    with prepared(stations, world["cfg"], world["run"], model=HOMOGENEOUS,
                  cache_dir=world["cache"]) as setup:
        frame = setup.stations.set_index("id").loc[stations["id"]]
    np.testing.assert_array_equal(frame["z"].to_numpy(), -stations["sensorElevM"].to_numpy() / 1e3)
    borehole = (stations["kind"] == "borehole").to_numpy()
    assert borehole.any() and (stations["sensorDepthM"].to_numpy()[borehole] > 100.0).all()
    assert not np.allclose(
        frame["z"].to_numpy()[borehole], -stations["surfaceElevM"].to_numpy()[borehole] / 1e3
    )


# --- association ----------------------------------------------------------------------------------


def test_synthetic_events_associated_with_the_right_picks(world: dict[str, Any]) -> None:
    result, counts, _ = _associate_toy(world)
    events, truth = world["events"], world["truth"]
    assert len(result.events) == N_EVENTS
    score = _score(result, truth)
    assert sorted(score["majority"]) == list(range(N_EVENTS))  # each true event exactly once
    assert score["precision"] >= 0.97, score
    assert score["recall"] >= 0.97, score
    assert not result.picks["pickId"].duplicated().any()  # no pick in two events
    # located where the synthetic events are (PyOcto's preliminary location, not the locator's)
    by_truth = result.events.set_index(result.events["assocId"].map(score["majority"]))
    ev = events.loc[by_truth.index]
    e, n, _ = to_enu(by_truth.latitude.to_numpy(), by_truth.longitude.to_numpy(),
                     by_truth.elevM.to_numpy(), world["run"].origin)
    dz = by_truth.elevM.to_numpy() - ev.elevM.to_numpy()
    dh = np.hypot(e - ev.e.to_numpy(), n - ev.n.to_numpy())
    dt = by_truth.t.to_numpy() - ev.t.to_numpy()
    assert np.median(np.abs(dz)) < 300.0 and np.max(np.abs(dz)) < 800.0, dz
    assert np.median(dh) < 300.0 and np.max(dh) < 800.0, dh
    assert np.max(np.abs(dt)) < 0.15, dt
    assert abs(np.median(dz)) < 250.0  # no datum offset (sea level vs surface would be ~1.6 km)
    counted = pick_counts(result.picks.rename(columns={"assocId": "eid"}).merge(
        world["picks"][["id", "stationId", "phase"]].rename(
            columns={"id": "pickId", "stationId": "station"}), on="pickId"))
    assert counted["nPicks"].tolist() == result.events.set_index("assocId")["nPicks"].tolist()
    assert counts["events"] == N_EVENTS and counts["assocPicks"] == len(result.picks)


def test_output_schema_and_deterministic_ids(world: dict[str, Any]) -> None:
    result, _, _ = _associate_toy(world)
    for frame, dtypes in ((result.events, EVENT_DTYPES), (result.picks, PICK_DTYPES)):
        assert list(frame.columns) == list(dtypes)
        assert {c: str(d) for c, d in frame.dtypes.items()} == {
            c: str(pd.Series([], dtype=d).dtype) for c, d in dtypes.items()
        }
    assert result.events["t"].is_monotonic_increasing
    assert list(result.events["assocId"]) == [f"assoc-{k:06d}" for k in range(N_EVENTS)]
    assert set(result.picks["assocId"]) == set(result.events["assocId"])
    ev = result.events.set_index("assocId")
    per = result.picks.merge(world["picks"][["id", "phase"]], left_on="pickId", right_on="id")
    assert (per.groupby("assocId").size() == ev["nPicks"]).all()
    assert (ev["nP"] + ev["nS"] == ev["nPicks"]).all()
    again, _, _ = _associate_toy(world)
    pd.testing.assert_frame_equal(again.events, result.events)
    pd.testing.assert_frame_equal(again.picks, result.picks)


def test_thread_count_does_not_change_results(world: dict[str, Any]) -> None:
    one, _, _ = _associate_toy(world, cfg=_cfg(world["cfg"], nThreads=1))
    many, _, _ = _associate_toy(world, cfg=_cfg(world["cfg"], nThreads=4))
    pd.testing.assert_frame_equal(one.events, many.events)
    pd.testing.assert_frame_equal(one.picks, many.picks)


def test_stalta_labelled_picks_associate_unchanged(world: dict[str, Any]) -> None:
    picks = world["picks"]
    stalta = picks.assign(
        picker="stalta", id="stalta:" + picks["id"].str.split(":", n=2).str[2]
    )
    rename = dict(zip(picks["id"], stalta["id"], strict=True))
    base, _, _ = _associate_toy(world)
    other, _, _ = _associate_toy(world, picks=stalta)
    pd.testing.assert_frame_equal(other.events, base.events)
    mapped = base.picks.assign(pickId=base.picks["pickId"].map(rename).astype("string"))
    pd.testing.assert_frame_equal(
        other.picks.sort_values(["assocId", "pickId"]).reset_index(drop=True),
        mapped.sort_values(["assocId", "pickId"]).reset_index(drop=True),
    )


def test_zero_picks_give_typed_zero_row_tables(world: dict[str, Any], tmp_path: Path) -> None:
    empty = pd.DataFrame({c: pd.Series([], dtype=d) for c, d in (
        ("id", "string"), ("stationId", "string"), ("phase", "string"), ("t", "float64"),
        ("prob", "float64"), ("picker", "string"))})
    low = world["picks"].assign(prob=0.1)  # everything below minPickProb
    noise = world["picks"][world["picks"]["id"].map(world["truth"]) == -1].head(5)
    for picks in (empty, low, noise):
        result, counts, _ = _associate_toy(world, picks=picks)
        assert len(result.events) == 0 and len(result.picks) == 0
        assert counts["events"] == 0
        write_table(result.events, tmp_path / "e.parquet", "AssocEvent")
        write_table(result.picks, tmp_path / "p.parquet", "AssocPick")
        schema = pq.read_schema(tmp_path / "e.parquet")
        assert {f.name: str(f.type) for f in schema} == {
            "assocId": "large_string", "t": "double", "latitude": "double", "longitude": "double",
            "elevM": "double", "nPicks": "int64", "nP": "int64", "nS": "int64"}
        schema = pq.read_schema(tmp_path / "p.parquet")
        assert {f.name: str(f.type) for f in schema} == {
            "assocId": "large_string", "pickId": "large_string"}


def test_public_signature_forge_model_and_cache(world: dict[str, Any]) -> None:
    """``associate(picks, stations, cfg, run)`` on the configured FORGE layer model.

    The forward model here is the associator's own tables (bilinear, as PyOcto reads them), so
    this checks the public call path and the layered model, not table accuracy (tested above).
    """
    run, cfg, stations = world["run"], world["cfg"], world["stations"]
    with prepared(stations, cfg, run, cache_dir=world["cache"]) as setup:
        spec, n_pad = setup.tables.spec, setup.tables.spec.n_padding
        tabs = {sid: read_tt_table(setup.tables.directory / f"{sid}.pyocto")
                for sid in stations["id"]}

    def travel_time(sid: str, sensor: float, r_m: float, elev: float, phase: str) -> float:
        _, p, s = tabs[sid]
        grid_t = p if phase == "P" else s
        x = r_m / spec.spacing_m
        z = (sensor - elev) / spec.spacing_m + n_pad
        ix, iz = int(x), int(z)
        ax, az = x - ix, z - iz
        return float(grid_t[ix, iz] * (1 - ax) * (1 - az) + grid_t[ix + 1, iz] * ax * (1 - az)
                     + grid_t[ix, iz + 1] * (1 - ax) * az + grid_t[ix + 1, iz + 1] * ax * az)

    picks, truth = make_picks(stations, world["events"], travel_time=travel_time)
    result = associate(picks, stations, cfg, run)  # exact docs/02 call: tables in a temp dir
    assert len(result.events) == N_EVENTS
    score = _score(result, truth)
    assert score["precision"] >= 0.97 and score["recall"] >= 0.97, score
    cached, _, rec = associate_detailed(picks, stations, cfg, run, cache_dir=world["cache"])
    pd.testing.assert_frame_equal(cached.events, result.events)
    assert rec["velocityModel"]["name"] == load_configured_model(cfg.velocity).name
    assert rec["tables"]["modelTopElevM"] >= stations["sensorElevM"].max()


def test_record_holds_every_pyocto_argument(world: dict[str, Any]) -> None:
    _, _, rec = _associate_toy(world)
    call = rec["pyocto"]["OctoAssociator"]
    expected = set(inspect.signature(pyocto.OctoAssociator.__init__).parameters) - {"self"}
    assert set(call) == expected
    vm = call["velocity_model"]
    vm_expected = set(inspect.signature(pyocto.StationSpecificVelocityModel1D.__init__).parameters)
    assert set(vm) - {"class"} == vm_expected - {"self"}
    acfg = world["cfg"].associator
    assert call["n_picks"] == acfg.nPicks and call["n_s_picks"] == acfg.nSPicks
    assert call["zlim"] == [-world["run"].refSurfaceElevM / 1000.0,
                            -acfg.volume.bottomElevM / 1000.0]
    assert vm["tolerance"] == acfg.toleranceS
    assert rec["picks"]["minPickProb"] == acfg.minPickProb
    assert "no per-pick weight" in rec["picks"]["weights"]
    assert rec["config"] == acfg.model_dump(mode="json")
    run = world["run"]
    assert (rec["window"]["startS"], rec["window"]["endS"]) == (run.window_start_s, run.window_end_s)
    assert rec["picks"]["nIn"] == len(world["picks"])
    assert rec["picks"]["tMinS"] == world["picks"]["t"].min()
    assert rec["picks"]["tMaxS"] == world["picks"]["t"].max()
    json.dumps(rec)  # ProcessingRun.associator is JSON


# --- duplicate merge and shared picks ------------------------------------------------------------


def _events(rows: list[tuple[int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"eid": [r[0] for r in rows], "t": [r[1] for r in rows], "x": 0.0, "y": 0.0, "z": 3.0}
    )


def _assign(rows: list[tuple[int, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eid": [r[0] for r in rows],
            "pickId": [r[1] for r in rows],
            "station": [r[1].split(":")[0] for r in rows],
            "phase": [r[1].split(":")[1] for r in rows],
            "residual": [r[2] for r in rows],
        }
    )


@pytest.mark.smoke
def test_duplicate_merge() -> None:
    a = [f"A{i}:P:1" for i in range(4)]
    events = _events([(0, 100.0), (1, 100.3), (2, 200.0), (3, 200.3), (4, 300.0), (5, 300.8)])
    assign = _assign(
        [(0, p, 0.01) for p in a]  # event 0: 4 picks
        + [(1, p, 0.02) for p in a[:3]] + [(1, "B9:S:1", 0.0)]  # event 1 shares 3 of 4: merge
        + [(2, f"C{i}:P:2", 0.0) for i in range(6)]
        + [(3, "C0:P:2", 0.0)] + [(3, f"D{i}:P:2", 0.0) for i in range(5)]  # shares 1 of 6: keep
        + [(4, f"E{i}:P:3", 0.0) for i in range(4)]
        + [(5, f"E{i}:P:3", 0.0) for i in range(4)]  # identical picks but 0.8 s apart: keep
    )
    out_e, out_a, merged = merge_duplicates(events, assign, 0.5, 0.5)
    assert merged == 1
    assert sorted(out_e["eid"]) == [0, 2, 3, 4, 5]
    assert set(out_a.loc[out_a["eid"] == 0, "pickId"]) == {*a, "B9:S:1"}  # union of picks
    assert out_e.loc[out_e["eid"] == 0, "t"].item() == 100.0  # larger event keeps its origin
    assert not out_a.duplicated(["eid", "station", "phase"]).any()


@pytest.mark.smoke
def test_duplicate_merge_keeps_one_pick_per_station_phase() -> None:
    events = _events([(0, 10.0), (1, 10.2), (2, 10.4)])
    assign = _assign(
        [(0, "A:P:1", 0.0), (0, "B:P:1", 0.0), (0, "C:P:1", 0.0)]
        + [(1, "A:P:1", 0.0), (1, "B:P:1", 0.0), (1, "C:P:2", 0.0), (1, "D:P:1", 0.0)]
        + [(2, "C:P:2", 0.0), (2, "D:P:1", 0.0)]
    )
    out_e, out_a, merged = merge_duplicates(events, assign, 0.5, 0.5)
    assert merged == 2 and list(out_e["eid"]) == [1]  # most picks keeps the origin
    got = dict(zip(out_a["station"], out_a["pickId"], strict=True))
    assert got == {"A": "A:P:1", "B": "B:P:1", "C": "C:P:2", "D": "D:P:1"}  # primary's C pick


@pytest.mark.smoke
def test_duplicate_merge_is_pairwise_not_transitive() -> None:
    """0 ~ 1 and 1 ~ 2, but 0 and 2 are 0.8 s apart: 1 joins 0 (ranked first), 2 stays."""
    events = _events([(0, 10.0), (1, 10.4), (2, 10.8)])
    assign = _assign(
        [(0, f"A{i}:P:1", 0.0) for i in range(5)]
        + [(1, "A3:P:1", 0.0), (1, "A4:P:1", 0.0), (1, "B0:P:1", 0.0), (1, "B1:P:1", 0.0)]
        + [(2, "B0:P:1", 0.0), (2, "B1:P:1", 0.0)] + [(2, f"C{i}:P:1", 0.0) for i in range(3)]
    )
    out_e, out_a, merged = merge_duplicates(events, assign, 0.5, 0.5)
    assert merged == 1 and list(out_e["eid"]) == [0, 2]
    assert set(out_a.loc[out_a["eid"] == 0, "station"]) == {"A0", "A1", "A2", "A3", "A4", "B0",
                                                             "B1"}


@pytest.mark.smoke
def test_shared_pick_never_lost_to_an_event_that_is_then_dropped(world: dict[str, Any]) -> None:
    """B wins pick x from A (smaller residual) but loses y to C and is dropped: x returns to A.

    Resolving before the minimums would leave A with 4 stations, below minStations 5.
    """
    acfg = world["base"].associator.model_copy(
        update={"minStations": 5, "nPicks": 4, "nPPicks": 2, "nSPicks": 1, "nPAndSPicks": 0}
    )
    events = _events([(0, 10.0), (1, 11.0), (2, 12.0)])
    assign = _assign(
        [(0, f"S{i}:P:a", 0.1) for i in range(1, 6)] + [(0, "S1:S:a", 0.1)]  # A: 5 stations
        + [(1, "S5:P:a", 0.0), (1, "T1:P:b", 0.0), (1, "T2:P:b", 0.0), (1, "T3:P:b", 0.0),
           (1, "T4:S:b", 0.2)]  # B: 5 stations with x = S5:P:a and y = T4:S:b
        + [(2, f"U{i}:P:c", 0.0) for i in range(1, 6)] + [(2, "T4:S:b", 0.0)]  # C: wins y
    )
    events_out, kept, shared, low_pyocto, low_stations = resolve_with_minimums(
        events, assign, acfg
    )
    assert list(events_out["eid"]) == [0, 2]
    assert kept.loc[kept["pickId"] == "S5:P:a", "eid"].tolist() == [0]
    assert kept.loc[kept["pickId"] == "T4:S:b", "eid"].tolist() == [2]
    assert low_pyocto + low_stations == 1 and shared == 0
    assert not kept["pickId"].duplicated().any()


@pytest.mark.smoke
def test_shared_pick_stays_with_smallest_residual() -> None:
    events = _events([(0, 10.0), (1, 12.0)])
    assign = _assign(
        [(0, "A:P:1", 0.30), (0, "B:P:1", 0.0), (1, "A:P:1", -0.05), (1, "C:P:1", 0.0)]
    )
    kept, removed = resolve_shared_picks(events, assign)
    assert removed == 1
    assert kept.loc[kept["pickId"] == "A:P:1", "eid"].tolist() == [1]


def test_duplicates_from_pyocto_are_merged_in_finish(world: dict[str, Any]) -> None:
    """A PyOcto event duplicated 0.2 s later with most of its picks becomes one event again."""
    cfg = world["cfg"].associator
    with prepared(world["stations"], world["cfg"], world["run"], model=HOMOGENEOUS,
                  cache_dir=world["cache"]) as setup:
        raw = run_pyocto(world["picks"], setup, cfg)
        clean, clean_counts = finish(raw, cfg, setup)
        first = int(raw.events["eid"].iloc[0])
        dup_rows = raw.assignments[raw.assignments["eid"] == first].iloc[1:]
        dup_eid = int(raw.events["eid"].max()) + 1
        dup_event = raw.events[raw.events["eid"] == first].assign(eid=dup_eid, t=lambda d: d.t + 0.2)
        doubled = type(raw)(
            events=pd.concat([raw.events, dup_event], ignore_index=True),
            assignments=pd.concat([raw.assignments, dup_rows.assign(eid=dup_eid)],
                                  ignore_index=True),
            picks_in=raw.picks_in, picks_used=raw.picks_used,
            picks_unused_stations=raw.picks_unused_stations, runtime_s=raw.runtime_s,
        )
        merged, counts = finish(doubled, cfg, setup)
    assert counts["mergedDuplicates"] == clean_counts["mergedDuplicates"] + 1
    pd.testing.assert_frame_equal(merged.events, clean.events)
    pd.testing.assert_frame_equal(merged.picks, clean.picks)


def test_events_outside_the_run_window_are_dropped(world: dict[str, Any]) -> None:
    """In-window picks of an event whose origin precedes windowStart form no candidate event."""
    run = world["run"]
    events = world["events"].head(1).assign(t=run.window_start_s - 1.0, elevM=-3000.0)
    picks, _ = make_picks(world["stations"], events, n_false=0)
    inside = picks[picks["t"] >= run.window_start_s].reset_index(drop=True)
    assert len(inside) >= 2 * world["cfg"].associator.minStations
    cfg = world["cfg"]
    with prepared(world["stations"], cfg, run, model=HOMOGENEOUS,
                  cache_dir=world["cache"]) as setup:
        raw = run_pyocto(inside, setup, cfg.associator)
        assert len(raw.events) >= 1 and float(raw.events["t"].min()) < run.window_start_s
        result, counts = finish(raw, cfg.associator, setup)
    assert len(result.events) == 0
    assert counts["droppedOutsideWindow"] == len(raw.events)
    # [start, end): an origin exactly at windowStart stays, one exactly at windowEnd goes
    ev = _events([(0, run.window_start_s), (1, run.window_end_s), (2, run.window_end_s - 1e-3)])
    assign = _assign([(0, "A:P:1", 0.0), (1, "B:P:1", 0.0), (2, "C:P:1", 0.0)])
    kept, kept_assign, dropped = drop_outside_window(
        ev, assign, (run.window_start_s, run.window_end_s)
    )
    assert kept["eid"].tolist() == [0, 2] and kept_assign["eid"].tolist() == [0, 2]
    assert dropped == 1


# --- input checks and config -------------------------------------------------------------------


def test_inputs_fail_loudly(world: dict[str, Any]) -> None:
    picks, stations = world["picks"], world["stations"]
    with pytest.raises(ValueError, match="missing from the stations table"):
        _associate_toy(world, picks=pd.concat([picks, pd.DataFrame(
            [pick_row("XX.NOPE", "P", float(picks.t.iloc[0]), 0.9)])]))
    with pytest.raises(ValueError, match="duplicate pick ids"):
        _associate_toy(world, picks=pd.concat([picks, picks.head(1)]))
    with pytest.raises(ValueError, match="P or S"):
        _associate_toy(world, picks=picks.assign(phase="Pg"))
    shifted = stations.copy()
    shifted.loc[1, "enu_u"] += 290.0  # enu at the wellhead, not the sensor
    with pytest.raises(ValueError, match="stored enu differs"):
        associate_detailed(picks, shifted, world["cfg"], world["run"], model=HOMOGENEOUS,
                           cache_dir=world["cache"])
    with pytest.raises(ValueError, match="timeBeforeS"):
        _associate_toy(world, cfg=_cfg(world["cfg"], timeBeforeS=1.0))


def test_picks_from_stations_not_used_in_the_run(world: dict[str, Any]) -> None:
    """H1 picks usedInRun-false stations on purpose: dropped and counted, or rejected (config)."""
    stations = world["stations"].copy()
    stations.loc[stations["id"] == "XX.S05", "usedInRun"] = False
    from_unused = int((world["picks"]["stationId"] == "XX.S05").sum())
    assert from_unused > 0
    with_all, counts_all, _ = _associate_toy(world)
    dropped, counts, rec = associate_detailed(
        world["picks"], stations, world["cfg"], world["run"], model=HOMOGENEOUS,
        cache_dir=world["cache"],
    )
    assert counts_all["picksFromUnusedStationsDropped"] == 0
    assert counts["picksFromUnusedStationsDropped"] == from_unused
    assert rec["stations"]["nStations"] == len(stations) - 1
    assert rec["picks"]["fromUnusedStations"].startswith("drop")
    kept_stations = set(
        world["picks"].set_index("id").loc[dropped.picks["pickId"], "stationId"].astype(str)
    )
    assert "XX.S05" not in kept_stations and len(dropped.events) == len(with_all.events)
    with pytest.raises(ValueError, match="usedInRun false"):
        associate_detailed(
            world["picks"], stations, _cfg(world["cfg"], picksFromUnusedStations="reject"),
            world["run"], model=HOMOGENEOUS, cache_dir=world["cache"],
        )


@pytest.mark.smoke
def test_config_rejects_values_pyocto_would_silently_raise(world: dict[str, Any]) -> None:
    acfg = world["base"].associator.model_dump()
    for bad in (
        {"nPicks": 3, "nPPicks": 2, "nSPicks": 2},  # PyOcto would use 4 picks
        {"nPAndSPicks": 1},  # sweep nSPicks 0 < n_p_and_s_picks
        {"nPicks": 5},  # above the smallest sweep minStations (4)
        {"locationSplitReturn": 6},
        {"minNodeSizeLocationKm": 5.0},
    ):
        with pytest.raises(ValueError):
            AssociatorConfig.model_validate({**acfg, **bad})


# --- stage and sweep -----------------------------------------------------------------------------


def _write_inputs(ctx: Any, world: dict[str, Any]) -> None:
    write_table(world["stations"], ctx.path("stations.parquet"), "Station")
    rows = world["picks"][["id", "stationId", "phase", "t", "prob", "picker"]]
    picks = [Pick(**r) for r in rows.to_dict("records")]
    write_models(picks, ctx.path("picks.parquet"), Pick)


def test_stage_writes_tables_record_and_counts_sweep(
    world: dict[str, Any], make_ctx: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hq.associate import core

    monkeypatch.setattr(core, "load_configured_model", lambda _cfg: HOMOGENEOUS)
    sweep = {"enabled": True, "minStations": [4, 8], "nSPicks": [1, 2], "minPickProb": [0.3]}
    seis = _cfg(world["cfg"], sweep=sweep)  # the configured point (0.3, nS 1) is a sweep point
    ctx = make_ctx(world["run"], seis)
    _write_inputs(ctx, world)
    ctx.path("sweep.parquet").write_bytes(b"stale")
    runs: list[float] = []
    real_run_pyocto = core.run_pyocto

    def counting_run_pyocto(*args: Any, **kwargs: Any) -> Any:
        runs.append(1.0)
        return real_run_pyocto(*args, **kwargs)

    monkeypatch.setattr("hq.associate.sweep.run_pyocto", counting_run_pyocto)
    stage_run(ctx)
    events = read_table(ctx.path("assoc_events.parquet"))
    picks = read_table(ctx.path("assoc_picks.parquet"))
    assert events.attrs["model"] == "AssocEvent" and len(events) == N_EVENTS
    assert list(picks.columns) == list(PICK_DTYPES) and not picks["pickId"].duplicated().any()
    assert not ctx.path("sweep.parquet").exists()  # the stale sweep of another association
    assert not list(ctx.run_dir.glob("*.part"))
    assert len(runs) == 1  # the sweep reran PyOcto only for nS 2, not the configured point
    (rec,) = ctx.records
    assert rec["stage"] == "associate" and rec["counts"]["events"] == N_EVENTS
    assert rec["counts"]["sweepPoints"] == 4
    params = rec["params"]
    assert params["sweep"]["grid"] == grid(seis.associator)
    assert [p["associated"] for p in params["sweep"]["points"]][1] == N_EVENTS  # (0.3, 1, 8)
    assert "LOC-06" in params["sweep"]["note"]
    assert (ctx.cache_dir / params["tables"]["directory"]).is_dir()
    assert str(ctx.cache_dir) not in json.dumps(params)  # no machine-specific path in run.json

    ctx.path("known").mkdir()
    write_table(world["stations"], ctx.path("known/picks.parquet"), "Station")
    with pytest.raises(ValueError, match="must hold 'Pick' rows"):
        stage_run(make_ctx(world["run"], _cfg(seis, picksTable="known/picks.parquet")))


def test_sweep_scores_every_point_with_the_evaluator(
    world: dict[str, Any], tmp_path: Path
) -> None:
    """LOC-06 calls run_sweep with an evaluator and writes sweep_points as sweep.parquet."""
    sweep = {"enabled": True, "minStations": [4, 8], "nSPicks": [1], "minPickProb": [0.3, 0.5]}
    acfg = _cfg(world["cfg"], sweep=sweep).associator
    calls: list[int] = []

    def evaluate(result: AssocResult) -> SweepScore:
        calls.append(len(result.events))
        return SweepScore(candidates=len(result.events) - 1,
                          recovered_public=len(result.events) // 2, tier_a=1)

    with prepared(world["stations"], world["cfg"], world["run"], model=HOMOGENEOUS,
                  cache_dir=world["cache"]) as setup:
        rows = run_sweep(world["picks"], setup, acfg, evaluate)
        unscored = run_sweep(world["picks"], setup, acfg)
    points = sweep_points(rows)
    assert len(points) == 4 == len(calls)
    for point, row, (prob, n_s, n_sta) in zip(
        points, rows, itertools.product([0.3, 0.5], [1], [4, 8]), strict=True
    ):
        assert point.params == {"minPickProb": prob, "nSPicks": n_s, "minStations": n_sta}
        assert point.candidates == row.candidates - 1  # the evaluator's located count
        assert point.recoveredPublic == row.candidates // 2 and point.tierA == 1
    assert rows[0].candidates >= rows[1].candidates  # minStations 4 keeps at least minStations 8
    write_models(points, tmp_path / "sweep.parquet", SweepPoint)  # docs/02 SweepPoint rows
    assert read_models(tmp_path / "sweep.parquet", SweepPoint) == points
    with pytest.raises(ValueError, match="no evaluator score"):
        sweep_points(unscored)


@pytest.mark.smoke
def test_stage_resolves_from_the_package_and_the_module() -> None:
    """The stage registry imports ``hq.associate`` and calls its ``run(ctx)`` (docs/01)."""
    import importlib

    assert importlib.import_module("hq.associate").run is stage_run
    assert importlib.import_module("hq.associate.run").run is stage_run
    assert list(inspect.signature(stage_run).parameters) == ["ctx"]  # docs/02 §4, exactly


# --- throughput (not smoke: about a minute) ------------------------------------------------------


def test_throughput_on_a_100k_pick_day(world: dict[str, Any]) -> None:
    """~100k picks over 24 h (2,000 events + 50,000 false picks) at the configured settings."""
    run, stations = world["run"], world["stations"]
    rng = np.random.default_rng(SEED + 9)
    n_events = 2000
    events = pd.DataFrame(
        {
            "t": np.sort(rng.uniform(run.window_start_s, run.window_end_s - 60.0, n_events)),
            "e": rng.normal(0.0, 2000.0, n_events),
            "n": rng.normal(0.0, 2000.0, n_events),
            "elevM": rng.uniform(-4500.0, -500.0, n_events),
        }
    )
    picks, truth = make_picks(stations, events, n_false=0)
    false = pd.DataFrame([
        pick_row(str(stations["id"].iloc[rng.integers(len(stations))]), "PS"[rng.integers(2)],
                 float(rng.uniform(run.window_start_s, run.window_end_s)), 0.5)
        for _ in range(50000)
    ]).drop_duplicates("id")
    picks = pd.concat([picks, false], ignore_index=True)
    started = time.perf_counter()
    result, counts, _ = _associate_toy(world, picks=picks)
    elapsed = time.perf_counter() - started
    print(f"\n{len(picks)} picks over 24 h: {counts['events']} events in {elapsed:.1f} s")
    score = _score(result, {**truth, **dict.fromkeys(false["id"], -1)})
    assert score["majority"][score["majority"] >= 0].nunique() >= 0.98 * n_events
    assert elapsed < 600.0

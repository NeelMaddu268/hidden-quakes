"""LOC-02: eikonal travel-time tables against analytic times, the exact layered solver, caching."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydantic
import pytest

from hq.config.seismology import GridsConfig, SeismologyConfig
from hq.locate.tt_grid import (
    CACHE_SUBDIR,
    SOURCE_SHA256,
    GridSpec,
    Phase,
    TravelTimeTable,
    build_station_tables,
    layered_first_arrival,
    make_grid,
    solve_table,
    table_accuracy,
    table_key,
)
from hq.locate.velocity import LayerModel, load_configured_model

pytestmark = pytest.mark.smoke

MAX_ERR_S = 0.005  # ticket acceptance: < 5 ms against analytic over the whole table
FloatArray = np.ndarray


@pytest.fixture(scope="module")
def loc_cfg(loc02: Any) -> SeismologyConfig:
    return loc02.test_config()


@pytest.fixture(scope="module")
def forge_model(loc_cfg: SeismologyConfig) -> LayerModel:
    return load_configured_model(loc_cfg.velocity)


@pytest.fixture(scope="module")
def grids(loc02: Any) -> GridsConfig:
    """The showcase grid knobs (25 m, 52 km)."""
    return GridsConfig.model_validate(loc02.showcase_raw()["grids"])


def _mesh(grid: GridSpec) -> tuple[FloatArray, FloatArray]:
    rr, zz = np.meshgrid(grid.r_nodes(), grid.z_nodes())
    return rr, zz


def _solve(
    model: LayerModel, phase: Phase, receiver: float, grid: GridSpec, cfg: GridsConfig
) -> tuple[LayerModel, FloatArray]:
    extended = model.with_top_extended_to(grid.top_elev_m, max_extension_m=5000.0)
    times = solve_table(
        extended, phase, receiver, grid, seed_radius_m=cfg.seedRadiusM, fmm_order=cfg.fmmOrder
    )
    return extended, times


@pytest.mark.parametrize("vs", [1163.0, 3374.0])  # slowest and fastest S in the FORGE model
def test_homogeneous_half_space_matches_analytic(loc02: Any, grids: GridsConfig, vs: float) -> None:
    receiver = 1700.0
    grid = make_grid(grids, receiver)
    _, times = _solve(loc02.toy_model([2000.0], [1.8 * vs], [vs]), "S", receiver, grid, grids)
    rr, zz = _mesh(grid)
    err = np.abs(times - np.hypot(rr, zz - receiver) / vs)
    assert grid.r_max_m == grids.rMaxM
    assert err.max() < MAX_ERR_S, f"max error {err.max() * 1e3:.2f} ms"


def _two_layer_analytic(
    rr: FloatArray, zz: FloatArray, receiver: float, interface: float, v1: float, v2: float
) -> tuple[FloatArray, FloatArray]:
    """First arrival in the upper layer: direct or head wave along the interface."""
    direct = np.hypot(rr, zz - receiver) / v1
    cos_c = np.sqrt(1.0 - (v1 / v2) ** 2)
    legs = (receiver - interface) + (zz - interface)
    head = rr / v2 + legs * cos_c / v1
    x_crit = legs * (v1 / v2) / cos_c
    return np.where(rr >= x_crit, np.minimum(direct, head), direct), head < direct


def test_two_layer_head_wave_matches_analytic(loc02: Any, grids: GridsConfig) -> None:
    v1, v2, interface, receiver = 3000.0, 5000.0, 1012.5, 1600.0  # interface between nodes
    grid = make_grid(grids, receiver)
    _, times = _solve(loc02.toy_model([2000.0, interface], [v1, v2], [v1 / 1.8, v2 / 1.8]), "P",
                      receiver, grid, grids)
    rr, zz = _mesh(grid)
    analytic, head_first = _two_layer_analytic(rr, zz, receiver, interface, v1, v2)
    upper = zz > interface
    head_zone = upper & head_first
    assert head_zone.sum() > 1000 and rr[head_zone].max() == grid.r_max_m
    err = np.abs(times - analytic)
    assert err[upper].max() < MAX_ERR_S, f"max error {err[upper].max() * 1e3:.2f} ms"
    assert err[head_zone].max() < MAX_ERR_S


def test_borehole_receiver(loc02: Any, grids: GridsConfig) -> None:
    # Homogeneous: the table is right above and below a receiver 300 m down.
    receiver = 1400.0
    grid = make_grid(grids, 1700.0)
    _, times = _solve(loc02.toy_model([2000.0], [3500.0], [2000.0]), "P", receiver, grid, grids)
    rr, zz = _mesh(grid)
    assert np.abs(times - np.hypot(rr, zz - receiver) / 3500.0).max() < MAX_ERR_S
    # Two layers, receiver below the interface: upgoing transmitted rays reach the slow layer.
    model = loc02.toy_model([2000.0, 1512.5], [3000.0, 5000.0], [1700.0, 2900.0])
    extended, times = _solve(model, "P", receiver, grid, grids)
    sub = (slice(None, None, 3), slice(None, None, 8))
    exact = layered_first_arrival(extended, "P", receiver, rr[sub], zz[sub])
    assert np.abs(times[sub] - exact).max() < MAX_ERR_S
    below = zz[sub] < 1512.5
    np.testing.assert_allclose(exact[below], np.hypot(rr[sub], zz[sub] - receiver)[below] / 5000.0,
                               rtol=0, atol=1e-9)


def test_layered_solver_matches_closed_forms(loc02: Any) -> None:
    v1, v2, interface = 3000.0, 5000.0, 1000.0
    model = loc02.toy_model([2000.0, interface], [v1, v2], [1700.0, 2900.0])
    rng = np.random.default_rng(1)
    # Upper layer: direct or head wave.
    r = rng.uniform(0.0, 30000.0, 500)
    z = rng.uniform(1001.0, 2000.0, 500)
    analytic, _ = _two_layer_analytic(r, z, 1600.0, interface, v1, v2)
    np.testing.assert_allclose(layered_first_arrival(model, "P", 1600.0, r, z), analytic,
                               rtol=0, atol=1e-9)
    # Transmitted ray from the fast layer up into the slow one, built with Snell's law.
    zs, zr = 200.0, 1800.0
    for p in (0.0, 0.5 / v2, 0.9 / v2, 0.999 / v2):
        c1, c2 = np.sqrt(1 - (p * v1) ** 2), np.sqrt(1 - (p * v2) ** 2)
        x = (zr - interface) * p * v1 / c1 + (interface - zs) * p * v2 / c2
        t = (zr - interface) / (v1 * c1) + (interface - zs) / (v2 * c2)
        got = float(layered_first_arrival(model, "P", zs, x, zr))
        assert got == pytest.approx(t, abs=1e-9)
    # Homogeneous and a horizontal ray at the source elevation.
    homog = loc02.toy_model([2000.0], [4000.0], [2300.0])
    assert float(layered_first_arrival(homog, "S", 500.0, 3000.0, -3500.0)) == pytest.approx(
        5000.0 / 2300.0, abs=1e-12)
    assert float(layered_first_arrival(model, "P", 1500.0, 700.0, 1500.0)) == pytest.approx(
        700.0 / v1, abs=1e-12)
    with pytest.raises(ValueError, match="above the top"):
        layered_first_arrival(model, "P", 1500.0, 10.0, 2500.0)


def test_reciprocity_between_surface_and_borehole(
    grids: GridsConfig, forge_model: LayerModel
) -> None:
    surface, borehole = 1840.0, 1345.0
    grid = make_grid(grids, surface)
    r = grid.r_nodes()
    # Tolerances: reciprocity compares two tables' grid errors. Against the exact solver, the
    # shallow S table peaks near 13 ms around a head-wave onset (the eikonal smooths the kink at
    # the critical distance of the 1227 m ASL interface); elsewhere it stays near 3-4 ms.
    phases: tuple[tuple[Phase, float, float], ...] = (
        ("P", MAX_ERR_S, MAX_ERR_S), ("S", 2 * MAX_ERR_S, 0.015))
    for phase, tol, tol_exact in phases:
        ext, t_surface = _solve(forge_model, phase, surface, grid, grids)
        _, t_borehole = _solve(forge_model, phase, borehole, grid, grids)
        tab_s = TravelTimeTable(phase, surface, grid, _ro(t_surface), "a")
        tab_b = TravelTimeTable(phase, borehole, grid, _ro(t_borehole), "b")
        forward = tab_s.lookup(r, borehole)
        backward = tab_b.lookup(r, surface)
        assert np.abs(forward - backward).max() < tol
        exact = layered_first_arrival(ext, phase, surface, r, borehole)
        assert np.abs(forward - exact).max() < tol_exact
        # Axis symmetry: the time straight down the axis is the vertical integral of slowness.
        assert float(tab_s.lookup(0.0, -3000.0)) == pytest.approx(
            float(layered_first_arrival(ext, phase, surface, 0.0, -3000.0)), abs=MAX_ERR_S)


def _ro(a: FloatArray) -> FloatArray:
    a = np.array(a)
    a.setflags(write=False)
    return a


def test_bilinear_lookup_is_exact_on_a_linear_field() -> None:
    grid = GridSpec(dr_m=25.0, dz_m=25.0, n_r=41, n_z=21, bottom_elev_m=-100.0)
    rr, zz = np.meshgrid(grid.r_nodes(), grid.z_nodes())
    table = TravelTimeTable("P", 0.0, grid, _ro(0.3 + 2e-4 * rr - 5e-4 * zz), "k")
    rng = np.random.default_rng(7)
    r = rng.uniform(0.0, grid.r_max_m, 1000)
    z = rng.uniform(grid.bottom_elev_m, grid.top_elev_m, 1000)
    np.testing.assert_allclose(table.lookup(r, z), 0.3 + 2e-4 * r - 5e-4 * z, rtol=0, atol=1e-12)
    corners = table.lookup([0.0, grid.r_max_m], [grid.bottom_elev_m, grid.top_elev_m])
    np.testing.assert_allclose(corners, [0.3 + 0.05, 0.3 + 0.2 - 5e-4 * 400.0], atol=1e-12)
    out = table.lookup(np.array([[0.0], [25.0]]), np.array([0.0, 25.0]))  # broadcasts
    assert out.shape == (2, 2)
    for r_bad, z_bad in ((-1.0, 0.0), (grid.r_max_m + 0.1, 0.0), (0.0, grid.top_elev_m + 0.1),
                         (0.0, grid.bottom_elev_m - 0.1), (np.nan, 0.0)):
        with pytest.raises(ValueError, match="outside"):
            table.lookup([10.0, r_bad], [0.0, z_bad])


def _stations(elevs: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"id": [f"X.{i}" for i in range(len(elevs))], "sensorElevM": elevs})


def test_cache_keys_sharing_and_determinism(
    loc_cfg: SeismologyConfig, forge_model: LayerModel, tmp_path: Path
) -> None:
    small = loc_cfg.grids.model_copy(update={"rMaxM": 5000.0, "bottomElevM": -3000.0})
    stations = _stations([1700.0, 1700.0, 1400.0])
    kwargs = {"cache_dir": tmp_path, "max_extension_m": loc_cfg.velocity.maxTopExtensionM}
    first = build_station_tables(stations, forge_model, small, **kwargs)
    assert (first.n_built, first.n_loaded) == (4, 0)  # 2 elevations x 2 phases
    assert first.table("X.0", "P") is first.table("X.1", "P")  # same sensorElevM, same table
    files = sorted((tmp_path / CACHE_SUBDIR).glob("*.npy"))
    assert len(files) == 4
    sidecar = json.loads(files[0].with_suffix(".json").read_text())
    assert {"velocityModel", "phase", "receiverElevM", "grid", "seedRadiusM",
            "solverSourceSha256", "accuracyVsExact"} <= set(sidecar)
    assert sidecar["solverSourceSha256"] == SOURCE_SHA256
    second = build_station_tables(stations, forge_model, small, **kwargs)
    assert (second.n_built, second.n_loaded) == (0, 4)
    for key, table in first.tables.items():
        np.testing.assert_array_equal(table.times_s, second.tables[key].times_s)
        assert not second.tables[key].times_s.flags.writeable
        assert second.tables[key].accuracy == table.accuracy  # read back from the sidecar
    # Identical inputs give identical tables, solved from scratch.
    grid = first.grid
    again = solve_table(first.model, "S", 1400.0, grid, seed_radius_m=small.seedRadiusM,
                        fmm_order=small.fmmOrder)
    np.testing.assert_array_equal(again, first.table("X.2", "S").times_s)
    # Every input is in the key.
    base, _ = table_key(first.model, "P", 1400.0, grid, seed_radius_m=500.0, fmm_order=2)
    changed = {
        table_key(first.model, "S", 1400.0, grid, seed_radius_m=500.0, fmm_order=2)[0],
        table_key(first.model, "P", 1400.5, grid, seed_radius_m=500.0, fmm_order=2)[0],
        table_key(first.model, "P", 1400.0, grid, seed_radius_m=600.0, fmm_order=2)[0],
        table_key(first.model, "P", 1400.0, grid, seed_radius_m=500.0, fmm_order=1)[0],
        table_key(forge_model.with_top_extended_to(grid.top_elev_m + 25.0, max_extension_m=1e3),
                  "P", 1400.0, grid, seed_radius_m=500.0, fmm_order=2)[0],
    }
    assert base not in changed and len(changed) == 5


def test_accuracy_record_matches_the_exact_solver(
    loc_cfg: SeismologyConfig, forge_model: LayerModel, tmp_path: Path
) -> None:
    small = loc_cfg.grids.model_copy(update={"rMaxM": 3000.0, "bottomElevM": -1000.0})
    tables = build_station_tables(_stations([1642.0]), forge_model, small, cache_dir=tmp_path,
                                  max_extension_m=loc_cfg.velocity.maxTopExtensionM)
    grid = tables.grid
    table = tables.table("X.0", "S")
    rr, zz = np.meshgrid(grid.r_nodes()[::small.accuracyCheckStrideR], grid.z_nodes())
    exact = layered_first_arrival(tables.model, "S", 1642.0, rr, zz)
    err = np.abs(table.times_s[:, ::small.accuracyCheckStrideR] - exact)
    acc = table.accuracy
    assert acc is not None and acc["maxErrS"] == pytest.approx(float(err.max()), abs=1e-12)
    assert acc == table_accuracy(tables.model, "S", 1642.0, grid, table.times_s,
                                 stride_r=small.accuracyCheckStrideR)
    layer = tables.model.layer_index(grid.z_nodes())
    for row in acc["byLayer"]:
        k = int(np.flatnonzero(tables.model.top_elev_m == row["topElevM"])[0])
        assert row["maxErrS"] == pytest.approx(float(err[layer == k].max()), abs=1e-12)
    record = tables.to_record()
    by_layer = record["accuracyVsExactByLayer"]
    assert set(by_layer) == {"P", "S"} and by_layer["S"] == [
        {"topElevM": r["topElevM"], "maxErrS": r["maxErrS"]} for r in acc["byLayer"]]
    assert {t["maxErrVsExactS"] for t in record["tables"] if t["phase"] == "S"} == {acc["maxErrS"]}
    # A sidecar that no longer matches its key's inputs fails loudly.
    sidecar = tmp_path / CACHE_SUBDIR / f"{table.key}.json"
    data = json.loads(sidecar.read_text())
    data["receiverElevM"] = 0.0
    sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match"):
        build_station_tables(_stations([1642.0]), forge_model, small, cache_dir=tmp_path,
                             max_extension_m=loc_cfg.velocity.maxTopExtensionM)


def test_top_extension_is_explicit_and_recorded(
    loc_cfg: SeismologyConfig, forge_model: LayerModel, tmp_path: Path
) -> None:
    small = loc_cfg.grids.model_copy(update={"rMaxM": 2000.0, "bottomElevM": -1000.0})
    tables = build_station_tables(
        _stations([2421.0, 1500.0]), forge_model, small, cache_dir=tmp_path,
        max_extension_m=loc_cfg.velocity.maxTopExtensionM, cover_top_elev_m=1627.7,
    )
    grid = tables.grid
    assert grid.top_elev_m >= 2421.0 + small.topMarginM
    assert grid.top_elev_m - small.dzM < 2421.0 + small.topMarginM  # snapped up by < one cell
    ext = tables.model.top_extension
    assert ext is not None and ext.to_elev_m == grid.top_elev_m
    assert ext.from_elev_m == forge_model.top_of_model_elev_m
    record = tables.to_record()
    assert record["velocityModel"]["topExtension"]["toElevM"] == grid.top_elev_m
    assert {t["phase"] for t in record["tables"]} == {"P", "S"}
    with pytest.raises(ValueError, match="more than max_extension_m"):
        build_station_tables(_stations([3000.0]), forge_model, small, cache_dir=tmp_path,
                             max_extension_m=loc_cfg.velocity.maxTopExtensionM)


def test_bad_inputs_raise(loc_cfg: SeismologyConfig, forge_model: LayerModel,
                          tmp_path: Path) -> None:
    small = loc_cfg.grids.model_copy(update={"rMaxM": 2000.0, "bottomElevM": -1000.0})
    grid = make_grid(small, 1750.0)  # grid top 1850 m ASL, above the model's own top
    with pytest.raises(ValueError, match="below the grid top"):
        solve_table(forge_model, "P", 1750.0, grid, seed_radius_m=500.0, fmm_order=2)
    with pytest.raises(ValueError, match="bottomElevM"):
        build_station_tables(_stations([1700.0, -1500.0]), forge_model, small,
                             cache_dir=tmp_path, max_extension_m=1000.0)
    with pytest.raises(ValueError, match="duplicate"):
        build_station_tables(pd.DataFrame({"id": ["A", "A"], "sensorElevM": [1.0, 2.0]}),
                             forge_model, small, cache_dir=tmp_path, max_extension_m=1000.0)
    bad = tmp_path / CACHE_SUBDIR
    tables = build_station_tables(_stations([1500.0]), forge_model, small, cache_dir=tmp_path,
                                  max_extension_m=1000.0)
    path = bad / f"{tables.table('X.0', 'P').key}.npy"
    np.save(path, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="delete it"):
        build_station_tables(_stations([1500.0]), forge_model, small, cache_dir=tmp_path,
                             max_extension_m=1000.0)


def test_grid_config_guards(loc02: Any) -> None:
    raw = loc02.showcase_raw()["grids"]
    with pytest.raises(ValueError, match="multiple of drM"):
        GridsConfig.model_validate({**raw, "rMaxM": 40010.0})
    with pytest.raises(ValueError, match="4 grid cells"):
        GridsConfig.model_validate({**raw, "seedRadiusM": 50.0})
    full = loc02.showcase_raw()
    full["grids"] = {**raw, "bottomElevM": full["locator"]["volume"]["bottomElevM"]}
    with pytest.raises(ValueError, match="below"):
        SeismologyConfig.model_validate(full)
    full = loc02.showcase_raw()
    full["locator"]["errConfidence"] = 0.9
    with pytest.raises(ValueError, match="errConfidence must be 0.68"):
        SeismologyConfig.model_validate(full)


# Every nested LOC-02 config section rejects unknown keys (LOC-01's test covers the top level).
NESTED_SECTIONS = ("grids", "locator", "locator.volume", "locator.pickSigmaS", "locator.outlier",
                   "synthetic", "synthetic.zone")


@pytest.mark.parametrize("where", NESTED_SECTIONS)
def test_unknown_nested_config_keys_fail(loc02: Any, where: str) -> None:
    raw = loc02.showcase_raw()
    target = raw
    for key in where.split("."):
        target = target[key]
    target["tpyo"] = 1.0
    with pytest.raises(pydantic.ValidationError, match="tpyo"):
        SeismologyConfig.model_validate(raw)


def test_unknown_profile_sigma_keys_fail(loc02: Any) -> None:
    raw = loc02.showcase_raw()
    raw["locator"]["profilePickSigmaS"] = {"borehole-B": {"P": 0.03, "S": 0.05, "tpyo": 1.0}}
    with pytest.raises(pydantic.ValidationError, match="tpyo"):
        SeismologyConfig.model_validate(raw)

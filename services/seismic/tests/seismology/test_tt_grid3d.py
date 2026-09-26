"""LOC-07: 3D travel-time tables (hq.locate.tt_grid3d) and the grid3d locator path.

Toy 3D models are built inside the tests on a small lattice around the showcase origin (tests may
build small synthetic data). The real GDR 1800 file is never read here.
"""

import copy
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydantic
import pytest

from hq.config.seismology import Grid3dConfig, SeismologyConfig
from hq.locate.coords import origin_utm
from hq.locate.locator import GRID3D, Locator, LocatorSetup, build_locator, make_volume
from hq.locate.synthetic import run_synthetic
from hq.locate.tt_grid import build_station_tables, layered_first_arrival
from hq.locate.tt_grid3d import (
    Grid3dSpec,
    Model3dSource,
    Table3d,
    VolumeBox,
    air_mask,
    build_station_tables3d,
    column_model,
    handle_air,
)
from hq.locate.velocity import LayerModel

MODEL_SPACING_M = 100.0
HALF_WIDTH_M = 7000.0  # toy model: +/- this around the origin, e and n
MODEL_BOTTOM_M = -5600.0
MODEL_TOP_M = 2500.0
# Layer tops on 100 m model cell edges (cells span node +/- 50 m), so a model column equals the
# layer model exactly.
TOPS = [2550.0, 1250.0, 450.0, -550.0]
VP = [2300.0, 3500.0, 5200.0, 5800.0]
VS = [1200.0, 2000.0, 3000.0, 3392.0]


def grid3d_config(loc02: Any, **grid3d: object) -> SeismologyConfig:
    """The LOC-02 test config (5 km volume) switched to grid3d on the toy model's range."""
    raw = loc02.test_config().model_dump(mode="json")
    raw["locator"]["method"] = GRID3D
    raw["grid3d"] = {**raw["grid3d"], "horizontalMarginM": 1000.0, "bottomElevM": -5400.0,
                     **grid3d}
    raw["locator"]["volume"]["bottomElevM"] = -5000.0
    return SeismologyConfig.model_validate(raw)


def toy_model(run: Any, vp_of: Any, vs_of: Any, name: str = "toy3d") -> Model3dSource:
    """A 3D model on the 100 m toy lattice; ``vp_of(e, n, z)`` / ``vs_of`` in ENU m and elevM."""
    x0, y0 = origin_utm(run.origin)
    ax = np.arange(-HALF_WIDTH_M, HALF_WIDTH_M + 1.0, MODEL_SPACING_M)
    z = np.arange(MODEL_BOTTOM_M, MODEL_TOP_M + 1.0, MODEL_SPACING_M)
    nn, ee, zz = np.meshgrid(ax, ax, z, indexing="ij")
    return Model3dSource.from_arrays(name, ax + x0, ax + y0, z, vp_of(ee, nn, zz),
                                     vs_of(ee, nn, zz))


def layered(values: list[float]) -> Any:
    tops = np.array(TOPS)

    def of(e: np.ndarray, n: np.ndarray, z: np.ndarray) -> np.ndarray:
        # layer i spans tops[i+1] < z <= tops[i]
        return np.array(values)[np.sum(z[..., None] <= tops[None, 1:], axis=-1)]

    return of


def few_stations(loc02: Any, run: Any, ids: tuple[str, ...]) -> pd.DataFrame:
    st = loc02.stations(run.origin.elevM)
    return st[st["id"].isin(ids)].reset_index(drop=True)


def build3d(loc02: Any, cfg: SeismologyConfig, run: Any, source: Model3dSource,
            stations: pd.DataFrame, cache: Path) -> Any:
    volume = make_volume(cfg.locator, run.refSurfaceElevM).box()
    return build_station_tables3d(
        stations, source, cfg.grid3d, origin_utm=origin_utm(run.origin), volume=volume,
        cache_dir=cache, vp_range_m_per_s=cfg.velocity.plausibleVpMPerS,
        vs_range_m_per_s=cfg.velocity.plausibleVsMPerS, column_grid=cfg.grids,
    )


def lattice_points(tables3d: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = tables3d.grid
    return np.meshgrid(g.z_nodes(), g.n_nodes(), g.e_nodes(), indexing="ij")


# --- smoke: lookups, air, config --------------------------------------------------------------


def _linear_table() -> tuple[Table3d, Any]:
    grid = Grid3dSpec(spacing_m=200.0, e0_m=-600.0, n0_m=-400.0, z0_m=-1000.0, n_e=6, n_n=5,
                      n_z=4, origin_utm_e_m=0.0, origin_utm_n_m=0.0)

    def f(e: Any, n: Any, z: Any) -> Any:
        return 0.3 + 2e-4 * e - 1.5e-4 * n + 3e-4 * z

    zz, nn, ee = np.meshgrid(grid.z_nodes(), grid.n_nodes(), grid.e_nodes(), indexing="ij")
    times = np.ascontiguousarray(f(ee, nn, zz))
    times.setflags(write=False)
    return Table3d("T.X", "P", (0.0, 0.0, 0.0), grid, times, "k"), f


@pytest.mark.smoke
def test_trilinear_lookup_is_exact_on_a_linear_field() -> None:
    table, f = _linear_table()
    rng = np.random.default_rng(7)
    e, n, z = rng.uniform(-600, 400, 50), rng.uniform(-400, 400, 50), rng.uniform(-1000, -400, 50)
    np.testing.assert_allclose(table.lookup(e, n, z), f(e, n, z), rtol=0, atol=1e-12)
    ea, na, za = np.sort(e[:7]), np.sort(n[:5]), np.sort(z[:4])
    box = table.lookup_box(ea, na, za)
    zz, nn, ee = np.meshgrid(za, na, ea, indexing="ij")
    assert box.shape == (4, 5, 7)
    np.testing.assert_allclose(box, table.lookup(ee, nn, zz), rtol=0, atol=1e-12)
    np.testing.assert_allclose(box, f(ee, nn, zz), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="outside the 3D table"):
        table.lookup(401.0, 0.0, -500.0)
    with pytest.raises(ValueError, match="elevM outside"):
        table.lookup_box(np.array([0.0]), np.array([0.0]), np.array([-1001.0]))


@pytest.mark.smoke
def test_air_is_the_constant_top_run_of_non_constant_columns() -> None:
    top_vp, top_vs = 732.71, 400.02
    basin_vp = np.array([5800.0, 3000.0, 1500.0, top_vp, top_vp, top_vp])  # bottom to top
    basin_vs = np.array([3392.0, 1600.0, 700.0, top_vs, top_vs, top_vs])
    vp = np.stack([basin_vp, np.full(6, 5800.0)])[None, :, :]  # (1 northing, 2 easting, 6 elev)
    vs = np.stack([basin_vs, np.full(6, 3392.0)])[None, :, :]
    air_values = (top_vp, top_vs)
    air = air_mask(vp, top_vp)
    assert air[0, 0].tolist() == [False, False, False, True, True, True]
    assert not air[0, 1].any()  # a constant column carries no ground surface
    fvp, fvs, counts = handle_air(vp, vs, "topSurfaceVelocity", air_values)
    assert fvp[0, 0].tolist() == [5800.0, 3000.0, 1500.0, 1500.0, 1500.0, 1500.0]
    assert fvs[0, 0, 3:].tolist() == [700.0] * 3
    np.testing.assert_array_equal(fvp[0, 1], vp[0, 1])
    assert counts["airNodes"] == 3 and counts["constantColumns"] == 1
    same_vp, _, _ = handle_air(vp, vs, "asFile", air_values)
    np.testing.assert_array_equal(same_vp, vp)
    # A genuine constant top layer is not air: only the model's air value is.
    no_air, _, counts = handle_air(vp, vs, "topSurfaceVelocity", (999.0, 500.0))
    np.testing.assert_array_equal(no_air, vp)
    assert counts["airNodes"] == 0
    bad_vs = vs.copy()
    bad_vs[0, 0, 5] = 401.0
    with pytest.raises(ValueError, match="Vs does not hold"):
        handle_air(vp, bad_vs, "topSurfaceVelocity", air_values)


@pytest.mark.smoke
def test_grid3d_config_validation(loc02: Any) -> None:
    base = loc02.showcase_raw()
    cfg = SeismologyConfig.model_validate(base)
    assert cfg.locator.method == "grid1d"  # the showcase default until the lead decides
    assert cfg.velocity.model3d.crs == "EPSG:32612" and cfg.velocity.model3d.units == "km/s"
    g3 = base["grid3d"]
    with pytest.raises(pydantic.ValidationError, match="4 grid cells"):
        Grid3dConfig.model_validate({**g3, "seedRadiusM": 3.0 * g3["spacingM"]})
    with pytest.raises(pydantic.ValidationError):
        Grid3dConfig.model_validate({**g3, "airHandling": "mask"})
    bad_cases: list[tuple[tuple[str, ...], object, str]] = [
        (("locator", "method"), "grid2d", "method"),
        (("grid3d", "bottomElevM"), base["locator"]["volume"]["bottomElevM"] + 100.0,
         "must not lie above"),
        (("grid3d", "spacingM"), 0.0, "spacingM"),
        (("velocity", "model3d", "crs"), "EPSG:26912", "crs"),
        (("velocity", "model3d", "units"), "ft/s", "units"),
        (("velocity", "model3d", "airVp"), 0.0, "airVp"),
    ]
    for path, value, match in bad_cases:
        bad = copy.deepcopy(base)
        node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(pydantic.ValidationError, match=match):
            SeismologyConfig.model_validate(bad)
    bad = copy.deepcopy(base)
    del bad["grid3d"]
    with pytest.raises(pydantic.ValidationError, match="grid3d"):
        SeismologyConfig.model_validate(bad)


# --- non-smoke: solver accuracy, fallback, cache, locator and synthetic test ---------------------


def test_homogeneous_3d_model_matches_analytic(loc02: Any, tmp_path: Path) -> None:
    run = loc02.run_section()
    cfg = grid3d_config(loc02)
    vp, vs = 4000.0, 2300.0
    source = toy_model(run, lambda e, n, z: np.full(e.shape, vp), lambda e, n, z: np.full(e.shape, vs))
    stations = few_stations(loc02, run, ("T.S01", "T.B02"))  # a surface and a borehole receiver
    t3 = build3d(loc02, cfg, run, source, stations, tmp_path)
    zz, nn, ee = lattice_points(t3)
    for sid, e0, n0, z0 in stations[["id", "enu_e", "enu_n", "sensorElevM"]].itertuples(index=False):
        for ph, v in (("P", vp), ("S", vs)):
            exact = np.sqrt((ee - e0) ** 2 + (nn - n0) ** 2 + (zz - z0) ** 2) / v
            err = np.abs(t3.table(sid, ph).times_s - exact)
            assert err.max() < 0.005, f"{sid} {ph}: max error {err.max() * 1e3:.2f} ms"


def test_layered_3d_model_equals_the_1d_tables(loc02: Any, tmp_path: Path) -> None:
    run = loc02.run_section()
    cfg = grid3d_config(loc02)
    source = toy_model(run, layered(VP), layered(VS))
    stations = few_stations(loc02, run, ("T.S04", "T.B01"))
    t3 = build3d(loc02, cfg, run, source, stations, tmp_path)
    layer_model = loc02.toy_model(TOPS, VP, VS)
    t1 = build_station_tables(stations, layer_model, cfg.grids, cache_dir=tmp_path,
                              max_extension_m=cfg.velocity.maxTopExtensionM)
    rng = np.random.default_rng(3)
    e = rng.uniform(-5000, 5000, 400)
    n = rng.uniform(-5000, 5000, 400)
    z = rng.uniform(-4500, -500, 400)  # the source depths the locator searches
    zz, nn, ee = lattice_points(t3)
    for sid, e0, n0, z0 in stations[["id", "enu_e", "enu_n", "sensorElevM"]].itertuples(index=False):
        # On the lattice nodes the 3D table is the 1D table: the 3D solve adds only the
        # 3D-minus-column difference, zero for a laterally uniform model up to float rounding.
        rr = np.hypot(ee - e0, nn - n0)
        for ph in ("P", "S"):
            np.testing.assert_allclose(t3.table(sid, ph).times_s,
                                       t1.table(sid, ph).lookup(rr, zz), rtol=0, atol=1e-9)
        # Between nodes, trilinear interpolation over 200 m cells: measured within ~8 ms (S) of
        # the 25 m 1D table and of the exact layered times at source depths (within a few cells
        # of the receiver it reaches ~30 ms; no hypocentre sits there).
        r = np.hypot(e - e0, n - n0)
        for ph in ("P", "S"):
            got = t3.table(sid, ph).lookup(e, n, z)
            np.testing.assert_allclose(got, t1.table(sid, ph).lookup(r, z), rtol=0, atol=0.012)
            exact = layered_first_arrival(t1.model, ph, z0, r, z)
            assert np.abs(got - exact).max() < 0.012


def test_tilted_gradient_3d_model_matches_analytic(loc02: Any, tmp_path: Path) -> None:
    """A laterally varying model with a closed-form first arrival: v = v0 + g . x."""
    run = loc02.run_section()
    cfg = grid3d_config(loc02)
    grad = np.array([0.06, -0.04, -0.35])  # 1/s along e, n, elevM: faster with depth and to the E
    v_ref = 3000.0

    def vel(e: Any, n: Any, z: Any) -> Any:
        return v_ref + grad[0] * e + grad[1] * n + grad[2] * (z - 1000.0)

    source = toy_model(run, vel, lambda e, n, z: vel(e, n, z) / 1.75)
    stations = few_stations(loc02, run, ("T.S03",))
    t3 = build3d(loc02, cfg, run, source, stations, tmp_path)
    sid, e0, n0, z0 = next(stations[["id", "enu_e", "enu_n", "sensorElevM"]].itertuples(index=False))
    g = float(np.linalg.norm(grad))
    rng = np.random.default_rng(11)
    e = rng.uniform(-4500, 4500, 500)
    n = rng.uniform(-4500, 4500, 500)
    z = rng.uniform(-4500, -500, 500)  # the source depths the locator searches
    d2 = (e - e0) ** 2 + (n - n0) ** 2 + (z - z0) ** 2
    exact = np.arccosh(1.0 + g * g * d2 / (2.0 * vel(e0, n0, z0) * vel(e, n, z))) / g
    err = t3.table(sid, "P").lookup(e, n, z) - exact
    # 200 m lattice, seed radius 1000 m: the column seed ignores the lateral gradient within
    # the seed radius; measured for this receiver median +1.5 ms, max 9 ms (P); 100 m: < 2 ms.
    assert abs(float(np.median(err))) < 0.005 and float(np.abs(err).max()) < 0.02, (
        f"median {np.median(err) * 1e3:+.1f} ms, max {np.abs(err).max() * 1e3:.1f} ms")


def test_station_outside_the_model_uses_its_1d_tables(loc02: Any, tmp_path: Path) -> None:
    run = loc02.run_section()
    cfg = grid3d_config(loc02)
    source = toy_model(run, layered(VP), layered(VS))
    setup = dataclasses.replace(loc02.setup(cfg), cache_dir=tmp_path, model3d=source)
    far = setup.stations.iloc[[0]].assign(id="T.FAR", enu_e=HALF_WIDTH_M + 1000.0, enu_n=0.0)
    stations = pd.concat([setup.stations, far], ignore_index=True)
    locator = build_locator(dataclasses.replace(setup, stations=stations))
    assert locator.method == GRID3D and locator.tables3d is not None
    t3 = locator.tables3d
    assert t3.fallback == {"T.FAR": "outside the model's horizontal extent"}
    assert set(t3.stations_3d) == set(setup.stations["id"])
    rec = locator.to_record()
    assert rec["method"] == GRID3D and rec["tables3d"]["fallback1d"] == t3.fallback
    assert locator.velocity_model_record()["fallback1d"]["stations"] == t3.fallback
    for e, n, z in ((0.0, 0.0, -2000.0), (1500.0, -800.0, -3500.0)):
        r = float(np.hypot(e - far["enu_e"].iloc[0], n))
        expected = float(locator.tables.table("T.FAR", "S").lookup(r, z))
        assert float(locator.station_times("T.FAR", "S", e, n, z)) == pytest.approx(expected,
                                                                                    abs=1e-12)
        assert float(locator.station_times("T.S01", "S", e, n, z)) == float(
            t3.table("T.S01", "S").lookup(e, n, z))


def test_cache_is_reused_and_keyed_by_the_model(loc02: Any, tmp_path: Path) -> None:
    run = loc02.run_section()
    cfg = grid3d_config(loc02)
    stations = few_stations(loc02, run, ("T.S01",))
    source = toy_model(run, layered(VP), layered(VS))
    first = build3d(loc02, cfg, run, source, stations, tmp_path)
    again = build3d(loc02, cfg, run, source, stations, tmp_path)
    assert (first.n_built, again.n_built, again.n_loaded) == (2, 0, 2)
    np.testing.assert_array_equal(first.table("T.S01", "P").times_s,
                                  again.table("T.S01", "P").times_s)
    assert again.resampling["airHandling"] == cfg.grid3d.airHandling
    faster = toy_model(run, layered([v * 1.1 for v in VP]), layered(VS), name="toy3d")
    other = build3d(loc02, cfg, run, faster, stations, tmp_path)
    assert other.n_built == 2 and other.table("T.S01", "P").key != first.table("T.S01", "P").key


def test_grid3d_locator_and_synthetic_test(loc02: Any, tmp_path: Path) -> None:
    run = loc02.run_section()
    cfg = grid3d_config(loc02)
    raw = cfg.model_dump(mode="json")
    raw["synthetic"] = {**raw["synthetic"], "nEvents": 6, "sKeepProb": 0.6, "pickProb": 1.0}
    cfg = SeismologyConfig.model_validate(raw)

    def vel(e: Any, n: Any, z: Any) -> Any:  # layered plus a lateral east-west contrast
        return layered(VP)(e, n, z) * (1.0 + 0.08 * np.tanh(e / 2000.0))

    source = toy_model(run, vel, lambda e, n, z: vel(e, n, z) / 1.8)
    setup = LocatorSetup(stations=loc02.stations(run.origin.elevM), model=loc02.toy_model(
        TOPS, VP, VS), config=cfg, run=run, cache_dir=tmp_path, model3d=source)
    result = run_synthetic(setup, geometry_label="TEST-3D")
    assert result.params["method"] == GRID3D and "3D tables" in result.params["forwardModel"]
    clean = result.params["noiseFree"]
    # Forward model and locator share the tables: noise-free events come back on the fine lattice.
    assert clean["medianHErrM"] < 30.0 and clean["medianVErrM"] < 30.0
    assert result.report.nEvents == 6
    locator: Locator = build_locator(setup)
    picks = loc02.exact_picks(locator, 0.0, 0.0, -2000.0, 1.0e9)
    exact_3d = [float(locator.station_times(r.stationId, r.phase, 0.0, 0.0, -2000.0))
                for r in picks.itertuples(index=False)]
    loc = locator.locate(picks.assign(t=1.0e9 + np.array(exact_3d)))
    assert loc.quality()["method"] == GRID3D
    assert abs(loc.elev_m + 2000.0) <= 25.0 and np.hypot(loc.e_m, loc.n_m) <= 25.0


def test_volume_outside_the_3d_model_fails_loudly(loc02: Any, tmp_path: Path) -> None:
    run = loc02.run_section()
    cfg = grid3d_config(loc02)
    source = toy_model(run, layered(VP), layered(VS))
    stations = few_stations(loc02, run, ("T.S01",))
    big = VolumeBox(-HALF_WIDTH_M - 1000.0, 0.0, -1000.0, 1000.0, -3000.0, 1000.0)
    with pytest.raises(ValueError, match="reaches outside"):
        build_station_tables3d(
            stations, source, cfg.grid3d, origin_utm=origin_utm(run.origin), volume=big,
            cache_dir=tmp_path, vp_range_m_per_s=cfg.velocity.plausibleVpMPerS,
            vs_range_m_per_s=cfg.velocity.plausibleVsMPerS, column_grid=cfg.grids)
    slow = toy_model(run, layered([v / 10.0 for v in VP]), layered([v / 10.0 for v in VS]))
    with pytest.raises(ValueError, match="unit guard"):
        build3d(loc02, cfg, run, slow, stations, tmp_path)


def test_toy_layer_model_is_the_column(loc02: Any) -> None:
    """The layered toy model's columns are exactly the 1D layer model (test premise)."""
    run = loc02.run_section()
    source = toy_model(run, layered(VP), layered(VS))
    vp, vs = source.load()
    col = column_model(source, vp[10, 10], vs[10, 10], "c")
    ref: LayerModel = loc02.toy_model(TOPS, VP, VS)
    np.testing.assert_array_equal(col.vp_m_per_s, ref.vp_m_per_s)
    np.testing.assert_array_equal(col.top_elev_m[1:], ref.top_elev_m[1:])

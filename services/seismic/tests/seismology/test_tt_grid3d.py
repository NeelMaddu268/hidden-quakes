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
    CONSTANT_COLUMN_REASON,
    Grid3dSpec,
    Model3dSource,
    Table3d,
    VolumeBox,
    air_mask,
    build_station_tables3d,
    column_model,
    ground_elev_m,
    handle_air,
    open_model3d,
    resample,
    station_columns,
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
    assert cfg.grid3d.constantColumns == "fallback1d"
    for spacing in (50.0, 250.0):  # the ticket's range is 100-200 m
        with pytest.raises(pydantic.ValidationError, match="spacingM"):
            Grid3dConfig.model_validate({**g3, "spacingM": spacing, "seedRadiusM": 4000.0})
    with pytest.raises(pydantic.ValidationError, match="constantColumns"):
        Grid3dConfig.model_validate({**g3, "constantColumns": "mask"})


@pytest.mark.smoke
def test_locate_without_cache_dir_finds_the_data_dir_for_grid3d(
    seismology_config: SeismologyConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/02 locate() has no cache_dir: grid3d takes the data dir's cache (the 3D model lives
    there), grid1d keeps its per-process temporary cache."""
    from hq.locate import default_cache_dir

    monkeypatch.setenv("HQ_DATA_DIR", str(tmp_path))
    cfg1 = seismology_config
    cfg3 = cfg1.model_copy(update={"locator": cfg1.locator.model_copy(update={"method": GRID3D})})
    assert default_cache_dir(cfg3) == tmp_path.resolve() / "cache"
    temp = default_cache_dir(cfg1)
    assert temp != tmp_path.resolve() / "cache" and temp.name.startswith("hq-locate-ttgrids-")


# --- non-smoke: solver accuracy, fallback, cache, locator and synthetic test ---------------------


def test_homogeneous_3d_model_matches_analytic(loc02: Any, tmp_path: Path) -> None:
    run = loc02.run_section()
    cfg = grid3d_config(loc02, constantColumns="asFile")  # every column holds one value
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


def test_stage_locate_with_grid3d_writes_the_1d_vs_3d_section(
    loc02: Any, make_ctx: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage locate with grid3d relocates the same association with grid1d in the same run and
    diagnostics.md compares the two (no earlier 1D run in the run dir needed)."""
    import importlib

    from hq_contracts.io import read_table, write_table

    from hq.associate.result import EVENT_DTYPES, PICK_DTYPES, typed_frame
    from hq.locate.coords import from_enu

    run = loc02.run_section()
    raw = grid3d_config(loc02).model_dump(mode="json")
    raw["diagnostics"]["datumCheck"]["elevM"] = [-2000.0]
    raw["synthetic"]["nEvents"] = 2
    cfg3 = SeismologyConfig.model_validate(raw)
    raw["locator"]["method"] = "grid1d"
    cfg1 = SeismologyConfig.model_validate(raw)

    def vel(e: Any, n: Any, z: Any) -> Any:  # the 1D layers plus an east-west contrast
        return layered(VP)(e, n, z) * (1.0 + 0.08 * np.tanh(e / 2000.0))

    source = toy_model(run, vel, lambda e, n, z: vel(e, n, z) / 1.8)
    monkeypatch.setattr("hq.locate.locator.open_model3d", lambda path, cfg: source)
    st = loc02.stations(run.origin.elevM)
    lat, lon, _ = from_enu(st["enu_e"], st["enu_n"], st["enu_u"], run.origin)
    stations = st.assign(latitude=lat, longitude=lon, usedInRun=True)
    locator = build_locator(dataclasses.replace(loc02.setup(cfg3, run), cache_dir=tmp_path,
                                                model3d=source))
    frames, events, links = [], [], []
    for k, (e, n, z) in enumerate(((500.0, -300.0, -2000.0), (-1200.0, 900.0, -3200.0))):
        t0 = run.window_start_s + 3600.0 * (k + 1)
        p = loc02.exact_picks(locator, e, n, z, t0, prob=0.8)
        p["t"] = [t0 + float(locator.station_times(s, ph, e, n, z))
                  for s, ph in zip(p["stationId"], p["phase"], strict=True)]
        p["id"] = [f"pick:{k}:{s}:{ph}" for s, ph in zip(p["stationId"], p["phase"], strict=True)]
        frames.append(p)
        la, lo, _ = from_enu(e, n, z - run.origin.elevM, run.origin)
        events.append({"assocId": f"a{k}", "t": t0, "latitude": float(la), "longitude": float(lo),
                       "elevM": z, "nPicks": len(p), "nP": int((p["phase"] == "P").sum()),
                       "nS": int((p["phase"] == "S").sum())})
        links += [{"assocId": f"a{k}", "pickId": i} for i in p["id"]]
    picks = pd.concat(frames, ignore_index=True).assign(picker="phasenet:test", eventId=None)
    ctx = dataclasses.replace(make_ctx(run, cfg1), cache_dir=tmp_path)
    write_table(picks, ctx.path(cfg1.associator.picksTable), "Pick")
    write_table(stations, ctx.path("stations.parquet"), "Station")
    write_table(typed_frame({c: [r[c] for r in events] for c in EVENT_DTYPES}, EVENT_DTYPES),
                ctx.path("assoc_events.parquet"), "AssocEvent")
    write_table(typed_frame({c: [r[c] for r in links] for c in PICK_DTYPES}, PICK_DTYPES),
                ctx.path("assoc_picks.parquet"), "AssocPick")
    stage = importlib.import_module("hq.locate.run").run
    ctx3 = dataclasses.replace(ctx, config=dataclasses.replace(ctx.config, seismology=cfg3))
    stage(ctx3)
    ev = read_table(ctx.path("events_located.parquet"))
    assert set(ev["quality_method"]) == {GRID3D}
    report = ctx.path("diagnostics.md").read_text(encoding="utf-8")
    assert "## 1D vs 3D travel times (LOC-07)" in report
    assert ("Against grid1d locations of the same association, made in this stage run with the "
            "same statics configuration (2 events") in report
    assert "method `grid3d`" in report and "3D minus 1D table time" in report
    assert "from the 3D tables themselves (grid3d)" in report  # row 2
    assert "the locator's own 3D tables" in report  # the synthetic section's forward model
    assert "Events above the 3D model's ground" in report
    assert "nothing here asks for 3D grids" not in report
    velocity, params = ctx3.records[-2]["params"], ctx3.records[-1]["params"]
    assert velocity["method"] == GRID3D and velocity["fallback1d"]["stations"] == {}
    assert params["method"] == GRID3D and params["tables3d"]["stations3d"]
    assert params["synthetic"]["method"] == GRID3D
    assert params["grid1dComparisonRuntimeS"] > 0
    # The picks are exact 3D times: grid3d puts the events back on the fine lattice.
    truth = {"a0": (500.0, -300.0, -2000.0), "a1": (-1200.0, 900.0, -3200.0)}
    flags = read_table(ctx.path("locate_flags.parquet"))
    for r in ev.merge(flags[["eventId", "assocId"]], left_on="id", right_on="eventId").itertuples():
        e, n, z = truth[r.assocId]
        assert abs(r.elevM - z) <= 25.0 and np.hypot(r.enu_e - e, r.enu_n - n) <= 25.0


# --- the NetCDF path (a tiny file written in the test) ------------------------------------------

NC_SPACING_M = 50.0  # the GDR 1800 file's node spacing
NC_E0_M, NC_N0_M = -1030.0, -1020.0  # ENU of the first node: off the 100 m lattice, as for real
NC_NODES = 41
NC_ELEV = np.arange(-1000.0, 1000.0 + 1.0, NC_SPACING_M)
NC_AIR = (0.73271, 0.40002)  # km/s, the file's above-ground value
NC_BASIN_E_M = -200.0  # basin columns east of this ENU easting, one-value columns west of it
NC_GROUND_M = 575.0  # basin columns: ground between the 550 m and 600 m nodes


def _nc_values() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(e, n, Vp, Vs): Vp/Vs in km/s, indexed (northing, easting, elevation)."""
    e = NC_E0_M + NC_SPACING_M * np.arange(NC_NODES)
    n = NC_N0_M + NC_SPACING_M * np.arange(NC_NODES)
    _, ee, zz = np.meshgrid(n, e, NC_ELEV, indexing="ij")
    basin = ee > NC_BASIN_E_M
    vp = np.full(ee.shape, 5.8)
    vs = np.full(ee.shape, 3.392)
    sediment = basin & (zz >= 0.0) & (zz < NC_GROUND_M)
    vp[sediment], vs[sediment] = 2.5, 1.2
    air = basin & (zz > NC_GROUND_M)
    vp[air], vs[air] = NC_AIR
    vp[-1, -1, :] = vs[-1, -1, :] = np.nan  # one column without values (the file's NaN strip)
    return e, n, vp, vs


def _write_nc(path: Path, run: Any, *, dims_ok: bool = True) -> None:
    import xarray as xr

    x0, y0 = origin_utm(run.origin)
    e, n, vp, vs = _nc_values()
    order = ("elevation", "easting", "northing")  # not the (northing, easting, elevation) we use
    data = {name: (order, np.transpose(v, (2, 1, 0))) for name, v in (("Vp", vp), ("Vs", vs))}
    if not dims_ok:
        data["Vs"] = (("elevation", "easting", "x"), np.transpose(vs, (2, 1, 0)))
    xr.Dataset(data, coords={"elevation": NC_ELEV, "easting": e + x0, "northing": n + y0}
               ).to_netcdf(path)


def _nc_setup(loc02: Any, tmp_path: Path) -> tuple[Any, SeismologyConfig, Model3dSource, Any]:
    run = loc02.run_section()
    base = loc02.test_config()
    path = tmp_path / "toy-model.nc"
    _write_nc(path, run)
    m3 = base.velocity.model3d.model_copy(update={
        "cacheFile": path.name, "expectedBytes": path.stat().st_size, "units": "km/s",
        "airVp": NC_AIR[0], "airVs": NC_AIR[1]})
    return run, base, open_model3d(path, m3), m3


def _nc_grid3d(constant_columns: str) -> Grid3dConfig:
    return Grid3dConfig(spacingM=100.0, horizontalMarginM=0.0, topMarginM=100.0,
                        bottomElevM=-900.0, airHandling="topSurfaceVelocity",
                        constantColumns=constant_columns, seedRadiusM=400.0, fmmOrder=2)


def _nc_stations() -> pd.DataFrame:
    rows = [("N.BAS", 300.0, 0.0, 500.0),  # basin column, sensor below its ground
            ("N.CON", -600.0, 100.0, 700.0),  # one-value column
            ("N.NAN", NC_E0_M + 40 * NC_SPACING_M, NC_N0_M + 40 * NC_SPACING_M, 0.0),
            ("N.EDGE", -1020.0, 0.0, 0.0),  # inside the model, west of the 100 m lattice's edge
            ("N.OUT", 2000.0, 0.0, 0.0)]  # outside the model
    return pd.DataFrame(rows, columns=["id", "enu_e", "enu_n", "sensorElevM"])


def test_model_file_units_dims_and_columns(loc02: Any, tmp_path: Path) -> None:
    run, _, source, m3 = _nc_setup(loc02, tmp_path)
    _, _, vp_kms, vs_kms = _nc_values()
    vp, vs = source.load()  # (northing, easting, elevation), m/s
    np.testing.assert_array_equal(vp, vp_kms * 1000.0)
    np.testing.assert_array_equal(vs, vs_kms * 1000.0)
    col_vp, col_vs = source.column(3, 30)
    np.testing.assert_array_equal(col_vs, vs[3, 30])
    np.testing.assert_array_equal(col_vp, vp[3, 30])
    many_vp, _ = source.columns([3, 7], [30, 2])
    np.testing.assert_array_equal(many_vp, vp[[3, 7], [30, 2]])
    assert source.units == "km/s" and source.air_m_per_s == pytest.approx(
        (NC_AIR[0] * 1000.0, NC_AIR[1] * 1000.0))
    np.testing.assert_allclose(ground_elev_m(vp[[3, 3], [30, 2]], source), [NC_GROUND_M, np.nan])
    with pytest.raises(ValueError, match="partial download"):
        open_model3d(Path(str(source.path)), m3.model_copy(update={"expectedBytes": 1}))
    with pytest.raises(FileNotFoundError, match="missing"):
        open_model3d(tmp_path / "absent.nc", m3)
    bad = tmp_path / "bad-dims.nc"
    _write_nc(bad, run, dims_ok=False)
    with pytest.raises(ValueError, match="expected variable Vs"):
        open_model3d(bad, m3.model_copy(update={"expectedBytes": bad.stat().st_size}))
    kinds = station_columns(source, _nc_stations(), origin_utm(run.origin))
    assert kinds["N.BAS"].kind == "basin" and kinds["N.BAS"].in_model
    assert (kinds["N.BAS"].ground_low_elev_m, kinds["N.BAS"].ground_high_elev_m) == (550.0, 600.0)
    assert kinds["N.CON"].kind == "constant" and kinds["N.CON"].in_model
    assert not kinds["N.NAN"].in_model and "no (or partial) values" in str(kinds["N.NAN"].reason)
    assert kinds["N.OUT"].reason == "outside the model's horizontal extent"


def test_resampling_weights_model_cells_by_overlap_on_a_misaligned_lattice(
    loc02: Any, tmp_path: Path
) -> None:
    run, _, source, _ = _nc_setup(loc02, tmp_path)
    vp, _ = source.load()
    grid = Grid3dSpec(spacing_m=100.0, e0_m=-900.0, n0_m=-900.0, z0_m=-900.0, n_e=17, n_n=17,
                      n_z=17, origin_utm_e_m=origin_utm(run.origin)[0],
                      origin_utm_n_m=origin_utm(run.origin)[1])
    speed = resample(vp, source, grid)
    e_cells = NC_E0_M + NC_SPACING_M * np.arange(NC_NODES)  # model node ENU
    n_cells = NC_N0_M + NC_SPACING_M * np.arange(NC_NODES)

    def overlap(x: float, cells: np.ndarray) -> np.ndarray:
        return np.clip(np.minimum(x + 50.0, cells + 25.0) - np.maximum(x - 50.0, cells - 25.0), 0,
                       None)

    for k, j, i in ((13, 9, 7), (14, 3, 8), (9, 12, 9), (15, 5, 12)):  # across the basin edge
        w = (overlap(grid.n_nodes()[j], n_cells)[:, None, None]
             * overlap(grid.e_nodes()[i], e_cells)[None, :, None]
             * overlap(grid.z_nodes()[k], NC_ELEV)[None, None, :])
        finite = np.isfinite(vp)
        expected = w[finite].sum() / (w[finite] / vp[finite]).sum()
        assert speed[k, j, i] == pytest.approx(expected, rel=1e-12)


def test_constant_columns_knob_and_model_edge_fallback(loc02: Any, tmp_path: Path) -> None:
    import types

    from hq.locate.statics import explain_terms, facts_grid3d

    run, base, source, _ = _nc_setup(loc02, tmp_path)
    stations = _nc_stations()
    volume = VolumeBox(-500.0, 500.0, -500.0, 500.0, -800.0, 400.0)
    kw = {"origin_utm": origin_utm(run.origin), "volume": volume, "cache_dir": tmp_path,
          "vp_range_m_per_s": base.velocity.plausibleVpMPerS,
          "vs_range_m_per_s": base.velocity.plausibleVsMPerS, "column_grid": base.grids}
    fall = build_station_tables3d(stations, source, _nc_grid3d("fallback1d"), **kw)
    as_file = build_station_tables3d(stations, source, _nc_grid3d("asFile"), **kw)
    assert fall.grid == as_file.grid  # the knob never moves the lattice
    assert fall.stations_3d == ("N.BAS",) and as_file.stations_3d == ("N.BAS", "N.CON")
    assert fall.fallback["N.CON"] == CONSTANT_COLUMN_REASON
    for t3 in (fall, as_file):
        assert t3.fallback["N.EDGE"] == "outside the 3D lattice (model edge)"
        assert set(t3.fallback) >= {"N.NAN", "N.OUT", "N.EDGE"}
    assert (as_file.n_built, as_file.n_loaded) == (2, 2)  # N.BAS's tables came from the cache
    np.testing.assert_array_equal(fall.table("N.BAS", "S").times_s,
                                  as_file.table("N.BAS", "S").times_s)
    assert fall.to_record()["constantColumns"] == "fallback1d"
    # Static explanations against the 3D model use the station's own 3D column.
    layer = loc02.toy_model(TOPS, VP, VS)
    st = stations.iloc[:2].assign(sensorElevM=[500.0, 700.0])
    facts = facts_grid3d(types.SimpleNamespace(tables3d=as_file,  # type: ignore[arg-type]
                                               tables=types.SimpleNamespace(model=layer)), st)
    assert facts["N.BAS"].vpvs == pytest.approx(2.5 / 1.2)
    assert facts["N.CON"].vpvs == pytest.approx(5.8 / 3.392)
    assert "between 550 and 600 m ASL" in facts["N.BAS"].elevation
    assert "one velocity at every elevation" in facts["N.CON"].elevation
    terms = pd.DataFrame([{"stationId": s, "phase": ph, "staticS": v, "rawS": v, "nEvents": 10,
                           "madS": 0.01} for s, p, sv in (("N.BAS", 0.2, 0.6), ("N.CON", 0.0, 0.0))
                          for ph, v in (("P", p), ("S", sv))])
    ex = explain_terms(terms, st, layer, (0.0, 0.0), 0.15, 3, base.statics.explain, facts)
    text = ex.set_index(["stationId", "phase"]).loc[("N.BAS", "S"), "explanation"]
    assert "against the 3D model" in text and "Vp/Vs 2.08 at the sensor" in text
    assert "extended upward" not in text
    fb = facts_grid3d(types.SimpleNamespace(tables3d=fall,  # type: ignore[arg-type]
                                            tables=types.SimpleNamespace(model=layer)), st)
    assert fb["N.CON"].against.startswith("its 1D tables")

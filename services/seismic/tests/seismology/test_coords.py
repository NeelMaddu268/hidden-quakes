"""hq.locate.coords: lat/lon/elevM <-> ENU (UTM 12N minus origin; u = elevM - origin.elevM)."""

import math

import numpy as np
import pytest
from pyproj import Geod

from hq.config.run import RunSection
from hq.locate.coords import from_enu, to_enu

pytestmark = pytest.mark.smoke

# UTM zone 12N definition, used only to derive independent expectations below.
UTM12_CENTRAL_MERIDIAN_DEG = -111.0
UTM_K0 = 0.9996


def test_origin_maps_to_zero(run_section: RunSection) -> None:
    o = run_section.origin
    e, n, u = to_enu(o.lat, o.lon, o.elevM, o)
    assert (float(e), float(n), float(u)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


def test_one_km_true_north_is_one_km_grid_north_rotated_by_convergence(
    run_section: RunSection,
) -> None:
    """1 km along the true meridian: |ENU| = k * 1000 m, rotated by the UTM grid convergence.

    e is about 20 m, not 0, because ENU axes are UTM grid axes (docs/01), and the grid is rotated
    by gamma = dLon * sin(lat) (about -1.18 degrees here) relative to true north.
    """
    o = run_section.origin
    lon2, lat2, _ = Geod(ellps="WGS84").fwd(o.lon, o.lat, 0.0, 1000.0)
    e, n, u = (float(v) for v in to_enu(lat2, lon2, o.elevM + 250.0, o))

    assert n == pytest.approx(1000.0, abs=1.0)
    assert abs(e) < 30.0
    assert u == pytest.approx(250.0, abs=1e-9)

    dlon = math.radians(o.lon - UTM12_CENTRAL_MERIDIAN_DEG)
    lat = math.radians(o.lat)
    gamma_deg = math.degrees(dlon * math.sin(lat))  # first-order grid convergence
    scale = UTM_K0 * (1.0 + (dlon * math.cos(lat)) ** 2 / 2.0)  # first-order point scale
    assert math.degrees(math.atan2(e, n)) == pytest.approx(-gamma_deg, abs=0.01)
    assert math.hypot(e, n) == pytest.approx(1000.0 * scale, abs=0.1)


def test_round_trip_is_exact_and_vectorized(run_section: RunSection) -> None:
    rng = np.random.default_rng(12)
    min_lon, min_lat, max_lon, max_lat = run_section.bbox
    lat = rng.uniform(min_lat, max_lat, 200)
    lon = rng.uniform(min_lon, max_lon, 200)
    elev = rng.uniform(-6000.0, 3000.0, 200)
    o = run_section.origin

    e, n, u = to_enu(lat, lon, elev, o)
    assert e.shape == n.shape == u.shape == (200,)
    np.testing.assert_allclose(u, elev - o.elevM, atol=1e-9)

    lat2, lon2, elev2 = from_enu(e, n, u, o)
    np.testing.assert_allclose(lat2, lat, atol=1e-9)
    np.testing.assert_allclose(lon2, lon, atol=1e-9)
    np.testing.assert_allclose(elev2, elev, atol=1e-9)


def test_non_finite_input_fails_loudly(run_section: RunSection) -> None:
    o = run_section.origin
    with pytest.raises(ValueError, match="lat"):
        to_enu([o.lat, float("nan")], [o.lon, o.lon], [0.0, 0.0], o)
    with pytest.raises(ValueError, match="u"):
        from_enu([0.0], [0.0], [float("inf")], o)

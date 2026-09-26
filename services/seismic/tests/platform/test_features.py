"""FEAT-01 acceptance: the showcase ``features:`` section loads through the real config loader,
``load_features`` turns it into contract ``GeoFeature`` rows that validate, every row carries a
citation and URL, nothing read through a copy claims to be verified, the minimum-curvature
trajectory is right on hand-made surveys, and a survey that contradicts its own printed positions
is refused. Offline: the real config CSVs plus tiny surveys built here."""

import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hq_contracts.models import GeoFeature
from pydantic import ValidationError

from hq.config import lax_models, load_config
from hq.config.export import ExportConfig, WellFeatureConfig
from hq.config.run import RunSection
from hq.export.features import (
    ENU_DECIMALS,
    M_PER_UNIT,
    load_features,
    minimum_curvature,
    read_survey_csv,
    resolve_survey_path,
)
from hq.locate.coords import to_enu

pytestmark = pytest.mark.smoke

SHOWCASE_DIR = Path(__file__).resolve().parents[2] / "configs" / "showcase"


@pytest.fixture(scope="module")
def run() -> RunSection:
    return load_config(SHOWCASE_DIR).run


@pytest.fixture(scope="module")
def export() -> ExportConfig:
    return load_config(SHOWCASE_DIR).export


@pytest.fixture(scope="module")
def features(export: ExportConfig, run: RunSection) -> list[GeoFeature]:
    return load_features(export, run)


def source(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "citation": "Test survey listing, 2026.",
        "url": "https://example.org/survey",
        "accessedOn": "2026-09-26",
    }
    return {**base, **overrides}


def well(**overrides: Any) -> dict[str, Any]:
    """A vertical two-station well 1 km east of the showcase origin, in UTM 12N metres."""
    base: dict[str, Any] = {
        "kind": "well",
        "id": "well-test",
        "name": "Test well",
        "verified": False,
        "crs": "EPSG:32612",
        "unit": "m",
        "headX": 335_000.0,
        "headY": 4_263_000.0,
        "depthRefElev": 1700.0,
        "northRef": "grid",
        "maxSurveyMismatchM": 0.5,
        "survey": [
            {"md": 0.0, "incDeg": 0.0, "aziDeg": 0.0},
            {"md": 1000.0, "incDeg": 0.0, "aziDeg": 0.0},
        ],
        "source": source(),
    }
    return {**base, **overrides}


# --- showcase config -----------------------------------------------------------------------


def test_showcase_config_has_typed_cited_features(export: ExportConfig) -> None:
    assert export.features, "export.yaml ships no reference features"
    assert lax_models(ExportConfig) == []  # every nested model rejects unknown keys
    for feature in export.features:
        assert feature.source.citation.strip()
        assert feature.source.url.startswith("http")
        if feature.source.readFrom is not None:
            assert feature.verified is False, feature.id
    ids = [f.id for f in export.features]
    assert len(set(ids)) == len(ids)


def test_showcase_features_validate_with_citation_and_url(
    features: list[GeoFeature], export: ExportConfig
) -> None:
    assert [f.id for f in features] == [f.id for f in export.features]
    for feature, cfg in zip(features, export.features, strict=True):
        GeoFeature.model_validate(feature.model_dump())  # round-trips through the contract
        assert feature.source.citation.strip()
        assert feature.source.url == cfg.source.url
        assert feature.source.verified is cfg.verified
        if not cfg.verified:
            assert "approximate" in feature.source.citation
        assert feature.kind == cfg.kind
        assert feature.path
        for point in feature.path:
            assert all(math.isfinite(v) for v in (point.e, point.n, point.u))
            assert round(point.e, ENU_DECIMALS) == point.e
            # inside the run's ~52 x 51 km region: no feature far from the scene
            assert abs(point.e) < 40_000 and abs(point.n) < 40_000


def test_showcase_wells_are_trajectories(features: list[GeoFeature], run: RunSection) -> None:
    wells = [f for f in features if f.kind == "well"]
    assert wells
    for feature in wells:
        assert len(feature.path) > 2
        u = np.array([p.u for p in feature.path])
        assert u[-1] < u[0]  # the toe is below the wellhead: elevM up-positive
        assert np.all(np.diff(u) <= 0.0)  # depth only increases along a survey
        assert u[0] + run.origin.elevM > run.refSurfaceElevM - 500.0  # head near the surface


def test_showcase_boundaries_are_closed_rings(features: list[GeoFeature], run: RunSection) -> None:
    rings = [f for f in features if f.kind == "boundary"]
    assert rings
    for feature in rings:
        assert feature.path[0] == feature.path[-1]
        assert len(feature.path) >= 4
        assert all(p.u == feature.path[0].u for p in feature.path)


def test_showcase_points_are_single(features: list[GeoFeature]) -> None:
    for feature in features:
        if feature.kind == "facility":
            assert len(feature.path) == 1


def test_showcase_survey_csvs_carry_their_source(export: ExportConfig) -> None:
    wells = [f for f in export.features if isinstance(f, WellFeatureConfig)]
    assert wells
    for feature in wells:
        assert feature.surveyCsv is not None
        path = resolve_survey_path(feature.surveyCsv)
        assert path.is_relative_to(SHOWCASE_DIR / "features")
        header = [line for line in path.read_text().splitlines() if line.startswith("#")]
        assert any("source:" in line and "http" in line for line in header)
        rows = read_survey_csv(path)
        assert rows[0].md == 0.0 and rows[0].tvd == 0.0
        assert all(r.tvd is not None for r in rows)


def test_load_features_is_deterministic(export: ExportConfig, run: RunSection) -> None:
    first = [f.model_dump() for f in load_features(export, run)]
    second = [f.model_dump() for f in load_features(export, run)]
    assert first == second


# --- minimum curvature ---------------------------------------------------------------------


def test_minimum_curvature_vertical_and_straight_holes() -> None:
    md = np.array([0.0, 100.0, 250.0])
    pos = minimum_curvature(md, np.zeros(3), np.zeros(3))
    assert np.allclose(pos.tvd, md) and np.allclose(pos.ns, 0.0) and np.allclose(pos.ew, 0.0)

    # straight hole at 30 deg inclination heading due east (dogleg zero, rf = 1)
    pos = minimum_curvature(md, np.full(3, 30.0), np.full(3, 90.0))
    assert np.allclose(pos.tvd, md * math.cos(math.radians(30.0)))
    assert np.allclose(pos.ew, md * math.sin(math.radians(30.0)))
    assert np.allclose(pos.ns, 0.0, atol=1e-9)


def test_minimum_curvature_quarter_circle_is_exact() -> None:
    # A build from vertical to horizontal along a circular arc of radius R heading north is
    # exactly what minimum curvature assumes, so the end sits at (N=R, TVD=R).
    radius = 500.0
    md = np.array([0.0, math.pi / 2.0 * radius])
    pos = minimum_curvature(md, np.array([0.0, 90.0]), np.array([0.0, 0.0]))
    assert pos.ns[-1] == pytest.approx(radius)
    assert pos.tvd[-1] == pytest.approx(radius)
    assert pos.ew[-1] == pytest.approx(0.0, abs=1e-9)


def test_minimum_curvature_rejects_bad_md() -> None:
    with pytest.raises(ValueError, match="start at md 0"):
        minimum_curvature(np.array([10.0, 20.0]), np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="increase strictly"):
        minimum_curvature(np.array([0.0, 20.0, 20.0]), np.zeros(3), np.zeros(3))


# --- conversions ---------------------------------------------------------------------------


def test_well_path_lands_on_the_wellhead_in_enu(run: RunSection) -> None:
    cfg = ExportConfig(features=[well()])
    [feature] = load_features(cfg, run)
    assert len(feature.path) == 2
    head, toe = feature.path
    assert head.u == pytest.approx(1700.0 - run.origin.elevM, abs=10**-ENU_DECIMALS)
    assert toe.u == pytest.approx(700.0 - run.origin.elevM, abs=10**-ENU_DECIMALS)
    assert toe.e == head.e and toe.n == head.n  # vertical hole
    # the head is where H2's helper puts the wellhead lat/lon (EPSG:32612 is the scene CRS)
    from pyproj import Transformer

    lon, lat = Transformer.from_crs("EPSG:32612", "EPSG:4326", always_xy=True).transform(
        335_000.0, 4_263_000.0
    )
    e, n, _ = to_enu(lat, lon, 1700.0, run.origin)
    assert head.e == pytest.approx(float(e), abs=10**-ENU_DECIMALS)
    assert head.n == pytest.approx(float(n), abs=10**-ENU_DECIMALS)


def test_us_survey_feet_convert_exactly(run: RunSection) -> None:
    usft = M_PER_UNIT["usft"]
    assert usft == 1200.0 / 3937.0
    metres = ExportConfig(features=[well()])
    feet = ExportConfig(
        features=[
            well(
                unit="usft",
                headX=335_000.0 / usft,
                headY=4_263_000.0 / usft,
                depthRefElev=1700.0 / usft,
                survey=[
                    {"md": 0.0, "incDeg": 0.0, "aziDeg": 0.0},
                    {"md": 1000.0 / usft, "incDeg": 0.0, "aziDeg": 0.0},
                ],
            )
        ]
    )
    [a] = load_features(metres, run)
    [b] = load_features(feet, run)
    assert [p.model_dump() for p in a.path] == [p.model_dump() for p in b.path]


def test_published_positions_must_match_minimum_curvature(run: RunSection) -> None:
    good = well(
        survey=[
            {"md": 0.0, "incDeg": 0.0, "aziDeg": 0.0, "tvd": 0.0, "ns": 0.0, "ew": 0.0},
            {"md": 1000.0, "incDeg": 0.0, "aziDeg": 0.0, "tvd": 1000.0, "ns": 0.0, "ew": 0.0},
        ]
    )
    assert len(load_features(ExportConfig(features=[good]), run)) == 1
    bad = well(
        survey=[
            {"md": 0.0, "incDeg": 0.0, "aziDeg": 0.0, "tvd": 0.0, "ns": 0.0, "ew": 0.0},
            {"md": 1000.0, "incDeg": 0.0, "aziDeg": 0.0, "tvd": 1000.0, "ns": 5.0, "ew": 0.0},
        ]
    )
    with pytest.raises(ValueError, match="differs from the minimum-curvature position"):
        load_features(ExportConfig(features=[bad]), run)


def test_survey_csv_reader_rejects_bad_files(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_text("# source: https://example.org\nmd,incDeg,aziDeg\n0,0,0\n100,1,45\n")
    rows = read_survey_csv(path)
    assert [r.md for r in rows] == [0.0, 100.0] and rows[1].tvd is None
    path.write_text("md,inc\n0,0\n")
    with pytest.raises(ValueError, match="columns must be"):
        read_survey_csv(path)
    path.write_text("md,incDeg,aziDeg\n0,0,north\n")
    with pytest.raises(ValueError, match="row 2"):
        read_survey_csv(path)


def test_point_and_polygon_features(run: RunSection) -> None:
    cfg = ExportConfig(
        features=[
            {
                "kind": "facility",
                "id": "pad-test",
                "name": "Test pad",
                "verified": False,
                "crs": "EPSG:4326",
                "unit": "deg",
                "x": run.origin.lon,
                "y": run.origin.lat,
                "elevM": run.origin.elevM + 10.0,
                "source": source(),
            },
            {
                "kind": "boundary",
                "id": "ring-test",
                "name": "Test ring",
                "verified": False,
                "crs": "EPSG:26912",
                "unit": "m",
                "vertices": [[334_000, 4_262_000], [336_000, 4_262_000], [335_000, 4_264_000]],
                "source": source(),
            },
        ]
    )
    pad, ring = load_features(cfg, run)
    assert pad.path == [pad.path[0]] and pad.path[0].e == 0.0 and pad.path[0].n == 0.0
    assert pad.path[0].u == pytest.approx(10.0)
    assert len(ring.path) == 4 and ring.path[0] == ring.path[-1]
    assert all(p.u == pytest.approx(run.refSurfaceElevM - run.origin.elevM) for p in ring.path)


# --- validation -----------------------------------------------------------------------------


def test_verified_feature_without_source_fails() -> None:
    feature = well(verified=True)
    del feature["source"]
    with pytest.raises(ValidationError, match="source"):
        ExportConfig(features=[feature])


def test_verified_feature_read_from_a_copy_fails() -> None:
    feature = well(verified=True, source=source(readFrom="https://mirror.example.org/copy"))
    with pytest.raises(ValidationError, match="read from source.url itself"):
        ExportConfig(features=[feature])
    ExportConfig(features=[well(verified=True)])  # read from the primary: allowed


def test_source_needs_citation_url_and_access_date() -> None:
    for broken in (source(citation=""), source(url="gdr.openei.org/x"), source(accessedOn=None)):
        with pytest.raises(ValidationError):
            ExportConfig(features=[well(source=broken)])


def test_unknown_keys_and_inconsistent_units_fail() -> None:
    with pytest.raises(ValidationError, match="extra"):
        ExportConfig(features=[well(datum="NAD83")])
    with pytest.raises(ValidationError, match="unit 'deg' goes with crs 'EPSG:4326'"):
        ExportConfig(features=[well(unit="deg")])
    with pytest.raises(ValidationError, match="needs a projected crs"):
        ExportConfig(features=[well(crs="EPSG:4326", unit="deg")])
    with pytest.raises(ValidationError, match="exactly one of survey and surveyCsv"):
        ExportConfig(features=[well(surveyCsv="x.csv")])
    with pytest.raises(ValidationError, match="unique"):
        ExportConfig(features=[well(), well()])
    with pytest.raises(ValidationError, match="published together"):
        ExportConfig(
            features=[
                well(
                    survey=[
                        {"md": 0.0, "incDeg": 0.0, "aziDeg": 0.0, "tvd": 0.0},
                        {"md": 10.0, "incDeg": 0.0, "aziDeg": 0.0},
                    ]
                )
            ]
        )
    assert date(2026, 9, 26) == ExportConfig(features=[well()]).features[0].source.accessedOn

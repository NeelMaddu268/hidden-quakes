"""Reference features for the bundle: ``configs/showcase/export.yaml`` ``features:`` ->
``GeoFeature`` rows in ENU (FEAT-01, H4).

Wells, well pads and site outlines are context for the scene, nothing more: every feature carries
the ``SourceRef`` its coordinates were read from, and ``verified`` is true only when they were
read from the authoritative publication itself. Nothing here attributes seismicity to anything.

Coordinates stay in their published datum and unit in the config. This module converts them with
pyproj to WGS84 latitude/longitude, then to ENU with H2's ``hq.locate.coords.to_enu`` (UTM 12N
minus the run origin; ``u = elevM - origin.elevM``, docs/01 -> Conventions). Well trajectories are
rebuilt from the survey (md, inclination, azimuth) with the minimum-curvature method and, when
the report also printed positions, checked against them.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from hq_contracts.models import Enu, GeoFeature, SourceRef
from pyproj import Transformer

import hq
from hq.config.export import (
    ExportConfig,
    FeatureCrs,
    LengthUnit,
    PointFeatureConfig,
    PolygonFeatureConfig,
    SurveyRow,
    WellFeatureConfig,
)
from hq.config.run import RunSection
from hq.locate.coords import GEOGRAPHIC_CRS, to_enu

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

# Unit definitions, not knobs. The US survey foot is exactly 1200/3937 m.
M_PER_UNIT: dict[str, float] = {"m": 1.0, "usft": 1200.0 / 3937.0}

# ENU output precision: metres to the centimetre. Survey reports print positions to 0.01 ft,
# so nothing finer is information; it keeps features.json compact and byte-stable.
ENU_DECIMALS = 2

# Directory that relative ``surveyCsv`` paths resolve against: ``services/seismic``, where
# ``configs/`` lives (the same root ``hq run configs/showcase`` is run from).
SEISMIC_ROOT = Path(hq.__file__).resolve().parents[1]

SURVEY_COLUMNS = ("md", "incDeg", "aziDeg")
SURVEY_PUBLISHED_COLUMNS = ("tvd", "ns", "ew")


@dataclass(frozen=True)
class SurveyPositions:
    """Minimum-curvature positions per survey station, in the survey's length unit."""

    ns: FloatArray  # displacement north of the wellhead
    ew: FloatArray  # displacement east of the wellhead
    tvd: FloatArray  # true vertical depth below the depth reference


def minimum_curvature(md: FloatArray, inc_deg: FloatArray, azi_deg: FloatArray) -> SurveyPositions:
    """Positions along a directional survey by the minimum-curvature method.

    Each interval between consecutive stations is a circular arc fitted to the two measured
    tangents; the dogleg ``dl`` sets the ratio factor ``rf = tan(dl/2) / (dl/2)`` (1 when the
    interval is straight). ``md`` must start at 0 and increase strictly. Angles in degrees,
    azimuth clockwise from north; lengths come back in the unit of ``md``.
    """
    md = np.asarray(md, dtype=np.float64)
    inc = np.deg2rad(np.asarray(inc_deg, dtype=np.float64))
    azi = np.deg2rad(np.asarray(azi_deg, dtype=np.float64))
    if md.ndim != 1 or md.shape != inc.shape or md.shape != azi.shape or md.size < 2:
        raise ValueError("md, inc_deg and azi_deg must be equal-length 1-d arrays of >= 2 rows")
    if md[0] != 0.0:
        raise ValueError(f"survey must start at md 0 (the depth reference), got {md[0]}")
    if np.any(np.diff(md) <= 0.0):
        raise ValueError("survey md must increase strictly")
    i1, i2 = inc[:-1], inc[1:]
    a1, a2 = azi[:-1], azi[1:]
    dmd = np.diff(md)
    cos_dl = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1))
    dl = np.arccos(np.clip(cos_dl, -1.0, 1.0))
    half = dl / 2.0
    rf = np.where(dl > 0.0, np.tan(half) / np.where(dl > 0.0, half, 1.0), 1.0)
    d_ns = dmd / 2.0 * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
    d_ew = dmd / 2.0 * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
    d_tvd = dmd / 2.0 * (np.cos(i1) + np.cos(i2)) * rf
    zero = np.zeros(1)
    return SurveyPositions(
        ns=np.concatenate([zero, np.cumsum(d_ns)]),
        ew=np.concatenate([zero, np.cumsum(d_ew)]),
        tvd=np.concatenate([zero, np.cumsum(d_tvd)]),
    )


def read_survey_csv(path: Path) -> list[SurveyRow]:
    """Survey stations from a CSV with header ``md,incDeg,aziDeg[,tvd,ns,ew]``.

    Lines starting with ``#`` are comments (the config CSVs open with their source), blank lines
    are skipped, and every row is validated as a ``SurveyRow``.
    """
    text = path.read_text(encoding="utf-8")
    lines = [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"{path}: no survey rows")
    reader = csv.DictReader(lines)
    fields = tuple(reader.fieldnames or ())
    expected = SURVEY_COLUMNS + SURVEY_PUBLISHED_COLUMNS
    if fields not in (SURVEY_COLUMNS, expected):
        raise ValueError(f"{path}: columns must be {SURVEY_COLUMNS} or {expected}, got {fields}")
    rows: list[SurveyRow] = []
    for number, record in enumerate(reader, start=2):
        try:
            rows.append(SurveyRow.model_validate({k: float(v) for k, v in record.items()}))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} row {number}: {exc}") from exc
    return rows


def resolve_survey_path(survey_csv: str) -> Path:
    """Absolute path of a ``surveyCsv`` entry; relative paths live under ``services/seismic``."""
    path = Path(survey_csv)
    return path if path.is_absolute() else SEISMIC_ROOT / path


@lru_cache(maxsize=8)
def _to_geographic(crs: str) -> Transformer:
    return Transformer.from_crs(crs, GEOGRAPHIC_CRS, always_xy=True)


def to_latlon(
    x: npt.ArrayLike, y: npt.ArrayLike, crs: FeatureCrs, unit: LengthUnit
) -> tuple[FloatArray, FloatArray]:
    """Published horizontal coordinates -> WGS84 ``(lat, lon)`` in degrees."""
    x_a = np.asarray(x, dtype=np.float64)
    y_a = np.asarray(y, dtype=np.float64)
    if unit == "deg":
        if crs != "EPSG:4326":
            raise ValueError(f"unit 'deg' needs crs EPSG:4326, got {crs}")
        return y_a, x_a
    scale = M_PER_UNIT[unit]
    lon, lat = _to_geographic(crs).transform(x_a * scale, y_a * scale, errcheck=True)
    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def _enu_path(lat: FloatArray, lon: FloatArray, elev_m: FloatArray, run: RunSection) -> list[Enu]:
    e, n, u = to_enu(lat, lon, elev_m, run.origin)
    return [
        Enu(
            e=round(float(ei), ENU_DECIMALS),
            n=round(float(ni), ENU_DECIMALS),
            u=round(float(ui), ENU_DECIMALS),
        )
        for ei, ni, ui in zip(np.atleast_1d(e), np.atleast_1d(n), np.atleast_1d(u), strict=True)
    ]


def _source_ref(
    feature: PointFeatureConfig | PolygonFeatureConfig | WellFeatureConfig,
) -> SourceRef:
    src = feature.source
    citation = src.citation
    if src.readFrom is not None:
        citation += (
            f" Coordinates read from a copy at {src.readFrom} on {src.accessedOn.isoformat()},"
            " not from the publication itself; location approximate until checked against it."
        )
    else:
        citation += f" Accessed {src.accessedOn.isoformat()}."
    if src.note:
        citation += f" {src.note}"
    return SourceRef(citation=citation, url=src.url, verified=feature.verified)


def well_path(feature: WellFeatureConfig, run: RunSection) -> list[Enu]:
    """ENU trajectory of a well: wellhead plus minimum-curvature displacements per station."""
    rows = (
        feature.survey
        if feature.survey is not None
        else read_survey_csv(resolve_survey_path(str(feature.surveyCsv)))
    )
    md = np.array([r.md for r in rows])
    pos = minimum_curvature(
        md, np.array([r.incDeg for r in rows]), np.array([r.aziDeg for r in rows])
    )
    scale = M_PER_UNIT[feature.unit]
    published = [r for r in rows if r.tvd is not None]
    if published:
        if len(published) != len(rows):
            raise ValueError(f"well {feature.id!r}: every row or no row carries tvd/ns/ew")
        p_ns = np.array([r.ns for r in rows], dtype=np.float64)
        p_ew = np.array([r.ew for r in rows], dtype=np.float64)
        p_tvd = np.array([r.tvd for r in rows], dtype=np.float64)
        mismatch_m = scale * np.sqrt(
            (pos.ns - p_ns) ** 2 + (pos.ew - p_ew) ** 2 + (pos.tvd - p_tvd) ** 2
        )
        worst = int(np.argmax(mismatch_m))
        log.info(
            "well %s: %d stations, max published-vs-minimum-curvature mismatch %.3f m at md %.1f",
            feature.id,
            len(rows),
            float(mismatch_m[worst]),
            float(md[worst]),
        )
        if float(mismatch_m[worst]) > feature.maxSurveyMismatchM:
            raise ValueError(
                f"well {feature.id!r}: published position at md {md[worst]} differs from the "
                f"minimum-curvature position by {mismatch_m[worst]:.2f} m "
                f"(> maxSurveyMismatchM {feature.maxSurveyMismatchM} m); check the survey table"
            )
    east = feature.headX + pos.ew
    north = feature.headY + pos.ns
    elev_m = (feature.depthRefElev - pos.tvd) * scale
    lat, lon = to_latlon(east, north, feature.crs, feature.unit)
    return _enu_path(lat, lon, elev_m, run)


def point_path(feature: PointFeatureConfig, run: RunSection) -> list[Enu]:
    """ENU of a single published point."""
    lat, lon = to_latlon(feature.x, feature.y, feature.crs, feature.unit)
    return _enu_path(np.atleast_1d(lat), np.atleast_1d(lon), np.array([feature.elevM]), run)


def polygon_path(feature: PolygonFeatureConfig, run: RunSection) -> list[Enu]:
    """ENU ring of an outline at ``elevM`` (else the run's ``refSurfaceElevM``), closed: the
    first vertex is repeated last."""
    xy = np.array(feature.vertices, dtype=np.float64)
    lat, lon = to_latlon(xy[:, 0], xy[:, 1], feature.crs, feature.unit)
    elev_m = run.refSurfaceElevM if feature.elevM is None else feature.elevM
    path = _enu_path(lat, lon, np.full(len(lat), elev_m), run)
    return path + [path[0]]


def load_features(cfg: ExportConfig, run: RunSection) -> list[GeoFeature]:
    """Every ``cfg.features`` entry as a contract ``GeoFeature`` in the run's ENU frame.

    Order follows the config. Wells become trajectories (``elevM`` along the path, up-positive),
    boundaries closed rings at their stated elevation, and pads/wellheads single points. Each
    ``SourceRef`` carries the citation, the authoritative URL and ``verified`` exactly as
    configured; features read through a copy rather than the publication say so in the citation.
    Identical config gives identical output; a survey table that contradicts its own printed
    positions is an error, never a silent fallback.
    """
    features: list[GeoFeature] = []
    for feature in cfg.features:
        if isinstance(feature, WellFeatureConfig):
            path = well_path(feature, run)
        elif isinstance(feature, PolygonFeatureConfig):
            path = polygon_path(feature, run)
        else:
            path = point_path(feature, run)
        features.append(
            GeoFeature(
                id=feature.id,
                kind=feature.kind,
                name=feature.name,
                path=path,
                source=_source_ref(feature),
            )
        )
        log.info(
            "feature %s (%s): %d points, verified=%s",
            feature.id,
            feature.kind,
            len(path),
            feature.verified,
        )
    log.info("loaded %d reference features", len(features))
    return features


__all__ = [
    "ENU_DECIMALS",
    "M_PER_UNIT",
    "SEISMIC_ROOT",
    "SurveyPositions",
    "load_features",
    "minimum_curvature",
    "point_path",
    "polygon_path",
    "read_survey_csv",
    "resolve_survey_path",
    "to_latlon",
    "well_path",
]

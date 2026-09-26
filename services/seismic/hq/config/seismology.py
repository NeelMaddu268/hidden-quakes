"""Seismology lane config (``configs/showcase/seismology.yaml``). Unknown keys are an error."""

import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

# services/seismic: config paths such as ``velocity.layerFile`` are relative to it. Found from this
# file, so ``hq`` must run from the source tree (uv's editable install); a wheel ships no configs/.
SEISMIC_ROOT = Path(__file__).resolve().parents[2]


class Velocity3dConfig(BaseModel):
    """The 3D velocity model file (GDR 1800), downloaded into ``data/cache/velocity/`` (LOC-07)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    submissionUrl: str = Field(min_length=1)  # GDR submission page
    url: str = Field(min_length=1)  # exact file URL
    cacheFile: str = Field(min_length=1)  # file name under data/cache/velocity/
    expectedBytes: int = Field(gt=0)  # Content-Length reported by the server
    citation: str = Field(min_length=1)
    license: str = Field(min_length=1)

    @field_validator("cacheFile")
    @classmethod
    def _bare_file_name(cls, value: str) -> str:
        if PurePosixPath(value).name != value or PureWindowsPath(value).name != value:
            raise ValueError(f"cacheFile must be a bare file name, got {value!r}")
        return value


class VelocityConfig(BaseModel):
    """Velocity models. The 1D layer file carries its own source and datum in its header."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layerFile: Path  # 1D layer file, relative to services/seismic
    # Unit guard for any layer file: every Vp and Vs (m/s) must lie inside these [min, max] ranges.
    plausibleVpMPerS: tuple[float, float]
    plausibleVsMPerS: tuple[float, float]
    minLayerThicknessM: float = Field(gt=0)  # unit guard: thinner layers mean topElevM is in km
    maxTopExtensionM: float = Field(gt=0)  # cap on with_top_extended_to, from the source top
    profilePlotBottomElevM: float  # m ASL; how far down the profile figure draws the half-space
    model3d: Velocity3dConfig

    @field_validator("layerFile")
    @classmethod
    def _relative(cls, value: Path) -> Path:
        # Rooted on either OS ("/x", "\\x", "C:\\x", "C:x"). Path.is_absolute() alone misses "/x"
        # on Windows (no drive), so check the POSIX and Windows forms explicitly.
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).anchor:
            raise ValueError(f"layerFile must be relative to services/seismic, got {value}")
        return value

    @model_validator(mode="after")
    def _ranges(self) -> "VelocityConfig":
        for label, (low, high) in (
            ("plausibleVpMPerS", self.plausibleVpMPerS),
            ("plausibleVsMPerS", self.plausibleVsMPerS),
        ):
            if not 0.0 < low < high:
                raise ValueError(f"{label} must be [min, max], 0 < min < max, got {[low, high]}")
        return self

    def layer_path(self) -> Path:
        """Absolute path of ``layerFile`` (resolved against services/seismic, not the CWD)."""
        return SEISMIC_ROOT / self.layerFile


# docs/02 LocationQuality: hErrM is the "68% horizontal semi-major axis", vErrM "68% vertical".
DOCS02_ERR_CONFIDENCE = 0.68


def _is_multiple(value: float, step: float) -> bool:
    """True when ``value`` is an integer multiple of ``step`` (to float rounding)."""
    ratio = value / step
    return abs(ratio - round(ratio)) <= 1e-9 * max(1.0, abs(ratio))


class GridsConfig(BaseModel):
    """Per-station 2D (r, elevM) eikonal travel-time tables (LOC-02, ``hq.locate.tt_grid``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    drM: float = Field(gt=0)  # horizontal node spacing (m)
    dzM: float = Field(gt=0)  # vertical node spacing (m)
    rMaxM: float = Field(gt=0)  # largest epicentral distance in a table (m); a multiple of drM
    # The grid top is the highest receiver sensorElevM (or the search-volume top, if higher) plus
    # this margin, snapped up onto the dzM lattice anchored at bottomElevM. The velocity model's
    # top layer is extended up to it explicitly (with_top_extended_to) and that is recorded.
    topMarginM: float = Field(ge=0)
    bottomElevM: float  # grid bottom (m ASL); must lie below the search volume (checked below)
    # Near-source initialisation: nodes within this distance of the receiver get exact 1D layered
    # times, and the eikonal solve starts from an isochron inside that region.
    seedRadiusM: float = Field(gt=0)
    fmmOrder: Literal[1, 2]  # scikit-fmm stencil order
    # After each solve, the table is compared with the exact 1D layered times at every elevation
    # node and every accuracyCheckStrideR-th distance node; the max error per model layer is kept
    # in the table's sidecar and in the run record (the check never changes the table).
    accuracyCheckStrideR: int = Field(ge=1)

    @model_validator(mode="after")
    def _check(self) -> "GridsConfig":
        if not _is_multiple(self.rMaxM, self.drM):
            raise ValueError(f"rMaxM {self.rMaxM} must be a multiple of drM {self.drM}")
        if self.seedRadiusM < 4.0 * max(self.drM, self.dzM):
            raise ValueError("seedRadiusM must be at least 4 grid cells so the seed isochron spans cells")
        return self


class SearchVolumeConfig(BaseModel):
    """Where the locator searches, in ENU metres around the run origin and in elevM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    halfWidthM: float = Field(gt=0)  # e and n both span [-halfWidthM, +halfWidthM]
    # Top of the search volume (m ASL). null means run.refSurfaceElevM: with a 1D model and no DEM,
    # the ground at the origin is the one surface we know, so hypocentres stay below it. The top
    # is then snapped down onto the fine lattice, so it never lies above this value.
    topElevM: float | None
    bottomElevM: float  # m ASL


class PhaseSigma(BaseModel):
    """Pick-time uncertainty per phase (s)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    P: float = Field(gt=0)
    S: float = Field(gt=0)


class OutlierConfig(BaseModel):
    """Outlier pass: drop picks with |residual| > max(madK * MAD, floorS), then relocate once."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    madK: float = Field(gt=0)
    floorS: float = Field(ge=0)


class LocatorConfig(BaseModel):
    """Grid-search locator (``hq.locate.locator``) and its PDF uncertainty (``uncertainty``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    volume: SearchVolumeConfig
    coarseSpacingM: float = Field(gt=0)  # full-volume grid search
    fineSpacingM: float = Field(gt=0)  # the reported PDF lives on this lattice
    # First fine box: +/- this around the best coarse node, clipped to the volume. It is only where
    # the search starts: the PDF region grows past it until it covers the mass (see locator.py).
    fineHalfWidthM: float = Field(gt=0)
    # The misfit is first evaluated every fineStageSpacingM, from the first fine box outward until
    # the region within pdfCutoff of the minimum clears every face that is not a volume face; the
    # fine nodes are then evaluated over that region and grown the same way (see locator.py).
    fineStageSpacingM: float = Field(gt=0)
    # Nodes whose PDF exponent (pdfMisfitScale * misfit) exceeds its minimum by more than this carry
    # less than exp(-pdfCutoff) of the peak node's mass and are treated as zero mass.
    pdfCutoff: float = Field(gt=0)
    # PDF = exp(-pdfMisfitScale * misfit). 1.0 is the Laplace likelihood with scale sigma / prob;
    # it changes the formal errors only, never the hypocentre (see hq.locate.uncertainty).
    pdfMisfitScale: float = Field(gt=0)
    # Most fine nodes one PDF may cover. A PDF whose region needs more is truncated: hErrM and vErrM
    # become None and the search record says pdfTruncated (see locator.py).
    maxPdfNodes: int = Field(ge=1)
    # Pick sigma per phase (s) for every preprocessing profile, and per-profile overrides keyed by
    # Station.preprocessProfile (empty: every profile uses pickSigmaS).
    pickSigmaS: PhaseSigma
    profilePickSigmaS: dict[str, PhaseSigma]
    outlier: OutlierConfig
    minPicks: int = Field(ge=4)  # fewer picks cannot constrain (e, n, elevM, t0)
    # hErrM is the semi-major axis at this level. docs/02 defines hErrM and vErrM as 68%, and vErrM
    # is the 1-sigma of the vertical marginal, so only 0.68 keeps the two consistent (checked).
    errConfidence: float
    # depthOnEdge: more than this PDF mass on the fine grid's top or bottom face.
    depthOnEdgeMassFraction: float = Field(gt=0, lt=1)
    # mapOnVolumeTop / mapOnVolumeBottom (separate from depthOnEdge): the MAP node lies within this
    # distance (m) of the search volume's top / bottom (0: only on the face row itself).
    mapOnVolumeFaceBandM: float = Field(ge=0)
    nWorkers: int = Field(ge=1)  # processes for locate_many; results do not depend on it
    evalChunkNodes: int = Field(ge=1)  # nodes per misfit block (memory only; results unchanged)
    enuConsistencyTolM: float = Field(gt=0)  # |enu_u + origin elevM - sensorElevM| must be below

    @field_validator("errConfidence")
    @classmethod
    def _docs02_confidence(cls, value: float) -> float:
        if value != DOCS02_ERR_CONFIDENCE:
            raise ValueError(
                f"errConfidence must be {DOCS02_ERR_CONFIDENCE}: docs/02 defines hErrM and vErrM "
                f"as 68% and vErrM is the vertical 1-sigma, got {value}"
            )
        return value

    @model_validator(mode="after")
    def _check(self) -> "LocatorConfig":
        fine = self.fineSpacingM
        for label, value in (
            ("coarseSpacingM", self.coarseSpacingM),
            ("fineStageSpacingM", self.fineStageSpacingM),
            ("fineHalfWidthM", self.fineHalfWidthM),
        ):
            if not _is_multiple(value, fine):
                raise ValueError(f"{label} {value} must be a multiple of fineSpacingM {fine}")
        if not _is_multiple(self.volume.halfWidthM, self.coarseSpacingM):
            raise ValueError("volume.halfWidthM must be a multiple of coarseSpacingM")
        first_box = (2 * round(self.fineHalfWidthM / fine) + 1) ** 3
        if self.maxPdfNodes < first_box:
            raise ValueError(
                f"maxPdfNodes {self.maxPdfNodes} must hold the first fine box ({first_box} nodes)"
            )
        top = self.volume.topElevM
        if top is not None and top - self.volume.bottomElevM < self.coarseSpacingM:
            raise ValueError("the search volume must be at least one coarse cell tall")
        return self


class SyntheticZoneConfig(BaseModel):
    """Where synthetic hypocentres are drawn: a vertical cylinder in ENU metres and elevM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    centerEM: float
    centerNM: float
    radiusM: float = Field(gt=0)  # uniform in area within this horizontal radius
    topElevM: float  # m ASL, uniform in elevation between bottomElevM and topElevM
    bottomElevM: float

    @model_validator(mode="after")
    def _check(self) -> "SyntheticZoneConfig":
        if not self.bottomElevM < self.topElevM:
            raise ValueError("synthetic zone bottomElevM must lie below topElevM")
        return self


class SyntheticConfig(BaseModel):
    """Synthetic recovery test (``hq.locate.synthetic``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nEvents: int = Field(ge=1)
    seed: int = Field(ge=0)
    zone: SyntheticZoneConfig
    sKeepProb: float = Field(ge=0, le=1)  # each station's S pick is kept with this probability
    pickProb: float = Field(gt=0, le=1)  # picker probability given to every synthetic pick


class CatalogDatum(BaseModel):
    """Depth datum of one catalog contributor (keyed by its QuakeML ``catalog:datasource``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    surfaceElevM: float  # m ASL of the surface depths are measured from; elevM = this - depth
    validFrom: AwareDatetime  # origins earlier than this used another datum and are rejected
    # Short statement of what the published depth is relative to; with sourceUrls[0] it becomes
    # every row's CatalogEvent.depthDatum. description and all sourceUrls go to the run record.
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)  # the documented evidence for that datum
    sourceUrls: list[str] = Field(min_length=1)  # where it is documented; the first is primary


class CatalogConfig(BaseModel):
    """Public regional catalog query (stage ``catalog``, MATCH-01)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)  # ObsPy FDSN client key, e.g. "USGS" (ComCat)
    providerLabel: str = Field(min_length=1)  # CatalogEvent.source = "<NET> via <providerLabel>"
    timeoutS: float = Field(gt=0)  # FDSN client timeout per request
    eventTypes: Annotated[list[str], Field(min_length=1)] | None  # kept types; None = all
    minMagnitude: float | None  # FDSN minmagnitude; None = no limit
    maxMagnitude: float | None  # FDSN maxmagnitude; None = no limit
    # ComCat product carrying picks/arrivals (analyst picks only when the origin is manual)
    arrivalsProductType: str = Field(min_length=1)
    datums: dict[str, CatalogDatum] = Field(min_length=1)  # per contributor, lowercase code


class SeismologyConfig(BaseModel):
    """Contents of ``seismology.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    velocity: VelocityConfig
    grids: GridsConfig
    locator: LocatorConfig
    synthetic: SyntheticConfig
    catalog: CatalogConfig

    @model_validator(mode="after")
    def _consistent(self) -> "SeismologyConfig":
        vol = self.locator.volume
        if self.grids.bottomElevM > vol.bottomElevM - self.grids.dzM:
            raise ValueError(
                f"grids.bottomElevM {self.grids.bottomElevM} must lie at least one dzM below "
                f"locator.volume.bottomElevM {vol.bottomElevM}"
            )
        corner = math.hypot(vol.halfWidthM, vol.halfWidthM)
        if corner >= self.grids.rMaxM:
            raise ValueError("grids.rMaxM must exceed the search volume's half diagonal")
        zone = self.synthetic.zone
        reach = math.hypot(zone.centerEM, zone.centerNM) + zone.radiusM
        if reach > vol.halfWidthM or zone.bottomElevM < vol.bottomElevM:
            raise ValueError("the synthetic zone must lie inside the search volume")
        if vol.topElevM is not None and zone.topElevM > vol.topElevM:
            raise ValueError("the synthetic zone must lie below the search volume top")
        return self

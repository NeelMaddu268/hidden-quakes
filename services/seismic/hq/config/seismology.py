"""Seismology lane config (``configs/showcase/seismology.yaml``). Unknown keys are an error."""

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


class AssociatorVolumeConfig(BaseModel):
    """PyOcto's search volume: the run bbox projected to ENU plus a margin, between two elevations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    horizontalMarginM: float = Field(ge=0)  # added on every side of the projected bbox (m)
    topElevM: float | None  # m ASL; null = run.refSurfaceElevM
    bottomElevM: float  # m ASL


class AssociatorTablesConfig(BaseModel):
    """PyOcto station-specific travel-time tables (``hq.associate.tables``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spacingM: float = Field(gt=0)  # node spacing, horizontal and vertical (m)
    seedRadiusM: float = Field(gt=0)  # straight-ray times inside this radius seed the eikonal solve
    fmmOrder: Literal[1, 2]  # scikit-fmm stencil order
    # Upper bound (s) on the seed error where a layer boundary crosses the seed disc; a station
    # whose bound exceeds this fails the build instead of producing a biased table.
    maxSeedErrorS: float = Field(gt=0)

    @model_validator(mode="after")
    def _seed_spans_cells(self) -> "AssociatorTablesConfig":
        if self.seedRadiusM < self.spacingM:
            raise ValueError("tables.seedRadiusM must be at least one node spacing")
        return self


class AssociatorSweepConfig(BaseModel):
    """Grid of association settings the sweep runs (docs/lanes/H2 -> Association (PyOcto))."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool  # false: the stage runs the configured point only
    minStations: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)
    nSPicks: list[Annotated[int, Field(ge=0)]] = Field(min_length=1)
    minPickProb: list[Annotated[float, Field(gt=0, le=1)]] = Field(min_length=1)


class AssociatorConfig(BaseModel):
    """Association with PyOcto 0.2.0 (stage ``associate``, LOC-03).

    PyOcto's own knobs keep its units: distances in km (``Km`` suffix), times in s.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    picksTable: Literal["known/picks.parquet", "picks.parquet"]  # stage input, in the run dir
    # Picks from a stations.parquet station with usedInRun false: drop them (counted, logged) or
    # reject the input. A pick from a station missing from stations.parquet always fails.
    picksFromUnusedStations: Literal["drop", "reject"]
    minPickProb: float = Field(gt=0, le=1)  # picks below this probability are not associated
    minStations: int = Field(ge=1)  # final events need picks from at least this many stations
    nPicks: int = Field(ge=1)  # PyOcto n_picks
    nPPicks: int = Field(ge=0)  # PyOcto n_p_picks (stations with a P pick)
    nSPicks: int = Field(ge=0)  # PyOcto n_s_picks (stations with an S pick)
    nPAndSPicks: int = Field(ge=0)  # PyOcto n_p_and_s_picks (stations with both)
    toleranceS: float = Field(gt=0)  # velocity model tolerance
    associationCutoffDistanceKm: Annotated[float, Field(gt=0)] | None  # null = every station
    timeBeforeS: float = Field(gt=0)  # checked against the largest S time in the volume
    timeSlicingS: float = Field(gt=0)
    minNodeSizeKm: float = Field(gt=0)
    minNodeSizeLocationKm: float = Field(gt=0)
    pickMatchToleranceS: float = Field(gt=0)
    minIntereventTimeS: float = Field(ge=0)
    exponentialEdt: bool
    edtPickStdS: float = Field(gt=0)
    maxPickOverlap: int = Field(ge=0)
    refinementIterations: int = Field(ge=1)
    locationSplitDepth: int = Field(ge=1)
    locationSplitReturn: int = Field(ge=0)
    minPickFraction: float = Field(ge=0, le=1)
    queueMemoryProtectionDfsSize: int = Field(ge=1)
    nodeLogInterval: int = Field(ge=0)
    nThreads: int = Field(ge=1)
    enuConsistencyTolM: float = Field(gt=0)  # stations: stored enu vs lat/lon/sensorElevM
    mergeWithinS: float = Field(ge=0)  # duplicate merge: origin times at most this far apart ...
    mergeMinSharedFraction: float = Field(gt=0, le=1)  # ... sharing this share of the smaller's picks
    volume: AssociatorVolumeConfig
    tables: AssociatorTablesConfig
    sweep: AssociatorSweepConfig

    @model_validator(mode="after")
    def _check(self) -> "AssociatorConfig":
        if self.locationSplitReturn >= self.locationSplitDepth:
            raise ValueError("locationSplitReturn must be smaller than locationSplitDepth (PyOcto)")
        if self.minNodeSizeLocationKm >= self.minNodeSizeKm:
            raise ValueError("minNodeSizeLocationKm must be smaller than minNodeSizeKm (PyOcto)")
        top = self.volume.topElevM
        if top is not None and top <= self.volume.bottomElevM:
            raise ValueError("volume.bottomElevM must lie below volume.topElevM")
        # PyOcto silently raises n_picks/n_p_picks/n_s_picks when they are inconsistent; require
        # consistency instead, for the configured point and every sweep point, so the recorded
        # arguments are the ones in effect.
        for n_s in {self.nSPicks, *self.sweep.nSPicks}:
            if self.nPicks < self.nPPicks + n_s:
                raise ValueError(f"nPicks {self.nPicks} < nPPicks {self.nPPicks} + nSPicks {n_s}")
            if n_s < self.nPAndSPicks:
                raise ValueError(f"nSPicks {n_s} < nPAndSPicks {self.nPAndSPicks}")
        if self.nPicks < 2 * self.nPAndSPicks or self.nPPicks < self.nPAndSPicks:
            raise ValueError("nPicks must be >= 2 * nPAndSPicks and nPPicks >= nPAndSPicks")
        # The station minimum must be the binding one: an event seen by minStations stations with
        # P only has minStations picks, and PyOcto must not reject it before the station filter.
        fewest = min(self.minStations, *self.sweep.minStations)
        if max(self.nPicks, self.nPPicks) > fewest:
            raise ValueError(f"nPicks and nPPicks must not exceed the smallest minStations {fewest}")
        return self


class SeismologyConfig(BaseModel):
    """Contents of ``seismology.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    velocity: VelocityConfig
    catalog: CatalogConfig
    associator: AssociatorConfig

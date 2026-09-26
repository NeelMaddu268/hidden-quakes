"""Seismology lane config (``configs/showcase/seismology.yaml``). Unknown keys are an error."""

from pathlib import Path
from typing import Annotated

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
        if Path(value).name != value:
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
        if value.is_absolute():
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


class SeismologyConfig(BaseModel):
    """Contents of ``seismology.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    velocity: VelocityConfig
    catalog: CatalogConfig

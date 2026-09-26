"""Seismology config: velocity models, catalog, association, location, tiers (``seismology.yaml``).

Parsed from ``configs/showcase/seismology.yaml``. Unknown keys are an error (docs/02 -> Config files).
Each section is added by the ticket that needs it; one field per line in ``SeismologyConfig``.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    profilePlotBottomElevM: float  # m ASL; how far down the profile figure draws the half-space
    model3d: Velocity3dConfig

    @field_validator("layerFile")
    @classmethod
    def _relative(cls, value: Path) -> Path:
        if value.is_absolute():
            raise ValueError(f"layerFile must be relative to services/seismic, got {value}")
        return value


class SeismologyConfig(BaseModel):
    """Contents of ``seismology.yaml``. Unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    velocity: VelocityConfig

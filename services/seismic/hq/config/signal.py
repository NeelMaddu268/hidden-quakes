"""Signal lane config (``configs/showcase/signal.yaml``), owned by H1.

One section per ticket, so parallel tickets edit separate classes and separate YAML blocks.
Every model rejects unknown keys (docs/02 -> Config files).
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- SEIS-01: station inventory and channel selection -------------------------------------------


class StationSelection(_Section):
    """Which stations and channel triplets enter the run, and how their sensor depth is resolved."""


# --- SEIS-05: full-window download and cache --------------------------------------------------------


class DownloadConfig(_Section):
    """Hour-chunk download, retries, cache and gap accounting."""

    client: str = Field(min_length=1)  # ObsPy FDSN client base, e.g. "EARTHSCOPE"
    timeoutS: float = Field(gt=0)  # per-request socket timeout
    chunkS: int = Field(gt=0)  # request length; chunks sit on multiples of this (epoch-aligned)
    padS: float = Field(ge=0)  # added before windowStart and after windowEnd
    maxWorkers: int = Field(ge=1)  # parallel (station, UTC day) download units
    maxRetries: int = Field(ge=0)  # retries per chunk after the first attempt
    backoffBaseS: float = Field(ge=0)  # wait before retry k is backoffBaseS * 2**k ...
    backoffMaxS: float = Field(ge=0)  # ... capped at this
    provisionalLagS: float = Field(ge=0)  # chunk ending < this before its fetch is refetched
    minGapSamples: float = Field(gt=1)  # spacing > this many sample intervals is a gap
    maxGapFraction: float = Field(gt=0, le=1)  # Check A: a useful station stays below this
    minUsefulStations: int = Field(ge=1)  # Check A: pass needs at least this many

    @model_validator(mode="after")
    def _backoff_ordered(self) -> "DownloadConfig":
        if self.backoffMaxS < self.backoffBaseS:
            raise ValueError(
                f"download.backoffMaxS ({self.backoffMaxS}) is below backoffBaseS "
                f"({self.backoffBaseS})"
            )
        return self


# --- SEIS-02: known-event windows -------------------------------------------------------------------


class KnownEventsConfig(_Section):
    """Windows around the largest public-catalog events, used for Check B and the weight A/B."""


# --- SEIS-03: preprocessing profiles ----------------------------------------------------------------


class PreprocessConfig(_Section):
    """Per-sensor-type preprocessing profiles that turn raw counts into 100 Hz model input."""


# --- SEIS-04 / SEIS-06: PhaseNet picking ------------------------------------------------------------


class PickerConfig(_Section):
    """PhaseNet weights, thresholds and gap-edge handling."""


# --- SEIS-07: STA/LTA baseline ----------------------------------------------------------------------


class BaselineConfig(_Section):
    """Classical recursive STA/LTA picker and its threshold sweep."""


class SignalConfig(_Section):
    """Contents of ``signal.yaml``."""

    stations: StationSelection
    download: DownloadConfig
    known: KnownEventsConfig
    preprocess: PreprocessConfig
    picker: PickerConfig
    baseline: BaselineConfig

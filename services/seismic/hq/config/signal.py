"""Signal lane config (``configs/showcase/signal.yaml``), owned by H1.

One section per ticket, so parallel tickets edit separate classes and separate YAML blocks.
Every model rejects unknown keys (docs/02 -> Config files).
"""

from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- SEIS-01: station inventory and channel selection -------------------------------------------


class StationSelection(_Section):
    """Which stations and channel triplets enter the run, and how their sensor depth is resolved."""


# --- SEIS-05: full-window download and cache --------------------------------------------------------


class DownloadConfig(_Section):
    """Hour-chunk download, retries, cache and gap accounting."""


# --- SEIS-02: known-event windows -------------------------------------------------------------------


class KnownEventsConfig(_Section):
    """Windows around the largest public-catalog events, used for Check B and the weight A/B."""

    nEvents: int = Field(ge=1)  # largest public events by magnitude to window
    preS: float = Field(ge=0.0)  # s of data before the catalog origin time
    postS: float = Field(gt=0.0)  # s of data after the catalog origin time
    minGapS: float = Field(gt=0.0)  # missing coverage shorter than this is jitter, not a gap
    maxGapFraction: float = Field(ge=0.0, le=1.0)  # usable only if the gappiest component <= this
    minStations: int = Field(ge=1)  # usable three-component stations a window needs to PASS
    usedInRunOnly: bool  # consider only stations.parquet rows with usedInRun = true


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

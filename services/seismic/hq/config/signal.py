"""Signal lane config (``configs/showcase/signal.yaml``), owned by H1.

One section per ticket, so parallel tickets edit separate classes and separate YAML blocks.
Every model rejects unknown keys (docs/02 -> Config files).
"""

from pydantic import BaseModel, ConfigDict


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

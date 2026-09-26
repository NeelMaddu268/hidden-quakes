"""Signal lane config (``configs/showcase/signal.yaml``), owned by H1.

One section per ticket, so parallel tickets edit separate classes and separate YAML blocks.
Every model rejects unknown keys (docs/02 -> Config files).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- SEIS-01: station inventory and channel selection -------------------------------------------


class StationQuery(_Section):
    """FDSN station query at channel level. Window and bbox come from ``run.yaml``."""

    fdsnClient: str = Field(min_length=1)  # ObsPy client key ("EARTHSCOPE") or base URL
    timeoutS: float = Field(gt=0)
    network: str = Field(min_length=1)  # FDSN pattern, e.g. "*"
    station: str = Field(min_length=1)
    location: str = Field(min_length=1)
    channel: str = Field(min_length=1)
    excludeNetworks: tuple[str, ...]  # e.g. SY: synthetic seismograms, never real data
    skipRestricted: bool  # drop channels whose restrictedStatus is "closed"
    cacheSubdir: str = Field(min_length=1)  # under cache_dir: raw XML, responses, DEM cache


class ChannelRules(_Section):
    """How three-component triplets are formed and ranked."""

    verticalComponents: tuple[str, ...] = Field(min_length=1)
    horizontalPairs: tuple[tuple[str, str], ...] = Field(min_length=1)  # preference order
    velocityCodes: tuple[str, ...] = Field(min_length=1)  # band+instrument, best first
    accelerometerCodes: tuple[str, ...]  # used only when a site has no velocity triplet
    componentDepthTolM: float = Field(ge=0)  # Z/H1/H2 depths must agree within this
    locationDepthTolM: float = Field(ge=0)  # two locations at one site count as distinct beyond
    coordTolM: float = Field(ge=0)  # warn when channel and station coordinates differ more

    @model_validator(mode="after")
    def _check(self) -> "ChannelRules":
        codes = (*self.velocityCodes, *self.accelerometerCodes)
        if any(len(c) != 2 for c in codes):
            raise ValueError(f"channel codes must be band+instrument (2 chars), got {codes}")
        if len(set(codes)) != len(codes):
            raise ValueError("velocityCodes and accelerometerCodes must not repeat or overlap")
        comps = (*self.verticalComponents, *(c for pair in self.horizontalPairs for c in pair))
        if any(len(c) != 1 for c in comps):
            raise ValueError(f"component codes must be single characters, got {comps}")
        return self


class KindRules(_Section):
    """``Station.kind``: borehole by depth first, then strong-motion by instrument code."""

    boreholeMinDepthM: float = Field(gt=0)
    strongMotionInstrumentCodes: tuple[str, ...]  # SEED instrument code, e.g. "N"
    boreholeLookingCodes: tuple[str, ...]  # flagged in the log when shallower than the minimum


class ProfileRule(_Section):
    """Sample-rate range (inclusive) mapped to a preprocessing profile; first match wins."""

    minRateHz: float = Field(gt=0)
    maxRateHz: float = Field(gt=0)
    profile: str = Field(min_length=1)  # key into the preprocess block (SEIS-03)

    @model_validator(mode="after")
    def _check(self) -> "ProfileRule":
        if self.maxRateHz < self.minRateHz:
            raise ValueError(f"maxRateHz {self.maxRateHz} < minRateHz {self.minRateHz}")
        return self


class ElevationCheck(_Section):
    """Resolve whether StationXML station elevation is the site surface or the sensor itself.

    Compared against a DEM at the sensor position; see ``hq.ingest.inventory.resolve_elevation``.
    """

    demUrl: str = Field(min_length=1)  # USGS 3DEP EPQS point query
    demTimeoutS: float = Field(gt=0)
    demRetries: int = Field(ge=0)
    demBackoffS: float = Field(ge=0)
    demNoDataValue: float  # EPQS sentinel for "no DEM here"
    demCacheFile: str = Field(min_length=1)  # JSON under the stationxml cache dir
    demKeyDecimals: int = Field(ge=0, le=8)  # lat/lon rounding for the cache key
    toleranceM: float = Field(gt=0)
    onAmbiguous: Literal["error", "surface", "skip"]


class AvailabilityCheck(_Section):
    """Per-channel data coverage of the window, from the MUSTANG daily availability metric."""

    url: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    timeoutS: float = Field(gt=0)
    onMissing: Literal["error", "unused"]  # a chosen channel-day with no measurement


class StationSelection(_Section):
    """Which stations and channel triplets enter the run, and how their sensor depth is resolved."""

    query: StationQuery
    channels: ChannelRules
    kind: KindRules
    profiles: tuple[ProfileRule, ...] = Field(min_length=1)
    elevation: ElevationCheck
    availability: AvailabilityCheck


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

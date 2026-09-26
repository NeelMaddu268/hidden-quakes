"""Signal lane config (``configs/showcase/signal.yaml``), owned by H1.

One section per ticket, so parallel tickets edit separate classes and separate YAML blocks.
Every model rejects unknown keys (docs/02 -> Config files).
"""

from typing import Annotated, Literal

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

    nEvents: int = Field(ge=1)  # largest public events by magnitude to window
    preS: float = Field(ge=0.0)  # s of data before the catalog origin time
    postS: float = Field(gt=0.0)  # s of data after the catalog origin time
    minGapSamples: float = Field(ge=1.0)  # hole > (minGapSamples - 1) sample intervals is a gap
    maxGapFraction: float = Field(ge=0.0, le=1.0)  # usable only if the gappiest component <= this
    minStations: int = Field(ge=1)  # usable three-component stations a window needs to PASS
    usedInRunOnly: bool  # consider only stations.parquet rows with usedInRun = true


# --- SEIS-03: preprocessing profiles ----------------------------------------------------------------


class TaperConfig(_Section):
    """ObsPy ``Trace.taper`` arguments, applied to every gap-separated segment."""

    type: Literal["hann", "hamming", "cosine", "blackman", "bartlett", "triang"]
    maxPercentage: float = Field(gt=0.0, le=0.5)  # fraction of the segment, per side
    maxLengthS: float = Field(gt=0.0)  # cap on the taper length, per side, in seconds


class AntiAliasConfig(_Section):
    """Chebyshev type II lowpass run zero-phase before every integer decimation.

    Edges are fractions of the OUTPUT Nyquist (``targetRateHz / 2``). The order is the minimum
    that meets both specs (``scipy.signal.cheb2ord``). Losses are per pass; zero-phase filtering
    runs the filter twice, so the stopband is attenuated by twice ``stopbandAttenuationDb``.
    """

    passbandEdgeFraction: float = Field(gt=0.0, lt=1.0)
    stopbandEdgeFraction: float = Field(gt=0.0, le=1.0)
    passbandLossDb: float = Field(gt=0.0)
    stopbandAttenuationDb: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> "AntiAliasConfig":
        if self.passbandEdgeFraction >= self.stopbandEdgeFraction:
            raise ValueError("antiAlias.passbandEdgeFraction must be below stopbandEdgeFraction")
        return self


class _ProfileBase(_Section):
    """Input sample rates a profile accepts; anything outside raises."""

    minRateHz: float = Field(gt=0.0)
    maxRateHz: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_rates(self) -> "_ProfileBase":
        if self.minRateHz > self.maxRateHz:
            raise ValueError(f"minRateHz {self.minRateHz} exceeds maxRateHz {self.maxRateHz}")
        return self


class PassthroughProfile(_ProfileBase):
    """Detrend and taper only; input must already be at ``targetRateHz`` (``surface-100``)."""

    method: Literal["passthrough"]


class DecimateProfile(_ProfileBase):
    """Detrend, taper, zero-phase Butterworth lowpass, anti-alias, resample to ``targetRateHz``.

    ``surface-hi`` and ``borehole-A``.
    """

    method: Literal["decimate"]
    lowpassHz: float = Field(gt=0.0)
    lowpassCorners: int = Field(ge=1)


class StretchProfile(_ProfileBase):
    """Detrend, taper, zero-phase Butterworth bandpass, then relabel to ``targetRateHz``.

    ``borehole-B``: no resampling, the waveform is time-stretched by ``rate / targetRateHz`` and
    picks come back through ``TimeMap.to_real``. ``bandpassHz`` is in real (unstretched) Hz.
    """

    method: Literal["stretch"]
    bandpassHz: tuple[float, float]
    bandpassCorners: int = Field(ge=1)


PreprocessProfile = Annotated[
    PassthroughProfile | DecimateProfile | StretchProfile, Field(discriminator="method")
]


class PreprocessConfig(_Section):
    """Per-sensor-type preprocessing profiles that turn raw counts into 100 Hz model input."""

    targetRateHz: float = Field(gt=0.0)
    rateRelTol: float = Field(gt=0.0, lt=1e-3)
    maxUpsampleFactor: int = Field(ge=1)
    minSegmentModelS: float = Field(gt=0.0)  # model seconds: real seconds x TimeMap factor
    joinMisalignmentSamples: float = Field(gt=0.0, lt=0.5)  # fraction of one input sample
    detrend: Literal["linear", "constant", "simple"]
    taper: TaperConfig
    antiAlias: AntiAliasConfig
    componentRename: dict[str, str]
    modelComponents: str = Field(min_length=1)
    profiles: dict[str, PreprocessProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "PreprocessConfig":
        if len(set(self.modelComponents)) != len(self.modelComponents):
            raise ValueError(f"modelComponents has duplicates: {self.modelComponents!r}")
        for src, dst in self.componentRename.items():
            if len(src) != 1 or len(dst) != 1:
                raise ValueError(f"componentRename maps single characters, got {src!r}: {dst!r}")
            if dst not in self.modelComponents:
                raise ValueError(f"componentRename target {dst!r} not in {self.modelComponents!r}")
        if len(set(self.componentRename.values())) != len(self.componentRename):
            raise ValueError("componentRename maps two components onto the same target")
        nyquist = self.targetRateHz / 2.0
        for name, prof in self.profiles.items():
            if prof.minRateHz < self.targetRateHz * (1.0 - self.rateRelTol):
                raise ValueError(f"profile {name}: minRateHz is below targetRateHz")
            if isinstance(prof, PassthroughProfile):
                if prof.minRateHz != self.targetRateHz or prof.maxRateHz != self.targetRateHz:
                    raise ValueError(f"profile {name}: passthrough needs min = max = targetRateHz")
            elif isinstance(prof, DecimateProfile):
                if prof.lowpassHz >= nyquist:
                    raise ValueError(f"profile {name}: lowpassHz must be below {nyquist} Hz")
            else:
                low, high = prof.bandpassHz
                if not 0.0 < low < high < prof.minRateHz / 2.0:
                    raise ValueError(
                        f"profile {name}: bandpassHz must satisfy 0 < low < high < minRateHz / 2"
                    )
        return self


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

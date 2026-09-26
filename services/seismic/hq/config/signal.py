"""Signal lane config (``configs/showcase/signal.yaml``), owned by H1.

One section per ticket, so parallel tickets edit separate classes and separate YAML blocks.
Every model rejects unknown keys (docs/02 -> Config files).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class SeisbenchArgs(_Section):
    """Keyword arguments passed verbatim to ``seisbench`` ``WaveformModel.classify``.

    Passed explicitly so the weights' own ``default_args`` (which differ per weight set) never
    decide a result silently.
    """

    overlap: int = Field(ge=0)  # samples of overlap between consecutive 3001-sample windows
    stacking: Literal["avg", "max"]  # how overlapping window predictions are combined
    blinding: tuple[int, int]  # prediction samples discarded at the start / end of every window
    strict: bool  # seisbench only annotates spans where all three components exist
    flexibleHorizontalComponents: bool  # treat 1/2 as N/E inside seisbench

    @model_validator(mode="after")
    def _check(self) -> "SeisbenchArgs":
        if min(self.blinding) < 0:
            raise ValueError(f"blinding must be non-negative, got {self.blinding}")
        if self.overlap < sum(self.blinding):
            # Blinded window edges would not be covered by the neighbouring window, leaving NaN
            # holes inside a block that split triggers. overlap < the model window is checked
            # against the loaded model (hq.pick.phasenet.check_model).
            raise ValueError(
                f"overlap {self.overlap} must be >= blinding[0] + blinding[1] = {sum(self.blinding)}"
            )
        return self


class RecordSectionConfig(_Section):
    """One PNG per known event: display copies sorted by epicentral distance, picks marked."""

    bandHz: tuple[float, float]  # display_copy bandpass
    windowS: tuple[float, float]  # plotted span, seconds relative to the catalog origin time
    padS: float = Field(ge=0.0)  # extra data read on each side so the filter taper stays off-plot
    component: str = Field(min_length=1, max_length=1)  # component letter plotted (e.g. "Z")
    widthIn: float = Field(gt=0.0)
    heightPerTraceIn: float = Field(gt=0.0)
    minHeightIn: float = Field(gt=0.0)
    dpi: int = Field(gt=0)
    traceHalfHeight: float = Field(gt=0.0)  # half-height of a normalized trace; rows are 1 apart

    @model_validator(mode="after")
    def _check(self) -> "RecordSectionConfig":
        if not 0.0 < self.bandHz[0] < self.bandHz[1]:
            raise ValueError(f"bandHz must be (low, high) with 0 < low < high, got {self.bandHz}")
        if not self.windowS[0] < self.windowS[1]:
            raise ValueError(f"windowS must be (start, end) with start < end, got {self.windowS}")
        return self


class ArrivalWindowConfig(_Section):
    """Which picks may belong to a known event, relative to its public-catalog origin time.

    Best P / best S, violations, rho and Check B only use picks with
    ``origin - preOriginS <= t <= origin + hypocentralDist / minVelocityMps + postMarginS``.
    Every pick is still written to ``known/picks.parquet``.
    """

    preOriginS: float = Field(ge=0.0)  # allowance for public-catalog origin-time error
    minVelocityMps: float = Field(gt=0.0)  # slowest apparent velocity considered (bounds late S)
    postMarginS: float = Field(ge=0.0)  # added to the slowest travel time


class CheckBConfig(_Section):
    """Check B thresholds (docs/lanes/H1-signal.md, SEIS-04 acceptance)."""

    minStationsPS: int = Field(ge=1)  # stations with both a P and an S pick, per event
    minRho: float = Field(ge=-1.0, le=1.0)  # Spearman rho, best-P time vs epicentral distance
    minEventsPass: int = Field(ge=1)  # events that must pass for Check B to pass


class PickerABConfig(_Section):
    """Weight A/B on the known-event windows (``hq.pick.ab``)."""

    # Extra preprocessing profiles tried on the stations of a base profile. A station whose data
    # the variant's preprocessing rejects (e.g. a rate outside its range) is logged, counted and
    # left out of that variant. A variant is adopted only when it beats the base profile on the
    # same A/B metric over the same station-windows; stations it rejected keep the base profile.
    profileVariants: dict[str, list[str]]
    minStationsForRho: int = Field(ge=2)  # fewer stations with a P pick -> rho is reported as NaN
    arrivalWindow: ArrivalWindowConfig
    recordSection: RecordSectionConfig
    checkB: CheckBConfig


class PickerConfig(_Section):
    """PhaseNet weights, thresholds and gap-edge handling."""

    model: Literal["seisbench.PhaseNet"]
    weightsVersion: str = Field(min_length=1)  # pinned seisbench weight version (no remote lookup)
    candidateWeights: list[str] = Field(min_length=1)  # weight sets compared by the A/B
    defaultWeights: str  # ProcessingRun.pickerWeights
    # Weights used per preprocessing profile. hq.pick.ab writes its recommendation to
    # runs/<id>/known/ab.csv and ab.json; the chosen values are copied here by hand.
    weightsByProfile: dict[str, str]
    pThreshold: float = Field(gt=0.0, le=1.0)
    sThreshold: float = Field(gt=0.0, le=1.0)
    gapEdgeS: float = Field(ge=0.0)  # real s; picks this close to a block edge are dropped
    batchSize: int = Field(ge=1)
    torchThreads: int = Field(ge=1)
    seed: int
    seisbench: SeisbenchArgs
    ab: PickerABConfig

    @model_validator(mode="after")
    def _check(self) -> "PickerConfig":
        if len(set(self.candidateWeights)) != len(self.candidateWeights):
            raise ValueError(f"candidateWeights has duplicates: {self.candidateWeights}")
        known = set(self.candidateWeights)
        if self.defaultWeights not in known:
            raise ValueError(f"defaultWeights {self.defaultWeights!r} not in candidateWeights")
        unknown = {p: w for p, w in self.weightsByProfile.items() if w not in known}
        if unknown:
            raise ValueError(f"weightsByProfile uses weights not in candidateWeights: {unknown}")
        for base, variants in self.ab.profileVariants.items():
            if base in variants:
                raise ValueError(f"profileVariants[{base!r}] lists the base profile itself")
        return self


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

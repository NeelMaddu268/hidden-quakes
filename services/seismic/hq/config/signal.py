"""Signal lane config (``configs/showcase/signal.yaml``), owned by H1.

One section per ticket, so parallel tickets edit separate classes and separate YAML blocks.
Every model rejects unknown keys (docs/02 -> Config files).
"""

from typing import Annotated, Literal

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
    retries: int = Field(ge=0)  # extra attempts after a transport error or timeout
    backoffS: float = Field(ge=0)  # sleep backoffS * attempt before each retry


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
    # A reading of the station elevation matches when it is within toleranceM of the DEM. For a
    # sensor no deeper than toleranceM the two readings are indistinguishable: "surface" is used.
    toleranceM: float = Field(gt=0)
    # Sensor deeper than toleranceM and neither reading matches. "dem" uses the DEM as the surface.
    onAmbiguous: Literal["error", "surface", "dem", "skip"]
    # Shallow sensor whose station elevation misses the DEM by more than toleranceM: kept and
    # flagged up to maxShallowMismatchM; beyond it, onShallowMismatch decides.
    maxShallowMismatchM: float = Field(gt=0)
    onShallowMismatch: Literal["error", "dem", "keep", "skip"]

    @model_validator(mode="after")
    def _check(self) -> "ElevationCheck":
        if self.maxShallowMismatchM < self.toleranceM:
            raise ValueError(
                f"maxShallowMismatchM {self.maxShallowMismatchM} < toleranceM {self.toleranceM}"
            )
        return self


class AvailabilityCheck(_Section):
    """Per-channel data coverage of the window, from the MUSTANG daily availability metric."""

    url: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    timeoutS: float = Field(gt=0)
    retries: int = Field(ge=0)  # extra attempts after a transport error, HTTP 429 or 5xx
    backoffS: float = Field(ge=0)  # sleep backoffS * attempt before each retry
    cacheFile: str = Field(min_length=1)  # raw replies, JSON under the stationxml cache dir
    # A chosen channel-day with no measurement. MUSTANG lags about two days and never measures
    # some channels. "used": usedInRun stays true with coverage null, and SEIS-05's measured gaps
    # decide. "unused": the station leaves the run. "error": the stage stops.
    onMissing: Literal["error", "unused", "used"]


class RateCheck(_Section):
    """Checks each chosen triplet's metadata sample rate against a few seconds of served data.

    StationXML can disagree with the data: a channel epoch that says 200 Hz can serve 100 Hz.
    The preprocessing profile follows the sample rate, so a wrong rate sends the station through
    the wrong profile, and for_picking rejects it.
    """

    probeS: float = Field(gt=0)  # seconds of data per probe request
    probeOffsetsS: tuple[float, ...] = Field(min_length=1)  # probe starts after windowStart
    relTol: float = Field(gt=0, lt=0.01)  # data and metadata rates agree within this fraction
    timeoutS: float = Field(gt=0)
    retries: int = Field(ge=0)  # extra attempts after a transport error or timeout
    backoffS: float = Field(ge=0)  # sleep backoffS * attempt before each retry
    cacheFile: str = Field(min_length=1)  # probe results, JSON under the stationxml cache dir
    # Data rate differs from metadata. "data": use the served rate and its profile, and flag it.
    # "skip": drop the station. "error": stop.
    onMismatch: Literal["error", "data", "skip"]
    # No probe returned data. "metadata": keep the StationXML rate, flagged. "error": stop.
    onNoData: Literal["error", "metadata"]


class StationSelection(_Section):
    """Which stations and channel triplets enter the run, and how their sensor depth is resolved."""

    query: StationQuery
    channels: ChannelRules
    kind: KindRules
    profiles: tuple[ProfileRule, ...] = Field(min_length=1)
    elevation: ElevationCheck
    availability: AvailabilityCheck
    rateCheck: RateCheck


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


class ChunkConfig(_Section):
    """Hour-scale tiling of a picking window (``hq.preprocess.chunks``), shared by both pickers.

    ``[t0, t1)`` is cut into keep intervals on multiples of ``lengthS`` since the epoch. Each
    is read with ``overlapS`` extra real seconds on both sides and preprocessed on its own, and
    a picker keeps only the picks inside its keep interval. ``edgeProbeS`` extends every read a
    little further so a real data edge can be told apart from the chunk's own cut.
    """

    lengthS: float = Field(gt=0.0)  # keep interval length, real seconds
    overlapS: float = Field(gt=0.0)  # extra real seconds read before and after each keep interval
    minOverlapS: float = Field(gt=0.0)  # floor for overlapS (edge effects must stay outside keep)
    edgeProbeS: float = Field(gt=0.0)  # must exceed one input sample interval (checked per trace)

    @model_validator(mode="after")
    def _check(self) -> "ChunkConfig":
        if self.overlapS < self.minOverlapS:
            raise ValueError(
                f"chunks.overlapS {self.overlapS} is below chunks.minOverlapS {self.minOverlapS}"
            )
        if self.edgeProbeS >= self.overlapS:
            raise ValueError(
                f"chunks.edgeProbeS {self.edgeProbeS} must be below overlapS {self.overlapS}"
            )
        return self


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
    chunks: ChunkConfig

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


def _strictly_increasing(name: str, values: tuple[float, ...]) -> None:
    if list(values) != sorted(set(values)):
        raise ValueError(f"{name} must be strictly increasing, got {values}")


class BaselinePhase(_Section):
    """Recursive STA/LTA windows for one phase, in REAL seconds (also for time-stretched input)."""

    components: str = Field(min_length=1)  # model component letters, e.g. "Z" or "NE"
    staS: float = Field(gt=0.0)
    ltaS: float = Field(gt=0.0)
    warmupS: float = Field(gt=0.0)  # CF is zeroed this long after every segment start

    @model_validator(mode="after")
    def _check(self) -> "BaselinePhase":
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"components has duplicates: {self.components!r}")
        if self.staS >= self.ltaS:
            raise ValueError(f"staS {self.staS} must be below ltaS {self.ltaS}")
        if self.warmupS < self.ltaS:
            raise ValueError(f"warmupS {self.warmupS} must be at least ltaS {self.ltaS}")
        return self


class BaselineBandpass(_Section):
    """Causal Butterworth bandpass applied before the STA/LTA, in real Hz."""

    lowHz: float = Field(gt=0.0)
    highHz: float = Field(gt=0.0)
    corners: int = Field(ge=1)

    @model_validator(mode="after")
    def _check(self) -> "BaselineBandpass":
        if self.lowHz >= self.highHz:
            raise ValueError(f"lowHz {self.lowHz} must be below highHz {self.highHz}")
        return self


class BaselineThresholds(_Section):
    """STA/LTA trigger on/off levels (``trigger_onset`` thres1/thres2) for P and S."""

    pOn: float = Field(gt=0.0)
    pOff: float = Field(gt=0.0)
    sOn: float = Field(gt=0.0)
    sOff: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> "BaselineThresholds":
        if self.pOff >= self.pOn or self.sOff >= self.sOn:
            raise ValueError(f"off thresholds must be below on thresholds: {self}")
        return self


class BaselineSweep(_Section):
    """Grid pOn x sOn x offLevels; each off level is used for both phases at that grid point."""

    pOn: tuple[float, ...] = Field(min_length=1)
    sOn: tuple[float, ...] = Field(min_length=1)
    offLevels: tuple[float, ...] = Field(min_length=1)
    scoreWithH2: bool  # score every grid point through H2's associate..assign_tiers, once merged

    @model_validator(mode="after")
    def _check(self) -> "BaselineSweep":
        for name in ("pOn", "sOn", "offLevels"):
            _strictly_increasing(f"sweep.{name}", getattr(self, name))
        if self.offLevels[0] <= 0.0:
            raise ValueError(f"sweep.offLevels must be positive, got {self.offLevels}")
        if self.offLevels[-1] >= min(self.pOn[0], self.sOn[0]):
            raise ValueError("every sweep.offLevels value must be below every sweep on level")
        return self


class BaselineConfig(_Section):
    """Classical recursive STA/LTA picker and its threshold sweep (``hq.baseline``).

    The gap-edge distance is not here: the baseline uses ``picker.gapEdgeS``, the same number
    PhaseNet's full-window stage passes to ``hq.preprocess.chunks.pick_is_kept``.
    """

    prob: float = Field(ge=0.0, le=1.0)  # Pick.prob of every trigger; see signal.yaml
    minSegmentMarginS: float = Field(ge=0.0)  # segments shorter than warmupS + this are skipped
    maxWorkers: int = Field(ge=1)  # stations processed in parallel (threads); 1 = inline
    settleLtaMultiple: float = Field(gt=0.0)  # overlapS must hold this many ltaS before a keep
    prefilter: dict[str, BaselineBandpass] = Field(min_length=1)  # per preprocess profile
    p: BaselinePhase
    s: BaselinePhase
    pHorizontalTolS: float = Field(ge=0.0)  # per horizontal: a trigger on this near a P is the P
    minSMinusPS: float = Field(ge=0.0)  # S is the first horizontal trigger in
    maxSMinusPS: float = Field(gt=0.0)  # [tP + minSMinusPS, tP + maxSMinusPS]
    chosen: BaselineThresholds  # thresholds of picks_stalta.parquet; must be a sweep grid point
    sweep: BaselineSweep

    @model_validator(mode="after")
    def _check(self) -> "BaselineConfig":
        if self.minSMinusPS >= self.maxSMinusPS:
            raise ValueError(
                f"minSMinusPS {self.minSMinusPS} must be below maxSMinusPS {self.maxSMinusPS}"
            )
        shared = set(self.p.components) & set(self.s.components)
        if shared:
            raise ValueError(f"p and s components overlap: {sorted(shared)}")
        c, sw = self.chosen, self.sweep
        on_grid = (
            c.pOff == c.sOff and c.pOn in sw.pOn and c.sOn in sw.sOn and c.pOff in sw.offLevels
        )
        if not on_grid:
            raise ValueError(
                f"chosen {c.model_dump()} is not a sweep grid point: needs pOn in sweep.pOn, "
                "sOn in sweep.sOn and pOff == sOff in sweep.offLevels"
            )
        return self


class SignalConfig(_Section):
    """Contents of ``signal.yaml``."""

    stations: StationSelection
    download: DownloadConfig
    known: KnownEventsConfig
    preprocess: PreprocessConfig
    picker: PickerConfig
    baseline: BaselineConfig

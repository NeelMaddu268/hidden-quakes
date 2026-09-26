"""Seismology lane config (``configs/showcase/seismology.yaml``). Unknown keys are an error."""

import math
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


# docs/02 LocationQuality: hErrM is the "68% horizontal semi-major axis", vErrM "68% vertical".
DOCS02_ERR_CONFIDENCE = 0.68


def _is_multiple(value: float, step: float) -> bool:
    """True when ``value`` is an integer multiple of ``step`` (to float rounding)."""
    ratio = value / step
    return abs(ratio - round(ratio)) <= 1e-9 * max(1.0, abs(ratio))


class GridsConfig(BaseModel):
    """Per-station 2D (r, elevM) eikonal travel-time tables (LOC-02, ``hq.locate.tt_grid``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    drM: float = Field(gt=0)  # horizontal node spacing (m)
    dzM: float = Field(gt=0)  # vertical node spacing (m)
    rMaxM: float = Field(gt=0)  # largest epicentral distance in a table (m); a multiple of drM
    # The grid top is the highest receiver sensorElevM (or the search-volume top, if higher) plus
    # this margin, snapped up onto the dzM lattice anchored at bottomElevM. The velocity model's
    # top layer is extended up to it explicitly (with_top_extended_to) and that is recorded.
    topMarginM: float = Field(ge=0)
    bottomElevM: float  # grid bottom (m ASL); must lie below the search volume (checked below)
    # Near-source initialisation: nodes within this distance of the receiver get exact 1D layered
    # times, and the eikonal solve starts from an isochron inside that region.
    seedRadiusM: float = Field(gt=0)
    fmmOrder: Literal[1, 2]  # scikit-fmm stencil order
    # After each solve, the table is compared with the exact 1D layered times at every elevation
    # node and every accuracyCheckStrideR-th distance node; the max error per model layer is kept
    # in the table's sidecar and in the run record (the check never changes the table).
    accuracyCheckStrideR: int = Field(ge=1)

    @model_validator(mode="after")
    def _check(self) -> "GridsConfig":
        if not _is_multiple(self.rMaxM, self.drM):
            raise ValueError(f"rMaxM {self.rMaxM} must be a multiple of drM {self.drM}")
        if self.seedRadiusM < 4.0 * max(self.drM, self.dzM):
            raise ValueError("seedRadiusM must be at least 4 grid cells so the seed isochron spans cells")
        return self


class SearchVolumeConfig(BaseModel):
    """Where the locator searches, in ENU metres around the run origin and in elevM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    halfWidthM: float = Field(gt=0)  # e and n both span [-halfWidthM, +halfWidthM]
    # Top of the search volume (m ASL). null means run.refSurfaceElevM: with a 1D model and no DEM,
    # the ground at the origin is the one surface we know, so hypocentres stay below it. The top
    # is then snapped down onto the fine lattice, so it never lies above this value.
    topElevM: float | None
    bottomElevM: float  # m ASL


class PhaseSigma(BaseModel):
    """Pick-time uncertainty per phase (s)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    P: float = Field(gt=0)
    S: float = Field(gt=0)


class OutlierConfig(BaseModel):
    """Outlier pass: drop picks with |residual| > max(madK * MAD, floorS), then relocate once."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    madK: float = Field(gt=0)
    floorS: float = Field(ge=0)


class LocatorConfig(BaseModel):
    """Grid-search locator (``hq.locate.locator``) and its PDF uncertainty (``uncertainty``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    volume: SearchVolumeConfig
    coarseSpacingM: float = Field(gt=0)  # full-volume grid search
    fineSpacingM: float = Field(gt=0)  # the reported PDF lives on this lattice
    # First fine box: +/- this around the best coarse node, clipped to the volume. It is only where
    # the search starts: the PDF region grows past it until it covers the mass (see locator.py).
    fineHalfWidthM: float = Field(gt=0)
    # The misfit is first evaluated every fineStageSpacingM, from the first fine box outward until
    # the region within pdfCutoff of the minimum clears every face that is not a volume face; the
    # fine nodes are then evaluated over that region and grown the same way (see locator.py).
    fineStageSpacingM: float = Field(gt=0)
    # Nodes whose PDF exponent (pdfMisfitScale * misfit) exceeds its minimum by more than this carry
    # less than exp(-pdfCutoff) of the peak node's mass and are treated as zero mass.
    pdfCutoff: float = Field(gt=0)
    # PDF = exp(-pdfMisfitScale * misfit). 1.0 is the Laplace likelihood with scale sigma / prob;
    # it changes the formal errors only, never the hypocentre (see hq.locate.uncertainty).
    pdfMisfitScale: float = Field(gt=0)
    # Most fine nodes one PDF may cover. A PDF whose region needs more is truncated: hErrM and vErrM
    # become None and the search record says pdfTruncated (see locator.py).
    maxPdfNodes: int = Field(ge=1)
    # Pick sigma per phase (s) for every preprocessing profile, and per-profile overrides keyed by
    # Station.preprocessProfile (empty: every profile uses pickSigmaS).
    pickSigmaS: PhaseSigma
    profilePickSigmaS: dict[str, PhaseSigma]
    outlier: OutlierConfig
    minPicks: int = Field(ge=4)  # fewer picks cannot constrain (e, n, elevM, t0)
    # hErrM is the semi-major axis at this level. docs/02 defines hErrM and vErrM as 68%, and vErrM
    # is the 1-sigma of the vertical marginal, so only 0.68 keeps the two consistent (checked).
    errConfidence: float
    # depthOnEdge: more than this PDF mass on the fine grid's top or bottom face.
    depthOnEdgeMassFraction: float = Field(gt=0, lt=1)
    # mapOnVolumeTop / mapOnVolumeBottom (separate from depthOnEdge): the MAP node lies within this
    # distance (m) of the search volume's top / bottom (0: only on the face row itself).
    mapOnVolumeFaceBandM: float = Field(ge=0)
    nWorkers: int = Field(ge=1)  # processes for locate_many; results do not depend on it
    evalChunkNodes: int = Field(ge=1)  # nodes per misfit block (memory only; results unchanged)
    enuConsistencyTolM: float = Field(gt=0)  # |enu_u + origin elevM - sensorElevM| must be below

    @field_validator("errConfidence")
    @classmethod
    def _docs02_confidence(cls, value: float) -> float:
        if value != DOCS02_ERR_CONFIDENCE:
            raise ValueError(
                f"errConfidence must be {DOCS02_ERR_CONFIDENCE}: docs/02 defines hErrM and vErrM "
                f"as 68% and vErrM is the vertical 1-sigma, got {value}"
            )
        return value

    @model_validator(mode="after")
    def _check(self) -> "LocatorConfig":
        fine = self.fineSpacingM
        for label, value in (
            ("coarseSpacingM", self.coarseSpacingM),
            ("fineStageSpacingM", self.fineStageSpacingM),
            ("fineHalfWidthM", self.fineHalfWidthM),
        ):
            if not _is_multiple(value, fine):
                raise ValueError(f"{label} {value} must be a multiple of fineSpacingM {fine}")
        if not _is_multiple(self.volume.halfWidthM, self.coarseSpacingM):
            raise ValueError("volume.halfWidthM must be a multiple of coarseSpacingM")
        first_box = (2 * round(self.fineHalfWidthM / fine) + 1) ** 3
        if self.maxPdfNodes < first_box:
            raise ValueError(
                f"maxPdfNodes {self.maxPdfNodes} must hold the first fine box ({first_box} nodes)"
            )
        top = self.volume.topElevM
        if top is not None and top - self.volume.bottomElevM < self.coarseSpacingM:
            raise ValueError("the search volume must be at least one coarse cell tall")
        return self


class SyntheticZoneConfig(BaseModel):
    """Where synthetic hypocentres are drawn: a vertical cylinder in ENU metres and elevM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    centerEM: float
    centerNM: float
    radiusM: float = Field(gt=0)  # uniform in area within this horizontal radius
    topElevM: float  # m ASL, uniform in elevation between bottomElevM and topElevM
    bottomElevM: float

    @model_validator(mode="after")
    def _check(self) -> "SyntheticZoneConfig":
        if not self.bottomElevM < self.topElevM:
            raise ValueError("synthetic zone bottomElevM must lie below topElevM")
        return self


class SyntheticConfig(BaseModel):
    """Synthetic recovery test (``hq.locate.synthetic``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nEvents: int = Field(ge=1)
    seed: int = Field(ge=0)
    zone: SyntheticZoneConfig
    # Each station's S pick is kept with this probability; every synthetic pick gets pickProb.
    # None: stage locate measures both from the run's located events (hq.locate.synthetic
    # .measured_pick_stats); a direct run_synthetic call then needs a config with numbers.
    sKeepProb: Annotated[float, Field(ge=0, le=1)] | None
    pickProb: Annotated[float, Field(gt=0, le=1)] | None


class DatumCheckConfig(BaseModel):
    """Diagnostics row 2: noise-free synthetic events at known elevM through the real stations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    eM: float  # ENU east of the run origin (m)
    nM: float  # ENU north of the run origin (m)
    elevM: list[float] = Field(min_length=1)  # m ASL; one synthetic event per value
    passTolM: float = Field(gt=0)  # largest |elevM| and horizontal error that still passes (m)


class DiagnosticsConfig(BaseModel):
    """Depth diagnostics written to ``diagnostics.md`` by stage ``locate`` (LOC-04)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    datumCheck: DatumCheckConfig
    minSForDepth: int = Field(ge=1)  # row 4 splits events at nS >= this vs nS < this
    # Fewest events (row 4) or picks (rows 5-7) a group needs before a row concludes from it.
    minGroupSize: int = Field(ge=1)
    # Rows 4 and 5: a group's spread counts as worse than another's above this ratio.
    degradationRatio: float = Field(gt=1)
    # Row 6: a station-phase median residual above this (s) is flagged (lane doc: every static
    # above 0.15 s needs a written explanation).
    stationResidualFlagS: float = Field(gt=0)
    # Row 5: profiles whose Station.preprocessProfile starts with this are the borehole profiles
    # the row's suspect is about (signal.yaml names them borehole-A, borehole-B).
    boreholeProfilePrefix: str = Field(min_length=1)
    # Row 7: an azimuthal residual amplitude (s) above this counts as a trend (1D misses structure).
    trendFlagS: float = Field(gt=0)
    # Row 7: at the catalog hypocentres, an S/P ratio of the trend amplitudes above the model's
    # Vp/Vs at the source depths times this factor counts as S-heavy (see diagnostics.py).
    trendSPRatioExcess: float = Field(gt=1)
    # Table-vs-exact travel-time error (s) at a located hypocentre above this counts as exposure to
    # the cell-mean interface bias of the tables (LOC-02 accuracy record).
    tableErrorFlagS: float = Field(gt=0)


class WellConstrainedConfig(BaseModel):
    """Events whose residuals estimate selfConsistent statics (``hq.locate.statics``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minStations: int = Field(ge=1)  # quality.nStations at least this
    minS: int = Field(ge=0)  # quality.nS at least this
    maxGapDeg: float = Field(gt=0, le=360)  # quality.gapDeg at most this


class StaticsExplainConfig(BaseModel):
    """Evidence rules for the written explanation of every static above the flag threshold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # The median term of a station's nearest other stations (same phase, at most
    # neighbourMaxDistM away) explains its term as lateral structure when it has the term's sign
    # and at least this fraction of its size.
    neighbours: int = Field(ge=1)
    neighbourMaxDistM: float = Field(gt=0)
    lateralFraction: float = Field(gt=0, le=1)
    # S term / P term of one station (same sign), compared only when |P term| is at least this.
    minRatioTermS: float = Field(gt=0)
    # S/P within a factor ratioBand of the model's Vp/Vs at the sensor: a path (velocity) anomaly;
    # above it: the local Vp/Vs differs from the model's; within a factor ratioBand of 1: equal P
    # and S delays, a timing offset is possible.
    ratioBand: float = Field(gt=1)
    # An early term at a station farther than this (m) from the events' median epicentre: rays
    # bottoming in the model's deepest (extrapolated) layers; a hypothesis, contradicted (and then
    # not a verdict) when another station that far has a late term above the flag.
    farStationM: float = Field(gt=0)


class StaticsConfig(BaseModel):
    """Station statics (LOC-05, ``hq.locate.statics``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["selfConsistent", "referenceEvents"]
    # selfConsistent: median residual per station-phase over well-constrained events, subtracted,
    # events relocated; ``iterations`` times, each static capped at +/- capS.
    iterations: int = Field(ge=1)
    capS: float = Field(gt=0)
    minEvents: int = Field(ge=1)  # fewer well-constrained events on a station-phase: static 0
    wellConstrained: WellConstrainedConfig
    # referenceEvents: terms at the public-catalog hypocentres of the matched events.
    minReferenceEvents: int = Field(ge=1)  # fewer reference events on a station-phase: term 0
    referenceCapS: float = Field(gt=0)  # every term capped at +/- this
    folds: Annotated[int, Field(ge=2)] | None  # null: leave-one-out; k: k-fold
    polishIterations: int = Field(ge=1)  # origin-time / term alternations (median polish)
    # Stage locate without catalog.parquet or with too few reference events: fail, or locate
    # without statics with a recorded WARNING.
    referenceFallback: Literal["fail", "noStatics"]
    explain: StaticsExplainConfig
    # Robust residual sigma above this multiple of locator.pickSigmaS is reported as well above.
    sigmaFlagRatio: float = Field(gt=1)


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


class TolerancePair(BaseModel):
    """One (time, distance) tolerance for the match sensitivity table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dtS: float = Field(gt=0)  # |origin-time difference| limit and cost scale (s)
    distM: float = Field(gt=0)  # epicentral-distance limit and cost scale (m)


class UnmatchedReasonsConfig(BaseModel):
    """Evidence thresholds for explaining unmatched public events (``hq.match.reasons``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minStations: int = Field(ge=1)  # fewer stations than this with data/picks/association fails
    minPickProb: float = Field(gt=0, le=1)  # picks below this count only for "below threshold"
    arrivalPadS: float = Field(ge=0)  # widens each side of every expected arrival window (s)
    maxCandidateDtS: float = Field(gt=0)  # a located candidate's |dt| limit for "out of tolerance"


class MatchingConfig(BaseModel):
    """One-to-one matching of located events to the public regional catalog (stage ``match``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dtScaleS: float = Field(gt=0)  # cost = |dt| / dtScaleS + distance / distScaleM
    distScaleM: float = Field(gt=0)
    maxDtS: float = Field(gt=0)  # admissible only if |dt| <= maxDtS ...
    maxDistM: float = Field(gt=0)  # ... and epicentral distance <= maxDistM
    sensitivity: list[TolerancePair] = Field(min_length=1)  # each pair is limit and cost scale
    enuConsistencyM: float = Field(gt=0)  # stored ENU vs ENU from lat/lon/elevation (stage check)
    reasons: UnmatchedReasonsConfig

    @model_validator(mode="after")
    def _pairs(self) -> "MatchingConfig":
        pairs = [(p.dtS, p.distM) for p in self.sensitivity]
        if len(set(pairs)) != len(pairs):
            raise ValueError(f"sensitivity pairs must be unique, got {pairs}")
        if (self.maxDtS, self.maxDistM) not in pairs:
            raise ValueError(
                f"sensitivity pairs {pairs} must include the headline tolerance "
                f"(maxDtS, maxDistM) = ({self.maxDtS}, {self.maxDistM})"
            )
        return self


class TierQuantiles(BaseModel):
    """Per tier, the share of the matched set allowed to fall on the worse side of each bar."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    A: float = Field(ge=0, lt=1)  # 0.25: p25 (higher is better) / p75 (lower is better)
    B: float = Field(ge=0, lt=1)  # 0.0: the worst matched event itself

    @model_validator(mode="after")
    def _b_not_stricter(self) -> "TierQuantiles":
        if self.B > self.A:
            raise ValueError(f"quantiles.B {self.B} must not exceed quantiles.A {self.A}")
        return self


class TierSweepConfig(BaseModel):
    """The association sweep the tier stage scores (``hq.tier.sweep``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool  # true: stage tier reruns associate -> locate -> match -> tiers per point


class TieringConfig(BaseModel):
    """Quality tiers from matched-event quantiles (stage ``tier``, LOC-06, ``hq.tier``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quantiles: TierQuantiles
    # Tier A also needs a station with a used pick within this many focal depths (epicentral).
    strictNearestStationFactor: float = Field(gt=0)
    minMatched: int = Field(ge=1)  # fewer matched events than this: derivation fails loudly
    # Stored vs recomputed depthKm and nearest used station distance must agree within this (m).
    consistencyTolM: float = Field(gt=0)
    sweep: TierSweepConfig


class MagnitudeWindowConfig(BaseModel):
    """Where amplitudes are measured, relative to each station's P and S anchor times (s)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sPreS: float = Field(ge=0)  # the S window starts this long before the S anchor ...
    sPostS: float = Field(gt=0)  # ... and ends this long after it
    noiseLenS: float = Field(gt=0)  # noise window length, ending noiseGapS before the P anchor
    noiseGapS: float = Field(ge=0)
    # Data read beyond both windows on each side; cosine-tapered for the FFT, never measured.
    padS: float = Field(gt=0)


class ResponseRemovalConfig(BaseModel):
    """Instrument response removal to ground displacement (ObsPy evalresp in the sensor's input
    units + water level, then integrated to displacement)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ObsPy pre_filt: cosine frequency taper, 0 below f1, 1 between f2 and f3, 0 above f4 (Hz).
    preFiltHz: tuple[float, float, float, float]
    # ObsPy water_level, dB below the maximum of the response in the sensor's own input units
    waterLevelDb: float = Field(gt=0)
    # |response sample rate / data sample rate - 1| above this excludes the station.
    rateRelTol: float = Field(gt=0)

    @field_validator("preFiltHz")
    @classmethod
    def _increasing(cls, value: tuple[float, float, float, float]) -> tuple[float, ...]:
        if not 0.0 < value[0] < value[1] < value[2] < value[3]:
            raise ValueError(f"preFiltHz must be 0 < f1 < f2 < f3 < f4, got {list(value)}")
        return value


class SaturationConfig(BaseModel):
    """Digitizer clipping screen: a window whose raw horizontal counts reach ``maxFraction`` of
    ``fullScaleCounts`` anywhere in its processed span gets status ``clipped`` (no magnitude)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fullScaleCounts: float = Field(gt=0)  # largest |count| the digitizer can output
    maxFraction: float = Field(gt=0, le=1)


class WoodAndersonConfig(BaseModel):
    """The simulated Wood-Anderson torsion seismometer (displacement in, trace amplitude out)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    periodS: float = Field(gt=0)  # free period
    damping: float = Field(gt=0, lt=1)  # fraction of critical
    gain: float = Field(gt=0)  # static magnification


class MagnitudeFitConfig(BaseModel):
    """Robust least squares for M_cat = a log10(A) + b log10(R) + c + station term."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"]  # scipy least_squares loss
    fScaleMag: float = Field(gt=0)  # scipy f_scale: residual (magnitude units) where it turns
    # null fits a; a number fixes it (1.0 is the Richter definition: M scales with log10 A).
    amplitudeSlope: Annotated[float, Field(gt=0)] | None
    # Ridge constraint on the station terms: the objective adds stationTermRidge * sum(s_j^2)
    # (quadratic, outside the robust loss); (residual sd / station-term sd)^2 in Gaussian terms.
    stationTermRidge: float = Field(gt=0)
    # A station gets a term only with at least this many calibration observations; stations with
    # fewer are left out of the fit and of every magnitude from it (logged).
    minStationObs: int = Field(ge=1)


class MagnitudeConfig(BaseModel):
    """Local magnitude calibrated on matched public events (stage ``magnitude``, MAG-01)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    calibrationMagType: str = Field(min_length=1)  # the one CatalogEvent.magType calibrated on
    maxLooMae: float = Field(gt=0)  # magnitudes are written only when the LOO MAE is at most this
    # Usable station amplitudes (above minSnr, station with a term) an event needs for a
    # magnitude; calibration events need as many.
    minStations: int = Field(ge=1)
    minCalibrationEvents: int = Field(ge=2)  # fewer calibration events: the stage fails
    minSnr: float = Field(gt=0)  # S-window peak / noise-window peak, on the same processed trace
    readChunkS: float = Field(gt=0)  # longest span read from the cache at once per station
    window: MagnitudeWindowConfig
    response: ResponseRemovalConfig
    saturation: SaturationConfig
    woodAnderson: WoodAndersonConfig
    fit: MagnitudeFitConfig


class SeismologyConfig(BaseModel):
    """Contents of ``seismology.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    velocity: VelocityConfig
    grids: GridsConfig
    locator: LocatorConfig
    synthetic: SyntheticConfig
    catalog: CatalogConfig
    associator: AssociatorConfig
    matching: MatchingConfig
    diagnostics: DiagnosticsConfig
    statics: StaticsConfig
    tiering: TieringConfig
    magnitude: MagnitudeConfig

    @model_validator(mode="after")
    def _consistent(self) -> "SeismologyConfig":
        vol = self.locator.volume
        # Every associated event has at least minStations stations, so at least that many picks;
        # the locator needs minPicks. Checked here so no associated event is left unlocatable.
        fewest = min(self.associator.minStations, *self.associator.sweep.minStations)
        if fewest < self.locator.minPicks:
            raise ValueError(
                f"associator minStations (smallest, sweep included) {fewest} is below "
                f"locator.minPicks {self.locator.minPicks}: such events could not be located"
            )
        datum = self.diagnostics.datumCheck
        if max(abs(datum.eM), abs(datum.nM)) > vol.halfWidthM or min(datum.elevM) < vol.bottomElevM:
            raise ValueError("diagnostics.datumCheck points must lie inside the search volume")
        if vol.topElevM is not None and max(datum.elevM) > vol.topElevM:
            raise ValueError("diagnostics.datumCheck elevM must lie below the search volume top")
        if self.grids.bottomElevM > vol.bottomElevM - self.grids.dzM:
            raise ValueError(
                f"grids.bottomElevM {self.grids.bottomElevM} must lie at least one dzM below "
                f"locator.volume.bottomElevM {vol.bottomElevM}"
            )
        corner = math.hypot(vol.halfWidthM, vol.halfWidthM)
        if corner >= self.grids.rMaxM:
            raise ValueError("grids.rMaxM must exceed the search volume's half diagonal")
        zone = self.synthetic.zone
        reach = math.hypot(zone.centerEM, zone.centerNM) + zone.radiusM
        if reach > vol.halfWidthM or zone.bottomElevM < vol.bottomElevM:
            raise ValueError("the synthetic zone must lie inside the search volume")
        if vol.topElevM is not None and zone.topElevM > vol.topElevM:
            raise ValueError("the synthetic zone must lie below the search volume top")
        return self

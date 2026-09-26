"""Export section: what the exporter writes and how (``configs/showcase/export.yaml``).

Consumed by ``hq.export`` (API-02) and ``hq.export.features`` (FEAT-01). Every value the exporter
uses is a field here, with its default documented in ``export.yaml``; nothing is hard-coded in
the exporter. Unknown keys are an error.

Sections: ``modes``, ``heroRule``, ``outputDir``, ``scene`` (SceneMeta knobs), ``evidence``
(snippets), ``rounding`` (decimals of every value the exporter derives, so a bundle is
byte-identical run to run) and ``features`` (FEAT-01).

Reference features (``features:``) are context only: wells, well pads and site outlines near the
region, each cited to the public document its coordinates were read from. They say nothing about
what caused any seismicity. Every coordinate stays in the datum and unit it was published in;
``hq.export.features`` converts to ENU at export time.
"""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Modes the exporter can write a bundle for. ``mock`` is excluded on purpose: synthetic bundles
# come only from ``scripts/mock-fixture.py`` (CLAUDE.md rule 5).
ExportMode = Literal["showcase", "live", "snapshot"]

# How ``SceneMeta.heroEventId`` is chosen.
#   tierA_most_stations: the Tier A event located with the most stations.
HeroRule = Literal["tierA_most_stations"]

# ``EventEvidence.traces`` holds at most this many snippets (docs/02 §1); the config cap.
MAX_EVIDENCE_TRACES = 16


class SceneConfig(BaseModel):
    """``SceneMeta`` knobs that are not derived from the run (the rest come from ``run.yaml``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verticalExaggeration: float = Field(default=1.0, gt=0.0)  # != 1 shows a "Vertical xN" badge
    # SceneMeta.depthLabel; ``{refSurfaceElevM}`` is filled from run.yaml (docs/01 -> Conventions).
    depthLabel: str = Field(
        default="Depth below site surface (ref {refSurfaceElevM:g} m ASL)", min_length=1
    )


class RoundingConfig(BaseModel):
    """Decimals of every value the exporter derives. Pipeline values are written as stored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timeS: int = Field(default=3, ge=0)  # WaveformSnippet.t0
    distM: int = Field(default=1, ge=0)  # WaveformSnippet.epiDistM
    sample: int = Field(default=3, ge=1)  # WaveformSnippet.samples (in [-1, 1])
    recall: int = Field(default=4, ge=1)  # AnalysisSummary.recall
    rmsS: int = Field(default=3, ge=1)  # AnalysisSummary.medianRmsS
    gain: int = Field(default=2, ge=1)  # AnalysisSummary.baseline.gain


class EvidenceConfig(BaseModel):
    """Waveform snippets for the evidence drawer (``EventEvidence`` / ``WaveformSnippet``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    beforeS: float = Field(default=1.5, gt=0.0)  # s of waveform before the predicted P at a station
    afterS: float = Field(default=3.5, gt=0.0)  # s after it; a snippet is beforeS + afterS long
    minLengthS: float = Field(default=4.0, gt=0.0)  # shortest snippet the drawer should show
    maxLengthS: float = Field(default=8.0, gt=0.0)  # longest; keeps evidence files small
    bandHz: tuple[float, float] = (2.0, 20.0)  # zero-phase bandpass (low, high) of the display copy
    maxTraces: int = Field(default=MAX_EVIDENCE_TRACES, ge=1, le=MAX_EVIDENCE_TRACES)
    displayRateHz: float = Field(default=100.0, gt=0.0)  # WaveformSnippet.dt = 1 / displayRateHz
    # How many evidence files the web app fetches up front is the web app's own knob
    # (apps/web/src/providers/config.ts EVIDENCE_PRELOAD_COUNT); the exporter has no say in it.
    # Extra seconds read on both sides of the window before filtering, so the display copy's
    # detrend/taper and the resampler never touch the samples that end up in the snippet.
    padS: float = Field(default=1.0, ge=0.0)
    # Component codes tried in order (last character of Station.channels); the first channel of
    # the station that matches and has data in the window becomes the trace.
    channelPriority: list[str] = Field(default_factory=lambda: ["Z"], min_length=1)
    # Hard cap on one evidence file (docs/01: 20-60 KB). A file over it drops its farthest trace
    # until it fits; the run fails if a single trace does not fit.
    maxFileBytes: int = Field(default=60 * 1024, gt=0)
    # Events that get an evidence file, in reveal order after the hero; null means every event
    # that has arrivals. The bundle is committed to git, so cap it when the run is large.
    maxEvents: int | None = Field(default=400, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "EvidenceConfig":
        if self.minLengthS > self.maxLengthS:
            raise ValueError(
                f"minLengthS {self.minLengthS} must not exceed maxLengthS {self.maxLengthS}"
            )
        length = self.beforeS + self.afterS
        if not self.minLengthS <= length <= self.maxLengthS:
            raise ValueError(
                f"beforeS + afterS = {length} s must lie within "
                f"[minLengthS, maxLengthS] = [{self.minLengthS}, {self.maxLengthS}] s"
            )
        low, high = self.bandHz
        if not 0.0 < low < high:
            raise ValueError(f"bandHz must be (low, high) with 0 < low < high, got {self.bandHz}")
        if high * 2.0 > self.displayRateHz:
            raise ValueError(
                f"displayRateHz {self.displayRateHz} cannot show bandHz top {high} Hz "
                f"(needs at least {high * 2.0} Hz, the Nyquist rate)"
            )
        bad = [c for c in self.channelPriority if len(c) != 1]
        if bad or len(set(self.channelPriority)) != len(self.channelPriority):
            raise ValueError(
                f"channelPriority must be distinct single component codes, got {self.channelPriority}"
            )
        return self


# Coordinate reference systems a feature may be published in. ``hq.export.features`` converts each
# to WGS84 latitude/longitude with pyproj, then to ENU through H2's ``hq.locate.coords``.
#   EPSG:4326   WGS84 latitude/longitude, degrees
#   EPSG:26912  NAD83 / UTM zone 12N, metres (the datum FORGE and UGS GIS products ship in)
#   EPSG:3742   NAD83(HARN) / UTM zone 12N, metres (the datum the 16B(78)-32 survey report states)
#   EPSG:32612  WGS84 / UTM zone 12N, metres (the project's own scene projection)
FeatureCrs = Literal["EPSG:4326", "EPSG:26912", "EPSG:3742", "EPSG:32612"]
PROJECTED_FEATURE_CRS: frozenset[str] = frozenset({"EPSG:26912", "EPSG:3742", "EPSG:32612"})

# Unit of the published horizontal coordinates, depths and elevations of one feature.
#   deg   degrees; only with EPSG:4326 (elevations stay in metres)
#   m     metres
#   usft  US survey feet (1200/3937 m), what COMPASS survey reports print
LengthUnit = Literal["deg", "m", "usft"]


class FeatureSource(BaseModel):
    """Where a feature's coordinates were read from. Becomes the contract's ``SourceRef``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation: str = Field(min_length=1)  # title, author/publisher, date of the publication
    # The authoritative publication; or, when ``primaryLocated`` is false, the copy that was read.
    url: str = Field(min_length=1, pattern=r"^https?://")
    accessedOn: date  # when the coordinates were read
    # The page actually read when it is not ``url`` itself (a mirror or copy of the publication).
    # A feature read through a mirror can never be ``verified``.
    readFrom: str | None = Field(default=None, min_length=1, pattern=r"^https?://")
    # False when the publication the coordinates originally come from could not be named: ``url``
    # is then the copy that was read, ``readFrom`` stays unset, and the feature is never
    # ``verified``. The bundle citation says so.
    primaryLocated: bool = True
    note: str = ""  # anything a reader needs to judge the numbers (datum caveats, what was checked)

    @model_validator(mode="after")
    def _check_source(self) -> "FeatureSource":
        if not self.primaryLocated and self.readFrom is not None:
            raise ValueError(
                "primaryLocated=false means url is the copy that was read; readFrom must be unset"
            )
        return self


class FeatureBase(BaseModel):
    """Fields every reference feature carries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")  # GeoFeature.id
    name: str = Field(min_length=1)  # display name; never a claim about what the feature does
    # True only when the coordinates were read from the authoritative publication (``source.url``)
    # itself. False means "location is approximate"; H3 renders it as such.
    verified: bool
    source: FeatureSource
    crs: FeatureCrs
    unit: LengthUnit

    @model_validator(mode="after")
    def _check_base(self) -> "FeatureBase":
        if self.verified and self.source.readFrom is not None:
            raise ValueError(
                f"feature {self.id!r}: verified=true requires the coordinates to be read from "
                f"source.url itself, but readFrom={self.source.readFrom!r} says they came from "
                "a copy; set verified=false or read them from the primary"
            )
        if self.verified and not self.source.primaryLocated:
            raise ValueError(
                f"feature {self.id!r}: verified=true requires the primary publication "
                "(source.primaryLocated=true); a copy of unknown origin is never verified"
            )
        if (self.crs == "EPSG:4326") != (self.unit == "deg"):
            raise ValueError(
                f"feature {self.id!r}: unit 'deg' goes with crs 'EPSG:4326' and a length unit "
                f"with a projected crs, got crs={self.crs!r} unit={self.unit!r}"
            )
        return self


class PointFeatureConfig(FeatureBase):
    """One point: a wellhead, pad or facility. ``GeoFeature.kind = "facility"``."""

    kind: Literal["facility"]
    x: float  # longitude (deg) or easting (unit)
    y: float  # latitude (deg) or northing (unit)
    elevM: float  # ground elevation at the point, m ASL


class PolygonFeatureConfig(FeatureBase):
    """A closed outline drawn at one elevation. ``GeoFeature.kind = "boundary"``."""

    kind: Literal["boundary"]
    # [x, y] vertices in ``crs``/``unit``, in ring order, first vertex not repeated.
    vertices: list[tuple[float, float]] = Field(min_length=3)
    # Elevation the outline is drawn at, m ASL; null draws it at the run's refSurfaceElevM.
    elevM: float | None = None

    @model_validator(mode="after")
    def _check_ring(self) -> "PolygonFeatureConfig":
        if self.vertices[0] == self.vertices[-1]:
            raise ValueError(f"feature {self.id!r}: do not repeat the first vertex; rings close")
        return self


class SurveyRow(BaseModel):
    """One directional-survey station. Lengths in the well's ``unit``; angles in degrees."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    md: float = Field(ge=0.0)  # measured depth along the hole from ``depthRefElev``
    incDeg: float = Field(ge=0.0, le=180.0)  # inclination from vertical
    aziDeg: float = Field(ge=0.0, le=360.0)  # azimuth clockwise from grid north (``northRef``)
    # Positions the report itself printed, if any; the loader recomputes them with minimum
    # curvature and refuses the well when they disagree by more than ``maxSurveyMismatchM``.
    tvd: float | None = None  # true vertical depth below ``depthRefElev``
    ns: float | None = None  # displacement north (+) of the wellhead
    ew: float | None = None  # displacement east (+) of the wellhead

    @model_validator(mode="after")
    def _check_published(self) -> "SurveyRow":
        published = (self.tvd, self.ns, self.ew)
        if any(v is None for v in published) and any(v is not None for v in published):
            raise ValueError("tvd, ns and ew are published together or not at all")
        return self


class WellFeatureConfig(FeatureBase):
    """A well trajectory from a directional survey. ``GeoFeature.kind = "well"``.

    The path is ``wellhead + minimum-curvature displacement`` per survey station, with
    ``elevM = depthRefElev - tvd`` (docs/01: elevM up-positive). Exactly one of ``survey`` and
    ``surveyCsv`` supplies the stations.
    """

    kind: Literal["well"]
    headX: float  # wellhead easting in ``unit`` (a projected ``crs`` is required)
    headY: float  # wellhead northing in ``unit``
    depthRefElev: float  # elevation of md = 0 (kelly bushing / rig floor), in ``unit``, ASL
    groundElev: float | None = None  # ground level at the wellhead, in ``unit``, ASL (context)
    northRef: Literal["grid"]  # survey azimuths must be grid-north referenced in ``crs``
    # Largest allowed distance, m, between a published station position and the one recomputed
    # from md/inc/azi. Catches transcription errors in the survey table.
    maxSurveyMismatchM: float = Field(gt=0.0)
    survey: list[SurveyRow] | None = Field(default=None, min_length=2)
    # CSV with columns md,incDeg,aziDeg[,tvd,ns,ew]; ``#`` lines are comments. Relative paths
    # resolve against the ``services/seismic`` directory (where ``configs/`` lives).
    surveyCsv: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_well(self) -> "WellFeatureConfig":
        if self.crs not in PROJECTED_FEATURE_CRS:
            raise ValueError(f"well {self.id!r}: a well needs a projected crs, got {self.crs!r}")
        if (self.survey is None) == (self.surveyCsv is None):
            raise ValueError(f"well {self.id!r}: give exactly one of survey and surveyCsv")
        return self


# ``kind`` picks the model, so a wrong key is reported against the right model.
FeatureConfig = Annotated[
    PointFeatureConfig | PolygonFeatureConfig | WellFeatureConfig, Field(discriminator="kind")
]


class ExportConfig(BaseModel):
    """Contents of ``export.yaml``. Unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    modes: list[ExportMode] = Field(default_factory=lambda: ["showcase"], min_length=1)
    heroRule: HeroRule = "tierA_most_stations"
    # Where bundles go: <outputDir>/<mode>/. Relative to the checkout that holds the ``hq``
    # package (the bundle is committed to git), or absolute.
    outputDir: str = Field(default="apps/web/public/data", min_length=1)
    # Cap on one whole bundle directory (docs/01 budgets the committed web bundle at < 30 MB);
    # the export fails before swapping the bundle in, and check_bundle flags it.
    maxBundleBytes: int = Field(default=30 * 1024 * 1024, gt=0)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    rounding: RoundingConfig = Field(default_factory=RoundingConfig)
    # Geothermal reference features (wells, well pads, boundaries), each with a cited source.
    # Empty means "export no features". Converted to ``GeoFeature`` by ``hq.export.features``.
    features: list[FeatureConfig] = Field(default_factory=list)
    # Decimal places of the ENU metres written for each feature point. Survey reports print
    # positions to 0.01 ft, so 2 (centimetres) loses nothing and keeps features.json byte-stable.
    featureDecimals: int = Field(default=2, ge=0, le=6)

    @model_validator(mode="after")
    def _check(self) -> "ExportConfig":
        if len(set(self.modes)) != len(self.modes):
            raise ValueError(f"modes must not repeat, got {self.modes}")
        ids = [feature.id for feature in self.features]
        if len(set(ids)) != len(ids):
            raise ValueError(f"feature ids must be unique, got {ids}")
        return self

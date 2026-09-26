"""Export section: what the exporter writes and how (``configs/showcase/export.yaml``).

Consumed by ``hq.export`` (API-02) and ``hq.export.features`` (FEAT-01). Every value the exporter
uses is a field here, with its default documented in ``export.yaml``; nothing is hard-coded in
the exporter. Unknown keys are an error.

Sections: ``modes``, ``heroRule``, ``outputDir``, ``scene`` (SceneMeta knobs), ``evidence``
(snippets), ``rounding`` (decimals of every value the exporter derives, so a bundle is
byte-identical run to run) and ``features`` (FEAT-01).
"""

from typing import Any, Literal

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
    preloadCount: int = Field(default=20, ge=0)  # evidence files the web app fetches up front
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
    maxEvents: int | None = Field(default=None, ge=1)

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


class ExportConfig(BaseModel):
    """Contents of ``export.yaml``. Unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    modes: list[ExportMode] = Field(default_factory=lambda: ["showcase"], min_length=1)
    heroRule: HeroRule = "tierA_most_stations"
    # Where bundles go: <outputDir>/<mode>/. Relative to the checkout that holds the ``hq``
    # package (the bundle is committed to git), or absolute.
    outputDir: str = Field(default="apps/web/public/data", min_length=1)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    rounding: RoundingConfig = Field(default_factory=RoundingConfig)
    # Geothermal reference features (wells, facilities, boundaries), each with a SourceRef.
    # Placeholder until FEAT-01 gives it a typed model; empty means "export no features".
    features: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ExportConfig":
        if len(set(self.modes)) != len(self.modes):
            raise ValueError(f"modes must not repeat, got {self.modes}")
        return self

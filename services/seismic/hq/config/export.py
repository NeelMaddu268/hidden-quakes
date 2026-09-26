"""Export section: what the exporter writes and how (``configs/showcase/export.yaml``).

Consumed by ``hq.export`` (API-02) and ``hq.export.features`` (FEAT-01). Every value the exporter
uses is a field here, with its default documented in ``export.yaml``; nothing is hard-coded in
the exporter. Unknown keys are an error.
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


class EvidenceConfig(BaseModel):
    """Waveform snippets for the evidence drawer (``EventEvidence`` / ``WaveformSnippet``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    beforeS: float = Field(default=2.0, gt=0.0)  # s of waveform before the predicted P at a station
    afterS: float = Field(default=4.0, gt=0.0)  # s after it; a snippet is beforeS + afterS long
    minLengthS: float = Field(default=4.0, gt=0.0)  # shortest snippet the drawer should show
    maxLengthS: float = Field(default=8.0, gt=0.0)  # longest; keeps evidence files small
    bandHz: tuple[float, float] = (2.0, 20.0)  # zero-phase bandpass (low, high) of the display copy
    maxTraces: int = Field(default=MAX_EVIDENCE_TRACES, ge=1, le=MAX_EVIDENCE_TRACES)
    displayRateHz: float = Field(default=100.0, gt=0.0)  # WaveformSnippet.dt = 1 / displayRateHz
    preloadCount: int = Field(default=20, ge=0)  # evidence files the web app fetches up front

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
        return self


class ExportConfig(BaseModel):
    """Contents of ``export.yaml``. Unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    modes: list[ExportMode] = Field(default_factory=lambda: ["showcase"], min_length=1)
    heroRule: HeroRule = "tierA_most_stations"
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    # Geothermal reference features (wells, facilities, boundaries), each with a SourceRef.
    # Placeholder until FEAT-01 gives it a typed model; empty means "export no features".
    features: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ExportConfig":
        if len(set(self.modes)) != len(self.modes):
            raise ValueError(f"modes must not repeat, got {self.modes}")
        return self

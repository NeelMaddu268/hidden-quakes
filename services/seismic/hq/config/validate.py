"""Validate section: the validation reruns H4 makes through H2's library API
(``configs/showcase/validate.yaml``). Unknown keys are an error.

Consumed by ``hq.validate`` (VAL-02 null test now; VAL-01 baseline table, sweep, G-R later).
Every value the stage uses is a field here, with its default documented in ``validate.yaml``;
nothing is hard-coded in the stage. The H2 knobs the reruns share (associator, locator, tiering,
matching) stay in ``seismology.yaml``: a rerun always uses the same ``SeismologyConfig`` as the
run of record, and this file only says how the inputs are perturbed or which picks go in.

An *association profile* names which picks are fed to ``associate``: ``full`` is every pick,
``p_only`` keeps P picks only (docs/02 ``BaselineRow.associationProfile``; the STA/LTA baseline
has no S picks, so the fair comparison feeds PhaseNet the same phase set).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AssociationProfile = Literal["full", "p_only"]
ASSOCIATION_PROFILES: tuple[AssociationProfile, ...] = ("full", "p_only")


class NullTestConfig(BaseModel):
    """Null test (VAL-02): rerun associate -> locate -> match -> assign_tiers on picks whose
    stations were shifted in time independently, so only chance coincidences survive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Reruns. The std is the sample standard deviation across reruns, so at least 2 are needed.
    nShuffles: int = Field(default=20, ge=2)
    # Each station's picks move together by one draw from uniform(-shiftS, +shiftS) seconds per
    # rerun; within-station P/S structure survives, inter-station coherence does not.
    shiftS: float = Field(default=30.0, gt=0.0)
    # Root seed; rerun i draws from ``numpy.random.default_rng([seed, i])``.
    seed: int = Field(default=0, ge=0)
    # Which picks go into the reruns (see the module docstring).
    profile: AssociationProfile = "full"


class BaselineConfig(BaseModel):
    """Baseline comparison (VAL-01): two pickers x two association profiles. Stub until VAL-01;
    the defaults name what docs/lanes/H4 asks for."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Association profiles every picker is run through; ``BaselineGain`` needs both.
    profiles: list[AssociationProfile] = Field(
        default_factory=lambda: list(ASSOCIATION_PROFILES), min_length=1
    )


class GRConfig(BaseModel):
    """Gutenberg-Richter curve (VAL-01). Stub until VAL-01; docs/lanes/H4 fixes the method:
    Aki-Utsu b with Shi-Bolt sigma, Mc by maximum curvature plus an offset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    magBinWidth: float = Field(default=0.1, gt=0.0)  # width of GRCurve.magBins
    mcOffsetMag: float = Field(default=0.2, ge=0.0)  # added to the maximum-curvature Mc


class ValidateConfig(BaseModel):
    """Contents of ``validate.yaml``. Unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nullTest: NullTestConfig = Field(default_factory=NullTestConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    gr: GRConfig = Field(default_factory=GRConfig)

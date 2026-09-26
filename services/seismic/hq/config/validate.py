"""Validate section: the validation reruns H4 makes through H2's library API
(``configs/showcase/validate.yaml``). Unknown keys are an error.

Consumed by ``hq.validate`` (VAL-02 null test; VAL-01 baseline table and G-R curve).
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
    """Baseline comparison (VAL-01): two pickers (PhaseNet, STA/LTA) x two association
    profiles through H2's associate -> locate -> match -> assign_tiers, each with the run's own
    ``SeismologyConfig`` (``hq.validate.baseline``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Association profiles every picker is run through. ``BaselineGain`` is quoted on ``full``
    # and only when the gain holds in both ``full`` and ``p_only``, so both must be listed for
    # the summary to carry one; fewer profiles still fill the table.
    profiles: list[AssociationProfile] = Field(
        default_factory=lambda: list(ASSOCIATION_PROFILES), min_length=1
    )
    # The gain (PhaseNet Tier A count / STA/LTA Tier A count) must exceed this in every profile
    # for ``BaselineGain`` to be claimed; the docs/lanes/H4 rule is "gain > 1".
    minGain: float = Field(default=1.0, ge=1.0)
    # docs/03 baseline kill switch: STA/LTA "within ~20% of PhaseNet's strict count" is logged
    # as a warning when |strictPhasenet - strictStalta| <= comparableFraction * strictPhasenet.
    # Logged only; it never decides the gain (that is the ``minGain`` rule above).
    comparableFraction: float = Field(default=0.2, ge=0.0, le=1.0)


class GRConfig(BaseModel):
    """Gutenberg-Richter curve (VAL-01, ``hq.validate.gr``). docs/lanes/H4 fixes the method:
    Aki-Utsu b with Shi-Bolt sigma, Mc by maximum curvature plus an offset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Width of GRCurve.magBins (magnitude units); also the histogram bin of the maximum-curvature
    # Mc and the ``bin_width / 2`` binning correction of the Aki-Utsu estimator.
    magBinWidth: float = Field(default=0.1, gt=0.0)
    # Added to the maximum-curvature magnitude of completeness (Mc = MaxC + offset).
    mcOffsetMag: float = Field(default=0.2, ge=0.0)
    # Fewest magnitudes for any estimate: below this the curve is not built at all (public and
    # recovered together), a set's Mc is null, and b / sigma are null when fewer than this many
    # recovered magnitudes lie above Mc.
    minEvents: int = Field(default=30, ge=2)
    # docs/03 magnitude kill switch: when H2's MagCalibration.looMae exceeds this, G-R is skipped
    # (with a warning) and only the calibration is embedded. H2's magnitude stage applies its own
    # gate (seismology.yaml magnitude.maxLooMae, recorded in ProcessingRun.matching["magnitude"]
    # ["gate"]["maxLooMae"]); when that record exists the stage applies H2's value and warns if
    # this one differs, so the two gates never disagree on a run (REQ-H2-13).
    maxLooMae: float = Field(default=0.4, gt=0.0)
    # The one CatalogEvent.magType the public curve (GRCurve.publicCum) is drawn from, so public
    # magnitudes are never binned across scales next to the candidates' calibrated magnitude
    # (REQ-H2-13). Used only when ProcessingRun.matching["magnitude"]["calibrationMagType"] is
    # absent (H2's magnitude stage has not run, or failed); with public magnitudes present and
    # neither, the stage fails. None: no fallback.
    publicMagType: str | None = Field(default=None, min_length=1)


class POnlyAssociatorConfig(BaseModel):
    """Associator fields overridden for the ``p_only`` profile (REQ-H2-7). The run's own
    ``SeismologyConfig.associator`` requires S picks (``nSPicks``, ``nPAndSPicks``), so feeding
    P-only picks through it associates nothing; the p_only reruns use a copy with these values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nSPicks: int = Field(default=0, ge=0)  # PyOcto n_s_picks for the p_only profile
    nPAndSPicks: int = Field(default=0, ge=0)  # PyOcto n_p_and_s_picks for the p_only profile


class ValidateConfig(BaseModel):
    """Contents of ``validate.yaml``. Unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nullTest: NullTestConfig = Field(default_factory=NullTestConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    gr: GRConfig = Field(default_factory=GRConfig)
    # Associator overrides for every p_only rerun (null test and baseline); the ``full`` profile
    # always uses the run's ``SeismologyConfig`` unchanged.
    pOnlyAssociator: POnlyAssociatorConfig = Field(default_factory=POnlyAssociatorConfig)

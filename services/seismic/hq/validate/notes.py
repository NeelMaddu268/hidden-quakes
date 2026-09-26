"""``validation_notes.json``: what the validation reruns could and could not do, next to the
numbers in ``null_test.json``, ``baseline.json`` and ``gr.json``.

The docs/02 contract models (``NullTest``, ``BaselineRow``, ``GRCurve``) are frozen and have no
room for provenance, so the stage writes this sidecar beside them (docs/02 §2). It is H4's own
model, not a contract: the exporter and the panels may read it, nothing else depends on it.

- ``nullTest`` / ``baseline`` (``RerunNotes``): the bars the reruns tiered against (the run's
  own, from ``ProcessingRun.tiering["thresholds"]``, REQ-H2-9), the Tier A rules H2's
  ``assign_tiers`` reports it applied (``tiering["rules"]``: how the nearest-station rule
  measured focal depth, whether the ``mapOnVolumeTop`` rule ran), that no station statics were
  applied (``hq.locate.locate`` has no match pass, FYI-H2-8), and the associator overrides the
  ``p_only`` profile reruns with (REQ-H2-7).
- ``gr`` (``GRNotes``): the one magnitude scale the public curve is drawn on and how many public
  magnitudes of other types were left out (REQ-H2-13), the kill-switch gate actually applied,
  and H2's leave-one-event-out MAE next to its null model (FYI-H2-7).

``rerun_notes`` reads the tiering dicts the reruns returned: they must carry the keys H2's
``assign_tiers`` documents, and a dict without them fails naming H2.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hq.validate.errors import ValidateError

log = logging.getLogger(__name__)

H2 = "H2 Seismology"
THRESHOLDS_RECORD = "ProcessingRun.tiering.thresholds"  # where the run's bars live (H2's tier stage)
FOCAL_DEPTH_SENSOR = "nearestUsedSensor"  # H2's value when arrivals= and stations= were given
FOCAL_DEPTH_REFERENCE = "refSurfaceElevM"  # H2's fallback value (the docs/02 3-argument call)
NO_STATICS_NOTE = (
    "Rerun events are located by hq.locate.locate, which has no match pass, so no station "
    "statics are applied (statics.mode referenceEvents needs one); rerun rmsS values are larger "
    "than the run's and the run's bars are harder for rerun events to meet (FYI-H2-8)."
)
MAP_ON_TOP_SKIPPED_NOTE = (
    "The Tier A mapOnVolumeTop rule was not applied: hq.locate.locate returns no locate flags. A "
    "skipped rule only lets more rerun events into Tier A (REQ-H2-9)."
)
MAP_ON_TOP_APPLIED_NOTE = "The Tier A mapOnVolumeTop rule was applied, as stage tier does."
FOCAL_SENSOR_NOTE = (
    "The Tier A nearest-station rule measured focal depth below the nearest used station's "
    "sensor (arrivals= and stations= given), as stage tier does."
)
FOCAL_REFERENCE_NOTE = (
    "The Tier A nearest-station rule fell back to the depth below run.refSurfaceElevM; this can "
    "move the Tier A count either way relative to stage tier (REQ-H2-9)."
)
THRESHOLDS_NOTE = (
    "Tiers use the run's own bars from ProcessingRun.tiering.thresholds (stage tier); a rerun's "
    "matched set is too small to derive bars from and none are invented (REQ-H2-9)."
)
NOT_TIERED_NOTE = "No rerun reached assign_tiers (every rerun associated or located nothing)."


class ThresholdsNote(BaseModel):
    """Where the reruns' bars came from (the run's ``ProcessingRun.tiering["thresholds"]``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: str = THRESHOLDS_RECORD
    nMatched: int  # the matched set the run's bars were derived from
    quantiles: dict[str, float]  # tier -> q, as recorded by H2's tier stage


class TieringRules(BaseModel):
    """The Tier A rules H2's ``assign_tiers`` reports it applied in the reruns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    focalDepthBelow: str  # tiering["rules"]["A"]["nearestStation"]["focalDepthBelow"]
    mapOnVolumeTopApplied: bool  # tiering["rules"]["A"]["mapOnVolumeTop"]["applied"]
    thresholdSource: str  # tiering["thresholdSource"]: "supplied" when the run's bars were used


class RerunNotes(BaseModel):
    """Provenance of one set of reruns (the null test's, or the baseline table's)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reruns: int  # pipeline reruns made
    rerunsTiered: int  # of them, the ones that reached assign_tiers (the rest found no event)
    thresholds: ThresholdsNote
    tieringRules: TieringRules | None  # None when no rerun reached assign_tiers
    staticsApplied: bool = False  # hq.locate.locate has no match pass (FYI-H2-8)
    # profile -> associator fields overridden for it ({} means the run's config unchanged)
    associatorOverrides: dict[str, dict[str, int]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class GRNotes(BaseModel):
    """Provenance of the G-R curve's magnitude sets and of the gate applied to it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    magType: str | None  # the one CatalogEvent.magType publicCum is drawn from
    magTypeSource: str | None  # which record or knob named it
    publicIncluded: int  # public magnitudes of magType (publicCum's sample)
    publicExcludedByType: dict[str, int]  # other magTypes -> magnitudes left out ("null": no type)
    recoveredMagTypes: list[str]  # SeismicEvent.magnitude.type values seen (logged, never compared)
    maxLooMae: float  # the kill-switch gate applied
    maxLooMaeSource: str  # which record or knob it came from
    looMae: float | None  # MagCalibration.looMae when H2 wrote magnitude.json
    nullModelMae: float | None  # ProcessingRun.matching.magnitude.leaveOneEventOut.nullModelMae
    skill: bool | None  # looMae < nullModelMae; None when either is unknown
    notes: list[str] = Field(default_factory=list)


class ValidationNotes(BaseModel):
    """Contents of ``validation_notes.json``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nullTest: RerunNotes
    baseline: RerunNotes | None  # None when H1's picks_stalta.parquet is absent (no reruns)
    gr: GRNotes | None  # None when the run has no magnitude anywhere


def thresholds_note(record: Mapping[str, Any]) -> ThresholdsNote:
    """The note on a run's ``tiering["thresholds"]`` record (H2's ``Thresholds.to_record``)."""
    try:
        return ThresholdsNote(
            nMatched=int(record["nMatched"]),
            quantiles={str(t): float(q) for t, q in record["quantiles"].items()},
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValidateError(
            f"{THRESHOLDS_RECORD} is not a tiering thresholds record (nMatched, quantiles, A, B; "
            f"written by H2's tier stage, owner {H2}): {exc!r}"
        ) from exc


def tiering_rules(tiering: Mapping[str, Any]) -> TieringRules:
    """The rules H2's ``assign_tiers`` recorded in one ``TierResult.tiering``."""
    try:
        a = tiering["rules"]["A"]
        return TieringRules(
            focalDepthBelow=str(a["nearestStation"]["focalDepthBelow"]),
            mapOnVolumeTopApplied=bool(a["mapOnVolumeTop"]["applied"]),
            thresholdSource=str(tiering["thresholdSource"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValidateError(
            "assign_tiers returned a tiering record without rules.A.nearestStation.focalDepthBelow, "
            f"rules.A.mapOnVolumeTop.applied or thresholdSource (owner: {H2}): {exc!r}"
        ) from exc


def rerun_notes(
    tierings: Sequence[Mapping[str, Any] | None],
    thresholds_record: Mapping[str, Any],
    associator_overrides: Mapping[str, Mapping[str, int]],
    what: str,
) -> RerunNotes:
    """The notes of one set of reruns from the tiering dicts they returned (``None`` for a rerun
    that ended before ``assign_tiers``). The rules are read from the first tiered rerun; a later
    rerun reporting other rules is a warning (H2 applies the same rules to every call made the
    same way)."""
    tiered = [t for t in tierings if t is not None]
    rules: TieringRules | None = None
    notes = [THRESHOLDS_NOTE, NO_STATICS_NOTE]
    if tiered:
        rules = tiering_rules(tiered[0])
        others = {tiering_rules(t) for t in tiered[1:]}
        if others - {rules}:
            log.warning(
                "%s: reruns reported different tiering rules (%s vs %s); the notes carry the "
                "first",
                what,
                rules,
                others,
            )
        notes.append(
            FOCAL_SENSOR_NOTE
            if rules.focalDepthBelow == FOCAL_DEPTH_SENSOR
            else FOCAL_REFERENCE_NOTE
        )
        notes.append(MAP_ON_TOP_APPLIED_NOTE if rules.mapOnVolumeTopApplied else MAP_ON_TOP_SKIPPED_NOTE)
    else:
        notes.append(NOT_TIERED_NOTE)
    result = RerunNotes(
        reruns=len(tierings),
        rerunsTiered=len(tiered),
        thresholds=thresholds_note(thresholds_record),
        tieringRules=rules,
        staticsApplied=False,
        associatorOverrides={p: dict(o) for p, o in associator_overrides.items()},
        notes=notes,
    )
    log.info(
        "%s: %d of %d reruns tiered against the run's bars (%d matched events; rules: focal depth "
        "below %s, mapOnVolumeTop applied %s); statics applied: %s; associator overrides %s",
        what,
        result.rerunsTiered,
        result.reruns,
        result.thresholds.nMatched,
        None if rules is None else rules.focalDepthBelow,
        None if rules is None else rules.mapOnVolumeTopApplied,
        result.staticsApplied,
        result.associatorOverrides or "none",
    )
    return result


__all__ = [
    "GRNotes",
    "RerunNotes",
    "ThresholdsNote",
    "TieringRules",
    "ValidationNotes",
    "rerun_notes",
    "thresholds_note",
    "tiering_rules",
]

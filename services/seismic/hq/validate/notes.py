"""``validation_notes.json``: what the validation reruns could and could not do, next to the
numbers in ``null_test.json``, ``baseline.json`` and ``gr.json``.

The docs/02 contract models (``NullTest``, ``BaselineRow``, ``GRCurve``) are frozen and have no
room for provenance, so the stage writes this sidecar beside them (docs/02 §2). It is H4's own
model, not a contract: the exporter and the panels may read it, nothing else depends on it.

- ``nullTest`` / ``baseline`` (``RerunNotes``): the bars the reruns tiered against and where
  they came from (``thresholds.source``: ``run``, the run's own
  ``ProcessingRun.tiering["thresholds"]``, REQ-H2-9, the default; or ``phasenetRerun``, H2's
  ``assign_tiers`` deriving them from the PhaseNet ``full`` rerun through the same path,
  ``validate.yaml`` ``rerunBars: reference``), the matched count they rest on and the bars
  themselves, the Tier A rules H2's ``assign_tiers`` reports it applied (``tiering["rules"]``:
  how the nearest-station rule measured focal depth, whether the ``mapOnVolumeTop`` rule ran),
  whether the run's station statics were applied (``staticsApplied``: true in the validate
  stage, which passes ``statics.parquet`` to H2's ``locate(..., statics=)``, REQ-H1-5 a; false
  for a caller that locates without them, FYI-H2-8), and the associator overrides the
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
# ``ThresholdsNote.source``: which bars the reruns were tiered against (validate.yaml rerunBars).
THRESHOLDS_SOURCE_RUN = "run"
THRESHOLDS_SOURCE_REFERENCE = "phasenetRerun"
THRESHOLDS_RECORD = "ProcessingRun.tiering.thresholds"  # where the run's bars live (H2's tier stage)
THRESHOLDS_RECORD_REFERENCE = (
    "assign_tiers(...).tiering.thresholds of the reference rerun: PhaseNet picks.parquet, profile "
    "full, through the same locate call as every other rerun (hq.validate.reference)"
)
THRESHOLDS_RECORDS: dict[str, str] = {
    THRESHOLDS_SOURCE_RUN: THRESHOLDS_RECORD,
    THRESHOLDS_SOURCE_REFERENCE: THRESHOLDS_RECORD_REFERENCE,
}
BARRED_TIERS: tuple[str, ...] = ("A", "B")  # the tiers a thresholds record holds bars for
FOCAL_DEPTH_SENSOR = "nearestUsedSensor"  # H2's value when arrivals= and stations= were given
FOCAL_DEPTH_REFERENCE = "refSurfaceElevM"  # H2's fallback value (the docs/02 3-argument call)
NO_STATICS_NOTE = (
    "Rerun events are located by hq.locate.locate without statics= (it has no match pass, so "
    "statics.mode referenceEvents cannot run); rerun rmsS values are larger than the run's and "
    "the run's bars are harder for rerun events to meet (FYI-H2-8)."
)
STATICS_NOTE = (
    "Rerun events are located by hq.locate.locate with the run's own station statics "
    "(statics.parquet passed as statics=, REQ-H1-5 a), the terms the run's events carry, so "
    "rerun residuals are on the run's scale and its tier bars apply to them as to the run's "
    "events; FYI-H2-8's no-statics caveat does not apply to these numbers."
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
THRESHOLDS_RUN_NOTE = (
    "Tiers use the run's own bars from ProcessingRun.tiering.thresholds (stage tier); a rerun's "
    "matched set is too small to derive bars from and none are invented (REQ-H2-9)."
)
THRESHOLDS_REFERENCE_NOTE = (
    "Tiers use bars H2's assign_tiers derived, without thresholds=, from the PhaseNet "
    "full-profile rerun through the same locate call every rerun makes (validate.yaml rerunBars: "
    "reference, REQ-H1-5 option (b)), so PhaseNet, STA/LTA and the null test are on one scale "
    "and nothing fitted to PhaseNet's picks enters; the run's own bars "
    "(ProcessingRun.tiering.thresholds) are recorded but not applied to these reruns. A "
    "shuffle's or an STA/LTA rerun's matched set is too small to derive bars from and none are "
    "invented (REQ-H2-9)."
)
THRESHOLDS_NOTES: dict[str, str] = {
    THRESHOLDS_SOURCE_RUN: THRESHOLDS_RUN_NOTE,
    THRESHOLDS_SOURCE_REFERENCE: THRESHOLDS_REFERENCE_NOTE,
}
THRESHOLDS_NOTE = THRESHOLDS_RUN_NOTE  # kept for callers that name the run-bars note
NOT_TIERED_NOTE = "No rerun reached assign_tiers (every rerun associated or located nothing)."


class ThresholdsNote(BaseModel):
    """The bars the reruns were tiered against and where they came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = THRESHOLDS_SOURCE_RUN  # THRESHOLDS_SOURCE_RUN or THRESHOLDS_SOURCE_REFERENCE
    record: str = THRESHOLDS_RECORD  # which record the bars were read from
    nMatched: int  # the matched set the bars were derived from
    quantiles: dict[str, float]  # tier -> q, as recorded by H2 (tiering.quantiles)
    # tier -> metric -> {op, value}: the bars themselves, as H2's Thresholds.to_record has them
    bars: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)


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
    staticsApplied: bool = False  # the run's statics.parquet went into every locate (REQ-H1-5 a)
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


def thresholds_note(
    record: Mapping[str, Any], source: str = THRESHOLDS_SOURCE_RUN
) -> ThresholdsNote:
    """The note on a ``tiering["thresholds"]`` record (H2's ``Thresholds.to_record``): the run's
    own (``source`` ``run``) or the reference rerun's (``phasenetNoStaticsRerun``)."""
    if source not in THRESHOLDS_RECORDS:
        raise ValidateError(f"unknown thresholds source {source!r}; one of {list(THRESHOLDS_RECORDS)}")
    try:
        bars = {
            tier: {
                str(metric): {"op": str(bar["op"]), "value": bar["value"]}
                for metric, bar in record[tier].items()
            }
            for tier in BARRED_TIERS
            if isinstance(record.get(tier), Mapping)
        }
        return ThresholdsNote(
            source=source,
            record=THRESHOLDS_RECORDS[source],
            nMatched=int(record["nMatched"]),
            quantiles={str(t): float(q) for t, q in record["quantiles"].items()},
            bars=bars,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValidateError(
            f"{THRESHOLDS_RECORDS[source]} is not a tiering thresholds record (nMatched, "
            f"quantiles, A, B; written by H2's assign_tiers, owner {H2}): {exc!r}"
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
    *,
    source: str = THRESHOLDS_SOURCE_RUN,
    statics_applied: bool = False,
) -> RerunNotes:
    """The notes of one set of reruns from the tiering dicts they returned (``None`` for a rerun
    that ended before ``assign_tiers``). ``thresholds_record`` is the bars record every rerun
    received and ``source`` says where it came from (``run``, or ``phasenetRerun`` when the
    reference rerun derived it: that rerun then reports ``thresholdSource`` "derived" and the
    others "supplied", and ``tieringRules.thresholdSource`` lists both). ``statics_applied``
    says whether every ``locate`` got the run's ``statics.parquet`` (the validate stage) or none
    (the default: a caller using the plain docs/02 call, FYI-H2-8). The rules are read from the
    first tiered rerun; a later rerun reporting other rules is a warning (H2 applies the same
    rules to every call made the same way)."""
    tiered = [t for t in tierings if t is not None]
    rules: TieringRules | None = None
    notes = [
        THRESHOLDS_NOTES.get(source, THRESHOLDS_RUN_NOTE),
        STATICS_NOTE if statics_applied else NO_STATICS_NOTE,
    ]
    if tiered:
        seen = [tiering_rules(t) for t in tiered]
        sources = sorted({r.thresholdSource for r in seen})
        rules = seen[0].model_copy(update={"thresholdSource": "/".join(sources)})
        without_source = {r.model_copy(update={"thresholdSource": ""}) for r in seen}
        if len(without_source) > 1:
            log.warning(
                "%s: reruns reported different tiering rules (%s); the notes carry the first",
                what,
                sorted(without_source, key=repr),
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
        thresholds=thresholds_note(thresholds_record, source),
        tieringRules=rules,
        staticsApplied=statics_applied,
        associatorOverrides={p: dict(o) for p, o in associator_overrides.items()},
        notes=notes,
    )
    log.info(
        "%s: %d of %d reruns tiered against the %s bars (%d matched events; rules: focal depth "
        "below %s, mapOnVolumeTop applied %s); statics applied: %s; associator overrides %s",
        what,
        result.rerunsTiered,
        result.reruns,
        result.thresholds.source,
        result.thresholds.nMatched,
        None if rules is None else rules.focalDepthBelow,
        None if rules is None else rules.mapOnVolumeTopApplied,
        result.staticsApplied,
        result.associatorOverrides or "none",
    )
    return result


__all__ = [
    "THRESHOLDS_SOURCE_REFERENCE",
    "THRESHOLDS_SOURCE_RUN",
    "GRNotes",
    "RerunNotes",
    "ThresholdsNote",
    "TieringRules",
    "ValidationNotes",
    "rerun_notes",
    "thresholds_note",
    "tiering_rules",
]

"""Where the validation reruns get H2's pipeline functions.

H2 provides ``hq.associate.associate``, ``hq.locate.locate``, ``hq.match.match`` and
``hq.tier.assign_tiers`` (docs/02 §5). The validate stage never imports them at module load:
``real_seismology_api`` resolves them when the stage runs, so the validate package imports (and
its tests run) before H2's modules are merged, and a missing one fails with a message naming H2.
Tests inject a ``SeismologyApi`` built from a toy associator instead (mirrors
``hq.export.waveforms``).

Beyond the docs/02 §5 positional shapes, the reruns use two keyword extensions H2 asked for:

- REQ-H2-8: ``hq.locate.locate(..., cache_dir=, run_id=)``. ``real_seismology_api`` binds both
  with ``functools.partial`` so every rerun's travel-time tables come from
  ``<cache_dir>/ttgrids/`` (without it each validate process builds them once in a temporary
  directory) and rerun events carry ``hq-<runId>-NNNNNN`` ids. The ``SeismologyApi.locate``
  signature stays the 5-positional docs/02 call.
- REQ-H1-5 (option (a), H2's PR #94): ``hq.locate.locate(..., statics=<statics.parquet rows>)``.
  ``real_seismology_api`` binds the run's own station terms the same way (``with_statics`` does
  it for an injected API), so every rerun's events are located WITH the statics the run's
  events carry and the run's tier bars apply to them on one scale; without it ``locate`` has no
  match pass and locates without statics (FYI-H2-8).
- REQ-H2-9: ``hq.tier.assign_tiers(..., thresholds=, arrivals=, stations=)``, passed at call
  time by ``hq.validate.null_test.rerun_pipeline``: the bars to apply (a shuffle's or an
  STA/LTA rerun's matched set is too small to derive bars from, and H2 never invents them) and
  the tables the nearest-station rule measures focal depth with. One call leaves ``thresholds=``
  out: the reference rerun (``hq.validate.reference``, REQ-H1-5 option (b)), whose matched set
  is the run's recovered public events, so H2 derives the bars every other rerun then receives.
"""

import functools
import importlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from hq.config.run import RunSection
from hq.validate.errors import ValidateError

log = logging.getLogger(__name__)

H2 = "H2 Seismology"
ASSOCIATE_MODULE, ASSOCIATE_NAME = "hq.associate", "associate"
LOCATE_MODULE, LOCATE_NAME = "hq.locate", "locate"
MATCH_MODULE, MATCH_NAME = "hq.match", "match"
TIER_MODULE, TIER_NAME = "hq.tier", "assign_tiers"


# The four docs/02 §5 results are frozen dataclasses of DataFrames; only the attributes named
# there are relied on, so a test's stand-in needs nothing more than these.
class AssocResult(Protocol):
    @property
    def events(self) -> pd.DataFrame: ...  # assoc_events.parquet schema
    @property
    def picks(self) -> pd.DataFrame: ...  # assoc_picks.parquet schema


class LocateResult(Protocol):
    @property
    def events(self) -> pd.DataFrame: ...  # events_located.parquet schema
    @property
    def arrivals(self) -> pd.DataFrame: ...
    @property
    def statics(self) -> pd.DataFrame: ...


class MatchResult(Protocol):
    @property
    def matches(self) -> pd.DataFrame: ...  # matches.parquet schema
    @property
    def sensitivity(self) -> pd.DataFrame: ...


class TierResult(Protocol):
    @property
    def events(self) -> pd.DataFrame: ...  # events.parquet schema (final SeismicEvent rows)
    @property
    def tiering(self) -> dict[str, Any]: ...


class SeismologyApi(Protocol):
    """The four H2 calls a validation rerun needs (docs/02 §5 signatures). ``cfg`` is H2's
    ``SeismologyConfig``; typed ``Any`` here because the model lives in H2's module."""

    def associate(
        self, picks: pd.DataFrame, stations: pd.DataFrame, cfg: Any, run: RunSection
    ) -> AssocResult: ...

    def locate(
        self,
        assoc: AssocResult,
        picks: pd.DataFrame,
        stations: pd.DataFrame,
        cfg: Any,
        run: RunSection,
    ) -> LocateResult: ...

    def match(
        self, events_located: pd.DataFrame, catalog: pd.DataFrame, cfg: Any
    ) -> MatchResult: ...

    def assign_tiers(
        self,
        events_located: pd.DataFrame,
        matches: pd.DataFrame,
        cfg: Any,
        *,
        # A ProcessingRun.tiering-shaped dict with the bars to apply (REQ-H2-9); left out by the
        # reference rerun only, so H2 derives them (or raises its TierError below minMatched).
        thresholds: Mapping[str, Any] = ...,
        arrivals: pd.DataFrame,  # LocateResult.arrivals of the same rerun
        stations: pd.DataFrame,  # the stations table the rerun located with
    ) -> TierResult: ...


@dataclass(frozen=True)
class LaneSeismologyApi:
    """A ``SeismologyApi`` over four plain functions (H2's, or a test's)."""

    associate: Callable[..., AssocResult]
    locate: Callable[..., LocateResult]
    match: Callable[..., MatchResult]
    assign_tiers: Callable[..., TierResult]


def _resolve(module: str, name: str) -> Callable[..., Any]:
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if module == missing or module.startswith(missing + "."):
            raise ValidateError(
                f"validation reruns need {module}.{name} (docs/02 §5), which is not merged yet "
                f"(owner: {H2})"
            ) from exc
        raise  # the module exists but one of its own imports is broken: a real error
    fn = getattr(mod, name, None)
    if not callable(fn):
        raise ValidateError(
            f"validation reruns need {module}.{name} (docs/02 §5), which {module} does not "
            f"expose (owner: {H2})"
        )
    return fn


def with_statics(api: SeismologyApi, statics: pd.DataFrame) -> LaneSeismologyApi:
    """``api`` with the run's ``statics.parquet`` rows bound into every ``locate`` call as
    ``statics=`` (REQ-H1-5 a), the other three calls untouched."""
    return LaneSeismologyApi(
        associate=api.associate,
        locate=functools.partial(api.locate, statics=statics),
        match=api.match,
        assign_tiers=api.assign_tiers,
    )


def real_seismology_api(
    *,
    cache_dir: Path | None = None,
    run_id: str | None = None,
    statics: pd.DataFrame | None = None,
) -> LaneSeismologyApi:
    """H2's four pipeline functions, imported now; a clear error names H2 if any is absent.

    ``cache_dir`` and ``run_id`` (REQ-H2-8) and ``statics`` (the run's ``statics.parquet``
    rows, REQ-H1-5 a) are bound into ``locate`` as keywords, so callers keep the docs/02 §5
    five-positional call, every rerun reads the run's travel-time table cache
    (``<cache_dir>/ttgrids/``) and locates with the run's station terms; any left None is not
    passed, so H2's own defaults apply (no statics: FYI-H2-8).
    """
    locate = _resolve(LOCATE_MODULE, LOCATE_NAME)
    bound: dict[str, Any] = {}
    if cache_dir is not None:
        bound["cache_dir"] = Path(cache_dir)
    if run_id is not None:
        bound["run_id"] = run_id
    if statics is not None:
        bound["statics"] = statics
    if bound:
        locate = functools.partial(locate, **bound)
    api = LaneSeismologyApi(
        associate=_resolve(ASSOCIATE_MODULE, ASSOCIATE_NAME),
        locate=locate,
        match=_resolve(MATCH_MODULE, MATCH_NAME),
        assign_tiers=_resolve(TIER_MODULE, TIER_NAME),
    )
    log.info(
        "validate: reruns use %s.%s, %s.%s (bound %s), %s.%s and %s.%s",
        ASSOCIATE_MODULE,
        ASSOCIATE_NAME,
        LOCATE_MODULE,
        LOCATE_NAME,
        {k: (f"{len(v)} statics rows" if k == "statics" else str(v)) for k, v in bound.items()}
        or "nothing",
        MATCH_MODULE,
        MATCH_NAME,
        TIER_MODULE,
        TIER_NAME,
    )
    return api

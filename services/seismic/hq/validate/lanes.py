"""Where the validation reruns get H2's pipeline functions.

H2 provides ``hq.associate.associate``, ``hq.locate.locate``, ``hq.match.match`` and
``hq.tier.assign_tiers`` (docs/02 §5). The validate stage never imports them at module load:
``real_seismology_api`` resolves them when the stage runs, so the validate package imports (and
its tests run) before H2's modules are merged, and a missing one fails with a message naming H2.
Tests inject a ``SeismologyApi`` built from a toy associator instead (mirrors
``hq.export.waveforms``).
"""

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
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
        self, events_located: pd.DataFrame, matches: pd.DataFrame, cfg: Any
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


def real_seismology_api() -> LaneSeismologyApi:
    """H2's four pipeline functions, imported now; a clear error names H2 if any is absent."""
    api = LaneSeismologyApi(
        associate=_resolve(ASSOCIATE_MODULE, ASSOCIATE_NAME),
        locate=_resolve(LOCATE_MODULE, LOCATE_NAME),
        match=_resolve(MATCH_MODULE, MATCH_NAME),
        assign_tiers=_resolve(TIER_MODULE, TIER_NAME),
    )
    log.info(
        "validate: reruns use %s.%s, %s.%s, %s.%s and %s.%s",
        ASSOCIATE_MODULE,
        ASSOCIATE_NAME,
        LOCATE_MODULE,
        LOCATE_NAME,
        MATCH_MODULE,
        MATCH_NAME,
        TIER_MODULE,
        TIER_NAME,
    )
    return api

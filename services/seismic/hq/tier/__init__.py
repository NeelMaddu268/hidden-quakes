"""Quality tiers (LOC-06): located candidate events + matches -> final events (docs/02 §5).

``assign_tiers(events_located, matches, cfg) -> TierResult`` is the docs/02 §5 API. It reads only
the tables it is given, so it runs unchanged on events located from PhaseNet or STA/LTA picks.
The stage wrapper (run dir in, ``events.parquet``, ``event_picks.parquet`` and optionally
``sweep.parquet`` out) is ``hq.tier.run``; ``run`` here is that stage function, because the stage
registry (H4's ``hq.runs.STAGES``) resolves stage ``tier`` as the attribute ``hq.tier.run``.

Tiers come from the data, never from textbook values
    The matched set M is the located events that ``matches`` pairs with a public regional catalog
    event. For every metric (``METRICS``: nStations, nP, nS higher is better; rmsS, hErrM, vErrM,
    gapDeg lower is better) and tier, the bar is an actual matched event's value: sort the events
    of M with a value on that metric from best to worst and take the ``ceil((1 - q) * n)``-th,
    with ``n`` their count and ``q`` the tier's ``tiering.quantiles`` entry. At least ``(1 - q)``
    of them therefore meets or beats the bar, and a bar is ``p(100 q)`` for higher-is-better
    metrics and ``p(100 (1 - q))`` for lower-is-better ones (numpy's ``inverted_cdf`` at
    ``1 - q`` on the lower-is-better side, its mirror ``-inverted_cdf(-v, 1 - q)`` on the
    higher-is-better side). With the showcase ``A: 0.25`` that is ">= p25" and "<= p75": at least
    as good as the 25th-percentile-worst matched event. ``B: 0.0`` is the worst matched event.
    Boundary equality passes.

    A (Strict): every A bar, ``quality.depthOnEdge`` false, ``mapOnVolumeTop`` false when a
    ``flags`` table (``locate_flags.parquet``) is given, and a station with a pick used in the
    final location within ``strictNearestStationFactor`` focal depths, epicentral:
    ``quality.minEpiDistM <= factor * depthKm * 1000``. Focal depth is ``depthKm * 1000 =
    refSurfaceElevM - elevM``: the depth below the run's reference ground surface.
    B (Good): every B bar. C (Candidate): associated and located, outside those ranges.

    A null ``hErrM`` or ``vErrM`` (the contract allows it: a truncated PDF) fails A and B on that
    metric. The bars on those metrics are drawn from the matched events with a value (``nUsed``
    of ``n``, ``nNull`` recorded), so a null in M never loosens a bar; fewer than ``minMatched``
    matched events with a value fails loudly like a small M. ``matchedSet.meetingEveryBar``
    counts the matched events that meet every bar of a tier at once (the per-metric shares above
    do not add up to a joint share).

    Fewer matched events than ``tiering.minMatched`` fails loudly (``TierError``): the bars are
    never invented. A caller that must tier a rerun whose M is small (the association sweep, H4's
    null test and baseline) passes ``thresholds=`` the bars of a run that had enough matches: a
    ``Thresholds`` or that run's ``ProcessingRun.tiering`` dict.

Outputs
    ``TierResult.events``: ``SeismicEvent`` rows (``events.parquet``, docs/02 §2 flattening and
    dtypes) in the input order, each validated through the model: the located columns unchanged,
    plus ``tier``, ``tierReasons`` (one string per metric, then every failed A rule),
    ``catalogMatch`` (``catalogId``, ``dtS``, ``distM`` from ``matches``) for matched events,
    ``magnitude`` null (MAG-01) and ``revealOrder`` -1. ``TierResult.tiering``: the bars with
    their source quantile and the size of M, the rules, the tier counts (all, additional =
    unmatched, and matched) and the caveat, for ``ProcessingRun.tiering``.
"""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import pandas as pd
from hq_contracts.io import columns_for, from_frame, to_frame
from hq_contracts.models import SeismicEvent

from hq.config.seismology import SeismologyConfig, TieringConfig

log = logging.getLogger(__name__)

TIERS: tuple[str, ...] = ("A", "B", "C")
BARRED_TIERS: tuple[str, ...] = ("A", "B")  # tiers with metric bars; C is everything else
Better = Literal["higher", "lower"]

# SeismicEvent fields events_located.parquet leaves out (docs/02 §2); this module adds them.
FINAL_ONLY_FIELDS: tuple[str, ...] = ("tier", "tierReasons", "catalogMatch", "magnitude")
FINAL_COLUMNS: tuple[str, ...] = tuple(columns_for(SeismicEvent))
LOCATED_COLUMNS: tuple[str, ...] = tuple(
    c for c in FINAL_COLUMNS if c.split("_")[0] not in FINAL_ONLY_FIELDS
)
MATCH_COLUMNS: tuple[str, ...] = ("catalogId", "eventId", "dtS", "distM")
FLAG_COLUMNS: tuple[str, ...] = ("eventId", "mapOnVolumeTop")
REVEAL_ORDER_UNSET = -1  # docs/02: H2 writes -1, the exporter assigns the real order

DEFINITION = (
    "Per metric and tier, the bar is an actual matched event's value: sort the events of the "
    "matched set M with a value on the metric from best to worst and take the ceil((1 - q) * n)-th, "
    "n = their count, q = tiering.quantiles[tier]; at least (1 - q) of them meets or beats the bar "
    "(boundary equality passes). A (Strict): every A bar, depthOnEdge false, mapOnVolumeTop "
    "false when locate flags are given, and "
    "quality.minEpiDistM <= strictNearestStationFactor * focal depth. B (Good): every B bar. "
    "C (Candidate): associated and located, outside those ranges."
)
CAVEAT = (
    "Public regional catalog events are the larger ones, so bars drawn from the recovered public "
    "events are conservative for small candidate events."
)
FOCAL_DEPTH = (
    "depthKm * 1000 = refSurfaceElevM - elevM: the depth below the run's reference ground "
    "surface (run.yaml refSurfaceElevM), the depth the scene shows"
)
NULL_RULE = (
    "a null hErrM or vErrM fails A and B on that metric; bars on those metrics are drawn from the "
    "matched events with a value (nUsed of n; nNull recorded), so a null never loosens a bar"
)
MEETING_EVERY_BAR = (
    "matched events meeting every metric bar of the tier at once (the A rules are not applied "
    "here; counts.matched has the matched events' tiers)"
)
QUANTILE_METHOD = (
    "rank ceil((1 - q) * n) from the best: numpy inverted_cdf at 1 - q for lower-is-better "
    "metrics; for higher-is-better ones its mirror -inverted_cdf(-v, 1 - q), the largest value "
    "at least (1 - q) of M meet or beat (one rank above inverted_cdf at q when q * n is an integer)"
)


def claim(q_a: float) -> str:
    """What Strict means, in words, for ``tiering.quantiles.A`` = ``q_a``."""
    share = float(1 - Fraction(repr(q_a)))
    return (
        f"Strict (Tier A): on each metric separately, at least as good as a bar that at least "
        f"{share:.0%} of the recovered public regional catalog events with a value on that metric "
        "meet or beat."
    )


class TierError(ValueError):
    """Inputs that cannot be tiered honestly (too few matches, inconsistent tables)."""


@dataclass(frozen=True)
class Metric:
    """One ``LocationQuality`` metric the tiers compare."""

    name: str
    better: Better
    nullable: bool  # docs/02 allows None (hErrM, vErrM)
    decimals: int  # tierReasons display; more when needed to tell a value from its bar

    @property
    def column(self) -> str:
        return f"quality_{self.name}"

    @property
    def op(self) -> str:
        return ">=" if self.better == "higher" else "<="

    @property
    def fail_op(self) -> str:
        return "<" if self.better == "higher" else ">"


METRICS: tuple[Metric, ...] = (
    Metric("nStations", "higher", False, 0),
    Metric("nP", "higher", False, 0),
    Metric("nS", "higher", False, 0),
    Metric("rmsS", "lower", False, 3),
    Metric("hErrM", "lower", True, 0),
    Metric("vErrM", "lower", True, 0),
    Metric("gapDeg", "lower", False, 0),
)
METRIC_BY_NAME: dict[str, Metric] = {m.name: m for m in METRICS}


@dataclass(frozen=True)
class Bar:
    """One tier's bar on one metric, with where it came from."""

    metric: str
    value: float
    level: float  # percentile level, 0-1: q (higher is better) or 1 - q (lower is better)
    label: str  # "p75 of matched" / "worst of matched"
    n: int  # size of M
    n_used: int  # matched events with a value on this metric: the bar is drawn from these
    n_null: int  # matched events with a null value on this metric (hErrM, vErrM)
    n_meeting: int  # matched events meeting this bar (a null never does)

    def passes(self, value: float | None) -> bool:
        return value is not None and bool(self.meets(np.array([value], dtype=np.float64))[0])

    def meets(self, values: np.ndarray) -> np.ndarray:
        """Elementwise ``passes`` on float64 values, NaN (null) never meeting the bar."""
        if METRIC_BY_NAME[self.metric].better == "higher":
            return values >= self.value
        return values <= self.value

    def to_record(self) -> dict[str, Any]:
        return {
            "op": METRIC_BY_NAME[self.metric].op,
            "value": self.value,
            "quantile": self.level,
            "label": self.label,
            "n": self.n,
            "nUsed": self.n_used,
            "nNull": self.n_null,
            "nMeeting": self.n_meeting,
        }


@dataclass(frozen=True)
class Thresholds:
    """The A and B bars on every metric, derived from one matched set."""

    quantiles: dict[str, float]  # tier -> q (tiering.quantiles)
    bars: dict[str, dict[str, Bar]]  # tier -> metric name -> bar
    n_matched: int

    def to_record(self) -> dict[str, Any]:
        return {
            "quantiles": dict(self.quantiles),
            "nMatched": self.n_matched,
            "method": QUANTILE_METHOD,
            **{
                tier: {name: bar.to_record() for name, bar in self.bars[tier].items()}
                for tier in BARRED_TIERS
            },
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Thresholds":
        """Inverse of ``to_record`` (``ProcessingRun.tiering["thresholds"]``)."""
        try:
            quantiles = {tier: float(record["quantiles"][tier]) for tier in BARRED_TIERS}
            n = int(record["nMatched"])
            bars: dict[str, dict[str, Bar]] = {}
            for tier in BARRED_TIERS:
                bars[tier] = {}
                for metric in METRICS:
                    rec = record[tier][metric.name]
                    if rec["op"] != metric.op:
                        raise TierError(f"{tier}.{metric.name}: op {rec['op']!r} != {metric.op}")
                    if rec["value"] is None:
                        raise TierError(f"{tier}.{metric.name}: a bar value is never null")
                    bars[tier][metric.name] = Bar(
                        metric=metric.name,
                        value=float(rec["value"]),
                        level=float(rec["quantile"]),
                        label=str(rec["label"]),
                        n=int(rec["n"]),
                        n_used=int(rec["nUsed"]),
                        n_null=int(rec["nNull"]),
                        n_meeting=int(rec["nMeeting"]),
                    )
        except (KeyError, TypeError) as exc:
            raise TierError(f"not a tiering thresholds record: {exc!r}") from exc
        return cls(quantiles=quantiles, bars=bars, n_matched=n)


@dataclass(frozen=True, eq=False)
class TierResult:
    """docs/02 §5: ``events`` (final ``SeismicEvent`` rows) and ``tiering`` (for the run record)."""

    events: pd.DataFrame
    tiering: dict[str, Any]


# --- inputs ---------------------------------------------------------------------------------------


def _missing(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise TierError(f"{what} lacks columns {missing}")


def _ids(series: pd.Series, what: str) -> pd.Series:
    if series.isna().any():
        raise TierError(f"{what}: null ids")
    ids = series.astype(str)
    if ids.duplicated().any():
        raise TierError(f"{what}: duplicate ids {sorted(set(ids[ids.duplicated()]))[:5]}")
    return ids


def _check_located(events: pd.DataFrame) -> pd.Series:
    _missing(events, LOCATED_COLUMNS, "events_located")
    final = [c for c in events.columns if c.split("_")[0] in FINAL_ONLY_FIELDS]
    if final:
        raise TierError(
            f"events_located carries final-event columns {final}: pass events_located.parquet "
            "rows (docs/02 §2), not a tiered table"
        )
    return _ids(events["id"], "events_located")


def matched_rows(matches: pd.DataFrame, event_ids: pd.Series) -> pd.DataFrame:
    """The matched rows of ``matches`` (``eventId`` not null), checked one-to-one and complete.

    Columns ``catalogId``, ``eventId`` (str), ``dtS``, ``distM`` (float64), in the input order.
    """
    _missing(matches, MATCH_COLUMNS, "matches")
    rows = matches[matches["eventId"].notna()]
    if rows["catalogId"].isna().any():
        raise TierError("matches: a matched row has a null catalogId")
    out = pd.DataFrame(
        {
            "catalogId": rows["catalogId"].astype(str).to_numpy(dtype=object),
            "eventId": rows["eventId"].astype(str).to_numpy(dtype=object),
            "dtS": pd.to_numeric(rows["dtS"]).to_numpy(dtype=np.float64),
            "distM": pd.to_numeric(rows["distM"]).to_numpy(dtype=np.float64),
        }
    )
    for col in ("eventId", "catalogId"):
        dup = out[col].duplicated()
        if dup.any():
            raise TierError(f"matches is not one-to-one: {col} {sorted(set(out[col][dup]))[:5]}")
    unknown = sorted(set(out["eventId"]) - set(event_ids))
    if unknown:
        raise TierError(f"matches name events missing from events_located: {unknown[:5]}")
    finite = np.isfinite(out[["dtS", "distM"]].to_numpy()).all(axis=1)
    if not finite.all():
        raise TierError(f"matches: non-finite dtS/distM for {list(out['eventId'][~finite])[:5]}")
    return out


def _map_on_top(flags: pd.DataFrame | None, event_ids: pd.Series) -> np.ndarray | None:
    """``mapOnVolumeTop`` per event, in ``event_ids`` order; None without a flags table."""
    if flags is None:
        return None
    _missing(flags, FLAG_COLUMNS, "locate flags")
    flag_ids = _ids(flags["eventId"], "locate flags")
    if set(flag_ids) != set(event_ids):
        missing = sorted(set(event_ids) - set(flag_ids))
        extra = sorted(set(flag_ids) - set(event_ids))
        raise TierError(
            f"locate flags do not cover events_located one-to-one: missing {missing[:5]}, "
            f"extra {extra[:5]}"
        )
    if flags["mapOnVolumeTop"].isna().any():
        raise TierError("locate flags: null mapOnVolumeTop")
    by_id = pd.Series(flags["mapOnVolumeTop"].astype(bool).to_numpy(), index=flag_ids.to_numpy())
    return by_id.reindex(event_ids.to_numpy()).to_numpy(dtype=bool)


def _metric_values(events: pd.DataFrame, metric: Metric) -> np.ndarray:
    """float64 values, NaN for null; a null in a required metric fails loudly."""
    values = pd.to_numeric(events[metric.column]).to_numpy(dtype=np.float64, na_value=np.nan)
    if not metric.nullable and np.isnan(values).any():
        raise TierError(f"events_located: null {metric.column} (docs/02 requires it)")
    if np.isinf(values).any():
        raise TierError(f"events_located: infinite {metric.column}")
    return values


# --- thresholds -----------------------------------------------------------------------------------


def _level_label(q: float, metric: Metric) -> tuple[float, str]:
    if q == 0.0:
        return (0.0 if metric.better == "higher" else 1.0), "worst of matched"
    level = float(Fraction(repr(q))) if metric.better == "higher" else float(1 - Fraction(repr(q)))
    return level, f"p{100 * level:g} of matched"


def derive_bar(values: np.ndarray, metric: Metric, q: float) -> Bar:
    """The bar at worst-side share ``q`` of the matched values with a value (NaN = null)."""
    finite = values[~np.isnan(values)]
    n, n_used = int(values.size), int(finite.size)
    if n_used == 0:
        raise TierError(f"cannot derive a {metric.name} bar: no matched event has a value")
    badness = np.sort(finite if metric.better == "lower" else -finite)
    rank = max(1, math.ceil((1 - Fraction(repr(q))) * n_used))  # exact for the yaml's decimal q
    worst_allowed = float(badness[rank - 1])
    value = worst_allowed if metric.better == "lower" else -worst_allowed
    level, label = _level_label(q, metric)
    bar = Bar(metric.name, value, level, label, n, n_used, n - n_used, 0)
    return replace(bar, n_meeting=int(bar.meets(values).sum()))


def derive_thresholds(matched: pd.DataFrame, tcfg: TieringConfig) -> Thresholds:
    """The A and B bars from the matched events' quality columns; fails below ``minMatched``."""
    n = len(matched)
    if n < tcfg.minMatched:
        raise TierError(
            f"the matched set has {n} event(s) and tiering.minMatched is {tcfg.minMatched}: "
            "quantiles of so few matched events are not meaningful, so no bars are derived. "
            "Pass thresholds= (a Thresholds, or the ProcessingRun.tiering of a run whose matched "
            "set was large enough) to apply that run's bars instead."
        )
    values = {m.name: _metric_values(matched, m) for m in METRICS}
    for m in METRICS:
        n_used = int(np.count_nonzero(~np.isnan(values[m.name])))
        if n_used < tcfg.minMatched:
            raise TierError(
                f"only {n_used} of the {n} matched events have a {m.name} value (the others are "
                f"null: truncated PDF) and tiering.minMatched is {tcfg.minMatched}: no bar is "
                "derived from so few values. Pass thresholds= to apply another run's bars."
            )
        if n_used < n:
            log.info("tier: bars on %s are drawn from the %d of %d matched events with a value "
                     "(%d null; a null fails A and B)", m.name, n_used, n, n - n_used)
    quantiles = {"A": tcfg.quantiles.A, "B": tcfg.quantiles.B}
    bars = {
        tier: {m.name: derive_bar(values[m.name], m, q) for m in METRICS}
        for tier, q in quantiles.items()
    }
    return Thresholds(quantiles=quantiles, bars=bars, n_matched=n)


def meeting_every_bar(matched: pd.DataFrame, thresholds: Thresholds) -> dict[str, int]:
    """Per tier, the matched events that meet every metric bar at once (rules not applied)."""
    out: dict[str, int] = {}
    for tier in BARRED_TIERS:
        meets = np.ones(len(matched), dtype=bool)
        for m in METRICS:
            meets &= thresholds.bars[tier][m.name].meets(_metric_values(matched, m))
        out[tier] = int(meets.sum())
    return out


def supplied_thresholds(
    thresholds: "Thresholds | Mapping[str, Any]", tcfg: TieringConfig
) -> Thresholds:
    """A ``Thresholds``, or one parsed from a ``ProcessingRun.tiering`` dict; quantiles must
    equal this config's, so bars from another tier definition are never mixed in."""
    if isinstance(thresholds, Thresholds):
        parsed = thresholds
    else:
        record = thresholds.get("thresholds")
        if not isinstance(record, Mapping):
            raise TierError(
                "thresholds= must be a Thresholds or a ProcessingRun.tiering dict with a "
                "'thresholds' record (the tier stage writes one)"
            )
        parsed = Thresholds.from_record(record)
    want = {"A": tcfg.quantiles.A, "B": tcfg.quantiles.B}
    if parsed.quantiles != want:
        raise TierError(
            f"supplied thresholds come from quantiles {parsed.quantiles}; this config has {want}"
        )
    return parsed


# --- tiering one event ----------------------------------------------------------------------------


def _fmt(value: float, other: float | None, decimals: int) -> str:
    """``value`` at ``decimals``, or more (up to 6) until it reads differently from ``other``."""
    d = decimals
    while other is not None and value != other and d < 6:
        if f"{value:.{d}f}" != f"{other:.{d}f}":
            break
        d += 1
    return f"{value:.{d}f}"


def _metric_reason(metric: Metric, value: float | None, bars: dict[str, Bar]) -> str:
    """One tierReasons string: the A comparison, and the B one when A fails."""
    if value is None:
        return f"{metric.name} null (no formal error): fails A and B"
    parts: list[str] = []
    for tier in BARRED_TIERS:
        bar = bars[tier]
        op = metric.op if bar.passes(value) else metric.fail_op
        bar_text = _fmt(bar.value, value, metric.decimals)
        source = f"{tier}: {bar.label}" + (f", n={bar.n_used}" if tier == "A" else "")
        parts.append(f"{op} {bar_text} ({source})")
        if bar.passes(value):
            break
    value_text = _fmt(value, next((b.value for b in bars.values()), None), metric.decimals)
    return f"{metric.name} {value_text} " + "; ".join(parts)


@dataclass(frozen=True)
class _Rules:
    factor: float
    map_on_top: np.ndarray | None


def _tier_event(
    row: dict[str, Any], k: int, thresholds: Thresholds, rules: _Rules
) -> tuple[str, list[str]]:
    values = {
        m.name: None if pd.isna(row[m.column]) else float(row[m.column]) for m in METRICS
    }
    reasons = [
        _metric_reason(
            m, values[m.name], {t: thresholds.bars[t][m.name] for t in BARRED_TIERS}
        )
        for m in METRICS
    ]
    passes = {
        tier: all(thresholds.bars[tier][m.name].passes(values[m.name]) for m in METRICS)
        for tier in BARRED_TIERS
    }
    failed_rules: list[str] = []
    if bool(row["quality_depthOnEdge"]):
        failed_rules.append("depthOnEdge (PDF mass on the grid's top or bottom face): fails A")
    if rules.map_on_top is not None and rules.map_on_top[k]:
        failed_rules.append("MAP on the search-volume top: fails A")
    depth_m = float(row["depthKm"]) * 1000.0
    nearest_m = float(row["quality_minEpiDistM"])
    if not nearest_m <= rules.factor * depth_m:
        failed_rules.append(
            f"nearest station {nearest_m:.0f} m > {rules.factor:g} x focal depth {depth_m:.0f} m "
            "(epicentral, depth below refSurfaceElevM): fails A"
        )
    if passes["A"] and not failed_rules:
        tier = "A"
    elif passes["B"]:
        tier = "B"
    else:
        tier = "C"
    return tier, reasons + failed_rules


# --- the API --------------------------------------------------------------------------------------


def _counts(tiers: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    return {tier: int(np.sum((tiers == tier) & mask)) for tier in TIERS}


def _final_frame(
    events: pd.DataFrame,
    event_ids: pd.Series,
    matched: pd.DataFrame,
    tiers: list[str],
    reasons: list[list[str]],
) -> pd.DataFrame:
    final = events[list(LOCATED_COLUMNS)].copy()
    by_event = matched.set_index("eventId")
    final["tier"] = tiers
    final["tierReasons"] = reasons
    keys = event_ids.to_numpy()
    final["catalogMatch_catalogId"] = by_event["catalogId"].reindex(keys).to_numpy(dtype=object)
    final["catalogMatch_dtS"] = by_event["dtS"].reindex(keys).to_numpy(dtype=np.float64)
    final["catalogMatch_distM"] = by_event["distM"].reindex(keys).to_numpy(dtype=np.float64)
    final["magnitude_value"] = np.nan  # MAG-01 fills magnitudes
    final["magnitude_type"] = None
    final["magnitude_sigma"] = np.nan
    final["revealOrder"] = REVEAL_ORDER_UNSET
    models = from_frame(final[list(FINAL_COLUMNS)], SeismicEvent)  # every row validated
    return to_frame(models, SeismicEvent).reset_index(drop=True)


def assign_tiers(
    events_located: pd.DataFrame,
    matches: pd.DataFrame,
    cfg: SeismologyConfig,
    *,
    flags: pd.DataFrame | None = None,
    thresholds: "Thresholds | Mapping[str, Any] | None" = None,
) -> TierResult:
    """Tier every located candidate event (docs/02 §5); see the module docstring.

    ``flags`` (keyword only; the docs/02 call leaves it out): ``locate_flags.parquet`` rows, one
    per event, to apply the ``mapOnVolumeTop`` rule. ``thresholds``: bars to apply instead of
    deriving them from this call's matched set (a ``Thresholds`` or a ``ProcessingRun.tiering``
    dict); their quantiles must equal ``cfg.tiering.quantiles``.
    """
    tcfg = cfg.tiering
    event_ids = _check_located(events_located)
    matched = matched_rows(matches, event_ids)
    map_on_top = _map_on_top(flags, event_ids)
    for metric in METRICS:  # every event, matched or not: nulls only where docs/02 allows them
        _metric_values(events_located, metric)
    is_matched = event_ids.isin(set(matched["eventId"])).to_numpy(dtype=bool)
    source = "derived" if thresholds is None else "supplied"
    rules = {
        "A": {
            "depthOnEdge": "must be false",
            "mapOnVolumeTop": {
                "rule": "must be false",
                "applied": map_on_top is not None,
                "note": "from locate_flags.parquet (MAP within locator.mapOnVolumeFaceBandM of "
                "the search-volume top)" if map_on_top is not None else
                "no locate flags given, so the rule was not applied",
            },
            "nearestStation": {
                "factor": tcfg.strictNearestStationFactor,
                "test": "quality.minEpiDistM <= factor * focal depth (epicentral distance to the "
                "nearest station with a pick used in the final location)",
                "focalDepth": FOCAL_DEPTH,
            },
        },
        "nullErrors": NULL_RULE,
    }
    base: dict[str, Any] = {
        "definition": DEFINITION,
        "claim": claim(tcfg.quantiles.A),
        "caveat": CAVEAT,
        "rules": rules,
        "thresholdSource": source,
        "matchedSet": {
            "n": int(is_matched.sum()),
            "definition": "located candidate events with a public regional catalog match "
            "(matches eventId not null)",
            "meetingEveryBarNote": MEETING_EVERY_BAR,
        },
    }

    if len(events_located) == 0:
        record = None if thresholds is None else supplied_thresholds(thresholds, tcfg).to_record()
        log.info("tier: no located events; nothing to tier")
        empty = to_frame([], SeismicEvent)
        zero = {tier: 0 for tier in TIERS}
        base["matchedSet"]["meetingEveryBar"] = {tier: 0 for tier in BARRED_TIERS}
        return TierResult(
            events=empty,
            tiering={
                **base,
                "thresholds": record,
                "note": "no located events, so no bars were derived",
                "counts": {"events": 0, "all": zero, "additional": zero, "matched": zero},
            },
        )

    if thresholds is None:
        bars = derive_thresholds(events_located[is_matched], tcfg)
    else:
        bars = supplied_thresholds(thresholds, tcfg)
    base["matchedSet"]["meetingEveryBar"] = meeting_every_bar(events_located[is_matched], bars)

    rule_set = _Rules(factor=tcfg.strictNearestStationFactor, map_on_top=map_on_top)
    rows = events_located[list(LOCATED_COLUMNS)].to_dict("records")
    tiered = [_tier_event(row, k, bars, rule_set) for k, row in enumerate(rows)]
    tiers = [t for t, _ in tiered]
    final = _final_frame(events_located, event_ids, matched, tiers, [r for _, r in tiered])

    tier_arr = np.asarray(tiers)
    everyone = np.ones(len(tier_arr), dtype=bool)
    counts = {
        "events": len(tier_arr),
        "all": _counts(tier_arr, everyone),
        "additional": _counts(tier_arr, ~is_matched),
        "matched": _counts(tier_arr, is_matched),
    }
    log.info(
        "tier: %d located candidate events: all A %d / B %d / C %d; additional (unmatched) "
        "A %d / B %d / C %d; matched A %d / B %d / C %d; bars %s from %d matched events",
        counts["events"], *counts["all"].values(), *counts["additional"].values(),
        *counts["matched"].values(), source, bars.n_matched,
    )
    return TierResult(
        events=final, tiering={**base, "thresholds": bars.to_record(), "counts": counts}
    )


# Last, so the package attribute ``run`` is the stage function, not the submodule.
from hq.tier.run import run

__all__ = [
    "METRICS",
    "Bar",
    "Thresholds",
    "TierError",
    "TierResult",
    "assign_tiers",
    "derive_thresholds",
    "run",
]

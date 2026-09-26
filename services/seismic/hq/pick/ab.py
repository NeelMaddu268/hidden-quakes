"""PhaseNet weight A/B on the known-event windows, plus the Check B evaluator (SEIS-04).

Reads ``runs/<id>/known/windows.json`` (SEIS-02), ``runs/<id>/stations.parquet`` (SEIS-01) and
waveforms via ``hq.ingest.cache.read_window`` (SEIS-05), keeping only each station's
``Station.channels``. For every event x weights x profile it picks every usable station.

Per station and event only picks inside the event's arrival window count
(``picker.ab.arrivalWindow``: from the catalog origin minus an allowance to the slowest travel
time over the hypocentral distance plus a margin). Inside it the best P is the max-probability P
and the best S the max-probability S after that P. A violation is a station whose max-probability
S is not after its best P. Per event: stations with P and S, and Spearman rho between best-P time
and epicentral distance over the stations with a best P (rhoPS over the P-and-S stations is
reported alongside, not gated).

A/B metric per (weights, profile), compared in this order: events consistent with Check B
(no violation and rho >= checkB.minRho), total P-and-S stations, fewer violations, events with a
defined rho, mean rho; a full tie goes to the earlier entry of ``candidateWeights``. A variant
profile (``borehole-B``) is tried on its base profile's stations; stations whose data the
variant's preprocessing rejects are counted and left out of it. The variant is adopted only when
it strictly beats the base profile on the same station-windows; stations it rejected keep the
base profile.

Writes to ``runs/<id>/known/``: ``ab.csv`` (one row per event x weights x profile, one summary
row per weights x profile, and the like-for-like variant comparison rows), ``ab.json`` (chosen
weights, adopted profiles, Check B), ``picks.parquet`` (every pick >= threshold from the chosen
combination, ``Pick`` schema) and one ``record_section_<eventId>.png`` per event; ``run(ctx)``
also writes ``pick_known.record.json`` (runtime, counts, params).

CLI: writes everything above except ``known/pick_known.record.json``, which only ``run(ctx)``
writes; neither touches ``run.json`` (pick_known is a sub-step, not a registered stage)::

    uv run python -m hq.pick.ab --run-dir <dir> --config-dir configs/showcase --cache-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import obspy
import pandas as pd
import yaml
from scipy.stats import spearmanr

from hq.config.signal import ArrivalWindowConfig, CheckBConfig, PickerConfig, SignalConfig
from hq.pick.phasenet import (
    ForPicking,
    PickDiagnostics,
    load_model,
    pick_prepared,
    prepare_station,
)

log = logging.getLogger(__name__)

ReadWindow = Callable[..., obspy.Stream]  # (station_id, t0, t1, *, cache_dir) -> Stream
DisplayCopy = Callable[[obspy.Stream, tuple[float, float]], obspy.Stream]
ModelLoader = Callable[[str, PickerConfig], Any]

KNOWN_DIR = "known"
WINDOWS_FILE = "windows.json"
AB_CSV = "ab.csv"
AB_JSON = "ab.json"
PICKS_FILE = "picks.parquet"
STAGE = "pick_known"

# Record-section styling (presentation only, never a pipeline parameter). Colours: categorical
# slots 1 and 2 of the dataviz reference palette; traces and text stay in neutral ink so colour
# only carries the phase identity.
_COLOR_P = "#2a78d6"
_COLOR_S = "#eb6834"
_INK = "#52514e"
_INK_MUTED = "#8a8985"
_SURFACE = "#fcfcfb"
_TRACE_LINEWIDTH = 0.6
_BEST_PICK_LINEWIDTH = 2.0
_OTHER_PICK_LINEWIDTH = 0.8
_OTHER_PICK_ALPHA = 0.35
_GRID_LINEWIDTH = 0.5
_GRID_ALPHA = 0.25
_LABEL_FONTSIZE = 8
_LEGEND_COLUMNS = 4


# --- inputs -------------------------------------------------------------------------------------

_DOC_KEYS = frozenset({"events", "params"})
_EVENT_KEYS = frozenset(
    {
        "eventId",
        "t",
        "latitude",
        "longitude",
        "depthKm",
        "mag",
        "magType",
        "windowStart",
        "windowEnd",
        "stations",
    }
)
_STATION_KEYS = frozenset(
    {"stationId", "components", "gapFraction", "epiDistM", "usable", "reason"}
)


@dataclass(frozen=True)
class KnownStation:
    stationId: str
    components: int
    gapFraction: float
    epiDistM: float
    usable: bool
    reason: str | None


@dataclass(frozen=True)
class KnownEvent:
    eventId: str
    t: float
    latitude: float
    longitude: float
    depthKm: float
    mag: float | None
    magType: str | None
    windowStart: float
    windowEnd: float
    stations: tuple[KnownStation, ...]


@dataclass(frozen=True)
class KnownWindows:
    events: tuple[KnownEvent, ...]
    params: dict[str, Any]


def _req(obj: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in obj:
        raise ValueError(f"{where}: missing key {key!r}")
    return obj[key]


def _no_extra(obj: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        raise ValueError(f"{where}: unknown keys {extra}")


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def parse_known_windows(doc: Mapping[str, Any]) -> KnownWindows:
    """Parse the shared ``known/windows.json`` document (SEIS-02 writes it; unknown keys fail)."""
    _no_extra(doc, _DOC_KEYS, "windows.json")
    events: list[KnownEvent] = []
    for i, ev in enumerate(_req(doc, "events", "windows.json")):
        where = f"windows.json events[{i}]"
        _no_extra(ev, _EVENT_KEYS, where)
        stations = []
        for j, st in enumerate(_req(ev, "stations", where)):
            swhere = f"{where}.stations[{j}]"
            _no_extra(st, _STATION_KEYS, swhere)
            reason = _req(st, "reason", swhere)
            stations.append(
                KnownStation(
                    stationId=str(_req(st, "stationId", swhere)),
                    components=int(_req(st, "components", swhere)),
                    gapFraction=float(_req(st, "gapFraction", swhere)),
                    epiDistM=float(_req(st, "epiDistM", swhere)),
                    usable=bool(_req(st, "usable", swhere)),
                    reason=None if reason is None else str(reason),
                )
            )
        mag_type = _req(ev, "magType", where)
        event = KnownEvent(
            eventId=str(_req(ev, "eventId", where)),
            t=float(_req(ev, "t", where)),
            latitude=float(_req(ev, "latitude", where)),
            longitude=float(_req(ev, "longitude", where)),
            depthKm=float(_req(ev, "depthKm", where)),
            mag=_opt_float(_req(ev, "mag", where)),
            magType=None if mag_type is None else str(mag_type),
            windowStart=float(_req(ev, "windowStart", where)),
            windowEnd=float(_req(ev, "windowEnd", where)),
            stations=tuple(stations),
        )
        if not event.windowStart < event.windowEnd:
            raise ValueError(f"{where}: windowStart must be before windowEnd")
        events.append(event)
    ids = [e.eventId for e in events]
    if len(set(ids)) != len(ids):
        raise ValueError(f"windows.json: duplicate eventIds {ids}")
    params = _req(doc, "params", "windows.json")
    return KnownWindows(events=tuple(events), params=dict(params))


def load_known_windows(path: Path) -> KnownWindows:
    with path.open(encoding="utf-8") as fh:
        return parse_known_windows(json.load(fh))


@dataclass(frozen=True)
class StationInfo:
    """The ``Station`` fields the A/B needs."""

    id: str
    preprocessProfile: str
    usedInRun: bool
    channels: tuple[str, ...]


def load_stations(path: Path) -> dict[str, StationInfo]:
    """Read ``stations.parquet`` through ``hq_contracts.io`` (docs/02 -> Tables on disk)."""
    from hq_contracts.io import from_frame, read_table
    from hq_contracts.models import Station

    rows = from_frame(read_table(path), Station)
    return {
        s.id: StationInfo(
            id=s.id,
            preprocessProfile=s.preprocessProfile,
            usedInRun=s.usedInRun,
            channels=tuple(s.channels),
        )
        for s in rows
    }


def write_picks(picks: list[dict[str, Any]], path: Path) -> None:
    """Write ``Pick`` rows through ``hq_contracts.io`` (validates every row; empty is fine)."""
    from hq_contracts.io import to_frame, write_table
    from hq_contracts.models import Pick

    write_table(to_frame([Pick(**p) for p in picks], model=Pick), path, "Pick")


def select_channels(raw: obspy.Stream, channels: Sequence[str]) -> tuple[obspy.Stream, int]:
    """Keep only the station's selected channels (``read_window`` returns every cached one)."""
    wanted = set(channels)
    kept = obspy.Stream([tr for tr in raw if tr.stats.channel in wanted])
    return kept, len(raw) - len(kept)


# --- metrics ------------------------------------------------------------------------------------


def arrival_window(
    event: KnownEvent, epi_dist_m: float, cfg: ArrivalWindowConfig
) -> tuple[float, float]:
    """Times at which a pick may belong to ``event`` at a station ``epi_dist_m`` away."""
    hypo_m = math.hypot(epi_dist_m, max(event.depthKm, 0.0) * 1000.0)
    return event.t - cfg.preOriginS, event.t + hypo_m / cfg.minVelocityMps + cfg.postMarginS


@dataclass(frozen=True)
class BestPS:
    """Best picks at one station for one event."""

    bestP: dict[str, Any] | None
    bestS: dict[str, Any] | None  # max-probability S strictly after bestP
    bestSRaw: dict[str, Any] | None  # max-probability S, before the S-after-P filter
    violation: bool  # bestSRaw exists and is not after bestP
    nOutsideWindow: int = 0  # picks at this station outside the event's arrival window

    @property
    def hasPS(self) -> bool:
        return self.bestP is not None and self.bestS is not None


def _argmax_pick(picks: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for p in picks:  # highest prob; on a tie the earlier pick
        if best is None or (p["prob"], -p["t"]) > (best["prob"], -best["t"]):
            best = p
    return best


def best_phases(
    picks: Sequence[dict[str, Any]], window: tuple[float, float] | None = None
) -> BestPS:
    """Best P, best S after it, and the S-before-P check, over the picks inside ``window``."""
    inside = (
        list(picks) if window is None else [p for p in picks if window[0] <= p["t"] <= window[1]]
    )
    n_out = len(picks) - len(inside)
    best_p = _argmax_pick(p for p in inside if p["phase"] == "P")
    s_picks = [p for p in inside if p["phase"] == "S"]
    best_s_raw = _argmax_pick(s_picks)
    if best_p is None:
        return BestPS(None, None, best_s_raw, violation=False, nOutsideWindow=n_out)
    best_s = _argmax_pick(p for p in s_picks if p["t"] > best_p["t"])
    violation = best_s_raw is not None and best_s_raw["t"] <= best_p["t"]
    return BestPS(best_p, best_s, best_s_raw, violation=violation, nOutsideWindow=n_out)


def spearman_rho(times: Sequence[float], dists: Sequence[float], min_n: int) -> float:
    """Spearman rho of best-P time vs epicentral distance; NaN when undefined or too few points."""
    if len(times) != len(dists):
        raise ValueError("times and dists differ in length")
    if len(times) < min_n or len(set(times)) < 2 or len(set(dists)) < 2:
        return math.nan
    return float(spearmanr(times, dists).statistic)


@dataclass(frozen=True)
class EventMetrics:
    nStations: int
    nWithPicks: int
    nP: int
    nS: int
    nPS: int
    violations: int
    rho: float  # over stations with a best P (gated by Check B)
    nRho: int
    rhoPS: float  # over stations with a best P and a best S (reported only)
    nRhoPS: int
    nOutsideWindow: int  # picks outside the arrival window, all stations


def event_metrics(
    best: Mapping[str, BestPS], dist_m: Mapping[str, float], min_n_rho: int
) -> EventMetrics:
    with_p = sorted(sid for sid, b in best.items() if b.bestP is not None)
    with_ps = [sid for sid in with_p if best[sid].hasPS]

    def rho_over(sids: Sequence[str]) -> float:
        times = [best[sid].bestP["t"] for sid in sids]  # type: ignore[index]
        return spearman_rho(times, [dist_m[sid] for sid in sids], min_n_rho)

    return EventMetrics(
        nStations=len(best),
        nWithPicks=sum(1 for b in best.values() if b.bestP is not None or b.bestSRaw is not None),
        nP=len(with_p),
        nS=sum(1 for b in best.values() if b.bestSRaw is not None),
        nPS=len(with_ps),
        violations=sum(1 for b in best.values() if b.violation),
        rho=rho_over(with_p),
        nRho=len(with_p),
        rhoPS=rho_over(with_ps),
        nRhoPS=len(with_ps),
        nOutsideWindow=sum(b.nOutsideWindow for b in best.values()),
    )


def _rho_key(rho: float) -> float:
    return -math.inf if math.isnan(rho) else rho


def is_consistent(m: EventMetrics, min_rho: float) -> bool:
    """The Check B criteria that apply to any station subset: no violation, rho >= minRho."""
    return m.violations == 0 and not math.isnan(m.rho) and m.rho >= min_rho


@dataclass(frozen=True)
class ComboSummary:
    weights: str
    profile: str
    nEvents: int
    nStationWindows: int
    nEventsConsistent: int  # events with no violation and rho >= checkB.minRho
    totalPS: int
    violations: int
    nEventsWithRho: int
    meanRho: float  # over events with a defined rho
    minRho: float

    @property
    def key(self) -> tuple[int, int, int, int, float]:
        """The A/B metric, larger is better (see the module docstring)."""
        return (
            self.nEventsConsistent,
            self.totalPS,
            -self.violations,
            self.nEventsWithRho,
            _rho_key(self.meanRho),
        )


def summarize(
    weights: str, profile: str, metrics: Sequence[EventMetrics], min_rho: float
) -> ComboSummary:
    rhos = [m.rho for m in metrics if not math.isnan(m.rho)]
    return ComboSummary(
        weights=weights,
        profile=profile,
        nEvents=len(metrics),
        nStationWindows=sum(m.nStations for m in metrics),
        nEventsConsistent=sum(1 for m in metrics if is_consistent(m, min_rho)),
        totalPS=sum(m.nPS for m in metrics),
        violations=sum(m.violations for m in metrics),
        nEventsWithRho=len(rhos),
        meanRho=float(np.mean(rhos)) if rhos else math.nan,
        minRho=float(np.min(rhos)) if rhos else math.nan,
    )


def choose_weights(
    summaries: Sequence[ComboSummary], candidate_order: Sequence[str]
) -> dict[str, str]:
    """Per profile: the weights with the largest ``ComboSummary.key``, then candidate order."""
    rank = {w: i for i, w in enumerate(candidate_order)}
    chosen: dict[str, ComboSummary] = {}
    for s in summaries:
        cur = chosen.get(s.profile)
        if cur is None or (s.key, -rank[s.weights]) > (cur.key, -rank[cur.weights]):
            chosen[s.profile] = s
    return {profile: s.weights for profile, s in chosen.items()}


@dataclass(frozen=True)
class VariantComparison:
    """Base and variant profile, each with its chosen weights, on the same station-windows."""

    base: str
    variant: str
    nStationWindows: int  # station-windows both profiles processed
    baseSummary: ComboSummary
    variantSummary: ComboSummary

    @property
    def variantWins(self) -> bool:
        return self.nStationWindows > 0 and self.variantSummary.key > self.baseSummary.key


def adopt_variants(
    bases: Iterable[str], comparisons: Sequence[VariantComparison]
) -> dict[str, str]:
    """Per base profile: the first variant (config order) that strictly beats it, else the base."""
    adopted = {base: base for base in bases}
    for c in comparisons:
        if c.base not in adopted:
            raise ValueError(f"comparison for unknown base profile {c.base!r}")
        if adopted[c.base] == c.base and c.variantWins:
            adopted[c.base] = c.variant
    return adopted


def _all_variants(profile_variants: Mapping[str, Sequence[str]]) -> set[str]:
    return {v for variants in profile_variants.values() for v in variants}


# --- Check B ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckBEvent:
    eventId: str
    nStations: int
    nPS: int
    violations: int
    rho: float
    nRho: int
    rhoPS: float
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class CheckBResult:
    events: tuple[CheckBEvent, ...]
    nPass: int
    passed: bool


def evaluate_check_b(metrics: Mapping[str, EventMetrics], cfg: CheckBConfig) -> CheckBResult:
    """Per event: P&S stations >= minStationsPS, no P-before-S violation, rho >= minRho."""
    events = []
    for event_id, m in metrics.items():
        failures = []
        if m.nPS < cfg.minStationsPS:
            failures.append(f"P&S stations {m.nPS} < {cfg.minStationsPS}")
        if m.violations > 0:
            failures.append(f"{m.violations} station(s) with S not after P")
        if math.isnan(m.rho):
            failures.append("rho undefined")
        elif m.rho < cfg.minRho:
            failures.append(f"rho {m.rho:.3f} < {cfg.minRho}")
        events.append(
            CheckBEvent(
                eventId=event_id,
                nStations=m.nStations,
                nPS=m.nPS,
                violations=m.violations,
                rho=m.rho,
                nRho=m.nRho,
                rhoPS=m.rhoPS,
                passed=not failures,
                failures=tuple(failures),
            )
        )
    n_pass = sum(1 for e in events if e.passed)
    return CheckBResult(events=tuple(events), nPass=n_pass, passed=n_pass >= cfg.minEventsPass)


def _fmt_rho(rho: float) -> str:
    return "nan" if math.isnan(rho) else f"{rho:.3f}"


def format_check_b(result: CheckBResult, cfg: CheckBConfig) -> str:
    header = (
        f"{'event':<24} {'stations':>8} {'P&S':>5} {'S<=P':>5} {'rho':>7} {'nRho':>5} "
        f"{'rhoPS':>7}  result"
    )
    lines = [
        (
            f"Check B (P&S >= {cfg.minStationsPS}, no S-before-P, rho >= {cfg.minRho}, "
            f">= {cfg.minEventsPass} events)"
        ),
        (
            "rho: Spearman(best-P time, epicentral distance) over stations with a best P in the "
            "arrival window; rhoPS: same over P&S stations (reported, not gated)"
        ),
        header,
        "-" * len(header),
    ]
    for e in result.events:
        verdict = "PASS" if e.passed else "FAIL: " + "; ".join(e.failures)
        lines.append(
            f"{e.eventId:<24} {e.nStations:>8} {e.nPS:>5} {e.violations:>5} "
            f"{_fmt_rho(e.rho):>7} {e.nRho:>5} {_fmt_rho(e.rhoPS):>7}  {verdict}"
        )
    overall = "PASS" if result.passed else "FAIL"
    lines.append(f"overall: {overall} ({result.nPass}/{len(result.events)} events pass)")
    return "\n".join(lines)


# --- orchestration ------------------------------------------------------------------------------


@dataclass(frozen=True)
class AbIO:
    """Everything the A/B reads or writes outside the run directory; injectable for tests."""

    read_window: ReadWindow
    for_picking: ForPicking
    display_copy: DisplayCopy
    load_model: ModelLoader
    load_stations: Callable[[Path], dict[str, StationInfo]]
    write_picks: Callable[[list[dict[str, Any]], Path], None]


def default_io() -> AbIO:
    """Real implementations: SEIS-05 cache, SEIS-03 preprocessing, seisbench, hq_contracts.io."""
    from hq.ingest.cache import read_window
    from hq.preprocess import display_copy, for_picking

    return AbIO(
        read_window=read_window,
        for_picking=for_picking,
        display_copy=display_copy,
        load_model=load_model,
        load_stations=load_stations,
        write_picks=write_picks,
    )


@dataclass(frozen=True)
class AbResult:
    eventRows: list[dict[str, Any]]
    summaryRows: list[dict[str, Any]]
    comparisonRows: list[dict[str, Any]]
    chosenWeights: dict[str, str]
    adoptedProfiles: dict[str, str]
    comparisons: list[VariantComparison]
    checkB: CheckBResult
    picks: list[dict[str, Any]]
    paths: dict[str, Path]
    counts: dict[str, int]
    notApplicable: dict[tuple[str, str, str], str] = field(default_factory=dict)
    runtimeS: float = 0.0


Key = tuple[str, str, str, str]  # eventId, stationId, profile, weights
_INT_COLUMNS = {
    *(f for f in EventMetrics.__dataclass_fields__ if not f.startswith("rho")),
    *("nNoData", "nNotApplicable", "picksP", "picksS", "nBlocks", "nBlocksTooShort"),
    *("droppedNearEdge", "nOverlaps"),
    *(f for f in ComboSummary.__dataclass_fields__ if f.startswith("n") or f == "totalPS"),
    "violations",
}


def _profiles_for(base: str, variants: Mapping[str, Sequence[str]]) -> list[str]:
    return [base, *variants.get(base, [])]


def _safe_name(event_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", event_id)


def run_ab(run_dir: Path, cache_dir: Path, cfg: SignalConfig, io: AbIO | None = None) -> AbResult:
    """Pick the known-event windows with every candidate, choose weights, evaluate Check B."""
    t_start = time.perf_counter()
    io = io if io is not None else default_io()
    picker = cfg.picker
    ab_cfg = picker.ab
    min_rho = ab_cfg.checkB.minRho
    known_dir = run_dir / KNOWN_DIR
    windows = load_known_windows(known_dir / WINDOWS_FILE)
    stations = io.load_stations(run_dir / "stations.parquet")
    variants = ab_cfg.profileVariants
    weights_list = list(picker.candidateWeights)
    log.info(
        "A/B: %d events, %d stations in stations.parquet, weights %s, variants %s",
        len(windows.events),
        len(stations),
        weights_list,
        dict(variants),
    )

    usable: dict[str, list[KnownStation]] = {}
    base_of: dict[str, str] = {}
    n_not_used_in_run = 0
    for event in windows.events:
        usable[event.eventId] = [s for s in event.stations if s.usable]
        for ks in usable[event.eventId]:
            info = stations.get(ks.stationId)
            if info is None:
                raise ValueError(
                    f"{event.eventId}: usable station {ks.stationId} missing from stations.parquet"
                )
            base_of[ks.stationId] = info.preprocessProfile
            if not info.usedInRun:
                n_not_used_in_run += 1
    clash = _all_variants(variants) & set(base_of.values())
    if clash:
        raise ValueError(f"variant profiles {sorted(clash)} are also native station profiles")
    if n_not_used_in_run:
        log.warning(
            "%d usable station-windows have usedInRun=false; picked anyway", n_not_used_in_run
        )
    win = {
        event.eventId: {
            ks.stationId: arrival_window(event, ks.epiDistM, ab_cfg.arrivalWindow)
            for ks in usable[event.eventId]
        }
        for event in windows.events
    }
    dist = {
        event.eventId: {ks.stationId: ks.epiDistM for ks in usable[event.eventId]}
        for event in windows.events
    }

    models = {w: io.load_model(w, picker) for w in weights_list}

    store: dict[Key, list[dict[str, Any]]] = {}
    diags: dict[Key, PickDiagnostics] = {}
    no_data: set[tuple[str, str]] = set()
    not_applicable: dict[tuple[str, str, str], str] = {}  # (eventId, stationId, profile) -> why
    n_channels_dropped = 0
    for event in windows.events:
        t_event = time.perf_counter()
        for ks in usable[event.eventId]:
            sid = ks.stationId
            base = base_of[sid]
            raw, n_dropped = select_channels(
                io.read_window(sid, event.windowStart, event.windowEnd, cache_dir=cache_dir),
                stations[sid].channels,
            )
            if n_dropped:
                n_channels_dropped += n_dropped
                log.info(
                    "%s %s: dropped %d traces outside channels %s",
                    event.eventId,
                    sid,
                    n_dropped,
                    list(stations[sid].channels),
                )
            profiles = _profiles_for(base, variants)
            if len(raw) == 0:
                no_data.add((event.eventId, sid))
                log.warning("%s %s: no data on the station channels", event.eventId, sid)
                for profile in profiles:
                    for w in weights_list:
                        store[(event.eventId, sid, profile, w)] = []
                continue
            for profile in profiles:
                try:
                    prepared = prepare_station(raw, sid, profile, cfg, for_picking=io.for_picking)
                except ValueError as exc:
                    if profile == base:
                        raise RuntimeError(
                            f"{event.eventId} {sid}: preprocessing with its own profile {base} "
                            "failed"
                        ) from exc
                    # The variant's preprocessing rejects this station's data (e.g. a rate
                    # outside the variant's range): not applicable here, counted and reported.
                    not_applicable[(event.eventId, sid, profile)] = str(exc)
                    log.warning(
                        "%s %s: variant %s not applicable, compared without it: %s",
                        event.eventId,
                        sid,
                        profile,
                        exc,
                    )
                    continue
                for w in weights_list:
                    picks, diag = pick_prepared(prepared, cfg, models[w], w)
                    store[(event.eventId, sid, profile, w)] = picks
                    diags[(event.eventId, sid, profile, w)] = diag
        log.info(
            "%s: picked %d usable stations x %d weights in %.1f s",
            event.eventId,
            len(usable[event.eventId]),
            len(weights_list),
            time.perf_counter() - t_event,
        )

    bases = sorted(set(base_of.values()))
    profile_order = [p for base in bases for p in _profiles_for(base, variants)]

    def stations_on(event_id: str, profile: str) -> list[str]:
        """Usable stations of ``event_id`` that ``profile`` processed."""
        return [
            ks.stationId
            for ks in usable[event_id]
            if profile in _profiles_for(base_of[ks.stationId], variants)
            and (event_id, ks.stationId, profile) not in not_applicable
        ]

    def metrics_for(event_id: str, profile: str, w: str, sids: Sequence[str]) -> EventMetrics:
        best = {
            sid: best_phases(store[(event_id, sid, profile, w)], win[event_id][sid]) for sid in sids
        }
        return event_metrics(best, dist[event_id], ab_cfg.minStationsForRho)

    event_rows: list[dict[str, Any]] = []
    per_combo: dict[tuple[str, str], list[EventMetrics]] = {}
    for event in windows.events:
        for w in weights_list:
            for profile in profile_order:
                sids = stations_on(event.eventId, profile)
                n_na = sum(1 for k in not_applicable if k[0] == event.eventId and k[2] == profile)
                if not sids and not n_na:
                    continue
                m = metrics_for(event.eventId, profile, w, sids)
                per_combo.setdefault((w, profile), []).append(m)
                ds = [diags[k] for sid in sids if (k := (event.eventId, sid, profile, w)) in diags]
                event_rows.append(
                    {
                        "rowType": "event",
                        "eventId": event.eventId,
                        "weights": w,
                        "profile": profile,
                        **asdict(m),
                        "nNoData": sum(1 for sid in sids if (event.eventId, sid) in no_data),
                        "nNotApplicable": n_na,
                        "picksP": sum(d.picksP for d in ds),
                        "picksS": sum(d.picksS for d in ds),
                        "nBlocks": sum(d.nBlocks for d in ds),
                        "nBlocksTooShort": sum(d.nBlocksTooShort for d in ds),
                        "droppedNearEdge": sum(d.droppedNearEdge for d in ds),
                        "secondsBlinded": round(sum(d.secondsBlinded for d in ds), 3),
                        "nOverlaps": sum(d.nOverlaps for d in ds),
                        "runtimeS": round(sum(d.runtimeS for d in ds), 3),
                    }
                )
                log.info(
                    "%s %s %s: stations %d, P %d, S %d, P&S %d, violations %d, rho %s, "
                    "outside arrival window %d",
                    event.eventId,
                    w,
                    profile,
                    m.nStations,
                    m.nP,
                    m.nS,
                    m.nPS,
                    m.violations,
                    _fmt_rho(m.rho),
                    m.nOutsideWindow,
                )

    summaries = [
        summarize(w, profile, per_combo[(w, profile)], min_rho)
        for w in weights_list
        for profile in profile_order
        if (w, profile) in per_combo and sum(m.nStations for m in per_combo[(w, profile)]) > 0
    ]
    chosen = choose_weights(summaries, weights_list)

    # Like-for-like: base and variant, each with its chosen weights, on the station-windows the
    # variant could process.
    comparisons: list[VariantComparison] = []
    for base in bases:
        for variant in variants.get(base, []):
            if variant not in chosen:
                log.warning("variant %s processed no %s station; not compared", variant, base)
                continue
            common = {e.eventId: stations_on(e.eventId, variant) for e in windows.events}
            common = {ev: sids for ev, sids in common.items() if sids}
            comparisons.append(
                VariantComparison(
                    base=base,
                    variant=variant,
                    nStationWindows=sum(len(s) for s in common.values()),
                    baseSummary=summarize(
                        chosen[base],
                        base,
                        [metrics_for(ev, base, chosen[base], s) for ev, s in common.items()],
                        min_rho,
                    ),
                    variantSummary=summarize(
                        chosen[variant],
                        variant,
                        [metrics_for(ev, variant, chosen[variant], s) for ev, s in common.items()],
                        min_rho,
                    ),
                )
            )
    adopted = adopt_variants(bases, comparisons)
    adopted_set = set(adopted.values())
    summary_rows = [
        {
            "rowType": "summary",
            "eventId": "",
            **asdict(s),
            "chosen": chosen.get(s.profile) == s.weights,
            "profileAdopted": s.profile in adopted_set,
        }
        for s in summaries
    ]
    comparison_rows = [
        {
            "rowType": "variantCompare",
            "eventId": "",
            **asdict(summary),
            "comparedWith": other,
            "profileAdopted": adopted[c.base] == summary.profile,
        }
        for c in comparisons
        for summary, other in ((c.baseSummary, c.variant), (c.variantSummary, c.base))
    ]
    for c in comparisons:
        log.info(
            "variant %s vs %s on %d station-windows: key %s vs %s -> %s",
            c.variant,
            c.base,
            c.nStationWindows,
            c.variantSummary.key,
            c.baseSummary.key,
            "adopted" if adopted[c.base] == c.variant else "not adopted",
        )
    for base, profile in adopted.items():
        log.info(
            "A/B choice: stations on %s -> profile %s, weights %s", base, profile, chosen[profile]
        )

    # Final picks: every usable station with its adopted profile and that profile's chosen
    # weights; a station the adopted variant could not process keeps its base profile.
    n_variant_fallback = 0

    def final_key(event_id: str, station_id: str) -> Key:
        base = base_of[station_id]
        profile = adopted[base]
        if (event_id, station_id, profile) in not_applicable:
            profile = base
        return (event_id, station_id, profile, chosen[profile])

    final_metrics: dict[str, EventMetrics] = {}
    final_best: dict[str, dict[str, BestPS]] = {}
    final_picks: list[dict[str, Any]] = []
    for event in windows.events:
        best = {}
        for ks in usable[event.eventId]:
            key = final_key(event.eventId, ks.stationId)
            if key[2] != adopted[base_of[ks.stationId]]:
                n_variant_fallback += 1
            picks = store[key]
            best[ks.stationId] = best_phases(picks, win[event.eventId][ks.stationId])
            final_picks.extend(picks)
        final_best[event.eventId] = best
        final_metrics[event.eventId] = event_metrics(
            best, dist[event.eventId], ab_cfg.minStationsForRho
        )
    if n_variant_fallback:
        log.warning(
            "%d station-windows kept their base profile because the adopted variant could not "
            "process them",
            n_variant_fallback,
        )
    check_b = evaluate_check_b(final_metrics, ab_cfg.checkB)

    unique: dict[str, dict[str, Any]] = {}
    for p in final_picks:
        unique.setdefault(p["id"], p)
    n_duplicates = len(final_picks) - len(unique)
    if n_duplicates:
        log.warning("%d duplicate pick ids from overlapping event windows dropped", n_duplicates)
    picks_out = sorted(unique.values(), key=lambda p: (p["t"], p["stationId"], p["phase"]))

    final_diags = [
        diags[k]
        for e in windows.events
        for ks in usable[e.eventId]
        if (k := final_key(e.eventId, ks.stationId)) in diags
    ]
    counts = {
        "events": len(windows.events),
        "stationWindows": sum(len(v) for v in usable.values()),
        "stationWindowsNoData": len(no_data),
        "tracesOutsideStationChannels": n_channels_dropped,
        "variantNotApplicable": len(not_applicable),
        "variantFallbackStationWindows": n_variant_fallback,
        "picks": len(picks_out),
        "picksP": sum(1 for p in picks_out if p["phase"] == "P"),
        "picksS": sum(1 for p in picks_out if p["phase"] == "S"),
        "picksOutsideArrivalWindow": sum(m.nOutsideWindow for m in final_metrics.values()),
        "duplicatePicksDropped": n_duplicates,
        "droppedNearEdge": sum(d.droppedNearEdge for d in final_diags),
        "overlapsTreatedAsGaps": sum(d.nOverlaps for d in final_diags),
        "checkBEventsPassed": check_b.nPass,
    }

    # Tables and the verdict first, so a plotting failure never hides them.
    known_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "abCsv": known_dir / AB_CSV,
        "abJson": known_dir / AB_JSON,
        "picks": known_dir / PICKS_FILE,
    }
    table = pd.DataFrame(event_rows + summary_rows + comparison_rows)
    for col in _INT_COLUMNS & set(table.columns):  # keep counts integer next to NaN cells
        table[col] = table[col].astype("Int64")
    table.to_csv(paths["abCsv"], index=False)
    with paths["abJson"].open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "metric": (
                    "per profile, larger is better in order: events with no S-before-P and "
                    "rho >= checkB.minRho, total P&S stations, fewer violations, events with a "
                    "defined rho, mean rho; tie -> candidateWeights order"
                ),
                "rhoStations": (
                    "rho: stations with a best P inside the arrival window (gated); rhoPS: "
                    "stations with P and S (reported only)"
                ),
                "variantRule": (
                    "a variant is adopted only if it strictly beats its base profile on the same "
                    "station-windows; stations it cannot process keep the base profile"
                ),
                "chosenWeightsByProfile": chosen,
                "adoptedProfileByBase": adopted,
                "variantComparisons": [_json_safe(asdict(c)) for c in comparisons],
                "variantNotApplicable": [
                    {"eventId": ev, "stationId": sid, "profile": prof, "reason": why}
                    for (ev, sid, prof), why in sorted(not_applicable.items())
                ],
                "checkB": {
                    "params": ab_cfg.checkB.model_dump(mode="json"),
                    "passed": check_b.passed,
                    "nPass": check_b.nPass,
                    "events": [_json_safe(asdict(e)) for e in check_b.events],
                },
                "counts": counts,
                "runtimeSBeforePlots": round(time.perf_counter() - t_start, 3),
                "picker": picker.model_dump(mode="json"),
            },
            fh,
            indent=2,
        )
    io.write_picks(picks_out, paths["picks"])

    for event in windows.events:
        png = known_dir / f"record_section_{_safe_name(event.eventId)}.png"
        plot_record_section(
            event,
            usable[event.eventId],
            {
                ks.stationId: store[final_key(event.eventId, ks.stationId)]
                for ks in usable[event.eventId]
            },
            final_best[event.eventId],
            {sid: stations[sid].channels for sid in dist[event.eventId]},
            cfg,
            io,
            cache_dir,
            png,
        )
        paths[f"recordSection:{event.eventId}"] = png

    runtime_s = time.perf_counter() - t_start
    log.info("A/B done in %.1f s: %s", runtime_s, counts)
    return AbResult(
        eventRows=event_rows,
        summaryRows=summary_rows,
        comparisonRows=comparison_rows,
        chosenWeights=chosen,
        adoptedProfiles=adopted,
        comparisons=comparisons,
        checkB=check_b,
        picks=picks_out,
        paths=paths,
        counts=counts,
        notApplicable=not_applicable,
        runtimeS=runtime_s,
    )


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    return obj


# --- record section -----------------------------------------------------------------------------


def plot_record_section(
    event: KnownEvent,
    stations: Sequence[KnownStation],
    picks_by_station: Mapping[str, Sequence[dict[str, Any]]],
    best_by_station: Mapping[str, BestPS],
    channels_by_station: Mapping[str, Sequence[str]],
    cfg: SignalConfig,
    io: AbIO,
    cache_dir: Path,
    path: Path,
) -> int:
    """Display copies sorted by epicentral distance, each normalized, with P and S picks marked.

    Best P / S (from the arrival window) are bold; every other pick is faint. Returns how many
    best picks fall outside the plotted span (logged; widen ``recordSection.windowS`` to see them).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    rs = cfg.picker.ab.recordSection
    w0, w1 = rs.windowS
    ordered = sorted(stations, key=lambda s: (s.epiDistM, s.stationId))
    height = max(rs.minHeightIn, rs.heightPerTraceIn * len(ordered))
    fig = Figure(figsize=(rs.widthIn, height), facecolor=_SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.set_facecolor(_SURFACE)
    labels = []
    n_skipped = 0
    n_best_off_plot = 0
    for row, ks in enumerate(ordered):
        labels.append(f"{ks.stationId}  {ks.epiDistM / 1000:.1f} km")
        raw, _ = select_channels(
            io.read_window(
                ks.stationId, event.t + w0 - rs.padS, event.t + w1 + rs.padS, cache_dir=cache_dir
            ),
            channels_by_station[ks.stationId],
        )
        segments = []
        if len(raw):
            shown = io.display_copy(raw, rs.bandHz).select(component=rs.component)
            shown = shown.slice(obspy.UTCDateTime(event.t + w0), obspy.UTCDateTime(event.t + w1))
            segments = [tr for tr in shown if tr.stats.npts > 1]
        peak = max((float(np.max(np.abs(tr.data))) for tr in segments), default=0.0)
        if not segments or peak == 0.0:
            n_skipped += 1
            labels[-1] += " (no trace)"
        else:
            for tr in segments:
                x = tr.times() + (tr.stats.starttime.timestamp - event.t)
                y = row + rs.traceHalfHeight * np.asarray(tr.data, dtype=float) / peak
                ax.plot(x, y, color=_INK, linewidth=_TRACE_LINEWIDTH)
        best = best_by_station.get(ks.stationId)
        best_picks = [] if best is None else [best.bestP, best.bestS]
        for p in picks_by_station.get(ks.stationId, []):
            if any(p is b for b in best_picks):
                continue
            ax.vlines(
                p["t"] - event.t,
                row - rs.traceHalfHeight / 2,
                row + rs.traceHalfHeight / 2,
                colors=_COLOR_P if p["phase"] == "P" else _COLOR_S,
                linewidth=_OTHER_PICK_LINEWIDTH,
                alpha=_OTHER_PICK_ALPHA,
            )
        for best_pick, color in zip(best_picks, (_COLOR_P, _COLOR_S), strict=False):
            if best_pick is None:
                continue
            if not w0 <= best_pick["t"] - event.t <= w1:
                n_best_off_plot += 1
            ax.vlines(
                best_pick["t"] - event.t,
                row - rs.traceHalfHeight,
                row + rs.traceHalfHeight,
                colors=color,
                linewidth=_BEST_PICK_LINEWIDTH,
            )
    if n_skipped:
        log.warning(
            "%s: %d stations without a plottable %s trace", event.eventId, n_skipped, rs.component
        )
    if n_best_off_plot:
        log.warning(
            "%s: %d best picks outside the plotted span %s s",
            event.eventId,
            n_best_off_plot,
            rs.windowS,
        )
    ax.set_xlim(w0, w1)
    ax.set_ylim(len(ordered) - 0.5, -0.5)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, fontsize=_LABEL_FONTSIZE, color=_INK)
    ax.tick_params(axis="x", colors=_INK)
    ax.set_xlabel("Seconds after public-catalog origin time", color=_INK)
    ax.grid(axis="x", color=_INK_MUTED, alpha=_GRID_ALPHA, linewidth=_GRID_LINEWIDTH)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    origin = datetime.fromtimestamp(event.t, tz=UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    band = f"{rs.bandHz[0]:g}-{rs.bandHz[1]:g} Hz"
    ax.set_title(f"{event.eventId}  {origin} UTC  ({rs.component}, {band})", color=_INK)
    fig.legend(
        handles=[
            Line2D([], [], color=_COLOR_P, linewidth=_BEST_PICK_LINEWIDTH, label="P (best)"),
            Line2D(
                [], [], color=_COLOR_S, linewidth=_BEST_PICK_LINEWIDTH, label="S (best, after P)"
            ),
            Line2D(
                [],
                [],
                color=_COLOR_P,
                linewidth=_OTHER_PICK_LINEWIDTH,
                alpha=_OTHER_PICK_ALPHA,
                label="other P picks",
            ),
            Line2D(
                [],
                [],
                color=_COLOR_S,
                linewidth=_OTHER_PICK_LINEWIDTH,
                alpha=_OTHER_PICK_ALPHA,
                label="other S picks",
            ),
        ],
        loc="outside lower center",
        ncol=_LEGEND_COLUMNS,
        fontsize=_LABEL_FONTSIZE,
        frameon=False,
    )
    fig.savefig(path, dpi=rs.dpi, facecolor=_SURFACE)
    log.info("wrote %s (%d stations)", path, len(ordered))
    return n_best_off_plot


# --- stage + CLI --------------------------------------------------------------------------------


class StageContext(Protocol):
    """The part of H4's ``hq.runs.RunContext`` this stage uses (docs/02 -> Stage API)."""

    @property
    def run_dir(self) -> Path: ...

    @property
    def cache_dir(self) -> Path: ...

    @property
    def config(self) -> Any: ...

    def path(self, name: str) -> Path: ...

    def record(
        self,
        stage: str,
        *,
        runtime_s: float,
        counts: dict[str, int],
        params: dict[str, Any] | None = None,
    ) -> None: ...


def run(ctx: StageContext) -> None:
    """Entry point for the known-event A/B on a run.

    It is a sub-step, not a pipeline stage (H4's registry rejects its name in ``ctx.record``), so
    its runtime, counts and params go to ``known/pick_known.record.json``.
    """
    from hq.ingest.windows import KNOWN_DIR, write_step_record

    signal = ctx.config.signal
    result = run_ab(ctx.run_dir, ctx.cache_dir, signal)
    params = {
        **signal.picker.model_dump(mode="json"),
        "preprocess": signal.preprocess.model_dump(mode="json"),  # the profiles decide the picks
    }
    write_step_record(ctx.path(KNOWN_DIR), STAGE, result.runtimeS, result.counts, params)


def load_signal_config(config_dir: Path) -> SignalConfig:
    with (config_dir / "signal.yaml").open(encoding="utf-8") as fh:
        return SignalConfig.model_validate(yaml.safe_load(fh))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hq.pick.ab", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_signal_config(args.config_dir)
    result = run_ab(args.run_dir, args.cache_dir, cfg)
    print()
    print("Chosen weights per profile (copy into signal.yaml picker.weightsByProfile):")
    for profile, weights in sorted(result.chosenWeights.items()):
        print(f"  {profile}: {weights}")
    print("Profile to use per station profile:")
    for base, profile in sorted(result.adoptedProfiles.items()):
        print(f"  {base} -> {profile}")
    print()
    print(format_check_b(result.checkB, cfg.picker.ab.checkB))
    print()
    for name, path in result.paths.items():
        print(f"{name}: {path}")
    print("(CLI run: known/pick_known.record.json not written; hq.pick.ab.run(ctx) writes it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

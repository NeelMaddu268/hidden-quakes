"""PhaseNet weight A/B on the known-event windows, plus the Check B evaluator (SEIS-04).

Reads ``runs/<id>/known/windows.json`` (SEIS-02), ``runs/<id>/stations.parquet`` (SEIS-01) and
waveforms via ``hq.ingest.cache.read_window`` (SEIS-05). For every event x weights x profile it
picks every usable station, then per station keeps the best P (max probability) and the best S
(max probability after that P). Metrics per (weights, profile): stations with P and S, Spearman
rho between best-P time and epicentral distance, and P-before-S violations (a station whose
max-probability S is not after its best P). Per profile the weights with the most P-and-S stations
win, ties broken by the higher mean rho; a variant profile (``borehole-B``) is adopted only when
it beats its base profile on the same metric.

Writes to ``runs/<id>/known/``: ``ab.csv`` (one row per event x weights x profile + one summary row
per weights x profile), ``ab.json`` (chosen weights, adopted profiles, Check B), ``picks.parquet``
(every pick >= threshold from the chosen combination, ``Pick`` schema) and one
``record_section_<eventId>.png`` per event.

CLI::

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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import obspy
import pandas as pd
import yaml
from scipy.stats import spearmanr

from hq.config.signal import CheckBConfig, PickerConfig, SignalConfig
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

# Record-section colours: categorical slots 1 and 2 of the dataviz reference palette; traces and
# text stay in neutral ink so colour only carries the phase identity.
_COLOR_P = "#2a78d6"
_COLOR_S = "#eb6834"
_INK = "#52514e"
_INK_MUTED = "#8a8985"
_SURFACE = "#fcfcfb"


# --- inputs -------------------------------------------------------------------------------------


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


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def parse_known_windows(doc: Mapping[str, Any]) -> KnownWindows:
    """Parse the shared ``known/windows.json`` document (SEIS-02 writes it)."""
    events: list[KnownEvent] = []
    for i, ev in enumerate(_req(doc, "events", "windows.json")):
        where = f"windows.json events[{i}]"
        stations = []
        for j, st in enumerate(_req(ev, "stations", where)):
            swhere = f"{where}.stations[{j}]"
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


def load_stations(path: Path) -> dict[str, StationInfo]:
    """Read ``stations.parquet`` through ``hq_contracts.io`` (docs/02 -> Tables on disk)."""
    from hq_contracts.io import from_frame, read_table
    from hq_contracts.models import Station

    rows = from_frame(read_table(path), Station)
    return {
        s.id: StationInfo(id=s.id, preprocessProfile=s.preprocessProfile, usedInRun=s.usedInRun)
        for s in rows
    }


def write_picks(picks: list[dict[str, Any]], path: Path) -> None:
    """Write ``Pick`` rows through ``hq_contracts.io`` (validates every row against the model)."""
    from hq_contracts.io import to_frame, write_table
    from hq_contracts.models import Pick

    write_table(to_frame([Pick(**p) for p in picks]), path, "Pick")


# --- metrics ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BestPS:
    """Best picks at one station for one event."""

    bestP: dict[str, Any] | None
    bestS: dict[str, Any] | None  # max-probability S strictly after bestP
    bestSRaw: dict[str, Any] | None  # max-probability S, before the S-after-P filter
    violation: bool  # bestSRaw exists and is not after bestP

    @property
    def hasPS(self) -> bool:
        return self.bestP is not None and self.bestS is not None


def _argmax_pick(picks: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for p in picks:  # highest prob; on a tie the earlier pick
        if best is None or (p["prob"], -p["t"]) > (best["prob"], -best["t"]):
            best = p
    return best


def best_phases(picks: Sequence[dict[str, Any]]) -> BestPS:
    best_p = _argmax_pick(p for p in picks if p["phase"] == "P")
    s_picks = [p for p in picks if p["phase"] == "S"]
    best_s_raw = _argmax_pick(s_picks)
    if best_p is None:
        return BestPS(bestP=None, bestS=None, bestSRaw=best_s_raw, violation=False)
    best_s = _argmax_pick(p for p in s_picks if p["t"] > best_p["t"])
    violation = best_s_raw is not None and best_s_raw["t"] <= best_p["t"]
    return BestPS(bestP=best_p, bestS=best_s, bestSRaw=best_s_raw, violation=violation)


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
    rho: float
    nRho: int


def event_metrics(
    best: Mapping[str, BestPS], dist_m: Mapping[str, float], min_n_rho: int
) -> EventMetrics:
    with_p = sorted(sid for sid, b in best.items() if b.bestP is not None)
    times = [best[sid].bestP["t"] for sid in with_p]  # type: ignore[index]
    dists = [dist_m[sid] for sid in with_p]
    return EventMetrics(
        nStations=len(best),
        nWithPicks=sum(1 for b in best.values() if b.bestP is not None or b.bestSRaw is not None),
        nP=len(with_p),
        nS=sum(1 for b in best.values() if b.bestSRaw is not None),
        nPS=sum(1 for b in best.values() if b.hasPS),
        violations=sum(1 for b in best.values() if b.violation),
        rho=spearman_rho(times, dists, min_n_rho),
        nRho=len(with_p),
    )


def _rho_key(rho: float) -> float:
    return -math.inf if math.isnan(rho) else rho


@dataclass(frozen=True)
class ComboSummary:
    weights: str
    profile: str
    nEvents: int
    totalPS: int
    meanRho: float
    minRho: float
    nEventsWithRho: int
    violations: int

    @property
    def key(self) -> tuple[int, float]:
        """The A/B metric: P-and-S stations first, then mean rho (NaN ranks last)."""
        return (self.totalPS, _rho_key(self.meanRho))


def summarize(weights: str, profile: str, metrics: Sequence[EventMetrics]) -> ComboSummary:
    rhos = [m.rho for m in metrics if not math.isnan(m.rho)]
    return ComboSummary(
        weights=weights,
        profile=profile,
        nEvents=len(metrics),
        totalPS=sum(m.nPS for m in metrics),
        meanRho=float(np.mean(rhos)) if rhos else math.nan,
        minRho=float(np.min(rhos)) if rhos else math.nan,
        nEventsWithRho=len(rhos),
        violations=sum(m.violations for m in metrics),
    )


def choose_weights(
    summaries: Sequence[ComboSummary], candidate_order: Sequence[str]
) -> dict[str, str]:
    """Per profile: most P-and-S stations, then higher mean rho, then candidate order."""
    rank = {w: i for i, w in enumerate(candidate_order)}
    chosen: dict[str, ComboSummary] = {}
    for s in summaries:
        cur = chosen.get(s.profile)
        if cur is None or (s.key, -rank[s.weights]) > (cur.key, -rank[cur.weights]):
            chosen[s.profile] = s
    return {profile: s.weights for profile, s in chosen.items()}


def adopt_variants(
    summaries: Sequence[ComboSummary],
    chosen: Mapping[str, str],
    profile_variants: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Per base profile, the profile to use: a variant only if it strictly beats the base."""
    by_combo = {(s.profile, s.weights): s for s in summaries}
    adopted: dict[str, str] = {}
    for base in sorted({s.profile for s in summaries} - _all_variants(profile_variants)):
        best_profile = base
        best_key = by_combo[(base, chosen[base])].key
        for variant in profile_variants.get(base, []):
            if variant not in chosen:
                continue
            key = by_combo[(variant, chosen[variant])].key
            if key > best_key:
                best_profile, best_key = variant, key
        adopted[base] = best_profile
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
                passed=not failures,
                failures=tuple(failures),
            )
        )
    n_pass = sum(1 for e in events if e.passed)
    return CheckBResult(events=tuple(events), nPass=n_pass, passed=n_pass >= cfg.minEventsPass)


def format_check_b(result: CheckBResult, cfg: CheckBConfig) -> str:
    header = f"{'event':<24} {'stations':>8} {'P&S':>5} {'P>=S':>5} {'rho':>7} {'nRho':>5}  result"
    lines = [
        (
            f"Check B (P&S >= {cfg.minStationsPS}, no S-before-P, rho >= {cfg.minRho}, "
            f">= {cfg.minEventsPass} events)"
        ),
        header,
        "-" * len(header),
    ]
    for e in result.events:
        rho = "nan" if math.isnan(e.rho) else f"{e.rho:.3f}"
        verdict = "PASS" if e.passed else "FAIL: " + "; ".join(e.failures)
        lines.append(
            f"{e.eventId:<24} {e.nStations:>8} {e.nPS:>5} {e.violations:>5} {rho:>7} "
            f"{e.nRho:>5}  {verdict}"
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
    chosenWeights: dict[str, str]
    adoptedProfiles: dict[str, str]
    checkB: CheckBResult
    picks: list[dict[str, Any]]
    paths: dict[str, Path]
    counts: dict[str, int]
    runtimeS: float


Key = tuple[str, str, str, str]  # eventId, stationId, profile, weights
_INT_COLUMNS = {
    *(f for f in EventMetrics.__dataclass_fields__ if f != "rho"),
    *("nNoData", "picksP", "picksS", "nBlocks", "nBlocksTooShort", "droppedNearEdge"),
    *("nEvents", "totalPS", "nEventsWithRho", "violations"),
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

    models = {w: io.load_model(w, picker) for w in weights_list}

    store: dict[Key, list[dict[str, Any]]] = {}
    diags: dict[Key, PickDiagnostics] = {}
    no_data: set[tuple[str, str]] = set()
    for event in windows.events:
        t_event = time.perf_counter()
        for ks in usable[event.eventId]:
            raw = io.read_window(
                ks.stationId, event.windowStart, event.windowEnd, cache_dir=cache_dir
            )
            profiles = _profiles_for(base_of[ks.stationId], variants)
            if len(raw) == 0:
                no_data.add((event.eventId, ks.stationId))
                log.warning("%s %s: read_window returned no data", event.eventId, ks.stationId)
                for profile in profiles:
                    for w in weights_list:
                        store[(event.eventId, ks.stationId, profile, w)] = []
                continue
            for profile in profiles:
                try:
                    prepared = prepare_station(
                        raw, ks.stationId, profile, cfg, for_picking=io.for_picking
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"{event.eventId} {ks.stationId} {profile}: preprocessing failed"
                    ) from exc
                for w in weights_list:
                    picks, diag = pick_prepared(prepared, cfg, models[w], w)
                    store[(event.eventId, ks.stationId, profile, w)] = picks
                    diags[(event.eventId, ks.stationId, profile, w)] = diag
        log.info(
            "%s: picked %d usable stations x %d weights in %.1f s",
            event.eventId,
            len(usable[event.eventId]),
            len(weights_list),
            time.perf_counter() - t_event,
        )

    profile_order = []
    for base in sorted(set(base_of.values())):
        profile_order.extend(_profiles_for(base, variants))

    def stations_on(event_id: str, profile: str) -> list[KnownStation]:
        return [
            ks
            for ks in usable[event_id]
            if profile in _profiles_for(base_of[ks.stationId], variants)
        ]

    event_rows: list[dict[str, Any]] = []
    per_combo: dict[tuple[str, str], list[EventMetrics]] = {}
    for event in windows.events:
        for w in weights_list:
            for profile in profile_order:
                sts = stations_on(event.eventId, profile)
                if not sts:
                    continue
                keys = [(event.eventId, ks.stationId, profile, w) for ks in sts]
                best = {k[1]: best_phases(store[k]) for k in keys}
                dist = {ks.stationId: ks.epiDistM for ks in sts}
                m = event_metrics(best, dist, ab_cfg.minStationsForRho)
                per_combo.setdefault((w, profile), []).append(m)
                ds = [diags[k] for k in keys if k in diags]
                row = {
                    "rowType": "event",
                    "eventId": event.eventId,
                    "weights": w,
                    "profile": profile,
                    **asdict(m),
                    "nNoData": sum(1 for ks in sts if (event.eventId, ks.stationId) in no_data),
                    "picksP": sum(d.picksP for d in ds),
                    "picksS": sum(d.picksS for d in ds),
                    "nBlocks": sum(d.nBlocks for d in ds),
                    "nBlocksTooShort": sum(d.nBlocksTooShort for d in ds),
                    "droppedNearEdge": sum(d.droppedNearEdge for d in ds),
                    "runtimeS": round(sum(d.runtimeS for d in ds), 3),
                }
                event_rows.append(row)
                log.info(
                    "%s %s %s: stations %d, P %d, S %d, P&S %d, violations %d, rho %s",
                    event.eventId,
                    w,
                    profile,
                    m.nStations,
                    m.nP,
                    m.nS,
                    m.nPS,
                    m.violations,
                    "nan" if math.isnan(m.rho) else f"{m.rho:.3f}",
                )

    summaries = [
        summarize(w, profile, per_combo[(w, profile)])
        for w in weights_list
        for profile in profile_order
        if (w, profile) in per_combo
    ]
    chosen = choose_weights(summaries, weights_list)
    adopted = adopt_variants(summaries, chosen, variants)
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
    for base, profile in adopted.items():
        log.info(
            "A/B choice: stations on %s -> profile %s, weights %s", base, profile, chosen[profile]
        )

    # Final picks: every usable station with its adopted profile and that profile's chosen weights.
    def final_key(event_id: str, station_id: str) -> Key:
        profile = adopted[base_of[station_id]]
        return (event_id, station_id, profile, chosen[profile])

    final_metrics: dict[str, EventMetrics] = {}
    final_picks: list[dict[str, Any]] = []
    for event in windows.events:
        best = {}
        for ks in usable[event.eventId]:
            picks = store[final_key(event.eventId, ks.stationId)]
            best[ks.stationId] = best_phases(picks)
            final_picks.extend(picks)
        dist = {ks.stationId: ks.epiDistM for ks in usable[event.eventId]}
        final_metrics[event.eventId] = event_metrics(best, dist, ab_cfg.minStationsForRho)
    check_b = evaluate_check_b(final_metrics, ab_cfg.checkB)

    unique: dict[str, dict[str, Any]] = {}
    for p in final_picks:
        unique.setdefault(p["id"], p)
    n_duplicates = len(final_picks) - len(unique)
    if n_duplicates:
        log.warning("%d duplicate pick ids from overlapping event windows dropped", n_duplicates)
    picks_out = sorted(unique.values(), key=lambda p: (p["t"], p["stationId"], p["phase"]))

    known_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "abCsv": known_dir / AB_CSV,
        "abJson": known_dir / AB_JSON,
        "picks": known_dir / PICKS_FILE,
    }
    table = pd.DataFrame(event_rows + summary_rows)
    for col in _INT_COLUMNS & set(table.columns):  # keep counts integer next to NaN cells
        table[col] = table[col].astype("Int64")
    table.to_csv(paths["abCsv"], index=False)
    io.write_picks(picks_out, paths["picks"])

    for event in windows.events:
        png = known_dir / f"record_section_{_safe_name(event.eventId)}.png"
        station_picks = {
            ks.stationId: store[final_key(event.eventId, ks.stationId)]
            for ks in usable[event.eventId]
        }
        plot_record_section(event, usable[event.eventId], station_picks, cfg, io, cache_dir, png)
        paths[f"recordSection:{event.eventId}"] = png

    counts = {
        "events": len(windows.events),
        "stationWindows": sum(len(v) for v in usable.values()),
        "stationWindowsNoData": len(no_data),
        "picks": len(picks_out),
        "picksP": sum(1 for p in picks_out if p["phase"] == "P"),
        "picksS": sum(1 for p in picks_out if p["phase"] == "S"),
        "duplicatePicksDropped": n_duplicates,
        "droppedNearEdge": sum(
            diags[final_key(e.eventId, ks.stationId)].droppedNearEdge
            for e in windows.events
            for ks in usable[e.eventId]
            if final_key(e.eventId, ks.stationId) in diags
        ),
        "checkBEventsPassed": check_b.nPass,
    }
    runtime_s = time.perf_counter() - t_start
    with paths["abJson"].open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "metric": "total P&S stations over the known events, tie -> higher mean rho",
                "chosenWeightsByProfile": chosen,
                "adoptedProfileByBase": adopted,
                "checkB": {
                    "params": ab_cfg.checkB.model_dump(mode="json"),
                    "passed": check_b.passed,
                    "nPass": check_b.nPass,
                    "events": [_json_safe(asdict(e)) for e in check_b.events],
                },
                "counts": counts,
                "runtimeS": round(runtime_s, 3),
                "picker": picker.model_dump(mode="json"),
            },
            fh,
            indent=2,
        )
    log.info("A/B done in %.1f s: %s", runtime_s, counts)
    return AbResult(
        eventRows=event_rows,
        summaryRows=summary_rows,
        chosenWeights=chosen,
        adoptedProfiles=adopted,
        checkB=check_b,
        picks=picks_out,
        paths=paths,
        counts=counts,
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
    cfg: SignalConfig,
    io: AbIO,
    cache_dir: Path,
    path: Path,
) -> None:
    """Display copies sorted by epicentral distance, each normalized, with P and S picks marked."""
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
    for row, ks in enumerate(ordered):
        labels.append(f"{ks.stationId}  {ks.epiDistM / 1000:.1f} km")
        raw = io.read_window(
            ks.stationId, event.t + w0 - rs.padS, event.t + w1 + rs.padS, cache_dir=cache_dir
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
                ax.plot(x, y, color=_INK, linewidth=0.6)
        best = best_phases(list(picks_by_station.get(ks.stationId, [])))
        for p in picks_by_station.get(ks.stationId, []):
            if p is best.bestP or p is best.bestS:
                continue
            color = _COLOR_P if p["phase"] == "P" else _COLOR_S
            ax.vlines(
                p["t"] - event.t,
                row - rs.traceHalfHeight / 2,
                row + rs.traceHalfHeight / 2,
                colors=color,
                linewidth=0.8,
                alpha=0.35,
            )
        for p, color in ((best.bestP, _COLOR_P), (best.bestS, _COLOR_S)):
            if p is not None:
                ax.vlines(
                    p["t"] - event.t,
                    row - rs.traceHalfHeight,
                    row + rs.traceHalfHeight,
                    colors=color,
                    linewidth=2.0,
                )
    if n_skipped:
        log.warning(
            "%s: %d stations without a plottable %s trace", event.eventId, n_skipped, rs.component
        )
    ax.set_xlim(w0, w1)
    ax.set_ylim(len(ordered) - 0.5, -0.5)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, fontsize=8, color=_INK)
    ax.tick_params(axis="x", colors=_INK)
    ax.set_xlabel("Seconds after public-catalog origin time", color=_INK)
    ax.grid(axis="x", color=_INK_MUTED, alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    origin = datetime.fromtimestamp(event.t, tz=UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    band = f"{rs.bandHz[0]:g}-{rs.bandHz[1]:g} Hz"
    ax.set_title(f"{event.eventId}  {origin} UTC  ({rs.component}, {band})", color=_INK)
    fig.legend(
        handles=[
            Line2D([], [], color=_COLOR_P, linewidth=2.0, label="P (best)"),
            Line2D([], [], color=_COLOR_S, linewidth=2.0, label="S (best, after P)"),
            Line2D([], [], color=_COLOR_P, linewidth=0.8, alpha=0.35, label="other P picks"),
            Line2D([], [], color=_COLOR_S, linewidth=0.8, alpha=0.35, label="other S picks"),
        ],
        loc="outside lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.savefig(path, dpi=rs.dpi, facecolor=_SURFACE)
    log.info("wrote %s (%d stations)", path, len(ordered))


# --- stage + CLI --------------------------------------------------------------------------------


def run(ctx: Any, io: AbIO | None = None) -> AbResult:
    """Stage-style entry point: ``ctx`` is an ``hq.runs.RunContext`` (docs/02 -> Stage API)."""
    result = run_ab(ctx.run_dir, ctx.cache_dir, ctx.config.signal, io)
    ctx.record(STAGE, runtime_s=result.runtimeS, counts=result.counts)
    return result


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
    return 0


if __name__ == "__main__":
    sys.exit(main())

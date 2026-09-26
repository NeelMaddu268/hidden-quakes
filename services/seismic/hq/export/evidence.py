"""Evidence snippets (``EventEvidence`` / ``WaveformSnippet``, docs/02 §1) for one event.

For every station with a predicted P in ``arrivals.parquet`` the window is
``[predP - beforeS, predP + afterS]``. The raw window (padded by ``padS`` on each side) comes from
the ``WaveformSource``, one channel is chosen by ``channelPriority``, the display copy is
bandpassed by H1's ``display_copy``, resampled to ``displayRateHz``, trimmed to the window,
scaled to ``[-1, 1]`` and rounded. A station with no data in the window, a channel with none of
the wanted components, a segment shorter than ``minLengthS`` or a flat trace is dropped and
logged, never filled. Picks come only from ``arrivals.pickId``; a null one shows no pick. Traces
are sorted by ``epiDistM`` (nearest first), at most ``maxTraces``, and the file must stay under
``maxFileBytes``: a file at or over it drops its farthest trace until it fits.
"""

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hq_contracts.models import EventEvidence, Pick, SeismicEvent, Station, WaveformSnippet
from obspy import Stream, Trace, UTCDateTime

from hq.config.export import EvidenceConfig, RoundingConfig
from hq.config.run import epoch_s
from hq.export.errors import ExportError
from hq.export.summary import rnd
from hq.export.tables import str_or_none
from hq.export.waveforms import WaveformSource

log = logging.getLogger(__name__)

JSON_SEPARATORS = (",", ":")  # compact: evidence files are fetched one per selection


@dataclass(frozen=True)
class StationArrivals:
    """What the event's arrivals and picks say about one station."""

    station: Station
    epi_dist_m: float
    pred_p: float
    pred_s: float | None
    pick_p: Pick | None
    pick_s: Pick | None


def epicentral_distance_m(event: SeismicEvent, station: Station) -> float:
    return math.hypot(station.enu.e - event.enu.e, station.enu.n - event.enu.n)


def _arrival_pick(picks: dict[str, Pick], pick_id: str | None) -> Pick | None:
    """The pick ``arrivals.pickId`` names, or None: a null pickId means the trace shows no pick
    (never a substitute from the event's pick list), while predP/predS stay filled."""
    return None if pick_id is None else picks[pick_id]


def station_arrivals(
    event: SeismicEvent,
    arrivals: pd.DataFrame,
    stations: dict[str, Station],
    picks: dict[str, Pick],
) -> list[StationArrivals]:
    """One entry per station with a predicted P, nearest first (``epiDistM``, then id)."""
    rows: dict[str, dict[str, tuple[float, str | None]]] = {}
    for row in arrivals.to_dict("records"):
        station_id, phase = str(row["stationId"]), str(row["phase"])
        per_station = rows.setdefault(station_id, {})
        if phase in per_station:
            raise ExportError(
                f"arrivals.parquet has several {phase} rows for event {event.id} at {station_id}"
            )
        t_pred = row["tPred"]
        if t_pred is None or not np.isfinite(float(t_pred)):
            raise ExportError(
                f"arrivals.parquet: event {event.id} at {station_id} {phase} has no tPred"
            )
        per_station[phase] = (float(t_pred), str_or_none(row["pickId"]))
    out: list[StationArrivals] = []
    for station_id, phases in rows.items():
        if "P" not in phases:
            log.info(
                "evidence %s: %s has an S prediction but no P; no window to anchor, skipped",
                event.id,
                station_id,
            )
            continue
        station = stations[station_id]
        pred_p, pick_id_p = phases["P"]
        pred_s, pick_id_s = phases.get("S", (None, None))
        out.append(
            StationArrivals(
                station=station,
                epi_dist_m=epicentral_distance_m(event, station),
                pred_p=pred_p,
                pred_s=pred_s,
                pick_p=_arrival_pick(picks, pick_id_p),
                pick_s=_arrival_pick(picks, pick_id_s),
            )
        )
    return sorted(out, key=lambda sa: (sa.epi_dist_m, sa.station.id))


def choose_channel(station: Station, st: Stream, priority: list[str]) -> str | None:
    """The first of ``station.channels`` whose component code comes earliest in ``priority``
    and that has data in ``st``; None when no wanted component has data."""
    present = {tr.stats.channel for tr in st if tr.stats.npts > 0}
    for component in priority:
        for channel in station.channels:
            if channel.endswith(component) and channel in present:
                return channel
    return None


def _longest(traces: list[Trace]) -> Trace | None:
    if not traces:
        return None
    return max(traces, key=lambda tr: (tr.stats.npts, -tr.stats.starttime.timestamp))


def snippet(
    sa: StationArrivals,
    event_id: str,
    cfg: EvidenceConfig,
    rounding: RoundingConfig,
    source: WaveformSource,
    cache_dir: Path,
) -> WaveformSnippet | None:
    """One trace for the station, or None (logged) when the cache cannot support one."""
    station = sa.station
    w0, w1 = sa.pred_p - cfg.beforeS, sa.pred_p + cfg.afterS
    raw = source.read_window(station.id, w0 - cfg.padS, w1 + cfg.padS, cache_dir=cache_dir)
    channel = choose_channel(station, raw, cfg.channelPriority)
    if channel is None:
        log.info(
            "evidence %s: %s has no data for components %s in [%.3f, %.3f]; trace dropped",
            event_id,
            station.id,
            cfg.channelPriority,
            w0,
            w1,
        )
        return None
    display = source.display_copy(
        Stream([tr for tr in raw if tr.stats.channel == channel]), cfg.bandHz
    )
    segments: list[Trace] = []
    for tr in display:
        if tr.stats.npts < 2:
            continue  # a lone sample is no segment; the resampler needs at least two
        if tr.stats.sampling_rate != cfg.displayRateHz:
            tr.resample(cfg.displayRateHz)
        tr.trim(UTCDateTime(w0), UTCDateTime(w1), nearest_sample=False)
        if tr.stats.npts > 1:
            segments.append(tr)
    tr = _longest(segments)
    if tr is None:
        log.info(
            "evidence %s: %s %s is empty after filtering; trace dropped",
            event_id,
            station.id,
            channel,
        )
        return None
    length_s = tr.stats.npts * tr.stats.delta
    if length_s < cfg.minLengthS:
        log.info(
            "evidence %s: %s %s covers %.2f s < minLengthS %.2f s (gap); trace dropped",
            event_id,
            station.id,
            channel,
            length_s,
            cfg.minLengthS,
        )
        return None
    data = np.asarray(tr.data, dtype=np.float64)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if not np.isfinite(peak) or peak == 0.0:
        log.info(
            "evidence %s: %s %s is flat or non-finite (peak %s); trace dropped",
            event_id,
            station.id,
            channel,
            peak,
        )
        return None
    samples = np.round(data / peak, rounding.sample)
    samples[samples == 0.0] = 0.0  # no -0.0 in the JSON
    return WaveformSnippet(
        stationId=station.id,
        channel=channel,
        epiDistM=rnd(sa.epi_dist_m, rounding.distM),
        t0=rnd(epoch_s(tr.stats.starttime.ns), rounding.timeS),
        dt=1.0 / cfg.displayRateHz,
        samples=samples.tolist(),
        pickP=None if sa.pick_p is None else sa.pick_p.t,
        pickS=None if sa.pick_s is None else sa.pick_s.t,
        probP=None if sa.pick_p is None else sa.pick_p.prob,
        probS=None if sa.pick_s is None else sa.pick_s.prob,
        predP=sa.pred_p,
        predS=sa.pred_s,
    )


def evidence_json(evidence: EventEvidence) -> bytes:
    """Compact, key-sorted JSON: the bytes written to ``evidence/<eventId>.json``."""
    text = json.dumps(evidence.model_dump(mode="json"), sort_keys=True, separators=JSON_SEPARATORS)
    return (text + "\n").encode("utf-8")


def fit_to_budget(evidence: EventEvidence, max_bytes: int) -> tuple[EventEvidence, bytes, int]:
    """Drop the farthest trace until the file is under ``max_bytes`` (strictly); returns
    (evidence, bytes, traces dropped). One trace that does not fit is a config error."""
    dropped = 0
    data = evidence_json(evidence)
    while len(data) >= max_bytes and len(evidence.traces) > 1:
        evidence = evidence.model_copy(update={"traces": evidence.traces[:-1]})
        data = evidence_json(evidence)
        dropped += 1
    if len(data) >= max_bytes:
        raise ExportError(
            f"evidence {evidence.eventId}: one trace alone is {len(data)} bytes, not under "
            f"{max_bytes}; shorten beforeS/afterS, lower displayRateHz or rounding.sample"
        )
    return evidence, data, dropped


def build_evidence(
    event: SeismicEvent,
    arrivals: pd.DataFrame,
    stations: dict[str, Station],
    picks: dict[str, Pick],
    cfg: EvidenceConfig,
    rounding: RoundingConfig,
    source: WaveformSource,
    cache_dir: Path,
    make_snippet: Callable[..., WaveformSnippet | None] = snippet,
) -> tuple[EventEvidence, bytes, dict[str, int]] | None:
    """The event's evidence and its file bytes, or None when no station yields a trace.

    Stations are tried nearest first until ``maxTraces`` traces exist, so a dropped near station
    is replaced by the next one out. The counts say how many were tried and dropped.
    """
    candidates = station_arrivals(event, arrivals, stations, picks)
    traces: list[WaveformSnippet] = []
    tried = 0
    for sa in candidates:
        if len(traces) >= cfg.maxTraces:
            break
        tried += 1
        trace = make_snippet(sa, event.id, cfg, rounding, source, cache_dir)
        if trace is not None:
            traces.append(trace)
    counts = {"stations": len(candidates), "tried": tried, "dropped": tried - len(traces)}
    if not traces:
        log.warning(
            "evidence %s: none of %d station(s) with a predicted P has usable data; no file",
            event.id,
            len(candidates),
        )
        return None
    evidence = EventEvidence(eventId=event.id, filterHz=cfg.bandHz, traces=traces)
    evidence, data, over_budget = fit_to_budget(evidence, cfg.maxFileBytes)
    if over_budget:
        log.info(
            "evidence %s: dropped %d farthest trace(s) to fit maxFileBytes %d (%d bytes)",
            event.id,
            over_budget,
            cfg.maxFileBytes,
            len(data),
        )
    counts["overBudget"] = over_budget
    return evidence, data, counts

"""Why each unmatched public event is unmatched, from the evidence in the run dir (MATCH-02).

``match()`` sees only the two tables, so it can say an event lost the one-to-one competition or
had no admissible candidate. ``explain_unmatched`` refines the second along the pipeline, using
whichever evidence tables exist; each check runs only when its inputs are present, and the result
records which checks ran and which were skipped (and why). Per unmatched public event, the first
check that fails gives the reason:

1. ``window``, ``bbox``: the public origin lies outside the run window or bbox.
2. ``oneToOne`` (from ``match``): an admissible candidate exists but every one was assigned to
   another public event. The reason is kept as ``match`` wrote it.
3. ``waveformData`` (stations, gaps): fewer than ``minStations`` used stations have waveform data
   in the event's expected arrival windows; a station has none when every channel's gaps cover
   both its P and its S window.
4. ``picks`` (stations, picks): fewer than ``minStations`` used stations have a pick in the
   expected windows (a P pick in the P window or an S pick in the S window).
5. ``association`` (stations, picks, assoc_picks): fewer than ``minStations`` stations have an
   associated pick among those.
6. ``location`` (stations, picks, ``events_located.pickIds``): the located events that carry
   in-window picks from at least ``minStations`` stations are this event's associated candidates.
   If any exists, the reason is "located out of tolerance" with the nearest one (lowest cost);
   if none does and check 5 ran, it is "associated, not located".
7. Otherwise the ``match`` reason stays: "no candidate within ..." with the nearest located event.

Expected arrival windows
    For a station at hypocentral distance ``R`` (ENU, sensor position to public hypocentre), the
    P window is ``[t + R / vpMax - pad, t + R / vpMin + pad]`` and the S window the same with Vs,
    where ``vMin``/``vMax`` are the slowest and fastest layer velocities of the configured 1D model
    and ``pad`` is ``reasons.arrivalPadS``. In that model every first arrival lies inside the
    unpadded bracket: no path is shorter than ``R`` or faster than ``vMax``, and the straight ray
    is never slower than ``R / vMin`` (extending the top layer upward keeps its velocity). The pad
    absorbs public origin-time and location error and pick error. Only stations with
    ``usedInRun`` count.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig
from hq.locate.velocity import LayerModel
from hq.match import REASONS, MatchResult, Offsets, Tolerance, checked, pair_offsets

# Evidence tables and the columns each check needs (docs/02 §2 names).
STATION_COLUMNS = ("id", "enu_e", "enu_n", "enu_u", "channels", "usedInRun")
GAP_COLUMNS = ("stationId", "channel", "gapStart", "gapEnd")
PICK_COLUMNS = ("id", "stationId", "phase", "t")
ASSOC_PICK_COLUMNS = ("assocId", "pickId")
PUBLIC_COLUMNS = ("id", "t", "latitude", "longitude", "enu_e", "enu_n", "enu_u")

# Check name -> the evidence inputs it needs, in pipeline order.
CHECK_INPUTS: dict[str, tuple[str, ...]] = {
    "waveformData": ("stations", "gaps"),
    "picks": ("stations", "picks"),
    "association": ("stations", "picks", "assoc_picks"),
    "location": ("stations", "picks", "events_located.pickIds"),
}
ALWAYS_RUN = ("window", "bbox", "oneToOne", "tolerance")


@dataclass(frozen=True, eq=False)
class Evidence:
    """Run-dir tables used to explain misses; ``None`` when the file is absent."""

    stations: pd.DataFrame | None = None  # stations.parquet
    gaps: pd.DataFrame | None = None  # gaps.parquet
    picks: pd.DataFrame | None = None  # picks.parquet
    assoc_picks: pd.DataFrame | None = None  # assoc_picks.parquet


@dataclass(frozen=True)
class SpeedBounds:
    """Slowest and fastest P and S layer velocities of a 1D model (m/s)."""

    model_name: str
    vp_min: float
    vp_max: float
    vs_min: float
    vs_max: float

    @classmethod
    def from_model(cls, model: LayerModel) -> "SpeedBounds":
        return cls(
            model_name=model.name,
            vp_min=float(np.min(model.vp_m_per_s)),
            vp_max=float(np.max(model.vp_m_per_s)),
            vs_min=float(np.min(model.vs_m_per_s)),
            vs_max=float(np.max(model.vs_m_per_s)),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "velocityModel": self.model_name,
            "vpMinMPerS": self.vp_min,
            "vpMaxMPerS": self.vp_max,
            "vsMinMPerS": self.vs_min,
            "vsMaxMPerS": self.vs_max,
        }


@dataclass(frozen=True, eq=False)
class Explained:
    """``matches`` with refined reasons, each unmatched public event's reason code, and the
    checks that ran or were skipped (check -> why)."""

    matches: pd.DataFrame
    codes: dict[str, str]
    checks_run: list[str]
    checks_skipped: dict[str, str] = field(default_factory=dict)


def _require(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def code_of(reason: str) -> str:
    """The code whose stable prefix starts ``reason``; fails if none does."""
    codes = [code for code, prefix in REASONS.items() if reason.startswith(prefix)]
    if len(codes) != 1:
        raise ValueError(f"reason {reason!r} does not start with exactly one known prefix")
    return codes[0]


@dataclass(frozen=True)
class _Stations:
    ids: NDArray[Any]
    enu: NDArray[np.float64]  # (n, 3)
    channels: list[list[str]]


def _used_stations(stations: pd.DataFrame) -> _Stations:
    _require(stations, STATION_COLUMNS, "stations")
    if stations["usedInRun"].isna().any():
        raise ValueError("stations: null usedInRun")
    used = stations[stations["usedInRun"].astype(bool)]
    enu = used[["enu_e", "enu_n", "enu_u"]].to_numpy(dtype=np.float64)
    if not np.isfinite(enu).all():
        raise ValueError("stations: non-finite ENU on a used station")
    channels = [list(c) for c in used["channels"]]
    empty = [sid for sid, c in zip(used["id"], channels, strict=True) if not c]
    if empty:
        raise ValueError(f"stations: used stations without channels: {empty}")
    return _Stations(ids=used["id"].astype(str).to_numpy(dtype=object), enu=enu, channels=channels)


def _merged_gaps(gaps: pd.DataFrame) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Per (station, channel): gap intervals merged where they overlap or touch."""
    _require(gaps, GAP_COLUMNS, "gaps")
    start, end = gaps["gapStart"].to_numpy(np.float64), gaps["gapEnd"].to_numpy(np.float64)
    if not (np.isfinite(start).all() and np.isfinite(end).all() and (end >= start).all()):
        raise ValueError("gaps: every gap needs finite gapStart <= gapEnd")
    merged: dict[tuple[str, str], list[tuple[float, float]]] = {}
    ordered = gaps.sort_values(["stationId", "channel", "gapStart"], kind="stable")
    for (sid, cha), group in ordered.groupby(["stationId", "channel"], sort=False):
        spans: list[tuple[float, float]] = []
        for start, end in zip(group["gapStart"], group["gapEnd"], strict=True):
            if spans and start <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(spans[-1][1], float(end)))
            else:
                spans.append((float(start), float(end)))
        merged[(str(sid), str(cha))] = spans
    return merged


def _covered(spans: list[tuple[float, float]], a: float, b: float) -> bool:
    return any(start <= a and end >= b for start, end in spans)


@dataclass(frozen=True)
class _PickIndex:
    """Per (station, phase): pick times (sorted) and ids."""

    t: dict[tuple[str, str], NDArray[np.float64]]
    ids: dict[tuple[str, str], NDArray[Any]]

    def within(self, sid: str, phase: str, a: float, b: float) -> NDArray[Any]:
        key = (sid, phase)
        if key not in self.t:
            return np.zeros(0, dtype=object)
        times = self.t[key]
        lo = np.searchsorted(times, a, side="left")
        hi = np.searchsorted(times, b, side="right")
        return self.ids[key][lo:hi]


def _pick_index(picks: pd.DataFrame, station_ids: NDArray[Any]) -> _PickIndex:
    _require(picks, PICK_COLUMNS, "picks")
    phases = set(picks["phase"].astype(str))
    if not phases <= {"P", "S"}:
        raise ValueError(f"picks: unknown phases {sorted(phases - {'P', 'S'})}")
    used = picks[picks["stationId"].astype(str).isin(set(station_ids))]
    ordered = used.sort_values(["stationId", "phase", "t", "id"], kind="stable")
    t: dict[tuple[str, str], NDArray[np.float64]] = {}
    ids: dict[tuple[str, str], NDArray[Any]] = {}
    for (sid, phase), group in ordered.groupby(["stationId", "phase"], sort=False):
        t[(str(sid), str(phase))] = group["t"].to_numpy(dtype=np.float64)
        ids[(str(sid), str(phase))] = group["id"].astype(str).to_numpy(dtype=object)
    return _PickIndex(t=t, ids=ids)


def _carriers(events_located: pd.DataFrame) -> dict[str, list[str]]:
    """pick id -> ids of the located events whose ``pickIds`` hold it."""
    out: dict[str, list[str]] = {}
    for event_id, pick_ids in zip(events_located["id"], events_located["pickIds"], strict=True):
        for pick_id in pick_ids:
            out.setdefault(str(pick_id), []).append(str(event_id))
    return out


def _nearest(
    candidates: list[str], j_of: dict[str, int], offsets: Offsets, i: int, tol: Tolerance
) -> tuple[str, float, float]:
    """The lowest-cost candidate for public row ``i``: (event id, dt, distance)."""
    cols = np.array([j_of[c] for c in candidates], dtype=np.intp)
    costs = tol.cost(offsets.dt[i, cols], offsets.dist[i, cols])
    k = int(np.argmin(costs))
    j = int(cols[k])
    return candidates[k], float(offsets.dt[i, j]), float(offsets.dist[i, j])


@dataclass(frozen=True)
class _Context:
    """Everything the evidence walk needs besides the public event itself."""

    stations: _Stations
    gaps: dict[tuple[str, str], list[tuple[float, float]]]
    picks: _PickIndex | None
    associated: set[str]
    carriers: dict[str, list[str]]
    ran: frozenset[str]
    offsets: Offsets
    j_of: dict[str, int]
    tol: Tolerance
    k_min: int
    pad: float
    speeds: SpeedBounds


def explain_unmatched(
    result: MatchResult,
    events_located: pd.DataFrame,
    catalog: pd.DataFrame,
    evidence: Evidence,
    cfg: SeismologyConfig,
    run: RunSection,
    speeds: SpeedBounds,
) -> Explained:
    """Refine the reason of every unmatched public event in ``result.matches`` (module docstring)."""
    _require(catalog, PUBLIC_COLUMNS, "catalog")
    public = checked(catalog, "catalog")
    located = checked(events_located, "events_located")
    matches = result.matches.copy()
    if len(matches) != len(public) or set(matches["catalogId"]) != set(public["id"]):
        raise ValueError("matches and catalog do not hold the same public events")
    by_id = catalog.set_index(catalog["id"].astype(str))
    i_of = {cid: i for i, cid in enumerate(public["id"])}

    present = {
        "stations": evidence.stations is not None,
        "gaps": evidence.gaps is not None,
        "picks": evidence.picks is not None,
        "assoc_picks": evidence.assoc_picks is not None,
        "events_located.pickIds": "pickIds" in events_located.columns,
    }
    checks_run = list(ALWAYS_RUN)
    checks_skipped: dict[str, str] = {}
    for check, inputs in CHECK_INPUTS.items():
        missing = [name for name in inputs if not present[name]]
        if missing:
            checks_skipped[check] = f"missing {', '.join(missing)}"
        else:
            checks_run.append(check)
    ran = frozenset(checks_run)

    ctx: _Context | None = None
    if evidence.stations is not None:
        stations = _used_stations(evidence.stations)
        associated: set[str] = set()
        if evidence.assoc_picks is not None and "association" in ran:
            _require(evidence.assoc_picks, ASSOC_PICK_COLUMNS, "assoc_picks")
            associated = set(evidence.assoc_picks["pickId"].astype(str))
        ctx = _Context(
            stations=stations,
            gaps=_merged_gaps(evidence.gaps)
            if evidence.gaps is not None and "waveformData" in ran
            else {},
            picks=_pick_index(evidence.picks, stations.ids) if evidence.picks is not None else None,
            associated=associated,
            carriers=_carriers(events_located) if "location" in ran else {},
            ran=ran,
            offsets=pair_offsets(located, public),
            j_of={eid: j for j, eid in enumerate(located["id"])},
            tol=Tolerance.from_config(cfg.matching),
            k_min=cfg.matching.reasons.minStations,
            pad=cfg.matching.reasons.arrivalPadS,
            speeds=speeds,
        )
    min_lon, min_lat, max_lon, max_lat = run.bbox

    codes: dict[str, str] = {}
    for row in matches.index[matches["eventId"].isna()]:
        cid = str(matches.at[row, "catalogId"])
        ev = by_id.loc[cid]
        reason = str(matches.at[row, "reason"])
        t = float(ev["t"])
        lat, lon = float(ev["latitude"]), float(ev["longitude"])
        if not run.window_start_s <= t < run.window_end_s:
            reason = (
                f"{REASONS['outsideWindow']} (origin {_iso(t)} not in "
                f"[{_iso(run.window_start_s)}, {_iso(run.window_end_s)}))"
            )
        elif not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
            reason = (
                f"{REASONS['outsideBbox']} (lat {lat:.4f}, lon {lon:.4f} not in "
                f"[{min_lon}, {min_lat}, {max_lon}, {max_lat}])"
            )
        elif code_of(reason) == "noCandidate" and ctx is not None:
            reason = _evidence_reason(reason, ev, i_of[cid], ctx)
        matches.at[row, "reason"] = reason
        codes[cid] = code_of(reason)
    return Explained(
        matches=matches, codes=codes, checks_run=checks_run, checks_skipped=checks_skipped
    )


def _evidence_reason(base: str, ev: pd.Series, i: int, ctx: _Context) -> str:
    """Walk checks 3-6 for one public event (row ``i``) that had no admissible candidate."""
    st, k_min, pad, speeds = ctx.stations, ctx.k_min, ctx.pad, ctx.speeds
    t = float(ev["t"])
    hypo = np.array([ev["enu_e"], ev["enu_n"], ev["enu_u"]], dtype=np.float64)
    r = np.sqrt(((st.enu - hypo) ** 2).sum(axis=1))
    p_lo, p_hi = t + r / speeds.vp_max - pad, t + r / speeds.vp_min + pad
    s_lo, s_hi = t + r / speeds.vs_max - pad, t + r / speeds.vs_min + pad
    n_used = len(st.ids)

    if "waveformData" in ctx.ran:
        with_data = 0
        for k, sid in enumerate(st.ids):
            gapped = all(
                _covered(ctx.gaps.get((sid, cha), []), p_lo[k], p_hi[k])
                and _covered(ctx.gaps.get((sid, cha), []), s_lo[k], s_hi[k])
                for cha in st.channels[k]
            )
            with_data += not gapped
        if with_data < k_min:
            return (
                f"{REASONS['noWaveformData']} (data on {with_data} of {n_used} used stations in "
                f"the expected arrival windows; need {k_min})"
            )

    if ctx.picks is None:  # checks 4-6 all need picks
        return base
    in_window = [
        np.concatenate(
            [
                ctx.picks.within(sid, "P", p_lo[k], p_hi[k]),
                ctx.picks.within(sid, "S", s_lo[k], s_hi[k]),
            ]
        )
        for k, sid in enumerate(st.ids)
    ]
    n_picked = sum(ids.size > 0 for ids in in_window)
    if n_picked < k_min:
        return (
            f"{REASONS['tooFewPicks']} (picks on {n_picked} of {n_used} used stations in the "
            f"expected arrival windows; need {k_min})"
        )

    n_assoc = 0
    if "association" in ctx.ran:
        n_assoc = sum(any(pid in ctx.associated for pid in ids) for ids in in_window)
        if n_assoc < k_min:
            return (
                f"{REASONS['picksNotAssociated']} (associated picks on {n_assoc} of the "
                f"{n_picked} stations with picks in the expected arrival windows; need {k_min})"
            )

    if "location" not in ctx.ran:
        return base
    stations_per_event: dict[str, int] = {}
    for ids in in_window:
        for eid in {e for pid in ids for e in ctx.carriers.get(str(pid), [])}:
            stations_per_event[eid] = stations_per_event.get(eid, 0) + 1
    candidates = sorted(e for e, n in stations_per_event.items() if n >= k_min)
    if candidates:
        eid, dt, dist = _nearest(candidates, ctx.j_of, ctx.offsets, i, ctx.tol)
        return (
            f"{REASONS['locatedOutOfTolerance']} {ctx.tol.label()} (nearest associated "
            f"candidate {eid}: dt {dt:+.2f} s, {dist / 1000.0:.2f} km; picks from "
            f"{stations_per_event[eid]} stations)"
        )
    if "association" in ctx.ran:
        return (
            f"{REASONS['associatedNotLocated']} (associated picks on {n_assoc} stations; no "
            f"located event carries picks from {k_min} of them)"
        )
    return base

"""Why each unmatched public regional catalog event is unmatched, from the run dir (MATCH-02).

``match()`` sees only the two tables, so it can say an event lost the one-to-one competition or
had no admissible candidate. ``explain_unmatched`` refines the second along the pipeline, using
whichever evidence tables exist; each check runs only when its inputs are present, and the result
records which checks ran and which were skipped (and why). Per unmatched public event, the first
check that fails gives the reason:

1. ``window``, ``bbox``: the public origin lies outside the run window or bbox.
2. ``oneToOne`` (from ``match``): an admissible candidate exists but every one was assigned to
   another public event. The reason is kept as ``match`` wrote it.
3. ``arrivalWindows`` (stations): the run has fewer than ``minStations`` used stations at all
   ("too few used stations"), or fewer than ``minStations`` used stations have an expected
   arrival window that overlaps the run window (an event near the end of the processed window).
4. ``waveformData`` (stations, gaps): fewer than ``minStations`` used stations have waveform data
   in their expected arrival windows. A station has none when, on every channel, the gaps cover
   both its P and its S window (windows are clipped to the run window, the downloaded span).
5. ``picks`` (stations, picks): fewer than ``minStations`` used stations have an in-window pick
   at ``minPickProb`` or above. The reason is "picks below threshold" when counting every stored
   pick would reach ``minStations``, and "too few picks" otherwise.
6. ``association`` (stations, picks, assoc_picks): no single associated event (``assocId``)
   holds in-window picks from ``minStations`` stations.
7. ``location`` (stations, picks, ``events_located.pickIds``): the located events within
   ``maxCandidateDtS`` of the public origin that carry in-window picks from ``minStations``
   stations are this event's associated candidates. If any exists, the reason is "located out of
   tolerance" with the lowest-cost one; if none does and check 6 ran, it is
   "associated, not located".
8. Otherwise the ``match`` reason stays: "no candidate within ..." with the lowest-cost located
   event.

In-window picks
    A station's in-window picks are its P picks inside its P window and its S picks inside its S
    window. S counts as well as P because the associator counts stations with P or S picks, so
    "too few picks" means too few for it. Contamination by neighbouring events stays small: a
    pick of another event counts only with the same phase inside these 1D-model windows (a few
    seconds wide), checks 6 and 7 need one associated or located event to hold the picks, and
    check 7 an origin time within ``maxCandidateDtS``.

Expected arrival windows
    For a station at hypocentral distance ``R`` (3D ENU, sensor to public hypocentre), the P window
    is ``[t + R / vpMax - pad, t + T_P + pad]``: ``vpMax`` is the fastest P velocity of the
    configured 1D model and ``T_P`` the P time along the straight ray through it (its top layer
    extended upward). The S window is the same with S. In that model every first arrival lies in
    the unpadded bracket: no path is shorter than ``R`` or faster than ``vMax``, and by Fermat's
    principle the first arrival is never later than the straight ray. ``pad`` is ``arrivalPadS``
    (public location and origin-time error, station delays, pick error). Windows are clipped to
    the run window, and only stations with ``usedInRun`` count.
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

FloatArray = NDArray[np.float64]

# Evidence tables and the columns each check needs (docs/02 §2 names).
STATION_COLUMNS = ("id", "enu_e", "enu_n", "enu_u", "channels", "usedInRun")
GAP_COLUMNS = ("stationId", "channel", "gapStart", "gapEnd")
PICK_COLUMNS = ("id", "stationId", "phase", "t", "prob")
ASSOC_PICK_COLUMNS = ("assocId", "pickId")
PUBLIC_COLUMNS = ("id", "t", "latitude", "longitude", "enu_e", "enu_n", "enu_u")
PHASES = ("P", "S")

# Check name -> the evidence inputs it needs, in pipeline order.
CHECK_INPUTS: dict[str, tuple[str, ...]] = {
    "arrivalWindows": ("stations",),
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


@dataclass(frozen=True, eq=False)
class ArrivalModel:
    """What the expected arrival windows take from the configured 1D model (m ASL, m/s)."""

    model_name: str
    tops: FloatArray  # layer tops, top-down; the top layer extends upward, the last downward
    vp: FloatArray
    vs: FloatArray

    @classmethod
    def from_model(cls, model: LayerModel) -> "ArrivalModel":
        return cls(
            model_name=model.name,
            tops=np.asarray(model.top_elev_m, dtype=np.float64),
            vp=np.asarray(model.vp_m_per_s, dtype=np.float64),
            vs=np.asarray(model.vs_m_per_s, dtype=np.float64),
        )

    def speeds(self, phase: str) -> FloatArray:
        return self.vp if phase == "P" else self.vs

    def straight_ray_s(
        self, phase: str, src_elev_m: float, rcv_elev_m: FloatArray, r_m: FloatArray
    ) -> FloatArray:
        """Travel time (s) along the straight ray of length ``r_m`` between the two elevations.

        Slowness along a straight ray depends only on elevation, so the time is ``r_m`` times the
        mean slowness over the ray's elevation span (the layer's slowness when it is horizontal).
        """
        v = self.speeds(phase)
        upper = np.concatenate([[np.inf], self.tops[1:]])  # layer i spans (lower_i, upper_i]
        lower = np.concatenate([self.tops[1:], [-np.inf]])
        rcv = np.asarray(rcv_elev_m, dtype=np.float64)
        z_lo = np.minimum(src_elev_m, rcv)[:, None]
        z_hi = np.maximum(src_elev_m, rcv)[:, None]
        overlap = np.clip(np.minimum(z_hi, upper) - np.maximum(z_lo, lower), 0.0, None)
        span = (z_hi - z_lo)[:, 0]
        layer_at = np.maximum((self.tops[None, :] >= z_lo).sum(axis=1) - 1, 0)
        sloped = span > 0
        mean_slowness = np.where(
            sloped,
            (overlap / v).sum(axis=1) / np.where(sloped, span, 1.0),
            1.0 / v[layer_at],
        )
        return np.asarray(r_m, dtype=np.float64) * mean_slowness

    def to_record(self, pad_s: float) -> dict[str, Any]:
        return {
            "velocityModel": self.model_name,
            "vpMaxMPerS": float(self.vp.max()),
            "vsMaxMPerS": float(self.vs.max()),
            "padS": pad_s,
            "rule": "[t + R / vMax - pad, t + straight-ray time + pad] per phase, clipped to the "
            "run window",
        }


def expected_windows(
    t: float,
    hypo_enu: FloatArray,
    sensor_enu: FloatArray,
    origin_elev_m: float,
    arrivals: ArrivalModel,
    pad_s: float,
) -> dict[str, tuple[FloatArray, FloatArray]]:
    """Unclipped ``{phase: (start, end)}`` epoch-s windows per sensor (module docstring)."""
    r = np.sqrt(((sensor_enu - hypo_enu) ** 2).sum(axis=1))
    src_elev = float(hypo_enu[2]) + origin_elev_m
    rcv_elev = sensor_enu[:, 2] + origin_elev_m
    out: dict[str, tuple[FloatArray, FloatArray]] = {}
    for phase in PHASES:
        fastest = float(arrivals.speeds(phase).max())
        slowest_path = arrivals.straight_ray_s(phase, src_elev, rcv_elev, r)
        out[phase] = (t + r / fastest - pad_s, t + slowest_path + pad_s)
    return out


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
    enu: FloatArray  # (n, 3)
    channels: list[list[str]]


def _used_stations(stations: pd.DataFrame) -> _Stations:
    _require(stations, STATION_COLUMNS, "stations")
    if stations["id"].isna().any() or stations["usedInRun"].isna().any():
        raise ValueError("stations: null id or usedInRun")
    dupes = sorted(stations.loc[stations["id"].duplicated(), "id"].astype(str))
    if dupes:
        raise ValueError(f"stations: duplicate ids {dupes}")
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
    starts, ends = gaps["gapStart"].to_numpy(np.float64), gaps["gapEnd"].to_numpy(np.float64)
    if not (np.isfinite(starts).all() and np.isfinite(ends).all() and (ends >= starts).all()):
        raise ValueError("gaps: every gap needs finite gapStart <= gapEnd")
    merged: dict[tuple[str, str], list[tuple[float, float]]] = {}
    ordered = gaps.sort_values(["stationId", "channel", "gapStart"], kind="stable")
    for (sid, cha), group in ordered.groupby(["stationId", "channel"], sort=False):
        spans: list[tuple[float, float]] = []
        for gap_start, gap_end in zip(group["gapStart"], group["gapEnd"], strict=True):
            if spans and gap_start <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(spans[-1][1], float(gap_end)))
            else:
                spans.append((float(gap_start), float(gap_end)))
        merged[(str(sid), str(cha))] = spans
    return merged


def _has_data(spans: list[tuple[float, float]], a: float, b: float) -> bool:
    """Whether [a, b] (non-empty) holds time that no gap covers."""
    return not any(start <= a and end >= b for start, end in spans)


@dataclass(frozen=True)
class _PickIndex:
    """Per (station, phase): pick times (sorted), ids and probabilities."""

    t: dict[tuple[str, str], FloatArray]
    ids: dict[tuple[str, str], NDArray[Any]]
    prob: dict[tuple[str, str], FloatArray]

    def within(self, sid: str, phase: str, a: float, b: float) -> tuple[NDArray[Any], FloatArray]:
        key = (sid, phase)
        if key not in self.t or not a < b:
            return np.zeros(0, dtype=object), np.zeros(0)
        times = self.t[key]
        lo = np.searchsorted(times, a, side="left")
        hi = np.searchsorted(times, b, side="right")
        return self.ids[key][lo:hi], self.prob[key][lo:hi]


def _pick_index(picks: pd.DataFrame, station_ids: NDArray[Any]) -> _PickIndex:
    _require(picks, PICK_COLUMNS, "picks")
    if picks[["id", "stationId", "phase"]].isna().any().any():
        raise ValueError("picks: null id, stationId or phase")
    phases = set(picks["phase"].astype(str))
    if not phases <= set(PHASES):
        raise ValueError(f"picks: unknown phases {sorted(phases - set(PHASES))}")
    for col in ("t", "prob"):
        values = picks[col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            bad = sorted(picks.loc[~np.isfinite(values), "id"].astype(str))
            raise ValueError(f"picks: non-finite {col} for {bad}")
    used = picks[picks["stationId"].astype(str).isin(set(station_ids))]
    ordered = used.sort_values(["stationId", "phase", "t", "id"], kind="stable")
    t: dict[tuple[str, str], FloatArray] = {}
    ids: dict[tuple[str, str], NDArray[Any]] = {}
    prob: dict[tuple[str, str], FloatArray] = {}
    for (sid, phase), group in ordered.groupby(["stationId", "phase"], sort=False):
        key = (str(sid), str(phase))
        t[key] = group["t"].to_numpy(dtype=np.float64)
        ids[key] = group["id"].astype(str).to_numpy(dtype=object)
        prob[key] = group["prob"].to_numpy(dtype=np.float64)
    return _PickIndex(t=t, ids=ids, prob=prob)


def _holders(assoc_picks: pd.DataFrame) -> dict[str, list[str]]:
    """pick id -> the assocIds holding it."""
    _require(assoc_picks, ASSOC_PICK_COLUMNS, "assoc_picks")
    if assoc_picks[list(ASSOC_PICK_COLUMNS)].isna().any().any():
        raise ValueError("assoc_picks: null assocId or pickId")
    out: dict[str, list[str]] = {}
    for assoc_id, pick_id in zip(assoc_picks["assocId"], assoc_picks["pickId"], strict=True):
        out.setdefault(str(pick_id), []).append(str(assoc_id))
    return out


def _carriers(events_located: pd.DataFrame) -> dict[str, list[str]]:
    """pick id -> ids of the located events whose ``pickIds`` hold it."""
    null = [
        str(eid)
        for eid, pick_ids in zip(events_located["id"], events_located["pickIds"], strict=True)
        if not isinstance(pick_ids, list | tuple | np.ndarray)
    ]
    if null:
        raise ValueError(f"events_located: null pickIds for {sorted(null)}")
    out: dict[str, list[str]] = {}
    for event_id, pick_ids in zip(events_located["id"], events_located["pickIds"], strict=True):
        for pick_id in pick_ids:
            out.setdefault(str(pick_id), []).append(str(event_id))
    return out


def _stations_per_group(
    in_window: list[NDArray[Any]], owners: dict[str, list[str]]
) -> dict[str, int]:
    """Group id -> number of stations whose in-window picks it holds."""
    counts: dict[str, int] = {}
    for ids in in_window:
        for group in {g for pid in ids for g in owners.get(str(pid), [])}:
            counts[group] = counts.get(group, 0) + 1
    return counts


def _best(counts: dict[str, int]) -> tuple[str | None, int]:
    """The group with the most stations (ties: smallest id), or (None, 0)."""
    if not counts:
        return None, 0
    group, n = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return group, n


@dataclass(frozen=True)
class _Context:
    """Everything the evidence walk needs besides the public event itself."""

    stations: _Stations
    gaps: dict[tuple[str, str], list[tuple[float, float]]]
    picks: _PickIndex | None
    holders: dict[str, list[str]]
    carriers: dict[str, list[str]]
    ran: frozenset[str]
    offsets: Offsets
    j_of: dict[str, int]
    tol: Tolerance
    k_min: int
    min_prob: float
    pad: float
    max_candidate_dt: float
    arrivals: ArrivalModel
    run: RunSection


def explain_unmatched(
    result: MatchResult,
    events_located: pd.DataFrame,
    catalog: pd.DataFrame,
    evidence: Evidence,
    cfg: SeismologyConfig,
    run: RunSection,
    arrivals: ArrivalModel | None,
) -> Explained:
    """Refine the reason of every unmatched public event in ``result.matches`` (module docstring).

    ``arrivals`` is needed only when ``evidence.stations`` is given (every window check needs it).
    """
    _require(catalog, PUBLIC_COLUMNS, "catalog")
    public = checked(catalog, "catalog")
    located = checked(events_located, "events_located")
    for col in ("latitude", "longitude", "enu_u"):
        values = pd.to_numeric(catalog[col], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            bad = sorted(catalog.loc[~np.isfinite(values), "id"].astype(str))
            raise ValueError(f"catalog: non-finite {col} for {bad}")
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
        if arrivals is None:
            raise ValueError("explain_unmatched: stations given without an arrival model")
        stations = _used_stations(evidence.stations)
        reasons_cfg = cfg.matching.reasons
        ctx = _Context(
            stations=stations,
            gaps=_merged_gaps(evidence.gaps) if evidence.gaps is not None else {},
            picks=_pick_index(evidence.picks, stations.ids) if evidence.picks is not None else None,
            holders=_holders(evidence.assoc_picks) if "association" in ran else {},
            carriers=_carriers(events_located) if "location" in ran else {},
            ran=ran,
            offsets=pair_offsets(located, public),
            j_of={eid: j for j, eid in enumerate(located["id"])},
            tol=Tolerance.from_config(cfg.matching),
            k_min=reasons_cfg.minStations,
            min_prob=reasons_cfg.minPickProb,
            pad=reasons_cfg.arrivalPadS,
            max_candidate_dt=reasons_cfg.maxCandidateDtS,
            arrivals=arrivals,
            run=run,
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
    """Walk checks 3-7 for one public event (row ``i``) that had no admissible candidate."""
    st, k_min = ctx.stations, ctx.k_min
    need = f"classifier minimum {k_min} stations"
    n_used = len(st.ids)
    if n_used < k_min:
        return f"{REASONS['tooFewUsedStations']} ({n_used} used in the run; {need})"
    t = float(ev["t"])
    hypo = np.array([ev["enu_e"], ev["enu_n"], ev["enu_u"]], dtype=np.float64)
    ws, we = ctx.run.window_start_s, ctx.run.window_end_s
    windows = {
        phase: (np.maximum(lo, ws), np.minimum(hi, we))  # clipped; empty where start >= end
        for phase, (lo, hi) in expected_windows(
            t, hypo, st.enu, ctx.run.origin.elevM, ctx.arrivals, ctx.pad
        ).items()
    }
    open_any = np.zeros(n_used, dtype=bool)
    for lo, hi in windows.values():
        open_any |= lo < hi

    n_inside = int(open_any.sum())
    if n_inside < k_min:
        return (
            f"{REASONS['arrivalsOutsideWindow']} (expected arrival windows of {n_inside} of "
            f"{n_used} used stations overlap [{_iso(ws)}, {_iso(we)}); {need})"
        )

    if "waveformData" in ctx.ran:
        with_data = 0
        for k, sid in enumerate(st.ids):
            with_data += any(
                lo[k] < hi[k] and _has_data(ctx.gaps.get((sid, cha), []), lo[k], hi[k])
                for cha in st.channels[k]
                for lo, hi in windows.values()
            )
        if with_data < k_min:
            return (
                f"{REASONS['noWaveformData']} (data on {with_data} of {n_used} used stations in "
                f"the expected arrival windows; {need})"
            )

    if ctx.picks is None:  # checks 5-7 all need picks
        return base
    in_window: list[NDArray[Any]] = []
    n_thr = 0
    for k, sid in enumerate(st.ids):
        found = [ctx.picks.within(sid, ph, lo[k], hi[k]) for ph, (lo, hi) in windows.items()]
        in_window.append(np.concatenate([ids for ids, _ in found]))
        n_thr += any((prob >= ctx.min_prob).any() for _, prob in found)
    n_any = sum(ids.size > 0 for ids in in_window)
    if n_thr < k_min:
        code = "picksBelowThreshold" if n_any >= k_min else "tooFewPicks"
        return (
            f"{REASONS[code]} (picks in the expected arrival windows on {n_any} of {n_used} used "
            f"stations, {n_thr} of them at prob >= {ctx.min_prob:g}; {need})"
        )

    assoc_id, n_assoc = None, 0
    if "association" in ctx.ran:
        assoc_id, n_assoc = _best(_stations_per_group(in_window, ctx.holders))
        if n_assoc < k_min:
            return (
                f"{REASONS['picksNotAssociated']} (at most {n_assoc} of the {n_any} stations with "
                f"picks in the expected arrival windows share one associated event; {need})"
            )

    if "location" not in ctx.ran:
        return base
    per_event = _stations_per_group(in_window, ctx.carriers)
    candidates = [
        eid
        for eid, n in sorted(per_event.items())
        if n >= k_min and abs(ctx.offsets.dt[i, ctx.j_of[eid]]) <= ctx.max_candidate_dt
    ]
    if candidates:
        cols = np.array([ctx.j_of[e] for e in candidates], dtype=np.intp)
        best = int(np.argmin(ctx.tol.cost(ctx.offsets.dt[i, cols], ctx.offsets.dist[i, cols])))
        eid, j = candidates[best], int(cols[best])
        return (
            f"{REASONS['locatedOutOfTolerance']} {ctx.tol.label()} (lowest-cost associated "
            f"candidate {eid}: dt {ctx.offsets.dt[i, j]:+.2f} s, "
            f"{ctx.offsets.dist[i, j] / 1000.0:.2f} km; picks in the expected arrival windows "
            f"from {per_event[eid]} stations)"
        )
    if "association" in ctx.ran:
        return (
            f"{REASONS['associatedNotLocated']} (associated event {assoc_id} holds picks in the "
            f"expected arrival windows from {n_assoc} stations; no located event within "
            f"{ctx.max_candidate_dt:g} s of the public origin carries such picks from {k_min} "
            "stations)"
        )
    return base

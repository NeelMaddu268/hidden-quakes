"""Catalog matching (MATCH-02): one-to-one matching of located events to the public regional
catalog.

``match(events_located, catalog, cfg) -> MatchResult`` is the docs/02 §5 API. It needs only the
``id``, ``t``, ``enu_e`` and ``enu_n`` columns of both tables, so it runs unchanged on events
located from PhaseNet or STA/LTA picks. The stage wrapper (reading and writing the run dir, and
explaining misses from the run's evidence files) is ``hq.match.run``.

Cost and admissibility
    For a located event and a public event, ``dt = t_event - t_public`` (s) and ``d`` is the
    epicentral distance (m). ``cost = |dt| / dtScaleS + d / distScaleM``. The pair is admissible
    only if ``|dt| <= maxDtS`` and ``d <= maxDistM``; both edges are admissible. Magnitude never
    enters the cost, because magnitudes are calibrated on these same matches.

Objective
    Among all one-to-one sets of admissible pairs, the chosen set has the most pairs and, among
    those, the smallest total cost. ``scipy.optimize.linear_sum_assignment`` solves it on the
    matrix restricted to rows and columns with at least one admissible pair, with every
    inadmissible entry set to the finite ``big = bound * (k + 1)``: ``bound`` bounds every
    admissible cost and ``k`` is the number of pairs the solver assigns, so an assignment with one
    more admissible pair is always cheaper by at least ``bound``. Assigned inadmissible pairs are
    then dropped, so an inadmissible pair is never matched. The recovered count therefore depends
    only on admissibility; the cost scales only choose between equally large sets.

Distance frame
    ``d`` is the horizontal distance between the ENU ``e, n`` of both tables: EPSG:32612 (UTM 12N)
    metres minus the run origin (docs/01). Both tables use the same frame, and depth (``u``) never
    enters. UTM grid distance differs from ellipsoidal distance by the grid scale factor, which
    lies within 1.7e-4 of 1 over the showcase bbox (0.99984-1.00006; checked in the tests), so
    under 1 m at 5 km: negligible against the tolerances.

Sensitivity
    Each configured ``(dtS, distM)`` pair reruns the full assignment with that pair as both the
    admissibility limit and the cost scale, and reports the recovered count.

Output
    ``MatchResult.matches``: one row per public event, sorted by ``(t, id)``, columns
    ``catalogId, eventId, dtS, distM, reason`` (docs/02 §2). A matched row has ``eventId``,
    ``dtS`` (event minus public, s) and ``distM`` (m), and a null ``reason``. An unmatched row has
    null ``eventId``, ``dtS`` and ``distM`` and a ``reason``: a stable prefix (``REASONS``) with
    the numbers in parentheses. ``match`` gives the reasons it can see from the two tables alone
    (lost in one-to-one competition, or no admissible candidate); ``hq.match.reasons`` refines the
    second from the run's evidence files. ``MatchResult.sensitivity``: ``dtS, distM, recovered``.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from hq.config.seismology import MatchingConfig, SeismologyConfig

log = logging.getLogger(__name__)

# Columns match() reads from each table (docs/02 §2 flattened names).
REQUIRED_COLUMNS = ("id", "t", "enu_e", "enu_n")

# docs/02 §2 schemas, as in-memory dtypes so zero-row and all-null frames keep them. Strings use
# pandas' default ``str`` semantics (missing is NaN) with python storage, which converts to Arrow
# string, as in hq.match.catalog.
STRING_DTYPE = pd.StringDtype("python", na_value=np.nan)
MATCH_DTYPES: dict[str, Any] = {
    "catalogId": STRING_DTYPE,
    "eventId": STRING_DTYPE,
    "dtS": "float64",
    "distM": "float64",
    "reason": STRING_DTYPE,
}
SENSITIVITY_DTYPES: dict[str, Any] = {"dtS": "float64", "distM": "float64", "recovered": "int64"}

# Stable reason prefixes, by code. Every unmatched reason starts with exactly one of them and
# carries its numbers in parentheses after it. match() writes the first two; hq.match.reasons
# replaces "noCandidate" with a more specific one when the run's evidence allows.
REASONS: dict[str, str] = {
    "lostOneToOne": "lost one-to-one",
    "noCandidate": "no candidate within",
    "outsideWindow": "outside run window",
    "outsideBbox": "outside run bbox",
    "tooFewUsedStations": "too few used stations",
    "arrivalsOutsideWindow": "arrivals outside run window",
    "noWaveformData": "no waveform data",
    "tooFewPicks": "too few picks",
    "picksBelowThreshold": "picks below threshold",
    "picksNotAssociated": "picks not associated",
    "associatedNotLocated": "associated, not located",
    "locatedOutOfTolerance": "located out of tolerance",
}

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]


@dataclass(frozen=True, eq=False)
class MatchResult:
    """docs/02 §5: ``matches`` and ``sensitivity`` in the docs/02 §2 table schemas."""

    matches: pd.DataFrame
    sensitivity: pd.DataFrame


@dataclass(frozen=True)
class Tolerance:
    """Admissibility limits and cost scales for one assignment."""

    max_dt_s: float
    max_dist_m: float
    dt_scale_s: float
    dist_scale_m: float

    @classmethod
    def from_config(cls, cfg: MatchingConfig) -> "Tolerance":
        return cls(cfg.maxDtS, cfg.maxDistM, cfg.dtScaleS, cfg.distScaleM)

    @classmethod
    def sensitivity_pair(cls, dt_s: float, dist_m: float) -> "Tolerance":
        """The pair as both the admissibility limit and the cost scale."""
        return cls(dt_s, dist_m, dt_s, dist_m)

    def cost(self, dt: FloatArray, dist: FloatArray) -> FloatArray:
        return np.abs(dt) / self.dt_scale_s + dist / self.dist_scale_m

    def admissible(self, dt: FloatArray, dist: FloatArray) -> NDArray[np.bool_]:
        return (np.abs(dt) <= self.max_dt_s) & (dist <= self.max_dist_m)

    @property
    def cost_bound(self) -> float:
        """Largest cost of any admissible pair (division and addition round monotonically)."""
        return self.max_dt_s / self.dt_scale_s + self.max_dist_m / self.dist_scale_m

    def label(self) -> str:
        return f"{self.max_dt_s:g} s / {self.max_dist_m / 1000.0:g} km"


@dataclass(frozen=True)
class Offsets:
    """Pairwise offsets, shape (n_public, n_located): ``dt`` = located t - public t (s), ``dist``
    = epicentral distance (m)."""

    dt: FloatArray
    dist: FloatArray


def checked(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """The required columns, sorted by (t, id); fails on missing, null, duplicate or non-finite."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    if out["id"].isna().any():
        raise ValueError(f"{name}: null ids")
    out["id"] = out["id"].astype(str)
    if out["id"].duplicated().any():
        dupes = sorted(out.loc[out["id"].duplicated(), "id"])
        raise ValueError(f"{name}: duplicate ids {dupes}")
    for col in ("t", "enu_e", "enu_n"):
        values = pd.to_numeric(out[col], errors="raise").astype("float64")
        if not np.isfinite(values.to_numpy()).all():
            bad = sorted(out.loc[~np.isfinite(values.to_numpy()), "id"])
            raise ValueError(f"{name}: non-finite {col} for {bad}")
        out[col] = values
    return out.sort_values(["t", "id"], kind="stable").reset_index(drop=True)


def pair_offsets(located: pd.DataFrame, public: pd.DataFrame) -> Offsets:
    """Offsets between every public event (rows) and every located event (columns)."""
    dt = located["t"].to_numpy()[None, :] - public["t"].to_numpy()[:, None]
    de = located["enu_e"].to_numpy()[None, :] - public["enu_e"].to_numpy()[:, None]
    dn = located["enu_n"].to_numpy()[None, :] - public["enu_n"].to_numpy()[:, None]
    return Offsets(dt=np.asarray(dt, dtype=np.float64), dist=np.hypot(de, dn))


def assign(offsets: Offsets, tol: Tolerance) -> tuple[IndexArray, IndexArray]:
    """One-to-one admissible pairs (public row, located column): the most pairs, then least cost.

    See the module docstring for why the finite ``big`` makes the solver maximise admissible pairs
    first. Inadmissible assignments are dropped before returning.
    """
    ok = tol.admissible(offsets.dt, offsets.dist)
    rows = np.flatnonzero(ok.any(axis=1))
    cols = np.flatnonzero(ok.any(axis=0))
    if rows.size == 0:  # no admissible pair at all (also every empty input)
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty
    sub_ok = ok[np.ix_(rows, cols)]
    cost = tol.cost(offsets.dt[np.ix_(rows, cols)], offsets.dist[np.ix_(rows, cols)])
    big = tol.cost_bound * (min(rows.size, cols.size) + 1)
    r, c = linear_sum_assignment(np.where(sub_ok, cost, big))
    keep = sub_ok[r, c]
    return rows[r[keep]].astype(np.intp), cols[c[keep]].astype(np.intp)


def _dt_km(dt: float, dist: float) -> str:
    return f"dt {dt:+.2f} s, {dist / 1000.0:.2f} km"


def no_candidate_reason(tol: Tolerance, nearest: tuple[float, float] | None) -> str:
    """``no candidate within 2 s / 5 km (nearest: dt +3.40 s, 7.90 km)``.

    "nearest" is the located event with the lowest cost ``|dt| / dtScaleS + d / distScaleM``.
    """
    detail = "no located events" if nearest is None else f"nearest: {_dt_km(*nearest)}"
    return f"{REASONS['noCandidate']} {tol.label()} ({detail})"


def _lost_reason(
    i: int,
    offsets: Offsets,
    tol: Tolerance,
    admissible: NDArray[np.bool_],
    public_for: IndexArray,
    located_ids: NDArray[Any],
    public_ids: NDArray[Any],
) -> str:
    candidates = np.flatnonzero(admissible)
    owners = public_for[candidates]
    if (owners < 0).any():
        # A free admissible candidate would make a larger set: the solver guarantees this.
        raise RuntimeError(f"public event {public_ids[i]} left unmatched beside a free candidate")
    costs = tol.cost(offsets.dt[i, candidates], offsets.dist[i, candidates])
    best = int(np.argmin(costs))
    j = int(candidates[best])
    return (
        f"{REASONS['lostOneToOne']} ({candidates.size} admissible candidate(s), each assigned to "
        f"another public event; best {located_ids[j]}: "
        f"{_dt_km(float(offsets.dt[i, j]), float(offsets.dist[i, j]))}, "
        f"assigned to {public_ids[owners[best]]})"
    )


def _matches_frame(
    public: pd.DataFrame,
    located_ids: NDArray[Any],
    located_for: IndexArray,
    offsets: Offsets,
    reasons: list[str | None],
) -> pd.DataFrame:
    matched = located_for >= 0
    idx = np.flatnonzero(matched)
    event_id = np.full(len(public), np.nan, dtype=object)
    dt = np.full(len(public), np.nan)
    dist = np.full(len(public), np.nan)
    event_id[idx] = located_ids[located_for[idx]]
    dt[idx] = offsets.dt[idx, located_for[idx]]
    dist[idx] = offsets.dist[idx, located_for[idx]]
    frame = pd.DataFrame(
        {
            "catalogId": public["id"].to_numpy(dtype=object),
            "eventId": event_id,
            "dtS": dt,
            "distM": dist,
            "reason": np.array([np.nan if r is None else r for r in reasons], dtype=object),
        }
    )
    return frame.astype(MATCH_DTYPES)


def sensitivity_frame(rows: list[tuple[float, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(SENSITIVITY_DTYPES)).astype(SENSITIVITY_DTYPES)


def match(
    events_located: pd.DataFrame, catalog: pd.DataFrame, cfg: SeismologyConfig
) -> MatchResult:
    """Match located events one-to-one to public events (docs/02 §5). See the module docstring."""
    started = time.perf_counter()
    public = checked(catalog, "catalog")
    located = checked(events_located, "events_located")
    public_ids = public["id"].to_numpy(dtype=object)
    located_ids = located["id"].to_numpy(dtype=object)
    offsets = pair_offsets(located, public)
    tol = Tolerance.from_config(cfg.matching)

    rows, cols = assign(offsets, tol)
    located_for = np.full(len(public), -1, dtype=np.intp)
    public_for = np.full(len(located), -1, dtype=np.intp)
    located_for[rows] = cols
    public_for[cols] = rows
    # One-to-one post-condition: no located event and no public event is used twice.
    if len(set(rows.tolist())) != rows.size or len(set(cols.tolist())) != cols.size:
        raise RuntimeError("assignment is not one-to-one")
    if not tol.admissible(offsets.dt[rows, cols], offsets.dist[rows, cols]).all():
        raise RuntimeError("an inadmissible pair was assigned")

    reasons: list[str | None] = []
    for i in range(len(public)):
        if located_for[i] >= 0:
            reasons.append(None)
            continue
        admissible = tol.admissible(offsets.dt[i], offsets.dist[i])
        if admissible.any():
            reasons.append(
                _lost_reason(i, offsets, tol, admissible, public_for, located_ids, public_ids)
            )
        elif len(located) == 0:
            reasons.append(no_candidate_reason(tol, None))
        else:
            j = int(np.argmin(tol.cost(offsets.dt[i], offsets.dist[i])))
            reasons.append(
                no_candidate_reason(tol, (float(offsets.dt[i, j]), float(offsets.dist[i, j])))
            )
    matches = _matches_frame(public, located_ids, located_for, offsets, reasons)

    sens_rows: list[tuple[float, float, int]] = []
    for pair in cfg.matching.sensitivity:
        pair_rows, _ = assign(offsets, Tolerance.sensitivity_pair(pair.dtS, pair.distM))
        sens_rows.append((pair.dtS, pair.distM, int(pair_rows.size)))
    sensitivity = sensitivity_frame(sens_rows)

    log.info(
        "match: recovered %d / %d public regional catalog events within %s from %d located "
        "candidate events; "
        "sensitivity %s; %.2f s",
        rows.size,
        len(public),
        tol.label(),
        len(located),
        ", ".join(f"({d:g} s, {m / 1000:g} km): {n}" for d, m, n in sens_rows),
        time.perf_counter() - started,
    )
    return MatchResult(matches=matches, sensitivity=sensitivity)

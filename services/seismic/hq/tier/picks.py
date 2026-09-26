"""``event_picks.parquet`` (docs/02 §2): the picks of every final event, with event and residual.

One ``Pick`` row per id in each final event's ``pickIds``: the picks used in its final location
(LOC-04 leaves outlier-dropped picks out of ``pickIds``; ``arrivals.parquet`` keeps them with
``usedInLocation`` false). Every field comes from the picks table the association read, except
``eventId`` (the event) and ``residualS`` (``arrivals.parquet``, observed minus predicted, statics
included). Rows follow the events' order, then each event's ``pickIds`` order.
"""

import numpy as np
import pandas as pd
from hq_contracts.io import columns_for, from_frame, to_frame
from hq_contracts.models import Pick

from hq.tier import TierError

ARRIVAL_COLUMNS: tuple[str, ...] = ("eventId", "pickId", "residualS", "usedInLocation")
PICK_COLUMNS: tuple[str, ...] = tuple(columns_for(Pick))


def event_picks(events: pd.DataFrame, arrivals: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    """``Pick`` rows (docs/02 §2 dtypes) for every pick id in ``events.pickIds``; see the module
    docstring. Fails on a pick id missing from ``picks`` or without a used arrival, a pick in two
    events, a used arrival missing from its event's ``pickIds``, or arrivals of unknown events."""
    missing = [c for c in ARRIVAL_COLUMNS if c not in arrivals.columns]
    if missing:
        raise TierError(f"arrivals lacks columns {missing}")
    missing = [c for c in PICK_COLUMNS if c not in picks.columns]
    if missing:
        raise TierError(f"picks lacks columns {missing}")

    event_ids = events["id"].astype(str).tolist()
    links = pd.DataFrame(
        [(eid, str(pid)) for eid, pids in zip(event_ids, events["pickIds"], strict=True)
         for pid in pids],
        columns=["eventId", "pickId"],
    )
    if links["pickId"].duplicated().any():
        dup = sorted(set(links["pickId"][links["pickId"].duplicated()]))
        raise TierError(f"picks in more than one event's pickIds: {dup[:5]}")

    stray = sorted(set(arrivals["eventId"].astype(str)) - set(event_ids))
    if stray:
        raise TierError(f"arrivals name events missing from the final events: {stray[:5]}")
    used = arrivals[arrivals["usedInLocation"].astype(bool) & arrivals["pickId"].notna()]
    used = pd.DataFrame(
        {
            "eventId": used["eventId"].astype(str).to_numpy(dtype=object),
            "pickId": used["pickId"].astype(str).to_numpy(dtype=object),
            "residualS": used["residualS"].to_numpy(dtype=np.float64),
        }
    )
    if used.duplicated(["eventId", "pickId"]).any():
        raise TierError("arrivals: a used pick appears twice for one event")
    joined = links.merge(used, on=["eventId", "pickId"], how="outer", indicator=True)
    no_arrival = joined[joined["_merge"] == "left_only"]
    if len(no_arrival):
        raise TierError(
            f"{len(no_arrival)} pickIds have no used arrival for their event, e.g. "
            f"{no_arrival.iloc[0]['eventId']} / {no_arrival.iloc[0]['pickId']}"
        )
    not_listed = joined[joined["_merge"] == "right_only"]
    if len(not_listed):
        raise TierError(
            f"{len(not_listed)} used arrivals are missing from their event's pickIds, e.g. "
            f"{not_listed.iloc[0]['eventId']} / {not_listed.iloc[0]['pickId']}"
        )
    residual = links.merge(used, on=["eventId", "pickId"], how="left")["residualS"]
    if not np.isfinite(residual.to_numpy(dtype=np.float64)).all():
        raise TierError("arrivals: a used pick has a non-finite residualS")

    pick_ids = picks["id"].astype(str)
    if pick_ids.duplicated().any():
        raise TierError(f"duplicate pick ids, e.g. {pick_ids[pick_ids.duplicated()].iloc[0]!r}")
    by_id = picks.set_index(pick_ids.to_numpy())
    absent = sorted(set(links["pickId"]) - set(by_id.index))
    if absent:
        raise TierError(f"{len(absent)} pickIds are not in the picks table: {absent[:5]}")
    rows = by_id.loc[links["pickId"].to_numpy(), list(PICK_COLUMNS)].reset_index(drop=True)
    rows["eventId"] = links["eventId"].to_numpy(dtype=object)
    rows["residualS"] = residual.to_numpy(dtype=np.float64)
    return to_frame(from_frame(rows, Pick), Pick)  # every row validated as a docs/02 Pick

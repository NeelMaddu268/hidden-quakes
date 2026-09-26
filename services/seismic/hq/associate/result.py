"""``AssocResult`` and the ``assoc_events`` / ``assoc_picks`` table schemas (docs/02 §2)."""

from dataclasses import dataclass

import pandas as pd

# docs/02 §2. Explicit dtypes so a zero-row frame (and its parquet file) keeps its column types.
EVENT_DTYPES: dict[str, str] = {
    "assocId": "string",
    "t": "float64",  # origin time, epoch s UTC
    "latitude": "float64",
    "longitude": "float64",
    "elevM": "float64",  # m ASL
    "nPicks": "Int64",
    "nP": "Int64",
    "nS": "Int64",
}
PICK_DTYPES: dict[str, str] = {"assocId": "string", "pickId": "string"}
EVENTS_MODEL = "AssocEvent"  # model name in the parquet metadata
PICKS_MODEL = "AssocPick"


def typed_frame(data: dict[str, object] | None, dtypes: dict[str, str]) -> pd.DataFrame:
    """A frame with exactly ``dtypes``' columns, in order, cast to those dtypes."""
    data = data or {}
    return pd.DataFrame(
        {name: pd.Series(data.get(name, []), dtype=dtype) for name, dtype in dtypes.items()}
    )


@dataclass(frozen=True, eq=False)
class AssocResult:
    """Association output: ``events`` in the assoc_events schema, ``picks`` in assoc_picks."""

    events: pd.DataFrame
    picks: pd.DataFrame

    def __post_init__(self) -> None:
        for name, frame, dtypes in (
            ("events", self.events, EVENT_DTYPES),
            ("picks", self.picks, PICK_DTYPES),
        ):
            got = {col: str(dtype) for col, dtype in frame.dtypes.items()}
            want = {col: str(pd.Series([], dtype=dtype).dtype) for col, dtype in dtypes.items()}
            if list(got) != list(want) or got != want:
                raise TypeError(f"AssocResult.{name} must have dtypes {want}, got {got}")


def empty_result() -> AssocResult:
    return AssocResult(typed_frame(None, EVENT_DTYPES), typed_frame(None, PICK_DTYPES))

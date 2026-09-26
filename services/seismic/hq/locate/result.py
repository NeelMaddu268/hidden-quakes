"""``LocateResult`` and the location table schemas (docs/02 §2), plus H2's ``locate_flags`` table.

Every frame here has explicit dtypes, so a zero-row frame (and its parquet file) keeps its column
types:

- ``events_located.parquet``: the ``SeismicEvent`` columns (``hq_contracts.io`` flattening and
  dtypes) without ``tier``, ``tierReasons``, ``catalogMatch_*`` and ``magnitude_*``.
- ``arrivals.parquet``: ``ARRIVAL_DTYPES`` (docs/02 §2 column list).
- ``statics.parquet``: ``STATIC_DTYPES`` (docs/02 §2 column list).
- ``locate_flags.parquet``: ``FLAG_DTYPES``, H2-internal (see ``hq.locate``).
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from hq_contracts.io import columns_for, dtypes_for
from hq_contracts.models import SeismicEvent

# SeismicEvent fields events_located.parquet leaves out (docs/02 §2); LOC-06 adds them.
FINAL_ONLY_FIELDS = ("tier", "tierReasons", "catalogMatch", "magnitude")
EVENT_COLUMNS: tuple[str, ...] = tuple(
    c for c in columns_for(SeismicEvent) if c.split("_")[0] not in FINAL_ONLY_FIELDS
)
EVENT_DTYPES: dict[str, str] = {c: dtypes_for(SeismicEvent)[c] for c in EVENT_COLUMNS}
ARRIVAL_DTYPES: dict[str, str] = {
    "eventId": "string",
    "stationId": "string",
    "phase": "string",
    "tPred": "float64",  # epoch s UTC: origin time + table travel time + static
    "tObs": "float64",  # null (NaN) where the station-phase has no pick
    "residualS": "float64",  # tObs - tPred; null where no pick
    "pickId": "string",  # null where no pick
    "usedInLocation": "bool",  # false for outlier-dropped picks and for rows without a pick
}
STATIC_DTYPES: dict[str, str] = {
    "stationId": "string",
    "phase": "string",
    "staticS": "float64",  # additive term the locator applies: tPred = t0 + T + staticS
    # the events the term was estimated from (hq.locate.statics: the reference events with a
    # pick there under referenceEvents, the well-constrained events under selfConsistent)
    "nEvents": "int64",
}
FLAG_DTYPES: dict[str, str] = {
    "eventId": "string",
    "assocId": "string",  # the association event it was located from
    "mapOnVolumeTop": "bool",  # MAP within locator.mapOnVolumeFaceBandM of the volume top
    "mapOnVolumeBottom": "bool",  # ... of the volume bottom
    "depthOnEdgeTop": "bool",  # > depthOnEdgeMassFraction of the PDF on the top face row
    "depthOnEdgeBottom": "bool",  # ... on the bottom face row
    "topFaceMass": "float64",  # PDF mass on the evaluated region's top face row
    "bottomFaceMass": "float64",
    "pdfTruncated": "bool",  # lateral volume face or node budget cut the PDF: h/vErrM null
    "nodeBudgetHit": "bool",  # locator.maxPdfNodes stopped the PDF region from growing
    "hErrGridLimited": "bool",  # hErrM floored at one fine cell (PDF narrower than the grid)
    "vErrGridLimited": "bool",  # vErrM floored at one fine cell
    "nDroppedPicks": "int64",  # picks dropped by the outlier pass
    "outlierPassSkipped": "bool",  # dropping would have left fewer than locator.minPicks
    # Local-ground proxy (no DEM): surfaceElevM of the epicentrally nearest usedInRun station.
    "nearestStationSurfaceElevM": "float64",
    "aboveNearestStationSurface": "bool",  # elevM above that proxy: hypocentre in the air
}
EVENTS_MODEL = "LocatedEvent"  # model names in the parquet metadata
ARRIVALS_MODEL = "Arrival"
STATICS_MODEL = "StationStatic"
FLAGS_MODEL = "LocateFlags"


def typed_frame(rows: list[dict[str, Any]], dtypes: dict[str, str]) -> pd.DataFrame:
    """A frame with exactly ``dtypes``' columns, in order, cast to those dtypes."""
    frame = pd.DataFrame.from_records(rows, columns=list(dtypes))
    return frame.astype(dtypes)


def check_dtypes(name: str, frame: pd.DataFrame, dtypes: dict[str, str]) -> None:
    """Raise unless ``frame`` has exactly ``dtypes``' columns, in order, with those dtypes."""
    got = {col: str(dtype) for col, dtype in frame.dtypes.items()}
    want = {col: str(pd.Series([], dtype=dtype).dtype) for col, dtype in dtypes.items()}
    if list(got) != list(want) or got != want:
        raise TypeError(f"{name} must have dtypes {want}, got {got}")


@dataclass(frozen=True, eq=False)
class LocateResult:
    """docs/02 §5: ``events``, ``arrivals`` and ``statics`` in the docs/02 §2 table schemas."""

    events: pd.DataFrame
    arrivals: pd.DataFrame
    statics: pd.DataFrame

    def __post_init__(self) -> None:
        check_dtypes("LocateResult.events", self.events, EVENT_DTYPES)
        check_dtypes("LocateResult.arrivals", self.arrivals, ARRIVAL_DTYPES)
        check_dtypes("LocateResult.statics", self.statics, STATIC_DTYPES)

"""Stage ``catalog``: the public regional catalog for the run window and bbox (MATCH-01).

Queries the FDSN event service named in ``seismology.yaml`` (ComCat, which carries the UUSS
solutions), saves the raw QuakeML exactly as served, and writes ``CatalogEvent`` rows (docs/02 §1)
with the published depth converted to ``elevM`` using the contributor's documented datum.

Window semantics: ``windowStart`` inclusive, ``windowEnd`` exclusive. FDSN ``endtime`` is
inclusive, so an origin exactly at ``windowEnd`` is served and then dropped (and counted) here.
"""

import hashlib
import logging
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from obspy import UTCDateTime, read_events
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.event import Catalog, Event

from hq.config.run import RunSection
from hq.config.seismology import CatalogConfig
from hq.locate.coords import to_enu

if TYPE_CHECKING:
    from hq.runs import RunContext

log = logging.getLogger(__name__)

STAGE = "catalog"
QUAKEML_NAME = "catalog.quakeml"  # docs/01 stage table
TABLE_NAME = "catalog.parquet"

_M_PER_KM = 1000.0  # unit conversion; QuakeML depths are metres

# docs/02 §2: CatalogEvent flattened (enu -> enu_e, enu_n, enu_u), field order as in the model.
_CATALOG_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("t", pa.float64(), nullable=False),
        pa.field("latitude", pa.float64(), nullable=False),
        pa.field("longitude", pa.float64(), nullable=False),
        pa.field("depthKm", pa.float64(), nullable=False),
        pa.field("depthDatum", pa.string(), nullable=False),
        pa.field("elevM", pa.float64(), nullable=False),
        pa.field("mag", pa.float64()),
        pa.field("magType", pa.string()),
        pa.field("enu_e", pa.float64(), nullable=False),
        pa.field("enu_n", pa.float64(), nullable=False),
        pa.field("enu_u", pa.float64(), nullable=False),
        pa.field("matchedEventId", pa.string()),
    ]
)
# Columns event_to_row fills; ENU and matchedEventId are added after selection.
_ROW_COLUMNS = [
    n for n in _CATALOG_SCHEMA.names if not n.startswith("enu_") and n != "matchedEventId"
]
_SCHEMA_VERSION = "1.0"  # docs/02 SCHEMA_VERSION
_MODEL_NAME = "CatalogEvent"


def query_params(run: RunSection, cfg: CatalogConfig) -> dict[str, Any]:
    """FDSN event-query arguments: the run window and bbox exactly, plus the configured limits."""
    min_lon, min_lat, max_lon, max_lat = run.bbox
    params: dict[str, Any] = {
        "starttime": UTCDateTime(run.windowStart),
        "endtime": UTCDateTime(run.windowEnd),  # inclusive on the server; see select_in_run
        "minlongitude": min_lon,
        "minlatitude": min_lat,
        "maxlongitude": max_lon,
        "maxlatitude": max_lat,
    }
    if cfg.minMagnitude is not None:
        params["minmagnitude"] = cfg.minMagnitude
    if cfg.maxMagnitude is not None:
        params["maxmagnitude"] = cfg.maxMagnitude
    return params


def fetch_quakeml(client: Client, params: dict[str, Any], path: Path) -> bytes:
    """Run the query and save the response bytes exactly as served to ``path``."""
    client.get_events(filename=str(path), **params)
    return path.read_bytes()


def ids_with_product(client: Client, params: dict[str, Any], product_type: str) -> set[str]:
    """ComCat ids of the query's events that carry ``product_type`` (one request, no download).

    Used to check whether analyst picks/arrivals (ComCat ``phase-data``) exist for the window.
    """
    try:
        served = client.get_events(producttype=product_type, **params)
    except FDSNNoDataException:
        return set()  # HTTP 204: no event in this query has the product
    return {comcat_id(event) for event in served}


def _anss_attr(obj: Any, name: str) -> str:
    """Value of an ANSS ``catalog:<name>`` XML attribute that ObsPy keeps in ``obj.extra``."""
    item = (getattr(obj, "extra", None) or {}).get(name)
    value = None if item is None else item.get("value")
    if not value:
        raise ValueError(f"{type(obj).__name__} {obj.resource_id} has no catalog:{name} attribute")
    return str(value)


def comcat_id(event: Event) -> str:
    """ComCat event id: ``catalog:eventsource`` + ``catalog:eventid``, e.g. ``uu80155936``."""
    return _anss_attr(event, "eventsource") + _anss_attr(event, "eventid")


def filter_event_types(catalog: Catalog, cfg: CatalogConfig) -> tuple[list[Event], Counter[str]]:
    """Keep events whose QuakeML type is in ``cfg.eventTypes``; count the rest by type."""
    if cfg.eventTypes is None:
        return list(catalog), Counter()
    kept: list[Event] = []
    dropped: Counter[str] = Counter()
    for event in catalog:
        if event.event_type is None:
            raise ValueError(f"event {comcat_id(event)} has no QuakeML type to filter on")
        if str(event.event_type) in cfg.eventTypes:
            kept.append(event)
        else:
            dropped[str(event.event_type)] += 1
            log.info("dropping %s: event type %r", comcat_id(event), str(event.event_type))
    return kept, dropped


def event_to_row(event: Event, cfg: CatalogConfig) -> dict[str, Any]:
    """One ``CatalogEvent`` row (without ENU) from the preferred origin and magnitude.

    Fails loudly on a missing preferred origin, position, time or depth, and on a contributor or
    origin time the configured depth datums do not cover.
    """
    event_id = comcat_id(event)
    origin = event.preferred_origin()
    if origin is None:
        raise ValueError(f"event {event_id} has no preferred origin")
    if origin.time is None or origin.latitude is None or origin.longitude is None:
        raise ValueError(f"event {event_id}: preferred origin lacks time or position")
    if origin.depth is None:
        raise ValueError(f"event {event_id}: preferred origin has no depth")

    contributor = _anss_attr(origin, "datasource")
    datum = cfg.datums.get(contributor)
    if datum is None:
        raise ValueError(
            f"event {event_id}: no depth datum configured for contributor {contributor!r}; "
            "document it under catalog.datums in seismology.yaml"
        )
    if origin.time < UTCDateTime(datum.validFrom):
        raise ValueError(
            f"event {event_id}: origin {origin.time} predates the documented {contributor!r} "
            f"datum (validFrom {datum.validFrom.isoformat()})"
        )

    magnitude = event.preferred_magnitude()
    if magnitude is not None and magnitude.mag is None:
        raise ValueError(f"event {event_id}: preferred magnitude has no value")

    depth_m = float(origin.depth)
    return {
        "id": event_id,
        "source": f"{contributor.upper()} via {cfg.providerLabel}",
        "t": float(origin.time.timestamp),
        "latitude": float(origin.latitude),
        "longitude": float(origin.longitude),
        "depthKm": depth_m / _M_PER_KM,
        "depthDatum": f"{datum.description} Sources: {', '.join(datum.sourceUrls)}",
        "elevM": datum.surfaceElevM - depth_m,
        "mag": None if magnitude is None else float(magnitude.mag),
        "magType": None if magnitude is None else magnitude.magnitude_type,
    }


def select_in_run(df: pd.DataFrame, run: RunSection) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep rows with ``windowStart <= t < windowEnd`` inside ``run.bbox`` (edges inclusive)."""
    min_lon, min_lat, max_lon, max_lat = run.bbox
    before = df["t"] < run.window_start_s
    at_or_after_end = df["t"] >= run.window_end_s
    in_time = ~before & ~at_or_after_end
    in_box = df["longitude"].between(min_lon, max_lon) & df["latitude"].between(min_lat, max_lat)
    counts = {
        "droppedBeforeWindowStart": int(before.sum()),
        "droppedAtOrAfterWindowEnd": int(at_or_after_end.sum()),
        "droppedOutsideBbox": int((in_time & ~in_box).sum()),
    }
    for reason, mask in (
        ("before windowStart", before),
        ("at or after windowEnd", at_or_after_end),
        ("outside bbox", in_time & ~in_box),
    ):
        for event_id in df.loc[mask, "id"]:
            log.info("dropping %s: %s", event_id, reason)
    return df[in_time & in_box].reset_index(drop=True), counts


def build_catalog(
    catalog: Catalog, run: RunSection, cfg: CatalogConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Served events -> ``CatalogEvent`` rows for the run, sorted by ``t``, plus filter counts."""
    events, dropped_types = filter_event_types(catalog, cfg)
    rows = pd.DataFrame([event_to_row(event, cfg) for event in events], columns=_ROW_COLUMNS)
    if rows["id"].duplicated().any():
        dupes = sorted(rows.loc[rows["id"].duplicated(), "id"])
        raise ValueError(f"duplicate catalog event ids: {dupes}")
    rows, dropped = select_in_run(rows, run)
    rows = rows.sort_values(["t", "id"], kind="stable").reset_index(drop=True)
    e, n, u = to_enu(rows["latitude"], rows["longitude"], rows["elevM"], run.origin)
    rows["enu_e"], rows["enu_n"], rows["enu_u"] = e, n, u
    rows["matchedEventId"] = None
    counts = {
        "served": len(catalog),
        "droppedEventType": sum(dropped_types.values()),
        **dropped,
        "kept": len(rows),
    }
    return rows[_CATALOG_SCHEMA.names], counts


def _write_table(df: pd.DataFrame, path: Path) -> None:
    # CONTRACT-01: replace with hq_contracts.io.write_table once it lands
    table = pa.Table.from_pandas(df, schema=_CATALOG_SCHEMA, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"schemaVersion"] = _SCHEMA_VERSION.encode()
    metadata[b"model"] = _MODEL_NAME.encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _client(cfg: CatalogConfig) -> Client:
    return Client(cfg.provider, timeout=cfg.timeoutS)


def run(ctx: "RunContext") -> None:
    """Stage ``catalog``: write ``catalog.quakeml`` (raw) and ``catalog.parquet`` to the run dir."""
    started = time.perf_counter()
    run_cfg: RunSection = ctx.config.run
    cfg: CatalogConfig = ctx.config.seismology.catalog
    params = query_params(run_cfg, cfg)
    window = f"[{run_cfg.windowStart.isoformat()}, {run_cfg.windowEnd.isoformat()})"
    log.info("catalog: querying %s for %s, bbox %s", cfg.provider, window, run_cfg.bbox)

    client = _client(cfg)
    quakeml_path = ctx.path(QUAKEML_NAME)
    raw = fetch_quakeml(client, params, quakeml_path)
    served = read_events(str(quakeml_path), format="QUAKEML")
    rows, counts = build_catalog(served, run_cfg, cfg)

    with_product = ids_with_product(client, params, cfg.arrivalsProductType)
    counts["withArrivalsProduct"] = int(rows["id"].isin(with_product).sum())

    _write_table(rows, ctx.path(TABLE_NAME))
    runtime_s = time.perf_counter() - started
    log.info(
        "catalog: %d events kept of %d served for %s, bbox %s (%d with %s) in %.1f s",
        counts["kept"],
        counts["served"],
        window,
        run_cfg.bbox,
        counts["withArrivalsProduct"],
        cfg.arrivalsProductType,
        runtime_s,
    )
    ctx.record(
        STAGE,
        runtime_s=runtime_s,
        counts=counts,
        params={
            "provider": cfg.provider,
            "providerUrl": client.base_url,
            "query": {k: str(v) for k, v in params.items()},
            "windowEndExclusive": True,
            "eventTypes": cfg.eventTypes,
            "arrivalsProductType": cfg.arrivalsProductType,
            "datums": {k: v.model_dump(mode="json") for k, v in cfg.datums.items()},
            "quakemlSha256": hashlib.sha256(raw).hexdigest(),
        },
    )

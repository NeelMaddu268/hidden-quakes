"""Stage ``catalog``: the public regional catalog for the run window and bbox (MATCH-01).

Queries the FDSN event service named in ``seismology.yaml`` (ComCat, which carries the UUSS
solutions), saves the raw QuakeML exactly as served, and writes ``CatalogEvent`` rows (docs/02 §1)
with the published depth converted to ``elevM`` using the contributor's documented datum.

Window semantics: ``windowStart`` inclusive, ``windowEnd`` exclusive. ComCat ignores fractional
seconds in ``starttime``/``endtime`` (it truncates both to the whole second, and ``endtime`` is
inclusive at whole seconds), so the query is padded outward to whole seconds and the exact bounds
are applied here; every event served but outside them is dropped, logged and counted.

An empty window is a valid result: when the provider answers "no data" (HTTP 204) to the main
query, the stage writes a fixed empty QuakeML document and a zero-row table, and records
``noDataFromProvider``. Both outputs are written under temporary names and moved into place only
after the whole stage has succeeded, so a failed rerun leaves the previous pair untouched.
"""

import hashlib
import io
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from obspy import UTCDateTime, read_events
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.event import Catalog, Event, ResourceIdentifier

from hq.config.run import RunSection, epoch_s
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
# Every field nullable, as hq_contracts.io.write_table writes them; event_to_row is the guard that
# required values are present (and ObsPy's reader rejects non-finite ones).
_CATALOG_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("t", pa.float64()),
        pa.field("latitude", pa.float64()),
        pa.field("longitude", pa.float64()),
        pa.field("depthKm", pa.float64()),
        pa.field("depthDatum", pa.string()),
        pa.field("elevM", pa.float64()),
        pa.field("mag", pa.float64()),
        pa.field("magType", pa.string()),
        pa.field("enu_e", pa.float64()),
        pa.field("enu_n", pa.float64()),
        pa.field("enu_u", pa.float64()),
        pa.field("matchedEventId", pa.string()),
    ]
)
# Columns event_to_row fills; ENU and matchedEventId are added after selection.
_ROW_COLUMNS = [
    n for n in _CATALOG_SCHEMA.names if not n.startswith("enu_") and n != "matchedEventId"
]
_SCHEMA_VERSION = "1.0"  # docs/02 SCHEMA_VERSION
_MODEL_NAME = "CatalogEvent"
# In-memory column types, so the frame itself carries the docs/02 types (any writer, zero rows,
# all-null columns). Strings use pandas' default ``str`` semantics (a missing value is NaN, as in
# every other run table read back with pandas) with python storage, which converts to Arrow
# string, not large_string.
_STRING = pd.StringDtype("python", na_value=np.nan)
_FRAME_DTYPES: dict[str, Any] = {
    f.name: _STRING if pa.types.is_string(f.type) else "float64" for f in _CATALOG_SCHEMA
}

_PART_SUFFIX = ".part"  # temporary output name while the stage runs; see run()
_ANALYST_MODE = "manual"  # QuakeML EvaluationMode of an analyst-reviewed origin
_QUAKEML_BED = "{http://quakeml.org/xmlns/bed/1.2}"  # QuakeML 1.2 BED namespace
_ANSS_CATALOG = "{http://anss.org/xmlns/catalog/0.1}"  # ANSS catalog:* attribute namespace
# Fixed publicID of the empty document written for a no-data answer (ObsPy would draw a random
# one), so an identical query gives byte-identical output.
_NO_DATA_PUBLIC_ID = "smi:local/hq/catalog/no-data"


def _floor_second(instant: datetime) -> datetime:
    return instant.replace(microsecond=0)


def _ceil_second(instant: datetime) -> datetime:
    floor = instant.replace(microsecond=0)
    return floor if floor == instant else floor + timedelta(seconds=1)


def query_params(run: RunSection, cfg: CatalogConfig) -> dict[str, Any]:
    """FDSN event-query arguments: the run bbox, the run window padded outward to whole seconds,
    and the configured limits.

    ComCat truncates ``starttime`` and ``endtime`` to the whole second and treats ``endtime`` as
    inclusive, so an unpadded fractional ``windowEnd`` would never serve origins in
    ``[floor(windowEnd), windowEnd)``. ``select_in_run`` applies the exact bounds afterwards.
    """
    min_lon, min_lat, max_lon, max_lat = run.bbox
    params: dict[str, Any] = {
        "starttime": UTCDateTime(_floor_second(run.windowStart)),
        "endtime": UTCDateTime(_ceil_second(run.windowEnd)),
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


def json_query(params: dict[str, Any]) -> dict[str, Any]:
    """The query as JSON values: times as ISO 8601 UTC strings, numbers kept as numbers."""
    return {k: str(v) if isinstance(v, UTCDateTime) else v for k, v in params.items()}


def fetch_quakeml(client: Client, params: dict[str, Any], path: Path) -> bool:
    """Run the query and save the response bytes exactly as served to ``path``.

    Returns False, with nothing written, when the provider has no data for the query (HTTP 204).
    Every other failure propagates.
    """
    try:
        client.get_events(filename=str(path), **params)
    except FDSNNoDataException:
        return False
    return True


def empty_quakeml(provider_url: str, params: dict[str, Any]) -> bytes:
    """A valid QuakeML document with no events, standing in for a no-data (HTTP 204) answer.

    Synthesized, not served: its description names the provider and the exact query. It has a
    fixed publicID and no creation time, so the same query always gives the same bytes.
    """
    query = json.dumps(json_query(params), sort_keys=True)
    catalog = Catalog(
        resource_id=ResourceIdentifier(_NO_DATA_PUBLIC_ID),
        description=f"No events: {provider_url} returned no data (HTTP 204) for query {query}",
    )
    buffer = io.BytesIO()
    catalog.write(buffer, format="QUAKEML")
    return buffer.getvalue()


def ids_with_product(client: Client, params: dict[str, Any], product_type: str) -> set[str]:
    """ComCat ids of the query's events that carry ``product_type``.

    One extra request for the same window and bbox, filtered by product type; it parses that
    event list, not the product itself. Used to count events with a ComCat ``phase-data`` product
    (picks/arrivals; analyst picks only where the origin is manual, see ``manual_origin_ids``).
    """
    try:
        served = client.get_events(producttype=product_type, **params)
    except FDSNNoDataException:
        return set()  # HTTP 204: no event in this query has the product
    return {comcat_id(event) for event in served}


def manual_origin_ids(events: Iterable[Event]) -> set[str]:
    """ComCat ids whose preferred origin is analyst-reviewed (QuakeML evaluationMode ``manual``)."""
    ids: set[str] = set()
    for event in events:
        origin = event.preferred_origin()
        if origin is not None and origin.evaluation_mode == _ANALYST_MODE:
            ids.add(comcat_id(event))
    return ids


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


def raw_event_types(raw: bytes) -> list[tuple[str, str | None]]:
    """(ComCat id, ``<type>`` text) of every ``<event>`` in a QuakeML document, in document order.

    Read from the bytes, independently of ObsPy, whose reader skips an event whose type is not a
    QuakeML 1.2 EventType value with only a UserWarning.
    """
    events: list[tuple[str, str | None]] = []
    for element in ET.fromstring(raw).iter(f"{_QUAKEML_BED}event"):
        source = element.get(f"{_ANSS_CATALOG}eventsource")
        number = element.get(f"{_ANSS_CATALOG}eventid")
        if not source or not number:
            raise ValueError(
                f"served event {element.get('publicID')} has no catalog:eventsource/eventid"
            )
        type_element = element.find(f"{_QUAKEML_BED}type")
        events.append((source + number, None if type_element is None else type_element.text))
    return events


def read_served(path: Path, cfg: CatalogConfig) -> tuple[Catalog, Counter[str]]:
    """Parse the saved QuakeML and account for every ``<event>`` element in it.

    ObsPy's reader skips an event whose type is outside the QuakeML 1.2 EventType enum (ComCat can
    serve e.g. "Rock Slide" verbatim). A skipped event whose raw type ``cfg.eventTypes`` would drop
    anyway is returned in the Counter, by raw type, to be counted as ``droppedEventType``; any
    other skipped event raises, since it would belong in the table.
    """
    served = read_events(str(path), format="QUAKEML")
    parsed = Counter(comcat_id(event) for event in served)
    skipped: Counter[str] = Counter()
    for event_id, event_type in raw_event_types(path.read_bytes()):
        if parsed[event_id] > 0:
            parsed[event_id] -= 1
            continue
        if cfg.eventTypes is None or event_type in cfg.eventTypes:
            raise ValueError(
                f"event {event_id} (type {event_type!r}) was served, but ObsPy's QuakeML reader "
                "skipped it, so it cannot be converted"
            )
        skipped[str(event_type)] += 1
        log.info("dropping %s: event type %r (not a QuakeML 1.2 event type)", event_id, event_type)
    return served, skipped


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
    origin time the configured depth datums do not cover. (ObsPy's reader already rejects a
    non-finite number.)
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
    if magnitude is not None and not magnitude.magnitude_type:
        raise ValueError(f"event {event_id}: preferred magnitude has no type")

    depth_m = float(origin.depth)
    return {
        "id": event_id,
        "source": f"{contributor.upper()} via {cfg.providerLabel}",
        "t": epoch_s(origin.time.ns),
        "latitude": float(origin.latitude),
        "longitude": float(origin.longitude),
        # Below the contributor's datum (sea level for UU), as published. Not comparable to
        # SeismicEvent.depthKm, which is below run.refSurfaceElevM; compare elevM instead.
        "depthKm": depth_m / _M_PER_KM,
        "depthDatum": f"{datum.label}; source: {datum.sourceUrls[0]}",
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
    catalog: Catalog,
    run: RunSection,
    cfg: CatalogConfig,
    skipped_by_reader: Counter[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Served events -> ``CatalogEvent`` rows for the run, sorted by ``t``, plus filter counts.

    ``skipped_by_reader`` holds served events ObsPy could not parse, by type (``read_served``);
    they count as served and as ``droppedEventType``. Every served event is either kept or counted
    under exactly one ``dropped*`` reason, which is checked. ``manualOrigin`` counts kept rows
    whose preferred origin is analyst-reviewed.

    Columns carry the docs/02 types (float64, string) even with zero rows or all-null columns. A
    missing value is NaN in string columns as in float columns (pandas' default ``str`` dtype, and
    what catalog.parquet reads back as): test it with ``pd.isna``, never ``is None`` or truthiness,
    or read rows through ``hq_contracts.io.from_frame``, which turns it into None.
    """
    skipped = skipped_by_reader or Counter()
    events, dropped_types = filter_event_types(catalog, cfg)
    rows = pd.DataFrame([event_to_row(event, cfg) for event in events], columns=_ROW_COLUMNS)
    rows = rows.astype({name: _FRAME_DTYPES[name] for name in _ROW_COLUMNS})
    if rows["id"].duplicated().any():
        dupes = sorted(rows.loc[rows["id"].duplicated(), "id"])
        raise ValueError(f"duplicate catalog event ids: {dupes}")
    rows, dropped = select_in_run(rows, run)
    rows = rows.sort_values(["t", "id"], kind="stable").reset_index(drop=True)
    e, n, u = to_enu(rows["latitude"], rows["longitude"], rows["elevM"], run.origin)
    rows["enu_e"], rows["enu_n"], rows["enu_u"] = e, n, u
    rows["matchedEventId"] = pd.Series(np.nan, index=rows.index, dtype=_STRING)
    counts = {
        "served": len(catalog) + skipped.total(),
        "droppedEventType": dropped_types.total() + skipped.total(),
        **dropped,
        "kept": len(rows),
        "manualOrigin": int(rows["id"].isin(manual_origin_ids(events)).sum()),
    }
    accounted = counts["kept"] + sum(v for k, v in counts.items() if k.startswith("dropped"))
    if accounted != counts["served"]:
        raise ValueError(f"catalog counts do not add up to the served events: {counts}")
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


def _part(path: Path) -> Path:
    return path.with_name(path.name + _PART_SUFFIX)


def run(ctx: "RunContext") -> None:
    """Stage ``catalog``: write ``catalog.quakeml`` (raw) and ``catalog.parquet`` to the run dir.

    Both files are built under ``.part`` names and replace the previous pair only once everything
    succeeded; on any failure the ``.part`` files are removed and the previous pair is untouched.
    """
    started = time.perf_counter()
    run_cfg: RunSection = ctx.config.run
    cfg: CatalogConfig = ctx.config.seismology.catalog
    params = query_params(run_cfg, cfg)
    window = f"[{run_cfg.windowStart.isoformat()}, {run_cfg.windowEnd.isoformat()})"
    log.info("catalog: querying %s for %s, bbox %s", cfg.provider, window, run_cfg.bbox)

    client = _client(cfg)
    quakeml_path, table_path = ctx.path(QUAKEML_NAME), ctx.path(TABLE_NAME)
    quakeml_part, table_part = _part(quakeml_path), _part(table_path)
    try:
        no_data = not fetch_quakeml(client, params, quakeml_part)
        if no_data:
            log.warning(
                "catalog: %s returned no data (HTTP 204) for %s, bbox %s; writing an empty catalog",
                client.base_url,
                window,
                run_cfg.bbox,
            )
            quakeml_part.write_bytes(empty_quakeml(client.base_url, params))
        raw = quakeml_part.read_bytes()
        served, skipped = read_served(quakeml_part, cfg)
        rows, counts = build_catalog(served, run_cfg, cfg, skipped)

        with_product = (
            set() if no_data else ids_with_product(client, params, cfg.arrivalsProductType)
        )
        kept_with_product = sorted(set(rows["id"]) & with_product)
        counts["withArrivalsProduct"] = len(kept_with_product)
        counts["withArrivalsProductManual"] = len(
            set(kept_with_product) & manual_origin_ids(served)
        )
        if counts["kept"] and not counts["withArrivalsProduct"]:
            log.warning(
                "catalog: none of the %d kept events carries a %r product; "
                "check arrivalsProductType in seismology.yaml",
                counts["kept"],
                cfg.arrivalsProductType,
            )

        _write_table(rows, table_part)
        os.replace(table_part, table_path)
        os.replace(quakeml_part, quakeml_path)
    finally:
        quakeml_part.unlink(missing_ok=True)
        table_part.unlink(missing_ok=True)

    runtime_s = time.perf_counter() - started
    log.info(
        "catalog: %d events kept of %d served for %s, bbox %s; %d with %s (%d of them with a "
        "manual origin) in %.1f s",
        counts["kept"],
        counts["served"],
        window,
        run_cfg.bbox,
        counts["withArrivalsProduct"],
        cfg.arrivalsProductType,
        counts["withArrivalsProductManual"],
        runtime_s,
    )
    ctx.record(
        STAGE,
        runtime_s=runtime_s,
        counts=counts,
        # Namespaced: the stage's params merge into ProcessingRun.matching, which the match stage
        # shares, so they land under matching.catalog and cannot collide with its keys.
        params={
            "catalog": {
                "providerUrl": client.base_url,
                "query": json_query(params),  # as sent: bounds padded outward to whole seconds
                "windowStart": run_cfg.windowStart.isoformat(),  # exact bounds applied to rows
                "windowEnd": run_cfg.windowEnd.isoformat(),
                "windowEndExclusive": True,
                "noDataFromProvider": no_data,
                "idsWithArrivalsProduct": kept_with_product,
                "quakemlSha256": hashlib.sha256(raw).hexdigest(),
                "config": cfg.model_dump(mode="json"),  # every catalog knob, recorded once
            }
        },
    )

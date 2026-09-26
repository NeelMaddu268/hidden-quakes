"""hq.match.catalog: QuakeML -> CatalogEvent rows, window/bbox selection, the stage, fail-loud cases.

The fixture is a real ComCat response for the showcase query trimmed to 3 events, served in
ComCat's newest-first order. Offline: the FDSN client is replaced by a fake.
"""

import copy
import hashlib
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from obspy import UTCDateTime, read_events
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException
from obspy.core.event import Catalog
from pydantic import ValidationError

from hq.config.run import RunSection, epoch_s
from hq.config.seismology import SeismologyConfig
from hq.locate.coords import to_enu
from hq.match import catalog as catalog_stage

pytestmark = pytest.mark.smoke

# Facts of the three fixture events, as published (fixture data, not product numbers).
LATEST_ID = "uu80155936"  # 2026-09-10T23:42:53.300Z, ml 1.34, depth 2740 m, lon -112.9005
MIDDLE_ID = "uu80155911"  # 2026-09-10T22:23:28.740Z, md 1.19, lat 38.52, lon -112.9005
EARLIEST_ID = "uu80155571"  # 2026-09-10T09:27:56.880Z, ml 2.14, lat 38.4928, lon -112.89667
EARLIEST_LON = -112.89666666667
# docs/02 CatalogEvent, flattened (docs/02 §2): str -> Arrow string, float -> Arrow double.
DOCS02_TYPES = {
    "id": pa.string(),
    "source": pa.string(),
    "t": pa.float64(),
    "latitude": pa.float64(),
    "longitude": pa.float64(),
    "depthKm": pa.float64(),
    "depthDatum": pa.string(),
    "elevM": pa.float64(),
    "mag": pa.float64(),
    "magType": pa.string(),
    "enu_e": pa.float64(),
    "enu_n": pa.float64(),
    "enu_u": pa.float64(),
    "matchedEventId": pa.string(),
}
DOCS02_COLUMNS = list(DOCS02_TYPES)
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
NS_PER_S = 1_000_000_000


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(UTC)


def _with(run: RunSection, **update: Any) -> RunSection:
    return RunSection.model_validate({**run.model_dump(), **update})


def _with_datum(cfg: SeismologyConfig, **update: Any) -> SeismologyConfig:
    datum = cfg.catalog.datums["uu"].model_copy(update=update)
    catalog = cfg.catalog.model_copy(update={"datums": {"uu": datum}})
    return cfg.model_copy(update={"catalog": catalog})


def _with_catalog(cfg: SeismologyConfig, **update: Any) -> SeismologyConfig:
    return cfg.model_copy(update={"catalog": cfg.catalog.model_copy(update=update)})


def _served(path: Path) -> Catalog:
    return read_events(str(path), format="QUAKEML")


def _by_id(cat: Catalog, event_id: str) -> Any:
    return next(ev for ev in cat if catalog_stage.comcat_id(ev) == event_id)


def _types(schema: pa.Schema) -> dict[str, pa.DataType]:
    return {f.name: f.type for f in schema}


# ---------------------------------------------------------------- config and query


def test_seismology_yaml_parses_and_rejects_unknown_keys(
    seismology_config: SeismologyConfig,
) -> None:
    cat = seismology_config.catalog
    assert cat.minMagnitude is None and cat.maxMagnitude is None
    assert "uu" in cat.datums
    # ComCat's product type name for picks/arrivals; a misspelling makes ComCat answer "no data".
    assert cat.arrivalsProductType == "phase-data"
    raw = seismology_config.model_dump(mode="json")
    datum = raw["catalog"]["datums"]["uu"]
    for bad in (
        {**raw, "unknown": 1},
        {**raw, "catalog": {**raw["catalog"], "unknown": 1}},
        {**raw, "catalog": {**raw["catalog"], "datums": {"uu": {**datum, "unknown": 1}}}},
    ):
        with pytest.raises(ValidationError) as exc:
            SeismologyConfig.model_validate(bad)
        assert [e["type"] for e in exc.value.errors()] == ["extra_forbidden"]


def test_query_uses_run_bbox_and_window_padded_to_whole_seconds(
    run_section: RunSection, seismology_config: SeismologyConfig
) -> None:
    params = catalog_stage.query_params(run_section, seismology_config.catalog)
    assert set(params) == {
        "starttime",
        "endtime",
        "minlongitude",
        "minlatitude",
        "maxlongitude",
        "maxlatitude",
    }
    # Whole-second bounds (the showcase run) are sent unchanged. Integer nanoseconds from datetime
    # arithmetic, independent of ObsPy's conversion.
    for key, bound in (("starttime", run_section.windowStart), ("endtime", run_section.windowEnd)):
        assert bound.microsecond == 0
        assert params[key].ns == (bound - EPOCH) // timedelta(microseconds=1) * 1000
    # Sub-second bounds are padded outward: start floored, end ceiled, to whole seconds.
    fractional = _with(
        run_section,
        windowStart=_utc("2026-09-10T00:00:00.999999Z"),
        windowEnd=_utc("2026-09-10T02:00:00.000001Z"),
    )
    params = catalog_stage.query_params(fractional, seismology_config.catalog)
    assert params["starttime"] == UTCDateTime("2026-09-10T00:00:00Z")
    assert params["endtime"] == UTCDateTime("2026-09-10T02:00:01Z")
    assert (
        params["minlongitude"],
        params["minlatitude"],
        params["maxlongitude"],
        params["maxlatitude"],
    ) == run_section.bbox
    limited = seismology_config.catalog.model_copy(update={"minMagnitude": 0.5, "maxMagnitude": 3})
    params = catalog_stage.query_params(run_section, limited)
    assert params["minmagnitude"] == 0.5 and params["maxmagnitude"] == 3


def test_client_uses_configured_provider_and_timeout(
    monkeypatch: pytest.MonkeyPatch, seismology_config: SeismologyConfig
) -> None:
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_client(*args: Any, **kwargs: Any) -> str:
        seen.append((args, kwargs))
        return "client"

    monkeypatch.setattr(catalog_stage, "Client", fake_client)
    cfg = seismology_config.catalog
    assert catalog_stage._client(cfg) == "client"
    assert seen == [((cfg.provider,), {"timeout": cfg.timeoutS})]


# ---------------------------------------------------------------- QuakeML -> rows


def test_fixture_rows_every_field(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    rows, counts = catalog_stage.build_catalog(
        _served(comcat_quakeml), run_section, seismology_config.catalog
    )
    assert list(rows.columns) == DOCS02_COLUMNS
    assert counts["served"] == 3 and counts["kept"] == 3 and counts["manualOrigin"] == 3
    assert rows["t"].dtype == np.float64

    datum = seismology_config.catalog.datums["uu"]
    assert set(rows["depthDatum"]) == {f"{datum.label}; source: {datum.sourceUrls[0]}"}

    row = rows.set_index("id").loc[LATEST_ID]
    assert row["source"] == "UU via USGS ComCat"
    assert row["t"] == _utc("2026-09-10T23:42:53.300Z").timestamp()
    assert row["latitude"] == 38.512166666667
    assert row["longitude"] == -112.9005
    assert row["depthKm"] == 2.74
    assert row["depthDatum"].startswith("km below sea level")
    assert row["depthDatum"].endswith("https://quake.utah.edu/resources/catalog-details/")
    assert row["elevM"] == -2740.0  # datum surface 0 m ASL minus 2740 m
    assert row["mag"] == 1.34 and row["magType"] == "ml"
    e, n, _ = to_enu(row["latitude"], row["longitude"], row["elevM"], run_section.origin)
    assert (row["enu_e"], row["enu_n"]) == (float(e), float(n))
    assert row["enu_u"] == pytest.approx(-2740.0 - run_section.origin.elevM, abs=1e-9)
    assert pd.isna(row["matchedEventId"])

    middle = rows.set_index("id").loc[MIDDLE_ID]
    assert middle["magType"] == "md" and middle["mag"] == 1.19


@pytest.mark.parametrize("empty", [False, True], ids=["fixture", "zero-rows"])
def test_frame_carries_docs02_types_without_a_writer_schema(
    empty: bool,
    run_section: RunSection,
    seismology_config: SeismologyConfig,
    comcat_quakeml: Path,
) -> None:
    """A schema-less writer (hq_contracts.io.write_table) must still produce docs/02 types."""
    served = Catalog() if empty else _served(comcat_quakeml)
    rows, counts = catalog_stage.build_catalog(served, run_section, seismology_config.catalog)
    assert counts["kept"] == (0 if empty else 3)
    table = pa.Table.from_pandas(rows, preserve_index=False)
    assert _types(table.schema) == DOCS02_TYPES


def test_elev_uses_the_configured_datum_surface(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    cfg = _with_datum(seismology_config, surfaceElevM=1500.0)
    rows, _ = catalog_stage.build_catalog(_served(comcat_quakeml), run_section, cfg.catalog)
    row = rows.set_index("id").loc[EARLIEST_ID]
    assert row["depthKm"] == 4.06  # as published, unchanged by the datum
    assert row["elevM"] == 1500.0 - 4060.0
    assert row["enu_u"] == pytest.approx(1500.0 - 4060.0 - run_section.origin.elevM, abs=1e-9)


def test_sorted_by_t_not_by_id(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    assert [catalog_stage.comcat_id(ev) for ev in served] == [LATEST_ID, MIDDLE_ID, EARLIEST_ID]
    rows, _ = catalog_stage.build_catalog(served, run_section, seismology_config.catalog)
    assert list(rows["id"]) == [EARLIEST_ID, MIDDLE_ID, LATEST_ID]
    assert np.all(np.diff(rows["t"].to_numpy()) > 0)

    # Move the highest id before the lowest in time, so id order and time order disagree.
    earliest_t = _by_id(served, EARLIEST_ID).preferred_origin().time
    _by_id(served, LATEST_ID).preferred_origin().time = earliest_t - 1.0
    rows, _ = catalog_stage.build_catalog(served, run_section, seismology_config.catalog)
    assert list(rows["id"]) == [LATEST_ID, EARLIEST_ID, MIDDLE_ID]
    assert np.all(np.diff(rows["t"].to_numpy()) > 0)


def test_window_start_inclusive_end_exclusive(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    t_latest = _utc("2026-09-10T23:42:53.300Z")
    at_end = _with(run_section, windowEnd=t_latest)
    rows, counts = catalog_stage.build_catalog(
        _served(comcat_quakeml), at_end, seismology_config.catalog
    )
    assert LATEST_ID not in set(rows["id"])
    assert counts["droppedAtOrAfterWindowEnd"] == 1 and counts["kept"] == 2

    just_after = _with(run_section, windowEnd=t_latest + timedelta(microseconds=1))
    rows, counts = catalog_stage.build_catalog(
        _served(comcat_quakeml), just_after, seismology_config.catalog
    )
    assert LATEST_ID in set(rows["id"]) and counts["droppedAtOrAfterWindowEnd"] == 0

    t_earliest = _utc("2026-09-10T09:27:56.880Z")
    starts_at_earliest = _with(run_section, windowStart=t_earliest)
    rows, counts = catalog_stage.build_catalog(
        _served(comcat_quakeml), starts_at_earliest, seismology_config.catalog
    )
    assert EARLIEST_ID in set(rows["id"]) and counts["droppedBeforeWindowStart"] == 0

    starts_after_earliest = _with(run_section, windowStart=t_earliest + timedelta(microseconds=1))
    rows, counts = catalog_stage.build_catalog(
        _served(comcat_quakeml), starts_after_earliest, seismology_config.catalog
    )
    assert list(rows["id"]) == [MIDDLE_ID, LATEST_ID]
    assert counts["droppedBeforeWindowStart"] == 1 and counts["kept"] == 2


def _instant_where_epoch_conversions_disagree(start: datetime) -> datetime:
    """First millisecond after ``start`` whose ObsPy epoch is below ``datetime.timestamp()``."""
    for ms in range(1, 10_000):
        instant = start + timedelta(milliseconds=ms)
        if UTCDateTime(instant).timestamp < instant.timestamp():
            return instant
    raise AssertionError("no disagreeing instant found")


def test_window_edges_are_exact_at_sub_second_instants(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    """An origin exactly on a window edge is judged by one epoch conversion on both sides.

    ``t`` and ``RunSection.window_*_s`` both come from ``hq.config.run.epoch_s``, including at an
    instant where ObsPy's own ``UTCDateTime.timestamp`` is one ulp low.
    """
    instant = _instant_where_epoch_conversions_disagree(_utc("2026-09-10T23:42:53.300Z"))
    served = _served(comcat_quakeml)
    _by_id(served, LATEST_ID).preferred_origin().time = UTCDateTime(instant)
    cfg = seismology_config.catalog
    assert epoch_s(UTCDateTime(instant).ns) == _with(run_section, windowEnd=instant).window_end_s

    rows, counts = catalog_stage.build_catalog(served, _with(run_section, windowEnd=instant), cfg)
    assert LATEST_ID not in set(rows["id"]) and counts["droppedAtOrAfterWindowEnd"] == 1

    rows, counts = catalog_stage.build_catalog(served, _with(run_section, windowStart=instant), cfg)
    assert list(rows["id"]) == [LATEST_ID] and counts["droppedBeforeWindowStart"] == 2


def test_bbox_is_respected_edges_inclusive(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    min_lon, min_lat, max_lon, max_lat = run_section.bbox
    cfg = seismology_config.catalog
    # minLat 38.50 excludes the event at 38.4928; maxLat 38.52 sits exactly on the middle event.
    box = _with(run_section, bbox=(min_lon, 38.50, max_lon, 38.52))
    rows, counts = catalog_stage.build_catalog(_served(comcat_quakeml), box, cfg)
    assert list(rows["id"]) == [MIDDLE_ID, LATEST_ID]
    assert counts["droppedOutsideBbox"] == 1

    # The origin must lie inside the bbox (RunSection); it does not affect selection.
    origin = run_section.origin.model_dump()
    # maxLon exactly on the two events at -112.9005 keeps them and drops the one further east.
    box = _with(
        run_section,
        bbox=(min_lon, min_lat, -112.9005, max_lat),
        origin={**origin, "lon": min_lon},
    )
    rows, counts = catalog_stage.build_catalog(_served(comcat_quakeml), box, cfg)
    assert list(rows["id"]) == [MIDDLE_ID, LATEST_ID]
    assert counts["droppedOutsideBbox"] == 1

    # minLon exactly on the eastern event keeps only it.
    box = _with(
        run_section,
        bbox=(EARLIEST_LON, min_lat, max_lon, max_lat),
        origin={**origin, "lon": max_lon},
    )
    rows, counts = catalog_stage.build_catalog(_served(comcat_quakeml), box, cfg)
    assert list(rows["id"]) == [EARLIEST_ID]
    assert counts["droppedOutsideBbox"] == 2


def test_event_type_filter_drops_and_counts(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    _by_id(served, MIDDLE_ID).event_type = "quarry blast"
    rows, counts = catalog_stage.build_catalog(served, run_section, seismology_config.catalog)
    assert MIDDLE_ID not in set(rows["id"])
    assert counts["droppedEventType"] == 1 and counts["kept"] == 2

    every_type = seismology_config.catalog.model_copy(update={"eventTypes": None})
    rows, counts = catalog_stage.build_catalog(served, run_section, every_type)
    assert MIDDLE_ID in set(rows["id"])
    assert counts["droppedEventType"] == 0 and counts["kept"] == 3


def test_missing_magnitude_gives_nulls(
    tmp_path: Path,
    run_section: RunSection,
    seismology_config: SeismologyConfig,
    comcat_quakeml: Path,
) -> None:
    served = _served(comcat_quakeml)
    event = _by_id(served, MIDDLE_ID)
    event.magnitudes = []
    event.preferred_magnitude_id = None
    rows, _ = catalog_stage.build_catalog(served, run_section, seismology_config.catalog)
    path = tmp_path / "catalog.parquet"
    catalog_stage.write_catalog(rows, path)
    table = pq.read_table(path).to_pylist()
    row = next(r for r in table if r["id"] == MIDDLE_ID)
    assert row["mag"] is None and row["magType"] is None
    assert all(r["mag"] is not None and r["magType"] for r in table if r["id"] != MIDDLE_ID)


def test_manual_origin_count(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    _by_id(served, MIDDLE_ID).preferred_origin().evaluation_mode = "automatic"
    assert catalog_stage.manual_origin_ids(served) == {LATEST_ID, EARLIEST_ID}
    _, counts = catalog_stage.build_catalog(served, run_section, seismology_config.catalog)
    assert counts["kept"] == 3 and counts["manualOrigin"] == 2


def test_origin_without_evaluation_mode_is_not_manual(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    _by_id(served, MIDDLE_ID).preferred_origin().evaluation_mode = None
    assert catalog_stage.manual_origin_ids(served) == {LATEST_ID, EARLIEST_ID}
    _, counts = catalog_stage.build_catalog(served, run_section, seismology_config.catalog)
    assert counts["kept"] == 3 and counts["manualOrigin"] == 2


def test_origin_exactly_at_datum_valid_from_is_kept(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    cfg = _with_datum(seismology_config, validFrom=_utc("2026-09-10T09:27:56.880Z"))
    rows, _ = catalog_stage.build_catalog(_served(comcat_quakeml), run_section, cfg.catalog)
    assert list(rows["id"]) == [EARLIEST_ID, MIDDLE_ID, LATEST_ID]


def test_each_dropped_event_is_counted_once(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    """An event outside both the window and the bbox counts under the window reason only."""
    served = _served(comcat_quakeml)
    _by_id(served, LATEST_ID).preferred_origin().latitude = run_section.bbox[3] + 0.5
    _by_id(served, EARLIEST_ID).preferred_origin().latitude = run_section.bbox[3] + 0.5
    run = _with(run_section, windowEnd=_utc("2026-09-10T23:00:00Z"))
    rows, counts = catalog_stage.build_catalog(served, run, seismology_config.catalog)
    assert list(rows["id"]) == [MIDDLE_ID]
    assert counts["droppedAtOrAfterWindowEnd"] == 1  # LATEST: after windowEnd and outside bbox
    assert counts["droppedOutsideBbox"] == 1  # EARLIEST: in the window, outside bbox
    dropped = sum(v for k, v in counts.items() if k.startswith("dropped"))
    assert counts["served"] == 3 == dropped + counts["kept"]


def test_counts_that_do_not_add_up_fail(
    monkeypatch: pytest.MonkeyPatch,
    run_section: RunSection,
    seismology_config: SeismologyConfig,
    comcat_quakeml: Path,
) -> None:
    real = catalog_stage.select_in_run

    def double_counting(df: pd.DataFrame, run: RunSection) -> tuple[pd.DataFrame, dict[str, int]]:
        rows, counts = real(df, run)
        return rows, {**counts, "droppedOutsideBbox": counts["droppedOutsideBbox"] + 1}

    monkeypatch.setattr(catalog_stage, "select_in_run", double_counting)
    with pytest.raises(ValueError, match="do not add up"):
        catalog_stage.build_catalog(_served(comcat_quakeml), run_section, seismology_config.catalog)


# ---------------------------------------------------------------- fail loudly


def _first_event_without(text: str, element: str) -> str:
    """Remove ``<element>...</element>`` from the first event only."""
    first = text.index("<event ")
    end = text.index("</event>", first)
    block = re.sub(rf"<{element}>.*?</{element}>", "", text[first:end], count=1, flags=re.DOTALL)
    return text[:first] + block + text[end:]


def test_origin_without_depth_fails(
    tmp_path: Path,
    run_section: RunSection,
    seismology_config: SeismologyConfig,
    comcat_quakeml: Path,
) -> None:
    broken = tmp_path / "no_depth.quakeml"
    broken.write_text(_first_event_without(comcat_quakeml.read_text("utf-8"), "depth"), "utf-8")
    with pytest.raises(ValueError, match="no depth"):
        catalog_stage.build_catalog(_served(broken), run_section, seismology_config.catalog)


def test_magnitude_without_type_fails(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    _by_id(served, LATEST_ID).preferred_magnitude().magnitude_type = None
    with pytest.raises(ValueError, match=f"{LATEST_ID}: preferred magnitude has no type"):
        catalog_stage.build_catalog(served, run_section, seismology_config.catalog)


def test_magnitude_without_value_fails(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    _by_id(served, LATEST_ID).preferred_magnitude().mag = None
    with pytest.raises(ValueError, match=f"{LATEST_ID}: preferred magnitude has no value"):
        catalog_stage.build_catalog(served, run_section, seismology_config.catalog)


@pytest.mark.parametrize("attr", ["latitude", "longitude"])
def test_origin_without_position_fails(
    attr: str, run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    setattr(_by_id(served, LATEST_ID).preferred_origin(), attr, None)
    with pytest.raises(ValueError, match=f"{LATEST_ID}: preferred origin lacks time or position"):
        catalog_stage.build_catalog(served, run_section, seismology_config.catalog)


def test_event_without_type_fails(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    _by_id(served, MIDDLE_ID).event_type = None
    with pytest.raises(ValueError, match=f"event {MIDDLE_ID} has no QuakeML type"):
        catalog_stage.build_catalog(served, run_section, seismology_config.catalog)


def test_event_without_preferred_origin_fails(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    _by_id(served, LATEST_ID).preferred_origin_id = None
    with pytest.raises(ValueError, match="no preferred origin"):
        catalog_stage.build_catalog(served, run_section, seismology_config.catalog)


def test_undocumented_contributor_fails(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    origin = _by_id(served, LATEST_ID).preferred_origin()
    origin.extra = copy.deepcopy(origin.extra)
    origin.extra["datasource"]["value"] = "nn"
    with pytest.raises(ValueError, match="no depth datum configured for contributor 'nn'"):
        catalog_stage.build_catalog(served, run_section, seismology_config.catalog)


def test_origin_before_datum_valid_from_fails(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    cfg = _with_datum(seismology_config, validFrom=_utc("2026-09-10T12:00:00Z"))
    with pytest.raises(ValueError, match="predates"):
        catalog_stage.build_catalog(_served(comcat_quakeml), run_section, cfg.catalog)


def test_missing_event_id_attribute_fails(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    del served[0].extra["eventid"]
    with pytest.raises(ValueError, match="catalog:eventid"):
        catalog_stage.build_catalog(served, run_section, seismology_config.catalog)


def test_duplicate_event_ids_fail(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    served.append(copy.deepcopy(served[0]))
    with pytest.raises(ValueError, match="duplicate"):
        catalog_stage.build_catalog(served, run_section, seismology_config.catalog)


# ---------------------------------------------------------------- stage


class FakeClient:
    """Stands in for obspy's FDSN Client: serves a QuakeML file, records every query.

    ``main_error`` is raised by the main (``filename=``) query; ``product_error`` by the
    product query. ``with_product=None`` makes the product query answer "no data" (HTTP 204).
    """

    base_url = "https://fdsn.invalid"

    def __init__(
        self,
        quakeml: Path,
        with_product: set[str] | None,
        *,
        main_error: Exception | None = None,
        product_error: Exception | None = None,
    ) -> None:
        self.quakeml = quakeml
        self.with_product = with_product
        self.main_error = main_error
        self.product_error = product_error
        self.calls: list[dict[str, Any]] = []

    def get_events(self, **kwargs: Any) -> Catalog | None:
        self.calls.append(kwargs)
        if "filename" in kwargs:
            if self.main_error is not None:
                raise self.main_error
            Path(kwargs["filename"]).write_bytes(self.quakeml.read_bytes())
            return None
        if self.product_error is not None:
            raise self.product_error
        if self.with_product is None:
            raise FDSNNoDataException("No data available for request.")
        served = _served(self.quakeml)
        return Catalog([ev for ev in served if catalog_stage.comcat_id(ev) in self.with_product])


def _use(monkeypatch: pytest.MonkeyPatch, fake: FakeClient) -> None:
    monkeypatch.setattr(catalog_stage, "_client", lambda cfg: fake)


def _no_part_files(run_dir: Path) -> bool:
    return not list(run_dir.glob("*.part"))


def test_stage_writes_both_files_and_records(
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
    seismology_config: SeismologyConfig,
    comcat_quakeml: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeClient(comcat_quakeml, with_product={LATEST_ID, EARLIEST_ID})
    _use(monkeypatch, fake)
    ctx = make_ctx(run_section)

    with caplog.at_level(logging.INFO, logger=catalog_stage.__name__):
        catalog_stage.run(ctx)

    raw = ctx.path("catalog.quakeml").read_bytes()
    assert raw == comcat_quakeml.read_bytes()
    table = pq.read_table(ctx.path("catalog.parquet"))
    assert table.schema.metadata[b"schemaVersion"] == b"1.0"
    assert table.schema.metadata[b"model"] == b"CatalogEvent"
    assert _types(table.schema) == DOCS02_TYPES
    assert table.column_names == DOCS02_COLUMNS
    df = table.to_pandas()
    assert list(df["id"]) == [EARLIEST_ID, MIDDLE_ID, LATEST_ID]
    assert df["matchedEventId"].isna().all()
    assert _no_part_files(ctx.run_dir)

    query, product_query = fake.calls
    assert query["starttime"] == UTCDateTime(run_section.windowStart)
    assert query["endtime"] == UTCDateTime(run_section.windowEnd)
    assert product_query["producttype"] == "phase-data"
    # The product query asks about exactly the same window and bbox.
    main_args = {k: v for k, v in query.items() if k != "filename"}
    assert {k: v for k, v in product_query.items() if k != "producttype"} == main_args

    (record,) = ctx.records
    assert record["stage"] == "catalog" and record["runtime_s"] >= 0
    assert record["counts"] == {
        "served": 3,
        "droppedEventType": 0,
        "droppedBeforeWindowStart": 0,
        "droppedAtOrAfterWindowEnd": 0,
        "droppedOutsideBbox": 0,
        "kept": 3,
        "manualOrigin": 3,
        "withArrivalsProduct": 2,
        "withArrivalsProductManual": 2,
    }
    # One namespaced key, so ProcessingRun.matching.catalog can't collide with the match stage.
    assert set(record["params"]) == {"catalog"}
    params = record["params"]["catalog"]
    assert set(params) == {
        "providerUrl",
        "query",
        "windowStart",
        "windowEnd",
        "windowEndExclusive",
        "noDataFromProvider",
        "idsWithArrivalsProduct",
        "quakemlSha256",
        "config",
    }
    assert params["providerUrl"] == FakeClient.base_url
    assert params["quakemlSha256"] == hashlib.sha256(raw).hexdigest()
    assert params["query"]["minlongitude"] == run_section.bbox[0]
    assert params["query"]["starttime"] == str(UTCDateTime(run_section.windowStart))
    assert params["windowStart"] == run_section.windowStart.isoformat()
    assert params["windowEnd"] == run_section.windowEnd.isoformat()
    assert params["windowEndExclusive"] is True
    assert params["noDataFromProvider"] is False
    assert params["idsWithArrivalsProduct"] == sorted([EARLIEST_ID, LATEST_ID])
    # Every knob once, inside config (eventTypes [earthquake], lead decision 2).
    assert params["config"] == seismology_config.catalog.model_dump(mode="json")
    assert params["config"]["eventTypes"] == ["earthquake"]

    # The stage summary (rule 12): kept and served counts, the window and the runtime.
    (summary,) = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and r.getMessage().startswith("catalog: 3 events kept of 3")
    ]
    assert run_section.windowStart.isoformat() in summary.getMessage()
    assert re.search(r"in \d+\.\d s$", summary.getMessage())


def test_arrivals_counts_only_kept_rows_and_manual_origins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
    comcat_quakeml: Path,
) -> None:
    """A product on a dropped event is not counted; analyst arrivals need a manual origin."""
    text = comcat_quakeml.read_text("utf-8")
    start = text.index(f'catalog:eventid="{MIDDLE_ID[2:]}"')
    manual = "<evaluationMode>manual</evaluationMode>"
    at = text.index(manual, start)  # the preferred origin's mode comes first in the event block
    edited = tmp_path / "middle_automatic.quakeml"
    edited.write_text(
        text[:at] + "<evaluationMode>automatic</evaluationMode>" + text[at + len(manual) :],
        "utf-8",
    )
    fake = FakeClient(edited, with_product={LATEST_ID, MIDDLE_ID, EARLIEST_ID})
    _use(monkeypatch, fake)
    ctx = make_ctx(_with(run_section, windowEnd=_utc("2026-09-10T23:42:53.300Z")))

    catalog_stage.run(ctx)

    (record,) = ctx.records
    counts = record["counts"]
    assert counts["kept"] == 2 and counts["droppedAtOrAfterWindowEnd"] == 1
    assert counts["manualOrigin"] == 1
    assert counts["withArrivalsProduct"] == 2
    assert counts["withArrivalsProductManual"] == 1
    assert record["params"]["catalog"]["idsWithArrivalsProduct"] == sorted([EARLIEST_ID, MIDDLE_ID])


def test_no_product_on_kept_events_warns(
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
    comcat_quakeml: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _use(monkeypatch, FakeClient(comcat_quakeml, with_product=None))
    ctx = make_ctx(run_section)
    with caplog.at_level(logging.WARNING, logger=catalog_stage.__name__):
        catalog_stage.run(ctx)
    (record,) = ctx.records
    assert record["counts"]["kept"] == 3 and record["counts"]["withArrivalsProduct"] == 0
    assert any(
        r.levelno == logging.WARNING and "'phase-data'" in r.getMessage() for r in caplog.records
    )


def test_empty_window_writes_empty_outputs_and_records_no_data(
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
    comcat_quakeml: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lead decision 1: HTTP 204 on the main query is a valid, empty result."""
    no_data = FDSNNoDataException("No data available for request.")
    fake = FakeClient(comcat_quakeml, with_product=set(), main_error=no_data)
    _use(monkeypatch, fake)
    ctx = make_ctx(run_section)

    with caplog.at_level(logging.WARNING, logger=catalog_stage.__name__):
        catalog_stage.run(ctx)

    assert len(fake.calls) == 1  # no product query for an empty window
    quakeml = ctx.path("catalog.quakeml")
    first = quakeml.read_bytes()
    assert len(read_events(str(quakeml), format="QUAKEML")) == 0
    table = pq.read_table(ctx.path("catalog.parquet"))
    assert table.num_rows == 0
    assert _types(table.schema) == DOCS02_TYPES
    assert table.schema.metadata[b"schemaVersion"] == b"1.0"
    assert table.schema.metadata[b"model"] == b"CatalogEvent"
    assert _no_part_files(ctx.run_dir)

    (record,) = ctx.records
    assert record["counts"] == {
        "served": 0,
        "droppedEventType": 0,
        "droppedBeforeWindowStart": 0,
        "droppedAtOrAfterWindowEnd": 0,
        "droppedOutsideBbox": 0,
        "kept": 0,
        "manualOrigin": 0,
        "withArrivalsProduct": 0,
        "withArrivalsProductManual": 0,
    }
    params = record["params"]["catalog"]
    assert params["noDataFromProvider"] is True
    assert params["idsWithArrivalsProduct"] == []
    assert params["quakemlSha256"] == hashlib.sha256(first).hexdigest()
    assert any(r.levelno == logging.WARNING and "no data" in r.getMessage() for r in caplog.records)

    # Identical query, identical bytes: the empty document has no random id or creation time.
    catalog_stage.run(make_ctx(run_section))
    assert quakeml.read_bytes() == first


def _write_previous_pair(ctx: Any) -> dict[str, bytes]:
    previous = {
        "catalog.quakeml": b"PREVIOUS QUAKEML",
        "catalog.parquet": b"PREVIOUS PARQUET",
    }
    for name, content in previous.items():
        ctx.path(name).write_bytes(content)
    return previous


@pytest.mark.parametrize(
    "failure",
    [
        {"product_error": FDSNException("Service temporarily unavailable")},
        {"main_error": FDSNException("Service temporarily unavailable")},
    ],
    ids=["product-query-fails", "main-query-fails"],
)
def test_failed_rerun_leaves_previous_pair_untouched(
    failure: dict[str, Exception],
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
    comcat_quakeml: Path,
) -> None:
    """Every failure other than no-data propagates, and nothing half-written is left behind."""
    _use(monkeypatch, FakeClient(comcat_quakeml, with_product=set(), **failure))
    ctx = make_ctx(run_section)
    previous = _write_previous_pair(ctx)

    with pytest.raises(FDSNException, match="temporarily unavailable"):
        catalog_stage.run(ctx)

    assert {name: ctx.path(name).read_bytes() for name in previous} == previous
    assert _no_part_files(ctx.run_dir)
    assert ctx.records == []


def _nan_depth(text: str) -> str:
    return text.replace("<depth>\n     <value>2740</value>", "<depth>\n     <value>NaN</value>", 1)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda text: _first_event_without(text, "depth"), "no depth"),
        # With every column nullable, ObsPy's reader is what keeps a NaN out of the table.
        (_nan_depth, "not a finite floating point value"),
    ],
    ids=["no-depth", "nan-depth"],
)
def test_malformed_data_on_rerun_leaves_previous_pair_untouched(
    edit: Any,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
    comcat_quakeml: Path,
) -> None:
    text = comcat_quakeml.read_text("utf-8")
    broken = tmp_path / "broken.quakeml"
    broken.write_text(edit(text), "utf-8")
    assert broken.read_text("utf-8") != text
    _use(monkeypatch, FakeClient(broken, with_product=set()))
    ctx = make_ctx(run_section)
    previous = _write_previous_pair(ctx)

    with pytest.raises(ValueError, match=message):
        catalog_stage.run(ctx)

    assert {name: ctx.path(name).read_bytes() for name in previous} == previous
    assert _no_part_files(ctx.run_dir)


def test_no_event_with_product_counts_zero(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    fake = FakeClient(comcat_quakeml, with_product=None)
    params = catalog_stage.query_params(run_section, seismology_config.catalog)
    assert catalog_stage.ids_with_product(fake, params, "phase-data") == set()  # type: ignore[arg-type]


# ---------------------------------------------------------------- provider quirks


def _only_events(text: str, keep: set[str]) -> str:
    """The QuakeML text with every ``<event>`` block not in ``keep`` removed."""

    def block(match: re.Match[str]) -> str:
        tag = re.match(
            r'  <event [^>]*catalog:eventsource="(\w+)" catalog:eventid="(\w+)"', match[0]
        )
        assert tag is not None
        return match[0] if tag[1] + tag[2] in keep else ""

    return re.sub(r"  <event .*?</event>\n", block, text, flags=re.DOTALL)


class ComCatLikeClient(FakeClient):
    """Serves only what ComCat serves: it truncates ``starttime`` and ``endtime`` to the whole
    second and returns origins with ``trunc(starttime) <= time <= trunc(endtime)``.

    That behaviour was checked against ComCat's /count endpoint with the showcase bbox and
    uu80155911 (22:23:28.740Z): endtime 22:23:28.999 -> 0, endtime 22:23:29 -> 1,
    starttime 22:23:28.999 -> 1, starttime 22:23:29 -> 0.
    """

    def get_events(self, **kwargs: Any) -> Catalog | None:
        if "filename" not in kwargs:
            return super().get_events(**kwargs)
        self.calls.append(kwargs)
        start, end = (
            UTCDateTime(ns=kwargs[key].ns // NS_PER_S * NS_PER_S)
            for key in ("starttime", "endtime")
        )
        keep = {
            catalog_stage.comcat_id(ev)
            for ev in _served(self.quakeml)
            if start <= ev.preferred_origin().time <= end
        }
        Path(kwargs["filename"]).write_text(
            _only_events(self.quakeml.read_text("utf-8"), keep), "utf-8"
        )
        return None


def test_fractional_window_bounds_are_padded_to_what_comcat_serves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
    comcat_quakeml: Path,
) -> None:
    """An origin in [floor(windowEnd), windowEnd) is served and kept; exact bounds apply after."""
    t_middle = _utc("2026-09-10T22:23:28.740Z")
    # MIDDLE is 0.21 s before a fractional windowEnd in the same whole second.
    run = _with(
        run_section,
        windowStart=t_middle - timedelta(seconds=1),
        windowEnd=_utc("2026-09-10T22:23:28.950Z"),
    )
    fake = ComCatLikeClient(comcat_quakeml, with_product=set())
    # Control: sent unpadded, these bounds get nothing from ComCat.
    unpadded = tmp_path / "unpadded.quakeml"
    fake.get_events(
        filename=str(unpadded),
        starttime=UTCDateTime(run.windowStart),
        endtime=UTCDateTime(run.windowEnd),
    )
    assert len(_served(unpadded)) == 0
    fake.calls.clear()
    _use(monkeypatch, fake)

    ctx = make_ctx(run)
    catalog_stage.run(ctx)

    query = fake.calls[0]
    assert query["starttime"] == UTCDateTime("2026-09-10T22:23:27Z")
    assert query["endtime"] == UTCDateTime("2026-09-10T22:23:29Z")
    (record,) = ctx.records
    assert record["counts"]["served"] == 1 and record["counts"]["kept"] == 1
    assert list(pq.read_table(ctx.path("catalog.parquet")).column("id").to_pylist()) == [MIDDLE_ID]
    params = record["params"]["catalog"]
    assert params["query"]["endtime"] == str(UTCDateTime("2026-09-10T22:23:29Z"))
    assert params["windowEnd"] == run.windowEnd.isoformat()

    # Start side: MIDDLE is 0.16 s before a fractional windowStart in the same second. ComCat
    # serves it (it truncates starttime); the exact bound drops it and counts it.
    run = _with(run_section, windowStart=_utc("2026-09-10T22:23:28.900Z"))
    ctx = make_ctx(run)
    catalog_stage.run(ctx)
    counts = ctx.records[0]["counts"]
    assert counts["droppedBeforeWindowStart"] == 1
    assert MIDDLE_ID not in set(pq.read_table(ctx.path("catalog.parquet")).column("id").to_pylist())


def _with_event_type(text: str, event_id: str, event_type: str) -> str:
    """Replace the event-level ``<type>earthquake</type>`` of one event block."""
    start = text.rindex("  <event ", 0, text.index(f'catalog:eventid="{event_id[2:]}"'))
    end = text.index("</event>", start)
    block = text[start:end].replace("<type>earthquake</type>", f"<type>{event_type}</type>", 1)
    assert block != text[start:end]
    return text[:start] + block + text[end:]


def test_event_type_obspy_cannot_parse_is_counted_by_its_raw_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
    comcat_quakeml: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ComCat can serve a type outside QuakeML 1.2 (e.g. "Rock Slide"); ObsPy skips that event."""
    edited = tmp_path / "rock_slide.quakeml"
    edited.write_text(
        _with_event_type(comcat_quakeml.read_text("utf-8"), MIDDLE_ID, "Rock Slide"), "utf-8"
    )
    with pytest.warns(UserWarning, match="does not comply"):
        assert len(_served(edited)) == 2  # the reader drops it, with only a warning
    _use(monkeypatch, FakeClient(edited, with_product=set()))
    ctx = make_ctx(run_section)

    with pytest.warns(UserWarning), caplog.at_level(logging.INFO, logger=catalog_stage.__name__):
        catalog_stage.run(ctx)

    counts = ctx.records[0]["counts"]
    assert counts["served"] == 3 and counts["droppedEventType"] == 1 and counts["kept"] == 2
    assert any(
        MIDDLE_ID in r.getMessage() and "'Rock Slide'" in r.getMessage() for r in caplog.records
    )


def test_event_obspy_cannot_parse_fails_when_it_would_be_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
    seismology_config: SeismologyConfig,
    comcat_quakeml: Path,
) -> None:
    edited = tmp_path / "rock_slide.quakeml"
    edited.write_text(
        _with_event_type(comcat_quakeml.read_text("utf-8"), MIDDLE_ID, "Rock Slide"), "utf-8"
    )
    _use(monkeypatch, FakeClient(edited, with_product=set()))
    ctx = make_ctx(run_section, _with_catalog(seismology_config, eventTypes=None))
    previous = _write_previous_pair(ctx)

    with pytest.warns(UserWarning), pytest.raises(ValueError, match=f"event {MIDDLE_ID}"):
        catalog_stage.run(ctx)

    assert {name: ctx.path(name).read_bytes() for name in previous} == previous
    assert _no_part_files(ctx.run_dir)


# ---------------------------------------------------------------- reading the table back


def test_null_strings_read_back_as_nan_like_every_run_table(
    tmp_path: Path,
    run_section: RunSection,
    seismology_config: SeismologyConfig,
    comcat_quakeml: Path,
) -> None:
    """Pins how a null comes back: NaN through pandas (never pd.NA), None through pyarrow."""
    served = _served(comcat_quakeml)
    event = _by_id(served, MIDDLE_ID)
    event.magnitudes = []
    event.preferred_magnitude_id = None
    rows, _ = catalog_stage.build_catalog(served, run_section, seismology_config.catalog)
    path = tmp_path / "catalog.parquet"
    catalog_stage.write_catalog(rows, path)

    default_str = pd.Series(["x", None]).dtype  # pandas' default string dtype (missing = NaN)
    string_columns = [c for c, t in DOCS02_TYPES.items() if t == pa.string()]
    for frame in (rows, pd.read_parquet(path), pq.read_table(path).to_pandas()):
        for column in string_columns:
            dtype = frame[column].dtype
            assert isinstance(dtype, pd.StringDtype) and dtype.na_value is default_str.na_value
        middle = frame.set_index("id").loc[MIDDLE_ID]
        for value in (middle["magType"], middle["matchedEventId"]):
            assert pd.isna(value) and value is not pd.NA  # NaN: pd.isna, never `is None`
    assert pd.read_parquet(path)["magType"].dtype == default_str

    records = pq.read_table(path).to_pylist()
    assert all(r["matchedEventId"] is None for r in records)
    assert next(r for r in records if r["id"] == MIDDLE_ID)["magType"] is None

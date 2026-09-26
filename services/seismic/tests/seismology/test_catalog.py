"""hq.match.catalog: QuakeML -> CatalogEvent rows, window/bbox selection, the stage, fail-loud cases.

The fixture is a real ComCat response for the showcase query trimmed to 3 events, served in
ComCat's newest-first order. Offline: the FDSN client is replaced by a fake.
"""

import copy
import hashlib
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from obspy import UTCDateTime, read_events
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.event import Catalog
from pydantic import ValidationError

from hq.config.run import RunSection
from hq.config.seismology import SeismologyConfig
from hq.locate.coords import to_enu
from hq.match import catalog as catalog_stage

pytestmark = pytest.mark.smoke

# Facts of the three fixture events, as published (fixture data, not product numbers).
LATEST_ID = "uu80155936"  # 2026-09-10T23:42:53.300Z, ml 1.34, depth 2740 m
MIDDLE_ID = "uu80155911"  # 2026-09-10T22:23:28.740Z, md 1.19, lat 38.52
EARLIEST_ID = "uu80155571"  # 2026-09-10T09:27:56.880Z, ml 2.14, lat 38.4928, depth 4060 m
DOCS02_COLUMNS = [
    "id",
    "source",
    "t",
    "latitude",
    "longitude",
    "depthKm",
    "depthDatum",
    "elevM",
    "mag",
    "magType",
    "enu_e",
    "enu_n",
    "enu_u",
    "matchedEventId",
]


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(UTC)


def _with(run: RunSection, **update: Any) -> RunSection:
    return RunSection.model_validate({**run.model_dump(), **update})


def _with_datum(cfg: SeismologyConfig, **update: Any) -> SeismologyConfig:
    datum = cfg.catalog.datums["uu"].model_copy(update=update)
    catalog = cfg.catalog.model_copy(update={"datums": {"uu": datum}})
    return cfg.model_copy(update={"catalog": catalog})


def _served(path: Path) -> Catalog:
    return read_events(str(path), format="QUAKEML")


def _by_id(cat: Catalog, event_id: str) -> Any:
    return next(ev for ev in cat if catalog_stage.comcat_id(ev) == event_id)


# ---------------------------------------------------------------- config and query


def test_seismology_yaml_parses_and_rejects_unknown_keys(
    seismology_config: SeismologyConfig,
) -> None:
    cat = seismology_config.catalog
    assert cat.minMagnitude is None and cat.maxMagnitude is None
    assert "uu" in cat.datums
    raw = seismology_config.model_dump(mode="json")
    with pytest.raises(ValidationError):
        SeismologyConfig.model_validate({**raw, "unknown": 1})
    with pytest.raises(ValidationError):
        SeismologyConfig.model_validate({"catalog": {**raw["catalog"], "unknown": 1}})


def test_query_uses_run_window_and_bbox_exactly(
    run_section: RunSection, seismology_config: SeismologyConfig
) -> None:
    params = catalog_stage.query_params(run_section, seismology_config.catalog)
    assert params == {
        "starttime": UTCDateTime(run_section.windowStart),
        "endtime": UTCDateTime(run_section.windowEnd),
        "minlongitude": run_section.bbox[0],
        "minlatitude": run_section.bbox[1],
        "maxlongitude": run_section.bbox[2],
        "maxlatitude": run_section.bbox[3],
    }
    limited = seismology_config.catalog.model_copy(update={"minMagnitude": 0.5, "maxMagnitude": 3})
    params = catalog_stage.query_params(run_section, limited)
    assert params["minmagnitude"] == 0.5 and params["maxmagnitude"] == 3


# ---------------------------------------------------------------- QuakeML -> rows


def test_fixture_rows_every_field(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    rows, counts = catalog_stage.build_catalog(
        _served(comcat_quakeml), run_section, seismology_config.catalog
    )
    assert list(rows.columns) == DOCS02_COLUMNS
    assert counts["served"] == 3 and counts["kept"] == 3
    assert rows["t"].dtype == np.float64

    row = rows.set_index("id").loc[LATEST_ID]
    assert row["source"] == "UU via USGS ComCat"
    assert row["t"] == _utc("2026-09-10T23:42:53.300Z").timestamp()
    assert row["latitude"] == 38.512166666667
    assert row["longitude"] == -112.9005
    assert row["depthKm"] == 2.74
    assert row["depthDatum"].startswith("km below sea level.")
    assert "https://quake.utah.edu/" in row["depthDatum"]
    assert row["elevM"] == -2740.0  # datum surface 0 m ASL minus 2740 m
    assert row["mag"] == 1.34 and row["magType"] == "ml"
    e, n, _ = to_enu(row["latitude"], row["longitude"], row["elevM"], run_section.origin)
    assert (row["enu_e"], row["enu_n"]) == (float(e), float(n))
    assert row["enu_u"] == pytest.approx(-2740.0 - run_section.origin.elevM, abs=1e-9)
    assert row["matchedEventId"] is None

    middle = rows.set_index("id").loc[MIDDLE_ID]
    assert middle["magType"] == "md" and middle["mag"] == 1.19


def test_elev_uses_the_configured_datum_surface(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    cfg = _with_datum(seismology_config, surfaceElevM=1500.0)
    rows, _ = catalog_stage.build_catalog(_served(comcat_quakeml), run_section, cfg.catalog)
    row = rows.set_index("id").loc[EARLIEST_ID]
    assert row["depthKm"] == 4.06  # as published, unchanged by the datum
    assert row["elevM"] == 1500.0 - 4060.0
    assert row["enu_u"] == pytest.approx(1500.0 - 4060.0 - run_section.origin.elevM, abs=1e-9)


def test_sorted_by_t(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    assert [catalog_stage.comcat_id(ev) for ev in served] == [LATEST_ID, MIDDLE_ID, EARLIEST_ID]
    rows, _ = catalog_stage.build_catalog(served, run_section, seismology_config.catalog)
    assert list(rows["id"]) == [EARLIEST_ID, MIDDLE_ID, LATEST_ID]
    assert np.all(np.diff(rows["t"].to_numpy()) > 0)


def test_window_end_is_exclusive(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    t_latest = _utc("2026-09-10T23:42:53.300Z")
    at_end = _with(run_section, windowEnd=t_latest)
    rows, counts = catalog_stage.build_catalog(
        _served(comcat_quakeml), at_end, seismology_config.catalog
    )
    assert at_end.window_end_s == t_latest.timestamp()
    assert LATEST_ID not in set(rows["id"])
    assert counts["droppedAtOrAfterWindowEnd"] == 1 and counts["kept"] == 2

    just_after = _with(run_section, windowEnd=t_latest + timedelta(microseconds=1))
    rows, counts = catalog_stage.build_catalog(
        _served(comcat_quakeml), just_after, seismology_config.catalog
    )
    assert LATEST_ID in set(rows["id"]) and counts["droppedAtOrAfterWindowEnd"] == 0

    starts_at_earliest = _with(run_section, windowStart=_utc("2026-09-10T09:27:56.880Z"))
    rows, counts = catalog_stage.build_catalog(
        _served(comcat_quakeml), starts_at_earliest, seismology_config.catalog
    )
    assert EARLIEST_ID in set(rows["id"]) and counts["droppedBeforeWindowStart"] == 0


def test_bbox_is_respected_edges_inclusive(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    min_lon, _, max_lon, _ = run_section.bbox
    # minLat 38.50 excludes the event at 38.4928; maxLat 38.52 sits exactly on the middle event.
    box = _with(run_section, bbox=(min_lon, 38.50, max_lon, 38.52))
    rows, counts = catalog_stage.build_catalog(
        _served(comcat_quakeml), box, seismology_config.catalog
    )
    assert list(rows["id"]) == [MIDDLE_ID, LATEST_ID]
    assert counts["droppedOutsideBbox"] == 1


def test_event_type_filter_drops_and_counts(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    served = _served(comcat_quakeml)
    _by_id(served, MIDDLE_ID).event_type = "quarry blast"
    rows, counts = catalog_stage.build_catalog(served, run_section, seismology_config.catalog)
    assert MIDDLE_ID not in set(rows["id"])
    assert counts["droppedEventType"] == 1 and counts["kept"] == 2


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
    catalog_stage._write_table(rows, path)
    table = pq.read_table(path).to_pylist()
    row = next(r for r in table if r["id"] == MIDDLE_ID)
    assert row["mag"] is None and row["magType"] is None
    assert all(r["mag"] is not None and r["magType"] for r in table if r["id"] != MIDDLE_ID)


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
    """Stands in for obspy's FDSN Client: serves the fixture, records every query."""

    base_url = "https://fdsn.invalid"

    def __init__(self, quakeml: Path, with_product: set[str] | None) -> None:
        self.quakeml = quakeml
        self.with_product = with_product
        self.calls: list[dict[str, Any]] = []

    def get_events(self, **kwargs: Any) -> Catalog | None:
        self.calls.append(kwargs)
        if "filename" in kwargs:
            Path(kwargs["filename"]).write_bytes(self.quakeml.read_bytes())
            return None
        if self.with_product is None:
            raise FDSNNoDataException("No data available for request.")
        served = _served(self.quakeml)
        return Catalog([ev for ev in served if catalog_stage.comcat_id(ev) in self.with_product])


def test_stage_writes_both_files_and_records(
    monkeypatch: pytest.MonkeyPatch,
    make_ctx: Any,
    run_section: RunSection,
    comcat_quakeml: Path,
) -> None:
    fake = FakeClient(comcat_quakeml, with_product={LATEST_ID, EARLIEST_ID})
    monkeypatch.setattr(catalog_stage, "_client", lambda cfg: fake)
    ctx = make_ctx(run_section)

    catalog_stage.run(ctx)

    raw = ctx.path("catalog.quakeml").read_bytes()
    assert raw == comcat_quakeml.read_bytes()
    table = pq.read_table(ctx.path("catalog.parquet"))
    assert table.schema.metadata[b"schemaVersion"] == b"1.0"
    assert table.schema.metadata[b"model"] == b"CatalogEvent"
    assert table.column_names == DOCS02_COLUMNS
    df = table.to_pandas()
    assert list(df["id"]) == [EARLIEST_ID, MIDDLE_ID, LATEST_ID]
    assert df["matchedEventId"].isna().all()
    assert str(table.schema.field("t").type) == "double"

    query, product_query = fake.calls
    assert query["starttime"] == UTCDateTime(run_section.windowStart)
    assert query["endtime"] == UTCDateTime(run_section.windowEnd)
    assert product_query["producttype"] == "phase-data"
    assert "filename" not in product_query

    (record,) = ctx.records
    assert record["stage"] == "catalog" and record["runtime_s"] >= 0
    assert record["counts"] == {
        "served": 3,
        "droppedEventType": 0,
        "droppedBeforeWindowStart": 0,
        "droppedAtOrAfterWindowEnd": 0,
        "droppedOutsideBbox": 0,
        "kept": 3,
        "withArrivalsProduct": 2,
    }
    params = record["params"]
    assert params["quakemlSha256"] == hashlib.sha256(raw).hexdigest()
    assert params["query"]["minlongitude"] == str(run_section.bbox[0])
    assert params["windowEndExclusive"] is True


def test_no_event_with_product_counts_zero(
    run_section: RunSection, seismology_config: SeismologyConfig, comcat_quakeml: Path
) -> None:
    fake = FakeClient(comcat_quakeml, with_product=None)
    params = catalog_stage.query_params(run_section, seismology_config.catalog)
    assert catalog_stage.ids_with_product(fake, params, "phase-data") == set()  # type: ignore[arg-type]

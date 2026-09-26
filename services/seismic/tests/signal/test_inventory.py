"""SEIS-01 station inventory, offline.

Fixtures under ``fixtures/inventory/`` are small recordings of public services for the showcase
window and bbox: ``channels.xml`` (EarthScope fdsnws-station, channel level, a subset of stations,
responses stripped), ``dem.json`` (USGS 3DEP EPQS values at the chosen sensors) and
``mustang_percent_availability.json`` (EarthScope MUSTANG responses per chosen triplet).
"""

import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from obspy import Inventory, read_inventory
from pydantic import ValidationError

from hq.config.run import RunSection
from hq.config.signal import ChannelRules, SignalConfig, StationSelection
from hq.ingest import inventory as inv
from hq.ingest.inventory import (
    AmbiguousElevationError,
    AvailabilityError,
    Chosen,
    DemError,
    EnuProjector,
    HttpResult,
    InventoryResult,
    Triplet,
    build_inventory,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "inventory"
CHANNELS_XML = FIXTURES / "channels.xml"
DEM = json.loads((FIXTURES / "dem.json").read_text(encoding="utf-8"))
MUSTANG = json.loads((FIXTURES / "mustang_percent_availability.json").read_text(encoding="utf-8"))

EXPECTED_IDS = [
    "6K.CS01",
    "6K.CS03",
    "NP.7225",
    "UU.FOR2",
    "UU.FOR6",
    "UU.FORK",
    "UU.FSB1",
    "UU.FSB4",
]
STATION_FIELDS = {  # docs/02 -> Station
    "id",
    "network",
    "station",
    "location",
    "latitude",
    "longitude",
    "surfaceElevM",
    "sensorDepthM",
    "sensorElevM",
    "kind",
    "channels",
    "sampleRateHz",
    "enu",
    "preprocessProfile",
    "usedInRun",
    "staticsS",
}


class FixtureClient:
    """Stands in for obspy's FDSN client: serves the recorded channel XML and response subsets."""

    def __init__(self, xml: Path = CHANNELS_XML) -> None:
        self.xml = xml
        self.inv = read_inventory(str(xml))
        self.levels: list[str] = []

    def get_stations(self, *, filename: str, level: str, **kw: Any) -> None:
        self.levels.append(level)
        if level == "channel":
            shutil.copy(self.xml, filename)
            return
        loc = "" if kw["location"] == "--" else kw["location"]
        sub = Inventory(networks=[], source="test")
        for cha in kw["channel"].split(","):
            sub += self.inv.select(
                network=kw["network"], station=kw["station"], location=loc, channel=cha
            )
        sub.write(filename, format="STATIONXML")


class NoNetworkClient:
    def get_stations(self, **kw: Any) -> None:
        raise AssertionError(f"station service called on a cache hit: {kw}")


def fixture_dem(lat: float, lon: float) -> float:
    return DEM[f"{lat:.5f},{lon:.5f}"]


def no_dem(lat: float, lon: float) -> float:
    raise AssertionError(f"DEM called on a cache hit: {lat},{lon}")


def mustang_http(url: str, params: Any, timeout_s: float) -> HttpResult:
    rec = MUSTANG[f"{params['net']}.{params['sta']}.{params['loc']}"]
    return HttpResult(status=rec["status"], text=rec["text"])


def build(run: RunSection, cfg: StationSelection, cache: Path, **kw: Any) -> InventoryResult:
    kw.setdefault("client", FixtureClient())
    kw.setdefault("dem", fixture_dem)
    kw.setdefault("http_get", mustang_http)
    return build_inventory(run, cfg, cache, **kw)


@pytest.fixture()
def cfg(signal_cfg: SignalConfig) -> StationSelection:
    return signal_cfg.stations


@pytest.fixture()
def result(run_section: RunSection, cfg: StationSelection, tmp_path: Path) -> InventoryResult:
    return build(run_section, cfg, tmp_path / "cache")


def by_id(res: InventoryResult) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in res.rows}


def detail(res: InventoryResult, sid: str) -> dict[str, Any]:
    return next(d for d in res.report["stations"] if d["id"] == sid)


# --- selection ------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_selected_set_and_skip_reasons(result: InventoryResult) -> None:
    assert [r["id"] for r in result.rows] == EXPECTED_IDS
    skipped = {s["site"]: s["reason"] for s in result.report["skippedSites"]}
    assert set(skipped) == {"2J.FS01", "UU.IMU"}
    assert "no velocity or accelerometer codes" in skipped["2J.FS01"]
    assert "incomplete triplet" in skipped["UU.IMU"]
    assert result.report["excludedNetworks"] == {"SY": ["R14A"]}
    assert not any(r["network"] in {"2J", "SY"} for r in result.rows)
    dropped = {d["triplet"]: d["reason"] for d in result.report["droppedTriplets"]}
    assert "lower priority than UU.FORK.01.GH?" in dropped["UU.FORK.01.EH?"]
    assert "site has a velocity triplet" in dropped["UU.FORK.01.GN?"]
    assert "site has a velocity triplet" in dropped["UU.FSB1.01.DN?"]
    assert "same kind (strong_motion)" in dropped["NP.7225.10.HN?"]


@pytest.mark.smoke
def test_rows_have_exactly_the_contract_fields(result: InventoryResult) -> None:
    for row in result.rows:
        assert set(row) == STATION_FIELDS
        assert set(row["enu"]) == {"e", "n", "u"}
        assert row["staticsS"] == {}
        assert row["sensorElevM"] == pytest.approx(row["surfaceElevM"] - row["sensorDepthM"])


@pytest.mark.smoke
def test_fork_borehole_geophone_with_sensor_level_station_elevation(
    result: InventoryResult,
) -> None:
    fork = by_id(result)["UU.FORK"]
    assert fork["channels"] == ["GHZ", "GH1", "GH2"]
    assert fork["kind"] == "borehole"
    assert fork["sampleRateHz"] == 1000.0
    assert fork["sensorDepthM"] == 281.0
    # StationXML station elevation 1408 is already the sensor; the naive rule would give ~1127.
    assert fork["sensorElevM"] == pytest.approx(1408.0)
    assert fork["surfaceElevM"] == pytest.approx(1689.0)
    assert fork["preprocessProfile"] == "borehole-A"
    elev = detail(result, "UU.FORK")["elevation"]
    assert (elev["convention"], elev["basis"]) == ("sensor", "dem")


@pytest.mark.smoke
def test_fsb4_sensor_convention_and_fsb1_surface_convention(result: InventoryResult) -> None:
    rows = by_id(result)
    fsb4 = rows["UU.FSB4"]
    assert detail(result, "UU.FSB4")["elevation"]["convention"] == "sensor"
    assert (fsb4["sensorDepthM"], fsb4["sensorElevM"], fsb4["surfaceElevM"]) == (
        41.0,
        1578.0,
        1619.0,
    )
    fsb1 = rows["UU.FSB1"]
    assert fsb1["channels"] == ["HHZ", "HH1", "HH2"]
    assert fsb1["sensorDepthM"] == 29.0
    assert fsb1["kind"] == "borehole"
    assert fsb1["preprocessProfile"] == "surface-hi"
    assert detail(result, "UU.FSB1")["elevation"]["convention"] == "surface"
    assert fsb1["sensorElevM"] == pytest.approx(1697.0 - 29.0)


@pytest.mark.smoke
def test_kinds_profiles_and_channel_order(result: InventoryResult) -> None:
    rows = by_id(result)
    assert rows["6K.CS01"]["kind"] == "borehole"  # GN accelerometer at 40 m
    assert rows["6K.CS01"]["sensorDepthM"] == 40.0
    assert rows["6K.CS01"]["channels"] == ["GNZ", "GN1", "GN2"]
    assert rows["6K.CS01"]["location"] == ""
    assert rows["6K.CS03"]["kind"] == "strong_motion"  # GN at 1 m
    assert rows["NP.7225"]["kind"] == "strong_motion"
    assert rows["NP.7225"]["channels"] == ["HNZ", "HNN", "HNE"]
    assert rows["NP.7225"]["location"] == "2C"  # 200 Hz beats 100 Hz at equal priority
    assert rows["UU.FOR2"]["kind"] == "surface"
    assert rows["UU.FOR2"]["channels"] == ["HHZ", "HHN", "HHE"]
    assert rows["UU.FOR2"]["preprocessProfile"] == "surface-hi"
    for row in result.rows:
        assert row["channels"][0].endswith("Z")
        if row["kind"] == "borehole":
            assert row["sensorDepthM"] > 0


@pytest.mark.smoke
def test_coverage_marks_stations_without_data_unused(result: InventoryResult) -> None:
    rows = by_id(result)
    assert rows["NP.7225"]["usedInRun"] is False
    assert detail(result, "NP.7225")["coverage"] == 0.0
    assert all(rows[i]["usedInRun"] for i in EXPECTED_IDS if i != "NP.7225")
    assert result.counts["withData"] == len(EXPECTED_IDS) - 1


@pytest.mark.smoke
def test_shallow_station_disagreeing_with_dem_is_flagged(result: InventoryResult) -> None:
    elev = detail(result, "UU.FOR6")["elevation"]
    assert (elev["convention"], elev["basis"]) == ("surface", "shallow")
    assert by_id(result)["UU.FOR6"]["surfaceElevM"] == 2421.0  # StationXML value kept
    assert any(f["station"] == "UU.FOR6" for f in result.report["flags"])


@pytest.mark.smoke
def test_enu_of_sensor(result: InventoryResult, run_section: RunSection) -> None:
    fork = by_id(result)["UU.FORK"]
    assert fork["enu"]["u"] == pytest.approx(1408.0 - run_section.origin.elevM)


# --- elevation rules ----------------------------------------------------------------------------


@pytest.mark.smoke
def test_resolve_elevation_rules() -> None:
    kw: dict[str, Any] = {"tol_m": 15.0, "on_ambiguous": "error", "what": "t"}
    d = inv.resolve_elevation(1634.7, 1634.7, 290.0, 1634.7, **kw)
    assert d is not None and (d.convention, d.sensor_elev_m) == ("surface", pytest.approx(1344.7))
    d = inv.resolve_elevation(1474.0, 1515.0, 41.0, 1512.6, **kw)
    assert d is not None and (d.convention, d.surface_elev_m) == ("sensor", 1515.0)
    d = inv.resolve_elevation(1840.1, 1840.1, 1.0, 1840.2, **kw)
    assert d is not None and (d.convention, d.basis) == ("surface", "both-match")
    with pytest.raises(AmbiguousElevationError):
        inv.resolve_elevation(1408.0, 1409.0, 281.0, 1500.0, **kw)
    kw["on_ambiguous"] = "skip"
    assert inv.resolve_elevation(1408.0, 1409.0, 281.0, 1500.0, **kw) is None
    kw["on_ambiguous"] = "surface"
    d = inv.resolve_elevation(1408.0, 1409.0, 281.0, 1500.0, **kw)
    assert d is not None and (d.basis, d.sensor_elev_m) == ("assumed", 1127.0)


@pytest.mark.smoke
def test_ambiguous_elevation_raises_by_default(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    assert cfg.elevation.onAmbiguous == "error"

    def shifted(lat: float, lon: float) -> float:
        value = fixture_dem(lat, lon)
        return value + 100.0 if (lat, lon) == (38.501227, -112.886489) else value

    with pytest.raises(AmbiguousElevationError, match="UU.FORK"):
        build(run_section, cfg, tmp_path / "cache", dem=shifted)


@pytest.mark.smoke
def test_borehole_looking_code_at_depth_zero_is_flagged(
    run_section: RunSection,
    cfg: StationSelection,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    edited = read_inventory(str(CHANNELS_XML))
    for ch in edited.select(station="FORK", channel="GH?")[0][0]:
        ch.depth = 0.0
    xml = tmp_path / "edited.xml"
    edited.write(str(xml), format="STATIONXML")
    caplog.set_level(logging.WARNING, logger=inv.__name__)
    res = build(run_section, cfg, tmp_path / "cache", client=FixtureClient(xml))
    fork = by_id(res)["UU.FORK"]
    assert (fork["channels"][0], fork["sensorDepthM"], fork["kind"]) == ("GHZ", 0.0, "surface")
    assert "looks like a borehole geophone" in caplog.text
    assert any("looks like a borehole geophone" in f["flag"] for f in res.report["flags"])


# --- caches and determinism ---------------------------------------------------------------------


@pytest.mark.smoke
def test_rerun_uses_caches_without_network(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    client = FixtureClient()
    first = build(run_section, cfg, cache, client=client)
    assert client.levels.count("channel") == 1
    assert client.levels.count("response") == len(EXPECTED_IDS)
    xml_dir = cache / cfg.query.cacheSubdir
    assert (xml_dir / cfg.elevation.demCacheFile).exists()
    for sid in EXPECTED_IDS:
        assert (xml_dir / f"{sid}.xml").exists()
    second = build(run_section, cfg, cache, client=NoNetworkClient(), dem=no_dem)
    assert second.rows == first.rows
    assert second.report == first.report


@pytest.mark.smoke
def test_response_cache_with_other_channels_is_refetched(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    build(run_section, cfg, cache)
    stale = read_inventory(str(CHANNELS_XML)).select(station="FORK", channel="EH?")
    stale.write(str(cache / cfg.query.cacheSubdir / "UU.FORK.xml"), format="STATIONXML")
    client = FixtureClient()
    build(run_section, cfg, cache, client=client)
    assert client.levels == ["response"]


# --- ENU, profiles, availability, DEM -----------------------------------------------------------


@pytest.mark.smoke
def test_enu_axes(run_section: RunSection) -> None:
    o = run_section.origin
    project = EnuProjector(o)
    at = project(o.lat, o.lon, o.elevM)
    assert at == pytest.approx({"e": 0.0, "n": 0.0, "u": 0.0}, abs=1e-6)
    east = project(o.lat, o.lon + 0.01, o.elevM)
    assert 800 < east["e"] < 950 and abs(east["n"]) < 50
    north = project(o.lat + 0.01, o.lon, o.elevM)
    assert 1050 < north["n"] < 1150 and abs(north["e"]) < 50
    assert project(o.lat, o.lon, o.elevM + 100.0)["u"] == pytest.approx(100.0)


@pytest.mark.smoke
def test_profile_rules(cfg: StationSelection) -> None:
    assert inv.match_profile(100.0, cfg) == "surface-100"
    assert inv.match_profile(200.0, cfg) == "surface-hi"
    assert inv.match_profile(250.0, cfg) == "surface-hi"
    assert inv.match_profile(1000.0, cfg) == "borehole-A"
    assert inv.match_profile(40.0, cfg) is None


def _triplet() -> Triplet:
    return Triplet(
        network="XX",
        station="TST",
        location="00",
        code="HH",
        family="velocity",
        rank=0,
        channels=("HHZ", "HHN", "HHE"),
        sample_rate_hz=100.0,
        depth_m=0.0,
        latitude=38.5,
        longitude=-112.9,
        channel_elev_m=1600.0,
        station_elev_m=1600.0,
        station_latitude=38.5,
        station_longitude=-112.9,
    )


def _row(cha: str, day: str, value: float, qual: str = "M") -> dict[str, Any]:
    return {"value": value, "loc": "00", "cha": cha, "qual": qual, "start": f"{day}T00:00:00"}


@pytest.mark.smoke
def test_partial_day_coverage_is_overlap_weighted(
    run_section: RunSection, cfg: StationSelection
) -> None:
    run = run_section.model_copy(
        update={
            "windowStart": datetime(2026, 9, 10, 12, tzinfo=UTC),
            "windowEnd": datetime(2026, 9, 11, 6, tzinfo=UTC),
        }
    )
    rows = [_row(c, "2026-09-10", 100.0) for c in ("HHZ", "HHN", "HHE")]
    rows += [_row(c, "2026-09-11", 50.0) for c in ("HHZ", "HHN", "HHE")]
    rows.append(_row("HHZ", "2026-09-11", 0.0, qual="D"))  # other quality code: best one wins
    seen: dict[str, Any] = {}

    def http(url: str, params: Any, timeout_s: float) -> HttpResult:
        seen.update(params)
        body = {"measurements": {cfg.availability.metric: rows}}
        return HttpResult(200, json.dumps(body))

    chosen = [Chosen(id="XX.TST", triplet=_triplet(), kind="surface")]
    cov = inv.fetch_coverage(chosen, run, cfg, http)
    assert seen["timewindow"] == "2026-09-10T00:00:00,2026-09-12T00:00:00"
    assert seen["cha"] == "HHZ,HHN,HHE"
    assert cov["XX.TST"]["HHZ"] == pytest.approx(12 / 18 + 0.5 * 6 / 18)
    assert inv.station_coverage(cov["XX.TST"]) == pytest.approx(12 / 18 + 0.5 * 6 / 18)


@pytest.mark.smoke
def test_missing_measurement_and_service_failure(
    run_section: RunSection, cfg: StationSelection
) -> None:
    chosen = [Chosen(id="XX.TST", triplet=_triplet(), kind="surface")]

    def no_content(url: str, params: Any, timeout_s: float) -> HttpResult:
        return HttpResult(204, "")

    as_error = cfg.model_copy(
        update={"availability": cfg.availability.model_copy(update={"onMissing": "error"})}
    )
    with pytest.raises(AvailabilityError, match="no percent_availability measurement"):
        inv.fetch_coverage(chosen, run_section, as_error, no_content)
    as_unused = cfg.model_copy(
        update={"availability": cfg.availability.model_copy(update={"onMissing": "unused"})}
    )
    cov = inv.fetch_coverage(chosen, run_section, as_unused, no_content)
    assert inv.station_coverage(cov["XX.TST"]) is None

    def down(url: str, params: Any, timeout_s: float) -> HttpResult:
        return HttpResult(503, "Service Unavailable")

    with pytest.raises(AvailabilityError, match="HTTP 503"):
        inv.fetch_coverage(chosen, run_section, cfg, down)


@pytest.mark.smoke
def test_epqs_dem_retries_then_parses_and_rejects_nodata(cfg: StationSelection) -> None:
    fast = cfg.model_copy(
        update={"elevation": cfg.elevation.model_copy(update={"demBackoffS": 0.0})}
    )
    replies = [HttpResult(500, "busy"), HttpResult(200, '{"value": "1686.542602539"}')]

    def http(url: str, params: Any, timeout_s: float) -> HttpResult:
        assert params["x"] == "-112.886489" and params["y"] == "38.501227"
        return replies.pop(0)

    assert inv.epqs_dem(fast, http)(38.501227, -112.886489) == pytest.approx(1686.542602539)

    def nodata(url: str, params: Any, timeout_s: float) -> HttpResult:
        return HttpResult(200, json.dumps({"value": fast.elevation.demNoDataValue}))

    with pytest.raises(DemError, match="no data"):
        inv.epqs_dem(fast, nodata)(38.5, -112.9)


# --- config -------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_station_selection_rejects_unknown_and_overlapping_codes(raw_signal_yaml: dict) -> None:
    raw = json.loads(json.dumps(raw_signal_yaml["stations"]))
    raw["notAKnob"] = 1
    with pytest.raises(ValidationError):
        StationSelection.model_validate(raw)
    rules = dict(raw_signal_yaml["stations"]["channels"])
    rules["accelerometerCodes"] = [*rules["accelerometerCodes"], "HH"]
    with pytest.raises(ValidationError, match="overlap"):
        ChannelRules.model_validate(rules)


# --- stage --------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_format_table_lists_every_station(result: InventoryResult) -> None:
    table = inv.format_table(result)
    for sid in EXPECTED_IDS:
        assert sid in table
    assert "with any data in window" in table and "2J.FS01" in table


@pytest.mark.smoke
def test_stage_writes_stations_parquet(
    fake_ctx: Any, cfg: StationSelection, monkeypatch: pytest.MonkeyPatch
) -> None:
    io = pytest.importorskip("hq_contracts.io")  # CONTRACT-01 (H4)
    build(fake_ctx.config.run, cfg, fake_ctx.cache_dir)  # seeds XML, DEM and response caches
    monkeypatch.setattr(inv, "requests_get", mustang_http)
    inv.run(fake_ctx)
    df = io.read_table(fake_ctx.path(inv.STATIONS_FILE))
    assert sorted(df["id"]) == EXPECTED_IDS
    assert {"enu_e", "enu_n", "enu_u", "sensorDepthM", "sensorElevM"} <= set(df.columns)
    report = json.loads(fake_ctx.path(inv.REPORT_FILE).read_text(encoding="utf-8"))
    assert report["counts"]["selected"] == len(EXPECTED_IDS)
    rec = fake_ctx.records[inv.STAGE]
    assert rec["counts"]["selected"] == len(EXPECTED_IDS)
    assert rec["params"] == cfg.model_dump(mode="json")


@pytest.mark.smoke
def test_missing_contracts_package_fails_loudly(
    tmp_path: Path, result: InventoryResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "hq_contracts.io", None)
    with pytest.raises(inv.InventoryError, match="CONTRACT-01"):
        inv.write_stations_parquet(result.rows, tmp_path / inv.STATIONS_FILE)

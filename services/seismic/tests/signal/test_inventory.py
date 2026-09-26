"""SEIS-01 station inventory, offline.

Fixtures under ``fixtures/inventory/`` are small recordings of public services for the showcase
window and bbox: ``channels.xml`` (EarthScope fdsnws-station, channel level, a subset of stations,
responses stripped), ``dem.json`` (USGS 3DEP EPQS values at the chosen sensors),
``mustang_percent_availability.json`` (EarthScope MUSTANG replies per chosen triplet) and
``run.yaml`` (the run section they were recorded for, pinned). Instrument responses served by the
fake client and the few edited inventories below are built inside the tests.
"""

import copy
import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import requests
import yaml
from obspy import Inventory, UTCDateTime, read_inventory
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException
from obspy.core.inventory.response import Response
from pydantic import ValidationError

from hq.config.run import RunSection
from hq.config.signal import ChannelRules, ElevationCheck, SignalConfig, StationSelection
from hq.ingest import inventory as inv
from hq.ingest.inventory import (
    AmbiguousElevationError,
    AvailabilityError,
    Chosen,
    DemError,
    EnuProjector,
    HttpResult,
    InventoryError,
    InventoryResult,
    JsonCache,
    SiteCandidates,
    Triplet,
    build_inventory,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "inventory"
CHANNELS_XML = FIXTURES / "channels.xml"
DEM = json.loads((FIXTURES / "dem.json").read_text(encoding="utf-8"))
MUSTANG = json.loads((FIXTURES / "mustang_percent_availability.json").read_text(encoding="utf-8"))
FOR6_DEM = DEM["38.48982,-112.78749"]

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


@pytest.fixture(scope="module")
def run_section() -> RunSection:
    """Overrides the conftest fixture for this module: the recorded fixtures' own window."""
    with (FIXTURES / "run.yaml").open(encoding="utf-8") as fh:
        return RunSection.model_validate(yaml.safe_load(fh))


def fake_response() -> Response:
    return Response.from_paz(
        zeros=[], poles=[], stage_gain=1.0, input_units="M/S", output_units="COUNTS"
    )


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
        for net in sub:
            for sta in net:
                for ch in sta:
                    ch.response = fake_response()
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


def no_http(url: str, params: Any, timeout_s: float) -> HttpResult:
    raise AssertionError(f"HTTP called on a cache hit: {url} {params}")


_FIXTURE_INV = read_inventory(str(CHANNELS_XML))


def metadata_rate_probe(
    net: str, sta: str, loc: str, channels: Any, t0: float, t1: float
) -> dict[str, float] | None:
    """Serves each channel at its StationXML rate: the data agree with the metadata."""
    out: dict[str, float] = {}
    for cha in channels:
        sel = _FIXTURE_INV.select(
            network=net, station=sta, location=loc, channel=cha, time=UTCDateTime(t0)
        )
        for n in sel:
            for st in n:
                for ch in st:
                    out[ch.code] = float(ch.sample_rate)
    return out or None


def assert_depths_match_stationxml(rows: list[dict[str, Any]], t: float) -> None:
    """Lane DoD: each row's sensorDepthM is the StationXML depth of every one of its channels."""
    assert rows
    for row in rows:
        for code in row["channels"]:
            sel = _FIXTURE_INV.select(
                network=row["network"],
                station=row["station"],
                location=row["location"],
                channel=code,
                time=UTCDateTime(t),
            )
            depths = {float(ch.depth) for net in sel for sta in net for ch in sta}
            assert depths == {row["sensorDepthM"]}, (row["id"], code, depths)


def no_probe(net: str, sta: str, loc: str, channels: Any, t0: float, t1: float) -> None:
    raise AssertionError(f"rate probe called on a cache hit: {net}.{sta}.{loc}")


def build(run: RunSection, cfg: StationSelection, cache: Path, **kw: Any) -> InventoryResult:
    kw.setdefault("client", FixtureClient())
    kw.setdefault("dem", fixture_dem)
    kw.setdefault("http_get", mustang_http)
    kw.setdefault("rate_probe", metadata_rate_probe)
    return build_inventory(run, cfg, cache, **kw)


def with_elevation(cfg: StationSelection, **update: Any) -> StationSelection:
    return cfg.model_copy(update={"elevation": cfg.elevation.model_copy(update=update)})


def with_availability(cfg: StationSelection, **update: Any) -> StationSelection:
    return cfg.model_copy(update={"availability": cfg.availability.model_copy(update=update)})


@pytest.fixture()
def cfg(signal_cfg: SignalConfig) -> StationSelection:
    return signal_cfg.stations


@pytest.fixture()
def fast(cfg: StationSelection) -> StationSelection:
    """The showcase config with every retry backoff set to zero."""
    return cfg.model_copy(
        update={
            "query": cfg.query.model_copy(update={"backoffS": 0.0}),
            "elevation": cfg.elevation.model_copy(update={"demBackoffS": 0.0}),
            "availability": cfg.availability.model_copy(update={"backoffS": 0.0}),
        }
    )


@pytest.fixture()
def result(run_section: RunSection, cfg: StationSelection, tmp_path: Path) -> InventoryResult:
    return build(run_section, cfg, tmp_path / "cache")


def by_id(res: InventoryResult) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in res.rows}


def detail(res: InventoryResult, sid: str) -> dict[str, Any]:
    return next(d for d in res.report["stations"] if d["id"] == sid)


def flags_of(res: InventoryResult, sid: str) -> list[str]:
    return [f["flag"] for f in res.report["flags"] if f["station"] == sid]


def triplet(
    station: str,
    location: str,
    code: str,
    *,
    family: str = "velocity",
    rank: int = 0,
    rate: float = 100.0,
    depth: float = 0.0,
) -> Triplet:
    return Triplet(
        network="XX",
        station=station,
        location=location,
        code=code,
        family=family,  # type: ignore[arg-type]
        rank=rank,
        channels=(f"{code}Z", f"{code}N", f"{code}E"),
        sample_rate_hz=rate,
        depth_m=depth,
        latitude=38.5,
        longitude=-112.9,
        channel_elev_m=1600.0,
        station_elev_m=1600.0,
        station_latitude=38.5,
        station_longitude=-112.9,
    )


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
    assert flags_of(result, "UU.FORK") == []


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
def test_every_row_depth_matches_its_stationxml_channels(
    result: InventoryResult, run_section: RunSection
) -> None:
    assert_depths_match_stationxml(result.rows, run_section.window_start_s)


@pytest.mark.smoke
def test_two_locations_at_one_site_get_raw_location_suffix(cfg: StationSelection) -> None:
    sites = {
        # a surface HH at the empty location and a borehole GH at 01: both kept
        "XX.A": SiteCandidates(
            triplets=[
                triplet("A", "", "HH", rank=6, rate=100.0, depth=0.0),
                triplet("A", "01", "GH", rank=0, rate=1000.0, depth=200.0),
            ]
        ),
        # two locations with the same kind and depth: only the higher priority one, plain id
        "XX.B": SiteCandidates(
            triplets=[
                triplet("B", "00", "HH", rank=6, rate=100.0),
                triplet("B", "10", "EH", rank=8, rate=100.0),
            ]
        ),
    }
    chosen, skipped, dropped = inv.choose_triplets(sites, cfg)
    assert skipped == []
    assert sorted(c.id for c in chosen) == ["XX.A.", "XX.A.01", "XX.B"]
    ids = {c.id: c for c in chosen}
    assert (ids["XX.A."].kind, ids["XX.A.01"].kind) == ("surface", "borehole")
    assert ids["XX.A.01"].profile == "borehole-A"
    reason = "same kind (surface) and depth as XX.B.00.HH?"
    assert dropped == [{"triplet": "XX.B.10.EH?", "reason": reason}]


@pytest.mark.smoke
def test_accelerometer_used_when_velocity_triplet_has_no_profile(cfg: StationSelection) -> None:
    sites = {
        "XX.C": SiteCandidates(
            triplets=[
                triplet("C", "00", "SH", rank=10, rate=50.0),
                triplet("C", "00", "HN", family="accelerometer", rank=3, rate=100.0),
            ]
        ),
        "XX.D": SiteCandidates(triplets=[triplet("D", "00", "SH", rank=10, rate=50.0)]),
    }
    chosen, skipped, dropped = inv.choose_triplets(sites, cfg)
    assert [(c.id, c.triplet.code, c.kind, c.profile) for c in chosen] == [
        ("XX.C", "HN", "strong_motion", "surface-100")
    ]
    assert {"triplet": "XX.C.00.SH?", "reason": "no preprocess profile for 50 Hz"} in dropped
    assert skipped == [{"site": "XX.D", "reason": "00.SH: no preprocess profile for 50 Hz"}]


@pytest.mark.smoke
def test_non_seismic_channel_without_sample_rate_is_ignored(
    run_section: RunSection, cfg: StationSelection
) -> None:
    edited = read_inventory(str(CHANNELS_XML))
    for ch in edited.select(network="2J")[0][0]:
        ch.sample_rate = None  # SampleRate is optional in StationXML
    for ch in edited.select(station="FOR2", channel="HHN")[0][0]:
        ch.sample_rate = None
    sites, _ = inv.collect_candidates(edited, run_section, cfg)
    assert sites["2J.FS01"].triplets == [] and sites["2J.FS01"].ignored_codes
    assert "01.HHN: StationXML lacks sample rate" in sites["UU.FOR2"].problems
    assert sites["UU.FOR2"].triplets == []


@pytest.mark.smoke
def test_epoch_change_inside_window_is_flagged(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    edited = read_inventory(str(CHANNELS_XML))
    uu = next(n for n in edited.networks if n.code == "UU")
    sta = next(s for s in uu.stations if s.code == "FSB1")  # select() would copy the list
    hhz = next(ch for ch in sta.channels if ch.code == "HHZ")
    later = copy.deepcopy(hhz)
    change = UTCDateTime(run_section.window_start_s + 14 * 3600)
    hhz.end_date = change
    later.start_date, later.depth = change, 35.0
    sta.channels.append(later)
    xml = tmp_path / "edited.xml"
    edited.write(str(xml), format="STATIONXML")
    res = build(run_section, cfg, tmp_path / "cache", client=FixtureClient(xml))
    fsb1 = by_id(res)["UU.FSB1"]
    assert fsb1["sensorDepthM"] == 29.0  # the epoch covering most of the window
    assert any("2 epochs in the window" in f for f in flags_of(res, "UU.FSB1"))


# --- coverage -------------------------------------------------------------------------------------


@pytest.mark.smoke
def test_coverage_marks_stations_without_data_unused(result: InventoryResult) -> None:
    rows = by_id(result)
    assert rows["NP.7225"]["usedInRun"] is False
    assert detail(result, "NP.7225")["coverage"] == 0.0
    assert all(rows[i]["usedInRun"] for i in EXPECTED_IDS if i != "NP.7225")
    assert result.counts["withData"] == len(EXPECTED_IDS) - 1
    assert result.counts["usedInRun"] == len(EXPECTED_IDS) - 1


@pytest.mark.smoke
def test_no_measurement_keeps_station_used_and_flagged(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    def cs01_unmeasured(url: str, params: Any, timeout_s: float) -> HttpResult:
        if params["sta"] == "CS01":
            return HttpResult(204, "")
        return mustang_http(url, params, timeout_s)

    res = build(run_section, cfg, tmp_path / "cache", http_get=cs01_unmeasured)
    assert cfg.availability.onMissing == "used"
    assert detail(res, "6K.CS01")["coverage"] is None
    assert by_id(res)["6K.CS01"]["usedInRun"] is True
    assert any("coverage unknown" in f for f in flags_of(res, "6K.CS01"))
    assert res.counts["noAvailabilityMeasurement"] == 1

    unused = with_availability(cfg, onMissing="unused")
    res = build(run_section, unused, tmp_path / "cache2", http_get=cs01_unmeasured)
    assert by_id(res)["6K.CS01"]["usedInRun"] is False


@pytest.mark.smoke
def test_unmeasured_station_with_no_served_data_is_unused(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    def cs01_unmeasured(url: str, params: Any, timeout_s: float) -> HttpResult:
        if params["sta"] == "CS01":
            return HttpResult(204, "")
        return mustang_http(url, params, timeout_s)

    res = build(
        run_section,
        cfg,
        tmp_path / "cache",
        http_get=cs01_unmeasured,
        rate_probe=serving({"6K.CS01": None}),
    )
    assert detail(res, "6K.CS01")["rateCheck"]["status"] == "nodata"
    assert by_id(res)["6K.CS01"]["usedInRun"] is False
    assert any("no data at any rate probe either" in f for f in flags_of(res, "6K.CS01"))


@pytest.mark.smoke
def test_no_station_with_data_fails_loudly(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    def nothing(url: str, params: Any, timeout_s: float) -> HttpResult:
        return HttpResult(204, "")

    unused = with_availability(cfg, onMissing="unused")
    with pytest.raises(InventoryError, match="none of the 8 selected stations has data"):
        build(run_section, unused, tmp_path / "cache", http_get=nothing)


@pytest.mark.smoke
def test_partial_day_coverage_is_overlap_weighted(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
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

    chosen = [_chosen()]
    cache = JsonCache(tmp_path / "avail.json")
    cov = inv.fetch_coverage(chosen, run, cfg.availability, http, cache)
    assert seen["timewindow"] == "2026-09-10T00:00:00,2026-09-12T00:00:00"
    assert seen["cha"] == "HHZ,HHN,HHE"
    assert cov["XX.TST"]["HHZ"] == pytest.approx(12 / 18 + 0.5 * 6 / 18)
    assert inv.station_coverage(cov["XX.TST"]) == pytest.approx(12 / 18 + 0.5 * 6 / 18)
    again = inv.fetch_coverage(chosen, run, cfg.availability, no_http, JsonCache(cache.path))
    assert again == cov


def _chosen() -> Chosen:
    return Chosen(
        id="XX.TST", triplet=triplet("TST", "00", "HH"), kind="surface", profile="surface-100"
    )


def _row(cha: str, day: str, value: float, qual: str = "M") -> dict[str, Any]:
    return {"value": value, "loc": "00", "cha": cha, "qual": qual, "start": f"{day}T00:00:00"}


@pytest.mark.smoke
def test_missing_measurement_and_service_failure(
    run_section: RunSection, fast: StationSelection, tmp_path: Path
) -> None:
    chosen = [_chosen()]

    def no_content(url: str, params: Any, timeout_s: float) -> HttpResult:
        return HttpResult(204, "")

    def fetch(acfg: Any, http: Any) -> dict[str, dict[str, float | None]]:
        return inv.fetch_coverage(chosen, run_section, acfg, http, JsonCache(tmp_path / "x.json"))

    acfg = fast.availability
    with pytest.raises(AvailabilityError, match="no percent_availability measurement"):
        fetch(acfg.model_copy(update={"onMissing": "error"}), no_content)
    (tmp_path / "x.json").unlink()
    for policy in ("unused", "used"):
        cov = fetch(acfg.model_copy(update={"onMissing": policy}), no_content)
        assert inv.station_coverage(cov["XX.TST"]) is None
        (tmp_path / "x.json").unlink()

    calls: list[int] = []

    def down(url: str, params: Any, timeout_s: float) -> HttpResult:
        calls.append(1)
        return HttpResult(503, "Service Unavailable")

    with pytest.raises(AvailabilityError, match="HTTP 503"):
        fetch(acfg, down)
    assert len(calls) == acfg.retries + 1
    assert not (tmp_path / "x.json").exists()  # failures are never cached

    def bad_request(url: str, params: Any, timeout_s: float) -> HttpResult:
        calls.append(1)
        return HttpResult(400, "bad")

    calls.clear()
    with pytest.raises(AvailabilityError, match="HTTP 400"):
        fetch(acfg, bad_request)
    assert len(calls) == 1  # a client error is not retried

    replies: list[Any] = [requests.ConnectionError("reset"), HttpResult(204, "")]

    def flaky(url: str, params: Any, timeout_s: float) -> HttpResult:
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    assert inv.station_coverage(fetch(acfg, flaky)["XX.TST"]) is None
    assert replies == []


# --- elevation rules ----------------------------------------------------------------------------


@pytest.mark.smoke
def test_shallow_station_far_from_dem_uses_dem_surface(
    run_section: RunSection, cfg: StationSelection, result: InventoryResult, tmp_path: Path
) -> None:
    assert cfg.elevation.onShallowMismatch == "dem"
    for6 = by_id(result)["UU.FOR6"]
    elev = detail(result, "UU.FOR6")["elevation"]
    assert (elev["convention"], elev["basis"]) == ("dem", "mismatch")
    assert elev["stationElevM"] == 2421.0  # the StationXML value stays in the report
    assert for6["surfaceElevM"] == pytest.approx(FOR6_DEM)
    assert for6["sensorElevM"] == pytest.approx(FOR6_DEM)
    assert for6["enu"]["u"] == pytest.approx(FOR6_DEM - run_section.origin.elevM)
    assert any("maxShallowMismatchM" in f for f in flags_of(result, "UU.FOR6"))
    assert result.counts["demSurface"] == 1

    keep = build(run_section, with_elevation(cfg, onShallowMismatch="keep"), tmp_path / "c1")
    assert by_id(keep)["UU.FOR6"]["surfaceElevM"] == 2421.0
    assert flags_of(keep, "UU.FOR6")
    skip = build(run_section, with_elevation(cfg, onShallowMismatch="skip"), tmp_path / "c2")
    assert "UU.FOR6" not in by_id(skip)
    with pytest.raises(AmbiguousElevationError, match="UU.FOR6"):
        build(run_section, with_elevation(cfg, onShallowMismatch="error"), tmp_path / "c3")


@pytest.mark.smoke
def test_enu_of_sensor(result: InventoryResult, run_section: RunSection) -> None:
    fork = by_id(result)["UU.FORK"]
    assert fork["enu"]["u"] == pytest.approx(1408.0 - run_section.origin.elevM)


def _ecfg(cfg: StationSelection, **update: Any) -> ElevationCheck:
    return cfg.elevation.model_copy(update=update)


@pytest.mark.smoke
def test_resolve_elevation_rules(cfg: StationSelection) -> None:
    e = _ecfg(cfg, toleranceM=15.0, maxShallowMismatchM=50.0, onAmbiguous="error")
    r = inv.resolve_elevation
    d = r(1634.7, 1634.7, 290.0, 1634.7, e, what="t")
    assert d is not None and (d.convention, d.sensor_elev_m) == ("surface", pytest.approx(1344.7))
    assert d.flag is None
    d = r(1474.0, 1515.0, 41.0, 1512.6, e, what="t")
    assert d is not None and (d.convention, d.surface_elev_m) == ("sensor", 1515.0)
    # both readings match and the depth is within tolerance: harmless, surface, no flag
    d = r(1840.1, 1840.1, 1.0, 1840.2, e, what="t")
    assert d is not None and (d.convention, d.basis, d.flag) == ("surface", "both-match", None)
    # both match but they differ by 29 m: the closer one wins, flagged
    d = r(1668.0, 1668.0, 29.0, 1683.0, e, what="t")
    assert d is not None and (d.convention, d.basis) == ("sensor", "both-match")
    assert d.sensor_elev_m == 1668.0 and d.flag is not None
    # shallow, off the DEM by less than maxShallowMismatchM: kept, flagged
    d = r(1853.0, 1853.0, 0.0, 1876.7, e, what="t")
    assert d is not None and (d.convention, d.basis, d.surface_elev_m) == (
        "surface",
        "shallow",
        1853.0,
    )
    assert d.flag is not None
    # deeper than tolerance and neither reading matches: ambiguous, even below 2 * tolerance
    with pytest.raises(AmbiguousElevationError):
        r(1700.0, 1700.0, 20.0, 1660.0, e, what="t")
    with pytest.raises(AmbiguousElevationError):
        r(1408.0, 1409.0, 281.0, 1500.0, e, what="t")
    assert r(1408.0, 1409.0, 281.0, 1500.0, _ecfg(cfg, onAmbiguous="skip"), what="t") is None
    d = r(1408.0, 1409.0, 281.0, 1500.0, _ecfg(cfg, onAmbiguous="surface"), what="t")
    assert d is not None and (d.basis, d.sensor_elev_m) == ("assumed", 1127.0)
    d = r(1408.0, 1409.0, 281.0, 1500.0, _ecfg(cfg, onAmbiguous="dem"), what="t")
    assert d is not None and (d.convention, d.surface_elev_m, d.sensor_elev_m) == (
        "dem",
        1500.0,
        1219.0,
    )
    for policy in ("dem", "keep"):
        d = r(2421.0, 2421.0, 0.0, 2261.7, _ecfg(cfg, onShallowMismatch=policy), what="t")
        assert d is not None and d.flag is not None
        assert d.sensor_elev_m == d.surface_elev_m - d.depth_m


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
    assert any("looks like a borehole geophone" in f for f in flags_of(res, "UU.FORK"))


# --- caches and determinism ---------------------------------------------------------------------


@pytest.mark.smoke
def test_rerun_is_a_pure_cache_hit(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    client = FixtureClient()
    first = build(run_section, cfg, cache, client=client)
    assert client.levels.count("channel") == 1
    assert client.levels.count("response") == len(EXPECTED_IDS)
    xml_dir = cache / inv.STATIONXML_DIR
    assert inv.dem_cache_path(xml_dir, cfg).exists()
    assert (xml_dir / cfg.availability.cacheFile).exists()
    for sid in EXPECTED_IDS:
        assert (xml_dir / f"{sid}.xml").exists()
    second = build(run_section, cfg, cache, client=NoNetworkClient(), dem=no_dem, http_get=no_http)
    assert second.rows == first.rows
    assert second.report == first.report


@pytest.mark.smoke
def test_dem_cache_file_is_keyed_by_source(cfg: StationSelection, tmp_path: Path) -> None:
    other = with_elevation(cfg, demUrl="https://example.invalid/dem")
    a, b = inv.dem_cache_path(tmp_path, cfg), inv.dem_cache_path(tmp_path, other)
    assert a != b and a.suffix == b.suffix == ".json"
    assert a.name.startswith(Path(cfg.elevation.demCacheFile).stem)


@pytest.mark.smoke
def test_stale_response_cache_is_refetched(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    build(run_section, cfg, cache)
    xml_dir = cache / inv.STATIONXML_DIR
    channel_level = read_inventory(str(CHANNELS_XML))
    # other channels
    channel_level.select(station="FORK", channel="EH?").write(
        str(xml_dir / "UU.FORK.xml"), format="STATIONXML"
    )
    # the right channels, but channel level (sensitivity only, no response stages)
    channel_level.select(station="FOR2", channel="HH?").write(
        str(xml_dir / "UU.FOR2.xml"), format="STATIONXML"
    )
    # the right channels with responses, but an epoch that ended before the window
    old = channel_level.select(station="FSB4", channel="HH?")
    for ch in old[0][0]:
        ch.response = fake_response()
        ch.end_date = UTCDateTime(run_section.window_start_s - 86400)
    old.write(str(xml_dir / "UU.FSB4.xml"), format="STATIONXML")
    client = FixtureClient()
    build(run_section, cfg, cache, client=client, http_get=no_http, dem=no_dem)
    assert client.levels == ["response"] * 3


@pytest.mark.smoke
def test_station_service_errors_are_retried_then_raised(
    run_section: RunSection, fast: StationSelection, tmp_path: Path
) -> None:
    class Flaky(FixtureClient):
        def __init__(self, errors: list[Exception]) -> None:
            super().__init__()
            self.errors = errors

        def get_stations(self, **kw: Any) -> None:
            if self.errors:
                raise self.errors.pop(0)
            super().get_stations(**kw)

    ok = build(run_section, fast, tmp_path / "a", client=Flaky([FDSNException("timeout")]))
    assert [r["id"] for r in ok.rows] == EXPECTED_IDS
    with pytest.raises(InventoryError, match="no data"):
        build(run_section, fast, tmp_path / "b", client=Flaky([FDSNNoDataException("none")]))
    errors: list[Exception] = [OSError("down")] * (fast.query.retries + 1)
    with pytest.raises(InventoryError, match="failed after"):
        build(run_section, fast, tmp_path / "c", client=Flaky(errors))


# --- ENU, profiles, DEM -------------------------------------------------------------------------


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
    assert inv.match_profile(500.0, cfg) == "borehole-A"
    assert inv.match_profile(1000.0, cfg) == "borehole-A"
    for rate in (40.0, 110.0, 260.0, 400.0, 1100.0, 2000.0):
        assert inv.match_profile(rate, cfg) is None


@pytest.mark.smoke
def test_epqs_dem_retries_then_parses_and_rejects_nodata(fast: StationSelection) -> None:
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
def test_station_selection_rejects_unknown_and_inconsistent_knobs(raw_signal_yaml: dict) -> None:
    raw = json.loads(json.dumps(raw_signal_yaml["stations"]))
    raw["notAKnob"] = 1
    with pytest.raises(ValidationError):
        StationSelection.model_validate(raw)
    rules = dict(raw_signal_yaml["stations"]["channels"])
    rules["accelerometerCodes"] = [*rules["accelerometerCodes"], "HH"]
    with pytest.raises(ValidationError, match="overlap"):
        ChannelRules.model_validate(rules)
    elevation = dict(raw_signal_yaml["stations"]["elevation"])
    elevation["maxShallowMismatchM"] = elevation["toleranceM"] / 2
    with pytest.raises(ValidationError, match="maxShallowMismatchM"):
        ElevationCheck.model_validate(elevation)


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
    from hq_contracts import io  # a hard dependency (pyproject.toml): a broken package fails

    seeded = build(fake_ctx.config.run, cfg, fake_ctx.cache_dir)  # seeds every cache

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access from run(ctx) on a warm cache")

    monkeypatch.setattr(inv, "requests_get", refuse)
    monkeypatch.setattr(inv._LazyClient, "get", refuse)
    monkeypatch.setattr(inv, "fdsn_rate_probe", lambda query, rcfg: refuse)  # rate probes too
    inv.run(fake_ctx)
    df = io.read_table(fake_ctx.path(inv.STATIONS_FILE))
    assert list(df["id"]) == EXPECTED_IDS
    assert {"enu_e", "enu_n", "enu_u", "sensorDepthM", "sensorElevM"} <= set(df.columns)
    fork = df[df["id"] == "UU.FORK"].iloc[0]
    assert (fork["sensorDepthM"], fork["sensorElevM"]) == (281.0, pytest.approx(1408.0))
    assert_depths_match_stationxml(df.to_dict("records"), fake_ctx.config.run.window_start_s)
    assert list(df["usedInRun"]) == [r["usedInRun"] for r in seeded.rows]
    report = json.loads(fake_ctx.path(inv.REPORT_FILE).read_text(encoding="utf-8"))
    assert report["counts"]["selected"] == len(EXPECTED_IDS)
    rec = fake_ctx.records[inv.STAGE]
    assert rec["counts"]["selected"] == len(EXPECTED_IDS)
    assert rec["params"] == {inv.STAGE: cfg.model_dump(mode="json")}


@pytest.mark.smoke
def test_missing_contracts_package_fails_loudly(
    tmp_path: Path, result: InventoryResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "hq_contracts.io", None)
    with pytest.raises(inv.InventoryError, match="CONTRACT-01"):
        inv.write_stations_parquet(result.rows, tmp_path / inv.STATIONS_FILE)
    assert not list(tmp_path.glob(inv.STATIONS_FILE + "*"))


# --- sample-rate check (data vs StationXML) ------------------------------------------------------


def serving(overrides: dict[str, float | None]) -> Any:
    """A probe that serves StationXML rates except for the stations in ``overrides``
    (NET.STA -> served rate, or None for no data)."""
    calls: list[tuple[str, float]] = []

    def probe(net: str, sta: str, loc: str, channels: Any, t0: float, t1: float) -> Any:
        calls.append((f"{net}.{sta}", t0))
        key = f"{net}.{sta}"
        if key in overrides:
            rate = overrides[key]
            return None if rate is None else {cha: rate for cha in channels}
        return metadata_rate_probe(net, sta, loc, channels, t0, t1)

    probe.calls = calls  # type: ignore[attr-defined]
    return probe


def with_rate(cfg: StationSelection, **update: Any) -> StationSelection:
    return cfg.model_copy(update={"rateCheck": cfg.rateCheck.model_copy(update=update)})


@pytest.mark.smoke
def test_rate_mismatch_uses_served_rate_and_its_profile(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    res = build(run_section, cfg, tmp_path / "cache", rate_probe=serving({"UU.FOR2": 100.0}))
    row = by_id(res)["UU.FOR2"]
    assert row["sampleRateHz"] == 100.0
    assert row["preprocessProfile"] == "surface-100"
    det = {d["id"]: d for d in res.report["stations"]}["UU.FOR2"]
    assert det["rateCheck"]["status"] == "mismatch"
    assert det["rateCheck"]["metadataHz"] == 200.0 and det["rateCheck"]["dataHz"] == 100.0
    assert any("serves 100 Hz" in f for f in flags_of(res, "UU.FOR2"))
    assert res.counts["rateMismatch"] == 1
    baseline = by_id(build(run_section, cfg, tmp_path / "baseline"))
    assert baseline["UU.FOR2"]["preprocessProfile"] == "surface-hi"
    for r in res.rows:
        if r["id"] != "UU.FOR2":
            assert r == baseline[r["id"]]


@pytest.mark.smoke
def test_rate_mismatch_skip_and_error_policies(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    probe = serving({"UU.FOR2": 100.0})
    res = build(run_section, with_rate(cfg, onMismatch="skip"), tmp_path / "a", rate_probe=probe)
    assert "UU.FOR2" not in by_id(res)
    assert any(s["site"] == "UU.FOR2" for s in res.report["skippedSites"])
    with pytest.raises(InventoryError, match="serves 100 Hz"):
        build(run_section, with_rate(cfg, onMismatch="error"), tmp_path / "b", rate_probe=probe)


@pytest.mark.smoke
def test_rate_unverified_keeps_metadata_or_raises(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    probe = serving({"UU.FOR2": None})
    res = build(run_section, cfg, tmp_path / "a", rate_probe=probe)
    assert by_id(res)["UU.FOR2"]["sampleRateHz"] == 200.0
    assert any("unverified" in f for f in flags_of(res, "UU.FOR2"))
    assert res.counts["rateUnverified"] == 1
    tried = [t0 for sid, t0 in probe.calls if sid == "UU.FOR2"]
    assert len(tried) == len(cfg.rateCheck.probeOffsetsS)  # every in-window offset was tried
    with pytest.raises(InventoryError, match="no data at any rate probe"):
        build(run_section, with_rate(cfg, onNoData="error"), tmp_path / "b", rate_probe=probe)


@pytest.mark.smoke
def test_rate_probe_uses_every_in_window_offset(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    first = run_section.window_start_s + cfg.rateCheck.probeOffsetsS[0]

    def probe(net: str, sta: str, loc: str, channels: Any, t0: float, t1: float) -> Any:
        if f"{net}.{sta}" == "UU.FOR2" and t0 == first:
            return None
        return serving({"UU.FOR2": 100.0})(net, sta, loc, channels, t0, t1)

    res = build(run_section, cfg, tmp_path / "cache", rate_probe=probe)
    det = detail(res, "UU.FOR2")
    starts = inv.probe_starts(run_section, cfg.rateCheck)
    assert det["rateCheck"]["probeTimes"] == starts[1:]
    assert det["rateCheck"]["dataHz"] == 100.0


@pytest.mark.smoke
def test_rate_change_inside_the_window_is_an_error(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    last = inv.probe_starts(run_section, cfg.rateCheck)[-1]

    def probe(net: str, sta: str, loc: str, channels: Any, t0: float, t1: float) -> Any:
        rate = 100.0 if f"{net}.{sta}" == "UU.FOR2" and t0 == last else None
        if rate is not None:
            return {cha: rate for cha in channels}
        return metadata_rate_probe(net, sta, loc, channels, t0, t1)

    with pytest.raises(InventoryError, match="changes inside the window"):
        build(run_section, cfg, tmp_path / "cache", rate_probe=probe)


@pytest.mark.smoke
def test_no_data_replies_are_not_cached(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    build(run_section, cfg, cache, rate_probe=serving({"UU.FOR2": None}))
    again = serving({})
    res = build(run_section, cfg, cache, rate_probe=again)
    assert [sid for sid, _ in again.calls] == ["UU.FOR2"] * len(
        inv.probe_starts(run_section, cfg.rateCheck)
    )
    assert detail(res, "UU.FOR2")["rateCheck"]["status"] == "match"


@pytest.mark.smoke
def test_station_measured_empty_is_not_probed(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    probe = serving({})
    res = build(run_section, cfg, tmp_path / "cache", rate_probe=probe)
    unprobed = [d["id"] for d in res.report["stations"] if d["rateCheck"]["status"] == "unprobed"]
    assert unprobed and all(not by_id(res)[sid]["usedInRun"] for sid in unprobed)
    assert not any(sid in unprobed for sid, _ in probe.calls)


@pytest.mark.smoke
def test_no_probe_offset_inside_the_window_is_an_error(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    span = run_section.window_end_s - run_section.window_start_s
    bad = with_rate(cfg, probeOffsetsS=(span,))
    with pytest.raises(InventoryError, match="fits inside"):
        build(run_section, bad, tmp_path / "cache")


class _FakeFdsn:
    """Stands in for obspy's FDSN Client in the default probe."""

    script: ClassVar[list[Any]] = []
    made: ClassVar[int] = 0

    def __init__(self, base: str, timeout: float) -> None:
        type(self).made += 1

    def get_waveforms(self, *args: Any) -> Any:
        step = type(self).script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _stream(rates: dict[str, list[float]]) -> Any:
    from obspy import Stream, Trace

    traces = []
    for cha, rs in rates.items():
        for r in rs:
            tr = Trace(data=np.zeros(10, dtype=np.float64))
            tr.stats.channel = cha
            tr.stats.sampling_rate = r
            traces.append(tr)
    return Stream(traces)


@pytest.mark.smoke
def test_default_probe_retries_parses_and_rejects_mixed_rates(
    cfg: StationSelection, monkeypatch: pytest.MonkeyPatch
) -> None:
    from obspy.clients import fdsn

    monkeypatch.setattr(fdsn, "Client", _FakeFdsn)
    monkeypatch.setattr(inv.time, "sleep", lambda s: None)
    probe = inv.fdsn_rate_probe(cfg.query, cfg.rateCheck)
    _FakeFdsn.script = [FDSNException("timeout"), _stream({"HHZ": [100.0], "HHN": [100.0]})]
    assert probe("UU", "X", "01", ["HHZ", "HHN"], 0.0, 5.0) == {"HHZ": 100.0, "HHN": 100.0}
    _FakeFdsn.script = [FDSNNoDataException("204")]
    assert probe("UU", "X", "01", ["HHZ"], 0.0, 5.0) is None
    _FakeFdsn.script = [_stream({"HHZ": [100.0, 200.0]})]
    with pytest.raises(InventoryError, match="mixed rates"):
        probe("UU", "X", "01", ["HHZ"], 0.0, 5.0)
    _FakeFdsn.script = [FDSNException("down")] * (cfg.rateCheck.retries + 1)
    with pytest.raises(InventoryError, match="failed after"):
        probe("UU", "X", "01", ["HHZ"], 0.0, 5.0)


@pytest.mark.smoke
def test_rate_probe_results_are_cached(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    first = build(run_section, cfg, cache, rate_probe=serving({"UU.FOR2": 100.0}))
    second = build(run_section, cfg, cache, rate_probe=no_probe)
    assert first.rows == second.rows


@pytest.mark.smoke
def test_components_serving_different_rates_is_an_error(
    run_section: RunSection, cfg: StationSelection, tmp_path: Path
) -> None:
    def probe(net: str, sta: str, loc: str, channels: Any, t0: float, t1: float) -> Any:
        rates = metadata_rate_probe(net, sta, loc, channels, t0, t1)
        if f"{net}.{sta}" == "UU.FOR2" and rates:
            rates[channels[0]] = 100.0
        return rates

    with pytest.raises(InventoryError, match="different rates"):
        build(run_section, cfg, tmp_path / "cache", rate_probe=probe)

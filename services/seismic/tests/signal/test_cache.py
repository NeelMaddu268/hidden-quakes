"""SEIS-05: full-window download, channel-day cache, read API, gaps and Check A.

Offline: every request goes to ``FakeClient``, which serves small deterministic synthetic
waveforms built inside the test (sample value = f(time), never zero), so duplicated or
fabricated samples are detectable.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import obspy
import pytest
from obspy import UTCDateTime
from obspy.clients.fdsn.header import (
    FDSNBadRequestException,
    FDSNNoDataException,
    FDSNServiceUnavailableException,
    FDSNTimeoutException,
)

from hq.config.signal import DownloadConfig
from hq.ingest.cache import (
    AmbiguousStationError,
    CacheMissError,
    Segment,
    read_inventory,
    read_window,
    stationxml_path,
)
from hq.ingest.download import (
    DownloadIncompleteError,
    StationRequest,
    channel_gaps,
    check_a_table,
    coverage,
    download_window,
    plan_chunks,
    plan_units,
)

pytestmark = pytest.mark.smoke

RATE = 10.0  # Hz; small so a day-scale test stays fast
T_MIDNIGHT = UTCDateTime("2026-09-10T00:00:00Z").timestamp
HOUR = 3600.0


def make_cfg(**overrides: object) -> DownloadConfig:
    base: dict[str, object] = {
        "client": "FAKE",
        "timeoutS": 1.0,
        "chunkS": 3600,
        "padS": 300.0,
        "maxWorkers": 2,
        "maxRetries": 2,
        "backoffBaseS": 1.0,
        "backoffMaxS": 1.5,
        "provisionalLagS": 3600.0,
        "minGapSamples": 1.5,
        "maxGapFraction": 0.2,
        "minUsefulStations": 2,
    }
    base.update(overrides)
    return DownloadConfig.model_validate(base)


def sta(code: str, loc: str = "00", net: str = "XX") -> StationRequest:
    return StationRequest(
        id=f"{net}.{code}",
        network=net,
        station=code,
        location=loc,
        channels=("HHZ", "HHN", "HHE"),
    )


def value_at(times: np.ndarray) -> np.ndarray:
    """Deterministic, never-zero sample values keyed on absolute time."""
    return (np.round(times * RATE).astype(np.int64) % 1000 + 1).astype(np.int32)


class FakeClient:
    """Serves ``coverage[(sta, cha)]`` intervals, overhanging each request by 2 s like
    whole miniSEED records do. ``errors`` is a queue of exceptions raised before serving."""

    def __init__(
        self,
        coverage: dict[tuple[str, str], list[tuple[float, float]]],
        errors: list[Exception] | None = None,
        fail_station: str | None = None,
        location: str = "00",
    ) -> None:
        self.coverage = coverage
        self.errors = list(errors or [])
        self.fail_station = fail_station
        self.location = location
        self.calls: list[tuple[str, str, float, float]] = []

    def get_waveforms(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
    ) -> obspy.Stream:
        a, b = starttime.timestamp, endtime.timestamp
        self.calls.append((station, location, a, b))
        if self.errors:
            raise self.errors.pop(0)
        if station == self.fail_station:
            raise FDSNServiceUnavailableException("503 in test")
        st = obspy.Stream()
        for cha in channel.split(","):
            for c, d in self.coverage.get((station, cha), []):
                lo, hi = max(c, a - 2.0), min(d, b + 2.0)
                if hi <= lo:
                    continue
                first = np.ceil(lo * RATE) / RATE
                times = np.arange(first, hi, 1.0 / RATE)
                if times.size == 0:
                    continue
                tr = obspy.Trace(data=value_at(times))
                tr.stats.network = network
                tr.stats.station = station
                tr.stats.location = "" if location == "--" else location
                tr.stats.channel = cha
                tr.stats.sampling_rate = RATE
                tr.stats.starttime = UTCDateTime(float(times[0]))
                st.append(tr)
        if not st:
            raise FDSNNoDataException("no data in test")
        return st


def full(stations: Sequence[str], t0: float, t1: float) -> dict[tuple[str, str], list]:
    return {(s, c): [(t0, t1)] for s in stations for c in ("HHZ", "HHN", "HHE")}


def factory(client: FakeClient) -> Callable[[], FakeClient]:
    return lambda: client


def no_network() -> FakeClient:
    raise AssertionError("a pure cache hit must not create a client or make any request")


def run_download(
    stations: Sequence[StationRequest],
    t0: float,
    t1: float,
    cfg: DownloadConfig,
    cache: Path,
    client_factory: Callable[[], FakeClient],
    now: float = T_MIDNIGHT + 30 * 86_400.0,
    sleeps: list[float] | None = None,
):
    return download_window(
        stations,
        t0,
        t1,
        cfg,
        cache_dir=cache,
        client_factory=client_factory,
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
        now=lambda: now,
    )


# --- planning ---------------------------------------------------------------------------------


def test_chunks_split_on_hours_across_midnight_with_pad() -> None:
    cfg = make_cfg()
    units = plan_units([sta("A")], T_MIDNIGHT, T_MIDNIGHT + 86_400.0, cfg)
    assert [u.day for u in units] == ["20260909", "20260910", "20260911"]
    assert units[0].chunks == ((T_MIDNIGHT - 300.0, T_MIDNIGHT),)
    assert len(units[1].chunks) == 24
    assert all(b - a == HOUR for a, b in units[1].chunks)
    assert units[2].chunks == ((T_MIDNIGHT + 86_400.0, T_MIDNIGHT + 86_400.0 + 300.0),)


def test_any_window_chunks_on_hour_boundaries() -> None:
    t1 = T_MIDNIGHT + 21 * HOUR + 37 * 60 + 12.5  # "last 2 hours" at an odd time
    chunks = plan_chunks(t1 - 2 * HOUR, t1, 3600)
    assert chunks[0] == (t1 - 2 * HOUR, T_MIDNIGHT + 20 * HOUR)
    assert chunks[1] == (T_MIDNIGHT + 20 * HOUR, T_MIDNIGHT + 21 * HOUR)
    assert chunks[2] == (T_MIDNIGHT + 21 * HOUR, t1)
    with pytest.raises(ValueError):
        plan_chunks(0.0, 10.0, 7000)  # does not divide a day


def test_duplicate_channel_is_rejected() -> None:
    with pytest.raises(ValueError):
        plan_units([sta("A"), sta("A")], T_MIDNIGHT, T_MIDNIGHT + HOUR, make_cfg())


# --- download, manifest, cache hit ------------------------------------------------------------


def test_download_writes_channel_day_files_then_rerun_is_pure_cache_hit(tmp_path: Path) -> None:
    cfg = make_cfg()
    t0, t1 = T_MIDNIGHT + HOUR, T_MIDNIGHT + 3 * HOUR
    client = FakeClient(full(["A"], t0 - HOUR, t1 + HOUR))
    res = run_download([sta("A")], t0, t1, cfg, tmp_path, factory(client))
    assert res.counts()["requests"] == 4  # 00:55-01, 01-02, 02-03, 03-03:05
    assert len(client.calls) == 4 and all(c[1] == "00" for c in client.calls)
    mseed = sorted(p.name for p in (tmp_path / "mseed").glob("*.mseed"))
    assert mseed == [f"XX.A.00.{c}.20260910.mseed" for c in ("HHE", "HHN", "HHZ")]
    man = json.loads((tmp_path / "mseed" / "XX.A.00.HHZ.20260910.json").read_text())
    assert [c["status"] for c in man["chunks"]] == ["ok"] * 4
    assert man["chunks"][0]["start"] == t0 - 300.0 and man["chunks"][-1]["end"] == t1 + 300.0

    # exactly the served samples: half-open chunks, no duplicates at chunk edges
    st = obspy.read(str(tmp_path / "mseed" / "XX.A.00.HHZ.20260910.mseed"))
    st.merge(method=-1)
    assert len(st) == 1 and st[0].stats.npts == int((t1 - t0 + 600.0) * RATE)

    again = run_download([sta("A")], t0, t1, cfg, tmp_path, no_network)
    assert again.counts()["requests"] == 0
    assert again.counts()["cacheHitUnits"] == again.counts()["units"]


def test_empty_location_uses_double_dash_and_two_dot_file_name(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + HOUR
    client = FakeClient(full(["CS01"], t0, t1))
    run_download([sta("CS01", loc="", net="6K")], t0, t1, cfg, tmp_path, factory(client))
    assert client.calls[0][1] == "--"
    assert (tmp_path / "mseed" / "6K.CS01..HHZ.20260910.mseed").is_file()
    assert len(read_window("6K.CS01", t0, t1, cache_dir=tmp_path)) == 3


def test_nodata_is_recorded_not_an_error(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + 2 * HOUR
    res = run_download([sta("A")], t0, t1, cfg, tmp_path, factory(FakeClient({})))
    assert res.counts()["chunksNodata"] == 2 and not res.failures
    assert not list((tmp_path / "mseed").glob("*.mseed"))
    man = json.loads((tmp_path / "mseed" / "XX.A.00.HHN.20260910.json").read_text())
    assert [c["status"] for c in man["chunks"]] == ["nodata", "nodata"]
    assert run_download([sta("A")], t0, t1, cfg, tmp_path, no_network).counts()["requests"] == 0


def test_one_missing_channel_is_nodata_for_that_channel_only(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + HOUR
    cov = full(["A"], t0, t1)
    del cov[("A", "HHE")]
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(FakeClient(cov)))
    for cha, status in (("HHZ", "ok"), ("HHE", "nodata")):
        man = json.loads((tmp_path / "mseed" / f"XX.A.00.{cha}.20260910.json").read_text())
        assert man["chunks"][0]["status"] == status


def test_retry_with_exponential_backoff_capped(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0, maxRetries=3, backoffBaseS=1.0, backoffMaxS=1.5)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + HOUR
    client = FakeClient(
        full(["A"], t0, t1),
        errors=[FDSNTimeoutException("t1"), OSError("reset"), FDSNTimeoutException("t2")],
    )
    sleeps: list[float] = []
    res = run_download([sta("A")], t0, t1, cfg, tmp_path, factory(client), sleeps=sleeps)
    assert sleeps == [1.0, 1.5, 1.5]
    assert res.counts()["requests"] == 4 and res.counts()["chunksOk"] == 1


def test_failures_listed_at_end_and_rerun_resumes_only_them(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0, maxRetries=1)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + 2 * HOUR
    cov = full(["A", "B"], t0, t1)
    sleeps: list[float] = []
    bad = FakeClient(cov, fail_station="B")
    with pytest.raises(DownloadIncompleteError) as err:
        run_download([sta("A"), sta("B")], t0, t1, cfg, tmp_path, factory(bad), sleeps=sleeps)
    failures = err.value.result.failures
    assert sorted((f.stationId, f.start) for f in failures) == [("XX.B", t0), ("XX.B", t0 + HOUR)]
    assert "XX.B" in str(err.value) and "2026-09-10T01:00:00Z" in str(err.value)
    assert not list((tmp_path / "mseed").glob("XX.B.*"))  # nothing recorded for failed chunks
    assert len(sleeps) == 2  # one retry per failing chunk

    good = FakeClient(cov)
    res = run_download([sta("A"), sta("B")], t0, t1, cfg, tmp_path, factory(good))
    assert {c[0] for c in good.calls} == {"B"} and res.counts()["requests"] == 2


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0, maxRetries=3)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + HOUR
    client = FakeClient(full(["A"], t0, t1), errors=[FDSNBadRequestException("400")])
    sleeps: list[float] = []
    with pytest.raises(DownloadIncompleteError):
        run_download([sta("A")], t0, t1, cfg, tmp_path, factory(client), sleeps=sleeps)
    assert sleeps == [] and len(client.calls) == 1


def test_provisional_chunks_are_refetched_and_replaced(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0, provisionalLagS=3600.0)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + 2 * HOUR
    # live mode: fetched 10 minutes after the window end, the latest data is still arriving
    partial = {k: [(t0, t1 - 1800.0)] for k in full(["A"], t0, t1)}
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(FakeClient(partial)), now=t1 + 600.0)
    # only the last chunk ended < 1 h before its fetch: it alone is provisional and refetched
    later = FakeClient(full(["A"], t0, t1))
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(later), now=t1 + 7200.0)
    assert [c[2] for c in later.calls] == [t0 + HOUR]
    st = read_window("XX.A", t0, t1 - 1.0 / RATE, cache_dir=tmp_path).select(channel="HHZ")
    assert len(st) == 1 and st[0].stats.npts == int((t1 - t0) * RATE)  # replaced, not doubled
    # now settled: nothing is provisional any more
    assert run_download([sta("A")], t0, t1, cfg, tmp_path, no_network).counts()["requests"] == 0


# --- read API ---------------------------------------------------------------------------------


def test_read_window_across_midnight_keeps_gap_as_separate_traces(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0)
    t0, t1 = T_MIDNIGHT - HOUR, T_MIDNIGHT + HOUR
    gap = (T_MIDNIGHT + 600.0, T_MIDNIGHT + 720.0)
    cov = {k: [(t0, gap[0]), (gap[1], t1)] for k in full(["A"], t0, t1)}
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(FakeClient(cov)))
    assert (tmp_path / "mseed" / "XX.A.00.HHZ.20260909.mseed").is_file()

    st = read_window("XX.A", t0 + 1800.0, t1 - 1800.0, cache_dir=tmp_path).select(channel="HHZ")
    assert len(st) == 2  # midnight joined, the gap kept
    first, second = st
    assert first.stats.starttime.timestamp == t0 + 1800.0
    assert first.stats.endtime.timestamp == pytest.approx(gap[0] - 1.0 / RATE)
    assert second.stats.starttime.timestamp == gap[1]
    assert second.stats.endtime.timestamp == t1 - 1800.0
    for tr in st:
        assert not np.ma.isMaskedArray(tr.data)
        assert np.all(tr.data != 0)  # never zero-filled
        np.testing.assert_array_equal(tr.data, value_at(tr.times("timestamp")))


def test_read_window_trims_to_request(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + HOUR
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(FakeClient(full(["A"], t0, t1))))
    st = read_window("XX.A", t0 + 100.05, t0 + 200.0, cache_dir=tmp_path)
    assert len(st) == 3
    for tr in st:
        assert tr.stats.starttime.timestamp == pytest.approx(t0 + 100.1)
        assert tr.stats.endtime.timestamp == pytest.approx(t0 + 200.0)
    assert len(read_window("XX.A", t1 + HOUR, t1 + 2 * HOUR, cache_dir=tmp_path)) == 0


def test_bare_station_id_with_two_locations_is_ambiguous(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + HOUR
    cov = full(["A"], t0, t1)
    for loc in ("00", "10"):
        s = StationRequest(f"XX.A.{loc}", "XX", "A", loc, ("HHZ", "HHN", "HHE"))
        run_download([s], t0, t1, cfg, tmp_path, factory(FakeClient(cov)))
    with pytest.raises(AmbiguousStationError):
        read_window("XX.A", t0, t1, cache_dir=tmp_path)
    assert {tr.stats.location for tr in read_window("XX.A.10", t0, t1, cache_dir=tmp_path)} == {
        "10"
    }


def test_nothing_cached_raises(tmp_path: Path) -> None:
    with pytest.raises(CacheMissError):
        read_window("XX.NOPE", T_MIDNIGHT, T_MIDNIGHT + 60.0, cache_dir=tmp_path)
    with pytest.raises(CacheMissError):
        read_inventory("XX.NOPE", cache_dir=tmp_path)


def test_read_inventory_reads_station_xml(tmp_path: Path) -> None:
    from obspy.core.inventory import Channel, Inventory, Network, Station

    cha = Channel("HHZ", "00", 38.5, -112.9, 1600.0, 150.0, sample_rate=RATE)
    inv = Inventory(
        networks=[Network("XX", stations=[Station("A", 38.5, -112.9, 1600.0, channels=[cha])])],
        source="test",
    )
    path = stationxml_path(tmp_path, "XX.A")
    path.parent.mkdir(parents=True)
    inv.write(str(path), format="STATIONXML")
    got = read_inventory("XX.A", cache_dir=tmp_path)
    assert got[0][0][0].depth == 150.0 and got[0][0].code == "A"


# --- gaps and Check A -------------------------------------------------------------------------


def test_channel_gaps_leading_trailing_interior_and_jitter() -> None:
    dt = 0.1
    t0, t1 = 0.0, 100.0
    segs = [
        Segment("HHZ", 10.0, 39.9, dt),  # leading gap 0-10
        Segment("HHZ", 40.04, 49.94, dt),  # 0.04 s late: jitter, not a gap
        Segment("HHZ", 60.0, 89.9, dt),  # interior gap 50.04-60, trailing gap 90-100
    ]
    gaps, overlaps = channel_gaps(segs, t0, t1, 1.5)
    assert [(round(a, 2), round(b, 2)) for a, b in gaps] == [(0, 10), (50.04, 60), (90, 100)]
    assert overlaps == 0
    assert channel_gaps([], t0, t1, 1.5) == ([(t0, t1)], 0)
    _, n_over = channel_gaps(
        [Segment("HHZ", 0, 59.9, dt), Segment("HHZ", 50, 99.9, dt)], 0, 100, 1.5
    )
    assert n_over == 1


def test_gap_rows_and_check_a_classification(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0, maxGapFraction=0.2, minUsefulStations=2)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + 2 * HOUR
    cov = full(["GOOD", "FAR"], t0, t1)
    cov.update({(s, c): v for (s, c), v in full(["MISS"], t0, t1).items() if c != "HHE"})
    cov.update({k: [(t0 + 0.3 * (t1 - t0), t1)] for k in full(["GAPPY"], t0, t1)})
    stations = [sta("GOOD"), sta("MISS"), sta("GAPPY"), sta("FAR")]
    run_download(stations, t0, t1, cfg, tmp_path, factory(FakeClient(cov)))

    coords = {"XX.GOOD": (38.5, -112.9), "XX.FAR": (40.0, -112.9)}
    rows, report = coverage(
        stations + [sta("NEVER")],
        t0,
        t1,
        cfg,
        cache_dir=tmp_path,
        bbox=(-113.2, 38.28, -112.6, 38.74),
        coords=coords,
    )
    by_id = {r.stationId: r for r in report}
    assert by_id["XX.GOOD"].useful and by_id["XX.GOOD"].inBbox is True
    assert by_id["XX.MISS"].componentsPresent == 2 and not by_id["XX.MISS"].useful
    assert by_id["XX.GAPPY"].maxGapFraction == pytest.approx(0.3) and not by_id["XX.GAPPY"].useful
    assert by_id["XX.FAR"].inBbox is False and not by_id["XX.FAR"].useful
    assert by_id["XX.NEVER"].componentsPresent == 0 and by_id["XX.NEVER"].maxGapFraction == 1.0

    miss = [r for r in rows if r["stationId"] == "XX.MISS"]
    assert miss == [{"stationId": "XX.MISS", "channel": "HHE", "gapStart": t0, "gapEnd": t1}]
    lead = [r for r in rows if r["stationId"] == "XX.GAPPY" and r["channel"] == "HHZ"]
    assert lead == [
        {"stationId": "XX.GAPPY", "channel": "HHZ", "gapStart": t0, "gapEnd": t0 + 2160.0}
    ]
    assert not [r for r in rows if r["stationId"] == "XX.GOOD"]
    table = check_a_table(report, cfg)
    assert "useful stations: 1 (need >= 2) -> FAIL" in table


def test_stage_run_writes_gaps_and_records(tmp_path: Path, fake_ctx, monkeypatch) -> None:
    io = pytest.importorskip("hq_contracts.io")  # CONTRACT-01 (H4) has not landed yet
    import pandas as pd

    import hq.ingest.download as dl

    run = fake_ctx.config.run
    t0, t1 = run.window_start_s, run.window_end_s
    client = FakeClient(full(["A"], t0 - HOUR, t1 + HOUR))
    monkeypatch.setattr(dl, "default_client_factory", lambda cfg: factory(client))
    stations = pd.DataFrame(
        [
            {
                "id": "XX.A",
                "network": "XX",
                "station": "A",
                "location": "00",
                "channels": ["HHZ", "HHN", "HHE"],
                "latitude": 38.5,
                "longitude": -112.9,
                "usedInRun": True,
            }
        ]
    )
    io.write_table(stations, fake_ctx.path("stations.parquet"), "Station")
    dl.run(fake_ctx)
    gaps = io.read_table(fake_ctx.path("gaps.parquet"))
    assert list(gaps.columns) == ["stationId", "channel", "gapStart", "gapEnd"] and gaps.empty
    rec = fake_ctx.records["download"]
    assert rec["counts"]["usefulStations"] == 1 and rec["params"]["chunkS"] == 3600
    assert fake_ctx.path("download_report.json").is_file()

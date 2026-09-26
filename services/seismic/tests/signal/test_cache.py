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
    main,
    plan_chunks,
    plan_units,
    verify_cache,
)

pytestmark = pytest.mark.smoke

RATE = 10.0  # Hz; small so a day-scale test stays fast
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "showcase"
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
    whole miniSEED records do. ``errors`` is a queue of exceptions raised before serving;
    ``fail_at`` maps a request start time to an exception raised on every such request."""

    def __init__(
        self,
        coverage: dict[tuple[str, str], list[tuple[float, float]]],
        errors: list[Exception] | None = None,
        fail_station: str | None = None,
        location: str = "00",
        fail_at: dict[float, Exception] | None = None,
    ) -> None:
        self.coverage = coverage
        self.fail_at = dict(fail_at or {})
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
        if a in self.fail_at:
            raise self.fail_at[a]
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


def test_unexpected_error_is_retried_listed_and_resume_keeps_file_in_time_order(
    tmp_path: Path,
) -> None:
    cfg = make_cfg(padS=0.0, maxRetries=2)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + 4 * HOUR
    cov = full(["A"], t0, t1)
    # ObsPy raises a bare Exception on a truncated response body: not an FDSN error
    bad = FakeClient(cov, fail_at={t0 + 2 * HOUR: Exception("Cannot open file/files")})
    sleeps: list[float] = []
    with pytest.raises(DownloadIncompleteError) as err:
        run_download([sta("A")], t0, t1, cfg, tmp_path, factory(bad), sleeps=sleeps)
    failures = err.value.result.failures
    assert [(f.start, f.end) for f in failures] == [(t0 + 2 * HOUR, t0 + 3 * HOUR)]
    assert "after 3 attempts" in failures[0].error and len(sleeps) == 2
    man = json.loads((tmp_path / "mseed" / "XX.A.00.HHZ.20260910.json").read_text())
    assert [c["start"] for c in man["chunks"]] == [t0, t0 + HOUR, t0 + 3 * HOUR]  # kept

    good = FakeClient(cov)
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(good))
    assert [c[2] for c in good.calls] == [t0 + 2 * HOUR]  # only the failed hour
    heads = obspy.read(str(tmp_path / "mseed" / "XX.A.00.HHZ.20260910.mseed"), headonly=True)
    starts = [tr.stats.starttime for tr in heads]
    assert starts == sorted(starts)
    st = read_window("XX.A", t0, t1 - 1.0 / RATE, cache_dir=tmp_path).select(channel="HHZ")
    assert len(st) == 1 and st[0].stats.npts == int((t1 - t0) * RATE)
    np.testing.assert_array_equal(st[0].data, value_at(st[0].times("timestamp")))


def test_a_broken_unit_is_listed_and_the_others_still_commit(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + HOUR
    broken = tmp_path / "mseed" / "XX.B.00.HHZ.20260910.json"
    broken.parent.mkdir(parents=True)
    broken.write_text(json.dumps({"version": 99, "chunks": []}))
    with pytest.raises(DownloadIncompleteError) as err:
        run_download(
            [sta("A"), sta("B")],
            t0,
            t1,
            cfg,
            tmp_path,
            factory(FakeClient(full(["A", "B"], t0, t1))),
        )
    failures = err.value.result.failures
    assert [f.stationId for f in failures] == ["XX.B"]
    assert "unit aborted: ValueError" in failures[0].error
    assert (tmp_path / "mseed" / "XX.A.00.HHZ.20260910.mseed").is_file()


def test_missing_or_truncated_data_file_is_fetched_again(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + 2 * HOUR
    cov = full(["A"], t0, t1)
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(FakeClient(cov)))
    z = tmp_path / "mseed" / "XX.A.00.HHZ.20260910.mseed"
    man = json.loads((tmp_path / "mseed" / "XX.A.00.HHZ.20260910.json").read_text())
    assert man["fileBytes"] == z.stat().st_size

    def z_is_complete() -> None:
        st = read_window("XX.A", t0, t1 - 1.0 / RATE, cache_dir=tmp_path).select(channel="HHZ")
        assert len(st) == 1 and st[0].stats.npts == int((t1 - t0) * RATE)

    z.unlink()  # e.g. a partial copy of the cache
    again = FakeClient(cov)
    res = run_download([sta("A")], t0, t1, cfg, tmp_path, factory(again))
    assert len(again.calls) == 2 and res.counts()["staleChannelDays"] == 1
    z_is_complete()

    z.write_bytes(z.read_bytes()[:-100])  # truncated
    third = FakeClient(cov)
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(third))
    assert len(third.calls) == 2
    z_is_complete()
    assert run_download([sta("A")], t0, t1, cfg, tmp_path, no_network).counts()["requests"] == 0


def test_verify_stamps_old_manifests_and_marks_short_files_stale(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + HOUR
    cov = full(["A"], t0, t1)
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(FakeClient(cov)))
    folder = tmp_path / "mseed"
    for m in folder.glob("*.json"):  # manifests written before fileBytes existed
        doc = json.loads(m.read_text())
        del doc["fileBytes"]
        m.write_text(json.dumps(doc))
    z = folder / "XX.A.00.HHZ.20260910.mseed"
    short = obspy.read(str(z))
    short.trim(UTCDateTime(t0), UTCDateTime(t0 + 1800.0))
    short.write(str(z), format="MSEED")  # valid miniSEED, but half the manifest's samples
    (folder / "XX.A.00.HHZ.20260910.abc.part").write_bytes(b"x" * 10)

    res = verify_cache(tmp_path)
    assert res.channelDays == 3 and res.stamped == 2
    assert list(res.stale) == ["XX.A.00.HHZ.20260910"] and "samples" in res.stale[z.stem]
    assert res.stray == {"XX.A.00.HHZ.20260910.abc.part": 10}
    assert (folder / "XX.A.00.HHZ.20260910.abc.part").is_file()  # listed, never deleted
    hhn = json.loads((folder / "XX.A.00.HHN.20260910.json").read_text())
    assert hhn["fileBytes"] == (folder / "XX.A.00.HHN.20260910.mseed").stat().st_size
    assert main(["--cache-dir", str(tmp_path), "--verify"]) == 1

    repair = FakeClient(cov)
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(repair))
    assert len(repair.calls) == 1
    st = read_window("XX.A", t0, t1 - 1.0 / RATE, cache_dir=tmp_path).select(channel="HHZ")
    assert st[0].stats.npts == int((t1 - t0) * RATE)
    res = verify_cache(tmp_path)
    assert not res.stale and res.stamped == 0


def test_provisional_refetch_that_returns_less_keeps_the_cached_samples(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0, provisionalLagS=3600.0)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + 2 * HOUR
    cov = full(["A"], t0, t1)
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(FakeClient(cov)), now=t1 + 600.0)
    empty = FakeClient({})  # a transient 204 on the refetch
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(empty), now=t1 + 7200.0)
    assert [c[2] for c in empty.calls] == [t0 + HOUR]
    st = read_window("XX.A", t0, t1 - 1.0 / RATE, cache_dir=tmp_path).select(channel="HHZ")
    assert st[0].stats.npts == int((t1 - t0) * RATE)
    man = json.loads((tmp_path / "mseed" / "XX.A.00.HHZ.20260910.json").read_text())
    assert man["chunks"][-1]["status"] == "ok" and man["chunks"][-1]["provisional"] is True

    settled = FakeClient(cov)
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(settled), now=t1 + 7200.0)
    assert len(settled.calls) == 1
    man = json.loads((tmp_path / "mseed" / "XX.A.00.HHZ.20260910.json").read_text())
    assert man["chunks"][-1]["provisional"] is False


def test_backoff_max_below_base_is_rejected() -> None:
    with pytest.raises(ValueError, match="backoffMaxS"):
        make_cfg(backoffBaseS=10.0, backoffMaxS=5.0)


def test_cli_check_a_uses_used_in_run_like_the_stage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = make_cfg(padS=0.0)
    t0, t1 = T_MIDNIGHT, T_MIDNIGHT + HOUR
    run_download([sta("A")], t0, t1, cfg, tmp_path, factory(FakeClient(full(["A"], t0, t1))))
    row = {"id": "XX.A", "network": "XX", "station": "A", "location": "00"}
    rows = [
        {**row, "channels": ["HHZ", "HHN", "HHE"], "usedInRun": True},
        {
            **row,
            "id": "XX.B",
            "station": "B",
            "channels": ["HHZ", "HHN", "HHE"],
            "usedInRun": False,
        },
    ]
    stations = tmp_path / "stations.json"
    stations.write_text(json.dumps(rows))
    code = main(
        [
            "--stations",
            str(stations),
            "--config-dir",
            str(CONFIG_DIR),
            "--cache-dir",
            str(tmp_path),
            "--start",
            UTCDateTime(t0).isoformat(),
            "--end",
            UTCDateTime(t1).isoformat(),
            "--report-only",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "1 usedInRun station(s) of 2 in stations.json" in out
    assert "XX.A " in out and "XX.B " not in out


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
    # gap accounting looks stations up by NET.STA.LOC, so a bare id is never ambiguous there
    bare = StationRequest("XX.A", "XX", "A", "00", ("HHZ", "HHN", "HHE"))
    rows, report = coverage([bare], t0, t1, cfg, cache_dir=tmp_path)
    assert rows == [] and report[0].useful


def test_ambiguity_is_judged_within_the_window_days(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0)
    later = T_MIDNIGHT + 5 * 86_400.0
    cov = full(["A"], T_MIDNIGHT, later + HOUR)
    for loc, t0 in (("00", T_MIDNIGHT), ("10", later)):
        s = StationRequest(f"XX.A.{loc}", "XX", "A", loc, ("HHZ", "HHN", "HHE"))
        run_download([s], t0, t0 + HOUR, cfg, tmp_path, factory(FakeClient(cov)))
    st = read_window("XX.A", T_MIDNIGHT, T_MIDNIGHT + 600.0, cache_dir=tmp_path)
    assert len(st) == 3 and {tr.stats.location for tr in st} == {"00"}
    with pytest.raises(AmbiguousStationError):
        read_window("XX.A", T_MIDNIGHT, later + 600.0, cache_dir=tmp_path)


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
    # window start between two grid samples: a first sample 0.07 s in is not a gap ...
    assert channel_gaps([Segment("HHZ", 0.07, 99.97, dt)], t0, t1, 1.5) == ([], 0)
    # ... but one whole missing sample at the start is
    gaps, _ = channel_gaps([Segment("HHZ", 0.13, 99.93, dt)], t0, t1, 1.5)
    assert [(round(a, 2), round(b, 2)) for a, b in gaps] == [(0, 0.13)]


def test_gap_rows_and_check_a_classification(tmp_path: Path) -> None:
    cfg = make_cfg(padS=0.0, maxGapFraction=0.2, minUsefulStations=3)
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
    assert by_id["XX.FAR"].inBbox is False and by_id["XX.FAR"].useful  # reported, not scored
    assert by_id["XX.NEVER"].componentsPresent == 0 and by_id["XX.NEVER"].maxGapFraction == 1.0

    miss = [r for r in rows if r["stationId"] == "XX.MISS"]
    assert miss == [{"stationId": "XX.MISS", "channel": "HHE", "gapStart": t0, "gapEnd": t1}]
    lead = [r for r in rows if r["stationId"] == "XX.GAPPY" and r["channel"] == "HHZ"]
    assert lead == [
        {"stationId": "XX.GAPPY", "channel": "HHZ", "gapStart": t0, "gapEnd": t0 + 2160.0}
    ]
    assert not [r for r in rows if r["stationId"] == "XX.GOOD"]
    table = check_a_table(report, cfg)
    assert "useful stations: 2 (need >= 3) -> FAIL" in table


def _exercise_stage(io, fake_ctx, monkeypatch) -> None:
    import pandas as pd

    import hq.ingest.download as dl

    run = fake_ctx.config.run
    cfg = fake_ctx.config.signal.download
    t0, t1 = run.window_start_s, run.window_end_s
    cov = full(["A"], t0 - HOUR, t1 + HOUR)
    del cov[("A", "HHE")]
    client = FakeClient(cov)
    monkeypatch.setattr(dl, "default_client_factory", lambda cfg: factory(client))
    row = {
        "id": "XX.A",
        "network": "XX",
        "station": "A",
        "location": "00",
        "channels": ["HHZ", "HHN", "HHE"],
        "latitude": 38.5,
        "longitude": -112.9,
        "usedInRun": True,
    }
    unused = {**row, "id": "XX.B", "station": "B", "usedInRun": False}
    io.write_table(pd.DataFrame([row, unused]), fake_ctx.path("stations.parquet"), "Station")
    dl.run(fake_ctx)
    assert {c[0] for c in client.calls} == {"A"}  # usedInRun=False is not downloaded
    gaps = io.read_table(fake_ctx.path("gaps.parquet"))
    assert list(gaps.columns) == ["stationId", "channel", "gapStart", "gapEnd"]
    assert gaps.to_dict(orient="records") == [
        {"stationId": "XX.A", "channel": "HHE", "gapStart": t0, "gapEnd": t1}
    ]
    rec = fake_ctx.records["download"]
    assert rec["counts"]["usefulStations"] == 0 and rec["counts"]["gapRows"] == 1
    assert rec["counts"]["requests"] == len(plan_chunks(t0 - cfg.padS, t1 + cfg.padS, cfg.chunkS))
    assert rec["params"] == cfg.model_dump(mode="json")
    report = json.loads(fake_ctx.path("download_report.json").read_text(encoding="utf-8"))
    assert report["pass"] is False and report["stations"][0]["componentsPresent"] == 2


def test_stage_run_writes_gaps_and_records(fake_ctx, monkeypatch) -> None:
    io = pytest.importorskip("hq_contracts.io")  # CONTRACT-01 (H4) has not landed yet
    _exercise_stage(io, fake_ctx, monkeypatch)


def test_stage_run_with_stand_in_table_io(fake_ctx, monkeypatch) -> None:
    """Same stage flow against a parquet stand-in for ``hq_contracts.io`` (docs/02 section 2)."""
    import sys
    import types

    import pandas as pd

    stub = types.ModuleType("hq_contracts.io")
    written: dict[str, str] = {}

    def write_table(df: pd.DataFrame, path: Path, model_name: str) -> None:
        written[Path(path).name] = model_name
        df.to_parquet(path)

    stub.write_table = write_table  # type: ignore[attr-defined]
    stub.read_table = pd.read_parquet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hq_contracts.io", stub)
    _exercise_stage(stub, fake_ctx, monkeypatch)
    assert written["gaps.parquet"] == "Gap"

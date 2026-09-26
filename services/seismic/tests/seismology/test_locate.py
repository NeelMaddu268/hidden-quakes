"""LOC-04: locate() and stage ``locate`` (hq.locate, hq.locate.run, hq.locate.diagnostics).

Synthetic data built here on the LOC-02 test geometry (12 stations, 3 borehole sensors): two
events at known hypocentres, exact 1D layered pick times plus seeded noise, one planted outlier,
a station with no pick, an unassociated pick and a station not used in the run. Offline, seeded.
"""

import dataclasses
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hq_contracts.io import columns_for, dtypes_for, from_frame, read_table, write_table
from hq_contracts.models import SeismicEvent

from hq.associate.result import EVENT_DTYPES as ASSOC_EVENT_DTYPES
from hq.associate.result import PICK_DTYPES as ASSOC_PICK_DTYPES
from hq.associate.result import AssocResult, empty_result, typed_frame
from hq.config.seismology import SeismologyConfig
from hq.locate import LocateDetails, locate, locate_detailed
from hq.locate.coords import from_enu, to_enu
from hq.locate.result import ARRIVAL_DTYPES, EVENT_COLUMNS, FLAG_DTYPES, STATIC_DTYPES
from hq.runs import resolve_stage, stage_spec

RUN_ID = "test-run"  # make_ctx's run id
SEED = 20260926
NOISE = {"P": 0.005, "S": 0.01}
OUTLIER_S = 0.6  # added to one P pick of the first event
# (e, n, elevM, seconds after windowStart, S stations, P stations (None: all))
EVENTS = (
    (500.0, -300.0, -2000.0, 3600.0, None, None),
    (-1200.0, 900.0, -3200.0, 7200.0, ("T.S01", "T.S03"), [f"T.S0{i}" for i in range(1, 9)]
     + ["T.B01", "T.B02", "T.B03"]),
)
NO_PICK_STATION = "T.S09"  # no pick in the second event: predicted-only rows
OUTLIER_STATION = "T.S05"
UNUSED_STATION = ("T.X99", 30000.0, 0.0, 1600.0)  # usedInRun false, beyond the table reach


def _config(loc02: Any) -> SeismologyConfig:
    raw = loc02.test_config().model_dump(mode="json")
    raw["diagnostics"]["datumCheck"]["elevM"] = [-2000.0]  # one datum-check event keeps it fast
    return SeismologyConfig.model_validate(raw)


def _stations(loc02: Any, run: Any) -> pd.DataFrame:
    st = loc02.stations(run.origin.elevM)
    extra = {"id": UNUSED_STATION[0], "surfaceElevM": UNUSED_STATION[3], "sensorDepthM": 0.0,
             "sensorElevM": UNUSED_STATION[3], "kind": "surface", "enu_e": UNUSED_STATION[1],
             "enu_n": UNUSED_STATION[2], "enu_u": UNUSED_STATION[3] - run.origin.elevM,
             "preprocessProfile": "surface"}
    st = pd.concat([st, pd.DataFrame([extra])], ignore_index=True)
    lat, lon, _ = from_enu(st["enu_e"], st["enu_n"], st["enu_u"], run.origin)
    return st.assign(latitude=lat, longitude=lon, usedInRun=st["id"] != UNUSED_STATION[0])


def _picks(loc02: Any, locator: Any, run: Any, picker: str) -> tuple[pd.DataFrame, list[list[str]]]:
    rng = np.random.default_rng(SEED)
    frames, members = [], []
    for k, (e, n, z, dt, s_st, p_st) in enumerate(EVENTS):
        p = loc02.exact_picks(locator, e, n, z, run.window_start_s + dt, s_stations=s_st,
                              p_stations=p_st, prob=0.8)
        p["t"] = p["t"] + rng.normal(0.0, 1.0, len(p)) * p["phase"].map(NOISE)
        if k == 0:
            hit = (p["stationId"] == OUTLIER_STATION) & (p["phase"] == "P")
            p.loc[hit, "t"] += OUTLIER_S
        p["prob"] = np.where(p["phase"] == "P", 0.8, 0.6)
        frames.append(p)
    picks = pd.concat(frames, ignore_index=True)
    stray = {"stationId": "T.S02", "phase": "P", "t": run.window_start_s + 100.0, "prob": 0.9}
    picks = pd.concat([picks, pd.DataFrame([stray])], ignore_index=True)
    prefix = "stalta" if picker == "stalta" else picker
    picks["id"] = [f"{prefix}:{s}:{ph}:{t:.3f}" for s, ph, t in
                   zip(picks["stationId"], picks["phase"], picks["t"], strict=True)]
    picks["picker"] = picker
    picks["eventId"] = None
    start = 0
    for f in frames:
        members.append(picks["id"].iloc[start : start + len(f)].tolist())
        start += len(f)
    return picks, members


def _assoc(picks: pd.DataFrame, members: list[list[str]], run: Any) -> AssocResult:
    rows, links = [], []
    for k, ((e, n, z, dt, _, _), ids) in enumerate(zip(EVENTS, members, strict=True)):
        aid = f"assoc-{k:06d}"
        # PyOcto-like preliminary position: 400 m east and 300 m shallower than the truth.
        lat, lon, _ = from_enu(e + 400.0, n, z + 300.0 - run.origin.elevM, run.origin)
        sub = picks[picks["id"].isin(ids)]
        rows.append({"assocId": aid, "t": run.window_start_s + dt, "latitude": float(lat),
                     "longitude": float(lon), "elevM": z + 300.0, "nPicks": len(ids),
                     "nP": int((sub["phase"] == "P").sum()), "nS": int((sub["phase"] == "S").sum())})
        links += [{"assocId": aid, "pickId": p} for p in ids]
    events = typed_frame({c: [r[c] for r in rows] for c in ASSOC_EVENT_DTYPES}, ASSOC_EVENT_DTYPES)
    picks_out = typed_frame({c: [r[c] for r in links] for c in ASSOC_PICK_DTYPES},
                            ASSOC_PICK_DTYPES)
    return AssocResult(events, picks_out)


@pytest.fixture(scope="module")
def world(loc02: Any) -> dict[str, Any]:
    from hq.locate.locator import build_locator

    run = loc02.run_section()
    cfg = _config(loc02)
    locator = build_locator(loc02.setup(cfg, run))
    stations = _stations(loc02, run)
    picks, members = _picks(loc02, locator, run, "phasenet:test")
    return {"run": run, "cfg": cfg, "stations": stations, "picks": picks, "members": members,
            "assoc": _assoc(picks, members, run), "cache": loc02.cache_dir}


@pytest.fixture(scope="module")
def located(world: dict[str, Any]) -> LocateDetails:
    return locate_detailed(world["assoc"], world["picks"], world["stations"], world["cfg"],
                           world["run"], run_id=RUN_ID, cache_dir=world["cache"])


def _dtypes(frame: pd.DataFrame) -> dict[str, str]:
    return {c: str(d) for c, d in frame.dtypes.items()}


def _want(dtypes: dict[str, str]) -> dict[str, str]:
    return {c: str(pd.Series([], dtype=d).dtype) for c, d in dtypes.items()}


# --- locate(): tables ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_events_are_contract_shaped(world: dict[str, Any], located: LocateDetails) -> None:
    ev = located.result.events
    excluded = ("tier", "tierReasons", "catalogMatch", "magnitude")
    assert list(ev.columns) == [c for c in columns_for(SeismicEvent)
                                if c.split("_")[0] not in excluded]
    assert _dtypes(ev) == {c: dtypes_for(SeismicEvent)[c] for c in EVENT_COLUMNS}
    assert list(ev["id"]) == [f"hq-{RUN_ID}-{k:06d}" for k in range(len(EVENTS))]
    assert ev["t"].is_monotonic_increasing
    assert (ev["runId"] == RUN_ID).all() and (ev["source"] == "hq-pipeline").all()
    assert (ev["revealOrder"] == -1).all() and (ev["quality_method"] == "grid1d").all()
    assert not ev["quality_statics"].any()
    run = world["run"]
    np.testing.assert_allclose(ev["depthKm"], (run.refSurfaceElevM - ev["elevM"]) / 1000.0)
    e, n, u = to_enu(ev["latitude"], ev["longitude"], ev["elevM"], run.origin)
    np.testing.assert_allclose(np.c_[e, n, u], ev[["enu_e", "enu_n", "enu_u"]], atol=0.01)
    for (te, tn, tz, *_), row in zip(EVENTS, ev.itertuples(index=False), strict=True):
        assert np.hypot(row.enu_e - te, row.enu_n - tn) < 150.0 and abs(row.elevM - tz) < 150.0
    # Rows are SeismicEvents once LOC-06 adds the tier fields.
    full = ev.assign(tier="C", tierReasons=[[] for _ in range(len(ev))])
    for col, dtype in dtypes_for(SeismicEvent).items():
        if col not in full:
            full[col] = pd.Series([None] * len(ev), dtype=dtype)
    assert [m.id for m in from_frame(full, SeismicEvent)] == list(ev["id"])


@pytest.mark.smoke
def test_pick_ids_outliers_and_mean_prob(world: dict[str, Any], located: LocateDetails) -> None:
    ev = located.result.events
    picks = world["picks"].set_index("id")
    first = ev.iloc[0]
    outlier = picks[(picks["stationId"] == OUTLIER_STATION) & (picks["phase"] == "P")
                    & picks.index.isin(world["members"][0])].index[0]
    assert outlier not in list(first["pickIds"])
    assert set(first["pickIds"]) == set(world["members"][0]) - {outlier}
    for row in ev.itertuples(index=False):
        assert row.meanPickProb == pytest.approx(picks.loc[list(row.pickIds), "prob"].mean())
    arr = located.result.arrivals
    row = arr[arr["pickId"] == outlier].iloc[0]
    assert not row["usedInLocation"] and row["residualS"] > 0.3
    assert located.flags.loc[0, "nDroppedPicks"] == 1


@pytest.mark.smoke
def test_arrivals_and_statics_tables(located: LocateDetails) -> None:
    arr = located.result.arrivals
    assert _dtypes(arr) == _want(ARRIVAL_DTYPES)
    n_used_stations = len(located.stations)
    assert len(arr) == len(EVENTS) * n_used_stations * 2
    assert not arr.duplicated(["eventId", "stationId", "phase"]).any()
    assert np.isfinite(arr["tPred"]).all()
    assert UNUSED_STATION[0] not in set(arr["stationId"])
    with_pick = arr["pickId"].notna()
    np.testing.assert_allclose(arr.loc[with_pick, "residualS"],
                               arr.loc[with_pick, "tObs"] - arr.loc[with_pick, "tPred"], atol=1e-9)
    blank = arr[(arr["eventId"] == located.result.events["id"].iloc[1])
                & (arr["stationId"] == NO_PICK_STATION)]
    assert len(blank) == 2 and blank["tObs"].isna().all() and blank["residualS"].isna().all()
    assert not blank["usedInLocation"].any()
    st = located.result.statics
    assert _dtypes(st) == _want(STATIC_DTYPES)
    assert len(st) == n_used_stations * 2 and (st["staticS"] == 0.0).all()
    used = arr[arr["usedInLocation"]].groupby(["stationId", "phase"]).size()
    got = st.set_index(["stationId", "phase"])["nEvents"]
    assert got.to_dict() == used.reindex(got.index, fill_value=0).to_dict()
    flags = located.flags
    assert _dtypes(flags) == _want(FLAG_DTYPES)
    assert list(flags["eventId"]) == list(located.result.events["id"])
    assert list(flags["assocId"]) == list(located.assoc_ids)
    assert located.counts["events"] == len(EVENTS)
    assert located.velocity_model["name"] and located.record["locate"]["runId"] == RUN_ID


@pytest.mark.smoke
def test_zero_events_give_typed_zero_row_tables(
    world: dict[str, Any], located: LocateDetails
) -> None:
    out = locate(empty_result(), world["picks"], world["stations"], world["cfg"], world["run"],
                 cache_dir=world["cache"])
    for name in ("events", "arrivals", "statics"):
        frame, full = getattr(out, name), getattr(located.result, name)
        assert len(frame) == 0 and _dtypes(frame) == _dtypes(full), name


@pytest.mark.smoke
def test_stalta_labelled_picks_locate_unchanged(
    world: dict[str, Any], located: LocateDetails, loc02: Any
) -> None:
    from hq.locate.locator import build_locator

    locator = build_locator(loc02.setup(world["cfg"], world["run"]))
    picks, members = _picks(loc02, locator, world["run"], "stalta")
    assoc = _assoc(picks, members, world["run"])
    one = AssocResult(assoc.events.iloc[:1].reset_index(drop=True),
                      assoc.picks[assoc.picks["assocId"] == "assoc-000000"].reset_index(drop=True))
    out = locate(one, picks, world["stations"], world["cfg"], world["run"],
                 cache_dir=world["cache"])
    ref = located.result.events.iloc[0]
    got = out.events.iloc[0]
    for col in ("t", "enu_e", "enu_n", "elevM", "quality_rmsS", "quality_vErrM"):
        assert got[col] == pytest.approx(ref[col], abs=1e-9), col
    assert all(p.startswith("stalta:") for p in got["pickIds"])
    assert got["id"] == f"hq-{world['run'].name}-000000"  # docs/02 call: run.name stands in


@pytest.mark.smoke
def test_inputs_fail_loudly(world: dict[str, Any]) -> None:
    args = (world["stations"], world["cfg"], world["run"])
    missing = world["picks"][~world["picks"]["id"].isin(world["members"][0][:1])]
    with pytest.raises(ValueError, match="not in the picks table"):
        locate(world["assoc"], missing, *args, cache_dir=world["cache"])
    moved = world["stations"].assign(enu_e=world["stations"]["enu_e"] + 5.0)
    with pytest.raises(ValueError, match="stored enu differs"):
        locate(world["assoc"], world["picks"], moved, world["cfg"], world["run"],
               cache_dir=world["cache"])


# --- stage and diagnostics ----------------------------------------------------------------------


def _stage(world: dict[str, Any], make_ctx: Any) -> Any:
    # The session's LOC-02 table cache, so the stage loads the tables instead of solving them.
    ctx = dataclasses.replace(make_ctx(world["run"], world["cfg"]), cache_dir=world["cache"])
    picks = world["picks"]
    write_table(picks, ctx.path(world["cfg"].associator.picksTable), "Pick")
    write_table(world["stations"], ctx.path("stations.parquet"), "Station")
    write_table(world["assoc"].events, ctx.path("assoc_events.parquet"), "AssocEvent")
    write_table(world["assoc"].picks, ctx.path("assoc_picks.parquet"), "AssocPick")
    importlib.import_module("hq.locate.run").run(ctx)
    return ctx


@pytest.mark.smoke
def test_stage_writes_tables_report_and_record(
    world: dict[str, Any], located: LocateDetails, make_ctx: Any
) -> None:
    ctx = _stage(world, make_ctx)
    models = {"events_located.parquet": "LocatedEvent", "arrivals.parquet": "Arrival",
              "statics.parquet": "StationStatic", "locate_flags.parquet": "LocateFlags"}
    for name, model in models.items():
        frame = read_table(ctx.path(name))
        assert frame.attrs["model"] == model and len(frame) > 0, name
    ev = read_table(ctx.path("events_located.parquet"))
    pd.testing.assert_series_equal(ev["elevM"], located.result.events["elevM"])
    assert not list(ctx.run_dir.glob("*.part"))
    assert [r["field"] for r in ctx.records] == ["velocityModel", None]
    velocity, locator = ctx.records
    assert velocity["params"]["name"] == located.velocity_model["name"]
    assert locator["counts"]["events"] == len(EVENTS)
    assert locator["params"]["method"] == "grid1d" and "diagnostics" in locator["params"]

    report = ctx.path("diagnostics.md").read_text(encoding="utf-8")
    rows = {}
    for line in report.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0].isdigit():
            rows[int(cells[0])] = cells
    assert sorted(rows) == list(range(1, 8))
    for number, cells in rows.items():
        assert cells[3] and cells[4], f"row {number} lacks a result or a conclusion"
    assert rows[1][4].startswith("Not the cause")  # boreholes carry their depths
    assert rows[2][4].startswith("Not the cause")  # the datum check recovers the event
    assert "catalog.parquet is not in the run dir" in report


@pytest.mark.smoke
def test_bad_borehole_depth_fails_row_one(world: dict[str, Any], located: LocateDetails) -> None:
    from hq.locate.diagnostics import DiagnosticsInputs, row_borehole

    stations = world["stations"].copy()
    stations.loc[stations["id"] == "T.B01", "sensorDepthM"] = 0.0
    row, _ = row_borehole(DiagnosticsInputs(RUN_ID, world["run"], world["cfg"], stations,
                                            located, world["assoc"].events))
    assert row.conclusion.startswith("FAIL") and "T.B01" in row.conclusion


@pytest.mark.smoke
def test_catalog_comparison_and_row_seven_at_catalog_hypocentres(
    world: dict[str, Any], located: LocateDetails
) -> None:
    from hq.locate.diagnostics import DiagnosticsInputs, build_diagnostics

    run = world["run"]
    rows = []
    for k, (e, n, z, dt, *_) in enumerate(EVENTS):
        lat, lon, _ = from_enu(e, n, z - run.origin.elevM, run.origin)
        rows.append({"id": f"pub{k}", "t": run.window_start_s + dt, "latitude": float(lat),
                     "longitude": float(lon), "elevM": z, "depthKm": -z / 1000.0,
                     "depthDatum": "km below sea level (test)", "mag": 1.0, "magType": "ml",
                     "enu_e": e, "enu_n": n, "enu_u": z - run.origin.elevM})
    far = {**rows[0], "id": "pub-far", "t": run.window_start_s + 50.0}  # no candidate
    catalog = pd.DataFrame([*rows, far])
    errors = pd.DataFrame({"id": ["pub0", "pub1"], "horizontalErrorM": [300.0, 300.0],
                           "depthErrorM": [400.0, 400.0]})
    report = build_diagnostics(DiagnosticsInputs(
        RUN_ID, run, world["cfg"], world["stations"], located, world["assoc"].events,
        catalog=catalog, catalog_errors=errors, known_ids=("pub0", "pub1")))
    table = [line for line in report.splitlines() if line.startswith("| pub")]
    assert [line.split("|")[1].strip() for line in table] == ["pub0", "pub1"]
    assert all(line.rstrip(" |").endswith("True / True") for line in table)
    assert "No located candidate within tolerance for 1 public event(s): pub-far" in report
    assert "every compared event lies within the catalog's stated horizontal uncertainty" in report
    row7 = next(line for line in report.splitlines() if line.startswith("| 7 |"))
    assert "hypocentre fixed at the public regional catalog's for the 2 compared" in row7
    assert "Lateral structure" not in row7  # truth hypocentres: noise-level residuals only


# --- stage registry -------------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("name", ["associate", "locate", "match"])
def test_stage_resolves_to_the_stage_function(name: str) -> None:
    module = importlib.import_module(f"hq.{name}.run")  # the submodule imported explicitly
    fn = resolve_stage(stage_spec(name))
    assert fn is module.run and fn.__module__ == f"hq.{name}.run"
    assert importlib.import_module(f"hq.{name}").run is fn


SUBMODULE_FIRST = """
import importlib
for name in ("locate", "match", "associate"):
    importlib.import_module(f"hq.{name}.run")
from hq.runs import resolve_stage, stage_spec
for name in ("associate", "locate", "match"):
    fn = resolve_stage(stage_spec(name))
    assert callable(fn) and fn.__module__ == f"hq.{name}.run", (name, fn)
print("ok")
"""


@pytest.mark.parametrize("first", ["hq.locate.run", "hq.match.run", "hq.locate.coords"])
def test_fresh_interpreter_submodule_first(first: str) -> None:
    """Not smoke (a fresh interpreter per case): the re-export survives any first import."""
    code = f"import importlib; importlib.import_module({first!r})\n" + SUBMODULE_FIRST
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[2], check=False)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-2000:]

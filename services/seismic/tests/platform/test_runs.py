"""RUN-01 acceptance: the config loader composes per-lane YAML, ``RunContext.record`` writes
runtime, counts and params into the run, and the stage runner names the owner of anything that
isn't merged yet. Offline; everything lives under ``tmp_path``."""

import json
import logging
import re
import shutil
import sys
import types
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hq_contracts.models import ProcessingRun
from pydantic import BaseModel, ConfigDict, create_model

from hq import cli, runs
from hq.config import ConfigError, RunConfig, load_config
from hq.config.export import ExportConfig

pytestmark = pytest.mark.smoke

SHOWCASE_DIR = Path(__file__).resolve().parents[2] / "configs" / "showcase"
NOW = datetime(2026, 9, 10, 1, 2, tzinfo=UTC)
RUN_ID_RE = re.compile(r"^\d{8}-\d{4}-(?:[0-9a-f]{7}|nogit00)$")


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A copy of the real run.yaml + export.yaml and no lane YAMLs."""
    target = tmp_path / "config"
    target.mkdir()
    for name in ("run.yaml", "export.yaml"):
        shutil.copy(SHOWCASE_DIR / name, target / name)
    return target


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


def install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> types.ModuleType:
    """Register a throwaway module in sys.modules so lazy imports find it."""
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def signal_model(*, forbid: bool = True) -> type[BaseModel]:
    config = ConfigDict(extra="forbid") if forbid else ConfigDict(extra="ignore")
    return create_model("SignalConfig", __config__=config, profiles=(list[str], ...))


def recording_stage(stage: str, **record_kwargs: object) -> Callable[[runs.RunContext], None]:
    def run(ctx: runs.RunContext) -> None:
        ctx.record(stage, **record_kwargs)  # type: ignore[arg-type]

    return run


# --- 1-3: load_config -----------------------------------------------------------------------------


def test_load_config_without_lane_yamls(config_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="hq.config")
    cfg = load_config(config_dir)
    assert isinstance(cfg, RunConfig)
    assert cfg.run.name == "showcase"
    assert cfg.run.window_end_s > cfg.run.window_start_s
    assert isinstance(cfg.export, ExportConfig)
    assert cfg.export.modes == ["showcase"]
    assert cfg.export.evidence.maxTraces <= 16
    assert cfg.signal is None
    assert cfg.seismology is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("signal" in w and "H1" in w for w in warnings)
    assert any("seismology" in w and "H2" in w for w in warnings)
    with pytest.raises(ConfigError, match="H1 Signal"):
        cfg.section("signal")
    with pytest.raises(ConfigError, match="H2 Seismology"):
        cfg.section("seismology")
    assert cfg.section("run") is cfg.run
    with pytest.raises(ConfigError, match="unknown config section"):
        cfg.section("nope")


def test_unknown_key_in_export_yaml_is_an_error(config_dir: Path) -> None:
    export = config_dir / "export.yaml"
    export.write_text(export.read_text() + "\nbogusKnob: 1\n")
    with pytest.raises(ConfigError, match="bogusKnob"):
        load_config(config_dir)


def test_unknown_nested_key_in_export_yaml_is_an_error(config_dir: Path) -> None:
    export = config_dir / "export.yaml"
    export.write_text(
        export.read_text().replace("  preloadCount: 20", "  preloadCount: 20\n  nope: 1")
    )
    with pytest.raises(ConfigError, match="nope"):
        load_config(config_dir)


def test_missing_required_section_names_owner(config_dir: Path) -> None:
    (config_dir / "run.yaml").unlink()
    with pytest.raises(ConfigError, match=r"run\.yaml.*H2 Seismology"):
        load_config(config_dir)


def test_signal_yaml_without_module_names_h1(config_dir: Path) -> None:
    (config_dir / "signal.yaml").write_text("profiles: [surface]\n")
    with pytest.raises(ConfigError, match=r"hq\.config\.signal.*H1 Signal"):
        load_config(config_dir)


def test_signal_module_without_yaml_names_h1(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_module(monkeypatch, "hq.config.signal", SignalConfig=signal_model())
    with pytest.raises(ConfigError, match=r"signal\.yaml.*H1 Signal"):
        load_config(config_dir)


def test_signal_section_loads_when_both_exist(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_module(monkeypatch, "hq.config.signal", SignalConfig=signal_model())
    (config_dir / "signal.yaml").write_text("profiles: [surface, borehole-A]\n")
    cfg = load_config(config_dir)
    assert cfg.section("signal").profiles == ["surface", "borehole-A"]
    (config_dir / "signal.yaml").write_text("profiles: [surface]\nextra: 1\n")
    with pytest.raises(ConfigError, match="extra"):
        load_config(config_dir)


def test_lane_model_must_forbid_extra_keys(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_module(monkeypatch, "hq.config.signal", SignalConfig=signal_model(forbid=False))
    (config_dir / "signal.yaml").write_text("profiles: [surface]\n")
    with pytest.raises(ConfigError, match="extra='forbid'.*H1 Signal"):
        load_config(config_dir)


# --- 4: a dummy stage records into run.json and stages.json --------------------------------------


def test_dummy_stage_records_runtime_counts_and_params(
    config_dir: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_module(
        monkeypatch,
        "hq_test_dummy_pick",
        run=recording_stage("pick", runtime_s=1.5, counts={"picks": 3}, params={"weights": "x"}),
    )
    registry = (runs.StageSpec("pick", "hq_test_dummy_pick", "H1 Signal"),)
    ctx = runs.create_run(config_dir, data_dir, now=NOW)
    assert ctx.path("run.json").is_file()  # run_dir + run.json exist before any stage runs
    assert ctx.cache_dir == data_dir.resolve() / "cache" and ctx.cache_dir.is_dir()

    runs.run_stage(ctx, "pick", registry=registry)

    run = ProcessingRun.model_validate_json(ctx.path("run.json").read_text())
    assert run.runtimeS["pick"] == 1.5
    assert run.picker["weights"] == "x"
    stages = json.loads(ctx.path("stages.json").read_text())
    assert stages == {"pick": {"runtimeS": 1.5, "counts": {"picks": 3}}}

    # A second record merges instead of replacing.
    ctx.record("pick", runtime_s=2.0, counts={"picks": 4}, params={"model": "seisbench.PhaseNet"})
    run = ctx.read_run()
    assert run.runtimeS == {"pick": 2.0}
    assert run.picker == {"weights": "x", "model": "seisbench.PhaseNet"}


def test_record_params_field_mapping(config_dir: Path, data_dir: Path) -> None:
    ctx = runs.create_run(config_dir, data_dir, now=NOW)
    ctx.record("catalog", runtime_s=0.2, counts={"events": 43}, params={"provider": "USGS"})
    ctx.record("locate", runtime_s=3.0, counts={}, params={"grid": "1d"})
    ctx.record("locate", runtime_s=3.0, counts={}, params={"name": "v1"}, field="velocityModel")
    run = ctx.read_run()
    assert run.matching == {"provider": "USGS"}
    assert run.locator == {"grid": "1d"}
    assert run.velocityModel == {"name": "v1"}
    with pytest.raises(runs.RunError, match="no ProcessingRun field"):
        ctx.record("download", runtime_s=1.0, counts={}, params={"x": 1})
    with pytest.raises(runs.RunError, match="not a ProcessingRun params dict"):
        ctx.record("pick", runtime_s=1.0, counts={}, params={"x": 1}, field="pickerModel")
    with pytest.raises(runs.RunError, match="counts"):
        ctx.record("download", runtime_s=1.0, counts={"n": "many"})  # type: ignore[dict-item]


def test_update_run_sets_top_level_fields(config_dir: Path, data_dir: Path) -> None:
    ctx = runs.create_run(config_dir, data_dir, now=NOW)
    ctx.update_run(stationIds=["UU.FOR1"], pickerModel="seisbench.PhaseNet", pickerWeights="w")
    run = ctx.read_run()
    assert run.stationIds == ["UU.FOR1"]
    assert (run.pickerModel, run.pickerWeights) == ("seisbench.PhaseNet", "w")
    with pytest.raises(runs.RunError, match="cannot set"):
        ctx.update_run(id="other")
    with pytest.raises(runs.RunError, match="cannot set"):
        ctx.update_run(picker={"a": 1})
    with pytest.raises(runs.RunError, match="invalid update"):
        ctx.update_run(stationIds="UU.FOR1")


# --- 5: missing lane stages name their owner ------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "owner", "module"),
    [
        ("pick", "H1 Signal", "hq.pick"),  # package exists, no run()
        ("inventory", "H1 Signal", "hq.ingest.inventory"),  # module missing
        ("catalog", "H2 Seismology", "hq.match.catalog"),
        ("associate", "H2 Seismology", "hq.associate"),
        ("validate", "H4 Platform", "hq.validate"),
    ],
)
def test_missing_stage_names_owner(
    config_dir: Path, data_dir: Path, stage: str, owner: str, module: str
) -> None:
    ctx = runs.create_run(config_dir, data_dir, now=NOW)
    with pytest.raises(runs.StageMissingError) as info:
        runs.run_stage(ctx, stage)
    message = str(info.value)
    assert stage in message and owner in message and module in message
    assert "not implemented yet" in message
    assert runs.stage_status(runs.stage_spec(stage)).startswith("missing:")


def test_unknown_stage_lists_registry(config_dir: Path, data_dir: Path) -> None:
    ctx = runs.create_run(config_dir, data_dir, now=NOW)
    with pytest.raises(runs.UnknownStageError, match="inventory, catalog, download, pick"):
        runs.run_stage(ctx, "bogus")
    with pytest.raises(runs.UnknownStageError):
        runs.select_stages(["pick", "bogus"])


def test_registry_matches_pipeline_table() -> None:
    assert [s.name for s in runs.STAGES] == [
        "inventory", "catalog", "download", "pick", "baseline", "associate",
        "locate", "match", "tier", "magnitude", "validate", "export",
    ]  # fmt: skip
    assert {s.owner for s in runs.STAGES} == {"H1 Signal", "H2 Seismology", "H4 Platform"}
    assert [s.name for s in runs.select_stages(["export", "pick"])] == ["pick", "export"]


def test_run_stages_stops_at_first_failure(
    config_dir: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[str] = []
    install_module(monkeypatch, "hq_test_a", run=lambda ctx: ran.append("a"))
    install_module(monkeypatch, "hq_test_c", run=lambda ctx: ran.append("c"))
    registry = (
        runs.StageSpec("a", "hq_test_a", "H4 Platform"),
        runs.StageSpec("b", "hq_test_missing_b", "H2 Seismology"),
        runs.StageSpec("c", "hq_test_c", "H4 Platform"),
    )
    ctx = runs.create_run(config_dir, data_dir, now=NOW)
    with pytest.raises(runs.StageMissingError, match="H2 Seismology"):
        runs.run_stages(ctx, registry=registry)
    assert ran == ["a"]
    assert "a" in ctx.read_run().runtimeS  # a forgot to record: the runner did it for it


# --- 6: run ids and the initial run.json ----------------------------------------------------------


def test_run_id_format_and_initial_run_json(config_dir: Path, data_dir: Path) -> None:
    ctx = runs.create_run(config_dir, data_dir, now=NOW)
    assert RUN_ID_RE.match(ctx.run_id), ctx.run_id
    assert ctx.run_id.startswith("20260910-0102-")
    assert ctx.run_dir == data_dir.resolve() / "showcase" / "runs" / ctx.run_id
    run = ProcessingRun.model_validate_json(ctx.path("run.json").read_text())
    assert run.id == ctx.run_id
    assert run.mode == "showcase"
    assert run.createdAt == "2026-09-10T01:02:00Z"
    assert ctx.run_id.endswith(run.gitSha[:7])
    assert run.windowStart == ctx.config.run.window_start_s
    assert run.windowEnd == ctx.config.run.window_end_s
    assert run.windowLabel == "2026-09-10 00:00-24:00 UTC"
    assert run.bbox == ctx.config.run.bbox
    assert run.stationIds == [] and run.pickerModel == "" and run.pickerWeights == ""
    assert run.softwareVersions["python"].count(".") == 2
    assert run.softwareVersions["numpy"]
    assert run.runtimeS == {} and run.picker == {} and run.matching == {}
    assert run.isSynthetic is False
    with pytest.raises(runs.RunError, match="already exists"):
        runs.create_run(config_dir, data_dir, now=NOW)
    with pytest.raises(ValueError, match="timezone-aware"):
        runs.create_run(config_dir, data_dir, now=NOW.replace(tzinfo=None))
    with pytest.raises(runs.RunError, match="mode"):
        runs.create_run(config_dir, data_dir, "bogus", now=NOW)  # type: ignore[arg-type]


def test_git_sha_outside_a_repo_is_nogit(tmp_path: Path) -> None:
    assert runs.git_sha(tmp_path) == "nogit00"


@pytest.mark.parametrize(
    ("start", "end", "label"),
    [
        ("2026-09-10T00:00:00Z", "2026-09-11T00:00:00Z", "2026-09-10 00:00-24:00 UTC"),
        ("2026-09-10T06:30:00Z", "2026-09-10T08:30:00Z", "2026-09-10 06:30-08:30 UTC"),
        ("2026-09-10T22:00:00Z", "2026-09-11T02:00:00Z", "2026-09-10 22:00-2026-09-11 02:00 UTC"),
    ],
)
def test_window_label(start: str, end: str, label: str) -> None:
    assert runs.window_label(datetime.fromisoformat(start), datetime.fromisoformat(end)) == label


def test_load_run_finds_existing_run(config_dir: Path, data_dir: Path) -> None:
    created = runs.create_run(config_dir, data_dir, now=NOW)
    loaded = runs.load_run(created.run_id, data_dir, config_dir)
    assert loaded.run_dir == created.run_dir
    assert loaded.cache_dir == created.cache_dir
    assert loaded.config.run == created.config.run
    with pytest.raises(runs.RunError, match="not found"):
        runs.load_run("20260910-0000-0000000", data_dir, config_dir)


# --- 7: the CLI -----------------------------------------------------------------------------------


def test_cli_stages_lists_registry(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["stages"]) == 0
    out = capsys.readouterr().out
    for spec in runs.STAGES:
        assert spec.name in out and spec.module in out and spec.owner in out
    assert "missing: hq.pick has no run(ctx)" in out
    assert "missing: module hq.ingest.inventory not found" in out


def test_cli_run_and_stage_with_a_fake_pick(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def fake_pick(ctx: runs.RunContext) -> None:
        calls.append(ctx.run_id)
        ctx.record("pick", runtime_s=0.1, counts={"picks": 1})

    install_module(monkeypatch, "hq.pick", run=fake_pick)
    data = tmp_path / "data-ok"
    assert cli.main(["run", str(config_dir), "--data-dir", str(data), "--stages", "pick"]) == 0
    run_id = capsys.readouterr().out.strip()
    assert RUN_ID_RE.match(run_id) and calls == [run_id]
    run_dir = data / "showcase" / "runs" / run_id
    assert ProcessingRun.model_validate_json((run_dir / "run.json").read_text()).runtimeS == {
        "pick": 0.1
    }

    argv = [
        "stage",
        "pick",
        "--run",
        run_id,
        "--config-dir",
        str(config_dir),
        "--data-dir",
        str(data),
    ]
    assert cli.main(argv) == 0
    assert calls == [run_id, run_id]
    assert cli.main(["stage", "bogus", "--run", run_id, "--data-dir", str(data)]) == 1
    assert (
        cli.main(
            [
                "stage",
                "pick",
                "--run",
                "nope",
                "--config-dir",
                str(config_dir),
                "--data-dir",
                str(data),
            ]
        )
        == 1
    )


def test_cli_run_fails_on_missing_stage(
    config_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR)
    data = tmp_path / "data-fail"
    assert cli.main(["run", str(config_dir), "--data-dir", str(data), "--stages", "validate"]) == 1
    errors = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert "stage 'validate' is not implemented yet" in errors and "H4 Platform" in errors
    assert cli.main(["run", str(config_dir), "--data-dir", str(data), "--stages", "bogus"]) == 1
    assert list((data / "showcase" / "runs").iterdir()) != []  # the failed run dir stays for reruns


def test_cli_data_dir_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    cfg = repo / "services" / "seismic" / "configs" / "showcase"
    cfg.mkdir(parents=True)
    monkeypatch.delenv(cli.DATA_DIR_ENV, raising=False)
    with pytest.raises(ConfigError, match="no .git found"):
        cli.resolve_data_dir(None, cfg)
    (repo / ".git").write_text("gitdir: elsewhere\n")  # worktrees keep .git as a file
    assert cli.resolve_data_dir(None, cfg) == repo / "data"
    monkeypatch.setenv(cli.DATA_DIR_ENV, str(tmp_path / "env-data"))
    assert cli.resolve_data_dir(None, cfg) == (tmp_path / "env-data").resolve()
    assert cli.resolve_data_dir(str(tmp_path / "flag"), cfg) == (tmp_path / "flag").resolve()
    assert cli.parse_stages(" pick, associate ,") == ["pick", "associate"]
    assert cli.parse_stages(None) is None
    with pytest.raises(ConfigError):
        cli.parse_stages(" , ")

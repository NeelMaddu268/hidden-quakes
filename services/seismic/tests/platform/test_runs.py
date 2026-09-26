"""RUN-01 acceptance: the config loader composes per-lane YAML, ``RunContext.record`` writes
runtime, counts and params into the run, and the stage runner names the owner of anything that
isn't merged yet. Offline; everything lives under ``tmp_path``.

Every test passes whether or not H1's / H2's config modules and YAMLs are merged: ``config_dir``
removes the lane YAMLs and blocks the lane modules, "present" cases inject fake modules, and
missing-stage cases use private registries or blocked modules, never the real lane packages."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import types
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hq_contracts.models import ProcessingRun
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from hq import cli, runs
from hq.config import SECTIONS, ConfigError, RunConfig, lax_models, load_config
from hq.config.export import EvidenceConfig, ExportConfig
from hq.config.validate import ValidateConfig

pytestmark = pytest.mark.smoke

SHOWCASE_DIR = Path(__file__).resolve().parents[2] / "configs" / "showcase"
NOW = datetime(2026, 9, 10, 1, 2, tzinfo=UTC)
RUN_ID_RE = re.compile(r"^\d{8}-\d{4}-(?:[0-9a-f]{7}|nogit00)$")
LANE_SECTIONS = tuple(spec for spec in SECTIONS if not spec.required)  # signal, seismology


@pytest.fixture
def showcase_config(tmp_path: Path) -> Path:
    """A copy of every YAML in configs/showcase, exactly as ``hq run configs/showcase`` sees it."""
    target = tmp_path / "config"
    target.mkdir()
    for path in sorted(SHOWCASE_DIR.glob("*.yaml")):
        shutil.copy(path, target / path.name)
    return target


@pytest.fixture
def config_dir(showcase_config: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """run.yaml + export.yaml only, lane modules blocked: the "nothing merged yet" state."""
    for spec in LANE_SECTIONS:
        (showcase_config / spec.filename).unlink(missing_ok=True)
        monkeypatch.setitem(sys.modules, spec.module, None)  # import -> ModuleNotFoundError
    return showcase_config


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


def lane_model(name: str, *, forbid: bool = True, **fields: object) -> type[BaseModel]:
    config = ConfigDict(extra="forbid") if forbid else ConfigDict(extra="ignore")
    return create_model(name, __config__=config, **fields)  # type: ignore[call-overload]


def signal_model(*, forbid: bool = True) -> type[BaseModel]:
    return lane_model("SignalConfig", forbid=forbid, profiles=(list[str], ...))


def recording_stage(stage: str, **record_kwargs: object) -> Callable[[runs.RunContext], None]:
    def run(ctx: runs.RunContext) -> None:
        ctx.record(stage, **record_kwargs)  # type: ignore[arg-type]

    return run


# --- 1-3: load_config -----------------------------------------------------------------------------


def test_load_config_without_lane_sections(
    config_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="hq.config")
    cfg = load_config(config_dir)
    assert isinstance(cfg, RunConfig)
    assert cfg.run.name == "showcase"
    assert cfg.run.window_end_s > cfg.run.window_start_s
    assert isinstance(cfg.export, ExportConfig)
    assert cfg.export.modes == ["showcase"]
    assert cfg.export.evidence.maxTraces <= 16
    assert isinstance(cfg.validate, ValidateConfig)
    assert cfg.validate.nullTest.nShuffles >= 2
    assert cfg.signal is None
    assert cfg.seismology is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("signal" in w and "H1 Signal" in w for w in warnings)
    assert any("seismology" in w and "H2 Seismology" in w for w in warnings)
    with pytest.raises(ConfigError, match="H1 Signal"):
        cfg.section("signal")
    with pytest.raises(ConfigError, match="H2 Seismology"):
        cfg.section("seismology")
    assert cfg.section("run") is cfg.run
    assert cfg.section("validate") is cfg.validate
    with pytest.raises(ConfigError, match="unknown config section"):
        cfg.section("nope")


def test_load_config_on_the_real_showcase_dir(showcase_config: Path) -> None:
    """Whatever lanes have merged, configs/showcase loads; each lane YAML implies its section."""
    cfg = load_config(showcase_config)
    assert cfg.run.name == "showcase"
    assert isinstance(cfg.export, ExportConfig)
    for spec in LANE_SECTIONS:
        loaded = getattr(cfg, spec.name) is not None
        assert loaded == (SHOWCASE_DIR / spec.filename).is_file(), spec.name


def test_all_four_sections_load_with_fake_lane_modules(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_module(monkeypatch, "hq.config.signal", SignalConfig=signal_model())
    install_module(
        monkeypatch,
        "hq.config.seismology",
        SeismologyConfig=lane_model("SeismologyConfig", velocityModel=(str, ...)),
    )
    (config_dir / "signal.yaml").write_text("profiles: [surface, borehole-A]\n")
    (config_dir / "seismology.yaml").write_text("velocityModel: utah1d\n")
    cfg = load_config(config_dir)
    assert cfg.section("signal").profiles == ["surface", "borehole-A"]
    assert cfg.section("seismology").velocityModel == "utah1d"
    assert cfg.section("run") is cfg.run and cfg.section("export") is cfg.export
    assert all(getattr(cfg, spec.name) is not None for spec in SECTIONS)


def test_unknown_key_in_export_yaml_is_an_error(config_dir: Path) -> None:
    export = config_dir / "export.yaml"
    export.write_text(export.read_text() + "\nbogusKnob: 1\n")
    with pytest.raises(ConfigError, match="bogusKnob"):
        load_config(config_dir)


def test_unknown_nested_key_in_export_yaml_is_an_error(config_dir: Path) -> None:
    export = config_dir / "export.yaml"
    text = export.read_text()
    assert "  preloadCount: 20" in text
    export.write_text(text.replace("  preloadCount: 20", "  preloadCount: 20\n  nope: 1"))
    with pytest.raises(ConfigError, match="nope"):
        load_config(config_dir)


def test_missing_required_section_names_owner(config_dir: Path) -> None:
    (config_dir / "run.yaml").unlink()
    with pytest.raises(ConfigError, match=r"run\.yaml.*H2 Seismology"):
        load_config(config_dir)


def test_missing_validate_yaml_names_h4(config_dir: Path) -> None:
    (config_dir / "validate.yaml").unlink()
    with pytest.raises(ConfigError, match=r"validate\.yaml.*H4 Platform"):
        load_config(config_dir)


def test_unknown_key_in_validate_yaml_is_an_error(config_dir: Path) -> None:
    validate = config_dir / "validate.yaml"
    text = validate.read_text()
    assert "  shiftS: 30.0" in text
    validate.write_text(text.replace("  shiftS: 30.0", "  shiftS: 30.0\n  shiftSec: 1"))
    with pytest.raises(ConfigError, match="shiftSec"):
        load_config(config_dir)
    validate.write_text(text.replace("  nShuffles: 20", "  nShuffles: 1"))
    with pytest.raises(ConfigError, match="nShuffles"):
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
    with pytest.raises(ConfigError, match="SignalConfig must set extra='forbid'.*H1 Signal"):
        load_config(config_dir)


def test_nested_lane_model_must_forbid_extra_keys(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = lane_model("Profile", forbid=False, bandHz=(tuple[float, float], ...))
    signal = lane_model(
        "SignalConfig",
        profiles=(dict[str, profile], ...),  # type: ignore[valid-type]
        fallback=(profile | None, None),  # type: ignore[valid-type]
    )
    assert [cls.__name__ for cls in lax_models(signal)] == ["Profile"]
    install_module(monkeypatch, "hq.config.signal", SignalConfig=signal)
    (config_dir / "signal.yaml").write_text("profiles: {surface: {bandHz: [2, 20]}}\n")
    with pytest.raises(ConfigError, match=r"SignalConfig: Profile must set extra='forbid'.*H1"):
        load_config(config_dir)
    assert lax_models(ExportConfig) == []


def test_export_config_snippet_length_bounds() -> None:
    assert EvidenceConfig().beforeS + EvidenceConfig().afterS >= EvidenceConfig().minLengthS
    with pytest.raises(ValidationError, match="minLengthS"):
        EvidenceConfig(beforeS=1.0, afterS=1.0)
    with pytest.raises(ValidationError, match="maxLengthS"):
        EvidenceConfig(beforeS=4.0, afterS=5.0)
    with pytest.raises(ValidationError, match="must not exceed"):
        EvidenceConfig(minLengthS=9.0, maxLengthS=8.0)
    with pytest.raises(ValidationError, match="Nyquist"):
        EvidenceConfig(bandHz=(2.0, 60.0))


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
    assert not list(ctx.run_dir.glob("*.tmp"))  # atomic writes leave nothing behind

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
    # H1's stages share `picker` and nest under their own key (REQ-H1-2), like catalog -> matching.
    ctx.record("inventory", runtime_s=0.5, counts={"stations": 12}, params={"inventory": {"q": 1}})
    ctx.record("pick", runtime_s=9.0, counts={"picks": 100}, params={"profile": "surface"})
    ctx.record("baseline", runtime_s=1.0, counts={}, params={"baseline": {"sta": 0.5}})
    assert ctx.read_run().picker == {
        "inventory": {"q": 1},
        "profile": "surface",
        "baseline": {"sta": 0.5},
    }
    with pytest.raises(runs.RunError, match="no ProcessingRun field"):
        ctx.record("validate", runtime_s=1.0, counts={}, params={"x": 1})
    with pytest.raises(runs.RunError, match="not a ProcessingRun params dict"):
        ctx.record("pick", runtime_s=1.0, counts={}, params={"x": 1}, field="pickerModel")
    with pytest.raises(runs.UnknownStageError, match="bogus"):
        ctx.record("bogus", runtime_s=1.0, counts={})
    for bad in ({"n": "many"}, {"n": 3.0}, {"n": True}):
        with pytest.raises(runs.RunError, match="counts"):
            ctx.record("download", runtime_s=1.0, counts=bad)  # type: ignore[arg-type]
    assert set(ctx.read_run().runtimeS) == {"catalog", "locate", "inventory", "pick", "baseline"}


def test_update_run_allowlist(config_dir: Path, data_dir: Path) -> None:
    assert set(runs.UPDATABLE_FIELDS) <= set(ProcessingRun.model_fields)
    ctx = runs.create_run(config_dir, data_dir, now=NOW)
    ctx.update_run(stationIds=["UU.FOR1"], pickerModel="seisbench.PhaseNet", pickerWeights="w")
    ctx.update_run(softwareVersions={**ctx.read_run().softwareVersions, "extra": "1.0"})
    run = ctx.read_run()
    assert run.stationIds == ["UU.FOR1"]
    assert (run.pickerModel, run.pickerWeights) == ("seisbench.PhaseNet", "w")
    assert run.softwareVersions["extra"] == "1.0"
    for rejected in (
        {"id": "other"},
        {"isSynthetic": True},
        {"gitSha": "deadbeef"},
        {"mode": "live"},
        {"createdAt": "2026-01-01T00:00:00Z"},
        {"windowStart": 0.0},
        {"picker": {"a": 1}},
        {"runtimeS": {}},
    ):
        with pytest.raises(runs.RunError, match="cannot set"):
            ctx.update_run(**rejected)
    with pytest.raises(runs.RunError, match="invalid update"):
        ctx.update_run(stationIds="UU.FOR1")
    assert ctx.read_run() == run  # rejected calls changed nothing


# --- 5: missing lane stages name their owner ------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "module", "owner", "reason"),
    [
        ("inventory", "hq_test_missing_inventory", "H1 Signal", "module hq_test_missing_inventory not found"),
        ("catalog", "hq_test_missing_catalog", "H2 Seismology", "module hq_test_missing_catalog not found"),
        ("validate", "hq_test_norun_validate", "H4 Platform", "hq_test_norun_validate has no run(ctx)"),
    ],
)  # fmt: skip
def test_missing_stage_names_owner(
    config_dir: Path,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    module: str,
    owner: str,
    reason: str,
) -> None:
    if "norun" in module:
        install_module(monkeypatch, module)  # the package exists but exposes no run()
    registry = (runs.StageSpec(stage, module, owner),)
    ctx = runs.create_run(config_dir, data_dir, now=NOW)
    with pytest.raises(runs.StageMissingError) as info:
        runs.run_stage(ctx, stage, registry=registry)
    message = str(info.value)
    assert f"stage '{stage}' is not implemented yet: {reason} (owner: {owner})" == message
    assert runs.stage_status(registry[0]) == f"missing: {reason}"


def test_missing_stage_in_the_real_registry(
    config_dir: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "hq.validate", None)  # as if H4 had not merged it
    ctx = runs.create_run(config_dir, data_dir, now=NOW)
    with pytest.raises(runs.StageMissingError, match=r"'validate'.*hq\.validate.*H4 Platform"):
        runs.run_stage(ctx, "validate")


def test_broken_import_inside_a_stage_is_a_real_error(
    config_dir: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = runs.create_run(config_dir, data_dir, now=NOW)
    real_import = runs.importlib.import_module

    def broken_import(name: str, package: str | None = None) -> types.ModuleType:
        if name == "hq.pick":  # as if hq/pick/__init__.py itself did `import torch_xyz`
            raise ModuleNotFoundError("No module named 'torch_xyz'", name="torch_xyz")
        return real_import(name, package)

    monkeypatch.setattr(runs.importlib, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError, match="torch_xyz"):
        runs.run_stage(ctx, "pick")


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
    assert set(runs.STAGE_PARAM_FIELDS) <= {s.name for s in runs.STAGES}
    assert set(runs.STAGE_PARAM_FIELDS.values()) <= set(runs.PARAM_FIELDS)
    assert [s.name for s in SECTIONS] == ["run", "signal", "seismology", "export", "validate"]
    assert [s.name for s in SECTIONS if s.required] == ["run", "export", "validate"]


def test_run_stages_stops_at_first_failure(
    config_dir: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[str] = []
    install_module(monkeypatch, "hq_test_inventory", run=lambda ctx: ran.append("inventory"))
    install_module(monkeypatch, "hq_test_download", run=lambda ctx: ran.append("download"))
    registry = (
        runs.StageSpec("inventory", "hq_test_inventory", "H1 Signal"),
        runs.StageSpec("catalog", "hq_test_missing_catalog", "H2 Seismology"),
        runs.StageSpec("download", "hq_test_download", "H1 Signal"),
    )
    ctx = runs.create_run(config_dir, data_dir, now=NOW)
    with pytest.raises(runs.StageMissingError, match="H2 Seismology") as info:
        runs.run_stages(ctx, registry=registry)
    assert ran == ["inventory"]
    assert "inventory" in ctx.read_run().runtimeS  # it forgot to record: the runner did it
    notes = getattr(info.value, "__notes__", [])
    assert any(f"hq stage catalog --run {ctx.run_id}" in note for note in notes)


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
    for package in ("numpy", "pandas", "pydantic", "pyproj"):
        assert run.softwareVersions[package]
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


def test_load_run_finds_existing_run_and_checks_config(config_dir: Path, data_dir: Path) -> None:
    created = runs.create_run(config_dir, data_dir, now=NOW)
    loaded = runs.load_run(created.run_id, data_dir, config_dir)
    assert loaded.run_dir == created.run_dir
    assert loaded.cache_dir == created.cache_dir
    assert loaded.config.run == created.config.run
    with pytest.raises(runs.RunError, match="not found"):
        runs.load_run("20260910-0000-0000000", data_dir, config_dir)
    run_yaml = config_dir / "run.yaml"
    text = run_yaml.read_text()
    assert 'windowEnd: "2026-09-11T00:00:00Z"' in text
    run_yaml.write_text(
        text.replace('windowEnd: "2026-09-11T00:00:00Z"', 'windowEnd: "2026-09-10T12:00:00Z"')
    )
    with pytest.raises(runs.RunError, match="no longer matches"):
        runs.load_run(created.run_id, data_dir, config_dir)


# --- 7: the CLI -----------------------------------------------------------------------------------


def test_cli_stages_lists_registry(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["stages"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["stage", "owner", "module", "status"]
    assert len(lines) == 1 + len(runs.STAGES)
    for spec, line in zip(runs.STAGES, lines[1:], strict=True):
        assert line.startswith(spec.name)
        assert spec.owner in line and spec.module in line
        assert "  implemented" in line or "  missing: " in line


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
    run = ProcessingRun.model_validate_json((run_dir / "run.json").read_text())
    assert run.runtimeS == {"pick": 0.1}

    common = ["--config-dir", str(config_dir), "--data-dir", str(data)]
    assert cli.main(["stage", "pick", "--run", run_id, *common]) == 0
    assert calls == [run_id, run_id]
    assert cli.main(["stage", "bogus", "--run", run_id, *common]) == 1
    assert cli.main(["stage", "pick", "--run", "nope", *common]) == 1


def test_cli_run_fails_once_on_a_missing_stage(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    monkeypatch.setitem(sys.modules, "hq.validate", None)  # as if H4 had not merged it
    data = tmp_path / "data-fail"
    assert cli.main(["run", str(config_dir), "--data-dir", str(data), "--stages", "validate"]) == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1, errors  # logged once, with the rerun hint
    assert "stage 'validate' is not implemented yet" in errors[0]
    assert "H4 Platform" in errors[0] and "hq stage validate --run " in errors[0]
    assert cli.main(["run", str(config_dir), "--data-dir", str(data), "--stages", "bogus"]) == 1
    assert list((data / "showcase" / "runs").iterdir()) != []  # the failed run dir stays


def test_cli_data_dir_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    cfg = repo / "services" / "seismic" / "configs" / "showcase"
    cfg.mkdir(parents=True)
    monkeypatch.delenv(cli.DATA_DIR_ENV, raising=False)
    with pytest.raises(ConfigError, match="no .git found"):
        cli.resolve_data_dir(None, cfg)
    (repo / ".git").mkdir()
    assert cli.resolve_data_dir(None, cfg) == repo / "data"
    monkeypatch.setenv(cli.DATA_DIR_ENV, str(tmp_path / "env-data"))
    assert cli.resolve_data_dir(None, cfg) == (tmp_path / "env-data").resolve()
    assert cli.resolve_data_dir(str(tmp_path / "flag"), cfg) == (tmp_path / "flag").resolve()
    assert cli.parse_stages(" pick, associate ,") == ["pick", "associate"]
    assert cli.parse_stages(None) is None
    with pytest.raises(ConfigError):
        cli.parse_stages(" , ")


def test_cli_worktrees_share_the_main_checkout_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    monkeypatch.delenv(cli.DATA_DIR_ENV, raising=False)
    main = tmp_path / "main"
    main.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    git = ["git", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q"], cwd=main, env=env, check=True)
    subprocess.run(
        [*git, "commit", "-q", "--allow-empty", "-m", "init"], cwd=main, env=env, check=True
    )
    worktree = tmp_path / "wt"
    subprocess.run([*git, "worktree", "add", "-q", str(worktree)], cwd=main, env=env, check=True)
    cfg = worktree / "services" / "seismic" / "configs" / "showcase"
    cfg.mkdir(parents=True)
    assert (worktree / ".git").is_file()
    assert cli.resolve_data_dir(None, cfg) == main.resolve() / "data"
    # A .git file that git can't resolve falls back to that directory rather than failing.
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / ".git").write_text("gitdir: elsewhere\n")
    assert cli.resolve_data_dir(None, broken / "configs") == broken / "data"

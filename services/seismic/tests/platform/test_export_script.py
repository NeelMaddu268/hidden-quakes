"""REQ-H3-7: ``scripts/export-showcase.sh`` under ``set -u`` with no optional arguments, the way
``make export RUN=<id>`` calls it, on any Bash (macOS ships 3.2, where expanding an empty array
is an "unbound variable" error). ``uv`` is a stub on PATH that records its command lines, so
nothing here runs the pipeline; the stub answers the modes query with two bundle directories."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCRIPT = SCRIPTS_DIR / "export-showcase.sh"
CONFIG_DIR = "configs/showcase"  # the script's default --config-dir
STUB_UV = """#!/usr/bin/env bash
set -eu
case "$*" in
  *"python -c"*) echo "/bundles/showcase /bundles/mock" ;;
  *) printf 'uv %s\\n' "$*" >> "$UV_LOG" ;;
esac
"""
# Bash 4+ syntax none of scripts/*.sh may use (macOS system Bash is 3.2): mapfile/readarray,
# associative arrays, `&>>`, `;;&`, `|&`, coproc, case-changing and negative-index expansions.
BASH4_ONLY = (
    r"\bmapfile\b",
    r"\breadarray\b",
    r"\b(declare|local|typeset)\s+-[A-Za-z]*A\b",
    r"&>>",
    r";;&",
    r"\|&",
    r"\bcoproc\b",
    r"\$\{[A-Za-z_][A-Za-z_0-9]*(\[[^]]*\])?(,,?|\^\^?)\}",
    r"\$\{[A-Za-z_][A-Za-z_0-9]*\[-[0-9]+\]\}",
)

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash on PATH")


@pytest.fixture
def uv_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "uv"
    stub.write_text(STUB_UV)
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    log = tmp_path / "uv.log"
    monkeypatch.setenv("UV_LOG", str(log))
    return log


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-u", str(SCRIPT), *args], capture_output=True, text=True, timeout=60, check=False
    )


def logged(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.is_file() else []


@needs_bash
def test_no_optional_arguments_runs_the_stage_and_checks_every_mode(uv_log: Path) -> None:
    result = run_script("run-1")
    assert result.returncode == 0, result.stderr
    assert "unbound variable" not in result.stderr
    assert logged(uv_log) == [
        f"uv run hq stage export --run run-1 --config-dir {CONFIG_DIR}",
        f"uv run --quiet python -m hq.export /bundles/showcase --config-dir {CONFIG_DIR}",
        f"uv run --quiet python -m hq.export /bundles/mock --config-dir {CONFIG_DIR}",
    ]


@needs_bash
def test_optional_arguments_pass_through(uv_log: Path) -> None:
    result = run_script("run-1", "--data-dir", "/d/x", "--config-dir", "cfg/alt")
    assert result.returncode == 0, result.stderr
    assert logged(uv_log) == [
        "uv run hq stage export --run run-1 --config-dir cfg/alt --data-dir /d/x",
        "uv run --quiet python -m hq.export /bundles/showcase --config-dir cfg/alt",
        "uv run --quiet python -m hq.export /bundles/mock --config-dir cfg/alt",
    ]


@needs_bash
@pytest.mark.parametrize(
    "args", [(), ("run-1", "--bogus"), ("run-1", "--data-dir"), ("run-1", "--config-dir", "")]
)
def test_bad_arguments_exit_before_running_anything(uv_log: Path, args: tuple[str, ...]) -> None:
    result = run_script(*args)
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert logged(uv_log) == []


def test_scripts_use_no_bash4_only_syntax() -> None:
    """A static guard for macOS system Bash: no Bash 4+ builtins or expansions, and no
    ``"${array[@]}"`` in export-showcase.sh (an empty array is unbound under ``set -u``)."""
    scripts = sorted(SCRIPTS_DIR.glob("*.sh"))
    assert SCRIPT in scripts
    for script in scripts:
        text = script.read_text()
        for pattern in BASH4_ONLY:
            hit = re.search(pattern, text)
            assert hit is None, f"{script.name} uses {hit.group(0)!r} (Bash 4+)"
    assert "[@]" not in SCRIPT.read_text()

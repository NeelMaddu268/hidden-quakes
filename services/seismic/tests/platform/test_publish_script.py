"""scripts/publish-run.sh refuses to overwrite a release that holds files the local run directory
lacks (REQ-H1-4), with a stub ``gh`` that serves a recorded tarball."""

import os
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "publish-run.sh"
RUN_ID = "20260101-0000-abcdef0"

STUB_GH = """#!/usr/bin/env bash
# Stub gh: `release view` succeeds when $GH_RELEASE_TGZ exists; `release download` copies it.
set -eu
echo "gh $*" >> "$GH_LOG"
case "$1 $2" in
  "release view") [ -f "${GH_RELEASE_TGZ:-}" ] ;;
  "release download") mkdir -p "$7"; cp "$GH_RELEASE_TGZ" "$7/run-$RUN_ID_FOR_STUB.tgz" ;;
  "release upload"|"release create") exit 0 ;;
  *) echo "unexpected gh call: $*" >&2; exit 9 ;;
esac
"""


def make_repo(tmp_path: Path, local_files: list[str]) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "publish-run.sh").write_text(SCRIPT.read_text(encoding="utf-8"))
    os.chmod(repo / "scripts" / "publish-run.sh", stat.S_IRWXU)
    run_dir = repo / "data" / "showcase" / "runs" / RUN_ID
    for name in local_files:
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return repo


def make_release(tmp_path: Path, files: list[str]) -> Path:
    src = tmp_path / "release" / RUN_ID
    for name in files:
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    tgz = tmp_path / "release" / f"run-{RUN_ID}.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(src, arcname=RUN_ID)
    return tgz


def publish(tmp_path: Path, repo: Path, release: Path | None, *args: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "gh").write_text(STUB_GH)
    os.chmod(bin_dir / "gh", stat.S_IRWXU)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GH_LOG": str(tmp_path / "gh.log"),
        "GH_RELEASE_TGZ": str(release) if release else "",
        "RUN_ID_FOR_STUB": RUN_ID,
    }
    return subprocess.run(
        ["bash", "-u", str(repo / "scripts" / "publish-run.sh"), RUN_ID, *args],
        capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip


def gh_calls(tmp_path: Path) -> list[str]:
    return (tmp_path / "gh.log").read_text().splitlines()


def test_first_publish_creates_the_release(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, ["run.json", "stations.parquet"])
    result = publish(tmp_path, repo, None)
    assert result.returncode == 0, result.stderr
    assert any(c.startswith("gh release create") for c in gh_calls(tmp_path))


def test_refuses_when_the_release_holds_files_the_local_copy_lacks(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, ["run.json", "stations.parquet"])
    release = make_release(tmp_path, ["run.json", "stations.parquet", "picks.parquet", "known/picks.parquet"])
    result = publish(tmp_path, repo, release)
    assert result.returncode == 3
    assert "picks.parquet" in result.stderr and "known/picks.parquet" in result.stderr
    assert "make fetch-run" in result.stderr
    assert not any(c.startswith("gh release upload") for c in gh_calls(tmp_path))
    forced = publish(tmp_path, repo, release, "--force")
    assert forced.returncode == 0, forced.stderr
    assert any(c.startswith("gh release upload") for c in gh_calls(tmp_path))


def test_uploads_when_the_local_copy_is_a_superset(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, ["run.json", "stations.parquet", "events.parquet"])
    release = make_release(tmp_path, ["run.json", "stations.parquet"])
    result = publish(tmp_path, repo, release)
    assert result.returncode == 0, result.stderr
    assert any(c.startswith("gh release upload") for c in gh_calls(tmp_path))


def test_usage_errors(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, ["run.json"])
    missing = subprocess.run(["bash", "-u", str(repo / "scripts" / "publish-run.sh")], capture_output=True, text=True, check=False)
    assert missing.returncode == 1 and "usage" in missing.stderr
    unknown = publish(tmp_path, repo, None, "--bogus")
    assert unknown.returncode == 2

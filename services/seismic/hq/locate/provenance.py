"""Which code and software an H2 stage ran with, for that stage's ``ProcessingRun`` record.

``ProcessingRun.gitSha`` and ``softwareVersions`` are set once, when ``hq.runs.create_run`` makes
the run dir, on whichever machine and commit created it. The H2 stages often run later, from
another commit and another interpreter, so ``run.json`` alone does not say which code produced
the H2 tables. Each H2 stage therefore records its own ``provenance``: the HEAD sha of the
checkout its code sits in (``hq.runs.git_sha``), whether that checkout had uncommitted changes,
and ``hq.runs.software_versions()`` of the running interpreter. Computed once per process.
"""

import functools
import subprocess
from pathlib import Path
from typing import Any

from hq.runs import git_sha, software_versions


def _uncommitted(cwd: Path) -> bool | None:
    """True when the checkout holding ``cwd`` has uncommitted changes; None without git."""
    try:
        proc = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                              text=True, check=False)
    except FileNotFoundError:
        return None
    return bool(proc.stdout.strip()) if proc.returncode == 0 else None


@functools.lru_cache(maxsize=1)
def _provenance() -> tuple[str, bool | None, tuple[tuple[str, str], ...]]:
    here = Path(__file__).resolve().parent
    return git_sha(here), _uncommitted(here), tuple(software_versions().items())


def stage_provenance() -> dict[str, Any]:
    """``{"gitSha", "uncommittedChanges", "softwareVersions"}`` for a stage record."""
    sha, dirty, versions = _provenance()
    return {"gitSha": sha, "uncommittedChanges": dirty, "softwareVersions": dict(versions),
            "note": "the code and software this stage ran with; ProcessingRun.gitSha and "
            "softwareVersions describe the run's creation"}

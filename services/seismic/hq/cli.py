"""``hq`` command line (docs/02 §4): ``hq run``, ``hq stage``, ``hq stages``.

    hq run <config_dir> [--stages a,b,c] [--data-dir DIR] [--mode showcase|live]
    hq stage <name> --run <runId> [--config-dir DIR] [--data-dir DIR]
    hq stages

The data directory is ``--data-dir``, else ``$HQ_DATA_DIR``, else ``<checkout root>/data`` where
the root is the first ancestor of the config directory holding ``.git``; in a linked git worktree
that is the main checkout, so every worktree on a laptop shares one ``data/``. Runs live at
``<data>/<mode>/runs/<runId>``, the waveform cache at ``<data>/cache``. Logs go to stderr at INFO.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from hq import runs
from hq.config import ConfigError

log = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = "configs/showcase"
DATA_DIR_ENV = "HQ_DATA_DIR"
DATA_DIRNAME = "data"
RUN_MODES: tuple[str, ...] = ("showcase", "live")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATA_DIR_HELP = f"runs and cache root (default: ${DATA_DIR_ENV} or <repo root>/{DATA_DIRNAME})"


def configure_logging() -> None:
    """INFO to stderr with UTC timestamps (like run ids). A no-op when the root logger already
    has handlers (tests, embedding); never touches global logging state."""
    root = logging.getLogger()
    if root.handlers:
        return
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S")
    formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def git_common_dir(cwd: Path) -> Path | None:
    """The ``.git`` directory shared by every worktree of the repo at ``cwd``; None without git."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return Path(proc.stdout.strip())


def find_repo_root(start: Path) -> Path | None:
    """Root of the checkout whose ``data/`` a run uses: the first ancestor of ``start`` holding
    ``.git``. In a linked worktree (``.git`` is a file) that is the main checkout, so worktrees
    share one ``data/`` (runs and the waveform cache) instead of each getting their own."""
    for candidate in (start, *start.parents):
        git = candidate / ".git"
        if git.is_dir():
            return candidate
        if git.is_file():
            common = git_common_dir(candidate)
            return common.parent if common is not None else candidate
    return None


def resolve_data_dir(explicit: str | None, config_dir: Path) -> Path:
    if explicit:
        return Path(explicit).resolve()
    from_env = os.environ.get(DATA_DIR_ENV)
    if from_env:
        return Path(from_env).resolve()
    root = find_repo_root(config_dir.resolve())
    if root is None:
        raise ConfigError(
            f"no .git found above {config_dir}; pass --data-dir or set {DATA_DIR_ENV}"
        )
    return root / DATA_DIRNAME


def parse_stages(text: str | None) -> list[str] | None:
    """``"a,b,c"`` → ``["a", "b", "c"]``; ``None`` or blank means every stage."""
    if text is None:
        return None
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise ConfigError("--stages needs at least one stage name, e.g. --stages pick,associate")
    return names


def cmd_run(args: argparse.Namespace) -> int:
    config_dir = Path(args.config_dir).resolve()
    data_dir = resolve_data_dir(args.data_dir, config_dir)
    specs = runs.select_stages(parse_stages(args.stages))  # validate names before making a run
    ctx = runs.create_run(config_dir, data_dir, args.mode)
    log.info("run %s: %d stage(s): %s", ctx.run_id, len(specs), ", ".join(s.name for s in specs))
    runs.run_stages(ctx, [spec.name for spec in specs])
    print(ctx.run_id)
    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    runs.stage_spec(args.name)  # unknown names fail before touching the run
    config_dir = Path(args.config_dir).resolve()
    data_dir = resolve_data_dir(args.data_dir, config_dir)
    ctx = runs.load_run(args.run, data_dir, config_dir)
    runs.run_stage(ctx, args.name)
    return 0


def cmd_stages(args: argparse.Namespace) -> int:
    rows = [("stage", "owner", "module", "status")]
    rows += [(s.name, s.owner, s.module, runs.stage_status(s)) for s in runs.STAGES]
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row[:3])) + "  " + row[3])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hq", description="Hidden Quakes pipeline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="create a run and execute stages in pipeline order")
    run.add_argument("config_dir", help="directory with run.yaml, export.yaml, ...")
    run.add_argument("--stages", help="comma-separated subset, run in pipeline order")
    run.add_argument("--data-dir", help=DATA_DIR_HELP)
    run.add_argument("--mode", choices=RUN_MODES, default="showcase")
    run.set_defaults(func=cmd_run)

    stage = sub.add_parser("stage", help="rerun one stage of an existing run in place")
    stage.add_argument("name", help="stage name (see `hq stages`)")
    stage.add_argument("--run", required=True, metavar="RUN_ID")
    stage.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    stage.add_argument("--data-dir", help=DATA_DIR_HELP)
    stage.set_defaults(func=cmd_stage)

    stages = sub.add_parser("stages", help="list the stage registry, owners and what's implemented")
    stages.set_defaults(func=cmd_stages)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, runs.RunError) as exc:  # logged once, with the runner's rerun hint
        log.error("%s", "; ".join([str(exc), *getattr(exc, "__notes__", [])]))
        return 1
    except Exception:  # a stage crashed: keep the traceback (notes included), exit nonzero
        log.exception("hq %s failed", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())

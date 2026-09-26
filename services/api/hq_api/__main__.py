"""``python -m hq_api`` / ``hq-api`` / ``make api``: the live worker's command line.

    hq-api [serve] [--config services/api/config.yaml] [--host H] [--port P] [--no-scheduler]
    hq-api freeze-snapshot [--config ...] [--allow-empty]

``serve`` (the default when no command is given) runs the worker under uvicorn. Every setting
comes from ``config.yaml`` (``hq_api.config.LiveConfig``); ``--host`` and ``--port`` override
the ``server`` section for one launch, ``--no-scheduler`` serves the last persisted window
without running the pipeline (useful to inspect state on a laptop).

``freeze-snapshot`` copies the last good live window into the snapshot bundle
(``paths.snapshotDir``, ``apps/web/public/data/snapshot/``) so it can be committed (API-05,
``hq_api.snapshot``).
"""

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from hq import cli as hq_cli
from hq_api.app import create_app
from hq_api.config import (
    DEFAULT_CONFIG_FILE,
    LiveConfig,
    LiveConfigError,
    checkout_root,
    load_live_config,
)
from hq_api.runner import HqPipelineRunner
from hq_api.snapshot import FreezeError, freeze_snapshot
from hq_api.worker import LiveWorker

log = logging.getLogger(__name__)

API_DIR = Path(__file__).resolve().parent.parent  # services/api
SERVE = "serve"
FREEZE_SNAPSHOT = "freeze-snapshot"
COMMANDS = (SERVE, FREEZE_SNAPSHOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hq-api",
        description="Hidden Quakes live worker",
        epilog=f"Without a command, `{SERVE}` is assumed: `hq-api --port 8001` serves.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def with_config(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sub.add_argument(
            "--config", default=str(API_DIR / DEFAULT_CONFIG_FILE), help="live config YAML"
        )
        return sub

    serve = with_config(commands.add_parser(SERVE, help="run the worker and the API (default)"))
    serve.add_argument("--host", help="override server.host for this launch")
    serve.add_argument("--port", type=int, help="override server.port for this launch")
    serve.add_argument(
        "--no-scheduler",
        action="store_true",
        help="serve the persisted last window; do not run the pipeline",
    )
    freeze = with_config(
        commands.add_parser(
            FREEZE_SNAPSHOT,
            help="copy the last good live window into paths.snapshotDir as the snapshot bundle",
        )
    )
    freeze.add_argument(
        "--allow-empty",
        action="store_true",
        help="freeze the newest good window even when it has no candidate events",
    )
    return parser


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """``serve`` is implied when the first argument is not a command (``hq-api --port 8001``)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in COMMANDS and args[0] not in ("-h", "--help"):
        args = [SERVE, *args]
    return build_parser().parse_args(args)


def main(argv: Sequence[str] | None = None) -> int:
    hq_cli.configure_logging()
    args = parse_args(argv)
    try:
        config = load_live_config(Path(args.config))
        root = checkout_root()
    except LiveConfigError as exc:
        log.error("%s", exc)
        return 1
    if args.command == FREEZE_SNAPSHOT:
        return freeze_command(config, root, allow_empty=args.allow_empty)
    return serve_command(config, root, args)


def freeze_command(config: LiveConfig, root: Path, *, allow_empty: bool) -> int:
    try:
        result = freeze_snapshot(config, root, allow_empty=allow_empty)
    except FreezeError as exc:
        log.error("freeze-snapshot: %s", exc)
        return 1
    print(
        f"snapshot bundle written to {result.target} from window {result.record.runId} "
        f"({result.record.eventCount} candidate events, {result.counts.get('bytes', 0)} bytes). "
        "Commit apps/web/public/data/snapshot/ through main (docs/deploy.md)."
    )
    return 0


def serve_command(config: LiveConfig, root: Path, args: argparse.Namespace) -> int:
    try:
        config_dir = Path(config.paths.configDir)
        config_dir = config_dir if config_dir.is_absolute() else root / config_dir
        data_dir = (
            Path(config.paths.dataDir).resolve()
            if config.paths.dataDir is not None
            else hq_cli.resolve_data_dir(None, config_dir)
        )
        window_config_dir = Path(config.paths.windowConfigDir)
        window_config_dir = (
            window_config_dir if window_config_dir.is_absolute() else root / window_config_dir
        )
    except hq_cli.ConfigError as exc:
        log.error("%s", exc)
        return 1
    runner = HqPipelineRunner(
        config_dir=config_dir,
        data_dir=data_dir,
        window_config_dir=window_config_dir,
        stages=config.window.stages,
    )
    log.info(
        "hq-api: config %s, pipeline config %s, data %s, runs every %.0f s over the last %.0f s",
        args.config,
        config_dir,
        data_dir,
        config.window.everyS,
        config.window.windowS,
    )
    app = create_app(config, runner, root=root, scheduler=not args.no_scheduler)
    uvicorn.run(
        app,
        host=args.host or config.server.host,
        port=args.port or config.server.port,
        timeout_keep_alive=int(config.server.keepAliveS),
        timeout_graceful_shutdown=int(config.server.gracefulShutdownS),
        log_config=None,  # keep hq's stderr logging; uvicorn's own would replace it
    )
    return exit_after_shutdown(app.state.worker)


def exit_after_shutdown(worker: LiveWorker) -> int:
    """After uvicorn returns (Ctrl-C), leave at once even if an abandoned window is still in
    its thread: the interpreter would otherwise join that thread at exit and block until the
    pipeline finishes. ``LiveWorker.stop`` already made sure it will not be committed."""
    if worker.running:
        log.warning(
            "hq-api: exiting while an abandoned window is still running; not waiting for it"
        )
        logging.shutdown()
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

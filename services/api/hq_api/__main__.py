"""``python -m hq_api`` / ``hq-api`` / ``make api``: run the live worker under uvicorn.

    hq-api [--config services/api/config.yaml] [--host H] [--port P] [--no-scheduler]

Every setting comes from ``config.yaml`` (``hq_api.config.LiveConfig``); ``--host`` and
``--port`` override the ``server`` section for one launch, ``--no-scheduler`` serves the last
persisted window without running the pipeline (useful to inspect state on a laptop).
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from hq import cli as hq_cli
from hq_api.app import create_app
from hq_api.config import DEFAULT_CONFIG_FILE, LiveConfigError, checkout_root, load_live_config
from hq_api.runner import HqPipelineRunner

log = logging.getLogger(__name__)

API_DIR = Path(__file__).resolve().parent.parent  # services/api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hq-api", description="Hidden Quakes live worker")
    parser.add_argument(
        "--config", default=str(API_DIR / DEFAULT_CONFIG_FILE), help="live config YAML"
    )
    parser.add_argument("--host", help="override server.host for this launch")
    parser.add_argument("--port", type=int, help="override server.port for this launch")
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="serve the persisted last window; do not run the pipeline",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    hq_cli.configure_logging()
    args = build_parser().parse_args(argv)
    try:
        config = load_live_config(Path(args.config))
        root = checkout_root()
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
    except (LiveConfigError, hq_cli.ConfigError) as exc:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())

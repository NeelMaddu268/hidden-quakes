"""DEV / TEST ONLY: the live API driven by the test suite's ``FakeRunner`` (API-05 drill).

    cd services/api && HQ_API_DEV=1 uv run python -m tests.drill_server --port 8765

Every window is synthetic (the same tiny tables ``tests/conftest.py`` builds for the offline
tests); nothing here touches the network or the real pipeline, and nothing it writes goes near
the checkout: runs, bundles, state and the snapshot all land in a temporary directory that is
printed at startup (``--data`` picks one). It exists so the kill-switch drill in
``services/api/README.md`` can be rehearsed before H1's and H2's stages are merged, and so the
Playwright acceptance for API-05 has a worker to kill. It refuses to start unless
``HQ_API_DEV=1`` is set, and it is not a test (pytest ignores it).
"""

import argparse
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import uvicorn
import yaml

from hq import cli as hq_cli
from hq_api.app import create_app
from hq_api.config import PathsConfig, load_live_config
from hq_api.worker import LiveWorker
from tests.conftest import CONFIG_YAML, SHOWCASE_DIR, FakeRunner, no_features

DEV_FLAG = "HQ_API_DEV"
DRILL_CONFIG = "drill.yaml"
MIN_EVERY_S = 60.0  # run ids have minute resolution (hq.runs.create_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.drill_server",
        description="DEV ONLY: serve synthetic live windows from the test suite's FakeRunner",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--events", type=int, default=3, help="candidate events per window")
    parser.add_argument(
        "--every",
        type=float,
        default=MIN_EVERY_S,
        help=f"seconds between windows (>= {MIN_EVERY_S:.0f})",
    )
    parser.add_argument("--window", type=float, default=7200.0, help="seconds of data per window")
    parser.add_argument(
        "--data", help="directory for runs, bundles, state and the snapshot (default: a temp dir)"
    )
    parser.add_argument(
        "--cors", default="*", help="serve.corsOrigins entry; the static site's origin, or *"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if os.environ.get(DEV_FLAG) != "1":
        print(
            f"drill_server: refusing to start without {DEV_FLAG}=1; it serves synthetic windows "
            "and is for the API-05 drill only",
            file=sys.stderr,
        )
        return 2
    hq_cli.configure_logging()
    args = build_parser().parse_args(argv)
    if args.every < MIN_EVERY_S:
        print(f"drill_server: --every must be >= {MIN_EVERY_S:.0f} s", file=sys.stderr)
        return 2
    root = Path(args.data) if args.data else Path(tempfile.mkdtemp(prefix="hq-api-drill-"))
    root.mkdir(parents=True, exist_ok=True)
    showcase = root / "showcase"
    showcase.mkdir(exist_ok=True)
    for path in sorted(SHOWCASE_DIR.glob("*.yaml")):
        shutil.copy(path, showcase / path.name)

    base = load_live_config(CONFIG_YAML)
    config = base.model_copy(
        update={
            "paths": PathsConfig(
                configDir=str(showcase),
                dataDir=str(root / "data"),
                windowConfigDir=str(root / "data" / "live" / "configs"),
                stateFile=str(root / "data" / "live" / "latest.json"),
                bundlesDir=str(root / "data" / "live" / "bundles"),
                snapshotDir=str(root / "web" / "snapshot"),
                keepBundles=base.paths.keepBundles,
            ),
            "window": base.window.model_copy(update={"everyS": args.every, "windowS": args.window}),
            "serve": base.serve.model_copy(update={"corsOrigins": [args.cors]}),
            "server": base.server.model_copy(update={"host": args.host, "port": args.port}),
        }
    )
    config_yaml = root / DRILL_CONFIG
    config_yaml.write_text(yaml.safe_dump(config.dump(), sort_keys=False), encoding="utf-8")
    runner = FakeRunner(config, n_events=args.events)
    worker = LiveWorker(config, runner, root=root, features_loader=no_features)
    app = create_app(config, runner, root=root, worker=worker)
    print(
        f"drill_server: SYNTHETIC windows of {args.events} candidate events every {args.every:.0f} s"
    )
    print(f"drill_server: data under {root}")
    print(f"drill_server: snapshot bundle (runtime write) at {config.paths.snapshotDir}")
    print(f"drill_server: freeze on demand: uv run hq-api freeze-snapshot --config {config_yaml}")
    print(f"drill_server: http://{args.host}:{args.port}/api/live/status  (Ctrl-C stops)")
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

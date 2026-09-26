#!/usr/bin/env bash
# Export one pipeline run to the web bundle and validate it (ticket API-02, H4).
#
#   scripts/export-showcase.sh <runId> [--config-dir DIR] [--data-dir DIR]
#
# Runs `hq stage export --run <runId>` (which writes apps/web/public/data/<mode>/ for every mode
# in configs/showcase/export.yaml, atomically) and then re-validates each bundle with
# `python -m hq.export <dir> --config-dir DIR` (hq.export.check_bundle, with the run's rounding
# and byte caps). Exit status is nonzero if either fails.
# `make export RUN=<runId>` calls this.
#
# Portable to macOS system Bash 3.2 (REQ-H3-7): the optional arguments are re-assembled into the
# positional parameters with `set --`, never held in an array, because expanding an empty array
# with `set -u` is an "unbound variable" error before Bash 4.4.
set -euo pipefail

usage() {
  echo "usage: $0 <runId> [--config-dir DIR] [--data-dir DIR]" >&2
  exit 2
}

if [[ $# -lt 1 ]]; then
  usage
fi
run_id="$1"
shift

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
seismic="$root/services/seismic"
config_dir="configs/showcase"
data_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-dir|--data-dir)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "$1 needs a directory" >&2
        usage
      fi
      ;;
  esac
  case "$1" in
    --config-dir) config_dir="$2"; shift 2 ;;
    --data-dir) data_dir="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

# The stage's arguments: --config-dir always, --data-dir only when given.
set -- --run "$run_id" --config-dir "$config_dir"
if [[ -n "$data_dir" ]]; then
  set -- "$@" --data-dir "$data_dir"
fi

cd "$seismic"
uv run hq stage export "$@"

# Every mode the config lists gets its own bundle; check each one the way the stage did.
modes="$(uv run --quiet python -c "
from pathlib import Path
from hq.config import load_config
from hq.export import output_root
cfg = load_config(Path('$config_dir')).export
print(' '.join(str(output_root(cfg) / mode) for mode in cfg.modes))
")"
status=0
for bundle in $modes; do
  uv run --quiet python -m hq.export "$bundle" --config-dir "$config_dir" || status=1
done
exit "$status"

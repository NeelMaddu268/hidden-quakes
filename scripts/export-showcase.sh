#!/usr/bin/env bash
# Export one pipeline run to the web bundle and validate it (ticket API-02, H4).
#
#   scripts/export-showcase.sh <runId> [--config-dir DIR] [--data-dir DIR]
#
# Runs `hq stage export --run <runId>` (which writes apps/web/public/data/<mode>/ for every mode
# in configs/showcase/export.yaml, atomically) and then re-validates each bundle with
# `python -m hq.export <dir>` (hq.export.check_bundle). Exit status is nonzero if either fails.
# `make export RUN=<runId>` calls this.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <runId> [--config-dir DIR] [--data-dir DIR]" >&2
  exit 2
fi
run_id="$1"
shift

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
seismic="$root/services/seismic"
config_dir="configs/showcase"
extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-dir) config_dir="$2"; extra+=("$1" "$2"); shift 2 ;;
    --data-dir) extra+=("$1" "$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$seismic"
uv run hq stage export --run "$run_id" --config-dir "$config_dir" "${extra[@]}"

# Every mode the config lists gets its own bundle; check each one the way the stage did.
modes="$(uv run --quiet python -c "
from pathlib import Path
from hq.config import load_config
from hq.export import output_root
cfg = load_config(Path('$config_dir')).export
print(' '.join(str(output_root(cfg) / mode) for mode in cfg.modes))
" 2>/dev/null)"
status=0
for bundle in $modes; do
  uv run --quiet python -m hq.export "$bundle" || status=1
done
exit "$status"

#!/usr/bin/env bash
# Share a run's tables with the team as a GitHub prerelease (make publish-run RUN=<id>), without
# dropping files other lanes added since (REQ-H1-4).
#
#   scripts/publish-run.sh <runId> [--force]
#
# When the release already exists, its current tarball is downloaded and its file list compared
# with the local run directory. If the release holds a file the local copy lacks, the publish is
# refused and the files are named: run `make fetch-run RUN=<id>` first, merge, then publish.
# `--force` (make publish-run RUN=<id> FORCE=1) uploads anyway. Works on macOS Bash 3.2.
set -euo pipefail

run_id="${1:-}"
force=0
for arg in "${@:2}"; do
  case "$arg" in
    --force) force=1 ;;
    *) echo "publish-run: unknown argument '$arg'" >&2; exit 2 ;;
  esac
done
[ -n "$run_id" ] || { echo "usage: make publish-run RUN=<runId> [FORCE=1]" >&2; exit 1; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs_dir="$root/data/showcase/runs"
run_dir="$runs_dir/$run_id"
[ -d "$run_dir" ] || { echo "publish-run: no such run: $run_dir" >&2; exit 1; }

tag="run-$run_id"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
tarball="$tmp/$tag.tgz"

if gh release view "$tag" >/dev/null 2>&1; then
  gh release download "$tag" -p "$tag.tgz" -D "$tmp/current" --clobber
  tar -tzf "$tmp/current/$tag.tgz" | sed -e 's#/$##' | grep -v "^$run_id$" | sed -e "s#^$run_id/##" | sort > "$tmp/remote.txt"
  (cd "$runs_dir" && find "$run_id" -type f -o -type d | sed -e "s#^$run_id/##" | grep -v "^$run_id$" | sort) > "$tmp/local.txt"
  missing="$(comm -23 "$tmp/remote.txt" "$tmp/local.txt" || true)"
  if [ -n "$missing" ] && [ "$force" != 1 ]; then
    echo "publish-run: refusing to overwrite release $tag: it holds files this run directory lacks:" >&2
    printf '  %s\n' $missing >&2
    echo "publish-run: run 'make fetch-run RUN=$run_id' first (it merges the release into your copy)," >&2
    echo "publish-run: or 'make publish-run RUN=$run_id FORCE=1' to drop them on purpose." >&2
    exit 3
  fi
  tar -czf "$tarball" -C "$runs_dir" "$run_id"
  gh release upload "$tag" "$tarball" --clobber
else
  tar -czf "$tarball" -C "$runs_dir" "$run_id"
  gh release create "$tag" "$tarball" --prerelease --title "run $run_id" --notes "Pipeline run tables. Fetch with: make fetch-run RUN=$run_id"
fi
echo "published $tag"

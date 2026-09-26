#!/usr/bin/env bash
# Serve the static export of the web app from this laptop, for a demo with Wi-Fi off (API-03).
#
#   scripts/serve-offline.sh              build the export, then serve apps/web/out/
#   scripts/serve-offline.sh --no-build   serve an existing apps/web/out/
#
# Env:
#   NEXT_PUBLIC_ALLOW_MOCK   "1" (default) lets ?mode=mock load; "0" builds the production shape
#   PORT                     local port (default 4173)
#   HOST                     bind address (default 127.0.0.1)
#
# The BUILD needs the network once: `next/font/google` downloads Inter and JetBrains Mono at build
# time and writes them into out/_next/static/media/, so the served page makes no request that
# leaves this machine. Build while online, then serve offline. Everything the page loads
# (HTML, JS, CSS, fonts, /data/<mode>/*.json, /terrain/*) is a file under apps/web/out/.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
web="$root/apps/web"
out="$web/out"
port="${PORT:-4173}"
host="${HOST:-127.0.0.1}"
build=1

for arg in "$@"; do
  case "$arg" in
    --no-build) build=0 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "serve-offline: unknown argument '$arg'" >&2; exit 2 ;;
  esac
done

if [ "$build" = 1 ]; then
  [ -d "$root/node_modules" ] || { echo "serve-offline: run 'pnpm install' at the repo root first" >&2; exit 1; }
  export NEXT_PUBLIC_ALLOW_MOCK="${NEXT_PUBLIC_ALLOW_MOCK:-1}"
  export NEXT_TELEMETRY_DISABLED=1
  echo "serve-offline: building static export (NEXT_PUBLIC_ALLOW_MOCK=$NEXT_PUBLIC_ALLOW_MOCK); this step needs the network for fonts"
  rm -rf "$out"
  (cd "$root" && pnpm --filter web build)
fi

[ -f "$out/index.html" ] || { echo "serve-offline: no export at $out (run without --no-build)" >&2; exit 1; }
command -v python3 >/dev/null || { echo "serve-offline: python3 not found" >&2; exit 1; }

modes=$(ls "$out/data" 2>/dev/null | tr '\n' ' ')
echo "serve-offline: serving $out"
echo "serve-offline: bundles present: ${modes:-none}"
echo
echo "  http://$host:$port/            (showcase, the default mode)"
echo "  http://$host:$port/?mode=mock  (synthetic bundle; needs NEXT_PUBLIC_ALLOW_MOCK=1 at build time)"
echo
echo "Ctrl-C stops the server."
exec python3 -m http.server "$port" --bind "$host" --directory "$out"

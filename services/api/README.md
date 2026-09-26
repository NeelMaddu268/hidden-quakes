# services/api · Live worker (API-04, API-05, H4)

A FastAPI service that runs the `hq` pipeline over a rolling window of the last few hours, every
few minutes, and serves the result as the Live API in `docs/02-contracts.md` §7. The web app's
`LiveProvider` (`apps/web/src/providers/live.ts`) reads it in `?mode=live` and fails over to the
snapshot bundle (`apps/web/public/data/snapshot/`, frozen from a good window by this worker or by
`hq-api freeze-snapshot`) when the API is unreachable (API-05, below).

Every knob is in `config.yaml` (`hq_api/config.py`, unknown keys are an error); the worker records
the whole config in every window record it serves.

## Run

```
make api                         # uvicorn on server.host:server.port from services/api/config.yaml
make api ARGS='--port 8001'      # one-launch overrides: --host, --port, --config, --no-scheduler
make api ARGS='freeze-snapshot'  # copy the last good live window into apps/web/public/data/snapshot/
cd services/api && uv run pytest -q -m smoke
```

`hq-api` (`python -m hq_api`) has two commands: `serve`, implied when none is given, and
`freeze-snapshot`. `--no-scheduler` serves the last persisted window without running the pipeline
(inspect state on a laptop). The project has its own `uv` environment (`uv sync --frozen` in
`services/api`); it depends on `hq` and `hq-contracts` by path, like `services/seismic` depends
on the contracts.

Ctrl-C stops the worker at once, even mid-window: the ticker and its tick tasks are cancelled,
the executor is shut down without waiting (queued futures cancelled), and a window still running
in its thread is logged as **abandoned**: its result is never committed to the state file, and the
process exits without waiting for the thread (the pipeline is synchronous and cannot be
interrupted; its half-written run directory under `data/live/runs/` is simply left behind).

## What a window does

Each tick (`window.everyS`, first one at startup), in one worker thread so the API stays
responsive:

1. Window `[now − dataLagS − windowS, now − dataLagS]`, whole seconds.
2. `HqPipelineRunner` copies `paths.configDir` into `paths.windowConfigDir/<window>/` with
   `run.yaml`'s `name`, `windowStart`, `windowEnd` replaced (bbox, origin, reference surface stay
   H2's), then `hq.runs.create_run(mode="live")` and `run_stages(window.stages)`. A stage that is
   not merged fails with the owner's name; the worker records that as the window's failure and
   keeps serving the last good window. The worker never imports a lane's stage.
3. `hq.export.export_bundle(..., mode="live")` writes and checks `paths.bundlesDir/<runId>/`
   (meta, events, evidence); the API serves from there.
4. `latencyS` = data end → results ready: the run measured on the monotonic clock plus
   `dataLagS`. Over `window.maxLatencyS` it is logged as slow and flagged in `/health`.
5. `export_bundle(..., mode="snapshot")` rewrites `paths.snapshotDir`
   (`apps/web/public/data/snapshot/`) when the window produced a valid bundle with candidate
   events; `snapshot.writeEmptyWindows` (default false) decides whether an empty window does
   too. A snapshot failure of any kind is logged and recorded in the window's `error`, never
   fails the window: its live bundle is already good. The deployed static site fails over to the
   snapshot **committed** in git (see "Freezing the snapshot" below); this runtime write matters
   for the local and offline builds, which serve `apps/web/public/data/snapshot/` as it is on disk.
6. The served window is swapped and `paths.stateFile` (`data/live/latest.json`) is written
   atomically: the last good record plus the last `serve.historyN` attempts. A restart serves the
   last window from it without waiting for a new run. Run ids have minute resolution
   (`hq.runs.create_run`), so restarting within the same minute as the last window records one
   failed attempt ("run directory already exists") and the next tick runs normally.

A tick that arrives while a window is still running is skipped and logged, never queued.

## Endpoints

| Route | Returns |
| --- | --- |
| `GET /api/live/meta` | `BundleMeta` of the latest good window (`mode: "live"`) |
| `GET /api/live/events` | `SeismicEvent[]` in reveal order, at most `serve.maxEvents` |
| `GET /api/live/evidence/{id}` | `EventEvidence`; 404 when the window has none for that id |
| `GET /api/live/status` | `LiveStatus` without `events` |
| `GET /health` | `status` (waiting / ok / stale), the served and last-attempted window records, history, worker counters, config |

`stationsOnline` in the status is the count of stations with `usedInRun` true in the run's
`stations.parquet`. When a window holds more than `serve.maxEvents` candidate events, `/events`
serves the first `maxEvents` in reveal order and the worker logs the truncation once per window.

Before the first good window every `/api/live/*` route answers 503 (the provider treats it as a
failed fetch and fails over to the snapshot). An empty window is a normal answer: `events` is `[]`
and the status is ok; the shell derives its copy from that. CORS allows `serve.corsOrigins` (add
the deployed site's origin there; the web build's `NEXT_PUBLIC_LIVE_API_BASE` must point at this
worker, see below).

## Where state lives

| What | Path (relative to the checkout root, gitignored) |
| --- | --- |
| Runs | `<dataDir>/live/runs/<runId>/` (`dataDir: null` = the main checkout's `data/`, the `hq` CLI rule) |
| Per-window configs | `data/live/configs/` (keep it inside the checkout so run ids carry the git sha) |
| Live bundles | `data/live/bundles/<runId>/`, newest `paths.keepBundles` kept |
| State file | `data/live/latest.json` |
| Snapshot bundle | `apps/web/public/data/snapshot/` (the one data product of this worker that is committed; see "Freezing the snapshot") |

## Failover (API-05): how the web app treats this worker

`ProviderRoot` (`apps/web/src/providers/root.tsx`) runs a two-state machine in `?mode=live`:

```
live ── any live request fails (connection refused, 503, no answer within LIVE_FETCH_TIMEOUT_MS) ──▶ snapshot
snapshot ── a heartbeat or poll of /api/live/status succeeds ──▶ live
```

- **Live** reads `/api/live/meta`, `/events`, `/evidence/{id}` from this worker and stations,
  catalog and features from the snapshot bundle (the API does not serve them). The label is
  `Live · last {windowS} · updated {n} min ago`, both values from `LiveStatus`.
- **Snapshot** is `StaticBundleProvider("snapshot")` reading `/data/snapshot/`, the same reader as
  showcase. The label is `Snapshot · generated {run.createdAt} by our pipeline · run {run.id}`
  from the snapshot's own `meta.json`. The LIVE pill stays pressed; the label alone says what is
  on screen. `useLiveStatus()` is `null` while failed over.
- The worker is polled every `LIVE_POLL_MS` (status + events) and probed every
  `LIVE_HEARTBEAT_MS` (status only), whichever state is active; each request has a
  `LIVE_FETCH_TIMEOUT_MS` deadline. All three are knobs in `apps/web/src/providers/config.ts`.
  A kill is therefore noticed within one heartbeat plus the failed request, and the snapshot
  (whose meta, stations, catalog and features are already in memory from live) is on screen
  right after; a worker that comes back with a served window is picked up the same way and the
  page switches back to live, reloading the bundle from the worker.
- Only a failed request fails over. A live `schemaVersion` mismatch, or a snapshot bundle that is
  missing after a failover, is the visible "Data unavailable" panel: never a silent fallback.

The web build must know where this worker is: the static export is same-origin by default
(`/api/live`), so a cross-origin worker needs `NEXT_PUBLIC_LIVE_API_BASE=http://<host>:<port>/api/live`
at build time (`docs/deploy.md`) and its origin in `serve.corsOrigins`.

## Freezing the snapshot (API-05)

The deployed site only has the snapshot that is **committed** in git. When a live window is worth
keeping on screen (candidate events, sane latency), freeze it and commit it through `main`:

```
make api ARGS='freeze-snapshot'                 # or: cd services/api && uv run hq-api freeze-snapshot
make api ARGS='freeze-snapshot --allow-empty'   # also accept the newest good window with no events
git add apps/web/public/data/snapshot && git commit -m 'API-05: freeze live window <runId> as the snapshot'
```

`freeze-snapshot` (`hq_api/snapshot.py`) reads `paths.stateFile`, takes the newest good window
whose live bundle still exists under `paths.bundlesDir` (windows with no candidate events are
skipped unless `--allow-empty`, the same rule as `snapshot.writeEmptyWindows`), re-checks it as
`live`, copies it with `meta.mode` set to `snapshot` (nothing else changes), checks the copy with
`check_bundle(mode="snapshot")` using the caps in `paths.configDir/export.yaml`, and swaps it into
`paths.snapshotDir` atomically. On any failure the previous snapshot is untouched. It works with
the worker stopped or running, and it is the same result as the worker's own runtime write after
a good window, on demand and for the window you pick.

The committed snapshot must come from a **real** window: the drill server below writes synthetic
windows and its snapshot lands in a temporary directory, never in the checkout.

## Kill-switch drill (API-05)

Rehearse the failover before the demo. Two terminals from the checkout root:

1. Start a worker. Real pipeline: `make api` (H1's and H2's stages must be merged; until then
   every window fails, see below). Dry run without them, on synthetic windows:
   `cd services/api && HQ_API_DEV=1 uv run python -m tests.drill_server --port 8765`
   (dev/test only; it refuses to start without `HQ_API_DEV=1`, keeps everything it writes in a
   temp dir it prints, and its windows carry the test suite's synthetic tables).
2. Build and serve the site against that worker (the build flag is inlined, so it is a rebuild):
   ```
   NEXT_PUBLIC_LIVE_ENABLED=1 NEXT_PUBLIC_ALLOW_MOCK=1 \
   NEXT_PUBLIC_LIVE_API_BASE=http://127.0.0.1:8765/api/live pnpm --filter web build
   make offline NO_BUILD=1                       # serves apps/web/out on http://127.0.0.1:4173
   ```
   For the dry run, copy the drill's snapshot into the export first
   (`cp -r <drill temp dir>/web/snapshot apps/web/out/data/snapshot`), never into
   `apps/web/public/`.
3. Open `http://127.0.0.1:4173/?mode=live`. Once the first window is served the label reads
   `Live · last … · updated … min ago`.
4. Ctrl-C the worker. The label must read `Snapshot · generated … by our pipeline · run …`
   within 5 s, with LIVE still the pressed pill. Start the worker again: after its first window
   the label returns to Live on its own.

`docs/03` → Live: unstable, or latency over about ten minutes → cut the LIVE pill
(`NEXT_PUBLIC_LIVE_ENABLED` unset in the web build). Read `/health`: `served.latencyS` and
`served.slow` against `config.window.maxLatencyS`, `worker.failures` and `history[*].error` for
stability. The snapshot bundle keeps the last good window on screen either way.

## Until H1/H2 stages land

`hq stages` lists what is merged. With `inventory`, `download`, `pick` (H1) and `associate`,
`locate`, `match`, `tier` (H2) missing, every window fails at the first missing stage with its
owner's name and nothing is served. Once they exist, the real runner needs nothing else: H1's
`hq.ingest.cache.read_window` and `hq.preprocess.display_copy` are resolved for evidence at
export time (a missing one names H1), and the run tables listed in `docs/02` §2 are what
`hq.export` reads.

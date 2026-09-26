# services/api · Live worker (API-04, H4)

A FastAPI service that runs the `hq` pipeline over a rolling window of the last few hours, every
few minutes, and serves the result as the Live API in `docs/02-contracts.md` §7. The web app's
`LiveProvider` (`apps/web/src/providers/live.ts`) reads it in `?mode=live`; API-05 fails over to
the snapshot bundle this worker writes when the API is unreachable.

Every knob is in `config.yaml` (`hq_api/config.py`, unknown keys are an error); the worker records
the whole config in every window record it serves.

## Run

```
make api                         # uvicorn on server.host:server.port from services/api/config.yaml
make api ARGS='--port 8001'      # one-launch overrides: --host, --port, --config, --no-scheduler
cd services/api && uv run pytest -q -m smoke
```

`--no-scheduler` serves the last persisted window without running the pipeline (inspect state on a
laptop). The project has its own `uv` environment (`uv sync --frozen` in `services/api`); it
depends on `hq` and `hq-contracts` by path, like `services/seismic` depends on the contracts.

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
   snapshot **committed** in git (API-05 commits it through `main`); this runtime write matters
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
failed fetch; API-05 fails over). An empty window is a normal answer: `events` is `[]` and the
status is ok; the shell derives its copy from that. CORS allows `serve.corsOrigins` (add the
deployed site's origin there).

## Where state lives

| What | Path (relative to the checkout root, gitignored) |
| --- | --- |
| Runs | `<dataDir>/live/runs/<runId>/` (`dataDir: null` = the main checkout's `data/`, the `hq` CLI rule) |
| Per-window configs | `data/live/configs/` (keep it inside the checkout so run ids carry the git sha) |
| Live bundles | `data/live/bundles/<runId>/`, newest `paths.keepBundles` kept |
| State file | `data/live/latest.json` |
| Snapshot bundle | `apps/web/public/data/snapshot/` (committed by API-05 when the window is worth freezing) |

## Kill switch

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

## API-05 to do

- The web app's `LIVE_API_BASE` (`apps/web/src/providers/config.ts`) is the same-origin relative
  `/api/live`, while this worker runs cross-origin: add a `NEXT_PUBLIC_LIVE_API_BASE` build flag
  (and keep CORS in `serve.corsOrigins`), or serve `apps/web/out/` from the API itself.
- Ctrl-C while a window is running blocks exit until the pipeline thread finishes: track the tick
  tasks and cancel them, and give the executor a way to abandon the run, before the demo.

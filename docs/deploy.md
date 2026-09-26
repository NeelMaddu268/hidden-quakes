# Deploying the web app (API-03)

The web app is a static export: `pnpm --filter web build` writes `apps/web/out/`, a folder of HTML, JS, CSS, self-hosted fonts and everything from `apps/web/public/` (the data bundles under `data/<mode>/` and the terrain tiles). Any static file server hosts it. There is no server-side code on the demo path.

## Vercel (the public URL)

Vercel's Git integration builds and deploys `main` on every merge. Nothing in the repo triggers it; the one-time setup is in the dashboard. Import the GitHub repo as a new project and set:

| Setting | Value | Why |
| --- | --- | --- |
| Framework Preset | Next.js | Detected; `apps/web/vercel.json` also pins it |
| Root Directory | `apps/web` | The app lives in a pnpm workspace; `apps/web/vercel.json` holds the install and build commands |
| Include source files outside of the Root Directory | on (the default) | The build needs `pnpm-lock.yaml`, `pnpm-workspace.yaml`, `packages/contracts/ts` and `packages/visualization` from the repo root |
| Production Branch (Settings → Git) | `main` | Only H4 merges into `main` |
| Environment variable `NEXT_PUBLIC_ALLOW_MOCK` | `1` (Production and Preview) | **Gate M:** lets `?mode=mock` load the synthetic bundle on the public URL until the showcase bundle exists |
| Environment variable `NEXT_PUBLIC_LIVE_ENABLED` | `1` only while the live worker runs | Shows the LIVE pill (API-04); unset is the kill switch (`docs/03`) |
| Environment variable `NEXT_PUBLIC_LIVE_API_BASE` | `http://<worker host>:<port>/api/live` | Where `?mode=live` fetches from (API-05). Unset means same-origin `/api/live`, which a static export has no server for; the worker's `serve.corsOrigins` must list the site's origin |

Everything else comes from `apps/web/vercel.json`: install `pnpm install --frozen-lockfile`, build `pnpm --filter web build`, output `out`, and an `ignoreCommand` that builds only `main` and `feat/*` (agent branches and PRs from them are skipped, so a push storm can't queue dozens of builds).

`?mode=` picks the bundle: the default is `showcase`, which shows the provider's "Data unavailable" panel until API-02 commits `apps/web/public/data/showcase/`.

### Live and its snapshot (API-05)

`?mode=live` reads the live worker (`services/api`, its README has the drill). It fails over to `apps/web/public/data/snapshot/` when the worker is unreachable, and only the snapshot that is committed in git reaches the deployed site: freeze a good window with `make api ARGS='freeze-snapshot'` and commit `apps/web/public/data/snapshot/` through `main`. The build flags are inlined by `next build`, so changing `NEXT_PUBLIC_LIVE_ENABLED` or `NEXT_PUBLIC_LIVE_API_BASE` means a rebuild (a redeploy on Vercel, `make offline` locally).

### The one-line switch after the showcase bundle lands

When `apps/web/public/data/showcase/` is on `main`, delete the `NEXT_PUBLIC_ALLOW_MOCK` environment variable in the Vercel dashboard (or set it to `0`) and redeploy. Production then refuses `?mode=mock` with the provider's visible "mode is disabled in this build" message, and the default URL loads the showcase run. Nothing in the repo changes.

### Checking a deploy

Open `<deployment url>/?mode=mock`: the red SYNTHETIC banner and the PUBLIC counter must show. Open `<deployment url>/` for the showcase run once it exists.

## Offline (Wi-Fi off on stage)

```
make offline              # build with NEXT_PUBLIC_ALLOW_MOCK=1, then serve apps/web/out on http://127.0.0.1:4173
make offline NO_BUILD=1   # serve an existing export
PORT=8080 scripts/serve-offline.sh
```

`scripts/serve-offline.sh` serves the export with `python3 -m http.server`. **Build while online, serve offline:** `next/font/google` downloads Inter and JetBrains Mono during `next build` and writes them into `out/_next/static/media/`, so the served page makes no request that leaves the laptop. Every fetch the page makes (`/data/<mode>/*.json`, `/terrain/*`, fonts, scripts) is a file under `apps/web/out/`. Build the export before leaving the venue's network, then `make offline NO_BUILD=1` works with the radio off.

## CI

`.github/workflows/check.yml` runs `make check` and the static export on every pull request and on every push to `main` and `feat/*`, and fails if `apps/web/out/index.html` or `apps/web/out/data/` is missing. It caches the pnpm store, the uv cache and `apps/web/.next/cache`.

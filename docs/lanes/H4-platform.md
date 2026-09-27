# H4 · Platform lane

**Mission:** turn four lanes into one product that never breaks on stage. You own the repo, the seams between lanes (contracts, config loader, stage runner, providers, exporter), deployment, validation data and the story. You're also the integration owner: only you merge into `main`.

## Paths

**You own:** repo root (`CLAUDE.md`, `README.md`, `Makefile`, `pnpm-workspace.yaml`, `.github/`), `packages/contracts/`, `packages/config/`, `services/seismic/hq/export/`, `hq/validate/`, `hq/cli.py`, `hq/runs.py`, `hq/config/__init__.py`, `hq/config/export.py`, `configs/showcase/export.yaml`, `services/seismic/tests/platform/`, `services/api/`, `apps/web/src/app/`, `apps/web/src/shell/`, `apps/web/src/providers/`, `apps/web/public/data/`, `scripts/` (except `bake-dem.py`), `data/fixtures/`, `docs/` (except other lanes' lane docs)

**You never write:** H1's and H2's `hq/` modules and YAML sections, H3's `apps/web/src/scene|drawer|state/`, `packages/visualization/`, `apps/web/public/terrain/`.

## What you consume

| What | From | Where | Until it exists |
| --- | --- | --- | --- |
| `run.yaml` shape | H2 | `hq/config/run.py`, `configs/showcase/run.yaml` | 8:05 PM |
| Stations, picks, cache API | H1 | `runs/<id>/`, `hq.ingest.cache`, `hq.preprocess` | mock fixture |
| Events, arrivals, event picks, matches, sweep, synthetic | H2 | `runs/<id>/` | mock fixture |
| `associate`, `locate`, `match`, `assign_tiers` | H2 | `docs/02` §5 | VAL tickets wait for them (~2 AM) |
| STA/LTA picks | H1 | `runs/<id>/picks_stalta.parquet` | VAL-01 waits (~4 AM) |
| Demo store, tokens, `<Scene/>`, `<EvidenceDrawer/>`, `<TimeScrubber/>` | H3 | `apps/web/src/state|scene|drawer/`, `packages/visualization/` | placeholder `<div>`s in `page.tsx` until 9:00 PM |

## What you produce

| What | For | Where | Due |
| --- | --- | --- | --- |
| Repo skeleton, worktree conventions, Makefile | everyone | repo root | 8:05 PM |
| Contracts (Py models, `io`, generated TS) | everyone | `packages/contracts/` | 8:25 PM |
| Config loader, `RunContext`, `hq run` / `hq stage` | H1, H2 | `hq/config/__init__.py`, `hq/runs.py`, `hq/cli.py` | 8:45 PM |
| Mock bundle | H3 | `apps/web/public/data/mock/` | 8:45 PM |
| Provider hooks | H3 | `apps/web/src/providers/hooks.ts` | 9:00 PM |
| Deployed URL (mock) | everyone | Vercel | 12:00 AM (Gate M) |
| Showcase bundle | H3 | `apps/web/public/data/showcase/` | 4:15 AM first, 8:00 AM hero + deployed |
| `validation.json` | shell panels | `runs/<id>/validation.json` → bundle | 2:00 PM Sat |

## Tickets, in order

### REPO-00 · P0 · 8:00–8:05 PM — Repo skeleton (done by bootstrap.sh)

- **Goal:** verify what `bootstrap.sh` created (see "What the bootstrap already did" below): repo, first commit, `apps/web` scaffold, four `feat/*` branches, collaborator invites, real usernames in `CODEOWNERS` and `docs/team.md`, one issue per ticket.
- **Files:** none new; fix anything the bootstrap missed
- **Accept:** everyone has accepted the invite and checked out their `feat/*` branch by 8:10; `make help` runs; `gh issue list` shows every ticket.

### CONTRACT-01 · P0 · 8:05–8:25 PM — Contracts

- **Goal:** implement `docs/02` §1–2 as `packages/contracts/python/hq_contracts/{models,io}.py`, `scripts/gen-contracts.sh`, the generated `packages/contracts/ts/src/index.ts`, and `apps/web/src/providers/types.ts` (§6).
- **In → out:** `docs/02` → installable Python package + generated TS + `schema.json`
- **Accept:** a round-trip test builds one of every model → JSON → parse; `to_frame` / `from_frame` round-trip every table model; `pnpm -r typecheck` passes on a file importing every type; regenerating TS gives no diff. H2 and H3 review at 8:20; merge by 8:25. That merge is the freeze.

### RUN-01 · P0 · 8:25–8:45 PM — Config loader and stage runner

- **Goal:** `load_config()` composing `run.yaml`, `signal.yaml`, `seismology.yaml`, `export.yaml` (unknown keys are errors); `RunContext` with `path()` and `record()`; `hq run` and `hq stage` CLIs with a stage registry that imports each lane's `run(ctx)`.
- **Files:** `hq/config/__init__.py`, `hq/config/export.py`, `hq/runs.py`, `hq/cli.py`, `tests/platform/test_runs.py`
- **Accept:** a dummy stage records runtime and counts into `run.json`; missing lane stages fail with a clear message naming the owner.

### FIX-01 · P0 · 8:25–8:45 PM — Mock fixture

- **Goal:** `scripts/mock-fixture.py`, written from scratch, generating a full mock bundle that exercises every UI path. Everything is invented; nothing derives from pre-event outputs.
- **Contents:** `meta.isSynthetic = true`; 43 public points; ~500 candidates (Tier A ~150 in a compact synthetic cluster, B ~200 around it, C ~150 scattered); origin times spread over 24 h with some clustering; 17 stations (14 surface in a ~20 km ring, one of them `usedInRun: false`, 3 borehole with nonzero `sensorDepthM`; 16 used, so the hero carries 16 traces); one synthetic well path with `verified: false`; 20 evidence files with synthetic wavelets at predicted times plus noise; a `validation.json` with every field filled; a `heroEventId`.
- **Accept:** every file validates with the Pydantic models; H3's scene loads it; the SYNTHETIC banner shows.

### API-01 · P0 · 8:45–9:15 PM — Providers and hooks

- **Goal:** `StaticBundleProvider(mode)` for mock / showcase / snapshot, a `LiveProvider` stub, `?mode=` selection, schema-version check, memoized fetches, evidence preload (hero + first 20), and the hooks from `docs/02` §6.
- **Files:** `apps/web/src/providers/*`
- **Accept:** a test swaps mock ↔ showcase (a copy of mock) with no component changes; a version mismatch renders a visible error.

### DEMO-01 · P0 · 9:15–11:00 PM — Shell

- **Goal:** page composition (`<Scene/>`, `<Shell/>`, `<EvidenceDrawer/>`, `<TimeScrubber/>`), fonts, top-left title + mode label, top-right counters (PUBLIC / RECOVERED / STRICT; the last two dimmed "—" before the reveal), REVEAL HIDDEN SIGNAL button, pills (PUBLIC / ALL / STRICT, SHOWCASE / LIVE), keyboard map from `docs/02` §6, SYNTHETIC banner.
- **Files:** `apps/web/src/app/*`, `apps/web/src/shell/*`
- **Accept:** keys drive the store; counters read only `AnalysisSummary` and `revealProgress`; a grep finds no numeric literals in UI copy.

### API-03 · P0 · by 12:00 AM — Deploy and offline

- **Goal:** `output: "export"` static build, Vercel deploy on every `main` merge, `scripts/serve-offline.sh`.
- **Accept:** the public URL loads mock mode; the offline build works with Wi-Fi off.

### FEAT-01 · P0 · by 6:00 AM — Geothermal reference features

- **Goal:** source a verified geothermal reference for `features.json`: a published well trajectory (search GDR for the well's survey data) or a region outline, each with a `SourceRef`. If you can't verify a location, mark `verified: false` and H3 renders it as approximate.
- **Files:** `configs/showcase/export.yaml` (features section), `hq/export/features.py`
- **Accept:** every feature has a citation and URL; nothing unverified claims precision.

### API-02 · P0 · First run ~4:15 AM; hero + deploy by 8:00 AM — Exporter

- **Goal:** run → bundle. Compute `revealOrder` (Tier A → B → C, time-ordered within), `heroEventId` (the Tier A event with the most stations), `AnalysisSummary`, `SceneMeta`, and evidence snippets from `arrivals.parquet` + `read_window` + `display_copy` (4–8 s, bandpassed, ~100 Hz display, ≤ 16 traces sorted by distance, picks and predicted arrivals filled).
- **Files:** `hq/export/`, `scripts/export-showcase.sh`, `tests/platform/test_export.py`
- **In → out:** `runs/<id>/` + cache → `apps/web/public/data/showcase/`
- **Accept:** every bundle file validates; showcase mode renders; each evidence file under 60 KB; summary counts equal counts recomputed from `events.json`.

### VAL-02 · P1 · ~4:00–8:00 AM — Null test

- **Goal:** 20 reruns of H2's `associate` + `locate` + `match` + `assign_tiers` with independent per-station pick shifts (uniform ±30 s), same config.
- **Files:** `hq/validate/null_test.py`
- **Accept:** `NullTest` written with mean and std of chance events and chance Tier A; seeded and reproducible.

### VAL-01 · P1 · ~8:00 AM–2:00 PM — Baseline comparison and validation.json

- **Goal:** two pickers × two association profiles (`full`, `p_only`) through H2's API; association sweep, synthetic test, null test, G-R (if H2's magnitudes pass) into `validation.json`; `BaselineGain` into the summary only when gain holds in both profiles.
- **Files:** `hq/validate/`
- **Accept:** the table reproduces from stored configs; G-R uses Aki–Utsu b with Shi–Bolt σ and Mc by maximum curvature + 0.2.

### DEMO-02 · P1 · after VAL-01 — Validation and Run details panels

- **Goal:** the tiny validation panel (table below) and a Run details panel that prints `ProcessingRun` verbatim plus the sweep plot.
- **Files:** `apps/web/src/shell/validation/`, `shell/run-details/`
- **Accept:** every row hides itself when its source field is null.
- **Keys:** D opens and closes Run details; Esc closes it (an addition beside the `docs/02` §6 map, which is unchanged).

### API-04 · P1 · after Gate E — Live worker

- **Goal:** FastAPI service that runs `hq run` on the last 2 hours every 10 minutes and serves `docs/02` §7.
- **Files:** `services/api/`
- **Accept:** `latencyS` measured and stored; an empty window renders "No high-confidence candidate events in this window."

### API-05 · P1 · after API-04 — Snapshot and failover

- **Goal:** freeze the last good live window to `apps/web/public/data/snapshot/`; `LiveProvider` fails over after 5 s and the label changes.
- **Accept:** killing the API flips the label to Snapshot within 5 s.

### DEMO-03 · P0 · 2:00 PM Sat → submission — Story

- **Goal:** pitch rehearsals, Devpost, README, 2-minute video, outsider tests at 4 PM and 10 PM. Content lives in `docs/demo/pitch-and-qa.md`.

### PLAT-99 · P1 · 2–6 PM Saturday — Lane hardening pass

- **Goal:** a fresh reviewer agent audits the lane against "Definition of done" and fixes what it finds.

## Domain notes (give these to your agent)

### What the bootstrap already did (REPO-00)

`bootstrap.sh` ran at 8:00 PM. It created the private repo with this plan and an empty skeleton as the first commit, scaffolded `apps/web` with create-next-app (no install), pushed `main` plus `feat/signal`, `feat/location`, `feat/web`, `feat/platform`, invited the other three as collaborators, filled real usernames into `CODEOWNERS` and `docs/team.md`, and opened one labeled issue per ticket. Verify all of that, then close REPO-00.

Already in the skeleton: `Makefile` (`check`, `contracts`, `publish-run`, `fetch-run`, `runs`), `services/seismic/pyproject.toml` (all Python deps, the `hq` entry point, the contracts package as an editable path dependency), `packages/contracts/python/pyproject.toml`, the TS workspace packages, and an empty `__init__.py` per lane subpackage. Add to them; don't replace them.

Still yours to add: web deps (`three @react-three/fiber @react-three/drei @react-three/postprocessing zustand`, dev `json-schema-to-typescript @types/three`), `scripts/gen-contracts.sh`, and Makefile targets `mock`, `run` (`hq run configs/showcase`), `export`, `dev`, `build`, `offline` as their tickets land. The Next.js scaffold ships its own `apps/web/AGENTS.md` telling agents to read `node_modules/next/dist/docs/` before writing Next.js code; keep it and follow it.

### Pipeline of record

This laptop runs the run of record. Around 10:30 PM, take H1's `data/cache/` folder (AirDrop or USB). Fetch lane runs with `make fetch-run RUN=<id>`, run `validate` and `export`, and commit the bundle through `main`. At the 6 PM pipeline freeze, rerun every stage end to end here from merged `main`, so the final numbers come from one reproducible run.

### Integration duty

- Merge each `feat/*` into `main` right after every integration meeting (10 PM, 12 AM, 4 AM, 8 AM, 2 PM, 6 PM, 12 AM). Merge only if `make check` passes on the branch.
- After merging, post in the team chat so everyone runs `git fetch && git merge origin/main`.
- Keep an eye on `docs/requests/`; route anything stuck to the owner in person.

### Providers

| Provider | Reads | Label | Notes |
| --- | --- | --- | --- |
| `StaticBundleProvider("mock")` | `/data/mock/` | red SYNTHETIC banner | Dev only; excluded from production by env flag |
| `StaticBundleProvider("showcase")` | `/data/showcase/` | "Showcase · {windowLabel} · run {runId}" | Default |
| `LiveProvider(apiBase)` | `/api/live/*` (`NEXT_PUBLIC_LIVE_API_BASE`) | "Live · last 2 h · updated {n} min ago" | Polls every 60 s, heartbeats every 2 s; a failed request (refused, 503, no answer within `LIVE_FETCH_TIMEOUT_MS`) fails over; a hang is noticed within one heartbeat plus that timeout; a later healthy heartbeat switches back (`services/api/README.md` → Failover) |
| `StaticBundleProvider("snapshot")` | `/data/snapshot/` | "Snapshot · generated {time} by our pipeline · run {runId}" | Automatic failover target; frozen with `hq-api freeze-snapshot`, committed through `main` |

### Validation panel

| Row | Source field | Shown only if |
| --- | --- | --- |
| Catalog recall X / N | `summary.recoveredCatalogCount`, `publicCatalogCount` | Always |
| Strict events "A (B not in public catalog)" | `summary.strictQualityCount`, `summary.strictAdditionalCount` (the parenthetical only when present) | Always |
| Median stations | `summary.medianStations` | Always |
| Median timing misfit | `summary.medianRmsS` | Always |
| Depth resolution (synthetic, all stations) ±m | `validation.synthetic.medianVErrM` | Always |
| STA/LTA strict events ("A of N candidates") | `validation.baseline[]` row with `method: "stalta"`, `associationProfile: "full"`: `tiers.A` and `candidates` | That row exists with finite counts (also when STA/LTA's strict count is zero and no gain can be claimed, REQ-H1-5). The rerun's PhaseNet strict count is never shown: it is not the STRICT counter |
| PhaseNet vs STA/LTA gain | `summary.baseline.gain` | Baseline ran and gain > 1 in both profiles |
| Chance associations, with a note "Mean of N timing scrambles; none (or about M per scramble) reached the strict tier" | `validation.nullTest.meanChanceEvents`, `nShuffles`, `meanChanceStrict` | Null test ran |

### Validation reruns and `validation_notes.json`

The null test and the baseline table rerun H2's `associate -> locate -> match -> assign_tiers` (docs/02 §5) with three keyword extensions: `locate(..., cache_dir=ctx.cache_dir, run_id=ctx.run_id, statics=<the run's statics.parquet rows>)` (REQ-H2-8 and REQ-H1-5 option (a), bound in `hq.validate.lanes.real_seismology_api`; `with_statics` binds the statics into an injected API) and `assign_tiers(events, matches, cfg, thresholds=<the run's ProcessingRun.tiering>, arrivals=located.arrivals, stations=<the stations table the rerun located with>)` (REQ-H2-9). Rerun events are located with the station terms the run's events carry, so they sit on the run's scale and the run's own bars, from H2's tier stage, apply to them; the stage fails naming stage `locate` when `statics.parquet` is absent and stage `tier` when `run.json` holds no bars, and never invents bars. `validate.yaml` `rerunBars` (default `run`) keeps H1's option (b) as an alternative (`reference`): the PhaseNet `full` rerun is made first with no `thresholds=`, so H2 derives bars from its matched set (a `TierError` below `tiering.minMatched` fails the stage naming the rule); those bars go as `thresholds=` to every other rerun and the rerun doubles as the baseline table's `(phasenet, full)` row (`hq.validate.reference`).

What a rerun did and could not do is written next to its numbers in `validation_notes.json` (`hq.validate.notes.ValidationNotes`, docs/02 §2): where the bars came from (`thresholds.source` `run` or `phasenetRerun`, the record, `nMatched`, the quantiles and the bars themselves as tier → metric → `{op, value}`), `staticsApplied: true` (the FYI-H2-8 caveat that reruns locate without statics no longer applies; a caller that uses the plain docs/02 call, such as H1's sweep scorer, records `false`), no locate flags (the `mapOnVolumeTop` rule is skipped, which only lets more rerun events into Tier A), how the nearest-station rule measured focal depth, and the `p_only` associator overrides. The G-R section records the one `magType` `publicCum` is drawn from (the calibration type H2 recorded, else `validate.yaml` `gr.publicMagType`; REQ-H2-13), the public magnitudes left out per type, the kill-switch gate applied (H2's recorded `gate.maxLooMae` when present) and `looMae` next to H2's `nullModelMae` (FYI-H2-7; a `looMae` not below it is logged as "no skill"). Panels that quote a null-test, baseline or G-R number should be able to show these notes beside it.

### Compliance

The README states plainly what was pre-event research (which data exists, which methods work) and what was built during HackGT (everything in the repo). Git history starts at 8:00 PM.

## Definition of done

- [ ] `make check` green on `main`; every lane's smoke tests run in it
- [ ] Contracts round-trip tests pass; TS regenerates with no diff
- [ ] Mock, showcase and snapshot load through the same components with zero code changes
- [ ] Exporter output validates; summary counts recompute from events
- [ ] Deployed URL and offline build both work with Wi-Fi off
- [ ] No numeric literals in UI copy; every panel row hides when its data is missing
- [ ] README and Devpost follow `docs/demo/pitch-and-qa.md`

## Kickoff prompt

```
You are the H4 Platform lane agent for Hidden Quakes (HackGT 13).
Before writing anything, read: CLAUDE.md, docs/00-project.md, docs/01-architecture.md,
docs/02-contracts.md, docs/03-schedule.md, docs/lanes/H4-platform.md and docs/demo/pitch-and-qa.md.
Write only inside the paths listed under "You own" in your lane doc, plus files your ticket names.
Your ticket: <TICKET-ID>. Start by restating its goal, files, inputs, outputs and acceptance test,
then write a short plan. Implement it with tests. Finish by running `make check` and the acceptance
test, and paste both outputs. If you need anything from another lane, write a request in
docs/requests/ and stop to tell me. Contracts change only through a reviewed PR.
```

Reviewer prompt (fresh session, after the builder finishes):

```
You are reviewing a Hidden Quakes H4 Platform PR. Read CLAUDE.md, docs/02-contracts.md and
docs/lanes/H4-platform.md, then review the diff on this branch against the ticket's acceptance test
and the Definition of done. Check: contracts match docs/02 exactly, no lane's code was edited, no
numeric literals in UI copy, synthetic data only via mock-fixture.py, exporter counts recompute from
events. List concrete findings with file:line, most serious first. Don't rewrite the code yourself.
```

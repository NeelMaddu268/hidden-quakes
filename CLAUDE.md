# Hidden Quakes — rules for every agent

You are one of several coding agents building **Hidden Quakes** at HackGT 13. Four humans each run their own agents, one lane per human. Your human told you which lane you're in. These rules beat convenience every time.

> Kickoff ran about 45 minutes late. Every Friday clock time before 10 PM in these docs means 45 minutes later; see the note at the top of `docs/03-schedule.md`.

Hidden Quakes rebuilds a denser earthquake catalog for Utah's geothermal-development region near Milford from **public** seismic waveforms (neural phase picking → association → relocation → quality tiers → public-catalog matching), then shows it as a 3D reveal against the sparse public regional catalog.

## Read before you write anything

1. This file.
2. `docs/00-project.md`: what we're building and the language rules.
3. `docs/01-architecture.md`: pipeline, repo tree, ownership, conventions.
4. `docs/02-contracts.md`: every interface you may consume or must produce.
5. `docs/03-schedule.md`: gates and freezes.
6. Your lane doc in `docs/lanes/`, and every section your ticket names.

## Lanes own paths

| Lane | Owns (write access) |
| --- | --- |
| **H1 Signal** | `services/seismic/hq/ingest/`, `hq/preprocess/`, `hq/pick/`, `hq/baseline/`, `hq/config/signal.py`, `services/seismic/configs/showcase/signal.yaml`, `services/seismic/tests/signal/`, `data/cache/mseed/`, `data/cache/stationxml/` |
| **H2 Seismology** | `services/seismic/hq/associate/`, `hq/locate/`, `hq/tier/`, `hq/match/`, `hq/magnitude/`, `hq/config/seismology.py`, `hq/config/run.py`, `services/seismic/configs/showcase/run.yaml`, `configs/showcase/seismology.yaml`, `services/seismic/configs/velocity/`, `services/seismic/tests/seismology/`, `data/cache/ttgrids/`, `data/cache/velocity/` |
| **H3 Visualization** | `apps/web/src/scene/`, `apps/web/src/drawer/`, `apps/web/src/state/`, `packages/visualization/`, `apps/web/public/terrain/`, `scripts/bake-dem.py` |
| **H4 Platform** | Everything else: repo root files, `packages/contracts/`, `packages/config/`, `services/seismic/hq/export/`, `hq/validate/`, `hq/cli.py`, `hq/runs.py`, `hq/config/__init__.py`, `hq/config/export.py`, `configs/showcase/export.yaml`, `services/api/`, `apps/web/src/app/`, `apps/web/src/shell/`, `apps/web/src/providers/`, `apps/web/public/data/`, `scripts/` (except `bake-dem.py`), `data/fixtures/`, `docs/` (except each lane's own lane doc) |

Run outputs under `data/showcase/runs/<runId>/` are written by whichever stage produces them (see `docs/01`), never edited by hand.

1. **Write only inside your lane's paths and the files your ticket lists.** Reading anything is fine and encouraged.
2. **Need something from another lane?** Append a request to `docs/requests/<OWNER>.md` (format in `docs/requests/README.md`) and tell your human. Never "just fix" another lane's file, even a one-liner.
3. **Contracts are frozen after 8:30 PM Friday.** That covers `packages/contracts/` and every interface in `docs/02`. If you believe one is wrong, stop and tell your human. Don't work around it.

## Honesty rules (code, UI copy, docs, commit messages)

4. **Never write a count, magnitude, date, station code or percentage as a fact** into UI copy, pitch text or docs. Numbers render from pipeline output.
5. **Synthetic data comes only from `scripts/mock-fixture.py`** and always carries `isSynthetic: true`. Tests may build small synthetic data inside the test.
6. **Language.** Say "candidate events", never "confirmed earthquakes". Never "caused by". Never attribute seismicity to FORGE, Cape Station or any operator. Never "predict". Say "public regional catalog", never "official catalog". The full list is in `docs/00`.
7. **No pre-event code, fixtures, notebooks or outputs.** Public libraries, public data, public models and published papers are fine.

## Engineering bar

8. **Every knob lives in config.** Pipeline parameters go in `services/seismic/configs/showcase/*.yaml` and get recorded in `ProcessingRun`. No magic constants in code.
9. **Typed and linted.** Python 3.11, type hints everywhere, `ruff` clean. TypeScript `strict`, `eslint` clean.
10. **Tested.** Every module ships unit tests. Mark the fast ones `@pytest.mark.smoke`; a lane's smoke tests finish in under 30 s. Tests never hit the network; use small synthetic data or tiny recorded fixtures under `tests/`.
11. **Deterministic.** Seed every source of randomness. Identical config gives identical outputs.
12. **Fail loudly.** No bare `except`, no silent fallbacks that change results. Log counts and runtimes at every stage.
13. **Prove it before you finish.** Run `make check` and your ticket's acceptance test, and paste both outputs into the PR description.

## Git

14. Work on an `agent/<TICKET>` branch cut from your lane's `feat/<lane>` branch, in its own worktree.
15. Prefix every commit with the ticket ID, e.g. `LOC-02: add eikonal travel-time tables`.
16. Fill in `.github/pull_request_template.md` for every PR, including `Closes #<issue>` for the ticket's GitHub issue.
17. **Only H4 merges into `main`**, at integration meetings. Two exceptions, because every lane is waiting on them: H2 pushes `run.yaml` + `hq/config/run.py` straight to `main` by 8:05 PM, and H4 lands CONTRACT-01, RUN-01, FIX-01 and API-01 on `main` as they pass review (8:25–9:00 PM).
18. After every change to `main`, merge it into your lane branch: `git fetch && git merge origin/main`.

## Data between laptops

19. Run outputs (`data/showcase/runs/`) and the waveform cache (`data/cache/`) are never committed.
20. Share a run with `make publish-run RUN=<runId>`; get one with `make fetch-run RUN=<runId>`; list them with `make runs`. Only the lane that ran the latest stage on a run publishes it.
21. The web bundle (`apps/web/public/data/showcase/`) is the one data product committed to git, by H4's exporter.

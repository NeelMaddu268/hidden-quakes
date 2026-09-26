# Hidden Quakes

*What the public can't see beneath Utah's geothermal frontier.* Built at HackGT 13.

Public regional earthquake catalogs show only a sparse slice of the microseismicity around Utah's geothermal-development region near Milford. Operators and research teams run dense downhole and fiber arrays that see far more; the public gets the regional catalog. Hidden Quakes is an open, public-data-only seismic layer: it rebuilds a denser, quality-tiered, inspectable catalog of **candidate events** directly from raw public waveforms (neural phase picking, multi-station association, relocation, quality tiers, matching against the public regional catalog) and shows it underground, in 3D, as a reveal against the public view. It is for people without operator data: researchers, journalists, regulators, local communities and energy observers.

Our contribution is product, pipeline, public access and visual explainability, not a new seismology algorithm. PhaseNet, PyOcto, QuakeFlow, GaMMA and published research catalogs for the region all exist, and we say so.

**Every number on screen is rendered from the exported run's `meta.json` and `validation.json`. This README therefore quotes none.** Where a number would go, open the deployed page or `apps/web/public/data/showcase/meta.json` (`summary` and `run`) and `validation.json` in the same folder.

## How the reveal works

1. The scene opens on dark terrain with one glowing geothermal reference and one counter: **PUBLIC**, the number of events the public regional catalog lists in the showcase window. That count comes from a catalog query saved in the run, never assumed.
2. **REVEAL HIDDEN SIGNAL** (or Space) plays the candidate events into the volume in `revealOrder` (strict tier first, then the rest, time-ordered within a tier). The **RECOVERED** counter climbs to `summary.candidateCount`; **STRICT** settles on `summary.strictQualityCount`.
3. **PUBLIC / ALL / STRICT** pills (or S) filter the view. **P** switches to plan view with a depth section. **T** turns on the time scrubber, which replays the window.
4. **E** opens the evidence drawer on the hero event (the strict event located with the most stations; any other dot is a click away): a record section sorted by distance, neural P and S picks, and the arrival times the final location implies. Agreement between the two is what makes a dot an event.
5. The **Validation** card and **Run details** (D) show the run's own checks and the full `ProcessingRun` config verbatim. Every row hides itself when its source field is missing, so nothing on screen is ever typed in.

Keyboard map: Space (next beat), R (reset), S (strict), T (time), E (evidence), P (plan/oblique), D (run details), Esc (close).

## Pipeline

Two halves joined by files (`docs/01-architecture.md`). A batch pipeline (Python package `hq` in `services/seismic`) writes an immutable run directory under `data/showcase/runs/<runId>/`; an exporter turns one chosen run into a static data bundle; the web app (`apps/web`, Next.js + React Three Fiber) reads only that bundle through a provider. Nothing on the demo path depends on a live service.

| Stage | Owner | Module | Writes |
| --- | --- | --- | --- |
| inventory | H1 Signal | `hq.ingest.inventory` | `stations.parquet` (+ StationXML in `data/cache/stationxml/`) |
| catalog | H2 Seismology | `hq.match.catalog` | `catalog.parquet`, `catalog.quakeml` |
| download | H1 Signal | `hq.ingest.download` | `gaps.parquet` (+ miniSEED in `data/cache/mseed/`) |
| pick | H1 Signal | `hq.pick` (PhaseNet via SeisBench) | `picks.parquet`, `known/` |
| baseline | H1 Signal | `hq.baseline` (STA/LTA) | `picks_stalta.parquet`, `baseline_sweep.parquet` |
| associate | H2 Seismology | `hq.associate` (PyOcto) | `assoc_events.parquet`, `assoc_picks.parquet`, `sweep.parquet` |
| locate | H2 Seismology | `hq.locate` | `events_located.parquet`, `residuals.parquet`, `statics.parquet`, `synthetic.json` |
| match | H2 Seismology | `hq.match` | `matches.parquet`, `match_sensitivity.parquet` |
| tier | H2 Seismology | `hq.tier` | `events.parquet` (final `SeismicEvent` rows) |
| magnitude | H2 Seismology | `hq.magnitude` | updates `events.parquet`, `magnitude.json` |
| validate | H4 Platform | `hq.validate` | `validation.json` |
| export | H4 Platform | `hq.export` | `apps/web/public/data/<mode>/` |

Every stage is `run(ctx: RunContext) -> None`. `hq run configs/showcase` runs them in order; `hq stage <name> --run <runId>` reruns one. Picks are stored once at a low probability floor and filtered downstream, waveforms download once into a shared cache, and every run records its full config in `run.json` (`ProcessingRun`), so every number on screen traces to a config.

Every knob lives in `services/seismic/configs/showcase/*.yaml` (`run.yaml` window, bbox and origin; `signal.yaml` stations, preprocessing profiles and picker; `seismology.yaml` velocity model, catalog query, association, location, tiers; `validate.yaml` null test, baseline and Gutenberg–Richter; `export.yaml` bundle, evidence snippets and reference features). Unknown keys are errors; no magic constants live in code.

## Running it

Prerequisites: `uv` (Python, see `requires-python` in `services/seismic/pyproject.toml`), `pnpm` (Node, see `packageManager` in `package.json`), and `gh` for sharing runs.

```bash
pnpm install                              # JS workspace (web app, contracts, tokens)
cd services/seismic && uv sync && cd -    # Python pipeline
cd services/api && uv sync && cd -        # live worker (optional)

make mock                 # synthetic bundle into apps/web/public/data/mock (SYNTHETIC banner)
make dev                  # web app on localhost; open /?mode=mock
make run                  # showcase pipeline: hq run configs/showcase (needs network + credentials)
make export RUN=<runId>   # run -> apps/web/public/data/showcase/, validated
make build                # static export into apps/web/out
make offline              # build, then serve apps/web/out locally; works with Wi-Fi off
make api                  # live worker + API (services/api/config.yaml); ?mode=live
make check                # typecheck + lint + smoke tests, before every PR
make check-copy           # scans README, docs/demo and the shell for numbers-as-facts and forbidden phrases
```

Data modes, picked with `?mode=`: `showcase` (default, the frozen final run), `mock` (synthetic, dev only, refused in production builds unless `NEXT_PUBLIC_ALLOW_MOCK=1`), `live` (behind `NEXT_PUBLIC_LIVE_ENABLED`, polls the API) and `snapshot` (the last good live window; automatic failover target, never a pill). Three of the four are the same static reader pointed at a different folder.

Run tables move between laptops with `make publish-run RUN=<runId>`, `make fetch-run RUN=<runId>` and `make runs` (GitHub prereleases); the waveform cache is copied by hand; the web bundle under `apps/web/public/data/showcase/` is the one data product committed to git. Deployment is a static export (`docs/deploy.md`).

## Data sources

All public. Named as they appear in the code and config.

| Source | Used for | Where |
| --- | --- | --- |
| EarthScope FDSN services (`fdsnClient: EARTHSCOPE`, station and dataselect; MUSTANG for availability) | Station metadata (StationXML, including channel depth for borehole sensors) and continuous waveforms | `configs/showcase/signal.yaml`, `hq.ingest` |
| Public regional catalog via the USGS ComCat FDSN event service (`provider: USGS`; UUSS solutions for the region; `CatalogEvent.source` = `<contributor> via USGS ComCat`) | The PUBLIC count, catalog recall, tier calibration, magnitude calibration | `configs/showcase/seismology.yaml` → `catalog`, `hq.match.catalog` |
| USGS 3DEP elevations (EPQS point query for the origin and station-elevation checks; terrain from AWS Terrain Tiles, Terrarium encoding, which carry 3DEP in the contiguous US) | Reference surface, DEM-checked station elevations, the terrain mesh | `run.yaml` → `origin`, `signal.yaml` → `stations`, `scripts/bake-dem.py` |
| DOE Geothermal Data Repository (GDR): a published FORGE 1D velocity model, the Cape EGS / Utah FORGE 3D velocity model (Nakata et al., CC BY 4.0), and well drilling-data submissions (directional surveys) | Travel times and location; well trajectories and a wellhead as reference features | `configs/velocity/forge_1d.csv`, `seismology.yaml` → `velocity`, `export.yaml` → `features` |
| Utah Geological Survey (UGS) FORGE extent layers | Project-area and study-area outlines as reference features | `export.yaml` → `features` |

Every reference feature carries a `SourceRef` (citation, URL, `verified`). A feature whose coordinates were not read from the primary publication is `verified: false` and renders as approximate. None of this says anything about what causes seismicity in the region.

## Candidate events and quality tiers

A **candidate event** is a set of picks that agree across multiple stations through a velocity model and locate with a stored uncertainty. It is not a verified earthquake, and we never call it one. Tiers come from the data, not from textbooks: the public-catalog events we recovered form the reference set, and each threshold is a quantile of that set, stored with its source quantile in `ProcessingRun.tiering`.

| Tier | Filter pill | Meaning |
| --- | --- | --- |
| A | STRICT | On every location metric, at least as good as three-quarters of the recovered public events; depth not pinned to a grid edge; a station close enough to constrain depth |
| B | (ALL) | On every metric, no worse than the worst recovered public event |
| C | (ALL) | Associated and located, but outside that range |

Caveat we own: public-catalog events are the larger ones, so tiers are conservative for small events. `meanPickProb` on each event is the picker's confidence, not a probability that the event is real. We never quote a false-positive rate: there is no ground truth for events the public catalog lacks. The proxies are the tiers and the null test below.

## Validation

Everything below is computed by the pipeline and written to `validation.json` and `meta.json`; the Validation card renders it and hides any row whose source is missing. The kill switches in `docs/03-schedule.md` remove a claim when its check fails; they never remove the reveal.

| Check | What it does | Bundle field | Claim it licenses | Kill switch |
| --- | --- | --- | --- | --- |
| Catalog recall | One-to-one matching of candidates against the public regional catalog for the exact window; every miss is listed | `summary.recoveredCatalogCount` / `publicCatalogCount`, `summary.unmatchedPublicIds` | "Using only public waveforms, we recovered X of N public events" | Poor recall after reasonable debugging ends the science track |
| Synthetic depth test | Synthetic events on the real station geometry, located with the same code; reports median horizontal and vertical error and depth bias | `validation.synthetic` (`medianVErrM`, `p90VErrM`, `medianDepthBiasM`) | "This station geometry resolves depth to about ±V m" | Depth: too many strict events pinned to the grid top, or a nonphysical vertical distribution → no structure claims; plan view becomes the hero |
| Null test | Reruns of associate → locate → match → tier with each station's picks time-shifted, same config, seeded | `validation.nullTest` (`meanChanceEvents`, `meanChanceStrict`) | "Chance associations on time-scrambled picks: about M" | Reported as is |
| Baseline comparison | Two pickers (PhaseNet, STA/LTA) × two association profiles (`full`, `p_only`) through the same downstream code | `validation.baseline`, `summary.baseline.gain` (present only when the gain holds in both profiles) | "At comparable quality, neural picking yields G× more strict events" | STA/LTA within a comparable fraction of PhaseNet's strict count → drop the claim, keep the table |
| Gutenberg–Richter | Aki–Utsu b with Shi–Bolt sigma, Mc by maximum curvature plus an offset, public curve versus recovered curve | `validation.gr`, `validation.magnitude` (`n`, `looMae`) | "Magnitudes extend the trend below the public catalog's completeness" | Leave-one-out MAE above the configured cap → no magnitude sizing, no G-R |
| Association sweep | Recall, candidates and strict count across the association grid; the knee is the chosen config | `validation.sweep` (plotted in Run details) | Why these thresholds | — |

What we never claim: that operators lack better monitoring, that any event was missed by anyone, fracture geometry (at most "structure", and only if the depth gate passes), or a mechanism.

## What this is not

> - **Not an attribution.** Several geothermal operations share this region. We never attribute any event to Utah FORGE, Cape Station or any operator; attribution needs operator data we don't have. Reference features on the map are context, with their sources cited.
> - **Not a catalog of verified earthquakes.** These are candidate events, tiered by location quality against the public regional catalog, and we don't claim every one is real. "Confirmed earthquake" is a phrase this project never uses. <!-- copy-ok -->
> - **Not a forecast.** We detect, associate and locate what already happened in a fixed window. Nothing here predicts anything. <!-- copy-ok -->
> - **Not a claim on the public regional catalog.** It lists what it lists for its own purposes; we say "the public regional catalog shows N here," and no more.
> - **Not a new algorithm.** The picker is pretrained PhaseNet (we trained nothing); the associator is PyOcto; the methods are published. What's ours is the product around them.

## Honesty rules the code enforces

- UI copy carries no digits: `apps/web/src/shell/copy.test.ts` parses every JSX text node and readable attribute in the shell and the page and fails on any number. Counters and panels read only `AnalysisSummary`, `Validation` and the demo store.
- Docs carry no numbers as facts: `scripts/check-copy.sh` scans this README, `docs/demo/*.md` and the shell for digit-bearing claims outside code and placeholders, and for the forbidden phrases in `docs/00-project.md`.
- Synthetic data comes only from `scripts/mock-fixture.py`, carries `isSynthetic: true`, and shows a full-width SYNTHETIC banner. Production builds refuse it.
- Every run is deterministic (seeded) and records its full config; the summary counts in `meta.json` are recomputed from `events.json` when the bundle is checked.
- Language: "candidate events", never verified ones; "public regional catalog", never a governmental or authoritative one; detect, associate and locate, never forecast; never name a cause.

## Compliance (HackGT)

- **Pre-event:** research only, into which public data exist (EarthScope waveforms, the public regional catalog, GDR velocity models and well surveys, UGS layers, 3DEP) and which published methods work (PhaseNet via SeisBench, PyOcto, grid and eikonal location, quantile-based quality tiers). That research produced the planning documents in `docs/`, which are the first commit.
- **Built during HackGT:** everything else in this repository: the `hq` pipeline and its tests, the contracts, the exporter and validator, the live worker, the web app (scene, shell, drawer, providers), the mock generator, the terrain bake, the scripts and this README. Git history starts at the kickoff commit (`REPO-00`), whose content is the plan and an empty skeleton. No pre-event code, fixtures, notebooks or outputs entered the repo.
- **Third-party components:** public libraries (ObsPy, SeisBench, PyOcto, scikit-fmm, SciPy, PyProj, FastAPI, Next.js, three.js, React Three Fiber, zustand), public pretrained model weights (PhaseNet through SeisBench; weights chosen by an A/B on public-catalog events, no training), public data and published papers only. Sources are cited in the config files and carried into the bundle as `SourceRef`s.
- Every number in the submission comes from the pipeline of record run during the event; none from pre-event research.

## Team and lanes

Four humans, one lane each, each running their own coding agents (`docs/team.md`, `CLAUDE.md`).

| Lane | Owns | GitHub |
| --- | --- | --- |
| H1 Signal | Station inventory, waveform download and cache, preprocessing profiles per sensor type, PhaseNet picking, STA/LTA baseline | @hueywinn |
| H2 Seismology | Public-catalog query, association, travel times and location with statics and uncertainty, matching, quality tiers, magnitudes, velocity models | @NeelMaddu268 |
| H3 Visualization | The 3D scene, terrain, the reveal, filters, plan view, evidence drawer, time scrubber, design tokens | @SN-P946 |
| H4 Platform | Repo, contracts, config loader and stage runner, providers, shell, exporter, validation, deploy, live worker, the story (this README, Devpost, pitch) | @Sririthishpalani-max |

## Repository layout

```
CLAUDE.md                          shared rules every agent follows (AGENTS.md links to it)
Makefile                           every command above
docs/                              00 project · 01 architecture · 02 contracts · 03 schedule · lanes/ · demo/ · requests/
apps/web/                          Next.js app: src/app + src/shell + src/providers (H4), src/scene + src/drawer + src/state (H3)
apps/web/public/data/<mode>/       generated bundles (showcase committed; mock from the fixture script)
apps/web/public/terrain/           baked DEM tiles (scripts/bake-dem.py)
services/seismic/                  Python pipeline package `hq`, configs/showcase/*.yaml, tests per lane
services/api/                      live worker (FastAPI) and the Live API
packages/contracts/                shared models: Python source of truth + generated TypeScript
packages/visualization/            design tokens
scripts/                           gen-contracts, mock-fixture, export-showcase, serve-offline, bake-dem, check-copy
data/                              runs and caches; never committed
```

Further reading: `docs/00-project.md` (what we may and must not claim), `docs/01-architecture.md`, `docs/02-contracts.md`, `docs/03-schedule.md`, `docs/deploy.md`, `services/api/README.md`, `docs/demo/pitch-and-qa.md`.

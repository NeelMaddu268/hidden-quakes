# Devpost submission (draft, DEMO-03)

Story and language follow `docs/demo/pitch-and-qa.md`; claims follow `docs/00-project.md`. Every number is a placeholder of the form `<from FILE: field>` and is filled on Saturday evening from the exported run of record (`apps/web/public/data/showcase/`). Nothing numeric is typed in by hand. The "Fill-in checklist" at the bottom lists every placeholder and its source; one grep finds them all:

```
grep -o '<[a-z][^>]*>' docs/demo/devpost.md | sort -u
```

A placeholder whose source field is null in the bundle means the claim was cut by a kill switch (`docs/03-schedule.md`); delete the sentence, don't estimate.

---

## Title

**Hidden Quakes**

## Tagline

What the public can't see beneath Utah's geothermal frontier.

## Thumbnail

Semi-transparent terrain, the geothermal reference outline, public points in white, strict candidate events in amber below, and the counter `<from meta.json: summary.publicCatalogCount> PUBLIC → <from meta.json: summary.candidateCount> RECOVERED`. It has to read with no video; if it looks like random dots, redesign it.

## Inspiration

Enhanced geothermal is expanding around Milford, Utah. Operators and research teams see underground: dense downhole geophones and fiber arrays give them detailed pictures of the microseismicity there. Someone without operator data, a county official asked at a meeting or a reporter checking a claim, gets the public regional catalog, which lists a sparse slice of that activity. Microseismicity is also what the oversight of geothermal operations (traffic-light protocols) is built on. We wanted to know how much of the underground picture the public seismic network is already hearing, and to show it in a way that person can read in five seconds, share by link, and download.

## What it does

The public regional catalog lists `<from meta.json: summary.publicCatalogCount>` events under Utah's geothermal frontier in the showcase window (`<from meta.json: run.windowLabel>`). We rebuilt the catalog from raw public seismometers with neural phase picking, multi-station association, relocation and quality tiers: `<from meta.json: summary.candidateCount>` candidate events, `<from meta.json: summary.strictQualityCount>` at strict quality, underground where the public view is nearly empty.

Three interactions:

1. **The reveal.** The scene opens on dark terrain, one glowing geothermal reference and one counter, PUBLIC. Press REVEAL HIDDEN SIGNAL and the candidate events play into the volume, strict tier first, while the counter climbs. PUBLIC / ALL / STRICT filters the view; plan view is one key away.
2. **The evidence.** Click any dot (or press E for the hero event, the strict event located with the most stations) and the drawer shows the waveforms that put it there: `<from events.json: hero quality.nStations>` stations sorted by distance, the neural P and S picks on each, and the arrival times implied by the final location. They agree, and that agreement is what makes a dot an event.
3. **Time.** The scrubber replays the window, so clustering in space and time is visible instead of described.

Everything on screen is a candidate event, tiered. Strict means every quality metric is within the range reached by three-quarters of the public-catalog events we recovered; each bar is set per metric, and the run stores how many of those events meet every bar at once (`run.json` → `tiering.matchedSet.meetingEveryBar.A` over `tiering.matchedSet.n`).

## How we built it

Two halves joined by files. A Python pipeline (`hq`) writes an immutable run directory, stage by stage:

- **Ingest (EarthScope FDSN):** station metadata at channel level, including sensor depth for borehole instruments, and continuous waveforms for every station in the box, cached once.
- **Preprocess:** one profile per sensor type with explicit anti-aliasing, because the borehole sensors sample far faster than the surface ones and the picker expects one rate.
- **Pick:** pretrained PhaseNet through SeisBench, weights chosen by an A/B on public-catalog events (`<from meta.json: run.pickerWeights>`). We trained no picker: site labels for picking don't exist.
- **Associate:** PyOcto turns single-station picks into events that agree across stations through a velocity model.
- **Locate:** our own grid locator (coarse-then-fine search over eikonal travel-time tables solved with scikit-fmm) on a published FORGE velocity model from the DOE Geothermal Data Repository (`<from meta.json: run.velocityModel.name>`), with station statics fitted on the recovered public events, per-event uncertainty and a synthetic recovery test on the real station geometry.
- **Match and tier:** one-to-one matching against the public regional catalog (USGS ComCat, UUSS solutions). The recovered public events set the quality bar: each tier threshold is a quantile of that set, stored in the run.
- **Magnitude:** a local magnitude (`ML_cal`) calibrated on the matched public events of one catalog magnitude type, with a leave-one-out error reported next to a null-model error; the stage nulls every magnitude when its gate fails, so no magnitude on screen outlives a bad calibration.
- **Decoy test (only if `confidence.json` is in the bundle):** one small model of our own, the `<from confidence.json: label>`, trained on decoy events made by scrambling station clocks. Each candidate's score says how much its timing looks like a real association rather than a scrambled-clock decoy; on held-out events the model's ROC AUC is `<from confidence.json: model.heldOut.rocAuc>`. The score is shown in the evidence drawer and is not a probability that an event is an earthquake; the tiers stay the quality call.
- **Validate and export:** a null test, an STA/LTA baseline, a Gutenberg–Richter curve whose public side counts only the catalog events of the calibration magnitude type, then a static bundle (`meta.json`, `events.json`, `validation.json`, evidence snippets) that the web app reads through a provider. Nothing on the demo path depends on a live service.

The web app is Next.js with React Three Fiber: terrain baked from public elevation tiles, events as instanced points with error halos, borehole sensors drawn at their true depth, reference features from GDR well surveys and UGS layers with their sources cited, a plan view with a true-scale depth section, an evidence drawer with the record section behind every dot, and a time scrubber that replays the window over a histogram of it. The shell renders no digit of its own: a test parses every text node and fails on a number. Every run records its full config; the Run details panel prints it verbatim.

Four humans, one lane each (signal, seismology, visualization, platform), each running their own coding agents against a shared plan with frozen contracts and one owner per path.

## Validation

Numbers below are read from the exported run's `meta.json` and `validation.json` (the same fields the deployed page's Validation card renders).

- Public-catalog recall: `<from meta.json: summary.recoveredCatalogCount>` of `<from meta.json: summary.publicCatalogCount>`; every miss is listed in the run.
- Candidate events: `<from meta.json: summary.candidateCount>`, of which `<from meta.json: summary.additionalCount>` are not in the public catalog; `<from meta.json: summary.strictAdditionalCount>` of those pass the strict tier.
- Median stations per event `<from meta.json: summary.medianStations>`; median travel-time residual `<from meta.json: summary.medianRmsS>` s.
- Depth resolution of this station geometry, from the synthetic test, where every event is recorded on every station: about ±`<from validation.json: synthetic.medianVErrM>` m. A typical candidate, recorded on fewer stations, resolves less finely.
- Null test: with each station's timing scrambled and the same config, association yields about `<from validation.json: nullTest.meanChanceEvents>` chance events, versus `<from meta.json: summary.candidateCount>` with real timing, and none of the chance events reached the strict tier in any of the `<from validation.json: nullTest.nShuffles>` scrambles (that clause only while `nullTest.meanChanceStrict` is exactly zero).
- Baseline, one of two sentences, both conditional (H1 is rescoring STA/LTA on the run's statics scale, so nothing is asserted until the run of record's `validation.json` is exported). Only if `summary.baseline` is present: at comparable quality, neural picking yields `<from meta.json: summary.baseline.gain>`× the strict events of STA/LTA. Otherwise, only if `validation.baseline` holds both `full` rows: at the strict tier, PhaseNet produced `<from validation.json: baseline[method=phasenet, associationProfile=full].tiers.A>` candidate events and STA/LTA produced `<from validation.json: baseline[method=stalta, associationProfile=full].tiers.A>`, through the same downstream code and the same tier bars. That PhaseNet count is a rerun that applies one statics table to every event; the strict count in the headline comes from the published run, which leaves each matched event out of its own statics, so the two differ by a few events. Whenever both appear in the same text, this sentence goes with them. If neither holds, no baseline sentence.
- Magnitudes (only if `validation.magnitude` is present): a local magnitude, `<from meta.json: run.matching.magnitude.calibrationMagType>`-calibrated `ML_cal`: typical error `<from validation.json: magnitude.looMae>` magnitude units (leave-one-out) versus `<from meta.json: run.matching.magnitude.leaveOneEventOut.nullModelMae>` for a no-skill baseline (a model that gives each event the mean magnitude of the others), from `<from validation.json: magnitude.n>` calibration events of that one catalog magnitude type; `<from meta.json: run.matching.magnitude.magnitudes.belowCalibratedRange>` of `<from meta.json: run.matching.magnitude.magnitudes.written>` candidate magnitudes are extrapolated below the smallest calibration event.
- Gutenberg–Richter (only if `validation.gr` is present): the public curve counts only the catalog events of the calibration magnitude type; the recovered curve is `ML_cal`. We claim no lower completeness magnitude: the public side has too few events of that type for a completeness estimate, and the recovered curve's low end is extrapolated and censored, so we describe the curve and quote no b-value from it.

## Limits

- **Candidate events, not verified earthquakes.** Each needs consistent picks across multiple stations; strict means every quality metric is within the range reached by three-quarters of the public events we recovered. We don't claim all of them are real, and there is no false-positive rate because there is no ground truth for events the public catalog lacks; the null test and the tiers are the proxies.
- **Sparse public geometry.** Depth is the weakest dimension. Published catalogs from downhole arrays are far denser and sharper than ours; every halo is that event's own error, and we show depths as located without reading a geometry or a mechanism into their pattern (that would need relative relocation, which we did not run).
- **No attribution.** Several operations share the region. We never name a cause for any event, and we never attribute seismicity to Utah FORGE, Cape Station or any operator.
- **Pretrained picker.** PhaseNet weights are public and unchanged; site-specific training needs labels we don't have.
- **One site, one window.** Config-driven, but run nowhere else yet.

## Challenges

- **Borehole sample rates.** The borehole sensors sample far faster than the surface stations. We built per-sensor-type preprocessing profiles with explicit anti-aliasing and A/B'd a time-stretch variant to keep the high-frequency band, rather than resampling blindly.
- **Depth credibility.** A sparse surface network trades depth against origin time. We required S picks and a near station for the strict tier, added station statics, measured depth resolution on synthetic events with the real geometry, and kept a plan-view hero ready in case the depth gate failed. The Saturday depth call passed on amended criteria: the 3D hero stays, and we say nothing about what the pattern of depths means.
- **Datums.** Public-catalog depths are relative to sea level, station elevations come from a DEM, borehole sensors sit far below their wellhead, and well surveys are published in survey feet on a state grid. One vertical convention (elevation above sea level, everywhere) and one horizontal one (UTM minus a fixed origin) made every comparison free.
- **Honesty at hackathon speed.** Every number on screen had to come from data, so the shell carries no digits and the docs quote none; a test and a script enforce it.

## Accomplishments that we're proud of

- A reproducible public-data pipeline from raw waveforms to a tiered catalog, with the whole config recorded in every run.
- A reveal that reads in five seconds and an evidence drawer that survives a skeptical seismologist.
- Quality tiers anchored to the public catalog rather than to hand-picked thresholds.
- A validation card that hides any row whose data is missing, so the demo never overstates.
- Four lanes, frozen contracts and one owner per path: four people and their agents shipped one product without stepping on each other.

## What we learned

- PhaseNet does one station at a time; the event only exists when picks agree across stations through a velocity model, so association, location, uncertainty and tiering are where the work is.
- Depth resolution is set by station geometry before any algorithm choice. Measuring it synthetically, on the real geometry, is the honest way to say how good the depths are.
- The public regional catalog is both the reference the public actually has and the only independent label set, which makes it the right thing to validate against and to calibrate tiers on.
- Saying "candidate event" every time is a feature, not a hedge.

## What's next

- Other public-network sites: the pipeline is config-driven (area, stations, velocity model); the real per-site cost is a velocity model and station QC.
- Relative relocation of the strict tier for sharper clusters.
- Continuous live operation: the live worker already reruns the pipeline on a rolling window and fails over to a snapshot; it needs hardening and a longer uptime record.
- A public-catalog recall check on a second, well-studied sequence.

## Built with

Pipeline (`services/seismic`, package `hq`): Python, ObsPy, SeisBench (pretrained PhaseNet), PyOcto, scikit-fmm, SciPy, NumPy, pandas, PyArrow, xarray, netCDF, PyProj, Pydantic, PyYAML, Requests, Matplotlib; uv, pytest, ruff. Live worker (`services/api`): FastAPI, Uvicorn. Contracts (`packages/contracts`): Pydantic models as the source of truth, JSON Schema, TypeScript generated with json-schema-to-typescript. Web app (`apps/web`): TypeScript, Next.js (static export), React, three.js, React Three Fiber, drei, react-postprocessing, zustand, Vitest, Testing Library, ESLint, pnpm; deployed on Vercel. Data: EarthScope FDSN services, USGS ComCat, USGS 3DEP elevations (AWS Terrain Tiles, Terrarium encoding), DOE Geothermal Data Repository, Utah Geological Survey.

## Links

- Live demo: `<deployed URL>`
- Repository: `<repo URL>`
- Video: `<video URL>`

## HackGT compliance

Pre-event work was research into which public data exist and which published methods work; it produced the planning documents that are the repository's first commit. Everything else in the repository (pipeline, contracts, exporter, web app, scripts, tests, docs) was built during HackGT by the four of us and our agents; git history starts at the kickoff commit. Public data, public pretrained models (PhaseNet via SeisBench) and published papers only; no pre-event code, fixtures or outputs.

---

## Fill-in checklist

Fill every placeholder from the exported run of record. `meta.json` and `validation.json` are in `apps/web/public/data/showcase/`; the evidence file is `evidence/<heroEventId>.json` in the same folder, with `heroEventId` in `meta.json` → `scene.heroEventId`. Conditional placeholders are removed, sentence and all, when their field is null.

| Placeholder | Source | Condition |
| --- | --- | --- |
| `<from meta.json: summary.publicCatalogCount>` | `meta.json` → `summary.publicCatalogCount` | always |
| `<from meta.json: summary.candidateCount>` | `meta.json` → `summary.candidateCount` | always |
| `<from meta.json: summary.strictQualityCount>` | `meta.json` → `summary.strictQualityCount` | always |
| `<from meta.json: summary.recoveredCatalogCount>` | `meta.json` → `summary.recoveredCatalogCount` | always |
| `<from meta.json: summary.additionalCount>` | `meta.json` → `summary.additionalCount` | always |
| `<from meta.json: summary.strictAdditionalCount>` | `meta.json` → `summary.strictAdditionalCount` | always |
| `<from meta.json: summary.medianStations>` | `meta.json` → `summary.medianStations` | always |
| `<from meta.json: summary.medianRmsS>` | `meta.json` → `summary.medianRmsS` | always |
| `<from meta.json: summary.baseline.gain>` | `meta.json` → `summary.baseline.gain` | only if `summary.baseline` is not null (VAL-01 writes it only when the gain holds in both association profiles); otherwise delete the sentence and fall back to the two strict counts below |
| `<from validation.json: baseline[method=phasenet, associationProfile=full].tiers.A>` / `<from validation.json: baseline[method=stalta, associationProfile=full].tiers.A>` | `validation.json` → `baseline[]`: the `tiers.A` of the row with `method: "phasenet"` and of the row with `method: "stalta"`, both with `associationProfile: "full"` (the Validation card's "Strict events, PhaseNet vs STA/LTA" row) | only if both rows exist; used only when `summary.baseline` is null; otherwise delete the sentence |
| `<from meta.json: run.windowLabel>` | `meta.json` → `run.windowLabel` | always |
| `<from meta.json: run.pickerWeights>` | `meta.json` → `run.pickerWeights` | always |
| `<from meta.json: run.velocityModel.name>` | `meta.json` → `run.velocityModel.name` | always (H2's dict; if the key is named differently, take the model name from `run.velocityModel`) |
| `<from validation.json: synthetic.medianVErrM>` | `validation.json` → `synthetic.medianVErrM` (synthetic test, every event on every station; keep the "fewer stations" sentence with it) | always |
| `<from validation.json: nullTest.meanChanceEvents>` | `validation.json` → `nullTest.meanChanceEvents` | only if `nullTest` is not null; otherwise delete the sentence |
| `<from validation.json: nullTest.nShuffles>` | `validation.json` → `nullTest.nShuffles` | only while `nullTest.meanChanceStrict` is exactly zero; otherwise delete the "none reached the strict tier" clause |
| `<from confidence.json: label>`, `<from confidence.json: model.heldOut.rocAuc>` | `confidence.json` (`hq.confidence/1`, ML-01, H2's `agent/ML-01-ui` PR) → `label`, `model.heldOut.rocAuc` two decimals | only if `confidence.json` is in the bundle with a held-out AUC; otherwise delete the Decoy test bullet |
| `<from validation.json: magnitude.n>` | `validation.json` → `magnitude.n` | only if `magnitude` is not null and the magnitude kill switch did not fire; otherwise delete the sentence |
| `<from validation.json: magnitude.looMae>` | `validation.json` → `magnitude.looMae`, rounded to two decimals, never with "±" | same as above |
| `<from meta.json: run.matching.magnitude.calibrationMagType>` | `meta.json` → `run.matching.magnitude.calibrationMagType` | same as above |
| `<from meta.json: run.matching.magnitude.leaveOneEventOut.nullModelMae>` | `meta.json` → `run.matching.magnitude.leaveOneEventOut.nullModelMae`, rounded to two decimals (`FYI-H2-7`) | same as above |
| `<from meta.json: run.matching.magnitude.magnitudes.belowCalibratedRange>` / `<from meta.json: run.matching.magnitude.magnitudes.written>` | `meta.json` → `run.matching.magnitude.magnitudes.belowCalibratedRange` and `.written` | same as above |
| `<from events.json: hero quality.nStations>` | `events.json` → the hero's `quality.nStations` (what the drawer header prints; the evidence file's trace count is capped by the exporter and only a fallback) | always |
| `<deployed URL>` | the production URL (`docs/deploy.md`) | always |
| `<repo URL>` | the GitHub repository | always |
| `<video URL>` | the uploaded video | always |

After filling: run `scripts/check-copy.sh docs/demo/devpost.md`. It will now flag the filled numbers, which is expected for the final pass; read each flagged line and confirm it is one of the placeholders above and nothing else.

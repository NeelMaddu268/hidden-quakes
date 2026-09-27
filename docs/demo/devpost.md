# Devpost submission

## Title

**Hidden Quakes**

## Tagline

The public regional catalog lists `<from meta.json: summary.publicCatalogCount>` quakes under Utah's geothermal frontier on `<from meta.json: run.windowStart, as a date>`. The same public data holds `<from meta.json: summary.candidateCount>` candidate events.

## Inspiration

Enhanced geothermal is expanding around Milford, Utah. Operators and research teams see underground: dense downhole geophones and fiber arrays give them detailed pictures of the microseismicity there. Someone without operator data, a county official asked at a meeting or a reporter checking a claim, gets the public regional catalog, which lists a sparse slice of that activity. Microseismicity is also what the oversight of geothermal operations (traffic-light protocols) is built on. We wanted to know how much of the underground picture the public seismic network is already hearing, and to show it in a way that person can read in five seconds, share by link, and download.

## What it does

**Results at a glance** (`<from meta.json: run.windowLabel>`):

- **`<from meta.json: summary.recoveredCatalogCount>` of `<from meta.json: summary.publicCatalogCount>`** public-catalog events recovered from raw public waveforms.
- **`<from meta.json: summary.candidateCount>` candidate events**, `<from meta.json: summary.strictQualityCount>` at strict quality, `<from meta.json: summary.strictAdditionalCount>` of them not in the public catalog.
- **Zero strict events** in `<from validation.json: nullTest.nShuffles>` timing-scrambled reruns (about `<from validation.json: nullTest.meanChanceEvents>` chance events each, all below strict).
- **`<from confidence.json: label>`: held-out ROC AUC `<from confidence.json: model.heldOut.rocAuc>`**, separating real pick timing from scrambled clocks.

The public regional catalog lists `<from meta.json: summary.publicCatalogCount>` events under Utah's geothermal frontier on `<from meta.json: run.windowStart, as a date>`. We rebuilt the catalog from raw public seismometers with neural phase picking, multi-station association, relocation and quality tiers: `<from meta.json: summary.candidateCount>` candidate events, `<from meta.json: summary.strictQualityCount>` at strict quality, a much denser view of the same volume the public events sit in.

The `<from confidence.json: label>` scores every candidate: a classifier trained on this run's `<from confidence.json: model.trainedOn.positives>` candidates and `<from confidence.json: model.trainedOn.decoys>` scrambled-clock decoys; held-out ROC AUC `<from confidence.json: model.heldOut.rocAuc>` (`<from confidence.json: model.heldOut.rocAucEqualStationCount>` even at equal station count); not a probability that an event is an earthquake.

**Try it in a minute** at `<deployed URL>`:

- **Space** reveals the candidate events: they play into the volume while the RECOVERED counter climbs next to PUBLIC; the legend tells the public regional catalog from candidate events (bright = strict), and the depth ruler runs to the deepest event. Space again keeps only the strict tier bright; once more replays the window in time order.
- **G** (or the Tour button) plays a guided tour with captions; any key, click or scroll stops it.
- **H** opens a strict event the public regional catalog doesn't have: its drawer says "Not in the public regional catalog" and shows the stations that agreed on it.
- **E** opens the evidence behind the hero event, the strict event located with the most stations: `<from events.json: hero quality.nStations>` stations sorted by distance, the neural P and S picks on each, and the arrival times implied by the final location. They agree, and that agreement is what makes a dot an event.
- **P** switches to plan view, with a true-scale depth section.
- **The `<from confidence.json: label>` score** sits in every event's drawer, next to its depth.
- **Download candidate catalog** (CSV or GeoJSON) appears once the reveal starts.
- **Share a link to any event:** the address bar follows the open drawer, so the link you copy opens that event's evidence for whoever you send it to.

Everything on screen is a candidate event, tiered. Strict means every quality metric is within the range reached by three-quarters of the public-catalog events we recovered; each bar is set per metric, so only `<from meta.json: run.tiering.matchedSet.meetingEveryBar.A>` of those `<from meta.json: run.tiering.matchedSet.n>` recovered public events clear every bar at once.

## How we built it

Two halves joined by files. A Python pipeline (`hq`) writes an immutable run directory, stage by stage:

- **Ingest (EarthScope FDSN):** station metadata at channel level, including sensor depth for borehole instruments, and continuous waveforms for every station in the box, cached once.
- **Preprocess:** one profile per sensor type with explicit anti-aliasing, because the borehole sensors sample far faster than the surface ones and the picker expects one rate.
- **Pick:** pretrained PhaseNet through SeisBench, weights chosen by an A/B on public-catalog events (`<from meta.json: run.pickerWeights>`). We trained no picker: site labels for picking don't exist.
- **Associate:** PyOcto turns single-station picks into events that agree across stations through a velocity model.
- **Locate:** our own grid locator (coarse-then-fine search over eikonal travel-time tables solved with scikit-fmm) on a published FORGE velocity model from the DOE Geothermal Data Repository (`<from meta.json: run.velocityModel.name>`), with station statics fitted on the recovered public events, per-event uncertainty and a synthetic recovery test on the real station geometry.
- **Match and tier:** one-to-one matching against the public regional catalog (USGS ComCat, UUSS solutions). The recovered public events set the quality bar: each tier threshold is a quantile of that set, stored in the run.
- **Magnitude:** a local magnitude (`ML_cal`) calibrated on the matched public events of one catalog magnitude type, with a leave-one-out error reported next to a null-model error; the stage nulls every magnitude when its gate fails, so no magnitude on screen outlives a bad calibration.
- **`<from confidence.json: label>`:** one small model of our own, a logistic regression on each candidate's station count, timing misfit and pick confidence, trained against decoy events made by scrambling each station's clock, with out-of-fold scores. Each candidate's score, shown in the evidence drawer, says how much its timing looks like a real association rather than a decoy; the tiers stay the quality call.
- **Validate and export:** a null test, an STA/LTA baseline, a Gutenberg–Richter curve whose public side counts only the catalog events of the calibration magnitude type, then a static bundle (`meta.json`, `events.json`, `validation.json`, `confidence.json`, evidence snippets) that the web app reads through a provider. Nothing on the demo path depends on a live service.

The web app is Next.js with React Three Fiber: terrain baked from public elevation tiles, events as instanced points with error halos, borehole sensors drawn at their true depth, reference features from GDR well surveys and UGS layers with their sources cited, a plan view with a true-scale depth section, an evidence drawer with the record section behind every dot, a time scrubber that replays the window over a histogram of it, a captioned guided tour, share links that open any event's drawer, and a catalog download built in the browser. The shell renders no digit of its own: a test parses every text node and fails on a number. Every run records its full config; the Run details panel prints it verbatim.

Four humans, one lane each (signal, seismology, visualization, platform), each running their own coding agents against a shared plan with frozen contracts and one owner per path.

## Validation

Every number below is read from the exported run, not typed by hand; the Validation card on the live page shows the same values.

- Public-catalog recall: `<from meta.json: summary.recoveredCatalogCount>` of `<from meta.json: summary.publicCatalogCount>`; every miss is listed in the run.
- Candidate events: `<from meta.json: summary.candidateCount>`, of which `<from meta.json: summary.additionalCount>` are not in the public catalog; `<from meta.json: summary.strictAdditionalCount>` of those pass the strict tier.
- Median stations per event `<from meta.json: summary.medianStations>`; median travel-time residual `<from meta.json: summary.medianRmsS>` s.
- Depth resolution of this station geometry, from the synthetic test, where every event is recorded on every station: about ±`<from validation.json: synthetic.medianVErrM>` m. A typical candidate, recorded on fewer stations, resolves less finely.
- Null test: with each station's timing scrambled and the same config, association yields about `<from validation.json: nullTest.meanChanceEvents>` chance events, versus `<from meta.json: summary.candidateCount>` with real timing, and none of the chance events reached the strict tier in any of the `<from validation.json: nullTest.nShuffles>` scrambles.
- `<from confidence.json: label>` (held-out ROC AUC): `<from confidence.json: model.heldOut.rocAuc>`, and `<from confidence.json: model.heldOut.rocAucEqualStationCount>` even at equal station count, telling real candidates from scrambled-clock decoys; not a probability that an event is an earthquake.
- Baseline: at the strict tier, PhaseNet produced `<from validation.json: baseline[method=phasenet, associationProfile=full].tiers.A>` candidate events and an energy-ratio picker (STA/LTA) produced `<from validation.json: baseline[method=stalta, associationProfile=full].tiers.A>`, through the same downstream code and the same tier bars. That PhaseNet count is a rerun that applies one statics table to every event; the strict count in the headline comes from the published run, which leaves each matched event out of its own statics, so the two differ slightly.
- Magnitudes: a local magnitude, `<from meta.json: run.matching.magnitude.calibrationMagType>`-calibrated `ML_cal`: typical error `<from validation.json: magnitude.looMae>` magnitude units (leave-one-out) versus `<from meta.json: run.matching.magnitude.leaveOneEventOut.nullModelMae>` for a no-skill baseline (a model that gives each event the mean magnitude of the others), from `<from validation.json: magnitude.n>` calibration events of that one catalog magnitude type; `<from meta.json: run.matching.magnitude.magnitudes.belowCalibratedRange>` of `<from meta.json: run.matching.magnitude.magnitudes.written>` candidate magnitudes are extrapolated below the smallest calibration event.
- Gutenberg–Richter: the public curve counts only the catalog events of the calibration magnitude type; the recovered curve is `ML_cal`. We claim no lower completeness magnitude: the public side has too few events of that type for a completeness estimate, and the recovered curve's low end is extrapolated and censored, so we describe the curve and quote no b-value from it.

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
- Live mode: the same pipeline on a rolling window of the latest public data, with a snapshot fallback. The worker in the repository is a start; it is not part of this demo.
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

For the team, not for the paste: everything above this section is final copy once rendered. Story and language follow `docs/demo/pitch-and-qa.md`; claims follow `docs/00-project.md`. Every number is a placeholder of the form `<from FILE: field>`; `make story` fills them from the exported run of record (`apps/web/public/data/showcase/`) into `data/story/devpost-filled.md`, and nothing numeric is typed by hand. `meta.json`, `validation.json` and `confidence.json` are in that folder; the evidence file is `evidence/<heroEventId>.json` in the same folder, with `heroEventId` in `meta.json` → `scene.heroEventId`.

The copy is written for the run of record as exported: every condition below holds for it. If a re-export breaks one, the renderer marks that placeholder as a condition not met; rewrite or delete the sentence (a kill switch fired, `docs/03-schedule.md`), never estimate.

Thumbnail (uploaded separately, not pasted): the STRICT frame from the video (`video-shot-list.md`), with the counter reading `<from meta.json: summary.publicCatalogCount> PUBLIC → <from meta.json: summary.candidateCount> RECOVERED`. It has to read with no video.

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
| Gain sentence (not in the copy) | `meta.json` → `summary.baseline.gain` | `summary.baseline` is null in the run of record (VAL-01 writes it only when the gain holds in both association profiles), so the Validation section quotes the two strict counts below; if a re-export fills it, the approved gain sentence ("at comparable quality, neural picking yields N× the strict events of STA/LTA") goes in their place |
| `<from validation.json: baseline[method=phasenet, associationProfile=full].tiers.A>` / `<from validation.json: baseline[method=stalta, associationProfile=full].tiers.A>` | `validation.json` → `baseline[]`: the `tiers.A` of the row with `method: "phasenet"` and of the row with `method: "stalta"`, both with `associationProfile: "full"` (the Validation card's "Strict events, PhaseNet vs STA/LTA" row) | only if both rows exist and `summary.baseline` is null; otherwise delete the baseline bullet |
| `<from meta.json: run.windowLabel>` | `meta.json` → `run.windowLabel` | always |
| `<from meta.json: run.windowStart, as a date>` | `meta.json` → `run.windowStart`, as its UTC day in words | only while the window is inside one UTC day |
| `<from meta.json: run.tiering.matchedSet.meetingEveryBar.A>` / `<from meta.json: run.tiering.matchedSet.n>` | `meta.json` → `run.tiering.matchedSet.meetingEveryBar.A` over `run.tiering.matchedSet.n` (recovered public events meeting every strict bar at once) | always |
| `<from meta.json: run.pickerWeights>` | `meta.json` → `run.pickerWeights` | always |
| `<from meta.json: run.velocityModel.name>` | `meta.json` → `run.velocityModel.name` | always (H2's dict; if the key is named differently, take the model name from `run.velocityModel`) |
| `<from validation.json: synthetic.medianVErrM>` | `validation.json` → `synthetic.medianVErrM` (synthetic test, every event on every station; keep the "fewer stations" sentence with it) | always |
| `<from validation.json: nullTest.meanChanceEvents>` | `validation.json` → `nullTest.meanChanceEvents` | only if `nullTest` is not null; otherwise delete the sentence |
| `<from validation.json: nullTest.nShuffles>` | `validation.json` → `nullTest.nShuffles` | only while `nullTest.meanChanceStrict` is exactly zero; otherwise delete the "none reached the strict tier" clause |
| `<from confidence.json: label>`, `<from confidence.json: model.trainedOn.positives>`, `<from confidence.json: model.trainedOn.decoys>`, `<from confidence.json: model.heldOut.rocAuc>`, `<from confidence.json: model.heldOut.rocAucEqualStationCount>` | `confidence.json` (`hq.confidence/1`, ML-01, H2) → `label`, `model.trainedOn.positives`, `model.trainedOn.decoys`, `model.heldOut.rocAuc`, `model.heldOut.rocAucEqualStationCount` (two decimals) | only if `confidence.json` is in the bundle; otherwise delete every scramble-test sentence (the glance line, the paragraph and the drawer-score bullet in What it does, the How-we-built bullet, the Validation line) |
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

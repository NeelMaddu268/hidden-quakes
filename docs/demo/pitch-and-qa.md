# Pitch, judges, Devpost, README

Every `{value}` is read off the screen at demo time, never memorized and never typed into copy. The reveal lands before any explanation of how.

## 10-second pitch

"The public catalog shows {publicCatalogCount} earthquakes under Utah's geothermal frontier on {windowLabel}. We rebuilt it from raw public seismometers with neural phase picking and found {candidateCount} candidate events, {strictQualityCount} at strict quality, underground where the public view is nearly empty."

## 30-second pitch

"Enhanced geothermal is expanding around Milford, Utah. Operators have dense downhole monitoring. The public gets the regional catalog: {publicCatalogCount} events here on {windowLabel}. *(REVEAL)* Hidden Quakes pulls raw waveforms from the public seismic network, picks P and S arrivals with a neural network, associates them across stations, relocates them underground and scores every one. We recover {recoveredCatalogCount} of the {publicCatalogCount} public events, plus {additionalCount} more candidates, {strictAdditionalCount} of them strict. Click any dot and you see the waveforms that put it there."

## 2-minute pitch

| Time | Screen | Say |
| --- | --- | --- |
| 0:00–0:10 | Surface view, PUBLIC {N} | "This is Utah's geothermal frontier near Milford. The public regional catalog recorded {publicCatalogCount} earthquakes here on {windowLabel}." |
| 0:10–0:25 | Press REVEAL | "That's what made it into the public catalog. We went straight to the raw public seismometers." *(Let the counter finish. Say nothing for two seconds.)* |
| 0:25–0:45 | Press S, orbit slowly | Depth gate passed: "The strict events concentrate {depthBand} below the surface. That's structure the public view doesn't show." Depth gate failed: "In map view, activity concentrates here, and each halo is that event's own location error." |
| 0:45–1:05 | Press E | "How do we know this dot is an earthquake? {nStations} stations. The neural picker marks P and S on each, and the lines are where physics says they should land from that location. They agree." |
| 1:05–1:25 | Validation panel | "We recover {recoveredCatalogCount} of {publicCatalogCount} public events. Strict means every quality metric is within the range reached by three-quarters of those. This station geometry resolves depth to about ±{medianVErrM} m." Add only if measured: the baseline gain and the null test. |
| 1:25–1:40 | LIVE pill, if it works | "It runs on rolling windows too. This is the last two hours, updated {n} minutes ago." If Live isn't working, spend this time in the drawer. |
| 1:40–2:00 | Full scene | "Operators can see underground. Regulators, journalists and communities mostly can't. This is an open, public-data-only window into what the public network is already hearing." Stop. |

## Hostile judge questions

Answer in one or two sentences, then show something on screen. If the honest answer is "we don't know," say it first.

| # | Question | Answer |
| --- | --- | --- |
| 1 | Isn't PhaseNet doing everything? | PhaseNet picks arrivals one station at a time. An event exists only when picks agree across stations through a velocity model, so association, relocation, statics, uncertainty, tiers and matching are ours, and they're most of the work. |
| 2 | Isn't this QuakeFlow? | Same family of methods, and we say so. QuakeFlow is a research workflow; we built a public-data product with reproducible runs, tiers anchored to the public catalog, evidence behind every dot, and a UI a non-seismologist can read. |
| 3 | Doesn't FORGE already have better sensors? | Yes. Downhole geophones and fiber give far denser catalogs than ours. Those belong to operators and research teams and usually arrive later. We show what the public stream supports, today. |
| 4 | Why compare to a public catalog? | It's the reference the public actually has, and our only independent label set: recovering it is our recall check, and its events set our quality bar. |
| 5 | Are the additional events real? | We call them candidate events. Each needs consistent picks across multiple stations; strict ones meet, on every quality metric, a bar that three-quarters of the recovered public-catalog events meet. We don't claim all {candidateCount} are real. If asked how many recovered public events pass every bar at once, say {strictJointShare}, never three-quarters. |
| 6 | How many do you actually trust? | *(Press S.)* {strictQualityCount}. |
| 7 | Why are your depths broader than published FORGE clouds? | Those use sensors much closer to the source, plus relative relocation. We use sparser public stations and absolute locations. Every halo is that event's own error, and we don't claim fracture geometry at this resolution. |
| 8 | Why care about tiny earthquakes? | Microseismicity is how an engineered reservoir becomes visible, and induced-seismicity oversight (traffic-light protocols) is built on it. Small events are the earliest public signal of how injection changes the subsurface. |
| 9 | What did you build? | Ingestion, preprocessing per sensor type, association config, our own locator with statics and uncertainty, tiering, catalog matching, the exporter, and the 3D product. |
| 10 | Did you train a neural network? | No. We used pretrained PhaseNet weights and chose between them by A/B on public-catalog events. Training needs site labels we don't have, so the time went into location quality. |
| 11 | How is this different from STA/LTA? | STA/LTA triggers on energy ratios; PhaseNet separates P from S and works at lower signal-to-noise. At comparable quality our baseline table shows {gain}× the strict events (only if measured). |
| 12 | Why 3D? | The question is where underground, and depth is the dimension the public catalog is weakest in. Plan view is one key away. |
| 13 | Isn't this established seismology? | The methods are established; we're not claiming a new algorithm. We built an open, near-real-time product over public data: ingestion, association, local relocation, quality scoring, catalog comparison and an inspectable interface. High-resolution geothermal catalogs usually depend on dedicated arrays; we test how much structure the public stream alone can surface. |
| 14 | What's your false-positive rate? | There's no ground truth for events the public catalog lacks, so no true rate. Our proxy is a null test: scramble each station's timing and rerun. That yields about {meanChanceEvents} chance events, versus {candidateCount} with real timing. |
| 15 | What's your external validation? | Public-catalog recall, a synthetic depth-resolution test, the null test, the STA/LTA baseline, and the published FORGE depth band. Ridgecrest only if it got done. |
| 16 | Could these belong to Cape Station instead of FORGE? | Possibly. The operations are close together and we deliberately don't attribute. We show verified facility locations and our locations with error; attribution needs operator data. |
| 17 | Why call this AI? | The detection step is a deep network trained on large labeled seismogram sets, and it's what makes small events visible. The rest is physics. We'd call it ML-assisted monitoring. |
| 18 | What did you build during HackGT? | Everything in the repo; git history starts at the kickoff commit. Pre-event work was research into which data and methods exist. No code, fixtures or outputs came over. |
| 19 | How near-real-time is it? | {latency} minutes for the last two hours on this laptop (only if measured). |
| 20 | Why that day ({windowLabel})? | It's a day with public-catalog activity in the region, which gives us reference events to validate against. |
| 21 | What about the high-rate borehole sensors? | We don't blindly resample. Each sensor type gets its own profile with explicit anti-aliasing, and we A/B'd a time-stretch variant to keep the high-frequency band. |
| 22 | Which velocity model? | A published FORGE 1D model from the Geothermal Data Repository, with a public 3D Cape/FORGE model as the upgrade path. The source is recorded in every run. |
| 23 | Aren't your depths just an artifact of the velocity model? | The synthetic test isolates geometry from the model, station statics absorb model error, and we compared 1D and 3D. How much depths shifted is in Run details. |
| 24 | Why not use the operator's catalog? | It isn't public in real time, and independence is the point. |
| 25 | What if Live shows nothing? | It says "No high-confidence candidate events in this window." We never fake activity. |
| 26 | Couldn't trucks or drilling noise fool it? | Noise at one station won't associate across many stations with consistent moveout. Persistent noise at one pad could, and that's what tiers and the evidence drawer are for. |
| 27 | How good are the magnitudes? | Calibrated on {n} matched public events, leave-one-out error ±{looMae}. If that had been bad, you wouldn't see magnitudes. |
| 28 | Does your catalog lower the completeness magnitude? | *(G-R panel, if built.)* The public curve stops at its Mc; ours continues below it. |
| 29 | Who would use this? | Operators have specialized monitoring; the public mostly doesn't. This is the public-data layer: an independently inspectable view for researchers, regulators, journalists and communities. |
| 30 | Does it scale to other sites? | It's config-driven: area, stations and velocity model. The real per-site cost is a velocity model and station QC. We haven't run it anywhere else. |
| 31 | Why PyOcto and not GaMMA? | PyOcto is fast on dense pick sets. Either would work; the associator isn't our contribution. |
| 32 | What's the biggest weakness? | Depth, because public station geometry is sparse. We'd rather say it before you do. |
| 33 | Why trust your tier thresholds? | They come from the public events we recovered, and they're stored in the run config. Open Run details. |
| 34 | Did you cherry-pick the hero event? | It's the strict event with the most stations. Click any other dot. |

## Devpost outline

1. **Title + tagline:** "Hidden Quakes: what the public can't see beneath Utah's geothermal frontier."
2. **Thumbnail:** semi-transparent terrain, the geothermal reference, public points in white, strict events in amber forming structure below, and a "{publicCatalogCount} PUBLIC → {candidateCount} RECOVERED" counter. It has to read with no video; if it looks like random dots, redesign.
3. **Inspiration:** enhanced geothermal is expanding; operators see underground and the public doesn't.
4. **What it does:** the 10-second pitch, then the three interactions (reveal, evidence, time).
5. **How we built it:** pipeline diagram, stack, data sources.
6. **Validation:** recall, strict count, depth resolution, plus baseline and null test if measured. Numbers copied from the validation panel.
7. **Limits (its own heading):** candidate events, sparse public geometry, no attribution, pretrained picker.
8. **Challenges:** borehole sample rates, depth credibility, datum handling.
9. **What's next:** other public-network sites, relative relocation, continuous live operation.
10. **Built with:** Python, ObsPy, SeisBench, PyOcto, scikit-fmm, FastAPI, Next.js, React Three Fiber.
11. **Links:** deployed URL, repo, 2-minute video.
12. **HackGT compliance line:** what was pre-event research vs what was built during the event.

## README outline

1. One-paragraph description plus the reveal screenshot.
2. Quickstart: `pnpm i && pnpm dev` for the web app; `uv sync && uv run hq run configs/showcase` for the pipeline.
3. Demo modes and how to switch them (`?mode=`).
4. Architecture diagram and repo layout.
5. Data sources and licenses: EarthScope waveforms, the public regional catalog, GDR velocity models (CC BY 4.0), DEM source.
6. Reproducing the showcase: the final `runId`, its config, and expected counts.
7. Scientific validity: what we claim, what we don't, and the validation results.
8. Team, roles, and the HackGT compliance statement.

## Numbers to fill Saturday evening

Every spoken `{value}` above is read off the screen at demo time. This table maps each one to the bundle field it renders from (`apps/web/public/data/showcase/`), so the Devpost and README fill-in is one lookup and the pitch never says a number the screen can't back. A field that is null in the bundle means the kill switch fired (`docs/03`): drop the sentence.

| Spoken | Bundle field | Where it shows on screen |
| --- | --- | --- |
| `{publicCatalogCount}` | `meta.json` → `summary.publicCatalogCount` | PUBLIC counter; Validation card (recall row) |
| `{candidateCount}` | `meta.json` → `summary.candidateCount` | RECOVERED counter after the reveal |
| `{strictQualityCount}` | `meta.json` → `summary.strictQualityCount` | STRICT counter; Validation card |
| `{recoveredCatalogCount}` | `meta.json` → `summary.recoveredCatalogCount` | Validation card (recall row) |
| `{additionalCount}` | `meta.json` → `summary.additionalCount` | Not on screen; read from `meta.json` |
| `{strictAdditionalCount}` | `meta.json` → `summary.strictAdditionalCount` | Not on screen; read from `meta.json` |
| `{nStations}` (hero event) | `evidence/<scene.heroEventId>.json` → `traces.length`; `events.json` → hero's `quality.nStations` | Evidence drawer header |
| `{medianVErrM}` | `validation.json` → `synthetic.medianVErrM` | Validation card (depth resolution row) |
| `{gain}` | `meta.json` → `summary.baseline.gain` (only when present) | Validation card (gain row, shown only when the baseline ran and the gain holds) |
| `{meanChanceEvents}` | `validation.json` → `nullTest.meanChanceEvents` (only when present) | Validation card (chance associations row) |
| `{n}` matched events, `{looMae}` (Q27) | `validation.json` → `magnitude.n`, `magnitude.looMae` (only when present) | Not on screen; read from `validation.json` |
| `{strictJointShare}` (Q5) | `run.json` → `tiering.matchedSet.meetingEveryBar.A` over `tiering.matchedSet.n` (the bars are per metric; this is the share of recovered public events that meets all of them) | Not on screen; read from the run's `run.json` |
| `{depthBand}` | No bundle field. Read it off the depth ruler with STRICT on; say what is visible, never a mechanism | Scene, depth ruler and slices |
| `{n}` minutes ago (Live) | `/api/live/status` → `updatedAt` | Live mode label |
| `{latency}` (Q19) | `/api/live/status` → `latencyS`, and `/health` → `served.latencyS` | Not on screen; read from the API |
| The window date spoken in the pitches | `meta.json` → `run.windowLabel` | Showcase mode label |
| Tier thresholds and their quantiles (Q33) | `meta.json` → `run.tiering` | Run details (D) |
| Velocity model source (Q22) | `meta.json` → `run.velocityModel` | Run details (D) |
| Depth shift between velocity models (Q23) | Only if H2 records it in `run.locator` or `run.velocityModel`; otherwise say "in H2's diagnostics", not "in Run details" | Run details (D), if recorded |
| G-R curve (Q28) | `validation.json` → `gr` (only when present) | No G-R panel exists in the shell yet; say "in the bundle" unless one lands |

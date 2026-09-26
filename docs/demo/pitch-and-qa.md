# Pitch, judges, Devpost, README

Every `{value}` is read off the screen at demo time, never memorized and never typed into copy. The reveal lands before any explanation of how.

## 10-second pitch

"The public regional catalog lists {publicCatalogCount} events under Utah's geothermal frontier on {windowLabel}. We rebuilt the catalog from raw public seismometers with neural phase picking, multi-station association and relocation: {candidateCount} candidate events, {strictQualityCount} at strict quality, underground where the public view is nearly empty."

## 30-second pitch

"Enhanced geothermal is expanding around Milford, Utah. Operators have dense downhole monitoring. The public gets the public regional catalog: {publicCatalogCount} events here on {windowLabel}. *(REVEAL)* Hidden Quakes pulls raw waveforms from the public seismic network, picks P and S arrivals with a neural network, associates them across stations, relocates them underground and scores every one. We recover {recoveredCatalogCount} of the {publicCatalogCount} public events, plus {additionalCount} more candidates, {strictAdditionalCount} of them strict. Click any dot and you see the waveforms that put it there."

## 2-minute pitch

| Time | Screen | Say |
| --- | --- | --- |
| 0:00–0:10 | Start frame: terrain, the geothermal reference, PUBLIC {N}, the REVEAL button | "This is Utah's geothermal frontier near Milford. The public regional catalog lists {publicCatalogCount} events here on {windowLabel}." |
| 0:10–0:25 | Space (or REVEAL HIDDEN SIGNAL): the counter climbs, the camera dollies to the side view | "That's what the public regional catalog shows. We went straight to the raw public seismometers." *(Let the counter finish. Say nothing for two seconds.)* |
| 0:25–0:40 | Space again (STRICT), orbit slowly | "These are the strict ones: every quality metric within the range reached by three-quarters of the public events we recovered. They sit {depthBand} below the surface; read it off the ruler." Say only what is on screen: the depth band, and whether the points bunch or spread. Never what the pattern means, never a geometry, never a mechanism. |
| 0:40–1:00 | E: the evidence drawer on the hero event | "How do we know this dot is an event and not noise? {nStations} stations agreed. The neural picker marks P and S on each, and the lines are the arrivals the final location implies. They agree." |
| 1:00–1:15 | Validation card (bottom-left since the reveal); D opens Run details if asked | "We recover {recoveredCatalogCount} of {publicCatalogCount} public events. This station geometry resolves depth to about ±{medianVErrM} m." Read any other row exactly as the card shows it: if the "Strict events, PhaseNet vs STA/LTA" row is there, say "{strictPhasenet} vs {strictStalta} at the strict tier" and stop; a gain only if the gain row is on the card; the null test only if the chance-associations row is. |
| 1:15–1:30 | Esc, then Space (TIME): the scrubber replays the window | "Same events, replayed over the day." Say what the histogram shows: bursts or a steady trickle, and when. Nothing about why. |
| 1:30–1:40 | Optional beat: the LIVE pill, only if it is up (docs/03 kill switch: unstable, or latency over about ten minutes, and the pill is cut) | "It runs on rolling windows too: the last two hours, updated {n} minutes ago." Without LIVE, spend this time in the drawer or the scrubber. |
| 1:40–2:00 | Full scene, orbit | "Operators can see underground. Regulators, journalists and communities mostly can't. This is an open, public-data-only window into what the public network is already hearing." Stop. |

## Hostile judge questions

Answer in one or two sentences, then show something on screen. If the honest answer is "we don't know," say it first.

| # | Question | Answer |
| --- | --- | --- |
| 1 | Isn't PhaseNet doing everything? | PhaseNet picks arrivals one station at a time. An event exists only when picks agree across stations through a velocity model, so association, relocation, statics, uncertainty, tiers and matching are ours, and they're most of the work. |
| 2 | Isn't this QuakeFlow? | Same family of methods, and we say so. QuakeFlow is a research workflow; we built a public-data product with reproducible runs, tiers anchored to the public catalog, evidence behind every dot, and a UI a non-seismologist can read. |
| 3 | Doesn't FORGE already have better sensors? | Yes. Downhole geophones and fiber give far denser catalogs than ours. Those belong to operators and research teams and usually arrive later. We show what the public stream supports, today. |
| 4 | Why compare to a public catalog? | It's the reference the public actually has, and our only independent label set: recovering it is our recall check, and its events set our quality bar. |
| 5 | Are the additional events real? | We call them candidate events. Each needs consistent picks across multiple stations; strict means every quality metric is within the range reached by three-quarters of the public events we recovered. We don't claim all {candidateCount} are real. If asked how many recovered public events pass every bar at once, say {strictJointShare}, never three-quarters. |
| 6 | How many do you actually trust? | *(Press S.)* {strictQualityCount}. |
| 7 | Why are your depths broader than published FORGE clouds? | Those use sensors much closer to the source, plus relative relocation. We use sparser public stations and absolute locations. Every halo is that event's own error. We show depths as located and read no geometry into the cloud; that would need relative relocation, which we didn't run. |
| 8 | Why care about tiny events? | Microseismicity is the public's only signal at this scale, and the oversight of geothermal operations (traffic-light protocols) is built on it. We show where and when candidate events occur; we don't say why. |
| 9 | What did you build? | Ingestion, preprocessing per sensor type, association config, our own locator with statics and uncertainty, tiering, catalog matching, the exporter, and the 3D product. |
| 10 | Did you train a neural network? | No. We used pretrained PhaseNet weights and chose between them by A/B on public-catalog events. Training needs site labels we don't have, so the time went into location quality. |
| 11 | How is this different from STA/LTA? | STA/LTA triggers on energy ratios; PhaseNet separates P from S and works at lower signal-to-noise. What that buys is measured, not assumed, and H1 is rescoring STA/LTA on the run's statics scale (`docs/requests/H4.md`, `REQ-H1-5`), so read the answer off the Validation card at demo time: a gain ("at comparable quality, {gain}× the strict events") only if the gain row is on the card; otherwise, if the "Strict events, PhaseNet vs STA/LTA" row is there, "at the strict tier, STA/LTA produced {strictStalta} candidate events while PhaseNet produced {strictPhasenet}"; say "at any threshold tested" only when the run's `baseline_reference.json` (H1's sweep) shows Tier A zero at every scored point. If neither row is on the card, the comparison isn't in the run of record: say so. |
| 12 | Why 3D? | The question is where underground, and depth is the dimension the public catalog is weakest in. Plan view is one key away. |
| 13 | Isn't this established seismology? | The methods are established; we're not claiming a new algorithm. We built an open, near-real-time product over public data: ingestion, association, local relocation, quality scoring, catalog comparison and an inspectable interface. High-resolution geothermal catalogs usually depend on dedicated arrays; we test how much of the seismicity, and at what depths, the public stream alone can surface. The Saturday depth call passed on amended criteria (`docs/lanes/H2-seismology.md`, "Depth call"): that licenses showing depths as located, not reading a pattern into them (no relative relocation was done). |
| 14 | What's your false-positive rate? | There's no ground truth for events the public catalog lacks, so no true rate. Our proxy is a null test: scramble each station's timing and rerun. That yields about {meanChanceEvents} chance events, versus {candidateCount} with real timing. |
| 15 | What's your external validation? | Public-catalog recall, a synthetic depth-resolution test, the null test, the STA/LTA baseline, and, for the events we recovered, the same-day public regional catalog's depths (published FORGE catalogs cover other dates and places, so they are context, not a check). Ridgecrest only if it got done. |
| 16 | Could these belong to Cape Station instead of FORGE? | We don't know, and we don't attribute to either. Several operations share the region; we show cited facility locations and our locations with their errors. Attribution needs operator data we don't have. |
| 17 | Why call this AI? | The detection step is a deep network trained on large labeled seismogram sets, and it's what makes small events visible. The rest is physics. We'd call it ML-assisted monitoring. |
| 18 | What did you build during HackGT? | Everything in the repo; git history starts at the kickoff commit. Pre-event work was research into which data and methods exist. No code, fixtures or outputs came over. |
| 19 | How near-real-time is it? | {latency} minutes for the last two hours on this laptop (only if measured). |
| 20 | Why that day ({windowLabel})? | It's a day with public-catalog activity in the region, which gives us reference events to validate against. |
| 21 | What about the high-rate borehole sensors? | We don't blindly resample. Each sensor type gets its own profile with explicit anti-aliasing, and we A/B'd a time-stretch variant to keep the high-frequency band. |
| 22 | Which velocity model? | A published FORGE 1D model from the Geothermal Data Repository, with a public 3D Cape/FORGE model as the upgrade path. The source is recorded in every run. |
| 23 | Aren't your depths just an artifact of the velocity model? | The synthetic test isolates geometry from the model, station statics absorb model error, and we compared 1D and 3D. How much depths shifted is in Run details. |
| 24 | Why not use the catalogs from the private downhole arrays? | They aren't public in real time, and independence is the point. |
| 25 | What if Live shows nothing? | It says "No high-confidence candidate events in this window." We never fake activity. |
| 26 | Couldn't trucks or drilling noise fool it? | Noise at one station won't associate across many stations with consistent moveout. Persistent noise at one pad could, and that's what tiers and the evidence drawer are for. |
| 27 | How good are the magnitudes? | They're a local magnitude, {magType}, calibrated on {n} matched public events of one catalog magnitude type; leave-one-out error ±{looMae}, read next to the null-model error ±{nullModelMae} (the error of a model that gives each event the mean magnitude of the others). {belowCalibratedRange} of the candidate magnitudes lie below the calibrated range, so they're extrapolated and, near the detection limit, biased upward. If the gate had failed there would be no magnitudes on screen. |
| 28 | Does your catalog lower the completeness magnitude? | *(No G-R panel is in the shell; say "in the bundle".)* The public curve counts only the catalog events of the calibration magnitude type ({magType}); ours is the candidates' {magType}. Ours continues below the public curve's Mc, with the caveat that its low end is extrapolated and censored, so we describe the curve and quote no b-value from it. |
| 29 | Who would use this? | Operators have specialized monitoring; the public mostly doesn't. This is the public-data layer: an independently inspectable view for researchers, regulators, journalists and communities. |
| 30 | Does it scale to other sites? | It's config-driven: area, stations and velocity model. The real per-site cost is a velocity model and station QC. We haven't run it anywhere else. |
| 31 | Why PyOcto and not GaMMA? | PyOcto is fast on dense pick sets. Either would work; the associator isn't our contribution. |
| 32 | What's the biggest weakness? | Depth, because public station geometry is sparse. We'd rather say it before you do. |
| 33 | Why trust your tier thresholds? | They come from the public events we recovered, and they're stored in the run config. Open Run details. |
| 34 | Did you cherry-pick the hero event? | It's the strict event with the most stations. Click any other dot. |

## Devpost outline

1. **Title + tagline:** "Hidden Quakes: what the public can't see beneath Utah's geothermal frontier."
2. **Thumbnail:** semi-transparent terrain, the geothermal reference, public points in white, strict candidate events in amber below, and a "{publicCatalogCount} PUBLIC → {candidateCount} RECOVERED" counter. It has to read with no video; if it looks like random dots, redesign.
3. **Inspiration:** enhanced geothermal is expanding; operators see underground and the public doesn't.
4. **What it does:** the 10-second pitch, then the three interactions (reveal, evidence, time).
5. **How we built it:** pipeline diagram, stack, data sources.
6. **Validation:** recall, strict count, depth resolution, plus baseline and null test if measured. Numbers copied from the validation panel.
7. **Limits (its own heading):** candidate events, sparse public geometry, no attribution, pretrained picker.
8. **Challenges:** borehole sample rates, depth credibility, datum handling.
9. **What's next:** other public-network sites, relative relocation, continuous live operation.
10. **Built with:** Python (ObsPy, SeisBench, PyOcto, scikit-fmm, SciPy, Pydantic, FastAPI), TypeScript (Next.js, React, three.js, React Three Fiber, zustand); the full list is in `docs/demo/devpost.md`.
11. **Links:** deployed URL, repo, 2-minute video.
12. **HackGT compliance line:** what was pre-event research vs what was built during the event.

## README outline

1. One-paragraph description plus the reveal screenshot.
2. Quickstart: `pnpm i && pnpm dev` for the web app; `uv sync && uv run hq run configs/showcase` for the pipeline.
3. Demo modes and how to switch them (`?mode=`).
4. Architecture diagram and repo layout.
5. Data sources and licenses: EarthScope waveforms, the public regional catalog, GDR velocity models (CC BY 4.0), DEM source.
6. Reproducing the showcase: the final `runId` and its config; counts are read from the bundle, never typed in.
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
| `{strictPhasenet}` vs `{strictStalta}` (Q11: "at the strict tier, STA/LTA produced {strictStalta} candidate events while PhaseNet produced {strictPhasenet}") | `validation.json` → `baseline[]` rows with `associationProfile: "full"`: `tiers.A` of `method: "phasenet"` and of `method: "stalta"` (only when both rows exist; never a hard-coded count) | Validation card ("Strict events, PhaseNet vs STA/LTA" row, shown whenever both `full` rows exist, even when the gain row is hidden) |
| `{meanChanceEvents}` | `validation.json` → `nullTest.meanChanceEvents` (only when present) | Validation card (chance associations row) |
| `{n}` matched events, `{looMae}` (Q27) | `validation.json` → `magnitude.n`, `magnitude.looMae` (only when present and the magnitude kill switch did not fire) | Not on screen; read from `validation.json` |
| `{magType}` (Q27, Q28) | `meta.json` → `run.matching.magnitude.calibrationMagType` (the one catalog magnitude type calibrated on); the candidates' own type is on every event's `magnitude.type` and in the drawer header | Evidence drawer header ("M x.x <type>") |
| `{nullModelMae}` (Q27) | `meta.json` → `run.matching.magnitude.leaveOneEventOut.nullModelMae` (`FYI-H2-7`: quote `looMae` only next to it) | Not on screen; read from `meta.json` |
| `{belowCalibratedRange}` (Q27) | `meta.json` → `run.matching.magnitude.magnitudes.belowCalibratedRange` (with `magnitudes.written` as the denominator) | Not on screen; read from `meta.json` |
| `{strictJointShare}` (Q5) | `run.json` → `tiering.matchedSet.meetingEveryBar.A` over `tiering.matchedSet.n` (the bars are per metric; this is the share of recovered public events that meets all of them) | Not on screen; read from the run's `run.json` |
| `{depthBand}` | No bundle field. Read it off the depth ruler with STRICT on; say what is visible (a depth band, bunched or spread), never a geometry and never a mechanism | Scene, depth ruler and slices |
| `{n}` minutes ago (Live) | `/api/live/status` → `updatedAt` | Live mode label |
| `{latency}` (Q19) | `/api/live/status` → `latencyS`, and `/health` → `served.latencyS` | Not on screen; read from the API |
| The window date spoken in the pitches | `meta.json` → `run.windowLabel` | Showcase mode label |
| Tier thresholds and their quantiles (Q33) | `meta.json` → `run.tiering` | Run details (D) |
| Velocity model source (Q22) | `meta.json` → `run.velocityModel` | Run details (D) |
| Depth shift between velocity models (Q23) | Only if H2 records it in `run.locator` or `run.velocityModel`; otherwise say "in H2's diagnostics", not "in Run details" | Run details (D), if recorded |
| G-R curve (Q28) | `validation.json` → `gr` (only when present); its `publicCum` counts the public regional catalog's magnitudes of `{magType}` only (`REQ-H2-13`) | No G-R panel exists in the shell; say "in the bundle" |

# Language audit (DEMO-03)

Text pass over `README.md`, `docs/demo/pitch-and-qa.md` and `docs/demo/devpost.md` after the Saturday depth call (PASS, amended: `docs/lanes/H2-seismology.md`, "Depth call"), the Strict wording fix (`docs/requests/H4.md`, `REQ-H2-11`), the G-R magnitude-type fix (`REQ-H2-13`), the magnitude caveats (`FYI-H2-7`) and H1's pending STA/LTA rescore (`REQ-H1-5`). Rules enforced, from the honesty rules in `CLAUDE.md` and `docs/00-project.md`:

1. The 3D hero with depths stays; no claim about what the pattern of depths means (no geometry, no mechanism), not even a conditional one. Depths, clustering and a band read off the ruler are what the data shows.
2. Strict is "every quality metric within the range reached by three-quarters of the public events we recovered", never a stronger reading.
3. "Candidate events" and "public regional catalog"; no attribution to any operation; nothing detected is called verified; no forecast wording; no false-positive rate.
4. Every baseline sentence is conditional on a bundle field (`summary.baseline`, or both `full` rows of `validation.baseline`); none asserts an outcome, because H1 is rescoring STA/LTA on the run's statics scale.
5. Every number is a placeholder that names its bundle field; the README quotes none.
6. Magnitudes only with their type (`ML_cal`, calibrated on one catalog magnitude type), `looMae` next to the null-model error, and the below-calibrated-range caveat; the public G-R curve counts the calibration magnitude type only.
7. Live mode is an optional beat, cut with the LIVE pill (`docs/03-schedule.md`, kill switches).

Each change below is the edited line as it now stands (`file:line`), with the old and new text in a code block so this file passes `scripts/check-copy.sh` unchanged (fenced blocks are exempt from its digit and phrase rules; the quoted old text is the reason each line changed).

## `README.md`

### `README.md:19`

```text
BEFORE
Keyboard map: Space (next beat), R (reset), S (strict), T (time), E (evidence), P (plan/oblique), D (run details), Esc (close).

AFTER
Keyboard map: Space (next beat: reveal, then strict, then time), R (reset), S (strict / all), T (time), E (evidence on the hero event), P (plan / oblique), D (run details), Esc (close).
```

### `README.md:88`

```text
BEFORE
| A | STRICT | Every quality metric within the range reached by three-quarters of the recovered public events (each bar is set per metric;

AFTER
| A | STRICT | Every quality metric within the range reached by three-quarters of the public events we recovered (each bar is set per metric;
```

### `README.md:101`

```text
BEFORE
| Depth: too many strict events pinned to the grid top, or a nonphysical vertical distribution → no structure claims; plan view becomes the hero |

AFTER
| Depth: too many strict events pinned to the grid top, or a nonphysical vertical distribution → plan view becomes the hero and the copy describes no pattern in the depths. The Saturday depth call passed on amended criteria (`docs/lanes/H2-seismology.md`, "Depth call"), so the 3D hero with depths stays; that licenses showing depths as located, not reading anything into their pattern (that would need relative relocation, which did not run) |
```

### `README.md:103`

```text
BEFORE
| "At comparable quality, neural picking yields G× more strict events" | STA/LTA within a comparable fraction of PhaseNet's strict count → drop the claim, keep the table |

AFTER
| "At comparable quality, neural picking yields G× more strict events", only while `summary.baseline` exists; without it the card shows the two strict counts side by side from the `full` rows and asserts no gain. H1 is rescoring STA/LTA on the run's statics scale (`docs/requests/H4.md`, `REQ-H1-5`), so no baseline outcome is written anywhere in this repository | STA/LTA within a comparable fraction of PhaseNet's strict count → drop the claim, keep the table |
```

### `README.md:104`

```text
BEFORE
| Gutenberg–Richter | Aki–Utsu b with Shi–Bolt sigma, Mc by maximum curvature plus an offset, public curve versus recovered curve | `validation.gr`, `validation.magnitude` (`n`, `looMae`) | "Magnitudes extend the trend below the public catalog's completeness" | Leave-one-out MAE above the configured cap → no magnitude sizing, no G-R |

AFTER
| Gutenberg–Richter | Aki–Utsu b with Shi–Bolt sigma, Mc by maximum curvature plus an offset; the public curve counts only the catalog events of the calibration magnitude type (`run.matching.magnitude.calibrationMagType`), the recovered curve is the candidates' `ML_cal` | `validation.gr`, `validation.magnitude` (`n`, `looMae`), `run.matching.magnitude` (`leaveOneEventOut.nullModelMae`, `magnitudes.belowCalibratedRange`) | "Magnitudes extend the trend below the public catalog's completeness", always with the magnitude type named, `looMae` read next to the null-model error, and the note that most candidate magnitudes lie below the calibrated range (extrapolated; near the detection limit, biased upward) | Leave-one-out MAE above the configured cap → no magnitude sizing, no G-R |
```

### `README.md:107`

```text
BEFORE
What we never claim: that operators lack better monitoring, that any event was missed by anyone, fracture geometry (at most "structure", and only if the depth gate passes), or a mechanism.

AFTER
What we never claim: that operators lack better monitoring, that any event was missed by anyone, what the pattern of candidate events means underground (we show depths and clustering as located and describe only what is on screen), or a mechanism.
```

## `docs/demo/pitch-and-qa.md`

### `docs/demo/pitch-and-qa.md:7`

```text
BEFORE
"The public catalog shows {publicCatalogCount} earthquakes under Utah's geothermal frontier on {windowLabel}. We rebuilt it from raw public seismometers with neural phase picking and found {candidateCount} candidate events, {strictQualityCount} at strict quality, underground where the public view is nearly empty."

AFTER
"The public regional catalog lists {publicCatalogCount} events under Utah's geothermal frontier on {windowLabel}. We rebuilt the catalog from raw public seismometers with neural phase picking, multi-station association and relocation: {candidateCount} candidate events, {strictQualityCount} at strict quality, underground where the public view is nearly empty."
```

### `docs/demo/pitch-and-qa.md:11`

```text
BEFORE
Operators have dense downhole monitoring. The public gets the regional catalog: {publicCatalogCount} events here on {windowLabel}.

AFTER
Operators have dense downhole monitoring. The public gets the public regional catalog: {publicCatalogCount} events here on {windowLabel}.
```

### `docs/demo/pitch-and-qa.md:17`

```text
BEFORE
| 0:00–0:10 | Surface view, PUBLIC {N} | "This is Utah's geothermal frontier near Milford. The public regional catalog recorded {publicCatalogCount} earthquakes here on {windowLabel}." |
| 0:10–0:25 | Press REVEAL | "That's what made it into the public catalog. We went straight to the raw public seismometers." *(Let the counter finish. Say nothing for two seconds.)* |
| 0:25–0:45 | Press S, orbit slowly | Depth gate passed: "The strict events concentrate {depthBand} below the surface. That's structure the public view doesn't show." Depth gate failed: "In map view, activity concentrates here, and each halo is that event's own location error." |
| 0:45–1:05 | Press E | "How do we know this dot is an earthquake? {nStations} stations. The neural picker marks P and S on each, and the lines are where physics says they should land from that location. They agree." |
| 1:05–1:25 | Validation panel | "We recover {recoveredCatalogCount} of {publicCatalogCount} public events. Strict means every quality metric is within the range reached by three-quarters of those. This station geometry resolves depth to about ±{medianVErrM} m." Add only if measured: the baseline gain and the null test. |
| 1:25–1:40 | LIVE pill, if it works | "It runs on rolling windows too. This is the last two hours, updated {n} minutes ago." If Live isn't working, spend this time in the drawer. |
| 1:40–2:00 | Full scene | "Operators can see underground. Regulators, journalists and communities mostly can't. This is an open, public-data-only window into what the public network is already hearing." Stop. |

AFTER
| 0:00–0:10 | Start frame: terrain, the geothermal reference, PUBLIC {N}, the REVEAL button | "This is Utah's geothermal frontier near Milford. The public regional catalog lists {publicCatalogCount} events here on {windowLabel}." |
| 0:10–0:25 | Space (or REVEAL HIDDEN SIGNAL): the counter climbs, the camera dollies to the side view | "That's what the public regional catalog shows. We went straight to the raw public seismometers." *(Let the counter finish. Say nothing for two seconds.)* |
| 0:25–0:40 | Space again (STRICT), orbit slowly | "These are the strict ones: every quality metric within the range reached by three-quarters of the public events we recovered. They sit {depthBand} below the surface; read it off the ruler." Say only what is on screen: the depth band, and whether the points bunch or spread. Never what the pattern means, never a geometry, never a mechanism. |
| 0:40–1:00 | E: the evidence drawer on the hero event | "How do we know this dot is an event and not noise? {nStations} stations agreed. The neural picker marks P and S on each, and the lines are the arrivals the final location implies. They agree." |
| 1:00–1:15 | Validation card (bottom-left since the reveal); D opens Run details if asked | "We recover {recoveredCatalogCount} of {publicCatalogCount} public events. This station geometry resolves depth to about ±{medianVErrM} m." Read any other row exactly as the card shows it: if the "Strict events, PhaseNet vs STA/LTA" row is there, say "{strictPhasenet} vs {strictStalta} at the strict tier" and stop; a gain only if the gain row is on the card; the null test only if the chance-associations row is. |
| 1:15–1:30 | Esc, then Space (TIME): the scrubber replays the window | "Same events, replayed over the day." Say what the histogram shows: bursts or a steady trickle, and when. Nothing about why. |
| 1:30–1:40 | Optional beat: the LIVE pill, only if it is up (docs/03 kill switch: unstable, or latency over about ten minutes, and the pill is cut) | "It runs on rolling windows too: the last two hours, updated {n} minutes ago." Without LIVE, spend this time in the drawer or the scrubber. |
| 1:40–2:00 | Full scene, orbit | "Operators can see underground. Regulators, journalists and communities mostly can't. This is an open, public-data-only window into what the public network is already hearing." Stop. |
```

### `docs/demo/pitch-and-qa.md:36`

```text
BEFORE
| 5 | Are the additional events real? | We call them candidate events. Each needs consistent picks across multiple stations; strict ones meet, on every quality metric, a bar that three-quarters of the recovered public-catalog events meet. We don't claim all {candidateCount} are real.

AFTER
| 5 | Are the additional events real? | We call them candidate events. Each needs consistent picks across multiple stations; strict means every quality metric is within the range reached by three-quarters of the public events we recovered. We don't claim all {candidateCount} are real.
```

### `docs/demo/pitch-and-qa.md:38`

```text
BEFORE
We use sparser public stations and absolute locations. Every halo is that event's own error, and we don't claim fracture geometry at this resolution. |

AFTER
We use sparser public stations and absolute locations. Every halo is that event's own error. We show depths as located and read no geometry into the cloud; that would need relative relocation, which we didn't run. |
```

### `docs/demo/pitch-and-qa.md:39`

```text
BEFORE
| 8 | Why care about tiny earthquakes? | Microseismicity is how an engineered reservoir becomes visible, and induced-seismicity oversight (traffic-light protocols) is built on it. Small events are the earliest public signal of how injection changes the subsurface. |

AFTER
| 8 | Why care about tiny events? | Microseismicity is the public's only signal at this scale, and the oversight of geothermal operations (traffic-light protocols) is built on it. We show where and when candidate events occur; we don't say why. |
```

### `docs/demo/pitch-and-qa.md:42`

```text
BEFORE
| 11 | How is this different from STA/LTA? | STA/LTA triggers on energy ratios; PhaseNet separates P from S and works at lower signal-to-noise. At comparable quality our baseline table shows {gain}× the strict events (only if measured). If no gain is claimed but the baseline ran: "at the strict tier, STA/LTA produced {strictStalta} candidate events while PhaseNet produced {strictPhasenet}", both read from the Validation card; say "at any threshold tested" only when the run's `baseline_reference.json` (H1's sweep) shows Tier A zero at every scored point. |

AFTER
| 11 | How is this different from STA/LTA? | STA/LTA triggers on energy ratios; PhaseNet separates P from S and works at lower signal-to-noise. What that buys is measured, not assumed, and H1 is rescoring STA/LTA on the run's statics scale (`docs/requests/H4.md`, `REQ-H1-5`), so read the answer off the Validation card at demo time: a gain ("at comparable quality, {gain}× the strict events") only if the gain row is on the card; otherwise, if the "Strict events, PhaseNet vs STA/LTA" row is there, "at the strict tier, STA/LTA produced {strictStalta} candidate events while PhaseNet produced {strictPhasenet}"; say "at any threshold tested" only when the run's `baseline_reference.json` (H1's sweep) shows Tier A zero at every scored point. If neither row is on the card, the comparison isn't in the run of record: say so. |
```

### `docs/demo/pitch-and-qa.md:44`

```text
BEFORE
High-resolution geothermal catalogs usually depend on dedicated arrays; we test how much of the seismicity the public stream alone can surface. *Only if the Saturday depth call passed (`docs/03-schedule.md`; it did, see `docs/lanes/H2-seismology.md`, "Depth call"):* "we test how much structure the public stream alone can surface" — and even then "structure" means the spatial pattern of candidate events, never fractures or a mechanism (no relative relocation was done). |

AFTER
High-resolution geothermal catalogs usually depend on dedicated arrays; we test how much of the seismicity, and at what depths, the public stream alone can surface. The Saturday depth call passed on amended criteria (`docs/lanes/H2-seismology.md`, "Depth call"): that licenses showing depths as located, not reading a pattern into them (no relative relocation was done). |
```

### `docs/demo/pitch-and-qa.md:46`

```text
BEFORE
| 15 | What's your external validation? | Public-catalog recall, a synthetic depth-resolution test, the null test, the STA/LTA baseline, and the published FORGE depth band. Ridgecrest only if it got done. |

AFTER
| 15 | What's your external validation? | Public-catalog recall, a synthetic depth-resolution test, the null test, the STA/LTA baseline, and, for the events we recovered, the same-day public regional catalog's depths (published FORGE catalogs cover other dates and places, so they are context, not a check). Ridgecrest only if it got done. |
```

### `docs/demo/pitch-and-qa.md:47`

```text
BEFORE
| 16 | Could these belong to Cape Station instead of FORGE? | Possibly. The operations are close together and we deliberately don't attribute. We show verified facility locations and our locations with error; attribution needs operator data. |

AFTER
| 16 | Could these belong to Cape Station instead of FORGE? | We don't know, and we don't attribute to either. Several operations share the region; we show cited facility locations and our locations with their errors. Attribution needs operator data we don't have. |
```

### `docs/demo/pitch-and-qa.md:55`

```text
BEFORE
| 24 | Why not use the operator's catalog? | It isn't public in real time, and independence is the point. |

AFTER
| 24 | Why not use the catalogs from the private downhole arrays? | They aren't public in real time, and independence is the point. |
```

### `docs/demo/pitch-and-qa.md:58`

```text
BEFORE
| 27 | How good are the magnitudes? | Calibrated on {n} matched public events, leave-one-out error ±{looMae}. If that had been bad, you wouldn't see magnitudes. |

AFTER
| 27 | How good are the magnitudes? | They're a local magnitude, {magType}, calibrated on {n} matched public events of one catalog magnitude type; leave-one-out error ±{looMae}, read next to the null-model error ±{nullModelMae} (the error of a model that gives each event the mean magnitude of the others). {belowCalibratedRange} of the candidate magnitudes lie below the calibrated range, so they're extrapolated and, near the detection limit, biased upward. If the gate had failed there would be no magnitudes on screen. |
```

### `docs/demo/pitch-and-qa.md:59`

```text
BEFORE
| 28 | Does your catalog lower the completeness magnitude? | *(G-R panel, if built.)* The public curve stops at its Mc; ours continues below it. |

AFTER
| 28 | Does your catalog lower the completeness magnitude? | *(No G-R panel is in the shell; say "in the bundle".)* The public curve counts only the catalog events of the calibration magnitude type ({magType}); ours is the candidates' {magType}. Ours continues below the public curve's Mc, with the caveat that its low end is extrapolated and censored, so we describe the curve and quote no b-value from it. |
```

### `docs/demo/pitch-and-qa.md:70`

```text
BEFORE
2. **Thumbnail:** semi-transparent terrain, the geothermal reference, public points in white, strict events in amber forming structure below, and a "{publicCatalogCount} PUBLIC → {candidateCount} RECOVERED" counter.

AFTER
2. **Thumbnail:** semi-transparent terrain, the geothermal reference, public points in white, strict candidate events in amber below, and a "{publicCatalogCount} PUBLIC → {candidateCount} RECOVERED" counter.
```

### `docs/demo/pitch-and-qa.md:78`

```text
BEFORE
10. **Built with:** Python, ObsPy, SeisBench, PyOcto, scikit-fmm, FastAPI, Next.js, React Three Fiber.

AFTER
10. **Built with:** Python (ObsPy, SeisBench, PyOcto, scikit-fmm, SciPy, Pydantic, FastAPI), TypeScript (Next.js, React, three.js, React Three Fiber, zustand); the full list is in `docs/demo/devpost.md`.
```

### `docs/demo/pitch-and-qa.md:89`

```text
BEFORE
6. Reproducing the showcase: the final `runId`, its config, and expected counts.

AFTER
6. Reproducing the showcase: the final `runId` and its config; counts are read from the bundle, never typed in.
```

### `docs/demo/pitch-and-qa.md:110`

```text
BEFORE
| `{n}` matched events, `{looMae}` (Q27) | `validation.json` → `magnitude.n`, `magnitude.looMae` (only when present) | Not on screen; read from `validation.json` |

AFTER
| `{n}` matched events, `{looMae}` (Q27) | `validation.json` → `magnitude.n`, `magnitude.looMae` (only when present and the magnitude kill switch did not fire) | Not on screen; read from `validation.json` |
| `{magType}` (Q27, Q28) | `meta.json` → `run.matching.magnitude.calibrationMagType` (the one catalog magnitude type calibrated on); the candidates' own type is on every event's `magnitude.type` and in the drawer header | Evidence drawer header ("M x.x <type>") |
| `{nullModelMae}` (Q27) | `meta.json` → `run.matching.magnitude.leaveOneEventOut.nullModelMae` (`FYI-H2-7`: quote `looMae` only next to it) | Not on screen; read from `meta.json` |
| `{belowCalibratedRange}` (Q27) | `meta.json` → `run.matching.magnitude.magnitudes.belowCalibratedRange` (with `magnitudes.written` as the denominator) | Not on screen; read from `meta.json` |
```

### `docs/demo/pitch-and-qa.md:115`

```text
BEFORE
| `{depthBand}` | No bundle field. Read it off the depth ruler with STRICT on; say what is visible, never a mechanism | Scene, depth ruler and slices |

AFTER
| `{depthBand}` | No bundle field. Read it off the depth ruler with STRICT on; say what is visible (a depth band, bunched or spread), never a geometry and never a mechanism | Scene, depth ruler and slices |
```

### `docs/demo/pitch-and-qa.md:122`

```text
BEFORE
| G-R curve (Q28) | `validation.json` → `gr` (only when present) | No G-R panel exists in the shell yet; say "in the bundle" unless one lands |

AFTER
| G-R curve (Q28) | `validation.json` → `gr` (only when present); its `publicCum` counts the public regional catalog's magnitudes of `{magType}` only (`REQ-H2-13`) | No G-R panel exists in the shell; say "in the bundle" |
```

## `docs/demo/devpost.md`

### `docs/demo/devpost.md:27`

```text
BEFORE
Enhanced geothermal is expanding around Milford, Utah. Operators and research teams see underground: dense downhole geophones and fiber arrays give them detailed pictures of the microseismicity that shows an engineered reservoir taking shape. The public gets the regional catalog, which lists a sparse slice of that activity. Microseismicity is also what induced-seismicity oversight (traffic-light protocols) is built on. We wanted to know how much of the underground picture the public seismic network is already hearing, and to show it in a way a non-seismologist can read in five seconds.

AFTER
Enhanced geothermal is expanding around Milford, Utah. Operators and research teams see underground: dense downhole geophones and fiber arrays give them detailed pictures of the microseismicity there. The public gets the public regional catalog, which lists a sparse slice of that activity. Microseismicity is also what the oversight of geothermal operations (traffic-light protocols) is built on. We wanted to know how much of the underground picture the public seismic network is already hearing, and to show it in a way a non-seismologist can read in five seconds.
```

### `docs/demo/devpost.md:31`

```text
BEFORE
The public regional catalog shows `<from meta.json: summary.publicCatalogCount>` earthquakes under Utah's geothermal frontier in the showcase window (`<from meta.json: run.windowLabel>`). We rebuilt the catalog from raw public seismometers with neural phase picking, multi-station association, relocation and quality tiers, and found `<from meta.json: summary.candidateCount>` candidate events, `<from meta.json: summary.strictQualityCount>` at strict quality, underground where the public view is nearly empty.

AFTER
The public regional catalog lists `<from meta.json: summary.publicCatalogCount>` events under Utah's geothermal frontier in the showcase window (`<from meta.json: run.windowLabel>`). We rebuilt the catalog from raw public seismometers with neural phase picking, multi-station association, relocation and quality tiers: `<from meta.json: summary.candidateCount>` candidate events, `<from meta.json: summary.strictQualityCount>` at strict quality, underground where the public view is nearly empty.
```

### `docs/demo/devpost.md:49`

```text
BEFORE
- **Locate:** our own grid locator on a published FORGE velocity model from the DOE Geothermal Data Repository (`<from meta.json: run.velocityModel.name>`), with station statics, per-event uncertainty and a synthetic recovery test on the real station geometry.

AFTER
- **Locate:** our own grid locator (coarse-then-fine search over eikonal travel-time tables solved with scikit-fmm) on a published FORGE velocity model from the DOE Geothermal Data Repository (`<from meta.json: run.velocityModel.name>`), with station statics fitted on the recovered public events, per-event uncertainty and a synthetic recovery test on the real station geometry.
```

### `docs/demo/devpost.md:51`

```text
BEFORE
- **Magnitude:** a local magnitude calibrated on the matched public events, with leave-one-out error, kept only if that error is acceptable.

AFTER
- **Magnitude:** a local magnitude (`ML_cal`) calibrated on the matched public events of one catalog magnitude type, with a leave-one-out error reported next to a null-model error; the stage nulls every magnitude when its gate fails, so no magnitude on screen outlives a bad calibration.
```

### `docs/demo/devpost.md:52`

```text
BEFORE
- **Validate and export:** a null test, an STA/LTA baseline, Gutenberg–Richter, then a static bundle (`meta.json`, `events.json`, `validation.json`, evidence snippets) that the web app reads through a provider. Nothing on the demo path depends on a live service.

AFTER
- **Validate and export:** a null test, an STA/LTA baseline, a Gutenberg–Richter curve whose public side counts only the catalog events of the calibration magnitude type, then a static bundle (`meta.json`, `events.json`, `validation.json`, evidence snippets) that the web app reads through a provider. Nothing on the demo path depends on a live service.
```

### `docs/demo/devpost.md:54`

```text
BEFORE
The web app is Next.js with React Three Fiber: terrain baked from public elevation tiles, events as instanced points with error halos, borehole sensors drawn at their true depth, reference features from GDR well surveys and UGS layers with their sources cited. The shell renders no digit of its own: a test parses every text node and fails on a number. Every run records its full config; the Run details panel prints it verbatim.

AFTER
The web app is Next.js with React Three Fiber: terrain baked from public elevation tiles, events as instanced points with error halos, borehole sensors drawn at their true depth, reference features from GDR well surveys and UGS layers with their sources cited, a plan view with a true-scale depth section, an evidence drawer with the record section behind every dot, and a time scrubber that replays the window over a histogram of it. The shell renders no digit of its own: a test parses every text node and fails on a number. Every run records its full config; the Run details panel prints it verbatim.
```

### `docs/demo/devpost.md:67`

```text
BEFORE
- Baseline (only if `summary.baseline` is present): at comparable quality, neural picking yields `<from meta.json: summary.baseline.gain>`× the strict events of STA/LTA.
- Magnitudes (only if `validation.magnitude` is present): calibrated on `<from validation.json: magnitude.n>` matched public events, leave-one-out error ±`<from validation.json: magnitude.looMae>`.

AFTER
- Baseline, one of two sentences, both conditional (H1 is rescoring STA/LTA on the run's statics scale, so nothing is asserted until the run of record's `validation.json` is exported). Only if `summary.baseline` is present: at comparable quality, neural picking yields `<from meta.json: summary.baseline.gain>`× the strict events of STA/LTA. Otherwise, only if `validation.baseline` holds both `full` rows: at the strict tier, PhaseNet produced `<from validation.json: baseline[method=phasenet, associationProfile=full].tiers.A>` candidate events and STA/LTA produced `<from validation.json: baseline[method=stalta, associationProfile=full].tiers.A>`, through the same downstream code and the same tier bars. If neither holds, no baseline sentence.
- Magnitudes (only if `validation.magnitude` is present): a local magnitude, `<from meta.json: run.matching.magnitude.calibrationMagType>`-calibrated `ML_cal`, fitted on `<from validation.json: magnitude.n>` matched public events of that one catalog magnitude type; leave-one-out error ±`<from validation.json: magnitude.looMae>` against a null-model error of ±`<from meta.json: run.matching.magnitude.leaveOneEventOut.nullModelMae>` (the error of a model that gives each event the mean magnitude of the others). `<from meta.json: run.matching.magnitude.magnitudes.belowCalibratedRange>` of the `<from meta.json: run.matching.magnitude.magnitudes.written>` candidate magnitudes lie below the calibrated range, so they are extrapolated and, near the detection limit, biased upward.
- Gutenberg–Richter (only if `validation.gr` is present): the public curve counts only the catalog events of the calibration magnitude type; the recovered curve is `ML_cal`. The recovered curve continues below the public curve's completeness magnitude; its low end is extrapolated and censored, so we describe the curve and quote no b-value from it.
```

### `docs/demo/devpost.md:73`

```text
BEFORE
- **Candidate events, not verified earthquakes.** Each needs consistent picks across multiple stations; strict ones meet, on every quality metric, a bar that three-quarters of the recovered public-catalog events meet. We don't claim all of them are real,

AFTER
- **Candidate events, not verified earthquakes.** Each needs consistent picks across multiple stations; strict means every quality metric is within the range reached by three-quarters of the public events we recovered. We don't claim all of them are real,
```

### `docs/demo/devpost.md:74`

```text
BEFORE
- **Sparse public geometry.** Depth is the weakest dimension. Published catalogs from downhole arrays are far denser and sharper than ours; every halo is that event's own error, and we claim no fracture geometry.

AFTER
- **Sparse public geometry.** Depth is the weakest dimension. Published catalogs from downhole arrays are far denser and sharper than ours; every halo is that event's own error, and we show depths as located without reading a geometry or a mechanism into their pattern (that would need relative relocation, which we did not run).
```

### `docs/demo/devpost.md:82`

```text
BEFORE
We required S picks and a near station for the strict tier, added station statics, measured depth resolution on synthetic events with the real geometry, and kept a plan-view hero ready in case the depth gate failed.

AFTER
We required S picks and a near station for the strict tier, added station statics, measured depth resolution on synthetic events with the real geometry, and kept a plan-view hero ready in case the depth gate failed. The Saturday depth call passed on amended criteria: the 3D hero stays, and we say nothing about what the pattern of depths means.
```

### `docs/demo/devpost.md:110`

```text
BEFORE
Python, ObsPy, SeisBench (PhaseNet), PyOcto, scikit-fmm, SciPy, NumPy, pandas, PyArrow, PyProj, Pydantic, FastAPI, uv; TypeScript, Next.js, React, three.js, React Three Fiber, zustand, Vitest, pnpm; EarthScope FDSN services, USGS ComCat, USGS 3DEP, DOE Geothermal Data Repository, Utah Geological Survey.

AFTER
Pipeline (`services/seismic`, package `hq`): Python, ObsPy, SeisBench (pretrained PhaseNet), PyOcto, scikit-fmm, SciPy, NumPy, pandas, PyArrow, xarray, netCDF, PyProj, Pydantic, PyYAML, Requests, Matplotlib; uv, pytest, ruff. Live worker (`services/api`): FastAPI, Uvicorn. Contracts (`packages/contracts`): Pydantic models as the source of truth, JSON Schema, TypeScript generated with json-schema-to-typescript. Web app (`apps/web`): TypeScript, Next.js (static export), React, three.js, React Three Fiber, drei, react-postprocessing, zustand, Vitest, Testing Library, ESLint, pnpm; deployed on Vercel. Data: EarthScope FDSN services, USGS ComCat, USGS 3DEP elevations (AWS Terrain Tiles, Terrarium encoding), DOE Geothermal Data Repository, Utah Geological Survey.
```

### `docs/demo/devpost.md:138`

```text
BEFORE
| `<from meta.json: summary.baseline.gain>` | `meta.json` → `summary.baseline.gain` | only if `summary.baseline` is not null (VAL-01 writes it only when the gain holds in both association profiles); otherwise delete the sentence |

AFTER
| `<from meta.json: summary.baseline.gain>` | `meta.json` → `summary.baseline.gain` | only if `summary.baseline` is not null (VAL-01 writes it only when the gain holds in both association profiles); otherwise delete the sentence and fall back to the two strict counts below |
| `<from validation.json: baseline[method=phasenet, associationProfile=full].tiers.A>` / `<from validation.json: baseline[method=stalta, associationProfile=full].tiers.A>` | `validation.json` → `baseline[]`: the `tiers.A` of the row with `method: "phasenet"` and of the row with `method: "stalta"`, both with `associationProfile: "full"` (the Validation card's "Strict events, PhaseNet vs STA/LTA" row) | only if both rows exist; used only when `summary.baseline` is null; otherwise delete the sentence |
```

### `docs/demo/devpost.md:146`

```text
BEFORE
| `<from validation.json: magnitude.looMae>` | `validation.json` → `magnitude.looMae` | same as above |

AFTER
| `<from validation.json: magnitude.looMae>` | `validation.json` → `magnitude.looMae` | same as above |
| `<from meta.json: run.matching.magnitude.calibrationMagType>` | `meta.json` → `run.matching.magnitude.calibrationMagType` | same as above |
| `<from meta.json: run.matching.magnitude.leaveOneEventOut.nullModelMae>` | `meta.json` → `run.matching.magnitude.leaveOneEventOut.nullModelMae` (`FYI-H2-7`) | same as above |
| `<from meta.json: run.matching.magnitude.magnitudes.belowCalibratedRange>` / `<from meta.json: run.matching.magnitude.magnitudes.written>` | `meta.json` → `run.matching.magnitude.magnitudes.belowCalibratedRange` and `.written` | same as above |
```

## Remaining hits of the forbidden-word grep

The grep below, run over the three files after the pass. Every hit is one of the following; none is a claim.

```
grep -n -i -E 'structure|fracture|fault|confirmed|caused|induced|triggered|predict|official|missed|operator' README.md docs/demo/pitch-and-qa.md docs/demo/devpost.md
```

| Hit | Where | Why it stays |
| --- | --- | --- |
| "Confirmed earthquake" | `README.md`, "What this is not" | Quoted in order to reject it; the line carries the `copy-ok` marker | <!-- copy-ok -->
| "predicts" | `README.md`, "What this is not" | Negation ("Nothing here predicts anything"); the line carries the `copy-ok` marker | <!-- copy-ok -->
| "fault" | `README.md`, "Data modes" | False positive: the substring inside "default" |
| "missed" | `README.md`, "Validation" | Negation ("we never claim ... that any event was missed by anyone") |
| "operator" / "operators" | `README.md` intro, "What this is not"; pitch (the thirty-second pitch, the closing line, Q3, Q16, Q29, the Devpost outline's Inspiration item); `docs/demo/devpost.md` "Inspiration" and "Limits" | Either the statement `docs/00-project.md` asks us to make first (operators and research teams run denser monitoring than the public gets) or a negation ("we never attribute ... to any operator", "attribution needs operator data we don't have"). No sentence attributes an event to an operation, and the phrase "the operator's" is gone |

Words that are now absent from all three files: structure, fracture, confirmed (outside the marked line), caused, induced, triggered, official, reservoir, conduit, "earthquakes we found", "detected earthquakes", "the catalog missed", "the network missed", "better than three-quarters". "False-positive rate" appears only in the sentences that refuse to quote one.

## H3 user-facing strings (drawer, scene, scrubber)

Read, not edited: `apps/web/src/drawer/{EvidenceDrawer,Header,RecordSection,Figures}.tsx` and `format.ts`; `apps/web/src/scene/plan/DepthSection.tsx` and `section.ts` (`outsideText`); `apps/web/src/scene/time/TimeScrubber.tsx`; `apps/web/src/scene/references/{CornerNote,VerticalBadge,DepthRuler,FeaturesLayer,StationsLayer}.tsx`, `badge.ts`, `ruler.ts`; `apps/web/src/scene/terrain/surface.ts`. H4's shell labels (`Counters`, `Pills`, `RevealButton`, `Shell`, validation rows, Run details) were read the same way.

**No user-facing string breaks a language rule.** Every label says "candidate event", "public regional catalog" (drawer: "in the public regional catalog"), "Tier A/B/C", "stations agreed", "modeled arrival", "Depth section", "Time", "public"; the depth-section header says "framed on Tier A and B", not what the frame contains; the magnitude line prints the type with the value (`M x.x ML_cal ±y`); the depth label comes from the bundle (`scene.depthLabel`). No request to H3 was filed. Two notes, for H3 to take or leave:

- `apps/web/src/drawer/Figures.tsx`: the SVG tooltips read `68% horizontal error ±N m` and `68% vertical error ±N m`, and `apps/web/src/scene/plan/DepthSection.tsx` reads `true scale (1 km = 1 km)`. These digits are definitions (`hErrM`/`vErrM` are the `68%` axes in `docs/02-contracts.md`; true scale is one-to-one), not facts about the data, so they don't break the numbers-as-facts rule; the shell's `copy.test.ts` would flag them if they were in H4's files. Wording without a digit ("one-sigma", "true scale") would make the drawer and section pass the same test.
- Internal names and comments still say "structure": `sectionStructureFit` and `framedOn: "structure"` in `apps/web/src/scene/plan/{geometry,section}.ts`, and comments in `scene/references/ruler.ts`, `scene/reveal/timeline.ts`, `scene/Canvas.tsx`, `scene/camera/bounds.ts`, `scene/plan/DepthSection.tsx`. None reaches the screen (the header says "framed on Tier A and B"). A rename to "cluster" or "tierAB" would keep the code aligned with the depth call, but nothing user-facing depends on it.

## What the shell does today, as the pitch describes it

Verified against `apps/web/src/shell/useKeyboard.ts`, `apps/web/src/state/demo.ts`, `apps/web/src/app/page.tsx` and the scrubber: Space steps public → reveal → strict → time (from "revealed", Space sets STRICT, then turns time mode on); E selects the hero event (`meta.scene.heroEventId`) and opens the drawer; Esc closes it (the scrubber hides while the drawer is open, so the flow presses Esc before the time beat); D toggles Run details; the Validation card mounts bottom-left once the reveal starts and shows only rows whose source field exists (recall, strict events, median stations, median residual, depth resolution, "Strict events, PhaseNet vs STA/LTA", gain, chance associations); the real `TimeScrubber` from `@/scene/time` is mounted in `page.tsx` and appears in time mode after the reveal; the LIVE pill exists only when the build sets `NEXT_PUBLIC_LIVE_ENABLED`.


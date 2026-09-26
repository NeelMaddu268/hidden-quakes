# 00 · Project

## What we're building

Public regional earthquake catalogs show only a sparse slice of the microseismicity around Utah's geothermal-development region near Milford. **Hidden Quakes** is an open, near-real-time, public-data-only seismic layer. It rebuilds a denser, quality-tiered, inspectable catalog from raw public waveforms and shows it underground.

It's for people without operator data: researchers, journalists, regulators, local communities and energy observers. Operators and research teams already run far denser downhole and fiber arrays, and we say so first.

The core sentence: *"The public regional catalog showed {publicCatalogCount} earthquakes here on September 10. We rebuilt the catalog directly from public seismic waveforms using neural phase picking, multi-station association, relocation and quality filtering, and recovered a much denser view of the microseismic activity beneath Utah's geothermal field."*

Our contribution is **product + pipeline + public access + visual explainability**, not a new seismology algorithm. PhaseNet, PyOcto, QuakeFlow, GaMMA and published FORGE research catalogs all exist, and we say so.

## The judge sequence we're building for

1. "I understand it instantly." (5-second test: dark terrain, one glowing geothermal reference, **PUBLIC {N}**, a giant **REVEAL HIDDEN SIGNAL**. Nothing else.)
2. "Whoa, there are way more events." (the reveal)
3. "Those points form actual underground structure." (only if the depth gate passes; see `docs/03`)
4. "You built the system that found and located them from raw public seismometers?" (the evidence drawer and validation)

Failing beat 1–2 loses. Failing beat 4 under questioning also loses. We need both.

## Claims we may make, and what licenses each

| Claim | Allowed only when |
| --- | --- |
| "The public regional catalog lists {publicCatalogCount} events here in this window." | The catalog query for the exact window is saved in the run |
| "Using only public waveforms, we recovered {recoveredCatalogCount} of {publicCatalogCount}." | One-to-one matching ran and every miss is listed |
| "We associated {additionalCount} more candidate events; {strictAdditionalCount} pass our strict tier." | Tiers were derived from matched-event quantiles |
| "Strict means located at least as well as three-quarters of the public events we recovered." | That derivation is stored in `ProcessingRun.tiering` |
| "This station geometry resolves depth to about ±{medianVErrM} m." | The synthetic recovery test ran on the real station geometry |
| "At comparable quality, neural picking yields {gain}× more strict events than STA/LTA." | The baseline table supports it in both association profiles |
| "Chance associations on time-scrambled picks: {meanChanceEvents}." | The null test ran |
| "Magnitudes extend the Gutenberg–Richter trend below the public catalog's completeness." | Leave-one-out MAE is acceptable and the curve shows it |
| "Activity clusters in space and time." | The time scrubber actually shows it. Describe what's visible; never interpret mechanism. |
| "Processes the last 2 hours in under {latency} minutes." | Live latency was measured on the demo machine |

## Claims we must never make

- "We discovered earthquakes scientists missed" or "FORGE didn't know these existed."
- "The official catalog missed hundreds of earthquakes." Say "the public regional catalog shows N."
- "We mapped FORGE's fractures", or any fracture-plane claim unless the depth gate passes. Even then, say "structure", not "fracture".
- "These were caused by FORGE / Cape Station / operator X." Several operations share the region; we never attribute.
- "{N} confirmed earthquakes." They're candidate events, tiered.
- "Our AI predicts earthquakes." We detect, associate and locate.
- "We invented ML earthquake detection", or anything implying operators lack better monitoring.
- A false-positive rate. There's no ground truth for events the public catalog lacks; quote the null test and tiers.
- Any number from pre-event research. Every number comes from the pipeline rebuilt during HackGT.

Location words to use: "Utah's geothermal field", "the geothermal-development region", "the geothermal frontier around Milford, Utah". When showing a well or facility, mark whether its location is verified from a cited source.

## Scope

| Tier | In scope |
| --- | --- |
| **P0** | Showcase pipeline end to end (ingest → picks → association → location → statics → matching → tiers) · mock and showcase providers · exporter with evidence snippets · 3D scene with terrain, depth ruler, stations and borehole sensors · the reveal · Public / All / Strict filter · evidence drawer · plan-view fallback · synthetic depth test · deployed static URL plus offline build · pitches |
| **P1** | STA/LTA baseline · time scrubber · null test · validation panel and Run details · magnitude plus Gutenberg–Richter · 3D velocity grids · relative relocation (if the depth gate needs it) · Live mode plus Snapshot · Devpost video |
| **P2** | Ridgecrest recall check · processing telemetry panel |
| **Never** | Chatbot, accounts, alerts, nationwide map, multiple sites, mobile, AR, LLM explainer, social features, auth, a database, a historical explorer |

We have ~36 hours and strong agents, so P1 is expected, not optional, but only after its gate (`docs/03`). Extra time goes into correctness, tests and polish, never into scope beyond this table.

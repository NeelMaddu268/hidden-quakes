# 2-minute video: shot list

The video is the 2-minute pitch in `pitch-and-qa.md`, recorded from the deployed site (or `make offline` with Wi-Fi off) at full HD, in a browser with no extensions and the SHOWCASE pill active. Screen only, no webcam. Voice-over recorded after the screen capture, read from the pitch table; every number in the voice-over comes from `make story` (`data/story/pitch-filled.md`), never from memory.

Before recording: `?mode=showcase` header reads "Showcase · {windowLabel} · run {runId}"; no SYNTHETIC banner; the console is clean; the drawer closes; Esc resets. Record two full takes; keep the one where the reveal counter finishes before the next key.

| # | Time | Keys | What is on screen | Voice-over (from `pitch-and-qa.md`) | Cut on |
| --- | --- | --- | --- | --- | --- |
| 1 | 0:00–0:10 | none (R first, to reset) | Start frame: terrain, the reference features labelled "(approximate)", PUBLIC {N}, REVEAL HIDDEN SIGNAL | "This is Utah's geothermal frontier near Milford. The public regional catalog lists {publicCatalogCount} events here on {windowLabel}." | the last word |
| 2 | 0:10–0:25 | Space | The counter climbs to {candidateCount}; the camera dollies to the side view; filter pills appear | "That's what the public regional catalog shows. We went straight to the raw public seismometers." A short silence while the counter finishes. | the counter settling |
| 3 | 0:25–0:40 | Space (STRICT), then slow orbit by mouse | Strict candidate events only; depth ruler on the left | "These are the strict ones: every quality metric within the range reached by three-quarters of the public events we recovered. They sit {depthBand} below the surface." Only what is on screen: the depth band, bunched or spread. No geometry, no mechanism. | the orbit completing a quarter turn |
| 4 | 0:40–1:00 | E | The evidence drawer on the hero: "{nStations} stations agreed", the record section with P and S picks and modeled arrivals, plan view and depth section | "How do we know this dot is an event and not noise? {nStations} stations agreed. The neural picker marks P and S on each, and the lines are the arrivals the final location implies. They agree." | the record section fully drawn |
| 5 | 1:00–1:15 | Esc, then hover the validation card (D only if the run-details beat is wanted) | Validation card, bottom-left | "We recover {recoveredCatalogCount} of {publicCatalogCount} public events. This station geometry resolves depth to about ±{medianVErrM} m." Read any other row exactly as the card shows it; a gain or null-test sentence only if that row is on the card. | the card in full view |
| 6 | 1:15–1:30 | Space (TIME) | The time scrubber along the bottom replays the window; the histogram and clock | "Same events, replayed over the day." Say what the histogram shows: bursts or a steady trickle, and when. Nothing about why. | the scrubber reaching the end |
| 7 | 1:30–1:40 | optional: click LIVE | Only if the LIVE pill is up (docs/03 kill switch) | "It runs on rolling windows too: the last two hours, updated {n} minutes ago." Without LIVE, stay in the scrubber or the drawer. | the label reading Live |
| 8 | 1:40–2:00 | R, then Space, slow orbit | Full scene after the reveal, all candidate events | "Operators can see underground. Regulators, journalists and communities mostly can't. This is an open, public-data-only window into what the public network is already hearing." Stop. | fade to the thumbnail frame |

Thumbnail (still, exported separately): the strict view (the STRICT shot) at the moment the counter reads "{publicCatalogCount} PUBLIC → {candidateCount} RECOVERED"; semi-transparent terrain, public points in white, strict candidate events in amber below. It has to read with no video.

Captions: burn in the two labels the audience must read, "Showcase · {windowLabel} · run {runId}" (top-left, already on screen) and "candidate events, not a confirmed catalog" (lower third, the reveal shot only) <!-- copy-ok -->. No other text.

Do not show: the MOCK pill, the SYNTHETIC banner, Run details unless asked, any browser chrome, any number that is not on screen or in `data/story/pitch-filled.md`.

Deliverable: `hidden-quakes-pitch.mp4` (a standard video file within the upload size limit), uploaded to the platform Devpost accepts; its URL goes into the `<video URL>` placeholder in `devpost.md`.

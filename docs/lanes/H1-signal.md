# H1 · Signal lane

**Mission:** turn public EarthScope waveforms into trustworthy phase picks. Everything downstream is only as good as your station metadata and your picks. Your two big moments are **Check A and Check B at 10:00 PM**; if they pass, the project is viable.

## Paths

**You own:** `services/seismic/hq/ingest/`, `hq/preprocess/`, `hq/pick/`, `hq/baseline/`, `hq/config/signal.py`, `services/seismic/configs/showcase/signal.yaml`, `services/seismic/tests/signal/`, `data/cache/mseed/`, `data/cache/stationxml/`

**You never write:** H2's `hq/associate|locate|tier|match|magnitude/`, `run.yaml`, `seismology.yaml`; H4's `hq/export|validate/`, `cli.py`, `runs.py`, `packages/contracts/`; anything in `apps/`.

## What you consume

| What | From | Where | Until it exists |
| --- | --- | --- | --- |
| Window, bbox, origin | H2 | `configs/showcase/run.yaml` | 8:05 PM; don't start downloads before it |
| Contracts + `hq_contracts.io` | H4 | `packages/contracts/` | code against the spec in `docs/02` |
| `RunContext` / `hq run` | H4 | `hq/runs.py`, `hq/cli.py` | a local `RunContext` stand-in in your tests |
| Public catalog (for the 3 known events) | H2 | `runs/<id>/catalog.parquet` | lands ~8:25 PM; do SEIS-01 and SEIS-03 first |
| Tier bars and station statics (STA/LTA sweep scoring only) | H2 | `runs/<id>/run.json` → `tiering`; `runs/<id>/statics.parquet` once `hq.baseline.score` passes it (REQ-H1-5; not read yet) | scoring is a CLI rerun on a run that went through `tier` (SEIS-07 → As built) |

## What you produce

| What | For | Where | Due |
| --- | --- | --- | --- |
| `stations.parquet` (with real sensor depths) | H2, H4 | `runs/<id>/stations.parquet` | 9:00 PM |
| Known-event picks + record sections | H2 (Check B) | `runs/<id>/known/` | 10:00 PM |
| Waveform cache + `read_window` / `read_inventory` / `display_copy` | H2 (magnitude), H4 (evidence) | `data/cache/mseed/`, `hq.ingest.cache`, `hq.preprocess` | 10:30 PM |
| Full-window PhaseNet picks | H2 | `runs/<id>/picks.parquet` | 12:30 AM |
| STA/LTA picks + threshold sweep | H4 (VAL-01) | `runs/<id>/picks_stalta.parquet`, `baseline_sweep.parquet`, `baseline_reference.json` (scored reruns only) | 4:00 AM |
| Audio clips + manifests (SEIS-09) | H3, through the lead | `hidden-quakes-busiest-hour.*`, `hidden-quakes-hero.*` (`.ogg`, `.mp3`, `.json`) in an output folder, never `apps/` | features freeze, Sat night |
| Station-day helicorder + manifest (SEIS-10) | H3 (WEB-10 panel, gallery) | `apps/web/public/helicorder/station-day.png` and `station-day.json` | features freeze, Sat night |

## Tickets, in order

### SEIS-01 · P0 · Start 8:05 PM — Station inventory

- **Goal:** query stations in the bbox at channel level for the window; build `Station` rows with `sensorDepthM` from each channel's `depth`, `kind`, sample rate and chosen channel triplet.
- **Files:** `hq/ingest/inventory.py`, `hq/config/signal.py` (station-selection section), `configs/showcase/signal.yaml`, `tests/signal/test_inventory.py`
- **In → out:** `run.yaml` → `data/cache/stationxml/*.xml`, `runs/<id>/stations.parquet`, `runs/<id>/inventory_report.json`
- **Depends on:** `run.yaml` (8:05)
- **Accept:** prints a station table (id, kind, rate, channels, sensorDepthM); every borehole station has `sensorDepthM > 0` or is flagged in the log; reports how many stations have any data in the window; test parses a small recorded StationXML fixture offline.

### SEIS-03 · P0 · Start 8:05 PM — Preprocessing profiles

- **Goal:** implement `surface-100`, `surface-hi`, `borehole-A`, `borehole-B` and `display_copy()`, with explicit anti-aliasing, plus `TimeMap` for profile B.
- **Files:** `hq/preprocess/__init__.py`, `hq/preprocess/profiles.py`, `tests/signal/test_preprocess.py`
- **In → out:** raw `Stream` → 100 Hz model-ready `Stream` + `TimeMap`
- **Depends on:** nothing (synthetic tests)
- **Accept:** a synthetic 1,000 Hz chirp keeps no energy above 50 Hz after profile A (spectrum test, < −40 dB); a known onset round-trips through profile B's `TimeMap` within 1 ms; gaps stay gaps (no zero-filled samples).

### SEIS-02 · P0 · Start ~8:30 PM — Known-event windows

- **Goal:** cut 10-minute windows around the 3 largest public events in the showcase window.
- **Files:** `hq/ingest/windows.py`, `tests/signal/test_windows.py`
- **In → out:** `catalog.parquet` + `stations.parquet` → `runs/<id>/known/windows.json` + a gap report per event, with waveforms read through `read_window` from the channel-day cache in `data/cache/mseed/`; `known/known_windows.record.json` (runtime, counts, params), written by the CLI
- **Depends on:** SEIS-01, SEIS-05 (cache), MATCH-01 (H2)
- **Decision (Fri night):** SEIS-02 never downloads. The channel-day cache has one writer (SEIS-05's downloader), and a second process writing the same channel-day files would lose chunks. If an event's hours aren't cached, the window reports "not downloaded" rather than fetching.
- **Accept:** at least 8 stations with three-component data per window; a gap report per window.

### SEIS-05 · P0 · Start ~8:30 PM — Full-window ingestion

- **Goal:** download the whole showcase window ±5 min in hour chunks, with retries, a cache keyed by channel-day, a gap log, and `read_window` / `read_inventory`. The same function must work for any window, including the last 2 hours (Live mode reuses it).
- **Files:** `hq/ingest/download.py`, `hq/ingest/cache.py`, `tests/signal/test_cache.py`
- **In → out:** `stations.parquet` → `data/cache/mseed/{net}.{sta}.{loc}.{cha}.{YYYYMMDD}.mseed`, `runs/<id>/gaps.parquet`, `runs/<id>/download_report.json` (the Check A table)
- **Depends on:** SEIS-01
- **Accept (Check A):** at least 12 useful stations (three components, < 20% gaps); a rerun is a pure cache hit; `read_window` returns gaps as separate traces. When the full download finishes (~10:30 PM), your human hands the whole `data/cache/` folder to H4 by AirDrop or USB.

### SEIS-04 · P0 · Start ~9:00 PM — PhaseNet on known events + weight A/B

- **Goal:** run PhaseNet on the known-event windows for each candidate weight set per profile, and pick the weights.
- **Files:** `hq/pick/__init__.py`, `hq/pick/phasenet.py`, `hq/pick/ab.py`, `tests/signal/test_pick.py`
- **In → out:** known-event windows → `runs/<id>/known/picks.parquet`, `known/ab.csv`, `known/ab.json` (chosen weights, adopted profiles, Check B), one record-section PNG per event. `known/pick_known.record.json` (runtime, counts, params) comes only from `hq.pick.ab.run(ctx)`, which no stage or CLI calls: a real run has no record of this sub-step (flagged in the SEIS-08 PR)
- **Depends on:** SEIS-02, SEIS-03
- **Accept (Check B):** P and S picks on ≥ 8 stations for ≥ 3 events; P before S on every station; Spearman ρ ≥ 0.8 between P time and epicentral distance; the A/B table names the chosen weights per profile. Then `make publish-run RUN=<runId>` and give H2 the runId.

### SEIS-06 · P0 · Start ~10:30 PM — Full-window picking

- **Goal:** pick the full window at probability ≥ 0.1 with the chosen weights, parallel by station, runtime recorded.
- **Files:** `hq/pick/run.py` (stage `run(ctx)`), `hq/preprocess/chunks.py`, `tests/signal/test_chunks.py`, `tests/signal/test_pick_run.py`
- **In → out:** cache → `runs/<id>/picks.parquet`, `runs/<id>/pick_report.json`; `ctx.record("pick", ...)`
- **Depends on:** SEIS-04, SEIS-05
- **Accept:** per-station pick counts printed; any station with zero picks has a stated reason; picks within 1 s of a gap edge are dropped and counted; rerun with the same config gives identical output. Then `make publish-run RUN=<runId>` and give H2 the runId.

### SEIS-07 · P1 · Start ~12:00 AM — STA/LTA baseline

- **Goal:** classical picks with `recursive_sta_lta` + `trigger_onset` on the same preprocessed traces, in the `Pick` schema, with a threshold sweep that maximizes the baseline's own Tier A count once H2's pipeline exists.
- **Files:** `hq/baseline/__init__.py`, `hq/baseline/run.py`, `hq/baseline/score.py`, `tests/signal/test_baseline.py`
- **In → out:** cache → `runs/<id>/picks_stalta.parquet`, `baseline_sweep.parquet` (an H1-local table, not a docs/02 model: its attrs label is `BaselineSweep`, and listing it in docs/02 is a request to H4 proposed in the SEIS-08 PR; scores null where a point was not scored); `baseline_reference.json`, written by a scored CLI rerun only
- **Depends on:** SEIS-03, SEIS-05
- **Accept:** picks load through H2's `associate()` unchanged; the sweep is saved; the chosen thresholds are in `signal.yaml`.
- **As built: scoring.** `hq run` never scores: stage `baseline` runs before `tier`, so the run's tier bars don't exist yet (`sweep.scoreMode: none`, pinned by a test). Scoring is a CLI rerun (`python -m hq.baseline.run --score-mode coordinate`) on a tiered run, through VAL-01's `rerun_tables` path against the run's own bars (`ProcessingRun.tiering`). The PhaseNet `picks.parquet` is scored first, alone, as the reference row. Coordinate mode stops at a coordinate-wise local optimum, so its best Tier A is a lower bound on the grid maximum. When every scored point has the same Tier A count the objective is flat and no best point is named: that means "not tuned by the sweep", not "tuned". `--score-only` scores without rewriting `picks_stalta.parquet` (which must already hold the chosen thresholds' picks); `--keep-scores` rewrites the picks at a new `baseline.chosen` and keeps the earlier scoring, after checking that it describes the same pick sets. `baseline_reference.json` holds the reference row, the best point (or null), the chosen-at-scoring row, the search path, per-point diagnostics and the rerun notes; `run.json` → `picker.baseline.sweepScoring` holds the same without the per-point list.
- **Scale (REQ-H1-5, decided Sat).** H2 chose option (a): validation reruns locate with the showcase run's own `statics.parquet` (`locate(..., statics=)`, H2 PR #94), and H4 wires it into VAL-01. `hq.baseline.score` must pass the table the same way before a scoring run counts as scale (a). Until then it locates without statics: under `statics.mode: referenceEvents`, `hq.locate.locate` has no match pass and applies none (its module docstring, and the flag in the SEIS-08 PR). Scale (b) is superseded: `baseline_tuning_b.json` in the showcase run is historical and read by no stage. The lead sets `baseline.chosen` from the scale-(a) rescoring; SEIS-08 implements no scale-(b) bars option.

### SEIS-08 · P1 · 2–6 PM Saturday — Lane hardening pass

- **Goal:** a fresh reviewer agent audits the whole lane against "Definition of done" below and fixes what it finds.
- **Accept:** every item in the checklist is true, with evidence pasted in the PR.

### SEIS-09 · P1 · Saturday night — Sonification (issue #102)

- **Goal:** sped-up audio renderings of the cached public waveforms for the web app: the busiest clock hour of candidate events at the borehole station with the most picks in it, and optionally a short window around the hero event at the nearest borehole station with data. Vertical channel only.
- **Files:** `hq/preprocess/sonify.py` (pure functions + CLI), `hq/config/signal.py` (`SonifyConfig`), `configs/showcase/signal.yaml` (`sonify:`), `tests/signal/test_sonify.py`
- **In → out:** the run's `stations.parquet` and `picks.parquet`, the committed bundle's `events.json` and `meta.json`, and the waveform cache (all read only) → `<fileStem>.ogg`, `.mp3` and `.json` (manifest) in an output folder. H1 never writes `apps/`: the lead hands the files to H3 for `apps/web/public/audio/`.
- **Run:** `uv run --with soundfile==0.14.0 python -m hq.preprocess.sonify --run-dir <run> --bundle-dir <bundle> --config-dir configs/showcase --cache-dir <cache> --out-dir <dir> [--clip hour|hero|all]`. python-soundfile (libsndfile, which writes OGG Vorbis and MP3) is not a project dependency; without it the CLI stops at once and names this command. The pin is the encoder that produced the delivered bytes: another version still encodes, with a logged warning, and the manifest's `encoder` records the soundfile and libsndfile versions actually used.
- **Selection:** busiest bin of `sonify.busiestHour.binS` seconds on epoch multiples (UTC clock hours) by candidate events, ties to the earliest; then the `usedInRun` station of `sonify.stationKind` with the most picks of any phase in that bin, ties in id order. Hero: `meta.json` `scene.heroEventId`, origin time from `events.json`; the nearest such station by epicentral ENU distance (ties in id order) whose source Nyquist is above the hero band's upper corner (a station sampled too slowly to carry the band is skipped and logged) and whose vertical channel covers at least `sonify.hero.minCoverageFraction` of the window. Both rules count clock hours and raw `picks.parquet` rows (associated or not), as the ticket asked.
- **As built:** each gap-separated segment is detrended, tapered, zero-phase bandpassed (`bandHz`, real Hz) and resampled with `resample_poly` to `audioRateHz / speed`, keeping only output samples at or before the segment's last real sample, then placed at its true time on a zero timeline, so gaps and missing data at the window ends are exact digital silence (never interpolated or bridged). Level: a percentile of `|x|` over samples with data. Compression: a soft-knee compressor on a peak envelope, the peak of `|x|` over the next `lookaheadMs`, released exponentially (`releaseMs`), then averaged over the past `lookaheadMs`. The envelope never falls below `|x|`, and the gain can start to fall at most `lookaheadMs` before a loud onset, so the noise just before an arrival and an emergent arrival's first motion keep their level (the first, centred envelope turned them down by tens of dB). Small events come up relative to large ones. Raised-cosine fades at every segment edge; peak normalization to `peakDbfs`. The render raises if a sample passes the peak target, a no-data sample is not exactly zero, or the duration differs from `window / speed` by more than one audio sample. Encoding: the Ogg stream serial is fixed (`oggStreamSerial`; libsndfile draws a random one), MP3 frame headers are checked for the configured constant bitrate, both files are decoded back and must have exactly the rendered length and a true peak (`truePeakOversample`x oversampled) at or below `maxTruePeakDbtp`, and each must stay under `maxBytes`. The manifest carries `stationId`, `channel`, `startUtc`, `endUtc`, `speed`, `sampleRateHz`, `filterHz`, `source: "EarthScope public waveforms"`, then `filterDomain` and `audioBandHz` (`filterHz` is ground-motion Hz; it plays at `filterHz x speed`), `seedId`, provenance (run, selection rule and its numbers, processing, file sizes and hashes) and `encoder`.
- **Not a stage:** it changes no station, pick, gap or bundle file, and no stage records the `sonify` block (inventory, download, pick and baseline record only their own sub-sections). Copy: a sped-up rendering of recorded ground motion, never "the sound of" an event.
- **Accept:** `tests/signal/test_sonify.py`: selection, rendering, envelope, validators, and the path from a synthetic run dir, bundle and cache through `load_inputs` and both selections. The encoding round trip and the CLI end to end run only under `uv run --with soundfile==0.14.0 pytest`, otherwise they are skipped. Both clips rendered from the frozen run with their manifests.
- **Hero clip length:** `(preS + postS) / speed` seconds of audio, under a second with the window and speed the ticket asked for. A longer clip is a config change only: lower `sonify.hero.render.speed` and keep `bandHz[1]` below `audioRateHz / speed / 2`.

### SEIS-10 · P1 · Saturday night — "A day at one station" helicorder

- **Goal:** a drum plot (helicorder) of one borehole station's vertical channel over the run's UTC day, with candidate events and public-catalog events marked at their origin times: the WEB-10 station-day panel and a gallery image.
- **Files:** `hq/preprocess/helicorder.py` (pure functions + CLI), `hq/config/signal.py` (`HelicorderConfig`), `configs/showcase/signal.yaml` (`helicorder:`), `tests/signal/test_helicorder.py`
- **In → out:** the run's `stations.parquet`, `picks.parquet` and `run.json`, the committed bundle's `events.json`, `catalog.json` and `meta.json`, and the waveform cache (all read only) → `station-day.png` (exactly `layout.widthPx` wide) and `station-day.json` (the station-day asset contract) in `apps/web/public/helicorder/` (writing there authorised by the lead for this ticket).
- **Run:** `cd services/seismic && uv run python -m hq.preprocess.helicorder --run-dir <run> --cache-dir <cache> --config-dir configs/showcase --bundle-dir ../../apps/web/public/data/showcase --out-dir ../../apps/web/public/helicorder`. Matplotlib (Agg) is already in the locked env through ObsPy. About a minute: the day is read one row at a time.
- **Station choice:** among `usedInRun` stations of `helicorder.stationKind`, the one with the most `picks.parquet` picks of `pickPhases` over the day, ties in station id order; its vertical channel. The manifest's `selection` records the rule, the winner's pick count and the runners-up, so the choice is never hard-coded anywhere.
- **As built:** the day is the bundle's run window (checked against `run.json`; it must be exactly one UTC day). Each row is read with `padS` at both ends; every gap-separated segment is detrended, tapered and zero-phase bandpassed (`bandHz`) on its own, never merged with fill or interpolated across a gap. Each segment becomes a per-pixel-column min / max envelope, so a spike shorter than a pixel keeps its height, and each is drawn as its own polygon: columns with no samples stay blank. One amplitude scale for the day (a percentile of the per-column peaks maps to `refHeightRows`), clipped at `clipRows` row spacings. Markers: amber ticks with the tier's opacity for candidate events, hollow diamonds for public-catalog events, from the bundle (the site's numbers); the run tables are cross-checked and a disagreement is logged. The PNG's size is read back from its header and must match the layout, and it must stay under `layout.maxBytes`. `gapSeconds` and `coverageFraction` come from the cache headers.
- **Not a stage:** it changes no station, pick, gap or bundle file. Copy: "candidate events" and "public-catalog events"; the caption and every number in the image come from the data at render time.
- **Accept:** `tests/signal/test_helicorder.py` (synthetic): station choice, gaps left blank with segments filtered on their own, row positions, marker selection by day and tier, the manifest's fields, the PNG size, and the config rejecting unknown keys.

## Domain notes (give these to your agent)

### Data access

- ObsPy ≥ 1.5: `Client("EARTHSCOPE")`. Older ObsPy: `Client("https://service.earthscope.org/")`. EarthScope FDSNWS credentials go in env vars, never in the repo.
- Request in hour chunks. Large dataselect requests can stall for minutes before returning; use timeouts, retries with backoff, and resume from the cache.
- Channel priority for picking: broadband/short-period velocity (`HH?`, `EH?`) and borehole geophones (`DP?` or similar) first; strong-motion (`HN?`) only if nothing else exists at that site. Three components or skip.
- **Sensor depth:** The sensor depth lives on each **channel's** `depth`. Ignoring it on deep borehole sensors shifts S arrivals by a large fraction of a second and smears every depth downstream. This is suspected cause #1 of the prototype's broad depths.
- **Served rate.** StationXML can disagree with the data (an epoch's metadata rate that the service doesn't serve). SEIS-01 probes a few seconds of dataselect per chosen triplet (`stations.rateCheck`); the profile follows the served rate (`onMismatch`), and a triplet no probe got data for keeps its StationXML rate, flagged (`onNoData`).
- **`usedInRun`** comes from MUSTANG daily `percent_availability` (coverage > 0). With no measurement, `stations.availability.onMissing` decides, and `onMissingNoProbeData` marks the station unused when no rate probe got data either.
- **Shallow sensors off the DEM.** A shallow sensor whose StationXML elevation misses the DEM by more than `maxShallowMismatchM` takes the DEM as its surface (`onShallowMismatch: dem`, coordinates trusted); every such case is flagged in `inventory_report.json`.
- **Station elevation is not always the surface.** Networks disagree on what StationXML's station elevation means: for some borehole stations in this region it is already the sensor level, so `stationElev − depth` would count the depth twice. SEIS-01 therefore checks every station against the USGS 3DEP DEM (EPQS) at the sensor position and resolves the convention per station: surface (`surfaceElevM = stationElev`) or sensor level (`surfaceElevM = stationElev + depth`). Ambiguous cases fail loudly. `sensorElevM = surfaceElevM − sensorDepthM` always holds, and every decision with its numbers is in `runs/<id>/inventory_report.json`.

### Gaps and noise

- **Never zero-fill gaps.** PhaseNet fires on the step at a gap edge. Keep gaps as separate traces and drop picks within 1 s of a gap boundary.
- Keep counts for picking; PhaseNet normalizes internally. Response removal is only for display copies and magnitudes.

### Preprocessing profiles

| Profile | Applies to | Steps |
| --- | --- | --- |
| `surface-100` | 100 Hz surface channels | Detrend, taper, no filter |
| `surface-hi` | 200–250 Hz surface channels | Detrend, taper, zero-phase lowpass 40 Hz (4 corners), decimate to 100 Hz |
| `borehole-A` (default) | ~1,000 Hz geophones | Detrend, taper, zero-phase lowpass 40 Hz (4 corners), decimate ×10 to 100 Hz |
| `borehole-B` (experiment) | Same | Bandpass, then relabel the sample rate as 100 Hz so PhaseNet sees a 10× slower waveform. Convert picks back: `t = t0 + (t' − t0) / 10`. |

Why `borehole-B` is worth an hour: small events near a borehole put much of their energy above 40 Hz, and profile A throws that away. Time-stretching moves that band into PhaseNet's comfort zone, and a 30-second model window covers 3 seconds of real data, plenty for near-field S−P times. Adopt it only if it beats A on the A/B metric.

Do the resampling yourself, before the model. Don't let a library default decide what the borehole sensors are worth. Borehole horizontals are often `1`/`2` with uncertain orientation; rename to `N`/`E` on a copy before picking and log that you did.

### PhaseNet

- `seisbench.models.PhaseNet.from_pretrained(name)`; list options with `PhaseNet.list_pretrained()`. A/B at least `instance`, `stead`, `original` and `scedc`, per profile.
- `model.classify(stream, P_threshold=0.1, S_threshold=0.1)` on the pre-resampled stream. Store every pick with its probability; H2 filters later.
- **A/B metric:** on the known events, count stations with P and S picks, and the P-moveout consistency (Spearman ρ vs distance). Once H2's locator exists, add median absolute residual after location. If the public catalog includes analyst arrivals, add median pick error against them. Surface and borehole can end up on different weights.
- **As built:** `hq.pick.ab` ranks on the Check B quantities only (`known/ab.json` → `metric`). The residual-after-location and analyst-arrival terms were not added, although the public catalog carries manual arrivals, and the weights were not rerun after H2's locator landed (a lead call, listed in the SEIS-08 PR).
- One day on ~20 stations is small. Parallelize by station; a GPU is optional. Cache model weights on first load.

### STA/LTA

- Short windows suit microseismic data: STA ~0.05–0.1 s, LTA ~2–5 s at 100 Hz. P onsets from Z; S from horizontals after the P.
- Tune on/off thresholds to maximize STA/LTA's own Tier A count, on the run's own bars and statics (SEIS-07 → Scale). Giving the baseline its best shot is what makes the comparison honest.
- **Pick probability.** STA/LTA has no calibrated confidence, so every baseline pick gets `baseline.prob`, a typical PhaseNet pick prob (its source is in the `signal.yaml` comment). It must stay ≥ `seismology.associator.minPickProb` (scoring checks it), and a baseline event's `meanPickProb` is that constant: it carries no confidence.
- **Statics caveat.** The run's station statics are residuals of PhaseNet picks at the public regional catalog's hypocentres of matched events. They absorb PhaseNet's station timing bias, not the late bias of STA/LTA onsets (trigger-level crossing, and the prefilter's group delay), so scale (a) still leans toward PhaseNet. Say so beside any comparison.

## Definition of done (the hardening pass checks every line)

- [x] `stations.parquet` sensor depths match StationXML channel depths for every borehole station
  - Evidence: every row of the showcase run checked against its StationXML channel epochs (SEIS-08 PR); pinned by `test_every_row_depth_matches_its_stationxml_channels`.
- [x] No zero-filled samples anywhere; gap-edge picks dropped and counted
  - Evidence: no fill in `read_window`, `for_picking` or `split_blocks`; drops counted in `pick_report.json` and `stages.json`; `test_missing_raw_sample_off_the_model_grid_drops_gap_edge_picks` (SEIS-08 PR).
- [x] Every knob in `signal.yaml`; every pick carries `picker` with the weights name
  - Evidence: `SignalConfig` forbids unknown keys and has no Python defaults; `run.json` records the whole `picker` and `preprocess` blocks; picker names checked in all three pick tables (SEIS-08 PR). `display_copy`'s filter shape stays in code (`DISPLAY_*` in `hq/preprocess/profiles.py`), because docs/02 fixes its signature with no config and it shapes display copies only.
- [ ] `hq run` from a clean cache reproduces identical picks
  - Partly shown (numbers in the SEIS-08 PR). Reruns from the warm cache are bit-identical on the same platform and `uv.lock`. No same-platform clean-cache run has been done. The one clean-cache run, on another platform, matched every STA/LTA pick; its PhaseNet probabilities differ in the last float32 digits, and one argmax tie moved one pick by a sample. Open: a small same-platform clean-cache check after the freeze.
- [x] Runtimes and counts recorded in `run.json` for inventory, download, pick, baseline
  - Evidence: `run.json` `runtimeS` and `picker.*` params for all four stages; their counts in `stages.json` (Notes below).
- [x] `pytest -m smoke tests/signal` < 30 s, offline
  - Evidence: suite-wide in-process network guard in `tests/signal/conftest.py`, pinned by `test_offline_guard.py` (spawned workers rely on injected fakes); under 30 s on an idle machine with warm imports at the SEIS-08 head, while a cold first run took longer (numbers in the SEIS-08 PR).
- [x] `read_window`, `read_inventory`, `display_copy`, `for_picking` match the `docs/02` signatures exactly
  - Evidence: `test_library_apis_match_docs02_signatures_exactly` (names, kinds, no defaults, return types).

**Notes (SEIS-08).**

- "Identical" means bit-identical on the same platform and `uv.lock`, shown so far for reruns from a warm cache. Across platforms, float32 PhaseNet probabilities differ in the last digits and can flip an argmax tie between adjacent samples (numbers in the SEIS-08 PR).
- A run whose inventory predates `stations.availability.onMissingNoProbeData` can differ from a rerun in `usedInRun` for a station that served no data, and in that station's `gaps.parquet` rows. Picks don't change.
- Counts live in `stages.json`, because `ProcessingRun` has no counts field (`hq/runs.py`, `RunContext.record`); runtimes and params are in `run.json`.
- The known-event sub-steps are not registered stages, so neither writes `run.json`. The windows CLI writes `known/known_windows.record.json`. The A/B's `known/pick_known.record.json` comes only from `hq.pick.ab.run(ctx)`, which no stage or CLI calls, so a real run has no record of it. Runs built before commit `c43b853` (the showcase run of record is one) carry both sub-steps in `run.json` and `stages.json` instead, and have no `known/*.record.json`.
- **Known limitation:** a clean `hq run configs/showcase` does not rerun the known-event sub-steps (`known_windows` and `pick_known` aren't in the `hq.runs.STAGES` registry). The pipeline of record gets `known/` only when these are run by hand on the run after `catalog`, `download` and `pick`: `uv run python -m hq.ingest.windows --run-dir <run> --config-dir configs/showcase --cache-dir <cache>`, then `uv run python -m hq.pick.ab --run-dir <run> --config-dir configs/showcase --cache-dir <cache>`. Nothing downstream reads `known/`: it is Check B and weight-A/B evidence only.
- The inventory stage sets `ProcessingRun.stationIds` (sorted `usedInRun` ids) through `RunContext.update_run`. Runs made before SEIS-08, including `20260926-0210-a04c611`, have `stationIds: []`.
- STA/LTA thresholds (SEIS-07, final): scored on the showcase run's statics scale (REQ-H1-5 option (a): H2's `locate(statics=)`, the run's own tier bars) on the busiest 6 h block. Tier A is 0 at all 15 coordinate-search points (pOn 3–12, sOn 3–12, off 1.0–2.0), so `baseline.chosen` stays at pOn 5 / sOn 5 / off 1.5. H4's full-day validate on the same scale agrees (STA/LTA 0 Tier A).
- The smoke budget holds on an idle machine with warm imports. Under heavy load the whole suite slows, not one test, so a loaded timing says nothing about the budget; the first run after a busy spell (cold import and bytecode caches) is slower too.

## Kickoff prompt

```
You are the H1 Signal lane agent for Hidden Quakes (HackGT 13).
Before writing anything, read: CLAUDE.md, docs/00-project.md, docs/01-architecture.md,
docs/02-contracts.md, docs/03-schedule.md, and docs/lanes/H1-signal.md.
Write only inside the paths listed under "You own" in your lane doc, plus files your ticket names.
Your ticket: <TICKET-ID>. Start by restating its goal, files, inputs, outputs and acceptance test,
then write a short plan. Implement it with tests. Finish by running `make check` and the acceptance
test, and paste both outputs. If you need anything from another lane, write a request in
docs/requests/ and stop to tell me. Never change a contract.
```

Reviewer prompt (fresh session, after the builder finishes):

```
You are reviewing a Hidden Quakes H1 Signal PR. Read CLAUDE.md, docs/02-contracts.md and
docs/lanes/H1-signal.md, then review the diff on this branch against the ticket's acceptance test
and the lane's Definition of done. Check: paths stay in-lane, contracts match docs/02 exactly,
no zero-filled gaps, no magic constants, tests are offline and deterministic. List concrete findings
with file:line, most serious first. Don't rewrite the code yourself.
```

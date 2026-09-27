# H2 · Seismology lane

**Mission:** turn picks into located, tiered, matched events whose depths survive a skeptical seismologist. This lane decides whether the reveal reads as "hidden subsurface structure" or "hundreds of orange dots". Your first two hours are synthetic on purpose: no real data touches the locator until it recovers synthetic events.

## Paths

**You own:** `services/seismic/hq/associate/`, `hq/locate/`, `hq/tier/`, `hq/match/` (including catalog fetch), `hq/magnitude/`, `hq/config/seismology.py`, `hq/config/run.py`, `services/seismic/configs/showcase/run.yaml`, `configs/showcase/seismology.yaml`, `services/seismic/configs/velocity/`, `services/seismic/tests/seismology/`, `data/cache/ttgrids/`, `data/cache/velocity/`

**You never write:** H1's `hq/ingest|preprocess|pick|baseline/`, `signal.yaml`; H4's `hq/export|validate/`, `cli.py`, `runs.py`, `packages/contracts/`; anything in `apps/`.

## What you consume

| What | From | Where | Until it exists |
| --- | --- | --- | --- |
| Contracts + `hq_contracts.io` | H4 | `packages/contracts/` | code against the spec in `docs/02` |
| `RunContext` / `hq run` | H4 | `hq/runs.py`, `hq/cli.py` | a local stand-in in your tests |
| Stations with real sensor depths | H1 | `runs/<id>/stations.parquet` | synthetic geometry (≥ 10 stations, 2–3 at depth) in tests |
| Known-event picks | H1 | `runs/<id>/known/picks.parquet` | synthetic picks from your own forward model |
| Full picks | H1 | `runs/<id>/picks.parquet` | known-event picks |
| `read_window`, `read_inventory` | H1 | `hq.ingest.cache` | synthetic traces (magnitude and LOC-08 only) |

## What you produce

| What | For | Where | Due |
| --- | --- | --- | --- |
| Window, bbox, origin, `refSurfaceElevM` | everyone | `configs/showcase/run.yaml` | 8:05 PM |
| Public catalog | H1, H4 | `runs/<id>/catalog.parquet` | 8:25 PM |
| Synthetic test report | H4 (validation panel) | `runs/<id>/synthetic.json` | 11:30 PM |
| `associate`, `locate`, `match`, `assign_tiers` | H4 (validation reruns) | `docs/02` §5 signatures | 2:00 AM |
| Final events + arrivals + event picks | H4 | `events.parquet`, `arrivals.parquet`, `event_picks.parquet` | 3:45 AM |
| Association sweep, match sensitivity | H4 | `sweep.parquet`, `match_sensitivity.parquet` | 3:45 AM |
| Magnitude calibration (P1) | H4 | `magnitude.json`, magnitudes in `events.parquet` | 2:00 PM Sat |

## Tickets, in order

### LOC-00 · P0 · 8:00–8:05 PM — Run section

- **Goal:** decide and write the showcase window (explicit UTC start and end), bbox, origin and `refSurfaceElevM`. Five minutes, no debate. Define `RunSection` in `hq/config/run.py`.
- **Files:** `configs/showcase/run.yaml`, `hq/config/run.py`
- **Accept:** `load_config()` parses it once H4's loader exists; the window matches the day the research counted. Push these two files **straight to `main`** by 8:05 PM (the one direct push this lane makes) so every lane can merge them.

### MATCH-01 · P0 · Start 8:05 PM — Public catalog

- **Goal:** fetch the public regional catalog for the exact window and bbox, e.g. UUSS events through the USGS ComCat FDSN event service (ObsPy `Client("USGS")`). Build `CatalogEvent` rows, convert depth to `elevM` using the catalog's stated datum, and check whether analyst arrivals are available.
- **Files:** `hq/match/catalog.py` (stage `run(ctx)` for `catalog`), `tests/seismology/test_catalog.py`
- **In → out:** `run.yaml` → `runs/<id>/catalog.parquet`, `catalog.quakeml`
- **Accept:** count printed with its window. The research phase counted 43 for this day; if the count differs, explain why (usually the day boundary or bbox) before moving on. The product never hard-codes 43. Raw QuakeML saved; `depthDatum` filled.

### LOC-01 · P0 · Start 8:05 PM — Velocity models

- **Goal:** build the 1D layer file from the GDR FORGE 1D model; start the 1.08 GB 3D NetCDF download; write the datum convention into the layer file and `SourceRef`.
- **Files:** `configs/velocity/forge_1d.csv`, `hq/locate/velocity.py`, `tests/seismology/test_velocity.py`
- **In → out:** GDR downloads → layer file + `SourceRef` in `data/cache/velocity/`
- **Accept:** Vp and Vs plotted against `elevM` with the datum stated; `SourceRef` filled with citation and URL.

### LOC-02 · P0 · Start ~8:30 PM — Travel-time tables + locator + synthetic test

- **Goal:** per-station 2D (r, z) eikonal tables via `skfmm` (reciprocity, true receiver elevation), the grid-search locator (200 m coarse → 25 m fine, L1, origin time removed analytically, PDF uncertainty), and the 200-event synthetic recovery test.
- **Files:** `hq/locate/tt_grid.py`, `hq/locate/locator.py`, `hq/locate/uncertainty.py`, `tests/seismology/test_tt_grid.py`, `tests/seismology/test_locator.py`
- **In → out:** layer file + station geometry → tables in `data/cache/ttgrids/`, locator, `synthetic.json`
- **Depends on:** LOC-01; real geometry from SEIS-01 when ready
- **Accept:** homogeneous half-space error < 5 ms vs analytic; two-layer head-wave test passes; synthetic test runs in < 2 min and reports median and p90 horizontal and vertical errors; noise-free depth bias within ±50 m.

### LOC-03 · P0 · Start ~10:00 PM (known events), ~12:30 AM (full) — Association

- **Goal:** PyOcto association fed the same 1D model; known-event windows first, then the full window, then the parameter sweep.
- **Files:** `hq/associate/__init__.py` (`associate()`), `hq/associate/run.py`, `tests/seismology/test_associate.py`
- **In → out:** picks + stations → `assoc_events.parquet`, `assoc_picks.parquet`, `sweep.parquet`
- **Depends on:** SEIS-04 (known), SEIS-06 (full)
- **Accept:** 3 / 3 known events associated with ≥ 8 stations; sweep saved; every PyOcto argument recorded via `ctx.record`. Get H1's runs with `make fetch-run RUN=<runId>`.

### LOC-04 · P0 · Start ~10:30 PM (known events) — Location + diagnostics

- **Goal:** locate every associated event, run the outlier pass, write arrivals, and fill diagnostics 1–5 below with real results.
- **Files:** `hq/locate/__init__.py` (`locate()`), `hq/locate/run.py`, `hq/locate/diagnostics.py`
- **In → out:** association → `events_located.parquet`, `arrivals.parquet`, `diagnostics.md` in the run dir
- **Depends on:** LOC-02, LOC-03
- **Accept:** every diagnostics row has a result and a conclusion; known events relocate within their catalog uncertainty.
- **Status (LOC-09):** the second half is not met on the showcase run `20260926-0210-a04c611`. With reference statics the known events sit within the catalog's stated depth error for only some of them, and outside its stated horizontal error for all of them; `diagnostics.md` row 2 now reports both halves (horizontal offsets next to the catalog's stated horizontal error). Over all matched events the offsets are about the size of the catalog's own stated errors, and the reference statics tie positions to the catalog's frame, so this reads as the limit of a 1D model plus terms, not a locator defect. Recorded here for the lead; no change to the locations.

### MATCH-02 · P0 · Start ~2:00 AM — Catalog matching

- **Goal:** one-to-one matching with `linear_sum_assignment`, sensitivity at three tolerance pairs, and a reason for every unmatched public event.
- **Files:** `hq/match/__init__.py` (`match()`), `hq/match/run.py`, `tests/seismology/test_match.py`
- **In → out:** located events + catalog → `matches.parquet`, `match_sensitivity.parquet`
- **Depends on:** LOC-04, MATCH-01
- **Accept:** X / N plus the unmatched list; sensitivity table; one-to-one guaranteed by test.

### LOC-05 · P0 · Start ~2:00 AM — Station statics

- **Goal:** three iterations of per-station, per-phase statics inside `locate()`, capped at 0.3 s.
- **Lead decision (during LOC-05):** both methods ship behind `statics.mode`. `selfConsistent` is the method above (cap `statics.capS`). `referenceEvents`, the showcase default, takes each term at the public regional catalog's hypocentres of the matched events, relocates every matched event with terms computed without it, and caps at `statics.referenceCapS` (chosen from the data; `diagnostics.md` justifies it per run). Its stage order is locate (pass 1) → match → locate (pass 2) → match → tier (Locator step 5).
- **Files:** `hq/locate/statics.py`
- **In → out:** residuals → `statics.parquet`, relocated events
- **Depends on:** LOC-04
- **Accept:** median rms drops; every static above 0.15 s has a written explanation in `diagnostics.md`.

### LOC-06 · P0 · Start ~3:15 AM — Quality tiers

- **Goal:** tiers from matched-event quantiles (method below), with `tierReasons` on every event; write the final `events.parquet` and `event_picks.parquet`.
- **Files:** `hq/tier/__init__.py` (`assign_tiers()`), `hq/tier/run.py`, `tests/seismology/test_tier.py`
- **In → out:** located events + matches → `events.parquet`, `event_picks.parquet`, `ProcessingRun.tiering`
- **Depends on:** LOC-05, MATCH-02
- **Accept:** thresholds and their source quantiles stored; tier counts printed; every matched event gets a tier; `depthKm` and `revealOrder = -1` set. Then `make publish-run RUN=<runId>` and give H4 the runId.

### LOC-07 · P1 · After Gate S, if depths look biased — 3D travel-time grids

- **Goal:** 3D grids from the GDR Cape/FORGE model resampled to 100–200 m, through the same locator (`method: "grid3d"`).
- **Files:** `hq/locate/tt_grid3d.py`, `tests/seismology/test_tt_grid3d.py`
- **Depends on:** LOC-02, the 3D download
- **Accept:** synthetic test passes with 3D grids; a 1D vs 3D residual and depth-shift comparison is saved in `diagnostics.md`.
- **Lead decision (Sat 1:35 PM, after LOC-07 merged):** keep `grid1d` with `referenceEvents` statics for the showcase; `grid3d` stays a cross-check.
- **Status (LOC-07, after review):** `locator.method: grid3d` runs the same locator on per-station 3D tables (`hq/locate/tt_grid3d.py`: the model's CRS, vertical datum and air values confirmed from the paper and the file, evidence in its docstring and in `diagnostics.md`). `grid1d` stays the default. Receivers in the model's one-value columns (no ground surface or basin data in the file) use their 1D tables unless `grid3d.constantColumns` is `asFile`. Every grid3d stage run also locates the same association with grid1d and the same statics, and `diagnostics.md` compares the two. On the showcase run's acceptance (`_runners/LOC-07_acceptance.out`), with `referenceEvents` statics grid3d and grid1d locate the events within the held-out scatter of each other, and grid3d leaves more unexplained statics. Without statics neither 3D variant removes the epicentre offset from the public regional catalog. With `asFile`, depths move far above the catalog's; the review traced that to the one-value columns under the Mineral Mountains outcrop stations (basement velocity right up to the sensor), not to the basin model. With those stations on 1D tables, depths sit somewhat deeper than the catalog's and the residuals grow. The basin model does remove most of the S-heavy azimuthal trend at the catalog's hypocentres (row 7). Recommendation: keep grid1d + statics for the showcase and use grid3d as a cross-check (lead call). Before any switch: grid3d has no above-ground mask (the report counts such events), its synthetic test has no table error (its forward model is the locator's own tables), and 100 m tables are the safer spacing.

### LOC-08 · P1 · Only if the depth gate fails at 4 AM — Relative relocation

- **Goal:** cross-correlation differential times on Tier A events (waveforms via `read_window`) plus GrowClust or HypoDD; `method: "relative"`.
- **Files:** `hq/locate/relative.py`
- **Depends on:** LOC-06, H1's `read_window`
- **Accept:** relocated Tier A events with before/after spread in `diagnostics.md`. Start only with 6+ hours before the pipeline freeze.

### MAG-01 · P1 · After Gate S — Magnitude

- **Goal:** amplitude–distance magnitude regression on matched events, leave-one-out MAE.
- **Files:** `hq/magnitude/__init__.py`, `hq/magnitude/run.py`, `tests/seismology/test_magnitude.py`
- **In → out:** matched events + response-removed waveforms → `magnitude.json`, magnitudes in `events.parquet`
- **Depends on:** MATCH-02, H1's `read_window` + `read_inventory`
- **Accept:** LOO MAE reported; magnitudes written to events only if MAE ≤ 0.4.

### LOC-09 · P1 · 2–6 PM Saturday — Lane hardening pass

- **Goal:** a fresh reviewer agent audits the lane against "Definition of done" and fixes what it finds.
- **Status (LOC-09 hardening, agent/LOC-09-hardening):** four audit lenses (physics, contracts, tests, checklist) ran on the lane and on run `20260926-0210-a04c611` v3. Fixed without changing any published value (on a rerun `catalog.parquet` and `matches.parquet` change string dtypes only): CI smoke now runs the locate stage (its synthetic test, `synthetic.json`, every diagnostics row), docs/02 `locate()` on STA/LTA-labelled picks, noise-free recovery, the one-to-one property and a docs/02 §5 signature test; stage tier fails without `locate_flags.parquet` and removes a stale `magnitude.json`; stage associate removes a stale `matches.parquet`; `diagnostics.md` wording (row 7 hedged after statics, both halves of the known-event check, sigma and origin-time notes); unmatched-reason windows move by the station statics; `run.json` records every table and velocity knob, each H2 stage's code and software, and no stale keys after a method switch; `catalog.parquet` and `matches.parquet` follow the docs/02 §2 dtype rule. Open lead decisions (each would change published numbers or needs the lead): association without station terms, synthetic station coverage, pick sigma (Locator step 2), `event_picks.parquet` scope, the locator vs association volume, and the sweep config below.
- **Sweep config (lead decision):** the published v3 `sweep.parquet` came from an out-of-repo overlay with `tiering.sweep.enabled: true`; the committed `seismology.yaml` has it false, and stage tier removes `sweep.parquet` when it is disabled. A pipeline-of-record rerun from the committed config therefore ships an empty `Validation.sweep`. Either enable it in the committed config (much longer tier runs) or hand H4 an override for the final rerun. That override has to be made from the committed `configs/showcase` with only this flag flipped: the overlay that produced v3 predates later changes to `signal.yaml`, `validate.yaml` and `seismology.yaml` and no longer loads. The configured point was kept without a recorded knee choice (Association: "pick the knee"); the lead records which point and why.

### LOC-10 · P1 · Saturday evening — Harvest unassociated picks at predicted arrivals (issue #96)

- **Decision (lead, 2026-09-26 18:08 EDT):** not adopted for the showcase run: 807 real picks recovered but locations and held-out catalog offsets essentially unchanged and the strict count would fall from 32 to 29 under the run's own bars; kept behind `harvest.enabled` for future runs. No v4: run `20260926-0210-a04c611` (v3) stays the run of record, and the code merges into `feat/location` with `harvest.enabled: false` in `configs/showcase/seismology.yaml`.
- **Why:** PyOcto associates without station terms (tolerance 0.3 s) while the reference terms reach 0.9 s (FSB5 S, FORW S, MHS2 S, FSB6 S), so real arrivals at those station-phases stay in no association event. The LOC-09 audit counted 828 free picks within ±0.15 s of the canonical run's predicted arrivals at station-phases with no pick.
- **What (option b, lead's choice):** `hq.locate.harvest`, behind `harvest.enabled` in `seismology.yaml` (**off** in the showcase config, so the run of record reproduces unchanged). When on, every statics-corrected locate of the whole association (stage locate pass 2 with each reference event's held-out terms, selfConsistent's last iteration, `locate(statics=...)` as the validation reruns call it, the tier sweep, and the grid1d comparison of a grid3d run) adds free picks (in no association event, `prob >= harvest.minProb`, same phase label) within `harvest.windowS` (P 0.10 s, S 0.15 s) of an event's `tPred` at a station-phase it has no pick for, and relocates the events that gained picks once with the same statics and outlier pass. Skipped and counted: picks within the window of two or more events' `tPred` (filled slots included; every such free pick is counted), slots with two or more remaining candidates, and the slots with a candidate in events with depthOnEdge, MAP on a volume face or a truncated PDF. Never: replacing or duplicating an event's own pick (outlier-dropped included), iterating, or estimating station terms from harvested picks (`statics.parquet` is the same with it on or off). Option (a), station terms inside PyOcto, would change the event set and the H4 runner; not pursued.
- **Config:** the `harvest` section is required, like every other section, so a `seismology.yaml` overlay copied from before LOC-10 no longer loads until it gains the block. Only when `enabled` is true, `windowS` must not exceed `locator.outlier.floorS` and `minProb` must not be below `associator.minPickProb` or any `associator.sweep.minPickProb`.
- **Outputs (contracts unchanged):** harvested picks appear as extra `pickIds`, filled `arrivals.parquet` rows and more `event_picks.parquet` rows; recover them as an event's `pickIds` minus its association's picks (join on `assocId` through `locate_flags.parquet`: event ids can renumber). Only when it ran: `run.json` `locator.harvest` (config, counts, per-phase counts, analytic and shifted-window chance, pre-harvest synthetic pick stats, every harvested pick), counts `picksHarvested` etc., `statics.reference` `afterNoHarvest*` offsets and `crossValidatedOffsets.afterNoHarvest`, and a `diagnostics.md` "Pick harvest (LOC-10)" section. `check_same_association` ignores located picks in no association event (so a pass 2 rerun on a harvested run dir works) but fails when none of a reference event's located picks is in an association event.
- **Caveats when on:** harvested picks are chosen because they agree with the current location, so pick counts, gap, rmsS, formal errors and the tiers and hero built on them are not independent evidence of a better location; the held-out reference offsets are. Also computed with harvested picks: the synthetic `sKeepProb`/`pickProb` when null in the config (pre-harvest values in `locator.harvest.pickStatsBeforeHarvest`), the robust pick sigma and selfConsistent history in the statics section (harvested residuals are cut at ±`windowS`, so that sigma reads low), and `nEvents` of the statics table `locate(statics=...)` returns (`statics.parquet` itself is not). The windows are centred on `tPred` with the station terms as estimated, so a biased term carries its bias into every pick it harvests. With the flag on, H4's validation notes should say that the reruns harvest.
- **Flag off (what merges):** a runner replay of run `20260926-0210-a04c611` (associate -> locate -> match -> locate -> match -> tier -> magnitude, committed `configs/showcase` with `nWorkers`/`nThreads` 4; recipe `_runners/LOC-10/replay.py`) on LOC-10 code 2a6a21f gives every H2 table, `synthetic.json`, `magnitude.json` and `diagnostics.md` byte-identical to the same replay of 360fa7d (the code before LOC-10); `run.json` differs only in runtimes and provenance. Both replays match the canonical copy by sha256 on every H2 table except `catalog.parquet` / `matches.parquet` (string dtypes, values equal, as LOC-09 found) and `sweep.parquet` (sweep off in the committed config). Later LOC-10 commits change no flag-off output: the one change on that path makes `check_same_association` raise again when none of a reference event's located picks is associated (360fa7d raised there too), and the harvest config checks run only when it is on.
- **Measured A/B (flag on vs v3, for the record):** a replay with `harvest.enabled: true`, same recipe; numbers from `_runners/LOC-10/eval/ab_metrics_implOn.{out,json}` and `_runners/LOC-10/ab-report.md`. Pairing is on `assocId` or `catalogId`.

  | metric | v3 (harvest off) | harvest on |
  | --- | --- | --- |
  | candidate events | 654 | 654, same assocIds |
  | recall | 43/43 | 43/43 at every tolerance |
  | picks harvested | 0 | 807 (P 149, S 658), all used after the relocation |
  | events gaining picks | – | 425 (v3 tiers A 32, B 167, C 226; all 43 matched) |
  | top station-phases (pre-relocation median offset) | – | FORW S 247 (+0.015 s), FOR1 S 176 (+0.090 s, term from 5 events), FSB6 S 132 (+0.022 s), FOR8 P 90 (-0.010 s) |
  | chance | – | analytic 2.4; shifted-window control 11 / 10 / 22 / 2 at -1 / -0.6 / +0.6 / +1 s; independent control at ±0.5 to ±2 s mean 6.6 |
  | skipped | – | 3 ambiguous picks, 14 untrusted slots with a candidate, 0 multi-candidate slots |
  | relocation of the gaining events | – | median 25 m horizontal and 25 m abs(dz); 152 did not move; no other event moved |
  | held-out reference offsets, median / p90 horizontal (m) | 309.8 / 614.5 | 303.0 / 606.8 (closer 12, farther 6, same 25; sign test p 0.24) |
  | held-out reference offsets, median / p90 abs(dz) (m) | 155 / 535 | 155 / 540 (closer 3, farther 14, same 26; sign test p 0.013) |
  | v3's 32 Tier A events, median hErrM / vErrM (m) | 53.6 / 66.0 | 51.5 / 65.2 |
  | all events median vErrM (m) | 172.1 | 169.5 |
  | v3's 32 Tier A events, median rmsS (s) | 0.0195 | 0.0264 (associated picks only: 0.0199) |
  | tiers, bars re-derived by the run | A 32 / B 171 / C 451 | A 29 / B 166 / C 459 |
  | tiers, v3 bars applied | A 32 / B 171 / C 451 | A 46 / B 166 / C 442 (15 of the 21 promotions to A had failed only count or gap bars; not a quality gain) |
  | hero (Tier A, most stations) | assoc-000301, 23 stations | assoc-000269, 24 stations (own bars; assoc-000267 under v3 bars) |
  | `statics.parquet` | – | byte-identical |
  | synthetic median v (m) | 59.2 | 61.2 (`sKeepProb` measured after the harvest, 0.778 -> 0.846) |
  | ML_cal LOO MAE | 0.1087 | 0.1086 |

  Associated FOR1 S picks already sit late in the harvest-on run (median residual +0.071 s over 291 picks), and its 176 harvested picks sit at +0.087 s after the relocation; the held-out depth offsets move deeper for 14 events and shallower for 3. The evaluator (`ab-report.md`) recommended not adopting it for a v4 for these reasons.
- **Before revisiting (from the evaluation and the three review lenses):** refit or gate the weak terms first (harvest only at station-phases whose term has enough events and whose harvested-offset median is near 0; FOR1 S's 5-event term is the clear case), pin the synthetic pick stats to the pre-harvest values, compute the pick sigma from associated picks only, have H4's validation notes state that the reruns harvest, and judge on held-out depth and horizontal offsets rather than counts, tiers or the hero. Not measured for this record: the validation null test and baseline with harvest on (on this branch the reruns call `locate(statics=...)`, so they would harvest).

### ML-01 · P1 · Saturday evening — Chance-association classifier (issue #100)

- **What it is:** a logistic regression trained this weekend on this run's own pipeline output. Positives: the 654 candidate events the pipeline built from the real picks. Negatives: 1880 decoy events the same pipeline built after shifting each station's picks by one uniform ±30 s draw (the null test's 20 published shuffles, `hq.validate.null_test`). It scores each candidate for how much its **station count, arrival-time fit (rms) and mean pick probability** look like an association on real pick timing rather than a scrambled-clock decoy. It is not a probability that an event is an earthquake.
- **Files:** `hq/tier/confidence_data.py` (training data: real rerun + decoys through one `rerun_pipeline` path, same statics and tier bars), `hq/tier/confidence.py` (models, folds, metrics, `confidence.json`), `tests/seismology/test_confidence_data.py`, `tests/seismology/test_confidence.py`.
- **Model choice:** three candidates, simplest first: logistic regression on 3 features (`quality_nStations`, `quality_rmsS`, `meanPickProb`), logistic regression on 34 features (37 minus 3 exact duplicates), and a 16-unit torch MLP on the 34. A larger one replaces the simpler only if it beats it by more than 0.01 on both the fold-mean ROC AUC and the equal-station-count AUC; neither did, so the 3-feature model ships. The 3-feature set was found by the independent review on these same folds, so it is not a pre-registered choice. Weights per standard deviation (full-data fit): station count +2.70, rms −1.18, mean pick probability −1.14.
- **Held-out numbers (run `20260926-0210-a04c611`):** 5 folds, seed 0; decoys split by whole shuffle (4 per fold), real events at random; every event is scored by the fold model that never saw it, and the imputer/standardizer are fitted on training rows only. Fold mean (min–max); "equal station count" is ROC AUC over real/decoy pairs with the same number of stations (63,339 pairs), where station count alone scores exactly 0.5.

  | Scorer | ROC AUC | Equal station count |
  |---|---|---|
  | **3-feature logistic (shipped)** | **0.990 (0.986–0.994)** | **0.956 (0.932–0.980)** |
  | 34-feature logistic | 0.992 (0.986–0.997) | 0.949 (0.901–0.981) |
  | 34-feature MLP | 0.992 (0.987–0.997) | 0.948 (0.907–0.977) |
  | 3-feature without pick probability (station count + rms) | 0.977 | 0.932 |
  | Station count alone (no training) | 0.948 | 0.5 |
  | Arrival-time rms alone (no training) | 0.843 | 0.932 |

  Shipped model, pooled out of fold: ROC AUC 0.990, AP 0.975, equal station count 0.961, equal used-pick count 0.949. By station count (pooled): 3 stations 0.89 (3 real), 4: 0.92 (8 real), 5: 0.96, 6: 0.98, 7: 0.98, 8: 1.00 (25 real vs 8 decoys). No decoy reached Tier B.
- **Caveats (say them with the numbers):**
  - Size carries most of the headline: station count alone reaches 0.948. The equal-station-count AUC is the number that shows learning beyond size, and most of it is timing coherence (rms alone 0.932), which is exactly what the scramble destroys.
  - Mean pick probability gets a **negative** weight because decoys are re-assembled strong picks from real events (the review found that scrambling only the picks left over after the real events gave no decoys in 5 shuffles). That is an artifact of how decoys are made, not physics; without it the model is station count + rms (0.977 / 0.932).
  - **Decoy support ends at 8 stations** (the largest station count with at least 5 decoys; 1871 of 1880 decoys have 7 or fewer, one has 10). 457 of the 654 candidates have 9 or more stations; their scores near 1 extrapolate the station-count weight and are not a comparison with decoys. The 197 candidates within support are all Tier C.
  - Single small-event scores are model-dependent: within the support, the 3-feature and 34-feature scores rank-correlate at only 0.55 although both models have similar AUCs. Do not quote counts of candidates below a decoy threshold as a finding.
  - Label 1 means "associated on real pick timing", not "real": the real set holds some chance associations itself (the null test's mean per shuffle is probably an upper bound, since decoys form mainly from recycled real-event picks).
  - The 43 public-catalog-matched events all have 15 or more stations, above the decoy support, so their scores near 1 are trivially met and are not evidence for the model.
- **UI:** label "Scramble test"; show the score only for events with at most `model.decoySupport.maxStations` stations, and "larger than any decoy" otherwise; tooltip is the file's `description`. Validation card: held-out ROC AUC quoted next to the station-count baseline.
- **Run:** from `services/seismic`, two steps. Step 1 rebuilds the training data (about 46 min with 6 workers; `--max-extra 0` keeps exactly the 20 published shuffles); step 2 trains, evaluates and writes the file (about 2 s, deterministic: two reruns give byte-identical `scores.parquet`).

  ```
  nice -n 5 uv run python -m hq.tier.confidence_data --run-dir <runs>/20260926-0210-a04c611 \
      --config-dir configs/showcase --out-dir <scratch>/data --workers 6 --max-extra 0
  uv run python -m hq.tier.confidence --run 20260926-0210-a04c611 \
      --data-dir <scratch>/data --out <scratch>/confidence.json
  ```

  Step 2 writes `confidence.json` (`hq.confidence/1`: `model` with the held-out numbers, baselines and decoy support; `label`; `description`; `events` = eventId → score, 3 decimals, all 654 canonical ids) and, beside it, `scores.parquet` (out-of-fold scores of real and decoy events for every candidate model) and `report.json` (every metric, per-fold summaries, thresholds, weights, permutation importance). It is copied into `runs/<runId>/confidence.json` only after the exporter change (branch `agent/ML-01-ui`) is merged; that is the lead's call.

## Domain notes (give these to your agent)

### Velocity models

- **1D (default):** the GDR DAS catalog for the 2023 16A/16B circulation test ships the 1D P/S model used for its locations, derived by joint hypocenter/velocity inversion: [GDR 1613](https://gdr.openei.org/submissions/1613).
- **3D (upgrade):** [GDR 1800](https://gdr.openei.org/submissions/1800), Cape EGS and Utah FORGE empirical 3D model (Nakata et al., LBNL, 2025, CC BY 4.0). It covers 30 × 30 km to 10 km depth with 3D topography and a sediment/basement contact. Basin velocities come from a log fit to borehole logs; basement is constant Vp 5.8 km/s, Vs 3.392 km/s. It's a 1.08 GB NetCDF on a 50 m grid; read it with xarray and resample to 100–200 m for eikonal solves.
- Published 2024 stimulation work found Vp/Vs ≈ 1.86 inside the earthquake cluster ([GDR 1723](https://gdr.openei.org/submissions/1723)), above the model's basement ratio. Prefer a per-phase S static over editing the model.
- Published FORGE surface-network work notes that a 1D model biases locations because it ignores the dipping basement contact, and sharpened locations with relative relocation ([Niemz et al.](https://www.sciencedirect.com/science/article/pii/S0375650524000373)).

### Travel-time grids

- **1D path:** for each station and phase, solve a 2D (r, z) eikonal with `skfmm.travel_time`, source at the sensor (reciprocity). For a laterally uniform model, first arrivals depend only on (r, z), so this is exact up to grid error. At 25 m spacing out to 40 km, that's under a million cells per table.
- **3D path:** the same on the resampled 3D model, cached as `.npy` under `data/cache/ttgrids/`. Stations outside the 3D model fall back to 1D tables.
- Receivers sit at their true `sensorElevM`. Extend the top layer up to the highest station. Only allow hypocenters below the local surface.

### Locator

1. Coarse grid search at 200 m over the volume, then a 25 m grid within ±1 km of the best node.
2. L1 misfit with origin time removed analytically: `t0` = weighted median of `t_obs − T_pred`. Weight = picker probability / σ for that phase and profile. Start σ at 0.02 s (P) and 0.04 s (S); update from the Tier A residual spread.
   *Status (LOC-09):* not applied. Stage locate compares the robust residual sigma with `pickSigmaS` and flags only a ratio above `statics.sigmaFlagRatio`; on the showcase run the robust sigma is below the configured values, so the formal `hErrM` / `vErrM` stay at the configured (larger) sigma. Changing σ would move the formal errors, the tier bars and the synthetic comparison: lead decision whether to keep this rule or rerun with an updated σ.
3. Drop picks with |residual| > max(3 × MAD, 0.15 s) and relocate once.
4. **Uncertainty:** normalize `exp(−misfit)` over the fine grid into a PDF, take its covariance, report `hErrM` (larger horizontal axis) and `vErrM`. Set `depthOnEdge` when > 5% of the mass sits on the top or bottom face.
5. **Statics:** after pass 1, take each station-phase's median residual over well-constrained events, subtract, relocate; three iterations; cap 0.3 s. This is `statics.mode: selfConsistent`. The showcase default, `referenceEvents` (LOC-05 lead decision), instead takes each station-phase term as the median residual at the public regional catalog's hypocentres of the matched events (hypocentre fixed, origin time by weighted median), relocates each matched event with terms computed without it (leave-one-out or `statics.folds`), gives unmatched events the terms from all matched events, and caps at `statics.referenceCapS`. It needs a match first, so the stages run locate (pass 1, no statics) → match → locate (pass 2) → match → tier; stage `locate` runs pass 2 when `matches.parquet` is in the run dir.

If someone already knows NonLinLoc well, it's an acceptable swap for steps 1–4. Nobody should learn it tonight.

### Diagnostics for smeared depths (LOC-04 runs these in order)

| # | Suspect | Quick test (~15 min) | Fix |
| --- | --- | --- | --- |
| 1 | Borehole sensor depth ignored | Print `surfaceElevM` and `sensorDepthM` per station; a borehole with depth 0 is a bug | Request H1 fix; use `sensorElevM` |
| 2 | Datum mixing | Locate a synthetic event at a known `elevM` and check the output | Everything in `elevM`; convert only at display |
| 3 | Coarse or homogeneous model used as the final location | Compare PyOcto depth with relocated depth | Relocate with the layered model |
| 4 | Too few S picks, so depth trades off against origin time | Depth spread for nS ≥ 3 vs nS < 3 | Require S for Tier A |
| 5 | Borehole picks degraded by resampling | Residuals split by preprocessing profile | Ask H1 to try `borehole-B`; per-profile σ |
| 6 | Station timing offsets or local sediment | Median residual per station and phase | Statics (LOC-05) |
| 7 | 1D can't represent the dipping basement | Residuals trend with azimuth or position | 3D grids (LOC-07) |

### Synthetic recovery test

200 synthetic hypocenters in the expected zone, the real station geometry, Gaussian pick noise at your σ values, S picks dropped to match the real nS distribution. Report median and p90 horizontal and vertical error plus median depth bias into `synthetic.json`. That becomes the slide saying "this station geometry resolves depth to ±N m."

### Depth gate (first check 4 AM, final call 10 AM)

- Tier A median `vErrM` is within about 2× the synthetic median.
- Under 20% of Tier A has `depthOnEdge` at the top face (the z = 0 collapse).
- Tier A depths show no systematic offset from the depth band in published FORGE catalogs. Compare the band only; those catalogs cover other dates.
- No unexplained station static above 0.15 s.

**Depth call, Saturday 10 AM: PASS (amended).** Decided by H2 on run `20260926-0210-a04c611` (evidence: that run's `diagnostics.md` and the lead's depth-gate report). Criteria 1 and 2 pass as written. Two amendments are on record:
- Criterion 3 is judged against the same-day public regional catalog's depths for the same events. The published FORGE catalogs sit at a different place and date, so the band comparison can't test bias.
- Criterion 4 is closed by a sensitivity test: relocating Tier A without the unexplained far-station statics barely moves them.

Consequences:
- The 3D scene with depths stays the hero.
- No "structure" or "fracture" claims. Those would need relative relocation (LOC-08).

**Fallback ladder if it fails:** (1) Tier A additionally requires a station within about 1–2× the focal depth; (2) 3D grids; (3) relative relocation (LOC-08); (4) still diffuse → tell H3 and H4 that the plan-view hero is permanent.

### Association (PyOcto)

- Build PyOcto's 1D velocity model from the same layer file (PyOcto precomputes its own table; follow its docs).
- Record every argument in `ProcessingRun.associator`: probability thresholds, min picks, min P, min S, min stations with both, depth limits, time window, tolerances.
- **Sweep, don't guess:** min stations {4, 5, 6, 8} × min S {0, 1, 2} × probability {0.1, 0.2, 0.3}. Record public recall, total candidates and Tier A count for each; pick the knee.
- Merge duplicates: two events within 0.5 s that share at least half their picks.

### Matching

- Cost = |Δt| / 2 s + epicentral distance / 5 km; admissible only if |Δt| ≤ 2 s and distance ≤ 5 km. Magnitude stays out of the cost, since magnitudes get calibrated on these same matches.
- Report recovered X / N, list every unmatched public event with a reason (no data, below threshold, outside window), and report counts at (1 s, 3 km) and (3 s, 8 km).

### Tiers (from data, not textbooks)

1. Take the matched set M and compute its distributions of `nStations`, `nP`, `nS`, `rmsS`, `hErrM`, `vErrM`, `gapDeg`.
2. **Tier A (Strict):** on every metric, at least as good as the 25th-percentile-worst event in M; no `depthOnEdge`; at least one station within about 2× the focal depth.
3. **Tier B (Good):** on every metric, no worse than the worst matched event.
4. **Tier C (Candidate):** associated but outside that range.
5. Store each threshold with its source quantile. How we say it: Strict = "on every quality metric, at least as good as a bar that three-quarters of the public-catalog events we recovered meet." Each bar is set per metric; this does not mean three-quarters of recovered events pass every bar at once (`ProcessingRun.tiering` stores how many do). Caveat to own: public events are the larger ones, so tiers are conservative for small events.

### Magnitude

`M_cat = a·log10(A_peak) + b·log10(R) + c + station term`, with A_peak the response-removed peak on the horizontals in the S window and R hypocentral distance. Event magnitude = median of station magnitudes. Calibrate against one catalog magnitude type only.

## Definition of done (the hardening pass checks every line)

- [x] Synthetic test passes in CI and its numbers are in `synthetic.json`
- [x] Every diagnostics row in `diagnostics.md` has a result and a conclusion
- [x] Every PyOcto and locator parameter is in `seismology.yaml` and `run.json`
- [x] Tier thresholds trace to quantiles of the matched set, stored in `ProcessingRun.tiering`
- [x] Every unmatched public event has a reason
- [x] `associate`, `locate`, `match`, `assign_tiers` match `docs/02` §5 exactly and run on STA/LTA picks unchanged
- [x] `pytest -m smoke tests/seismology` < 30 s, offline, seeded

Checked by LOC-09 (status above). In CI (`make check`, smoke only): the locate stage's synthetic test and `synthetic.json`, every diagnostics row, noise-free recovery, the docs/02 §5 signatures, docs/02 `locate()` on STA/LTA-labelled picks and the one-to-one property. The STA/LTA association test (every PyOcto call reloads its tables, too slow for the smoke budget) and the larger `run_synthetic` test stay in the full suite (`pytest tests/seismology`).

## Kickoff prompt

```
You are the H2 Seismology lane agent for Hidden Quakes (HackGT 13).
Before writing anything, read: CLAUDE.md, docs/00-project.md, docs/01-architecture.md,
docs/02-contracts.md, docs/03-schedule.md, and docs/lanes/H2-seismology.md.
Write only inside the paths listed under "You own" in your lane doc, plus files your ticket names.
Your ticket: <TICKET-ID>. Start by restating its goal, files, inputs, outputs and acceptance test,
then write a short plan. Build the synthetic test before touching real data. Implement it with tests.
Finish by running `make check` and the acceptance test, and paste both outputs. If you need anything
from another lane, write a request in docs/requests/ and stop to tell me. Never change a contract.
```

Reviewer prompt (fresh session, after the builder finishes):

```
You are reviewing a Hidden Quakes H2 Seismology PR. Read CLAUDE.md, docs/02-contracts.md and
docs/lanes/H2-seismology.md, then review the diff on this branch against the ticket's acceptance test
and the lane's Definition of done. Check the physics: elevation/datum handling, receiver depths,
origin-time elimination, uncertainty math, tier derivation. Check that paths stay in-lane, contracts
match docs/02 exactly, and tests are offline and seeded. List concrete findings with file:line,
most serious first. Don't rewrite the code yourself.
```

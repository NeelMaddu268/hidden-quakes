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

### MATCH-02 · P0 · Start ~2:00 AM — Catalog matching

- **Goal:** one-to-one matching with `linear_sum_assignment`, sensitivity at three tolerance pairs, and a reason for every unmatched public event.
- **Files:** `hq/match/__init__.py` (`match()`), `hq/match/run.py`, `tests/seismology/test_match.py`
- **In → out:** located events + catalog → `matches.parquet`, `match_sensitivity.parquet`
- **Depends on:** LOC-04, MATCH-01
- **Accept:** X / N plus the unmatched list; sensitivity table; one-to-one guaranteed by test.

### LOC-05 · P0 · Start ~2:00 AM — Station statics

- **Goal:** three iterations of per-station, per-phase statics inside `locate()`, capped at 0.3 s.
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
3. Drop picks with |residual| > max(3 × MAD, 0.15 s) and relocate once.
4. **Uncertainty:** normalize `exp(−misfit)` over the fine grid into a PDF, take its covariance, report `hErrM` (larger horizontal axis) and `vErrM`. Set `depthOnEdge` when > 5% of the mass sits on the top or bottom face.
5. **Statics:** after pass 1, take each station-phase's median residual over well-constrained events, subtract, relocate; three iterations; cap 0.3 s.

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
5. Store each threshold with its source quantile. How we say it: Strict = "located at least as well as three-quarters of the public-catalog events we recovered, on every metric." Caveat to own: public events are the larger ones, so tiers are conservative for small events.

### Magnitude

`M_cat = a·log10(A_peak) + b·log10(R) + c + station term`, with A_peak the response-removed peak on the horizontals in the S window and R hypocentral distance. Event magnitude = median of station magnitudes. Calibrate against one catalog magnitude type only.

## Definition of done (the hardening pass checks every line)

- [ ] Synthetic test passes in CI and its numbers are in `synthetic.json`
- [ ] Every diagnostics row in `diagnostics.md` has a result and a conclusion
- [ ] Every PyOcto and locator parameter is in `seismology.yaml` and `run.json`
- [ ] Tier thresholds trace to quantiles of the matched set, stored in `ProcessingRun.tiering`
- [ ] Every unmatched public event has a reason
- [ ] `associate`, `locate`, `match`, `assign_tiers` match `docs/02` §5 exactly and run on STA/LTA picks unchanged
- [ ] `pytest -m smoke tests/seismology` < 30 s, offline, seeded

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

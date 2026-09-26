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

## What you produce

| What | For | Where | Due |
| --- | --- | --- | --- |
| `stations.parquet` (with real sensor depths) | H2, H4 | `runs/<id>/stations.parquet` | 9:00 PM |
| Known-event picks + record sections | H2 (Check B) | `runs/<id>/known/` | 10:00 PM |
| Waveform cache + `read_window` / `read_inventory` / `display_copy` | H2 (magnitude), H4 (evidence) | `data/cache/mseed/`, `hq.ingest.cache`, `hq.preprocess` | 10:30 PM |
| Full-window PhaseNet picks | H2 | `runs/<id>/picks.parquet` | 12:30 AM |
| STA/LTA picks + threshold sweep | H4 (VAL-01) | `runs/<id>/picks_stalta.parquet`, `baseline_sweep.parquet` | 4:00 AM |

## Tickets, in order

### SEIS-01 · P0 · Start 8:05 PM — Station inventory

- **Goal:** query stations in the bbox at channel level for the window; build `Station` rows with `sensorDepthM` from each channel's `depth`, `kind`, sample rate and chosen channel triplet.
- **Files:** `hq/ingest/inventory.py`, `hq/config/signal.py` (station-selection section), `configs/showcase/signal.yaml`, `tests/signal/test_inventory.py`
- **In → out:** `run.yaml` → `data/cache/stationxml/*.xml`, `runs/<id>/stations.parquet`
- **Depends on:** `run.yaml` (8:05)
- **Accept:** prints a station table (id, kind, rate, channels, sensorDepthM); every borehole station has `sensorDepthM > 0` or is flagged in the log; reports how many stations have any data in the window; test parses a small recorded StationXML fixture offline.

### SEIS-03 · P0 · Start 8:05 PM — Preprocessing profiles

- **Goal:** implement `surface-100`, `surface-hi`, `borehole-A`, `borehole-B` and `display_copy()`, with explicit anti-aliasing, plus `TimeMap` for profile B.
- **Files:** `hq/preprocess/__init__.py`, `hq/preprocess/profiles.py`, `tests/signal/test_preprocess.py`
- **In → out:** raw `Stream` → 100 Hz model-ready `Stream` + `TimeMap`
- **Depends on:** nothing (synthetic tests)
- **Accept:** a synthetic 1,000 Hz chirp keeps no energy above 50 Hz after profile A (spectrum test, < −40 dB); a known onset round-trips through profile B's `TimeMap` within 1 ms; gaps stay gaps (no zero-filled samples).

### SEIS-02 · P0 · Start ~8:30 PM — Known-event windows

- **Goal:** download 10-minute windows around the 3 largest public events in the showcase window.
- **Files:** `hq/ingest/windows.py`, `tests/signal/test_windows.py`
- **In → out:** `catalog.parquet` + `stations.parquet` → mseed windows in `data/cache/mseed/`
- **Depends on:** SEIS-01, MATCH-01 (H2)
- **Accept:** at least 8 stations with three-component data per window; a gap report per window.

### SEIS-05 · P0 · Start ~8:30 PM — Full-window ingestion

- **Goal:** download the whole showcase window ±5 min in hour chunks, with retries, a cache keyed by channel-day, a gap log, and `read_window` / `read_inventory`. The same function must work for any window, including the last 2 hours (Live mode reuses it).
- **Files:** `hq/ingest/download.py`, `hq/ingest/cache.py`, `tests/signal/test_cache.py`
- **In → out:** `stations.parquet` → `data/cache/mseed/{net}.{sta}.{loc}.{cha}.{YYYYMMDD}.mseed`, `runs/<id>/gaps.parquet`
- **Depends on:** SEIS-01
- **Accept (Check A):** at least 12 useful stations (three components, < 20% gaps); a rerun is a pure cache hit; `read_window` returns gaps as separate traces. When the full download finishes (~10:30 PM), your human hands the whole `data/cache/` folder to H4 by AirDrop or USB.

### SEIS-04 · P0 · Start ~9:00 PM — PhaseNet on known events + weight A/B

- **Goal:** run PhaseNet on the known-event windows for each candidate weight set per profile, and pick the weights.
- **Files:** `hq/pick/__init__.py`, `hq/pick/phasenet.py`, `hq/pick/ab.py`, `tests/signal/test_pick.py`
- **In → out:** known-event windows → `runs/<id>/known/picks.parquet`, `known/ab.csv`, one record-section PNG per event
- **Depends on:** SEIS-02, SEIS-03
- **Accept (Check B):** P and S picks on ≥ 8 stations for ≥ 3 events; P before S on every station; Spearman ρ ≥ 0.8 between P time and epicentral distance; the A/B table names the chosen weights per profile. Then `make publish-run RUN=<runId>` and give H2 the runId.

### SEIS-06 · P0 · Start ~10:30 PM — Full-window picking

- **Goal:** pick the full window at probability ≥ 0.1 with the chosen weights, parallel by station, runtime recorded.
- **Files:** `hq/pick/run.py` (stage `run(ctx)`)
- **In → out:** cache → `runs/<id>/picks.parquet`; `ctx.record("pick", ...)`
- **Depends on:** SEIS-04, SEIS-05
- **Accept:** per-station pick counts printed; any station with zero picks has a stated reason; picks within 1 s of a gap edge are dropped and counted; rerun with the same config gives identical output. Then `make publish-run RUN=<runId>` and give H2 the runId.

### SEIS-07 · P1 · Start ~12:00 AM — STA/LTA baseline

- **Goal:** classical picks with `recursive_sta_lta` + `trigger_onset` on the same preprocessed traces, in the `Pick` schema, with a threshold sweep that maximizes the baseline's own Tier A count once H2's pipeline exists.
- **Files:** `hq/baseline/__init__.py`, `hq/baseline/run.py`, `tests/signal/test_baseline.py`
- **In → out:** cache → `runs/<id>/picks_stalta.parquet`, `baseline_sweep.parquet`
- **Depends on:** SEIS-03, SEIS-05
- **Accept:** picks load through H2's `associate()` unchanged; the sweep is saved; the chosen thresholds are in `signal.yaml`.

### SEIS-08 · P1 · 2–6 PM Saturday — Lane hardening pass

- **Goal:** a fresh reviewer agent audits the whole lane against "Definition of done" below and fixes what it finds.
- **Accept:** every item in the checklist is true, with evidence pasted in the PR.

## Domain notes (give these to your agent)

### Data access

- ObsPy ≥ 1.5: `Client("EARTHSCOPE")`. Older ObsPy: `Client("https://service.earthscope.org/")`. EarthScope FDSNWS credentials go in env vars, never in the repo.
- Request in hour chunks. Large dataselect requests can stall for minutes before returning; use timeouts, retries with backoff, and resume from the cache.
- Channel priority for picking: broadband/short-period velocity (`HH?`, `EH?`) and borehole geophones (`DP?` or similar) first; strong-motion (`HN?`) only if nothing else exists at that site. Three components or skip.
- **Sensor depth:** StationXML station elevation is the surface (the wellhead for boreholes). The sensor depth lives on each **channel's** `depth`. Ignoring it on deep borehole sensors shifts S arrivals by a large fraction of a second and smears every depth downstream. This is suspected cause #1 of the prototype's broad depths.

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
- One day on ~20 stations is small. Parallelize by station; a GPU is optional. Cache model weights on first load.

### STA/LTA

- Short windows suit microseismic data: STA ~0.05–0.1 s, LTA ~2–5 s at 100 Hz. P onsets from Z; S from horizontals after the P.
- Tune on/off thresholds to maximize STA/LTA's own Tier A count. Giving the baseline its best shot is what makes the comparison honest.

## Definition of done (the hardening pass checks every line)

- [ ] `stations.parquet` sensor depths match StationXML channel depths for every borehole station
- [ ] No zero-filled samples anywhere; gap-edge picks dropped and counted
- [ ] Every knob in `signal.yaml`; every pick carries `picker` with the weights name
- [ ] `hq run` from a clean cache reproduces identical picks
- [ ] Runtimes and counts recorded in `run.json` for inventory, download, pick, baseline
- [ ] `pytest -m smoke tests/signal` < 30 s, offline
- [ ] `read_window`, `read_inventory`, `display_copy`, `for_picking` match the `docs/02` signatures exactly

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

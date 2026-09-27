# 03 · Schedule, gates, freezes, kill switches

T+0 = **8:00 PM Friday**. The event runs to Sunday morning. H4 confirms the exact submission deadline at opening ceremony and edits the last rows of this file if it differs from 8:00 AM Sunday.

> **Late start.** Kickoff ran at about 8:45 PM, not 8:00. Read every Friday clock time before 10 PM in these docs (the 8:05, 8:25, 8:30, 8:45 and 9:00 hand-offs, including the contract freeze) as 45 minutes later, and Gate V as 10:45 PM. Everything from Gate M (12:00 AM) on stays where it is.

We have ~36 hours and strong agents, so code volume isn't the constraint. Two things are: the science chain is serial and data-dependent (picks → association → location → tiers), and the two go/no-go decisions come early (10 PM and 4 AM). Spend slack on review agents, tests and polish, never on new scope.

## Gates

| Gate | Clock | Pass means | Owner |
| --- | --- | --- | --- |
| **V · viability** | 10:00 PM Fri | Check A and Check B pass | H1 + H2 |
| **M · mock** | 12:00 AM Sat | Mock reveal runs in the deployed build with shell and providers | H3 + H4 |
| **S · science** | 4:00 AM Sat | Hour-8 table below passes; the first real bundle renders | H2 |
| **E · end to end** | 8:00 AM Sat | Showcase mode on the public URL; drawer opens real evidence | H4 |
| **Depth call** | 10:00 AM Sat | Depth gate passes, or the plan-view hero becomes permanent | H2 |
| **D · demo** | 2:00 PM Sat | Full 2-minute flow works; an outsider explains it in 20 seconds | Everyone |
| **P · pipeline freeze** | Passed early, about 6:45 PM Sat (H2's call: run `20260926-0210-a04c611` is final and no v4 follows; the 9 PM slot from the 4:50 PM move is void) | Final showcase `runId` exported and deployed | H2 + H4 |
| **F · feature freeze** | 12:30 AM Sun (H2's call at about 8:50 PM Sat, finishing early; UI copy freeze 1:30 AM, code freeze 2:30 AM, submission by about 3 AM) | Nothing new after this | H4 |

### Check A and Check B (10:00 PM)

- **Check A:** EarthScope serves September 10 data for at least 12 useful stations. Useful = three components, under 20% gaps in the window, inside the bbox.
- **Check B:** for at least 3 known public-catalog events, PhaseNet gives P and S picks on at least 8 stations each, with P before S on every station and P times increasing with distance (Spearman ρ ≥ 0.8).
- **Fail:** take 30 minutes to find the cause. If it's fundamental (the data isn't there, or no station gives usable picks), switch to Nightflight by 10:30 PM.

### Hour-8 science gate (4:00 AM)

| Check | Pass | If it fails |
| --- | --- | --- |
| Public recall | ≥ 38 / 43, or ≥ 88% of the re-queried count | Debug up to 2 h; still poor → Nightflight |
| Additional candidates | ≥ ~2× the public count | Lead with Strict and honesty; never pad with Tier C |
| Locations | Tier A median `vErrM` ≤ ~2× the synthetic median; under 20% of Tier A pinned at the grid top (z = 0 collapse); depths inside the published FORGE band | Downgrade geometric claims now, run the fallback ladder, final call at 10 AM |
| Magnitude–frequency | Not screaming noise (if magnitudes exist yet) | Cut G-R; keep going |
| Coherence | Events cluster instead of scattering uniformly through the volume | Treat as a location bug first; if unresolved → Nightflight |

## Schedule

| Clock | T+ | H1 Signal | H2 Seismology | H3 Visualization | H4 Platform |
| --- | --- | --- | --- | --- | --- |
| 8:00–8:30 PM | 0–0.5 | Env, credentials, SEIS-01, SEIS-03 | LOC-00 `run.yaml` by 8:05, MATCH-01, LOC-01 | Scaffold, WEB-01 on placeholder types | REPO-00 by 8:05, CONTRACT-01 merged 8:25 |
| 8:30–10 PM | 0.5–2 | SEIS-02, SEIS-05 start, SEIS-04 | LOC-02 + synthetic test | WEB-01, WEB-03 skeleton | RUN-01, FIX-01, API-01, DEMO-01 skeleton |
| **10 PM** | **2** | **Gate V + integration meeting 1 (15 min)** | | | |
| 10 PM–12 AM | 2–4 | SEIS-05 finish, SEIS-06 start | LOC-03 + LOC-04 on known events, diagnostics | WEB-02 terrain, WEB-03 reveal | API-03 deploy mock, DEMO-01 |
| **12 AM** | **4** | **Gate M + integration meeting 2** | | | |
| 12–4 AM | 4–8 | SEIS-06 full picks, SEIS-07 baseline | LOC-03/04 full, LOC-05, MATCH-02, LOC-06 | Queue WEB-04 + WEB-05 agents by 12:45, sleep 1–5 AM | API-02 on partial runs |
| **4 AM** | **8** | **Gate S + integration meeting 3** | | | |
| 4–8 AM | 8–12 | Sleep 5–9 AM | Fallback ladder or LOC-07, reviewer passes | Review WEB-04/05, WEB-07 plan view, real data in scene | API-02 hero + deploy showcase, FEAT-01, VAL-02 |
| **8 AM** | **12** | **Gate E + integration meeting 4** | | | |
| 8 AM–2 PM | 12–18 | Baseline tuning, support MAG-01 | Depth call at 10 AM, MAG-01, LOC-08 if needed · sleep 11 AM–3 PM | WEB-06 time scrubber, labels, depth cues | VAL-01, DEMO-02, API-04 · sleep 9 AM–1 PM |
| **2 PM** | **18** | **Gate D; outsider test at 4 PM** | | | |
| 2–6 PM | 18–22 | Lane hardening pass | Final run, thresholds locked | Polish from outsider test | Pitch v1, API-05 |
| **6 PM** | **22** | **Integration meeting 6** (Gate P moved to 9 PM) | | | |
| **~6:45 PM** | **23** | **Gate P: data freeze, called early by H2** (run `20260926-0210-a04c611` final; no v4) | | | |
| 6 PM–12:30 AM | 22–28.5 | Everyone: features for the judging rubric (comprehension, labels, who it is for, what the ML is), Devpost screenshots, language audit; every push to `main` redeploys the site on Neel's side | | | |
| **12:30 AM Sun** | **28.5** | **Gate F: feature freeze** (finishing early: H2's call at about 8:50 PM Sat) | | | |
| **1:30 AM Sun** | **29.5** | UI copy and claims freeze | | | |
| **2:30 AM Sun** | **30.5** | Code freeze; the repo goes public | | | |
| **~3 AM Sun** | **31** | Submission done: Devpost first, then the Expo form (expo.hexlabs.org) | | | |
| 12:30–3 AM | 28.5–31 | Devpost fill (`make story`), video, Expo form, README, multi-browser + Wi-Fi-off tests; emergency fixes only after 2:30 AM | | | |
| 7–8 AM | 35–36 | Pitch practice; nothing else | | | |

H2 and H3 are never asleep at the same time before the 10 AM depth call.

**Hardening pass (2–6 PM Saturday):** each human runs one fresh reviewer agent over their whole lane against their lane doc's "Definition of done", then fixes what it finds before the pipeline freeze.

## Integration meetings

15 minutes, at the team table: **10 PM Fri, 12 AM, 4 AM, 8 AM, 2 PM, 6 PM Sat.** From Saturday evening H4 merges to `main` as soon as a change is green, and every push to `main` redeploys the site on Neel's side (`docs/deploy.md`). Each lane answers three things: what's merged, what's blocked (with a `docs/requests/` link), and what lands before the next meeting. H4 merges `feat/*` into `main` right after each one, then everyone runs `git fetch && git merge origin/main`.

Before the first meeting, foundations land on `main` directly so nobody waits: `run.yaml` (H2, 8:05 PM), contracts (H4, 8:25), config loader + stage runner and mock bundle (H4, 8:45), provider hooks (H4, 9:00).

## Freezes

| What | Frozen at | Allowed after |
| --- | --- | --- |
| Contracts (`packages/contracts`, `docs/02`) | 8:30 PM Fri | PR + approval from affected lanes + version bump |
| Tier definitions and match tolerances | 10:00 AM Sat | Only with a rerun under a new `runId` |
| Showcase run | Frozen about 6:45 PM Sat (run `20260926-0210-a04c611`; H2 called no v4) | Nothing: the bundle on `main` is final |
| Features | 12:30 AM Sun (finishing early) | Bugs, polish, docs |
| UI copy and claims | 1:30 AM Sun | Typo fixes |
| Code | 2:30 AM Sun; the repo goes public | Emergency fixes only |
| Submission | By about 3 AM Sun: Devpost first, then the Expo form (expo.hexlabs.org) | — |

## Kill switches

Only two switches end Hidden Quakes, and both fire before 4 AM. Everything else downgrades a claim or cuts a feature. **The reveal is never cut.**

| Switch | Trigger | Checked | Action |
| --- | --- | --- | --- |
| Pipeline | Check A or B fails for a fundamental reason | 10 PM Fri (10:30 at latest) | Switch to Nightflight |
| Science | Recall poor or events incoherent after reasonable debugging | 4 AM Sat | Switch to Nightflight |
| Depth | Over 20% of Tier A pinned at the grid top, or the vertical distribution is nonphysical | 4 AM first, final 10 AM | No structure or fracture claims. Hero becomes plan view + depth section + halos + Strict subset. |
| Baseline | STA/LTA within ~20% of PhaseNet's strict count at comparable rms | 2 PM Sat | Drop the neural-advantage claim; keep the table as context |
| Magnitude | Leave-one-out MAE above ~0.4 | 2 PM Sat | Cut magnitude sizing and G-R |
| Ridgecrest | Not started by the pipeline freeze | ~6:45 PM Sat (freeze called early) | Cut |
| Live | Unstable, or latency over ~10 min | Decided by H2 at about 8:50 PM Sat: **cut** | The LIVE pill stays off (`NEXT_PUBLIC_LIVE_ENABLED` unset in the web build, `docs/deploy.md`); the pitch and shot list keep their "only if LIVE is up" lines, which now read as omitted |
| Evidence drawer | Too slow or heavy | Any time | Drop to 8 traces per event |
| Terrain | DEM work slows integration | 12 AM Sat | Clean abstract slab, labeled as such |
| UI numbers | Any number not from provider data | Always | Bug; fix before the pipeline freeze |

Bad depths alone never trigger Nightflight. They downgrade claims.

**Nightflight (backup only, not architected):** Atlanta KFFC NEXRAD → Py-ART → dual-pol + MistNet (described honestly as a pretrained component) → weather/biology separation → the same WebGL reveal. Use the mixed rain-and-birds night for the reveal and October 8, 2025 only as the record-migration showcase. Use current Unidata NEXRAD storage paths and "Birds Georgia" as the organization name. Don't claim a classifier beats thresholds unless you trained and evaluated one. The scene, shell and providers carry over.

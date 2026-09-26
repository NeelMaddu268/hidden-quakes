# H3 · Visualization lane

**Mission:** the reveal. A judge must understand the screen in 5 seconds and say "whoa" in 10. You build entirely on mock data until ~4 AM; switching to real data has to be a mode change (`?mode=showcase`), never a rewrite. The reveal is the one thing the team never cuts.

## Paths

**You own:** `apps/web/src/scene/`, `apps/web/src/drawer/`, `apps/web/src/state/`, `packages/visualization/`, `apps/web/public/terrain/`, `scripts/bake-dem.py`

**You never write:** H4's `apps/web/src/app|shell|providers/`, `apps/web/public/data/`, `packages/contracts/`, repo config; anything in `services/`.

The layout split: **you own the canvas, the drawer and the time scrubber.** H4's shell owns everything overlaid on the canvas (title, counters, REVEAL button, pills, panels, SYNTHETIC banner, keyboard). You both talk through the demo store.

## What you consume

| What | From | Where | Until it exists |
| --- | --- | --- | --- |
| Next.js app scaffold | H4 | `apps/web/` (REPO-00) | 8:05 PM |
| Generated types | H4 | `packages/contracts/ts/` | placeholder types copied from the `docs/02` spec, deleted at 8:30 |
| Mock bundle | H4 | `apps/web/public/data/mock/` | a tiny hand-made JSON inside your test, deleted once mock lands (~8:45) |
| Provider hooks | H4 | `apps/web/src/providers/hooks.ts` | import mock JSON directly behind a `useBundle`-shaped shim, deleted at 9:00 |
| Origin + bbox | H2 | `services/seismic/configs/showcase/run.yaml` | 8:05 PM |
| Showcase bundle | H4 | `apps/web/public/data/showcase/` | mock bundle; real by ~4:15 AM |
| Features (wells, region outline) | H4 | `features.json` in the bundle | draw nothing |

## What you produce

| What | For | Where | Due |
| --- | --- | --- | --- |
| Demo store | H4 shell | `apps/web/src/state/demo.ts` (`docs/02` §6) | 9:00 PM |
| Design tokens | H4 shell | `packages/visualization/tokens.ts` | 9:00 PM |
| `<Scene/>` | H4 page | `apps/web/src/scene/index.ts` | mock reveal by 12:00 AM (Gate M) |
| `<EvidenceDrawer/>` | H4 page | `apps/web/src/drawer/index.ts` | 6:00 AM (reviewed after your 1–5 AM sleep) |
| `<TimeScrubber/>` | H4 page | `apps/web/src/scene/time/index.ts` | 2:00 PM Sat |
| Baked terrain | Scene | `apps/web/public/terrain/` | 11:00 PM |

## Tickets, in order

### WEB-01 · P0 · Start 8:05 PM — Scene, store, tokens

- **Goal:** R3F canvas with one `InstancedMesh` of events placed via ENU → scene axes, orbit camera with presets (oblique, side, plan), the demo store from `docs/02` §6 (zustand), and `tokens.ts` from the table below.
- **Files:** `apps/web/src/scene/index.ts`, `scene/Canvas.tsx`, `scene/events/`, `scene/camera/`, `scene/coords.ts`, `apps/web/src/state/demo.ts`, `packages/visualization/tokens.ts`, tests beside each
- **In → out:** `useBundle()` events → rendered points; store actions → camera presets
- **Accept:** 2,000 events at 60 fps on the demo laptop; a unit test puts a known ENU point at the expected scene coordinate; store actions behave per `docs/02`.

### WEB-02 · P0 · Start ~9:30 PM — Terrain and references

- **Goal:** `scripts/bake-dem.py` (public DEM tiles → 16-bit `height.png`, a hillshade you compute, `meta.json` with ENU bounds and source); terrain mesh with contour shader and an opacity uniform; depth ruler; horizontal slices every 1 km; surface stations; borehole sensors at true depth with a line to the wellhead; features from `features.json`; vertical-exaggeration badge logic; abstract-slab fallback flag.
- **Files:** `scripts/bake-dem.py`, `apps/web/public/terrain/*`, `scene/terrain/`, `scene/references/`
- **In → out:** DEM tiles + `run.yaml` origin + stations + features → terrain and reference layers
- **Accept:** the ruler label reads `SceneMeta.depthLabel`; borehole markers sit at `sensorElevM`; unverified features render dashed with "approximate" or not at all.

### WEB-03 · P0 · Start ~9:00 PM — The reveal

- **Goal:** the ~7-second choreography below on one `revealProgress` value: terrain fade, camera dolly, per-instance `revealAt` from `revealOrder`, bloom pass. The shell's counter reads the same progress.
- **Files:** `scene/reveal/`, `scene/post/`
- **In → out:** store `phase` + events → animation
- **Accept:** identical every run; `revealProgress` hits exactly 1 and the phase becomes `revealed`; `reset()` returns to a pixel-identical start frame.

### WEB-04 · P0 · Queue by 12:45 AM — Filters and halos

- **Goal:** PUBLIC / ALL / STRICT with tier opacity, and uncertainty halos sized from `hErrM` / `vErrM` on Tier A under STRICT.
- **Files:** `scene/filters/`, `scene/events/halos.ts`
- **Accept:** STRICT shows exactly `strictQualityCount` instances (unit test on the filter selector).

### WEB-05 · P0 · Queue by 12:45 AM — Evidence drawer

- **Goal:** click to select (raycast on instances with a screen-space threshold); the drawer slides in at 40% width with a record section sorted by distance, P and S picks animating in, predicted-arrival lines, header stats, a mini map with station-to-epicenter lines, and a small depth section.
- **Files:** `apps/web/src/drawer/*`
- **In → out:** `useEvidence(selectedEventId)` → drawer
- **Accept:** opens in under 200 ms from preloaded evidence; E opens the hero event; missing `predP` / `pickS` render gracefully, never as `NaN`.

### WEB-07 · P0 · Start ~5:00 AM — Plan view and depth section (insurance)

- **Goal:** a first-class plan-view mode (top-down map of events with halos) plus a side depth section, reachable from the `plan` camera preset. If the depth gate fails, this becomes the hero view, so it has to look as good as the 3D view.
- **Files:** `scene/plan/`
- **Accept:** P toggles it; the reveal also works starting from plan view.

### WEB-06 · P1 · After Gate E — Time scrubber

- **Goal:** `tNow` uniform with decaying glow on the last ~30 minutes, 1 h/s playback, and a per-10-minute histogram strip of public vs recovered.
- **Files:** `scene/time/`
- **Accept:** scrubbing to the window start shows zero events and to the end shows all of them; playback is smooth at 60 fps.

### WEB-08 · P1 · 2–6 PM Saturday — Visual QA and hardening

- **Goal:** a fresh reviewer agent plus a human run the QA checklist below on the demo laptop and fix everything that fails.

## Domain notes (give these to your agent)

### Next.js version

The app was scaffolded with the latest create-next-app, which ships `apps/web/AGENTS.md`: it says this Next.js version has breaking changes and to read the relevant guide in `node_modules/next/dist/docs/` before writing code. Follow it; don't trust memory of older Next.js APIs.

### Coordinates

- ENU comes precomputed in every record (`enu.e`, `enu.n`, `enu.u`, meters). Scene: x = e / 1000, y = u / 1000 × verticalExaggeration, z = −n / 1000 (1 unit = 1 km, y-up).
- Never recompute positions from lat/lon in the browser.
- Any vertical exaggeration other than 1.0 shows a permanent "Vertical ×N" badge.

### Scene composition

- **Terrain:** ~16 × 16 km around the origin, 512 × 512 segments, dark base, thin contours every 100 m, opacity uniform. No licensed imagery; compute your own hillshade. AWS Terrain Tiles (Terrarium encoding) or USGS 3DEP are fine public DEM sources.
- **Subsurface:** depth ruler at the scene edge (0–6 km below site surface, 1 km ticks) with the depth label beside it; faint slices every 1 km.
- **Stations:** small inverted triangles on the surface. Borehole sensors: a marker at true depth plus a thin line up to the wellhead. That one detail shows judges why depth is constrained.
- **Events:** public = cool white, recovered = amber, tier sets opacity (A 1.0, B 0.6, C 0.3). Size and halo carry tier too, not color alone.
- **Post:** bloom with a luminance threshold so only events glow; light vignette; DPR capped at 2; exponential fog increasing with depth.

### The reveal (~7 s)

| Time (s) | What happens |
| --- | --- |
| 0.0 | `reveal()` called; phase → `revealing` |
| 0.0–1.2 | Terrain opacity 1 → 0.12; contours stay |
| 0.4–2.0 | Camera dollies from the oblique surface view to a low side view below ground |
| 1.0–6.0 | Recovered events appear in `revealOrder`, each popping (scale 2 → 1, brightness spike, settle) |
| 1.0–6.0 | `revealProgress` drives the shell's counter from `publicCatalogCount` to `candidateCount` |
| 6.0–7.0 | Settle; phase → `revealed` |

`revealOrder` comes from the exporter: Tier A first (time-ordered within), then B, then C. Structure forms first, then fills in. Never re-sort in the browser.

### Strict, time, drawer

- **STRICT:** B and C fade to 0.05 over 600 ms; halos appear on Tier A.
- **TIME:** events with `t ≤ tNow` visible; recent ones glow and decay. Show only what the data supports; never add a narrative the data doesn't show.
- **Drawer:** normalized traces, P ticks in `pickP` blue, S ticks in `pickS` orange on a 150 ms stagger, faint predicted-arrival lines. Header: "{nStations} stations agreed", tier, rms, ±h / ±v error, matched public id if any. All numbers from the data.

### Design tokens (`packages/visualization/tokens.ts`)

| Token | Value | Use |
| --- | --- | --- |
| `bg` | #07090C | Page and scene background |
| `surface` | #0E1217 | Drawer, panels |
| `terrain` | #1A2027 | Terrain base |
| `contour` | #2A333D | Contours, slices, ruler ticks |
| `text` | #E6E9ED | Primary text, counters |
| `textDim` | #8A94A0 | Labels, secondary |
| `public` | #DCE6F2 | Public-catalog events |
| `recovered` | #FFB547 | Recovered events: the only accent |
| `strictHalo` | #FFD08A | Tier A halos |
| `geo` | #7FE0CF | Geothermal reference |
| `pickP` | #5AA9FF | P picks |
| `pickS` | #FF8A4C | S picks |
| `station` | #7C8B99 | Station glyphs |
| `alert` | #FF4D4D | SYNTHETIC banner and errors only |

Type: Inter for UI, JetBrains Mono with tabular figures for numbers. Motion: 150 ms micro, 600 ms state, 1,200 ms scene moves, cubic-out; nothing loops except the Live pulse. Banned: decorative gradients, cards everywhere, rainbow colormaps, particles that aren't events.

### Performance budget

- 60 fps with 2,000 events + terrain + bloom on the demo laptop.
- First frame under 1.5 s from a warm cache; the reveal never waits on a network request.
- No per-frame allocations in `useFrame`; per-instance data lives in typed-array attributes.

### Visual QA checklist (WEB-08)

- [ ] 5-second test: pre-reveal frame shows only dark terrain, one geothermal reference, PUBLIC count, REVEAL button
- [ ] Depth label and ruler visible in every camera preset
- [ ] Reveal identical across 10 runs; reset returns to the exact start frame
- [ ] STRICT count matches the shell's STRICT counter
- [ ] Drawer handles events with 4 traces and with 16
- [ ] Plan view looks finished, not like a debug view
- [ ] Works in Chrome and Safari; 60 fps on the demo laptop; nothing breaks at 1280 × 720 or 4K
- [ ] A screenshot of the revealed state passes the "not random dots" test

## Definition of done

- [ ] Every number on the canvas or drawer comes from provider data
- [ ] Switching `?mode=mock` → `?mode=showcase` needs zero code changes
- [ ] Store matches `docs/02` §6 exactly; H4's shell works against it
- [ ] Plan view is production quality
- [ ] QA checklist passes; `pnpm -r typecheck && pnpm -r lint` clean

## Kickoff prompt

```
You are the H3 Visualization lane agent for Hidden Quakes (HackGT 13).
Before writing anything, read: CLAUDE.md, docs/00-project.md, docs/01-architecture.md,
docs/02-contracts.md, docs/03-schedule.md, and docs/lanes/H3-visualization.md.
Write only inside the paths listed under "You own" in your lane doc, plus files your ticket names.
Your ticket: <TICKET-ID>. Start by restating its goal, files, inputs, outputs and acceptance test,
then write a short plan. Work against the mock bundle through the provider hooks; never read
pipeline files directly. Implement it with tests. Finish by running `make check` and the acceptance
test, and paste both outputs. If you need anything from another lane, write a request in
docs/requests/ and stop to tell me. Never change a contract.
```

Reviewer prompt (fresh session, after the builder finishes):

```
You are reviewing a Hidden Quakes H3 Visualization PR. Read CLAUDE.md, docs/02-contracts.md and
docs/lanes/H3-visualization.md, then review the diff on this branch against the ticket's acceptance
test, the performance budget and the Definition of done. Check: coordinates follow the ENU → scene
mapping exactly, no hard-coded numbers in rendered text, no per-frame allocations, store matches
docs/02, paths stay in-lane. List concrete findings with file:line, most serious first. Don't
rewrite the code yourself.
```

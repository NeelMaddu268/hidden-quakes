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
- **As built:** the "16-bit `height.png`" is **rg16**: an 8-bit RGB PNG with v = R·256 + G (B = 0) spanning `elevMinM..elevMaxM`, because browsers truncate 16-bit PNGs to 8 bits in canvas and WebGL. `meta.json` documents the encoding, bounds (pixel centres, row 0 = north), source and attribution, plus checksums the browser checks so a lossy decode falls back loudly. Rebake with `uv run scripts/bake-dem.py` (`--self-test`, `--half-width-km`, `--size`, `--offline`). The "Terrain" kill switch (docs/03) is `FORCE_ABSTRACT_SLAB` in `scene/terrain/surface.ts`; `?terrain=slab` previews it.

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
- **Subsurface:** depth ruler at the scene edge (0–6 km below site surface, 1 km ticks) with the depth label beside it; faint slices every 1 km. *As built (WEB-08 follow-up):* the ruler and slices reach the deepest displayed event (candidates and public catalog, from provider data, rounded up to a whole kilometre), not a fixed 6 km.
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

### Demo store semantics (what H4's shell can rely on)

`apps/web/src/state/demo.ts` implements the docs/02 §6 interface exactly. Where docs/02 is silent, it behaves like this:

| Action | Effect |
| --- | --- |
| `reveal()` | Only from `"public"` (a no-op mid-reveal or after, so Space can't restart it). Sets `phase: "revealing"`, `revealProgress: 0`, `filter: "all"` (or keeps `"strict"` if S was pressed first), and `view: "side"` because the reveal dollies into the side view (plan view stays `"plan"`). |
| `reset()` | Every field back to the start frame: `phase "public"`, progress 0, filter `"public"`, selection null, time mode off, `tNow` null, not playing, view `"oblique"`. The camera tweens back to the preset. |
| `setFilter(f)` | Sets it in any phase. Before the reveal the scene still hides candidates; a pre-reveal S sticks through `reveal()`. |
| `setTimeMode(false)` | Also stops playback and sets `tNow` null (every event visible). `setTimeMode(true)` leaves `tNow` for the scrubber to set. |
| `setPlaying(true)` | Also turns time mode on. |
| `setView(v)` / `select(id)` / `setTNow(t)` | Set the field; nothing else. |

**Reveal timing for the counter:** `revealProgress` is the counter clock. It stays 0 until events start appearing (~1.0 s after `reveal()`), reaches exactly 1 as the last event appears (~6.0 s), and the counter should read `publicCatalogCount + (candidateCount − publicCatalogCount) × revealProgress`, rounded. `phase` becomes `"revealed"` only after the settle (~7.0 s). Gate "next beat" logic on `phase`, never on `revealProgress === 1`.

**Tokens for the shell:** `colors`, `fonts` (reads `--font-inter` / `--font-jetbrains-mono`), `numeric` (tabular figures), `motion`, `easeOutCubic`, `tierStyle`, `strictFadeOpacity`, and `cssVariables()` (every token as a `--hq-*` custom property).

### Strict, time, drawer

- **STRICT:** B and C fade to 0.05 over 600 ms; halos appear on Tier A.
  - *As built (WEB-04):* "shown" means drawn above the 0.05 background weight, so STRICT shows exactly `strictQualityCount` (Tier A) instances; B and C stay faintly on screen as context and aren't counted. The selector is derived from the renderer's own look table (`scene/filters/fade.ts`), so the count and the pixels can't drift.
  - The public regional catalog steps back to 0.4 weight under STRICT (after the reveal only); a STRICT pressed before the reveal changes nothing on the start frame and applies when the reveal starts.
  - Halos are ellipsoids with semi-axes hErrM (horizontal, drawn as a circle: the contract has no orientation) and vErrM × VE, drawn as a soft rim in `strictHalo`. Overlapping halos combine with MAX blending, so a dense cluster is never brighter than one rim, and the rim stays below the bloom threshold: halos never glow and never outshine their events. Tier A events without a 68% error get no halo (count logged).
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
- No per-frame allocations in `useFrame`; per-instance data lives in typed-array attributes. One deliberate exception: during the ~5 s reveal the scene calls `setRevealProgress` once per frame, which is one small zustand `set` (docs/02 routes `revealProgress` to the shell through the store).

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

## WEB-07 implementation plan · prepared before the 1–5 AM sleep window

Status (04:10 EDT Sat): implemented and accepted on `agent/WEB-07` (headed Chrome 153 + WebKit 26.5); see
"WEB-07 completion checkpoint" at the end. Merge after the 1–5 AM no-merge window. Gate S (the depth call) is
still undecided; plan view is ready to become the hero view if it fails, with no code change.

### Starting point and invariant

WEB-02, WEB-04, the WEB-01 provider cleanup and WEB-05 are integrated into `feat/web`. The scene and
drawer share H4's provider; the existing shell already maps P to `setView("plan")`, E to the hero,
and R to reset. `reveal()` preserves a selected plan view. Camera-director tests already cover
starting the reveal top-down and interrupting a dolly with a view change.

Build a first-class grid-north-up plan view plus a coordinated east-versus-depth section. Both views
must show exactly the same population, appearance times, filter state and selection as the 3D scene.
They must consume `useBundle` with no additional fetching or new store/contract fields. Every event
position comes from ENU; every displayed depth comes from `(refSurfaceElevM - elevM) / 1000`.
Never use catalog published depth as display depth, rotate to true north, snap sensors to terrain,
or change inclusion because the depth gate failed.

### Implementation order

1. Fetch origin and inspect the integration state. If main advanced, merge it into feat/web before
   merging the refreshed lane baseline into this branch. Read the Gate S outcome and current
   requests; use the real showcase bundle through `?mode=showcase` when it exists. A missing bundle
   remains an explicit provider error. Keep the synthetic banner for mock data.
2. Add pure projection/layout helpers and tests under `scene/plan/`. Fit the plan to the same framed
   candidate population as the 3D view with a data-derived scale bar; include the selected event
   without silently changing membership. Define the depth section as all events projected onto east
   versus site depth, explicitly labelled as a projection (no invented slice width or spatial cutoff).
   Include borehole wellhead-to-sensor lines and finite error extents when fitting it.
3. Add an orthographic plan camera under `scene/plan/`, wired from H3's Canvas. Use east right and
   grid north up, exact top-down orientation, and equal horizontal scales. Reuse the existing
   instanced glyphs, filter driver, selection and reveal clock. Give the plan camera explicit
   ownership while active; update the H3 camera rig seam so its perspective preset/dolly and orbit
   momentum cannot overwrite the plan framing. Pan/zoom remain available, rotation does not.
   Restore the perspective camera and controls cleanly on P/reset, including rapid repeated toggles.
4. Render the coordinated depth section in a compact, labelled Canvas2D panel in `scene/plan/`.
   Precompute typed position/error arrays on bundle changes. Its frame callback reads existing
   appearance times, filter look and selected id, reuses scratch state, and allocates nothing.
   Use the same token colors and opacity rules as the event material; public points begin visible,
   candidates appear on the shared reveal clock, and STRICT keeps B/C at background weight.
   Give the panel a true-scale default; if a deliberate exaggeration is needed to fit, label its
   numerical factor permanently and independently of the 3D scene's factor. Use SceneMeta.depthLabel
   verbatim, data-derived ticks, grid-east distance and a visible explanation of the projection.
5. Plan halos must convey horizontal error whenever hErrM is usable, even if vErrM is null. Do not
   reuse the 3D ellipsoid's requirement for both errors. Missing uncertainty has no invented halo;
   the depth section omits only the unavailable vertical error. Implement the plan halo instances in
   H3 paths and preserve the existing 3D halo behavior. Use MAX blending so overlaps never brighten
   the uncertainty into event-like signals.
6. Coordinate picking from both views through `select(id)`. Keep the drawer's E/Escape behavior and
   the existing ring; use stable ids and the renderer's appearance/filter gates. Reserve space for
   shell controls and move/size the section so it never covers the evidence drawer or capture its
   pointer events. Datum, scale, synthetic warning and any exaggeration must remain readable at
   1280×720 and 4K. Use the stable scene label portal introduced by WEB-01 for 3D HTML labels.

### Acceptance and review before a PR

- Pure tests: grid-north/east orientation, equal plan scale, elevation-derived depths despite a
  conflicting depthKm, true borehole sensor positions, zero/null uncertainty, degenerate extents,
  aspect changes and deterministic framing. No contract changes or new synthetic bundle generator.
- State/browser sequences: P before reveal; P then Space; S then P then Space; P during the reveal;
  repeated P; resize in plan; E and click from both views; Escape; R while revealing; R after pan/zoom.
  Counts and appearance times must agree with the same providers/selector used in 3D. Reset restores
  the initial oblique frame and releases all camera/control ownership.
- Validate the insurance use case explicitly: missing vErrM still permits honest plan uncertainty;
  depth-on-edge flags and large/null vertical errors remain visible in evidence without asserting
  that depth is constrained. Gate S failure changes presentation priority, not data or scientific claims.
- Headed Chrome and Safari checks at 1280×720 and 4K: readable axes/datum, no panel/drawer overlap,
  no shader or page exceptions, no new network requests during reveal, and at least 60 fps with
  2,000 generator-produced events plus terrain/bloom while the depth panel runs. Record actual
  browser/device, DPR, median fps and slow-frame measurements. If the second view exceeds budget,
  optimize its drawing batches before reducing visual detail; do not silently thin events.
- Run `make check`, ticket acceptance and a full self-review against coordinates, frozen store,
  lane boundaries, determinism and allocation budget. Push and open the template PR into feat/web
  with Closes #27 only when the implementation is ready; manually close #27 after its approved merge.

### Handoffs to preserve

Main 2803a8a includes all four lanes, VAL-01, API-05 and the scheduler shutdown fix. H4 completed drawer mounting (REQ-H3-5), mock evidence
preload coverage (REQ-H3-3), and ENU-to-geographic UTM conversion with the matching DEM projection
(REQ-H3-4). The stock mock can now render the real terrain; retain `?terrain=slab` as the explicit
fallback. FEAT-01 reference features remain honestly unverified and must stay dashed.

The public Gate M check still needs the one-time Vercel connection and a deployment URL. Offline
static-export acceptance is distinct from that deployed gate. The refreshed shared run
`20260926-0210-a04c611` was fetched successfully on this laptop. Its 04:05:21 UTC archive includes
full-run picks and baseline outputs, but still lacks events, arrivals and matches. An actual export
fails on missing events.parquet. H2 must publish the completed event tables (REQ-H3-8); H1's waveform
cache is also absent locally (REQ-H3-9). The default make export invocation has a macOS Bash 3.2
empty-array failure (REQ-H3-7); passing the script's explicit --data-dir reaches Python safely.
The user authorized committing a validated generated showcase bundle directly to feat/platform,
but there is no generated bundle to commit yet. Use /Users/snp/hq-worktrees/platform-export for it.

After WEB-07, WEB-06 remains scheduled after Gate E and WEB-08 retains the full ten-run, two-browser
hardening checklist. The 1–5 AM no-merge window remains in force.

### Integrated baseline verification after main 544c869

H3 independently built the static export and tested it in headed Chrome at 1280×720, DPR 2: all
21 evidence preloads succeeded; core bundle files each loaded once; real terrain rendered without
fallback; reveal and STRICT counts matched provider/selector values; E opened the mounted hero
drawer (21.3 ms trace paint); validation, Run details and reset worked. There were zero external
requests, HTTP failures or page/console errors during the tested flow. Explicit `?terrain=slab`
also displayed its label. This verifies the local production export, not a Vercel deployment.

The web gate passes (404 web tests, 12 token tests, lint and typecheck). Full `make check` on this
laptop reports seismic 453 passed / 2 skipped and API 18 passed / 1 failed. The scheduler shutdown
failure reproduces alone and is reported to H4 in REQ-H3-6; do not mark the aggregate gate green.

### Latest integration and export checkpoint · 2026-09-26 01:03 EDT

Supersedes the failing 544c869 test result above. feat/web is synchronized and pushed at 2803a8a;
agent/WEB-07 merged that baseline before 1 AM. Full make check passes: seismic 465 passed / 2 skipped,
API 26, web 420, visualization tokens 12, with lint/typecheck/copy checks clean. The production build
passes. Headed Chrome offline acceptance again passed at 1280×720 DPR 2: 21 evidence preloads,
one fetch per core bundle file, actual terrain, monotonic reveal and exact STRICT count, E hero
(19.9 ms trace paint), panels and reset. No external requests, HTTP failures or page/console errors.
Explicit slab fallback is labelled. Dense feature-label overlap remains a WEB-08 polish item.
This is mock/local verification; the real showcase and public deployment are still unverified.

The failed export left feat/platform clean at 2803a8a. No source changes outside the H3 lane and no
bundle commit. The detailed next-agent handoff in the task's outputs directory includes release
identity, precise export errors, workaround, shared-data/worktree layout and retry steps. Browser
and local server were stopped after acceptance. WEB-07 remains plan-only; preserve 1–5 AM no merges.

### WEB-07 foundation checkpoint · 2026-09-26 03:34 EDT

The user asked to use the remaining usage and stop fully with saved work before the limit. H3 began
bounded implementation ahead of the original ~5 AM estimate; no merges occurred during 1–5 AM.

- `scene/plan/geometry.ts`: tested orthographic framing, independent untrimmed depth clipping,
  site-elevation depth projection, true-scale fitting of all candidates/catalog events/errors and
  borehole endpoints. It preserves input order and never uses published catalog depth for display.
- `scene/plan/halos.ts`: typed horizontal uncertainty instances use hErrM alone, preserve exact
  candidate positions/appearance times, and remain valid when vErrM is null. No invented error radius.
- `scene/picking/pick.ts`: orthographic stacked-event ties use NDC depth, because clip w is constant;
  perspective picking retains its original distance key. Three regression tests use a real camera.
- Full `make check` PASS: seismic 465 passed / 2 skipped, API 26 passed, web 453 passed, token tests 12;
  lint/typecheck/copy checks clean. New foundation coverage is 33 tests. Third-party SeisBench emitted
  one Python 3.14 invalid-escape SyntaxWarning during fresh environment import; checks still pass.
- Self-review fixed an edge-tolerance assertion, narrowed geometry input types to the fields read,
  and separated camera composition from full-data clipping so pan/zoom does not discard deep outliers.

This is a tested foundation, not a completed WEB-07. No PlanCamera, PlanHalos renderer or depth-section
panel is mounted yet, and no browser/performance acceptance is claimed for these new modules. Next:
explicit camera ownership, stable plan portal/panel layout, MAX-blended horizontal rings, allocation-free
Canvas2D depth projection, shared selection/filter/reveal, and the previously documented full acceptance.
Retain the existing SceneHtml portal and exact glyph clamp; the event shader already supports orthographic
projection. The current CameraRig assumes perspective preset ownership, so do not simply replace its
camera without suspending its director and OrbitControls. Test P/Space/R, rapid toggles and resize.

The release was replaced again at 07:27:29 UTC (03:27 EDT): 945,879 bytes, SHA-256
`8de669484030182f1efe650e3297619f82486abeaa7f69d75262300f3284d432`. It adds synthetic.json but drops
root picks and baseline outputs that were in the earlier 04:05 archive; events/arrivals/matches remain
absent. Scratch inspection did not extract over the earlier fetched run. REQ-H3-8 has an appended FYI.
Do not overlay versions into a misleading mixed run or infer a real-data depth gate from synthetic.json.

### WEB-07 completion checkpoint · 2026-09-26 04:10 EDT

Implemented on `agent/WEB-07` (6bd8d93 → 7e64650) on top of the 3038c14 foundation:

- **Camera ownership.** `plan/PlanCamera.tsx`: orthographic, straight down, grid north up (`up = [0,0,-1]`),
  MapControls pan/zoom, no rotation. Canvas mounts it *instead of* CameraRig when `view === "plan"`, so
  presets, the reveal dolly and orbit momentum can't touch it; drei restores the perspective camera and
  OrbitControls on P/reset. The pose is declarative (props): drei rebuilds MapControls for whichever camera
  is default at render time, and a one-shot effect had set the target on the stale perspective-bound
  instance (plan tilted ~7.4°). `PlanInvariant` now console.errors on entry unless the live camera is
  orthographic, straight down and grid-north-up; a probe build of the old code proved it fires.
- **Framing.** `plan/view.ts`: R3F's pixel frustum → zoom = px/km; the framed structure is centered in the
  region right of the depth-section panel (`plan/layout.ts`); clipping covers every event.
- **Horizontal uncertainty.** `plan/PlanRingsLayer.tsx` + `ringMaterial.ts`: hErrM-only rings (valid when
  vErrM is null; none without a usable hErrM), MAX blending, under the bloom threshold, same reveal/STRICT
  gates as the 3D halos; the 3D ellipsoids are hidden in plan.
- **Depth section.** `plan/section.ts` + `DepthSection.tsx`: docked Canvas2D panel, every event projected
  onto grid east vs `(refSurfaceElevM − elevM)/1000`, true scale, SceneMeta.depthLabel verbatim, boreholes at
  sensorElevM with wellhead lines, STRICT uncertainty arms only where errors exist, shared reveal clock,
  filter look and selection; click selects with the 3D picker's gates; redraws only on change. It carries
  the abstract-surface note in plan (the 3D ruler, a point seen end-on, and the stacked slices are hidden).
- **Layout.** The panel clears the drawer, title/Run details, counters and filter pills, REVEAL, and the
  bottom-left validation panel + mode pills at 1280×720 → 4K (unit tests + real DOM boxes in the browser).
- **Safari terrain (found here).** WebKit altered the grayscale `hillshade.png` on decode (non-deterministic
  checksum), so Safari fell back to the slab. The bake now stores the hillshade in R = G = B of an RGB PNG;
  height.png byte-identical, checksums unchanged; both engines load the real terrain.

Acceptance (static export, `NEXT_PUBLIC_ALLOW_MOCK=1`; 2,000 events from `scripts/mock-fixture.py` knobs,
local only): P before the reveal (public only in the section), Space from plan (section fills on the reveal
clock), STRICT (rings, B/C to background, arms), E hero + drawer (no overlap, also after resizing to
1920×1080 and back), Esc, pan + 7 rapid P toggles, R → exact start-frame hash (Chrome `c46fd1ca…`, WebKit
`b307d155…`), P mid-reveal from 3D, slab fallback note in plan, 4K layout; no page/console errors, HTTP
failures or external requests. Chrome 153 (Apple M4 Pro, 120 Hz): median 120 fps in every phase with 2,000
events + terrain + bloom + panel, 0 frames > 20 ms. WebKit 26.5: median 58.8 fps (rAF capped at 60),
0 frames > 33 ms. `make check`: seismic 465 passed / 2 skipped, API 26, web 492, tokens 12, copy clean.

WEB-08 items found while testing (not WEB-07 scope): the VE badge's CornerNote sits in the canvas's
bottom-left, under H4's mode pills (hidden today only because the mock's VE is 1); in 3D oblique the depth
ruler title and unverified-well labels can overlap; the true-scale section shows a compact cluster when
Tier C events scatter wide (a "fit structure" toggle would help, clearly labelled).

### WEB-08 hardening checkpoint · 2026-09-26 05:45 EDT

Fixed ahead of the Saturday QA pass (branch `agent/WEB-08`, PR into feat/web "Refs #29"; the ticket stays
open for the 2–6 PM pass with a fresh reviewer and a human on the demo laptop with the real bundle):

- **VE badge.** The permanent "Vertical ×N" corner note sat bottom-left, under H4's mode pills and validation
  panel. It is bottom-right now (`cornerNoteAnchor`), and the abstract-slab note stacks above it.
- **5-second test.** The depth ruler (line, ticks, title) drew over the opaque pre-reveal terrain. It now fades
  in with the reveal's terrain fade (`rulerRevealOpacity`) and out on reset. The slab note moved from the
  ruler title to the corner stack, permanent in every view including plan.
- **Label overlap.** In the side view at 1280×720 the mock's pad and well labels printed on top of each
  other. Feature labels are now placed per frame in screen space (`references/labelPlacement.ts`):
  - Default: right of the anchor.
  - A label moves to the next free slot if it would cover another label, the ruler's title or tick
    labels, another feature's anchor, or the plan panel, or would run off screen. Slots go left side
    first, then 1–3 lines up or down with a thin leader.
  - Labels are never hidden; with no free slot, the one with the least overlap wins.
  - Slots are held only while the camera moves, so a still camera always gets the same layout.
  - DepthRuler publishes its label boxes as obstacles each frame (priority −1). Sizes come from a
    ResizeObserver (`useLabelElements`). There are no per-frame allocations, and style is written only
    on change.

QA checklist on the dev machine (Apple M4 Pro, headed Chrome 153 + WebKit 26.5, mock bundle; the local
Playwright harness is outside the repo):

| Item | Status |
| --- | --- |
| 5-second test | pass: no ruler on the pre-reveal frame; terrain, geothermal reference, PUBLIC, REVEAL |
| Depth label and ruler in every preset | pass: side (after reveal), oblique, plan (depth-section axis title), 1280×720 and 3840×2160, both engines; zero label overlaps incl. pre-reveal and plan |
| Reveal identical ×10; reset → exact start | pass: 10/10 canvas hashes (Chrome `97a502f9…`, WebKit `84d63c41…`) **and** DOM label layout, both engines |
| STRICT count matches the shell | pass (43 / 500 / 150 on mock) |
| Drawer with 4 and 16 traces | pass: both engines, 1280×720 and 4K; every row inside the record, no overflow, 16 traces ≥ 13 px tall at 720p |
| Plan view finished | pass: WEB-07 acceptance re-run on this build |
| Chrome + Safari, 1280×720 and 4K, 60 fps | pass on dev (Chrome 120 fps median with 2,000 events; WebKit at its 60 Hz cap, 0 frames > 33 ms); **re-measure on the demo laptop** |
| "Not random dots" screenshot | **open**: needs the real showcase bundle (REQ-H3-8) |

`make check`: seismic 686 passed / 2 skipped, API 26, web 523, tokens 12, copy clean.

### WEB-06 checkpoint · 2026-09-26 10:30 EDT

Time mode (branch `agent/WEB-06`, PR into feat/web "Closes #28"). Semantics follow docs/02 §6, README ("T turns
on the time scrubber, which replays the window") and docs/01 (the scrubber "appears after the reveal"):

- **Replay.** Time mode takes effect only once the reveal has finished; a T on the pre-reveal frame changes nothing (the
  start frame stays pixel-identical). When it takes effect, `TimeDriver` (priority −2) sets `tNow = windowStart` and
  plays at `LOOK.time.playbackRate` (1 h/s: a day in 24 s), stopping at `windowEnd`. T again, or R, shows every event.
- **Gating everywhere.** An event is shown when `t <= tNow`, in every place that draws or picks events: the event shader
  (`aTime` / `uTimeNow`), the Tier A halos, the plan rings, the depth section (draw and click) and the picker. They all
  read one value, `sceneFx.timeNowRel` (seconds since windowStart; `TIME_ALL` when off). Events from the last
  `glowWindowS` (30 min of data time) glow brighter and slightly larger in their own hue (no whitening, so recent amber
  never reads as a white public event).
- **Scrubber** (`scene/time/TimeScrubber`, H4 mounts it: REQ-H3-10). Bottom band between the validation card and the
  corner notes (`time/layout.ts`; hidden while the open drawer leaves no room). Per-10-minute histogram, recovered above
  the baseline and the public regional catalog below, one shared scale, future bins dimmed, playhead; UTC clock;
  play/pause; "N public · M recovered" on screen so far (Tier A under STRICT). Drag or click to scrub (pauses); with the
  strip focused, arrows step a bin (Shift: an hour), Home/End. Every number is counted from the bundle.

Acceptance (`work/web06-time-check.mjs`, headed Chrome 153 and WebKit 26.5; a local-only page.tsx swap stands in for
REQ-H3-10): at 1280×720 and 3840×2160, T before the reveal leaves the start frame unchanged. After the reveal, T shows
0 events at windowStart (0 amber / 0 white pixels) and all of them at windowEnd; counts match the bundle at 12:00 and
under STRICT; playback measured 3,589–3,605 data-s/s and stops at windowEnd; the depth section is empty at windowStart
and full at windowEnd; the scrubber clears the shell and the drawer; R returns the exact start frame. With 2,000 events,
playback holds 120 fps (Chrome, 0 frames > 20 ms) and WebKit's 60 Hz cap (0 frames > 33 ms). The production page
without the scrubber mounted replays correctly with no errors. Time mode off is inert: the 10-run determinism hashes
equal WEB-08's in both engines. `make check`: seismic 778 / 3 skipped, API 26, web 575, tokens 12, copy clean.

### WEB-08 real-data checkpoint · 2026-09-26 11:05 EDT

The real showcase run (20260926-0210-a04c611, republished 12:07Z with events, arrivals and matches) exports on this
laptop up to evidence, which needs H1's waveform cache (REQ-H3-9: `CacheMissError`, nothing written). To QA the scene on
real data before the cache arrives, H4's exporter was run unchanged with a no-waveform source into scratch (a
local-only preview, never committed, never presented as evidence). Its shape is a compact near-vertical column several
km below the site, and it exposed four defects the mock never did, fixed on `agent/WEB-08` (#29 stays open):

- **Revealed frame.** The side preset framed only the event cloud, so the surface, the ruler's 0 km and the wellhead
  labels fell off the top. It now targets the middle of the column from the surface to the deepest framed event and
  frames the surface.
- **Depth section.** It fitted the whole population including stations (tens of km of grid east), so the structure was a
  few pixels wide. `sectionStructureFit` now frames Tier A and B (trimmed like the plan camera) from the surface down at
  true scale. Drawing is clipped to the plot, and the header states how many events lie outside the frame.
- **Label under the counters.** Feature labels now also keep clear of the shell's overlay blocks and H3's panels
  (`overlayObstacles.ts`: read-only rects of the shell root's children and H3 panels, re-measured 4× a second).
- **Ruler among Tier C.** `RULER_GAP_FRACTION` 0.12 → 0.3.

Verified on the mock and the real preview, headed Chrome 153 + WebKit 26.5:
- Strict preset label-overlap check passes.
- 10-run determinism passes in both engines. The start frames are unchanged. The revealed frames have new baselines,
  since the side view was reframed on purpose: Chrome `3c71a6c6…`, WebKit `f4f6a0ec…`.
- Full plan acceptance passes, including 2,000 events at 120 fps (Chrome) and the 60 Hz cap (WebKit).
- The real-bundle check passes (`PREVIEW_NO_EVIDENCE=1`: the evidence and E steps are skipped).

`make check`: seismic 778 / 3 skipped, API 26, web 584, tokens 12, copy clean.

### WEB-08 QA checkpoint on the real bundle · 2026-09-26 5:00 PM EDT

Real showcase bundle on main (981b746, run 20260926-0210-a04c611, v3), static export of feat/web, headed Chrome 153
and WebKit 26.5 on the dev machine (Apple M4 Pro). Local harness: `work/web08-qa-real.mjs` and `web08-determinism.mjs`.

A fresh reviewer subagent, started by the lane agent (not a separate session opened by a human), ran the lane doc's
reviewer prompt over the H3 paths on the real bundle. Findings and fixes:

- **Blocker, fixed.** 254 Tier C events have no evidence file (the exporter caps evidence at `maxEvents`), and their
  drawer printed the provider's raw "…json: HTTP 404" in red and hid the figures. The drawer now says "No waveform
  evidence was exported for this event." as a neutral note (other failures: a plain error, logged), and keeps the plan
  and depth figures with lines to the stations that picked the event.
- **Fixed.** Record-section caption: "N traces from the closest stations, k with a pick" (it claimed "the stations that
  picked this event").
- **Fixed.** Real tier reasons (300–700 characters) pushed the record section below the fold at 720p. They are clamped
  to two lines with "Show all reasons".
- **Fixed.** Before the reveal, clicking a public point opened its matched candidate's drawer. Public points now select
  only after the reveal.
- **Fixed.** Time playback allocated per frame: `stepTime` writes into a preallocated step, and the scrubber builds
  text only on change.
- **Fixed, polish.** Typographic minus on depth-section ticks. Copy without digits: "One-sigma … error", "true scale".
  Unverified labels read "Name · approximate". Dead code and stale comments removed.
- **Not changed.**
  - The ruler stops at 6 km; 15 Tier C events go deeper.
  - Five labelled features show before the reveal.
  - Tier reasons are verbatim pipeline text (e.g. "hErrM null (no formal error)"), reported to H2.

QA results on the dev machine:
- **5-second frame:** no ruler, no SYNTHETIC banner, PUBLIC = meta.
- **Counters:** RECOVERED and STRICT equal meta after the reveal; STRICT = `strictQualityCount`.
- **Drawer:** E opens the hero with 16 real traces. Null-field events (Tier C null hErrM/vErrM, depthOnEdge, null
  pickP/pickS, a non-hero catalog match, no evidence file) render without NaN/undefined/Infinity.
- **Plan view:** the depth section shows its outside count. The time scrubber shows 0 events at windowStart and all at
  windowEnd, and plays. Reset returns the exact start frame.
- **Speed:** 120 fps median in every phase in Chrome, and 58.8 in WebKit (its 60 Hz cap), at 1280×720 and 3840×2160.
- **Determinism:** 10/10 identical start and revealed frames (canvas + DOM labels) in both engines.
- **Still to do by hand:** fps on the demo laptop, and Safari itself (the runs used WebKit).

Manual run sheet for the demo laptop (Chrome, then Safari; 1280×720, then a 4K display):
1. `make offline` (or `NEXT_PUBLIC_ALLOW_MOCK=1 pnpm --filter web build` and serve `apps/web/out`); open `/` (showcase is
   the default mode). Expect no SYNTHETIC banner and the header "Showcase · <window> · run <id>".
2. 5-second frame: dark terrain, the glowing wells labelled "· approximate", the PUBLIC count, REVEAL. No depth ruler.
3. Press Space. The reveal runs about 7 s and the counters climb. It ends in the side view with 0 km and the wellhead
   labels at the top and the event column below.
4. fps: paste this in the console right after pressing Space, and again later in each view. Record the median and the
   frames over 20 ms:
   `(()=>{const d=[];let l=performance.now(),t=l;const f=n=>{d.push(n-l);l=n;if(n-t<5000)requestAnimationFrame(f);else{d.shift();const s=[...d].sort((a,b)=>a-b);console.log('median fps',(1000/s[s.length>>1]).toFixed(1),'frames>20ms',d.filter(x=>x>20).length,'of',d.length)}};requestAnimationFrame(f)})()`
5. Press S. STRICT equals the Validation card's strict count; B and C fade; Tier A halos are small, tight rings. Press S
   again to return to ALL.
6. Press E. The hero drawer shows "N stations agreed", Tier A, and 16 traces whose picks tick in. There is no
   NaN/undefined anywhere. "Show all reasons" expands and collapses. Press Esc.
7. Click a few faint Tier C points away from the column. Some show "No waveform evidence was exported for this event."
   with the plan and depth figures still drawn; none shows a URL or "404".
8. Press P for the straight-down plan view (north up). The left panel says "framed on Tier A and B" plus the outside
   count. Clicking an event in the panel opens the drawer. Press P to go back.
9. Press T. The scrubber appears at the bottom and replays from 00:00 UTC, with counts and histogram building. Drag the
   strip, use the arrow keys, play and pause. T closes it.
10. Press R: exactly the frame from step 2.
11. Reload three times and run step 3 each time: the revealed frame looks identical.
12. Resize to 1280×720 and to 4K. Repeat steps 3, 6, 8 and 9: nothing overlaps the counters, validation card, drawer or
    scrubber.
13. Repeat steps 1–12 in Safari.

### WEB-08 follow-up · depth ruler from the data · 2026-09-26 5:30 PM EDT

At H2's request, the depth ruler and the depth slices now reach the deepest displayed event instead of a fixed
6 km. `rulerMaxDepthKm` takes the candidates' and the public catalog's `elevM`, turns them into display depth
((refSurfaceElevM − elevM)/1000), and rounds up to a whole tick, with a minimum of one tick. On the real v3 bundle
that is 9 km: 15 Tier C events lie deeper than 6 km, the deepest at 8.63 km. The label-obstacle list is sized
from the same depth. The five pre-reveal feature labels and the verbatim tier reasons stay as they are (H2's
call).

### WEB-09 Tour (story mode) · 2026-09-26 6:45 PM EDT

H2's first overnight item: the judge sequence plays by itself, for a presenter who wants hands off and for
recording the video. **G** starts it (H4's Tour button too, once mounted: REQ-H3-15). Steps, in `scene/tour/script.ts`:

| Step | Store action (as the presenter's keys would) | Holds for |
| --- | --- | --- |
| public | `reset()`: the start frame, pixel-identical to a fresh load | 4.5 s |
| reveal | `reveal()` | until phase "revealed" (the settle, ≈7 s), then 3 s |
| strict | `setFilter("strict")` | 6.5 s |
| hidden | `selectHiddenHero(events)`: the drawer on a Tier A event with no catalog match (skipped if none) | 8 s, then `select(null)` |
| validation | outlines H4's Validation card (skipped if not on screen) | 6 s |
| replay | `setTimeMode(true)`: the window replays at `LOOK.time.playbackRate` | until it stops at the window end, then 1.5 s |
| full | `select(null)`, `setTimeMode(false)`, `setFilter("all")` | 6.5 s, then the tour ends |

- **Stopping.** Any key, click or scroll stops it and leaves the scene where it is. The key is captured on
  `window` before the shell's presenter keys and prevented, so it does nothing else. Bare modifiers and
  Cmd/Ctrl/Alt shortcuts don't stop it, so a screen recorder's hotkeys can't end a take. Clicks on the tour's
  own button (`data-tour-control`) are the button's.
- **Captions.** One per step, from `scene/tour/copy.ts`, the copy file **H4 may edit** (REQ-H3-15; the rest of
  `scene/tour/` stays H3's). Numbers come only from `{placeholders}`, filled from `meta.summary`, the hidden
  event's `quality.nStations` and the playback rate. They are drawn in the counters' mono face and tones (public
  cool white, pipeline amber). `copy.test.ts` fails on a digit, a docs/00 forbidden phrase, an unqualified
  "catalog", an unknown placeholder or a caption over 170 characters. The caption is a lower third
  (`tour/layout.ts` `captionBox`), centered, and keeps clear of every measured overlay: it steps up above the
  scrubber or REVEAL when either is under the center, and sits left of the open drawer. Scene labels avoid it
  too, through `H3_OVERLAY_SELECTORS`. It fades in on each step, except under `prefers-reduced-motion`.
- **State.** A separate `useTour` store (running, step, runs), so docs/02 §6's `DemoState` stays as frozen.
  `TourRun` is pure: the clock is passed in, and the overlay ticks it once per animation frame, so a hidden tab
  pauses the tour with the scene.

### WEB-11 · H opens the hidden hero · 2026-09-26 9:45 PM EDT

The live site had no key for the hidden hero (REQ-H3-12 was still open). H3 binds **H** in the scene itself
(`scene/HiddenHeroKey.tsx`), the same way the tour binds G. It calls `selectHiddenHero(events)` once the reveal
has begun, and does nothing on the public frame, where no candidate event is on screen to open. It follows the
shell's key rules (`scene/keys.ts` `isPlainPress`: text fields, repeats and Cmd/Ctrl/Alt combos are left alone).
While the tour plays, H stops the tour like any other key and opens nothing. H4 lists H in docs/02 §6 and the README.

### WEB-10 · hover tooltip on any event · 2026-09-26 10:00 PM EDT

H2's overnight item 2. Hovering a glyph shows a small card that follows the pointer (`scene/picking/HoverTooltip.tsx`,
content in `picking/hover.ts`), formatted with the drawer's own functions so the two never disagree.

- **Candidate event:** tier ("Tier A (strict)"), origin time to the second in UTC, depth below the site surface (the
  display-depth rule), magnitude with its type and sigma, stations that agreed, and "In / Not in the public regional
  catalog".
- **Public regional catalog event:** time, depth and the catalog's magnitude. After the reveal it also says whether
  our pipeline recovered it, and at what tier. Before the reveal it doesn't, so the reveal isn't given away.

The Picker's hover pick (at most once per frame, only while the pointer moves) sees any drawn glyph, including public
points before the reveal. The click pick is unchanged, so public points still select nothing before the reveal. The
card flips to the pointer's other side at the viewport edge and before the open drawer. It clears on a drag, a zoom
and on any phase, filter, time-mode or view change, and stays hidden while the tour plays. On the real bundle
(`work/web10-hover-check.mjs`) every value matches the bundle in Chrome and WebKit, and a pointer sweep across the
cluster costs no frame rate.

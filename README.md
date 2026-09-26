# Hidden Quakes

HackGT 13 · Oracle of the Deep · Friday 8:00 PM → Sunday morning (~36 h)

This repo was created at 8:00 PM by `bootstrap.sh`. Its first commit holds the plan, the rules and an empty skeleton; every line of code after that is written by our agents during the event. Every GitHub issue is one ticket from a lane doc.

## Start here (every teammate)

1. Accept the GitHub invite for this repo.
2. Open your prompt: `docs/prompts/H1.md`, `H2.md`, `H3.md` or `H4.md` (who's who is in `docs/team.md`).
3. Paste the prompt into a fresh agent session on your laptop. The agent clones the repo, reads the plan, lists your issues, and starts on your first ticket.

That prompt makes your agent the lead for your lane. For extra parallel sessions later, use the per-ticket prompt at the bottom of your lane doc.

## Layout

```
CLAUDE.md                          shared rules every agent follows (AGENTS.md links to it)
Makefile                           make check · make publish-run · make fetch-run · make runs
.github/CODEOWNERS                 who owns which paths
.github/pull_request_template.md   what every PR must show before it merges
docs/
  team.md                          who is H1–H4
  prompts/H1.md … H4.md            the one prompt each teammate pastes into their agent
  00-project.md                    what we're building, what we may and must not claim, scope
  01-architecture.md               pipeline, repo tree, ownership, conventions, handoffs, data sharing
  02-contracts.md                  every cross-lane interface: data models, files, APIs, web hooks
  03-schedule.md                   gates, clock times, freezes, kill switches
  lanes/H1-signal.md               waveforms → picks
  lanes/H2-seismology.md           picks → located, tiered, matched events
  lanes/H3-visualization.md        the 3D scene, the reveal, the evidence drawer
  lanes/H4-platform.md             contracts, providers, exporter, deploy, story
  demo/pitch-and-qa.md             pitches, hostile Q&A, Devpost and README outlines
  requests/README.md               how one lane asks another for something
apps/web/                          Next.js app (scaffolded at 8:00 PM)
services/seismic/                  Python pipeline package `hq` (one subpackage per lane)
services/api/                      live worker (P1)
packages/contracts/                shared models: Python source of truth + generated TS
packages/visualization/            design tokens
```

## How four lanes stay out of each other's way

1. **Paths.** Every path has exactly one owner (`CLAUDE.md`, `docs/01`, `CODEOWNERS`). Agents write only inside their lane.
2. **Contracts.** Every cross-lane interface lives in `docs/02` and freezes at 8:30 PM Friday. Changing one takes a PR approved by every lane it touches.
3. **Stubs.** Every lane can run against a stub or mock of what it consumes, so nobody waits on anybody.
4. **Requests.** Need something from another lane? Write it in `docs/requests/`. Never edit their code.
5. **Data travels by command, not by git.** Run tables move with `make publish-run` / `make fetch-run`; the web bundle comes through `main`.

## Working pattern

- **One ticket = one issue = one branch.** Agents work on `agent/<TICKET>` in their own worktree, cut from the lane's `feat/<lane>` branch.
- **Every ticket gets a reviewer.** Before a PR, a fresh reviewer agent checks the diff against the lane doc and the ticket's acceptance test.
- **Humans approve merges** into `feat/<lane>`. H4 merges `feat/*` into `main` at each integration meeting.
- **Time is generous; the critical path isn't.** The science chain is serial and data-dependent. Spend slack on tests, review and polish, never on new scope.

## Compliance

Everything in the first commit is planning material and an empty skeleton, written before the coding period. All code, fixtures and outputs are produced after 8:00 PM by agents working from these docs. Nothing from pre-event prototypes enters the repo, and git history starts at 8:00 PM. Where a doc shows a schema or signature, it's a spec the agent implements, not code to paste.

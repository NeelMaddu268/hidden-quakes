# Requests between lanes

Lanes never edit each other's files. When your agent needs something from another lane, it writes a request here instead, and tells its human.

## Where

One file per owner: `docs/requests/H1.md`, `H2.md`, `H3.md`, `H4.md`. Create the file if it doesn't exist yet. Append only; never rewrite someone else's entry.

## Format

```
### REQ-<your lane>-<n> · <short title>
- From: H3 (agent on WEB-05)
- Needed by: 12:00 AM Sat
- What: `EventEvidence.traces[].predP` is always null in the mock bundle; drawer can't draw moveout lines
- Why it blocks: WEB-05 acceptance needs predicted arrivals
- Status: open
```

The owning lane changes `Status:` to `accepted`, `done (<commit>)` or `declined: <reason>`.

## Rules

- **Contract changes** go through a PR to `packages/contracts/` approved by every affected lane, not through a request alone.
- **Blocking requests:** ping the owner in person or chat too. A file nobody reads doesn't unblock anyone.
- **While you wait,** keep working against the stub named in `docs/01` → Handoffs.

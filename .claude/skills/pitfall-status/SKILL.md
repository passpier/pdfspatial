---
name: pitfall-status
description: Report the live Stage 3 pitfall scoreboard and whether the generated docs (roadmap checklist, fixtures/README coverage tables, README status row) are in sync with it. Read-only. Use when the user asks "where do we stand on the pitfall checklist", "is the scoreboard in sync", or before/after a Stage 4 heuristic change.
---

Read-only status pass over the Stage 3 pitfall scoreboard. Makes no code or doc edits.

1. Run:
   ```sh
   cargo run --quiet --example stage3_scoreboard --features stage3
   ```
   Report the global `P / N cases failing` line and, for any failing case, its pitfall
   and mismatch(es) — group by pitfall (`docs/pitfall_registry.json`'s `title` field
   gives the human-readable name for each slug).

2. Run:
   ```sh
   cargo run --quiet --example stage3_scoreboard --features stage3 -- --check
   ```
   - Exit 0: tell the user the roadmap, `fixtures/README.md`, and `README.md` generated
     blocks match the live corpus.
   - Exit 1: it prints which files are stale. Offer to run
     `cargo run --example stage3_scoreboard --features stage3 -- --write` to
     regenerate them, but don't run it without the user's go-ahead — write actions
     aren't part of a status check.

3. If the user wants per-pitfall detail instead of the aggregate, re-run with
   `-- --format json` and pick out the named pitfall's `cases` array.

Do not edit `crates/pdfspatial-core/src/**`, `fixtures/**`, or any of the four docs in
this skill — that's `/pitfall-fix`'s job, not this one.

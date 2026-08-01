---
name: pitfall-fix
description: Autonomously iterate Stage 4 heuristic fixes for one Stage 3 pitfall (assemble.rs/layout.rs/extract.rs) until the corpus scoreboard for it is green, a hard stop condition is hit, or the pitfall is refused as vision-model-blocked. Edits crates/pdfspatial-core/src/**, never commits. Use when the user says "fix <pitfall>", "work on the <slug> pitfall", or wants a Stage 3→4 loop run for a specific checklist item.
argument-hint: <pitfall-slug>
---

Runs a bounded, file-backed loop that turns one Stage 3 pitfall from failing/partial to
passing on the regression corpus, or exhausts its hypotheses and reports why. Edits
`crates/pdfspatial-core/src/**` directly across iterations; **never runs `git commit`**
— the user reviews and commits the final diff themselves.

## 0. Resolve the slug and preflight

The argument is a `Pitfall` slug (`multi_column`, `footnote`, `header_footer`,
`multi_line_table_cell`, `merged_table_cell`, `borderless_table`, `nested_formula`,
`super_subscript`, `rotated_text`, `figure_caption`, `list_nesting`,
`cross_page_continuation`, `section_header_vs_bold`, `embedded_font`,
`overlapping_text` — see `fixtures/README.md`). If no argument was given or it doesn't
match one of these, ask which pitfall to work on.

Read `docs/pitfall_registry.json`'s entry for the slug. **If `blocked.loop_refuses` is
`true`, stop immediately** — print `blocked.reason` and `blocked.unblocked_by`, make
zero edits, and tell the user this pitfall needs the out-of-scope ONNX vision-model
detector (`CLAUDE.md`: "an intentional, documented `unimplemented!()` stub — it's out
of scope by design, not a bug or a TODO to silently fill in"). Do not touch
`layout::classify_regions`'s `unimplemented!()` path to work around this.

If `blocked` is present but `loop_refuses` is `false` (extraction-layer gaps,
cross-page stitching, the font-weight schema gap, footnote serialization), the loop may
proceed — read `blocked.unblocked_by` as a hint for where the fix likely lives.

## 1. Set up the ledger

Ledger path: `.claude/pitfall-loop/<slug>.md` (create the directory if needed; it's
gitignored, so this never needs cleanup). If it already exists, read it in full first —
a prior run's "Exhausted hypotheses" section is binding: do not re-attempt a listed
hypothesis.

If the ledger is new, seed it with today's baseline:
```sh
cargo run --quiet --example stage3_scoreboard --features stage3 -- --format json
```
Record the target pitfall's `passing`/`total` and the global `passing`/`total` as the
`# <slug> — baseline P/N this pitfall, P/N global` header.

## 2. Iteration protocol (repeat until a stop condition below is hit)

Each iteration is identical:

1. **Baseline.** Re-run `--format json`, note current target and global `passing`/`total`.
2. **Plan.** From the target pitfall's `cases[].mismatches` in that JSON (and
   `docs/pitfall_registry.json`'s `note`/`blocked` context), read the responsible
   heuristic — `classification` root cause points at `layout.rs`, `ordering` at
   `assemble.rs`, `geometric` at `extract.rs` (per `CLAUDE.md`'s module list). State
   one smallest hypothesis not already in the ledger's "Exhausted hypotheses" section.
   Read the actual case JSON under `fixtures/<slug>/` to see the exact geometry/text
   driving the mismatch before touching code.
3. **Implement.** Make the smallest edit that plausibly fixes the stated hypothesis.
4. **Verify.** Invoke the `verify` skill (fmt → clippy → tiered tests). If it fails,
   treat this as a failed iteration (step 6's gate can't pass without green verify).
5. **Re-measure.** Re-run `--format json`.
6. **Gate:** target `passing` strictly increased **and** global `passing` did not
   decrease **and** verify was green.
   - **Pass:** run `cargo run --example stage3_scoreboard --features stage3 -- --write`
     to resync the generated docs, append an iteration entry to the ledger (hypothesis,
     files touched, verify result, score delta, verdict KEPT), and continue to the next
     iteration if the pitfall isn't fully green yet.
   - **Fail:** revert the touched files (`git checkout -- <files>`, only the files this
     iteration touched — never a broader reset), append the hypothesis to the ledger's
     "Exhausted hypotheses" list with a one-line reason, and continue to the next
     iteration.

## 3. Stop conditions

Stop the loop (do not start another iteration) when any of these hold:

- The target pitfall's `passing == total` on the corpus — success.
- Two consecutive iterations both hit the gate-fail branch.
- 5 iterations have run this session.
- The only remaining hypothesis would require the vision-model stub or another
  out-of-scope prerequisite — treat this like the step-0 refusal from here on.

## 4. Report

Always end with:
- The scoreboard delta for this pitfall and globally (baseline → final).
- The current uncommitted diff (`git status --porcelain` + a summary of what changed),
  left unstaged and uncommitted for the user to review.
- The ledger path, so the user (or the next `/pitfall-fix <slug>` run) can see the full
  iteration history.
- If stopped short of green: a proposed `blocked` entry (reason/unblocked_by) for
  `docs/pitfall_registry.json`, for the user to accept or reject — do not edit the
  registry yourself.

For an unattended run, the user can wrap this as `/loop /pitfall-fix <slug>`; the
ledger is what makes a fresh loop firing resume exactly where the last one left off
instead of re-deriving state.

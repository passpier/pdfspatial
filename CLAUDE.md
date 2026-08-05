# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`pdfspatial` is a Rust workspace implementing a four-stage PDF→Markdown pipeline (extraction →
layout classification → reading-order assembly → serialization) grounded in spatial
(bounding-box) extraction rather than layout guesswork, plus a Stage 5 comparative-benchmarking
gate that runs outside that loop. Two workspace members: `crates/pdfspatial-core` (the library —
Stages 1-4) and `crates/pdfspatial-cli` (the `pdfspatial` binary — single-file and batch
PDF→Markdown conversion, built on `pdfspatial-core::pdf_to_markdown` /
`serialize::to_markdown_pipeline`; this is what Stage 5's bench harness drives).

- Stage 1 (baseline extraction) and the algorithmic core of Stages 2/4 (metrics, heuristic layout
  classifier, XY-cut reading-order assembly, Markdown serializer) are fully implemented, pure,
  dependency-free Rust — no ML runtime involved.
- Stage 1b (`extract::extract_graphics`, `graphics.rs`) pulls non-text page objects (ruling
  lines, images, fills) from PDFium's page-object API and deterministically detects `Table`
  and `Picture` regions from them — no vision model needed for those two classes. This closed
  a real gap: earlier revisions of this doc (and `docs/pitfall_registry.json`) claimed `Table`/
  `Picture`/`Formula` were all unreachable without a vision model, which was true only because
  `extract.rs` used to consume just PDFium's text layer (`page.text()`), never its page-object
  API (`page.objects()`) — an unused-API gap, not a fundamental PDFium limitation.
- `RegionClass::Formula` is the one region class still genuinely vision-model-shaped: a formula
  has no ruling-line or XObject signal to key a geometric heuristic off, unlike a table or
  picture. The ONNX RT-DETR layout detector the roadmap's Stage 4b describes for it remains
  unimplemented and out of scope by design — not a bug or a TODO to silently fill in. (There is
  no literal `unimplemented!()` in the source; the gap is simply the absence of that code path.)
- Propose a plan before implementing non-trivial changes rather than diving straight into code.
- Stage 5 (`bench/opendataloader/`, `scripts/run-opendataloader-bench.sh`) benchmarks
  `pdfspatial` against other local, model-free PDF→Markdown engines on the real,
  external opendataloader-bench corpus (200 PDFs) and publishes a hardware-labelled
  results table in `README.md`'s `## Benchmarks` section, backed by a committed
  `bench/opendataloader/results/results.json`. **This never runs in CI** — it needs
  `uv`/Python 3.13, network access, and an otherwise-idle machine for the speed column
  to mean anything. See `bench/opendataloader/README.md` for the reproduction
  methodology and `docs/benchmark-analysis.md` for the metric-by-metric analysis and
  known scoring asymmetries before trusting the numbers too literally.

## Build, test, lint

- `cargo test` — runs `tests/stage2_pipeline.rs` only (synthetic in-code fixtures, no PDFium or
  feature flags needed).
- `cargo test --features doclaynet` — also runs the DocLayNet dataset harness
  (`tests/stage2_doclaynet.rs`), scoring `layout::classify_regions` against a vendored fixture.
- `cargo test --features stage3` — also runs the Stage 3 regression corpus harness
  (`tests/stage3_corpus.rs`), which loads hand-authored, minimal-repro cases from `fixtures/`
  (via `eval::corpus`) and checks their integrity, including 6 PDF-backed cases whose
  `page`/`pages` is a frozen `extract_baseline` snapshot (no PDFium needed just to load
  them). The corpus's behavioral spec (`corpus_cases_meet_expected_behavior`) is
  `#[ignore]`d — it asserts each case's *desired* post-Stage-4 behavior and fails today
  by design; run `cargo test --features stage3 -- --ignored` to print the full
  scoreboard. See `fixtures/README.md`. `tests/stage3_mining.rs` (gated on `doclaynet`
  AND `stage3`, so run by `cargo test --all-features`) exercises the ranked-page →
  minimal-repro → draft-case mining pipeline (`eval::minimize_reorder_repro`,
  `eval::corpus::write_draft_case`) end to end against the vendored DocLayNet fixture.
  `tests/stage3_pdf_fixtures.rs` (also gated on `stage3`, but needs a native PDFium
  library to do real work — see the next bullet) re-extracts each PDF-backed case's
  source PDF and checks the committed snapshot hasn't drifted; without PDFium set up it
  prints a skip notice unless `PDFSPATIAL_PDFIUM_LIB`/`CI` is set, in which case it fails
  loudly instead. `examples/stage3_pdf_cases.rs` (also gated on `stage3`) is what
  produces the 6 fixture PDFs under `crates/pdfspatial-core/tests/fixtures/stage3/` and
  their frozen corpus snapshots in the first place — see `fixtures/README.md`'s
  "PDF-backed cases" section before touching either.
- `cargo test --all-features` (matches CI) — also runs `tests/stage1_baseline.rs`, which
  **requires** a native PDFium library: set `PDFSPATIAL_PDFIUM_LIB` to the lib file/dir, or place
  it on the OS dynamic-loader path (`DYLD_LIBRARY_PATH` on macOS). `pdfium-render` does not bundle
  this library. `scripts/fetch-pdfium.sh --write-cargo-config` downloads a pinned release and
  writes a gitignored `.cargo/config.toml` so this works with no shell export; CI's PDFium
  download step calls the same script. `scripts/` follows the same workspace-root convention as
  `benches/`/`examples/` (see below), not the crate's own layout.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` and
  `cargo fmt --all -- --check` — both enforced in CI; no custom `rustfmt.toml`/`clippy.toml`, so
  defaults apply. `lib.rs` also sets `#![warn(missing_docs)]` and `#![warn(clippy::all)]`.
- `cargo bench --features doclaynet` — Criterion benches for the Stage 2 eval pipeline and metric
  primitives (`benches/`, see `benches/README.md`). Set `DOCLAYNET_DIR=/path/to/doclaynet` to bench
  against a real unpacked DocLayNet sample instead of the vendored fixture.
- Bench and example targets (`benches/`, `examples/`) live at the workspace root, not inside the
  crate — declared via relative paths in the crate's `Cargo.toml`, not the standard Cargo layout.

## Conventions

- All crate errors flow through a single `PipelineError` enum (`thiserror`).
- Modules: `extract`, `layout`, `assemble`, `metrics`, `serialize`, `eval` (with `eval::doclaynet`
  gated behind the `doclaynet` feature and `eval::corpus`/`eval::scoreboard` gated behind the
  `stage3` feature).
- Public APIs carry doc comments with runnable `# Examples` doctests — keep this pattern for new
  public functions.
- `Cargo.lock` is gitignored (this is a library crate).

## Stage 3 pitfall scoreboard — generated, not hand-maintained

The roadmap's Document-Structure Pitfall Checklist, `fixtures/README.md`'s coverage
tables, and `README.md`'s corpus status row are **generated** from the live Stage 3
regression corpus by `eval::scoreboard` / `examples/stage3_scoreboard.rs`, spliced into
each doc between `<!-- BEGIN GENERATED: name --> … <!-- END GENERATED: name -->`
markers. **Never hand-edit the text inside those markers** — it will be silently
overwritten by the next `--write` and, before that, flagged as stale by
`tests/stage3_docs.rs::generated_blocks_are_in_sync` (part of `--features stage3`, so
CI's `--all-features` run enforces it) and by a `Stop` hook that runs `-- --check`
whenever a turn touched `crates/pdfspatial-core/src/` or `fixtures/`.

- `cargo run --example stage3_scoreboard --features stage3` — text scoreboard (same
  report `cargo test --features stage3 -- --ignored` panics with).
- `-- --format json` — machine-readable per-pitfall/per-case detail.
- `-- --write` — regenerate every doc's generated blocks from the live corpus.
- `-- --check` — exit 1 (with a diff) if any doc has drifted; no exit-code doc write.
- Human judgement that no test can derive (why a corpus-green pitfall is still only
  `[~]` partial, what unblocks a blocked pitfall, PDF-backed rationale) lives in
  `docs/pitfall_registry.json`, one entry per `assemble::Pitfall` slug, merged in by the
  same generator — edit that file by hand, never the generated Markdown.
- `.claude/skills/pitfall-status` is a read-only status pass over the scoreboard/sync
  state; `.claude/skills/pitfall-fix` is a bounded, file-backed loop
  (`.claude/pitfall-loop/<slug>.md`, gitignored) that iterates Stage 4 heuristic fixes
  for one named pitfall until its corpus cases pass, a stop condition is hit, or the
  pitfall is refused outright (any pitfall whose `docs/pitfall_registry.json` entry sets
  `blocked.loop_refuses: true` needs the out-of-scope vision-model stub, not a
  heuristic — see the note above about `unimplemented!()`). It edits
  `crates/pdfspatial-core/src/**` across iterations but never runs `git commit`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`pdfspatial` is a Rust workspace with a single crate, `crates/pdfspatial-core`, implementing a
four-stage PDF→Markdown pipeline (extraction → layout classification → reading-order assembly →
serialization) grounded in spatial (bounding-box) extraction rather than layout guesswork.

- Stage 1 (baseline extraction) and the algorithmic core of Stages 2/4 (metrics, heuristic layout
  classifier, XY-cut reading-order assembly, Markdown serializer) are fully implemented, pure,
  dependency-free Rust — no ML runtime involved.
- The vision-model layout detector (ONNX RT-DETR, for `Table`/`Picture`/`Formula` region classes)
  is an intentional, documented `unimplemented!()` stub — it's out of scope by design, not a bug
  or a TODO to silently fill in.
- Propose a plan before implementing non-trivial changes rather than diving straight into code.

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
  gated behind the `doclaynet` feature and `eval::corpus` gated behind the `stage3` feature).
- Public APIs carry doc comments with runnable `# Examples` doctests — keep this pattern for new
  public functions.
- `Cargo.lock` is gitignored (this is a library crate).

# pdfspatial

A closed-loop, four-stage pipeline for turning PDFs into Markdown, grounded in spatial
(bounding-box) extraction rather than layout guesswork. Built in Rust on top of
[`pdfium-render`](https://docs.rs/pdfium-render), the idiomatic Rust wrapper around
Google's PDFium engine.

This repository currently ships one crate, [`pdfspatial-core`](crates/pdfspatial-core),
implementing **Stage 1** in full, the **algorithmic core of Stages 2 and 4**
(validation metrics, a deterministic heuristic layout classifier, column-aware
reading-order assembly, and structural Markdown output — all pure, dependency-free
Rust), and a **seeded Stage 3 regression corpus and harness**. The one piece still
unimplemented is the vision-model layout detector the roadmap describes for Stage 2/4b
(an ONNX RT-DETR-style detector over rendered page rasters, needed for
`Table`/`Picture`/`Formula` classes); it remains a documented `unimplemented!()` stub
since it needs an inference runtime, model weights, and a DocLayNet-backed evaluation
harness that are out of scope for this pass.

## The four-stage loop

```
PDF → [pdfium: char/word bboxes + page raster] → [Layout Model: region classification]
    → [Reading-order assembly] → [Markdown serializer] → .md
```

1. **Baseline extraction** (implemented) — a deterministic, OCR-free text-extraction
   floor built directly on PDFium's native text layer: char → word → line → block
   grouping via geometric heuristics only (no ML). See
   [`crates/pdfspatial-core/src/extract.rs`](crates/pdfspatial-core/src/extract.rs).
2. **Validation** (metrics and dataset harness implemented) — score structural fidelity:
   TEDS, TEDS(IOU), and TEDS-Struct (via a restricted table-HTML parser and a
   Zhang–Shasha tree-edit-distance implementation), plus GIoU and per-class region F1.
   Wired up against held-out [DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1)
   samples via the `eval` module (`crates/pdfspatial-core/src/eval/`, gated behind the
   `doclaynet` cargo feature) and scored into Criterion benches in
   [`benches/`](benches). TEDS itself stays exercised only by its own unit tests, since no
   table-structure predictor yet exists to feed it real predictions. See
   [`metrics.rs`](crates/pdfspatial-core/src/metrics.rs) and
   [`eval/`](crates/pdfspatial-core/src/eval).
3. **Error analysis** (seed corpus + harness implemented) — a reproducible failure-mode
   taxonomy (multi-column gutters, footnotes, borderless tables, ...), each pitfall tied
   to a minimal-repro regression case tagged with a root cause (`geometric`,
   `classification`, `ordering`). See [`assemble.rs`](crates/pdfspatial-core/src/assemble.rs)
   for the `Pitfall`/`RootCause` taxonomy, [`eval/corpus.rs`](crates/pdfspatial-core/src/eval/corpus.rs)
   (gated behind the `stage3` feature) for the loader/checker, and
   [`fixtures/`](fixtures) for the corpus itself — 19 seeded cases across 9 pitfalls
   reachable through the pure-Rust classify/assemble surface, with the remaining
   extraction-layer pitfalls documented as deferred pending real PDFium/DocLayNet
   fixtures. Each case's expected behavior is the desired *post-Stage-4* outcome, so the
   corpus's behavioral test is `#[ignore]`d and fails today by design for 18 of the 19
   cases — the exception, a `multi_column` case with a gutter wide enough for the
   XY-cut heuristic to recover correct order, quantifies the roadmap's predicted
   reading-order edit-distance degradation (via `metrics::reading_order_edit_distance`,
   now surfaced by `eval::corpus::CaseOutcome`) alongside its recovery. See
   `fixtures/README.md`. The same naive-vs-assembled edit-distance delta now also drives
   real-sample mining: `eval::rank_pages_by_reorder` (and, behind the `doclaynet`
   feature, `eval::doclaynet::mine_reading_order_failures`) ranks a DocLayNet sample's
   pages by how heavily `assemble_reading_order` reordered them — an unsupervised proxy
   for "which pages most likely have a Stage 1 reading-order failure worth mining into a
   `fixtures/` regression case next," since DocLayNet itself ships no gold reading order.
   See `examples/doclaynet_mine.rs`. `eval::minimize_reorder_repro` and
   `eval::corpus::write_draft_case` carry that ranking the rest of the way: shrinking a
   ranked page to the handful of blocks that actually drive the reordering and emitting
   it as an unreviewed **draft** `fixtures/`-schema case (`"draft": true`) for a human to
   verify, re-tag, and promote. Both examples accept real DocLayNet-core's on-disk
   naming/schema unmodified, and an opt-in `--grouped` flag reconstructs pages via Stage
   1's real word/line/block grouping (`extract::group_chars_into_blocks`,
   `eval::doclaynet::document_from_cells_grouped`) instead of one-cell-per-block, for more
   legible drafts at the cost of merging text across column gutters like real Stage 1
   does. See `examples/doclaynet_drafts.rs` and `fixtures/README.md`'s "Mining drafts from
   a real sample" section.
4. **Refinement** (heuristics implemented; fine-tuning stubbed) — [`layout.rs`](crates/pdfspatial-core/src/layout.rs)
   classifies regions with deterministic text-layer heuristics (Title, SectionHeader,
   ListItem, Caption, PageHeader/Footer, Text — `Table`/`Picture`/`Formula` need the
   still-unimplemented ONNX detector) and [`assemble.rs`](crates/pdfspatial-core/src/assemble.rs)
   reorders blocks via column-aware XY-cut recursion. Targeted model fine-tuning,
   validated against a Stage 3 regression corpus, remains future work.

After Stage 4, the loop returns to Stage 2 on a fresh held-out split — this is a
continuous cycle, not a linear pipeline.

## Stage 1 metrics (exit criteria)

| Metric | Target |
|---|---|
| Character extraction recall | ≥ 99% |
| Line-grouping accuracy (single-column) | ≥ 95% |
| Throughput | Recorded only — establishes the perf floor for later stages |
| Reading-order edit distance | Recorded only — expected to be poor on multi-column layouts; drives Stage 3 |

**Exit criterion:** bbox extraction is lossless and fast on single-column, non-tabular
PDFs. Multi-column layouts, tables, and formulas are *expected* to fail at Stage 1 —
characterizing that failure surface is Stage 3's job.

## Getting started

### Prerequisites

`pdfium-render` binds to the native PDFium library at run time; it is not bundled with
this crate. Quickest path (macOS/Linux):

```sh
./scripts/fetch-pdfium.sh --write-cargo-config
cargo test --all-features
```

This downloads a pinned release from
[bblanchon/pdfium-binaries](https://github.com/bblanchon/pdfium-binaries/releases) into a
gitignored `third_party/pdfium/`, and `--write-cargo-config` writes a gitignored
`.cargo/config.toml` that sets `PDFSPATIAL_PDFIUM_LIB` for every `cargo` invocation in this
workspace — no shell export needed. Re-run the script any time; it's a no-op if the library
is already present (`--force` to re-download).

To set it up manually instead, download a prebuilt binary for your platform from the same
releases page, then either:

- place it somewhere on your system's standard dynamic-library search path
  (`DYLD_LIBRARY_PATH` on macOS, `LD_LIBRARY_PATH` on Linux, `PATH` on Windows), or
- set the `PDFSPATIAL_PDFIUM_LIB` environment variable to the library file (or the
  directory containing it).

### Usage

```rust,no_run
use std::path::Path;

let document = pdfspatial_core::extract_baseline(Path::new("input.pdf"))?;
println!("{}", document.reading_order_text());
# Ok::<(), pdfspatial_core::PipelineError>(())
```

Run the bundled example:

```sh
PDFSPATIAL_PDFIUM_LIB=/path/to/libpdfium.dylib \
  cargo run --example basic_extract -- path/to/input.pdf
```

### Running tests

```sh
cargo test --all-features
```

This works with no environment variable once `scripts/fetch-pdfium.sh --write-cargo-config`
has been run (see Prerequisites above); otherwise set `PDFSPATIAL_PDFIUM_LIB` explicitly:

```sh
PDFSPATIAL_PDFIUM_LIB=/path/to/libpdfium.dylib cargo test --all-features
```

Stage 1's integration tests (`crates/pdfspatial-core/tests/stage1_baseline.rs`) extract a
small, hand-authored, dependency-free fixture PDF
(`crates/pdfspatial-core/tests/fixtures/single_column.pdf`) and assert both roadmap
targets — char recall ≥ 99%, line-grouping accuracy ≥ 95% — against its known ground
truth. These require the native PDFium library, as shown above.

Stage 2/4's integration test (`crates/pdfspatial-core/tests/stage2_pipeline.rs`)
constructs a synthetic multi-column `Document` in code and runs it through
`classify_regions` → `assemble_reading_order` → `to_markdown_structured`. It needs no
PDF and no PDFium library, so a plain `cargo test` (no environment variable required)
exercises it.

Stage 2's DocLayNet dataset harness (`crates/pdfspatial-core/tests/stage2_doclaynet.rs`)
loads a vendored, hand-authored DocLayNet-format fixture and scores
`layout::classify_regions`'s text-only predictions against it. It's gated behind the
`doclaynet` cargo feature:

```sh
cargo test --features doclaynet
```

The same fixture also exercises the Stage 3 mining signal end-to-end
(`mine_reading_order_failures`), and `examples/doclaynet_mine.rs` prints it as a ranked
report against a real DocLayNet sample:

```sh
cargo run --example doclaynet_mine --features doclaynet -- <coco.json> <cells_dir>
```

`examples/doclaynet_drafts.rs` (gated on both `doclaynet` and `stage3`, so also exercised
by `tests/stage3_mining.rs` under `cargo test --all-features`) takes that ranking further,
mining the top-N ranked pages into minimal, reviewable draft `fixtures/`-schema cases:

```sh
cargo run --example doclaynet_drafts --features doclaynet,stage3 -- \
  <coco.json> <cells_dir> --out <dir> [--top-n 5] [--grouped]
```

Both examples accept a real, unpacked DocLayNet-core tree's naming and cell schema
directly (no renaming needed), and `--grouped` switches page reconstruction to Stage 1's
real word/line/block grouping instead of one-cell-per-block. See `fixtures/README.md`'s
"Getting a real DocLayNet-core sample" and "Mining drafts from a real sample" sections
for setup, the `--grouped` trade-off, and the review checklist before promoting a draft
into the real corpus.

### Running benchmarks

[Criterion](https://docs.rs/criterion) benches score the Stage 2 eval pipeline and metric
primitives against the vendored DocLayNet fixture (or a real sample via `DOCLAYNET_DIR`);
see [`benches/README.md`](benches/README.md) for details:

```sh
cargo bench --features doclaynet
```

## Project layout

```
pdfspatial/
├── crates/pdfspatial-core/   # the published crate: extract, layout, assemble,
│                             # serialize, metrics
├── benches/                  # Criterion benches for Stage 2 validation
├── fixtures/                 # Stage 3 minimal-repro regression corpus (seeded)
├── scripts/fetch-pdfium.sh   # downloads/wires up the native PDFium library
├── examples/basic_extract.rs
└── .github/workflows/ci.yml
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

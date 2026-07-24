# pdfspatial

A closed-loop, four-stage pipeline for turning PDFs into Markdown, grounded in spatial
(bounding-box) extraction rather than layout guesswork. Built in Rust on top of
[`pdfium-render`](https://docs.rs/pdfium-render), the idiomatic Rust wrapper around
Google's PDFium engine.

This repository currently ships one crate, [`pdfspatial-core`](crates/pdfspatial-core),
implementing **Stage 1** in full and the **algorithmic core of Stages 2 and 4**:
validation metrics, a deterministic heuristic layout classifier, column-aware
reading-order assembly, and structural Markdown output — all pure, dependency-free Rust.
The one piece still unimplemented is the vision-model layout detector the roadmap
describes for Stage 2/4b (an ONNX RT-DETR-style detector over rendered page rasters,
needed for `Table`/`Picture`/`Formula` classes); it remains a documented
`unimplemented!()` stub since it needs an inference runtime, model weights, and a
DocLayNet-backed evaluation harness that are out of scope for this pass.

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
3. **Error analysis** (stubbed) — cluster Stage 2 shortfalls into a reproducible
   failure-mode taxonomy (multi-column gutters, footnotes, cross-page tables, ...), each
   tied to a minimal-repro regression fixture. See
   [`assemble.rs`](crates/pdfspatial-core/src/assemble.rs) for the full pitfall
   checklist this stage organizes around.
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
this crate. Download a prebuilt binary for your platform from
[bblanchon/pdfium-binaries](https://github.com/bblanchon/pdfium-binaries/releases), then
either:

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
├── fixtures/                 # Stage 3 minimal-repro regression corpus (placeholder)
├── examples/basic_extract.rs
└── .github/workflows/ci.yml
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

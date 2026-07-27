# pdfspatial

[![CI](https://github.com/passpier/pdfspatial/actions/workflows/ci.yml/badge.svg)](https://github.com/passpier/pdfspatial/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](Cargo.toml)

**Turn a PDF into faithful Markdown by reasoning about where things are on the
page, not by guessing at layout from text alone.**

Most PDF-to-text tools flatten a page into a single stream of characters and
lose the structure — columns, headings, tables, and reading order all
collapse into one blob. `pdfspatial` instead extracts every character's exact
bounding box (via [`pdfium-render`](https://docs.rs/pdfium-render), the Rust
wrapper around Google's PDFium engine), then uses that spatial information —
not heuristic guesswork — to classify regions, reconstruct reading order, and
serialize structured Markdown.

This repo ships one crate, [`pdfspatial-core`](crates/pdfspatial-core). Today
it can reliably extract text with bounding boxes and reassemble multi-column
reading order; table/figure/formula *detection* (the vision-model stage) is
an intentional, documented stub — see [Status](#status).

## Status

| Capability | State |
|---|---|
| Bounding-box text extraction (char → word → line → block) | ✅ Stable |
| Reading-order assembly (column-aware XY-cut) | ✅ Stable |
| Heuristic layout classification (Title, Header, List, Caption, ...) | ✅ Stable |
| Structural Markdown serialization | ✅ Stable |
| Validation metrics (TEDS, GIoU, region F1) against DocLayNet | ✅ Stable |
| Regression corpus of known failure modes (26 seeded cases) | ✅ Stable |
| Vision-model region detection (`Table`/`Picture`/`Formula`) | 📋 Planned — documented `unimplemented!()` stub, needs an ONNX runtime + model weights, out of scope for now |

See [`docs/PDF-to-Markdown Pipeline Roadmap.md`](docs/PDF-to-Markdown%20Pipeline%20Roadmap.md)
for the full stage-by-stage plan, metrics, and exit criteria behind this table.

## Install

**Prerequisites:** Rust ≥ 1.85, and PDFium's native library (not bundled by
`pdfium-render`).

```bash
git clone https://github.com/passpier/pdfspatial.git
cd pdfspatial
./scripts/fetch-pdfium.sh --write-cargo-config
```

This downloads a pinned PDFium release from
[bblanchon/pdfium-binaries](https://github.com/bblanchon/pdfium-binaries/releases)
and writes a gitignored `.cargo/config.toml` so every `cargo` command in this
workspace can find it — no manual environment variables needed.

Verify the setup:

```bash
cargo test --all-features
```

<details>
<summary>Manual PDFium setup (if you'd rather not run the script)</summary>

Download a prebuilt binary for your platform from the
[bblanchon/pdfium-binaries releases](https://github.com/bblanchon/pdfium-binaries/releases),
then either:

- place it on your OS's dynamic-library search path (`DYLD_LIBRARY_PATH` on
  macOS, `LD_LIBRARY_PATH` on Linux, `PATH` on Windows), or
- set `PDFSPATIAL_PDFIUM_LIB` to the library file (or its containing
  directory).

</details>

## Usage

```rust,no_run
use std::path::Path;

let document = pdfspatial_core::extract_baseline(Path::new("input.pdf"))?;
println!("{}", document.reading_order_text());
# Ok::<(), pdfspatial_core::PipelineError>(())
```

Or run the bundled example against your own PDF:

```bash
cargo run --example basic_extract -- path/to/input.pdf
```

## How it works

```
PDF → [pdfium: char/word bboxes + page raster] → [Layout Model: region classification]
    → [Reading-order assembly] → [Markdown serializer] → .md
```

1. **Extract** — pull every character's bounding box straight from PDFium's
   text layer, then group chars → words → lines → blocks geometrically. No
   OCR, no ML. (`extract.rs`)
2. **Classify** — assign each block a region type (Title, Section Header,
   List Item, Caption, ...) using deterministic text-layer heuristics.
   (`layout.rs`)
3. **Assemble** — reorder blocks into correct reading order with a
   column-aware recursive XY-cut, so multi-column pages don't read
   left-then-right across the gutter. (`assemble.rs`)
4. **Serialize** — emit structured Markdown from the classified, ordered
   blocks. (`serialize.rs`)

Accuracy is tracked against the [DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1)
dataset (TEDS, GIoU, per-class F1), and known failure modes (multi-column
gutters, footnotes, rotated text, borderless tables, ...) are captured as a
26-case regression corpus under [`fixtures/`](fixtures) so fixes can be
validated instead of eyeballed.

For the full detail — per-stage exit criteria, the failure-mode taxonomy, and
how the loop closes — see the [roadmap doc](docs/PDF-to-Markdown%20Pipeline%20Roadmap.md).

## Tech stack

- **[Rust](https://www.rust-lang.org/)** (edition 2024, MSRV 1.85) — a single
  dependency-free algorithmic core (no ML runtime) for Stages 1/2/4, chosen
  for deterministic, testable geometry rather than a Python heuristic
  pipeline.
- **[PDFium](https://pdfium.googlesource.com/pdfium/)** via
  [`pdfium-render`](https://docs.rs/pdfium-render) — Google's production PDF
  renderer, for character-level bounding boxes and page rasters.
- **[DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1)** —
  human-annotated layout dataset used to validate classification accuracy
  (opt-in via the `doclaynet` feature).
- **[Criterion](https://docs.rs/criterion)** — benchmarks for the validation
  pipeline and metric primitives (`benches/`).

## Development

```bash
cargo test --all-features                # full suite, needs PDFium (see Install)
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
cargo bench --features doclaynet          # Criterion benches, see benches/README.md
```

`cargo test` alone (no flags) runs the PDFium-free, synthetic-fixture pipeline
tests. Feature flags (`doclaynet`, `stage3`) gate the dataset harness and
regression-corpus tests respectively — see [`CLAUDE.md`](CLAUDE.md) for the
full breakdown of what each test target covers, and
[`fixtures/README.md`](fixtures/README.md) for the regression-corpus schema
and how to mine new cases from real DocLayNet samples.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

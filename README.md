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

This repo ships one crate, [`pdfspatial-core`](crates/pdfspatial-core). It
reliably extracts text with bounding boxes, reassembles multi-column reading
order, and detects tables and pictures from a page's non-text graphics (ruling
lines, embedded images) — all deterministic, dependency-free geometry, no
OCR and no vision model. Only formula detection remains genuinely
model-shaped — see [Status](#status).

## Status

| Capability | State |
|---|---|
| Bounding-box text extraction (char → word → line → block) | ✅ Stable |
| Reading-order assembly (column-aware XY-cut) | ✅ Stable |
| Heuristic layout classification (Title, Header, List, Caption, ...) | ✅ Stable |
| Table/picture detection from ruling lines & embedded images (`graphics`) | ✅ Stable |
| Structural Markdown serialization (incl. GFM tables, image placeholders) | ✅ Stable |
| Validation metrics (TEDS, GIoU, region F1) against DocLayNet | ✅ Stable |
<!-- BEGIN GENERATED: pitfall-status-row -->
| Regression corpus of known failure modes (26 seeded cases) | ✅ Stable |
<!-- END GENERATED: pitfall-status-row -->
| Formula detection (`RegionClass::Formula`) | 🚧 Partial — display (block-level) formulas classified via a geometric heuristic (`layout::is_display_formula`: centered, narrow, isolated, symbol-dense); *inline* formulas embedded mid-sentence still need an ONNX runtime + model weights, out of scope for now |

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
PDF → [pdfium: char/word bboxes + page graphics] → [Layout: text heuristics + graphics-layer table/picture detection]
    → [Reading-order assembly] → [Markdown serializer] → .md
```

1. **Extract** — pull every character's bounding box straight from PDFium's
   text layer, then group chars → words → lines → blocks geometrically; also
   pull every non-text page object (ruling lines, images, fills) as a
   `Graphic`. No OCR, no ML. (`extract.rs`)
2. **Classify** — assign each block a region type (Title, Section Header,
   List Item, Caption, ...) using deterministic text-layer heuristics, and
   detect `Table`/`Picture` regions from ruling lines and embedded images
   (`layout.rs`, `graphics.rs`).
3. **Assemble** — reorder blocks into correct reading order with a
   column-aware recursive XY-cut, so multi-column pages don't read
   left-then-right across the gutter. (`assemble.rs`)
4. **Serialize** — emit structured Markdown from the classified, ordered
   blocks — headings, lists, captions, footnotes, GFM pipe tables
   reconstructed from a table's ruling-line grid, and image placeholders.
   (`serialize.rs`)

Accuracy is tracked against the [DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1)
dataset (TEDS, GIoU, per-class F1), and known failure modes (multi-column
gutters, footnotes, rotated text, borderless tables, ...) are captured as a
26-case regression corpus under [`fixtures/`](fixtures) so fixes can be
validated instead of eyeballed.

For the full detail — per-stage exit criteria, the failure-mode taxonomy, and
how the loop closes — see the [roadmap doc](docs/PDF-to-Markdown%20Pipeline%20Roadmap.md).

## Benchmarks

Scored against the real, external [opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench)
corpus (200 real-world PDFs, Apache-2.0) — every engine below re-run on one
machine (Apple M2 Pro, macOS 15) on 2026-08-04 — alongside
[`pdf-inspector`](https://github.com/firecrawl/pdf-inspector), the most
directly analogous competitor (a dependency-light, model-free, deterministic
Rust extractor):

| Engine | Overall | Reading order (NID) | Table (TEDS) | Heading (MHS) | s/doc | License |
|---|---|---|---|---|---|---|
| pdf-inspector | 0.875 | 0.915 | 0.814 | 0.788 | 0.007s | MIT OR Apache-2.0 |
| opendataloader | 0.831 | 0.902 | 0.489 | 0.739 | 0.014s | Apache-2.0 |
| pdfspatial (compact) | 0.602 | 0.811 | 0.064 | 0.229 | 0.003s | MIT OR Apache-2.0 |
| **pdfspatial** | **0.600** | **0.809** | **0.064** | **0.229** | **0.005s** | MIT OR Apache-2.0 |
| markitdown | 0.589 | 0.844 | 0.273 | 0.000 | 0.125s | MIT |
| liteparse | 0.582 | 0.873 | 0.000 | 0.000 | 2.683s | Apache-2.0 |

The **pdfspatial** row (bold) is the headline: the library's default,
faithful Markdown output (`---` page breaks, `![]()` picture placeholders).
`pdfspatial (compact)` is the same binary run with `--no-page-breaks
--no-image-placeholders`, isolating what that Markdown syntax costs against
a scorer that treats it as document text — see "Known asymmetries" below.

Two honest caveats before reading too much into the ranking:

- **Speed isn't apples to apples.** `pdfspatial` runs the entire 200-PDF
  corpus in **one process** (its CLI has a real batch mode); `pdf-inspector`'s
  CLI has no batch mode and pays 200 process spawns inside the timer. Both are
  real properties of the tools being measured, not something this harness
  imposes — see [`bench/opendataloader/README.md`](bench/opendataloader/README.md).
- **`evaluator_reading_order.py`'s NID only collapses whitespace** — it never
  strips Markdown syntax, so `pdfspatial`'s faithful `---`/`![]()` output is
  scored as inserted document text. TEDS is also `pdfspatial`'s weakest
  column by a wide margin: tables are reconstructed from ruling lines
  (`graphics::table_grid_cells`), so a borderless/whitespace-delimited table
  produces no GFM table at all on some corpus documents — a known, tracked
  limitation (`docs/pitfall_registry.json`'s `borderless_table` entry), not
  measurement noise. MHS is structurally capped too: `to_markdown_structured`
  only emits `#`/`##`, with no heading class for `###` or deeper.

See [`bench/opendataloader/README.md`](bench/opendataloader/README.md) to
reproduce this table (`./scripts/run-opendataloader-bench.sh`) and
[`bench/opendataloader/results/results.json`](bench/opendataloader/results/results.json)
for the raw numbers, hardware, and corpus revision behind it.

## Tech stack

- **[Rust](https://www.rust-lang.org/)** (edition 2024, MSRV 1.85) — a single
  dependency-free algorithmic core (no ML runtime) for Stages 1/2/4, chosen
  for deterministic, testable geometry rather than a Python heuristic
  pipeline.
- **[PDFium](https://pdfium.googlesource.com/pdfium/)** via
  [`pdfium-render`](https://docs.rs/pdfium-render) — Google's production PDF
  renderer, for character-level bounding boxes, page rasters, and non-text
  page objects (ruling lines, images) via its page-object API.
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

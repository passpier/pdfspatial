# opendataloader-bench comparison

Runs `pdfspatial` against the real, external
[opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench)
corpus (200 real-world PDFs, Apache-2.0) and a handful of local, model-free comparison
engines, the way [firecrawl/pdf-inspector](https://github.com/firecrawl/pdf-inspector)
publishes its own numbers against the same corpus. The headline table lives in the
repo's [`README.md`](../../README.md#benchmarks); this directory is what produces it.

## Reproducing

```sh
./scripts/fetch-pdfium.sh --write-cargo-config   # if you haven't already, see the main README
./scripts/run-opendataloader-bench.sh
```

That one command clones the upstream corpus into `third_party/opendataloader-bench/`
(already gitignored -- see the main repo's `.gitignore`), builds a release `pdfspatial`
binary, installs the comparison engines via `uv`, registers `pdfspatial` /
`pdfspatial-compact` / `pdf-inspector` as bench engines by appending to a fresh clone's
`src/engine_registry.py` (no fork needed -- see `registry_patch.py`), runs every engine
sequentially over all 200 PDFs, and writes `results/results.json` plus a ready-to-paste
Markdown table. See `scripts/run-opendataloader-bench.sh --help` for flags -- notably
`--doc-id <stem>` for a ~2-minute smoke test instead of a multi-hour full run, and
`--engines "pdfspatial markitdown"` to run a subset.

**Prerequisites**: `git`, `uv` (Python 3.13+, provisioned automatically by `uv sync`),
`cargo`. No Java is needed for the default `opendataloader` engine (it uses the
`opendataloader-pdf` Python package; only its optional JAR mode needs a JVM, which this
harness doesn't use).

**This never runs in CI.** It needs a multi-GB Python environment (the corpus's base
dependencies pull in `easyocr`, which drags in PyTorch, regardless of which comparison
engines you actually select), network access, and — for a speed column to mean
anything — an otherwise-idle machine for what can be hours. A number measured on a
shared, noisy CI runner would be worse than no number at all. If you re-run this, do it
locally and record the hardware (see below).

## What's measured

Per engine, from the bench's own evaluator (`evaluator.py`), read out of
`prediction/<engine>/evaluation.json` after `run.py` merges in `speed`:

- **Overall** — mean of NID, TEDS, and MHS per document, averaged across documents.
- **NID** (reading order) — `rapidfuzz.fuzz.ratio` similarity between ground-truth and
  predicted Markdown, after collapsing whitespace only. It does **not** strip Markdown
  syntax — see "Known asymmetries" below.
- **TEDS** (table structure) — tree-edit-distance similarity between ground-truth and
  predicted GFM/HTML tables.
- **MHS** (heading structure) — tree-edit-distance similarity between heading-level
  sequences (`#`, `##`, ...).
- **s/doc** — `elapsed_per_doc` from the bench's own `summary.json`: total wall time for
  the whole corpus run, divided by document count. Reported per-document, not per-page,
  because per-page counts aren't uniformly available from every engine's output — only
  `pdfspatial`'s own CLI reports those (see its stderr summary line after a run).

## Engines compared, and why each is measured the way it is

- **`pdfspatial`** (this crate, via `crates/pdfspatial-cli`) — the headline row. Runs as
  **one process for the whole 200-PDF corpus** (`pdfspatial --out DIR <200 paths>`),
  because that's how the CLI's batch mode is meant to be used, and because
  `pdf_parser.py` times only the `to_markdown(...)` adapter call — which encloses
  process spawn and PDFium dylib load — so paying that cost once instead of 200 times is
  the honest measurement. Run at `--jobs 1` (see below).
- **`pdfspatial-compact`** — the same binary with `--no-page-breaks
  --no-image-placeholders`. Exists to quantify what the faithful default's Markdown
  syntax costs against a scorer that treats `---` and `![]()` as inserted document text
  (see "Known asymmetries"). Not the headline row — the library's default output stays
  faithful (see `pdfspatial_core::serialize::MarkdownOptions`'s doc comment).
- **`pdf-inspector`** — the most directly analogous competitor: a dependency-light,
  model-free, deterministic Rust extractor (`lopdf`-based). Its `pdf2md` CLI has **no
  batch mode**, so this adapter spawns one subprocess *per document* — 200 spawns inside
  the timer. That's a real property of the tool, not a handicap this harness imposes,
  but it is the single biggest confound in the speed comparison. Its default output also
  carries no page-break markers (`--pages` opts in), so on that one axis it's closer to
  `pdfspatial-compact` than to `pdfspatial`'s own default.
- **`markitdown`**, **`liteparse`**, **`opendataloader`** — run via the upstream repo's
  own adapters, unmodified.

### Why `pdfspatial`'s headline row runs at `--jobs 1`

Every comparison engine above processes its 200 documents sequentially, in a single
process. `pdfspatial`'s own PDFium binding also serializes all real extraction work
behind a process-wide lock (`extract.rs`'s `PDFIUM_CALL_LOCK`) regardless of how many
threads ask for it, so cross-document parallelism doesn't buy real extraction speedup
here anyway — only the pure post-extraction stages (classify/reorder/serialize) overlap
across threads. Running the headline row single-threaded keeps the comparison apples to
apples; a multi-job number, if recorded, is a separate footnote, not the table's number.

## Known asymmetries (read before trusting the table too literally)

- **`evaluator_reading_order.py`'s `_normalize` only collapses whitespace** — it never
  strips Markdown syntax. `pdfspatial`'s faithful default emits a `---` thematic break
  between every page and a `![]()` placeholder for every detected picture; both count as
  inserted document text against ground truth. The `pdfspatial-compact` row exists to
  isolate exactly this cost.
- **MHS is structurally capped for `pdfspatial`.** `serialize::to_markdown_structured`
  only emits `#` (Title) and `##` (SectionHeader) — no `RegionClass` maps to `###` or
  deeper. Any ground-truth document with a third heading level caps our MHS-S no matter
  how good the classification is.
- **TEDS is likely `pdfspatial`'s weakest column.** Tables are reconstructed from ruling
  lines (`graphics::table_grid_cells`); a borderless/whitespace-delimited table (the
  `borderless_table` pitfall) produces no GFM table at all on some corpus documents.
  This is a known, tracked limitation, not measurement noise — see
  `docs/pitfall_registry.json`'s `borderless_table` entry.
- **`PageHeader`/`PageFooter` regions are dropped entirely** from `pdfspatial`'s
  structured output. Depending on whether the bench's own ground truth retains running
  headers/footers, this cuts for or against us; check a few ground-truth files directly
  if this matters for your read of the numbers.
- **Thermal throttling.** A multi-hour, multi-engine run on a single laptop can make
  later engines look slower for reasons that have nothing to do with the engine.
  `s/doc` here is order-of-magnitude evidence, not a precision benchmark result.

## `results/results.json`

Committed after every real run, alongside the engines' raw
`prediction/<engine>/evaluation.json` copies under `results/raw/` (not automated by
`collect.py` today — copy them by hand if you want them preserved past the next
`third_party/` clean). Records `corpus_commit` (the upstream clone's own git SHA),
`documents`, `date`, a `hardware` block (`processor`, `os`, `jobs`), and one entry per
engine with every metric above. `hardware` and `corpus_commit` are non-negotiable: every
number in the README table is machine- and corpus-revision-specific, and without them
the table is unfalsifiable. `crates/pdfspatial-core/tests/stage5_bench_results.rs`
checks the README table against this file.

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

- **Overall** — mean of NID, TEDS, and MHS per document, averaged across documents. This
  is a **ragged** mean, not a simple average of the three published column means: TEDS
  is `None` (excluded from that document's mean, not scored `0.0`) when the ground truth
  has no table, and likewise MHS when it has no heading. On this corpus TEDS is scored
  on only 42 of the 200 documents and MHS on 107, all 200 score NID — so Overall can't be
  reconstructed by hand from the three published column means, and a change concentrated
  in a small set of documents can move Overall by more than its share of the column mean
  would suggest.
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

The metric-by-metric analysis — why NID scores `pdfspatial`'s Markdown syntax as
inserted text, the two tracked causes behind its weak TEDS column, why MHS isn't
capped by heading depth, the speed asymmetry between batch and per-document CLIs, and
what this benchmark doesn't measure at all — lives in
[`docs/benchmark-analysis.md`](../../docs/benchmark-analysis.md), not here. This file
stays scoped to reproduction methodology.

## `results/results.json`

Committed after every real run, alongside the engines' raw
`prediction/<engine>/evaluation.json` copies under `results/raw/` (not automated by
`collect.py` today — copy them by hand if you want them preserved past the next
`third_party/` clean). Records `corpus_commit` (the upstream clone's own git SHA),
`documents`, `date`, a `hardware` block (`processor`, `os`, `jobs`), a `versions` block
(`pdfspatial_git_sha`, `rustc`, `python`, `uv_lock_sha256_12` -- see below), and one
entry per engine with every metric above, including that engine's real installed
`version` (read from `importlib.metadata`/`cargo install --list` at run time by
`registry_patch.py`, not the possibly-stale literal upstream's own
`engine_registry.py` hardcodes). `hardware` and `corpus_commit` are non-negotiable:
every number in the README table is machine- and corpus-revision-specific, and without
them the table is unfalsifiable. `crates/pdfspatial-core/tests/stage5_bench_results.rs`
checks the README table against this file.

`versions.uv_lock_sha256_12` is a fingerprint of the upstream clone's `uv.lock` at
collection time -- the actual pin for every Python engine's full dependency tree, which
this repo doesn't vendor. If a future run's per-engine `version` fields match today's
but this hash differs, something in the dependency tree moved without a version bump
visible in the table.

# Benchmark analysis

The headline table in [`README.md`](../README.md#benchmarks) is scored against the real,
external [opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench)
corpus (200 real-world PDFs). This doc is the metric-by-metric, architectural read of
those numbers — why `pdfspatial` lands where it does on each column, and every known
asymmetry in how the scorer treats different engines' output. For *how to reproduce* the
table, see [`bench/opendataloader/README.md`](../bench/opendataloader/README.md) instead.

## How to read "Overall"

`evaluator.py` computes Overall as the mean of NID, TEDS, and MHS **per document**,
averaged across documents. This is a **ragged** mean, not a simple average of the three
published column means: TEDS is scored `None` (excluded from that document's mean, not
scored `0.0`) whenever the ground truth has no table, and likewise MHS when it has no
heading. On this corpus, NID is scored on all 200 documents, TEDS on only 42, and MHS on
107 — so Overall can't be reconstructed by hand from the three column means, and a
change concentrated in a small set of documents can move Overall by more than its share
of a column mean would suggest.

## Reading order (NID)

`evaluator_reading_order.py`'s NID is `rapidfuzz.fuzz.ratio` similarity between
ground-truth and predicted Markdown, after collapsing whitespace only — it never strips
Markdown syntax. `pdfspatial`'s faithful default emits a `---` thematic break between
every page and a `![]()` placeholder for every detected picture; both count as inserted
document text against ground truth that has neither. The `pdfspatial (compact)` row
(`--no-page-breaks --no-image-placeholders`) exists to isolate exactly this cost —
measured on this corpus, it's small: compact vs. default differ by ~0.002 Overall and
~0.003 NID, not the dominant factor a naive reading of "Markdown syntax is scored as
document text" might suggest.

## Table (TEDS)

`pdfspatial`'s weakest column by a wide margin, and the gap is real, not measurement
noise. Two distinct, tracked causes (`docs/pitfall_registry.json`):

- **`borderless_table`** — a borderless/whitespace-delimited table produces no GFM table
  at all on some documents. Partially closed: `layout::whitespace_column_corridors` now
  merges adjacent compatible row bands into one multi-row table instead of N degenerate
  one-row tables.
- **`multi_line_table_cell`** — real Stage 1 block grouping
  (`extract::group_chars_into_blocks`) sometimes merges a bordered table's row cells
  across the column gap into one block *before* `graphics::table_grid_cells` ever sees
  them as separate cells to assign to a grid. Still open — `blocked` in the registry,
  needs a Stage 1 `extract.rs` change (stop a block merge at a ruling-line boundary),
  not a Stage 4 classification fix.

## Heading (MHS)

MHS is **not** structurally capped by only emitting `#`/`##`. (An earlier revision of
this analysis claimed `serialize::to_markdown_structured` emitting no `RegionClass` past
`##` caps MHS whenever a ground-truth document has a third heading level — that was
wrong on inspection: `evaluator_heading_level.py`'s `HeadingTree` discards the
`#`-count entirely and treats every heading level as structurally equivalent. Separately,
this corpus's own ground truth generator only ever emits a level-1 `#` — there is no
`##`+ anywhere in the 200 reference files — so heading depth was never actually in play
here.)

The real driver of a low MHS score was headings **trapped as an interior line of a
merged Stage 1 paragraph block** — never becoming their own block, so
`classify_block`'s heading rules (which only ever fire on a whole block) never saw them.
`layout::split_blocks_at_style_breaks` closes most of that gap as a Stage 2 pre-pass.

## Speed (s/doc)

`pdfspatial` runs the entire 200-PDF corpus in **one process** (its CLI has a real batch
mode), matching every comparison engine's single-process, sequential-document execution
— except `pdf-inspector`, whose `pdf2md` CLI has no batch mode and pays 200 process
spawns inside the timer. That's a real property of the tool being measured, not a
handicap this harness imposes, but it's the single biggest confound in the speed
comparison; `pdf-inspector`'s number would very likely drop with a batch mode.

`pdfspatial`'s own headline row runs at `--jobs 1`: its PDFium binding serializes all
real extraction work behind a process-wide lock (`extract.rs`'s `PDFIUM_CALL_LOCK`)
regardless of thread count, so cross-document parallelism doesn't buy real extraction
speedup here anyway — only the post-extraction stages (classify/reorder/serialize)
overlap across threads. A multi-job number, if recorded, is a footnote, not the table's
number.

A multi-hour, multi-engine run on a single laptop can also make later engines look
slower for reasons that have nothing to do with the engine (thermal throttling). Treat
`s/doc` as order-of-magnitude evidence, not a precision benchmark result.

## What this benchmark does not measure

- **`PageHeader`/`PageFooter` regions are dropped entirely** from `pdfspatial`'s
  structured output. Depending on whether the bench's own ground truth retains running
  headers/footers, this cuts for or against `pdfspatial` on NID — check a few
  ground-truth files directly if this matters for your read of the numbers.
- **No formula scoring.** `RegionClass::Formula` (see the main [README](../README.md#status))
  is the one region class this crate leaves genuinely vision-model-shaped, but this
  corpus's evaluator doesn't score formulas at all — so that gap is invisible in every
  column above, not just underweighted.

## Version provenance

Every number above is tied to a specific corpus revision, hardware, and set of engine
versions — see `bench/opendataloader/results/results.json`'s `corpus_commit`,
`hardware`, `versions`, and each engine's own `version` field. The Python engines
(`opendataloader`, `markitdown`, `liteparse`) are pinned only by the upstream clone's own
`uv.lock`, which this repo doesn't vendor; `results.json`'s `versions.uv_lock_sha256_12`
is a fingerprint of that lockfile at collection time, and each engine's `version` field
is read from the installed package (`importlib.metadata`/`cargo install --list`) rather
than trusted from upstream's `engine_registry.py` literals, which otherwise go stale the
moment a `uv sync --upgrade` moves a package underneath them.

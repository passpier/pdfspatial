# Stage 3 regression corpus

Hand-authored, minimal-repro regression cases for Stage 3 error analysis, loaded by
`eval::corpus` (`crates/pdfspatial-core/src/eval/corpus.rs`) behind the `stage3` cargo
feature and checked by `crates/pdfspatial-core/tests/stage3_corpus.rs`. Each case is
tagged with one of the 15 checklist items in `crates/pdfspatial-core/src/assemble.rs`
(the `Pitfall` enum) and a root-cause tag (`assemble::RootCause`: `geometric`,
`classification`, or `ordering`).

This is distinct from `crates/pdfspatial-core/tests/fixtures/`, which holds small,
hand-authored PDFs used by Stage 1's unit tests.

## Why synthetic `Document`s, not real PDFs

The roadmap's Stage 3 process describes mining failures from a real DocLayNet sample
and extracting a minimal-repro *PDF page* for each cluster. That requires a native
PDFium library and a DocLayNet download, neither of which is guaranteed to be available
in every environment this crate builds in. Each case here instead encodes an
already-extracted `Document` directly as JSON (the same PDFium-free substrate
`tests/stage2_pipeline.rs` uses), so the corpus runs anywhere `cargo test` runs, with no
setup. This narrows coverage to pitfalls reachable through the
`layout::classify_regions` / `assemble::assemble_reading_order` surface.

Mining real DocLayNet failures into real PDF fixtures remains future work once PDFium
and a DocLayNet sample are available; the `eval::doclaynet` loader and the
`giou`/`region_f1`/`match_regions` metrics are the intended bridge for that pass.

## Directory layout

```
fixtures/
├── <pitfall-slug>/
│   ├── case_one.json
│   └── case_two.json
└── ...
```

Every case must live directly under the directory matching its own `pitfall` field
(enforced by `corpus_is_wellformed` in `tests/stage3_corpus.rs`).

## Case JSON schema

```json
{
  "id": "unique-case-id",
  "pitfall": "multi_column",
  "root_cause": "ordering",
  "description": "Human-readable explanation of the failure this case reproduces.",
  "page": {
    "width": 600.0,
    "height": 800.0,
    "blocks": [
      {
        "lines": [
          { "text": "...", "bbox": [40.0, 700.0, 290.0, 750.0], "font_size": 10.0 }
        ]
      }
    ]
  },
  "expected": {
    "reading_order": ["...", "..."],
    "classes": [
      { "block_text": "...", "class": "footnote" }
    ]
  }
}
```

- `bbox` is `[left, bottom, right, top]`, already in the crate's own bottom-left-origin
  coordinate space (no COCO-style flip needed -- these are hand-authored directly).
- A block is authored as one or more `lines` (each its own line/word/char run) so a
  case can control line count directly -- several pitfalls (running headers/footers,
  multi-line table cells, fragmented formulas) hinge on `Block::lines().len()`, which a
  flat one-line-per-block shape can't vary.
- `expected.reading_order` and `expected.classes[].block_text` identify blocks by their
  exact joined text (`Block::text()`, lines joined by `\n`); block texts must be unique
  within a case.
- `expected.classes` states the class `classify_regions` *should* assign once Stage 4
  fixes the pitfall -- not what it assigns today. See "Desired behavior, not current
  behavior" below.
- Either or both of `expected.reading_order`/`expected.classes` may be present,
  depending on whether the case exercises ordering, classification, or both.
- A case authors exactly one of `page` (single page) or `pages` (a JSON array of the
  same page shape, one entry per page in document order) -- `pages` exists for
  `cross_page_continuation`, the one pitfall that's inherently about more than one page.
- `source_pdf`: present only on **PDF-backed cases** (see below) -- a workspace-relative
  path to the fixture PDF this case's `page`/`pages` was extracted from. Absent for
  hand-authored cases.
- `expected.requires_extraction_fix` (default `false`): set on a PDF-backed case whose
  `expected` names text the current extractor doesn't produce yet (e.g. a desired
  `"SIDEBAR LABEL"` where extraction today shatters the rotated glyphs into one
  single-character line each). It exempts that case from `corpus_is_wellformed`'s normal
  "every expected string resolves to a real block" check -- but the test still requires a
  `source_pdf` and at least one entry that genuinely doesn't resolve, so the flag can't be
  set spuriously. See "PDF-backed cases" below.

### `pitfall` slugs

`multi_column`, `footnote`, `header_footer`, `multi_line_table_cell`,
`merged_table_cell`, `borderless_table`, `nested_formula`, `super_subscript`,
`rotated_text`, `figure_caption`, `list_nesting`, `cross_page_continuation`,
`section_header_vs_bold`, `embedded_font`, `overlapping_text` (mirrors
`assemble::Pitfall`'s variants).

### `root_cause` slugs

`geometric`, `classification`, `ordering` (mirrors `assemble::RootCause`).

### `class` slugs (for `expected.classes[].class`)

`caption`, `footnote`, `formula`, `list_item`, `page_footer`, `page_header`, `picture`,
`section_header`, `table`, `text`, `title` (mirrors `layout::RegionClass`).

## Desired behavior, not current behavior

Every seeded case's `expected` block states what the pipeline *should* do once Stage 4
lands the corresponding fix -- not a snapshot of today's (wrong) output. Running
`cargo test --features stage3 -- --ignored` executes
`corpus_cases_meet_expected_behavior`, which is `#[ignore]`d for exactly this reason: it
currently fails for some cases, printing a scoreboard of every mismatch -- see the
generated coverage block above (or `cargo run --example stage3_scoreboard --features
stage3`) for the live per-pitfall pass/fail counts, rather than a number restated here
by hand. `multi_column-wide-gutter-recovers-column-order`
(`fixtures/multi_column/`) is a *positive contrast* case -- its gutter is wide enough for
`assemble::MIN_CUT_FRACTION` to fire regardless of the em-relative rule below -- and is
checked unconditionally by `tests/stage3_corpus.rs`'s
`multi_column_wide_gutter_recovers_reading_order`, not the `#[ignore]`d scoreboard.
All three narrow-gutter ordering cases in `multi_column`/`list_nesting` now pass too --
`assemble::widest_vertical_gutter`'s qualifying threshold is no longer a fixed fraction of
page width (`MIN_CUT_FRACTION`, still used on the horizontal axis and as the vertical
threshold's upper bound); it scales to the page's own median character size
(`MIN_GUTTER_EMS` × the page's median `Char::font_size`, floored by `MIN_GUTTER_ABS_PT`),
so a 1-em column gutter in small type qualifies as a cut even though it's well under 3%
of the page width. A companion guard (`vertical_extents_overlap`) requires the blocks on
either side of a candidate gutter to actually coexist vertically before it counts as a
column boundary, so two unrelated blocks in opposite page corners aren't mistaken for
columns now that the threshold is looser. Column *ordering* is fixed by this change;
list *nesting* (hierarchy depth, not order) remains open --
`two-outlines-narrow-gutter-interleaved` and its short variant now read each outline
fully before the next, but both still flatten into one list rather than a nested one.
`super_subscript-formula-baseline-clustering` (`fixtures/super_subscript/`) is a *fixed*
case -- Stage 1's `extract::group_words` now recognizes a smaller, baseline-offset,
nearby character as a super/subscript of its predecessor and attaches it to its base
instead of splitting it off as its own word, so it flows through the ordinary scoreboard
and simply passes. All three `header_footer` cases (`running-header-exceeds-line-limit`,
`running-footer-exceeds-line-limit`, `repeated-running-header-across-pages`) now pass as
well -- `layout::classify_block`'s band rule was relaxed from a hard line-count cap to a
shape test (thin strip, detached from the body by a minimum gap), and `classify_regions`
gained a cross-page repeated-content pass for running headers/footers too tall or too
close to the body for the single-page geometric rule to catch on its own. Both `footnote`
cases (`footnote-marker-not-classified`, `footnote-adjacent-to-footer-band`) now pass too
-- `classify_block` gained a `Footnote` rule requiring a block to sit low on the page,
carry a smaller font than the rest of the page (compared against a median that excludes
the candidate block itself, so a long footnote can't inflate its own baseline), and open
with a recognized footnote marker (a bare digit/symbol, never a bullet). That failing
run *is* the executable Stage 3 error analysis -- re-run it after any Stage 4 heuristic
change to see which cases flip to passing.

For every case with `expected.reading_order`, `CaseOutcome` (from `eval::corpus`) also
reports `reading_order_edit_distance` (post-`assemble_reading_order`, using
`metrics::reading_order_edit_distance`) and `naive_reading_order_edit_distance` (the
as-authored input order vs. ground truth) -- quantifying the roadmap's Stage 1 "reading-order
edit distance" prediction rather than only reporting a pass/fail boolean. The scoreboard
prints both numbers for every failing ordering case.

Two tests run unconditionally (not `#[ignore]`d) to guard the corpus's own integrity
regardless of pipeline behavior: `corpus_is_wellformed` (every case parses, ids are
unique, cases live under the directory matching their own pitfall, every
`block_text`/`reading_order` reference resolves to a real block) and
`corpus_covers_seeded_pitfalls` (at least one case per seeded pitfall, with a
per-pitfall coverage report on stderr).

## Coverage

Generated from the live corpus by `eval::scoreboard`/`examples/stage3_scoreboard.rs`
-- **never hand-edit the text between the `<!-- BEGIN/END GENERATED -->` markers**;
run `cargo run --example stage3_scoreboard --features stage3 -- --write` to
regenerate after any corpus change. Per-pitfall "why it needs a real PDF" prose lives
in [`docs/pitfall_registry.json`](../docs/pitfall_registry.json).

<!-- BEGIN GENERATED: pitfall-coverage -->
All 15 `assemble::Pitfall` variants are seeded, 26 cases total:

**Hand-authored, synthetic `Document`s** -- reachable through the synthetic `classify_regions`/`assemble_reading_order` surface, no PDF or PDFium involved:

| Pitfall | Root cause |
|---|---|
| `multi_column` (Multi-column layout / gutter detection) | `ordering` |
| `footnote` (Footnotes) | `classification` |
| `header_footer` (Page headers/footers repeated across pages) | `classification` |
| `merged_table_cell` (Merged table cells (rowspan/colspan)) | `classification` |
| `borderless_table` (Borderless / whitespace-delimited tables) | `classification` |
| `nested_formula` (Nested mathematical formulas) | `classification` |
| `figure_caption` (Figure/caption association) | `classification` |
| `list_nesting` (List-item nesting) | `ordering` |
| `section_header_vs_bold` (Section headers vs. bold body text) | `classification` |

**PDF-backed, frozen-extraction snapshots** -- see "PDF-backed cases" below:

| Pitfall | Root cause | Why it needs a real PDF |
|---|---|---|
| `multi_line_table_cell` | `classification` | Real Stage 1 block grouping merges the row's cells together across the column gap in a way a hand-authored already-split `Document` can't demonstrate. |
| `super_subscript` | `geometric` | A character-extraction/baseline-clustering failure in the real PDFium text layer; a synthetic already-grouped `Document` gives Stage 1's baseline clustering nothing to get wrong. **Fixed** — this is the one PDF-backed case that already passes the scoreboard. |
| `rotated_text` | `geometric` | Needs real glyph rotation data from PDFium's text layer. |
| `cross_page_continuation` | `ordering` | Needs a real multi-page extraction; `assemble_reading_order` only operates within a single page, and a synthetic multi-page `Document` wouldn't exercise the actual stitching gap PDFium extraction produces. |
| `embedded_font` | `geometric` | Needs a real custom-encoded/CID-keyed font to reproduce dropped/garbled glyphs. |
| `overlapping_text` | `geometric` | Needs real z-ordered text objects from a PDF. |
<!-- END GENERATED: pitfall-coverage -->

The roadmap's "≥ 20 samples per category" target is scoped as future work for a
real-data DocLayNet mining pass (once a DocLayNet sample is available in this
environment); this corpus establishes the format, harness, and a seed per pitfall.

## PDF-backed cases

The 6 pitfalls above are fundamentally about what PDFium's *real* text layer produces
(glyph baselines, rotation matrices, encoding tables, z-order, cross-page stitching) --
a hand-assembled `Document` gives Stage 1 nothing to get wrong, since it's already
correctly grouped by construction. These cases instead carry a `source_pdf` pointing at
a small fixture PDF under `crates/pdfspatial-core/tests/fixtures/stage3/`, and their
`page`/`pages` is a **frozen snapshot** of `extract_baseline`'s real output on that PDF.

The corpus itself stays PDFium-free to load -- the snapshot is committed as ordinary
JSON, so `corpus_is_wellformed`/`corpus_covers_seeded_pitfalls`
(`tests/stage3_corpus.rs`) run under plain `cargo test --features stage3`, same as
every hand-authored case. A separate test, `tests/stage3_pdf_fixtures.rs` (needs
PDFium), re-extracts each `source_pdf` and asserts the committed snapshot hasn't
drifted -- this is what keeps "frozen snapshot" honest across PDFium version bumps or
edits to the fixture PDFs.

Both the PDFs and their snapshots are produced by `examples/stage3_pdf_cases.rs`,
never hand-typed:

```sh
cargo run --example stage3_pdf_cases --features stage3
```

This writes the 6 PDFs (a small in-example PDF object writer, the same hand-rollable
PDF 1.7 shape `tests/fixtures/single_column.pdf` uses, just with the `xref` offset
bookkeeping automated), then -- if PDFium is available -- extracts each one and writes
its case JSON. Regeneration is idempotent and never clobbers hand review: if a target
case file already exists, its `description` and `expected` are loaded and carried
forward unchanged, and only the frozen `page`/`pages` geometry is replaced. Pass
`--pdfs-only` to skip the PDFium-dependent step (e.g. to inspect the PDFs without
PDFium set up).

**Never hand-edit a PDF-backed case's `page`/`pages` block** -- it's a snapshot of real
extraction output, not hand-authored geometry; editing the JSON directly desyncs it
from the PDF it claims to represent, and `stage3_pdf_fixtures.rs` will catch the drift.
To change one, edit the PDF content (or `examples/stage3_pdf_cases.rs`'s content-stream
string for that case) and re-run the generator. `description` and `expected` **are**
meant to be hand-edited -- that's the human-review step, and the generator preserves
your edits across regeneration as described above.

## Getting a real DocLayNet-core sample

The mining commands below need an unpacked
[DocLayNet-core](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1) download
(COCO annotations + per-page `pdf_cells` JSON; DocLayNet-core ships no source PDFs, so
`doclaynet_mine`'s `--pdfium` mode isn't reachable against it). Point `DOCLAYNET_DIR` at
the unpacked root:

```
$DOCLAYNET_DIR/
├── COCO/{train,val,test}.json
└── JSON/<page_hash>.json      # pdf_cells, one file per page
```

`load_sample` accepts this layout unmodified: it tries `{stem}.cells.json` (the vendored
test fixtures' own naming) and falls back to `{stem}.json` (real DocLayNet-core's
naming), and reads font metrics from either a flat `font_size` field or a nested
`font.size` object. No renaming or pre-processing needed.

## Mining drafts from a real sample

`eval::rank_pages_by_reorder` / `eval::doclaynet::mine_reading_order_failures` rank a
DocLayNet sample's pages by reading-order reordering severity (see `examples/doclaynet_mine.rs`).
`examples/doclaynet_drafts.rs` consumes that ranking end to end: it takes the top-N
most-reordered pages, shrinks each one via `eval::minimize_reorder_repro` (a greedy
backward prune that keeps removing blocks as long as the survivors still reorder --
turning a page's worth of `pdf_cells` into the handful of blocks that actually drive the
reordering), and writes each result as a **draft** case via `eval::corpus::write_draft_case`:

```sh
# Rank first, to see what's worth mining:
export DOCLAYNET_DIR=/path/to/DocLayNet-core
cargo run --example doclaynet_mine --features doclaynet

# Then mine the top pages into drafts (positional args default the same way):
cargo run --example doclaynet_drafts --features doclaynet,stage3 -- --out <dir> [--top-n 5]
```

Both examples also accept an explicit `<coco.json> <cells_dir>` pair instead of
`DOCLAYNET_DIR`. `--out` is required on `doclaynet_drafts` and is never `fixtures/`
itself -- a draft is unreviewed output, not a regression case yet. A draft carries
`"draft": true` and an `expected.reading_order` that is a **snapshot of
`assemble_reading_order`'s current output**, not a desired post-fix order like every
hand-authored case above. That's why `draft` cases are excluded from
`corpus_covers_seeded_pitfalls` and the `corpus_cases_meet_expected_behavior` scoreboard
(`tests/stage3_corpus.rs`) -- they'd otherwise pass trivially and mask real regressions --
though `corpus_is_wellformed` still checks their shape.

### `--grouped`: two ways to reconstruct a page from `pdf_cells`

By default, both examples map **one `pdf_cells` cell to one block** (`document_from_cells`)
-- DocLayNet cells are word- or phrase-sized, so a draft mined this way is a pile of
word-blocks needing heavy editing before it reads as a real repro. Passing `--grouped`
switches to `document_from_cells_grouped`, which runs the cells through Stage 1's real
char → word → line → block grouping (`extract::group_chars_into_blocks`) instead --
producing realistically-sized blocks, at the cost of merging same-baseline text across
column gutters exactly as real Stage 1 does (which can *reduce* the reordering signal on
some pages; see `eval::doclaynet`'s module docs for the full trade-off). Run both and
compare before deciding which to mine with:

```sh
cargo run --example doclaynet_mine --features doclaynet -- --grouped
cargo run --example doclaynet_drafts --features doclaynet,stage3 -- --out <dir> --grouped
```

Promoting a draft into the real corpus is a manual review step:

1. Open the draft JSON and confirm it's actually a minimal, legible repro of a real
   reordering failure. If it was mined without `--grouped`, watch for artifacts of
   `document_from_cells`'s one-cell-per-block under-grouping -- see `eval::doclaynet`'s
   module docs.
2. Re-tag `pitfall`/`root_cause` -- the mining pipeline always guesses `multi_column`/
   `ordering` (its own hypothesis, since that's what the ranking signal measures), which
   may not be the actual failure mode.
3. Rewrite `expected.reading_order` to the *desired* order, not the current one --
   `assemble_reading_order`'s snapshot is deliberately wrong for exactly this file until
   Stage 4 fixes the pitfall.
4. Delete the `"draft": true` field and move the file into `fixtures/<pitfall>/`.

## Running

```sh
# Corpus integrity, no PDFium needed (runs in CI via --all-features):
cargo test --features stage3

# PDF-backed snapshot drift check, needs PDFium (runs in CI via --all-features):
cargo test --all-features

# Full Stage 3 scoreboard (expected to fail until Stage 4 lands fixes):
cargo test --features stage3 -- --ignored
```

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
currently fails for all 18 seeded cases, printing a scoreboard of every mismatch. That
failing run *is* the executable Stage 3 error analysis -- re-run it after any Stage 4
heuristic change to see which cases flip to passing.

Two tests run unconditionally (not `#[ignore]`d) to guard the corpus's own integrity
regardless of pipeline behavior: `corpus_is_wellformed` (every case parses, ids are
unique, cases live under the directory matching their own pitfall, every
`block_text`/`reading_order` reference resolves to a real block) and
`corpus_covers_seeded_pitfalls` (at least one case per seeded pitfall, with a
per-pitfall coverage report on stderr).

## Coverage: seeded vs. deferred

Seeded (9 pitfalls, 2 cases each, 18 total -- reachable through the synthetic
`classify_regions`/`assemble_reading_order` surface):

| Pitfall | Root cause |
|---|---|
| `multi_column` | `ordering` |
| `footnote` | `classification` |
| `header_footer` | `classification` |
| `borderless_table` | `classification` |
| `merged_table_cell` | `classification` |
| `nested_formula` | `classification` |
| `figure_caption` | `classification` |
| `section_header_vs_bold` | `classification` |
| `list_nesting` | `ordering` |

Deferred (index-only; documented here, not yet seeded with executable JSON):

| Pitfall | Root cause | Why it's deferred |
|---|---|---|
| `multi_line_table_cell` | `classification` | Same `Table`-unreachable gap as `borderless_table`/`merged_table_cell`; not seeded separately to avoid redundant cases pending real table-structure predictions. |
| `super_subscript` | `geometric` | A character-extraction/baseline-clustering failure in the real PDFium text layer; a synthetic already-grouped `Document` gives Stage 1's baseline clustering nothing to get wrong. |
| `rotated_text` | `geometric` | Same -- needs real glyph rotation data from PDFium. |
| `embedded_font` | `geometric` | Same -- needs a real embedded/CID-keyed font to reproduce dropped/garbled glyphs. |
| `overlapping_text` | `geometric` | Same -- needs real z-ordered text objects from a PDF. |
| `cross_page_continuation` | `ordering` | Needs a multi-page real extraction; `assemble_reading_order` today only operates within a single page, and a synthetic multi-page `Document` wouldn't exercise the actual stitching gap PDFium extraction produces. |

The roadmap's "≥ 20 samples per category" target is scoped as future work for a
real-data DocLayNet mining pass (once PDFium and a DocLayNet sample are available in
this environment); this corpus establishes the format, harness, and a 2-case-per-pitfall
seed for the reachable subset.

## Running

```sh
# Corpus integrity (runs in CI via --all-features):
cargo test --features stage3

# Full Stage 3 scoreboard (expected to fail until Stage 4 lands fixes):
cargo test --features stage3 -- --ignored
```

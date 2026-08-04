# Technical Roadmap: High-Performance PDF-to-Markdown Extraction Pipeline (pdfium-render)

## Overview

This roadmap defines a closed-loop, four-stage development cycle for a Rust-based PDF-to-Markdown pipeline built on [`pdfium-render`](https://github.com/ajrcarey/pdfium-render), the idiomatic Rust wrapper around Google's PDFium engine, plus a Stage 5 comparative-benchmarking gate that sits outside that loop (see below). `pdfium-render` binds to PDFium at run-time, extracts character-level bounding boxes via `PdfPageTextChar`, and can render page bitmaps for downstream vision-model inference — giving us both the "text layer" and "visual layer" needed for spatial document reconstruction ([GitHub: ajrcarey/pdfium-render](https://github.com/ajrcarey/pdfium-render), [pdfium-render#27: word-level bbox extraction](https://github.com/ajrcarey/pdfium-render/issues/27)).

The architecture mirrors the proven pattern used by Docling's Rust backend — `pdfium` extracts text cells + renders page images, an object-detection stack classifies regions, and cells are reassembled in reading order into a structured document ([docs.rs: docling-pdf](https://docs.rs/docling-pdf/latest/docling_pdf/)).

```
PDF → [pdfium: char/word bboxes + page raster] → [Layout Model: region classification]
    → [Reading-order assembly] → [Markdown serializer] → .md
```

---

## Stage 1 — Baseline: pdfium Bounding-Box Extraction

**Goal:** Establish a deterministic, OCR-free text extraction floor using PDFium's native text layer.

### Implementation
- Use `FPDFText_CountRects` / `FPDFText_GetRect` (exposed in `pdfium-render` as page text object APIs) to pull per-character bounding boxes, font name, size, weight, rotation, and fill/stroke color ([Stack Overflow: pdfium bounded text](https://stackoverflow.com/questions/49726817/retrieve-the-text-content-with-bounds-from-pdf-document-using-pdfium-library), [PyShine: LiteParse architecture](https://pyshine.com/LiteParse-Fast-Lightweight-PDF-Parsing-Bounding-Boxes/)).
- Iterate `PdfPageTextChar` within each `PdfPageTextObject` to reconstruct word-level boxes (a text object may span multiple words on one baseline but never crosses a line break) ([pdfium-render#27](https://github.com/ajrcarey/pdfium-render/issues/27)).
- Group chars → words → lines via baseline clustering (y-coordinate tolerance + x-gap thresholding), then lines → blocks via vertical gap heuristics (naive column/paragraph detection, no ML).
- Render each page to a bitmap in parallel (needed later for layout-model inference and QA overlays).
- Emit raw text in reading order (top-to-bottom, left-to-right) with zero structural tagging — this is intentionally "dumb" text, no headings/tables/lists yet.

### Baseline Metrics
| Metric | Definition | Target |
|---|---|---|
| **Character extraction recall** | % of ground-truth characters recovered vs. dropped (ligatures, embedded fonts, CID-keyed text) | ≥ 99% |
| **Line-grouping accuracy** | % of lines correctly segmented by baseline clustering vs. manual annotation | ≥ 95% on single-column docs |
| **Throughput** | Pages/sec on reference hardware (single core, no OCR) | Established: **~333 pages/sec** (200 pages / 0.60s, Apple M2 Pro, single thread, `--jobs 1`) on the opendataloader-bench corpus's real-world PDFs — see [Stage 5](#stage-5--comparative-benchmarking). This is the perf floor for later stages, not a fixed target. |
| **Reading-order edit distance** | Levenshtein distance between extracted token order and ground-truth order | Record only — this is expected to be poor on multi-column layouts and drives Stage 3 |

**Exit criterion:** Character/word bbox extraction is lossless and fast on single-column, non-tabular PDFs (e.g., a plain-text arXiv preprint). Multi-column, tables, and formulas are *expected* to fail here — that failure surface is what Stage 3 characterizes.

---

## Stage 2 — Validation: TEDS + Layout Fidelity on DocLayNet

**Goal:** Quantify structural correctness against held-out ground truth, separating table-structure errors from generic layout errors.

### Datasets
- **[DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1)** (IBM Research): 80,863 human-annotated pages across 6 document categories (financial, scientific, patents, manuals, laws, tenders), with 11 class labels — `Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title` — each with pixel-precise bounding boxes ([HF: DocLayNet-v1.1 README](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1/blob/a00c59a78023ad82af42e3c2795e09025c7a80c0/README.md), [Hansimov/doc-layout-net category list](https://github.com/Hansimov/doc-layout-net)). Use the `pdf_cells` field (PDF text cells per bbox) to cross-check against our pdfium extraction.
- Reserve a stratified sample across all 6 document categories — layout error modes differ sharply between, e.g., financial reports (dense tables) and scientific papers (multi-column + formulas).

### Metric 1 — Table Structure: TEDS
- **TEDS (Tree-Edit-Distance-based Similarity)**, from PubTabNet, represents each table as an HTML tree and computes normalized tree-edit distance between predicted and ground-truth structure, capturing both cell topology and content simultaneously ([arXiv:1911.10683 — Image-based table recognition](https://arxiv.org/pdf/1911.10683), [blog.lomin.ai TEDS explainer](https://blog.lomin.ai/awesome-table-structure-recognition-33597)).
- Use **TEDS-Struct** (structure-only, ignoring cell text) to isolate row/column/span detection errors from downstream text-extraction errors, and standard **TEDS** for end-to-end fidelity.
- Also compute **TEDS(IOU)**, a text-independent variant that replaces string edit-distance on cell content with IoU distance between predicted/ground-truth cell bounding boxes — useful since our pipeline is OCR-free and bbox-driven by construction ([arXiv:2208.00385 — Evaluating Table Structure Recognition](http://arxiv.org/pdf/2208.00385.pdf)).
- Reference implementations: [IBM PubTabNet TEDS](https://github.com/ibm-aur-nlp/PubTabNet) or [SWHL/TableRecognitionMetric](https://github.com/SWHL/TableRecognitionMetric).

| Metric | Definition | Target |
|---|---|---|
| TEDS-Struct | Tree-edit similarity, structure only | ≥ 0.90 |
| TEDS (full) | Tree-edit similarity, structure + content | ≥ 0.85 |
| TEDS(IOU) | IoU-based cell matching variant | ≥ 0.88 |

### Metric 2 — Character/Region-Level Layout Fidelity: GIoU + F1
- **GIoU (Generalized IoU)** extends standard IoU with a penalty term for the smallest enclosing box around predicted and ground-truth boxes, so it remains informative even for non-overlapping boxes (critical for early-epoch layout models that predict boxes with no overlap at all). Compute per-region-class GIoU between predicted and DocLayNet ground-truth boxes for every one of the 11 categories.
- **F1-score** at the region-classification level: precision/recall of predicted region boxes matched to ground truth at a GIoU (or IoU) threshold (standard COCO-style matching, e.g., threshold 0.5), broken out **per class** — footnotes and page-headers/footers are the classes most likely to underperform and need isolated tracking.
- **Character-level layout fidelity**: for text regions, measure the fraction of ground-truth characters whose extracted bbox falls within the correctly classified parent region (i.e., did the character get assigned to "Text" vs. incorrectly merged into "Caption" or "Footnote").

| Metric | Definition | Target |
|---|---|---|
| Mean GIoU (all classes) | Average GIoU across 11 DocLayNet categories | ≥ 0.75 |
| Region F1 @ IoU 0.5 | Detection F1 per class, then macro-averaged | ≥ 0.85 macro-F1 |
| Footnote/header F1 (isolated) | F1 restricted to `Footnote`, `Page-header`, `Page-footer` | Track separately — historically the weakest classes |
| Character-to-region assignment accuracy | % characters assigned to correct semantic region | ≥ 92% |

**Exit criterion:** TEDS-Struct ≥ 0.90 on table-containing DocLayNet samples, mean GIoU ≥ 0.75 across all 11 classes, with per-class F1 reported (not just macro-averaged) so weak classes are visible going into Stage 3.

---

## Stage 3 — Error Analysis: Failure Modes in Spatial Parsing

**Goal:** Convert aggregate metric shortfalls into a taxonomy of concrete, reproducible failure modes, each tied to a minimal repro PDF.

### Process
1. Filter Stage 2 outputs to all samples below per-class GIoU/F1 threshold.
2. Cluster failures by geometric signature (e.g., "box merged across column gutter," "box split mid-cell," "box assigned wrong class despite correct geometry").
3. For each cluster, extract a minimal-reproduction PDF page and log it as a regression fixture.
4. Tag every failure with root cause: **geometric** (pdfium bbox wrong), **classification** (region model wrong), or **ordering** (reading-order assembly wrong) — this determines whether Stage 4 fixes are heuristic or model-level.

### Document-Structure Pitfall Checklist
Use this checklist to drive systematic sample collection — each item should have ≥ 20 DocLayNet (or supplementary) samples in the error-analysis corpus.

Legend: `[x]` fixed · `[~]` partial · `[ ]` open. This checklist and the table below are
generated from the live Stage 3 corpus by `eval::scoreboard`/
`examples/stage3_scoreboard.rs` — **never hand-edit the text between the
`<!-- BEGIN/END GENERATED -->` markers**; run
`cargo run --example stage3_scoreboard --features stage3 -- --write` to regenerate
them after any corpus or heuristic change (`-- --check` fails CI if they've drifted).
Per-pitfall prose (title, one-line summary, why-partial notes, blocker prerequisites)
lives in [`docs/pitfall_registry.json`](pitfall_registry.json) and is merged in by the
same generator; only the counts and pass/fail markers are re-derived every run.

<!-- BEGIN GENERATED: pitfall-checklist -->
Status reflects the Stage 3 regression corpus under [`fixtures/`](../fixtures) — run `cargo run --example stage3_scoreboard --features stage3` for the live scoreboard (currently 25/26 cases passing); see [`fixtures/README.md`](../fixtures/README.md) for per-case detail.

- [x] **Multi-column layout / gutter detection** — text lines merged across column boundaries, or column order inverted (right column read before left) *(3 cases, `ordering`; 3/3 passing — assemble::assemble_reading_order's XY-cut now qualifies a vertical gutter against the page's own median character size (MIN_GUTTER_EMS), not just a fixed page-width fraction (MIN_CUT_FRACTION), so narrow (1 em) column gutters recover column-major order the same way wide ones already did)*
- [x] **Footnotes** — footnote text merged into body `Text` region instead of classified as `Footnote`; footnote markers (superscript numerals) not linked to their note *(2 cases, `classification`; 2/2 passing — classified via layout::classify_block's footnote rule (requires a block to sit low on the page, carry a smaller font than the rest of the page, and open with a recognized footnote marker). serialize::render_block now has a Footnote arm too (`[^marker]: text`, marker taken from the block's own leading digit/symbol) -- marker-to-note *linking* (resolving a superscript reference elsewhere in the body back to its note) is still open.)*
- [x] **Page headers/footers repeated across pages** — header/footer text leaking into the body text stream; running headers misclassified as `Section-header` *(3 cases, `classification`; 3/3 passing — layout::band_of's shape test (thin strip, detached from the body by a minimum gap) plus a cross-page repeated_running_bands pass, suppressed in serialize::render_block)*
- [x] **Nested/multi-line table cells** — cells spanning multiple visual lines merged incorrectly with adjacent rows; rowspan/colspan not detected *(1 case, `classification`; 1/1 passing)*
- [x] **Merged table cells (rowspan/colspan)** — TEDS-Struct penalizes tree-topology mismatches here most heavily *(2 cases, `classification`; 2/2 passing — graphics::detect_table_regions + graphics::table_grid_cells (Stage 1b) now detect a bordered table from its ruling lines and assign a block spanning multiple grid cells to every cell it overlaps, resolving a merged/rowspan cell correctly -- see the corpus cases now carrying synthetic `graphics` ruling-line fixtures.)*
- [x] **Borderless / whitespace-delimited tables** — no ruling lines for the layout model to key off; frequently misclassified as plain `Text` *(2 cases, `classification`; 2/2 passing — Closed by layout::whitespace_column_corridors — a shared-whitespace-corridor heuristic over a single block's lines, complementing detect_borderless_table_regions's side-by-side-blocks shape.)*
- [~] **Nested mathematical formulas** — inline formulas embedded mid-sentence not separated from surrounding text; multi-line/stacked formulas (fractions, summations, matrices) fragmented into multiple bounding boxes *(2 cases, `classification`; 2/2 passing — Display (block-level) formulas are now classified via layout::is_display_formula -- a geometric-plus-lexical heuristic requiring a block to be centered, narrow relative to the page, vertically isolated from its neighbours by roughly a line height, dense with math symbols/digits, and not end like a sentence. Both seeded cases (an isolated display formula and a two-line stacked fraction) now pass. *Inline* formulas embedded mid-sentence in a body paragraph are still open -- see blocked below.)*
- [x] **Superscript/subscript handling** — footnote markers, exponents, and chemical/math subscripts causing baseline-clustering errors in Stage 1's line grouping *(1 case, `geometric`; 1/1 passing — extract::group_words's is_script_continuation check now recognizes a smaller, baseline-offset, nearby character as a super/subscript of its predecessor and attaches it to its base instead of splitting it off)*
- [x] **Rotated or vertical text** — sidebar labels, rotated table headers, CJK vertical text (relevant given Traditional Chinese/Japanese document support) *(1 case, `geometric`; 1/1 passing — Handles 90-degree-rotated Latin runs (sidebar labels) via extract::merge_rotated_text_runs. Char::angle_degrees now also carries PDFium's own reported glyph rotation directly (extract::extract_char), rather than only the geometric run-inference merge_rotated_text_runs falls back to when it's unavailable. CJK vertical writing mode is still unhandled.)*
- [x] **Figure/caption association** — captions not correctly linked to their parent `Picture`/`Table` region, or associated with the wrong figure when multiple figures share a page *(2 cases, `classification`; 2/2 passing — layout::is_caption covers the caption half. graphics::detect_picture_regions (Stage 1b) now emits RegionClass::Picture from Image/XObjectForm page objects (plus large text-free Fill graphics) -- see the corpus cases now carrying synthetic `graphics` image fixtures. Associating a caption with the *correct* Picture when multiple figures share a page (nearest-neighbor matching) is still open.)*
- [~] **List-item nesting** — multi-level bullet/numbered lists collapsed into flat `List-item` blocks, losing hierarchy *(2 cases, `ordering`; 2/2 passing — reading order now passes after the narrow-gutter fix shared with multi-column above -- each side-by-side outline is read fully before the next; hierarchy/depth itself is still not modeled, so nested lists still flatten into one flat sequence)*
- [x] **Cross-page table/paragraph continuation** — tables or paragraphs split across a page boundary not stitched back together *(1 case, `ordering`; 1/1 passing)*
- [x] **Section headers vs. bold body text** — `Section-header` misclassified as `Title` or emphasized `Text` when font-weight is the only visual cue *(2 cases, `classification`; 2/2 passing — Char::font_weight now carries PDFium's own reported font weight (extract::extract_char, via PdfPageTextChar::font_weight) alongside the existing font-*name* heuristic (layout::is_bold_font_name). The classifier itself does not yet threshold on the numeric weight -- see unblocked_by.)*
- [ ] **Embedded/CID-keyed fonts and ligatures** — pdfium character extraction dropping or garbling glyphs from non-standard font encodings *(1 case, `geometric`; 0/1 passing)*
- [x] **Overlapping/z-ordered text objects** — watermarks, stamps, or redaction boxes overlapping real text and corrupting bbox clustering *(1 case, `geometric`; 1/1 passing)*
<!-- END GENERATED: pitfall-checklist -->

#### Status & next steps

<!-- BEGIN GENERATED: pitfall-blockers -->
| Blocker | Pitfalls | Next step |
|---|---|---|
| This pitfall's PDF-backed corpus case fails one step upstream of classification: real Stage 1 block grouping (extract::group_chars_into_blocks) already merges the row's cells together across the column gap into one block, so no separate per-cell block exists left for graphics::table_grid_cells to assign to a grid cell. graphics.rs (Stage 1b) now detects the table and reconstructs its grid from ruling lines correctly wherever cells survive as distinct blocks -- see merged_table_cell and figure_caption, unblocked by the same code path -- but this case's problem is upstream of that. | Nested/multi-line table cells | Teach extract::group_chars_into_blocks to stop a line/block merge at a ruling line boundary from Page::graphics, instead of only using vertical-gap/x-gap heuristics blind to the table grid. |
| An inline formula shares its block with ordinary prose, so there is no separate bounding box or whitespace/isolation signal for a geometric heuristic to key off the way there is for a display formula -- distinguishing 'e=mc^2' embedded mid-sentence from surrounding text needs actual layout/vision understanding, not just geometry. | Nested mathematical formulas | Stage 4b vision detector -- inline formula segmentation is the one sub-case in this registry that still needs it. |
| `assemble_reading_order` is single-page | Cross-page table/paragraph continuation | Cross-page stitching (Stage 4a, third bullet) |
| layout::classify_block's bold-heading signal still only reads the font *name* (is_bold_font_name), not the new Char::font_weight field -- the signal now exists in the data model but classify_block hasn't been updated to prefer it over the name heuristic. | Section headers vs. bold body text | Thread a block's median Char::font_weight into classify_block and prefer it over is_bold_font_name when present (a font that declares its weight numerically is a stronger signal than a name-string guess). |
| pdfium-render 0.9.3 (and PDFium's FPDF_TEXTPAGE API) expose no raw charcode, /Encoding, or /Differences accessor, so Stage 1 cannot re-resolve glyph-name-derived Unicode against the base encoding. Separately, the corpus case's `page` is a frozen extract_baseline snapshot, so an extract.rs fix cannot move the scoreboard until examples/stage3_pdf_cases.rs regenerates it. Note this is a narrower gap than it first appears: PDFium resolves ToUnicode internally for the common case, so this is specifically about non-standard/custom /Differences encodings PDFium's own resolution doesn't cover -- not a general 'PDFium can't read fonts' problem. | Embedded/CID-keyed fonts and ligatures | A charcode-level PDF reader independent of PDFium -- e.g. `lopdf` (xref + FlateDecode + content-stream tokenizing) paired with an Adobe Glyph List table, walking each page's /Font resource dict's /Encoding /Differences to re-resolve glyph names the way firecrawl/pdf-inspector does on the same lopdf substrate -- or an upstream pdfium-render API exposing per-char charcodes; then regenerate the PDF-backed snapshot. |
| Page::graphics now carries a real page-object paint-order z-index (extract::extract_graphics), and Char::fill_color now carries each character's own fill color -- both signals this pitfall needs are now in the data model. Nothing yet reads them to resolve an overlap (e.g. drop a low-z, faint-fill-colored watermark run before it corrupts bbox clustering). | Overlapping/z-ordered text objects | In extract.rs or layout.rs, use Char::fill_color (a very light gray/faint fill is a watermark signal) and relative z-order between overlapping text runs to suppress or de-prioritize the watermark run before block clustering. |
<!-- END GENERATED: pitfall-blockers -->

Per-category sample volume (≥ 20 DocLayNet samples per pitfall, per the target
above) remains future work pending a real DocLayNet mining pass — see
[`fixtures/README.md`](../fixtures/README.md)'s "Getting a real DocLayNet-core
sample" section. Today's corpus seeds 1–3 hand-authored or PDF-backed cases per
pitfall, enough to drive and validate Stage 4 fixes but not yet at sampling
scale.

### Failure-Mode Metrics
| Metric | Purpose |
|---|---|
| Failure count per pitfall category | Prioritization signal for Stage 4 backlog |
| Root-cause split (geometric / classification / ordering) | Determines heuristic vs. model-retraining response |
| Regression corpus size per category | Ensures Stage 4 fixes are validated, not just anecdotal |

**Exit criterion:** Every pitfall category above has a labeled regression corpus and a root-cause tag; failure counts are ranked to define the Stage 4 priority order. <!-- BEGIN GENERATED: pitfall-exit-criterion -->
**Met** — all 15 pitfalls are seeded in the corpus with a root-cause tag and a live pass/fail scoreboard (25/26 cases passing today, 6 pitfalls blocked on a named prerequisite above); the one open gap is volume — each category has 1–3 cases, short of the ≥ 20-samples-per-category target, which needs a real DocLayNet mining pass to close.
<!-- END GENERATED: pitfall-exit-criterion -->

---

## Stage 4 — Refinement: Heuristics + Targeted Fine-Tuning

**Goal:** Close the gaps identified in Stage 3 via the cheapest effective fix first, escalating to model fine-tuning only where heuristics plateau.

### 4a. Heuristic Adjustments (fast iteration, no retraining)
- **Geometric root causes** → fix directly in the pdfium assembly layer:
  - Column-gutter detection via vertical whitespace-density histograms before line-grouping.
  - Baseline-clustering tolerance tuned per font-size bucket to fix superscript/subscript misgrouping.
  - Cross-page stitching via bottom-of-page/top-of-next-page bbox adjacency + incomplete-sentence detection.
  - Header/footer suppression via repeated-content detection across N consecutive pages (same text at same y-position range).
- **Ordering root causes** → adjust reading-order graph construction (e.g., switch from pure top-to-bottom scan to column-aware XY-cut recursive segmentation for multi-column pages).
- Validate each heuristic change against the Stage 3 regression corpus **before** re-running full Stage 2 metrics, to catch regressions cheaply.

### 4b. Targeted Object-Detection Fine-Tuning (classification root causes)
- Fine-tune a lightweight detector (e.g., RT-DETR-style, matching the approach used in Docling's ONNX layout stage: resize to 640×640, sigmoid + top-k class×query matching, box decode to page scale — [docs.rs: docling-pdf](https://docs.rs/docling-pdf/latest/docling_pdf/)) on the specific under-performing classes identified in Stage 3 (typically `Footnote`, `Page-header`/`Page-footer`, `Formula`).
- Use **hard-negative mining**: oversample the regression corpus pages where these classes were confused with `Text`/`Section-header`.
- Fine-tune incrementally per class cluster rather than full retraining — freeze backbone, fine-tune detection head on the augmented DocLayNet subset + custom hard-negative samples.
- For nested formulas specifically, consider a two-stage approach: coarse `Formula` region detection, then recursive sub-box detection within formula regions for stacked/multi-line structures.

### Refinement-Loop Metrics (same metrics as Stage 2, tracked as deltas)
| Metric | Before → After tracking |
|---|---|
| Mean GIoU (targeted classes) | e.g., Footnote GIoU 0.61 → target ≥ 0.75 |
| Per-class F1 (targeted classes) | e.g., Page-header F1 0.70 → target ≥ 0.85 |
| TEDS-Struct | Should not regress from heuristic changes to reading-order/column logic |
| Regression corpus pass rate | % of Stage 3 minimal-repro cases now passing | Target 100% before closing the loop |
| Throughput delta | Heuristics should add negligible latency; fine-tuned models add inference cost — track pages/sec regression |

**Exit criterion:** Targeted classes hit Stage 2 target thresholds on the regression corpus with no throughput regression beyond an agreed budget (e.g., <10% latency increase from added inference).

---

## Stage 5 — Comparative Benchmarking

**Goal:** Measure end-to-end Markdown quality against other local, model-free PDF extractors on a real-world corpus — not just intrinsic fidelity against DocLayNet's own ground-truth boxes (Stage 2), but whether the *shipped* tool is actually competitive. Stage 2 asks "are our region boxes and table structure accurate against annotated ground truth?"; Stage 5 asks "does the Markdown we emit read as well as a competitor's, on PDFs nobody hand-picked for us?" Keeping the two separate preserves that distinction rather than diluting Stage 2 with an external corpus it wasn't designed around.

**Corpus:** [opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench) (Apache-2.0), 200 real-world PDFs with hand-authored ground-truth Markdown.

**Metrics** (from the bench's own evaluator, `[0, 1]` unless noted):
| Metric | What it scores |
|---|---|
| Overall | Mean of NID, TEDS, MHS per document, averaged across documents |
| NID / NID-S | Reading-order text similarity (whitespace-normalized, table content included/excluded) |
| TEDS / TEDS-S | Table structure similarity (tree-edit distance, content included/excluded) |
| MHS / MHS-S | Heading-level sequence similarity |
| s/doc | Wall time per document, single process, sequential |

**Method:** `./scripts/run-opendataloader-bench.sh` clones the corpus, builds a release `pdfspatial` binary, registers it (and `pdfspatial-compact`, `pdf-inspector`) as bench engines without forking the upstream repo, runs every engine over all 200 PDFs, and collects `bench/opendataloader/results/results.json` plus the README table. See [`bench/opendataloader/README.md`](../bench/opendataloader/README.md) for the full methodology, engine list, and known scoring asymmetries (the bench's NID doesn't strip Markdown syntax; our TEDS/MHS are structurally capped by what `serialize::to_markdown_structured` emits today).

**Non-goal:** Stage 5 does not gate merges, and it never runs in CI — it needs a multi-GB Python environment, network access, and an otherwise-idle machine for a speed number to mean anything (see the bench README). Run it per release, by hand, not per pull request.

**Exit criterion:** None committed at 0.1.0 — the goal is a reproducible, hardware-labelled table checked into the repo, not a numeric threshold. A future iteration may set one once there's a baseline trend to compare against.

---

## Closing the Loop

After Stage 4, re-run the **full Stage 2 validation suite** on a fresh DocLayNet sample split (not the one used for fine-tuning) to confirm generalization, then return to Stage 3 error analysis on the new metric shortfalls; re-run Stage 5 per release, independently of this loop's cadence. This is a continuous loop, not a linear pipeline — each iteration should:

1. Re-run Stage 1 baseline only if pdfium extraction logic changed.
2. Re-score Stage 2 metrics (TEDS-Struct, TEDS, GIoU, F1) on held-out data.
3. Re-cluster Stage 3 failures against the updated pitfall checklist — expect new pitfalls to surface as old ones close.
4. Prioritize Stage 4 fixes by failure count × user-facing impact (e.g., table errors typically outrank footnote errors for most downstream RAG/LLM use cases).

## Summary Metrics Dashboard (track every iteration)

| Stage | Primary Metrics | Target |
|---|---|---|
| 1. Baseline | Char recall, line-grouping accuracy, throughput | ≥99% recall, ≥95% line accuracy |
| 2. Validation | TEDS-Struct, TEDS, TEDS(IOU), mean GIoU, region F1 | ≥0.90 TEDS-Struct, ≥0.75 GIoU, ≥0.85 macro-F1 |
<!-- BEGIN GENERATED: pitfall-dashboard-row -->
| 3. Error Analysis | Failure count/category, root-cause split | Full pitfall-checklist coverage — **met**: 15/15 pitfalls seeded, root-cause-tagged, scoreboard-tracked (25/26 cases passing); ≥20-samples/category volume still outstanding |
<!-- END GENERATED: pitfall-dashboard-row -->
| 4. Refinement | Per-class GIoU/F1 delta, regression pass rate, throughput delta | 100% regression pass, <10% latency cost |
| 5. Comparative Benchmarking | Overall/NID/TEDS/MHS + s/doc vs. external engines on opendataloader-bench | Reproducible, hardware-labelled table checked in; no numeric threshold committed at 0.1.0 |

---

### Key References
- [ajrcarey/pdfium-render (GitHub)](https://github.com/ajrcarey/pdfium-render)
- [pdfium-render Issue #27 — word-level bbox extraction](https://github.com/ajrcarey/pdfium-render/issues/27)
- [Stack Overflow — pdfium bounded text extraction](https://stackoverflow.com/questions/49726817/retrieve-the-text-content-with-bounds-from-pdf-document-using-pdfium-library)
- [docling-pdf Rust crate docs](https://docs.rs/docling-pdf/latest/docling_pdf/)
- [DocLayNet-v1.1 dataset (Hugging Face)](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1/blob/a00c59a78023ad82af42e3c2795e09025c7a80c0/README.md)
- [DocLayNet category schema (Hansimov/doc-layout-net)](https://github.com/Hansimov/doc-layout-net)
- [Zhong et al., "Image-based table recognition: data, model, and evaluation" (TEDS origin), arXiv:1911.10683](https://arxiv.org/pdf/1911.10683)
- ["Evaluating Table Structure Recognition" — TEDS(IOU), arXiv:2208.00385](http://arxiv.org/pdf/2208.00385.pdf)
- [PubTabNet TEDS reference implementation (IBM)](https://github.com/ibm-aur-nlp/PubTabNet)
- [SWHL/TableRecognitionMetric](https://github.com/SWHL/TableRecognitionMetric)
- [LiteParse architecture writeup (PyShine)](https://pyshine.com/LiteParse-Fast-Lightweight-PDF-Parsing-Bounding-Boxes/)

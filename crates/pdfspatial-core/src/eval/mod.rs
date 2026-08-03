//! Stage 2 dataset evaluation harness: aggregates the per-comparison metrics in
//! [`crate::metrics`] across a page-level dataset into a single [`Stage2Report`]
//! dashboard.
//!
//! This module is intentionally split in parts:
//!
//! - This file (`eval`) is the pure aggregation core: [`evaluate_pages`] takes
//!   already-built `(predicted, ground_truth)` region pairs and produces a
//!   [`Stage2Report`]. It has no dependencies beyond [`crate::metrics`] and
//!   [`crate::layout`], so it stays compiled unconditionally and is reusable by
//!   benches, examples, and tests alike.
//! - [`doclaynet`] (behind the `doclaynet` cargo feature) loads a real
//!   [DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1) sample
//!   from disk and turns it into the `(predicted, ground_truth)` pairs this module
//!   consumes.
//! - [`corpus`] (behind the `stage3` cargo feature) loads the Stage 3 regression
//!   corpus — hand-authored, minimal-repro cases tagged by failure-mode taxonomy — and
//!   checks the pipeline's actual behavior against each case's desired post-fix
//!   behavior.
//! - [`scoreboard`] (behind the `stage3` cargo feature) aggregates [`corpus`]'s
//!   per-case outcomes by pitfall and renders them into the generated blocks embedded
//!   in the roadmap and `fixtures/README.md` -- see `examples/stage3_scoreboard.rs`.

#[cfg(feature = "doclaynet")]
pub mod doclaynet;

#[cfg(feature = "stage3")]
pub mod corpus;

#[cfg(feature = "stage3")]
pub mod scoreboard;

use crate::layout::{Region, RegionClass};
use crate::metrics;
use crate::{Block, Document, Page, assemble::assemble_reading_order};
use std::collections::BTreeMap;

/// A Stage 2 validation dashboard: region-detection fidelity aggregated over a set of
/// pages, matching the roadmap's Stage 2 metric table.
///
/// Produced by [`evaluate_pages`]. Uses [`BTreeMap`] (rather than
/// [`std::collections::HashMap`]) for its per-class fields so iteration order — and
/// therefore any printed report — is deterministic.
#[derive(Debug, Clone, PartialEq)]
pub struct Stage2Report {
    /// Number of pages the report was aggregated over.
    pub page_count: usize,
    /// The IoU threshold used for matching predictions to ground truth.
    pub iou_threshold: f32,
    /// Per-class F1, with true/false positive/negative counts pooled across every page
    /// before scoring (not averaged per-page). Mirrors [`crate::metrics::region_f1`].
    pub region_f1: BTreeMap<RegionClass, f32>,
    /// Macro-average of [`region_f1`](Self::region_f1) — the mean F1 over classes with
    /// at least one prediction or ground-truth region anywhere in the dataset.
    pub macro_region_f1: f32,
    /// Per-class mean [`crate::metrics::giou`] restricted to true-positive matched
    /// pairs (i.e. localization tightness of correctly detected regions, not detection
    /// recall/precision — that's what [`region_f1`](Self::region_f1) captures).
    pub mean_giou_per_class: BTreeMap<RegionClass, f32>,
    /// Macro-average of [`mean_giou_per_class`](Self::mean_giou_per_class) over classes
    /// with at least one true-positive match. Classes with zero matches (e.g. classes
    /// [`crate::layout::classify_regions`] never predicts) are excluded from this
    /// average but remain visible with `F1 = 0.0` in [`region_f1`](Self::region_f1).
    pub mean_giou: f32,
    /// F1 for [`RegionClass::Footnote`] alone, tracked in isolation per the roadmap —
    /// historically one of the weakest classes.
    pub footnote_f1: f32,
    /// F1 for [`RegionClass::PageHeader`] alone, tracked in isolation.
    pub page_header_f1: f32,
    /// F1 for [`RegionClass::PageFooter`] alone, tracked in isolation.
    pub page_footer_f1: f32,
    /// Ground-truth region count per class, pooled across every page. Lets a reader
    /// distinguish "this class scored 0 because it never occurs in the data" from "this
    /// class scored 0 despite occurring" (e.g. classes the heuristic classifier can
    /// never produce).
    pub support: BTreeMap<RegionClass, usize>,
    /// Table-structure TEDS score, if computed. `None` in a harness whose predictions
    /// never include table-structure HTML (e.g. the text-only [`doclaynet`] path, since
    /// [`crate::layout::classify_regions`] never emits [`RegionClass::Table`]) — the
    /// TEDS family ([`crate::metrics::teds`], [`crate::metrics::teds_struct`],
    /// [`crate::metrics::teds_iou`]) stays exercised by its own unit tests and is wired
    /// into a report only once a table-structure predictor exists to score.
    pub teds: Option<f64>,
}

/// Aggregates the Stage 2 dashboard over a dataset of per-page `(predicted,
/// ground_truth)` region pairs.
///
/// Matching is performed independently per page via [`crate::metrics::match_regions`]
/// (regions on different pages never match each other); true/false positive/negative
/// counts are then pooled across all pages before per-class F1 is computed, so a page
/// with many ground-truth regions is weighted proportionally to its size rather than
/// averaged equally with a sparse page.
///
/// Returns a [`Stage2Report`] with every count at zero and every rate at `0.0` if
/// `pages` is empty.
pub fn evaluate_pages(pages: &[(Vec<Region>, Vec<Region>)], iou_threshold: f32) -> Stage2Report {
    let mut pooled: BTreeMap<RegionClass, (usize, usize, usize)> = BTreeMap::new();
    let mut giou_sum: BTreeMap<RegionClass, f64> = BTreeMap::new();
    let mut giou_count: BTreeMap<RegionClass, usize> = BTreeMap::new();
    let mut support: BTreeMap<RegionClass, usize> = BTreeMap::new();

    for (predicted, ground_truth) in pages {
        for gt in ground_truth {
            *support.entry(gt.class).or_insert(0) += 1;
        }

        for (class, class_match) in metrics::match_regions(predicted, ground_truth, iou_threshold) {
            let entry = pooled.entry(class).or_insert((0, 0, 0));
            entry.0 += class_match.true_positives;
            entry.1 += class_match.false_positives;
            entry.2 += class_match.false_negatives;

            for &(pred_index, gt_index) in &class_match.matches {
                let score = metrics::giou(predicted[pred_index].bbox, ground_truth[gt_index].bbox);
                *giou_sum.entry(class).or_insert(0.0) += score as f64;
                *giou_count.entry(class).or_insert(0) += 1;
            }
        }
    }

    let region_f1: BTreeMap<RegionClass, f32> = pooled
        .iter()
        .map(|(&class, &(tp, fp, fn_))| {
            let denom = 2 * tp + fp + fn_;
            let f1 = if denom > 0 {
                (2 * tp) as f32 / denom as f32
            } else {
                0.0
            };
            (class, f1)
        })
        .collect();

    let macro_region_f1 = mean(region_f1.values().copied());

    let mean_giou_per_class: BTreeMap<RegionClass, f32> = giou_sum
        .iter()
        .map(|(&class, &sum)| {
            let count = giou_count[&class] as f64;
            (class, (sum / count) as f32)
        })
        .collect();

    let mean_giou = mean(mean_giou_per_class.values().copied());

    let class_f1 = |class: RegionClass| region_f1.get(&class).copied().unwrap_or(0.0);

    Stage2Report {
        page_count: pages.len(),
        iou_threshold,
        macro_region_f1,
        mean_giou_per_class,
        mean_giou,
        footnote_f1: class_f1(RegionClass::Footnote),
        page_header_f1: class_f1(RegionClass::PageHeader),
        page_footer_f1: class_f1(RegionClass::PageFooter),
        region_f1,
        support,
        teds: None,
    }
}

/// One page's reading-order reordering severity: how far
/// [`assemble_reading_order`]'s XY-cut order diverges from the naive, as-extracted
/// block order it started from.
///
/// This is an **unsupervised reordering-magnitude proxy**, not an accuracy score
/// against a gold reading order — see [`rank_pages_by_reorder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageReorderRank {
    /// The page's [`crate::Page::index`].
    pub page_index: usize,
    /// Number of blocks on the page (context for interpreting the distance below --
    /// a distance of 3 means little on a 3-block page but a lot on a 30-block one).
    pub block_count: usize,
    /// [`metrics::reading_order_edit_distance`] between the page's naive (as-extracted)
    /// block order and [`assemble_reading_order`]'s output, using each block's
    /// [`crate::Block::text()`] as its identity (so two distinct blocks with identical
    /// text are indistinguishable to this signal -- acceptable for a prioritizer, since
    /// it only ever under- or over-counts a tie, never flips the ranking direction).
    pub reorder_edit_distance: usize,
}

/// Ranks a document's pages by reading-order reordering severity, most-reordered
/// first (ties broken by ascending [`PageReorderRank::page_index`] for determinism).
///
/// # Why this, not a gold-order accuracy score
///
/// The roadmap's Stage 3 process needs a signal for *which* pages to mine into
/// minimal-repro regression fixtures first. Real datasets like DocLayNet ship region
/// bounding boxes, not an authored reading-order sequence, so there is no gold order to
/// score against outside the hand-authored `fixtures/` corpus (see
/// [`crate::eval::corpus`]). Instead, this measures how many block moves
/// [`assemble_reading_order`] applied relative to the naive top-to-bottom,
/// left-to-right order [`crate::extract`] emits: a page where XY-cut barely touches the
/// order is (by construction) one where the naive order was already plausible, while a
/// page XY-cut heavily reorders is exactly the kind of multi-column/complex layout
/// Stage 1's reading-order metric is expected to fail on -- the highest-value page to
/// mine next.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::eval::rank_pages_by_reorder;
/// use pdfspatial_core::{BBox, Block, Document, Line, Page, Word};
///
/// fn block(text: &str, bbox: BBox) -> Block {
///     let word = Word { text: text.into(), bbox, chars: vec![] };
///     let line = Line { text: text.into(), bbox, words: vec![word] };
///     Block { bbox, lines: vec![line] }
/// }
///
/// // Two columns, authored in interleaved (naive) order -- XY-cut recovers
/// // column-major order, so this page reorders heavily.
/// let multi_column = Page {
///     index: 0,
///     width: 600.0,
///     height: 800.0,
///     blocks: vec![
///         block("Right one", BBox { left: 320.0, bottom: 700.0, right: 560.0, top: 750.0 }),
///         block("Left one", BBox { left: 40.0, bottom: 700.0, right: 280.0, top: 750.0 }),
///         block("Right two", BBox { left: 320.0, bottom: 640.0, right: 560.0, top: 690.0 }),
///         block("Left two", BBox { left: 40.0, bottom: 640.0, right: 280.0, top: 690.0 }),
///     ],
///     ..Default::default()
/// };
/// // A single-column page already in top-to-bottom order -- XY-cut leaves it alone.
/// let single_column = Page {
///     index: 1,
///     width: 600.0,
///     height: 800.0,
///     blocks: vec![
///         block("First", BBox { left: 40.0, bottom: 700.0, right: 560.0, top: 750.0 }),
///         block("Second", BBox { left: 40.0, bottom: 600.0, right: 560.0, top: 650.0 }),
///     ],
///     ..Default::default()
/// };
///
/// let document = Document { pages: vec![multi_column, single_column] };
/// let ranked = rank_pages_by_reorder(&document);
///
/// assert_eq!(ranked[0].page_index, 0);
/// assert!(ranked[0].reorder_edit_distance > 0);
/// assert_eq!(ranked[1].page_index, 1);
/// assert_eq!(ranked[1].reorder_edit_distance, 0);
/// ```
pub fn rank_pages_by_reorder(document: &Document) -> Vec<PageReorderRank> {
    let assembled = assemble_reading_order(document);

    let mut ranks: Vec<PageReorderRank> = document
        .pages
        .iter()
        .zip(&assembled.pages)
        .map(|(naive_page, assembled_page)| {
            let naive_order: Vec<String> = naive_page.blocks.iter().map(|b| b.text()).collect();
            let assembled_order: Vec<String> =
                assembled_page.blocks.iter().map(|b| b.text()).collect();
            let naive_refs: Vec<&str> = naive_order.iter().map(String::as_str).collect();
            let assembled_refs: Vec<&str> = assembled_order.iter().map(String::as_str).collect();

            PageReorderRank {
                page_index: naive_page.index,
                block_count: naive_page.blocks.len(),
                reorder_edit_distance: metrics::reading_order_edit_distance(
                    &assembled_refs,
                    &naive_refs,
                ),
            }
        })
        .collect();

    ranks.sort_by(|a, b| {
        b.reorder_edit_distance
            .cmp(&a.reorder_edit_distance)
            .then(a.page_index.cmp(&b.page_index))
    });
    ranks
}

/// A minimized reading-order repro: the smallest subset of a page's blocks (found by
/// [`minimize_reorder_repro`]) that [`assemble_reading_order`] still reorders relative
/// to its naive (as-extracted) order.
#[derive(Debug, Clone, PartialEq)]
pub struct ReproMinimization {
    /// The minimized page: a subset of the original's blocks (in their original,
    /// naive order), with the original `index`/`width`/`height` preserved -- XY-cut's
    /// gutter/gap thresholds are page-relative, so shrinking the page dimensions would
    /// change *why* the repro reorders, not just its size.
    pub page: Page,
    /// The number of blocks on the page passed to [`minimize_reorder_repro`], before
    /// minimization -- context for judging how much smaller [`page`](Self::page) is.
    pub original_block_count: usize,
    /// [`metrics::reading_order_edit_distance`] between the minimized page's naive
    /// order and [`assemble_reading_order`]'s output on it -- always `> 0`, since a
    /// page that doesn't reorder is never returned (see [`minimize_reorder_repro`]).
    pub reorder_edit_distance: usize,
}

/// Finds a minimal subset of `page`'s blocks that still reproduces a reading-order
/// reordering, for turning a [`PageReorderRank`]-flagged real-dataset page into a
/// reviewable `fixtures/` regression case instead of dumping the whole (often
/// hundreds-of-blocks) page.
///
/// # Why this, and why greedy
///
/// Real pages mined via [`crate::eval::doclaynet::document_from_cells`] reconstruct to
/// one block per PDF text cell -- far too large to hand-author as a minimal repro, and
/// the corpus schema (see [`crate::eval::corpus`]) identifies blocks by their exact
/// text, which real pages routinely repeat (page numbers, running headers, "Table 1").
/// This function first deduplicates by block text (the same identity
/// [`metrics::reading_order_edit_distance`] already uses via
/// [`PageReorderRank::reorder_edit_distance`], so a duplicate-text block already makes
/// that signal ambiguous -- not merely a corpus-schema workaround), then greedily
/// removes blocks one at a time (from the end, backward), keeping each removal only if
/// the remaining blocks still reorder. This is a textbook greedy shrink: `O(n)`
/// predicate evaluations for an `n`-block page, each evaluation itself
/// [`assemble_reading_order`] plus an `O(n)` sequence comparison. It only guarantees
/// 1-minimality (no single further block can be dropped), not a globally smallest
/// witness -- a full delta-debugging (`ddmin`) pass could shrink further in some cases,
/// but is disproportionate here: greedy already collapses a page-full of cells down to
/// the handful of blocks that actually drive the reorder.
///
/// Returns `None` if `page` doesn't reorder to begin with (nothing to reproduce), or if
/// deduplication alone removes the reordering signal (e.g. it depended on two blocks
/// sharing text) -- both cases the caller should report as skipped rather than emit as
/// an empty or misleading case.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::eval::minimize_reorder_repro;
/// use pdfspatial_core::{BBox, Block, Line, Page, Word};
///
/// fn block(text: &str, bbox: BBox) -> Block {
///     let word = Word { text: text.into(), bbox, chars: vec![] };
///     let line = Line { text: text.into(), bbox, words: vec![word] };
///     Block { bbox, lines: vec![line] }
/// }
///
/// fn bbox(left: f32, bottom: f32, right: f32, top: f32) -> BBox {
///     BBox { left, bottom, right, top }
/// }
///
/// // A two-column pair (interleaved -- reorders) plus several single-column "noise"
/// // blocks that don't contribute to the reordering.
/// let page = Page {
///     index: 0,
///     width: 600.0,
///     height: 800.0,
///     blocks: vec![
///         block("Right one", bbox(320.0, 700.0, 560.0, 750.0)),
///         block("Left one", bbox(40.0, 700.0, 280.0, 750.0)),
///         block("Right two", bbox(320.0, 640.0, 560.0, 690.0)),
///         block("Left two", bbox(40.0, 640.0, 280.0, 690.0)),
///         block("Noise A", bbox(40.0, 500.0, 560.0, 550.0)),
///         block("Noise B", bbox(40.0, 400.0, 560.0, 450.0)),
///     ],
///     ..Default::default()
/// };
///
/// let minimized = minimize_reorder_repro(&page).expect("page reorders");
/// assert!(minimized.page.blocks.len() < minimized.original_block_count);
/// assert!(minimized.reorder_edit_distance > 0);
/// ```
pub fn minimize_reorder_repro(page: &Page) -> Option<ReproMinimization> {
    let original_block_count = page.blocks.len();

    let mut seen = std::collections::HashSet::new();
    let mut blocks: Vec<Block> = Vec::new();
    for block in &page.blocks {
        if seen.insert(block.text()) {
            blocks.push(block.clone());
        }
    }

    if !reorders(&blocks, page.width, page.height) {
        return None;
    }

    let mut index = blocks.len();
    while index > 0 {
        index -= 1;
        let removed = blocks.remove(index);
        if !reorders(&blocks, page.width, page.height) {
            blocks.insert(index, removed);
        }
    }

    let naive_order: Vec<String> = blocks.iter().map(|b| b.text()).collect();
    let scratch = Page {
        index: page.index,
        width: page.width,
        height: page.height,
        blocks: blocks.clone(),
        ..Default::default()
    };
    let assembled = assemble_reading_order(&Document {
        pages: vec![scratch],
    });
    let assembled_order: Vec<String> = assembled.pages[0].blocks.iter().map(|b| b.text()).collect();
    let naive_refs: Vec<&str> = naive_order.iter().map(String::as_str).collect();
    let assembled_refs: Vec<&str> = assembled_order.iter().map(String::as_str).collect();

    Some(ReproMinimization {
        page: Page {
            index: page.index,
            width: page.width,
            height: page.height,
            blocks,
            ..Default::default()
        },
        original_block_count,
        reorder_edit_distance: metrics::reading_order_edit_distance(&assembled_refs, &naive_refs),
    })
}

/// `true` if [`assemble_reading_order`] changes `blocks`' text order at all, i.e. the
/// reordering predicate [`minimize_reorder_repro`]'s greedy shrink preserves.
///
/// Note: this compares block *text* before and after assembly, on the assumption that
/// assembly only reorders blocks. [`assemble_reading_order`] also runs
/// [`crate::extract::merge_rotated_text_runs`], which can rewrite a block's text (merging
/// a rotated run of lines) without moving the block at all. That pass is structurally
/// inert on DocLayNet-derived pages (each block there is already one line), so it cannot
/// affect this function today -- but a future caller feeding multi-line blocks through
/// `reorders`/[`minimize_reorder_repro`] should not assume a `true`/nonzero result means
/// the blocks moved.
fn reorders(blocks: &[Block], page_width: f32, page_height: f32) -> bool {
    if blocks.is_empty() {
        return false;
    }
    let naive_order: Vec<String> = blocks.iter().map(|b| b.text()).collect();
    let scratch = Page {
        index: 0,
        width: page_width,
        height: page_height,
        blocks: blocks.to_vec(),
        ..Default::default()
    };
    let assembled = assemble_reading_order(&Document {
        pages: vec![scratch],
    });
    let assembled_order: Vec<String> = assembled.pages[0].blocks.iter().map(|b| b.text()).collect();
    naive_order != assembled_order
}

/// Arithmetic mean of an iterator of `f32`s, or `0.0` if it's empty.
fn mean(values: impl Iterator<Item = f32> + Clone) -> f32 {
    let count = values.clone().count();
    if count == 0 {
        return 0.0;
    }
    values.sum::<f32>() / count as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BBox;

    fn region(class: RegionClass, bbox: BBox, confidence: f32) -> Region {
        Region {
            class,
            bbox,
            confidence,
        }
    }

    fn bbox(left: f32, bottom: f32, right: f32, top: f32) -> BBox {
        BBox {
            left,
            bottom,
            right,
            top,
        }
    }

    #[test]
    fn empty_dataset_yields_zeroed_report() {
        let report = evaluate_pages(&[], 0.5);
        assert_eq!(report.page_count, 0);
        assert_eq!(report.macro_region_f1, 0.0);
        assert_eq!(report.mean_giou, 0.0);
        assert!(report.region_f1.is_empty());
        assert!(report.support.is_empty());
        assert_eq!(report.teds, None);
    }

    #[test]
    fn perfect_match_across_pages_scores_one() {
        let b = bbox(0.0, 0.0, 10.0, 10.0);
        let page = (
            vec![region(RegionClass::Text, b, 1.0)],
            vec![region(RegionClass::Text, b, 1.0)],
        );
        let report = evaluate_pages(&[page.clone(), page], 0.5);

        assert_eq!(report.page_count, 2);
        assert_eq!(report.region_f1[&RegionClass::Text], 1.0);
        assert_eq!(report.macro_region_f1, 1.0);
        assert_eq!(report.mean_giou_per_class[&RegionClass::Text], 1.0);
        assert_eq!(report.mean_giou, 1.0);
        assert_eq!(report.support[&RegionClass::Text], 2);
    }

    #[test]
    fn unproduced_class_stays_visible_with_zero_f1() {
        // Ground truth has a Table region; predictions never include one -- this is
        // exactly what happens when scoring `classify_regions` output, which never
        // emits `RegionClass::Table`.
        let gt_bbox = bbox(0.0, 0.0, 10.0, 10.0);
        let page = (Vec::new(), vec![region(RegionClass::Table, gt_bbox, 1.0)]);
        let report = evaluate_pages(&[page], 0.5);

        assert_eq!(report.region_f1[&RegionClass::Table], 0.0);
        assert_eq!(report.support[&RegionClass::Table], 1);
        // No true-positive pairs for Table, so it's excluded from the GIoU macro.
        assert!(!report.mean_giou_per_class.contains_key(&RegionClass::Table));
    }

    #[test]
    fn isolated_class_f1_fields_track_their_class() {
        let b = bbox(0.0, 0.0, 10.0, 10.0);
        let page = (
            vec![region(RegionClass::Footnote, b, 1.0)],
            vec![region(RegionClass::Footnote, b, 1.0)],
        );
        let report = evaluate_pages(&[page], 0.5);

        assert_eq!(report.footnote_f1, 1.0);
        assert_eq!(report.page_header_f1, 0.0);
        assert_eq!(report.page_footer_f1, 0.0);
    }

    fn text_block(text: &str, bbox: BBox) -> crate::Block {
        let word = crate::Word {
            text: text.into(),
            bbox,
            chars: vec![],
        };
        let line = crate::Line {
            text: text.into(),
            bbox,
            words: vec![word],
        };
        crate::Block {
            bbox,
            lines: vec![line],
        }
    }

    #[test]
    fn rank_pages_by_reorder_ranks_multi_column_page_above_single_column() {
        // Two columns, authored interleaved -- gutter is 40pt on a 600pt page, well
        // above assemble::MIN_CUT_FRACTION, so XY-cut recovers column-major order and
        // this page reorders heavily.
        let multi_column = crate::Page {
            index: 0,
            width: 600.0,
            height: 800.0,
            blocks: vec![
                text_block("Right one", bbox(320.0, 700.0, 560.0, 750.0)),
                text_block("Left one", bbox(40.0, 700.0, 280.0, 750.0)),
                text_block("Right two", bbox(320.0, 640.0, 560.0, 690.0)),
                text_block("Left two", bbox(40.0, 640.0, 280.0, 690.0)),
            ],
            ..Default::default()
        };
        // Already top-to-bottom, single column -- XY-cut leaves it untouched.
        let single_column = crate::Page {
            index: 1,
            width: 600.0,
            height: 800.0,
            blocks: vec![
                text_block("First", bbox(40.0, 700.0, 560.0, 750.0)),
                text_block("Second", bbox(40.0, 600.0, 560.0, 650.0)),
            ],
            ..Default::default()
        };

        let document = crate::Document {
            pages: vec![multi_column, single_column],
        };
        let ranked = rank_pages_by_reorder(&document);

        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].page_index, 0);
        assert_eq!(ranked[0].block_count, 4);
        assert!(ranked[0].reorder_edit_distance > 0);
        assert_eq!(ranked[1].page_index, 1);
        assert_eq!(ranked[1].block_count, 2);
        assert_eq!(ranked[1].reorder_edit_distance, 0);
    }

    #[test]
    fn rank_pages_by_reorder_breaks_ties_by_ascending_page_index() {
        // Two identical single-column pages: both have distance 0, so the ordering
        // must fall back to page_index.
        let page = |index: usize| crate::Page {
            index,
            width: 600.0,
            height: 800.0,
            blocks: vec![text_block("Only block", bbox(40.0, 700.0, 560.0, 750.0))],
            ..Default::default()
        };

        let document = crate::Document {
            pages: vec![page(1), page(0)],
        };
        let ranked = rank_pages_by_reorder(&document);

        assert_eq!(ranked[0].page_index, 0);
        assert_eq!(ranked[1].page_index, 1);
    }

    #[test]
    fn minimize_reorder_repro_shrinks_noisy_page_to_the_reordering_blocks() {
        // A reordering two-column pair plus several single-column "noise" blocks that
        // don't contribute to the reordering at all -- the greedy shrink should drop
        // (at least some of) the noise while keeping the page reordering.
        let page = crate::Page {
            index: 0,
            width: 600.0,
            height: 800.0,
            blocks: vec![
                text_block("Right one", bbox(320.0, 700.0, 560.0, 750.0)),
                text_block("Left one", bbox(40.0, 700.0, 280.0, 750.0)),
                text_block("Right two", bbox(320.0, 640.0, 560.0, 690.0)),
                text_block("Left two", bbox(40.0, 640.0, 280.0, 690.0)),
                text_block("Noise A", bbox(40.0, 500.0, 560.0, 550.0)),
                text_block("Noise B", bbox(40.0, 400.0, 560.0, 450.0)),
                text_block("Noise C", bbox(40.0, 300.0, 560.0, 350.0)),
            ],
            ..Default::default()
        };

        let minimized = minimize_reorder_repro(&page).expect("page should reorder");

        assert_eq!(minimized.original_block_count, 7);
        assert!(minimized.page.blocks.len() < minimized.original_block_count);
        assert!(minimized.reorder_edit_distance > 0);
        assert_eq!(minimized.page.width, page.width);
        assert_eq!(minimized.page.height, page.height);
        assert!(reorders(&minimized.page.blocks, page.width, page.height));
    }

    #[test]
    fn minimize_reorder_repro_returns_none_for_already_ordered_page() {
        let page = crate::Page {
            index: 0,
            width: 600.0,
            height: 800.0,
            blocks: vec![
                text_block("First", bbox(40.0, 700.0, 560.0, 750.0)),
                text_block("Second", bbox(40.0, 600.0, 560.0, 650.0)),
            ],
            ..Default::default()
        };

        assert_eq!(minimize_reorder_repro(&page), None);
    }

    #[test]
    fn minimize_reorder_repro_returns_none_when_dedup_removes_the_signal() {
        // Both "columns" share identical text, so deduplicating by block text collapses
        // the page to a single block before any reordering can be observed.
        let page = crate::Page {
            index: 0,
            width: 600.0,
            height: 800.0,
            blocks: vec![
                text_block("Same", bbox(320.0, 700.0, 560.0, 750.0)),
                text_block("Same", bbox(40.0, 700.0, 280.0, 750.0)),
            ],
            ..Default::default()
        };

        assert_eq!(minimize_reorder_repro(&page), None);
    }
}

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

#[cfg(feature = "doclaynet")]
pub mod doclaynet;

#[cfg(feature = "stage3")]
pub mod corpus;

use crate::layout::{Region, RegionClass};
use crate::metrics;
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
}

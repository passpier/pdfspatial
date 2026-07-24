//! Validation metrics for every stage of the pipeline.
//!
//! [`char_recall`] and [`line_grouping_accuracy`] are implemented and back the Stage 1
//! unit tests in this crate, matching the roadmap's Stage 1 metric table exactly
//! (character extraction recall ≥ 99%, line-grouping accuracy ≥ 95% on single-column
//! documents). Everything else here — TEDS, TEDS(IOU), and GIoU/region-F1 — is a Stage
//! 2/4 stub: the function shapes are fixed so downstream code can be written against a
//! stable API, but the bodies are not yet implemented.

use crate::BBox;
use std::collections::HashMap;

/// Character extraction recall: the fraction of ground-truth characters recovered by
/// extraction, as a multiset intersection over non-whitespace characters.
///
/// This matches the roadmap's Stage 1 definition — "% of ground-truth characters
/// recovered vs. dropped (ligatures, embedded fonts, CID-keyed text)" — with a target of
/// ≥ 99% on single-column, non-tabular documents.
///
/// Returns `1.0` if `ground_truth` contains no non-whitespace characters.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::metrics::char_recall;
///
/// let recall = char_recall("Helo world", "Hello world");
/// assert!((0.0..1.0).contains(&recall));
/// assert_eq!(char_recall("Hello world", "Hello world"), 1.0);
/// ```
pub fn char_recall(extracted: &str, ground_truth: &str) -> f32 {
    let gt_counts = char_histogram(ground_truth);
    let total: usize = gt_counts.values().sum();

    if total == 0 {
        return 1.0;
    }

    let extracted_counts = char_histogram(extracted);
    let recovered: usize = gt_counts
        .iter()
        .map(|(ch, gt_count)| (*gt_count).min(*extracted_counts.get(ch).unwrap_or(&0)))
        .sum();

    recovered as f32 / total as f32
}

fn char_histogram(text: &str) -> HashMap<char, usize> {
    let mut counts = HashMap::new();
    for ch in text.chars().filter(|c| !c.is_whitespace()) {
        *counts.entry(ch).or_insert(0) += 1;
    }
    counts
}

/// Line-grouping accuracy: the fraction of ground-truth lines that Stage 1's baseline
/// clustering reproduced exactly (after trimming surrounding whitespace).
///
/// Matches the roadmap's Stage 1 definition — "% of lines correctly segmented by
/// baseline clustering vs. manual annotation" — with a target of ≥ 95% on single-column
/// documents.
///
/// Returns `1.0` if `ground_truth_lines` is empty.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::metrics::line_grouping_accuracy;
///
/// let extracted = vec!["Hello world".to_string(), "second line".to_string()];
/// let ground_truth = vec!["Hello world".to_string(), "second line".to_string()];
/// assert_eq!(line_grouping_accuracy(&extracted, &ground_truth), 1.0);
/// ```
pub fn line_grouping_accuracy(extracted_lines: &[String], ground_truth_lines: &[String]) -> f32 {
    if ground_truth_lines.is_empty() {
        return 1.0;
    }

    let extracted_trimmed: Vec<&str> = extracted_lines.iter().map(|l| l.trim()).collect();
    let matched = ground_truth_lines
        .iter()
        .filter(|gt| extracted_trimmed.contains(&gt.trim()))
        .count();

    matched as f32 / ground_truth_lines.len() as f32
}

/// Tree-Edit-Distance-based Similarity, structure only (row/column/span topology,
/// ignoring cell text), as defined by PubTabNet.
///
/// # Panics
///
/// Always panics — this is a Stage 2 stub, not yet implemented.
pub fn teds_struct(_predicted_html: &str, _ground_truth_html: &str) -> f64 {
    unimplemented!("TEDS-Struct is not yet implemented")
}

/// Tree-Edit-Distance-based Similarity, structure and content combined.
///
/// # Panics
///
/// Always panics — this is a Stage 2 stub, not yet implemented.
pub fn teds(_predicted_html: &str, _ground_truth_html: &str) -> f64 {
    unimplemented!("TEDS is not yet implemented")
}

/// TEDS(IOU): a text-independent TEDS variant that scores cell content by bounding-box
/// IoU instead of string edit distance, appropriate for an OCR-free, bbox-driven
/// pipeline like this one.
///
/// # Panics
///
/// Always panics — this is a Stage 2 stub, not yet implemented.
pub fn teds_iou(_predicted_html: &str, _ground_truth_html: &str) -> f64 {
    unimplemented!("TEDS(IOU) is not yet implemented")
}

/// Generalized IoU between a predicted and ground-truth bounding box.
///
/// Unlike plain IoU, GIoU stays informative for non-overlapping boxes via a penalty term
/// based on the smallest enclosing box — important for early-epoch layout models that
/// may predict boxes with no overlap at all.
///
/// # Panics
///
/// Always panics — this is a Stage 2 stub, not yet implemented.
pub fn giou(_predicted: BBox, _ground_truth: BBox) -> f32 {
    unimplemented!("GIoU is not yet implemented")
}

/// Region-classification F1 at a given IoU/GIoU matching threshold (COCO-style),
/// broken out per [`crate::layout::RegionClass`].
///
/// # Panics
///
/// Always panics — this is a Stage 2 stub, not yet implemented.
pub fn region_f1(
    _predicted: &[crate::layout::Region],
    _ground_truth: &[crate::layout::Region],
    _iou_threshold: f32,
) -> HashMap<crate::layout::RegionClass, f32> {
    unimplemented!("Region F1 is not yet implemented")
}

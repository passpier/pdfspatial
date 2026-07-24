//! Stage 2/4: layout region classification.
//!
//! **Status: not implemented.** This module defines the shape Stage 2 validation and
//! Stage 4 refinement will fill in — a region classifier trained/evaluated against
//! [DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1)'s 11-class
//! schema — so the rest of the crate can be written against a stable API. Every function
//! here panics via `unimplemented!()`.

use crate::{BBox, Document};

/// DocLayNet's 11 region categories.
///
/// Mirrors the class schema used by DocLayNet-v1.1 exactly, since Stage 2 validation
/// scores predictions against DocLayNet ground truth. See the roadmap's Stage 2 section
/// for per-class GIoU/F1 targets, and note that `Footnote`, `Page-header`, and
/// `Page-footer` are called out there as the historically weakest classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionClass {
    /// A caption associated with a `Picture` or `Table` region.
    Caption,
    /// Footnote text, distinct from body `Text`.
    Footnote,
    /// A mathematical formula, inline or block.
    Formula,
    /// A single item within a list.
    ListItem,
    /// A running page footer.
    PageFooter,
    /// A running page header.
    PageHeader,
    /// A figure, photo, chart, or other non-text graphic.
    Picture,
    /// A section heading.
    SectionHeader,
    /// A table.
    Table,
    /// Ordinary body text.
    Text,
    /// A document or chapter title.
    Title,
}

/// A classified layout region: a geometric area with a predicted [`RegionClass`] and
/// confidence score.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// The predicted category.
    pub class: RegionClass,
    /// The region's bounding box in page space.
    pub bbox: BBox,
    /// Model confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// Classifies every geometric [`crate::Block`] in `document` into a [`Region`].
///
/// # Stage 2/4 design intent
///
/// The eventual implementation is expected to follow Docling's ONNX layout stage: resize
/// each page raster to 640×640, run an RT-DETR-style detector (sigmoid + top-k
/// class×query matching), then decode predicted boxes back to page scale. See
/// [`crate::extract::render_pages_parallel`] for the page-rendering path this will
/// consume.
///
/// # Panics
///
/// Always panics — this is a Stage 2/4 stub, not yet implemented.
pub fn classify_regions(_document: &Document) -> Vec<Region> {
    unimplemented!("Stage 2 layout classification is not yet implemented")
}

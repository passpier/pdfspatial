//! Stage 3/4: reading-order assembly.
//!
//! **Status: not implemented.** Stage 1's [`crate::extract`] module produces blocks in
//! naive top-to-bottom, left-to-right order, which is known to be wrong for multi-column
//! layouts and several other structural patterns. This module is where a real
//! reading-order solver (e.g. column-aware XY-cut recursive segmentation, per the
//! roadmap's Stage 4a) will eventually live, informed by the failure taxonomy Stage 3
//! error analysis produces.
//!
//! The checklist below mirrors the roadmap's "Document-Structure Pitfall Checklist"
//! verbatim. Each item is a category of reading-order/geometric/classification failure
//! that Stage 3 collects a labeled regression corpus for (≥ 20 samples each, tagged
//! `geometric`, `classification`, or `ordering`), and that Stage 4 fixes — heuristically
//! first, with targeted fine-tuning only where heuristics plateau.
//!
//! - [ ] **Multi-column layout / gutter detection** — text lines merged across column
//!   boundaries, or column order inverted (right column read before left).
//! - [ ] **Footnotes** — footnote text merged into body `Text` instead of classified as
//!   `Footnote`; footnote markers (superscript numerals) not linked to their note.
//! - [ ] **Page headers/footers repeated across pages** — header/footer text leaking into
//!   the body text stream; running headers misclassified as `Section-header`.
//! - [ ] **Nested/multi-line table cells** — cells spanning multiple visual lines merged
//!   incorrectly with adjacent rows; rowspan/colspan not detected.
//! - [ ] **Merged table cells (rowspan/colspan)** — TEDS-Struct penalizes tree-topology
//!   mismatches here most heavily.
//! - [ ] **Borderless / whitespace-delimited tables** — no ruling lines for the layout
//!   model to key off; frequently misclassified as plain `Text`.
//! - [ ] **Nested mathematical formulas** — inline formulas embedded mid-sentence not
//!   separated from surrounding text; multi-line/stacked formulas (fractions,
//!   summations, matrices) fragmented into multiple bounding boxes.
//! - [ ] **Superscript/subscript handling** — footnote markers, exponents, and
//!   chemical/math subscripts causing baseline-clustering errors in Stage 1's line
//!   grouping.
//! - [ ] **Rotated or vertical text** — sidebar labels, rotated table headers, CJK
//!   vertical text.
//! - [ ] **Figure/caption association** — captions not correctly linked to their parent
//!   `Picture`/`Table` region, or associated with the wrong figure when multiple figures
//!   share a page.
//! - [ ] **List-item nesting** — multi-level bullet/numbered lists collapsed into flat
//!   `List-item` blocks, losing hierarchy.
//! - [ ] **Cross-page table/paragraph continuation** — tables or paragraphs split across
//!   a page boundary not stitched back together.
//! - [ ] **Section headers vs. bold body text** — `Section-header` misclassified as
//!   `Title` or emphasized `Text` when font-weight is the only visual cue.
//! - [ ] **Embedded/CID-keyed fonts and ligatures** — pdfium character extraction
//!   dropping or garbling glyphs from non-standard font encodings.
//! - [ ] **Overlapping/z-ordered text objects** — watermarks, stamps, or redaction boxes
//!   overlapping real text and corrupting bbox clustering.

use crate::Document;
use crate::layout::Region;

/// Root cause tag attached to each Stage 3 failure-mode regression case.
///
/// Determines whether the Stage 4 fix is heuristic (`Geometric`, `Ordering`) or requires
/// model retraining (`Classification`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RootCause {
    /// The underlying pdfium bounding box is wrong.
    Geometric,
    /// The region layout model assigned the wrong class.
    Classification,
    /// Reading-order assembly sequenced correctly-classified regions incorrectly.
    Ordering,
}

/// Reassembles classified [`Region`]s into reading-order [`Document`] structure.
///
/// # Stage 3/4 design intent
///
/// The naive top-to-bottom scan used by [`crate::extract`] is expected to be replaced
/// here with column-aware XY-cut recursive segmentation (roadmap Stage 4a), validated
/// against the pitfall checklist above before each full Stage 2 metrics re-run.
///
/// # Panics
///
/// Always panics — this is a Stage 3/4 stub, not yet implemented.
pub fn assemble_reading_order(_regions: &[Region]) -> Document {
    unimplemented!("Stage 3/4 reading-order assembly is not yet implemented")
}

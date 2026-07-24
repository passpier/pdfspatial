//! Stage 3/4: reading-order assembly.
//!
//! [`assemble_reading_order`] implements the roadmap's Stage 4a reading-order fix:
//! column-aware XY-cut recursive segmentation, replacing Stage 1's naive top-to-bottom,
//! left-to-right block order (which is known to be wrong for multi-column layouts and
//! several other structural patterns). It reorders each page's blocks in place; nothing
//! about block/line/word/char content changes. Further refinement, informed by the
//! failure taxonomy Stage 3 error analysis produces, remains future work.
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

use crate::{BBox, Block, Document, Page};

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

/// A category from the [module-level pitfall checklist](self), one variant per checklist
/// item, in the checklist's listed order.
///
/// This is the taxonomy the Stage 3 regression corpus (`fixtures/`, loaded by
/// `eval::corpus` behind the `stage3` cargo feature) organizes its cases around: every
/// regression case is tagged with exactly one `Pitfall` plus a [`RootCause`].
///
/// # Examples
///
/// ```
/// use pdfspatial_core::assemble::Pitfall;
///
/// let pitfall = Pitfall::MultiColumn;
/// assert_eq!(pitfall, Pitfall::MultiColumn);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Pitfall {
    /// Text lines merged across column boundaries, or column order inverted (right
    /// column read before left).
    MultiColumn,
    /// Footnote text merged into body `Text` instead of classified as `Footnote`;
    /// footnote markers (superscript numerals) not linked to their note.
    Footnote,
    /// Header/footer text leaking into the body text stream; running headers
    /// misclassified as `Section-header`.
    HeaderFooter,
    /// Table cells spanning multiple visual lines merged incorrectly with adjacent
    /// rows; rowspan/colspan not detected.
    MultiLineTableCell,
    /// Merged table cells (rowspan/colspan) whose tree-topology mismatch TEDS-Struct
    /// penalizes most heavily.
    MergedTableCell,
    /// Borderless / whitespace-delimited tables with no ruling lines for the layout
    /// model to key off; frequently misclassified as plain `Text`.
    BorderlessTable,
    /// Inline formulas embedded mid-sentence not separated from surrounding text;
    /// multi-line/stacked formulas fragmented into multiple bounding boxes.
    NestedFormula,
    /// Footnote markers, exponents, and chemical/math subscripts causing
    /// baseline-clustering errors in Stage 1's line grouping.
    SuperSubscript,
    /// Sidebar labels, rotated table headers, CJK vertical text.
    RotatedText,
    /// Captions not correctly linked to their parent `Picture`/`Table` region, or
    /// associated with the wrong figure when multiple figures share a page.
    FigureCaption,
    /// Multi-level bullet/numbered lists collapsed into flat `List-item` blocks,
    /// losing hierarchy.
    ListNesting,
    /// Tables or paragraphs split across a page boundary not stitched back together.
    CrossPageContinuation,
    /// `Section-header` misclassified as `Title` or emphasized `Text` when
    /// font-weight is the only visual cue.
    SectionHeaderVsBold,
    /// Embedded/CID-keyed fonts and ligatures causing pdfium character extraction to
    /// drop or garble glyphs from non-standard font encodings.
    EmbeddedFont,
    /// Watermarks, stamps, or redaction boxes overlapping real text and corrupting
    /// bbox clustering.
    OverlappingText,
}

/// A vertical (column-gutter) or horizontal (row) whitespace gap smaller than this,
/// expressed as a multiple of the page width/height, is not considered a qualifying
/// XY-cut -- it's normal inter-paragraph spacing, not a structural column/row boundary.
const MIN_CUT_FRACTION: f32 = 0.03;

/// Reassembles a [`Document`]'s blocks into reading order via recursive, column-aware
/// XY-cut segmentation, replacing Stage 1's naive top-to-bottom scan.
///
/// # Why this takes a [`Document`], not `&[Region]`
///
/// The roadmap's original stub signature was `fn(&[Region]) -> Document`, but
/// [`crate::layout::Region`] carries neither text nor a page index -- there is no way to
/// reconstruct page boundaries or block content from a bare region list. XY-cut is
/// inherently a per-page operation over blocks that already own their text and geometry,
/// so this function operates directly on [`Document`]/[`Page`]/[`Block`] and returns a
/// new [`Document`] with each page's `blocks` reordered (block/line/word/char content is
/// unchanged).
///
/// # Algorithm
///
/// Within each page, blocks are recursively partitioned: at each level, find the widest
/// vertical whitespace gutter (an x-range with no block spanning it) and the widest
/// horizontal gap (a y-range with no block spanning it) among the blocks in scope. If a
/// vertical gutter at least [`MIN_CUT_FRACTION`] of the page width is found, split into
/// left/right groups and recurse on each, left before right (read a full column
/// top-to-bottom before moving to the next). Otherwise, if a horizontal gap at least
/// [`MIN_CUT_FRACTION`] of the page height is found, split into top/bottom groups and
/// recurse, top before bottom. When neither cut qualifies (or only one block remains),
/// order the blocks by descending top-y, then ascending left-x.
pub fn assemble_reading_order(document: &Document) -> Document {
    let pages = document
        .pages
        .iter()
        .map(|page| {
            let mut blocks = page.blocks.clone();
            xy_cut_order(&mut blocks, page.width, page.height);
            Page {
                index: page.index,
                width: page.width,
                height: page.height,
                blocks,
            }
        })
        .collect();

    Document { pages }
}

/// Recursively reorders `blocks` in place into XY-cut reading order.
fn xy_cut_order(blocks: &mut Vec<Block>, page_width: f32, page_height: f32) {
    if blocks.len() <= 1 {
        return;
    }

    if let Some(cut_x) = widest_vertical_gutter(blocks, page_width) {
        let (mut left, mut right): (Vec<Block>, Vec<Block>) =
            blocks.drain(..).partition(|b| center_x(b.bbox) < cut_x);
        xy_cut_order(&mut left, page_width, page_height);
        xy_cut_order(&mut right, page_width, page_height);
        left.extend(right);
        *blocks = left;
        return;
    }

    if let Some(cut_y) = widest_horizontal_gap(blocks, page_height) {
        let (mut top, mut bottom): (Vec<Block>, Vec<Block>) =
            blocks.drain(..).partition(|b| center_y(b.bbox) >= cut_y);
        xy_cut_order(&mut top, page_width, page_height);
        xy_cut_order(&mut bottom, page_width, page_height);
        top.extend(bottom);
        *blocks = top;
        return;
    }

    blocks.sort_by(|a, b| {
        b.bbox
            .top
            .partial_cmp(&a.bbox.top)
            .unwrap()
            .then(a.bbox.left.partial_cmp(&b.bbox.left).unwrap())
    });
}

/// Finds the x-coordinate of the widest vertical whitespace gutter that separates
/// `blocks` into a non-empty left group and non-empty right group, provided that gutter
/// is at least [`MIN_CUT_FRACTION`] of `page_width` wide. A "gutter" here is a gap
/// between one block's right edge and the next block's left edge when blocks are sorted
/// left to right; it does not require that every block avoid the gap vertically, since a
/// true multi-row column gutter often has some blocks spanning only part of the page's
/// height on each side.
fn widest_vertical_gutter(blocks: &[Block], page_width: f32) -> Option<f32> {
    let mut edges: Vec<(f32, f32)> = blocks.iter().map(|b| (b.bbox.left, b.bbox.right)).collect();
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Merge overlapping horizontal spans, then look at the gaps between the merged
    // spans -- those gaps are candidate column gutters.
    let mut merged: Vec<(f32, f32)> = Vec::new();
    for (left, right) in edges {
        match merged.last_mut() {
            Some((_, prev_right)) if left <= *prev_right => {
                *prev_right = prev_right.max(right);
            }
            _ => merged.push((left, right)),
        }
    }

    if merged.len() < 2 {
        return None;
    }

    let min_gap = page_width * MIN_CUT_FRACTION;
    merged
        .windows(2)
        .map(|w| (w[0].1, w[1].0 - w[0].1))
        .filter(|&(_, gap)| gap >= min_gap)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(right_edge, gap)| right_edge + gap / 2.0)
}

/// Finds the y-coordinate of the widest horizontal gap that separates `blocks` into a
/// non-empty top group and non-empty bottom group, provided that gap is at least
/// [`MIN_CUT_FRACTION`] of `page_height` tall. Mirrors [`widest_vertical_gutter`] on the
/// vertical axis.
fn widest_horizontal_gap(blocks: &[Block], page_height: f32) -> Option<f32> {
    let mut edges: Vec<(f32, f32)> = blocks.iter().map(|b| (b.bbox.bottom, b.bbox.top)).collect();
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut merged: Vec<(f32, f32)> = Vec::new();
    for (bottom, top) in edges {
        match merged.last_mut() {
            Some((_, prev_top)) if bottom <= *prev_top => {
                *prev_top = prev_top.max(top);
            }
            _ => merged.push((bottom, top)),
        }
    }

    if merged.len() < 2 {
        return None;
    }

    let min_gap = page_height * MIN_CUT_FRACTION;
    merged
        .windows(2)
        .map(|w| (w[0].1, w[1].0 - w[0].1))
        .filter(|&(_, gap)| gap >= min_gap)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(top_edge, gap)| top_edge + gap / 2.0)
}

fn center_x(bbox: BBox) -> f32 {
    (bbox.left + bbox.right) / 2.0
}

fn center_y(bbox: BBox) -> f32 {
    (bbox.top + bbox.bottom) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Line;

    fn block(bbox: BBox) -> Block {
        Block {
            bbox,
            lines: vec![Line {
                text: "line".into(),
                bbox,
                words: vec![],
            }],
        }
    }

    #[test]
    fn two_column_page_reads_left_column_before_right() {
        // Two columns of two stacked blocks each, on a 600pt-wide page. Column gutter
        // runs from x=280 to x=320 (40pt, well above the 3% = 18pt threshold).
        let left_top = block(BBox {
            left: 20.0,
            bottom: 600.0,
            right: 280.0,
            top: 700.0,
        });
        let left_bottom = block(BBox {
            left: 20.0,
            bottom: 400.0,
            right: 280.0,
            top: 590.0,
        });
        let right_top = block(BBox {
            left: 320.0,
            bottom: 600.0,
            right: 580.0,
            top: 700.0,
        });
        let right_bottom = block(BBox {
            left: 320.0,
            bottom: 400.0,
            right: 580.0,
            top: 590.0,
        });

        // Deliberately out of reading order in the input.
        let page = Page {
            index: 0,
            width: 600.0,
            height: 800.0,
            blocks: vec![
                right_top.clone(),
                left_bottom.clone(),
                right_bottom.clone(),
                left_top.clone(),
            ],
        };
        let doc = Document { pages: vec![page] };

        let ordered = assemble_reading_order(&doc);
        let blocks = &ordered.pages[0].blocks;

        assert_eq!(blocks[0].bbox, left_top.bbox);
        assert_eq!(blocks[1].bbox, left_bottom.bbox);
        assert_eq!(blocks[2].bbox, right_top.bbox);
        assert_eq!(blocks[3].bbox, right_bottom.bbox);
    }

    #[test]
    fn single_column_page_preserves_top_to_bottom_order() {
        let first = block(BBox {
            left: 20.0,
            bottom: 700.0,
            right: 400.0,
            top: 750.0,
        });
        let second = block(BBox {
            left: 20.0,
            bottom: 600.0,
            right: 400.0,
            top: 650.0,
        });
        let third = block(BBox {
            left: 20.0,
            bottom: 500.0,
            right: 400.0,
            top: 550.0,
        });

        let page = Page {
            index: 0,
            width: 600.0,
            height: 800.0,
            blocks: vec![second.clone(), third.clone(), first.clone()],
        };
        let doc = Document { pages: vec![page] };

        let ordered = assemble_reading_order(&doc);
        let blocks = &ordered.pages[0].blocks;

        assert_eq!(blocks[0].bbox, first.bbox);
        assert_eq!(blocks[1].bbox, second.bbox);
        assert_eq!(blocks[2].bbox, third.bbox);
    }
}

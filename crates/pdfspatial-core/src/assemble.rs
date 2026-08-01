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

/// A horizontal (row) whitespace gap smaller than this, expressed as a multiple of the
/// page height, is not considered a qualifying XY-cut -- it's normal inter-paragraph
/// spacing, not a structural row boundary. Also serves as the *upper bound* on the
/// vertical gutter threshold (see [`MIN_GUTTER_EMS`]): a page-relative fraction is too
/// coarse to recognize a narrow column gutter in small type, but it's still a sensible
/// ceiling above which any gap is obviously a gutter regardless of font size.
const MIN_CUT_FRACTION: f32 = 0.03;

/// Minimum vertical (column-gutter) whitespace gap, expressed as a multiple of the
/// page's own median character size ("em"), for it to qualify as a structural column
/// boundary rather than ordinary inter-word spacing (~0.25 em).
///
/// A fixed page-width fraction (as [`MIN_CUT_FRACTION`] alone would give) is blind to
/// the page's text scale: an 18pt gutter (3% of a 600pt-wide page) rejects a
/// perfectly legible 1-em column gutter in 10pt body text. Scaling the threshold to
/// the page's own font size instead recognizes gutters at whatever size the page's
/// text actually is, while [`MIN_CUT_FRACTION`] still caps how loose that can get on a
/// very small-font page.
const MIN_GUTTER_EMS: f32 = 0.8;

/// Absolute floor for the em-relative vertical gutter threshold, in points, so a
/// pathologically tiny median font size can't shrink the minimum gutter to near zero.
const MIN_GUTTER_ABS_PT: f32 = 2.0;

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
/// horizontal gap (a y-range with no block spanning it) among the blocks in scope. A
/// vertical gutter qualifies as a cut when it is both at least [`MIN_GUTTER_EMS`] times
/// the page's own median character size (clamped to [`MIN_GUTTER_ABS_PT`] and
/// [`MIN_CUT_FRACTION`] of the page width -- see [`median_font_size`]) *and* the blocks
/// on either side of it actually coexist vertically (their y-extents overlap), so a
/// block in one corner and an unrelated block in the opposite corner aren't mistaken
/// for two columns. If a vertical gutter qualifies, split into left/right groups and
/// recurse on each, left before right (read a full column top-to-bottom before moving
/// to the next). Otherwise, if a horizontal gap at least [`MIN_CUT_FRACTION`] of the
/// page height is found, split into top/bottom groups and recurse, top before bottom.
/// When neither cut qualifies (or only one block remains), order the blocks by
/// descending top-y, then ascending left-x.
pub fn assemble_reading_order(document: &Document) -> Document {
    let pages = document
        .pages
        .iter()
        .map(|page| {
            let mut blocks = page.blocks.clone();
            let params = CutParams::for_page(page);
            xy_cut_order(&mut blocks, &params);
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

/// Per-page constants threaded through the XY-cut recursion, computed once per page
/// rather than re-derived at every recursion level.
struct CutParams {
    page_height: f32,
    /// Minimum vertical gutter width, in points, for this page (see [`MIN_GUTTER_EMS`]).
    min_gutter: f32,
}

impl CutParams {
    fn for_page(page: &Page) -> Self {
        let upper_bound = page.width * MIN_CUT_FRACTION;
        // Guard against a pathologically narrow page where the absolute floor would
        // exceed the page-relative ceiling -- clamp's bounds must satisfy low <= high.
        let lower_bound = MIN_GUTTER_ABS_PT.min(upper_bound);
        let min_gutter = match median_font_size(page) {
            Some(em) => (MIN_GUTTER_EMS * em).clamp(lower_bound, upper_bound),
            // No character data to measure a text scale from (e.g. hand-built test
            // fixtures with empty lines) -- fall back to the page-relative rule.
            None => upper_bound,
        };
        CutParams {
            page_height: page.height,
            min_gutter,
        }
    }
}

/// Returns the median `font_size` across every [`Char`](crate::Char) on the page, or
/// `None` if the page has no characters to measure. Used as a proxy for the page's
/// typical text scale ("em"), which a gutter must be some multiple of to count as a
/// structural column boundary -- see [`MIN_GUTTER_EMS`].
///
/// Line-box height is deliberately not used as the proxy: a line's bounding box can be
/// much taller than its font size (e.g. loose leading, or a hand-authored test fixture
/// with an oversized box around small type), which would inflate the gutter threshold
/// and defeat the point of scaling it to text size.
fn median_font_size(page: &Page) -> Option<f32> {
    let mut sizes: Vec<f32> = page
        .blocks
        .iter()
        .flat_map(|b| &b.lines)
        .flat_map(|l| &l.words)
        .flat_map(|w| &w.chars)
        .map(|c| c.font_size)
        .collect();

    if sizes.is_empty() {
        return None;
    }

    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(sizes[sizes.len() / 2])
}

/// Recursively reorders `blocks` in place into XY-cut reading order.
fn xy_cut_order(blocks: &mut Vec<Block>, params: &CutParams) {
    if blocks.len() <= 1 {
        return;
    }

    if let Some(cut_x) = widest_vertical_gutter(blocks, params.min_gutter) {
        let (mut left, mut right): (Vec<Block>, Vec<Block>) =
            blocks.drain(..).partition(|b| center_x(b.bbox) < cut_x);
        xy_cut_order(&mut left, params);
        xy_cut_order(&mut right, params);
        left.extend(right);
        *blocks = left;
        return;
    }

    if let Some(cut_y) = widest_horizontal_gap(blocks, params.page_height) {
        let (mut top, mut bottom): (Vec<Block>, Vec<Block>) =
            blocks.drain(..).partition(|b| center_y(b.bbox) >= cut_y);
        xy_cut_order(&mut top, params);
        xy_cut_order(&mut bottom, params);
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
/// is at least `min_gutter` points wide *and* the blocks strictly left of it and
/// strictly right of it have overlapping y-extents -- i.e. the two sides actually
/// coexist vertically, which is what makes a gap a column gutter rather than two
/// unrelated blocks in opposite corners of the page. A "gutter" here is a gap between
/// one block's right edge and the next block's left edge when blocks are sorted left to
/// right; it does not require that every individual block avoid the gap vertically,
/// since a true multi-row column gutter often has some blocks spanning only part of the
/// page's height on each side.
fn widest_vertical_gutter(blocks: &[Block], min_gutter: f32) -> Option<f32> {
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

    merged
        .windows(2)
        .map(|w| (w[0].1, w[1].0 - w[0].1))
        .filter(|&(_, gap)| gap >= min_gutter)
        .filter(|&(right_edge, _)| vertical_extents_overlap(blocks, right_edge))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(right_edge, gap)| right_edge + gap / 2.0)
}

/// Returns `true` if the blocks strictly left of `cut_x` and the blocks strictly right
/// of `cut_x` have overlapping y-extents, i.e. content on both sides of the candidate
/// gutter coexists vertically rather than sitting in unrelated corners of the page.
fn vertical_extents_overlap(blocks: &[Block], cut_x: f32) -> bool {
    let mut left_extent: Option<BBox> = None;
    let mut right_extent: Option<BBox> = None;

    for block in blocks {
        let extent = if block.bbox.right <= cut_x {
            &mut left_extent
        } else if block.bbox.left >= cut_x {
            &mut right_extent
        } else {
            // A block straddles the cut -- shouldn't happen for a merged-span gap, but
            // guard against treating a spanning block as evidence either side is empty.
            continue;
        };
        *extent = Some(match extent {
            Some(existing) => existing.union(&block.bbox),
            None => block.bbox,
        });
    }

    match (left_extent, right_extent) {
        (Some(left), Some(right)) => left.vertically_overlaps(&right),
        _ => false,
    }
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
    use crate::{Char, Line, Word};

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

    /// Like [`block`], but with a single character at `font_size` so
    /// [`median_font_size`] has something to measure -- needed to exercise the
    /// em-relative gutter threshold rather than its no-chars fallback.
    fn block_with_font(bbox: BBox, font_size: f32) -> Block {
        let ch = Char {
            unicode: Some('x'),
            bbox,
            font_name: "Test".into(),
            font_size,
        };
        let word = Word {
            text: "line".into(),
            bbox,
            chars: vec![ch],
        };
        Block {
            bbox,
            lines: vec![Line {
                text: "line".into(),
                bbox,
                words: vec![word],
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

    #[test]
    fn narrow_em_relative_gutter_still_reads_columns_in_order() {
        // A 10pt gutter at 10pt font (1 em) -- well below the old fixed 18pt
        // threshold (3% of 600pt), but comfortably above 0.8 em = 8pt.
        let left = block_with_font(
            BBox {
                left: 40.0,
                bottom: 400.0,
                right: 290.0,
                top: 600.0,
            },
            10.0,
        );
        let right = block_with_font(
            BBox {
                left: 300.0,
                bottom: 420.0,
                right: 550.0,
                top: 620.0,
            },
            10.0,
        );

        let page = Page {
            index: 0,
            width: 600.0,
            height: 800.0,
            // Right column's top edge sits higher, so the naive top-y leaf sort would
            // read it first -- only a qualifying vertical cut reads left-before-right.
            blocks: vec![right.clone(), left.clone()],
        };
        let doc = Document { pages: vec![page] };

        let ordered = assemble_reading_order(&doc);
        let blocks = &ordered.pages[0].blocks;

        assert_eq!(blocks[0].bbox, left.bbox);
        assert_eq!(blocks[1].bbox, right.bbox);
    }

    #[test]
    fn sub_threshold_gutter_falls_through_to_leaf_sort() {
        // A 3pt gutter at 10pt font -- below 0.8 em = 8pt -- must not qualify as a
        // column cut, so blocks fall through to the naive top-y sort.
        let left = block_with_font(
            BBox {
                left: 40.0,
                bottom: 400.0,
                right: 290.0,
                top: 600.0,
            },
            10.0,
        );
        let right = block_with_font(
            BBox {
                left: 293.0,
                bottom: 420.0,
                right: 550.0,
                top: 620.0,
            },
            10.0,
        );

        let page = Page {
            index: 0,
            width: 600.0,
            height: 800.0,
            blocks: vec![left.clone(), right.clone()],
        };
        let doc = Document { pages: vec![page] };

        let ordered = assemble_reading_order(&doc);
        let blocks = &ordered.pages[0].blocks;

        // Right's top edge (620) is higher than left's (600), so the leaf sort
        // (descending top-y) reads it first -- confirming no vertical cut fired.
        assert_eq!(blocks[0].bbox, right.bbox);
        assert_eq!(blocks[1].bbox, left.bbox);
    }

    #[test]
    fn no_char_data_falls_back_to_page_fraction_threshold() {
        // A 10pt gutter with no character data at all (the plain `block` helper) --
        // below the fixed 3% = 18pt fallback threshold, so it must not cut, exactly
        // as it didn't before this change.
        let left = block(BBox {
            left: 40.0,
            bottom: 400.0,
            right: 290.0,
            top: 600.0,
        });
        let right = block(BBox {
            left: 300.0,
            bottom: 420.0,
            right: 550.0,
            top: 620.0,
        });

        let page = Page {
            index: 0,
            width: 600.0,
            height: 800.0,
            blocks: vec![left.clone(), right.clone()],
        };
        let doc = Document { pages: vec![page] };

        let ordered = assemble_reading_order(&doc);
        let blocks = &ordered.pages[0].blocks;

        assert_eq!(blocks[0].bbox, right.bbox);
        assert_eq!(blocks[1].bbox, left.bbox);
    }

    #[test]
    fn non_coexisting_corners_use_horizontal_cut_not_vertical() {
        // A top-left block and a bottom-right block share a wide x-gap but never
        // coexist vertically -- this must read as top-then-bottom (horizontal cut),
        // not be split into left/right columns.
        let top_left = block_with_font(
            BBox {
                left: 40.0,
                bottom: 700.0,
                right: 200.0,
                top: 750.0,
            },
            10.0,
        );
        let bottom_right = block_with_font(
            BBox {
                left: 400.0,
                bottom: 100.0,
                right: 560.0,
                top: 150.0,
            },
            10.0,
        );

        let page = Page {
            index: 0,
            width: 600.0,
            height: 800.0,
            blocks: vec![bottom_right.clone(), top_left.clone()],
        };
        let doc = Document { pages: vec![page] };

        let ordered = assemble_reading_order(&doc);
        let blocks = &ordered.pages[0].blocks;

        assert_eq!(blocks[0].bbox, top_left.bbox);
        assert_eq!(blocks[1].bbox, bottom_right.bbox);
    }
}

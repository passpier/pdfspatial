//! Stage 3/4: reading-order assembly.
//!
//! [`assemble_reading_order`] implements the roadmap's Stage 4a reading-order fix:
//! column-aware XY-cut recursive segmentation, replacing Stage 1's naive top-to-bottom,
//! left-to-right block order (which is known to be wrong for multi-column layouts and
//! several other structural patterns), followed by a cross-page paragraph-stitching pass
//! (see [`assemble_reading_order`]'s docs) that merges a sentence split across a page
//! boundary back into one block. Further refinement, informed by the failure taxonomy
//! Stage 3 error analysis produces, remains future work.
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

use crate::{BBox, Block, Document, Line, Page};

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

/// A block's median font size must be at least this multiple of another block's to be a
/// candidate overlay of it (see [`xy_cut_order`]'s fallback ordering). A watermark or stamp
/// is typically drawn much larger than the text it's stamped across; this keeps ordinary
/// same-scale neighbors (a heading a couple of points larger than its body, say) from ever
/// qualifying.
const OVERLAY_FONT_SIZE_RATIO_MIN: f32 = 2.0;

/// Two blocks' bounding-box intersection must cover at least this fraction of the smaller
/// block's area for the larger-font one to count as drawn *across* the other, rather than
/// merely adjacent to or barely clipping it.
const OVERLAY_AREA_OVERLAP_FRACTION: f32 = 0.5;

/// Reassembles a [`Document`]'s blocks into reading order via recursive, column-aware
/// XY-cut segmentation, replacing Stage 1's naive top-to-bottom scan.
///
/// Before the XY-cut, each page's blocks pass through
/// [`crate::extract::merge_rotated_text_runs`], which repairs rotated/vertical text that
/// Stage 1's baseline clustering shatters into one line per glyph. This is the only place
/// besides cross-page stitching (below) where a block's lines/words are rebuilt rather
/// than just reordered; block bounding boxes are unchanged, so it never affects which cut
/// the XY-cut algorithm makes.
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
///
/// # Cross-page continuation stitching
///
/// After every page is independently XY-cut, a second pass walks the ordered pages and
/// merges a page's last block into the next page's first block when they look like one
/// paragraph split by the page boundary -- see [`stitch_cross_page_continuations`] for
/// the exact predicate. This is the only case where content crosses a page boundary or a
/// block's lines/words are edited rather than just reordered.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::assemble::assemble_reading_order;
/// use pdfspatial_core::{BBox, Block, Char, Document, Line, Page, Word};
///
/// fn line(text: &str, bbox: BBox, font_size: f32) -> Line {
///     let chars = vec![Char { unicode: Some('x'), bbox, font_name: "Test".into(), font_size }];
///     let word = Word { text: text.into(), bbox, chars };
///     Line { text: text.into(), bbox, words: vec![word] }
/// }
///
/// let tail_bbox = BBox { left: 72.0, bottom: 697.5, right: 328.8, top: 710.9 };
/// let page0 = Page {
///     index: 0,
///     width: 612.0,
///     height: 792.0,
///     blocks: vec![Block {
///         bbox: tail_bbox,
///         lines: vec![line("This sentence continues on the next", tail_bbox, 12.0)],
///     }],
/// };
///
/// let head_bbox = BBox { left: 72.0, bottom: 697.5, right: 307.5, top: 710.9 };
/// let page1 = Page {
///     index: 1,
///     width: 612.0,
///     height: 792.0,
///     blocks: vec![Block {
///         bbox: head_bbox,
///         lines: vec![line("page without a heading to signal a restart.", head_bbox, 12.0)],
///     }],
/// };
///
/// let document = Document { pages: vec![page0, page1] };
/// let assembled = assemble_reading_order(&document);
///
/// assert_eq!(assembled.pages[0].blocks.len(), 1);
/// assert_eq!(
///     assembled.pages[0].blocks[0].text(),
///     "This sentence continues on the next page without a heading to signal a restart."
/// );
/// assert!(assembled.pages[1].blocks.is_empty());
/// ```
pub fn assemble_reading_order(document: &Document) -> Document {
    let mut pages: Vec<Page> = document
        .pages
        .iter()
        .map(|page| {
            // Repair rotated/vertical text before ordering: Stage 1's center-y line
            // grouping shatters a rotated label into one line per glyph, and neither the
            // XY-cut nor the cross-page stitcher below can put those back together.
            let mut blocks = crate::extract::merge_rotated_text_runs(&page.blocks);
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

    stitch_cross_page_continuations(&mut pages);

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

    // Neither cut qualifies: a watermark/stamp overlay's box can still beat a real block's
    // on raw `top` (it's often the larger of the two), which would otherwise sort it first
    // -- wrong for something drawn *across* existing content, not above it. Compute overlay
    // status once per block against every other block in scope before sorting, since the
    // ordinary top/left comparator can't see other blocks to make that call itself.
    let geometry: Vec<(BBox, Option<f32>)> = blocks
        .iter()
        .map(|b| (b.bbox, block_font_size(b)))
        .collect();
    let is_overlay: Vec<bool> = geometry
        .iter()
        .enumerate()
        .map(|(i, &(bbox, font_size))| {
            geometry
                .iter()
                .enumerate()
                .any(|(j, &(other_bbox, other_font_size))| {
                    i != j && overlays(bbox, font_size, other_bbox, other_font_size)
                })
        })
        .collect();

    let mut tagged: Vec<(bool, Block)> = blocks
        .drain(..)
        .zip(is_overlay)
        .map(|(b, o)| (o, b))
        .collect();
    tagged.sort_by(|(a_overlay, a), (b_overlay, b)| {
        a_overlay.cmp(b_overlay).then_with(|| {
            b.bbox
                .top
                .partial_cmp(&a.bbox.top)
                .unwrap()
                .then(a.bbox.left.partial_cmp(&b.bbox.left).unwrap())
        })
    });
    *blocks = tagged.into_iter().map(|(_, b)| b).collect();
}

/// `true` if `bbox` (with median font size `font_size`) reads as an overlay drawn across
/// `other_bbox` (median font size `other_font_size`): its text is at least
/// [`OVERLAY_FONT_SIZE_RATIO_MIN`] times larger, and their boxes intersect over at least
/// [`OVERLAY_AREA_OVERLAP_FRACTION`] of the smaller box's area. `None` font sizes (a block
/// with no character data, e.g. a hand-built test fixture) never qualify -- there's nothing
/// to compare.
fn overlays(
    bbox: BBox,
    font_size: Option<f32>,
    other_bbox: BBox,
    other_font_size: Option<f32>,
) -> bool {
    let (Some(font_size), Some(other_font_size)) = (font_size, other_font_size) else {
        return false;
    };
    if font_size < other_font_size * OVERLAY_FONT_SIZE_RATIO_MIN {
        return false;
    }

    let overlap_x = (bbox.right.min(other_bbox.right) - bbox.left.max(other_bbox.left)).max(0.0);
    let overlap_y = (bbox.top.min(other_bbox.top) - bbox.bottom.max(other_bbox.bottom)).max(0.0);
    let overlap_area = overlap_x * overlap_y;
    if overlap_area <= 0.0 {
        return false;
    }

    let smaller_area = (bbox.width() * bbox.height())
        .min(other_bbox.width() * other_bbox.height())
        .max(1.0);
    overlap_area / smaller_area >= OVERLAY_AREA_OVERLAP_FRACTION
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

// --- Cross-page continuation stitching --------------------------------------------

/// A left-edge (indentation) mismatch smaller than this, expressed as a multiple of the
/// blocks' own font size, is not considered evidence the blocks are a different column
/// or a differently indented paragraph -- see [`left_edges_align`].
const STITCH_LEFT_ALIGN_EMS: f32 = 1.0;

/// Absolute floor for [`STITCH_LEFT_ALIGN_EMS`], in points, so a pathologically tiny
/// font size can't shrink the left-alignment tolerance to near zero.
const STITCH_LEFT_ALIGN_ABS_PT: f32 = 3.0;

/// Maximum relative difference between two blocks' median font sizes for them to still
/// be considered the same running paragraph -- see [`font_sizes_similar`].
const STITCH_FONT_SIZE_TOLERANCE: f32 = 0.15;

/// Tolerance, in points, for treating a block's bbox edge as *the* extreme edge on its
/// page (see [`is_page_bottom`]/[`is_page_top`]), to absorb float round-off from
/// extraction rather than requiring bit-exact equality.
const STITCH_EDGE_EPSILON_PT: f32 = 0.5;

/// Walks `pages` in document order and merges a page's last-in-reading-order block into
/// the next page's first-in-reading-order block wherever [`should_stitch`] says they
/// look like one paragraph split by the page boundary -- the Stage 4a fix for the
/// `CrossPageContinuation` pitfall (roadmap: "Cross-page stitching via
/// bottom-of-page/top-of-next-page bbox adjacency + incomplete-sentence detection").
///
/// `left` deliberately does *not* always advance to `right`: when a stitch happens, the
/// merged content now lives in page `left`'s block, so a paragraph spanning three or
/// more pages keeps extending that same block instead of losing the chain the moment an
/// intermediate page is emptied out.
fn stitch_cross_page_continuations(pages: &mut [Page]) {
    if pages.len() < 2 {
        return;
    }

    let mut left = 0usize;
    for right in 1..pages.len() {
        let (left_slice, right_slice) = pages.split_at_mut(right);
        let left_page = &mut left_slice[left];
        let right_page = &mut right_slice[0];

        let Some(tail_idx) = left_page.blocks.len().checked_sub(1) else {
            left = right;
            continue;
        };
        if right_page.blocks.is_empty() {
            left = right;
            continue;
        }

        let stitch = should_stitch(
            left_page,
            &left_page.blocks[tail_idx],
            right_page,
            &right_page.blocks[0],
        );

        if stitch {
            let head_block = right_page.blocks.remove(0);
            let merged = merge_blocks(&left_page.blocks[tail_idx], &head_block);
            left_page.blocks[tail_idx] = merged;
            // `left` stays put -- see doc comment above.
        } else {
            left = right;
        }
    }
}

/// `true` if `tail_block` (the last block, in reading order, of `tail_page`) and
/// `head_block` (the first block, in reading order, of `head_page`) look like a single
/// paragraph split by the page boundary between them:
///
/// - `tail_block` sits at the bottom of its own page's content and `head_block` sits at
///   the top of its own page's content ([`is_page_bottom`]/[`is_page_top`]) -- expressed
///   relative to the page's *own* content rather than a fixed physical margin, so it
///   fires even when the split line sits far from the physical page edge (e.g. a mostly
///   blank page).
/// - The two blocks share a left edge, within [`STITCH_LEFT_ALIGN_EMS`]
///   ([`left_edges_align`]) -- rules out a different column or a differently indented
///   paragraph.
/// - The two blocks' font sizes are close, within [`STITCH_FONT_SIZE_TOLERANCE`]
///   ([`font_sizes_similar`]) -- rules out gluing body text to a running
///   header/footer/caption of a different size.
/// - `tail_block`'s last line does not end in sentence-final punctuation, and
///   `head_block`'s first line starts with a lowercase letter -- the incomplete-sentence
///   signal the roadmap calls for, and specifically what keeps a new heading, a running
///   header, or a bulleted/numbered list item on the next page from being swallowed into
///   the previous page's last paragraph.
fn should_stitch(
    tail_page: &Page,
    tail_block: &Block,
    head_page: &Page,
    head_block: &Block,
) -> bool {
    let (Some(tail_line), Some(head_line)) = (tail_block.lines.last(), head_block.lines.first())
    else {
        return false;
    };

    is_page_bottom(tail_page, tail_block)
        && is_page_top(head_page, head_block)
        && left_edges_align(tail_block, head_block)
        && font_sizes_similar(tail_block, head_block)
        && !ends_with_sentence_final_punct(&tail_line.text)
        && starts_with_lowercase_letter(&head_line.text)
}

/// `true` if no other block on `page` has a lower `bbox.bottom` than `block`'s, i.e.
/// `block` is (one of) the bottom-most block(s) on its page -- see
/// [`STITCH_EDGE_EPSILON_PT`] for the tolerance.
fn is_page_bottom(page: &Page, block: &Block) -> bool {
    let min_bottom = page
        .blocks
        .iter()
        .fold(f32::INFINITY, |acc, b| acc.min(b.bbox.bottom));
    block.bbox.bottom <= min_bottom + STITCH_EDGE_EPSILON_PT
}

/// `true` if no other block on `page` has a higher `bbox.top` than `block`'s, i.e.
/// `block` is (one of) the top-most block(s) on its page. Mirrors [`is_page_bottom`].
fn is_page_top(page: &Page, block: &Block) -> bool {
    let max_top = page
        .blocks
        .iter()
        .fold(f32::NEG_INFINITY, |acc, b| acc.max(b.bbox.top));
    block.bbox.top >= max_top - STITCH_EDGE_EPSILON_PT
}

/// `true` if `tail`'s and `head`'s left edges differ by no more than
/// [`STITCH_LEFT_ALIGN_EMS`] times whichever block has font-size data available
/// (clamped to [`STITCH_LEFT_ALIGN_ABS_PT`]), falling back to a plain 12pt em if neither
/// block carries character data (e.g. a hand-built test fixture with empty lines).
fn left_edges_align(tail: &Block, head: &Block) -> bool {
    let em = block_font_size(tail)
        .or_else(|| block_font_size(head))
        .unwrap_or(12.0);
    let threshold = (STITCH_LEFT_ALIGN_EMS * em).max(STITCH_LEFT_ALIGN_ABS_PT);
    (tail.bbox.left - head.bbox.left).abs() <= threshold
}

/// `true` if `tail`'s and `head`'s median font sizes are within
/// [`STITCH_FONT_SIZE_TOLERANCE`] of each other, or if either block has no character
/// data to measure a font size from (nothing to contradict a match).
fn font_sizes_similar(tail: &Block, head: &Block) -> bool {
    match (block_font_size(tail), block_font_size(head)) {
        (Some(a), Some(b)) => {
            let diff = (a - b).abs();
            diff / a.max(b).max(f32::EPSILON) <= STITCH_FONT_SIZE_TOLERANCE
        }
        _ => true,
    }
}

/// Returns the median `font_size` across every [`crate::Char`] in `block`, or `None` if
/// the block has no characters to measure -- the block-scoped counterpart of
/// [`median_font_size`], used to compare two blocks' text scale directly rather than
/// against a whole page's.
fn block_font_size(block: &Block) -> Option<f32> {
    let mut sizes: Vec<f32> = block
        .lines
        .iter()
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

/// `true` if `text`, right-trimmed, ends in a character that plausibly closes a
/// sentence (or a quotation/parenthetical following one) -- the guard against stitching
/// a genuinely finished paragraph into whatever starts the next page.
fn ends_with_sentence_final_punct(text: &str) -> bool {
    const TERMINATORS: &[char] = &['.', '!', '?', ':', ';', '"', '\'', ')', '»', '”'];
    matches!(text.trim_end().chars().last(), Some(c) if TERMINATORS.contains(&c))
}

/// `true` if `text`, left-trimmed, starts with a lowercase letter -- the
/// incomplete-sentence signal that a stitch candidate is a continuation rather than a
/// new heading, running header, or list item (which conventionally start uppercase,
/// with a digit, or with a bullet glyph).
fn starts_with_lowercase_letter(text: &str) -> bool {
    text.trim_start()
        .chars()
        .next()
        .is_some_and(char::is_lowercase)
}

/// Merges `head` into `tail`, joining their last/first lines into one line (see
/// [`join_with_dehyphenation`]) and keeping `tail`'s bbox -- a cross-page union would be
/// geometrically meaningless, and holding it steady keeps
/// [`crate::serialize::to_markdown_structured`]'s bbox-equality region lookup working
/// for the merged block.
fn merge_blocks(tail: &Block, head: &Block) -> Block {
    let mut lines = tail.lines.clone();
    // Checked non-empty by `should_stitch` before this is called.
    let tail_last = lines
        .pop()
        .expect("should_stitch checked tail has a last line");
    let head_first = head
        .lines
        .first()
        .expect("should_stitch checked head has a first line");

    let mut words = tail_last.words.clone();
    words.extend(head_first.words.iter().cloned());
    lines.push(Line {
        text: join_with_dehyphenation(&tail_last.text, &head_first.text),
        bbox: tail_last.bbox,
        words,
    });
    lines.extend(head.lines[1..].iter().cloned());

    Block {
        bbox: tail.bbox,
        lines,
    }
}

/// Joins `tail_text` and `head_text` with a single space, unless `tail_text` ends in a
/// hyphen immediately preceded by a letter, in which case the hyphen is dropped and the
/// two are joined directly -- standard end-of-line-hyphenation reversal.
fn join_with_dehyphenation(tail_text: &str, head_text: &str) -> String {
    let trimmed_tail = tail_text.trim_end();
    let head_text = head_text.trim_start();

    if let Some(stripped) = trimmed_tail
        .strip_suffix('-')
        .or_else(|| trimmed_tail.strip_suffix('\u{2010}'))
    {
        if stripped
            .chars()
            .next_back()
            .is_some_and(char::is_alphabetic)
        {
            return format!("{stripped}{head_text}");
        }
    }

    format!("{trimmed_tail} {head_text}")
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

    // --- Cross-page continuation stitching ------------------------------------

    /// Builds a one-line block with real text and character data at `font_size`, so
    /// text-content predicates ([`ends_with_sentence_final_punct`],
    /// [`starts_with_lowercase_letter`]) and font-size predicates
    /// ([`font_sizes_similar`], [`left_edges_align`]) both have something to key off,
    /// unlike [`block`]/[`block_with_font`]'s placeholder `"line"` text.
    fn text_block(text: &str, bbox: BBox, font_size: f32) -> Block {
        let chars: Vec<Char> = text
            .chars()
            .map(|c| Char {
                unicode: Some(c),
                bbox,
                font_name: "Test".into(),
                font_size,
            })
            .collect();
        let word = Word {
            text: text.into(),
            bbox,
            chars,
        };
        Block {
            bbox,
            lines: vec![Line {
                text: text.into(),
                bbox,
                words: vec![word],
            }],
        }
    }

    fn page_of(index: usize, blocks: Vec<Block>) -> Page {
        Page {
            index,
            width: 612.0,
            height: 792.0,
            blocks,
        }
    }

    #[test]
    fn continuation_split_across_two_pages_is_stitched_into_one_block() {
        let tail_bbox = BBox {
            left: 72.0,
            bottom: 697.5,
            right: 328.8,
            top: 710.9,
        };
        let head_bbox = BBox {
            left: 72.0,
            bottom: 697.5,
            right: 307.5,
            top: 710.9,
        };
        let page0 = page_of(
            0,
            vec![text_block(
                "This sentence continues on the next",
                tail_bbox,
                12.0,
            )],
        );
        let page1 = page_of(
            1,
            vec![text_block(
                "page without a heading to signal a restart.",
                head_bbox,
                12.0,
            )],
        );

        let doc = Document {
            pages: vec![page0, page1],
        };
        let assembled = assemble_reading_order(&doc);

        assert_eq!(assembled.pages[0].blocks.len(), 1);
        assert_eq!(
            assembled.pages[0].blocks[0].text(),
            "This sentence continues on the next page without a heading to signal a restart."
        );
        assert!(assembled.pages[1].blocks.is_empty());
    }

    #[test]
    fn continuation_chains_across_three_pages() {
        // Each page's block continues into the next; the merge should keep extending
        // page 0's block rather than losing the chain once page 1 is emptied out.
        let bbox = BBox {
            left: 72.0,
            bottom: 697.5,
            right: 328.8,
            top: 710.9,
        };
        let doc = Document {
            pages: vec![
                page_of(0, vec![text_block("one two three", bbox, 12.0)]),
                page_of(1, vec![text_block("four five six", bbox, 12.0)]),
                page_of(2, vec![text_block("seven eight nine.", bbox, 12.0)]),
            ],
        };

        let assembled = assemble_reading_order(&doc);

        assert_eq!(assembled.pages[0].blocks.len(), 1);
        assert_eq!(
            assembled.pages[0].blocks[0].text(),
            "one two three four five six seven eight nine."
        );
        assert!(assembled.pages[1].blocks.is_empty());
        assert!(assembled.pages[2].blocks.is_empty());
    }

    #[test]
    fn hyphenated_line_break_is_dehyphenated_on_stitch() {
        let bbox = BBox {
            left: 72.0,
            bottom: 697.5,
            right: 328.8,
            top: 710.9,
        };
        let doc = Document {
            pages: vec![
                page_of(0, vec![text_block("a hyphen-", bbox, 12.0)]),
                page_of(1, vec![text_block("ated word.", bbox, 12.0)]),
            ],
        };

        let assembled = assemble_reading_order(&doc);

        assert_eq!(assembled.pages[0].blocks[0].text(), "a hyphenated word.");
    }

    #[test]
    fn finished_sentence_is_not_stitched_to_next_page() {
        // Tail ends in a period -- a complete sentence, not a split one.
        let bbox = BBox {
            left: 72.0,
            bottom: 697.5,
            right: 328.8,
            top: 710.9,
        };
        let doc = Document {
            pages: vec![
                page_of(0, vec![text_block("A finished sentence.", bbox, 12.0)]),
                page_of(1, vec![text_block("a new paragraph starts.", bbox, 12.0)]),
            ],
        };

        let assembled = assemble_reading_order(&doc);

        assert_eq!(assembled.pages[0].blocks.len(), 1);
        assert_eq!(assembled.pages[1].blocks.len(), 1);
    }

    #[test]
    fn capitalized_next_page_start_is_not_stitched() {
        // Head starts uppercase -- looks like a new sentence/heading, not a
        // continuation, even though the tail doesn't end in terminal punctuation.
        let bbox = BBox {
            left: 72.0,
            bottom: 697.5,
            right: 328.8,
            top: 710.9,
        };
        let doc = Document {
            pages: vec![
                page_of(0, vec![text_block("An unfinished lead-in", bbox, 12.0)]),
                page_of(1, vec![text_block("New Section Heading", bbox, 12.0)]),
            ],
        };

        let assembled = assemble_reading_order(&doc);

        assert_eq!(assembled.pages[0].blocks.len(), 1);
        assert_eq!(assembled.pages[1].blocks.len(), 1);
    }

    #[test]
    fn mismatched_left_edge_is_not_stitched() {
        let tail_bbox = BBox {
            left: 72.0,
            bottom: 697.5,
            right: 328.8,
            top: 710.9,
        };
        let head_bbox = BBox {
            left: 200.0,
            bottom: 697.5,
            right: 450.0,
            top: 710.9,
        };
        let doc = Document {
            pages: vec![
                page_of(0, vec![text_block("indented differently", tail_bbox, 12.0)]),
                page_of(1, vec![text_block("on the next page", head_bbox, 12.0)]),
            ],
        };

        let assembled = assemble_reading_order(&doc);

        assert_eq!(assembled.pages[0].blocks.len(), 1);
        assert_eq!(assembled.pages[1].blocks.len(), 1);
    }

    #[test]
    fn mismatched_font_size_is_not_stitched() {
        let bbox = BBox {
            left: 72.0,
            bottom: 697.5,
            right: 328.8,
            top: 710.9,
        };
        let doc = Document {
            pages: vec![
                page_of(0, vec![text_block("a body paragraph that", bbox, 10.0)]),
                page_of(1, vec![text_block("running header text", bbox, 22.0)]),
            ],
        };

        let assembled = assemble_reading_order(&doc);

        assert_eq!(assembled.pages[0].blocks.len(), 1);
        assert_eq!(assembled.pages[1].blocks.len(), 1);
    }

    #[test]
    fn header_footer_repeated_across_pages_is_not_stitched() {
        // Regression guard for the header_footer corpus case's shape: a short,
        // uppercase-starting running header sits at the top of each page, and body
        // text (ending in a period) sits lower. Neither the header-to-header nor the
        // body-to-header pairing should stitch.
        let header_bbox = BBox {
            left: 40.0,
            bottom: 760.0,
            right: 300.0,
            top: 775.0,
        };
        let body_bbox = BBox {
            left: 40.0,
            bottom: 300.0,
            right: 560.0,
            top: 755.0,
        };
        let doc = Document {
            pages: vec![
                page_of(
                    0,
                    vec![
                        text_block("Chapter 4: Results", header_bbox, 10.0),
                        text_block("Body text for page one.", body_bbox, 10.0),
                    ],
                ),
                page_of(
                    1,
                    vec![
                        text_block("Chapter 4: Results", header_bbox, 10.0),
                        text_block("Body text for page two.", body_bbox, 10.0),
                    ],
                ),
            ],
        };

        let assembled = assemble_reading_order(&doc);

        assert_eq!(assembled.pages[0].blocks.len(), 2);
        assert_eq!(assembled.pages[1].blocks.len(), 2);
    }
}

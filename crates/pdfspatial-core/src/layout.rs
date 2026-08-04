//! Stage 2/4: layout region classification.
//!
//! [`classify_regions`] combines two independent, deterministic signals -- no vision
//! model or inference runtime involved:
//!
//! - A **text-layer-only heuristic** classifier (the roadmap's Stage 4a "heuristics
//!   first" approach) over geometry, font metrics, and text, emitting
//!   [`RegionClass::Title`], [`RegionClass::SectionHeader`], [`RegionClass::ListItem`],
//!   [`RegionClass::Caption`], [`RegionClass::PageHeader`], [`RegionClass::PageFooter`],
//!   [`RegionClass::Footnote`], and [`RegionClass::Text`].
//! - A **graphics-layer** pass ([`crate::graphics::detect_table_regions`],
//!   [`crate::graphics::detect_picture_regions`]), run first over each page's
//!   [`crate::Graphic`]s (ruling lines, images, fills), emitting
//!   [`RegionClass::Table`] and [`RegionClass::Picture`]. A block whose center falls
//!   inside a detected table or picture is excluded from the text-layer pass so it
//!   isn't double-classified.
//!
//! [`RegionClass::Formula`] is produced for **display** (block-level) formulas via
//! [`is_display_formula`] -- a centered, narrow, vertically isolated block whose text is
//! dense with math symbols/digits and doesn't read as a sentence. *Inline* formula
//! segmentation (a formula embedded mid-sentence in a body paragraph) still requires a
//! genuine layout/vision model -- there is no geometric signal to key off there, since
//! the formula shares its block with ordinary prose -- and is never produced here; see
//! `docs/pitfall_registry.json`'s `nested_formula` entry.
//!
//! [`RegionClass::PageHeader`]/[`RegionClass::PageFooter`] come from two independent
//! signals, either of which is sufficient: a single-page geometric shape test (thin
//! strip in the top/bottom band, detached from the body by a minimum gap — see
//! [`classify_block`]), and cross-page repeated-content detection (the same
//! digit-normalized text recurring at the same page-relative position across
//! [`REPEATED_BAND_MIN_PAGES`] consecutive pages — see [`repeated_running_bands`]) for
//! running headers/footers too tall or too close to the body for the geometric rule
//! alone to catch.
//!
//! [`RegionClass::Footnote`] requires three independent signals to hold at once: the
//! block sits in the lower [`FOOTNOTE_BAND_FRACTION`] of the page, its font is smaller
//! than the rest of the page's text (see [`body_font_size_excluding`] for why the
//! comparison excludes the candidate block itself), and its text opens with a
//! recognized footnote marker (a bare digit/symbol, not a bullet — see
//! [`starts_with_footnote_marker`]). It is checked after the header/footer band rules,
//! so a block already claimed as a running footer never also matches as a footnote.

use crate::{BBox, Block, Document, Line};

/// DocLayNet's 11 region categories.
///
/// Mirrors the class schema used by DocLayNet-v1.1 exactly, since Stage 2 validation
/// scores predictions against DocLayNet ground truth. See the roadmap's Stage 2 section
/// for per-class GIoU/F1 targets, and note that `Footnote`, `Page-header`, and
/// `Page-footer` are called out there as the historically weakest classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    /// Zero-based index of the [`crate::Page`] this region belongs to (matches
    /// [`crate::Page::index`]). `bbox` alone is not enough to place a region on the
    /// right page: two different pages' coordinate spaces both start near `(0, 0)`, so
    /// without this field a region from one page can numerically collide with a block on
    /// another page -- see [`crate::serialize::render_page`], which filters regions to
    /// their own page before matching them against that page's blocks.
    pub page: usize,
}

/// A block's font size is "large" relative to the page body when it exceeds the body
/// median by at least this factor.
const SECTION_HEADER_FONT_FACTOR: f32 = 1.15;

/// A block sitting above this fraction of the page height is in the header band.
const HEADER_BAND_FRACTION: f32 = 0.92;

/// A block sitting below this fraction of the page height is in the footer band.
const FOOTER_BAND_FRACTION: f32 = 0.08;

/// A running header/footer is a thin strip; a band block taller than this fraction of
/// the page is body text that merely starts (or ends) at the page edge, not a running
/// header/footer.
const HEADER_FOOTER_MAX_HEIGHT_FRACTION: f32 = 0.10;

/// A running header/footer is visually detached from the body. A band block closer than
/// this fraction of the page height to its nearest neighbour is part of the body flow
/// rather than a separate running header/footer.
const HEADER_FOOTER_MIN_BODY_GAP_FRACTION: f32 = 0.02;

/// How many consecutive pages a band block's (normalized) text must recur on to count as
/// running header/footer content, per the roadmap's repeated-content signal.
const REPEATED_BAND_MIN_PAGES: usize = 2;

/// Page-relative vertical tolerance for "same y-position range" when matching a band
/// block's position across pages.
const REPEATED_BAND_Y_TOLERANCE: f32 = 0.02;

/// A caption/list block spanning more than this many lines is treated as body text.
const SHORT_BLOCK_MAX_LINES: usize = 2;

/// A block sitting below this fraction of the page height is eligible to be a footnote --
/// wider than [`FOOTER_BAND_FRACTION`] since footnotes commonly sit just above a running
/// footer rather than sharing its narrow strip.
const FOOTNOTE_BAND_FRACTION: f32 = 0.25;

/// A footnote's font must be no larger than this factor of the surrounding body text's
/// font size (see [`body_font_size_excluding`]).
const FOOTNOTE_FONT_FACTOR: f32 = 0.95;

/// A block (or page) counts as "bold" once at least this fraction of its characters
/// carry a bold font name (see [`is_bold_font_name`]) -- a dominant-fraction test rather
/// than "any bold char," so a single bold run-in word inside an otherwise-regular
/// paragraph doesn't get promoted, and a heading with one non-bold stray glyph still
/// qualifies.
const BOLD_CHAR_FRACTION: f32 = 0.8;

/// A borderless-table row candidate's tallest member block must be no taller than this
/// fraction of the page height -- real body/column text runs the height of the page,
/// while a table row's cells are short. Same order of magnitude as
/// [`HEADER_FOOTER_MAX_HEIGHT_FRACTION`], but named separately since the two guard
/// unrelated heuristics.
const BORDERLESS_TABLE_ROW_MAX_HEIGHT_FRACTION: f32 = 0.15;

/// A whitespace corridor (see [`whitespace_column_corridors`]) only counts as a column
/// separator once it's at least this many multiples of the block's own font size wide.
/// Ordinary inter-word spacing is well under one font size, so this cleanly separates a
/// table gutter from a single wide space in running text; it also matches the same
/// "gutter must dominate" spirit as [`detect_borderless_table_regions`]'s narrowest-gutter
/// test, just expressed per-corridor instead of per-row.
const BORDERLESS_TABLE_MIN_CORRIDOR_FACTOR: f32 = 1.5;

/// [`whitespace_column_corridors`] never fires on a single line -- one line has no other
/// line to corroborate a corridor against, so any "column" it appears to have could just
/// be irregular inter-word spacing.
const BORDERLESS_TABLE_MIN_LINES: usize = 2;

/// A display formula's horizontal center must fall within this fraction of the page
/// width from the true page center -- a display formula is centered on the page (or the
/// text column), unlike a left-aligned body paragraph or heading.
const FORMULA_CENTER_TOLERANCE_FRACTION: f32 = 0.08;

/// A display formula's block must be no wider than this fraction of the page width.
/// This is the load-bearing discriminator between a formula and an ordinary centered
/// body paragraph above/below it (also centered, but running most of the page's text
/// width) -- see [`is_display_formula`].
const FORMULA_MAX_WIDTH_FRACTION: f32 = 0.6;

/// A display formula must be separated from its vertical neighbours by at least this
/// multiple of its own font size -- ordinary body text wraps line-to-line with far less
/// clearance than a formula set apart from surrounding prose.
const FORMULA_MIN_ISOLATION_FACTOR: f32 = 1.2;

/// The fraction of a candidate formula block's non-whitespace characters that must fall
/// in the math-symbol/digit set (see [`is_display_formula`]) for it to count as
/// symbol-dense rather than ordinary prose that merely happens to be centered and
/// narrow.
const FORMULA_MIN_MATH_DENSITY: f32 = 0.15;

/// A line-to-line font-size ratio at or above this counts as a style break for
/// [`split_blocks_at_style_breaks`] -- the same magnitude [`classify_block`]'s
/// size-based heading rule already uses, so a split only ever separates lines that
/// [`classify_block`] would also treat as differently sized.
const HEADING_SPLIT_FONT_FACTOR: f32 = SECTION_HEADER_FONT_FACTOR;

/// Splits a `document`'s blocks at internal style breaks (a font-size jump of at least
/// [`HEADING_SPLIT_FONT_FACTOR`], or a boldness flip, between two consecutive lines),
/// returning a new [`Document`] -- the input is never mutated.
///
/// This exists because real Stage 1 block grouping ([`crate::extract::group_blocks`])
/// merges lines purely on vertical-gap geometry, blind to a font-size or weight change
/// between them. A heading line sitting flush against the paragraph it introduces (no
/// extra vertical gap) ends up trapped as an interior line of one large `Text` block
/// instead of its own block -- and [`classify_block`]'s heading rules only ever fire on
/// a whole block, never a line within one, so that heading is never classified as a
/// heading and never reaches `#`/`##` in the rendered Markdown.
///
/// Splitting here, as a Stage 2 pass over a *copy* of `document`, rather than in
/// [`crate::extract::group_blocks`] itself, keeps Stage 1's own extraction output --
/// and every PDF-backed regression-corpus snapshot frozen against it -- unchanged. Only
/// [`crate::serialize::to_markdown_pipeline`] (and therefore the `pdfspatial` CLI) calls
/// this; a caller driving [`classify_regions`]/[`crate::assemble::assemble_reading_order`]
/// manually gets Stage 1's original, unsplit blocks unless it calls this first itself.
///
/// A split only ever separates lines *within* one block -- it never merges lines across
/// two different Stage 1 blocks, so it can only recover structure Stage 1 already threw
/// away by over-merging, never invent structure that wasn't there.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::layout::split_blocks_at_style_breaks;
/// use pdfspatial_core::{BBox, Block, Document, Line, Page, Word};
///
/// fn line(text: &str, font_size: f32, bbox: BBox) -> Line {
///     use pdfspatial_core::Char;
///     let chars = text
///         .chars()
///         .map(|c| Char {
///             unicode: Some(c),
///             bbox,
///             font_size,
///             font_name: "Helvetica".into(),
///             ..Default::default()
///         })
///         .collect();
///     let word = Word { text: text.into(), bbox, chars };
///     Line { text: text.into(), bbox, words: vec![word] }
/// }
///
/// let heading = line(
///     "A Heading",
///     16.0,
///     BBox { left: 0.0, bottom: 720.0, right: 200.0, top: 736.0 },
/// );
/// let body = line(
///     "Body text that follows immediately, no gap.",
///     10.0,
///     BBox { left: 0.0, bottom: 700.0, right: 400.0, top: 716.0 },
/// );
/// let block = Block {
///     bbox: BBox { left: 0.0, bottom: 700.0, right: 400.0, top: 736.0 },
///     lines: vec![heading, body],
/// };
/// let doc = Document {
///     pages: vec![Page { index: 0, width: 612.0, height: 792.0, blocks: vec![block], ..Default::default() }],
/// };
///
/// let split = split_blocks_at_style_breaks(&doc);
/// assert_eq!(split.pages[0].blocks.len(), 2);
/// assert_eq!(split.pages[0].blocks[0].text(), "A Heading");
/// ```
pub fn split_blocks_at_style_breaks(document: &Document) -> Document {
    Document {
        pages: document
            .pages
            .iter()
            .map(|page| crate::Page {
                blocks: page.blocks.iter().flat_map(split_block).collect(),
                ..page.clone()
            })
            .collect(),
    }
}

/// Splits one [`Block`] into one or more blocks at internal style breaks -- see
/// [`split_blocks_at_style_breaks`]. Returns `vec![block.clone()]` unchanged when no
/// break qualifies (the common case), so this is safe to call unconditionally over every
/// block.
fn split_block(block: &Block) -> Vec<Block> {
    if block.lines.len() < 2 {
        return vec![block.clone()];
    }

    let mut groups: Vec<Vec<Line>> = vec![vec![block.lines[0].clone()]];
    for line in &block.lines[1..] {
        let prev = groups
            .last()
            .and_then(|g| g.last())
            .expect("group is never empty");
        if is_style_break(prev, line) {
            groups.push(vec![line.clone()]);
        } else {
            groups
                .last_mut()
                .expect("group is never empty")
                .push(line.clone());
        }
    }

    if groups.len() == 1 {
        return vec![block.clone()];
    }

    groups
        .into_iter()
        .filter_map(|lines| {
            let bbox = lines.iter().map(|l| l.bbox).reduce(|a, b| a.union(&b))?;
            Some(Block { bbox, lines })
        })
        .collect()
}

/// `true` if `prev` and `curr` (two consecutive lines within one Stage 1 block) differ
/// enough in font size or boldness that they belong in different blocks -- see
/// [`split_blocks_at_style_breaks`].
fn is_style_break(prev: &Line, curr: &Line) -> bool {
    let prev_size = line_font_size(prev);
    let curr_size = line_font_size(curr);
    if prev_size > 0.0 && curr_size > 0.0 {
        let ratio = (curr_size / prev_size).max(prev_size / curr_size);
        if ratio >= HEADING_SPLIT_FONT_FACTOR {
            return true;
        }
    }
    line_is_bold(prev) != line_is_bold(curr)
}

/// The character-weighted median font size of `line`'s own characters.
fn line_font_size(line: &Line) -> f32 {
    median_font_size(
        line.words
            .iter()
            .flat_map(|w| &w.chars)
            .map(|c| c.font_size)
            .collect(),
    )
}

/// `true` if at least [`BOLD_CHAR_FRACTION`] of `line`'s characters carry a bold font
/// name.
fn line_is_bold(line: &Line) -> bool {
    bold_char_fraction(line.words.iter().flat_map(|w| &w.chars)) >= BOLD_CHAR_FRACTION
}

/// Classifies every geometric [`crate::Block`] in `document` into a [`Region`], one
/// region per block (in document order, page-major), using only heuristics over Stage
/// 1's own geometry and text — no vision model. See the [module docs](self) for which
/// classes this can and cannot produce.
///
/// Each returned [`Region`] carries the source block's own bounding box unchanged, so
/// callers (e.g. [`crate::serialize::to_markdown_structured`]) can align a block back to
/// its region by exact bbox equality.
pub fn classify_regions(document: &Document) -> Vec<Region> {
    let mut regions = Vec::new();
    let repeated_bands = repeated_running_bands(document);
    // Threaded across every page, not reset per page, so a multi-page document gets at
    // most one `Title` overall (the first oversized block near the top of its first
    // page) -- a long PDF shouldn't emit one spurious `#` per page.
    let mut seen_title = false;

    for page in &document.pages {
        let body_font_size = body_median_font_size(page);
        let page_predominantly_bold = page_is_predominantly_bold(page);

        let table_regions = crate::graphics::detect_table_regions(&page.graphics, page);
        let picture_regions = crate::graphics::detect_picture_regions(&page.graphics, page);

        let ungraphed_blocks: Vec<&Block> = page
            .blocks
            .iter()
            .filter(|block| {
                !table_regions
                    .iter()
                    .chain(&picture_regions)
                    .any(|r| r.bbox.contains_center(&block.bbox))
            })
            .collect();
        let borderless_table_regions =
            detect_borderless_table_regions(&ungraphed_blocks, page.height, page.index);

        for block in &page.blocks {
            // A block claimed by the graphics layer (its center sits inside a detected
            // table or picture) is excluded from the text-layer heuristic entirely --
            // its content is represented by the table/picture region itself, not by a
            // separate per-block region. See `crate::serialize` for how a table's member
            // blocks are folded back into one rendered GFM table.
            let claimed_by_graphics = table_regions
                .iter()
                .chain(&picture_regions)
                .chain(&borderless_table_regions)
                .any(|r| r.bbox.contains_center(&block.bbox));
            if claimed_by_graphics {
                continue;
            }

            let repeated_band = band_of(block, page.height).filter(|band| {
                repeated_bands.contains(&(*band, normalize_running_text(&block.text())))
            });
            let (class, confidence) = classify_block(
                block,
                page,
                body_font_size,
                page_predominantly_bold,
                repeated_band,
                &mut seen_title,
            );
            regions.push(Region {
                class,
                bbox: block.bbox,
                confidence,
                page: page.index,
            });
        }

        regions.extend(table_regions);
        regions.extend(picture_regions);
        regions.extend(borderless_table_regions);
    }

    regions
}

/// Detects borderless tables from text geometry alone: bands of `blocks` that share the
/// same vertical span, sit side by side with no vertical overlap between neighbours, and
/// are separated by a gutter wide enough that it can't be ordinary word-wrapping --
/// stacked vertically-adjacent bands with matching column counts and aligned column
/// starts are merged into one multi-row [`Region`], rather than emitted as a separate
/// single-row region each (see [`rows_are_compatible`]): a real table has several rows,
/// and treating each one as its own table produced as many degenerate one-row GFM tables
/// as the source had rows -- see `docs/pitfall_registry.json`'s `borderless_table` entry.
///
/// This is the text-layer counterpart to [`crate::graphics::detect_table_regions`], for
/// tables with no ruling lines to key a grid off of -- see `docs/pitfall_registry.json`'s
/// `multi_line_table_cell` entry. Unlike the ruling-line detector, this one has no grid
/// geometry to reconstruct cells from, so [`crate::serialize`]'s renderer re-derives rows
/// from the region's own member blocks (clustering them back into rows the same way this
/// function does, via [`cluster_blocks_by_vertical_band`]) when
/// [`crate::graphics::table_grid_cells`] finds no ruling lines to work from.
///
/// A candidate band qualifies as a table row only when *all* of these hold, each guarding
/// against a specific way ordinary body text can look like a row of side-by-side blocks:
///
/// - at least two blocks share the band (a single block is never a "row");
/// - every pair of blocks in the band is horizontally disjoint (no two cells overlap
///   in x -- if they did, this is one paragraph's wrapped lines caught by the vertical
///   overlap test, not two cells);
/// - the *narrowest* gutter between horizontally adjacent blocks is at least as wide as
///   the band's *widest* block -- multi-column prose (see the `multi_column`/
///   `list_nesting`/`figure_caption` fixtures) runs a 10-40pt gutter next to
///   200pt-plus-wide columns, while a short table cell's gutter routinely exceeds its own
///   width, so this ratio is the load-bearing discriminator between the two;
/// - every block in the band is shorter than [`BORDERLESS_TABLE_ROW_MAX_HEIGHT_FRACTION`]
///   of the page (a real body column runs most of the page's height; a table row's cells
///   don't);
/// - no block in the band sits in the header/footer band (so a left/right running
///   header or footer pair, which is otherwise exactly this shape, is never misread as a
///   table row).
///
/// Confidence is fixed at `0.5`: weaker evidence than
/// [`crate::graphics::detect_table_regions`]'s `0.75`, since ruling lines are direct
/// structural evidence and this is inferred purely from a gutter-width heuristic.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::layout::{detect_borderless_table_regions, RegionClass};
/// use pdfspatial_core::{BBox, Block, Line, Word};
///
/// fn cell(text: &str, bbox: BBox) -> Block {
///     let word = Word { text: text.into(), bbox, chars: vec![] };
///     let line = Line { text: text.into(), bbox, words: vec![word] };
///     Block { bbox, lines: vec![line] }
/// }
///
/// // A narrow left cell and a far-off right cell sharing a row, wide gutter between them.
/// let left = cell("Item", BBox { left: 72.0, bottom: 690.0, right: 165.0, top: 709.0 });
/// let right = cell("42", BBox { left: 340.0, bottom: 698.0, right: 351.0, top: 709.0 });
/// let blocks = [&left, &right];
///
/// let regions = detect_borderless_table_regions(&blocks, 792.0, 0);
/// assert_eq!(regions.len(), 1);
/// assert_eq!(regions[0].class, RegionClass::Table);
/// ```
pub fn detect_borderless_table_regions(
    blocks: &[&Block],
    page_height: f32,
    page: usize,
) -> Vec<Region> {
    let mut valid_rows: Vec<Vec<&Block>> = cluster_blocks_by_vertical_band(blocks)
        .into_iter()
        .filter_map(|mut band| {
            if band.len() < 2 {
                return None;
            }
            band.sort_by(|a, b| a.bbox.left.partial_cmp(&b.bbox.left).unwrap());

            let widest = band.iter().map(|b| b.bbox.width()).fold(0.0_f32, f32::max);
            let narrowest_gutter = band
                .windows(2)
                .map(|pair| pair[1].bbox.left - pair[0].bbox.right)
                .fold(f32::INFINITY, f32::min);
            if narrowest_gutter < widest {
                return None;
            }

            let tallest = band.iter().map(|b| b.bbox.height()).fold(0.0_f32, f32::max);
            if tallest > page_height * BORDERLESS_TABLE_ROW_MAX_HEIGHT_FRACTION {
                return None;
            }

            if band.iter().any(|b| band_of(b, page_height).is_some()) {
                return None;
            }

            Some(band)
        })
        .collect();

    // Top to bottom, so adjacent rows in reading order sit next to each other below,
    // ready to be merged into one multi-row table.
    valid_rows.sort_by(|a, b| {
        let top_a = a.iter().map(|blk| blk.bbox.top).fold(f32::MIN, f32::max);
        let top_b = b.iter().map(|blk| blk.bbox.top).fold(f32::MIN, f32::max);
        top_b
            .partial_cmp(&top_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Each group is a list of rows (not a flattened list of blocks): compatibility is
    // checked against the *last row added*, not the group's whole accumulated block
    // list, so a 3-row merge chain (row 1 compatible with row 2, row 2 compatible with
    // row 3) doesn't spuriously break because row 3's cell count no longer matches the
    // *combined* cell count of rows 1+2.
    let mut groups: Vec<Vec<Vec<&Block>>> = Vec::new();
    for row in valid_rows {
        match groups.last().and_then(|g| g.last()) {
            Some(prev_row) if rows_are_compatible(prev_row, &row) => {
                groups.last_mut().unwrap().push(row);
            }
            _ => groups.push(vec![row]),
        }
    }

    groups
        .into_iter()
        .filter_map(|group| {
            let bbox = group
                .iter()
                .flatten()
                .map(|b| b.bbox)
                .reduce(|a, b| a.union(&b))?;
            Some(Region {
                class: RegionClass::Table,
                bbox,
                confidence: 0.5,
                page,
            })
        })
        .collect()
}

/// The largest difference, in points, between two rows' matching column's left edges
/// that still counts as "the same columns" for [`rows_are_compatible`]. Generous enough
/// to tolerate ordinary cell-content-width jitter (a longer value in one row nudging a
/// later column's actual text start) without being so wide it merges two structurally
/// unrelated row bands that simply happen to sit near the same x-positions.
const ROW_COLUMN_ALIGNMENT_TOLERANCE_PT: f32 = 20.0;

/// `true` if two already-validated table-row bands (each already sorted left to right by
/// [`detect_borderless_table_regions`]) look like consecutive rows of the *same* table:
/// the same number of cells, each column's left edge aligned within
/// [`ROW_COLUMN_ALIGNMENT_TOLERANCE_PT`] of its counterpart in the other row. A
/// differing cell count or misaligned columns means the two bands are structurally
/// different -- e.g. one genuinely is a table row and the other is an unrelated
/// two-column caption pair that happens to sit nearby -- so they stay separate tables
/// rather than being merged into one inconsistent grid.
fn rows_are_compatible(a: &[&Block], b: &[&Block]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x.bbox.left - y.bbox.left).abs() <= ROW_COLUMN_ALIGNMENT_TOLERANCE_PT)
}

/// Clusters `blocks` into vertically-overlapping bands via union-find: any two blocks
/// whose bboxes vertically overlap end up in the same band, transitively. Used both to
/// find row candidates ([`detect_borderless_table_regions`]) and, on the serialization
/// side, to re-derive a detected borderless table's rows from its own member blocks (see
/// [`crate::serialize`]).
pub(crate) fn cluster_blocks_by_vertical_band<'a>(blocks: &[&'a Block]) -> Vec<Vec<&'a Block>> {
    let mut parent: Vec<usize> = (0..blocks.len()).collect();

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    for i in 0..blocks.len() {
        for j in (i + 1)..blocks.len() {
            if blocks[i].bbox.vertically_overlaps(&blocks[j].bbox) {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    let mut bands: std::collections::BTreeMap<usize, Vec<&Block>> = Default::default();
    for (i, &block) in blocks.iter().enumerate() {
        let root = find(&mut parent, i);
        bands.entry(root).or_default().push(block);
    }
    bands.into_values().collect()
}

/// Splits `line`'s text into non-whitespace tokens, each carrying its own x-extent in
/// page space, left to right.
///
/// A line with two or more [`Word`](crate::Word)s already has real per-word geometry
/// (see [`Line::words`](crate::Line::words)'s own doc comment: "left to right"), so each
/// word's own bbox is used directly. A line with at most one word -- the shape
/// [`crate::eval::corpus`]'s fixture loader produces (one whole-line `Word`), and what a
/// monospaced or kerning-collapsed run from real extraction can look like too -- has no
/// per-word geometry to key off, so this falls back to interpolating a uniform character
/// grid across the line's own bbox and splitting on runs of whitespace. Uniform
/// interpolation is exact for monospaced text; for proportional text it's an
/// approximation, but the corridors [`whitespace_column_corridors`] looks for are wide
/// enough (see [`BORDERLESS_TABLE_MIN_CORRIDOR_FACTOR`]) to survive the resulting jitter.
pub(crate) fn whitespace_line_tokens(line: &Line) -> Vec<(f32, f32, String)> {
    if line.words.len() >= 2 {
        return line
            .words
            .iter()
            .map(|w| (w.bbox.left, w.bbox.right, w.text.clone()))
            .collect();
    }

    let (bbox, text) = match line.words.first() {
        Some(word) => (word.bbox, word.text.as_str()),
        None => (line.bbox, line.text.as_str()),
    };
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() || bbox.width() <= 0.0 {
        return Vec::new();
    }
    let n = chars.len() as f32;
    let x_at = |i: usize| bbox.left + bbox.width() * (i as f32) / n;

    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && !chars[i].is_whitespace() {
            i += 1;
        }
        tokens.push((x_at(start), x_at(i), chars[start..i].iter().collect()));
    }
    tokens
}

/// The x-intervals between `line`'s consecutive tokens (see [`whitespace_line_tokens`]),
/// i.e. every whitespace gap wide enough to separate two tokens -- including ordinary
/// single-space word gaps, which [`whitespace_column_corridors`]'s width filter is
/// responsible for discarding, not this function.
fn whitespace_line_gaps(line: &Line) -> Vec<(f32, f32)> {
    let tokens = whitespace_line_tokens(line);
    tokens
        .windows(2)
        .filter_map(|pair| {
            let gap = (pair[0].1, pair[1].0);
            (gap.1 > gap.0).then_some(gap)
        })
        .collect()
}

/// The pairwise intersection of every interval in `a` against every interval in `b`.
fn intersect_intervals(a: &[(f32, f32)], b: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    for &(a0, a1) in a {
        for &(b0, b1) in b {
            let lo = a0.max(b0);
            let hi = a1.min(b1);
            if lo < hi {
                out.push((lo, hi));
            }
        }
    }
    out
}

/// Detects vertical whitespace corridors shared by every line of `block` -- the
/// text-layer signal for a borderless table whose columns live *inside* one block's
/// lines (as runs of aligned spaces), rather than as several side-by-side blocks (see
/// [`detect_borderless_table_regions`] for that shape). See `docs/pitfall_registry.json`'s
/// `borderless_table` entry.
///
/// Each line contributes the x-gaps between its own consecutive tokens (see
/// [`whitespace_line_gaps`]); a corridor survives only if every line has a gap covering
/// it, so a column that only some lines observe (e.g. a short trailing line, or a
/// heading that doesn't share the body's columns) is not a corridor. This also means a
/// corridor never floats past either edge of any line's own content, since a gap only
/// exists between two real tokens -- ordinary ragged-right prose, where any given
/// vertical slice usually lands inside some line's word rather than in every line's
/// gap, essentially never survives the intersection.
///
/// Surviving corridors are then filtered to those at least
/// [`BORDERLESS_TABLE_MIN_CORRIDOR_FACTOR`] times the block's own font size wide, to
/// discard ordinary single-space word gaps that happen to align by coincidence across a
/// couple of short lines.
///
/// Returns an empty `Vec` for a block with fewer than [`BORDERLESS_TABLE_MIN_LINES`]
/// lines, or one with no shared corridor at all (the common case for ordinary body text).
///
/// # Examples
///
/// ```
/// use pdfspatial_core::layout::whitespace_column_corridors;
/// use pdfspatial_core::{BBox, Block, Char, Line, Word};
///
/// fn char_at(bbox: BBox) -> Char {
///     Char { unicode: None, bbox, font_size: 10.0, ..Default::default() }
/// }
///
/// fn line(text: &str, bbox: BBox) -> Line {
///     let chars = vec![char_at(bbox); text.len()];
///     let word = Word { text: text.into(), bbox, chars };
///     Line { text: text.into(), bbox, words: vec![word] }
/// }
///
/// // Two lines whose columns line up in a wide shared corridor, no ruling lines at all.
/// let bbox = BBox { left: 40.0, bottom: 500.0, right: 400.0, top: 515.0 };
/// let header = line("Name        Score", bbox);
/// let bbox2 = BBox { left: 40.0, bottom: 485.0, right: 400.0, top: 500.0 };
/// let row = line("Alice          92", bbox2);
///
/// let block = Block { bbox: bbox.union(&bbox2), lines: vec![header, row] };
/// assert!(!whitespace_column_corridors(&block).is_empty());
/// ```
pub fn whitespace_column_corridors(block: &Block) -> Vec<(f32, f32)> {
    if block.lines.len() < BORDERLESS_TABLE_MIN_LINES {
        return Vec::new();
    }

    let mut lines = block.lines.iter().map(whitespace_line_gaps);
    let Some(first) = lines.next() else {
        return Vec::new();
    };
    let mut corridors = first;
    if corridors.is_empty() {
        return Vec::new();
    }
    for gaps in lines {
        if gaps.is_empty() {
            return Vec::new();
        }
        corridors = intersect_intervals(&corridors, &gaps);
        if corridors.is_empty() {
            return Vec::new();
        }
    }

    let min_width = block_font_size(block) * BORDERLESS_TABLE_MIN_CORRIDOR_FACTOR;
    corridors.retain(|&(lo, hi)| hi - lo >= min_width);
    corridors
}

/// Which page edge a block sits against, per [`HEADER_BAND_FRACTION`]/[`FOOTER_BAND_FRACTION`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Band {
    /// The block sits in the top-of-page header band.
    Header,
    /// The block sits in the bottom-of-page footer band.
    Footer,
}

/// Classifies a single block. `page` supplies the block's geometric context (its
/// siblings, width, and height) -- `block` itself is expected to be one of `page.blocks`.
/// `seen_title` is threaded across the whole document's blocks (not reset per page) so
/// at most one [`RegionClass::Title`] is emitted overall (the first oversized block near
/// the top of the first page); every later oversized block, on any page, falls back to
/// [`RegionClass::SectionHeader`]. `repeated_band` is `Some` when this
/// block's band and (normalized) text were found to recur across consecutive pages by
/// [`repeated_running_bands`] -- the strongest signal, checked first.
/// `page_predominantly_bold` gates the bold-heading signal (see
/// [`page_is_predominantly_bold`]): bold text only marks a heading when it stands out
/// against the page's own body.
fn classify_block(
    block: &Block,
    page: &crate::Page,
    body_font_size: f32,
    page_predominantly_bold: bool,
    repeated_band: Option<Band>,
    seen_title: &mut bool,
) -> (RegionClass, f32) {
    let page_blocks = &page.blocks;
    let page_width = page.width;
    let page_height = page.height;
    let line_count = block.lines.len();
    let font_size = block_font_size(block);
    let text = block.text();

    match repeated_band {
        Some(Band::Header) => return (RegionClass::PageHeader, 0.8),
        Some(Band::Footer) => return (RegionClass::PageFooter, 0.8),
        None => {}
    }

    let in_header_band = block.bbox.top >= page_height * HEADER_BAND_FRACTION;
    let in_footer_band = block.bbox.bottom <= page_height * FOOTER_BAND_FRACTION;
    let is_thin_strip = block.bbox.height() <= page_height * HEADER_FOOTER_MAX_HEIGHT_FRACTION;

    if in_header_band && is_thin_strip {
        let gap = gap_to_nearest_block(block, page_blocks, /* below */ true);
        if gap >= page_height * HEADER_FOOTER_MIN_BODY_GAP_FRACTION {
            return (RegionClass::PageHeader, 0.6);
        }
    }
    if in_footer_band && is_thin_strip {
        let gap = gap_to_nearest_block(block, page_blocks, /* below */ false);
        if gap >= page_height * HEADER_FOOTER_MIN_BODY_GAP_FRACTION {
            return (RegionClass::PageFooter, 0.6);
        }
    }

    if block.bbox.bottom <= page_height * FOOTNOTE_BAND_FRACTION
        && starts_with_footnote_marker(&text)
    {
        let body_ref = body_font_size_excluding(page_blocks, block);
        if body_ref > 0.0 && font_size <= body_ref * FOOTNOTE_FONT_FACTOR {
            return (RegionClass::Footnote, 0.55);
        }
    }

    if is_list_item(&text) {
        return (RegionClass::ListItem, 0.7);
    }

    if is_caption(&text) && line_count <= SHORT_BLOCK_MAX_LINES {
        return (RegionClass::Caption, 0.6);
    }

    // A numbered/lettered outline heading ("7 Variants of...", "III. Regulatory...",
    // "Appendix A: ...") carries no font-size cue at all when it's set at body size, the
    // same gap the bold-heading rule below fills for bold-at-body-size headings. Checked
    // after `is_list_item` (a single-digit-run-then-period/paren, "1. Introduction", is
    // that rule's shape, not this one's -- see `starts_with_heading_number`) and before
    // the size-based rule, since a numbered heading is never the document `Title`. The
    // "no wide internal gap" guard excludes a running header/footer's "Chapter 4 ...
    // Page 12" pagination shape (chapter label, wide gutter, page number), which opens
    // with the same "Chapter N" prefix as a real numbered heading but is a two-column
    // band, not a title -- an ordinary heading is a short title-cased phrase with no
    // multi-space internal run.
    if starts_with_heading_number(&text)
        && line_count <= SHORT_BLOCK_MAX_LINES
        && !text.trim_end().ends_with(['.', '?', '!'])
        && !text.contains("   ")
    {
        return (RegionClass::SectionHeader, 0.55);
    }

    if body_font_size > 0.0
        && font_size >= body_font_size * SECTION_HEADER_FONT_FACTOR
        && line_count <= SHORT_BLOCK_MAX_LINES
    {
        if !*seen_title {
            *seen_title = true;
            return (RegionClass::Title, 0.65);
        }
        return (RegionClass::SectionHeader, 0.6);
    }

    // A heading set in bold at body size has no font-size cue at all, so it needs its
    // own signal below the size-based branch above. Guarded by
    // `page_predominantly_bold` (bold only means something when it contrasts with the
    // body), a line-count cap (same short-block rule as the size-based branch), a
    // size floor (excludes bold captions/footnotes, which are smaller than body text),
    // and a "doesn't end in sentence punctuation" check (headings aren't sentences).
    // Never sets `seen_title`: a body-sized bold block is always a subheading, never
    // the document title.
    if !page_predominantly_bold
        && block_is_bold(block)
        && line_count <= SHORT_BLOCK_MAX_LINES
        && font_size >= body_font_size
        && !text.trim_end().ends_with(['.', '?', '!'])
    {
        return (RegionClass::SectionHeader, 0.55);
    }

    // A display (block-level) formula: centered, narrow, vertically isolated, and dense
    // with math symbols/digits. Checked before the borderless-table fallback below,
    // since a short two-line formula (e.g. a stacked fraction) can otherwise present the
    // same aligned-whitespace shape `whitespace_column_corridors` looks for.
    if is_display_formula(block, page_blocks, page_width, &text) {
        return (RegionClass::Formula, 0.5);
    }

    // A borderless table whose columns live inside this block's own lines (aligned
    // whitespace runs, no ruling lines and no separate side-by-side blocks for
    // `detect_borderless_table_regions` to key off) -- see `whitespace_column_corridors`.
    // Checked last, like every other fallback here, so it only ever reclassifies a block
    // that would otherwise land in the `Text` default below.
    if line_count >= BORDERLESS_TABLE_MIN_LINES
        && band_of(block, page_height).is_none()
        && !whitespace_column_corridors(block).is_empty()
    {
        return (RegionClass::Table, 0.5);
    }

    (RegionClass::Text, 0.5)
}

/// Returns `true` if `name` (a PDF font's own name, e.g. `"Helvetica-Bold"` or
/// `"ABCDEF+TimesNewRomanPS-BoldMT"`) looks like a bold weight. This is the honest
/// ceiling for weight detection over this crate's extraction path: PDF exposes no
/// numeric font-weight value here, only the font's own name, so a case-insensitive
/// substring match against common weight keywords is what's available.
fn is_bold_font_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    ["bold", "black", "heavy"]
        .iter()
        .any(|marker| lower.contains(marker))
}

/// A font's own declared weight (`Char::font_weight`) at or above this counts as bold --
/// the standard OpenType/CSS "Bold" weight class. Preferred over [`is_bold_font_name`]'s
/// name-substring guess whenever the font program declares a *plausible* numeric weight
/// (see [`MIN_PLAUSIBLE_FONT_WEIGHT`]), per the `section_header_vs_bold` entry in
/// `docs/pitfall_registry.json`: a font that declares its weight numerically is a
/// stronger signal than a name-string guess.
const BOLD_FONT_WEIGHT_THRESHOLD: u32 = 700;

/// The lowest `Char::font_weight` value trusted as a real declared weight rather than a
/// PDFium sentinel for "unknown". OpenType's own `usWeightClass` is valid from 1-1000,
/// but real font programs cluster at 100 ("Thin") and up; PDFium has been observed to
/// report `Some(0)` for fonts it can't actually resolve a weight for (e.g. a real
/// `Lato-Bold` in the wild reports weight `0`, not `700` and not `None`) -- trusting that
/// literally would silently override [`is_bold_font_name`]'s correct, working guess with
/// a wrong "not bold" answer. `0` (and anything below this floor) falls back to the name
/// heuristic instead of being read as "declared regular weight".
const MIN_PLAUSIBLE_FONT_WEIGHT: u32 = 100;

/// `true` if `c` is bold: its own declared [`crate::Char::font_weight`] when the font
/// program provides a plausible one (see [`MIN_PLAUSIBLE_FONT_WEIGHT`]), falling back to
/// [`is_bold_font_name`]'s name-substring guess otherwise (many fonts never declare a
/// numeric weight at all, and some PDFium reports one that isn't trustworthy).
fn char_is_bold(c: &crate::Char) -> bool {
    match c.font_weight {
        Some(weight) if weight >= MIN_PLAUSIBLE_FONT_WEIGHT => weight >= BOLD_FONT_WEIGHT_THRESHOLD,
        _ => is_bold_font_name(&c.font_name),
    }
}

/// The fraction of `chars` that are bold (see [`char_is_bold`]), or `0.0` for an empty
/// iterator.
fn bold_char_fraction<'a>(chars: impl Iterator<Item = &'a crate::Char>) -> f32 {
    let mut total = 0usize;
    let mut bold = 0usize;
    for c in chars {
        total += 1;
        if char_is_bold(c) {
            bold += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        bold as f32 / total as f32
    }
}

/// Returns `true` if at least [`BOLD_CHAR_FRACTION`] of `block`'s characters carry a
/// bold font name.
fn block_is_bold(block: &Block) -> bool {
    bold_char_fraction(
        block
            .lines
            .iter()
            .flat_map(|l| &l.words)
            .flat_map(|w| &w.chars),
    ) >= BOLD_CHAR_FRACTION
}

/// Returns `true` if at least [`BOLD_CHAR_FRACTION`] of `page`'s characters carry a bold
/// font name -- the guard that keeps [`block_is_bold`] from firing on every block of a
/// page that's simply set in a bold font throughout (a whole-page style choice, not a
/// heading signal).
fn page_is_predominantly_bold(page: &crate::Page) -> bool {
    bold_char_fraction(
        page.blocks
            .iter()
            .flat_map(|b| &b.lines)
            .flat_map(|l| &l.words)
            .flat_map(|w| &w.chars),
    ) >= BOLD_CHAR_FRACTION
}

/// Vertical distance from `block` to the nearest other block on the page in the given
/// direction: `below == true` looks for the closest block whose top edge is at or below
/// `block`'s bottom edge (the header-candidate direction); `below == false` looks for the
/// closest block whose bottom edge is at or above `block`'s top edge (the
/// footer-candidate direction). Returns `f32::INFINITY` when nothing lies that way -- a
/// lone block on the page counts as fully detached. A vertically overlapping block
/// yields a gap of `0.0` (saturating, since overlapping bboxes can subtract negative).
fn gap_to_nearest_block(block: &Block, page_blocks: &[Block], below: bool) -> f32 {
    page_blocks
        .iter()
        .filter(|other| !std::ptr::eq(*other, block))
        .map(|other| {
            if below {
                (block.bbox.bottom - other.bbox.top).max(0.0)
            } else {
                (other.bbox.bottom - block.bbox.top).max(0.0)
            }
        })
        .filter(|gap| gap.is_finite())
        .fold(f32::INFINITY, f32::min)
}

/// Classifies which page-edge band `block` sits in, if any -- the same
/// [`HEADER_BAND_FRACTION`]/[`FOOTER_BAND_FRACTION`] geometry [`classify_block`] uses, but
/// without the shape/gap tests, so a block that fails those can still be considered for
/// [`repeated_running_bands`]'s cross-page repetition signal.
fn band_of(block: &Block, page_height: f32) -> Option<Band> {
    if block.bbox.top >= page_height * HEADER_BAND_FRACTION {
        Some(Band::Header)
    } else if block.bbox.bottom <= page_height * FOOTER_BAND_FRACTION {
        Some(Band::Footer)
    } else {
        None
    }
}

/// Lowercased, whitespace-collapsed, digit-run-masked form of `text`, so e.g. "Page 12"
/// and "Page 13" normalize to the same string and are recognized as the same running
/// header/footer despite the per-page page-number digits.
fn normalize_running_text(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut prev_was_digit = false;
    for word in text.split_whitespace() {
        if !normalized.is_empty() {
            normalized.push(' ');
        }
        for c in word.chars() {
            let is_digit = c.is_ascii_digit();
            if is_digit {
                if !prev_was_digit {
                    normalized.push('#');
                }
            } else {
                normalized.push(c.to_ascii_lowercase());
            }
            prev_was_digit = is_digit;
        }
    }
    normalized
}

/// Band texts (see [`Band`]) that recur at a stable page-relative vertical offset on at
/// least [`REPEATED_BAND_MIN_PAGES`] *consecutive* pages -- the roadmap's "repeated
/// content across N consecutive pages" signal for running headers/footers that are too
/// tall, or too close to the body, for the single-page geometric rule in
/// [`classify_block`] to catch on its own.
fn repeated_running_bands(document: &Document) -> std::collections::HashSet<(Band, String)> {
    let mut positions: std::collections::HashMap<(Band, String), Vec<(usize, f32)>> =
        std::collections::HashMap::new();

    for page in &document.pages {
        if page.height <= 0.0 {
            continue;
        }
        for block in &page.blocks {
            let Some(band) = band_of(block, page.height) else {
                continue;
            };
            let normalized = normalize_running_text(&block.text());
            if normalized.is_empty() {
                continue;
            }
            let rel_y = match band {
                Band::Header => (page.height - block.bbox.top) / page.height,
                Band::Footer => block.bbox.bottom / page.height,
            };
            positions
                .entry((band, normalized))
                .or_default()
                .push((page.index, rel_y));
        }
    }

    positions
        .into_iter()
        .filter(|(_, occurrences)| has_consecutive_run(occurrences))
        .map(|(key, _)| key)
        .collect()
}

/// Returns `true` if `occurrences` (page index, page-relative y) contains a run of at
/// least [`REPEATED_BAND_MIN_PAGES`] consecutive page indices all within
/// [`REPEATED_BAND_Y_TOLERANCE`] of the run's first y-position.
fn has_consecutive_run(occurrences: &[(usize, f32)]) -> bool {
    if occurrences.len() < REPEATED_BAND_MIN_PAGES {
        return false;
    }

    let mut sorted = occurrences.to_vec();
    sorted.sort_by_key(|(page_index, _)| *page_index);

    // `run_start` indexes the first element of the current consecutive-page run; every
    // element in the run is compared back to its y-position, not to its immediate
    // predecessor, so the tolerance can't drift across a long run one small step at a time.
    let mut run_start = 0;
    for i in 1..sorted.len() {
        let (prev_page, _) = sorted[i - 1];
        let (page_index, rel_y) = sorted[i];
        let (_, run_start_y) = sorted[run_start];

        let continues_run =
            page_index == prev_page + 1 && (rel_y - run_start_y).abs() <= REPEATED_BAND_Y_TOLERANCE;
        if !continues_run {
            run_start = i;
        }

        if i - run_start + 1 >= REPEATED_BAND_MIN_PAGES {
            return true;
        }
    }

    false
}

/// The median of a collection of font sizes, or `0.0` if empty. Shared by
/// [`body_median_font_size`], [`block_font_size`], and [`body_font_size_excluding`], all
/// of which differ only in which characters they collect sizes from.
fn median_font_size(mut sizes: Vec<f32>) -> f32 {
    if sizes.is_empty() {
        return 0.0;
    }

    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sizes[sizes.len() / 2]
}

/// The median font size across every character on the page — a cheap proxy for "body
/// text size" that [`classify_block`] compares individual blocks against to spot
/// oversized headings.
fn body_median_font_size(page: &crate::Page) -> f32 {
    median_font_size(
        page.blocks
            .iter()
            .flat_map(|b| &b.lines)
            .flat_map(|l| &l.words)
            .flat_map(|w| &w.chars)
            .map(|c| c.font_size)
            .collect(),
    )
}

/// The dominant font size within a single block (its own median), used to compare
/// against the page body's median.
fn block_font_size(block: &Block) -> f32 {
    median_font_size(
        block
            .lines
            .iter()
            .flat_map(|l| &l.words)
            .flat_map(|w| &w.chars)
            .map(|c| c.font_size)
            .collect(),
    )
}

/// The character-weighted median font size across every *other* block on the page,
/// excluding `block` itself.
///
/// [`body_median_font_size`] can't be reused for footnote detection: it's computed over
/// the *whole* page, so a long footnote block's own small-font characters can dominate
/// the median it's then compared against, making the block indistinguishable from "the
/// body." Excluding the candidate block from its own reference avoids that
/// self-domination and asks the right question -- "is this smaller than the rest of the
/// page" -- the same `std::ptr::eq`-based exclusion [`gap_to_nearest_block`] already uses
/// to skip comparing a block against itself.
fn body_font_size_excluding(page_blocks: &[Block], block: &Block) -> f32 {
    median_font_size(
        page_blocks
            .iter()
            .filter(|other| !std::ptr::eq(*other, block))
            .flat_map(|b| &b.lines)
            .flat_map(|l| &l.words)
            .flat_map(|w| &w.chars)
            .map(|c| c.font_size)
            .collect(),
    )
}

/// Returns `true` if `text` looks like a bulleted or ordered list item: starts with a
/// bullet glyph (`-`, `*`, `•`) or an ordered marker (`1.` / `1)`), each followed by
/// whitespace.
fn is_list_item(text: &str) -> bool {
    let trimmed = text.trim_start();

    if let Some(rest) = trimmed
        .strip_prefix('-')
        .or_else(|| trimmed.strip_prefix('*'))
        .or_else(|| trimmed.strip_prefix('\u{2022}'))
    {
        return rest.starts_with(char::is_whitespace);
    }

    let digits: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return false;
    }
    let rest = &trimmed[digits.len()..];
    matches!(rest.chars().next(), Some('.') | Some(')'))
        && rest.chars().nth(1).is_some_and(char::is_whitespace)
}

/// Returns `true` if `text` looks like a figure/table caption: starts with `Figure` or
/// `Table` followed by a number (case-insensitive).
/// Returns `true` if `block` has the geometric and textual signature of a **display**
/// (block-level) mathematical formula -- see the [module docs](self) for why this can't
/// cover *inline* formulas. Every one of the following must hold:
///
/// - **Centered**: the block's horizontal center falls within
///   [`FORMULA_CENTER_TOLERANCE_FRACTION`] of the page width from true center.
/// - **Narrow**: the block is no wider than [`FORMULA_MAX_WIDTH_FRACTION`] of the page --
///   this, not centering, is what separates a formula from an ordinary centered body
///   paragraph sitting right above/below it (a paragraph is also centered, but runs most
///   of the page's text width).
/// - **Vertically isolated**: at least [`FORMULA_MIN_ISOLATION_FACTOR`] font-sizes of
///   clearance to the nearest neighbour above *and* below (via
///   [`gap_to_nearest_block`]; a block with no neighbour on one side, e.g. the only block
///   on its page, is trivially isolated on that side).
/// - **Symbol-dense**: at least [`FORMULA_MIN_MATH_DENSITY`] of `text`'s non-whitespace
///   characters fall in the math-symbol/digit set below.
/// - **Not a sentence**: `text` doesn't end in `.`/`?`/`!`/`:` -- rules out an ordinary
///   sentence that happens to be short, centered, and narrow (e.g. a centered pull quote
///   or a lead-in line ending in a colon).
///
/// Deliberately *no* "low prose-word ratio" check: a hand-authored/ASCII-transcribed
/// formula (e.g. `"integral from 0 to infinity of ... dx = sqrt(pi) / 2"`) spells math
/// out in ordinary words, so its prose-word ratio is high even though it's genuinely a
/// formula. Symbol/digit density is the signal that actually discriminates here.
fn is_display_formula(block: &Block, page_blocks: &[Block], page_width: f32, text: &str) -> bool {
    if page_width <= 0.0 {
        return false;
    }

    let center_x = (block.bbox.left + block.bbox.right) / 2.0;
    let is_centered =
        (center_x - page_width / 2.0).abs() <= page_width * FORMULA_CENTER_TOLERANCE_FRACTION;
    if !is_centered {
        return false;
    }

    if block.bbox.width() > page_width * FORMULA_MAX_WIDTH_FRACTION {
        return false;
    }

    let font_size = block_font_size(block);
    if font_size > 0.0 {
        let min_gap = font_size * FORMULA_MIN_ISOLATION_FACTOR;
        let gap_below = vertical_gap_in_direction(block, page_blocks, /* below */ true);
        let gap_above = vertical_gap_in_direction(block, page_blocks, /* below */ false);
        if gap_below < min_gap || gap_above < min_gap {
            return false;
        }
    }

    let trimmed = text.trim_end();
    if trimmed.ends_with(['.', '?', '!', ':']) {
        return false;
    }

    let non_whitespace: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if non_whitespace.is_empty() {
        return false;
    }
    let math_count = non_whitespace
        .iter()
        .filter(|c| is_math_symbol_or_digit(**c))
        .count();
    let density = math_count as f32 / non_whitespace.len() as f32;

    density >= FORMULA_MIN_MATH_DENSITY
}

/// Returns the vertical gap between `block` and its nearest neighbour strictly in the
/// given direction (`below = true` looks for blocks below `block`, `below = false` looks
/// above), or `f32::INFINITY` if no block exists in that direction at all.
///
/// This differs from [`gap_to_nearest_block`] in exactly the case that matters for
/// isolation checks: that helper takes a `min` over *every* other block's signed gap
/// clamped to `0.0`, so a block with no neighbour on one side (e.g. the last block on a
/// page) reports a false "touching" gap of `0.0` from a block that's actually on the
/// *other* side, rather than "no neighbour here." Filtering to blocks genuinely on the
/// requested side first avoids that false positive.
fn vertical_gap_in_direction(block: &Block, page_blocks: &[Block], below: bool) -> f32 {
    page_blocks
        .iter()
        .filter(|other| !std::ptr::eq(*other, block))
        .filter_map(|other| {
            if below {
                (other.bbox.top <= block.bbox.bottom).then_some(block.bbox.bottom - other.bbox.top)
            } else {
                (other.bbox.bottom >= block.bbox.top).then_some(other.bbox.bottom - block.bbox.top)
            }
        })
        .filter(|gap| gap.is_finite())
        .fold(f32::INFINITY, f32::min)
}

/// Returns `true` if `c` is a digit or a symbol commonly found in mathematical notation:
/// ASCII operators/grouping (`= + - * / ^ ( ) [ ] { }`), the Greek block (formula
/// variables like `π`, `θ`, `Σ`), and the Unicode math-operator block (`U+2200`-`U+22FF`,
/// covering `∫ ∑ ∏ √` and friends). The seeded corpus cases are ASCII transcriptions
/// (`"sqrt(pi)"`, not `"√π"`), so only the ASCII/digit checks are exercised today, but
/// real extraction hands back actual Unicode math glyphs, so both ranges are checked.
fn is_math_symbol_or_digit(c: char) -> bool {
    c.is_ascii_digit()
        || matches!(
            c,
            '=' | '+' | '-' | '*' | '/' | '^' | '(' | ')' | '[' | ']' | '{' | '}'
        )
        || ('\u{0370}'..='\u{03FF}').contains(&c)
        || ('\u{2200}'..='\u{22FF}').contains(&c)
}

fn is_caption(text: &str) -> bool {
    let trimmed = text.trim_start();
    for prefix in ["Figure", "Table", "Fig."] {
        if let Some(rest) = strip_prefix_ignore_case(trimmed, prefix) {
            let rest = rest.trim_start();
            if rest.starts_with(|c: char| c.is_ascii_digit()) {
                return true;
            }
        }
    }
    false
}

/// Returns `true` if `text` opens with a heading-style numbering or label: a bare or
/// dotted multi-level number ("7 Variants...", "7.2 Simple..."), a roman numeral
/// followed by `.` ("III. Regulatory..."), or a `Chapter`/`Section`/`Part`/`Appendix`
/// label immediately followed by a number or single-letter identifier ("Chapter 4",
/// "Appendix A"). The label form requires that identifier -- "Section lead-in" or
/// "Chapter 4" running as ordinary body text (a header/footer band, say) starts with the
/// same keyword as a real "Section 5.1: ..." heading, so the keyword alone isn't
/// sufficient; it's the identifier after it that's the actual signal. See
/// [`strip_multilevel_number`] for why a single-digit-run-then-period/paren ("1.
/// Introduction") is deliberately excluded from the bare-number form: that's
/// `is_list_item`'s shape, and the two are indistinguishable from text alone.
fn starts_with_heading_number(text: &str) -> bool {
    let trimmed = text.trim_start();

    if let Some(rest) = strip_multilevel_number(trimmed) {
        if rest.starts_with(char::is_whitespace) || rest.starts_with(':') {
            return true;
        }
    }

    if let Some(rest) = strip_roman_numeral(trimmed) {
        if let Some(after_dot) = rest.strip_prefix('.') {
            if after_dot.starts_with(char::is_whitespace) {
                return true;
            }
        }
    }

    ["Chapter", "Section", "Part", "Appendix"]
        .iter()
        .any(|keyword| {
            strip_prefix_ignore_case(trimmed, keyword).is_some_and(|rest| {
                let rest = rest.trim_start();
                rest.starts_with(|c: char| c.is_ascii_digit())
                    || starts_with_lettered_identifier(rest)
            })
        })
}

/// `true` if `text` opens with a single uppercase ASCII letter used as an identifier
/// ("A" in "Appendix A", "Appendix A:") rather than the first letter of an ordinary word
/// ("Overview" in "Appendix Overview") -- i.e. the letter is immediately followed by the
/// end of the text, whitespace, `:`, or `.`, never by another letter.
fn starts_with_lettered_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_uppercase() {
        return false;
    }
    match chars.next() {
        None => true,
        Some(c) => matches!(c, ':' | '.' | ' ' | '\t'),
    }
}

/// Strips a bare or dotted multi-level number ("7", "7.2", "7.2.1") from the front of
/// `text`, returning what follows -- or `None` if `text` doesn't open with one. A
/// *single*-level number immediately followed by `.` or `)` ("1.", "1)") is excluded:
/// that exact shape belongs to [`is_list_item`], checked earlier in `classify_block`, and
/// a single-level numbered list item ("1. Buy milk") and a single-level numbered heading
/// ("1. Introduction") aren't distinguishable from text shape alone.
fn strip_multilevel_number(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let mut end = 0;
    let mut levels = 0u32;
    loop {
        let level_start = end;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end == level_start {
            break;
        }
        levels += 1;
        if bytes.get(end) == Some(&b'.') && bytes.get(end + 1).is_some_and(u8::is_ascii_digit) {
            end += 1; // consume the '.' and continue into the next level
        } else {
            break;
        }
    }
    if levels == 0 {
        return None;
    }
    if levels == 1 && matches!(bytes.get(end), Some(b'.') | Some(b')')) {
        return None;
    }
    Some(&text[end..])
}

/// Strips a short (at most 6 letters -- real outline numbering rarely runs past
/// "XX"/"XXV") run of uppercase roman-numeral letters from the front of `text`,
/// returning what follows, or `None` if `text` doesn't open with one.
fn strip_roman_numeral(text: &str) -> Option<&str> {
    let end = text
        .char_indices()
        .take_while(|(_, c)| matches!(c, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M'))
        .take(6)
        .last()
        .map(|(i, c)| i + c.len_utf8())?;
    Some(&text[end..])
}

/// Returns `true` if `text` opens with a recognized footnote marker: a short bare digit
/// run (optionally followed by `.` or `)`, then whitespace), a conventional footnote
/// symbol (`*`, `†`, `‡`, `§`, `¶`), or a superscript digit (`⁰`-`⁹`). Unlike
/// [`is_list_item`], bullet glyphs (`-`, `*` as a bullet, `•`) are deliberately *not*
/// markers here -- `*` only counts as a footnote marker when it isn't followed by
/// whitespace (which would instead read as a `-`/`*`/`•` bulleted list item), and `-`/`•`
/// are never footnote markers at all.
fn starts_with_footnote_marker(text: &str) -> bool {
    let trimmed = text.trim_start();

    if let Some(first) = trimmed.chars().next() {
        if matches!(first, '†' | '‡' | '§' | '¶') {
            return true;
        }
        if first == '*' && !trimmed[first.len_utf8()..].starts_with(char::is_whitespace) {
            return true;
        }
        if ('\u{2070}'..='\u{2079}').contains(&first) {
            return true;
        }
    }

    let digits: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() || digits.len() > 3 {
        return false;
    }
    let rest = &trimmed[digits.len()..];
    match rest.chars().next() {
        Some('.') | Some(')') => rest.chars().nth(1).is_some_and(char::is_whitespace),
        Some(c) => c.is_whitespace(),
        None => false,
    }
}

fn strip_prefix_ignore_case<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    // `text.get(..prefix.len())` (rather than `text.split_at`/direct slicing) returns
    // `None` both when `text` is shorter than `prefix` and when `prefix.len()` lands
    // inside a multi-byte character -- a real PDF's caption/heading text can carry
    // non-ASCII characters (e.g. "Ω" in a physics caption) before the prefix-length
    // byte offset, which `split_at` would otherwise panic on.
    let head = text.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| &text[prefix.len()..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Char, Line, Page, Word};

    fn char_at(bbox: BBox, font_size: f32) -> Char {
        Char {
            unicode: Some('x'),
            bbox,
            font_name: "Test".into(),
            font_size,
            ..Default::default()
        }
    }

    fn char_at_named(bbox: BBox, font_size: f32, font_name: &str) -> Char {
        Char {
            unicode: Some('x'),
            bbox,
            font_name: font_name.into(),
            font_size,
            ..Default::default()
        }
    }

    fn word(text: &str, bbox: BBox, font_size: f32) -> Word {
        // Repeat the char to give the block's/page's median enough weight to be
        // meaningful in tests -- one-char blocks would make every "body" text block's
        // font size tie with a lone title char.
        let chars =
            std::iter::repeat_n(char_at(bbox, font_size), text.chars().count().max(1)).collect();
        Word {
            text: text.into(),
            bbox,
            chars,
        }
    }

    fn word_named(text: &str, bbox: BBox, font_size: f32, font_name: &str) -> Word {
        let chars = std::iter::repeat_n(
            char_at_named(bbox, font_size, font_name),
            text.chars().count().max(1),
        )
        .collect();
        Word {
            text: text.into(),
            bbox,
            chars,
        }
    }

    fn line(text: &str, bbox: BBox, font_size: f32) -> Line {
        Line {
            text: text.into(),
            bbox,
            words: vec![word(text, bbox, font_size)],
        }
    }

    fn line_named(text: &str, bbox: BBox, font_size: f32, font_name: &str) -> Line {
        Line {
            text: text.into(),
            bbox,
            words: vec![word_named(text, bbox, font_size, font_name)],
        }
    }

    fn line_weighted(text: &str, bbox: BBox, font_size: f32, font_weight: u32) -> Line {
        let chars = std::iter::repeat_n(
            Char {
                font_weight: Some(font_weight),
                ..char_at(bbox, font_size)
            },
            text.chars().count().max(1),
        )
        .collect();
        Line {
            text: text.into(),
            bbox,
            words: vec![Word {
                text: text.into(),
                bbox,
                chars,
            }],
        }
    }

    fn block(lines: Vec<Line>) -> Block {
        let bbox = lines
            .iter()
            .map(|l| l.bbox)
            .reduce(|a, b| a.union(&b))
            .unwrap();
        Block { bbox, lines }
    }

    fn page(blocks: Vec<Block>) -> Page {
        Page {
            index: 0,
            width: 612.0,
            height: 792.0,
            blocks,
            ..Default::default()
        }
    }

    /// Like [`page`], but with an explicit index/height for multi-page documents and
    /// fixture-matching geometry (the corpus's `header_footer` cases use an 800pt-tall
    /// page, not this module's default 792pt).
    fn page_at(index: usize, height: f32, blocks: Vec<Block>) -> Page {
        Page {
            index,
            width: 600.0,
            height,
            blocks,
            ..Default::default()
        }
    }

    #[test]
    fn large_font_top_block_is_title() {
        let title_bbox = BBox {
            left: 50.0,
            bottom: 700.0,
            right: 300.0,
            top: 720.0,
        };
        let body_bbox = BBox {
            left: 50.0,
            bottom: 600.0,
            right: 300.0,
            top: 615.0,
        };

        let title = block(vec![line("Big Heading", title_bbox, 24.0)]);
        let body = block(vec![line("Body text here", body_bbox, 10.0)]);

        let doc = Document {
            pages: vec![page(vec![title, body])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Title);
        assert_eq!(regions[1].class, RegionClass::Text);
    }

    #[test]
    fn bullet_line_is_list_item() {
        let bbox = BBox {
            left: 50.0,
            bottom: 400.0,
            right: 300.0,
            top: 415.0,
        };
        let b = block(vec![line("- first point", bbox, 10.0)]);
        let doc = Document {
            pages: vec![page(vec![b])],
        };

        let regions = classify_regions(&doc);
        assert_eq!(regions[0].class, RegionClass::ListItem);
    }

    #[test]
    fn top_band_short_block_is_page_header() {
        let bbox = BBox {
            left: 50.0,
            bottom: 775.0,
            right: 300.0,
            top: 785.0,
        };
        let b = block(vec![line("Running Header", bbox, 9.0)]);
        let doc = Document {
            pages: vec![page(vec![b])],
        };

        let regions = classify_regions(&doc);
        assert_eq!(regions[0].class, RegionClass::PageHeader);
    }

    #[test]
    fn multi_line_running_header_is_page_header() {
        // Mirrors fixtures/header_footer/running_header_exceeds_line_limit.json: a
        // 3-line running header, previously rejected outright by the old
        // HEADER_FOOTER_MAX_LINES(2) cutoff.
        let header = block(vec![
            line(
                "Company Name",
                BBox {
                    left: 40.0,
                    bottom: 760.0,
                    right: 300.0,
                    top: 775.0,
                },
                10.0,
            ),
            line(
                "Confidential Draft",
                BBox {
                    left: 40.0,
                    bottom: 745.0,
                    right: 300.0,
                    top: 760.0,
                },
                10.0,
            ),
            line(
                "Do Not Distribute",
                BBox {
                    left: 40.0,
                    bottom: 730.0,
                    right: 300.0,
                    top: 745.0,
                },
                10.0,
            ),
        ]);
        let body = block(vec![line(
            "Body text of the document that continues on this page.",
            BBox {
                left: 40.0,
                bottom: 300.0,
                right: 560.0,
                top: 400.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![header, body])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::PageHeader);
    }

    #[test]
    fn multi_line_running_footer_is_page_footer() {
        // Mirrors fixtures/header_footer/running_footer_exceeds_line_limit.json.
        let body = block(vec![line(
            "Body text of the document continues here.",
            BBox {
                left: 40.0,
                bottom: 300.0,
                right: 560.0,
                top: 400.0,
            },
            10.0,
        )]);
        let footer = block(vec![
            line(
                "Page 12",
                BBox {
                    left: 40.0,
                    bottom: 50.0,
                    right: 300.0,
                    top: 60.0,
                },
                8.0,
            ),
            line(
                "Section 3: Results",
                BBox {
                    left: 40.0,
                    bottom: 35.0,
                    right: 300.0,
                    top: 50.0,
                },
                8.0,
            ),
            line(
                "www.example.com/docs",
                BBox {
                    left: 40.0,
                    bottom: 20.0,
                    right: 300.0,
                    top: 35.0,
                },
                8.0,
            ),
        ]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![body, footer])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[1].class, RegionClass::PageFooter);
    }

    #[test]
    fn tall_top_of_page_block_is_not_page_header() {
        // A block that starts in the header band but spans most of the page is body
        // text that merely reaches the top of the page, not a running header.
        let bbox = BBox {
            left: 40.0,
            bottom: 100.0,
            right: 560.0,
            top: 785.0,
        };
        let b = block(vec![line(
            "A very long article that starts near the top.",
            bbox,
            10.0,
        )]);
        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![b])],
        };

        let regions = classify_regions(&doc);
        assert_eq!(regions[0].class, RegionClass::Text);
    }

    #[test]
    fn header_adjacent_to_body_is_not_page_header() {
        // Thin and in-band, but only 5pt above the body block below it (well under the
        // 16pt/2%-of-800 gap threshold) -- this looks like the body's own first line,
        // not a detached running header.
        let header = block(vec![line(
            "Section lead-in",
            BBox {
                left: 40.0,
                bottom: 760.0,
                right: 300.0,
                top: 775.0,
            },
            10.0,
        )]);
        let body = block(vec![line(
            "Body text starting immediately below the heading above.",
            BBox {
                left: 40.0,
                bottom: 300.0,
                right: 560.0,
                top: 755.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![header, body])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Text);
    }

    #[test]
    fn repeated_band_text_across_pages_is_page_header() {
        // Same shape as header_adjacent_to_body_is_not_page_header (geometrically
        // rejected on every single page), but the header text recurs at the same
        // page-relative position across 3 consecutive pages, varying only by page
        // number -- the roadmap's repeated-content signal should catch it where the
        // single-page geometric rule can't.
        fn running_header_page(index: usize, page_num: u32, body_text: &str) -> Page {
            let header = block(vec![line(
                &format!("Chapter 4: Results - Page {page_num}"),
                BBox {
                    left: 40.0,
                    bottom: 760.0,
                    right: 300.0,
                    top: 775.0,
                },
                10.0,
            )]);
            let body = block(vec![line(
                body_text,
                BBox {
                    left: 40.0,
                    bottom: 300.0,
                    right: 560.0,
                    top: 755.0,
                },
                10.0,
            )]);
            page_at(index, 800.0, vec![header, body])
        }

        // Body text varies by more than a digit (unlike the header) so it doesn't
        // accidentally normalize to a matching repeated key of its own.
        let doc = Document {
            pages: vec![
                running_header_page(0, 1, "Revenue grew across every operating region."),
                running_header_page(1, 2, "Currency headwinds offset seasonal demand."),
                running_header_page(2, 3, "New product launches are planned overseas."),
            ],
        };
        let regions = classify_regions(&doc);

        // Two regions per page, page-major order: header, body, header, body, ...
        for page in 0..3 {
            assert_eq!(
                regions[page * 2].class,
                RegionClass::PageHeader,
                "page {page} header"
            );
            assert_eq!(
                regions[page * 2 + 1].class,
                RegionClass::Text,
                "page {page} body"
            );
        }
    }

    #[test]
    fn unrepeated_band_text_on_multipage_document_stays_text() {
        // Same geometry as the repeated case (including the adjacent body block, so
        // the single-page geometric rule can't accept it on its own) but the band text
        // differs per page, not just by a digit -- repetition must not fire on every
        // band block in a multi-page document.
        fn distinct_header_page(index: usize, header_text: &str, body_text: &str) -> Page {
            let header = block(vec![line(
                header_text,
                BBox {
                    left: 40.0,
                    bottom: 760.0,
                    right: 300.0,
                    top: 775.0,
                },
                10.0,
            )]);
            let body = block(vec![line(
                body_text,
                BBox {
                    left: 40.0,
                    bottom: 300.0,
                    right: 560.0,
                    top: 755.0,
                },
                10.0,
            )]);
            page_at(index, 800.0, vec![header, body])
        }

        let doc = Document {
            pages: vec![
                distinct_header_page(0, "Introduction", "Body content for page one."),
                distinct_header_page(1, "Methodology", "Body content for page two."),
                distinct_header_page(2, "Results", "Body content for page three."),
            ],
        };
        let regions = classify_regions(&doc);

        for region in &regions {
            assert_eq!(region.class, RegionClass::Text);
        }
    }

    #[test]
    fn small_marked_block_near_page_bottom_is_footnote() {
        // Mirrors fixtures/footnote/footnote_marker_not_classified.json.
        let body = block(vec![line(
            "This references a footnote marker superscript.",
            BBox {
                left: 40.0,
                bottom: 300.0,
                right: 400.0,
                top: 320.0,
            },
            10.0,
        )]);
        let footnote = block(vec![line(
            "1 This is a footnote clarifying the claim above.",
            BBox {
                left: 40.0,
                bottom: 90.0,
                right: 400.0,
                top: 105.0,
            },
            8.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![body, footnote])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[1].class, RegionClass::Footnote);
    }

    #[test]
    fn footnote_just_above_footer_band_is_footnote() {
        // Mirrors fixtures/footnote/footnote_adjacent_to_footer_band.json: the block
        // sits above the 8%-of-800 = 64pt footer band (bottom = 70), so it's ineligible
        // for the PageFooter geometric rule and must be caught by the footnote rule
        // instead.
        let body = block(vec![line(
            "Body text referencing note two.",
            BBox {
                left: 40.0,
                bottom: 200.0,
                right: 400.0,
                top: 220.0,
            },
            10.0,
        )]);
        let footnote = block(vec![line(
            "2 Another footnote near the bottom margin explaining a term.",
            BBox {
                left: 40.0,
                bottom: 70.0,
                right: 400.0,
                top: 85.0,
            },
            8.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![body, footnote])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[1].class, RegionClass::Footnote);
    }

    #[test]
    fn body_font_block_near_page_bottom_is_not_footnote() {
        // Same geometry and marker as small_marked_block_near_page_bottom_is_footnote,
        // but body-sized font -- the font-ratio gate must reject it.
        let body = block(vec![line(
            "This references a footnote marker superscript.",
            BBox {
                left: 40.0,
                bottom: 300.0,
                right: 400.0,
                top: 320.0,
            },
            10.0,
        )]);
        let not_footnote = block(vec![line(
            "1 This looks like a footnote but is body-sized text.",
            BBox {
                left: 40.0,
                bottom: 90.0,
                right: 400.0,
                top: 110.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![body, not_footnote])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[1].class, RegionClass::Text);
    }

    #[test]
    fn unmarked_small_block_near_page_bottom_is_not_footnote() {
        // Same geometry and font as the footnote fixtures, but no leading marker -- the
        // marker gate must reject it (this is just small text at the bottom of the page,
        // not a footnote).
        let body = block(vec![line(
            "This references a footnote marker superscript.",
            BBox {
                left: 40.0,
                bottom: 300.0,
                right: 400.0,
                top: 320.0,
            },
            10.0,
        )]);
        let not_footnote = block(vec![line(
            "Small print near the bottom margin with no marker at all.",
            BBox {
                left: 40.0,
                bottom: 90.0,
                right: 400.0,
                top: 105.0,
            },
            8.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![body, not_footnote])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[1].class, RegionClass::Text);
    }

    #[test]
    fn bullet_list_near_page_bottom_stays_list_item() {
        // A bulleted list item can legitimately sit low on a small-font page -- `-`/`•`
        // must never be treated as footnote markers.
        let body = block(vec![line(
            "This references a footnote marker superscript.",
            BBox {
                left: 40.0,
                bottom: 300.0,
                right: 400.0,
                top: 320.0,
            },
            10.0,
        )]);
        let bullet = block(vec![line(
            "- a small-print bullet point near the bottom margin",
            BBox {
                left: 40.0,
                bottom: 90.0,
                right: 400.0,
                top: 105.0,
            },
            8.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![body, bullet])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[1].class, RegionClass::ListItem);
    }

    #[test]
    fn footer_band_block_outranks_footnote() {
        // A marked, small-font block inside the footer band (bottom = 20, well under the
        // 8%-of-800 = 64pt footer band) must still resolve to PageFooter -- the footer
        // rule is checked first, so a running footer that happens to look like a
        // footnote is never reclassified out from under it.
        let body = block(vec![line(
            "Body text of the document continues here.",
            BBox {
                left: 40.0,
                bottom: 300.0,
                right: 560.0,
                top: 400.0,
            },
            10.0,
        )]);
        let footer = block(vec![line(
            "1 Legal disclaimer text repeated on every page.",
            BBox {
                left: 40.0,
                bottom: 20.0,
                right: 300.0,
                top: 35.0,
            },
            8.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![body, footer])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[1].class, RegionClass::PageFooter);
    }

    #[test]
    fn bold_body_sized_heading_is_section_header() {
        // Same font size as the body -- the only signal here is the bold font name.
        let heading = block(vec![line_named(
            "Results and Discussion",
            BBox {
                left: 40.0,
                bottom: 600.0,
                right: 560.0,
                top: 615.0,
            },
            10.0,
            "Helvetica-Bold",
        )]);
        let body = block(vec![line(
            "This section presents the results of the experiment in detail.",
            BBox {
                left: 40.0,
                bottom: 500.0,
                right: 560.0,
                top: 595.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![heading, body])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::SectionHeader);
    }

    #[test]
    fn bold_heading_on_all_bold_page_is_not_promoted() {
        // When the whole page is set in bold, boldness carries no heading signal --
        // `page_is_predominantly_bold` must switch the branch off.
        let heading = block(vec![line_named(
            "Results and Discussion",
            BBox {
                left: 40.0,
                bottom: 600.0,
                right: 560.0,
                top: 615.0,
            },
            10.0,
            "Helvetica-Bold",
        )]);
        let body = block(vec![line_named(
            "This section presents the results of the experiment in detail.",
            BBox {
                left: 40.0,
                bottom: 500.0,
                right: 560.0,
                top: 595.0,
            },
            10.0,
            "Helvetica-Bold",
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![heading, body])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Text);
    }

    #[test]
    fn font_weight_outranks_font_name_for_boldness() {
        // A font that never puts "bold"/"black"/"heavy" in its own name (so
        // `is_bold_font_name` would say no) but *does* declare a numeric weight of 700+
        // must still be treated as bold -- `Char::font_weight` outranks the name guess
        // whenever the font program provides it (see `char_is_bold`).
        let heading = block(vec![line_weighted(
            "Results and Discussion",
            BBox {
                left: 40.0,
                bottom: 600.0,
                right: 560.0,
                top: 615.0,
            },
            10.0,
            700,
        )]);
        let body = block(vec![line(
            "This section presents the results of the experiment in detail.",
            BBox {
                left: 40.0,
                bottom: 500.0,
                right: 560.0,
                top: 595.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![heading, body])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::SectionHeader);
    }

    #[test]
    fn declared_regular_weight_overrides_a_misleadingly_named_font() {
        // The inverse: a font *named* "...Bold..." but whose program declares a regular
        // (400) weight should not be treated as bold once a numeric weight is present --
        // the declared weight is trusted over the name.
        let text = block(vec![line_weighted(
            "A font whose name lies about its own weight.",
            BBox {
                left: 40.0,
                bottom: 600.0,
                right: 560.0,
                top: 615.0,
            },
            10.0,
            400,
        )]);
        assert!(!char_is_bold(&text.lines[0].words[0].chars[0]));
    }

    #[test]
    fn zero_font_weight_falls_back_to_font_name() {
        // Real PDFium has been observed to report `font_weight: Some(0)` for a font it
        // can't actually resolve a weight for -- a `Lato-Bold` character reporting
        // weight 0, not 700 and not `None`. Trusting that literally would read a
        // genuinely bold, correctly-named font as "not bold" and silently override
        // `is_bold_font_name`'s correct answer. `0` must fall back to the name guess.
        let bold_named_but_zero_weight = Char {
            font_name: "Lato-Bold".into(),
            font_weight: Some(0),
            ..char_at(BBox::ZERO, 12.0)
        };
        assert!(char_is_bold(&bold_named_but_zero_weight));
    }

    #[test]
    fn bare_number_heading_is_section_header() {
        // No font-size or bold cue at all -- the numbering itself ("7 Variants...") is
        // the only signal, same shape as the real corpus docs this rule was added for.
        let heading = block(vec![line(
            "7 Variants of SJ Observer Models",
            BBox {
                left: 40.0,
                bottom: 600.0,
                right: 560.0,
                top: 615.0,
            },
            10.0,
        )]);
        let body = block(vec![line(
            "Body text discussing the variants in detail.",
            BBox {
                left: 40.0,
                bottom: 500.0,
                right: 560.0,
                top: 595.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![heading, body])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::SectionHeader);
    }

    #[test]
    fn roman_numeral_heading_is_section_header() {
        let heading = block(vec![line(
            "III. Regulatory Cholesterol",
            BBox {
                left: 40.0,
                bottom: 600.0,
                right: 560.0,
                top: 615.0,
            },
            10.0,
        )]);
        let body = block(vec![line(
            "Body text about regulation follows here.",
            BBox {
                left: 40.0,
                bottom: 500.0,
                right: 560.0,
                top: 595.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![heading, body])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::SectionHeader);
    }

    #[test]
    fn appendix_letter_heading_is_section_header() {
        let heading = block(vec![line(
            "Appendix A: Supplementary Tables",
            BBox {
                left: 40.0,
                bottom: 600.0,
                right: 560.0,
                top: 615.0,
            },
            10.0,
        )]);
        let body = block(vec![line(
            "Additional tables are provided below.",
            BBox {
                left: 40.0,
                bottom: 500.0,
                right: 560.0,
                top: 595.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![heading, body])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::SectionHeader);
    }

    #[test]
    fn plain_sentence_starting_with_a_heading_keyword_is_not_promoted() {
        // "Section lead-in" opens with the same keyword a real "Section 5.1: ..."
        // heading does, but isn't followed by a number or lettered identifier -- must
        // stay Text, not get promoted just because it starts with "Section".
        let text = block(vec![line(
            "Section lead-in text continues normally from here on.",
            BBox {
                left: 40.0,
                bottom: 600.0,
                right: 560.0,
                top: 615.0,
            },
            10.0,
        )]);
        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![text])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Text);
    }

    #[test]
    fn single_document_title_survives_across_pages() {
        // `seen_title` is threaded across the whole document now, not reset per page --
        // a second oversized block on page 2 must fall back to `SectionHeader`, not
        // become a second `Title`.
        let title = block(vec![line(
            "The Document Title",
            BBox {
                left: 40.0,
                bottom: 700.0,
                right: 560.0,
                top: 730.0,
            },
            20.0,
        )]);
        let page1_body = block(vec![line(
            "Page one body text at ordinary size.",
            BBox {
                left: 40.0,
                bottom: 400.0,
                right: 560.0,
                top: 416.0,
            },
            10.0,
        )]);
        let page2_heading = block(vec![line(
            "A Second Oversized Block",
            BBox {
                left: 40.0,
                bottom: 700.0,
                right: 560.0,
                top: 730.0,
            },
            20.0,
        )]);
        let page2_body = block(vec![line(
            "Page two body text at ordinary size.",
            BBox {
                left: 40.0,
                bottom: 400.0,
                right: 560.0,
                top: 416.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![
                page_at(0, 800.0, vec![title, page1_body]),
                page_at(1, 800.0, vec![page2_heading, page2_body]),
            ],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Title);
        assert_eq!(regions[2].class, RegionClass::SectionHeader);
    }

    #[test]
    fn bold_sentence_is_not_promoted() {
        // Headings aren't sentences -- a bold block ending in sentence punctuation
        // (e.g. an emphasized run-in sentence) should stay Text.
        let sentence = block(vec![line_named(
            "This is an important bold sentence.",
            BBox {
                left: 40.0,
                bottom: 600.0,
                right: 560.0,
                top: 615.0,
            },
            10.0,
            "Helvetica-Bold",
        )]);
        let body = block(vec![line(
            "This section presents the results of the experiment in detail.",
            BBox {
                left: 40.0,
                bottom: 500.0,
                right: 560.0,
                top: 595.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![sentence, body])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Text);
    }

    fn h_line(y: f32, left: f32, right: f32) -> crate::Graphic {
        crate::Graphic {
            kind: crate::GraphicKind::Stroke,
            bbox: BBox {
                left,
                bottom: y,
                right,
                top: y,
            },
            z: 0,
            stroke_width: 1.0,
            is_stroked: true,
            is_filled: false,
        }
    }

    fn v_line(x: f32, bottom: f32, top: f32) -> crate::Graphic {
        crate::Graphic {
            kind: crate::GraphicKind::Stroke,
            bbox: BBox {
                left: x,
                bottom,
                right: x,
                top,
            },
            z: 0,
            stroke_width: 1.0,
            is_stroked: true,
            is_filled: false,
        }
    }

    #[test]
    fn block_inside_a_detected_table_is_not_separately_classified() {
        let cell_bbox = BBox {
            left: 45.0,
            bottom: 65.0,
            right: 115.0,
            top: 95.0,
        };
        let cell = block(vec![line("Ada", cell_bbox, 10.0)]);

        let mut p = page(vec![cell]);
        p.graphics = vec![
            h_line(100.0, 40.0, 200.0),
            h_line(60.0, 40.0, 200.0),
            v_line(40.0, 60.0, 100.0),
            v_line(120.0, 60.0, 100.0),
            v_line(200.0, 60.0, 100.0),
        ];
        let doc = Document { pages: vec![p] };

        let regions = classify_regions(&doc);

        // Exactly one region: the table itself. The cell's own block never gets a
        // separate Text/whatever region -- it's represented by the table region only.
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].class, RegionClass::Table);
    }

    #[test]
    fn whitespace_aligned_block_is_table() {
        // Same shape as the borderless_table corpus fixtures: one block, several lines,
        // each line's columns separated by a much wider gutter than a single word-space.
        let b = block(vec![
            line(
                "Name        Score",
                BBox {
                    left: 40.0,
                    bottom: 500.0,
                    right: 400.0,
                    top: 515.0,
                },
                10.0,
            ),
            line(
                "Alice          92",
                BBox {
                    left: 40.0,
                    bottom: 485.0,
                    right: 400.0,
                    top: 500.0,
                },
                10.0,
            ),
            line(
                "Bob            77",
                BBox {
                    left: 40.0,
                    bottom: 470.0,
                    right: 400.0,
                    top: 485.0,
                },
                10.0,
            ),
        ]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![b])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Table);
    }

    #[test]
    fn ordinary_wrapped_prose_stays_text() {
        // Ragged-right word-wrapped prose: no shared vertical corridor across all three
        // lines, since each line's words fall at different x-positions.
        let b = block(vec![
            line(
                "The quick brown fox jumps over the lazy dog and",
                BBox {
                    left: 40.0,
                    bottom: 500.0,
                    right: 400.0,
                    top: 515.0,
                },
                10.0,
            ),
            line(
                "then keeps running until it reaches the far end",
                BBox {
                    left: 40.0,
                    bottom: 485.0,
                    right: 400.0,
                    top: 500.0,
                },
                10.0,
            ),
            line(
                "of the field before stopping to rest completely.",
                BBox {
                    left: 40.0,
                    bottom: 470.0,
                    right: 400.0,
                    top: 485.0,
                },
                10.0,
            ),
        ]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![b])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Text);
    }

    #[test]
    fn single_whitespace_padded_line_stays_text() {
        // Same column shape as whitespace_aligned_block_is_table's header line alone --
        // but with no second line to corroborate a corridor against, BORDERLESS_TABLE_MIN_LINES
        // should keep this Text.
        let b = block(vec![line(
            "Name        Score",
            BBox {
                left: 40.0,
                bottom: 500.0,
                right: 400.0,
                top: 515.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![b])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Text);
    }

    #[test]
    fn header_band_aligned_lines_are_not_promoted_to_table() {
        // Two whitespace-aligned lines sitting in the header band with only a small gap
        // to the body below -- too close for the earlier PageHeader gap check to fire,
        // so this reaches the new whitespace-corridor check. Left/right running
        // header/footer pairs are exactly the shape
        // `detect_borderless_table_regions`'s own doc comment calls out as a false
        // positive to guard against via `band_of`; the whitespace-corridor path needs
        // the same guard, or this would wrongly turn into a Table.
        let b = block(vec![
            line(
                "Chapter 4                              Page 12",
                BBox {
                    left: 40.0,
                    bottom: 760.0,
                    right: 560.0,
                    top: 775.0,
                },
                10.0,
            ),
            line(
                "Results Section                        Draft",
                BBox {
                    left: 40.0,
                    bottom: 744.0,
                    right: 560.0,
                    top: 759.0,
                },
                10.0,
            ),
        ]);
        let body = block(vec![line(
            "Body text starting immediately below the heading above.",
            BBox {
                left: 40.0,
                bottom: 300.0,
                right: 560.0,
                top: 743.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page_at(0, 800.0, vec![b, body])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Text);
    }

    #[test]
    fn image_graphic_is_classified_as_picture_region() {
        let mut p = page(vec![]);
        p.graphics = vec![crate::Graphic {
            kind: crate::GraphicKind::Image,
            bbox: BBox {
                left: 40.0,
                bottom: 40.0,
                right: 200.0,
                top: 200.0,
            },
            z: 0,
            stroke_width: 0.0,
            is_stroked: false,
            is_filled: true,
        }];
        let doc = Document { pages: vec![p] };

        let regions = classify_regions(&doc);

        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].class, RegionClass::Picture);
    }

    #[test]
    fn centered_isolated_symbol_dense_block_is_formula() {
        // Centered, narrow (112pt of a 612pt page), the only block on its page (so
        // trivially isolated), and dense with math operators/digits.
        let formula = block(vec![line(
            "x + y = z^2",
            BBox {
                left: 250.0,
                bottom: 400.0,
                right: 362.0,
                top: 415.0,
            },
            12.0,
        )]);

        let doc = Document {
            pages: vec![page(vec![formula])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Formula);
    }

    #[test]
    fn centered_wide_paragraph_is_not_formula() {
        // Centered like a formula would be, but 520pt wide on a 612pt page --
        // `FORMULA_MAX_WIDTH_FRACTION` is what has to reject this, since centering
        // alone doesn't discriminate a formula from a centered body paragraph.
        let para = block(vec![line(
            "This sentence is centered but far too wide to read as a formula",
            BBox {
                left: 46.0,
                bottom: 400.0,
                right: 566.0,
                top: 415.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page(vec![para])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Text);
    }

    #[test]
    fn centered_narrow_line_ending_in_colon_is_not_formula() {
        // Centered and narrow enough to pass the geometric tests, but reads as a
        // sentence (a lead-in line), not a formula.
        let intro = block(vec![line(
            "The result follows:",
            BBox {
                left: 200.0,
                bottom: 400.0,
                right: 412.0,
                top: 415.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page(vec![intro])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Text);
    }

    #[test]
    fn centered_large_font_block_stays_title_not_formula() {
        // Also centered and narrow enough to pass the formula geometry tests, but the
        // oversized-font title rule runs first in `classify_block` and must win the tie.
        let title = block(vec![line(
            "System Overview",
            BBox {
                left: 200.0,
                bottom: 600.0,
                right: 412.0,
                top: 630.0,
            },
            24.0,
        )]);
        let body = block(vec![line(
            "This is ordinary body text describing the system in more detail.",
            BBox {
                left: 46.0,
                bottom: 400.0,
                right: 566.0,
                top: 415.0,
            },
            10.0,
        )]);

        let doc = Document {
            pages: vec![page(vec![title, body])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Title);
    }

    #[test]
    fn whitespace_delimited_table_stays_table_not_formula() {
        // Guards against the corridor collision the formula rule is deliberately
        // ordered around: a genuine borderless table's aligned columns can otherwise
        // look like the same shared-whitespace-corridor shape a short stacked formula
        // has. This block is also off-center (left-anchored, not centered on the page),
        // which independently keeps it out of `is_display_formula`.
        let table = block(vec![
            line(
                "Name        Score",
                BBox {
                    left: 40.0,
                    bottom: 500.0,
                    right: 400.0,
                    top: 515.0,
                },
                10.0,
            ),
            line(
                "Alice          92",
                BBox {
                    left: 40.0,
                    bottom: 485.0,
                    right: 400.0,
                    top: 500.0,
                },
                10.0,
            ),
            line(
                "Bob            77",
                BBox {
                    left: 40.0,
                    bottom: 470.0,
                    right: 400.0,
                    top: 485.0,
                },
                10.0,
            ),
        ]);

        let doc = Document {
            pages: vec![page(vec![table])],
        };
        let regions = classify_regions(&doc);

        assert_eq!(regions[0].class, RegionClass::Table);
    }

    #[test]
    fn strip_prefix_ignore_case_does_not_panic_on_multibyte_boundary() {
        // Regression test: `strip_prefix_ignore_case` used to `str::split_at` at
        // `prefix.len()` unconditionally, which panics if that byte offset lands
        // inside a multi-byte character. "AΩΩ text" puts the second (2-byte) Ω at
        // byte offset 3..5, so slicing at byte 4 -- "Fig."'s length -- lands mid-glyph,
        // reproducing the exact panic a real corpus PDF hit (a physics caption
        // opening with "Ω") in `pdfspatial-core`'s opendataloader-bench run.
        assert_eq!(strip_prefix_ignore_case("AΩΩ text", "Fig."), None);
        assert_eq!(strip_prefix_ignore_case("AΩΩ text", "Table"), None);
        assert_eq!(strip_prefix_ignore_case("AΩΩ text", "Figure"), None);

        // Still strips a genuine ASCII-prefixed match.
        assert_eq!(
            strip_prefix_ignore_case("FIG. 3: caption text", "Fig."),
            Some(" 3: caption text")
        );
    }

    #[test]
    fn is_caption_does_not_panic_on_multibyte_prefix_candidate() {
        // A block that isn't actually a "Figure"/"Table"/"Fig." caption, but opens
        // with a multi-byte character positioned to misalign a byte-length prefix
        // check, must be classified rather than panicking.
        assert!(!is_caption("AΩΩ text that just happens to start this way."));
    }

    fn cell(text: &str, bbox: BBox) -> Block {
        block(vec![line(text, bbox, 10.0)])
    }

    #[test]
    fn stacked_compatible_rows_merge_into_one_borderless_table() {
        // Three row bands, each two side-by-side cells, aligned column starts and equal
        // cell counts -- must merge into ONE table region spanning all three rows, not
        // three separate single-row regions (the defect that made every borderless
        // table degenerate into N header-only 1-row GFM tables).
        let row = |y: f32, left_text: &str, right_text: &str| {
            vec![
                cell(
                    left_text,
                    BBox {
                        left: 72.0,
                        bottom: y,
                        right: 165.0,
                        top: y + 15.0,
                    },
                ),
                cell(
                    right_text,
                    BBox {
                        left: 340.0,
                        bottom: y,
                        right: 351.0,
                        top: y + 15.0,
                    },
                ),
            ]
        };
        let mut all = Vec::new();
        all.extend(row(690.0, "Item", "42"));
        all.extend(row(660.0, "Widget", "17"));
        all.extend(row(630.0, "Gadget", "9"));
        let refs: Vec<&Block> = all.iter().collect();

        let regions = detect_borderless_table_regions(&refs, 792.0, 0);
        assert_eq!(
            regions.len(),
            1,
            "expected one merged table, got {regions:?}"
        );
    }

    #[test]
    fn incompatible_adjacent_rows_stay_separate_tables() {
        // A 2-cell row directly above a 3-cell row -- different cell counts, so these
        // are structurally different bands and must not be merged into one grid.
        let two_cell = vec![
            cell(
                "Item",
                BBox {
                    left: 72.0,
                    bottom: 690.0,
                    right: 165.0,
                    top: 705.0,
                },
            ),
            cell(
                "42",
                BBox {
                    left: 340.0,
                    bottom: 690.0,
                    right: 351.0,
                    top: 705.0,
                },
            ),
        ];
        let three_cell = vec![
            cell(
                "A",
                BBox {
                    left: 72.0,
                    bottom: 660.0,
                    right: 100.0,
                    top: 675.0,
                },
            ),
            cell(
                "B",
                BBox {
                    left: 200.0,
                    bottom: 660.0,
                    right: 228.0,
                    top: 675.0,
                },
            ),
            cell(
                "C",
                BBox {
                    left: 340.0,
                    bottom: 660.0,
                    right: 368.0,
                    top: 675.0,
                },
            ),
        ];
        let mut all = Vec::new();
        all.extend(two_cell);
        all.extend(three_cell);
        let refs: Vec<&Block> = all.iter().collect();

        let regions = detect_borderless_table_regions(&refs, 792.0, 0);
        assert_eq!(
            regions.len(),
            2,
            "expected two separate tables, got {regions:?}"
        );
    }
}

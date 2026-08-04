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
//! [`RegionClass::Formula`] still requires a genuine layout/vision model -- inline
//! formula segmentation needs more than ruling lines or geometry can offer -- and is
//! never produced here; see `docs/pitfall_registry.json`'s `nested_formula` entry.
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

use crate::{BBox, Block, Document};

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

    for page in &document.pages {
        let body_font_size = body_median_font_size(page);
        let page_predominantly_bold = page_is_predominantly_bold(page);
        let mut seen_title = false;

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
            detect_borderless_table_regions(&ungraphed_blocks, page.height);

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
                &page.blocks,
                page.height,
                body_font_size,
                page_predominantly_bold,
                repeated_band,
                &mut seen_title,
            );
            regions.push(Region {
                class,
                bbox: block.bbox,
                confidence,
            });
        }

        regions.extend(table_regions);
        regions.extend(picture_regions);
        regions.extend(borderless_table_regions);
    }

    regions
}

/// Detects borderless table rows from text geometry alone: a band of `blocks` that share
/// the same vertical span, sit side by side with no vertical overlap between neighbours,
/// and are separated by a gutter wide enough that it can't be ordinary word-wrapping.
///
/// This is the text-layer counterpart to [`crate::graphics::detect_table_regions`], for
/// tables with no ruling lines to key a grid off of -- see `docs/pitfall_registry.json`'s
/// `multi_line_table_cell` entry. Unlike the ruling-line detector, this one has no grid
/// geometry to reconstruct cells from, so it only ever emits one [`Region`] per row band;
/// [`crate::serialize`]'s renderer falls back to ordering a row band's own blocks
/// left-to-right when [`crate::graphics::table_grid_cells`] finds no ruling lines to work
/// from.
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
/// let regions = detect_borderless_table_regions(&blocks, 792.0);
/// assert_eq!(regions.len(), 1);
/// assert_eq!(regions[0].class, RegionClass::Table);
/// ```
pub fn detect_borderless_table_regions(blocks: &[&Block], page_height: f32) -> Vec<Region> {
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

    bands
        .into_values()
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

            let bbox = band.iter().map(|b| b.bbox).reduce(|a, b| a.union(&b))?;
            Some(Region {
                class: RegionClass::Table,
                bbox,
                confidence: 0.5,
            })
        })
        .collect()
}

/// Which page edge a block sits against, per [`HEADER_BAND_FRACTION`]/[`FOOTER_BAND_FRACTION`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Band {
    /// The block sits in the top-of-page header band.
    Header,
    /// The block sits in the bottom-of-page footer band.
    Footer,
}

/// Classifies a single block. `seen_title` is threaded through a page's blocks so at
/// most one [`RegionClass::Title`] is emitted per page (the first oversized block near
/// the top); later oversized blocks fall back to [`RegionClass::SectionHeader`].
/// `repeated_band` is `Some` when this block's band and (normalized) text were found to
/// recur across consecutive pages by [`repeated_running_bands`] -- the strongest signal,
/// checked first. `page_predominantly_bold` gates the bold-heading signal (see
/// [`page_is_predominantly_bold`]): bold text only marks a heading when it stands out
/// against the page's own body.
fn classify_block(
    block: &Block,
    page_blocks: &[Block],
    page_height: f32,
    body_font_size: f32,
    page_predominantly_bold: bool,
    repeated_band: Option<Band>,
    seen_title: &mut bool,
) -> (RegionClass, f32) {
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

/// The fraction of `chars`' font names that are bold (see [`is_bold_font_name`]), or
/// `0.0` for an empty iterator.
fn bold_char_fraction<'a>(chars: impl Iterator<Item = &'a crate::Char>) -> f32 {
    let mut total = 0usize;
    let mut bold = 0usize;
    for c in chars {
        total += 1;
        if is_bold_font_name(&c.font_name) {
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
    if text.len() < prefix.len() {
        return None;
    }
    let (head, tail) = text.split_at(prefix.len());
    head.eq_ignore_ascii_case(prefix).then_some(tail)
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
}

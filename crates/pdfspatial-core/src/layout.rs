//! Stage 2/4: layout region classification.
//!
//! [`classify_regions`] implements a **deterministic, text-layer-only heuristic**
//! classifier — the roadmap's Stage 4a "heuristics first" approach — rather than the
//! ONNX RT-DETR detector described in the roadmap's Stage 2/4b design intent. That
//! detector needs an inference runtime, model weights, and a vision signal
//! ([`crate::extract::render_pages_parallel`] produces the page rasters it would
//! consume) and remains unimplemented; it is the one part of this crate's Stage 2/4
//! surface that isn't a pure algorithm.
//!
//! Because the heuristic only has geometry, font metrics, and text to work with, it can
//! only ever emit the classes derivable from those signals: [`RegionClass::Title`],
//! [`RegionClass::SectionHeader`], [`RegionClass::ListItem`], [`RegionClass::Caption`],
//! [`RegionClass::PageHeader`], [`RegionClass::PageFooter`], and [`RegionClass::Text`].
//! [`RegionClass::Table`], [`RegionClass::Picture`], and [`RegionClass::Formula`] require
//! a genuine layout/vision model and are never produced here — see the module docs above
//! for why.

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

/// A header/footer-band block spanning more than this many lines is treated as body
/// text instead (long text repeated near the page edge is unlikely to be a running
/// header/footer).
const HEADER_FOOTER_MAX_LINES: usize = 2;

/// A caption/list block spanning more than this many lines is treated as body text.
const SHORT_BLOCK_MAX_LINES: usize = 2;

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

    for page in &document.pages {
        let body_font_size = body_median_font_size(page);
        let mut seen_title = false;

        for block in &page.blocks {
            let (class, confidence) =
                classify_block(block, page.height, body_font_size, &mut seen_title);
            regions.push(Region {
                class,
                bbox: block.bbox,
                confidence,
            });
        }
    }

    regions
}

/// Classifies a single block. `seen_title` is threaded through a page's blocks so at
/// most one [`RegionClass::Title`] is emitted per page (the first oversized block near
/// the top); later oversized blocks fall back to [`RegionClass::SectionHeader`].
fn classify_block(
    block: &Block,
    page_height: f32,
    body_font_size: f32,
    seen_title: &mut bool,
) -> (RegionClass, f32) {
    let line_count = block.lines.len();
    let font_size = block_font_size(block);
    let text = block.text();

    let in_header_band = block.bbox.top >= page_height * HEADER_BAND_FRACTION;
    let in_footer_band = block.bbox.bottom <= page_height * FOOTER_BAND_FRACTION;

    if in_header_band && line_count <= HEADER_FOOTER_MAX_LINES {
        return (RegionClass::PageHeader, 0.6);
    }
    if in_footer_band && line_count <= HEADER_FOOTER_MAX_LINES {
        return (RegionClass::PageFooter, 0.6);
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

    (RegionClass::Text, 0.5)
}

/// The median font size across every character on the page — a cheap proxy for "body
/// text size" that [`classify_block`] compares individual blocks against to spot
/// oversized headings.
fn body_median_font_size(page: &crate::Page) -> f32 {
    let mut sizes: Vec<f32> = page
        .blocks
        .iter()
        .flat_map(|b| &b.lines)
        .flat_map(|l| &l.words)
        .flat_map(|w| &w.chars)
        .map(|c| c.font_size)
        .collect();

    if sizes.is_empty() {
        return 0.0;
    }

    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sizes[sizes.len() / 2]
}

/// The dominant font size within a single block (its own median), used to compare
/// against the page body's median.
fn block_font_size(block: &Block) -> f32 {
    let mut sizes: Vec<f32> = block
        .lines
        .iter()
        .flat_map(|l| &l.words)
        .flat_map(|w| &w.chars)
        .map(|c| c.font_size)
        .collect();

    if sizes.is_empty() {
        return 0.0;
    }

    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sizes[sizes.len() / 2]
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

    fn line(text: &str, bbox: BBox, font_size: f32) -> Line {
        Line {
            text: text.into(),
            bbox,
            words: vec![word(text, bbox, font_size)],
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
}

//! Markdown serialization.
//!
//! [`to_markdown`] renders a [`Document`] to Markdown at Stage 1 fidelity: each geometric
//! [`crate::Block`] becomes a paragraph, separated by a blank line, with no headings,
//! lists, or tables — because Stage 1 performs no structural classification (see
//! [`crate::extract`]). It is deliberately left unchanged by Stage 2/4 work so callers
//! that only want the lossless Stage 1 floor keep getting it.
//!
//! [`to_markdown_structured`] renders the Stage 2/4a structural Markdown instead: `#`/`##`
//! headings from [`crate::layout::classify_regions`]'s `Title`/`SectionHeader` output,
//! `-` list items, and italicized captions, over blocks already reordered by
//! [`crate::assemble::assemble_reading_order`].

use crate::Document;
use crate::layout::{Region, RegionClass};

/// Renders `document` to Markdown at Stage 1 fidelity: one paragraph per geometric
/// block, blocks separated by a blank line, and pages separated by a Markdown thematic
/// break (`---`). No structural tagging is applied.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::serialize::to_markdown;
/// use pdfspatial_core::{BBox, Block, Document, Line, Page, Word};
///
/// let word = Word { text: "Hello".into(), bbox: BBox::ZERO, chars: vec![] };
/// let line = Line { text: "Hello".into(), bbox: BBox::ZERO, words: vec![word] };
/// let block = Block { bbox: BBox::ZERO, lines: vec![line] };
/// let page = Page { index: 0, width: 612.0, height: 792.0, blocks: vec![block] };
/// let doc = Document { pages: vec![page] };
///
/// assert_eq!(to_markdown(&doc), "Hello");
/// ```
pub fn to_markdown(document: &Document) -> String {
    document
        .pages
        .iter()
        .map(|page| {
            page.blocks
                .iter()
                .map(|block| block.text())
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n")
}

/// Renders `document` to structural Markdown, using `regions` (as produced by
/// [`crate::layout::classify_regions`]) to decide each block's Markdown treatment:
///
/// - [`RegionClass::Title`] → `# ` heading
/// - [`RegionClass::SectionHeader`] → `## ` heading
/// - [`RegionClass::ListItem`] → `- ` list item
/// - [`RegionClass::Caption`] → italicized (`*...*`) paragraph
/// - [`RegionClass::PageHeader`] / [`RegionClass::PageFooter`] → omitted entirely
/// - everything else (including [`RegionClass::Text`] and any class the heuristic
///   classifier never produces) → a plain paragraph, same as [`to_markdown`]
///
/// Each block is matched to its region by exact bounding-box equality — the contract
/// [`crate::layout::classify_regions`] documents for its output. A block with no
/// matching region (e.g. `regions` came from a different document) falls back to a
/// plain paragraph. Pages are separated by a Markdown thematic break (`---`), as in
/// [`to_markdown`].
///
/// # Examples
///
/// ```
/// use pdfspatial_core::layout::{Region, RegionClass};
/// use pdfspatial_core::serialize::to_markdown_structured;
/// use pdfspatial_core::{BBox, Block, Document, Line, Page, Word};
///
/// let bbox = BBox { left: 0.0, bottom: 0.0, right: 100.0, top: 20.0 };
/// let word = Word { text: "Title".into(), bbox, chars: vec![] };
/// let line = Line { text: "Title".into(), bbox, words: vec![word] };
/// let block = Block { bbox, lines: vec![line] };
/// let page = Page { index: 0, width: 612.0, height: 792.0, blocks: vec![block] };
/// let doc = Document { pages: vec![page] };
///
/// let regions = vec![Region { class: RegionClass::Title, bbox, confidence: 1.0 }];
/// assert_eq!(to_markdown_structured(&doc, &regions), "# Title");
/// ```
pub fn to_markdown_structured(document: &Document, regions: &[Region]) -> String {
    document
        .pages
        .iter()
        .map(|page| {
            page.blocks
                .iter()
                .filter_map(|block| render_block(block, regions))
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n")
}

/// Renders a single block per its matched region's class, or `None` if the region is a
/// running header/footer that should be dropped from the Markdown output.
fn render_block(block: &crate::Block, regions: &[Region]) -> Option<String> {
    let class = regions
        .iter()
        .find(|r| r.bbox == block.bbox)
        .map(|r| r.class);
    let text = block.text();

    match class {
        Some(RegionClass::PageHeader) | Some(RegionClass::PageFooter) => None,
        Some(RegionClass::Title) => Some(format!("# {text}")),
        Some(RegionClass::SectionHeader) => Some(format!("## {text}")),
        Some(RegionClass::ListItem) => Some(
            text.lines()
                .map(|line| format!("- {}", strip_list_marker(line)))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        Some(RegionClass::Caption) => Some(format!("*{text}*")),
        _ => Some(text),
    }
}

/// Strips a leading bullet/ordered-list marker (already detected by
/// [`crate::layout::classify_regions`]) so `render_block` doesn't double it up with the
/// Markdown `- ` it adds.
fn strip_list_marker(line: &str) -> &str {
    let trimmed = line.trim_start();
    for marker in ["- ", "* ", "\u{2022} "] {
        if let Some(rest) = trimmed.strip_prefix(marker) {
            return rest;
        }
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BBox, Line, Word};

    fn bbox(top: f32) -> BBox {
        BBox {
            left: 0.0,
            bottom: top - 10.0,
            right: 100.0,
            top,
        }
    }

    fn block(text: &str, bbox: BBox) -> crate::Block {
        let word = Word {
            text: text.into(),
            bbox,
            chars: vec![],
        };
        let line = Line {
            text: text.into(),
            bbox,
            words: vec![word],
        };
        crate::Block {
            bbox,
            lines: vec![line],
        }
    }

    #[test]
    fn list_item_renders_as_markdown_bullet() {
        let bbox = bbox(100.0);
        let doc = Document {
            pages: vec![crate::Page {
                index: 0,
                width: 612.0,
                height: 792.0,
                blocks: vec![block("- first point", bbox)],
            }],
        };
        let regions = vec![Region {
            class: RegionClass::ListItem,
            bbox,
            confidence: 1.0,
        }];

        assert_eq!(to_markdown_structured(&doc, &regions), "- first point");
    }

    #[test]
    fn page_header_is_omitted_from_structured_output() {
        let header_bbox = bbox(780.0);
        let body_bbox = bbox(100.0);
        let doc = Document {
            pages: vec![crate::Page {
                index: 0,
                width: 612.0,
                height: 792.0,
                blocks: vec![
                    block("Running Header", header_bbox),
                    block("Body text", body_bbox),
                ],
            }],
        };
        let regions = vec![
            Region {
                class: RegionClass::PageHeader,
                bbox: header_bbox,
                confidence: 1.0,
            },
            Region {
                class: RegionClass::Text,
                bbox: body_bbox,
                confidence: 1.0,
            },
        ];

        assert_eq!(to_markdown_structured(&doc, &regions), "Body text");
    }
}

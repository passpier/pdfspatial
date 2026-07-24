//! Markdown serialization.
//!
//! [`to_markdown`] renders a [`Document`] to Markdown at Stage 1 fidelity: each geometric
//! [`crate::Block`] becomes a paragraph, separated by a blank line, with no headings,
//! lists, or tables — because Stage 1 performs no structural classification (see
//! [`crate::extract`]). Structural Markdown (`#` headings from `Title`/`Section-header`
//! regions, `-`/`1.` lists, table syntax) is Stage 2/4 work, gated on
//! [`crate::layout::classify_regions`] and [`crate::assemble::assemble_reading_order`]
//! landing first.

use crate::Document;

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

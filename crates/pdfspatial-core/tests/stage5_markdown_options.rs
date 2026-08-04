//! `MarkdownOptions` regression tests: proves the two output-shape switches
//! (`page_breaks`, `image_placeholders`) do what they say without breaking
//! `to_markdown_structured`'s frozen default output.
//!
//! These flags exist for Stage 5's comparative benchmark (see `bench/opendataloader/`),
//! whose scorer treats Markdown syntax as document text -- a `---` thematic break or a
//! `![]()` placeholder counts as inserted content against ground truth. This file is the
//! corpus-free proof of the flags' semantics, since the 200-PDF benchmark corpus itself
//! isn't vendored into this repo.

use pdfspatial_core::layout::{Region, RegionClass};
use pdfspatial_core::serialize::{to_markdown_structured, to_markdown_structured_with};
use pdfspatial_core::{BBox, Block, Document, Line, MarkdownOptions, Page, Word};

fn bbox(top: f32) -> BBox {
    BBox {
        left: 0.0,
        bottom: top - 10.0,
        right: 100.0,
        top,
    }
}

fn text_block(text: &str, bbox: BBox) -> Block {
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
    Block {
        bbox,
        lines: vec![line],
    }
}

fn page(index: usize, blocks: Vec<Block>) -> Page {
    Page {
        index,
        width: 600.0,
        height: 800.0,
        blocks,
        ..Default::default()
    }
}

#[test]
fn default_options_are_byte_identical_to_to_markdown_structured() {
    let doc = Document {
        pages: vec![
            page(0, vec![text_block("Alpha", bbox(700.0))]),
            page(1, vec![text_block("Beta", bbox(700.0))]),
        ],
    };
    let regions: Vec<Region> = vec![];

    assert_eq!(
        to_markdown_structured_with(&doc, &regions, MarkdownOptions::default()),
        to_markdown_structured(&doc, &regions),
    );
}

#[test]
fn page_breaks_false_removes_the_thematic_break() {
    let doc = Document {
        pages: vec![
            page(0, vec![text_block("Alpha", bbox(700.0))]),
            page(1, vec![text_block("Beta", bbox(700.0))]),
        ],
    };
    let regions: Vec<Region> = vec![];

    let with_breaks = to_markdown_structured_with(&doc, &regions, MarkdownOptions::default());
    assert!(with_breaks.contains("---"));

    let options = MarkdownOptions {
        page_breaks: false,
        ..Default::default()
    };
    let without_breaks = to_markdown_structured_with(&doc, &regions, options);
    assert!(!without_breaks.contains("---"));
    assert_eq!(without_breaks, "Alpha\n\nBeta");
}

#[test]
fn image_placeholders_false_removes_picture_markers_but_still_drops_overlapping_text() {
    let picture_bbox = BBox {
        left: 0.0,
        bottom: 100.0,
        right: 200.0,
        top: 300.0,
    };
    // A stray text block sitting inside the picture's own bbox -- render_page always
    // drops this (it's the picture's own OCR-noise / caption-adjacent text, not a
    // separate item), regardless of whether the placeholder itself is emitted.
    let overlapping_text = text_block(
        "caption noise",
        BBox {
            left: 50.0,
            bottom: 150.0,
            right: 150.0,
            top: 200.0,
        },
    );
    let doc = Document {
        pages: vec![page(0, vec![overlapping_text])],
    };
    let regions = vec![Region {
        class: RegionClass::Picture,
        bbox: picture_bbox,
        confidence: 1.0,
    }];

    let with_placeholder = to_markdown_structured_with(&doc, &regions, MarkdownOptions::default());
    assert_eq!(with_placeholder, "![]()");

    let options = MarkdownOptions {
        image_placeholders: false,
        ..Default::default()
    };
    let without_placeholder = to_markdown_structured_with(&doc, &regions, options);
    assert!(!without_placeholder.contains("![]()"));
    // The overlapping text block is still dropped, not resurrected as its own paragraph.
    assert!(!without_placeholder.contains("caption noise"));
    assert_eq!(without_placeholder, "");
}

#[test]
fn empty_page_with_breaks_disabled_does_not_leave_a_blank_line_run() {
    let doc = Document {
        pages: vec![
            page(0, vec![text_block("Alpha", bbox(700.0))]),
            page(1, vec![]), // genuinely blank page
            page(2, vec![text_block("Gamma", bbox(700.0))]),
        ],
    };
    let regions: Vec<Region> = vec![];

    let options = MarkdownOptions {
        page_breaks: false,
        ..Default::default()
    };
    let markdown = to_markdown_structured_with(&doc, &regions, options);

    // The blank page contributes nothing, and its absence must not leave a stray blank
    // line run between Alpha and Gamma.
    assert_eq!(markdown, "Alpha\n\nGamma");
    assert!(!markdown.contains("\n\n\n"));
}

#[test]
fn empty_page_with_breaks_enabled_still_gets_its_own_separator() {
    let doc = Document {
        pages: vec![
            page(0, vec![text_block("Alpha", bbox(700.0))]),
            page(1, vec![]), // genuinely blank page
            page(2, vec![text_block("Gamma", bbox(700.0))]),
        ],
    };
    let regions: Vec<Region> = vec![];

    let markdown = to_markdown_structured_with(&doc, &regions, MarkdownOptions::default());

    // Every page keeps its own `---` separator, including the blank one, so page count
    // stays inferable from the output.
    assert_eq!(markdown, "Alpha\n\n---\n\n\n\n---\n\nGamma");
}

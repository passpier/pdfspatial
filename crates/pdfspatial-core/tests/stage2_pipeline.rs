//! Stage 2/4 pipeline integration test: `classify_regions` -> `assemble_reading_order`
//! -> `to_markdown_structured`, exercised end to end on a synthetic multi-column
//! `Document`.
//!
//! Unlike `stage1_baseline.rs`, this test builds its `Document` directly in code instead
//! of extracting a PDF, so it requires no native PDFium library and runs in any CI
//! environment.

use pdfspatial_core::layout::classify_regions;
use pdfspatial_core::serialize::{to_markdown_pipeline, to_markdown_structured};
use pdfspatial_core::{
    BBox, Block, Char, Document, Line, MarkdownOptions, Page, Word,
    assemble::assemble_reading_order,
};

fn char_run(text: &str, bbox: BBox, font_size: f32) -> Vec<Char> {
    text.chars()
        .map(|c| Char {
            unicode: Some(c),
            bbox,
            font_name: "Test".into(),
            font_size,
            ..Default::default()
        })
        .collect()
}

fn text_block(text: &str, bbox: BBox, font_size: f32) -> Block {
    let word = Word {
        text: text.to_string(),
        bbox,
        chars: char_run(text, bbox, font_size),
    };
    let line = Line {
        text: text.to_string(),
        bbox,
        words: vec![word],
    };
    Block {
        bbox,
        lines: vec![line],
    }
}

/// A page with a large title, then two columns of body text (out of reading order in
/// the input, mirroring how Stage 1's naive top-to-bottom scan can interleave columns).
fn synthetic_two_column_page() -> Page {
    let title_bbox = BBox {
        left: 40.0,
        bottom: 700.0,
        right: 400.0,
        top: 730.0,
    };
    let title = text_block("Quarterly Report", title_bbox, 22.0);

    let left_bbox = BBox {
        left: 40.0,
        bottom: 400.0,
        right: 280.0,
        top: 650.0,
    };
    let left = text_block("Left column body text here", left_bbox, 10.0);

    let right_bbox = BBox {
        left: 320.0,
        bottom: 400.0,
        right: 560.0,
        top: 650.0,
    };
    let right = text_block("Right column body text here", right_bbox, 10.0);

    Page {
        index: 0,
        width: 600.0,
        height: 800.0,
        // Deliberately out of reading order: right column before left.
        blocks: vec![title, right, left],
        ..Default::default()
    }
}

#[test]
fn stage2_pipeline_reorders_columns_and_emits_structural_markdown() {
    let document = Document {
        pages: vec![synthetic_two_column_page()],
    };

    let regions = classify_regions(&document);
    let reordered = assemble_reading_order(&document);
    let markdown = to_markdown_structured(&reordered, &regions);

    // Reading order: title first, then left column before right column.
    let blocks = &reordered.pages[0].blocks;
    assert_eq!(blocks[0].text(), "Quarterly Report");
    assert_eq!(blocks[1].text(), "Left column body text here");
    assert_eq!(blocks[2].text(), "Right column body text here");

    // The oversized top block should be classified and serialized as a heading.
    assert!(
        markdown.starts_with("# Quarterly Report"),
        "expected structural markdown to open with a heading, got: {markdown:?}"
    );
    assert!(markdown.contains("Left column body text here"));
    assert!(markdown.contains("Right column body text here"));

    // Left column text should still precede right column text in the serialized output.
    let left_pos = markdown.find("Left column").unwrap();
    let right_pos = markdown.find("Right column").unwrap();
    assert!(left_pos < right_pos);
}

#[test]
fn to_markdown_pipeline_matches_the_manual_classify_assemble_serialize_chain() {
    let document = Document {
        pages: vec![synthetic_two_column_page()],
    };

    // The manual chain this integration test already exercises above, expected to be
    // exactly what `to_markdown_pipeline` does internally.
    let regions = classify_regions(&document);
    let reordered = assemble_reading_order(&document);
    let manual = to_markdown_structured(&reordered, &regions);

    let pipeline = to_markdown_pipeline(&document, MarkdownOptions::default());

    assert_eq!(pipeline, manual);
}

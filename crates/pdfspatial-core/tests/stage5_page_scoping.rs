//! Regression test for `Region::page`: proves a picture/table region detected on one page
//! cannot suppress or duplicate content on a different page.
//!
//! Before `Region` carried a `page` field, [`crate::serialize::render_page`] was handed the
//! *whole document's* flat region list for every page it rendered. Two different pages'
//! bbox coordinate spaces both start near `(0, 0)`, so a `Picture`/`Table` region from page
//! 1 could numerically collide with a block on page 2 and silently drop it (via
//! `contains_center`), and every picture placeholder was duplicated once per page instead
//! of rendered once. This file is the corpus-free proof that scoping regions to their own
//! page fixes both failure modes -- see the `opendataloader-bench` investigation that
//! diagnosed it (docs `01030000000198`/`01030000000199`/`01030000000200`, though the
//! precise real-world trigger there turned out to be a full-page background `Image` rather
//! than cross-page bbox collision; this test targets the cross-page mechanism directly).

use pdfspatial_core::layout::{Region, RegionClass};
use pdfspatial_core::serialize::to_markdown_structured;
use pdfspatial_core::{BBox, Block, Document, Line, Page, Word};

fn bbox(left: f32, bottom: f32, right: f32, top: f32) -> BBox {
    BBox {
        left,
        bottom,
        right,
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
fn picture_region_on_one_page_does_not_suppress_text_on_another_page() {
    // Page 0 has a large Picture region; page 1's text block's bbox happens to share the
    // exact same coordinates (both pages are 600x800 -- a realistic collision, not a
    // contrived one, since every page's blocks are laid out in the same page-local space).
    let picture_bbox = bbox(50.0, 50.0, 550.0, 750.0);
    let page1_text_bbox = bbox(50.0, 50.0, 550.0, 750.0);

    let doc = Document {
        pages: vec![
            page(0, vec![]),
            page(1, vec![text_block("Page two survives", page1_text_bbox)]),
        ],
    };
    let regions = vec![Region {
        class: RegionClass::Picture,
        bbox: picture_bbox,
        confidence: 0.7,
        page: 0,
    }];

    let out = to_markdown_structured(&doc, &regions);
    assert!(
        out.contains("Page two survives"),
        "page 1's text was suppressed by page 0's picture region: {out:?}"
    );
    // Exactly one `![]()` -- for page 0's own picture -- not one per page.
    assert_eq!(out.matches("![]()").count(), 1, "output: {out:?}");
}

#[test]
fn table_region_on_one_page_does_not_swallow_a_block_on_another_page() {
    let table_bbox = bbox(50.0, 50.0, 550.0, 750.0);
    let page1_text_bbox = bbox(50.0, 50.0, 550.0, 750.0);

    let doc = Document {
        pages: vec![
            page(0, vec![]),
            page(
                1,
                vec![text_block("Not part of page 0's table", page1_text_bbox)],
            ),
        ],
    };
    let regions = vec![Region {
        class: RegionClass::Table,
        bbox: table_bbox,
        confidence: 0.75,
        page: 0,
    }];

    let out = to_markdown_structured(&doc, &regions);
    assert!(
        out.contains("Not part of page 0's table"),
        "page 1's text was folded into page 0's table region: {out:?}"
    );
}

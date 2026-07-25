//! Stage 1 integration tests: char-recall and line-grouping-accuracy targets, checked
//! against the roadmap's exact thresholds (>= 99% char recall, >= 95% line-grouping
//! accuracy on a single-column document).
//!
//! These tests extract a hand-authored, dependency-free fixture PDF
//! (`tests/fixtures/single_column.pdf`) and score the result against its known ground
//! truth (`tests/fixtures/single_column.lines.txt`).
//!
//! Running these tests requires the native PDFium library. Point
//! `PDFSPATIAL_PDFIUM_LIB` at a PDFium shared library (file or containing directory) --
//! see the crate README for setup instructions and a link to prebuilt binaries.

use pdfspatial_core::{PdfiumSource, extract_baseline_with_source, metrics, serialize};
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn extract_fixture(name: &str) -> pdfspatial_core::Document {
    extract_baseline_with_source(&fixture_path(name), &PdfiumSource::default()).expect(
        "extraction failed -- set PDFSPATIAL_PDFIUM_LIB to a PDFium shared library \
         (see README.md for setup instructions)",
    )
}

const GROUND_TRUTH: &str = include_str!("fixtures/single_column.lines.txt");

#[test]
fn stage1_char_recall_meets_roadmap_target() {
    let document = extract_fixture("single_column.pdf");
    let extracted = document.reading_order_text();

    let recall = metrics::char_recall(&extracted, GROUND_TRUTH);

    assert!(
        recall >= 0.99,
        "char extraction recall {recall:.4} below Stage 1 target of 0.99 \
         (extracted: {extracted:?})"
    );
}

#[test]
fn stage1_line_grouping_meets_roadmap_target() {
    let ground_truth_lines: Vec<String> = GROUND_TRUTH.lines().map(str::to_string).collect();
    let document = extract_fixture("single_column.pdf");

    let extracted_lines: Vec<String> = document
        .pages
        .iter()
        .flat_map(|page| &page.blocks)
        .flat_map(|block| &block.lines)
        .map(|line| line.text.clone())
        .collect();

    let accuracy = metrics::line_grouping_accuracy(&extracted_lines, &ground_truth_lines);

    assert!(
        accuracy >= 0.95,
        "line-grouping accuracy {accuracy:.4} below Stage 1 target of 0.95 \
         (extracted lines: {extracted_lines:?})"
    );
    assert_eq!(
        extracted_lines.len(),
        ground_truth_lines.len(),
        "expected {} lines from baseline clustering, got {}: {extracted_lines:?}",
        ground_truth_lines.len(),
        extracted_lines.len()
    );
}

#[test]
fn stage1_output_carries_zero_structural_tagging() {
    let document = extract_fixture("single_column.pdf");
    let markdown = serialize::to_markdown(&document);

    assert!(
        !markdown.contains('#'),
        "Stage 1 output must not contain heading syntax"
    );
    assert!(
        !markdown.contains("| --"),
        "Stage 1 output must not contain table syntax"
    );
    assert!(!markdown.trim().is_empty());
}

#[test]
fn stage1_block_grouping_separates_paragraphs_on_vertical_gap() {
    let document = extract_fixture("single_column.pdf");

    assert_eq!(document.pages.len(), 1);
    let blocks = &document.pages[0].blocks;

    // The fixture places a title line, then a 3-line paragraph, then a visibly larger
    // vertical gap before a second 2-line paragraph -- the vertical-gap heuristic should
    // separate at least the two paragraphs into distinct blocks.
    assert!(
        blocks.len() >= 2,
        "expected the vertical-gap heuristic to find at least 2 blocks, found {}",
        blocks.len()
    );

    let total_lines: usize = blocks.iter().map(|b| b.lines.len()).sum();
    assert_eq!(
        total_lines, 6,
        "all 6 source lines should still be present across blocks"
    );
}

#[test]
fn stage1_reading_order_is_lossless_on_single_column() {
    let document = extract_fixture("single_column.pdf");

    let extracted_tokens: Vec<&str> = document
        .pages
        .iter()
        .flat_map(|page| &page.blocks)
        .flat_map(|block| &block.lines)
        .flat_map(|line| &line.words)
        .map(|word| word.text.as_str())
        .collect();

    let ground_truth_tokens: Vec<&str> = GROUND_TRUTH.split_whitespace().collect();

    let distance = metrics::reading_order_edit_distance(&extracted_tokens, &ground_truth_tokens);
    eprintln!(
        "reading-order edit distance (single-column, record only): {distance} \
         ({} extracted tokens vs. {} ground-truth tokens)",
        extracted_tokens.len(),
        ground_truth_tokens.len()
    );

    assert_eq!(
        distance, 0,
        "single-column reading order should be lossless -- this is the Stage 1 exit \
         criterion; extracted: {extracted_tokens:?}, ground truth: {ground_truth_tokens:?}"
    );
}

#[test]
fn stage1_throughput_is_recorded() {
    const ITERATIONS: usize = 5;

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        let document = extract_fixture("single_column.pdf");
        std::hint::black_box(&document);
    }
    let elapsed = start.elapsed();

    let page_count = ITERATIONS; // one page per extraction
    let throughput = metrics::throughput_pages_per_sec(page_count, elapsed);
    eprintln!(
        "throughput (single core, no OCR, record only): {throughput:.2} pages/sec \
         ({page_count} pages in {elapsed:?})"
    );

    assert!(
        throughput.is_finite() && throughput > 0.0,
        "expected a positive, finite throughput reading, got {throughput}"
    );
}

#[test]
fn stage1_renders_pages_in_parallel() {
    let rendered = pdfspatial_core::extract::render_pages_parallel(
        &fixture_path("single_column.pdf"),
        &PdfiumSource::default(),
        200,
    )
    .expect(
        "rendering failed -- set PDFSPATIAL_PDFIUM_LIB to a PDFium shared library \
         (see README.md for setup instructions)",
    );

    assert_eq!(rendered.len(), 1);
    let page = &rendered[0];
    assert_eq!(page.width, 200);
    assert!(page.height > 0);
    assert_eq!(page.pixels.len(), (page.width * page.height * 4) as usize);
}

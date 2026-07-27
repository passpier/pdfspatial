//! Re-extracts every PDF-backed Stage 3 corpus case's `source_pdf` and checks the
//! frozen `page`/`pages` snapshot committed under `fixtures/` hasn't drifted.
//!
//! Gated behind the `stage3` cargo feature like `tests/stage3_corpus.rs`, but unlike
//! that file, this one needs the native PDFium library -- freezing a snapshot doesn't
//! remove the need to keep it honest across PDFium version bumps or edits to
//! `examples/stage3_pdf_cases.rs`'s fixture PDFs. `cargo test --features stage3` alone
//! (no PDFium) still runs the corpus-integrity checks in `stage3_corpus.rs`; this file
//! only runs for real once PDFium is available, which `cargo test --all-features`
//! (CI's invocation) always provides. If PDFium can't be loaded and neither
//! `PDFSPATIAL_PDFIUM_LIB` nor `CI` is set, the test prints a skip notice and returns
//! rather than failing, the same accommodation `stage1_baseline.rs`'s README section
//! documents for local `cargo test --features stage3` runs.
//!
//! See `examples/stage3_pdf_cases.rs` and `fixtures/README.md`'s "PDF-backed cases"
//! section for how these snapshots are produced.

#![cfg(feature = "stage3")]

use pdfspatial_core::eval::corpus::load_corpus;
use pdfspatial_core::{PdfiumSource, extract_baseline_with_source};
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn corpus_dir() -> PathBuf {
    workspace_root().join("fixtures")
}

fn pdfium_should_be_required() -> bool {
    std::env::var_os("PDFSPATIAL_PDFIUM_LIB").is_some() || std::env::var_os("CI").is_some()
}

#[test]
fn pdf_backed_cases_match_frozen_snapshot() {
    let cases = load_corpus(&corpus_dir()).expect("corpus should load without error");
    let pdf_cases: Vec<_> = cases.iter().filter(|c| c.source_pdf.is_some()).collect();
    assert!(
        !pdf_cases.is_empty(),
        "expected at least one PDF-backed corpus case"
    );

    let source = PdfiumSource::default();
    let root = workspace_root();
    let mut mismatches: Vec<String> = Vec::new();

    for case in pdf_cases {
        let source_pdf = case
            .source_pdf
            .as_ref()
            .expect("filtered to cases with a source_pdf");
        let pdf_path = root.join(source_pdf);

        let document = match extract_baseline_with_source(&pdf_path, &source) {
            Ok(document) => document,
            Err(err) => {
                if pdfium_should_be_required() {
                    panic!(
                        "failed to extract {} for case {:?}: {err} -- set \
                         PDFSPATIAL_PDFIUM_LIB to a PDFium shared library (see README.md)",
                        pdf_path.display(),
                        case.id
                    );
                }
                eprintln!(
                    "skipping pdf_backed_cases_match_frozen_snapshot: no native PDFium \
                     library available ({err}). Set PDFSPATIAL_PDFIUM_LIB to run this \
                     test for real -- see README.md's Prerequisites section."
                );
                return;
            }
        };

        assert_eq!(
            document.pages.len(),
            case.document.pages.len(),
            "case {:?}: re-extracting {} produced {} page(s), but the frozen snapshot has \
             {} -- regenerate via `cargo run --example stage3_pdf_cases --features stage3`",
            case.id,
            source_pdf.display(),
            document.pages.len(),
            case.document.pages.len()
        );

        for (page_index, (fresh_page, frozen_page)) in document
            .pages
            .iter()
            .zip(case.document.pages.iter())
            .enumerate()
        {
            let fresh_texts: Vec<String> = fresh_page.blocks.iter().map(|b| b.text()).collect();
            let frozen_texts: Vec<String> = frozen_page.blocks.iter().map(|b| b.text()).collect();
            if fresh_texts != frozen_texts {
                mismatches.push(format!(
                    "case {:?}, page {page_index}: re-extracted block text {fresh_texts:?} \
                     doesn't match the frozen snapshot's {frozen_texts:?} -- regenerate via \
                     `cargo run --example stage3_pdf_cases --features stage3`",
                    case.id
                ));
                continue;
            }

            for (block_index, (fresh_block, frozen_block)) in fresh_page
                .blocks
                .iter()
                .zip(frozen_page.blocks.iter())
                .enumerate()
            {
                // Tolerance scales with font size: these fixtures use non-embedded base-14
                // fonts, so PDFium substitutes a different physical font per platform (macOS
                // locally vs Linux in CI) and the loose glyph box's ascent/descent shifts by a
                // few percent of an em, while the standard advance widths stay exact. Applied
                // to all four edges because rotated glyphs put that variation on the x axis.
                const FONT_METRIC_TOLERANCE_FACTOR: f32 = 0.15;
                const MIN_TOLERANCE: f32 = 0.5;
                let max_font_size = fresh_block
                    .lines
                    .iter()
                    .flat_map(|l| l.words.iter())
                    .flat_map(|w| w.chars.iter())
                    .map(|c| c.font_size)
                    .fold(0.0_f32, f32::max);
                let max_font_size = if max_font_size > 0.0 {
                    max_font_size
                } else {
                    12.0
                };
                let epsilon = (FONT_METRIC_TOLERANCE_FACTOR * max_font_size).max(MIN_TOLERANCE);
                let close = |a: f32, b: f32| (a - b).abs() <= epsilon;
                if !(close(fresh_block.bbox.left, frozen_block.bbox.left)
                    && close(fresh_block.bbox.bottom, frozen_block.bbox.bottom)
                    && close(fresh_block.bbox.right, frozen_block.bbox.right)
                    && close(fresh_block.bbox.top, frozen_block.bbox.top))
                {
                    mismatches.push(format!(
                        "case {:?}, page {page_index}, block {block_index}: re-extracted bbox \
                         {:?} drifted from the frozen snapshot's {:?} by more than {epsilon}pt -- \
                         regenerate via `cargo run --example stage3_pdf_cases --features stage3`",
                        case.id, fresh_block.bbox, frozen_block.bbox
                    ));
                }
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} PDF-backed snapshot mismatch(es):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

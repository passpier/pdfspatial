//! End-to-end test of the Stage 3 mining pipeline: DocLayNet sample →
//! `mine_reading_order_failures` → `minimize_reorder_repro` → `write_draft_case` → back
//! through `load_corpus`, exercising the same path `examples/doclaynet_drafts.rs` drives.
//!
//! Gated on both `doclaynet` and `stage3` (compiled by CI's `--all-features`), since it
//! needs both the DocLayNet loader and the corpus writer/reader. Runs with no download
//! and no native PDFium.

#![cfg(all(feature = "doclaynet", feature = "stage3"))]

use pdfspatial_core::BBox;
use pdfspatial_core::assemble::{Pitfall, RootCause};
use pdfspatial_core::eval::corpus::{DraftCase, load_corpus, write_draft_case};
use pdfspatial_core::eval::doclaynet::{
    DocLayNetPage, DocLayNetSample, PdfCell, document_from_cells, load_sample,
    mine_reading_order_failures,
};
use pdfspatial_core::eval::minimize_reorder_repro;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/doclaynet")
}

/// A temp directory unique to this test process, cleaned up on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("pdfspatial_stage3_mining_{name}"));
        std::fs::remove_dir_all(&path).ok();
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

fn cell(text: &str, left: f32, bottom: f32, right: f32, top: f32) -> PdfCell {
    PdfCell {
        text: text.into(),
        bbox: BBox {
            left,
            bottom,
            right,
            top,
        },
        font_size: 10.0,
    }
}

/// The vendored `tests/fixtures/doclaynet/` sample doesn't itself reorder (its one page
/// is already single-column-ordered -- see `stage2_doclaynet.rs`), so it alone can't
/// exercise the "a draft gets written" path deterministically. This builds a synthetic
/// multi-column page alongside it, the same pattern `eval::doclaynet`'s own unit tests
/// use, to guarantee at least one page mines into a draft regardless of the vendored
/// fixture's content.
fn sample_with_a_reordering_page() -> DocLayNetSample {
    let mut sample =
        load_sample(&fixture_dir().join("coco.json"), &fixture_dir()).expect("fixture loads");

    sample.pages.push(DocLayNetPage {
        image_id: 999,
        file_name: "synthetic_multi_column.png".into(),
        width: 600.0,
        height: 800.0,
        ground_truth: Vec::new(),
        cells: vec![
            cell("Right one", 320.0, 700.0, 560.0, 750.0),
            cell("Left one", 40.0, 700.0, 280.0, 750.0),
            cell("Right two", 320.0, 640.0, 560.0, 690.0),
            cell("Left two", 40.0, 640.0, 280.0, 690.0),
            cell("Noise A", 40.0, 500.0, 560.0, 550.0),
            cell("Noise B", 40.0, 400.0, 560.0, 450.0),
        ],
    });
    sample
}

#[test]
fn mining_pipeline_emits_drafts_loadable_by_load_corpus() {
    let sample = sample_with_a_reordering_page();
    let mined = mine_reading_order_failures(&sample);
    assert!(mined.len() >= 2, "fixture page + synthetic page");

    let out = TempDir::new("emits_drafts");
    let mut written = 0;
    let mut skipped = 0;

    for mined_page in &mined {
        let source_page = sample
            .pages
            .iter()
            .find(|p| p.image_id == mined_page.image_id)
            .unwrap();
        let document = document_from_cells(source_page);
        let page = &document.pages[0];
        let original_block_count = page.blocks.len();

        let Some(minimized) = minimize_reorder_repro(page) else {
            skipped += 1;
            continue;
        };

        // The minimizer should never grow the page, and every returned repro must
        // actually still reorder (`reorder_edit_distance > 0`).
        assert!(minimized.page.blocks.len() <= original_block_count);
        assert!(minimized.reorder_edit_distance > 0);

        let id = format!("mined-{}", mined_page.image_id);
        let draft = DraftCase {
            id: &id,
            pitfall: Pitfall::MultiColumn,
            root_cause: RootCause::Ordering,
            description: "test-mined draft",
            page: &minimized.page,
        };

        write_draft_case(&out.0, &draft).expect("draft should write");
        written += 1;
    }

    // The synthetic page is designed to always reorder, so at least one draft must
    // have been written regardless of the vendored fixture's own (non-reordering)
    // content.
    assert!(written >= 1, "expected the synthetic page to mine a draft");
    eprintln!("mined {written} draft(s), skipped {skipped} page(s)");

    let cases = load_corpus(&out.0).expect("emitted drafts should load back via load_corpus");
    assert_eq!(cases.len(), written);
    for case in &cases {
        assert!(case.draft, "case {:?} should be marked draft", case.id);
        assert_eq!(case.pitfall, Pitfall::MultiColumn);
        assert_eq!(case.root_cause, RootCause::Ordering);
        assert!(!case.document.pages[0].blocks.is_empty());
    }
}

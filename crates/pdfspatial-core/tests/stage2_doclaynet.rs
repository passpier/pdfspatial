//! Stage 2 dataset harness integration test: loads a tiny, hand-authored DocLayNet-format
//! fixture (vendored under `tests/fixtures/doclaynet/`, not a real dataset download),
//! scores `layout::classify_regions`'s text-only predictions against it, and checks the
//! resulting [`Stage2Report`] dashboard against hand-computed expected values.
//!
//! This runs deterministically with no network access and no native PDFium, so it's
//! safe for any CI environment; it is gated behind the `doclaynet` cargo feature (which
//! CI compiles via `--all-features`) because it exercises `eval::doclaynet`.

#![cfg(feature = "doclaynet")]

use pdfspatial_core::BBox;
use pdfspatial_core::eval::Stage2Report;
use pdfspatial_core::eval::doclaynet::{
    coco_bbox_to_bbox, evaluate_sample, load_sample, mine_reading_order_failures,
};
use pdfspatial_core::layout::RegionClass;
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/doclaynet")
}

#[test]
fn coco_bbox_to_bbox_matches_hand_computed_flip() {
    // Same box used by the fixture's page-header annotation: [100, 20, 300, 25] in a
    // 1025-tall COCO frame.
    let bbox = coco_bbox_to_bbox([100.0, 20.0, 300.0, 25.0], 1025.0);
    assert_eq!(
        bbox,
        BBox {
            left: 100.0,
            bottom: 980.0,
            right: 400.0,
            top: 1005.0,
        }
    );
}

#[test]
fn load_sample_reads_one_page_with_five_ground_truth_regions() {
    let dir = fixtures_dir();
    let sample = load_sample(&dir.join("coco.json"), &dir).expect("fixture should load");

    assert_eq!(sample.pages.len(), 1);
    let page = &sample.pages[0];
    assert_eq!(page.image_id, 1);
    assert_eq!(page.ground_truth.len(), 5);
    assert_eq!(page.cells.len(), 4);

    let has_class = |class: RegionClass| page.ground_truth.iter().any(|r| r.class == class);
    assert!(has_class(RegionClass::PageHeader));
    assert!(has_class(RegionClass::Title));
    assert!(has_class(RegionClass::Text));
    assert!(has_class(RegionClass::Table));
    assert!(has_class(RegionClass::PageFooter));
}

/// The fixture is deliberately constructed so every ground-truth region except `Table`
/// has a matching `pdf_cells` cell at the *exact* same bbox, and `classify_regions`'s
/// heuristics land on the matching class for each (large-font top block -> `Title`,
/// top-band short block -> `PageHeader`, bottom-band short block -> `PageFooter`,
/// everything else -> `Text`). No cell falls inside the `Table` box, mirroring the
/// documented fact that `classify_regions` never emits `RegionClass::Table` — so `Table`
/// must stay visible in the report with `F1 = 0.0`, not silently disappear.
#[test]
fn evaluate_sample_scores_text_only_predictions_against_fixture() {
    let dir = fixtures_dir();
    let sample = load_sample(&dir.join("coco.json"), &dir).expect("fixture should load");

    let report: Stage2Report = evaluate_sample(&sample, 0.5);

    assert_eq!(report.page_count, 1);
    assert_eq!(report.iou_threshold, 0.5);
    assert_eq!(report.teds, None);

    // Every class except Table is a perfect (identical-bbox) match.
    assert_eq!(report.region_f1[&RegionClass::PageHeader], 1.0);
    assert_eq!(report.region_f1[&RegionClass::Title], 1.0);
    assert_eq!(report.region_f1[&RegionClass::Text], 1.0);
    assert_eq!(report.region_f1[&RegionClass::PageFooter], 1.0);

    // The heuristic classifier never produces `Table`, so it stays a visible false
    // negative rather than being omitted from the report.
    assert_eq!(report.region_f1[&RegionClass::Table], 0.0);
    assert_eq!(report.support[&RegionClass::Table], 1);

    // macro F1 over the 5 classes present in ground truth: (1+1+1+1+0)/5.
    assert_eq!(report.macro_region_f1, 0.8);

    // GIoU is only defined over true-positive matches; Table has none, so it's excluded
    // from both the per-class map and the macro mean, which is 1.0 over the other 4.
    assert!(!report.mean_giou_per_class.contains_key(&RegionClass::Table));
    assert_eq!(report.mean_giou_per_class[&RegionClass::Title], 1.0);
    assert_eq!(report.mean_giou, 1.0);

    // Isolated Footnote/Page-header/Page-footer tracking per the roadmap.
    assert_eq!(report.footnote_f1, 0.0);
    assert_eq!(report.page_header_f1, 1.0);
    assert_eq!(report.page_footer_f1, 1.0);

    // Ground-truth support is visible per class, independent of whether it was matched.
    assert_eq!(report.support[&RegionClass::PageHeader], 1);
    assert_eq!(report.support[&RegionClass::Title], 1);
    assert_eq!(report.support[&RegionClass::Text], 1);
    assert_eq!(report.support[&RegionClass::PageFooter], 1);
}

/// The fixture's 4 cells are single-column and already top-to-bottom in the naive
/// (as-authored) order, so `assemble_reading_order`'s XY-cut leaves them untouched --
/// this is a real, if quiet, end-to-end check that the mining wiring runs the whole
/// load -> reconstruct -> rank pipeline correctly, not just that it compiles.
#[test]
fn mine_reading_order_failures_ranks_the_fixture_page_with_zero_distance() {
    let dir = fixtures_dir();
    let sample = load_sample(&dir.join("coco.json"), &dir).expect("fixture should load");

    let mined = mine_reading_order_failures(&sample);

    assert_eq!(mined.len(), 1);
    assert_eq!(mined[0].image_id, 1);
    assert_eq!(mined[0].file_name, "page_0001.png");
    assert_eq!(mined[0].rank.block_count, 4);
    assert_eq!(mined[0].rank.reorder_edit_distance, 0);
}

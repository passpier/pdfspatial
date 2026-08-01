//! Stage 3 regression corpus harness: loads the hand-authored, minimal-repro cases
//! vendored under the workspace root's `fixtures/` directory and checks them against
//! `layout::classify_regions` / `assemble::assemble_reading_order`.
//!
//! Gated behind the `stage3` cargo feature (compiled by CI's `--all-features`), which
//! pulls in `eval::corpus`. Runs with no native PDFium and no network access.
//!
//! This file has two kinds of tests:
//!
//! - **Corpus integrity** (`corpus_is_wellformed`, `corpus_covers_seeded_pitfalls`) —
//!   always run; these guard the *shape* of the corpus itself (every case parses, ids
//!   are unique, cases live under the directory matching their own pitfall, expected
//!   block references resolve) independent of whether the pipeline behavior they
//!   describe has been implemented yet.
//! - **Behavioral scoreboard** (`corpus_cases_meet_expected_behavior`) — `#[ignore]`d,
//!   since every seeded case currently describes *desired* post-Stage-4 behavior that
//!   the heuristic classifier/assembler doesn't implement yet. Run it explicitly via
//!   `cargo test --features stage3 -- --ignored` to print the full per-case scoreboard;
//!   it's expected to fail today by design, and should be re-run after any Stage 4 fix
//!   to see which cases flipped to passing.
//!
//! Auto-mined draft cases (`eval::corpus::DraftCase`, `RegressionCase::draft`) still
//! parse and get checked by `corpus_is_wellformed` -- they must still be shaped like a
//! valid case -- but are excluded from `corpus_covers_seeded_pitfalls` (they're
//! unreviewed, not seeded coverage) and `corpus_cases_meet_expected_behavior` (their
//! `expected` is a current-behavior snapshot, so they'd pass trivially and mask real
//! regressions). See `fixtures/README.md`'s "Mining drafts from a real sample" section.

#![cfg(feature = "stage3")]

use pdfspatial_core::eval::corpus::{evaluate_case, load_corpus, pitfall_slug};
use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

/// The workspace-root `fixtures/` directory (this crate lives at
/// `crates/pdfspatial-core`, so it's two levels up).
fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// Every pitfall the corpus currently seeds cases for -- both the pitfalls reachable
/// through the synthetic `classify_regions`/`assemble_reading_order` surface and the
/// PDF-backed extraction-layer pitfalls (`source_pdf`-carrying cases whose `page`/
/// `pages` is a frozen real-PDFium snapshot; see `fixtures/README.md`'s "PDF-backed
/// cases" section). This is every variant of `assemble::Pitfall`.
const SEEDED_PITFALL_SLUGS: &[&str] = &[
    "multi_column",
    "footnote",
    "header_footer",
    "borderless_table",
    "merged_table_cell",
    "nested_formula",
    "figure_caption",
    "section_header_vs_bold",
    "list_nesting",
    "multi_line_table_cell",
    "super_subscript",
    "rotated_text",
    "embedded_font",
    "overlapping_text",
    "cross_page_continuation",
];

#[test]
fn corpus_is_wellformed() {
    let cases = load_corpus(&corpus_dir()).expect("corpus should load without error");
    assert!(!cases.is_empty(), "expected at least one seeded case");

    let mut seen_ids = HashSet::new();
    for case in &cases {
        assert!(
            seen_ids.insert(case.id.clone()),
            "duplicate case id {:?} (source: {:?})",
            case.id,
            case.source_path
        );

        assert!(
            case.document.pages.iter().any(|p| !p.blocks.is_empty()),
            "case {:?} has an empty document",
            case.id
        );

        // Every case must live directly under fixtures/<its own pitfall slug>/.
        let expected_dir_name = pitfall_slug(case.pitfall);
        let actual_dir_name = case
            .source_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("");
        assert_eq!(
            actual_dir_name, expected_dir_name,
            "case {:?} tagged pitfall {:?} (slug {expected_dir_name:?}) but lives under \
             directory {actual_dir_name:?} (source: {:?})",
            case.id, case.pitfall, case.source_path
        );

        // A PDF-backed case's source_pdf must point at a real, present file (workspace-
        // relative), so the frozen snapshot has a checkable provenance.
        if let Some(source_pdf) = &case.source_pdf {
            let workspace_root = corpus_dir()
                .parent()
                .expect("fixtures/ has a workspace-root parent")
                .to_path_buf();
            let resolved = workspace_root.join(source_pdf);
            assert!(
                resolved.is_file(),
                "case {:?} names source_pdf {source_pdf:?}, which doesn't exist at {resolved:?}",
                case.id
            );
        }

        // requires_extraction_fix must never be set spuriously: it's only meaningful
        // (and only exempts the block-reference checks below) for PDF-backed cases,
        // and it must actually name at least one string that doesn't resolve today --
        // otherwise the flag is dead weight and the case should just be a normal
        // desired-behavior assertion.
        if case.expected.requires_extraction_fix {
            assert!(
                case.source_pdf.is_some(),
                "case {:?} sets requires_extraction_fix but has no source_pdf",
                case.id
            );
        }

        // Every expected.classes[].block_text / reading_order entry must name a block
        // that actually exists -- unless requires_extraction_fix says the expectation
        // deliberately names text the current extractor doesn't produce yet, in which
        // case at least one entry must genuinely fail to resolve (or the flag is
        // spurious).
        let block_texts: HashSet<String> = case
            .document
            .pages
            .iter()
            .flat_map(|p| p.blocks.iter())
            .map(|b| b.text())
            .collect();

        let mut any_unresolved = false;
        for expected_class in &case.expected.classes {
            if !block_texts.contains(&expected_class.block_text) {
                any_unresolved = true;
                assert!(
                    case.expected.requires_extraction_fix,
                    "case {:?} expects a class for block text {:?}, but no such block \
                     exists (and requires_extraction_fix isn't set)",
                    case.id, expected_class.block_text
                );
            }
        }
        if let Some(order) = &case.expected.reading_order {
            for text in order {
                if !block_texts.contains(text) {
                    any_unresolved = true;
                    assert!(
                        case.expected.requires_extraction_fix,
                        "case {:?} expects {:?} in its reading order, but no such block \
                         exists (and requires_extraction_fix isn't set)",
                        case.id, text
                    );
                }
            }
        }
        assert!(
            !case.expected.requires_extraction_fix || any_unresolved,
            "case {:?} sets requires_extraction_fix but every expected string already \
             resolves to a real block -- the flag is spurious",
            case.id
        );
    }
}

#[test]
fn corpus_covers_seeded_pitfalls() {
    let cases = load_corpus(&corpus_dir()).expect("corpus should load without error");

    // Draft cases (see `eval::corpus::DraftCase`) are unreviewed mining output, not
    // hand-authored coverage -- they don't count toward the seeded checklist.
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for case in cases.iter().filter(|c| !c.draft) {
        *counts.entry(pitfall_slug(case.pitfall)).or_insert(0) += 1;
    }

    for slug in SEEDED_PITFALL_SLUGS {
        assert!(
            counts.get(slug).copied().unwrap_or(0) >= 1,
            "expected at least one seeded case for pitfall {slug:?}; coverage so far: {counts:?}"
        );
    }

    eprintln!("Stage 3 corpus coverage (cases per pitfall): {counts:#?}");
}

#[test]
#[ignore = "Stage 3 failing specs -- flip green as Stage 4 lands fixes; run with \
            `cargo test --features stage3 -- --ignored` for the full scoreboard, or \
            `cargo run --example stage3_scoreboard --features stage3` for the same \
            report outside a panic message"]
fn corpus_cases_meet_expected_behavior() {
    // `eval::scoreboard::score_corpus` is the single implementation of "aggregate
    // non-draft case outcomes by pitfall" -- this test's only job is to turn that
    // aggregate into a pass/fail assertion with a readable panic message, via
    // `render_text`, which `examples/stage3_scoreboard.rs` shares verbatim.
    let cases = load_corpus(&corpus_dir()).expect("corpus should load without error");
    let board = pdfspatial_core::eval::scoreboard::score_corpus(&cases);

    if board.passing < board.total {
        panic!("{}", pdfspatial_core::eval::scoreboard::render_text(&board));
    }
}

/// Positive contrast to the (still-failing, `#[ignore]`d) scoreboard above: a wide
/// column gutter clears `assemble::MIN_CUT_FRACTION`, so `assemble_reading_order`
/// recovers the correct column-major order even though the case is authored in the
/// naive (interleaved) order Stage 1's baseline would emit. This demonstrates the
/// roadmap's predicted reading-order edit-distance degradation (Stage 1 baseline,
/// `naive_reading_order_edit_distance`) alongside its recovery
/// (`reading_order_edit_distance == 0` post-assembly) -- not a desired-but-unimplemented
/// behavior, so unlike `corpus_cases_meet_expected_behavior` this runs unconditionally.
#[test]
fn multi_column_wide_gutter_recovers_reading_order() {
    let cases = load_corpus(&corpus_dir()).expect("corpus should load without error");
    let case = cases
        .iter()
        .find(|c| c.id == "multi_column-wide-gutter-recovers-column-order")
        .expect("wide-gutter contrast fixture should be present in the corpus");

    let outcome = evaluate_case(case);

    let naive = outcome
        .naive_reading_order_edit_distance
        .expect("case has expected.reading_order");
    let assembled = outcome
        .reading_order_edit_distance
        .expect("case has expected.reading_order");
    eprintln!(
        "multi-column wide-gutter contrast: naive reading-order edit distance = {naive}, \
         post-assembly = {assembled}"
    );

    assert!(
        naive > 0,
        "naive (as-authored) block order should already be degraded relative to ground truth"
    );
    assert_eq!(
        assembled, 0,
        "assemble_reading_order should fully recover column-major order once the gutter \
         clears MIN_CUT_FRACTION"
    );
    assert!(outcome.passed);
}

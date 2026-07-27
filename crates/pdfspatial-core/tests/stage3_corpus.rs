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

/// Every pitfall the corpus currently seeds cases for (the subset reachable through the
/// synthetic `classify_regions`/`assemble_reading_order` surface -- see
/// `fixtures/README.md` for the pitfalls deliberately left unseeded and why).
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
            !case.document.pages[0].blocks.is_empty(),
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

        // Every expected.classes[].block_text must name a block that actually exists.
        let block_texts: HashSet<String> = case.document.pages[0]
            .blocks
            .iter()
            .map(|b| b.text())
            .collect();
        for expected_class in &case.expected.classes {
            assert!(
                block_texts.contains(&expected_class.block_text),
                "case {:?} expects a class for block text {:?}, but no such block exists",
                case.id,
                expected_class.block_text
            );
        }
        if let Some(order) = &case.expected.reading_order {
            for text in order {
                assert!(
                    block_texts.contains(text),
                    "case {:?} expects {:?} in its reading order, but no such block exists",
                    case.id,
                    text
                );
            }
        }
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
            `cargo test --features stage3 -- --ignored` for the full scoreboard"]
fn corpus_cases_meet_expected_behavior() {
    let cases = load_corpus(&corpus_dir()).expect("corpus should load without error");

    // Draft cases (see `eval::corpus::DraftCase`) encode `expected` as a snapshot of
    // *current* pipeline behavior, not desired post-fix behavior -- they pass this
    // scoreboard trivially by construction, so including them would mask real
    // regressions. They stay out of `fixtures/` until reviewed and re-authored anyway;
    // this filter is a defensive guard against that convention slipping.
    let outcomes: Vec<_> = cases
        .iter()
        .filter(|c| !c.draft)
        .map(evaluate_case)
        .collect();
    let failing: Vec<_> = outcomes.iter().filter(|o| !o.passed).collect();

    if !failing.is_empty() {
        let mut report = String::from("Stage 3 regression corpus scoreboard -- failing cases:\n");
        for outcome in &failing {
            report.push_str(&format!("  - {}:\n", outcome.case_id));
            for mismatch in &outcome.mismatches {
                report.push_str(&format!("      {mismatch}\n"));
            }
            if let Some(distance) = outcome.reading_order_edit_distance {
                report.push_str(&format!(
                    "      reading-order edit distance: {distance} (naive input order: {})\n",
                    outcome
                        .naive_reading_order_edit_distance
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "?".to_string())
                ));
            }
        }
        report.push_str(&format!(
            "\n{} / {} cases failing.\n",
            failing.len(),
            outcomes.len()
        ));
        panic!("{report}");
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

//! Loader and checker for the Stage 3 regression corpus (`fixtures/` at the workspace
//! root): a set of hand-authored, minimal-repro JSON cases, each tagged with a
//! [`Pitfall`](crate::assemble::Pitfall) category and a
//! [`RootCause`](crate::assemble::RootCause), and each carrying a small synthetic
//! [`Document`] plus its *desired* post-fix [`classify_regions`]/
//! [`assemble_reading_order`] behavior.
//!
//! Gated behind the `stage3` cargo feature, which pulls in `serde`/`serde_json` — the
//! same pattern [`super::doclaynet`] uses to keep the rest of `pdfspatial-core`
//! dependency-free. Neither [`crate::BBox`] nor [`Document`]/[`Region`]/[`RegionClass`]
//! gain serde derives; this module deserializes into its own private mirror structs and
//! constructs the crate's real types by hand.
//!
//! # Why synthetic `Document`s, not real PDFs
//!
//! The roadmap's Stage 3 process describes extracting a minimal-repro *PDF page* for
//! each failure cluster. Building and running those requires a native PDFium library
//! and a real DocLayNet sample to mine failures from — neither is available in every
//! environment this crate builds in. Each case here instead encodes an
//! already-extracted [`Document`] directly (the same PDFium-free substrate
//! `tests/stage2_pipeline.rs` uses), so the corpus stays runnable anywhere `cargo test`
//! runs. This narrows coverage to pitfalls reachable through the
//! [`classify_regions`]/[`assemble_reading_order`] surface — pitfalls that are
//! fundamentally about character-level extraction (rotated text, CID-keyed fonts,
//! overlapping z-ordered text) or cross-page stitching cannot be reproduced this way,
//! and are deliberately left unseeded (see `fixtures/README.md`).
//!
//! # Desired-behavior assertions, not current-behavior snapshots
//!
//! Each case's `expected` block states what the pipeline *should* do once Stage 4 fixes
//! the corresponding pitfall — not what it does today. [`evaluate_case`] is a checker,
//! not an oracle: it reports mismatches between desired and actual behavior, so a
//! failing [`CaseOutcome`] is expected for any pitfall Stage 4 hasn't fixed yet. See
//! `tests/stage3_corpus.rs`, whose `corpus_cases_meet_expected_behavior` test is
//! `#[ignore]`d for exactly this reason: it becomes an executable Stage 3 scoreboard,
//! flipping green case by case as fixes land.

use crate::assemble::{Pitfall, RootCause, assemble_reading_order};
use crate::layout::{RegionClass, classify_regions};
use crate::{BBox, Block, Char, Document, Line, Page, Word};
use std::path::{Path, PathBuf};

/// Errors returned by [`load_case`] and [`load_corpus`].
#[derive(Debug, thiserror::Error)]
pub enum CorpusError {
    /// Failed to read a file from disk.
    #[error("I/O error reading {path}: {source}")]
    Io {
        /// The path that failed to read.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Failed to parse a file as JSON.
    #[error("JSON error in {path}: {source}")]
    Json {
        /// The path that failed to parse.
        path: PathBuf,
        /// The underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// A case's `pitfall` field didn't match any known [`Pitfall`] slug.
    #[error("unknown pitfall slug {0:?}")]
    UnknownPitfall(String),
    /// A case's `root_cause` field didn't match any known [`RootCause`] slug.
    #[error("unknown root cause slug {0:?}")]
    UnknownRootCause(String),
    /// An `expected.classes[].class` field didn't match any known [`RegionClass`] slug.
    #[error("unknown region class slug {0:?}")]
    UnknownClass(String),
    /// Two blocks in the same case's `page.blocks` had identical text, so
    /// `expected.classes[].block_text`/`expected.reading_order` entries can't
    /// unambiguously identify a single block.
    #[error("duplicate block text {0:?} makes block references ambiguous")]
    DuplicateBlockText(String),
}

/// A single Stage 3 regression case: a minimal-repro [`Document`] tagged with the
/// [`Pitfall`] and [`RootCause`] it exercises, plus its desired post-fix behavior.
#[derive(Debug, Clone)]
pub struct RegressionCase {
    /// A short, unique, human-readable identifier for this case.
    pub id: String,
    /// Which checklist item (see [`crate::assemble`]) this case reproduces.
    pub pitfall: Pitfall,
    /// Whether the fix belongs in the geometric extraction layer, the region
    /// classifier, or the reading-order assembler.
    pub root_cause: RootCause,
    /// A human-readable description of the failure this case reproduces.
    pub description: String,
    /// The minimal-repro document: a single synthetic page built directly in the
    /// crate's coordinate space, deliberately shaped to trigger the pitfall.
    pub document: Document,
    /// The behavior [`classify_regions`]/[`assemble_reading_order`] should exhibit on
    /// [`document`](Self::document) once the pitfall is fixed.
    pub expected: ExpectedBehavior,
    /// The file this case was loaded from, kept for diagnostics.
    pub source_path: PathBuf,
}

/// The desired post-fix behavior a [`RegressionCase`] checks for.
#[derive(Debug, Clone, Default)]
pub struct ExpectedBehavior {
    /// The desired block-text sequence after [`assemble_reading_order`], if this case
    /// exercises reading-order behavior. `None` if the case doesn't check ordering.
    pub reading_order: Option<Vec<String>>,
    /// The desired [`RegionClass`] for each named block, if this case exercises
    /// classification behavior. Empty if the case doesn't check classification.
    pub classes: Vec<ExpectedClass>,
}

/// One block's desired classification, identified by its exact block text.
#[derive(Debug, Clone)]
pub struct ExpectedClass {
    /// The text of the block this expectation applies to (must be unique within the
    /// case's document; see [`CorpusError::DuplicateBlockText`]).
    pub block_text: String,
    /// The class [`classify_regions`] should assign this block once the pitfall is
    /// fixed.
    pub class: RegionClass,
}

/// The result of checking one [`RegressionCase`] against the pipeline's actual
/// behavior, produced by [`evaluate_case`].
#[derive(Debug, Clone)]
pub struct CaseOutcome {
    /// The [`RegressionCase::id`] this outcome is for.
    pub case_id: String,
    /// `true` if every expectation in the case held; `false` if any mismatched.
    pub passed: bool,
    /// Human-readable descriptions of every mismatch found, empty if `passed`.
    pub mismatches: Vec<String>,
    /// [`reading_order_edit_distance`](crate::metrics::reading_order_edit_distance)
    /// between [`assemble_reading_order`]'s output and `expected.reading_order`, i.e.
    /// the post-assembly residual. `None` if the case has no `expected.reading_order`.
    /// Quantifies the roadmap's Stage 1 "reading-order edit distance" metric on the
    /// *fixed* (or still-broken) order, as opposed to
    /// [`Self::naive_reading_order_edit_distance`]'s pre-fix baseline.
    pub reading_order_edit_distance: Option<usize>,
    /// [`reading_order_edit_distance`](crate::metrics::reading_order_edit_distance)
    /// between the case's authored (as-emitted) block order and
    /// `expected.reading_order`, i.e. the naive degradation the roadmap predicts before
    /// [`assemble_reading_order`] runs. `None` if the case has no
    /// `expected.reading_order`.
    pub naive_reading_order_edit_distance: Option<usize>,
}

// --- On-disk schema (private; the corpus's own JSON authoring format) -----------

#[derive(serde::Deserialize)]
struct CaseFile {
    id: String,
    pitfall: String,
    root_cause: String,
    description: String,
    page: PageFile,
    expected: ExpectedFile,
}

#[derive(serde::Deserialize)]
struct PageFile {
    width: f32,
    height: f32,
    blocks: Vec<BlockFile>,
}

/// One geometric block, authored as one or more lines so cases can control
/// `Block::lines().len()` directly -- several pitfalls (running headers/footers,
/// multi-line table cells, fragmented formulas) hinge on line count, which a
/// single-line-per-block shape can't vary.
#[derive(serde::Deserialize)]
struct BlockFile {
    lines: Vec<LineFile>,
}

#[derive(serde::Deserialize)]
struct LineFile {
    text: String,
    /// `[left, bottom, right, top]`, already in the crate's own bottom-left-origin
    /// coordinate space (these fixtures are hand-authored directly in that space, not
    /// converted from an external frame like DocLayNet's COCO annotations are).
    bbox: [f32; 4],
    #[serde(default = "default_font_size")]
    font_size: f32,
}

fn default_font_size() -> f32 {
    10.0
}

#[derive(serde::Deserialize, Default)]
struct ExpectedFile {
    #[serde(default)]
    reading_order: Option<Vec<String>>,
    #[serde(default)]
    classes: Vec<ExpectedClassFile>,
}

#[derive(serde::Deserialize)]
struct ExpectedClassFile {
    block_text: String,
    class: String,
}

// --- Slug <-> enum mappings -------------------------------------------------------

fn pitfall_from_slug(slug: &str) -> Result<Pitfall, CorpusError> {
    Ok(match slug {
        "multi_column" => Pitfall::MultiColumn,
        "footnote" => Pitfall::Footnote,
        "header_footer" => Pitfall::HeaderFooter,
        "multi_line_table_cell" => Pitfall::MultiLineTableCell,
        "merged_table_cell" => Pitfall::MergedTableCell,
        "borderless_table" => Pitfall::BorderlessTable,
        "nested_formula" => Pitfall::NestedFormula,
        "super_subscript" => Pitfall::SuperSubscript,
        "rotated_text" => Pitfall::RotatedText,
        "figure_caption" => Pitfall::FigureCaption,
        "list_nesting" => Pitfall::ListNesting,
        "cross_page_continuation" => Pitfall::CrossPageContinuation,
        "section_header_vs_bold" => Pitfall::SectionHeaderVsBold,
        "embedded_font" => Pitfall::EmbeddedFont,
        "overlapping_text" => Pitfall::OverlappingText,
        other => return Err(CorpusError::UnknownPitfall(other.to_string())),
    })
}

/// The canonical directory-name slug for a [`Pitfall`], the inverse of
/// [`pitfall_from_slug`]. Used to check that a case file lives under the directory
/// matching its own `pitfall` field.
pub fn pitfall_slug(pitfall: Pitfall) -> &'static str {
    match pitfall {
        Pitfall::MultiColumn => "multi_column",
        Pitfall::Footnote => "footnote",
        Pitfall::HeaderFooter => "header_footer",
        Pitfall::MultiLineTableCell => "multi_line_table_cell",
        Pitfall::MergedTableCell => "merged_table_cell",
        Pitfall::BorderlessTable => "borderless_table",
        Pitfall::NestedFormula => "nested_formula",
        Pitfall::SuperSubscript => "super_subscript",
        Pitfall::RotatedText => "rotated_text",
        Pitfall::FigureCaption => "figure_caption",
        Pitfall::ListNesting => "list_nesting",
        Pitfall::CrossPageContinuation => "cross_page_continuation",
        Pitfall::SectionHeaderVsBold => "section_header_vs_bold",
        Pitfall::EmbeddedFont => "embedded_font",
        Pitfall::OverlappingText => "overlapping_text",
    }
}

fn root_cause_from_slug(slug: &str) -> Result<RootCause, CorpusError> {
    Ok(match slug {
        "geometric" => RootCause::Geometric,
        "classification" => RootCause::Classification,
        "ordering" => RootCause::Ordering,
        other => return Err(CorpusError::UnknownRootCause(other.to_string())),
    })
}

fn class_from_slug(slug: &str) -> Result<RegionClass, CorpusError> {
    Ok(match slug {
        "caption" => RegionClass::Caption,
        "footnote" => RegionClass::Footnote,
        "formula" => RegionClass::Formula,
        "list_item" => RegionClass::ListItem,
        "page_footer" => RegionClass::PageFooter,
        "page_header" => RegionClass::PageHeader,
        "picture" => RegionClass::Picture,
        "section_header" => RegionClass::SectionHeader,
        "table" => RegionClass::Table,
        "text" => RegionClass::Text,
        "title" => RegionClass::Title,
        other => return Err(CorpusError::UnknownClass(other.to_string())),
    })
}

// --- Loading -----------------------------------------------------------------------

/// Loads a single regression case from a JSON file.
///
/// # Errors
///
/// Returns [`CorpusError::Io`]/[`CorpusError::Json`] if the file can't be read or
/// parsed, [`CorpusError::UnknownPitfall`]/[`CorpusError::UnknownRootCause`]/
/// [`CorpusError::UnknownClass`] if a tag field doesn't match a known slug, or
/// [`CorpusError::DuplicateBlockText`] if two blocks in the case share identical text
/// (which would make `expected` block references ambiguous).
pub fn load_case(path: &Path) -> Result<RegressionCase, CorpusError> {
    let text = std::fs::read_to_string(path).map_err(|source| CorpusError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let case_file: CaseFile = serde_json::from_str(&text).map_err(|source| CorpusError::Json {
        path: path.to_path_buf(),
        source,
    })?;

    let pitfall = pitfall_from_slug(&case_file.pitfall)?;
    let root_cause = root_cause_from_slug(&case_file.root_cause)?;

    let document = document_from_blocks(&case_file.page);

    let mut seen_texts = std::collections::HashSet::new();
    for block in &document.pages[0].blocks {
        if !seen_texts.insert(block.text()) {
            return Err(CorpusError::DuplicateBlockText(block.text()));
        }
    }

    let classes = case_file
        .expected
        .classes
        .into_iter()
        .map(|c| {
            Ok(ExpectedClass {
                block_text: c.block_text,
                class: class_from_slug(&c.class)?,
            })
        })
        .collect::<Result<Vec<_>, CorpusError>>()?;

    Ok(RegressionCase {
        id: case_file.id,
        pitfall,
        root_cause,
        description: case_file.description,
        document,
        expected: ExpectedBehavior {
            reading_order: case_file.expected.reading_order,
            classes,
        },
        source_path: path.to_path_buf(),
    })
}

/// Loads every `*.json` case file found (recursively) under `dir`, sorted by path for
/// deterministic ordering.
///
/// # Errors
///
/// Returns [`CorpusError::Io`] if `dir` itself can't be read, or any error
/// [`load_case`] can return for an individual file.
pub fn load_corpus(dir: &Path) -> Result<Vec<RegressionCase>, CorpusError> {
    let mut paths = Vec::new();
    collect_json_paths(dir, &mut paths)?;
    paths.sort();

    paths.iter().map(|path| load_case(path)).collect()
}

fn collect_json_paths(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), CorpusError> {
    let entries = std::fs::read_dir(dir).map_err(|source| CorpusError::Io {
        path: dir.to_path_buf(),
        source,
    })?;

    for entry in entries {
        let entry = entry.map_err(|source| CorpusError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_json_paths(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "json") {
            out.push(path);
        }
    }

    Ok(())
}

/// Reconstructs a single-page [`Document`] from a case's `page.blocks`, so
/// [`classify_regions`]/[`assemble_reading_order`] can run on it directly.
///
/// Maps each authored line to its own [`Line`] with a single whole-line [`Word`], one
/// [`Char`] per character (all sharing the line's own bbox and font size) --
/// mirroring the `text_block` helper `tests/stage2_pipeline.rs` uses, extended to allow
/// several lines per block. A block's own bbox is the union of its lines' bboxes, the
/// same invariant [`crate::extract`]'s real block grouping maintains.
fn document_from_blocks(page: &PageFile) -> Document {
    let blocks = page
        .blocks
        .iter()
        .map(|block_file| {
            let lines: Vec<Line> = block_file
                .lines
                .iter()
                .map(|line_file| {
                    let bbox = BBox {
                        left: line_file.bbox[0],
                        bottom: line_file.bbox[1],
                        right: line_file.bbox[2],
                        top: line_file.bbox[3],
                    };
                    let chars: Vec<Char> = line_file
                        .text
                        .chars()
                        .map(|ch| Char {
                            unicode: Some(ch),
                            bbox,
                            font_name: "Stage3Corpus".to_string(),
                            font_size: line_file.font_size,
                        })
                        .collect();
                    let word = Word {
                        text: line_file.text.clone(),
                        bbox,
                        chars,
                    };
                    Line {
                        text: line_file.text.clone(),
                        bbox,
                        words: vec![word],
                    }
                })
                .collect();

            let bbox = lines
                .iter()
                .map(|l| l.bbox)
                .reduce(|a, b| a.union(&b))
                .unwrap_or(BBox::ZERO);

            Block { bbox, lines }
        })
        .collect();

    Document {
        pages: vec![Page {
            index: 0,
            width: page.width,
            height: page.height,
            blocks,
        }],
    }
}

// --- Checking ------------------------------------------------------------------

/// Runs [`classify_regions`] and [`assemble_reading_order`] on `case.document` and
/// compares the result against `case.expected`, collecting a human-readable mismatch
/// for every expectation that didn't hold.
///
/// A failing [`CaseOutcome`] is the expected result for any pitfall Stage 4 hasn't
/// fixed yet — see the [module docs](self) on desired- vs. current-behavior.
pub fn evaluate_case(case: &RegressionCase) -> CaseOutcome {
    let mut mismatches = Vec::new();
    let mut reading_order_edit_distance = None;
    let mut naive_reading_order_edit_distance = None;

    if let Some(expected_order) = &case.expected.reading_order {
        let reordered = assemble_reading_order(&case.document);
        let actual_order: Vec<String> =
            reordered.pages[0].blocks.iter().map(|b| b.text()).collect();
        if &actual_order != expected_order {
            mismatches.push(format!(
                "reading order mismatch: expected {expected_order:?}, got {actual_order:?}"
            ));
        }

        let naive_order: Vec<String> = case.document.pages[0]
            .blocks
            .iter()
            .map(|b| b.text())
            .collect();
        let expected_refs: Vec<&str> = expected_order.iter().map(String::as_str).collect();
        let actual_refs: Vec<&str> = actual_order.iter().map(String::as_str).collect();
        let naive_refs: Vec<&str> = naive_order.iter().map(String::as_str).collect();
        reading_order_edit_distance = Some(crate::metrics::reading_order_edit_distance(
            &actual_refs,
            &expected_refs,
        ));
        naive_reading_order_edit_distance = Some(crate::metrics::reading_order_edit_distance(
            &naive_refs,
            &expected_refs,
        ));
    }

    if !case.expected.classes.is_empty() {
        let regions = classify_regions(&case.document);
        let blocks = &case.document.pages[0].blocks;

        for expected_class in &case.expected.classes {
            let actual = blocks
                .iter()
                .zip(&regions)
                .find(|(block, _)| block.text() == expected_class.block_text)
                .map(|(_, region)| region.class);

            match actual {
                None => mismatches.push(format!(
                    "no block with text {:?} found in document",
                    expected_class.block_text
                )),
                Some(actual_class) if actual_class != expected_class.class => {
                    mismatches.push(format!(
                        "class mismatch for block {:?}: expected {:?}, got {:?}",
                        expected_class.block_text, expected_class.class, actual_class
                    ));
                }
                Some(_) => {}
            }
        }
    }

    CaseOutcome {
        case_id: case.id.clone(),
        passed: mismatches.is_empty(),
        mismatches,
        reading_order_edit_distance,
        naive_reading_order_edit_distance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_case(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    const MINIMAL_CASE: &str = r#"
    {
        "id": "test-case",
        "pitfall": "footnote",
        "root_cause": "classification",
        "description": "A footnote block should be classified as Footnote, not Text.",
        "page": {
            "width": 600.0,
            "height": 800.0,
            "blocks": [
                { "lines": [
                    { "text": "Body text.", "bbox": [40.0, 400.0, 400.0, 420.0], "font_size": 10.0 }
                ] },
                { "lines": [
                    { "text": "1. A footnote.", "bbox": [40.0, 40.0, 400.0, 55.0], "font_size": 8.0 }
                ] }
            ]
        },
        "expected": {
            "classes": [
                { "block_text": "1. A footnote.", "class": "footnote" }
            ]
        }
    }
    "#;

    #[test]
    fn load_case_parses_minimal_case() {
        let dir = std::env::temp_dir().join("pdfspatial_corpus_test_load_case");
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_case(&dir, "case.json", MINIMAL_CASE);

        let case = load_case(&path).expect("case should load");
        assert_eq!(case.id, "test-case");
        assert_eq!(case.pitfall, Pitfall::Footnote);
        assert_eq!(case.root_cause, RootCause::Classification);
        assert_eq!(case.document.pages[0].blocks.len(), 2);
        assert_eq!(case.expected.classes.len(), 1);
        assert_eq!(case.expected.classes[0].class, RegionClass::Footnote);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn evaluate_case_reports_mismatch_for_unfixed_footnote_pitfall() {
        let dir = std::env::temp_dir().join("pdfspatial_corpus_test_evaluate_case");
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_case(&dir, "case.json", MINIMAL_CASE);

        let case = load_case(&path).expect("case should load");
        let outcome = evaluate_case(&case);

        // `classify_regions` never emits `Footnote` today, so this desired-behavior
        // spec is expected to fail until Stage 4 adds footnote classification.
        assert!(!outcome.passed);
        assert_eq!(outcome.mismatches.len(), 1);
        assert!(outcome.mismatches[0].contains("class mismatch"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_case_rejects_duplicate_block_text() {
        let dir = std::env::temp_dir().join("pdfspatial_corpus_test_duplicate");
        std::fs::create_dir_all(&dir).unwrap();
        let contents = r#"
        {
            "id": "dup",
            "pitfall": "footnote",
            "root_cause": "classification",
            "description": "duplicate text",
            "page": {
                "width": 600.0,
                "height": 800.0,
                "blocks": [
                    { "lines": [
                        { "text": "Same", "bbox": [0.0, 0.0, 10.0, 10.0], "font_size": 10.0 }
                    ] },
                    { "lines": [
                        { "text": "Same", "bbox": [0.0, 20.0, 10.0, 30.0], "font_size": 10.0 }
                    ] }
                ]
            },
            "expected": {}
        }
        "#;
        let path = write_case(&dir, "case.json", contents);

        let result = load_case(&path);
        assert!(matches!(result, Err(CorpusError::DuplicateBlockText(_))));

        std::fs::remove_dir_all(&dir).ok();
    }

    const MULTI_COLUMN_CASE: &str = r#"
    {
        "id": "test-multi-column",
        "pitfall": "multi_column",
        "root_cause": "ordering",
        "description": "Two columns authored in interleaved (naive) order, with a gutter wide enough for xy_cut_order to recover column-major order.",
        "page": {
            "width": 600.0,
            "height": 800.0,
            "blocks": [
                { "lines": [ { "text": "Left row one", "bbox": [40.0, 700.0, 280.0, 750.0], "font_size": 10.0 } ] },
                { "lines": [ { "text": "Right row one", "bbox": [320.0, 700.0, 560.0, 750.0], "font_size": 10.0 } ] },
                { "lines": [ { "text": "Left row two", "bbox": [40.0, 640.0, 280.0, 690.0], "font_size": 10.0 } ] },
                { "lines": [ { "text": "Right row two", "bbox": [320.0, 640.0, 560.0, 690.0], "font_size": 10.0 } ] }
            ]
        },
        "expected": {
            "reading_order": ["Left row one", "Left row two", "Right row one", "Right row two"]
        }
    }
    "#;

    #[test]
    fn evaluate_case_reports_reading_order_edit_distances() {
        let dir = std::env::temp_dir().join("pdfspatial_corpus_test_edit_distance");
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_case(&dir, "case.json", MULTI_COLUMN_CASE);

        let case = load_case(&path).expect("case should load");
        let outcome = evaluate_case(&case);

        // The wide gutter clears `MIN_CUT_FRACTION`, so `assemble_reading_order`
        // recovers the expected column-major order exactly.
        assert!(outcome.passed);
        assert_eq!(outcome.reading_order_edit_distance, Some(0));
        // The naive (as-authored, interleaved) order is worse than the assembled one --
        // this is the roadmap's predicted Stage 1 degradation on multi-column layouts.
        assert!(outcome.naive_reading_order_edit_distance.unwrap() > 0);
        assert!(outcome.naive_reading_order_edit_distance > outcome.reading_order_edit_distance);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_corpus_sorts_by_path_and_loads_all() {
        let dir = std::env::temp_dir().join("pdfspatial_corpus_test_load_corpus");
        std::fs::create_dir_all(dir.join("footnote")).unwrap();
        write_case(&dir.join("footnote"), "a.json", MINIMAL_CASE);
        write_case(&dir.join("footnote"), "b.json", MINIMAL_CASE);

        let cases = load_corpus(&dir).expect("corpus should load");
        assert_eq!(cases.len(), 2);

        std::fs::remove_dir_all(&dir).ok();
    }
}

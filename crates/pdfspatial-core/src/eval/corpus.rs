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
//! Most cases here encode an already-extracted [`Document`] directly (the same
//! PDFium-free substrate `tests/stage2_pipeline.rs` uses), so the corpus stays runnable
//! anywhere `cargo test` runs — no native PDFium library or DocLayNet download needed.
//! This is the only way to reach pitfalls that live entirely in the
//! [`classify_regions`]/[`assemble_reading_order`] surface.
//!
//! A minority of cases (geometric extraction-layer pitfalls: rotated text, CID-keyed
//! fonts, overlapping z-ordered text, cross-page stitching) are fundamentally about
//! what PDFium's real text layer produces, which a hand-assembled `Document` can't
//! reproduce. Those cases instead carry a `source_pdf` pointing at a small fixture PDF
//! under `crates/pdfspatial-core/tests/fixtures/stage3/`, and their `page`/`pages` is a
//! **frozen snapshot** of `extract_baseline`'s real output on that PDF, generated once
//! by `examples/stage3_pdf_cases.rs` — not hand-typed. The corpus itself stays
//! PDFium-free to load (the snapshot is just JSON); a separate test
//! (`tests/stage3_pdf_fixtures.rs`, needing PDFium) re-extracts each `source_pdf` and
//! checks the snapshot hasn't drifted. See `fixtures/README.md`'s "PDF-backed cases"
//! section.
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
    /// Failed to serialize a [`DraftCase`] to JSON.
    #[error("JSON serialization error: {0}")]
    Serialize(#[source] serde_json::Error),
    /// [`write_draft_case`] was asked to write a case whose target path already
    /// exists; drafts are never silently overwritten.
    #[error("draft case already exists at {path}")]
    DraftExists {
        /// The path that already existed.
        path: PathBuf,
    },
    /// A case's on-disk `page`/`pages` fields didn't specify exactly one of the two.
    #[error("case at {path} must specify exactly one of `page` or `pages`")]
    PageSpec {
        /// The file that specified zero or both.
        path: PathBuf,
    },
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
    /// The minimal-repro document: either a single synthetic page built directly in
    /// the crate's coordinate space (deliberately shaped to trigger the pitfall), or —
    /// for cases with a [`source_pdf`](Self::source_pdf) — a frozen snapshot of
    /// [`crate::extract_baseline`]'s real output on that PDF.
    pub document: Document,
    /// The behavior [`classify_regions`]/[`assemble_reading_order`] should exhibit on
    /// [`document`](Self::document) once the pitfall is fixed.
    pub expected: ExpectedBehavior,
    /// The file this case was loaded from, kept for diagnostics.
    pub source_path: PathBuf,
    /// For extraction-layer (`geometric`) cases whose `document` is a frozen
    /// [`crate::extract_baseline`] snapshot rather than a hand-authored `Document`: the
    /// workspace-relative path to the PDF it was extracted from. `None` for
    /// hand-authored cases. See the [module docs](self).
    pub source_pdf: Option<PathBuf>,
    /// `true` if this case was auto-mined by [`write_draft_case`] and not yet
    /// human-reviewed. A draft's `expected` is a snapshot of the pipeline's *current*
    /// behavior (from [`assemble_reading_order`] at mining time), not desired post-fix
    /// behavior -- the opposite of every hand-authored case's convention (see the
    /// [module docs](self)). Consumers that treat `expected` as a desired-behavior
    /// assertion (e.g. `corpus_cases_meet_expected_behavior`,
    /// `corpus_covers_seeded_pitfalls`) must exclude draft cases.
    pub draft: bool,
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
    /// `true` if this case's expectations name text that the current extractor
    /// doesn't yet produce (e.g. a desired `"x² + y² = z²"` where extraction today
    /// yields several split blocks) — i.e. the case documents a desired *post-Stage-1-
    /// fix* extraction, not just a post-Stage-4 reordering/classification. Cases
    /// setting this flag are exempt from `corpus_is_wellformed`'s "every expected
    /// string resolves to a real block" check, but must still name at least one string
    /// that genuinely doesn't resolve (enforced by that same test) and must carry a
    /// [`RegressionCase::source_pdf`], so the flag can't be set spuriously.
    pub requires_extraction_fix: bool,
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

#[derive(serde::Deserialize, serde::Serialize)]
struct CaseFile {
    id: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    draft: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_pdf: Option<String>,
    pitfall: String,
    root_cause: String,
    description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    page: Option<PageFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pages: Option<Vec<PageFile>>,
    #[serde(default, skip_serializing_if = "is_default_expected")]
    expected: ExpectedFile,
}

fn is_default_expected(expected: &ExpectedFile) -> bool {
    expected.reading_order.is_none()
        && expected.classes.is_empty()
        && !expected.requires_extraction_fix
}

#[derive(serde::Deserialize, serde::Serialize)]
struct PageFile {
    width: f32,
    height: f32,
    blocks: Vec<BlockFile>,
}

/// One geometric block, authored as one or more lines so cases can control
/// `Block::lines().len()` directly -- several pitfalls (running headers/footers,
/// multi-line table cells, fragmented formulas) hinge on line count, which a
/// single-line-per-block shape can't vary.
#[derive(serde::Deserialize, serde::Serialize)]
struct BlockFile {
    lines: Vec<LineFile>,
}

#[derive(serde::Deserialize, serde::Serialize)]
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

#[derive(serde::Deserialize, serde::Serialize, Default)]
struct ExpectedFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reading_order: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    classes: Vec<ExpectedClassFile>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    requires_extraction_fix: bool,
}

#[derive(serde::Deserialize, serde::Serialize)]
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

/// The canonical slug for a [`RootCause`], the inverse of [`root_cause_from_slug`].
/// Used by [`draft_case_json`] to render a [`DraftCase`]'s tag back into the on-disk
/// schema.
fn root_cause_slug(root_cause: RootCause) -> &'static str {
    match root_cause {
        RootCause::Geometric => "geometric",
        RootCause::Classification => "classification",
        RootCause::Ordering => "ordering",
    }
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

/// The canonical slug for a [`RegionClass`], the inverse of [`class_from_slug`]. Used
/// by [`snapshot_case_json`] to render a [`SnapshotCase`]'s expected classes back into
/// the on-disk schema.
fn class_slug(class: RegionClass) -> &'static str {
    match class {
        RegionClass::Caption => "caption",
        RegionClass::Footnote => "footnote",
        RegionClass::Formula => "formula",
        RegionClass::ListItem => "list_item",
        RegionClass::PageFooter => "page_footer",
        RegionClass::PageHeader => "page_header",
        RegionClass::Picture => "picture",
        RegionClass::SectionHeader => "section_header",
        RegionClass::Table => "table",
        RegionClass::Text => "text",
        RegionClass::Title => "title",
    }
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

    let pages = match (case_file.page, case_file.pages) {
        (Some(page), None) => vec![page],
        (None, Some(pages)) => pages,
        (None, None) | (Some(_), Some(_)) => {
            return Err(CorpusError::PageSpec {
                path: path.to_path_buf(),
            });
        }
    };
    let document = document_from_pages(&pages);

    let mut seen_texts = std::collections::HashSet::new();
    for page in &document.pages {
        for block in &page.blocks {
            if !seen_texts.insert(block.text()) {
                return Err(CorpusError::DuplicateBlockText(block.text()));
            }
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
            requires_extraction_fix: case_file.expected.requires_extraction_fix,
        },
        source_path: path.to_path_buf(),
        source_pdf: case_file.source_pdf.map(PathBuf::from),
        draft: case_file.draft,
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

/// Reconstructs a (possibly multi-page) [`Document`] from a case's `page`/`pages`
/// blocks, so [`classify_regions`]/[`assemble_reading_order`] can run on it directly.
///
/// Maps each authored line to its own [`Line`] with a single whole-line [`Word`], one
/// [`Char`] per character (all sharing the line's own bbox and font size) --
/// mirroring the `text_block` helper `tests/stage2_pipeline.rs` uses, extended to allow
/// several lines per block. A block's own bbox is the union of its lines' bboxes, the
/// same invariant [`crate::extract`]'s real block grouping maintains. `Page::index` is
/// assigned by position in `pages`.
fn document_from_pages(pages: &[PageFile]) -> Document {
    let pages = pages
        .iter()
        .enumerate()
        .map(|(index, page)| {
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

            Page {
                index,
                width: page.width,
                height: page.height,
                blocks,
            }
        })
        .collect();

    Document { pages }
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
        let actual_order: Vec<String> = reordered
            .pages
            .iter()
            .flat_map(|p| p.blocks.iter())
            .map(|b| b.text())
            .collect();
        if &actual_order != expected_order {
            mismatches.push(format!(
                "reading order mismatch: expected {expected_order:?}, got {actual_order:?}"
            ));
        }

        let naive_order: Vec<String> = case
            .document
            .pages
            .iter()
            .flat_map(|p| p.blocks.iter())
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
        let blocks: Vec<&Block> = case
            .document
            .pages
            .iter()
            .flat_map(|p| p.blocks.iter())
            .collect();

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

// --- Draft emission (mining pipeline) ----------------------------------------------

/// The inputs needed to render an auto-mined page as a draft corpus case, consumed by
/// [`draft_case_json`]/[`write_draft_case`].
///
/// A draft's `expected.reading_order` is filled in automatically from
/// [`assemble_reading_order`]'s *current* output on `page` -- i.e. it is a
/// current-behavior snapshot, not a desired-behavior assertion like every hand-authored
/// case's `expected` block. See [`RegressionCase::draft`].
pub struct DraftCase<'a> {
    /// A short, unique, human-readable identifier for this case.
    pub id: &'a str,
    /// The mining pipeline's best guess at which checklist item this page
    /// exercises -- a hypothesis for the human reviewer to confirm or correct, not a
    /// verified tag.
    pub pitfall: Pitfall,
    /// The mining pipeline's best guess at this page's root cause -- likewise a
    /// hypothesis to confirm or correct.
    pub root_cause: RootCause,
    /// A human-readable description of where this page came from (e.g. its source
    /// dataset, image id, and file name) and why it was flagged, so a reviewer knows
    /// what to verify.
    pub description: &'a str,
    /// The (already minimized) page to encode as the case's document.
    pub page: &'a Page,
}

/// Renders `draft` as corpus-schema JSON: `draft: true`, and `expected.reading_order`
/// pre-filled from running [`assemble_reading_order`] on `draft.page` right now (see
/// [`DraftCase`]'s docs on why that's a snapshot, not a desired-behavior assertion).
///
/// # Errors
///
/// Returns [`CorpusError::Serialize`] if JSON serialization fails (in practice only
/// reachable via non-finite `f32` coordinates, which `serde_json` refuses to encode).
pub fn draft_case_json(draft: &DraftCase) -> Result<String, CorpusError> {
    let document = Document {
        pages: vec![draft.page.clone()],
    };
    let assembled = assemble_reading_order(&document);
    let reading_order: Vec<String> = assembled.pages[0].blocks.iter().map(|b| b.text()).collect();

    let case_file = CaseFile {
        id: draft.id.to_string(),
        draft: true,
        source_pdf: None,
        pitfall: pitfall_slug(draft.pitfall).to_string(),
        root_cause: root_cause_slug(draft.root_cause).to_string(),
        description: draft.description.to_string(),
        page: Some(page_file_from_page(draft.page)),
        pages: None,
        expected: ExpectedFile {
            reading_order: Some(reading_order),
            classes: Vec::new(),
            requires_extraction_fix: false,
        },
    };

    serde_json::to_string_pretty(&case_file).map_err(CorpusError::Serialize)
}

/// Converts a real [`Page`] (e.g. mined from DocLayNet, or extracted by
/// [`crate::extract_baseline`]) into the on-disk `PageFile` schema, dropping only
/// `Page::index` (which the schema derives from position in `page`/`pages`) and
/// collapsing each line's per-char font sizes down to its first char's, since the
/// schema authors one `font_size` per line, not per char.
fn page_file_from_page(page: &Page) -> PageFile {
    let blocks = page
        .blocks
        .iter()
        .map(|block| BlockFile {
            lines: block
                .lines
                .iter()
                .map(|line| LineFile {
                    text: line.text.clone(),
                    bbox: [
                        line.bbox.left,
                        line.bbox.bottom,
                        line.bbox.right,
                        line.bbox.top,
                    ],
                    font_size: line
                        .words
                        .first()
                        .and_then(|w| w.chars.first())
                        .map(|c| c.font_size)
                        .unwrap_or_else(default_font_size),
                })
                .collect(),
        })
        .collect();

    PageFile {
        width: page.width,
        height: page.height,
        blocks,
    }
}

/// The inputs needed to render a PDF-backed pitfall as a corpus case, consumed by
/// [`snapshot_case_json`].
///
/// Unlike [`DraftCase`], a `SnapshotCase`'s `expected` is written verbatim: it's a
/// human-authored, desired-behavior assertion (the corpus's normal convention), not an
/// auto-filled current-behavior snapshot. Only `pages` itself is a snapshot — a frozen
/// copy of [`crate::extract_baseline`]'s real output on `source_pdf`, taken once by
/// `examples/stage3_pdf_cases.rs` so the corpus stays PDFium-free to load. See the
/// [module docs](self).
pub struct SnapshotCase<'a> {
    /// A short, unique, human-readable identifier for this case.
    pub id: &'a str,
    /// Which checklist item (see [`crate::assemble`]) this case reproduces.
    pub pitfall: Pitfall,
    /// Whether the fix belongs in the geometric extraction layer, the region
    /// classifier, or the reading-order assembler.
    pub root_cause: RootCause,
    /// A human-readable description of the failure this case reproduces.
    pub description: &'a str,
    /// Workspace-relative path to the PDF this case's `pages` was extracted from.
    pub source_pdf: &'a str,
    /// The frozen [`crate::extract_baseline`] output to encode as the case's document,
    /// one entry per page in extraction order.
    pub pages: &'a [Page],
    /// The desired post-fix behavior this case checks for, written to the case file
    /// as-is.
    pub expected: &'a ExpectedBehavior,
}

/// Renders `case` as corpus-schema JSON carrying a `source_pdf` and a frozen
/// `page`/`pages` snapshot, for cases whose pitfall lives in the real PDFium text
/// layer (see the [module docs](self)).
///
/// # Examples
///
/// ```
/// use pdfspatial_core::eval::corpus::{ExpectedBehavior, SnapshotCase, snapshot_case_json};
/// use pdfspatial_core::assemble::{Pitfall, RootCause};
/// use pdfspatial_core::{BBox, Block, Char, Line, Page, Word};
///
/// let bbox = BBox { left: 40.0, bottom: 700.0, right: 200.0, top: 712.0 };
/// let ch = Char { unicode: Some('x'), bbox, font_name: "Helvetica".into(), font_size: 12.0 };
/// let word = Word { text: "x".into(), bbox, chars: vec![ch] };
/// let line = Line { text: "x".into(), bbox, words: vec![word] };
/// let page = Page { index: 0, width: 600.0, height: 800.0, blocks: vec![Block { bbox, lines: vec![line] }] };
///
/// let expected = ExpectedBehavior::default();
/// let case = SnapshotCase {
///     id: "example-case",
///     pitfall: Pitfall::RotatedText,
///     root_cause: RootCause::Geometric,
///     description: "example",
///     source_pdf: "crates/pdfspatial-core/tests/fixtures/stage3/rotated_text.pdf",
///     pages: std::slice::from_ref(&page),
///     expected: &expected,
/// };
///
/// let json = snapshot_case_json(&case).unwrap();
/// assert!(json.contains("\"source_pdf\""));
/// ```
///
/// # Errors
///
/// Returns [`CorpusError::Serialize`] if JSON serialization fails (in practice only
/// reachable via non-finite `f32` coordinates, which `serde_json` refuses to encode).
pub fn snapshot_case_json(case: &SnapshotCase) -> Result<String, CorpusError> {
    let (page, pages) = match case.pages {
        [single] => (Some(page_file_from_page(single)), None),
        many => (None, Some(many.iter().map(page_file_from_page).collect())),
    };

    let case_file = CaseFile {
        id: case.id.to_string(),
        draft: false,
        source_pdf: Some(case.source_pdf.to_string()),
        pitfall: pitfall_slug(case.pitfall).to_string(),
        root_cause: root_cause_slug(case.root_cause).to_string(),
        description: case.description.to_string(),
        page,
        pages,
        expected: ExpectedFile {
            reading_order: case.expected.reading_order.clone(),
            classes: case
                .expected
                .classes
                .iter()
                .map(|c| ExpectedClassFile {
                    block_text: c.block_text.clone(),
                    class: class_slug(c.class).to_string(),
                })
                .collect(),
            requires_extraction_fix: case.expected.requires_extraction_fix,
        },
    };

    serde_json::to_string_pretty(&case_file).map_err(CorpusError::Serialize)
}

/// Writes `draft` as a draft case file to `dir/<pitfall_slug>/<id>.json`, creating the
/// pitfall subdirectory if needed.
///
/// # Errors
///
/// Returns [`CorpusError::Serialize`] if rendering fails (see [`draft_case_json`]),
/// [`CorpusError::DraftExists`] if the target path already exists (drafts are never
/// silently overwritten -- a human review may have already edited that file), or
/// [`CorpusError::Io`] if creating the directory or writing the file fails.
pub fn write_draft_case(dir: &Path, draft: &DraftCase) -> Result<PathBuf, CorpusError> {
    let json = draft_case_json(draft)?;

    let subdir = dir.join(pitfall_slug(draft.pitfall));
    std::fs::create_dir_all(&subdir).map_err(|source| CorpusError::Io {
        path: subdir.clone(),
        source,
    })?;

    let path = subdir.join(format!("{}.json", draft.id));
    if path.exists() {
        return Err(CorpusError::DraftExists { path });
    }

    std::fs::write(&path, json).map_err(|source| CorpusError::Io {
        path: path.clone(),
        source,
    })?;

    Ok(path)
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

    fn draft_page() -> Page {
        fn line(text: &str, bbox: [f32; 4]) -> Line {
            let bbox = BBox {
                left: bbox[0],
                bottom: bbox[1],
                right: bbox[2],
                top: bbox[3],
            };
            let chars: Vec<Char> = text
                .chars()
                .map(|ch| Char {
                    unicode: Some(ch),
                    bbox,
                    font_name: "Draft".to_string(),
                    font_size: 10.0,
                })
                .collect();
            let word = Word {
                text: text.to_string(),
                bbox,
                chars,
            };
            Line {
                text: text.to_string(),
                bbox,
                words: vec![word],
            }
        }
        fn block(text: &str, bbox: [f32; 4]) -> Block {
            let line = line(text, bbox);
            Block {
                bbox: line.bbox,
                lines: vec![line],
            }
        }

        Page {
            index: 0,
            width: 600.0,
            height: 800.0,
            blocks: vec![
                block("Right one", [320.0, 700.0, 560.0, 750.0]),
                block("Left one", [40.0, 700.0, 280.0, 750.0]),
                block("Right two", [320.0, 640.0, 560.0, 690.0]),
                block("Left two", [40.0, 640.0, 280.0, 690.0]),
            ],
        }
    }

    #[test]
    fn write_draft_case_round_trips_through_load_case() {
        let dir = std::env::temp_dir().join("pdfspatial_corpus_test_draft_round_trip");
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();

        let page = draft_page();
        let draft = DraftCase {
            id: "mined-example",
            pitfall: Pitfall::MultiColumn,
            root_cause: RootCause::Ordering,
            description: "mined from a synthetic multi-column page",
            page: &page,
        };

        let path = write_draft_case(&dir, &draft).expect("draft should write");
        assert_eq!(path, dir.join("multi_column").join("mined-example.json"));

        let case = load_case(&path).expect("draft should load back");
        assert_eq!(case.id, "mined-example");
        assert!(case.draft);
        assert_eq!(case.pitfall, Pitfall::MultiColumn);
        assert_eq!(case.root_cause, RootCause::Ordering);
        assert_eq!(case.document.pages[0].blocks.len(), page.blocks.len());

        // A draft's expected.reading_order is a snapshot of assemble_reading_order's
        // current output, so it must already pass evaluate_case trivially -- this is
        // exactly why the behavioral scoreboard must exclude drafts.
        let outcome = evaluate_case(&case);
        assert!(outcome.passed);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_draft_case_refuses_to_overwrite_existing_file() {
        let dir = std::env::temp_dir().join("pdfspatial_corpus_test_draft_exists");
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();

        let page = draft_page();
        let draft = DraftCase {
            id: "mined-example",
            pitfall: Pitfall::MultiColumn,
            root_cause: RootCause::Ordering,
            description: "mined from a synthetic multi-column page",
            page: &page,
        };

        write_draft_case(&dir, &draft).expect("first write should succeed");
        let result = write_draft_case(&dir, &draft);
        assert!(matches!(result, Err(CorpusError::DraftExists { .. })));

        std::fs::remove_dir_all(&dir).ok();
    }
}

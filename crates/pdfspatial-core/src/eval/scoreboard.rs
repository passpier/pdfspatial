//! Machine-readable Stage 3 pitfall scoreboard, built on top of [`super::corpus`].
//!
//! `eval::corpus` answers "does this one case pass?"; this module aggregates that
//! per-case answer across the whole corpus, grouped by
//! [`Pitfall`](crate::assemble::Pitfall), and renders the result into the exact
//! artifacts that used to be hand-maintained prose: the roadmap's pitfall checklist,
//! its blocker table, `fixtures/README.md`'s coverage tables, and `README.md`'s
//! status-table row. The generator is the single source of truth for every count that
//! text used to restate by hand; see `examples/stage3_scoreboard.rs` for the CLI that
//! drives it and `.claude/skills/pitfall-{status,fix}/SKILL.md` for the workflows built
//! on top of it.
//!
//! Two kinds of input feed the renderers:
//!
//! - **Live facts** — [`score_corpus`] over [`super::corpus::load_corpus`]'s output.
//!   Pass/fail, per-pitfall counts, and edit distances always come from actually
//!   running [`super::corpus::evaluate_case`]; nothing here is ever hand-typed.
//! - **Human judgement** — [`Registry`], loaded from `docs/pitfall_registry.json` by
//!   [`load_registry`]. A pitfall can be 100% green on the corpus and still be only
//!   partially fixed in a way no test can detect (the `footnote` pitfall: both cases
//!   pass classification, but `serialize::render_block` has no `Footnote` arm yet) --
//!   [`RegistryEntry::status_override`] is the escape hatch for exactly that gap.
//!
//! Generated text is spliced into a document between a pair of
//! `<!-- BEGIN GENERATED: name --> … <!-- END GENERATED: name -->` HTML-comment
//! markers via [`splice_block`], so everything outside those markers stays
//! hand-written and hand-reviewed.

use super::corpus::{CaseOutcome, RegressionCase, evaluate_case, pitfall_slug};
use crate::assemble::{Pitfall, RootCause};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Every [`Pitfall`] variant, in the roadmap checklist's listed order (which matches
/// the enum's own declaration order -- see [`Pitfall`]'s doc comment).
pub const ALL_PITFALLS: [Pitfall; 15] = [
    Pitfall::MultiColumn,
    Pitfall::Footnote,
    Pitfall::HeaderFooter,
    Pitfall::MultiLineTableCell,
    Pitfall::MergedTableCell,
    Pitfall::BorderlessTable,
    Pitfall::NestedFormula,
    Pitfall::SuperSubscript,
    Pitfall::RotatedText,
    Pitfall::FigureCaption,
    Pitfall::ListNesting,
    Pitfall::CrossPageContinuation,
    Pitfall::SectionHeaderVsBold,
    Pitfall::EmbeddedFont,
    Pitfall::OverlappingText,
];

/// Errors returned by [`load_registry`] and [`splice_block`].
#[derive(Debug, thiserror::Error)]
pub enum ScoreboardError {
    /// Failed to read the registry file from disk.
    #[error("I/O error reading {path}: {source}")]
    Io {
        /// The path that failed to read.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Failed to parse the registry file as JSON.
    #[error("JSON error in {path}: {source}")]
    Json {
        /// The path that failed to parse.
        path: PathBuf,
        /// The underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// A registry key didn't match any known [`Pitfall`] slug.
    #[error("unknown pitfall slug {0:?} in registry")]
    UnknownPitfall(String),
    /// [`splice_block`] couldn't find a `<!-- BEGIN/END GENERATED: name -->` marker
    /// pair in the target document.
    #[error("missing `<!-- {which} GENERATED: {name} -->` marker")]
    MissingMarker {
        /// The generated-block name that was being spliced.
        name: String,
        /// Which half of the pair (`"BEGIN"` or `"END"`) was missing.
        which: &'static str,
    },
    /// [`splice_block`]'s end marker appeared before its begin marker.
    #[error("`<!-- END GENERATED: {0} -->` appears before its matching BEGIN marker")]
    MarkerOrder(String),
}

/// One pitfall's outcomes, aggregated from every non-draft [`RegressionCase`] tagged
/// with it.
#[derive(Debug, Clone)]
pub struct PitfallScore {
    /// The pitfall this score is for.
    pub pitfall: Pitfall,
    /// The root cause tagged on this pitfall's cases (taken from the first case; the
    /// corpus convention is one root cause per pitfall).
    pub root_cause: RootCause,
    /// Every non-draft case's outcome, in corpus load order.
    pub cases: Vec<CaseOutcome>,
    /// Number of `cases` entries with `passed: true`.
    pub passing: usize,
    /// `cases.len()`.
    pub total: usize,
}

/// The full corpus scoreboard: every [`Pitfall`]'s aggregated outcome, plus the global
/// pass/fail totals repeated throughout the roadmap and `fixtures/README.md`.
#[derive(Debug, Clone)]
pub struct Scoreboard {
    /// One entry per [`ALL_PITFALLS`], in that order.
    pub pitfalls: Vec<PitfallScore>,
    /// Sum of every [`PitfallScore::passing`].
    pub passing: usize,
    /// Sum of every [`PitfallScore::total`].
    pub total: usize,
}

/// Runs [`evaluate_case`] over every non-draft case in `cases` and aggregates the
/// result by [`Pitfall`], in [`ALL_PITFALLS`] order.
///
/// Draft cases ([`RegressionCase::draft`]) are excluded -- their `expected` is a
/// snapshot of current behavior, not a desired-behavior assertion, so they'd pass
/// trivially and mask real regressions. This mirrors
/// `tests/stage3_corpus.rs::corpus_cases_meet_expected_behavior`'s own filter.
///
/// A pitfall with zero non-draft cases still gets a [`PitfallScore`] entry (with
/// `total: 0`, `passing: 0`, and an arbitrary [`RootCause`] placeholder of
/// [`RootCause::Geometric`]) so every renderer can iterate [`ALL_PITFALLS`] uniformly;
/// today every pitfall is seeded, so this is defensive rather than expected.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::eval::corpus::load_corpus;
/// use pdfspatial_core::eval::scoreboard::score_corpus;
/// # fn corpus_dir() -> std::path::PathBuf {
/// #     std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
/// # }
///
/// let cases = load_corpus(&corpus_dir()).expect("corpus should load");
/// let board = score_corpus(&cases);
/// assert_eq!(board.pitfalls.len(), 15);
/// assert!(board.passing <= board.total);
/// ```
pub fn score_corpus(cases: &[RegressionCase]) -> Scoreboard {
    let mut by_pitfall: BTreeMap<&'static str, Vec<&RegressionCase>> = BTreeMap::new();
    for case in cases.iter().filter(|c| !c.draft) {
        by_pitfall
            .entry(pitfall_slug(case.pitfall))
            .or_default()
            .push(case);
    }

    let mut passing = 0;
    let mut total = 0;
    let pitfalls = ALL_PITFALLS
        .iter()
        .map(|&pitfall| {
            let slug = pitfall_slug(pitfall);
            let matching = by_pitfall.get(slug).cloned().unwrap_or_default();
            let root_cause = matching
                .first()
                .map(|c| c.root_cause)
                .unwrap_or(RootCause::Geometric);
            let cases: Vec<CaseOutcome> = matching.iter().map(|c| evaluate_case(c)).collect();
            let pass_count = cases.iter().filter(|o| o.passed).count();
            passing += pass_count;
            total += cases.len();
            PitfallScore {
                pitfall,
                root_cause,
                total: cases.len(),
                passing: pass_count,
                cases,
            }
        })
        .collect();

    Scoreboard {
        pitfalls,
        passing,
        total,
    }
}

// --- Registry (human judgement) -----------------------------------------------------

/// Extra prose/context for a pitfall that no test can derive: why it's only
/// partially fixed despite a green scoreboard, what unblocks it, whether it needs a
/// real PDF, etc. Loaded from `docs/pitfall_registry.json` by [`load_registry`].
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RegistryEntry {
    /// The pitfall's checklist title, e.g. `"Multi-column layout / gutter detection"`.
    pub title: String,
    /// One-sentence description of the failure mode, used as the checklist bullet's
    /// body text.
    pub summary: String,
    /// Freeform prose appended to the checklist bullet explaining *why* the live
    /// pass/fail counts look the way they do (which heuristic changed, what's still
    /// open). `None` if the counts speak for themselves.
    #[serde(default)]
    pub note: Option<String>,
    /// Overrides the checklist status marker (`"fixed"`, `"partial"`, or `"open"`)
    /// that would otherwise be derived purely from `passing`/`total`. Needed when a
    /// pitfall is corpus-green but genuinely incomplete outside what the corpus checks
    /// (see the `footnote` entry).
    #[serde(default)]
    pub status_override: Option<String>,
    /// Set when this pitfall is blocked on a named prerequisite (the vision-model
    /// stub, an extraction-layer gap, ...). Feeds the roadmap's blocker table and
    /// [`RegistryEntry::blocked`]'s `loop_refuses` gate for `/pitfall-fix`.
    #[serde(default)]
    pub blocked: Option<BlockedInfo>,
    /// For the 6 PDF-backed pitfalls: why a hand-authored synthetic `Document` can't
    /// reproduce this failure mode, feeding `fixtures/README.md`'s PDF-backed coverage
    /// table. `None` for hand-authored pitfalls.
    #[serde(default)]
    pub pdf_backed_reason: Option<String>,
}

/// A pitfall's blocking prerequisite, shared by every registry entry blocked on the
/// same underlying gap (grouped by identical `reason` text in
/// [`render_blocker_table`]).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct BlockedInfo {
    /// The blocker table's "Blocker" column text, e.g.
    /// `` "no code path emits `RegionClass::Table`/`Formula`/`Picture`" ``. Entries
    /// sharing identical `reason` text are grouped into one blocker-table row.
    pub reason: String,
    /// The blocker table's "Next step" column text.
    pub unblocked_by: String,
    /// `true` if `/pitfall-fix` must refuse to iterate on this pitfall outright (the
    /// fix requires the out-of-scope ONNX vision-model stub documented in
    /// `CLAUDE.md`, not a heuristic change). `false` for prerequisites a heuristic
    /// change genuinely could clear (extraction-layer gaps, missing serializer arms).
    #[serde(default)]
    pub loop_refuses: bool,
}

/// A [`Pitfall`] slug (see [`pitfall_slug`]) to [`RegistryEntry`] map, the parsed form
/// of `docs/pitfall_registry.json`.
pub type Registry = BTreeMap<String, RegistryEntry>;

/// Loads and validates the pitfall registry from `path`.
///
/// # Errors
///
/// [`ScoreboardError::Io`]/[`ScoreboardError::Json`] if the file can't be read or
/// parsed as JSON, or [`ScoreboardError::UnknownPitfall`] if a key doesn't match any
/// [`pitfall_slug`] (this validation lives here rather than in
/// `registry_covers_every_pitfall` alone so every renderer gets it for free).
pub fn load_registry(path: &Path) -> Result<Registry, ScoreboardError> {
    let text = std::fs::read_to_string(path).map_err(|source| ScoreboardError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let registry: Registry =
        serde_json::from_str(&text).map_err(|source| ScoreboardError::Json {
            path: path.to_path_buf(),
            source,
        })?;

    let known_slugs: std::collections::HashSet<&'static str> =
        ALL_PITFALLS.iter().copied().map(pitfall_slug).collect();
    for slug in registry.keys() {
        if !known_slugs.contains(slug.as_str()) {
            return Err(ScoreboardError::UnknownPitfall(slug.clone()));
        }
    }

    Ok(registry)
}

fn root_cause_slug(root_cause: RootCause) -> &'static str {
    match root_cause {
        RootCause::Geometric => "geometric",
        RootCause::Classification => "classification",
        RootCause::Ordering => "ordering",
    }
}

/// `"fixed"`, `"partial"`, or `"open"`, derived from `score.passing`/`score.total`
/// unless [`RegistryEntry::status_override`] is set.
fn derived_status(score: &PitfallScore, entry: Option<&RegistryEntry>) -> &'static str {
    if let Some(status) = entry.and_then(|e| e.status_override.as_deref()) {
        return match status {
            "fixed" => "fixed",
            "partial" => "partial",
            _ => "open",
        };
    }
    if score.total > 0 && score.passing == score.total {
        "fixed"
    } else if score.passing > 0 {
        "partial"
    } else {
        "open"
    }
}

fn status_marker(status: &str) -> &'static str {
    match status {
        "fixed" => "[x]",
        "partial" => "[~]",
        _ => "[ ]",
    }
}

// --- Rendering -----------------------------------------------------------------------

/// Renders a JSON report of the full scoreboard: global totals plus, per pitfall, its
/// slug, root cause, passing/total counts, and every case's id/pass/mismatches/edit
/// distances. Consumed by `--format json` (`examples/stage3_scoreboard.rs`) and the
/// `/pitfall-fix` skill's gate check.
pub fn render_json(board: &Scoreboard) -> String {
    let pitfalls: Vec<serde_json::Value> = board
        .pitfalls
        .iter()
        .map(|score| {
            let cases: Vec<serde_json::Value> = score
                .cases
                .iter()
                .map(|outcome| {
                    serde_json::json!({
                        "case_id": outcome.case_id,
                        "passed": outcome.passed,
                        "mismatches": outcome.mismatches,
                        "reading_order_edit_distance": outcome.reading_order_edit_distance,
                        "naive_reading_order_edit_distance": outcome.naive_reading_order_edit_distance,
                    })
                })
                .collect();
            serde_json::json!({
                "pitfall": pitfall_slug(score.pitfall),
                "root_cause": root_cause_slug(score.root_cause),
                "passing": score.passing,
                "total": score.total,
                "cases": cases,
            })
        })
        .collect();

    let value = serde_json::json!({
        "passing": board.passing,
        "total": board.total,
        "pitfalls": pitfalls,
    });
    serde_json::to_string_pretty(&value).expect("Scoreboard JSON is always serializable")
}

/// Renders the plain-text scoreboard report: every failing case's id and mismatches
/// (plus its reading-order edit distance, if it has one), followed by the
/// `P / N cases failing.` summary line.
///
/// This is `tests/stage3_corpus.rs::corpus_cases_meet_expected_behavior`'s panic
/// message, factored out so the CLI and the test share one implementation.
pub fn render_text(board: &Scoreboard) -> String {
    let mut report = String::from("Stage 3 regression corpus scoreboard -- failing cases:\n");
    for score in &board.pitfalls {
        for outcome in &score.cases {
            if outcome.passed {
                continue;
            }
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
    }
    let failing = board.total - board.passing;
    report.push_str(&format!("\n{failing} / {} cases failing.\n", board.total));
    report
}

/// Renders the roadmap's 15-item Document-Structure Pitfall Checklist, one bullet per
/// [`ALL_PITFALLS`] entry, as generated Markdown for [`splice_block`]'s
/// `pitfall-checklist` block.
pub fn render_checklist(board: &Scoreboard, registry: &Registry) -> String {
    let mut out = format!(
        "Status reflects the Stage 3 regression corpus under \
         [`fixtures/`](../fixtures) — run `cargo run --example stage3_scoreboard \
         --features stage3` for the live scoreboard (currently {}/{} cases \
         passing); see [`fixtures/README.md`](../fixtures/README.md) for per-case \
         detail.\n\n",
        board.passing, board.total
    );

    for score in &board.pitfalls {
        let slug = pitfall_slug(score.pitfall);
        let entry = registry.get(slug);
        let status = derived_status(score, entry);
        let marker = status_marker(status);
        let title = entry.map(|e| e.title.as_str()).unwrap_or(slug);
        let summary = entry.map(|e| e.summary.as_str()).unwrap_or_default();
        let case_word = if score.total == 1 { "case" } else { "cases" };
        let note = entry
            .and_then(|e| e.note.as_deref())
            .map(|n| format!(" — {n}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "- {marker} **{title}** — {summary} *({} {case_word}, `{}`; {}/{} \
             passing{note})*\n",
            score.total,
            root_cause_slug(score.root_cause),
            score.passing,
            score.total,
        ));
    }
    out
}

/// Renders the roadmap's "Status & next steps" blocker table for [`splice_block`]'s
/// `pitfall-blockers` block: one row per distinct [`BlockedInfo::reason`] across the
/// registry, grouping every pitfall blocked on the same prerequisite.
pub fn render_blocker_table(board: &Scoreboard, registry: &Registry) -> String {
    let mut groups: Vec<(&str, &str, Vec<&str>)> = Vec::new();
    for score in &board.pitfalls {
        let slug = pitfall_slug(score.pitfall);
        let Some(entry) = registry.get(slug) else {
            continue;
        };
        let Some(blocked) = &entry.blocked else {
            continue;
        };
        if let Some(group) = groups.iter_mut().find(|(reason, unblocked_by, _)| {
            *reason == blocked.reason && *unblocked_by == blocked.unblocked_by
        }) {
            group.2.push(&entry.title);
        } else {
            groups.push((&blocked.reason, &blocked.unblocked_by, vec![&entry.title]));
        }
    }

    let mut out = String::from("| Blocker | Pitfalls | Next step |\n|---|---|---|\n");
    for (reason, unblocked_by, titles) in groups {
        out.push_str(&format!(
            "| {reason} | {} | {unblocked_by} |\n",
            titles.join(", ")
        ));
    }
    out
}

/// Renders `fixtures/README.md`'s two coverage tables (hand-authored synthetic
/// pitfalls, then PDF-backed frozen-snapshot pitfalls) for [`splice_block`]'s
/// `pitfall-coverage` block, plus the total seeded-case count.
pub fn render_coverage_tables(
    board: &Scoreboard,
    registry: &Registry,
    cases: &[RegressionCase],
) -> String {
    let total_cases = cases.iter().filter(|c| !c.draft).count();
    let pdf_backed: std::collections::HashSet<&'static str> = cases
        .iter()
        .filter(|c| !c.draft && c.source_pdf.is_some())
        .map(|c| pitfall_slug(c.pitfall))
        .collect();

    let mut out =
        format!("All 15 `assemble::Pitfall` variants are seeded, {total_cases} cases total:\n\n");

    out.push_str(
        "**Hand-authored, synthetic `Document`s** -- reachable through the synthetic \
         `classify_regions`/`assemble_reading_order` surface, no PDF or PDFium \
         involved:\n\n| Pitfall | Root cause |\n|---|---|\n",
    );
    for score in &board.pitfalls {
        let slug = pitfall_slug(score.pitfall);
        if pdf_backed.contains(slug) {
            continue;
        }
        let title = registry.get(slug).map(|e| e.title.as_str()).unwrap_or(slug);
        out.push_str(&format!(
            "| `{slug}` ({title}) | `{}` |\n",
            root_cause_slug(score.root_cause)
        ));
    }

    out.push_str(
        "\n**PDF-backed, frozen-extraction snapshots** -- see \"PDF-backed cases\" \
         below:\n\n| Pitfall | Root cause | Why it needs a real PDF |\n|---|---|---|\n",
    );
    for score in &board.pitfalls {
        let slug = pitfall_slug(score.pitfall);
        if !pdf_backed.contains(slug) {
            continue;
        }
        let entry = registry.get(slug);
        let reason = entry
            .and_then(|e| e.pdf_backed_reason.as_deref())
            .unwrap_or("");
        out.push_str(&format!(
            "| `{slug}` | `{}` | {reason} |\n",
            root_cause_slug(score.root_cause)
        ));
    }

    out
}

/// Replaces the content between a `<!-- BEGIN GENERATED: name --> … <!-- END
/// GENERATED: name -->` marker pair in `doc` with `body`, leaving everything outside
/// the markers (and the markers themselves) untouched. Used by
/// `examples/stage3_scoreboard.rs --write` to keep the roadmap, `fixtures/README.md`,
/// and `README.md` in sync with the live corpus without disturbing their hand-written
/// prose.
///
/// # Errors
///
/// [`ScoreboardError::MissingMarker`] if either half of the pair isn't present, or
/// [`ScoreboardError::MarkerOrder`] if the end marker appears before the begin marker.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::eval::scoreboard::splice_block;
///
/// let doc = "intro\n<!-- BEGIN GENERATED: foo -->\nold\n<!-- END GENERATED: foo -->\noutro\n";
/// let updated = splice_block(doc, "foo", "new").unwrap();
/// assert_eq!(
///     updated,
///     "intro\n<!-- BEGIN GENERATED: foo -->\nnew\n<!-- END GENERATED: foo -->\noutro\n"
/// );
/// ```
pub fn splice_block(doc: &str, name: &str, body: &str) -> Result<String, ScoreboardError> {
    let begin = format!("<!-- BEGIN GENERATED: {name} -->");
    let end = format!("<!-- END GENERATED: {name} -->");

    let begin_at = doc
        .find(&begin)
        .ok_or_else(|| ScoreboardError::MissingMarker {
            name: name.to_string(),
            which: "BEGIN",
        })?;
    let end_at = doc
        .find(&end)
        .ok_or_else(|| ScoreboardError::MissingMarker {
            name: name.to_string(),
            which: "END",
        })?;
    if end_at < begin_at {
        return Err(ScoreboardError::MarkerOrder(name.to_string()));
    }

    let content_start = begin_at + begin.len();
    let mut out = String::with_capacity(doc.len() + body.len());
    out.push_str(&doc[..content_start]);
    out.push('\n');
    out.push_str(body.trim_end());
    out.push('\n');
    out.push_str(&doc[end_at..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splice_block_replaces_only_the_marked_region() {
        let doc = "a\n<!-- BEGIN GENERATED: x -->\nold\n<!-- END GENERATED: x -->\nb\n";
        let out = splice_block(doc, "x", "new").unwrap();
        assert_eq!(
            out,
            "a\n<!-- BEGIN GENERATED: x -->\nnew\n<!-- END GENERATED: x -->\nb\n"
        );
    }

    #[test]
    fn splice_block_is_idempotent() {
        let doc = "a\n<!-- BEGIN GENERATED: x -->\nold\n<!-- END GENERATED: x -->\nb\n";
        let once = splice_block(doc, "x", "new").unwrap();
        let twice = splice_block(&once, "x", "new").unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn splice_block_errors_on_missing_marker() {
        let doc = "no markers here\n";
        assert!(matches!(
            splice_block(doc, "x", "new"),
            Err(ScoreboardError::MissingMarker { which: "BEGIN", .. })
        ));
    }

    #[test]
    fn splice_block_errors_when_end_precedes_begin() {
        let doc = "<!-- END GENERATED: x -->\n<!-- BEGIN GENERATED: x -->\n";
        assert!(matches!(
            splice_block(doc, "x", "new"),
            Err(ScoreboardError::MarkerOrder(_))
        ));
    }

    #[test]
    fn derived_status_matches_pass_ratio_without_override() {
        let full = PitfallScore {
            pitfall: Pitfall::MultiColumn,
            root_cause: RootCause::Ordering,
            cases: vec![],
            passing: 2,
            total: 2,
        };
        let partial = PitfallScore {
            passing: 1,
            ..clone_score(&full)
        };
        let open = PitfallScore {
            passing: 0,
            ..clone_score(&full)
        };
        assert_eq!(derived_status(&full, None), "fixed");
        assert_eq!(derived_status(&partial, None), "partial");
        assert_eq!(derived_status(&open, None), "open");
    }

    fn clone_score(score: &PitfallScore) -> PitfallScore {
        PitfallScore {
            pitfall: score.pitfall,
            root_cause: score.root_cause,
            cases: score.cases.clone(),
            passing: score.passing,
            total: score.total,
        }
    }
}

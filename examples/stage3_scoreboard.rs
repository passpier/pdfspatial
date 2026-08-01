//! CLI over `eval::scoreboard`: renders the live Stage 3 pitfall scoreboard, and can
//! rewrite the generated blocks embedded in the roadmap, `fixtures/README.md`, and
//! `README.md` so they never drift from what the corpus actually reports.
//!
//! ```sh
//! cargo run --example stage3_scoreboard --features stage3                 # text report (default)
//! cargo run --example stage3_scoreboard --features stage3 -- --format json
//! cargo run --example stage3_scoreboard --features stage3 -- --write      # rewrite generated blocks
//! cargo run --example stage3_scoreboard --features stage3 -- --check      # exit 1 if docs are stale
//! ```
//!
//! Needs no PDFium: the corpus (including its 6 PDF-backed cases) loads from committed
//! JSON snapshots under `fixtures/`.

use pdfspatial_core::eval::corpus::load_corpus;
use pdfspatial_core::eval::scoreboard::{
    Registry, Scoreboard, load_registry, render_blocker_table, render_checklist,
    render_coverage_tables, render_json, render_text, score_corpus, splice_block,
};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

struct GeneratedBlock {
    doc: PathBuf,
    name: &'static str,
}

fn generated_blocks(workspace_root: &Path) -> Vec<(GeneratedBlock, String)> {
    vec![
        (
            GeneratedBlock {
                doc: workspace_root.join("docs/PDF-to-Markdown Pipeline Roadmap.md"),
                name: "pitfall-checklist",
            },
            String::new(),
        ),
        (
            GeneratedBlock {
                doc: workspace_root.join("docs/PDF-to-Markdown Pipeline Roadmap.md"),
                name: "pitfall-blockers",
            },
            String::new(),
        ),
        (
            GeneratedBlock {
                doc: workspace_root.join("docs/PDF-to-Markdown Pipeline Roadmap.md"),
                name: "pitfall-exit-criterion",
            },
            String::new(),
        ),
        (
            GeneratedBlock {
                doc: workspace_root.join("docs/PDF-to-Markdown Pipeline Roadmap.md"),
                name: "pitfall-dashboard-row",
            },
            String::new(),
        ),
        (
            GeneratedBlock {
                doc: workspace_root.join("fixtures/README.md"),
                name: "pitfall-coverage",
            },
            String::new(),
        ),
        (
            GeneratedBlock {
                doc: workspace_root.join("README.md"),
                name: "pitfall-status-row",
            },
            String::new(),
        ),
    ]
}

fn bodies(
    board: &Scoreboard,
    registry: &Registry,
    cases: &[pdfspatial_core::eval::corpus::RegressionCase],
) -> Vec<(&'static str, String)> {
    let blocked_count = registry.values().filter(|e| e.blocked.is_some()).count();
    vec![
        ("pitfall-checklist", render_checklist(board, registry)),
        ("pitfall-blockers", render_blocker_table(board, registry)),
        (
            "pitfall-exit-criterion",
            format!(
                "**Met** — all 15 pitfalls are seeded in the corpus with a root-cause tag \
                 and a live pass/fail scoreboard ({}/{} cases passing today, {blocked_count} \
                 pitfalls blocked on a named prerequisite above); the one open gap is volume \
                 — each category has 1–3 cases, short of the ≥ 20-samples-per-category target, \
                 which needs a real DocLayNet mining pass to close.",
                board.passing, board.total
            ),
        ),
        (
            "pitfall-dashboard-row",
            format!(
                "| 3. Error Analysis | Failure count/category, root-cause split | Full \
                 pitfall-checklist coverage — **met**: 15/15 pitfalls seeded, \
                 root-cause-tagged, scoreboard-tracked ({}/{} cases passing); \
                 ≥20-samples/category volume still outstanding |",
                board.passing, board.total
            ),
        ),
        (
            "pitfall-coverage",
            render_coverage_tables(board, registry, cases),
        ),
        (
            "pitfall-status-row",
            format!(
                "| Regression corpus of known failure modes ({} seeded cases) | ✅ Stable |",
                board.total
            ),
        ),
    ]
}

fn corpus_dir(workspace_root: &Path) -> PathBuf {
    workspace_root.join("fixtures")
}

/// The workspace root -- this example is compiled as part of `pdfspatial-core`, whose
/// manifest lives at `crates/pdfspatial-core`, two levels below the workspace root
/// where `docs/`, `fixtures/`, and `README.md` actually live (same convention
/// `tests/stage3_corpus.rs::corpus_dir` uses).
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn registry_path(workspace_root: &Path) -> PathBuf {
    workspace_root.join("docs/pitfall_registry.json")
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let format_json = args.iter().any(|a| a == "--format")
        && args
            .windows(2)
            .any(|w| w[0] == "--format" && w[1] == "json");
    let write = args.iter().any(|a| a == "--write");
    let check = args.iter().any(|a| a == "--check");

    let root = workspace_root();
    let cases = match load_corpus(&corpus_dir(&root)) {
        Ok(cases) => cases,
        Err(err) => {
            eprintln!("failed to load Stage 3 corpus: {err}");
            return ExitCode::FAILURE;
        }
    };
    let board = score_corpus(&cases);
    let registry = match load_registry(&registry_path(&root)) {
        Ok(registry) => registry,
        Err(err) => {
            eprintln!("failed to load pitfall registry: {err}");
            return ExitCode::FAILURE;
        }
    };

    if !write && !check {
        if format_json {
            println!("{}", render_json(&board));
        } else {
            print!("{}", render_text(&board));
        }
        return ExitCode::SUCCESS;
    }

    let rendered: std::collections::HashMap<&str, String> =
        bodies(&board, &registry, &cases).into_iter().collect();

    let mut stale = Vec::new();
    let mut by_doc: std::collections::BTreeMap<PathBuf, Vec<&'static str>> =
        std::collections::BTreeMap::new();
    for (block, _) in generated_blocks(&root) {
        by_doc.entry(block.doc).or_default().push(block.name);
    }

    for (doc_path, names) in by_doc {
        let original = match std::fs::read_to_string(&doc_path) {
            Ok(text) => text,
            Err(err) => {
                eprintln!("failed to read {}: {err}", doc_path.display());
                return ExitCode::FAILURE;
            }
        };
        let mut updated = original.clone();
        for name in names {
            let body = rendered.get(name).map(String::as_str).unwrap_or("");
            updated = match splice_block(&updated, name, body) {
                Ok(updated) => updated,
                Err(err) => {
                    eprintln!("failed to splice {name} into {}: {err}", doc_path.display());
                    return ExitCode::FAILURE;
                }
            };
        }

        if updated != original {
            if check {
                stale.push(doc_path.clone());
            } else if write {
                if let Err(err) = std::fs::write(&doc_path, &updated) {
                    eprintln!("failed to write {}: {err}", doc_path.display());
                    return ExitCode::FAILURE;
                }
                println!("wrote {}", doc_path.display());
            }
        }
    }

    if check && !stale.is_empty() {
        eprintln!("Stage 3 generated docs are stale:");
        for path in &stale {
            eprintln!("  - {}", path.display());
        }
        eprintln!(
            "\nRun `cargo run --example stage3_scoreboard --features stage3 -- --write` \
             to regenerate."
        );
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

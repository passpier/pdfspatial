//! Guards that the generated doc blocks (roadmap checklist/blockers/exit-criterion,
//! `fixtures/README.md`'s coverage tables, `README.md`'s status row) stay in sync with
//! what `eval::scoreboard` actually reports for the live corpus, and that
//! `docs/pitfall_registry.json` covers every seeded pitfall.
//!
//! Gated behind the `stage3` cargo feature, same as `tests/stage3_corpus.rs` and
//! `eval::scoreboard` itself, so CI's `cargo test --workspace --all-features` step
//! enforces this with no `ci.yml` change. On drift, `generated_blocks_are_in_sync`
//! fails with the exact `cargo run --example stage3_scoreboard --features stage3 --
//! --write` command needed to fix it -- **never hand-edit the text between a
//! `<!-- BEGIN/END GENERATED -->` marker pair**; that's exactly the drift this test
//! exists to catch.

#![cfg(feature = "stage3")]

use pdfspatial_core::eval::corpus::{load_corpus, pitfall_slug};
use pdfspatial_core::eval::scoreboard::{
    ALL_PITFALLS, load_registry, render_blocker_table, render_checklist, render_coverage_tables,
    score_corpus, splice_block,
};
use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn registry_path() -> PathBuf {
    workspace_root().join("docs/pitfall_registry.json")
}

#[test]
fn generated_blocks_are_in_sync() {
    let cases = load_corpus(&corpus_dir()).expect("corpus should load without error");
    let board = score_corpus(&cases);
    let registry = load_registry(&registry_path()).expect("registry should load without error");

    let blocked_count = registry.values().filter(|e| e.blocked.is_some()).count();

    let checks: Vec<(PathBuf, &str, String)> = vec![
        (
            workspace_root().join("docs/PDF-to-Markdown Pipeline Roadmap.md"),
            "pitfall-checklist",
            render_checklist(&board, &registry),
        ),
        (
            workspace_root().join("docs/PDF-to-Markdown Pipeline Roadmap.md"),
            "pitfall-blockers",
            render_blocker_table(&board, &registry),
        ),
        (
            workspace_root().join("docs/PDF-to-Markdown Pipeline Roadmap.md"),
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
            workspace_root().join("docs/PDF-to-Markdown Pipeline Roadmap.md"),
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
            workspace_root().join("fixtures/README.md"),
            "pitfall-coverage",
            render_coverage_tables(&board, &registry, &cases),
        ),
        (
            workspace_root().join("README.md"),
            "pitfall-status-row",
            format!(
                "| Regression corpus of known failure modes ({} seeded cases) | ✅ Stable |",
                board.total
            ),
        ),
    ];

    let mut stale = Vec::new();
    for (path, name, body) in checks {
        let committed = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        let regenerated = splice_block(&committed, name, &body)
            .unwrap_or_else(|err| panic!("failed to splice {name} into {}: {err}", path.display()));
        if regenerated != committed {
            stale.push(format!("{} [{name}]", path.display()));
        }
    }

    assert!(
        stale.is_empty(),
        "generated doc blocks are stale: {stale:?}\n\nRun `cargo run --example \
         stage3_scoreboard --features stage3 -- --write` to regenerate them."
    );
}

#[test]
fn registry_covers_every_pitfall() {
    let registry = load_registry(&registry_path()).expect("registry should load without error");

    for pitfall in ALL_PITFALLS {
        let slug = pitfall_slug(pitfall);
        assert!(
            registry.contains_key(slug),
            "docs/pitfall_registry.json is missing an entry for pitfall {slug:?}"
        );
    }
    assert_eq!(
        registry.len(),
        ALL_PITFALLS.len(),
        "docs/pitfall_registry.json has an entry that doesn't match any Pitfall variant \
         (load_registry validates keys, so this indicates a count mismatch)"
    );
}

//! Checks that `README.md`'s `## Benchmarks` table matches the committed
//! `bench/opendataloader/results/results.json` it's rendered from, so the two can't
//! silently drift after a re-run of `scripts/run-opendataloader-bench.sh`.
//!
//! Gated behind the `stage3` cargo feature purely to reuse `serde_json` as a dev
//! dependency already wired up for that feature -- this test has no other relationship
//! to the Stage 3 pitfall corpus. Unlike the pitfall scoreboard's generated doc blocks
//! (`eval::scoreboard`), the benchmark table is hand-written, not spliced in from a
//! renderer: see `bench/opendataloader/README.md` for why (a `benchmark-results`
//! generated block would need a serde type for bench output inside the library crate,
//! plus three new entries kept in lockstep across `examples/stage3_scoreboard.rs` and
//! `tests/stage3_docs.rs`, where a typo panics CI). This test is the cheaper
//! alternative: it only guards that the two files agree, not how the table looks.

#![cfg(feature = "stage3")]

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct TableRow {
    engine: String,
    overall: String,
    nid: String,
    teds: String,
    mhs: String,
    s_per_doc: String,
}

/// Strips Markdown bold markers (`**pdfspatial**` -> `pdfspatial`) -- the README bolds
/// the headline `pdfspatial` row to distinguish it from the `pdfspatial (compact)` row.
fn strip_bold(cell: &str) -> String {
    cell.trim().trim_matches('*').trim().to_string()
}

/// Parses the `## Benchmarks` GFM table out of `readme`, keyed on its exact header row.
fn parse_benchmark_table(readme: &str) -> Vec<TableRow> {
    const HEADER: &str = "| Engine | Overall | Reading order (NID) | Table (TEDS) | Heading (MHS) | s/doc | License |";

    let header_line = readme
        .lines()
        .position(|line| line.trim() == HEADER)
        .unwrap_or_else(|| {
            panic!("README.md: benchmark table header not found -- expected:\n{HEADER}")
        });

    readme
        .lines()
        .skip(header_line + 2) // header row + the `|---|...` separator row
        .take_while(|line| line.trim_start().starts_with('|'))
        .map(|line| {
            let cells: Vec<&str> = line.trim().trim_matches('|').split('|').collect();
            assert_eq!(
                cells.len(),
                7,
                "README.md: malformed benchmark table row (expected 7 cells): {line:?}"
            );
            TableRow {
                engine: strip_bold(cells[0]),
                overall: strip_bold(cells[1]),
                nid: strip_bold(cells[2]),
                teds: strip_bold(cells[3]),
                mhs: strip_bold(cells[4]),
                s_per_doc: strip_bold(cells[5]).trim_end_matches('s').to_string(),
            }
        })
        .collect()
}

#[test]
fn readme_benchmark_table_matches_committed_results_json() {
    let root = workspace_root();
    let readme =
        std::fs::read_to_string(root.join("README.md")).expect("README.md should be readable");
    let results_path = root.join("bench/opendataloader/results/results.json");
    let results_raw = std::fs::read_to_string(&results_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", results_path.display()));
    let results: serde_json::Value =
        serde_json::from_str(&results_raw).expect("results.json should be valid JSON");

    let regen_hint = "regenerate both via `./scripts/run-opendataloader-bench.sh` (or, if \
                       results.json is already up to date, `./scripts/run-opendataloader-bench.sh \
                       --collect-only`), then update README.md's Benchmarks table by hand";

    let table_rows = parse_benchmark_table(&readme);
    let json_engines = results["engines"]
        .as_array()
        .expect("results.json's `engines` should be an array");

    assert_eq!(
        table_rows.len(),
        json_engines.len(),
        "README.md's benchmark table has {} row(s) but results.json has {} engine(s) -- {regen_hint}",
        table_rows.len(),
        json_engines.len(),
    );

    for (row, engine) in table_rows.iter().zip(json_engines.iter()) {
        let display_name = engine["display_name"]
            .as_str()
            .expect("each engine should have a display_name");
        assert_eq!(
            row.engine, display_name,
            "README.md benchmark table row order/name doesn't match results.json \
             (expected engines sorted by `overall` descending) -- {regen_hint}"
        );

        let fmt = |value: &serde_json::Value| -> String {
            format!("{:.3}", value.as_f64().expect("expected a numeric metric"))
        };

        assert_eq!(
            row.overall,
            fmt(&engine["overall"]),
            "engine {display_name:?}: overall mismatch -- {regen_hint}"
        );
        assert_eq!(
            row.nid,
            fmt(&engine["nid"]),
            "engine {display_name:?}: NID mismatch -- {regen_hint}"
        );
        assert_eq!(
            row.teds,
            fmt(&engine["teds"]),
            "engine {display_name:?}: TEDS mismatch -- {regen_hint}"
        );
        assert_eq!(
            row.mhs,
            fmt(&engine["mhs"]),
            "engine {display_name:?}: MHS mismatch -- {regen_hint}"
        );
        assert_eq!(
            row.s_per_doc,
            fmt(&engine["s_per_doc"]),
            "engine {display_name:?}: s/doc mismatch -- {regen_hint}"
        );
    }
}

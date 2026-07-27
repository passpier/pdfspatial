//! Mines a real, unpacked [DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1)
//! sample's most-reordered pages (per
//! [`pdfspatial_core::eval::rank_pages_by_reorder`]) into minimal-repro **draft** Stage 3
//! regression cases under `fixtures/`-compatible directories, using
//! [`pdfspatial_core::eval::minimize_reorder_repro`] to shrink each page down to the
//! handful of blocks that actually drive the reordering, and
//! [`pdfspatial_core::eval::corpus::write_draft_case`] to render them as reviewable JSON.
//!
//! ```sh
//! cargo run --example doclaynet_drafts --features doclaynet,stage3 \
//!   -- <coco.json> <cells_dir> --out <dir> [--top-n 5]
//! ```
//!
//! `<coco.json>` and `<cells_dir>` are the same DocLayNet-core inputs
//! [`pdfspatial_core::eval::doclaynet::load_sample`] expects (see `examples/doclaynet_eval.rs`).
//! With no positional arguments, both default to `$DOCLAYNET_DIR/COCO/val.json` and
//! `$DOCLAYNET_DIR/JSON`, matching `examples/doclaynet_mine.rs`.
//!
//! `--out` is required and is never defaulted to the workspace's own `fixtures/` --
//! every draft this emits is unreviewed (see [`pdfspatial_core::eval::corpus::DraftCase`]) and
//! must be reviewed, re-tagged, and moved into `fixtures/<pitfall>/` by hand before it
//! becomes a real regression case; see `fixtures/README.md`.
//!
//! For each of the top `--top-n` (default 5) most-reordered pages, this reconstructs the
//! page text-only from `pdf_cells` (no native PDFium needed), minimizes it, and either
//! writes a draft case or reports why the page was skipped (already ordered, or
//! duplicate-text blocks destroyed the reordering signal once deduplicated).

use pdfspatial_core::assemble::{Pitfall, RootCause};
use pdfspatial_core::eval::corpus::{DraftCase, write_draft_case};
use pdfspatial_core::eval::doclaynet::{
    document_from_cells, load_sample, mine_reading_order_failures,
};
use pdfspatial_core::eval::minimize_reorder_repro;
use std::path::PathBuf;
use std::process::ExitCode;

const DEFAULT_TOP_N: usize = 5;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut positional = Vec::new();
    let mut out_dir: Option<PathBuf> = None;
    let mut top_n = DEFAULT_TOP_N;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                let Some(value) = args.get(i + 1) else {
                    eprintln!("--out requires a directory");
                    return ExitCode::FAILURE;
                };
                out_dir = Some(PathBuf::from(value));
                i += 2;
            }
            "--top-n" => {
                let Some(value) = args.get(i + 1) else {
                    eprintln!("--top-n requires a number");
                    return ExitCode::FAILURE;
                };
                match value.parse() {
                    Ok(n) => top_n = n,
                    Err(_) => {
                        eprintln!("--top-n expects a positive integer, got {value:?}");
                        return ExitCode::FAILURE;
                    }
                }
                i += 2;
            }
            other => {
                positional.push(other.to_string());
                i += 1;
            }
        }
    }

    let Some(out_dir) = out_dir else {
        eprintln!(
            "usage: doclaynet_drafts <coco.json> <cells_dir> --out <dir> [--top-n N]\n\
             (or set DOCLAYNET_DIR to default the positional args to \
             $DOCLAYNET_DIR/COCO/val.json and $DOCLAYNET_DIR/JSON)"
        );
        return ExitCode::FAILURE;
    };

    let (coco_json, cells_dir) = match positional.as_slice() {
        [coco, cells] => (PathBuf::from(coco), PathBuf::from(cells)),
        [] => {
            let Some(base) = std::env::var_os("DOCLAYNET_DIR") else {
                eprintln!(
                    "usage: doclaynet_drafts <coco.json> <cells_dir> --out <dir> [--top-n N]\n\
                     (or set DOCLAYNET_DIR to default to $DOCLAYNET_DIR/COCO/val.json and \
                     $DOCLAYNET_DIR/JSON)"
                );
                return ExitCode::FAILURE;
            };
            let base = PathBuf::from(base);
            (base.join("COCO/val.json"), base.join("JSON"))
        }
        _ => {
            eprintln!("usage: doclaynet_drafts <coco.json> <cells_dir> --out <dir> [--top-n N]");
            return ExitCode::FAILURE;
        }
    };

    let sample = match load_sample(&coco_json, &cells_dir) {
        Ok(sample) => sample,
        Err(err) => {
            eprintln!("failed to load DocLayNet sample: {err}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "loaded {} page(s) from {}",
        sample.pages.len(),
        coco_json.display()
    );

    let mined = mine_reading_order_failures(&sample);
    let candidates = mined.into_iter().take(top_n);

    println!(
        "{:<6} {:>10} {:<24} {:>10} {:>10} {:>10}  result",
        "rank", "image_id", "file_name", "orig_blk", "min_blk", "distance"
    );

    let mut written = 0usize;
    let mut skipped = 0usize;

    for (position, mined_page) in candidates.enumerate() {
        let source_page = sample
            .pages
            .iter()
            .find(|p| p.image_id == mined_page.image_id)
            .expect("mined page must come from the loaded sample");
        let document = document_from_cells(source_page);
        let page = &document.pages[0];

        match minimize_reorder_repro(page) {
            None => {
                skipped += 1;
                println!(
                    "{:<6} {:>10} {:<24} {:>10} {:>10} {:>10}  skipped (no repro after dedup)",
                    position + 1,
                    mined_page.image_id,
                    mined_page.file_name,
                    page.blocks.len(),
                    "-",
                    "-"
                );
            }
            Some(minimized) => {
                let id = format!(
                    "mined-{}",
                    mined_page
                        .file_name
                        .rsplit('/')
                        .next()
                        .unwrap_or(&mined_page.file_name)
                        .trim_end_matches(".png")
                );
                let description = format!(
                    "DRAFT: mined from DocLayNet image_id {} ({}), ranked #{} by reorder_edit_distance \
                     ({} block moves on the full page). Pitfall/root_cause below are the mining \
                     signal's own hypothesis (reordering => multi_column/ordering) -- review and \
                     re-tag before promoting out of draft.",
                    mined_page.image_id,
                    mined_page.file_name,
                    position + 1,
                    mined_page.rank.reorder_edit_distance,
                );
                let draft = DraftCase {
                    id: &id,
                    pitfall: Pitfall::MultiColumn,
                    root_cause: RootCause::Ordering,
                    description: &description,
                    page: &minimized.page,
                };

                match write_draft_case(&out_dir, &draft) {
                    Ok(path) => {
                        written += 1;
                        println!(
                            "{:<6} {:>10} {:<24} {:>10} {:>10} {:>10}  wrote {}",
                            position + 1,
                            mined_page.image_id,
                            mined_page.file_name,
                            minimized.original_block_count,
                            minimized.page.blocks.len(),
                            minimized.reorder_edit_distance,
                            path.display()
                        );
                    }
                    Err(err) => {
                        eprintln!("failed to write draft for {}: {err}", mined_page.file_name);
                        return ExitCode::FAILURE;
                    }
                }
            }
        }
    }

    println!();
    println!("{written} draft case(s) written, {skipped} page(s) skipped.");
    println!(
        "Review each draft in {} before moving it into fixtures/<pitfall>/ -- see \
         fixtures/README.md's \"Mining drafts from a real sample\" section.",
        out_dir.display()
    );

    ExitCode::SUCCESS
}

//! Ranks a real, unpacked [DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1)
//! sample's pages by reading-order **reordering severity** -- the Stage 3
//! failure-cluster prioritization signal described in
//! [`pdfspatial_core::eval::rank_pages_by_reorder`] -- and prints a ranked "mine these
//! first" report.
//!
//! ```sh
//! cargo run --example doclaynet_mine --features doclaynet -- <coco.json> <cells_dir> [--pdfium <pdf_dir>]
//! ```
//!
//! `<coco.json>` and `<cells_dir>` are the same DocLayNet-core inputs
//! [`pdfspatial_core::eval::doclaynet::load_sample`] expects (see `examples/doclaynet_eval.rs`
//! for the exact expected shape). With no arguments, both paths default to
//! `$DOCLAYNET_DIR/COCO/val.json` and `$DOCLAYNET_DIR/JSON`.
//!
//! # What this ranks, and why
//!
//! DocLayNet ships region bounding boxes, not an authored reading-order sequence, so
//! there is no gold order to score pages against outside the hand-authored `fixtures/`
//! corpus. Instead, each page is scored by how many block moves
//! `assemble::assemble_reading_order`'s XY-cut applied relative to the naive
//! top-to-bottom, left-to-right order Stage 1 emits. A page barely touched by XY-cut is
//! one where the naive order was already plausible; a page XY-cut heavily reorders is
//! exactly the kind of multi-column/complex layout Stage 1's reading-order metric is
//! expected to fail on -- the highest-value page to mine into a `fixtures/`
//! minimal-repro regression case next (see `fixtures/README.md`).
//!
//! By default, pages are reconstructed text-only from DocLayNet's own `pdf_cells` (via
//! `document_from_cells`), so this needs no native PDFium. Passing `--pdfium <pdf_dir>`
//! instead ranks the real Stage 1 extraction (`extract_baseline`) against
//! `<pdf_dir>/{image_file_stem}.pdf`, closer to the real pipeline but requiring
//! `PDFSPATIAL_PDFIUM_LIB` to be set. Passing `--grouped` instead of `--pdfium` still
//! needs no PDFium: it reconstructs each page via `document_from_cells_grouped`, running
//! Stage 1's real word/line/block grouping over DocLayNet's cells rather than treating
//! each cell as its own block (see `eval::doclaynet`'s module docs for the trade-off).
//! `--pdfium` and `--grouped` are mutually exclusive.

use pdfspatial_core::eval::doclaynet::{
    DocLayNetPage, load_sample, mine_reading_order_failures, mine_reading_order_failures_grouped,
};
use pdfspatial_core::eval::{PageReorderRank, rank_pages_by_reorder};
use pdfspatial_core::{PdfiumSource, extract_baseline_with_source};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// One ranked row in the mining report, unified across the text-only and `--pdfium`
/// paths (which differ only in how each page's `Document` is reconstructed).
struct Row {
    image_id: i64,
    file_name: String,
    rank: PageReorderRank,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut positional = Vec::new();
    let mut pdf_dir: Option<PathBuf> = None;
    let mut grouped = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pdfium" => {
                let Some(value) = args.get(i + 1) else {
                    eprintln!("--pdfium requires a directory");
                    return ExitCode::FAILURE;
                };
                pdf_dir = Some(PathBuf::from(value));
                i += 2;
            }
            "--grouped" => {
                grouped = true;
                i += 1;
            }
            other => {
                positional.push(other.to_string());
                i += 1;
            }
        }
    }

    if pdf_dir.is_some() && grouped {
        eprintln!("--pdfium and --grouped are mutually exclusive");
        return ExitCode::FAILURE;
    }

    let (coco_json, cells_dir) = match positional.as_slice() {
        [coco, cells] => (PathBuf::from(coco), PathBuf::from(cells)),
        [] => {
            let Some(base) = std::env::var_os("DOCLAYNET_DIR") else {
                eprintln!(
                    "usage: doclaynet_mine <coco.json> <cells_dir> [--pdfium <pdf_dir>] [--grouped]\n\
                     (or set DOCLAYNET_DIR to default to $DOCLAYNET_DIR/COCO/val.json and $DOCLAYNET_DIR/JSON)"
                );
                return ExitCode::FAILURE;
            };
            let base = PathBuf::from(base);
            (base.join("COCO/val.json"), base.join("JSON"))
        }
        _ => {
            eprintln!(
                "usage: doclaynet_mine <coco.json> <cells_dir> [--pdfium <pdf_dir>] [--grouped]"
            );
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

    let rows: Vec<Row> = match &pdf_dir {
        None if grouped => mine_reading_order_failures_grouped(&sample)
            .into_iter()
            .map(|mined| Row {
                image_id: mined.image_id,
                file_name: mined.file_name,
                rank: mined.rank,
            })
            .collect(),
        None => mine_reading_order_failures(&sample)
            .into_iter()
            .map(|mined| Row {
                image_id: mined.image_id,
                file_name: mined.file_name,
                rank: mined.rank,
            })
            .collect(),
        Some(pdf_dir) => match rank_via_pdfium(&sample.pages, pdf_dir) {
            Ok(rows) => rows,
            Err(err) => {
                eprintln!("failed to rank via PDFium: {err}");
                return ExitCode::FAILURE;
            }
        },
    };

    print_report(&rows);
    ExitCode::SUCCESS
}

/// Runs the real Stage 1 extraction against each page's source PDF and ranks the
/// resulting `Document`s by reordering severity via `rank_pages_by_reorder` directly --
/// no gold reading order is needed on this path either.
fn rank_via_pdfium(pages: &[DocLayNetPage], pdf_dir: &Path) -> Result<Vec<Row>, String> {
    let source = PdfiumSource::default();
    let mut rows = Vec::with_capacity(pages.len());

    for page in pages {
        let stem = Path::new(&page.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&page.file_name);
        let pdf_path = pdf_dir.join(format!("{stem}.pdf"));

        let document = extract_baseline_with_source(&pdf_path, &source)
            .map_err(|err| format!("{}: {err}", pdf_path.display()))?;
        if document.pages.is_empty() {
            return Err(format!("{}: no pages extracted", pdf_path.display()));
        }

        let Some(rank) = rank_pages_by_reorder(&document).into_iter().next() else {
            continue;
        };
        rows.push(Row {
            image_id: page.image_id,
            file_name: page.file_name.clone(),
            rank,
        });
    }

    rows.sort_by(|a, b| {
        b.rank
            .reorder_edit_distance
            .cmp(&a.rank.reorder_edit_distance)
            .then(a.image_id.cmp(&b.image_id))
    });
    Ok(rows)
}

fn print_report(rows: &[Row]) {
    println!(
        "Stage 3 mining report ({} page(s)), most-reordered first",
        rows.len()
    );
    println!(
        "(reorder_dist = block moves assemble_reading_order applied vs. the naive extraction order --"
    );
    println!(
        "no DocLayNet gold reading order exists, so this ranks reordering severity, not accuracy)"
    );
    println!();
    println!(
        "{:<6} {:>10} {:<24} {:>8} {:>12}",
        "rank", "image_id", "file_name", "blocks", "reorder_dist"
    );
    for (position, row) in rows.iter().enumerate() {
        println!(
            "{:<6} {:>10} {:<24} {:>8} {:>12}",
            position + 1,
            row.image_id,
            row.file_name,
            row.rank.block_count,
            row.rank.reorder_edit_distance
        );
    }
}

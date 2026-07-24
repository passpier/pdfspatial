//! Scores `pdfspatial_core::layout::classify_regions` against a real, unpacked
//! [DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1) sample
//! and prints the resulting Stage 2 dashboard.
//!
//! ```sh
//! cargo run --example doclaynet_eval --features doclaynet -- <coco.json> <cells_dir> [--iou 0.5] [--pdfium <pdf_dir>]
//! ```
//!
//! `<coco.json>` is a DocLayNet-core COCO object-detection file (`images` /
//! `categories` / `annotations`), and `<cells_dir>` must contain one
//! `{image_file_stem}.cells.json` file per image listed in it — see
//! [`pdfspatial_core::eval::doclaynet::load_sample`] for the exact expected shape; a
//! real unpacked DocLayNet-core `JSON/` directory may need renaming to match.
//!
//! With no arguments, both paths default to `$DOCLAYNET_DIR/COCO/val.json` and
//! `$DOCLAYNET_DIR/JSON`.
//!
//! By default, predictions are produced text-only — by reconstructing a `Document`
//! from DocLayNet's own `pdf_cells` and running `classify_regions` on it, so this needs
//! no native PDFium. Passing `--pdfium <pdf_dir>` instead runs the real Stage 1
//! extraction (`extract_baseline`) against `<pdf_dir>/{image_file_stem}.pdf` and scores
//! that against ground truth rescaled into the PDF's own point space — closer to the
//! real pipeline, but requires `PDFSPATIAL_PDFIUM_LIB` to be set.

use pdfspatial_core::eval::doclaynet::{DocLayNetPage, load_sample, predict_regions_textonly};
use pdfspatial_core::eval::{Stage2Report, evaluate_pages};
use pdfspatial_core::layout::{Region, classify_regions};
use pdfspatial_core::{BBox, PdfiumSource, extract_baseline_with_source};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut positional = Vec::new();
    let mut iou_threshold = 0.5_f32;
    let mut pdf_dir: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--iou" => {
                let Some(value) = args.get(i + 1) else {
                    eprintln!("--iou requires a value");
                    return ExitCode::FAILURE;
                };
                match value.parse() {
                    Ok(v) => iou_threshold = v,
                    Err(_) => {
                        eprintln!("--iou value must be a float, got {value:?}");
                        return ExitCode::FAILURE;
                    }
                }
                i += 2;
            }
            "--pdfium" => {
                let Some(value) = args.get(i + 1) else {
                    eprintln!("--pdfium requires a directory");
                    return ExitCode::FAILURE;
                };
                pdf_dir = Some(PathBuf::from(value));
                i += 2;
            }
            other => {
                positional.push(other.to_string());
                i += 1;
            }
        }
    }

    let (coco_json, cells_dir) = match positional.as_slice() {
        [coco, cells] => (PathBuf::from(coco), PathBuf::from(cells)),
        [] => {
            let Some(base) = std::env::var_os("DOCLAYNET_DIR") else {
                eprintln!(
                    "usage: doclaynet_eval <coco.json> <cells_dir> [--iou 0.5] [--pdfium <pdf_dir>]\n\
                     (or set DOCLAYNET_DIR to default to $DOCLAYNET_DIR/COCO/val.json and $DOCLAYNET_DIR/JSON)"
                );
                return ExitCode::FAILURE;
            };
            let base = PathBuf::from(base);
            (base.join("COCO/val.json"), base.join("JSON"))
        }
        _ => {
            eprintln!(
                "usage: doclaynet_eval <coco.json> <cells_dir> [--iou 0.5] [--pdfium <pdf_dir>]"
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

    let report = match &pdf_dir {
        None => {
            let pages: Vec<(Vec<Region>, Vec<Region>)> = sample
                .pages
                .iter()
                .map(|page| (predict_regions_textonly(page), page.ground_truth.clone()))
                .collect();
            evaluate_pages(&pages, iou_threshold)
        }
        Some(pdf_dir) => match evaluate_via_pdfium(&sample.pages, pdf_dir, iou_threshold) {
            Ok(report) => report,
            Err(err) => {
                eprintln!("failed to evaluate via PDFium: {err}");
                return ExitCode::FAILURE;
            }
        },
    };

    print_report(&report);
    ExitCode::SUCCESS
}

/// Runs the real Stage 1 extraction against each page's source PDF and scores it
/// against ground truth rescaled from DocLayNet's COCO pixel frame into that PDF's own
/// point space (`scale = pdf_dimension / coco_dimension`, applied per axis).
fn evaluate_via_pdfium(
    pages: &[DocLayNetPage],
    pdf_dir: &Path,
    iou_threshold: f32,
) -> Result<Stage2Report, String> {
    let source = PdfiumSource::default();
    let mut evaluated = Vec::with_capacity(pages.len());

    for page in pages {
        let stem = Path::new(&page.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&page.file_name);
        let pdf_path = pdf_dir.join(format!("{stem}.pdf"));

        let document = extract_baseline_with_source(&pdf_path, &source)
            .map_err(|err| format!("{}: {err}", pdf_path.display()))?;
        let Some(pdf_page) = document.pages.first() else {
            return Err(format!("{}: no pages extracted", pdf_path.display()));
        };

        let scale_x = pdf_page.width / page.width;
        let scale_y = pdf_page.height / page.height;
        let ground_truth: Vec<Region> = page
            .ground_truth
            .iter()
            .map(|region| Region {
                bbox: rescale(region.bbox, scale_x, scale_y),
                ..region.clone()
            })
            .collect();

        let predicted = classify_regions(&document);
        evaluated.push((predicted, ground_truth));
    }

    Ok(evaluate_pages(&evaluated, iou_threshold))
}

fn rescale(bbox: BBox, scale_x: f32, scale_y: f32) -> BBox {
    BBox {
        left: bbox.left * scale_x,
        right: bbox.right * scale_x,
        bottom: bbox.bottom * scale_y,
        top: bbox.top * scale_y,
    }
}

fn print_report(report: &Stage2Report) {
    println!(
        "Stage 2 dashboard ({} page(s), IoU >= {})",
        report.page_count, report.iou_threshold
    );
    println!();
    println!(
        "{:<16} {:>8} {:>10} {:>10}",
        "class", "support", "F1", "mean GIoU"
    );
    for (class, support) in &report.support {
        let f1 = report.region_f1.get(class).copied().unwrap_or(0.0);
        let giou = report
            .mean_giou_per_class
            .get(class)
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "--".to_string());
        println!("{class:<16?} {support:>8} {f1:>10.3} {giou:>10}");
    }
    println!();
    println!("macro region F1:   {:.3}", report.macro_region_f1);
    println!("mean GIoU:         {:.3}", report.mean_giou);
    println!("footnote F1:       {:.3}", report.footnote_f1);
    println!("page-header F1:    {:.3}", report.page_header_f1);
    println!("page-footer F1:    {:.3}", report.page_footer_f1);
    match report.teds {
        Some(score) => println!("TEDS:              {score:.3}"),
        None => println!("TEDS:              n/a (no table-structure predictions)"),
    }
}

//! Writes the small, hand-authored PDF fixtures behind the Stage 3 corpus's
//! extraction-layer (`geometric`, plus `cross_page_continuation`'s `ordering`)
//! pitfalls, then extracts each one through the real Stage 1 pipeline
//! ([`pdfspatial_core::extract_baseline`]) and freezes the result as a corpus case via
//! [`pdfspatial_core::eval::corpus::snapshot_case_json`].
//!
//! ```sh
//! cargo run --example stage3_pdf_cases --features stage3
//! ```
//!
//! Both steps are idempotent: re-running with unchanged PDF content and unchanged
//! `CASES` metadata below regenerates byte-identical output. Writing the PDFs needs no
//! native library (they're built directly, object-by-object, the same way
//! `tests/fixtures/single_column.pdf` was hand-written); refreshing the snapshots needs
//! PDFium, so pass `--pdfs-only` to skip that step in an environment without it set up
//! (see the crate README's Prerequisites section).
//!
//! If a target case file under `--out` (default the workspace's own `fixtures/`)
//! already exists, this loads it first and carries its `description` and `expected`
//! forward untouched -- regenerating a snapshot (e.g. after a PDFium version bump or a
//! wording tweak to the PDF text) must never clobber a human-reviewed `expected` block.
//! Only the frozen `page`/`pages` geometry is replaced. See `fixtures/README.md`'s
//! "PDF-backed cases" section.
//!
//! `--pdf-dir` overrides where the PDFs are written/read from (default
//! `crates/pdfspatial-core/tests/fixtures/stage3/`).

use pdfspatial_core::assemble::{Pitfall, RootCause};
use pdfspatial_core::eval::corpus::{
    ExpectedBehavior, ExpectedClass, SnapshotCase, load_case, snapshot_case_json,
};
use pdfspatial_core::layout::RegionClass;
use pdfspatial_core::{PdfiumSource, extract_baseline_with_source};
use std::path::PathBuf;
use std::process::ExitCode;

// --- Minimal PDF object writer -----------------------------------------------------
//
// Builds a PDF 1.7 file object-by-object (no compression, literal `Tm`/`Tj` content
// streams), the same shape `tests/fixtures/single_column.pdf` was hand-written in --
// this just automates computing the `xref` byte offsets, which is the tedious part to
// keep correct by hand across edits.

struct PdfWriter {
    objects: Vec<Vec<u8>>,
}

impl PdfWriter {
    fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    /// Reserves the next object number without content yet; fill it in later with
    /// [`Self::set_object`]. Used for `/Pages`, whose `/Kids` array can only be known
    /// once every page object has been added.
    fn reserve(&mut self) -> usize {
        self.objects.push(Vec::new());
        self.objects.len()
    }

    fn set_object(&mut self, num: usize, body: &str) {
        self.objects[num - 1] = body.as_bytes().to_vec();
    }

    fn add_object(&mut self, body: &str) -> usize {
        let num = self.reserve();
        self.set_object(num, body);
        num
    }

    /// A content-stream object: `<< /Length N >> stream ... endstream`.
    fn add_stream(&mut self, content: &str) -> usize {
        let num = self.reserve();
        let mut body = format!("<< /Length {} >>\nstream\n", content.len());
        body.push_str(content);
        body.push_str("\nendstream");
        self.set_object(num, &body);
        num
    }

    fn finish(&self, root: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");

        let mut offsets = Vec::with_capacity(self.objects.len());
        for (i, body) in self.objects.iter().enumerate() {
            offsets.push(buf.len());
            buf.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
            buf.extend_from_slice(body);
            buf.extend_from_slice(b"\nendobj\n");
        }

        let xref_offset = buf.len();
        buf.extend_from_slice(format!("xref\n0 {}\n", self.objects.len() + 1).as_bytes());
        buf.extend_from_slice(b"0000000000 65535 f \n");
        for off in &offsets {
            buf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        buf.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root {} 0 R >>\nstartxref\n{}\n%%EOF",
                self.objects.len() + 1,
                root,
                xref_offset
            )
            .as_bytes(),
        );

        buf
    }
}

/// Builds a single- or multi-page PDF where every page shares the same font resource
/// dict body (`font_body`, e.g. a plain Helvetica `/Type1` dict or one with a custom
/// `/Encoding`) and each page's own content stream and media box.
fn build_pdf(font_body: &str, pages: &[(&str, [f32; 2])]) -> Vec<u8> {
    let mut w = PdfWriter::new();
    let pages_num = w.reserve();
    let font_num = w.add_object(font_body);

    let mut kids = Vec::new();
    for (content, [width, height]) in pages {
        let content_num = w.add_stream(content);
        let page_num = w.add_object(&format!(
            "<< /Type /Page /Parent {pages_num} 0 R /MediaBox [0 0 {width} {height}] \
             /Resources << /Font << /F1 {font_num} 0 R >> >> /Contents {content_num} 0 R >>"
        ));
        kids.push(page_num);
    }

    let kids_refs: String = kids
        .iter()
        .map(|n| format!("{n} 0 R"))
        .collect::<Vec<_>>()
        .join(" ");
    w.set_object(
        pages_num,
        &format!(
            "<< /Type /Pages /Kids [{kids_refs}] /Count {} >>",
            kids.len()
        ),
    );

    let catalog_num = w.add_object(&format!("<< /Type /Catalog /Pages {pages_num} 0 R >>"));
    w.finish(catalog_num)
}

const HELVETICA: &str =
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>";

/// A `/Type1` Helvetica font whose `/Encoding /Differences` swaps the glyph names for
/// `A`/`B` and omits `/ToUnicode` -- with no ToUnicode CMap, a text extractor has to
/// fall back to resolving glyph names through the Adobe Glyph List, so swapping the
/// names swaps the recovered Unicode too, without needing a real embedded font
/// program. This reproduces the `embedded_font` pitfall's dropped/garbled glyphs using
/// only standard fonts, since a real custom/CID-keyed embedded font isn't something a
/// hand-written object stream can produce.
const SWAPPED_ENCODING_HELVETICA: &str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
     /Encoding << /Type /Encoding /BaseEncoding /WinAnsiEncoding \
     /Differences [65 /B 66 /A] >> >>";

struct CaseSpec {
    id: &'static str,
    pdf_name: &'static str,
    pitfall: Pitfall,
    root_cause: RootCause,
    description: &'static str,
    font_body: &'static str,
    pages: &'static [(&'static str, [f32; 2])],
    expected: fn() -> ExpectedBehavior,
}

const PAGE: [f32; 2] = [612.0, 792.0];

fn super_subscript_expected() -> ExpectedBehavior {
    ExpectedBehavior {
        reading_order: Some(vec!["x2 + y2 = z2".to_string()]),
        classes: Vec::new(),
        requires_extraction_fix: true,
    }
}

fn rotated_text_expected() -> ExpectedBehavior {
    ExpectedBehavior {
        reading_order: Some(vec![
            "Body paragraph next to a rotated sidebar label.".to_string(),
            "SIDEBAR LABEL".to_string(),
        ]),
        classes: Vec::new(),
        requires_extraction_fix: true,
    }
}

fn embedded_font_expected() -> ExpectedBehavior {
    ExpectedBehavior {
        reading_order: Some(vec!["AB swapped by a Differences encoding.".to_string()]),
        classes: Vec::new(),
        requires_extraction_fix: true,
    }
}

fn overlapping_text_expected() -> ExpectedBehavior {
    ExpectedBehavior {
        reading_order: Some(vec![
            "Overlapping text stress test line one.".to_string(),
            "Overlapping text stress test line two.".to_string(),
            "DRAFT".to_string(),
        ]),
        classes: Vec::new(),
        requires_extraction_fix: true,
    }
}

fn multi_line_table_cell_expected() -> ExpectedBehavior {
    ExpectedBehavior {
        reading_order: None,
        classes: vec![
            ExpectedClass {
                block_text: "Item description that\nspans two table lines".to_string(),
                class: RegionClass::Table,
            },
            ExpectedClass {
                block_text: "42".to_string(),
                class: RegionClass::Table,
            },
        ],
        requires_extraction_fix: true,
    }
}

fn cross_page_continuation_expected() -> ExpectedBehavior {
    ExpectedBehavior {
        reading_order: Some(vec![
            "This sentence deliberately continues on the next page without any heading to \
             signal a restart."
                .to_string(),
        ]),
        classes: Vec::new(),
        requires_extraction_fix: true,
    }
}

fn cases() -> Vec<CaseSpec> {
    vec![
        CaseSpec {
            id: "super_subscript-formula-baseline-clustering",
            pdf_name: "super_subscript.pdf",
            pitfall: Pitfall::SuperSubscript,
            root_cause: RootCause::Geometric,
            description: "A short algebraic formula whose exponents are set at 7pt on a \
                baseline raised ~6pt above the 12pt body characters (real superscript \
                positioning, not a synthetic already-grouped block). Word grouping keeps \
                the whole formula on one line, but treats each exponent as its own word \
                separated by a spurious space ('x 2 + y 2 = z 2') instead of attaching it \
                directly to its base ('x2 + y2 = z2') -- Stage 1 has no font-size-aware \
                notion of \"this word is a superscript of the previous one\".",
            font_body: HELVETICA,
            pages: &[(
                "BT\n/F1 12 Tf\n1 0 0 1 72 700 Tm\n(x) Tj\n\
                 /F1 7 Tf\n1 0 0 1 82 706 Tm\n(2) Tj\n\
                 /F1 12 Tf\n1 0 0 1 90 700 Tm\n( + y) Tj\n\
                 /F1 7 Tf\n1 0 0 1 116 706 Tm\n(2) Tj\n\
                 /F1 12 Tf\n1 0 0 1 124 700 Tm\n( = z) Tj\n\
                 /F1 7 Tf\n1 0 0 1 150 706 Tm\n(2) Tj\nET",
                PAGE,
            )],
            expected: super_subscript_expected,
        },
        CaseSpec {
            id: "rotated_text-vertical-sidebar-label",
            pdf_name: "rotated_text.pdf",
            pitfall: Pitfall::RotatedText,
            root_cause: RootCause::Geometric,
            description: "A normal horizontal paragraph next to a 90-degree-rotated \
                sidebar label (`0 1 -1 0 x y Tm`). PDFium reports each rotated glyph's own \
                tall-and-narrow (page-space) bounding box; Stage 1's line grouping assumes \
                wide-and-flat horizontal runs, so it shatters the vertical label into one \
                single-character line per glyph (occasionally two, when adjacent glyphs \
                happen to overlap enough), instead of the one 'SIDEBAR LABEL' line a \
                rotation-aware grouping pass would recover.",
            font_body: HELVETICA,
            pages: &[(
                "BT\n/F1 12 Tf\n1 0 0 1 72 700 Tm\n\
                 (Body paragraph next to a rotated sidebar label.) Tj\nET\n\
                 BT\n/F1 10 Tf\n0 1 -1 0 560 620 Tm\n(SIDEBAR LABEL) Tj\nET",
                PAGE,
            )],
            expected: rotated_text_expected,
        },
        CaseSpec {
            id: "embedded_font-differences-encoding-swap",
            pdf_name: "embedded_font.pdf",
            pitfall: Pitfall::EmbeddedFont,
            root_cause: RootCause::Geometric,
            description: "A font whose /Encoding /Differences remaps the glyph names for \
                codes 65/66 (normally A/B) to B/A, with no /ToUnicode CMap -- the same \
                shape a subsetted/CID-keyed embedded font takes when its glyph-name-based \
                Unicode recovery doesn't match the font's own internal glyph order. Stage 1 \
                has no CMap-aware fallback, so extraction recovers the swapped letters.",
            font_body: SWAPPED_ENCODING_HELVETICA,
            pages: &[(
                "BT\n/F1 12 Tf\n1 0 0 1 72 700 Tm\n\
                 (AB swapped by a Differences encoding.) Tj\nET",
                PAGE,
            )],
            expected: embedded_font_expected,
        },
        CaseSpec {
            id: "overlapping_text-watermark-over-body",
            pdf_name: "overlapping_text.pdf",
            pitfall: Pitfall::OverlappingText,
            root_cause: RootCause::Geometric,
            description: "A two-line body paragraph with a large 'DRAFT' watermark text \
                object drawn across the same vertical band as the second line, both real \
                z-ordered text objects in one content stream (not a synthetic pre-grouped \
                block). Stage 1's line grouping has no z-order/overlap awareness, so the \
                watermark's glyphs get spliced into the middle of the second line by x \
                position ('Overlapping DRAFT text stress test line two.') instead of staying \
                a separate overlay.",
            font_body: HELVETICA,
            pages: &[(
                "BT\n/F1 12 Tf\n1 0 0 1 72 700 Tm\n\
                 (Overlapping text stress test line one.) Tj\n\
                 1 0 0 1 72 686 Tm\n(Overlapping text stress test line two.) Tj\nET\n\
                 BT\n/F1 40 Tf\n1 0 0 1 120 685 Tm\n(DRAFT) Tj\nET",
                PAGE,
            )],
            expected: overlapping_text_expected,
        },
        CaseSpec {
            id: "multi_line_table_cell-wrapped-left-cell",
            pdf_name: "multi_line_table_cell.pdf",
            pitfall: Pitfall::MultiLineTableCell,
            root_cause: RootCause::Classification,
            description: "A borderless two-column row: the left cell wraps onto two \
                closely-spaced lines while the right cell ('42') is a single line at the \
                same row as the left cell's first line. Stage 1's block grouping only \
                clusters by vertical gaps, with no notion of column gutters at the block \
                level, so the right cell's single line gets merged straight into the left \
                cell's block as a third line instead of staying its own cell -- losing the \
                row/column structure entirely, not just the (still-unimplemented) Table \
                classification on top of it.",
            font_body: HELVETICA,
            pages: &[(
                "BT\n/F1 10 Tf\n1 0 0 1 72 700 Tm\n(Item description that) Tj\n\
                 1 0 0 1 72 686 Tm\n(spans two table lines) Tj\n\
                 1 0 0 1 340 700 Tm\n(42) Tj\nET",
                PAGE,
            )],
            expected: multi_line_table_cell_expected,
        },
        CaseSpec {
            id: "cross_page_continuation-sentence-split-across-pages",
            pdf_name: "cross_page_continuation.pdf",
            pitfall: Pitfall::CrossPageContinuation,
            root_cause: RootCause::Ordering,
            description: "A two-page document whose single sentence is split across the \
                page boundary with no heading to signal a restart on page two. \
                `assemble_reading_order` only reorders blocks within a page, so it has no \
                mechanism to notice or stitch a sentence that continues onto the next page.",
            font_body: HELVETICA,
            pages: &[
                (
                    "BT\n/F1 12 Tf\n1 0 0 1 72 700 Tm\n\
                     (This sentence deliberately continues on the next) Tj\nET",
                    PAGE,
                ),
                (
                    "BT\n/F1 12 Tf\n1 0 0 1 72 700 Tm\n\
                     (page without any heading to signal a restart.) Tj\nET",
                    PAGE,
                ),
            ],
            expected: cross_page_continuation_expected,
        },
    ]
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut pdf_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stage3");
    let mut out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("fixtures");
    let mut pdfs_only = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pdf-dir" => {
                let Some(value) = args.get(i + 1) else {
                    eprintln!("--pdf-dir requires a directory");
                    return ExitCode::FAILURE;
                };
                pdf_dir = PathBuf::from(value);
                i += 2;
            }
            "--out" => {
                let Some(value) = args.get(i + 1) else {
                    eprintln!("--out requires a directory");
                    return ExitCode::FAILURE;
                };
                out_dir = PathBuf::from(value);
                i += 2;
            }
            "--pdfs-only" => {
                pdfs_only = true;
                i += 1;
            }
            other => {
                eprintln!("unrecognized argument {other:?}");
                eprintln!("usage: stage3_pdf_cases [--pdf-dir <dir>] [--out <dir>] [--pdfs-only]");
                return ExitCode::FAILURE;
            }
        }
    }

    if let Err(err) = std::fs::create_dir_all(&pdf_dir) {
        eprintln!("failed to create {}: {err}", pdf_dir.display());
        return ExitCode::FAILURE;
    }

    let specs = cases();

    println!(
        "Writing {} PDF fixture(s) to {}",
        specs.len(),
        pdf_dir.display()
    );
    for spec in &specs {
        let bytes = build_pdf(spec.font_body, spec.pages);
        let path = pdf_dir.join(spec.pdf_name);
        if let Err(err) = std::fs::write(&path, &bytes) {
            eprintln!("failed to write {}: {err}", path.display());
            return ExitCode::FAILURE;
        }
        println!("  wrote {}", path.display());
    }

    if pdfs_only {
        println!("--pdfs-only set; skipping snapshot extraction.");
        return ExitCode::SUCCESS;
    }

    println!(
        "\nExtracting via PDFium and writing corpus snapshots to {}",
        out_dir.display()
    );
    let source = PdfiumSource::default();
    let mut failures = 0usize;

    for spec in &specs {
        let pdf_path = pdf_dir.join(spec.pdf_name);
        let document = match extract_baseline_with_source(&pdf_path, &source) {
            Ok(document) => document,
            Err(err) => {
                eprintln!("  failed to extract {}: {err}", pdf_path.display());
                failures += 1;
                continue;
            }
        };

        let case_path = out_dir
            .join(pdfspatial_core::eval::corpus::pitfall_slug(spec.pitfall))
            .join(format!("{}.json", spec.id));

        let (description, expected) = if case_path.exists() {
            match load_case(&case_path) {
                Ok(existing) => (existing.description, existing.expected),
                Err(err) => {
                    eprintln!(
                        "  {} exists but failed to load ({err}); refusing to overwrite \
                         a possibly hand-edited file -- fix or delete it first",
                        case_path.display()
                    );
                    failures += 1;
                    continue;
                }
            }
        } else {
            (spec.description.to_string(), (spec.expected)())
        };

        // Workspace-relative path, independent of where this example was invoked from.
        let source_pdf = format!(
            "crates/pdfspatial-core/tests/fixtures/stage3/{}",
            spec.pdf_name
        );

        let snapshot = SnapshotCase {
            id: spec.id,
            pitfall: spec.pitfall,
            root_cause: spec.root_cause,
            description: &description,
            source_pdf: &source_pdf,
            pages: &document.pages,
            expected: &expected,
        };

        let json = match snapshot_case_json(&snapshot) {
            Ok(json) => json,
            Err(err) => {
                eprintln!("  failed to render snapshot for {}: {err}", spec.id);
                failures += 1;
                continue;
            }
        };

        if let Some(parent) = case_path.parent() {
            if let Err(err) = std::fs::create_dir_all(parent) {
                eprintln!("  failed to create {}: {err}", parent.display());
                failures += 1;
                continue;
            }
        }
        if let Err(err) = std::fs::write(&case_path, format!("{json}\n")) {
            eprintln!("  failed to write {}: {err}", case_path.display());
            failures += 1;
            continue;
        }
        println!("  wrote {}", case_path.display());
    }

    if failures > 0 {
        eprintln!("\n{failures} case(s) failed -- see above.");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

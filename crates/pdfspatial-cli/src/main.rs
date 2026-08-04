//! `pdfspatial`: a command-line PDF-to-Markdown converter built on
//! [`pdfspatial_core`] -- spatially-grounded, deterministic, OCR-free.
//!
//! ```sh
//! pdfspatial input.pdf                       # Markdown to stdout
//! pdfspatial --out out/ a.pdf b.pdf some_dir  # batch mode, one .md per PDF
//! ```
//!
//! See [`args::USAGE`] for the full flag surface. Requires the native PDFium library to
//! be available at runtime -- set `PDFSPATIAL_PDFIUM_LIB`, or place it on the OS's
//! dynamic-library search path. See the crate README for setup instructions.

#![warn(missing_docs)]
#![warn(clippy::all)]

mod args;

use args::{Config, ParseOutcome};
use pdfspatial_core::{extract_baseline, serialize};
use rayon::prelude::*;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let config = match args::parse(&args) {
        Ok(ParseOutcome::Help) => {
            print!("{}", args::USAGE);
            return ExitCode::SUCCESS;
        }
        Ok(ParseOutcome::Version) => {
            println!("pdfspatial {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        Ok(ParseOutcome::Run(config)) => config,
        Err(message) => {
            eprintln!("pdfspatial: {message}\n\n{}", args::USAGE);
            return ExitCode::FAILURE;
        }
    };

    let files = match resolve_inputs(&config.inputs) {
        Ok(files) => files,
        Err(message) => {
            eprintln!("pdfspatial: {message}");
            return ExitCode::FAILURE;
        }
    };

    // A single input PDF with no --out prints straight to stdout, matching
    // examples/basic_extract.rs's convention -- no directory is created for the
    // common one-off "convert this file and look at it" case.
    if config.out.is_none() {
        let [path] = files.as_slice() else {
            eprintln!(
                "pdfspatial: --out DIR is required when converting more than one PDF ({} given)",
                files.len()
            );
            return ExitCode::FAILURE;
        };
        return match pdfspatial_core::pdf_to_markdown(path, config.markdown) {
            Ok(markdown) => {
                println!("{markdown}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("pdfspatial: failed to convert {}: {err}", path.display());
                ExitCode::FAILURE
            }
        };
    }

    let out_dir = config.out.as_ref().unwrap();
    if let Err(err) = std::fs::create_dir_all(out_dir) {
        eprintln!(
            "pdfspatial: failed to create output directory {}: {err}",
            out_dir.display()
        );
        return ExitCode::FAILURE;
    }

    run_batch(&files, out_dir, &config)
}

/// Expands `inputs` (files and/or directories) into a flat, sorted list of PDF paths.
/// Directories are scanned non-recursively for `*.pdf` (case-sensitive extension, as
/// PDFium itself doesn't care about case but a mixed-case corpus is rare enough not to
/// special-case here). A path that is neither an existing file nor an existing
/// directory is an error -- silently skipping a typo'd path would be worse than failing
/// fast, since a batch's success is judged by output *count* as much as content.
fn resolve_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for input in inputs {
        let metadata = std::fs::metadata(input)
            .map_err(|err| format!("cannot read {}: {err}", input.display()))?;
        if metadata.is_dir() {
            let mut dir_pdfs: Vec<PathBuf> = std::fs::read_dir(input)
                .map_err(|err| format!("cannot read directory {}: {err}", input.display()))?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "pdf"))
                .collect();
            dir_pdfs.sort();
            files.extend(dir_pdfs);
        } else {
            files.push(input.clone());
        }
    }
    if files.is_empty() {
        return Err("no .pdf files found among the given inputs".to_string());
    }
    Ok(files)
}

/// One file's conversion result, kept deliberately small: batch mode must not hold
/// hundreds of extracted [`pdfspatial_core::Document`]s live at once (gigabytes for a
/// real corpus), so each worker extracts, serializes, writes, and drops its `Document`
/// before returning this.
struct FileOutcome {
    path: PathBuf,
    pages: usize,
    chars: usize,
    elapsed: Duration,
    ok: bool,
}

fn run_batch(files: &[PathBuf], out_dir: &Path, config: &Config) -> ExitCode {
    let total = files.len();
    let progress = AtomicUsize::new(0);
    let quiet = config.quiet;
    let markdown_options = config.markdown;

    let convert_one = |path: &PathBuf| -> FileOutcome {
        let started = Instant::now();
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "output".to_string());
        let out_path = out_dir.join(format!("{stem}.md"));

        // A single malformed PDF must not take the other 199 down with it: neither a
        // returned PipelineError nor a panic inside the PDFium binding is allowed to
        // abort the batch.
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            let document = extract_baseline(path)?;
            let markdown = serialize::to_markdown_pipeline(&document, markdown_options);
            Ok::<_, pdfspatial_core::PipelineError>((
                document.pages.len(),
                document.char_count(),
                markdown,
            ))
        }));

        let (pages, chars, ok) = match result {
            Ok(Ok((pages, chars, markdown))) => {
                if let Err(err) = std::fs::write(&out_path, markdown) {
                    eprintln!("pdfspatial: failed to write {}: {err}", out_path.display());
                    let _ = std::fs::write(&out_path, "");
                    (pages, chars, false)
                } else {
                    (pages, chars, true)
                }
            }
            Ok(Err(err)) => {
                eprintln!("pdfspatial: failed to convert {}: {err}", path.display());
                let _ = std::fs::write(&out_path, "");
                eprintln!(
                    "pdfspatial: wrote empty {} (conversion failed)",
                    out_path.display()
                );
                (0, 0, false)
            }
            Err(_) => {
                eprintln!("pdfspatial: panicked while converting {}", path.display());
                let _ = std::fs::write(&out_path, "");
                eprintln!(
                    "pdfspatial: wrote empty {} (conversion panicked)",
                    out_path.display()
                );
                (0, 0, false)
            }
        };

        let elapsed = started.elapsed();
        if !quiet {
            let index = progress.fetch_add(1, Ordering::Relaxed) + 1;
            eprintln!(
                "[{index:>4}/{total}] {}  {pages}p  {}ms{}",
                path.display(),
                elapsed.as_millis(),
                if ok { "" } else { "  FAILED" },
            );
        }

        FileOutcome {
            path: path.clone(),
            pages,
            chars,
            elapsed,
            ok,
        }
    };

    let wall_started = Instant::now();
    let outcomes: Vec<FileOutcome> = if let Some(jobs) = config.jobs {
        match rayon::ThreadPoolBuilder::new().num_threads(jobs).build() {
            Ok(pool) => pool.install(|| files.par_iter().map(convert_one).collect()),
            Err(err) => {
                eprintln!("pdfspatial: failed to build a {jobs}-thread pool: {err}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        files.par_iter().map(convert_one).collect()
    };
    let wall = wall_started.elapsed();

    let failed = outcomes.iter().filter(|o| !o.ok).count();
    let total_pages: usize = outcomes.iter().map(|o| o.pages).sum();
    let total_chars: usize = outcomes.iter().map(|o| o.chars).sum();
    if let Some(slowest) = outcomes.iter().max_by_key(|o| o.elapsed) {
        eprintln!(
            "pdfspatial: slowest file: {} ({}ms)",
            slowest.path.display(),
            slowest.elapsed.as_millis(),
        );
    }
    if failed > 0 {
        eprintln!("pdfspatial: failed files:");
        for outcome in outcomes.iter().filter(|o| !o.ok) {
            eprintln!("  {}", outcome.path.display());
        }
    }
    let s_per_page = if total_pages > 0 {
        wall.as_secs_f64() / total_pages as f64
    } else {
        0.0
    };
    let s_per_doc = if total > 0 {
        wall.as_secs_f64() / total as f64
    } else {
        0.0
    };

    eprintln!(
        "pdfspatial: {total} file(s), {failed} failed, {total_pages} page(s), {total_chars} char(s), \
         {:.2}s wall, {s_per_page:.4} s/page, {s_per_doc:.4} s/doc",
        wall.as_secs_f64(),
    );

    if failed > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

//! Extracts a PDF's Stage 1 baseline text and prints it as Markdown.
//!
//! ```sh
//! cargo run --example basic_extract -- path/to/input.pdf
//! ```
//!
//! Requires the native PDFium library to be available; set `PDFSPATIAL_PDFIUM_LIB` to
//! its path if it is not on your system's standard library search path. See the
//! crate README for setup instructions and a link to prebuilt binaries.

use pdfspatial_core::{extract_baseline, serialize};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = std::env::args_os().nth(1) else {
        eprintln!("usage: basic_extract <path-to.pdf>");
        return ExitCode::FAILURE;
    };
    let path = PathBuf::from(path);

    let document = match extract_baseline(&path) {
        Ok(document) => document,
        Err(err) => {
            eprintln!("failed to extract {}: {err}", path.display());
            return ExitCode::FAILURE;
        }
    };

    println!("{}", serialize::to_markdown(&document));
    eprintln!(
        "\n--- {} page(s), {} character(s) extracted ---",
        document.pages.len(),
        document.char_count()
    );

    ExitCode::SUCCESS
}

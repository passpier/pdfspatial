//! Integration test for the `pdfspatial` binary's batch mode: spawns the real compiled
//! binary (not a library call) over the 6 real PDF fixtures already committed for the
//! Stage 3 corpus (`crates/pdfspatial-core/tests/fixtures/stage3/*.pdf`), the same
//! invocation shape `bench/opendataloader/adapters/pdf_parser_pdfspatial.py` uses.
//!
//! Needs the native PDFium library, so it follows
//! `crates/pdfspatial-core/tests/stage3_pdf_fixtures.rs`'s accommodation: skip with a
//! printed notice unless `PDFSPATIAL_PDFIUM_LIB` or `CI` is set, in which case fail
//! loudly instead of silently skipping.

use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixtures_dir() -> PathBuf {
    workspace_root().join("crates/pdfspatial-core/tests/fixtures/stage3")
}

fn pdfium_should_be_required() -> bool {
    std::env::var_os("PDFSPATIAL_PDFIUM_LIB").is_some() || std::env::var_os("CI").is_some()
}

/// A fresh, unique temp directory under the OS temp dir, cleaned up on drop. Avoids a
/// `tempfile` dependency in a crate that otherwise depends on nothing but
/// `pdfspatial-core` and `rayon`.
struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "pdfspatial-cli-test-{label}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("failed to create temp dir");
        TempDir(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn batch_mode_converts_every_pdf_and_never_aborts_on_one_bad_input() {
    let fixtures = fixtures_dir();
    let mut pdfs: Vec<PathBuf> = std::fs::read_dir(&fixtures)
        .expect("fixtures dir should exist")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "pdf"))
        .collect();
    pdfs.sort();
    assert!(!pdfs.is_empty(), "expected at least one fixture PDF");
    let expected_count = pdfs.len();

    // A bogus, non-PDF input mixed into the batch: the batch must still finish and
    // still produce an (empty) output for it, rather than aborting the other real PDFs.
    let out = TempDir::new("batch");
    let bogus_path = out.path().join("not-a-pdf.pdf");
    std::fs::write(&bogus_path, b"this is not a PDF").expect("failed to write bogus file");
    pdfs.push(bogus_path.clone());

    let output_dir = TempDir::new("batch-out");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_pdfspatial"));
    cmd.arg("--out")
        .arg(output_dir.path())
        .arg("--quiet")
        .args(&pdfs);

    let result = cmd.output();
    let output = match result {
        Ok(output) => output,
        Err(err) => {
            if pdfium_should_be_required() {
                panic!("failed to spawn pdfspatial binary: {err}");
            }
            eprintln!("skipping batch_mode test: failed to spawn pdfspatial binary: {err}");
            return;
        }
    };

    let stderr = String::from_utf8_lossy(&output.stderr);
    if !pdfium_should_be_required() && stderr.contains("failed to bind PDFium") {
        eprintln!(
            "skipping batch_mode_converts_every_pdf_and_never_aborts_on_one_bad_input: \
             no native PDFium library available. Set PDFSPATIAL_PDFIUM_LIB to run this \
             test for real -- see README.md's Prerequisites section.\nstderr:\n{stderr}"
        );
        return;
    }

    // The bogus PDF is expected to fail conversion, so the process should report
    // failure overall -- but every real PDF, plus an empty stand-in for the bogus one,
    // must still have been written.
    assert!(
        !output.status.success(),
        "expected a nonzero exit due to the bogus input, got success.\nstderr:\n{stderr}"
    );

    let written: Vec<PathBuf> = std::fs::read_dir(output_dir.path())
        .expect("output dir should exist")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .collect();
    assert_eq!(
        written.len(),
        expected_count + 1,
        "expected {} .md file(s) (one per real PDF, plus one empty stand-in for the \
         bogus input), got {}.\nstderr:\n{stderr}",
        expected_count + 1,
        written.len()
    );

    // Every real fixture PDF produced non-empty Markdown.
    for pdf in &pdfs[..expected_count] {
        let stem = pdf.file_stem().unwrap().to_string_lossy();
        let md_path = output_dir.path().join(format!("{stem}.md"));
        let content = std::fs::read_to_string(&md_path)
            .unwrap_or_else(|_| panic!("expected {} to exist", md_path.display()));
        assert!(
            !content.trim().is_empty(),
            "expected {} to contain Markdown, got an empty file",
            md_path.display()
        );
    }

    // The bogus, non-PDF input still produced a file -- empty, but present, so a
    // bench-style evaluator never silently treats it as a missing prediction.
    let bogus_md = output_dir.path().join("not-a-pdf.md");
    let bogus_content = std::fs::read_to_string(&bogus_md)
        .unwrap_or_else(|_| panic!("expected {} to exist even on failure", bogus_md.display()));
    assert!(
        bogus_content.is_empty(),
        "expected the failed conversion to write an empty file, got: {bogus_content:?}"
    );
}

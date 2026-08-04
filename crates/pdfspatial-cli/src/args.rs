//! Argument parsing for the `pdfspatial` binary, split out from `main.rs` so the flag
//! surface is testable without spawning a process or touching PDFium (see
//! `tests/cli_args.rs`).
//!
//! Hand-rolled rather than built on a crate like `clap`, matching the convention every
//! example under `examples/` already uses (`while i < args.len()` / `match
//! args[i].as_str()`): the workspace advertises a dependency-minimal algorithmic core,
//! and a 6-flag surface doesn't earn a dependency's MSRV/publish-surface cost.

use pdfspatial_core::MarkdownOptions;
use std::path::PathBuf;

pub(crate) const USAGE: &str = "\
pdfspatial [OPTIONS] <INPUT>...

Converts PDFs to structural Markdown: spatially-grounded, deterministic, OCR-free.

INPUT is a .pdf file, or a directory (batch mode: every *.pdf found directly inside
it, not recursively). Multiple inputs are allowed and files/directories may be mixed.

OPTIONS
  --out DIR                 Write <stem>.md per input PDF into DIR (created if it
                             doesn't exist). Required when there is more than one
                             input PDF. Without it, a single input PDF is written to
                             stdout.
  --no-page-breaks          Omit the `---` thematic break between pages.
  --no-image-placeholders   Omit `![]()` placeholders for detected Picture regions.
  --jobs N                  Worker threads for batch mode (default: rayon's own
                             default; 1 = fully serial).
  --quiet                   Suppress per-file progress on stderr (the final summary
                             line is still printed).
  --version                 Print the version and exit.
  --help                    Print this message and exit.

Both `--flag VALUE` and `--flag=VALUE` are accepted.
";

/// What `parse` decided the process should do.
#[derive(Debug)]
pub(crate) enum ParseOutcome {
    /// Print `USAGE` and exit successfully (e.g. `--help`, or no arguments at all).
    Help,
    /// Print the crate version and exit successfully.
    Version,
    /// Convert the given inputs under the given configuration.
    Run(Config),
}

/// A fully validated set of options for one `pdfspatial` invocation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Config {
    /// Files and/or directories given on the command line, in the order given.
    pub inputs: Vec<PathBuf>,
    /// `--out DIR`. `None` means "write the single input's Markdown to stdout" --
    /// `resolve_inputs` (in `main.rs`) is what rejects `None` with more than one PDF.
    pub out: Option<PathBuf>,
    /// The `MarkdownOptions` derived from `--no-page-breaks` / `--no-image-placeholders`.
    pub markdown: MarkdownOptions,
    /// `--jobs N`. `None` lets rayon pick its own default thread count.
    pub jobs: Option<usize>,
    /// `--quiet`: suppress per-file stderr progress lines.
    pub quiet: bool,
}

/// Parses `args` (already stripped of `argv[0]`, i.e. `std::env::args().skip(1)`).
///
/// Returns `Err(message)` for any malformed flag or missing value; the caller is
/// expected to print `message` (and `USAGE`) to stderr and exit with a failure code.
pub(crate) fn parse(args: &[String]) -> Result<ParseOutcome, String> {
    if args.is_empty() {
        return Ok(ParseOutcome::Help);
    }

    let mut inputs = Vec::new();
    let mut out = None;
    let mut page_breaks = true;
    let mut image_placeholders = true;
    let mut jobs = None;
    let mut quiet = false;

    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();

        // Supports both `--flag value` and `--flag=value` for every valued option.
        let (flag, inline_value) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (arg, None),
        };

        macro_rules! take_value {
            ($flag:expr) => {{
                match inline_value {
                    Some(v) => {
                        i += 1;
                        v
                    }
                    None => {
                        let Some(v) = args.get(i + 1) else {
                            return Err(format!("{} requires a value", $flag));
                        };
                        i += 2;
                        v.clone()
                    }
                }
            }};
        }

        match flag {
            "--help" | "-h" => return Ok(ParseOutcome::Help),
            "--version" | "-V" => return Ok(ParseOutcome::Version),
            "--out" => out = Some(PathBuf::from(take_value!("--out"))),
            "--no-page-breaks" => {
                page_breaks = false;
                i += 1;
            }
            "--no-image-placeholders" => {
                image_placeholders = false;
                i += 1;
            }
            "--jobs" => {
                let value = take_value!("--jobs");
                jobs = Some(value.parse::<usize>().map_err(|_| {
                    format!("--jobs value must be a positive integer, got {value:?}")
                })?);
                if jobs == Some(0) {
                    return Err("--jobs value must be at least 1".to_string());
                }
            }
            "--quiet" => {
                quiet = true;
                i += 1;
            }
            other if other.starts_with('-') && other != "-" => {
                return Err(format!("unknown flag: {other}"));
            }
            _ => {
                inputs.push(PathBuf::from(arg));
                i += 1;
            }
        }
    }

    if inputs.is_empty() {
        return Err("no input PDFs or directories given".to_string());
    }

    Ok(ParseOutcome::Run(Config {
        inputs,
        out,
        markdown: MarkdownOptions {
            page_breaks,
            image_placeholders,
        },
        jobs,
        quiet,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> Config {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        match parse(&args).expect("expected successful parse") {
            ParseOutcome::Run(config) => config,
            _ => panic!("expected ParseOutcome::Run"),
        }
    }

    #[test]
    fn no_args_is_help() {
        assert!(matches!(parse(&[]).unwrap(), ParseOutcome::Help));
    }

    #[test]
    fn help_and_version_flags() {
        assert!(matches!(
            parse(&["--help".to_string()]).unwrap(),
            ParseOutcome::Help
        ));
        assert!(matches!(
            parse(&["--version".to_string()]).unwrap(),
            ParseOutcome::Version
        ));
    }

    #[test]
    fn single_input_defaults() {
        let config = run(&["input.pdf"]);
        assert_eq!(config.inputs, vec![PathBuf::from("input.pdf")]);
        assert_eq!(config.out, None);
        assert_eq!(config.markdown, MarkdownOptions::default());
        assert_eq!(config.jobs, None);
        assert!(!config.quiet);
    }

    #[test]
    fn out_flag_space_form() {
        let config = run(&["--out", "outdir", "a.pdf", "b.pdf"]);
        assert_eq!(config.out, Some(PathBuf::from("outdir")));
        assert_eq!(
            config.inputs,
            vec![PathBuf::from("a.pdf"), PathBuf::from("b.pdf")]
        );
    }

    #[test]
    fn out_flag_equals_form() {
        let config = run(&["--out=outdir", "a.pdf"]);
        assert_eq!(config.out, Some(PathBuf::from("outdir")));
    }

    #[test]
    fn output_mode_flags() {
        let config = run(&["--no-page-breaks", "--no-image-placeholders", "a.pdf"]);
        assert!(!config.markdown.page_breaks);
        assert!(!config.markdown.image_placeholders);
    }

    #[test]
    fn jobs_flag_both_forms() {
        assert_eq!(run(&["--jobs", "4", "a.pdf"]).jobs, Some(4));
        assert_eq!(run(&["--jobs=1", "a.pdf"]).jobs, Some(1));
    }

    #[test]
    fn jobs_flag_rejects_zero() {
        let args: Vec<String> = ["--jobs", "0", "a.pdf"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let err = parse(&args).unwrap_err();
        assert!(err.contains("at least 1"), "unexpected error: {err}");
    }

    #[test]
    fn jobs_flag_rejects_non_numeric() {
        let args: Vec<String> = ["--jobs", "many", "a.pdf"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let err = parse(&args).unwrap_err();
        assert!(err.contains("positive integer"), "unexpected error: {err}");
    }

    #[test]
    fn quiet_flag() {
        assert!(run(&["--quiet", "a.pdf"]).quiet);
    }

    #[test]
    fn out_requires_a_value() {
        let args: Vec<String> = ["--out"].iter().map(|s| s.to_string()).collect();
        let err = parse(&args).unwrap_err();
        assert!(err.contains("--out requires a value"), "{err}");
    }

    #[test]
    fn unknown_flag_is_rejected() {
        let args: Vec<String> = ["--bogus", "a.pdf"].iter().map(|s| s.to_string()).collect();
        let err = parse(&args).unwrap_err();
        assert!(err.contains("unknown flag: --bogus"), "{err}");
    }

    #[test]
    fn no_inputs_is_rejected() {
        let args: Vec<String> = ["--quiet"].iter().map(|s| s.to_string()).collect();
        let err = parse(&args).unwrap_err();
        assert!(err.contains("no input"), "{err}");
    }

    #[test]
    fn directories_and_files_can_be_mixed() {
        let config = run(&["a.pdf", "some_dir", "b.pdf"]);
        assert_eq!(
            config.inputs,
            vec![
                PathBuf::from("a.pdf"),
                PathBuf::from("some_dir"),
                PathBuf::from("b.pdf"),
            ]
        );
    }
}

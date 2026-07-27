//! Stage 1: deterministic, OCR-free baseline extraction on top of PDFium.
//!
//! This module binds [`pdfium-render`](https://docs.rs/pdfium-render) to pull
//! character-level bounding boxes straight out of a PDF's native text layer, then
//! reconstructs words, lines, and blocks from pure geometry:
//!
//! - **char → word**: x-gap thresholding along a shared baseline.
//! - **word → line**: y-coordinate (baseline) tolerance clustering.
//! - **line → block**: a vertical-gap heuristic between consecutive lines.
//!
//! No machine-learning model, layout classifier, or reading-order solver is involved —
//! that is intentional. Stage 1's job is to establish a lossless, fast extraction floor
//! on well-behaved (single-column, non-tabular) documents; multi-column layouts, tables,
//! and formulas are expected to degrade here, and that failure surface is what Stage 3
//! (see [`crate::assemble`]) characterizes.
//!
//! ## Locating the native PDFium library
//!
//! `pdfium-render` does not bundle PDFium; the native library must be present on the
//! machine running this code. Use [`PdfiumSource`] to control how it is located, or set
//! the `PDFSPATIAL_PDFIUM_LIB` environment variable to a path to the library file (or a
//! directory containing it) and call [`extract_baseline`], which honors it by default.
//!
//! Prebuilt PDFium binaries are published at
//! <https://github.com/bblanchon/pdfium-binaries>.
//!
//! ## Process-wide binding
//!
//! PDFium's C API can only be bound once per process. `pdfspatial-core` binds it lazily
//! on first use and reuses that binding for every subsequent call in the same process —
//! including calls that pass a different [`PdfiumSource`]. If you need to switch which
//! PDFium library is loaded, do so before the first extraction call.
//!
//! ## PDFium is not thread-safe
//!
//! Upstream PDFium makes no thread-safety guarantees, and its authors recommend
//! multi-*process* parallelism over multi-threading. `pdfium-render`'s `thread_safe`
//! feature (enabled here) makes the [`Pdfium`] handle `Send + Sync` so it can be shared
//! across threads, but that alone does not make *concurrent* calls into PDFium safe —
//! two threads calling into the library at the same instant can corrupt its internal
//! state. `pdfspatial-core` therefore serializes every call into PDFium behind an
//! internal lock. [`render_pages_parallel`] still fans work out across a thread pool —
//! for I/O overlap and to keep the call shape ready for a future truly-parallel
//! backend — but the actual PDFium calls it makes are serialized, not concurrent.

use crate::{BBox, Block, Char, Document, Line, Page, PipelineError, Word};
use pdfium_render::prelude::*;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// A horizontal gap larger than this, expressed as a multiple of the current font size,
/// ends the current word.
const WORD_GAP_FACTOR: f32 = 0.28;

/// Two words whose vertical centers differ by more than this, expressed as a multiple of
/// the larger of the two font sizes, are placed on different lines.
const LINE_Y_TOLERANCE_FACTOR: f32 = 0.45;

/// A vertical gap between consecutive lines larger than this, expressed as a multiple of
/// the shorter line's height, starts a new block.
const BLOCK_GAP_FACTOR: f32 = 1.5;

/// Where to obtain the native PDFium library from.
///
/// `pdfium-render` binds to PDFium at run time rather than linking it statically, so this
/// crate needs to be told where to find it.
#[derive(Debug, Clone)]
pub enum PdfiumSource {
    /// Search the operating system's standard dynamic-library search paths (for example
    /// `DYLD_LIBRARY_PATH` on macOS, `LD_LIBRARY_PATH` on Linux, or `PATH` on Windows) for
    /// a PDFium library.
    System,
    /// Load PDFium from an explicit location: either the path to the shared library file
    /// itself, or a directory containing a platform-conventionally-named PDFium library
    /// (`libpdfium.dylib`, `libpdfium.so`, or `pdfium.dll`).
    Path(PathBuf),
}

impl Default for PdfiumSource {
    /// Reads the `PDFSPATIAL_PDFIUM_LIB` environment variable, if set, as a
    /// [`PdfiumSource::Path`]; otherwise falls back to [`PdfiumSource::System`].
    fn default() -> Self {
        match std::env::var_os("PDFSPATIAL_PDFIUM_LIB") {
            Some(path) => PdfiumSource::Path(PathBuf::from(path)),
            None => PdfiumSource::System,
        }
    }
}

static PDFIUM: OnceLock<Pdfium> = OnceLock::new();
static PDFIUM_INIT: Mutex<()> = Mutex::new(());

/// Serializes every call into PDFium's C API across the whole process.
///
/// See the [module docs](self#pdfium-is-not-thread-safe) for why this exists: PDFium is
/// not safe for concurrent access even via a `Send + Sync` handle, so every function in
/// this module that touches PDFium must hold this lock for the duration of that access.
static PDFIUM_CALL_LOCK: Mutex<()> = Mutex::new(());

/// Returns the process-wide [`Pdfium`] instance, binding it on first use.
fn pdfium_instance(source: &PdfiumSource) -> Result<&'static Pdfium, PipelineError> {
    if let Some(pdfium) = PDFIUM.get() {
        return Ok(pdfium);
    }

    // Double-checked locking: only one thread performs the (fallible) bind + bindings
    // registration, since `Pdfium::new` panics if called more than once per process.
    let _guard = PDFIUM_INIT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(pdfium) = PDFIUM.get() {
        return Ok(pdfium);
    }

    let bindings = match source {
        PdfiumSource::System => Pdfium::bind_to_system_library(),
        PdfiumSource::Path(path) => {
            let library_path = if path.is_dir() {
                Pdfium::pdfium_platform_library_name_at_path(path)
            } else {
                path.clone()
            };
            Pdfium::bind_to_library(library_path)
        }
    }
    .map_err(PipelineError::PdfiumBinding)?;

    Ok(PDFIUM.get_or_init(|| Pdfium::new(bindings)))
}

/// Extracts a [`Document`] from the PDF at `path` using Stage 1's baseline pipeline.
///
/// This is the crate's primary entry point: it locates the native PDFium library (via
/// [`PdfiumSource::default`], which honors the `PDFSPATIAL_PDFIUM_LIB` environment
/// variable), opens the file, and runs char → word → line → block grouping over every
/// page. The result carries zero structural tagging — see the [module docs](self) for
/// what that means and why.
///
/// # Errors
///
/// Returns [`PipelineError::PdfiumBinding`] if the native PDFium library cannot be
/// located or bound, or [`PipelineError::Document`] if the file cannot be opened or
/// parsed as a PDF.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
///
/// # fn main() -> Result<(), pdfspatial_core::PipelineError> {
/// let document = pdfspatial_core::extract_baseline(Path::new("input.pdf"))?;
/// for page in &document.pages {
///     println!("page {}: {} blocks", page.index, page.blocks.len());
/// }
/// # Ok(())
/// # }
/// ```
pub fn extract_baseline(path: &Path) -> Result<Document, PipelineError> {
    extract_baseline_with_source(path, &PdfiumSource::default())
}

/// Like [`extract_baseline`], but with explicit control over how PDFium is located.
///
/// See the [module docs](self#process-wide-binding) for the process-wide binding caveat:
/// `source` is only honored on the first call in a process.
pub fn extract_baseline_with_source(
    path: &Path,
    source: &PdfiumSource,
) -> Result<Document, PipelineError> {
    let pdfium = pdfium_instance(source)?;
    let _call_guard = PDFIUM_CALL_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let document = pdfium
        .load_pdf_from_file(path, None)
        .map_err(PipelineError::Document)?;

    let pages = document
        .pages()
        .iter()
        .enumerate()
        .map(|(index, page)| extract_page(index, &page))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Document { pages })
}

/// Extracts a single page: chars, then char → word → line → block grouping.
fn extract_page(index: usize, page: &PdfPage) -> Result<Page, PipelineError> {
    let width = page.width().value;
    let height = page.height().value;

    let text = page.text().map_err(PipelineError::Document)?;

    let chars: Vec<Char> = text
        .chars()
        .iter()
        .filter_map(|raw| extract_char(&raw))
        .collect();

    let blocks = group_chars_into_blocks(&chars);

    Ok(Page {
        index,
        width,
        height,
        blocks,
    })
}

/// Runs Stage 1's char → word → line → block grouping over an already-extracted
/// character sequence, with no PDFium involvement.
///
/// This is the same geometric grouping [`extract_baseline`] applies to PDFium's native
/// text layer, factored out as a pure function so callers that reconstruct characters
/// from another source (for example, a dataset's own text cells) can run the real Stage
/// 1 grouping instead of approximating it. See
/// [`crate::eval::doclaynet::document_from_cells_grouped`] for such a caller.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::{BBox, Char};
/// use pdfspatial_core::extract::group_chars_into_blocks;
///
/// let make_char = |ch: char, left: f32, right: f32| Char {
///     unicode: Some(ch),
///     bbox: BBox { left, bottom: 0.0, right, top: 10.0 },
///     font_name: "Test".to_string(),
///     font_size: 10.0,
/// };
///
/// // "Hi" as two characters on one baseline.
/// let chars = vec![make_char('H', 0.0, 6.0), make_char('i', 6.0, 9.0)];
/// let blocks = group_chars_into_blocks(&chars);
///
/// assert_eq!(blocks.len(), 1);
/// assert_eq!(blocks[0].text(), "Hi");
/// ```
pub fn group_chars_into_blocks(chars: &[Char]) -> Vec<Block> {
    let words = group_words(chars);
    let lines = group_lines(words);
    group_blocks(lines)
}

/// Converts a single PDFium character into our owned [`Char`] type.
///
/// Returns `None` only when PDFium cannot produce *any* usable bounding box for the
/// glyph (neither loose nor tight bounds); such glyphs are dropped and count against the
/// character-extraction-recall metric (see [`crate::metrics::char_recall`]).
fn extract_char(raw: &PdfPageTextChar) -> Option<Char> {
    // Loose bounds (full glyph advance box) are preferred over tight bounds because they
    // resolve for whitespace and other "inkless" glyphs that tight bounds rejects, and
    // because word/line clustering cares about advance geometry, not ink shape.
    let rect = raw.loose_bounds().or_else(|_| raw.tight_bounds()).ok()?;

    Some(Char {
        unicode: raw.unicode_char(),
        bbox: BBox {
            left: rect.left().value,
            bottom: rect.bottom().value,
            right: rect.right().value,
            top: rect.top().value,
        },
        font_name: raw.font_name(),
        font_size: raw.unscaled_font_size().value,
    })
}

/// Groups characters into words via whitespace and x-gap thresholding.
///
/// A word boundary occurs when: the character is whitespace, the horizontal gap since
/// the previous character exceeds [`WORD_GAP_FACTOR`] times the local font size, or the
/// previous character sits on a visibly different baseline (a line break mid-stream).
fn group_words(chars: &[Char]) -> Vec<Word> {
    let mut words = Vec::new();
    let mut current: Vec<Char> = Vec::new();
    let mut prev: Option<&Char> = None;

    for ch in chars {
        let is_space = ch.unicode.map(char::is_whitespace).unwrap_or(false);
        let mut break_word = is_space;

        if let Some(p) = prev {
            let font_size = ch.font_size.max(p.font_size).max(1.0);
            let x_gap = ch.bbox.left - p.bbox.right;
            let same_baseline = ch.bbox.vertically_overlaps(&p.bbox);

            if !same_baseline || x_gap > font_size * WORD_GAP_FACTOR {
                break_word = true;
            }
        }

        if break_word && !current.is_empty() {
            words.push(finalize_word(std::mem::take(&mut current)));
        }

        if !is_space {
            current.push(ch.clone());
        }

        prev = Some(ch);
    }

    if !current.is_empty() {
        words.push(finalize_word(current));
    }

    words
}

fn finalize_word(chars: Vec<Char>) -> Word {
    let text: String = chars.iter().filter_map(|c| c.unicode).collect();
    let bbox = union_all(chars.iter().map(|c| c.bbox)).unwrap_or(BBox::ZERO);
    Word { text, bbox, chars }
}

/// Groups words into lines via baseline (vertical-center) tolerance clustering.
///
/// Words are appended to the most recently opened line as long as their vertical center
/// stays within [`LINE_Y_TOLERANCE_FACTOR`] times the font size of the running line; a
/// larger vertical shift starts a new line. Within each finished line, words are ordered
/// left to right by their x-position, since PDF content-stream order does not guarantee
/// horizontal ordering.
fn group_lines(words: Vec<Word>) -> Vec<Line> {
    let mut line_groups: Vec<Vec<Word>> = Vec::new();

    for word in words {
        let attaches_to_last = line_groups.last().is_some_and(|line| {
            let anchor = &line[0];
            let font_size = word_font_size(&word).max(word_font_size(anchor)).max(1.0);
            let tol = font_size * LINE_Y_TOLERANCE_FACTOR;
            (center_y(word.bbox) - center_y(anchor.bbox)).abs() <= tol
        });

        if attaches_to_last {
            line_groups.last_mut().unwrap().push(word);
        } else {
            line_groups.push(vec![word]);
        }
    }

    line_groups.into_iter().map(finalize_line).collect()
}

fn finalize_line(mut words: Vec<Word>) -> Line {
    words.sort_by(|a, b| a.bbox.left.partial_cmp(&b.bbox.left).unwrap());
    let text = words
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let bbox = union_all(words.iter().map(|w| w.bbox)).unwrap_or(BBox::ZERO);
    Line { text, bbox, words }
}

/// Groups consecutive lines into blocks via a vertical-gap heuristic: a gap between one
/// line's bottom and the next line's top larger than [`BLOCK_GAP_FACTOR`] times the
/// shorter line's height starts a new block. This is a naive, ML-free paragraph/column
/// detector by design — see [`crate::assemble`] for where a real reading-order solver
/// eventually replaces it.
fn group_blocks(lines: Vec<Line>) -> Vec<Block> {
    let mut block_groups: Vec<Vec<Line>> = Vec::new();

    for line in lines {
        let attaches_to_last = block_groups.last().is_some_and(|block: &Vec<Line>| {
            let prev = block.last().unwrap();
            let gap = prev.bbox.bottom - line.bbox.top;
            let shorter_height = prev.bbox.height().min(line.bbox.height()).max(1.0);
            gap <= shorter_height * BLOCK_GAP_FACTOR
        });

        if attaches_to_last {
            block_groups.last_mut().unwrap().push(line);
        } else {
            block_groups.push(vec![line]);
        }
    }

    block_groups
        .into_iter()
        .map(|lines| {
            let bbox = union_all(lines.iter().map(|l| l.bbox)).unwrap_or(BBox::ZERO);
            Block { bbox, lines }
        })
        .collect()
}

fn union_all(mut boxes: impl Iterator<Item = BBox>) -> Option<BBox> {
    let first = boxes.next()?;
    Some(boxes.fold(first, |acc, b| acc.union(&b)))
}

fn center_y(bbox: BBox) -> f32 {
    (bbox.top + bbox.bottom) / 2.0
}

fn word_font_size(word: &Word) -> f32 {
    word.chars.first().map(|c| c.font_size).unwrap_or(0.0)
}

/// A page rendered to an RGBA bitmap, for downstream layout-model inference or QA
/// overlays. Produced by [`render_pages_parallel`].
#[derive(Debug, Clone)]
pub struct RenderedPage {
    /// Zero-based page index within the document.
    pub index: usize,
    /// Bitmap width in pixels.
    pub width: i32,
    /// Bitmap height in pixels.
    pub height: i32,
    /// Raw RGBA8 pixel data, `width * height * 4` bytes, row-major from the top-left.
    pub pixels: Vec<u8>,
}

/// Renders every page of the PDF at `path` to an RGBA bitmap, in parallel.
///
/// Stage 1 does not yet consume these bitmaps (no layout model is wired up), but the
/// rendering path is established now so Stage 2/4 can add vision-model inference and QA
/// overlays without touching the extraction pipeline. Each page is rendered at
/// `target_width` pixels wide, with height scaled to preserve the page's aspect ratio.
///
/// Rendering fans out across a Rayon thread pool. Because PDFium's page and document
/// handles are not thread-safe, each worker opens its own short-lived handle to the file
/// rather than sharing one across threads; only the process-wide PDFium *binding* (see
/// [module docs](self#process-wide-binding)) is shared.
///
/// # Errors
///
/// Returns [`PipelineError::PdfiumBinding`] or [`PipelineError::Document`] under the same
/// conditions as [`extract_baseline`].
pub fn render_pages_parallel(
    path: &Path,
    source: &PdfiumSource,
    target_width: i32,
) -> Result<Vec<RenderedPage>, PipelineError> {
    let pdfium = pdfium_instance(source)?;

    let page_count: PdfPageIndex = {
        let _guard = PDFIUM_CALL_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let document = pdfium
            .load_pdf_from_file(path, None)
            .map_err(PipelineError::Document)?;
        document.pages().len()
    };

    (0..page_count)
        .into_par_iter()
        .map(|index| render_page(pdfium, path, index, target_width))
        .collect()
}

fn render_page(
    pdfium: &Pdfium,
    path: &Path,
    index: PdfPageIndex,
    target_width: i32,
) -> Result<RenderedPage, PipelineError> {
    let _guard = PDFIUM_CALL_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let document = pdfium
        .load_pdf_from_file(path, None)
        .map_err(PipelineError::Document)?;
    let page = document
        .pages()
        .get(index)
        .map_err(PipelineError::Document)?;

    let aspect_ratio = page.height().value / page.width().value;
    let target_height = (target_width as f32 * aspect_ratio).round() as i32;

    let bitmap = page
        .render(target_width, target_height, None)
        .map_err(PipelineError::Document)?;

    Ok(RenderedPage {
        index: index as usize,
        width: bitmap.width(),
        height: bitmap.height(),
        pixels: bitmap.as_rgba_bytes(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn char_at(ch: char, left: f32, bottom: f32, right: f32, top: f32, font_size: f32) -> Char {
        Char {
            unicode: Some(ch),
            bbox: BBox {
                left,
                bottom,
                right,
                top,
            },
            font_name: "Test".to_string(),
            font_size,
        }
    }

    /// `group_chars_into_blocks` needs no PDFium at all — it operates purely on already
    /// -extracted [`Char`]s — so this exercises Stage 1's real grouping without the
    /// native library `tests/stage1_baseline.rs` requires.
    #[test]
    fn group_chars_into_blocks_groups_one_line_into_one_block() {
        // "Hi" on one baseline, no gaps large enough to break word/line/block grouping.
        let chars = vec![
            char_at('H', 0.0, 0.0, 6.0, 10.0, 10.0),
            char_at('i', 6.0, 0.0, 9.0, 10.0, 10.0),
        ];

        let blocks = group_chars_into_blocks(&chars);

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].lines.len(), 1);
        assert_eq!(blocks[0].text(), "Hi");
    }

    /// A vertical gap larger than `BLOCK_GAP_FACTOR` times line height starts a new
    /// block, even though both lines individually group cleanly. Characters are given
    /// in top-to-bottom reading order (matching the order PDFium's content stream and
    /// `document_from_cells_grouped` both feed this function), so `B` sits below `A`.
    #[test]
    fn group_chars_into_blocks_splits_on_large_vertical_gap() {
        let chars = vec![
            char_at('A', 0.0, 100.0, 6.0, 110.0, 10.0),
            char_at('B', 0.0, 0.0, 6.0, 10.0, 10.0),
        ];

        let blocks = group_chars_into_blocks(&chars);

        assert_eq!(blocks.len(), 2);
    }
}

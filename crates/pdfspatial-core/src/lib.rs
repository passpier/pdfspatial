//! `pdfspatial-core` extracts spatially-grounded, reading-order text from PDF documents.
//!
//! This crate is the foundation of the `pdfspatial` project: a closed-loop, four-stage
//! pipeline that turns PDFs into Markdown by grounding every extracted token in its
//! on-page bounding box. The stages are:
//!
//! 1. **Baseline extraction** (implemented) — a deterministic, OCR-free text extraction
//!    floor built directly on PDFium's native text layer, plus a graphics-layer pass
//!    ([`extract::extract_graphics`]) over PDFium's non-text page objects (ruling lines,
//!    images, fills) that [`graphics::detect_table_regions`]/
//!    [`graphics::detect_picture_regions`] key `Table`/`Picture` detection off. See
//!    [`extract`].
//! 2. **Validation** — structural-fidelity scoring: [`metrics::giou`],
//!    [`metrics::region_f1`], and the TEDS family ([`metrics::teds_struct`],
//!    [`metrics::teds`], [`metrics::teds_iou`]) are pure, tested functions, aggregated
//!    across a page dataset by [`eval::evaluate_pages`] into a [`eval::Stage2Report`].
//!    The `doclaynet` cargo feature adds [`eval::doclaynet`], a loader that scores
//!    [`layout::classify_regions`] against real DocLayNet ground truth.
//! 3. **Error analysis** — clustering Stage 2 shortfalls into a reproducible failure-mode
//!    taxonomy. See [`assemble`] for the pitfall checklist this stage is organized around.
//! 4. **Refinement** — closing the gaps found in Stage 3, heuristics first
//!    ([`layout::classify_regions`] for region classification,
//!    [`assemble::assemble_reading_order`] for column-aware reading order), targeted
//!    model fine-tuning second.
//!
//! Stage 1 is fully implemented, and Stage 2/4's *algorithmic* core — validation
//! metrics, a heuristic layout classifier plus graphics-layer table/picture detection
//! ([`graphics`]), XY-cut reading-order assembly, and structural Markdown output
//! ([`serialize::to_markdown_structured`], including GFM tables and image placeholders)
//! — is implemented as pure, dependency-free Rust; no vision model is involved. The one
//! region class still unimplemented is [`layout::RegionClass::Formula`], via the
//! roadmap's Stage 2/4b vision-model layout detector (an ONNX RT-DETR-style detector
//! over rendered page rasters) — see [`layout`] for why a formula, unlike a table or
//! picture, has no non-text page-object signal to key a heuristic off.
//!
//! # Quick start
//!
//! ```no_run
//! use std::path::Path;
//!
//! # fn main() -> Result<(), pdfspatial_core::PipelineError> {
//! let document = pdfspatial_core::extract_baseline(Path::new("input.pdf"))?;
//! println!("{}", document.reading_order_text());
//! # Ok(())
//! # }
//! ```
//!
//! Extraction requires the native PDFium library to be available at runtime; see
//! [`extract::PdfiumSource`] for the ways to point `pdfspatial-core` at it.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod assemble;
pub mod eval;
pub mod extract;
pub mod graphics;
pub mod layout;
pub mod metrics;
pub mod serialize;

pub use extract::{PdfiumSource, extract_baseline, extract_baseline_with_source};
pub use metrics::iou;
pub use serialize::to_markdown_structured;

/// Errors that can occur anywhere in the `pdfspatial` pipeline.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// The native PDFium library could not be loaded or bound.
    #[error("failed to bind PDFium library: {0}")]
    PdfiumBinding(#[source] pdfium_render::prelude::PdfiumError),

    /// PDFium returned an error while opening, parsing, or reading the document.
    #[error("PDFium document error: {0}")]
    Document(#[source] pdfium_render::prelude::PdfiumError),

    /// An I/O error occurred while reading the input file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A rectangular bounding box in PDF user-space points.
///
/// The coordinate space matches PDFium's page space: the origin `(0.0, 0.0)` is at the
/// bottom-left of the page, `x` increases to the right, and `y` increases upward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// Left edge, in points from the page origin.
    pub left: f32,
    /// Bottom edge, in points from the page origin.
    pub bottom: f32,
    /// Right edge, in points from the page origin.
    pub right: f32,
    /// Top edge, in points from the page origin.
    pub top: f32,
}

impl BBox {
    /// Returns a zero-sized bounding box at the origin.
    pub const ZERO: BBox = BBox {
        left: 0.0,
        bottom: 0.0,
        right: 0.0,
        top: 0.0,
    };

    /// Width of the box, in points. Never negative.
    pub fn width(&self) -> f32 {
        (self.right - self.left).max(0.0)
    }

    /// Height of the box, in points. Never negative.
    pub fn height(&self) -> f32 {
        (self.top - self.bottom).max(0.0)
    }

    /// Returns the smallest [`BBox`] that fully encloses both `self` and `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdfspatial_core::BBox;
    ///
    /// let a = BBox { left: 0.0, bottom: 0.0, right: 10.0, top: 10.0 };
    /// let b = BBox { left: 5.0, bottom: 5.0, right: 20.0, top: 20.0 };
    /// let union = a.union(&b);
    /// assert_eq!(union, BBox { left: 0.0, bottom: 0.0, right: 20.0, top: 20.0 });
    /// ```
    pub fn union(&self, other: &BBox) -> BBox {
        BBox {
            left: self.left.min(other.left),
            bottom: self.bottom.min(other.bottom),
            right: self.right.max(other.right),
            top: self.top.max(other.top),
        }
    }

    /// Returns `true` if the vertical center of `other` falls within `self`'s vertical
    /// extent (or vice versa), a cheap proxy for "on the same text baseline."
    pub fn vertically_overlaps(&self, other: &BBox) -> bool {
        self.bottom < other.top && other.bottom < self.top
    }

    /// Returns `true` if `other`'s center point falls within `self`'s extent (inclusive
    /// of the boundary). Used to test membership of a small box (e.g. a text block)
    /// inside a larger one (e.g. a detected table or picture region) without requiring
    /// full containment, since a block on a table's edge may extend a point or two past
    /// the table's own ruling lines.
    pub fn contains_center(&self, other: &BBox) -> bool {
        let cx = (other.left + other.right) / 2.0;
        let cy = (other.bottom + other.top) / 2.0;
        self.left <= cx && cx <= self.right && self.bottom <= cy && cy <= self.top
    }
}

/// A single extracted character, tied to its glyph, font, and on-page position.
#[derive(Debug, Clone, PartialEq)]
pub struct Char {
    /// The Unicode scalar value PDFium resolved for this glyph, if any.
    ///
    /// `None` for glyphs PDFium could not map to a Unicode code point (e.g. some
    /// symbolic or malformed embedded fonts); such glyphs still contribute a bounding
    /// box but no text.
    pub unicode: Option<char>,
    /// The character's tight bounding box in page space.
    pub bbox: BBox,
    /// PostScript name of the font used to render this character.
    pub font_name: String,
    /// Font size in points, as scaled on the page.
    pub font_size: f32,
    /// The font's declared weight (e.g. `400` regular, `700` bold), if the font program
    /// exposes one. `None` when PDFium can't resolve a weight (many fonts don't declare
    /// one), in which case [`crate::layout::is_bold_font_name`]'s name-based heuristic is
    /// the only remaining signal.
    pub font_weight: Option<u32>,
    /// The glyph's rotation, in degrees counter-clockwise from upright text. `0.0` for
    /// ordinary horizontal text. Defaults to `0.0` when PDFium can't report an angle
    /// (e.g. characters synthesized outside real extraction, as in the Stage 3 synthetic
    /// corpus) rather than failing extraction over a cosmetic signal.
    pub angle_degrees: f32,
    /// The character's fill color as `(r, g, b, a)`, `0-255` per channel. Defaults to
    /// opaque black (`(0, 0, 0, 255)`) when PDFium can't report a color.
    pub fill_color: (u8, u8, u8, u8),
}

impl Default for Char {
    /// A blank, opaque-black, upright character at the page origin. Existing call sites
    /// that predate `font_weight`/`angle_degrees`/`fill_color` can add `..Default::default()`
    /// to a literal that already sets `unicode`/`bbox`/`font_name`/`font_size` to pick up
    /// sensible values for the rest without touching every construction site by hand.
    fn default() -> Self {
        Char {
            unicode: None,
            bbox: BBox::ZERO,
            font_name: String::new(),
            font_size: 0.0,
            font_weight: None,
            angle_degrees: 0.0,
            fill_color: (0, 0, 0, 255),
        }
    }
}

/// A run of [`Char`]s grouped into a word by x-gap thresholding.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    /// The word's text, concatenated from its characters in extraction order.
    pub text: String,
    /// The union bounding box of all characters in the word.
    pub bbox: BBox,
    /// The characters that make up this word.
    pub chars: Vec<Char>,
}

/// A single line of text, formed by clustering [`Word`]s on a shared baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    /// The line's text, with words joined by single spaces.
    pub text: String,
    /// The union bounding box of all words in the line.
    pub bbox: BBox,
    /// The words that make up this line, left to right.
    pub words: Vec<Word>,
}

/// A paragraph-like group of [`Line`]s, formed by a vertical-gap heuristic.
///
/// Stage 1 blocks carry no semantic tag (heading, list, table, ...); they are a purely
/// geometric grouping. Semantic classification is Stage 2/4 work — see [`layout`].
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    /// The union bounding box of all lines in the block.
    pub bbox: BBox,
    /// The lines that make up this block, top to bottom.
    pub lines: Vec<Line>,
}

impl Block {
    /// Returns the block's text with lines joined by newlines.
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// What kind of mark a [`Graphic`] represents on the page.
///
/// Derived from PDFium's `PdfPageObjectType`, collapsed to the categories the Stage 1b
/// graphics layer actually distinguishes between. Text objects are deliberately excluded
/// here — they already flow through [`Char`]/[`Word`]/[`Line`]/[`Block`], and counting
/// them again as graphics would double-count ink when [`crate::graphics`] unions bboxes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicKind {
    /// A stroked path segment (a ruling line, a table border, an underline).
    Stroke,
    /// A filled path (a shaded box, a highlight, a solid-color rule).
    Fill,
    /// A raster or form XObject (an embedded photo, chart, or scanned figure).
    Image,
    /// A shading pattern (a gradient fill).
    Shading,
}

/// A single non-text mark on the page: a vector path, image, or shading.
///
/// This is the signal PDFium's text layer alone can't provide: ruling lines that key
/// table-grid detection, image placeholders for figures, and z-order for resolving
/// overlapping content. See [`crate::graphics`] for what's built from it.
#[derive(Debug, Clone, PartialEq)]
pub struct Graphic {
    /// What kind of mark this is.
    pub kind: GraphicKind,
    /// The graphic's bounding box in page space.
    pub bbox: BBox,
    /// This graphic's index among the page's objects, in PDFium's own object order --
    /// which is paint order, so a higher `z` was drawn (and so visually sits) on top.
    pub z: usize,
    /// The stroke width in points, `0.0` for unstroked graphics.
    pub stroke_width: f32,
    /// Whether this graphic is stroked (has a visible outline).
    pub is_stroked: bool,
    /// Whether this graphic is filled (has a visible interior).
    pub is_filled: bool,
}

/// A single extracted page.
#[derive(Debug, Clone, PartialEq)]
pub struct Page {
    /// Zero-based page index within the document.
    pub index: usize,
    /// Page width in points.
    pub width: f32,
    /// Page height in points.
    pub height: f32,
    /// Geometric blocks on this page, in raw reading order
    /// (top-to-bottom, left-to-right; see [`extract`] for the exact rule).
    pub blocks: Vec<Block>,
    /// Non-text marks on this page (vector paths, images, shadings), in PDFium's paint
    /// order. Empty for pages built without a graphics-object pass (e.g. most of the
    /// Stage 3 synthetic corpus, which authors [`Block`]s directly).
    pub graphics: Vec<Graphic>,
}

impl Default for Page {
    /// An empty page at the origin with no graphics. Existing call sites that predate
    /// `graphics` can add `..Default::default()` to a literal that already sets
    /// `index`/`width`/`height`/`blocks` to pick up an empty graphics list without
    /// touching every construction site by hand.
    fn default() -> Self {
        Page {
            index: 0,
            width: 0.0,
            height: 0.0,
            blocks: Vec::new(),
            graphics: Vec::new(),
        }
    }
}

impl Page {
    /// Returns the page's text with blocks joined by blank lines.
    pub fn text(&self) -> String {
        self.blocks
            .iter()
            .map(|b| b.text())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// A fully extracted document: the top-level result of [`extract_baseline`].
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    /// Pages in document order.
    pub pages: Vec<Page>,
}

impl Document {
    /// Returns the whole document's raw reading-order text, with pages joined by a
    /// form-feed-style blank-line separator.
    ///
    /// This text carries **zero structural tagging** by design (no headings, lists, or
    /// tables) — Stage 1's job is establishing a lossless extraction floor, not
    /// structure. See the crate-level docs for the four-stage pipeline this feeds into.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdfspatial_core::{BBox, Block, Document, Line, Page, Word};
    ///
    /// let word = Word { text: "Hello".into(), bbox: BBox::ZERO, chars: vec![] };
    /// let line = Line { text: "Hello".into(), bbox: BBox::ZERO, words: vec![word] };
    /// let block = Block { bbox: BBox::ZERO, lines: vec![line] };
    /// let page = Page { index: 0, width: 612.0, height: 792.0, blocks: vec![block], ..Default::default() };
    /// let doc = Document { pages: vec![page] };
    ///
    /// assert_eq!(doc.reading_order_text(), "Hello");
    /// ```
    pub fn reading_order_text(&self) -> String {
        self.pages
            .iter()
            .map(|p| p.text())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Total number of characters extracted across all pages.
    pub fn char_count(&self) -> usize {
        self.pages
            .iter()
            .flat_map(|p| &p.blocks)
            .flat_map(|b| &b.lines)
            .flat_map(|l| &l.words)
            .map(|w| w.chars.len())
            .sum()
    }
}

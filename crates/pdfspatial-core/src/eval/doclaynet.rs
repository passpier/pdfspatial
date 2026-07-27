//! Loader for [DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1)
//! samples in its "DocLayNet-core" on-disk layout: a COCO object-detection JSON file
//! (`images`/`categories`/`annotations`) plus one `pdf_cells` JSON file per page (the
//! PDF text cells DocLayNet ships alongside its layout annotations).
//!
//! Gated behind the `doclaynet` cargo feature, which pulls in `serde`/`serde_json` —
//! the rest of `pdfspatial-core` stays dependency-free. Neither [`crate::BBox`] nor
//! [`crate::layout::Region`]/[`crate::layout::RegionClass`] gain serde derives; this
//! module deserializes into its own private mirror structs and constructs the crate's
//! real types by hand, so serde stays entirely behind this feature gate.
//!
//! # Coordinate frame
//!
//! DocLayNet ground-truth boxes and `pdf_cells` boxes are both given in the COCO image
//! frame: top-left origin, `y` increasing downward, `[x, y, width, height]`. This
//! crate's [`crate::BBox`] is bottom-left origin, `y` increasing upward (see the
//! [`crate::BBox`] docs). [`coco_bbox_to_bbox`] converts one box given the frame's
//! total height; [`load_sample`] applies it to both the ground truth and the cells
//! using the *same* height per page (the COCO `images[].height`), which is required for
//! predicted and ground-truth boxes to be comparable under IoU/GIoU. A caller that
//! instead wants to score predictions from a real PDF via [`crate::extract_baseline`]
//! (PDF-point space) must first rescale the ground truth by `pdf_height / coco_height`
//! before flipping — this loader does not do that for you.
//!
//! Real DocLayNet-core `pdf_cells` files carry font metrics under a nested `font.size`
//! rather than a flat `font_size`, and separately track the original PDF page's point
//! dimensions in a `metadata` object. [`load_sample`] reads both, and computes
//! [`DocLayNetPage::coco_scale`] from the latter for [`document_from_cells_grouped`] to
//! use (see "Prediction granularity" below).
//!
//! # Prediction granularity
//!
//! Two reconstructions turn a [`DocLayNetPage`]'s cells into a [`crate::Document`],
//! trading off legibility against under/over-grouping:
//!
//! - [`document_from_cells`] maps **one `pdf_cells` cell per [`crate::Line`] per
//!   [`crate::Block`]**, skipping Stage 1's word/line/block grouping entirely. DocLayNet
//!   cells are typically word- or short-phrase-sized, so this under-groups relative to a
//!   real extraction — predictions from [`predict_regions_textonly`] (which always uses
//!   this path) honestly score lower than the real pipeline would, and drafts mined via
//!   [`mine_reading_order_failures`] are piles of word-blocks needing heavy editing to
//!   review. This remains the default because it makes no assumption about DocLayNet's
//!   authored cell order being reading-order-adjacent.
//! - [`document_from_cells_grouped`] instead runs the cells through Stage 1's real char →
//!   word → line → block grouping ([`crate::extract::group_chars_into_blocks`]), the
//!   same pure geometry [`crate::extract_baseline`] applies to PDFium's text layer. This
//!   produces realistically-sized blocks, but it also means multi-column pages get their
//!   columns merged line-by-line exactly as real Stage 1 does — which can *reduce* the
//!   reordering signal [`mine_reading_order_failures_grouped`] ranks pages by, since a
//!   merged line can no longer itself be reordered. Use this path (and
//!   `--grouped` on `examples/doclaynet_mine.rs`/`examples/doclaynet_drafts.rs`) to
//!   compare against the default on a real sample before deciding which one to mine
//!   with.

use crate::extract::group_chars_into_blocks;
use crate::layout::{Region, RegionClass, classify_regions};
use crate::{BBox, Block, Char, Document, Line, Page, Word};
use std::collections::HashMap;
use std::path::Path;

use super::{PageReorderRank, Stage2Report, evaluate_pages, rank_pages_by_reorder};

/// A single-cell font size assumed for `pdf_cells` entries that don't carry one (the
/// DocLayNet `pdf_cells` schema doesn't always include font metrics), used only so
/// [`crate::layout::classify_regions`]'s font-size heuristics have a non-degenerate
/// input.
const DEFAULT_FONT_SIZE: f32 = 10.0;

/// Errors returned by [`load_sample`].
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// Failed to read a file from disk.
    #[error("I/O error reading {path}: {source}")]
    Io {
        /// The path that failed to read.
        path: std::path::PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Failed to parse a file as JSON.
    #[error("JSON error in {path}: {source}")]
    Json {
        /// The path that failed to parse.
        path: std::path::PathBuf,
        /// The underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// A COCO annotation referenced a `category_id` not in DocLayNet's 11-class schema.
    #[error("unknown DocLayNet category id {0}")]
    UnknownCategory(i64),
    /// An image listed in the COCO JSON has no corresponding `pdf_cells` file under
    /// either of the naming conventions [`load_sample`] tries.
    #[error("no pdf_cells file for image {file_name:?}; tried {tried:?}")]
    MissingCells {
        /// The image's `file_name`, per the COCO JSON.
        file_name: String,
        /// Every candidate path [`load_sample`] checked before giving up.
        tried: Vec<std::path::PathBuf>,
    },
}

/// A loaded DocLayNet sample: every page's ground truth and reconstructable text
/// content, ready to be scored by [`evaluate_sample`].
#[derive(Debug, Clone)]
pub struct DocLayNetSample {
    /// Pages in the sample, in the order the COCO JSON listed their images.
    pub pages: Vec<DocLayNetPage>,
}

/// One page of a [`DocLayNetSample`]: DocLayNet's ground-truth layout regions and PDF
/// text cells, both already converted into this crate's coordinate frame (see the
/// [module docs](self) for the conversion this applies).
#[derive(Debug, Clone)]
pub struct DocLayNetPage {
    /// DocLayNet's COCO `image_id` for this page.
    pub image_id: i64,
    /// The `file_name` DocLayNet's COCO JSON recorded for this page's rendered image;
    /// also used to locate its `pdf_cells` file (see [`load_sample`]).
    pub file_name: String,
    /// Page width in the COCO image frame (pixels).
    pub width: f32,
    /// Page height in the COCO image frame (pixels); the frame height used to convert
    /// every box on this page from COCO's top-left frame to [`crate::BBox`]'s
    /// bottom-left frame.
    pub height: f32,
    /// Ground-truth regions, converted from the COCO annotations for this image.
    /// `confidence` is always `1.0` (ground truth, not a prediction).
    pub ground_truth: Vec<Region>,
    /// PDF text cells for this page, converted from its `pdf_cells` file.
    pub cells: Vec<PdfCell>,
    /// Ratio of the COCO image frame's height to the original PDF page's height in
    /// points (`coco_height / original_height`), if the page's `pdf_cells` file carried
    /// enough `metadata` to compute it; `1.0` otherwise. [`document_from_cells_grouped`]
    /// uses this to convert [`PdfCell::font_size`] into the COCO pixel frame before
    /// running Stage 1's real, font-size-relative grouping thresholds on cell
    /// geometry that is itself already in that frame -- see the [module
    /// docs](self#coordinate-frame). [`document_from_cells`] ignores it, so the
    /// default (one-cell-per-block) reconstruction is unaffected.
    pub coco_scale: f32,
}

/// One PDF text cell from DocLayNet's `pdf_cells` annotation, converted into this
/// crate's coordinate frame.
#[derive(Debug, Clone)]
pub struct PdfCell {
    /// The cell's text content.
    pub text: String,
    /// The cell's bounding box, already converted to [`crate::BBox`]'s bottom-left
    /// frame.
    pub bbox: BBox,
    /// Font size in points, if `pdf_cells` carried one; otherwise
    /// [`DEFAULT_FONT_SIZE`].
    pub font_size: f32,
}

// --- On-disk schema (private; mirrors DocLayNet's JSON, not this crate's types) -----

#[derive(serde::Deserialize)]
struct CocoDataset {
    images: Vec<CocoImage>,
    categories: Vec<CocoCategory>,
    annotations: Vec<CocoAnnotation>,
}

#[derive(serde::Deserialize)]
struct CocoImage {
    id: i64,
    file_name: String,
    width: f32,
    height: f32,
}

#[derive(serde::Deserialize)]
struct CocoCategory {
    id: i64,
    name: String,
}

#[derive(serde::Deserialize)]
struct CocoAnnotation {
    image_id: i64,
    category_id: i64,
    bbox: [f32; 4],
}

#[derive(serde::Deserialize)]
struct PdfCellsPage {
    cells: Vec<PdfCellRaw>,
    /// Present on real DocLayNet-core page JSON (absent from the flat
    /// `{stem}.cells.json` shape this crate's own vendored fixtures use); carries the
    /// original PDF page dimensions alongside the COCO frame's, letting [`load_sample`]
    /// compute [`DocLayNetPage::coco_scale`]. See the [module docs](self#coordinate-frame).
    #[serde(default)]
    metadata: Option<PdfCellsMetadataRaw>,
}

#[derive(serde::Deserialize)]
struct PdfCellsMetadataRaw {
    #[serde(default)]
    original_width: Option<f32>,
    #[serde(default)]
    original_height: Option<f32>,
}

#[derive(serde::Deserialize)]
struct PdfCellRaw {
    text: String,
    bbox: [f32; 4],
    #[serde(default)]
    font_size: Option<f32>,
    /// Real DocLayNet-core cells nest font metrics under this object rather than a flat
    /// `font_size` field; [`load_sample`] falls back to it when `font_size` is absent.
    #[serde(default)]
    font: Option<PdfCellFontRaw>,
}

#[derive(serde::Deserialize)]
struct PdfCellFontRaw {
    #[serde(default)]
    size: Option<f32>,
}

// --- Conversions ----------------------------------------------------------------

/// Converts a COCO-style `[x, y, width, height]` box (top-left origin, `y`-down) to a
/// [`crate::BBox`] (bottom-left origin, `y`-up), given the total frame height.
///
/// ```text
/// left   = x
/// right  = x + width
/// top    = frame_height - y
/// bottom = frame_height - y - height
/// ```
///
/// # Examples
///
/// ```
/// use pdfspatial_core::eval::doclaynet::coco_bbox_to_bbox;
/// use pdfspatial_core::BBox;
///
/// let bbox = coco_bbox_to_bbox([100.0, 20.0, 300.0, 25.0], 1025.0);
/// assert_eq!(bbox, BBox { left: 100.0, bottom: 980.0, right: 400.0, top: 1005.0 });
/// ```
pub fn coco_bbox_to_bbox(xywh: [f32; 4], frame_height: f32) -> BBox {
    let [x, y, width, height] = xywh;
    BBox {
        left: x,
        bottom: frame_height - y - height,
        right: x + width,
        top: frame_height - y,
    }
}

/// Maps a DocLayNet COCO `category_id` to its [`RegionClass`], by DocLayNet's
/// documented id→name mapping (ids `1..=11`, matching [`RegionClass`]'s declaration
/// order: `Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture,
/// Section-header, Table, Text, Title`). Returns `None` for any other id.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::eval::doclaynet::category_to_class;
/// use pdfspatial_core::layout::RegionClass;
///
/// assert_eq!(category_to_class(11), Some(RegionClass::Title));
/// assert_eq!(category_to_class(0), None);
/// ```
pub fn category_to_class(category_id: i64) -> Option<RegionClass> {
    Some(match category_id {
        1 => RegionClass::Caption,
        2 => RegionClass::Footnote,
        3 => RegionClass::Formula,
        4 => RegionClass::ListItem,
        5 => RegionClass::PageFooter,
        6 => RegionClass::PageHeader,
        7 => RegionClass::Picture,
        8 => RegionClass::SectionHeader,
        9 => RegionClass::Table,
        10 => RegionClass::Text,
        11 => RegionClass::Title,
        _ => return None,
    })
}

/// Maps a DocLayNet COCO category *name* (e.g. `"Section-header"`) to its
/// [`RegionClass`], as a fallback for callers that only have the `categories` array's
/// `name` field rather than a numeric id matching [`category_to_class`]'s assumed
/// ordering.
fn category_name_to_class(name: &str) -> Option<RegionClass> {
    Some(match name {
        "Caption" => RegionClass::Caption,
        "Footnote" => RegionClass::Footnote,
        "Formula" => RegionClass::Formula,
        "List-item" => RegionClass::ListItem,
        "Page-footer" => RegionClass::PageFooter,
        "Page-header" => RegionClass::PageHeader,
        "Picture" => RegionClass::Picture,
        "Section-header" => RegionClass::SectionHeader,
        "Table" => RegionClass::Table,
        "Text" => RegionClass::Text,
        "Title" => RegionClass::Title,
        _ => return None,
    })
}

/// Loads a DocLayNet sample from a COCO JSON file and a directory of per-page
/// `pdf_cells` JSON files.
///
/// `cells_dir` is expected to contain one file per image named either
/// `{file_stem}.cells.json` (the vendored test fixtures' naming, under
/// `tests/fixtures/doclaynet/`) or `{file_stem}.json` (a real unpacked DocLayNet-core
/// `JSON/` directory's own naming), where `file_stem` is `coco_json`'s
/// `images[].file_name` with its extension removed (e.g. image `page_0001.png` → cells
/// file `page_0001.cells.json` or `page_0001.json`). Both are tried, in that order, so a
/// real download needs no renaming.
///
/// Each cell's font size is read from a flat `font_size` field if present, else a
/// nested `font.size` object (real DocLayNet-core's shape); either missing falls back to
/// [`DEFAULT_FONT_SIZE`].
///
/// # Errors
///
/// Returns [`EvalError::Io`]/[`EvalError::Json`] if either file can't be read or
/// parsed, [`EvalError::UnknownCategory`] if an annotation's `category_id` isn't one of
/// DocLayNet's 11 known ids/names, or [`EvalError::MissingCells`] if an image has no
/// cells file under either naming convention.
pub fn load_sample(coco_json: &Path, cells_dir: &Path) -> Result<DocLayNetSample, EvalError> {
    let coco = read_json::<CocoDataset>(coco_json)?;

    let mut category_names: HashMap<i64, &str> = HashMap::new();
    for category in &coco.categories {
        category_names.insert(category.id, category.name.as_str());
    }
    let class_for = |category_id: i64| -> Result<RegionClass, EvalError> {
        category_to_class(category_id)
            .or_else(|| {
                category_names
                    .get(&category_id)
                    .and_then(|n| category_name_to_class(n))
            })
            .ok_or(EvalError::UnknownCategory(category_id))
    };

    let mut annotations_by_image: HashMap<i64, Vec<&CocoAnnotation>> = HashMap::new();
    for annotation in &coco.annotations {
        annotations_by_image
            .entry(annotation.image_id)
            .or_default()
            .push(annotation);
    }

    let mut pages = Vec::with_capacity(coco.images.len());
    for image in &coco.images {
        let ground_truth = annotations_by_image
            .get(&image.id)
            .into_iter()
            .flatten()
            .map(|annotation| {
                Ok(Region {
                    class: class_for(annotation.category_id)?,
                    bbox: coco_bbox_to_bbox(annotation.bbox, image.height),
                    confidence: 1.0,
                })
            })
            .collect::<Result<Vec<_>, EvalError>>()?;

        let stem = Path::new(&image.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&image.file_name);
        let candidates = [
            cells_dir.join(format!("{stem}.cells.json")),
            cells_dir.join(format!("{stem}.json")),
        ];
        let Some(cells_path) = candidates.iter().find(|p| p.exists()) else {
            return Err(EvalError::MissingCells {
                file_name: image.file_name.clone(),
                tried: candidates.to_vec(),
            });
        };
        let cells_raw = read_json::<PdfCellsPage>(cells_path)?;

        let coco_scale = cells_raw
            .metadata
            .as_ref()
            .and_then(|m| Some((m.original_width?, m.original_height?)))
            .filter(|(w, h)| w.is_finite() && h.is_finite() && *w > 0.0 && *h > 0.0)
            .map(|(_, original_height)| image.height / original_height)
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .unwrap_or(1.0);

        let cells = cells_raw
            .cells
            .into_iter()
            .map(|c| PdfCell {
                text: c.text,
                bbox: coco_bbox_to_bbox(c.bbox, image.height),
                font_size: c
                    .font_size
                    .or(c.font.and_then(|f| f.size))
                    .unwrap_or(DEFAULT_FONT_SIZE),
            })
            .collect();

        pages.push(DocLayNetPage {
            image_id: image.id,
            file_name: image.file_name.clone(),
            width: image.width,
            height: image.height,
            ground_truth,
            cells,
            coco_scale,
        });
    }

    Ok(DocLayNetSample { pages })
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, EvalError> {
    let text = std::fs::read_to_string(path).map_err(|source| EvalError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| EvalError::Json {
        path: path.to_path_buf(),
        source,
    })
}

/// Reconstructs a single-page [`Document`] from a [`DocLayNetPage`]'s PDF text cells,
/// so [`crate::layout::classify_regions`] can run on it without native PDFium.
///
/// Maps each cell to its own [`Line`] inside its own [`Block`] — see the [module
/// docs](self#prediction-granularity) for why this under-groups relative to a real
/// extraction, and its effect on prediction quality.
pub fn document_from_cells(page: &DocLayNetPage) -> Document {
    let blocks = page
        .cells
        .iter()
        .filter(|cell| !cell.text.trim().is_empty())
        .map(|cell| {
            let chars: Vec<Char> = cell
                .text
                .chars()
                .map(|ch| Char {
                    unicode: Some(ch),
                    bbox: cell.bbox,
                    font_name: "DocLayNet".to_string(),
                    font_size: cell.font_size,
                })
                .collect();
            let word = Word {
                text: cell.text.clone(),
                bbox: cell.bbox,
                chars,
            };
            let line = Line {
                text: cell.text.clone(),
                bbox: cell.bbox,
                words: vec![word],
            };
            Block {
                bbox: cell.bbox,
                lines: vec![line],
            }
        })
        .collect();

    Document {
        pages: vec![Page {
            index: 0,
            width: page.width,
            height: page.height,
            blocks,
        }],
    }
}

/// Reconstructs a single-page [`Document`] from a [`DocLayNetPage`]'s PDF text cells,
/// running Stage 1's real char → word → line → block grouping
/// ([`crate::extract::group_chars_into_blocks`]) instead of [`document_from_cells`]'s
/// one-cell-per-block mapping.
///
/// Cells are concatenated in the order `page.cells` lists them (the same order
/// [`document_from_cells`] preserves), each cell's characters carrying its bbox and its
/// font size scaled into the COCO pixel frame via [`DocLayNetPage::coco_scale`] --
/// grouping's thresholds are expressed as multiples of font size, so this keeps them
/// comparable to bbox geometry that is itself already in that frame.
///
/// See the [module docs](self#prediction-granularity) for the trade-off against
/// [`document_from_cells`]: real grouping merges same-baseline text across column
/// gutters exactly as Stage 1's PDFium-backed path does, which is more honest but can
/// erase the reordering signal [`mine_reading_order_failures_grouped`] looks for on some
/// pages.
pub fn document_from_cells_grouped(page: &DocLayNetPage) -> Document {
    let chars: Vec<Char> = page
        .cells
        .iter()
        .filter(|cell| !cell.text.trim().is_empty())
        .flat_map(|cell| {
            let font_size = cell.font_size * page.coco_scale;
            cell.text.chars().map(move |ch| Char {
                unicode: Some(ch),
                bbox: cell.bbox,
                font_name: "DocLayNet".to_string(),
                font_size,
            })
        })
        .collect();

    let blocks = group_chars_into_blocks(&chars);

    Document {
        pages: vec![Page {
            index: 0,
            width: page.width,
            height: page.height,
            blocks,
        }],
    }
}

/// Runs [`crate::layout::classify_regions`] on the [`Document`] reconstructed by
/// [`document_from_cells`], producing the text-only predictions [`evaluate_sample`]
/// scores against `page.ground_truth`.
pub fn predict_regions_textonly(page: &DocLayNetPage) -> Vec<Region> {
    classify_regions(&document_from_cells(page))
}

/// Scores [`predict_regions_textonly`]'s predictions against `sample`'s ground truth
/// for every page, aggregated into a single [`Stage2Report`] via [`evaluate_pages`].
pub fn evaluate_sample(sample: &DocLayNetSample, iou_threshold: f32) -> Stage2Report {
    let pages: Vec<(Vec<Region>, Vec<Region>)> = sample
        .pages
        .iter()
        .map(|page| (predict_regions_textonly(page), page.ground_truth.clone()))
        .collect();
    evaluate_pages(&pages, iou_threshold)
}

/// One DocLayNet page's Stage 3 mining rank: its
/// [`PageReorderRank`](super::PageReorderRank) plus the identifiers needed to locate
/// the source page for minimal-repro extraction.
#[derive(Debug, Clone)]
pub struct MinedPage {
    /// The page's DocLayNet COCO `image_id`, matching [`DocLayNetPage::image_id`].
    pub image_id: i64,
    /// The page's rendered-image file name, matching [`DocLayNetPage::file_name`] --
    /// also the key for locating a source PDF (see `examples/doclaynet_mine.rs`'s
    /// `--pdfium` mode).
    pub file_name: String,
    /// The reordering-severity rank for this page.
    pub rank: PageReorderRank,
}

/// Ranks every page in `sample` by reading-order reordering severity -- the Stage 3
/// failure-cluster prioritization signal described in [`super::rank_pages_by_reorder`]
/// -- reconstructing each page's [`Document`] text-only via [`document_from_cells`], so
/// this needs no native PDFium. Most-reordered-first, i.e. the pages worth mining into
/// minimal-repro `fixtures/` cases next.
///
/// A caller with a real PDFium library available can instead rank real
/// [`crate::extract_baseline`] output by calling [`super::rank_pages_by_reorder`]
/// directly on documents built from the source PDFs (see `examples/doclaynet_mine.rs`'s
/// `--pdfium` mode) -- this function's text-only path exists so mining stays runnable
/// with nothing but a DocLayNet download.
pub fn mine_reading_order_failures(sample: &DocLayNetSample) -> Vec<MinedPage> {
    mine_reading_order_failures_with(sample, document_from_cells)
}

/// Like [`mine_reading_order_failures`], but reconstructing each page via
/// [`document_from_cells_grouped`]'s real Stage 1 grouping instead of
/// [`document_from_cells`]'s one-cell-per-block mapping. See
/// [`document_from_cells_grouped`]'s docs for the trade-off this implies for the
/// reordering-severity signal itself.
pub fn mine_reading_order_failures_grouped(sample: &DocLayNetSample) -> Vec<MinedPage> {
    mine_reading_order_failures_with(sample, document_from_cells_grouped)
}

/// Shared body of [`mine_reading_order_failures`] and
/// [`mine_reading_order_failures_grouped`], parameterized by which reconstruction to
/// rank so the two entry points can't drift apart.
fn mine_reading_order_failures_with(
    sample: &DocLayNetSample,
    reconstruct: impl Fn(&DocLayNetPage) -> Document,
) -> Vec<MinedPage> {
    let mut mined: Vec<MinedPage> = sample
        .pages
        .iter()
        .map(|page| {
            let document = reconstruct(page);
            // Each DocLayNet page reconstructs to a single-page `Document`, so there is
            // exactly one rank to pull out.
            let rank = rank_pages_by_reorder(&document)
                .into_iter()
                .next()
                .unwrap_or(PageReorderRank {
                    page_index: 0,
                    block_count: 0,
                    reorder_edit_distance: 0,
                });
            MinedPage {
                image_id: page.image_id,
                file_name: page.file_name.clone(),
                rank,
            }
        })
        .collect();

    mined.sort_by(|a, b| {
        b.rank
            .reorder_edit_distance
            .cmp(&a.rank.reorder_edit_distance)
            .then(a.image_id.cmp(&b.image_id))
    });
    mined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coco_bbox_to_bbox_flips_top_left_to_bottom_left() {
        let bbox = coco_bbox_to_bbox([100.0, 20.0, 300.0, 25.0], 1025.0);
        assert_eq!(
            bbox,
            BBox {
                left: 100.0,
                bottom: 980.0,
                right: 400.0,
                top: 1005.0,
            }
        );
    }

    #[test]
    fn category_to_class_covers_all_eleven_ids_in_declared_order() {
        assert_eq!(category_to_class(1), Some(RegionClass::Caption));
        assert_eq!(category_to_class(2), Some(RegionClass::Footnote));
        assert_eq!(category_to_class(3), Some(RegionClass::Formula));
        assert_eq!(category_to_class(4), Some(RegionClass::ListItem));
        assert_eq!(category_to_class(5), Some(RegionClass::PageFooter));
        assert_eq!(category_to_class(6), Some(RegionClass::PageHeader));
        assert_eq!(category_to_class(7), Some(RegionClass::Picture));
        assert_eq!(category_to_class(8), Some(RegionClass::SectionHeader));
        assert_eq!(category_to_class(9), Some(RegionClass::Table));
        assert_eq!(category_to_class(10), Some(RegionClass::Text));
        assert_eq!(category_to_class(11), Some(RegionClass::Title));
        assert_eq!(category_to_class(12), None);
        assert_eq!(category_to_class(0), None);
    }

    #[test]
    fn category_name_to_class_matches_id_mapping() {
        for id in 1..=11 {
            let by_id = category_to_class(id).unwrap();
            let name = match by_id {
                RegionClass::Caption => "Caption",
                RegionClass::Footnote => "Footnote",
                RegionClass::Formula => "Formula",
                RegionClass::ListItem => "List-item",
                RegionClass::PageFooter => "Page-footer",
                RegionClass::PageHeader => "Page-header",
                RegionClass::Picture => "Picture",
                RegionClass::SectionHeader => "Section-header",
                RegionClass::Table => "Table",
                RegionClass::Text => "Text",
                RegionClass::Title => "Title",
            };
            assert_eq!(category_name_to_class(name), Some(by_id));
        }
    }

    #[test]
    fn document_from_cells_skips_blank_cells_and_preserves_bbox() {
        let page = DocLayNetPage {
            image_id: 1,
            file_name: "page_0001.png".into(),
            width: 100.0,
            height: 100.0,
            ground_truth: vec![],
            cells: vec![
                PdfCell {
                    text: "Hello".into(),
                    bbox: BBox {
                        left: 0.0,
                        bottom: 0.0,
                        right: 10.0,
                        top: 10.0,
                    },
                    font_size: 10.0,
                },
                PdfCell {
                    text: "   ".into(),
                    bbox: BBox::ZERO,
                    font_size: 10.0,
                },
            ],
            coco_scale: 1.0,
        };

        let document = document_from_cells(&page);
        assert_eq!(document.pages.len(), 1);
        assert_eq!(document.pages[0].blocks.len(), 1);
        assert_eq!(document.pages[0].blocks[0].text(), "Hello");
    }

    fn cell(text: &str, left: f32, bottom: f32, right: f32, top: f32) -> PdfCell {
        PdfCell {
            text: text.into(),
            bbox: BBox {
                left,
                bottom,
                right,
                top,
            },
            font_size: DEFAULT_FONT_SIZE,
        }
    }

    #[test]
    fn mine_reading_order_failures_ranks_multi_column_page_above_single_column() {
        // Two columns, cells authored interleaved (naive reading order) with a 40pt
        // gutter on a 600pt-wide page -- well above assemble::MIN_CUT_FRACTION, so
        // assemble_reading_order recovers column-major order and this page reorders
        // heavily relative to the naive order document_from_cells preserves.
        let multi_column = DocLayNetPage {
            image_id: 2,
            file_name: "page_0002.png".into(),
            width: 600.0,
            height: 800.0,
            ground_truth: vec![],
            cells: vec![
                cell("Right one", 320.0, 700.0, 560.0, 750.0),
                cell("Left one", 40.0, 700.0, 280.0, 750.0),
                cell("Right two", 320.0, 640.0, 560.0, 690.0),
                cell("Left two", 40.0, 640.0, 280.0, 690.0),
            ],
            coco_scale: 1.0,
        };
        let single_column = DocLayNetPage {
            image_id: 1,
            file_name: "page_0001.png".into(),
            width: 600.0,
            height: 800.0,
            ground_truth: vec![],
            cells: vec![
                cell("First", 40.0, 700.0, 560.0, 750.0),
                cell("Second", 40.0, 600.0, 560.0, 650.0),
            ],
            coco_scale: 1.0,
        };

        let sample = DocLayNetSample {
            pages: vec![single_column, multi_column],
        };
        let mined = mine_reading_order_failures(&sample);

        assert_eq!(mined.len(), 2);
        assert_eq!(mined[0].image_id, 2);
        assert_eq!(mined[0].file_name, "page_0002.png");
        assert!(mined[0].rank.reorder_edit_distance > 0);
        assert_eq!(mined[1].image_id, 1);
        assert_eq!(mined[1].rank.reorder_edit_distance, 0);
    }
}

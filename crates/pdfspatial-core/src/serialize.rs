//! Markdown serialization.
//!
//! [`to_markdown`] renders a [`Document`] to Markdown at Stage 1 fidelity: each geometric
//! [`crate::Block`] becomes a paragraph, separated by a blank line, with no headings,
//! lists, or tables — because Stage 1 performs no structural classification (see
//! [`crate::extract`]). It is deliberately left unchanged by Stage 2/4 work so callers
//! that only want the lossless Stage 1 floor keep getting it.
//!
//! [`to_markdown_structured`] renders the Stage 2/4a structural Markdown instead: `#`/`##`
//! headings from [`crate::layout::classify_regions`]'s `Title`/`SectionHeader` output,
//! `-` list items, italicized captions, footnotes, GFM pipe tables, and image
//! placeholders, over blocks already reordered by
//! [`crate::assemble::assemble_reading_order`].

use crate::layout::{Region, RegionClass};
use crate::{Block, Document, Page};
use crate::{graphics, layout, metrics};

/// Renders `document` to Markdown at Stage 1 fidelity: one paragraph per geometric
/// block, blocks separated by a blank line, and pages separated by a Markdown thematic
/// break (`---`). No structural tagging is applied.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::serialize::to_markdown;
/// use pdfspatial_core::{BBox, Block, Document, Line, Page, Word};
///
/// let word = Word { text: "Hello".into(), bbox: BBox::ZERO, chars: vec![] };
/// let line = Line { text: "Hello".into(), bbox: BBox::ZERO, words: vec![word] };
/// let block = Block { bbox: BBox::ZERO, lines: vec![line] };
/// let page = Page { index: 0, width: 612.0, height: 792.0, blocks: vec![block], ..Default::default() };
/// let doc = Document { pages: vec![page] };
///
/// assert_eq!(to_markdown(&doc), "Hello");
/// ```
pub fn to_markdown(document: &Document) -> String {
    document
        .pages
        .iter()
        .map(|page| {
            page.blocks
                .iter()
                .map(|block| block.text())
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n")
}

/// Renders `document` to structural Markdown, using `regions` (as produced by
/// [`crate::layout::classify_regions`]) to decide each block's Markdown treatment:
///
/// - [`RegionClass::Title`] → `# ` heading
/// - [`RegionClass::SectionHeader`] → `## ` heading
/// - [`RegionClass::ListItem`] → `- ` list item
/// - [`RegionClass::Caption`] → italicized (`*...*`) paragraph
/// - [`RegionClass::Footnote`] → a `[^n]:`-style Markdown footnote definition
/// - [`RegionClass::PageHeader`] / [`RegionClass::PageFooter`] → omitted entirely
/// - [`RegionClass::Table`] → a GFM pipe table, reconstructed by
///   [`crate::graphics::table_grid_cells`] from the region's ruling lines and every
///   block it contains -- rendered once per table, not once per member block
/// - [`RegionClass::Picture`] → a Markdown image placeholder (`![]()`, no `src`: Stage
///   1b extracts an image's bbox, not its raster bytes)
/// - everything else (including [`RegionClass::Text`] and any class the heuristic
///   classifier never produces) → a plain paragraph, same as [`to_markdown`]
///
/// A block is matched to its region by highest-overlap IoU (via [`crate::metrics::iou`])
/// rather than exact bounding-box equality, so a block whose bbox was minutely
/// recomputed upstream (e.g. [`crate::assemble::stitch_cross_page_continuations`]) still
/// finds its region instead of silently degrading to a plain paragraph; a block with no
/// region overlapping it by more than [`MATCH_IOU_THRESHOLD`] falls back to one. Pages
/// are separated by a Markdown thematic break (`---`), as in [`to_markdown`].
///
/// # Examples
///
/// ```
/// use pdfspatial_core::layout::{Region, RegionClass};
/// use pdfspatial_core::serialize::to_markdown_structured;
/// use pdfspatial_core::{BBox, Block, Document, Line, Page, Word};
///
/// let bbox = BBox { left: 0.0, bottom: 0.0, right: 100.0, top: 20.0 };
/// let word = Word { text: "Title".into(), bbox, chars: vec![] };
/// let line = Line { text: "Title".into(), bbox, words: vec![word] };
/// let block = Block { bbox, lines: vec![line] };
/// let page = Page { index: 0, width: 612.0, height: 792.0, blocks: vec![block], ..Default::default() };
/// let doc = Document { pages: vec![page] };
///
/// let regions = vec![Region { class: RegionClass::Title, bbox, confidence: 1.0, page: 0 }];
/// assert_eq!(to_markdown_structured(&doc, &regions), "# Title");
/// ```
pub fn to_markdown_structured(document: &Document, regions: &[Region]) -> String {
    to_markdown_structured_with(document, regions, MarkdownOptions::default())
}

/// Output-shape switches for [`to_markdown_structured_with`] and [`to_markdown_pipeline`].
///
/// The library default (both fields `true`) is *faithful*: every page break and every
/// detected picture is represented, matching [`to_markdown_structured`]'s frozen output
/// exactly. The flags exist for downstream consumers -- e.g. a scoring harness that
/// diffs plain text -- that treat Markdown syntax as document content and would
/// otherwise penalize a thematic break or an image placeholder as inserted text. Turning
/// a flag off discards information (page boundaries, or the presence of a picture), so
/// prefer the default unless a specific consumer needs it.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::MarkdownOptions;
///
/// assert_eq!(MarkdownOptions::default(), MarkdownOptions { page_breaks: true, image_placeholders: true });
///
/// let compact = MarkdownOptions { page_breaks: false, ..Default::default() };
/// assert!(!compact.page_breaks);
/// assert!(compact.image_placeholders);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkdownOptions {
    /// Separate pages with a Markdown thematic break (`---`). Default `true`.
    pub page_breaks: bool,
    /// Emit a `![]()` placeholder for each detected [`RegionClass::Picture`]. Default
    /// `true`.
    pub image_placeholders: bool,
}

impl Default for MarkdownOptions {
    fn default() -> Self {
        Self {
            page_breaks: true,
            image_placeholders: true,
        }
    }
}

/// As [`to_markdown_structured`], but honoring `options`. See [`MarkdownOptions`] for
/// what each switch does and why the default differs from it.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::layout::{Region, RegionClass};
/// use pdfspatial_core::serialize::to_markdown_structured_with;
/// use pdfspatial_core::{BBox, Block, Document, Line, MarkdownOptions, Page, Word};
///
/// fn page(index: usize, text: &str) -> Page {
///     let bbox = BBox { left: 0.0, bottom: 700.0, right: 500.0, top: 740.0 };
///     let word = Word { text: text.into(), bbox, chars: vec![] };
///     let line = Line { text: text.into(), bbox, words: vec![word] };
///     Page { index, width: 612.0, height: 792.0, blocks: vec![Block { bbox, lines: vec![line] }], ..Default::default() }
/// }
/// let doc = Document { pages: vec![page(0, "Alpha"), page(1, "Beta")] };
///
/// let options = MarkdownOptions { page_breaks: false, ..Default::default() };
/// assert_eq!(to_markdown_structured_with(&doc, &[], options), "Alpha\n\nBeta");
/// ```
pub fn to_markdown_structured_with(
    document: &Document,
    regions: &[Region],
    options: MarkdownOptions,
) -> String {
    let separator = if options.page_breaks {
        "\n\n---\n\n"
    } else {
        "\n\n"
    };
    document
        .pages
        .iter()
        .map(|page| render_page(page, regions, options))
        // With page breaks off there is no separator carrying an empty page's "I exist"
        // signal, so an empty render (a genuinely blank page) must be dropped here --
        // otherwise adjacent separators collapse into a spurious blank-line run. With
        // page breaks on, every page -- even an empty one -- keeps its own `---`, so
        // nothing is filtered.
        .filter(|rendered| options.page_breaks || !rendered.is_empty())
        .collect::<Vec<_>>()
        .join(separator)
}

/// Runs the full Stage 2/4a chain in one call: splits blocks at internal style breaks
/// ([`crate::layout::split_blocks_at_style_breaks`]), classifies regions
/// ([`crate::layout::classify_regions`]), reassembles reading order
/// ([`crate::assemble::assemble_reading_order`]), and serializes the result
/// ([`to_markdown_structured_with`]).
///
/// The style-break split runs first, over a *copy* of `document` -- see
/// [`crate::layout::split_blocks_at_style_breaks`]'s docs for why this lives here rather
/// than in Stage 1 extraction. Classification then runs on the *split* document, before
/// reassembly: it operates on each block's own geometry (font size, position, shape) and
/// doesn't depend on cross-block ordering, so the sequence doesn't affect its output.
/// Serialization, though, uses the *reassembled* document's blocks matched back against
/// those regions by IoU -- [`to_markdown_structured_with`]'s docs. That match is
/// deliberately tolerant (see `MATCH_IOU_THRESHOLD`) rather than exact, because
/// reassembly can union bounding boxes together (e.g. stitching a paragraph split across
/// a page boundary), which nudges a block's bbox just enough that exact equality would
/// silently drop the match.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::serialize::to_markdown_pipeline;
/// use pdfspatial_core::{BBox, Block, Document, Line, MarkdownOptions, Page, Word};
///
/// let bbox = BBox { left: 0.0, bottom: 380.0, right: 500.0, top: 420.0 };
/// let word = Word { text: "Hello".into(), bbox, chars: vec![] };
/// let line = Line { text: "Hello".into(), bbox, words: vec![word] };
/// let block = Block { bbox, lines: vec![line] };
/// let page = Page { index: 0, width: 612.0, height: 792.0, blocks: vec![block], ..Default::default() };
/// let doc = Document { pages: vec![page] };
///
/// assert_eq!(to_markdown_pipeline(&doc, MarkdownOptions::default()), "Hello");
/// ```
pub fn to_markdown_pipeline(document: &Document, options: MarkdownOptions) -> String {
    let split = layout::split_blocks_at_style_breaks(document);
    let regions = layout::classify_regions(&split);
    let reordered = crate::assemble::assemble_reading_order(&split);
    to_markdown_structured_with(&reordered, &regions, options)
}

/// The minimum IoU between a block and a region for them to be considered "the same
/// block" when matching (see [`to_markdown_structured`]'s docs). Well above what any two
/// distinct, non-overlapping blocks could reach, but tolerant of the sub-point float
/// drift a bbox can pick up from being recomputed (e.g. unioned during stitching).
const MATCH_IOU_THRESHOLD: f32 = 0.9;

/// Renders one page: blocks claimed by a detected table are folded into a single GFM
/// table at that table's position (top-y of its bbox); a picture region with no
/// containing/overlapping block is inserted as its own item at its own position
/// (unless `options.image_placeholders` is off); everything else renders block-by-block
/// via [`render_block`].
fn render_page(page: &Page, regions: &[Region], options: MarkdownOptions) -> String {
    enum Item<'a> {
        Block(&'a Block),
        Table(&'a Region),
        Picture,
    }

    // `regions` is the whole document's region list (one flat `Vec` covering every
    // page -- `Region::page` is what disambiguates them, since two different pages'
    // bbox coordinate spaces both start near `(0, 0)` and can otherwise collide). Scope
    // to this page *before* doing anything else below, or a picture/table on another
    // page can suppress this page's text or duplicate its own placeholder once per page.
    let page_regions: Vec<&Region> = regions.iter().filter(|r| r.page == page.index).collect();

    let table_regions: Vec<&Region> = page_regions
        .iter()
        .copied()
        .filter(|r| r.class == RegionClass::Table)
        .collect();
    let picture_regions: Vec<&Region> = page_regions
        .iter()
        .copied()
        .filter(|r| r.class == RegionClass::Picture)
        .collect();

    let mut items: Vec<(f32, Item)> = Vec::new();
    let mut rendered_tables: Vec<&Region> = Vec::new();

    'blocks: for block in &page.blocks {
        for table in &table_regions {
            if table.bbox.contains_center(&block.bbox) {
                if !rendered_tables.iter().any(|r| std::ptr::eq(*r, *table)) {
                    items.push((table.bbox.top, Item::Table(table)));
                    rendered_tables.push(table);
                }
                continue 'blocks;
            }
        }
        if picture_regions
            .iter()
            .any(|p| p.bbox.contains_center(&block.bbox))
        {
            // A block sitting inside a picture's own bbox (rare -- most pictures have
            // no overlapping text block at all) is dropped rather than duplicated: the
            // picture placeholder below already represents that region of the page.
            continue;
        }
        items.push((block.bbox.top, Item::Block(block)));
    }

    if options.image_placeholders {
        for picture in &picture_regions {
            items.push((picture.bbox.top, Item::Picture));
        }
    }

    items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    items
        .into_iter()
        .filter_map(|(_, item)| match item {
            Item::Block(block) => render_block(block, &page_regions),
            Item::Table(table) => {
                graphics::table_grid_cells(table.bbox, &page.graphics, &page.blocks)
                    .or_else(|| borderless_table_row(table.bbox, &page.blocks))
                    .map(|grid| render_gfm_table(&grid))
            }
            Item::Picture => Some("![]()".to_string()),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Builds a table grid for a borderless-table region with no ruling lines to reconstruct
/// a grid from. Two distinct shapes reach here (see `docs/pitfall_registry.json`'s
/// `borderless_table` entry), told apart by how many blocks the region covers:
///
/// - Several blocks side by side ([`crate::layout::detect_borderless_table_regions`]):
///   there's no per-line column signal to split on, so each of the region's own member
///   blocks is re-clustered back into rows by vertical band (the same clustering
///   [`crate::layout::detect_borderless_table_regions`] used to detect the table in the
///   first place -- [`crate::layout::cluster_blocks_by_vertical_band`]), each row's
///   blocks ordered left to right, rows ordered top to bottom.
/// - One block whose own lines carry the columns as aligned whitespace runs
///   ([`crate::layout::whitespace_column_corridors`]): each line becomes its own row,
///   split at the block's shared corridors, with the first line's row doubling as the
///   GFM header row (the same "first row is the header" convention
///   [`render_gfm_table`] applies everywhere else).
///
/// Returns `None` if no block's center falls inside `table_bbox` (should not happen for
/// a region either detector above itself produced, since both always enclose the blocks
/// they were built from).
///
/// Only called when [`graphics::table_grid_cells`] already returned `None`, so a
/// ruling-line table always prefers its own reconstructed grid over this fallback.
fn borderless_table_row(table_bbox: crate::BBox, blocks: &[Block]) -> Option<Vec<Vec<String>>> {
    let cells: Vec<&Block> = blocks
        .iter()
        .filter(|b| table_bbox.contains_center(&b.bbox))
        .collect();
    if cells.is_empty() {
        return None;
    }

    if let [block] = cells.as_slice() {
        let corridors = layout::whitespace_column_corridors(block);
        if !corridors.is_empty() {
            return Some(whitespace_table_rows(block, &corridors));
        }
    }

    let mut rows = layout::cluster_blocks_by_vertical_band(&cells);
    rows.sort_by(|a, b| {
        let top_a = a.iter().map(|blk| blk.bbox.top).fold(f32::MIN, f32::max);
        let top_b = b.iter().map(|blk| blk.bbox.top).fold(f32::MIN, f32::max);
        top_b
            .partial_cmp(&top_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Some(
        rows.into_iter()
            .map(|mut row| {
                row.sort_by(|a, b| a.bbox.left.partial_cmp(&b.bbox.left).unwrap());
                row.into_iter()
                    .map(|b| b.text().replace('\n', "<br>"))
                    .collect()
            })
            .collect(),
    )
}

/// Splits each of `block`'s lines into cells at `corridors` (as detected by
/// [`crate::layout::whitespace_column_corridors`]), producing one row per line. A
/// token (see [`crate::layout::whitespace_line_tokens`]) is assigned to the column
/// after the last corridor that ends at or before its horizontal center -- so text
/// falling between corridor *k* and corridor *k+1* lands in column *k+1*, giving
/// `corridors.len() + 1` columns overall. Multiple tokens landing in the same column
/// (e.g. two words on one side of every corridor) are joined with a single space.
fn whitespace_table_rows(block: &Block, corridors: &[(f32, f32)]) -> Vec<Vec<String>> {
    let mut corridors = corridors.to_vec();
    corridors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    block
        .lines
        .iter()
        .map(|line| {
            let mut row = vec![String::new(); corridors.len() + 1];
            for (start, end, text) in layout::whitespace_line_tokens(line) {
                let center = (start + end) / 2.0;
                let column = corridors.iter().filter(|c| c.1 <= center).count();
                if !row[column].is_empty() {
                    row[column].push(' ');
                }
                row[column].push_str(&text);
            }
            row
        })
        .collect()
}

/// Renders a `rows`-by-columns grid (as produced by [`graphics::table_grid_cells`]) as a
/// GFM pipe table. The first row is treated as the header row -- table ruling-line
/// detection has no way to distinguish a header row from a body row by geometry alone,
/// so this mirrors the common case (a table's first row is its header) rather than
/// guessing per-table.
fn render_gfm_table(rows: &[Vec<String>]) -> String {
    let Some(header) = rows.first() else {
        return String::new();
    };
    let escape = |cell: &str| cell.replace('|', "\\|").replace('\n', "<br>");
    let render_row = |row: &[String]| {
        format!(
            "| {} |",
            row.iter()
                .map(|c| escape(c))
                .collect::<Vec<_>>()
                .join(" | ")
        )
    };

    let mut lines = vec![render_row(header)];
    lines.push(format!(
        "| {} |",
        std::iter::repeat_n("---", header.len())
            .collect::<Vec<_>>()
            .join(" | ")
    ));
    lines.extend(rows.iter().skip(1).map(|row| render_row(row)));
    lines.join("\n")
}

/// Renders a single block per its matched region's class (matched by IoU, see
/// [`to_markdown_structured`]), or `None` if the region is a running header/footer that
/// should be dropped from the Markdown output. [`RegionClass::Table`]/
/// [`RegionClass::Picture`] blocks are handled by [`render_page`] before this is called
/// and never reach the fallback arm here as anything but a plain paragraph if somehow
/// unmatched.
fn render_block(block: &Block, regions: &[&Region]) -> Option<String> {
    let matchable = regions
        .iter()
        .filter(|r| r.class != RegionClass::Table && r.class != RegionClass::Picture);

    // An exact bbox match (the common case: classification and serialization see the
    // same block) is preferred over the fuzzy IoU fallback below, so a block never gets
    // silently downgraded to a plain paragraph by float drift in an *unrelated*
    // region's bbox happening to also clear MATCH_IOU_THRESHOLD.
    let class = matchable
        .clone()
        .find(|r| r.bbox == block.bbox)
        .map(|r| r.class)
        .or_else(|| {
            matchable
                .map(|r| (r, metrics::iou(r.bbox, block.bbox)))
                .filter(|(_, iou)| *iou >= MATCH_IOU_THRESHOLD)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(r, _)| r.class)
        });
    let text = block.text();

    match class {
        Some(RegionClass::PageHeader) | Some(RegionClass::PageFooter) => None,
        Some(RegionClass::Title) => Some(format!("# {text}")),
        Some(RegionClass::SectionHeader) => Some(format!("## {text}")),
        Some(RegionClass::ListItem) => Some(
            text.lines()
                .map(|line| format!("- {}", strip_list_marker(line)))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        Some(RegionClass::Caption) => Some(format!("*{text}*")),
        Some(RegionClass::Footnote) => {
            let marker = footnote_marker(&text).unwrap_or(text.as_str());
            Some(format!("[^{marker}]: {text}"))
        }
        _ => Some(text),
    }
}

/// Extracts a footnote block's leading marker (the digit/symbol
/// [`crate::layout::classify_block`]'s footnote rule already required it to open with)
/// for use as the footnote's Markdown reference id, e.g. `"1"` from `"1 See appendix
/// A."`. Falls back to `None` (the caller then uses the full text as the id) if the text
/// doesn't start with a short marker token.
fn footnote_marker(text: &str) -> Option<&str> {
    let marker = text.split_whitespace().next()?;
    (marker.len() <= 3).then_some(marker)
}

/// Strips a leading bullet/ordered-list marker (already detected by
/// [`crate::layout::classify_regions`]) so `render_block` doesn't double it up with the
/// Markdown `- ` it adds.
fn strip_list_marker(line: &str) -> &str {
    let trimmed = line.trim_start();
    for marker in ["- ", "* ", "\u{2022} "] {
        if let Some(rest) = trimmed.strip_prefix(marker) {
            return rest;
        }
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BBox, Line, Word};

    fn bbox(top: f32) -> BBox {
        BBox {
            left: 0.0,
            bottom: top - 10.0,
            right: 100.0,
            top,
        }
    }

    fn block(text: &str, bbox: BBox) -> crate::Block {
        let word = Word {
            text: text.into(),
            bbox,
            chars: vec![],
        };
        let line = Line {
            text: text.into(),
            bbox,
            words: vec![word],
        };
        crate::Block {
            bbox,
            lines: vec![line],
        }
    }

    #[test]
    fn list_item_renders_as_markdown_bullet() {
        let bbox = bbox(100.0);
        let doc = Document {
            pages: vec![crate::Page {
                index: 0,
                width: 612.0,
                height: 792.0,
                blocks: vec![block("- first point", bbox)],
                ..Default::default()
            }],
        };
        let regions = vec![Region {
            class: RegionClass::ListItem,
            bbox,
            confidence: 1.0,
            page: 0,
        }];

        assert_eq!(to_markdown_structured(&doc, &regions), "- first point");
    }

    #[test]
    fn page_header_is_omitted_from_structured_output() {
        let header_bbox = bbox(780.0);
        let body_bbox = bbox(100.0);
        let doc = Document {
            pages: vec![crate::Page {
                index: 0,
                width: 612.0,
                height: 792.0,
                blocks: vec![
                    block("Running Header", header_bbox),
                    block("Body text", body_bbox),
                ],
                ..Default::default()
            }],
        };
        let regions = vec![
            Region {
                class: RegionClass::PageHeader,
                bbox: header_bbox,
                confidence: 1.0,
                page: 0,
            },
            Region {
                class: RegionClass::Text,
                bbox: body_bbox,
                confidence: 1.0,
                page: 0,
            },
        ];

        assert_eq!(to_markdown_structured(&doc, &regions), "Body text");
    }

    #[test]
    fn whitespace_aligned_block_renders_as_gfm_table() {
        // One block, no ruling lines -- the same shape
        // `layout::whitespace_column_corridors` detects, and the fallback
        // `borderless_table_row`/`whitespace_table_rows` must turn into a proper
        // multi-row GFM table, not a single row with the whole block's text crammed in.
        fn multiline(lines: &[(&str, BBox)]) -> crate::Block {
            let block_lines: Vec<Line> = lines
                .iter()
                .map(|(text, bbox)| {
                    let word = Word {
                        text: (*text).into(),
                        bbox: *bbox,
                        chars: vec![],
                    };
                    Line {
                        text: (*text).into(),
                        bbox: *bbox,
                        words: vec![word],
                    }
                })
                .collect();
            let bbox = block_lines
                .iter()
                .map(|l| l.bbox)
                .reduce(|a, b| a.union(&b))
                .unwrap();
            crate::Block {
                bbox,
                lines: block_lines,
            }
        }

        let lines = [
            (
                "Name        Score",
                BBox {
                    left: 40.0,
                    bottom: 500.0,
                    right: 400.0,
                    top: 515.0,
                },
            ),
            (
                "Alice          92",
                BBox {
                    left: 40.0,
                    bottom: 485.0,
                    right: 400.0,
                    top: 500.0,
                },
            ),
            (
                "Bob            77",
                BBox {
                    left: 40.0,
                    bottom: 470.0,
                    right: 400.0,
                    top: 485.0,
                },
            ),
        ];
        let block = multiline(&lines);
        let doc = Document {
            pages: vec![crate::Page {
                index: 0,
                width: 612.0,
                height: 792.0,
                blocks: vec![block.clone()],
                ..Default::default()
            }],
        };
        let regions = vec![Region {
            class: RegionClass::Table,
            bbox: block.bbox,
            confidence: 0.5,
            page: 0,
        }];

        assert_eq!(
            to_markdown_structured(&doc, &regions),
            "| Name | Score |\n| --- | --- |\n| Alice | 92 |\n| Bob | 77 |"
        );
    }

    fn h_line(y: f32, left: f32, right: f32) -> crate::Graphic {
        crate::Graphic {
            kind: crate::GraphicKind::Stroke,
            bbox: BBox {
                left,
                bottom: y,
                right,
                top: y,
            },
            z: 0,
            stroke_width: 1.0,
            is_stroked: true,
            is_filled: false,
        }
    }

    fn v_line(x: f32, bottom: f32, top: f32) -> crate::Graphic {
        crate::Graphic {
            kind: crate::GraphicKind::Stroke,
            bbox: BBox {
                left: x,
                bottom,
                right: x,
                top,
            },
            z: 0,
            stroke_width: 1.0,
            is_stroked: true,
            is_filled: false,
        }
    }

    #[test]
    fn table_region_renders_as_one_gfm_table_not_one_paragraph_per_cell() {
        let table_bbox = BBox {
            left: 40.0,
            bottom: 60.0,
            right: 200.0,
            top: 100.0,
        };
        let graphics = vec![
            h_line(100.0, 40.0, 200.0),
            h_line(80.0, 40.0, 200.0),
            h_line(60.0, 40.0, 200.0),
            v_line(40.0, 60.0, 100.0),
            v_line(120.0, 60.0, 100.0),
            v_line(200.0, 60.0, 100.0),
        ];
        let cells = [
            (
                "Name",
                BBox {
                    left: 45.0,
                    bottom: 85.0,
                    right: 115.0,
                    top: 95.0,
                },
            ),
            (
                "Age",
                BBox {
                    left: 125.0,
                    bottom: 85.0,
                    right: 195.0,
                    top: 95.0,
                },
            ),
            (
                "Ada",
                BBox {
                    left: 45.0,
                    bottom: 65.0,
                    right: 115.0,
                    top: 75.0,
                },
            ),
            (
                "36",
                BBox {
                    left: 125.0,
                    bottom: 65.0,
                    right: 195.0,
                    top: 75.0,
                },
            ),
        ];
        let doc = Document {
            pages: vec![crate::Page {
                index: 0,
                width: 612.0,
                height: 792.0,
                blocks: cells.iter().map(|(t, b)| block(t, *b)).collect(),
                graphics,
            }],
        };
        let regions = vec![Region {
            class: RegionClass::Table,
            bbox: table_bbox,
            confidence: 0.75,
            page: 0,
        }];

        let out = to_markdown_structured(&doc, &regions);
        assert_eq!(out, "| Name | Age |\n| --- | --- |\n| Ada | 36 |");
    }

    #[test]
    fn borderless_multi_row_table_renders_as_one_gfm_table_with_body_rows() {
        // Three row bands of side-by-side blocks, no ruling lines at all -- must render
        // as ONE GFM table with a header row plus two body rows, not three separate
        // header-only 1-row tables (the defect `layout::detect_borderless_table_regions`
        // merging compatible adjacent rows into one `Region` fixes).
        let row = |y: f32, left_text: &str, right_text: &str| {
            vec![
                block(
                    left_text,
                    BBox {
                        left: 72.0,
                        bottom: y,
                        right: 165.0,
                        top: y + 15.0,
                    },
                ),
                block(
                    right_text,
                    BBox {
                        left: 340.0,
                        bottom: y,
                        right: 351.0,
                        top: y + 15.0,
                    },
                ),
            ]
        };
        let mut blocks = Vec::new();
        blocks.extend(row(690.0, "Item", "Count"));
        blocks.extend(row(660.0, "Widget", "17"));
        blocks.extend(row(630.0, "Gadget", "9"));
        let table_bbox = blocks
            .iter()
            .map(|b| b.bbox)
            .reduce(|a, b| a.union(&b))
            .unwrap();

        let doc = Document {
            pages: vec![crate::Page {
                index: 0,
                width: 612.0,
                height: 792.0,
                blocks,
                ..Default::default()
            }],
        };
        let regions = vec![Region {
            class: RegionClass::Table,
            bbox: table_bbox,
            confidence: 0.5,
            page: 0,
        }];

        let out = to_markdown_structured(&doc, &regions);
        assert_eq!(
            out,
            "| Item | Count |\n| --- | --- |\n| Widget | 17 |\n| Gadget | 9 |"
        );
    }

    #[test]
    fn picture_region_renders_as_image_placeholder() {
        let picture_bbox = BBox {
            left: 40.0,
            bottom: 40.0,
            right: 200.0,
            top: 200.0,
        };
        let doc = Document {
            pages: vec![crate::Page {
                index: 0,
                width: 612.0,
                height: 792.0,
                ..Default::default()
            }],
        };
        let regions = vec![Region {
            class: RegionClass::Picture,
            bbox: picture_bbox,
            confidence: 0.7,
            page: 0,
        }];

        assert_eq!(to_markdown_structured(&doc, &regions), "![]()");
    }

    #[test]
    fn footnote_renders_with_marker_reference() {
        let footnote_bbox = bbox(50.0);
        let doc = Document {
            pages: vec![crate::Page {
                index: 0,
                width: 612.0,
                height: 792.0,
                blocks: vec![block("1 See appendix A.", footnote_bbox)],
                ..Default::default()
            }],
        };
        let regions = vec![Region {
            class: RegionClass::Footnote,
            bbox: footnote_bbox,
            confidence: 0.55,
            page: 0,
        }];

        assert_eq!(
            to_markdown_structured(&doc, &regions),
            "[^1]: 1 See appendix A."
        );
    }

    #[test]
    fn block_matches_region_despite_minor_bbox_drift() {
        // A bbox that's off by a fraction of a point (e.g. from upstream float
        // recomputation) should still match its region via IoU tolerance, not fall
        // through to a plain paragraph.
        let region_bbox = bbox(100.0);
        let drifted_bbox = BBox {
            left: region_bbox.left,
            bottom: region_bbox.bottom + 0.001,
            right: region_bbox.right,
            top: region_bbox.top,
        };
        let doc = Document {
            pages: vec![crate::Page {
                index: 0,
                width: 612.0,
                height: 792.0,
                blocks: vec![block("Heading", drifted_bbox)],
                ..Default::default()
            }],
        };
        let regions = vec![Region {
            class: RegionClass::Title,
            bbox: region_bbox,
            confidence: 1.0,
            page: 0,
        }];

        assert_eq!(to_markdown_structured(&doc, &regions), "# Heading");
    }
}

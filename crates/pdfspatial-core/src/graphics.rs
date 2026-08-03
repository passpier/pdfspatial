//! Stage 1b/4: table and picture region detection from non-text page objects.
//!
//! [`crate::layout::classify_regions`] can only ever emit [`RegionClass::Table`],
//! [`RegionClass::Picture`], or [`RegionClass::Formula`] if *something* first tells it
//! where those regions are -- the text-layer-only heuristics in [`crate::layout`] have no
//! signal for a table border or an embedded photo, because a table border and a photo
//! aren't text. This module is that signal: it works over [`Graphic`], the non-text page
//! objects (`extract::extract_graphics`) pulls straight out of PDFium's page-object API
//! (vector paths, images, shadings) -- the same signal
//! [`firecrawl/pdf-inspector`](https://github.com/firecrawl/pdf-inspector) uses via
//! `lopdf`'s raw content-stream access, deterministically and without any vision model.
//!
//! Two independent passes:
//!
//! - [`detect_table_regions`] unions stroked ruling lines into grid components (the same
//!   union-find approach pdf-inspector's README describes) and keeps components with at
//!   least [`MIN_GRID_LINES`] distinct lines on each axis -- a real table grid, not a
//!   stray underline or box rule.
//! - [`detect_picture_regions`] treats every `Image`/`XObjectForm` graphic as a picture
//!   outright, plus any sufficiently large `Fill` graphic that contains no text (a solid
//!   photo backdrop or vector illustration with no ruling lines to key a table off).
//!
//! Borderless (whitespace-delimited) tables have no ruling lines by definition, so they
//! are out of scope for this module -- see `borderless_table` in
//! `docs/pitfall_registry.json` for that heuristic's own text-alignment approach.
//!
//! [`table_grid_cells`] is the second half of table detection: given a region this module
//! already flagged as a table, it reconstructs the row/column grid and assigns each
//! overlapping [`Block`] to a cell, for [`crate::serialize`] to render as a GFM pipe
//! table. A block spanning more than one grid cell (rowspan/colspan) is assigned to every
//! cell its center-line crosses, which is what lets `merged_table_cell` /
//! `multi_line_table_cell` (see `docs/pitfall_registry.json`) resolve correctly instead of
//! silently dropping into just one cell.

use crate::layout::{Region, RegionClass};
use crate::{BBox, Block, Graphic, GraphicKind, Page};

/// A stroked graphic's degenerate axis (its thickness) must be at or below this, in
/// points, to be treated as a ruling line rather than a filled bar or box.
const LINE_THICKNESS_MAX_PT: f32 = 2.0;

/// A ruling line's long axis must be at least this long, in points, to count -- filters
/// out stray dots, bullet glyphs rendered as tiny filled paths, and other non-table
/// marks that happen to have a thin bounding box.
const MIN_LINE_LENGTH_PT: f32 = 10.0;

/// Two ruling lines (or a line and a growing grid component) within this many points of
/// each other are treated as touching/overlapping for union-find clustering, and two
/// lines within this tolerance on their shared axis are treated as the same grid line
/// (deduplicated) rather than two distinct rows/columns.
const GRID_TOLERANCE_PT: f32 = 3.0;

/// A grid component needs at least this many *distinct* horizontal lines, and the same
/// number of distinct vertical lines, to qualify as a bordered table rather than a
/// decorative rule, a single box, or an underline.
const MIN_GRID_LINES: usize = 2;

/// A filled graphic must cover at least this area, in square points, and contain no text
/// blocks, to be treated as a picture on its own (as opposed to a small decorative
/// highlight or table-cell shading).
const MIN_PICTURE_FILL_AREA_PT2: f32 = 900.0; // 30pt x 30pt

/// Detects table regions from ruling lines: [`Graphic`]s that are stroked, thin on one
/// axis, and long on the other. Lines are unioned into connected grid components (any two
/// lines whose bounding boxes lie within [`GRID_TOLERANCE_PT`] of touching are merged);
/// a component qualifies as a table when it has at least [`MIN_GRID_LINES`] distinct
/// horizontal lines *and* [`MIN_GRID_LINES`] distinct vertical lines. The region's bbox is
/// the union of every line in the component.
///
/// Returns regions in no particular order. Confidence is fixed at `0.75` -- this is a
/// deterministic geometric rule, not a probabilistic model, so the confidence is a flat
/// signal that "this passed the grid-line test" rather than a per-instance score.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::graphics::detect_table_regions;
/// use pdfspatial_core::{BBox, Graphic, GraphicKind, Page};
///
/// fn h_line(y: f32, left: f32, right: f32) -> Graphic {
///     Graphic {
///         kind: GraphicKind::Stroke,
///         bbox: BBox { left, bottom: y, right, top: y },
///         z: 0,
///         stroke_width: 1.0,
///         is_stroked: true,
///         is_filled: false,
///     }
/// }
/// fn v_line(x: f32, bottom: f32, top: f32) -> Graphic {
///     Graphic {
///         kind: GraphicKind::Stroke,
///         bbox: BBox { left: x, bottom, right: x, top },
///         z: 0,
///         stroke_width: 1.0,
///         is_stroked: true,
///         is_filled: false,
///     }
/// }
///
/// // A simple 2x2 grid: two horizontal rules, two vertical rules.
/// let graphics = vec![
///     h_line(100.0, 40.0, 200.0),
///     h_line(80.0, 40.0, 200.0),
///     h_line(60.0, 40.0, 200.0),
///     v_line(40.0, 60.0, 100.0),
///     v_line(120.0, 60.0, 100.0),
///     v_line(200.0, 60.0, 100.0),
/// ];
/// let page = Page { index: 0, width: 600.0, height: 800.0, ..Default::default() };
///
/// let tables = detect_table_regions(&graphics, &page);
/// assert_eq!(tables.len(), 1);
/// assert_eq!(tables[0].bbox, BBox { left: 40.0, bottom: 60.0, right: 200.0, top: 100.0 });
/// ```
pub fn detect_table_regions(graphics: &[Graphic], _page: &Page) -> Vec<Region> {
    let lines: Vec<&Graphic> = graphics.iter().filter(|g| is_ruling_line(g)).collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let components = cluster_lines(&lines);

    components
        .into_iter()
        .filter_map(|indices| {
            let component_lines: Vec<&Graphic> = indices.iter().map(|&i| lines[i]).collect();
            let (horizontals, verticals) = distinct_grid_lines(&component_lines);
            if horizontals.len() < MIN_GRID_LINES || verticals.len() < MIN_GRID_LINES {
                return None;
            }

            let bbox = union_all(component_lines.iter().map(|g| g.bbox))?;
            Some(Region {
                class: RegionClass::Table,
                bbox,
                confidence: 0.75,
            })
        })
        .collect()
}

/// Detects picture regions: every `Image`/`XObjectForm` [`Graphic`] outright, plus any
/// `Fill` graphic covering at least [`MIN_PICTURE_FILL_AREA_PT2`] whose bbox contains no
/// `page`'s text blocks (a vector illustration or solid photo backdrop with no embedded
/// raster to key off).
///
/// # Examples
///
/// ```
/// use pdfspatial_core::graphics::detect_picture_regions;
/// use pdfspatial_core::{BBox, Graphic, GraphicKind, Page};
///
/// let image = Graphic {
///     kind: GraphicKind::Image,
///     bbox: BBox { left: 50.0, bottom: 50.0, right: 250.0, top: 250.0 },
///     z: 0,
///     stroke_width: 0.0,
///     is_stroked: false,
///     is_filled: true,
/// };
/// let page = Page { index: 0, width: 600.0, height: 800.0, ..Default::default() };
///
/// let pictures = detect_picture_regions(&[image], &page);
/// assert_eq!(pictures.len(), 1);
/// ```
pub fn detect_picture_regions(graphics: &[Graphic], page: &Page) -> Vec<Region> {
    graphics
        .iter()
        .filter_map(|g| match g.kind {
            GraphicKind::Image => Some(Region {
                class: RegionClass::Picture,
                bbox: g.bbox,
                confidence: 0.7,
            }),
            GraphicKind::Fill
                if g.bbox.width() * g.bbox.height() >= MIN_PICTURE_FILL_AREA_PT2
                    && !page.blocks.iter().any(|b| block_center_in(b, g.bbox)) =>
            {
                Some(Region {
                    class: RegionClass::Picture,
                    bbox: g.bbox,
                    confidence: 0.5,
                })
            }
            _ => None,
        })
        .collect()
}

/// Reconstructs a table's row/column grid within `table_bbox` from its ruling lines in
/// `graphics`, and assigns each of `blocks` whose center falls inside `table_bbox` to
/// every cell its bbox overlaps -- so a block spanning multiple grid cells (a merged or
/// multi-line cell) appears in each one, rather than being forced into a single cell.
///
/// Returns `None` if fewer than [`MIN_GRID_LINES`] distinct lines are found on either
/// axis within `table_bbox` (the caller should not have called this on a bbox
/// [`detect_table_regions`] didn't itself flag as a table).
///
/// Cells are returned top-to-bottom, left-to-right, each joined from its assigned
/// blocks' text with `"<br>"` (GFM's convention for a multi-line table cell, since a
/// literal newline is not valid inside a pipe-table cell).
///
/// # Examples
///
/// ```
/// use pdfspatial_core::graphics::table_grid_cells;
/// use pdfspatial_core::{BBox, Block, Graphic, GraphicKind, Line, Word};
///
/// fn h_line(y: f32) -> Graphic {
///     Graphic { kind: GraphicKind::Stroke, bbox: BBox { left: 40.0, bottom: y, right: 200.0, top: y }, z: 0, stroke_width: 1.0, is_stroked: true, is_filled: false }
/// }
/// fn v_line(x: f32) -> Graphic {
///     Graphic { kind: GraphicKind::Stroke, bbox: BBox { left: x, bottom: 60.0, right: x, top: 100.0 }, z: 0, stroke_width: 1.0, is_stroked: true, is_filled: false }
/// }
/// fn cell(text: &str, bbox: BBox) -> Block {
///     let word = Word { text: text.into(), bbox, chars: vec![] };
///     let line = Line { text: text.into(), bbox, words: vec![word] };
///     Block { bbox, lines: vec![line] }
/// }
///
/// // Two horizontal rules (one row band) and three vertical rules (two column bands).
/// let graphics = vec![h_line(100.0), h_line(60.0), v_line(40.0), v_line(120.0), v_line(200.0)];
/// let blocks = vec![
///     cell("A", BBox { left: 45.0, bottom: 65.0, right: 115.0, top: 95.0 }),
///     cell("B", BBox { left: 125.0, bottom: 65.0, right: 195.0, top: 95.0 }),
/// ];
/// let table_bbox = BBox { left: 40.0, bottom: 60.0, right: 200.0, top: 100.0 };
///
/// let grid = table_grid_cells(table_bbox, &graphics, &blocks).unwrap();
/// assert_eq!(grid.len(), 1);
/// assert_eq!(grid[0], vec!["A".to_string(), "B".to_string()]);
/// ```
pub fn table_grid_cells(
    table_bbox: BBox,
    graphics: &[Graphic],
    blocks: &[Block],
) -> Option<Vec<Vec<String>>> {
    let lines: Vec<&Graphic> = graphics
        .iter()
        .filter(|g| is_ruling_line(g) && bboxes_touch(g.bbox, table_bbox, GRID_TOLERANCE_PT))
        .collect();

    let (mut row_cuts, mut col_cuts) = distinct_grid_lines(&lines);
    if row_cuts.len() < MIN_GRID_LINES || col_cuts.len() < MIN_GRID_LINES {
        return None;
    }
    row_cuts.sort_by(|a, b| b.partial_cmp(a).unwrap()); // top to bottom (descending y)
    col_cuts.sort_by(|a, b| a.partial_cmp(b).unwrap()); // left to right

    let row_bands: Vec<(f32, f32)> = row_cuts.windows(2).map(|w| (w[1], w[0])).collect();
    let col_bands: Vec<(f32, f32)> = col_cuts.windows(2).map(|w| (w[0], w[1])).collect();

    let mut grid: Vec<Vec<Vec<String>>> = vec![vec![Vec::new(); col_bands.len()]; row_bands.len()];

    for block in blocks {
        if !block_center_in(block, table_bbox) {
            continue;
        }
        for (row_idx, &(bottom, top)) in row_bands.iter().enumerate() {
            if block.bbox.top <= bottom || block.bbox.bottom >= top {
                continue; // no vertical overlap with this row band
            }
            for (col_idx, &(left, right)) in col_bands.iter().enumerate() {
                if block.bbox.right <= left || block.bbox.left >= right {
                    continue; // no horizontal overlap with this column band
                }
                grid[row_idx][col_idx].push(block.text());
            }
        }
    }

    Some(
        grid.into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|cell_blocks| cell_blocks.join("<br>"))
                    .collect()
            })
            .collect(),
    )
}

/// `true` if `g` is a stroked graphic thin enough on one axis and long enough on the
/// other to be a ruling line rather than a filled bar, box, or decorative mark.
fn is_ruling_line(g: &Graphic) -> bool {
    if g.kind != GraphicKind::Stroke || !g.is_stroked {
        return false;
    }
    let w = g.bbox.width();
    let h = g.bbox.height();
    (w <= LINE_THICKNESS_MAX_PT && h >= MIN_LINE_LENGTH_PT)
        || (h <= LINE_THICKNESS_MAX_PT && w >= MIN_LINE_LENGTH_PT)
}

/// `true` if `line` is closer to horizontal (wide, thin) than vertical.
fn is_horizontal(line: &Graphic) -> bool {
    line.bbox.width() >= line.bbox.height()
}

/// Groups `lines` into connected components via union-find: two lines are connected if
/// their bounding boxes lie within [`GRID_TOLERANCE_PT`] of touching or overlapping.
/// Returns each component as a list of indices into `lines`.
fn cluster_lines(lines: &[&Graphic]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..lines.len()).collect();

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            if bboxes_touch(lines[i].bbox, lines[j].bbox, GRID_TOLERANCE_PT) {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    let mut components: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for i in 0..lines.len() {
        let root = find(&mut parent, i);
        components.entry(root).or_default().push(i);
    }
    components.into_values().collect()
}

/// Splits `lines` into distinct horizontal and vertical grid-line positions (deduplicated
/// within [`GRID_TOLERANCE_PT`] on their shared axis), returning `(horizontal_ys,
/// vertical_xs)`.
fn distinct_grid_lines(lines: &[&Graphic]) -> (Vec<f32>, Vec<f32>) {
    let mut ys: Vec<f32> = Vec::new();
    let mut xs: Vec<f32> = Vec::new();

    for &line in lines {
        if is_horizontal(line) {
            let y = (line.bbox.bottom + line.bbox.top) / 2.0;
            if !ys
                .iter()
                .any(|&existing| (existing - y).abs() <= GRID_TOLERANCE_PT)
            {
                ys.push(y);
            }
        } else {
            let x = (line.bbox.left + line.bbox.right) / 2.0;
            if !xs
                .iter()
                .any(|&existing| (existing - x).abs() <= GRID_TOLERANCE_PT)
            {
                xs.push(x);
            }
        }
    }

    (ys, xs)
}

/// `true` if `a` and `b`, each expanded by `tol` on every side, overlap or touch.
fn bboxes_touch(a: BBox, b: BBox, tol: f32) -> bool {
    let a = BBox {
        left: a.left - tol,
        bottom: a.bottom - tol,
        right: a.right + tol,
        top: a.top + tol,
    };
    !(a.right < b.left || b.right < a.left || a.top < b.bottom || b.top < a.bottom)
}

/// The smallest [`BBox`] enclosing every box in `boxes`, or `None` if `boxes` is empty.
fn union_all(mut boxes: impl Iterator<Item = BBox>) -> Option<BBox> {
    let first = boxes.next()?;
    Some(boxes.fold(first, |acc, b| acc.union(&b)))
}

/// `true` if `block`'s bbox center falls inside `bbox`. Thin wrapper over
/// [`BBox::contains_center`] kept for call-site readability (`block_center_in(b, bbox)`
/// reads more naturally at each use than `bbox.contains_center(&b.bbox)`).
fn block_center_in(block: &Block, bbox: BBox) -> bool {
    bbox.contains_center(&block.bbox)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Line, Word};

    fn h_line(y: f32, left: f32, right: f32) -> Graphic {
        Graphic {
            kind: GraphicKind::Stroke,
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

    fn v_line(x: f32, bottom: f32, top: f32) -> Graphic {
        Graphic {
            kind: GraphicKind::Stroke,
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

    fn grid_2x2() -> Vec<Graphic> {
        vec![
            h_line(100.0, 40.0, 200.0),
            h_line(80.0, 40.0, 200.0),
            h_line(60.0, 40.0, 200.0),
            v_line(40.0, 60.0, 100.0),
            v_line(120.0, 60.0, 100.0),
            v_line(200.0, 60.0, 100.0),
        ]
    }

    fn page() -> Page {
        Page {
            index: 0,
            width: 600.0,
            height: 800.0,
            ..Default::default()
        }
    }

    fn cell(text: &str, bbox: BBox) -> Block {
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
        Block {
            bbox,
            lines: vec![line],
        }
    }

    #[test]
    fn two_by_two_grid_is_detected_as_one_table() {
        let tables = detect_table_regions(&grid_2x2(), &page());
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].class, RegionClass::Table);
        assert_eq!(
            tables[0].bbox,
            BBox {
                left: 40.0,
                bottom: 60.0,
                right: 200.0,
                top: 100.0
            }
        );
    }

    #[test]
    fn a_single_underline_rule_is_not_a_table() {
        let graphics = vec![h_line(100.0, 40.0, 200.0)];
        assert!(detect_table_regions(&graphics, &page()).is_empty());
    }

    #[test]
    fn a_lone_box_border_is_not_a_table_two_lines_per_axis_is_the_floor() {
        // A single rectangle: 2 horizontal + 2 vertical lines is the minimum grid --
        // this *does* qualify (a bordered single-cell "table"), documenting the floor.
        let graphics = vec![
            h_line(100.0, 40.0, 200.0),
            h_line(60.0, 40.0, 200.0),
            v_line(40.0, 60.0, 100.0),
            v_line(200.0, 60.0, 100.0),
        ];
        assert_eq!(detect_table_regions(&graphics, &page()).len(), 1);
    }

    #[test]
    fn two_far_apart_grids_are_detected_as_two_separate_tables() {
        let mut graphics = grid_2x2();
        // A second, unrelated 2x2 grid far away on the page.
        graphics.extend([
            h_line(700.0, 40.0, 200.0),
            h_line(680.0, 40.0, 200.0),
            h_line(660.0, 40.0, 200.0),
            v_line(40.0, 660.0, 700.0),
            v_line(120.0, 660.0, 700.0),
            v_line(200.0, 660.0, 700.0),
        ]);
        assert_eq!(detect_table_regions(&graphics, &page()).len(), 2);
    }

    #[test]
    fn image_graphic_is_detected_as_a_picture() {
        let image = Graphic {
            kind: GraphicKind::Image,
            bbox: BBox {
                left: 50.0,
                bottom: 50.0,
                right: 250.0,
                top: 250.0,
            },
            z: 0,
            stroke_width: 0.0,
            is_stroked: false,
            is_filled: true,
        };
        let pictures = detect_picture_regions(&[image], &page());
        assert_eq!(pictures.len(), 1);
        assert_eq!(pictures[0].class, RegionClass::Picture);
    }

    #[test]
    fn small_fill_is_not_a_picture() {
        let fill = Graphic {
            kind: GraphicKind::Fill,
            bbox: BBox {
                left: 0.0,
                bottom: 0.0,
                right: 5.0,
                top: 5.0,
            },
            z: 0,
            stroke_width: 0.0,
            is_stroked: false,
            is_filled: true,
        };
        assert!(detect_picture_regions(&[fill], &page()).is_empty());
    }

    #[test]
    fn large_fill_containing_text_is_not_a_picture() {
        let fill = Graphic {
            kind: GraphicKind::Fill,
            bbox: BBox {
                left: 0.0,
                bottom: 0.0,
                right: 100.0,
                top: 100.0,
            },
            z: 0,
            stroke_width: 0.0,
            is_stroked: false,
            is_filled: true,
        };
        let mut p = page();
        p.blocks.push(cell(
            "Body text",
            BBox {
                left: 10.0,
                bottom: 10.0,
                right: 90.0,
                top: 90.0,
            },
        ));
        assert!(detect_picture_regions(&[fill], &p).is_empty());
    }

    #[test]
    fn large_fill_with_no_text_is_a_picture() {
        let fill = Graphic {
            kind: GraphicKind::Fill,
            bbox: BBox {
                left: 0.0,
                bottom: 0.0,
                right: 100.0,
                top: 100.0,
            },
            z: 0,
            stroke_width: 0.0,
            is_stroked: false,
            is_filled: true,
        };
        let pictures = detect_picture_regions(&[fill], &page());
        assert_eq!(pictures.len(), 1);
    }

    #[test]
    fn table_grid_cells_assigns_each_block_to_its_cell() {
        let table_bbox = BBox {
            left: 40.0,
            bottom: 60.0,
            right: 200.0,
            top: 100.0,
        };
        let blocks = vec![
            cell(
                "A1",
                BBox {
                    left: 45.0,
                    bottom: 85.0,
                    right: 115.0,
                    top: 95.0,
                },
            ),
            cell(
                "B1",
                BBox {
                    left: 125.0,
                    bottom: 85.0,
                    right: 195.0,
                    top: 95.0,
                },
            ),
            cell(
                "A2",
                BBox {
                    left: 45.0,
                    bottom: 65.0,
                    right: 115.0,
                    top: 75.0,
                },
            ),
            cell(
                "B2",
                BBox {
                    left: 125.0,
                    bottom: 65.0,
                    right: 195.0,
                    top: 75.0,
                },
            ),
        ];

        let grid = table_grid_cells(table_bbox, &grid_2x2(), &blocks).unwrap();
        assert_eq!(
            grid,
            vec![
                vec!["A1".to_string(), "B1".to_string()],
                vec!["A2".to_string(), "B2".to_string()],
            ]
        );
    }

    #[test]
    fn a_block_spanning_two_columns_appears_in_both_cells() {
        let table_bbox = BBox {
            left: 40.0,
            bottom: 60.0,
            right: 200.0,
            top: 100.0,
        };
        // One block spans the full width of both columns in the top row (a merged cell).
        let blocks = vec![cell(
            "Merged",
            BBox {
                left: 45.0,
                bottom: 85.0,
                right: 195.0,
                top: 95.0,
            },
        )];

        let grid = table_grid_cells(table_bbox, &grid_2x2(), &blocks).unwrap();
        assert_eq!(grid[0], vec!["Merged".to_string(), "Merged".to_string()]);
    }

    #[test]
    fn too_few_grid_lines_returns_none() {
        let table_bbox = BBox {
            left: 40.0,
            bottom: 60.0,
            right: 200.0,
            top: 100.0,
        };
        let single_line = vec![h_line(100.0, 40.0, 200.0)];
        assert!(table_grid_cells(table_bbox, &single_line, &[]).is_none());
    }
}

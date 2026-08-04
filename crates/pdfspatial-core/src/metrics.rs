//! Validation metrics for every stage of the pipeline.
//!
//! [`char_recall`] and [`line_grouping_accuracy`] back the Stage 1 unit tests in this
//! crate, matching the roadmap's Stage 1 metric table exactly (character extraction
//! recall ≥ 99%, line-grouping accuracy ≥ 95% on single-column documents).
//! [`reading_order_edit_distance`] and [`throughput_pages_per_sec`] round out that same
//! table's two record-only rows (no pass/fail target; recorded to establish the perf
//! floor and the ordering-error signal that drives Stage 3).
//!
//! [`iou`], [`giou`], and [`region_f1`] (built on [`match_regions`]) implement the
//! roadmap's Stage 2 region-level metrics (mean GIoU, per-class F1 at a COCO-style IoU
//! threshold). The TEDS family ([`teds_struct`], [`teds`], [`teds_iou`]) implements the
//! roadmap's table-structure metrics on a restricted table-HTML dialect, backed by a
//! private Zhang–Shasha tree-edit-distance implementation (see the `teds` submodule).
//! [`crate::eval`] aggregates [`match_regions`]/[`giou`] across a page dataset into a
//! dashboard, and [`crate::eval::doclaynet`] wires that up against real DocLayNet
//! ground truth; the TEDS family remains exercised only by its own unit tests below, as
//! no table-structure predictor exists yet to score against PubTabNet/DocLayNet table
//! ground truth.

use crate::BBox;
use std::collections::HashMap;

/// Character extraction recall: the fraction of ground-truth characters recovered by
/// extraction, as a multiset intersection over non-whitespace characters.
///
/// This matches the roadmap's Stage 1 definition — "% of ground-truth characters
/// recovered vs. dropped (ligatures, embedded fonts, CID-keyed text)" — with a target of
/// ≥ 99% on single-column, non-tabular documents.
///
/// Returns `1.0` if `ground_truth` contains no non-whitespace characters.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::metrics::char_recall;
///
/// let recall = char_recall("Helo world", "Hello world");
/// assert!((0.0..1.0).contains(&recall));
/// assert_eq!(char_recall("Hello world", "Hello world"), 1.0);
/// ```
pub fn char_recall(extracted: &str, ground_truth: &str) -> f32 {
    let gt_counts = char_histogram(ground_truth);
    let total: usize = gt_counts.values().sum();

    if total == 0 {
        return 1.0;
    }

    let extracted_counts = char_histogram(extracted);
    let recovered: usize = gt_counts
        .iter()
        .map(|(ch, gt_count)| (*gt_count).min(*extracted_counts.get(ch).unwrap_or(&0)))
        .sum();

    recovered as f32 / total as f32
}

fn char_histogram(text: &str) -> HashMap<char, usize> {
    let mut counts = HashMap::new();
    for ch in text.chars().filter(|c| !c.is_whitespace()) {
        *counts.entry(ch).or_insert(0) += 1;
    }
    counts
}

/// Line-grouping accuracy: the fraction of ground-truth lines that Stage 1's baseline
/// clustering reproduced exactly (after trimming surrounding whitespace).
///
/// Matches the roadmap's Stage 1 definition — "% of lines correctly segmented by
/// baseline clustering vs. manual annotation" — with a target of ≥ 95% on single-column
/// documents.
///
/// Returns `1.0` if `ground_truth_lines` is empty.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::metrics::line_grouping_accuracy;
///
/// let extracted = vec!["Hello world".to_string(), "second line".to_string()];
/// let ground_truth = vec!["Hello world".to_string(), "second line".to_string()];
/// assert_eq!(line_grouping_accuracy(&extracted, &ground_truth), 1.0);
/// ```
pub fn line_grouping_accuracy(extracted_lines: &[String], ground_truth_lines: &[String]) -> f32 {
    if ground_truth_lines.is_empty() {
        return 1.0;
    }

    let extracted_trimmed: Vec<&str> = extracted_lines.iter().map(|l| l.trim()).collect();
    let matched = ground_truth_lines
        .iter()
        .filter(|gt| extracted_trimmed.contains(&gt.trim()))
        .count();

    matched as f32 / ground_truth_lines.len() as f32
}

/// Reading-order edit distance: the raw Levenshtein distance between the extracted token
/// sequence and the ground-truth token sequence.
///
/// Matches the roadmap's Stage 1 definition verbatim — "Levenshtein distance between
/// extracted token order and ground-truth order" — word-level (not char-level) and
/// unnormalized (a raw edit count, not a ratio), since this metric is **record only**: the
/// roadmap expects it to be poor on multi-column layouts and treats it as the signal that
/// drives Stage 3 error analysis, not a pass/fail target like [`char_recall`] or
/// [`line_grouping_accuracy`].
///
/// # Examples
///
/// ```
/// use pdfspatial_core::metrics::reading_order_edit_distance;
///
/// let extracted = ["the", "quick", "fox"];
/// let ground_truth = ["the", "quick", "brown", "fox"];
/// assert_eq!(reading_order_edit_distance(&extracted, &ground_truth), 1);
/// assert_eq!(reading_order_edit_distance(&extracted, &extracted), 0);
/// ```
pub fn reading_order_edit_distance(extracted: &[&str], ground_truth: &[&str]) -> usize {
    let mut dp = vec![vec![0usize; ground_truth.len() + 1]; extracted.len() + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=extracted.len() {
        for j in 1..=ground_truth.len() {
            dp[i][j] = if extracted[i - 1] == ground_truth[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1])
            };
        }
    }
    dp[extracted.len()][ground_truth.len()]
}

/// Throughput, in pages per second, for a batch of `page_count` pages extracted in
/// `elapsed` wall-clock time.
///
/// Matches the roadmap's Stage 1 definition — "Pages/sec on reference hardware (single
/// core, no OCR)" — and is **record only**: the roadmap treats it as establishing the perf
/// floor for later stages, not a pass/fail target.
///
/// Returns `0.0` if `elapsed` is zero (avoids dividing by zero on a degenerate timing).
///
/// # Examples
///
/// ```
/// use pdfspatial_core::metrics::throughput_pages_per_sec;
/// use std::time::Duration;
///
/// let throughput = throughput_pages_per_sec(10, Duration::from_secs(2));
/// assert_eq!(throughput, 5.0);
/// assert_eq!(throughput_pages_per_sec(10, Duration::ZERO), 0.0);
/// ```
pub fn throughput_pages_per_sec(page_count: usize, elapsed: std::time::Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs == 0.0 {
        return 0.0;
    }
    page_count as f64 / secs
}

/// Area of a [`BBox`], in square points. Never negative (mirrors [`BBox::width`]/
/// [`BBox::height`]).
fn area(b: BBox) -> f32 {
    b.width() * b.height()
}

/// Area of the intersection of two [`BBox`]es, or `0.0` if they don't overlap.
fn intersection_area(a: BBox, b: BBox) -> f32 {
    let left = a.left.max(b.left);
    let right = a.right.min(b.right);
    let bottom = a.bottom.max(b.bottom);
    let top = a.top.min(b.top);
    (right - left).max(0.0) * (top - bottom).max(0.0)
}

/// Standard Intersection-over-Union between two bounding boxes, in `[0.0, 1.0]`.
///
/// Returns `0.0` if both boxes are degenerate (zero area) and disjoint; used by
/// [`region_f1`] for COCO-style matching and by [`teds_iou`] to score bbox-annotated
/// table cells.
///
/// # Examples
///
/// ```
/// use pdfspatial_core::BBox;
/// use pdfspatial_core::metrics::iou;
///
/// let a = BBox { left: 0.0, bottom: 0.0, right: 10.0, top: 10.0 };
/// assert_eq!(iou(a, a), 1.0);
///
/// let disjoint = BBox { left: 20.0, bottom: 20.0, right: 30.0, top: 30.0 };
/// assert_eq!(iou(a, disjoint), 0.0);
/// ```
pub fn iou(a: BBox, b: BBox) -> f32 {
    let intersection = intersection_area(a, b);
    let union = area(a) + area(b) - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Generalized IoU between a predicted and ground-truth bounding box, in `[-1.0, 1.0]`.
///
/// Unlike plain IoU, GIoU stays informative for non-overlapping boxes via a penalty term
/// based on the smallest enclosing box — important for early-epoch layout models that
/// may predict boxes with no overlap at all. See Rezatofighi et al., "Generalized
/// Intersection over Union" (CVPR 2019).
///
/// Returns `0.0` if the enclosing box is degenerate (both inputs zero-area at the same
/// point).
///
/// # Examples
///
/// ```
/// use pdfspatial_core::BBox;
/// use pdfspatial_core::metrics::giou;
///
/// let a = BBox { left: 0.0, bottom: 0.0, right: 10.0, top: 10.0 };
/// assert_eq!(giou(a, a), 1.0);
///
/// let disjoint = BBox { left: 20.0, bottom: 0.0, right: 30.0, top: 10.0 };
/// assert!(giou(a, disjoint) < 0.0);
/// ```
pub fn giou(predicted: BBox, ground_truth: BBox) -> f32 {
    let intersection = intersection_area(predicted, ground_truth);
    let union = area(predicted) + area(ground_truth) - intersection;
    let enclosing = predicted.union(&ground_truth);
    let enclosing_area = area(enclosing);

    if enclosing_area <= 0.0 {
        return 0.0;
    }

    let iou = if union <= 0.0 {
        0.0
    } else {
        intersection / union
    };
    iou - (enclosing_area - union) / enclosing_area
}

/// Confusion counts and matched `(predicted_index, ground_truth_index)` pairs for one
/// [`crate::layout::RegionClass`], as computed by [`match_regions`].
///
/// The indices in [`matches`](Self::matches) refer to positions in the *original*
/// `predicted`/`ground_truth` slices passed to [`match_regions`], not to any
/// per-class-filtered subset, so callers can look up the full [`crate::layout::Region`]
/// (e.g. its `bbox`) for every matched pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassMatch {
    /// Predictions greedily matched to a ground-truth region of the same class at or
    /// above the matching threshold.
    pub true_positives: usize,
    /// Predictions left unmatched.
    pub false_positives: usize,
    /// Ground-truth regions left unmatched.
    pub false_negatives: usize,
    /// `(predicted_index, ground_truth_index)` pairs, in the order they were matched
    /// (predictions taken highest-confidence first). Indices are into the original
    /// slices passed to [`match_regions`].
    pub matches: Vec<(usize, usize)>,
}

/// Greedily matches `predicted` regions to `ground_truth` regions of the same
/// [`crate::layout::RegionClass`] at a given IoU threshold (COCO-style), broken out per
/// class.
///
/// For each class present in `predicted` or `ground_truth`: predictions are sorted by
/// confidence (highest first) and greedily matched to the highest-IoU unmatched
/// ground-truth region of the same class with `IoU >= iou_threshold`. Unmatched
/// predictions count as false positives, unmatched ground-truth regions as false
/// negatives. [`region_f1`] and [`crate::eval`]'s dashboard aggregation are both built
/// on this same matching, so per-class F1 and mean GIoU stay consistent with each
/// other.
///
/// Classes with no predictions and no ground truth are omitted from the result.
pub fn match_regions(
    predicted: &[crate::layout::Region],
    ground_truth: &[crate::layout::Region],
    iou_threshold: f32,
) -> HashMap<crate::layout::RegionClass, ClassMatch> {
    use crate::layout::RegionClass;

    let mut classes: Vec<RegionClass> = predicted
        .iter()
        .map(|r| r.class)
        .chain(ground_truth.iter().map(|r| r.class))
        .collect();
    classes.sort_by_key(|c| *c as u8 as u32 + region_class_rank(*c));
    classes.dedup();

    let mut results = HashMap::new();

    for class in classes {
        let mut preds: Vec<(usize, &crate::layout::Region)> = predicted
            .iter()
            .enumerate()
            .filter(|(_, r)| r.class == class)
            .collect();
        preds.sort_by(|(_, a), (_, b)| b.confidence.partial_cmp(&a.confidence).unwrap());

        let gts: Vec<(usize, &crate::layout::Region)> = ground_truth
            .iter()
            .enumerate()
            .filter(|(_, r)| r.class == class)
            .collect();
        let mut gt_matched = vec![false; gts.len()];

        let mut matches = Vec::new();
        for (pred_index, pred) in &preds {
            let mut best: Option<(usize, f32)> = None;
            for (gt_slot, (_, gt)) in gts.iter().enumerate() {
                if gt_matched[gt_slot] {
                    continue;
                }
                let score = iou(pred.bbox, gt.bbox);
                if score >= iou_threshold && best.is_none_or(|(_, b)| score > b) {
                    best = Some((gt_slot, score));
                }
            }
            if let Some((gt_slot, _)) = best {
                gt_matched[gt_slot] = true;
                matches.push((*pred_index, gts[gt_slot].0));
            }
        }

        let true_positives = matches.len();
        let false_positives = preds.len() - true_positives;
        let false_negatives = gts.len() - true_positives;

        results.insert(
            class,
            ClassMatch {
                true_positives,
                false_positives,
                false_negatives,
                matches,
            },
        );
    }

    results
}

/// Region-classification F1 at a given IoU matching threshold (COCO-style), broken out
/// per [`crate::layout::RegionClass`].
///
/// Matches `predicted` to `ground_truth` via [`match_regions`] and scores each class as
/// `F1 = 2*TP / (2*TP + FP + FN)`. See [`match_regions`] for the matching rule.
///
/// Classes with no predictions and no ground truth are omitted from the result.
pub fn region_f1(
    predicted: &[crate::layout::Region],
    ground_truth: &[crate::layout::Region],
    iou_threshold: f32,
) -> HashMap<crate::layout::RegionClass, f32> {
    match_regions(predicted, ground_truth, iou_threshold)
        .into_iter()
        .filter_map(|(class, m)| {
            let denom = 2 * m.true_positives + m.false_positives + m.false_negatives;
            (denom > 0).then_some((class, (2 * m.true_positives) as f32 / denom as f32))
        })
        .collect()
}

/// Stable ordering key for [`crate::layout::RegionClass`] used only to make
/// [`region_f1`]'s internal class enumeration deterministic; not a meaningful ranking.
fn region_class_rank(class: crate::layout::RegionClass) -> u32 {
    use crate::layout::RegionClass::*;
    match class {
        Caption => 0,
        Footnote => 1,
        Formula => 2,
        ListItem => 3,
        PageFooter => 4,
        PageHeader => 5,
        Picture => 6,
        SectionHeader => 7,
        Table => 8,
        Text => 9,
        Title => 10,
    }
}

/// Tree-Edit-Distance-based Similarity, structure only (row/column/span topology,
/// ignoring cell text), as defined by PubTabNet.
///
/// `predicted_html` and `ground_truth_html` are parsed as the restricted table-HTML
/// dialect documented on the [`teds`] submodule. Returns a score in `[0.0, 1.0]`, where
/// `1.0` means identical row/column/span topology.
pub fn teds_struct(predicted_html: &str, ground_truth_html: &str) -> f64 {
    teds::score(
        predicted_html,
        ground_truth_html,
        teds::CostMode::StructureOnly,
    )
}

/// Tree-Edit-Distance-based Similarity, structure and content combined.
///
/// Like [`teds_struct`], but matching cells are further scored by normalized string edit
/// distance between their text content.
pub fn teds(predicted_html: &str, ground_truth_html: &str) -> f64 {
    teds::score(predicted_html, ground_truth_html, teds::CostMode::WithText)
}

/// TEDS(IOU): a text-independent TEDS variant that scores cell content by bounding-box
/// IoU instead of string edit distance, appropriate for an OCR-free, bbox-driven
/// pipeline like this one.
///
/// Cells are expected to carry a `bbox="left bottom right top"` attribute (see the
/// [`teds`] submodule); cells missing it score maximal substitution cost against any
/// counterpart.
pub fn teds_iou(predicted_html: &str, ground_truth_html: &str) -> f64 {
    teds::score(predicted_html, ground_truth_html, teds::CostMode::Iou)
}

/// TEDS scoring machinery: a minimal restricted-HTML table parser and a Zhang–Shasha
/// tree-edit-distance implementation, shared by [`super::teds_struct`], [`super::teds`],
/// and [`super::teds_iou`].
///
/// ## Supported HTML dialect
///
/// Only `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<td>`, and `<th>` tags are recognized;
/// any other markup is treated as an error by the caller (all three public functions
/// require well-formed input from this dialect — this is a scoring utility for
/// controlled evaluation harnesses, not a general HTML parser). Supported attributes:
/// `colspan`, `rowspan` (integers, default `1`), and `bbox="left bottom right top"`
/// (four whitespace-separated floats, used only by [`super::teds_iou`]). Cell text
/// content is everything between a cell's open and close tag, whitespace-trimmed.
mod teds {
    use crate::BBox;

    /// Which substitution-cost policy [`score`] uses when comparing two matched cell
    /// nodes; see the parent module docs for what each variant means.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum CostMode {
        StructureOnly,
        WithText,
        Iou,
    }

    /// A parsed table-tree node.
    #[derive(Debug, Clone, PartialEq)]
    pub(super) struct Node {
        pub tag: String,
        pub colspan: u32,
        pub rowspan: u32,
        pub text: String,
        pub bbox: Option<BBox>,
        pub children: Vec<Node>,
    }

    /// Computes the TEDS-family score for `predicted_html` vs. `ground_truth_html` under
    /// `mode`.
    pub(super) fn score(predicted_html: &str, ground_truth_html: &str, mode: CostMode) -> f64 {
        let predicted = parse_table(predicted_html);
        let ground_truth = parse_table(ground_truth_html);

        let max_nodes = node_count(&predicted).max(node_count(&ground_truth)).max(1);
        let distance = tree_edit_distance(&predicted, &ground_truth, mode);

        (1.0 - distance / max_nodes as f64).clamp(0.0, 1.0)
    }

    fn node_count(node: &Node) -> usize {
        1 + node.children.iter().map(node_count).sum::<usize>()
    }

    // --- Minimal restricted-HTML table parser -----------------------------------

    fn parse_table(html: &str) -> Node {
        let mut stack: Vec<Node> = Vec::new();
        let mut root: Option<Node> = None;
        let mut cursor = 0usize;

        while cursor < html.len() {
            let Some(lt) = html[cursor..].find('<').map(|o| cursor + o) else {
                // Trailing text with no more tags: attribute it to whatever is open.
                if let Some(top) = stack.last_mut() {
                    top.text.push_str(&html[cursor..]);
                }
                break;
            };

            // Text between the previous tag and this one belongs to the innermost
            // currently-open node.
            if lt > cursor {
                if let Some(top) = stack.last_mut() {
                    top.text.push_str(&html[cursor..lt]);
                }
            }

            let gt = html[lt..].find('>').map(|o| lt + o).unwrap_or(html.len());
            let tag_src = &html[lt + 1..gt];

            if let Some(name) = tag_src.strip_prefix('/') {
                let name = name.trim();
                if let Some(mut finished) = stack.pop() {
                    debug_assert_eq!(finished.tag, name, "mismatched closing tag");
                    finished.text = finished.text.trim().to_string();
                    match stack.last_mut() {
                        Some(parent) => parent.children.push(finished),
                        None => root = Some(finished),
                    }
                }
            } else {
                let mut parts = tag_src.split_whitespace();
                let name = parts.next().unwrap_or("").to_string();
                let mut colspan = 1;
                let mut rowspan = 1;
                let mut bbox = None;

                for attr in parts {
                    if let Some((key, val)) = attr.split_once('=') {
                        let val = val.trim_matches('"');
                        match key {
                            "colspan" => colspan = val.parse().unwrap_or(1),
                            "rowspan" => rowspan = val.parse().unwrap_or(1),
                            "bbox" => bbox = parse_bbox(val),
                            _ => {}
                        }
                    }
                }

                stack.push(Node {
                    tag: name,
                    colspan,
                    rowspan,
                    text: String::new(),
                    bbox,
                    children: Vec::new(),
                });
            }

            cursor = gt + 1;
        }

        root.unwrap_or(Node {
            tag: "table".to_string(),
            colspan: 1,
            rowspan: 1,
            text: String::new(),
            bbox: None,
            children: Vec::new(),
        })
    }

    fn parse_bbox(val: &str) -> Option<BBox> {
        let mut nums = val.split_whitespace().filter_map(|n| n.parse::<f32>().ok());
        Some(BBox {
            left: nums.next()?,
            bottom: nums.next()?,
            right: nums.next()?,
            top: nums.next()?,
        })
    }

    // --- Tree edit distance (Zhang-Shasha) ---------------------------------------

    /// Flattens a tree into postorder, alongside each node's "leftmost leaf descendant"
    /// index (`l(i)` in Zhang-Shasha notation) needed for the keyroot recursion.
    fn postorder(node: &Node) -> (Vec<&Node>, Vec<usize>) {
        let mut nodes = Vec::new();
        let mut leftmost = Vec::new();
        postorder_visit(node, &mut nodes, &mut leftmost);
        (nodes, leftmost)
    }

    fn postorder_visit<'a>(node: &'a Node, nodes: &mut Vec<&'a Node>, leftmost: &mut Vec<usize>) {
        let mut first_child_leftmost = None;
        for child in &node.children {
            if first_child_leftmost.is_none() {
                first_child_leftmost = Some(nodes.len());
            }
            postorder_visit(child, nodes, leftmost);
        }
        let this_index = nodes.len();
        let this_leftmost = if node.children.is_empty() {
            this_index
        } else {
            leftmost[first_child_leftmost.unwrap()]
        };
        nodes.push(node);
        leftmost.push(this_leftmost);
    }

    fn keyroots(leftmost: &[usize]) -> Vec<usize> {
        let mut seen = vec![false; leftmost.len()];
        let mut roots = Vec::new();
        for i in (0..leftmost.len()).rev() {
            if !seen[leftmost[i]] {
                roots.push(i);
                seen[leftmost[i]] = true;
            }
        }
        roots.sort_unstable();
        roots
    }

    fn substitution_cost(a: &Node, b: &Node, mode: CostMode) -> f64 {
        if a.tag != b.tag || a.colspan != b.colspan || a.rowspan != b.rowspan {
            return 1.0;
        }
        match mode {
            CostMode::StructureOnly => 0.0,
            CostMode::WithText => normalized_edit_distance(&a.text, &b.text),
            CostMode::Iou => match (a.bbox, b.bbox) {
                (Some(x), Some(y)) => 1.0 - super::iou(x, y) as f64,
                _ => 1.0,
            },
        }
    }

    /// Classic Levenshtein distance, normalized by the longer string's length; `0.0` for
    /// two empty strings.
    fn normalized_edit_distance(a: &str, b: &str) -> f64 {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }

        let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
        for (i, row) in dp.iter_mut().enumerate() {
            row[0] = i;
        }
        for (j, cell) in dp[0].iter_mut().enumerate() {
            *cell = j;
        }
        for i in 1..=a.len() {
            for j in 1..=b.len() {
                dp[i][j] = if a[i - 1] == b[j - 1] {
                    dp[i - 1][j - 1]
                } else {
                    1 + dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1])
                };
            }
        }

        dp[a.len()][b.len()] as f64 / a.len().max(b.len()) as f64
    }

    /// Zhang-Shasha tree edit distance between two ordered, labeled trees, with
    /// insertion/deletion cost `1.0` and substitution cost per `mode` (via
    /// [`substitution_cost`]).
    fn tree_edit_distance(root_a: &Node, root_b: &Node, mode: CostMode) -> f64 {
        let (nodes_a, leftmost_a) = postorder(root_a);
        let (nodes_b, leftmost_b) = postorder(root_b);
        let n = nodes_a.len();
        let m = nodes_b.len();

        // `treedist[i][j]` (1-indexed) once computed by the keyroot loop below.
        let mut treedist = vec![vec![0.0f64; m + 1]; n + 1];

        for &i in &keyroots(&leftmost_a) {
            for &j in &keyroots(&leftmost_b) {
                let li = leftmost_a[i];
                let lj = leftmost_b[j];
                let width = i - li + 2;
                let height = j - lj + 2;
                let mut forest_dist = vec![vec![0.0f64; height]; width];

                for di in 1..width {
                    forest_dist[di][0] = forest_dist[di - 1][0] + 1.0;
                }
                for dj in 1..height {
                    forest_dist[0][dj] = forest_dist[0][dj - 1] + 1.0;
                }

                for di in 1..width {
                    for dj in 1..height {
                        let node_i = li + di - 1;
                        let node_j = lj + dj - 1;

                        if leftmost_a[node_i] == li && leftmost_b[node_j] == lj {
                            let sub = substitution_cost(nodes_a[node_i], nodes_b[node_j], mode);
                            let cost = (forest_dist[di - 1][dj] + 1.0)
                                .min(forest_dist[di][dj - 1] + 1.0)
                                .min(forest_dist[di - 1][dj - 1] + sub);
                            forest_dist[di][dj] = cost;
                            treedist[node_i + 1][node_j + 1] = cost;
                        } else {
                            let ii = leftmost_a[node_i] - li;
                            let jj = leftmost_b[node_j] - lj;
                            let cost = (forest_dist[di - 1][dj] + 1.0)
                                .min(forest_dist[di][dj - 1] + 1.0)
                                .min(forest_dist[ii][jj] + treedist[node_i + 1][node_j + 1]);
                            forest_dist[di][dj] = cost;
                        }
                    }
                }
            }
        }

        treedist[n][m]
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parses_simple_table() {
            let node = parse_table("<table><tr><td>a</td><td>b</td></tr></table>");
            assert_eq!(node.tag, "table");
            assert_eq!(node.children.len(), 1);
            assert_eq!(node.children[0].children.len(), 2);
            assert_eq!(node.children[0].children[0].text, "a");
        }

        #[test]
        fn identical_tables_score_one() {
            let html = "<table><tr><td>a</td><td>b</td></tr></table>";
            assert_eq!(score(html, html, CostMode::StructureOnly), 1.0);
            assert_eq!(score(html, html, CostMode::WithText), 1.0);
        }

        #[test]
        fn rowspan_mismatch_lowers_structure_score() {
            let a = "<table><tr><td>a</td><td>b</td></tr></table>";
            let b = "<table><tr><td rowspan=\"2\">a</td><td>b</td></tr></table>";
            assert!(score(a, b, CostMode::StructureOnly) < 1.0);
        }

        #[test]
        fn text_mismatch_only_affects_teds_not_teds_struct() {
            let a = "<table><tr><td>a</td></tr></table>";
            let b = "<table><tr><td>z</td></tr></table>";
            assert_eq!(score(a, b, CostMode::StructureOnly), 1.0);
            assert!(score(a, b, CostMode::WithText) < 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Region, RegionClass};

    fn bbox(left: f32, bottom: f32, right: f32, top: f32) -> BBox {
        BBox {
            left,
            bottom,
            right,
            top,
        }
    }

    #[test]
    fn giou_identical_boxes_is_one() {
        let a = bbox(0.0, 0.0, 10.0, 10.0);
        assert_eq!(giou(a, a), 1.0);
    }

    #[test]
    fn giou_disjoint_boxes_is_negative() {
        let a = bbox(0.0, 0.0, 10.0, 10.0);
        let b = bbox(20.0, 0.0, 30.0, 10.0);
        assert!(giou(a, b) < 0.0);
    }

    #[test]
    fn region_f1_perfect_match_is_one() {
        let region = |bbox| Region {
            class: RegionClass::Text,
            bbox,
            confidence: 1.0,
            page: 0,
        };
        let predicted = vec![region(bbox(0.0, 0.0, 10.0, 10.0))];
        let ground_truth = vec![region(bbox(0.0, 0.0, 10.0, 10.0))];

        let scores = region_f1(&predicted, &ground_truth, 0.5);
        assert_eq!(scores[&RegionClass::Text], 1.0);
    }

    #[test]
    fn region_f1_with_miss_and_false_positive() {
        let region = |bbox| Region {
            class: RegionClass::Title,
            bbox,
            confidence: 1.0,
            page: 0,
        };
        // One predicted region matches, one is a false positive; ground truth has one
        // extra miss.
        let predicted = vec![
            region(bbox(0.0, 0.0, 10.0, 10.0)),
            region(bbox(100.0, 100.0, 110.0, 110.0)),
        ];
        let ground_truth = vec![
            region(bbox(0.0, 0.0, 10.0, 10.0)),
            region(bbox(50.0, 50.0, 60.0, 60.0)),
        ];

        let scores = region_f1(&predicted, &ground_truth, 0.5);
        // TP=1, FP=1, FN=1 -> F1 = 2/(2+1+1) = 0.5
        assert_eq!(scores[&RegionClass::Title], 0.5);
    }
}

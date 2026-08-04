//! Criterion micro-benches for the Stage 2 metric primitives in
//! `pdfspatial_core::metrics`: [`giou`], [`region_f1`] (which drives `match_regions`
//! internally), and the TEDS family ([`teds`], [`teds_struct`], [`teds_iou`]).
//!
//! Uses small, hand-built inputs rather than a dataset — these are pure-function
//! micro-benches, not pipeline throughput benches (see `stage2_eval.rs` for those). Needs
//! no cargo feature, since none of these APIs are gated behind `doclaynet`.
//!
//! ```sh
//! cargo bench --bench metrics
//! ```

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use pdfspatial_core::BBox;
use pdfspatial_core::layout::{Region, RegionClass};
use pdfspatial_core::metrics::{giou, region_f1, teds, teds_iou, teds_struct};

fn bbox(left: f32, bottom: f32, right: f32, top: f32) -> BBox {
    BBox {
        left,
        bottom,
        right,
        top,
    }
}

fn bench_giou(c: &mut Criterion) {
    let overlapping = (bbox(0.0, 0.0, 10.0, 10.0), bbox(5.0, 5.0, 15.0, 15.0));
    let disjoint = (bbox(0.0, 0.0, 10.0, 10.0), bbox(20.0, 20.0, 30.0, 30.0));

    let mut group = c.benchmark_group("giou");
    group.bench_function("overlapping", |b| {
        b.iter(|| giou(black_box(overlapping.0), black_box(overlapping.1)))
    });
    group.bench_function("disjoint", |b| {
        b.iter(|| giou(black_box(disjoint.0), black_box(disjoint.1)))
    });
    group.finish();
}

fn region(class: RegionClass, b: BBox) -> Region {
    Region {
        class,
        bbox: b,
        confidence: 1.0,
        page: 0,
    }
}

fn bench_region_f1(c: &mut Criterion) {
    let predicted = vec![
        region(RegionClass::Title, bbox(0.0, 90.0, 100.0, 100.0)),
        region(RegionClass::Text, bbox(0.0, 40.0, 100.0, 85.0)),
        region(RegionClass::PageFooter, bbox(0.0, 0.0, 100.0, 5.0)),
    ];
    let ground_truth = vec![
        region(RegionClass::Title, bbox(0.0, 91.0, 100.0, 100.0)),
        region(RegionClass::Text, bbox(0.0, 40.0, 100.0, 84.0)),
        region(RegionClass::PageFooter, bbox(0.0, 0.0, 100.0, 6.0)),
        region(RegionClass::Table, bbox(0.0, 10.0, 100.0, 39.0)),
    ];

    c.bench_function("region_f1", |b| {
        b.iter(|| {
            region_f1(
                black_box(&predicted),
                black_box(&ground_truth),
                black_box(0.5),
            )
        })
    });
}

/// A small predicted/ground-truth table pair: identical topology, one cell's text and
/// bbox perturbed, so the TEDS family has non-trivial edit distance to compute.
fn table_pair() -> (String, String) {
    let predicted = "<table><tr><td bbox=\"0 0 10 10\">A</td><td bbox=\"10 0 20 10\">B</td></tr>\
                      <tr><td bbox=\"0 10 10 20\">C</td><td bbox=\"10 10 20 20\">D2</td></tr></table>"
        .to_string();
    let ground_truth = "<table><tr><td bbox=\"0 0 10 10\">A</td><td bbox=\"10 0 20 10\">B</td></tr>\
                         <tr><td bbox=\"0 10 10 20\">C</td><td bbox=\"10 10 20 20\">D</td></tr></table>"
        .to_string();
    (predicted, ground_truth)
}

fn bench_teds(c: &mut Criterion) {
    let (predicted, ground_truth) = table_pair();

    let mut group = c.benchmark_group("teds");
    group.bench_function("teds_struct", |b| {
        b.iter(|| teds_struct(black_box(&predicted), black_box(&ground_truth)))
    });
    group.bench_function("teds", |b| {
        b.iter(|| teds(black_box(&predicted), black_box(&ground_truth)))
    });
    group.bench_function("teds_iou", |b| {
        b.iter(|| teds_iou(black_box(&predicted), black_box(&ground_truth)))
    });
    group.finish();
}

criterion_group!(benches, bench_giou, bench_region_f1, bench_teds);
criterion_main!(benches);

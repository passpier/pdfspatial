//! Criterion benches for the Stage 2 DocLayNet eval pipeline: [`load_sample`] (I/O +
//! JSON parsing), [`evaluate_sample`] (classify + metric aggregation end-to-end), and
//! [`evaluate_pages`] alone (aggregation only, isolated from loading/classifying).
//!
//! Defaults to the vendored fixture in `tests/fixtures/doclaynet/` (deterministic, no
//! network access, safe for CI). Set `DOCLAYNET_DIR` to bench against a real unpacked
//! DocLayNet-core sample instead — same layout `doclaynet_eval` expects:
//! `$DOCLAYNET_DIR/COCO/val.json` and `$DOCLAYNET_DIR/JSON`.
//!
//! ```sh
//! cargo bench --features doclaynet
//! DOCLAYNET_DIR=/path/to/doclaynet cargo bench --features doclaynet
//! ```

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use pdfspatial_core::eval::doclaynet::{
    DocLayNetSample, evaluate_sample, load_sample, predict_regions_textonly,
};
use pdfspatial_core::eval::evaluate_pages;
use pdfspatial_core::layout::Region;
use std::path::PathBuf;

/// Resolves `(coco_json, cells_dir)`: `$DOCLAYNET_DIR/COCO/val.json` +
/// `$DOCLAYNET_DIR/JSON` when `DOCLAYNET_DIR` is set (mirrors
/// `examples/doclaynet_eval.rs`'s default), else the vendored fixture under
/// `tests/fixtures/doclaynet/`.
fn sample_paths() -> (PathBuf, PathBuf) {
    if let Some(base) = std::env::var_os("DOCLAYNET_DIR") {
        let base = PathBuf::from(base);
        return (base.join("COCO/val.json"), base.join("JSON"));
    }

    // `CARGO_MANIFEST_DIR` is the crate directory (`crates/pdfspatial-core`) even for
    // this out-of-crate bench path, same as it would be for an in-crate `tests/*.rs`.
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/doclaynet");
    (fixtures.join("coco.json"), fixtures)
}

fn load_fixture() -> DocLayNetSample {
    let (coco_json, cells_dir) = sample_paths();
    load_sample(&coco_json, &cells_dir).expect("DocLayNet sample should load")
}

fn bench_load_sample(c: &mut Criterion) {
    let (coco_json, cells_dir) = sample_paths();

    c.bench_function("load_sample", |b| {
        b.iter(|| load_sample(black_box(&coco_json), black_box(&cells_dir)).unwrap())
    });
}

fn bench_evaluate_sample(c: &mut Criterion) {
    let sample = load_fixture();

    let mut group = c.benchmark_group("evaluate_sample");
    group.throughput(Throughput::Elements(sample.pages.len() as u64));
    group.bench_function("evaluate_sample", |b| {
        b.iter(|| evaluate_sample(black_box(&sample), black_box(0.5)))
    });
    group.finish();
}

fn bench_evaluate_pages(c: &mut Criterion) {
    let sample = load_fixture();
    let pairs: Vec<(Vec<Region>, Vec<Region>)> = sample
        .pages
        .iter()
        .map(|page| (predict_regions_textonly(page), page.ground_truth.clone()))
        .collect();

    let mut group = c.benchmark_group("evaluate_pages");
    group.throughput(Throughput::Elements(pairs.len() as u64));
    group.bench_function("evaluate_pages", |b| {
        b.iter(|| evaluate_pages(black_box(&pairs), black_box(0.5)))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_load_sample,
    bench_evaluate_sample,
    bench_evaluate_pages
);
criterion_main!(benches);

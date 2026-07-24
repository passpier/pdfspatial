# Stage 2 validation benchmarks

[Criterion](https://docs.rs/criterion) benches that exercise Stage 2 validation
(`crates/pdfspatial-core/src/metrics.rs`, `crates/pdfspatial-core/src/eval/`), registered as
out-of-crate `[[bench]]` targets in `crates/pdfspatial-core/Cargo.toml` (the same pattern
already used for the `[[example]]` targets under `examples/`).

- **`stage2_eval.rs`** — the DocLayNet eval pipeline: `eval::doclaynet::load_sample` (I/O +
  JSON parsing), `eval::doclaynet::evaluate_sample` (classify + metric aggregation
  end-to-end), and `eval::evaluate_pages` alone (aggregation only, isolated from
  loading/classifying). Requires the `doclaynet` cargo feature.
- **`metrics.rs`** — micro-benches for the metric primitives themselves: `giou`,
  `region_f1` (which drives `match_regions` internally), and the TEDS family
  (`teds`/`teds_struct`/`teds_iou`). No cargo feature required.

## Running

```sh
# Both benches, against the vendored fixture in
# crates/pdfspatial-core/tests/fixtures/doclaynet/ (deterministic, no network access):
cargo bench --features doclaynet

# metrics.rs alone needs no feature:
cargo bench --bench metrics

# Against a real unpacked DocLayNet-core sample instead of the fixture:
DOCLAYNET_DIR=/path/to/doclaynet cargo bench --features doclaynet
```

`DOCLAYNET_DIR` is expected to contain `COCO/val.json` and a `JSON/` directory of
`{image_file_stem}.cells.json` files — the same layout `examples/doclaynet_eval.rs` expects.

Stage 1's throughput metric (pages/sec, single core, no OCR — recorded per the roadmap, not
gated on a target) is exercised separately by the integration tests in
`crates/pdfspatial-core/tests/`, not by these benches.

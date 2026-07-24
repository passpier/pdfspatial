# Stage 2 validation benchmarks

Placeholder. Stage 2 validation metrics (TEDS-Struct, TEDS, TEDS(IOU), mean GIoU, region
F1 — see `crates/pdfspatial-core/src/metrics.rs`) are now wired up against DocLayNet via
the `eval` module (`crates/pdfspatial-core/src/eval/`, `doclaynet` cargo feature; see
`crates/pdfspatial-core/tests/stage2_doclaynet.rs` and `examples/doclaynet_eval.rs`), but
there are no Criterion benchmarks here yet — that's what this directory is still a
placeholder for.

Once benchmarks land, this directory will hold [Criterion](https://docs.rs/criterion)
benches that score pipeline output against a held-out DocLayNet sample via
`eval::evaluate_pages`/`eval::doclaynet::evaluate_sample`, tracked here as `.bench.rs`
targets registered in `crates/pdfspatial-core/Cargo.toml`.

In the meantime, Stage 1's throughput metric (pages/sec, single core, no OCR — recorded
per the roadmap, not gated on a target) is exercised by the integration tests in
`crates/pdfspatial-core/tests/`.

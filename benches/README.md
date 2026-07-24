# Stage 2 validation benchmarks

Placeholder. Stage 2 validation (TEDS-Struct, TEDS, TEDS(IOU), mean GIoU, region F1
against DocLayNet — see `crates/pdfspatial-core/src/metrics.rs`) is not yet implemented,
so there are no benchmarks here yet.

Once Stage 2 metrics land, this directory will hold [Criterion](https://docs.rs/criterion)
benches that score pipeline output against a held-out DocLayNet sample, tracked here as
`.bench.rs` targets registered in `crates/pdfspatial-core/Cargo.toml`.

In the meantime, Stage 1's throughput metric (pages/sec, single core, no OCR — recorded
per the roadmap, not gated on a target) is exercised by the integration tests in
`crates/pdfspatial-core/tests/`.

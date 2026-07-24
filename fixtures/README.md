# Stage 3 regression corpus

Placeholder. Stage 3 error analysis is not yet implemented (see
`crates/pdfspatial-core/src/assemble.rs` for the pitfall checklist it will collect
against), so there is no regression corpus here yet.

Once Stage 3 lands, this directory will hold minimal-repro PDF pages — at least 20 per
pitfall category — each tagged with its root cause (`geometric`, `classification`, or
`ordering`) and the DocLayNet-derived sample it was extracted from, per the roadmap's
Stage 3 process.

This is distinct from `crates/pdfspatial-core/tests/fixtures/`, which holds small,
hand-authored PDFs used by today's Stage 1 unit tests.

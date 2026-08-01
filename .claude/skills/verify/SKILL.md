---
name: verify
description: Run the full check suite for pdfspatial (format, lint, and all-feature tests). Use before considering a change in this repo done, or when the user asks to "verify", "run checks", or "make sure this passes CI".
---

Run these in order and report results. Stop and report at the first failure rather than
continuing past it.

1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
3. Tests, tiered by what's available:
   - `cargo test --features doclaynet` always runs (no external dependency).
   - `cargo test --all-features` additionally exercises `tests/stage1_baseline.rs`, which needs a
     native PDFium library. Only run this tier if `PDFSPATIAL_PDFIUM_LIB` is set in the
     environment, a gitignored `.cargo/config.toml` sets it (written by
     `scripts/fetch-pdfium.sh --write-cargo-config` — check for `[env]` /
     `PDFSPATIAL_PDFIUM_LIB` in that file), or the library is reachable via the OS
     dynamic-loader path. If none of those hold, run the `doclaynet`-only tier, tell the user
     PDFium-dependent tests were skipped, and point them at
     `./scripts/fetch-pdfium.sh --write-cargo-config`.

A plain `cargo test` alone is insufficient — it silently skips the `doclaynet`-gated dataset
harness test.

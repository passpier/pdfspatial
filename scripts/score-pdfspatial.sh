#!/usr/bin/env bash
# Fast pdfspatial-only scoring loop against the opendataloader-bench corpus, for
# iterating on the score-raising work tracked in the "Raising pdfspatial's
# opendataloader-bench score" plan. NOT part of the full six-engine benchmark
# (scripts/run-opendataloader-bench.sh) and never run in CI -- see CLAUDE.md's Stage 5
# section. This script only builds pdfspatial, runs it over the 200-PDF corpus, and
# reproduces evaluator.py's exact per-document ragged mean using the already-vendored
# upstream clone and venv -- it never touches bench/opendataloader/results/results.json
# or the README table.
#
# Usage:
#   ./scripts/score-pdfspatial.sh                  # faithful default MarkdownOptions
#   ./scripts/score-pdfspatial.sh --no-page-breaks --no-image-placeholders
#   ./scripts/score-pdfspatial.sh --save-baseline   # also snapshot as the diff baseline
#
# Requires: third_party/opendataloader-bench already cloned with its .venv synced (see
# scripts/run-opendataloader-bench.sh, run at least once with --skip-clone off) and the
# ground-truth/markdown + pdfs directories present.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_CLONE="$ROOT/third_party/opendataloader-bench"
VENV_PY="$BENCH_CLONE/.venv/bin/python"
SCRATCH="$ROOT/target/score-pdfspatial"
BASELINE_FILE="$ROOT/target/score-pdfspatial-baseline.json"

SAVE_BASELINE=0
PDFSPATIAL_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--save-baseline" ]]; then
        SAVE_BASELINE=1
    else
        PDFSPATIAL_ARGS+=("$arg")
    fi
done

if [[ ! -x "$VENV_PY" ]]; then
    echo "error: $VENV_PY not found -- run scripts/run-opendataloader-bench.sh at least once to clone/sync the upstream bench" >&2
    exit 1
fi
if [[ ! -d "$BENCH_CLONE/pdfs" || ! -d "$BENCH_CLONE/ground-truth/markdown" ]]; then
    echo "error: $BENCH_CLONE/{pdfs,ground-truth/markdown} missing" >&2
    exit 1
fi

echo "==> cargo build --release -p pdfspatial" >&2
cargo build --release -p pdfspatial --manifest-path "$ROOT/Cargo.toml"
BIN="$ROOT/target/release/pdfspatial"

# .cargo/config.toml's [env] table only applies to `cargo run`/`cargo test`, not to a
# bare binary spawned directly -- same reason run-opendataloader-bench.sh exports this
# explicitly for its adapter (see CLAUDE.md / bench/opendataloader/README.md).
if [[ -z "${PDFSPATIAL_PDFIUM_LIB:-}" && -f "$ROOT/.cargo/config.toml" ]]; then
    export PDFSPATIAL_PDFIUM_LIB="$(grep -m1 PDFSPATIAL_PDFIUM_LIB "$ROOT/.cargo/config.toml" | sed -E 's/.*"(.*)".*/\1/')"
fi

rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"

echo "==> running pdfspatial over the 200-PDF corpus (args: ${PDFSPATIAL_ARGS[*]:-<default>})" >&2
START=$(date +%s.%N)
"$BIN" --out "$SCRATCH" --quiet --jobs 1 "${PDFSPATIAL_ARGS[@]+"${PDFSPATIAL_ARGS[@]}"}" "$BENCH_CLONE"/pdfs/*.pdf
END=$(date +%s.%N)

# Fill in an empty .md for any PDF that failed to produce output, matching
# bench/opendataloader/adapters/pdf_parser_pdfspatial.py's own defensive behavior --
# a missing prediction is dropped from evaluator.py's mean rather than scored zero.
for pdf in "$BENCH_CLONE"/pdfs/*.pdf; do
    stem="$(basename "$pdf" .pdf)"
    [[ -f "$SCRATCH/$stem.md" ]] || : > "$SCRATCH/$stem.md"
done

SAVE_BASELINE_FLAG=()
[[ "$SAVE_BASELINE" == "1" ]] && SAVE_BASELINE_FLAG=(--save-baseline "$BASELINE_FILE")

BASELINE_ARG=()
[[ -f "$BASELINE_FILE" ]] && BASELINE_ARG=(--baseline "$BASELINE_FILE")

"$VENV_PY" "$ROOT/scripts/score_pdfspatial_eval.py" \
    --gt-dir "$BENCH_CLONE/ground-truth/markdown" \
    --pred-dir "$SCRATCH" \
    --elapsed "$(echo "$END - $START" | bc)" \
    --doc-count "$(ls "$BENCH_CLONE"/pdfs/*.pdf | wc -l | tr -d ' ')" \
    "${SAVE_BASELINE_FLAG[@]+"${SAVE_BASELINE_FLAG[@]}"}" \
    "${BASELINE_ARG[@]+"${BASELINE_ARG[@]}"}"

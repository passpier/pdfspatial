#!/bin/sh
# Runs the real opendataloader-bench corpus (200 PDFs) against pdfspatial and a handful
# of comparison engines, and collects the results into
# bench/opendataloader/results/results.json + a ready-to-paste README table.
#
# Usage:
#   scripts/run-opendataloader-bench.sh [--dest DIR] [--engines LIST] [--doc-id ID]
#                                       [--skip-clone] [--skip-build] [--skip-sync]
#                                       [--force] [--collect-only] [--jobs N] [--help]
#
# Env overrides:
#   BENCH_REPO              git URL to clone (default: opendataloader-project/opendataloader-bench)
#   BENCH_REF               branch/tag to check out (default: main)
#   PDFSPATIAL_PDFIUM_LIB   PDFium library for the release `pdfspatial` binary (default:
#                            <repo>/third_party/pdfium/lib/<libname>, from fetch-pdfium.sh)
#   PDF_INSPECTOR_BIN       path to a pre-built pdf2md binary (default: cargo-install one)
#
# This never runs in CI: it needs uv + Python 3.13, network access, and (for a
# meaningful speed column) an otherwise-idle machine for potentially hours. See
# bench/opendataloader/README.md.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)

BENCH_REPO="${BENCH_REPO:-https://github.com/opendataloader-project/opendataloader-bench}"
BENCH_REF="${BENCH_REF:-main}"

dest="$repo_root/third_party/opendataloader-bench"
engines="pdfspatial pdfspatial-compact pdf-inspector markitdown liteparse opendataloader"
doc_id=""
skip_clone=0
skip_build=0
skip_sync=0
force=0
collect_only=0
jobs=1

usage() {
    cat <<EOF
Usage: $0 [--dest DIR] [--engines LIST] [--doc-id ID] [--skip-clone] [--skip-build]
          [--skip-sync] [--force] [--collect-only] [--jobs N] [--help]

Clones opendataloader-bench (200 real PDFs + ground truth, no LFS pull required as of
this writing -- see bench/opendataloader/README.md), builds the release pdfspatial
binary, registers pdfspatial/pdfspatial-compact/pdf-inspector as bench engines, runs
each engine in --engines over the corpus, and collects results.

  --dest DIR        Clone destination (default: $dest)
  --engines LIST     Space-separated engine names to run (default: "$engines")
  --doc-id ID        Restrict to one document stem, for a fast smoke test
  --skip-clone       Reuse an existing clone at --dest as-is (no clone/pull)
  --skip-build       Reuse an existing release pdfspatial binary (no cargo build)
  --skip-sync        Reuse the bench's existing uv environment (no uv sync)
  --force            Pass --force to run.py (re-run engines with existing evaluation.json)
  --collect-only     Skip running engines; just collect existing results into
                      bench/opendataloader/results/results.json
  --jobs N           Worker threads for pdfspatial's headline row (default: 1, matching
                      every comparison engine's single-process, sequential-document
                      execution -- see bench/opendataloader/README.md)
  --help             Show this help.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dest) dest=$2; shift 2 ;;
        --dest=*) dest=${1#--dest=}; shift ;;
        --engines) engines=$2; shift 2 ;;
        --engines=*) engines=${1#--engines=}; shift ;;
        --doc-id) doc_id=$2; shift 2 ;;
        --doc-id=*) doc_id=${1#--doc-id=}; shift ;;
        --skip-clone) skip_clone=1; shift ;;
        --skip-build) skip_build=1; shift ;;
        --skip-sync) skip_sync=1; shift ;;
        --force) force=1; shift ;;
        --collect-only) collect_only=1; shift ;;
        --jobs) jobs=$2; shift 2 ;;
        --jobs=*) jobs=${1#--jobs=}; shift ;;
        --help | -h) usage; exit 0 ;;
        *)
            echo "run-opendataloader-bench.sh: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

log() { echo "run-opendataloader-bench.sh: $*"; }

if [ "$collect_only" -eq 0 ]; then
    for tool in git uv cargo; do
        command -v "$tool" >/dev/null 2>&1 || {
            echo "run-opendataloader-bench.sh: '$tool' is required but not on PATH." >&2
            exit 1
        }
    done

    # --- 1. Clone / update the upstream corpus ------------------------------------
    if [ "$skip_clone" -eq 0 ]; then
        if [ -d "$dest/.git" ]; then
            log "updating existing clone at $dest"
            git -C "$dest" checkout -- src/engine_registry.py 2>/dev/null || true
            git -C "$dest" fetch origin "$BENCH_REF"
            git -C "$dest" checkout "$BENCH_REF"
            git -C "$dest" pull --ff-only origin "$BENCH_REF"
        else
            log "cloning $BENCH_REPO@$BENCH_REF into $dest"
            mkdir -p "$(dirname "$dest")"
            git clone --branch "$BENCH_REF" "$BENCH_REPO" "$dest"
        fi

        if command -v git-lfs >/dev/null 2>&1; then
            (cd "$dest" && git lfs install --local >/dev/null 2>&1 && git lfs pull) || \
                log "git lfs pull reported an error (continuing -- see the PDF check below)"
        else
            log "git-lfs not found; skipping (this corpus has not required it historically)"
        fi

        pdf_count=$(find "$dest/pdfs" -maxdepth 1 -name '*.pdf' | wc -l | tr -d ' ')
        log "found $pdf_count PDF(s) in $dest/pdfs"
        [ "$pdf_count" -ge 1 ] || {
            echo "run-opendataloader-bench.sh: no PDFs found under $dest/pdfs" >&2
            exit 1
        }
        first_pdf=$(find "$dest/pdfs" -maxdepth 1 -name '*.pdf' | sort | head -n1)
        if [ "$(head -c 9 "$first_pdf")" = "version h" ]; then
            echo "run-opendataloader-bench.sh: $first_pdf looks like an LFS pointer" \
                "file, not a real PDF -- 'git lfs pull' didn't fetch real content." >&2
            exit 1
        fi
    fi

    # --- 2. Python environment ------------------------------------------------------
    if [ "$skip_sync" -eq 0 ]; then
        log "uv sync (base + opendataloader + markitdown + liteparse extras)"
        (cd "$dest" && uv sync --extra opendataloader --extra markitdown --extra liteparse)
    fi

    # --- 3. Build the release pdfspatial binary --------------------------------------
    if [ "$skip_build" -eq 0 ]; then
        log "cargo build --release -p pdfspatial"
        (cd "$repo_root" && cargo build --release -p pdfspatial)
    fi
    pdfspatial_bin="$repo_root/target/release/pdfspatial"
    [ -x "$pdfspatial_bin" ] || {
        echo "run-opendataloader-bench.sh: $pdfspatial_bin not found -- build it first" \
            "(cargo build --release -p pdfspatial) or drop --skip-build." >&2
        exit 1
    }
    export PDFSPATIAL_BIN="$pdfspatial_bin"

    # cargo's own [env] table (.cargo/config.toml, written by
    # fetch-pdfium.sh --write-cargo-config) applies to `cargo run`/`cargo test`, NOT to
    # a bare binary spawned from Python -- this must be exported explicitly here.
    if [ -z "${PDFSPATIAL_PDFIUM_LIB:-}" ]; then
        os=$(uname -s)
        case "$os" in
            Darwin) lib_name="libpdfium.dylib" ;;
            Linux) lib_name="libpdfium.so" ;;
            *) lib_name="" ;;
        esac
        candidate="$repo_root/third_party/pdfium/lib/$lib_name"
        if [ -n "$lib_name" ] && [ -f "$candidate" ]; then
            PDFSPATIAL_PDFIUM_LIB="$candidate"
        fi
    fi
    if [ -z "${PDFSPATIAL_PDFIUM_LIB:-}" ] || [ ! -f "$PDFSPATIAL_PDFIUM_LIB" ]; then
        echo "run-opendataloader-bench.sh: PDFSPATIAL_PDFIUM_LIB is not set to an" \
            "existing file. Run ./scripts/fetch-pdfium.sh first, or export it yourself." >&2
        exit 1
    fi
    export PDFSPATIAL_PDFIUM_LIB
    log "PDFSPATIAL_PDFIUM_LIB=$PDFSPATIAL_PDFIUM_LIB"

    # --- 4. pdf-inspector -------------------------------------------------------------
    if [ -z "${PDF_INSPECTOR_BIN:-}" ]; then
        if command -v pdf2md >/dev/null 2>&1; then
            PDF_INSPECTOR_BIN=$(command -v pdf2md)
        else
            log "cargo install pdf-inspector --locked"
            cargo install pdf-inspector --locked
            PDF_INSPECTOR_BIN=$(command -v pdf2md || echo "$HOME/.cargo/bin/pdf2md")
        fi
    fi
    export PDF_INSPECTOR_BIN
    log "PDF_INSPECTOR_BIN=$PDF_INSPECTOR_BIN"

    # --- 5. Register pdfspatial/pdfspatial-compact/pdf-inspector as bench engines ----
    cp "$repo_root"/bench/opendataloader/adapters/*.py "$dest/src/"
    registry="$dest/src/engine_registry.py"
    marker="# --- BEGIN pdfspatial bench engines"
    if ! grep -qF "$marker" "$registry"; then
        cat "$repo_root/bench/opendataloader/registry_patch.py" >>"$registry"
        log "patched $registry"
    else
        log "$registry already patched"
    fi

    # --- 6. Run each engine, sequentially, on an otherwise-idle machine --------------
    for engine in $engines; do
        log "=== running engine: $engine ==="
        run_args="--engine $engine"
        [ -n "$doc_id" ] && run_args="$run_args --doc-id $doc_id"
        [ "$force" -eq 1 ] && run_args="$run_args --force"
        case "$engine" in
            pdfspatial)
                (cd "$dest" && PDFSPATIAL_ARGS="--jobs $jobs" uv run src/run.py $run_args)
                ;;
            *)
                (cd "$dest" && uv run src/run.py $run_args)
                ;;
        esac
    done
fi

# --- 7. Collect ------------------------------------------------------------------
log "collecting results"
BENCH_DIR="$dest" python3 "$repo_root/bench/opendataloader/collect.py" \
    --bench-dir "$dest" --engines $engines --jobs "$jobs"

log "done -- see bench/opendataloader/results/results.json"

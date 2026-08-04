#!/usr/bin/env python3
"""Collects opendataloader-bench per-engine `evaluation.json` reports into this repo's
committed `results/results.json`, and prints the Markdown table pasted into README.md.

Run from a shell with BENCH_DIR pointing at the upstream clone (see
scripts/run-opendataloader-bench.sh, which invokes this after every engine has run):

    BENCH_DIR=third_party/opendataloader-bench python3 bench/opendataloader/collect.py \
        --engines pdfspatial pdfspatial-compact pdf-inspector markitdown liteparse opendataloader

Each engine's `<BENCH_DIR>/prediction/<engine>/evaluation.json` is the one file this
script reads for scores: `src/run.py`'s `run_pipeline` merges the evaluator's NID/TEDS/MHS
means with the run's own `speed` block (from `summary.json`: `elapsed_per_doc`,
`total_elapsed`, `document_count`, `processor`) into that same file before this script
ever runs, so per-document speed and per-document accuracy come from one consistent
source. Speed is reported as **seconds per document** (`elapsed_per_doc`, the bench's own
unit), not per page -- page counts aren't uniformly available across every engine's
prediction output, only pdfspatial's own CLI reports those (see its stderr summary line),
so per-page numbers are a footnote, not a shared table column.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# name -> (display name, license) for the table's License column. Versions come from
# each engine's own evaluation.json `summary.engine_version` at collection time.
ENGINE_META = {
    "pdfspatial": ("pdfspatial", "MIT OR Apache-2.0"),
    "pdfspatial-compact": ("pdfspatial (compact)", "MIT OR Apache-2.0"),
    "pdf-inspector": ("pdf-inspector", "MIT OR Apache-2.0"),
    "markitdown": ("markitdown", "MIT"),
    "liteparse": ("liteparse", "Apache-2.0"),
    "opendataloader": ("opendataloader", "Apache-2.0"),
}


def _corpus_commit(bench_dir: Path) -> str:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(bench_dir), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_engine(bench_dir: Path, engine: str) -> dict | None:
    eval_path = bench_dir / "prediction" / engine / "evaluation.json"
    if not eval_path.is_file():
        print(f"collect.py: no evaluation.json for {engine!r} at {eval_path}", file=sys.stderr)
        return None

    with eval_path.open(encoding="utf-8") as f:
        data = json.load(f)

    score = data.get("metrics", {}).get("score", {})
    speed = data.get("speed") or data.get("summary") or {}
    summary = data.get("summary", {})
    display_name, license_ = ENGINE_META.get(engine, (engine, "unknown"))

    return {
        "name": engine,
        "display_name": display_name,
        "version": summary.get("engine_version", "unknown"),
        "license": license_,
        "overall": score.get("overall_mean"),
        "nid": score.get("nid_mean"),
        "nid_s": score.get("nid_s_mean"),
        "teds": score.get("teds_mean"),
        "teds_s": score.get("teds_s_mean"),
        "mhs": score.get("mhs_mean"),
        "mhs_s": score.get("mhs_s_mean"),
        "s_per_doc": speed.get("elapsed_per_doc"),
        "document_count": speed.get("document_count"),
        "missing_predictions": data.get("metrics", {}).get("missing_predictions"),
        "processor": speed.get("processor"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", default=None, help="Path to the upstream clone (default: $BENCH_DIR or third_party/opendataloader-bench)")
    parser.add_argument("--engines", nargs="+", required=True, help="Engine names to collect, as registered in engine_registry.py")
    parser.add_argument("--jobs", type=int, default=None, help="Worker threads pdfspatial's headline row was run with, recorded for the record")
    args = parser.parse_args()

    import os

    bench_dir = Path(
        args.bench_dir or os.environ.get("BENCH_DIR", REPO_ROOT / "third_party" / "opendataloader-bench")
    ).resolve()
    if not bench_dir.is_dir():
        print(f"collect.py: bench directory not found: {bench_dir}", file=sys.stderr)
        return 1

    engines = []
    for name in args.engines:
        entry = _load_engine(bench_dir, name)
        if entry is not None:
            engines.append(entry)
    if not engines:
        print("collect.py: no engine results collected -- nothing to write", file=sys.stderr)
        return 1

    engines.sort(key=lambda e: (e["overall"] is None, -(e["overall"] or 0)))

    documents = max((e["document_count"] or 0) for e in engines)
    processor = next((e["processor"] for e in engines if e["processor"]), platform.processor())

    results = {
        "corpus": "opendataloader-bench",
        "corpus_commit": _corpus_commit(bench_dir),
        "documents": documents,
        "date": date.today().isoformat(),
        "hardware": {
            "processor": processor,
            "os": f"{platform.system()} {platform.release()} {platform.machine()}",
            "jobs": args.jobs,
        },
        "engines": engines,
    }

    out_path = Path(__file__).resolve().parent / "results" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"collect.py: wrote {out_path}", file=sys.stderr)

    # Ready-to-paste Markdown table, mirroring the README's Benchmarks section.
    print("| Engine | Overall | Reading order (NID) | Table (TEDS) | Heading (MHS) | s/doc | License |")
    print("|---|---|---|---|---|---|---|")
    for e in engines:
        def fmt(v: float | None) -> str:
            return f"{v:.3f}" if v is not None else "N/A"

        print(
            f"| {e['display_name']} | {fmt(e['overall'])} | {fmt(e['nid'])} | "
            f"{fmt(e['teds'])} | {fmt(e['mhs'])} | "
            f"{e['s_per_doc']:.3f}s | {e['license']} |"
            if e["s_per_doc"] is not None
            else f"| {e['display_name']} | {fmt(e['overall'])} | {fmt(e['nid'])} | "
            f"{fmt(e['teds'])} | {fmt(e['mhs'])} | N/A | {e['license']} |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

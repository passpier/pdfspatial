#!/usr/bin/env python3
"""Score a pdfspatial prediction directory against opendataloader-bench ground truth,
reproducing third_party/opendataloader-bench/src/evaluator.py's exact per-document
ragged mean (Overall = mean of whichever of [nid, teds, mhs] are not None for that
document; TEDS/MHS are None when the ground truth has no table/heading).

Run via scripts/score-pdfspatial.sh, which builds the binary and supplies --gt-dir /
--pred-dir. Not part of the CI-facing test suite; a fast local iteration tool for the
score-raising work described in the plan referenced from CLAUDE.md's Stage 5 section.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean

# The upstream bench's own evaluator modules -- imported directly rather than
# reimplemented, so this script can never silently drift from what
# run-opendataloader-bench.sh actually scores with.
BENCH_SRC = Path(__file__).resolve().parent.parent / "third_party" / "opendataloader-bench" / "src"
sys.path.insert(0, str(BENCH_SRC))

from evaluator_heading_level import evaluate_heading_level  # noqa: E402
from evaluator_reading_order import evaluate_reading_order  # noqa: E402
from evaluator_table import evaluate_table  # noqa: E402


def score_all(gt_dir: Path, pred_dir: Path) -> dict:
    per_doc = {}
    for gt_path in sorted(gt_dir.glob("*.md")):
        doc_id = gt_path.stem
        pred_path = pred_dir / f"{doc_id}.md"
        gt = gt_path.read_text(encoding="utf-8")
        pred = pred_path.read_text(encoding="utf-8") if pred_path.is_file() else ""

        nid, _ = evaluate_reading_order(gt, pred)
        teds, _ = evaluate_table(gt, pred)
        mhs, _ = evaluate_heading_level(gt, pred)

        components = [v for v in (nid, teds, mhs) if v is not None]
        overall = fmean(components) if components else None

        per_doc[doc_id] = {
            "overall": overall,
            "nid": nid,
            "teds": teds,
            "mhs": mhs,
            "prediction_available": pred_path.is_file(),
        }
    return per_doc


def aggregate(per_doc: dict) -> dict:
    def mean_of(key):
        vals = [d[key] for d in per_doc.values() if d[key] is not None]
        return fmean(vals) if vals else None, len(vals)

    overall_mean, overall_n = mean_of("overall")
    nid_mean, nid_n = mean_of("nid")
    teds_mean, teds_n = mean_of("teds")
    mhs_mean, mhs_n = mean_of("mhs")
    return {
        "overall": overall_mean,
        "overall_n": overall_n,
        "nid": nid_mean,
        "nid_n": nid_n,
        "teds": teds_mean,
        "teds_n": teds_n,
        "mhs": mhs_mean,
        "mhs_n": mhs_n,
        "documents": len(per_doc),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--elapsed", type=float, default=None, help="wall-clock seconds for the whole batch run")
    ap.add_argument("--doc-count", type=int, default=None)
    ap.add_argument("--baseline", type=Path, default=None, help="a previously --save-baseline JSON to diff against")
    ap.add_argument("--save-baseline", type=Path, default=None, help="write this run's per-doc scores here")
    args = ap.parse_args()

    per_doc = score_all(args.gt_dir, args.pred_dir)
    agg = aggregate(per_doc)

    print(f"Overall  {agg['overall']:.4f}  (n={agg['overall_n']}/{agg['documents']})")
    print(f"NID      {agg['nid']:.4f}  (n={agg['nid_n']})")
    print(f"TEDS     {agg['teds']:.4f}  (n={agg['teds_n']})")
    print(f"MHS      {agg['mhs']:.4f}  (n={agg['mhs_n']})")
    if args.elapsed is not None and args.doc_count:
        print(f"s/doc    {args.elapsed / args.doc_count:.4f}  ({args.elapsed:.2f}s / {args.doc_count} docs)")

    if args.baseline and args.baseline.is_file():
        baseline = json.loads(args.baseline.read_text())
        base_per_doc = baseline["per_doc"]
        deltas = []
        for doc_id, scores in per_doc.items():
            base = base_per_doc.get(doc_id)
            if not base or base["overall"] is None or scores["overall"] is None:
                continue
            deltas.append((scores["overall"] - base["overall"], doc_id))
        deltas.sort()
        base_agg = baseline["aggregate"]
        print(f"\nvs baseline: Overall {base_agg['overall']:.4f} -> {agg['overall']:.4f} "
              f"({agg['overall'] - base_agg['overall']:+.4f})")
        regressions = [d for d in deltas if d[0] < -0.01]
        improvements = [d for d in deltas if d[0] > 0.01]
        if regressions:
            print(f"\n{len(regressions)} regressed doc(s), worst first:")
            for delta, doc_id in regressions[:15]:
                print(f"  {doc_id}  {delta:+.3f}")
        if improvements:
            print(f"\n{len(improvements)} improved doc(s), best first:")
            for delta, doc_id in sorted(improvements, reverse=True)[:15]:
                print(f"  {doc_id}  {delta:+.3f}")

    if args.save_baseline:
        args.save_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.save_baseline.write_text(json.dumps({"aggregate": agg, "per_doc": per_doc}, indent=2))
        print(f"\nsaved baseline to {args.save_baseline}")


if __name__ == "__main__":
    main()

"""opendataloader-bench adapter for pdfspatial's "compact" output mode:
`--no-page-breaks --no-image-placeholders`.

evaluator_reading_order.py's NID/NID-S only collapse whitespace (`_normalize`) -- they
never strip Markdown syntax. pdfspatial's faithful default output includes a `---`
thematic break between every page and a `![]()` placeholder for every detected picture,
both of which the scorer counts as inserted document text relative to ground truth. This
variant measures the delta that costs: same extraction, same classification, just a
leaner serialization aimed at a scorer that treats Markdown syntax as content. See
`bench/opendataloader/README.md` for the measured delta and why the faithful-default row
stays the headline number.

The flags are hardcoded here (rather than read from PDFSPATIAL_ARGS, which
pdf_parser_pdfspatial.py does support) so both variants can be registered as distinct
engines and run from one `run.py` invocation without one overwriting the other's env var.
"""

import os
import subprocess
from pathlib import Path

BIN = os.environ.get("PDFSPATIAL_BIN", "pdfspatial")
EXTRA_ARGS = ["--no-page-breaks", "--no-image-placeholders"]


def to_markdown(doc_paths, _input_path, output_dir):
    command = [BIN, "--out", str(output_dir), "--quiet", *EXTRA_ARGS]
    command += [str(p) for p in doc_paths]
    subprocess.run(command, check=False)

    for doc_path in doc_paths:
        out_path = Path(output_dir) / f"{Path(doc_path).stem}.md"
        if not out_path.exists():
            out_path.write_text("", encoding="utf-8")

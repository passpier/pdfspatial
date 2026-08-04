"""opendataloader-bench adapter for pdfspatial (crates/pdfspatial-cli), default/faithful
output mode: page breaks (`---`) and picture placeholders (`![]()`) are both kept.

Unlike most adapters in this repo, this one spawns a *single* subprocess for the whole
corpus rather than one per document: `pdf_parser.py` times only the `to_markdown(...)`
call below, and that timer encloses process spawn and PDFium dylib load. `pdfspatial`'s
CLI has a real batch mode (`--out DIR <pdf>...`), so paying that startup cost once for
all N documents -- not N times -- is the honest measurement of how the tool is actually
used, and is what makes the reported speed comparable to engines that keep one process
warm across a whole run.

Env vars:
  PDFSPATIAL_BIN    Path to the built `pdfspatial` binary (default: "pdfspatial", i.e.
                     whatever's on PATH).
  PDFSPATIAL_ARGS   Extra space-separated flags to pass through, e.g.
                     "--no-page-breaks --no-image-placeholders" for the compact variant
                     (see pdf_parser_pdfspatial_compact.py, which hardcodes this instead
                     of relying on the env var so both variants run from one `run.py`
                     invocation without clobbering each other).
"""

import os
import shlex
import subprocess
from pathlib import Path

BIN = os.environ.get("PDFSPATIAL_BIN", "pdfspatial")
EXTRA_ARGS = shlex.split(os.environ.get("PDFSPATIAL_ARGS", ""))


def to_markdown(doc_paths, _input_path, output_dir):
    command = [BIN, "--out", str(output_dir), "--quiet", *EXTRA_ARGS]
    command += [str(p) for p in doc_paths]
    subprocess.run(command, check=False)

    # Belt and braces: evaluator.py drops any document with no prediction file from the
    # mean rather than scoring it zero (see evaluator.py's `prediction_available`
    # filtering). Silently omitting a file on a crash would therefore *inflate* our
    # score -- pdfspatial's own batch mode already writes an empty <stem>.md on
    # failure, but this loop is a second line of defense if the whole subprocess
    # failed to start (e.g. wrong PDFSPATIAL_BIN) and produced nothing at all.
    for doc_path in doc_paths:
        out_path = Path(output_dir) / f"{Path(doc_path).stem}.md"
        if not out_path.exists():
            out_path.write_text("", encoding="utf-8")

"""opendataloader-bench adapter for pdf-inspector (https://github.com/firecrawl/pdf-inspector),
a Rust `lopdf`-based extractor -- the most directly analogous competitor to pdfspatial in
this comparison (both are dependency-light, model-free, deterministic Rust extractors).

pdf-inspector's `pdf2md` CLI has no batch mode, so this adapter -- unlike
pdf_parser_pdfspatial.py -- spawns one subprocess *per document*. That is a real property
of the tool being measured, not a handicap this adapter imposes; it is the single biggest
confound in the speed comparison and is called out explicitly in
`bench/opendataloader/README.md`. `pdf2md`'s default output has no page-break markers
(`--pages` opts *into* them), so pdf-inspector's default is closer to pdfspatial's
`--no-page-breaks` compact variant than to pdfspatial's own faithful default on that one
axis.

Env vars:
  PDF_INSPECTOR_BIN   Path to the `pdf2md` binary (default: "pdf2md", i.e. whatever's on
                       PATH -- `cargo install pdf-inspector` puts it there).
"""

import os
import subprocess
from pathlib import Path

BIN = os.environ.get("PDF_INSPECTOR_BIN", "pdf2md")


def to_markdown(doc_paths, _input_path, output_dir):
    for doc_path in doc_paths:
        out_path = Path(output_dir) / f"{Path(doc_path).stem}.md"
        result = subprocess.run(
            [BIN, str(doc_path)], capture_output=True, text=True, check=False
        )
        out_path.write_text(
            result.stdout if result.returncode == 0 else "", encoding="utf-8"
        )

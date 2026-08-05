
# --- BEGIN pdfspatial bench engines (appended by scripts/run-opendataloader-bench.sh) ---
# Upstream's engine_registry.py keeps three module-level dicts (ENGINES,
# _ENGINE_MODULES, ALL_CHART_ENGINES) plus a lazy ENGINE_DISPATCH that resolves
# handlers through _ENGINE_MODULES on first access -- registering a new engine is a
# pure append, no fork and no edits to any existing line. This block is applied
# idempotently by the run script (checkout -> pull -> append-if-absent), so it's safe
# to re-run against a fresh clone.
import importlib.metadata
import os

ENGINES["pdfspatial"] = os.environ.get("PDFSPATIAL_VERSION") or "0.1.0"
ENGINES["pdfspatial-compact"] = os.environ.get("PDFSPATIAL_VERSION") or "0.1.0"
ENGINES["pdf-inspector"] = os.environ.get("PDF_INSPECTOR_VERSION") or "0.1.7"

_ENGINE_MODULES["pdfspatial"] = "pdf_parser_pdfspatial"
_ENGINE_MODULES["pdfspatial-compact"] = "pdf_parser_pdfspatial_compact"
_ENGINE_MODULES["pdf-inspector"] = "pdf_parser_pdf_inspector"

for _name in ("pdfspatial", "pdfspatial-compact", "pdf-inspector"):
    ALL_CHART_ENGINES[_name] = ENGINES[_name]

# `run.py`/`pdf_parser.py` read ENGINES[name] straight into results.json's `version`
# field (via summary.engine_version) -- upstream's own literals (e.g. liteparse
# "1.2.1") go stale the moment `uv sync --upgrade` re-resolves uv.lock to a newer
# release, silently mislabelling every number that release produced. Override with
# whatever's actually installed in this environment, for every upstream Python engine
# this repo's harness runs.
for _pkg, _engine in (
    ("opendataloader-pdf", "opendataloader"),
    ("markitdown", "markitdown"),
    ("liteparse", "liteparse"),
):
    try:
        ENGINES[_engine] = importlib.metadata.version(_pkg)
        if _engine in ALL_CHART_ENGINES:
            ALL_CHART_ENGINES[_engine] = ENGINES[_engine]
    except importlib.metadata.PackageNotFoundError:
        pass
# --- END pdfspatial bench engines ---

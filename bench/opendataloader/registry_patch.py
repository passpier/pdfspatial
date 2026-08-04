
# --- BEGIN pdfspatial bench engines (appended by scripts/run-opendataloader-bench.sh) ---
# Upstream's engine_registry.py keeps three module-level dicts (ENGINES,
# _ENGINE_MODULES, ALL_CHART_ENGINES) plus a lazy ENGINE_DISPATCH that resolves
# handlers through _ENGINE_MODULES on first access -- registering a new engine is a
# pure append, no fork and no edits to any existing line. This block is applied
# idempotently by the run script (checkout -> pull -> append-if-absent), so it's safe
# to re-run against a fresh clone.
ENGINES["pdfspatial"] = "0.1.0"
ENGINES["pdfspatial-compact"] = "0.1.0"
ENGINES["pdf-inspector"] = "0.1.7"

_ENGINE_MODULES["pdfspatial"] = "pdf_parser_pdfspatial"
_ENGINE_MODULES["pdfspatial-compact"] = "pdf_parser_pdfspatial_compact"
_ENGINE_MODULES["pdf-inspector"] = "pdf_parser_pdf_inspector"

for _name in ("pdfspatial", "pdfspatial-compact", "pdf-inspector"):
    ALL_CHART_ENGINES[_name] = ENGINES[_name]
# --- END pdfspatial bench engines ---

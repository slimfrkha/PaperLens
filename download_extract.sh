#!/usr/bin/env bash
# Manual ingestion entrypoint. The paper list is no longer hardcoded here — it
# lives in config.yaml. This delegates to the Python pipeline, which downloads,
# converts to markdown (Docling), indexes into the RAG DB, and tags each paper.
#
# Requires deps installed first: `uv sync`
# Usage: bash download_extract.sh   (or: uv run paperlens-ingest)
set -e
exec paperlens-ingest "$@"

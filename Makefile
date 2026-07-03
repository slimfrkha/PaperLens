.PHONY: install dev ingest build serve

# The active config. Override on the CLI, e.g. `make serve CONFIG=configs/examples/anthropic.yaml`.
CONFIG ?= configs/recent-oss-agentic-models.yaml

# Install Python (uv-managed venv + locked deps) + frontend deps.
install:
	uv sync
	npm --prefix web install

# Run backend + frontend dev server together (Ctrl-C stops both).
dev:
	uv run paperlens-serve --config $(CONFIG) & npm --prefix web run dev; kill %1 2>/dev/null || true

# Ingest configured papers not yet in the DB.
ingest:
	uv run paperlens-ingest --config $(CONFIG)

# Backend only.
serve:
	uv run paperlens-serve --config $(CONFIG)

# Production build of the frontend (served by the backend at cfg.paths.web_dist).
build:
	npm --prefix web run build

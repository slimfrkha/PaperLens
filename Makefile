.PHONY: install dev ingest build serve

# Install Python (uv-managed venv + locked deps) + frontend deps.
install:
	uv sync
	npm --prefix web install

# Run backend + frontend dev server together (Ctrl-C stops both).
dev:
	uv run paperlens-serve & npm --prefix web run dev; kill %1 2>/dev/null || true

# Ingest configured papers not yet in the DB.
ingest:
	uv run paperlens-ingest

# Backend only.
serve:
	uv run paperlens-serve

# Production build of the frontend (served by the backend at cfg.paths.web_dist).
build:
	npm --prefix web run build

.PHONY: install dev ingest build serve

# Install Python (editable) + frontend deps.
install:
	pip install -e .
	npm --prefix web install

# Run backend + frontend dev server together (Ctrl-C stops both).
dev:
	paperlens-serve & npm --prefix web run dev; kill %1 2>/dev/null || true

# Ingest configured papers not yet in the DB.
ingest:
	paperlens-ingest

# Backend only.
serve:
	paperlens-serve

# Production build of the frontend (served by the backend at cfg.paths.web_dist).
build:
	npm --prefix web run build

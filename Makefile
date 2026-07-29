.PHONY: install dev ingest build serve

# The active config, e.g. `make serve CONFIG=configs/examples/anthropic.yaml`. Unset
# falls through to paperlens-serve/-ingest's own discovery (PAPERLENS_CONFIG env var,
# then an upward search for config.yaml) — there is no default config.
CONFIG_FLAG = $(if $(CONFIG),--config_path $(CONFIG),)

# Install Python (uv-managed venv + locked deps) + frontend deps.
install:
	uv sync
	npm --prefix web install

# Run backend + frontend dev server together (Ctrl-C stops both).
dev:
	uv run paperlens-serve $(CONFIG_FLAG) & npm --prefix web run dev; kill %1 2>/dev/null || true

# Ingest configured papers not yet in the DB.
ingest:
	uv run paperlens-ingest $(CONFIG_FLAG)

# Backend only.
serve:
	uv run paperlens-serve $(CONFIG_FLAG)

# Production build of the frontend (served by the backend at cfg.paths.web_dist).
build:
	npm --prefix web run build

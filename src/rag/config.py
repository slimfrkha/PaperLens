"""Typed configuration loaded from config.yaml — the app's single source of truth.

Everything (paths, embedder, reranker, tagging/chat LLMs, ingestion, server,
and the paper list) is configured here so nothing is hardcoded in scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load a local .env (e.g. ANTHROPIC_API_KEY=...) so every entrypoint that reads
# config — server, ingest, tagger, chat — picks up credentials automatically.
load_dotenv()

CONFIG_FILENAME = "config.yaml"
CONFIG_ENV_VAR = "PAPERLENS_CONFIG"

# Path fields anchored to the project root (relative values) at load time.
_PATH_FIELDS = ("rag_db", "pdf_dir", "markdown_dir", "chat_history", "web_dist")


class Paths(BaseModel):
    rag_db: str = "data/rag_db"
    pdf_dir: str = "data/papers/pdf"
    markdown_dir: str = "data/papers/text"
    chat_history: str = "data/chat_history"  # per-session ChatML JSON files
    web_dist: str = "web/dist"  # built frontend SPA served by the backend


class EmbeddingCfg(BaseModel):
    model: str = "BAAI/bge-m3"
    type: str = "hf"  # hf | openai
    max_seq_length: int = 1024
    batch_size: int = 32
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"


class RerankerCfg(BaseModel):
    model: str = "BAAI/bge-reranker-v2-m3"
    enabled: bool = True


class LLMSpec(BaseModel):
    provider: str = "anthropic"  # anthropic | openai
    model: str = "claude-opus-4-8"
    api_base: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 2048
    temperature: float = 0.0


class LLMCfg(BaseModel):
    tagging: LLMSpec = Field(
        default_factory=lambda: LLMSpec(model="claude-haiku-4-5-20251001")
    )
    chat: LLMSpec = Field(default_factory=lambda: LLMSpec(model="claude-opus-4-8"))


class IngestionCfg(BaseModel):
    auto_start: bool = True
    concurrency: int = 1


class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class Paper(BaseModel):
    name: str
    arxiv_id: str


class Config(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    collection: str = "arxiv_papers"
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    reranker: RerankerCfg = Field(default_factory=RerankerCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    ingestion: IngestionCfg = Field(default_factory=IngestionCfg)
    server: ServerCfg = Field(default_factory=ServerCfg)
    papers: list[Paper] = Field(default_factory=list)
    # Resolved project root (directory of the loaded config.yaml). Set by
    # load_config, not read from YAML; excluded from serialization.
    root: Path = Field(default_factory=Path.cwd, exclude=True)


def _find_config(path: str | None) -> Path:
    """Locate config.yaml: explicit path -> env var -> upward search from CWD.

    An explicit path (e.g. --config) is resolved relative to the CWD. This makes
    every entrypoint CWD-independent instead of assuming CWD == repo root.
    """
    if path:
        return Path(path)
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        return Path(env)
    cwd = Path.cwd()
    for d in (cwd, *cwd.parents):
        candidate = d / CONFIG_FILENAME
        if candidate.exists():
            return candidate
    return Path(CONFIG_FILENAME)  # not found -> falls back to Config defaults


def load_config(path: str | None = None) -> Config:
    cfg_path = _find_config(path)
    data = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg = Config(**(data or {}))

    # Anchor every relative path to the config file's directory (the project
    # root); absolute paths are left untouched.
    cfg.root = cfg_path.resolve().parent if cfg_path.exists() else Path.cwd()
    for field in _PATH_FIELDS:
        p = Path(getattr(cfg.paths, field))
        if not p.is_absolute():
            setattr(cfg.paths, field, str((cfg.root / p).resolve()))
    return cfg

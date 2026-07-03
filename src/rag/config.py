"""Typed configuration loaded from config.yaml — the app's single source of truth.

Everything (paths, embedder, reranker, tagging/chat LLMs, ingestion, server,
and the paper list) is configured here so nothing is hardcoded in scripts.

The schema is plain dataclasses decoded by ``draccus``. The embedder, reranker,
and LLM sections are ``draccus.ChoiceRegistry`` bases: a ``type`` string in the
YAML selects the variant subclass, which carries only that backend's fields
(e.g. ``max_seq_length`` is hf-only; ``api_base`` only on OpenAI-compatible LLMs).

Two entry points:

* ``load_config(path)`` — programmatic (server internals, ingest, tests). YAML ->
  OmegaConf ``${...}`` resolution -> ``draccus.decode`` -> path anchoring.
* ``parse_config(argv)`` — the CLI (``paperlens-serve`` / ``paperlens-ingest``).
  Adds draccus's per-field ``--help`` and ``--field.subfield=value`` overrides on
  top of the file, resolving ``${...}`` *after* the file+CLI merge so CLI
  overrides feed interpolation.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import draccus
import mergedeep
import yaml
from dotenv import load_dotenv
from draccus import cfgparsing, utils
from draccus.argparsing import ArgumentParser
from draccus.parsers import decoding
from omegaconf import OmegaConf

# Load a local .env (e.g. ANTHROPIC_API_KEY=...) so every entrypoint that reads
# config — server, ingest, tagger, chat — picks up credentials automatically.
load_dotenv()

CONFIG_FILENAME = "config.yaml"
CONFIG_ENV_VAR = "PAPERLENS_CONFIG"

# Path fields anchored to the project root (relative values) at load time.
_PATH_FIELDS = ("rag_db", "pdf_dir", "markdown_dir", "chat_history", "web_dist")


@dataclass
class Paths:
    rag_db: str = "data/rag_db"
    pdf_dir: str = "data/papers/pdf"
    markdown_dir: str = "data/papers/text"
    chat_history: str = "data/chat_history"  # per-session ChatML JSON files
    web_dist: str = "web/dist"  # built frontend SPA served by the backend


# --- Embedder: a `type` string selects the variant; each carries only its fields ---
@dataclass
class EmbeddingCfg(draccus.ChoiceRegistry):
    model: str = "BAAI/bge-m3"
    batch_size: int = 32

    @classmethod
    def default_choice_name(cls) -> str:  # `type:` may be omitted -> hf
        return "hf"


@EmbeddingCfg.register_subclass("hf")
@dataclass
class HFEmbeddingCfg(EmbeddingCfg):
    # Token cap guarding the MPS 2**32 tensor limit on Apple Silicon (hf only).
    max_seq_length: int = 1024


@EmbeddingCfg.register_subclass("openai")
@dataclass
class OpenAIEmbeddingCfg(EmbeddingCfg):
    model: str = "text-embedding-3-small"
    api_base: str = ""  # any OpenAI-compatible endpoint ("" = provider default)
    api_key_env: str = "OPENAI_API_KEY"


@EmbeddingCfg.register_subclass("gemini")
@dataclass
class GeminiEmbeddingCfg(EmbeddingCfg):
    model: str = "gemini-embedding-001"
    api_key_env: str = "GEMINI_API_KEY"


@EmbeddingCfg.register_subclass("ollama")
@dataclass
class OllamaEmbeddingCfg(EmbeddingCfg):
    model: str = "nomic-embed-text"
    api_base: str = ""  # "" = http://localhost:11434


# --- Reranker: second retrieval stage; `enabled` toggles the whole stage ---
@dataclass
class RerankerCfg(draccus.ChoiceRegistry):
    enabled: bool = True

    @classmethod
    def default_choice_name(cls) -> str:
        return "hf"


@RerankerCfg.register_subclass("hf")
@dataclass
class HFRerankerCfg(RerankerCfg):
    model: str = "BAAI/bge-reranker-v2-m3"  # local cross-encoder


@RerankerCfg.register_subclass("llm")
@dataclass
class LLMRerankerCfg(RerankerCfg):
    pass  # reuses the chat LLM; no extra fields


# --- LLM backends: a `type` string selects the provider variant ---
@dataclass
class LLMSpec(draccus.ChoiceRegistry):
    model: str = "claude-opus-4-8"
    max_tokens: int = 2048
    temperature: float = 0.0
    api_key_env: str = "ANTHROPIC_API_KEY"

    @classmethod
    def default_choice_name(cls) -> str:
        return "anthropic"


@LLMSpec.register_subclass("anthropic")
@dataclass
class AnthropicSpec(LLMSpec):
    pass


@LLMSpec.register_subclass("openai")
@dataclass
class OpenAISpec(LLMSpec):
    api_base: str = ""  # endpoint URL for any OpenAI-compatible server ("" = OpenAI)
    api_key_env: str = "OPENAI_API_KEY"


@LLMSpec.register_subclass("vllm")
@dataclass
class VLLMSpec(OpenAISpec):
    pass


@LLMSpec.register_subclass("sglang")
@dataclass
class SGLangSpec(OpenAISpec):
    pass


@LLMSpec.register_subclass("gemini")
@dataclass
class GeminiSpec(LLMSpec):
    api_key_env: str = "GEMINI_API_KEY"


@dataclass
class LLMCfg:
    tagging: LLMSpec = field(
        default_factory=lambda: AnthropicSpec(model="claude-haiku-4-5-20251001")
    )
    chat: LLMSpec = field(default_factory=lambda: AnthropicSpec(model="claude-opus-4-8"))


@dataclass
class IngestionCfg:
    auto_start: bool = True
    concurrency: int = 1


@dataclass
class ServerCfg:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Paper:
    name: str = ""
    arxiv_id: str = ""


@dataclass(frozen=True)
class IngestConfig:
    """Narrow, read-only view of Config with only the fields ingestion consumes.

    Built via ``Config.for_ingest()``; never decoded from YAML directly. Frozen
    because it is a snapshot: it aliases Config's sub-objects (paths, embedding,
    papers) by reference, so treating it as an independently-mutable config would
    be a bug — writes wouldn't propagate. There is deliberately no ServeConfig:
    serve uses the full Config because ``create_app`` hosts the ingestion worker,
    which needs every field ingestion needs (see ``Config.for_ingest``)."""

    paths: Paths
    collection: str
    embedding: EmbeddingCfg
    tagging: LLMSpec  # flattened from Config.llm.tagging
    papers: list[Paper]


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    collection: str = "arxiv_papers"
    # Interpolation base for paths, e.g. `rag_db: ${data_path}/${collection}/rag_db`.
    # Not itself a path field (not anchored); configs that reference ${data_path} must set it.
    data_path: str = "data"
    embedding: EmbeddingCfg = field(default_factory=HFEmbeddingCfg)
    reranker: RerankerCfg = field(default_factory=HFRerankerCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    ingestion: IngestionCfg = field(default_factory=IngestionCfg)
    server: ServerCfg = field(default_factory=ServerCfg)
    papers: list[Paper] = field(default_factory=list)
    # Resolved project root (directory of the loaded config.yaml). Set by the
    # loader, not read from YAML; init=False keeps it off the CLI and decode input.
    root: Path = field(default_factory=Path.cwd, init=False)

    def for_ingest(self) -> IngestConfig:
        """Project this Config to the ingestion-only view (CLI + in-process worker).

        Serve keeps using the full Config directly — there is no ServeConfig, since
        the server hosts the ingestion worker and thus reads every field."""
        return IngestConfig(
            paths=self.paths,
            collection=self.collection,
            embedding=self.embedding,
            tagging=self.llm.tagging,
            papers=self.papers,
        )


def _find_config(path: str | None) -> Path:
    """Locate config.yaml: explicit path -> env var -> upward search from CWD.

    An explicit path (e.g. --config_path) is resolved relative to the CWD. This
    makes every entrypoint CWD-independent instead of assuming CWD == repo root.
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


def _load_yaml(cfg_path: Path | None) -> dict:
    if cfg_path is None or not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text()) or {}


def _resolve(data: dict) -> dict:
    """Resolve OmegaConf ``${...}`` interpolation, returning a plain dict."""
    resolved = OmegaConf.to_container(OmegaConf.create(data), resolve=True)
    assert isinstance(resolved, dict)
    return resolved


def _anchor(cfg: Config, cfg_path: Path) -> Config:
    """Anchor every relative path to the config file's directory (project root);
    absolute paths are left untouched. Also records ``cfg.root``."""
    cfg.root = cfg_path.resolve().parent if cfg_path.exists() else Path.cwd()
    for name in _PATH_FIELDS:
        p = Path(getattr(cfg.paths, name))
        if not p.is_absolute():
            setattr(cfg.paths, name, str((cfg.root / p).resolve()))
    return cfg


def load_config(path: str | None = None) -> Config:
    """Load and decode config.yaml (no CLI overrides). ``${...}`` is resolved
    against the file's own values."""
    cfg_path = _find_config(path)
    data = _resolve(_load_yaml(cfg_path))
    cfg = decoding.decode(Config, data)
    return _anchor(cfg, cfg_path)


class _InterpolatingArgumentParser(ArgumentParser):
    """draccus parser that resolves ``${...}`` AFTER the file+CLI merge, so CLI
    overrides (``--embedding.batch_size=64``) feed interpolation.

    Only ``_postprocessing`` (parsed-args -> dataclass) is reimplemented; arg
    registration and ``--help`` remain draccus's. This couples us to draccus's
    internal merge/decode step, hence the pin in pyproject.toml and the
    regression test in tests/unit/test_config.py.
    """

    def _postprocessing(self, parsed_args):
        vals = {k: cfgparsing.parse_string(v) for k, v in vars(parsed_args).items()}
        # draccus's own --config_path (CONFIG_ARG) wins over the discovered default.
        raw = vals.pop(utils.CONFIG_ARG, None) or self.config_path
        config_path: str | None = str(raw) if raw else None
        cfg_path = Path(config_path) if config_path else Path.cwd()
        file_args = _load_yaml(Path(config_path) if config_path else None)
        merged = cast(dict, mergedeep.merge(file_args, utils.deflatten(vals, sep=".")))
        cfg = decoding.decode(self.config_class, _resolve(merged))
        return _anchor(cfg, cfg_path)


def parse_config(argv: Sequence[str] | None = None) -> Config:
    """CLI config loader: file + draccus per-field overrides, ``${...}`` resolved
    after the merge so overrides feed interpolation. Honors ``--config_path`` and
    falls back to the same discovery as ``load_config``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg_path = _find_config(None)
    parser = _InterpolatingArgumentParser(
        Config, config_path=str(cfg_path) if cfg_path.exists() else None
    )
    return parser.parse_args(argv)

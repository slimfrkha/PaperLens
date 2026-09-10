"""index_markdown removes chunks orphaned by a changed chunking config.

Offline against a real temp Chroma (fake embedder) — no model download, no network. See
also ``tests/unit/test_eval_index_isolated.py``, whose contamination test proves the same
bug exists one layer down in ``upsert_chunks`` (deliberately left unfixed there — the eval
harness works around it with a throwaway collection per config); this file proves the fix
at ``index_markdown``, the layer production ingestion (``pipeline.ingest_paper``) calls.
"""

from __future__ import annotations

from rag.config import ChunkingCfg
from rag.index import collect_chunks, index_markdown, open_collection, remove_paper_chunks

# Same body shape as test_eval_index_isolated.py's _POOL: one long numbered section that
# packs into fewer chunks at a large max_tokens than at a small one, so chunk count is
# observably config-dependent.
_PARA = "the model applies latent attention over compressed key value cache states here now"
_BODY = "\n\n".join(_PARA for _ in range(12))
_MARKDOWN = f"## My Paper\n\nAuthor One, Author Two\n\n## 1. Method\n\n{_BODY}\n"


def _write_md(tmp_path, name: str = "paper-a") -> tuple[str, str]:
    path = tmp_path / f"{name}.md"
    path.write_text(_MARKDOWN)
    return str(path), name


def test_index_markdown_reindex_under_changed_chunking_removes_stale_chunks(
    tmp_path, fake_embedder
):
    md_path, paper_id = _write_md(tmp_path)
    collection = open_collection(str(tmp_path / "db"), "papers", embedder_name=fake_embedder.name())

    n_small = index_markdown(
        collection, fake_embedder, md_path, paper_id, chunking=ChunkingCfg(max_tokens=128)
    )
    assert collection.count() == n_small

    n_large = index_markdown(
        collection, fake_embedder, md_path, paper_id, chunking=ChunkingCfg(max_tokens=512)
    )
    # The precondition the bug rests on: the larger max_tokens really packs into fewer parts.
    assert n_large < n_small
    # THE FIX: the collection reflects only the new (fewer) chunks, not the union of both.
    assert collection.count() == n_large


def test_index_markdown_same_config_rerun_is_idempotent(tmp_path, fake_embedder):
    md_path, paper_id = _write_md(tmp_path)
    collection = open_collection(str(tmp_path / "db"), "papers", embedder_name=fake_embedder.name())
    chunking = ChunkingCfg(max_tokens=128)

    n1 = index_markdown(collection, fake_embedder, md_path, paper_id, chunking=chunking)
    n2 = index_markdown(collection, fake_embedder, md_path, paper_id, chunking=chunking)

    assert n1 == n2
    assert collection.count() == n1  # unchanged content -> nothing deleted, nothing duplicated


def test_index_markdown_fresh_ingest_unaffected(tmp_path, fake_embedder):
    md_path, paper_id = _write_md(tmp_path)
    collection = open_collection(str(tmp_path / "db"), "papers", embedder_name=fake_embedder.name())

    n = index_markdown(
        collection, fake_embedder, md_path, paper_id, chunking=ChunkingCfg(max_tokens=512)
    )

    assert n > 0
    assert collection.count() == n  # no prior chunks for this paper_id -> nothing to clean up


def test_remove_paper_chunks_deletes_only_target_paper(tmp_path, fake_embedder):
    md_path_a, paper_a = _write_md(tmp_path, name="paper-a")
    md_path_b, paper_b = _write_md(tmp_path, name="paper-b")
    collection = open_collection(str(tmp_path / "db"), "papers", embedder_name=fake_embedder.name())

    n_a = index_markdown(collection, fake_embedder, md_path_a, paper_a, chunking=ChunkingCfg())
    n_b = index_markdown(collection, fake_embedder, md_path_b, paper_b, chunking=ChunkingCfg())
    assert collection.count() == n_a + n_b

    remove_paper_chunks(collection, paper_a)

    assert collection.count() == n_b
    assert collection.get(where={"paper_id": paper_a}, include=[])["ids"] == []
    assert len(collection.get(where={"paper_id": paper_b}, include=[])["ids"]) == n_b


def test_remove_paper_chunks_unknown_paper_is_a_noop(tmp_path, fake_embedder):
    md_path, paper_id = _write_md(tmp_path)
    collection = open_collection(str(tmp_path / "db"), "papers", embedder_name=fake_embedder.name())
    n = index_markdown(collection, fake_embedder, md_path, paper_id, chunking=ChunkingCfg())

    remove_paper_chunks(collection, "no-such-paper")

    assert collection.count() == n


def test_collect_chunks_ignores_display_markdown(tmp_path):
    (tmp_path / "paper-a.md").write_text(_MARKDOWN)
    (tmp_path / "paper-a_display.md").write_text(_MARKDOWN + "\n![Figure](paper-a.assets/x.png)")

    chunks = collect_chunks(str(tmp_path))

    assert chunks
    assert {chunk.metadata["paper_id"] for chunk in chunks} == {"paper-a"}


def test_index_markdown_accepts_repeated_unnumbered_section_titles(tmp_path, fake_embedder):
    repeated = " ".join(["distinct section content"] * 20)
    path = tmp_path / "paper-a.md"
    path.write_text(
        f"## Paper A\n\n## Agent.\n\nfirst {repeated}\n\n## Agent.\n\nsecond {repeated}\n"
    )
    collection = open_collection(str(tmp_path / "db"), "papers", embedder_name=fake_embedder.name())

    n_chunks = index_markdown(
        collection,
        fake_embedder,
        str(path),
        "paper-a",
        chunking=ChunkingCfg(),
    )

    assert n_chunks == 2
    assert collection.count() == 2

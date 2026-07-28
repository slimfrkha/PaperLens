"""Guard 1: isolated re-index cells never contaminate, and the count guard holds.

Offline against a real temp Chroma (fake embedder) — no model download, no network. The
contamination test *is* the plan's 10-minute proof, turned into a regression: re-chunking
into the same collection inflates its count; ``build_isolated_searcher`` keeps it exact.
"""

from __future__ import annotations

from eval.index_isolated import build_isolated_searcher, cell_signature, chunks_for
from rag.config import ChunkingCfg, HFEmbeddingCfg
from rag.index import open_collection, upsert_chunks

# A paper whose one numbered section is long enough (many paragraphs) to split into several
# chunks at a small max_tokens but pack into fewer at a large one — so chunk COUNT depends on
# the chunking config, which is exactly what makes contamination observable.
_PARA = "the model applies latent attention over compressed key value cache states here now"
_BODY = "\n\n".join(_PARA for _ in range(12))
_POOL = {"p1": f"## My Paper\n\nAuthor One, Author Two\n\n## 1. Method\n\n{_BODY}\n"}


def test_chunk_count_depends_on_max_tokens():
    # Precondition the contamination test rests on: the two configs really cut differently.
    assert len(chunks_for(_POOL, ChunkingCfg(max_tokens=128))) > len(
        chunks_for(_POOL, ChunkingCfg(max_tokens=512))
    )


def test_reusing_one_collection_across_chunkings_contaminates(tmp_path, fake_embedder):
    # THE BUG (Guard 1): upsert two chunkings into the SAME collection. _chunk_id encodes
    # `part`, so the finer cut's later parts get new ids and pile up beside the coarse cut —
    # the collection becomes a union of two arms and every retrieval number after is garbage.
    big = chunks_for(_POOL, ChunkingCfg(max_tokens=512))  # coarse: one chunk
    small = chunks_for(_POOL, ChunkingCfg(max_tokens=128))  # fine: several chunks
    coll = open_collection(
        str(tmp_path / "db"), "contam_cell", embedder_name=fake_embedder.name(), reset=True
    )
    upsert_chunks(coll, fake_embedder, big)
    assert coll.count() == len(big)  # clean: reflects exactly the coarse cut
    upsert_chunks(coll, fake_embedder, small)
    # Re-indexing the finer cut into the SAME collection inflates it past the coarse cut's
    # true size (the plan's "count goes up" proof) — the collection is now a union of two
    # arms, so any retrieval number read off it belongs to neither config.
    assert coll.count() > len(big)


def test_build_isolated_searcher_keeps_count_exact(tmp_path, fake_embedder):
    # THE FIX: each config gets its own fresh collection (reset=True), so count == exactly the
    # chunks that config produced — no pileup, no cross-arm contamination.
    for mt in (128, 512):
        chunking = ChunkingCfg(max_tokens=mt)
        searcher = build_isolated_searcher(
            _POOL, chunking, HFEmbeddingCfg(), db_dir=str(tmp_path / "db"), embedder=fake_embedder
        )
        assert searcher.collection.count() == len(chunks_for(_POOL, chunking))


def test_isolated_cells_do_not_leak_into_each_other(tmp_path, fake_embedder):
    # Two different chunkings in the SAME db_dir land in DIFFERENT collections (distinct
    # signatures), so building the second never changes the first's count.
    s_small = build_isolated_searcher(
        _POOL,
        ChunkingCfg(max_tokens=128),
        HFEmbeddingCfg(),
        db_dir=str(tmp_path / "db"),
        embedder=fake_embedder,
    )
    n_small = s_small.collection.count()
    build_isolated_searcher(
        _POOL,
        ChunkingCfg(max_tokens=512),
        HFEmbeddingCfg(),
        db_dir=str(tmp_path / "db"),
        embedder=fake_embedder,
    )
    assert s_small.collection.count() == n_small  # untouched by the second cell


def test_cell_signature_stable_and_sensitive():
    emb = HFEmbeddingCfg()
    a = ChunkingCfg(max_tokens=512)
    assert cell_signature(a, emb) == cell_signature(ChunkingCfg(max_tokens=512), emb)  # stable
    assert cell_signature(a, emb) != cell_signature(ChunkingCfg(max_tokens=256), emb)  # chunking
    assert cell_signature(a, emb) != cell_signature(a, HFEmbeddingCfg(model="other"))  # embedder
    assert cell_signature(a, emb).startswith("cell_")
